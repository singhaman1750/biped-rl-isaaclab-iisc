"""Co-optimisation on-policy runner.

Extends ``OnPolicyRunner`` with an evolutionary algorithm (EA) outer loop that
periodically replaces the robot morphology population and restarts environment
episodes.  The policy (PPO + ActorCritic) is unchanged.

Usage::

    runner = CoptOnPolicyRunner(env, design_generator, agent_cfg_dict,
                                log_dir=log_dir, device=device)
    runner.learn(num_learning_iterations=50000)
"""

from __future__ import annotations

import os
import time
import torch
import warnings
from collections import deque
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner

from co_optimisation.modules import CoptActorCritic
from co_optimisation.runners.usd_generator import DesignGeneratorBase, Population
from co_optimisation.utils.respawn import apply_actuator_params, respawn_robots


class CoptOnPolicyRunner(OnPolicyRunner):
    """On-policy runner with evolutionary design co-optimisation.

    Adds an outer EA loop on top of the standard PPO training loop.  Every
    ``ea_update_interval`` policy-update iterations the runner:

    1. Evaluates per-individual fitness from accumulated episode returns.
    2. Calls ``design_generator.update_with_fitness()`` so the generator can
       improve future designs.
    3. Generates a new design population.
    4. Hot-swaps all robot articulations via the 10-step respawn sequence.
    5. Patches ``IdentifiedActuator`` tensor attributes for the new designs.
    6. Resets fitness accumulators and fetches fresh observations.

    Args:
        env: The wrapped ``VecEnv`` (``RslRlVecEnvWrapper``).
        design_generator: A :class:`DesignGeneratorBase` that produces
            :class:`Population` objects.
        train_cfg: RSL-RL training configuration dict (same as
            ``OnPolicyRunner``).  May contain an optional ``"copt"`` sub-dict
            with keys:

            - ``"ea_update_interval"`` (int, default 100): policy-update
              iterations between EA generations.
            - ``"num_individuals"`` (int, default 16): population size.
            - ``"ea_late_start"`` (int, default -1): delay for policy warm
              start before EA generations are updated.

        log_dir: Directory for TensorBoard / checkpoint logging.
        device: Torch device string.
    """

    def __init__(
        self,
        env: VecEnv,
        design_generator: DesignGeneratorBase,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        # Extract COPT-specific config before calling super().__init__
        copt_cfg: dict = train_cfg.get("copt", {})
        self._ea_update_interval: int = copt_cfg.get("ea_update_interval", 100)
        self._ea_late_start: int = copt_cfg.get("ea_late_start", -1)
        self._num_individuals: int = copt_cfg.get("num_individuals", 16)
        self._design_generator = design_generator
        # Per-generation state
        self.generation: int = 0
        self.current_population: Population | None = None

        # Fitness accumulators indexed by individual
        self._individual_fitness = torch.zeros(
            self._num_individuals, dtype=torch.float, device=device
        )
        self._individual_episode_counts = torch.zeros(
            self._num_individuals, dtype=torch.long, device=device
        )
        self.encoder_cfg = train_cfg["encoder"]

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        # Maps env_idx → individual_idx (round-robin)
        # The round robin is setup in co_optimisation.utils.respawn.respawn_robots
        self._env_to_individual: list[int] = self._assign_individuals_to_envs()

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        """Run training with periodic EA morphology updates."""
        # Initialise logging writer
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Generate initial population before the first rollout and start learning
        self.respawn_robots = respawn_robots()
        with torch.inference_mode():
            self._reload_morphology()

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        # Create buffers for logging extrinsic and intrinsic rewards
        if self.alg.rnd:
            erewbuffer: deque = deque(maxlen=100)
            irewbuffer: deque = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(
                self.env.num_envs, dtype=torch.float, device=self.device
            )
            cur_ireward_sum = torch.zeros(
                self.env.num_envs, dtype=torch.float, device=self.device
            )

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Sample actions
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if self.alg.rnd else None
                    )
                    # Book keeping

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        # Update rewards
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        # Update episode length
                        cur_episode_length += 1
                        # Clear data for completed episodes
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # ---- COPT: accumulate per-individual fitness --------
                        completed_env_ids = new_ids[:, 0]
                        for env_idx in completed_env_ids.tolist():
                            ind_idx = self._env_to_individual[env_idx]
                            self._individual_fitness[ind_idx] += cur_reward_sum[env_idx]
                            self._individual_episode_counts[ind_idx] += 1
                        # ---- end COPT injection ----------------------------

                        rewbuffer.extend(
                            cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        lenbuffer.extend(
                            cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                        )
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        if self.alg.rnd:
                            erewbuffer.extend(
                                cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            )
                            irewbuffer.extend(
                                cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist()
                            )
                            cur_ereward_sum[new_ids] = 0
                            cur_ireward_sum[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())
                # Save model
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            # Clear episode infos
            ep_infos.clear()

            if it == start_iter and not self.disable_logs:
                from rsl_rl.utils import store_code_state

                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

            # ---- COPT: EA generation update ---------------------------------
            if ((it + 1 - start_iter) % self._ea_update_interval == 0) and (
                self._ea_late_start < (it + 1 - start_iter)
            ):
                if self._ea_late_start > 0 and self._design_generator.late_start:
                    print(
                        f"toggling late start at {it + 1 - start_iter} steps with configure threshold at {self._ea_late_start}"
                    )
                    self._design_generator.toggle_late_start()
                with torch.inference_mode():
                    self._reload_morphology()
                # Refresh observations after respawn
                obs = self.env.get_observations().to(self.device)
            # ---- end COPT injection -----------------------------------------

        if self.log_dir is not None and not self.disable_logs:
            self.save(
                os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
       # TODO Check if we can just run this method from the parent class by updating the configuration?
        # In such a case we would only have to update the logging and the rest of the pipeline from the parent class gets used.
        """Construct the actor-critic algorithm."""
        # Resolve RND config
        self.alg_cfg = resolve_rnd_config(
                self.alg_cfg, obs, self.cfg["obs_groups"], self.env
                )

        # Resolve symmetry config
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Resolve deprecated normalization config
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                    "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                    "`critic_obs_normalization` as part of the `policy` configuration instead.",
                    DeprecationWarning,
                    )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg[
                        "empirical_normalization"
                        ]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg[
                        "empirical_normalization"
                        ]

        # Initialize the policy
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))
        actor_critic: CoptActorCritic = actor_critic_class(
                obs,
                self.cfg["obs_groups"],
                self.env.num_actions,
                self.encoder_cfg,
                **self.policy_cfg,
                ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: PPO = alg_class(
                actor_critic,
                device=self.device,
                **self.alg_cfg,
                multi_gpu_cfg=self.multi_gpu_cfg,
                )

        # Initialize the storage
        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg

    # ------------------------------------------------------------------
    # COPT helpers
    # ------------------------------------------------------------------

    def _reload_morphology(self) -> None:
        """Run one EA cycle: evaluate fitness, generate new population, respawn."""
        if self.current_population is not None:
            fitness = self._compute_individual_fitness()
            print("Updating Design Population")
            self._design_generator.update_with_fitness(fitness)

        elif self.current_population is None:
            self._design_generator.sample_batch()

        # Generate new population
        print("Sampling New Population")
        self.current_population = self._design_generator.generate_population(
            self.generation
        )
        self.generation += 1

        # Respawn robots with new USD files
        unwrapped_env = self.env.unwrapped  # ManagerBasedRLEnv
        print("Spawning sampled population")
        self.respawn_robots(unwrapped_env, self.current_population.get_usd_files())

        # Patch actuator params
        print("Applying Sampled Actuator Parameters")
        apply_actuator_params(
            unwrapped_env, self.current_population.get_actuator_params()
        )

        # Reset fitness accumulators
        print("Resetting Accumulators")
        self._individual_fitness.zero_()
        self._individual_episode_counts.zero_()

    def _compute_individual_fitness(self) -> list[float]:
        """Return mean episode return per individual (0.0 if no episodes completed)."""
        fitness: list[float] = []
        for i in range(self._num_individuals):
            count = self._individual_episode_counts[i].item()
            if count > 0:
                fitness.append((self._individual_fitness[i] / count).item())
            else:
                fitness.append(0.0)
        return fitness

    def _assign_individuals_to_envs(self) -> list[int]:
        """Round-robin assignment: env_idx → individual_idx."""
        return [i % self._num_individuals for i in range(self.env.num_envs)]
