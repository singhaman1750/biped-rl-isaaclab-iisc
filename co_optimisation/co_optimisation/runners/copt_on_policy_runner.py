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

import pandas as pd
from isaaclab.sim.utils.stage import get_current_stage
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner

from co_optimisation.algorithms import CoptLearnedModelPPO, CoptPPO
from co_optimisation.modules import CoptActorCritic, CoptLearnedModelActorCritic
from co_optimisation.runners.usd_generator import (
    DesignGeneratorBase,
    Population,
    RandomPopulation,
)
from co_optimisation.utils.analysis import log_prototype_and_instance_info
from co_optimisation.utils.env_state import capture_env_state, restore_env_state
from co_optimisation.utils.respawn import apply_actuator_params, respawn_robots
from co_optimisation.utils.update import apply_link_length_params


def _population_to_dict(pop: Population) -> dict:
    """Serialise a :class:`Population` to plain Python lists for checkpointing.

    A :class:`RandomPopulation` is fully described by three parallel lists, so the
    snapshot is loss-less and rebuilds an identical population on resume.
    """
    return {
        "usd_files": pop.get_usd_files(),
        "actuator_params": pop.get_actuator_params(),
        "link_length_params": pop.get_link_length_params(),
    }


def _population_from_dict(state: dict) -> RandomPopulation:
    """Reconstruct a :class:`RandomPopulation` from :func:`_population_to_dict`."""
    return RandomPopulation(
        state["usd_files"],
        state["actuator_params"],
        state["link_length_params"],
    )


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
        self._randomise_before_late_start: bool = copt_cfg.get(
            "randomise_before_late_start", False
        )
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
        self.decoder_cfg: dict | None = train_cfg.get("decoder", None)
        self._copt_started = False

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        # Maps env_idx → individual_idx (round-robin)
        # The round robin is setup in co_optimisation.utils.respawn.respawn_robots
        self._env_to_individual: list[int] = self._assign_individuals_to_envs()
        # Set by ``load`` when a checkpoint carries the extended COPT state, and
        # consumed once by ``learn`` to apply the restored population and env state.
        self._resumed_from_checkpoint: bool = False
        self._restored_population: Population | None = None
        self._restored_env_state: dict | None = None

    # ------------------------------------------------------------------
    # Checkpoint persistence
    # ------------------------------------------------------------------

    def save(self, path: str, infos: dict | None = None) -> None:
        """Persist policy weights plus the full co-optimisation and env state.

        The base implementation is called first so the policy, optimiser, and
        iteration counter are written exactly as before, preserving compatibility
        with the inherited loader and with non-COPT tooling.  The saved file is then
        reopened and augmented with the residual PPO state (the adaptive learning
        rate and the logging totals), the evolutionary state (generation, population,
        fitness accumulators, and the CMA-ES search distribution), and the IsaacLab
        environment state (curriculum, command, reward, counters, and RNG).
        """
        super().save(path, infos)
        saved = torch.load(path, weights_only=False)
        saved["learning_rate"] = self.alg.learning_rate
        saved["tot_timesteps"] = self.tot_timesteps
        saved["tot_time"] = self.tot_time
        saved["copt"] = {
            "generation": self.generation,
            "copt_started": self._copt_started,
            "individual_fitness": self._individual_fitness.detach().cpu().clone(),
            "individual_episode_counts": self._individual_episode_counts.detach()
            .cpu()
            .clone(),
            "current_population": _population_to_dict(self.current_population),
            "design_generator": self._design_generator.get_state(),
        }
        saved["env_state"] = capture_env_state(self.env)
        torch.save(saved, path)

        # Re-upload the augmented checkpoint to any external logger, mirroring the
        # base behaviour which uploads the policy-only file.
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Restore policy weights plus the full co-optimisation and env state.

        The base loader restores the policy, optimiser, and iteration counter.  When
        the checkpoint additionally carries the extended COPT state this method
        restores the adaptive learning rate (and writes it onto every optimiser
        parameter group so the first adaptive update does not silently reset it), the
        logging totals, the evolutionary state, and the CMA-ES search distribution,
        and it stashes the population and env state for ``learn`` to apply once the
        simulation is live.  Legacy policy-only checkpoints load unchanged.
        """
        infos = super().load(path, load_optimizer, map_location)
        loaded = torch.load(path, weights_only=False, map_location=map_location)

        if "copt" not in loaded:
            # Legacy or non-COPT checkpoint: nothing further to restore.
            return infos

        self.alg.learning_rate = loaded["learning_rate"]
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        self.tot_timesteps = loaded["tot_timesteps"]
        self.tot_time = loaded["tot_time"]

        c = loaded["copt"]
        self.generation = c["generation"]
        self._copt_started = c["copt_started"]
        self._individual_fitness.copy_(c["individual_fitness"].to(self.device))
        self._individual_episode_counts.copy_(
            c["individual_episode_counts"].to(self.device)
        )
        self._design_generator.load_state(c["design_generator"])

        self._restored_population = _population_from_dict(c["current_population"])
        self._restored_env_state = loaded["env_state"]
        self._resumed_from_checkpoint = True
        return infos

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
        # print("[respawn] Logging prototype and instance info...")
        # log_prototype_and_instance_info(get_current_stage())
        self.respawn_robots = respawn_robots()
        if self.log_dir is not None:
            if not os.path.exists(os.path.join(self.log_dir, "designs")):
                os.mkdir(os.path.join(self.log_dir, "designs"))
        with torch.inference_mode():
            if getattr(self, "_resumed_from_checkpoint", False):
                # Resume: re-spawn the checkpointed population and restore the env,
                # manager, curriculum, and RNG state instead of generating a fresh
                # generation-zero population.
                assert self._restored_population is not None, (
                    "resume checkpoint carries no design population, so the run "
                    "cannot re-establish one USD prototype per design and every "
                    "environment would silently share a single morphology"
                )
                self._respawn_population(self._restored_population)
                self._apply_restored_state()
            else:
                # Bootstrap generation zero through the respawn path, so that each
                # design receives its own USD reference arc and therefore its own
                # instancing prototype. This is done as SOLEFOOT_IDENTIFIED_CFG_URDF
                # is used to spawn the base environment which contains only a single 
                # prototype
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
            # Schedule against the ABSOLUTE iteration ``it`` rather than the
            # resume-relative ``(it - start_iter + 1)`` so that a run resumed at,
            # for example, iteration eight thousand enters the CMA-ES phase
            # immediately instead of repeating the random phase. On a fresh run
            # ``start_iter == 0`` so the two forms coincide.
            is_late_start_toggle_time = self._ea_late_start <= (it + 1)
            is_morph_update_time = (
                (it + 1) % self._ea_update_interval == 0
            ) and (is_late_start_toggle_time or self._randomise_before_late_start)
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
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
                        if (step < self.num_steps_per_env - 1) or (
                            not is_morph_update_time
                        ):
                            # ---- COPT: accumulate per-individual fitness --------
                            completed_env_ids = new_ids[:, 0]
                            for env_idx in completed_env_ids.tolist():
                                ind_idx = self._env_to_individual[env_idx]
                                self._individual_fitness[ind_idx] += cur_reward_sum[
                                    env_idx
                                ]
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
                                    cur_ereward_sum[new_ids][:, 0]
                                    .cpu()
                                    .numpy()
                                    .tolist()
                                )
                                irewbuffer.extend(
                                    cur_ireward_sum[new_ids][:, 0]
                                    .cpu()
                                    .numpy()
                                    .tolist()
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

            # ---- COPT: EA generation update ---------------------------------
            if is_morph_update_time:
                if (
                    self._ea_late_start > 0
                    and self._design_generator.late_start
                    and is_late_start_toggle_time
                ):
                    print(
                        f"toggling late start at absolute iteration {it + 1} with configured threshold at {self._ea_late_start} iterations"
                    )
                    self._design_generator.toggle_late_start()
                with torch.inference_mode():
                    self._update_morphology(it + 1)
                # Refresh observations after in-place update
                obs = self.env.get_observations().to(self.device)
                if self.log_dir is not None:
                    # Discard partial (truncated) returns so that rewbuffer contains
                    # only genuinely completed episodes from the normal done path and
                    # the logged Train/mean_reward is not corrupted by mid-episode sums.
                    cur_reward_sum[:] = 0
                    cur_episode_length[:] = 0

                    if self.alg.rnd:
                        cur_ereward_sum[:] = 0
                        cur_ireward_sum[:] = 0
                    link_lengths = self.current_population.get_link_length_params()
                    keys = None
                    if len(link_lengths) > 0:
                        keys = link_lengths[0].keys()
                    df = pd.DataFrame(self.current_population.get_link_length_params())
                    df.to_csv(
                        os.path.join(self.log_dir, "designs", f"link_lengths_{it}.csv"),
                        header=False,
                        columns=keys,
                    )
            # ---- end COPT injection -----------------------------------------

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
        if self.decoder_cfg is not None:
            actor_critic: CoptActorCritic = actor_critic_class(
                obs,
                self.cfg["obs_groups"],
                self.env.num_actions,
                self.encoder_cfg,
                self.decoder_cfg,
                **self.policy_cfg,
            ).to(self.device)
        else:
            actor_critic: CoptActorCritic = actor_critic_class(
                obs,
                self.cfg["obs_groups"],
                self.env.num_actions,
                self.encoder_cfg,
                **self.policy_cfg,
            ).to(self.device)

        # Initialize the algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        alg: CoptPPO = alg_class(
            actor_critic,
            self._num_individuals,
            [i % self._num_individuals for i in range(self.env.num_envs)],
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

    def _respawn_population(self, population: Population) -> None:
        """Spawn one USD per design, with the terrain curriculum suppressed.

        This is the only pathway that gives each design its own USD reference arc and
        therefore its own USD instancing prototype.  ``apply_link_length_params`` edits
        prototypes rather than composed prims, so where a single asset backs every
        environment the whole population resolves to one prototype and the per-design
        writes overwrite one another, leaving every environment on the design authored
        last.  Establishing the prototypes here is what makes the in-place path correct
        for every generation that follows.

        The terrain curriculum is suppressed across the reset that ``respawn_robots``
        performs at its step 11, because the episodes are truncated by the spawn and
        the walked-distance signal ``terrain_levels_vel`` reads is meaningless at that
        boundary.  This mirrors the guard :meth:`_update_morphology` already applies
        around its own reset, and is applied here rather than inside ``respawn_robots``
        so that the behaviour of that shared utility is unchanged for any other caller.

        Args:
            population: The population whose ``get_usd_files()`` supplies one USD path
                per individual.  Environments are assigned round-robin by
                ``env_idx % num_individuals``.
        """
        unwrapped_env = self.env.unwrapped
        unwrapped_env._suppress_terrain_curriculum = True
        try:
            self.respawn_robots(unwrapped_env, population.get_usd_files())
        finally:
            unwrapped_env._suppress_terrain_curriculum = False

    def _reload_morphology(self) -> None:
        """Run one EA cycle: evaluate fitness, generate new population, respawn."""
        if self.current_population is not None :
            fitness = self._compute_individual_fitness()
            print("Updating Design Population")
            self._design_generator.update_with_fitness(fitness)

        elif self.current_population is None and self._ea_late_start < 0:
            self._design_generator.sample_batch()

        # Generate new population
        print("Sampling New Population")
        self.current_population = self._design_generator.generate_population(
            self.generation
        )
        self.generation += 1
        if self.current_population is not None:
            # Respawn robots with new USD files
            unwrapped_env = self.env.unwrapped  # ManagerBasedRLEnv
            print("Spawning sampled population")
            self._respawn_population(self.current_population)

            # Patch actuator params
            print("Applying Sampled Actuator Parameters")
            apply_actuator_params(
                unwrapped_env, self.current_population.get_actuator_params()
            )

            # Reset fitness accumulators
            print("Resetting Accumulators")
            self._individual_fitness.zero_()
            self._individual_episode_counts.zero_()

    def _apply_restored_state(self) -> None:
        """Re-spawn the checkpointed population and restore env state on resume.

        Mirrors the in-place sequence of :meth:`_update_morphology` but without
        advancing the design generator, incrementing the generation counter, or
        zeroing the fitness accumulators, all of which were already restored in
        :meth:`load`.  The usual ``unwrapped_env.reset()`` is replaced by
        :func:`restore_env_state`, which runs ``reset_to`` internally so the robots
        are placed at their checkpointed poses and the curriculum, command, reward,
        counter, and RNG state are written back.
        """
        unwrapped_env = self.env.unwrapped
        sim = unwrapped_env.sim
        if sim.is_playing():
            sim._disable_app_control_on_stop_handle = True
            sim.stop()
            sim._disable_app_control_on_stop_handle = False

        self.current_population = self._restored_population
        print("Re-applying checkpointed morphology population")
        apply_link_length_params(
            unwrapped_env, self.current_population.get_link_length_params()
        )

        print("Restoring environment, curriculum, and RNG state")
        restore_env_state(self.env, self._restored_env_state)

        print("Applying checkpointed actuator parameters")
        apply_actuator_params(
            unwrapped_env, self.current_population.get_actuator_params()
        )

        # Consume the restore flags so a subsequent in-loop save/checkpoint does not
        # re-trigger the restore path.
        self._resumed_from_checkpoint = False
        self._restored_population = None
        self._restored_env_state = None

    def _update_morphology(self, it: int) -> None:
        """In-place EA cycle: no delete/respawn, no manager re-binding."""
        unwrapped_env = self.env.unwrapped
        sim = unwrapped_env.sim

        if sim.is_playing():  # 1. stop (Fabric off)
            sim._disable_app_control_on_stop_handle = True
            sim.stop()
            sim._disable_app_control_on_stop_handle = False

        if self._ea_late_start <= it:
            if not self._copt_started:
                self.alg.reinit_learning_rate()
                self._design_generator.sample_batch()
                self._copt_started = True
            else:
                fitness = self._compute_individual_fitness()
                print("Updating Design Population")
                self._design_generator.update_with_fitness(fitness)
        else:
            self.alg.reinit_learning_rate()
        self.current_population = self._design_generator.generate_population(
            self.generation
        )
        self.generation += 1
        if self.current_population is not None:
            print(
                "Applying link length parameters in place"
            )  # 4. author + sim.reset() (inside)
            apply_link_length_params(
                unwrapped_env, self.current_population.get_link_length_params()
            )

            # 5. reset episodes — suppress terrain promotion/demotion because the
            # episodes are truncated by the morphology swap and the walked-distance
            # signal is meaningless at this boundary.
            unwrapped_env._suppress_terrain_curriculum = True
            unwrapped_env.reset()
            unwrapped_env._suppress_terrain_curriculum = False

            print("Applying Sampled Actuator Parameters")  # 6. actuators AFTER reset
            apply_actuator_params(
                unwrapped_env, self.current_population.get_actuator_params()
            )

            self._individual_fitness.zero_()  # 7. zero accumulators
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
