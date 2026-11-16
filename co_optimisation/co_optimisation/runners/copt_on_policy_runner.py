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

import contextlib
import itertools
import math
import os
import shutil
import time
import torch
import warnings
from collections import deque
from tensordict import TensorDict

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import pandas as pd
from isaaclab.sim.utils.stage import get_current_stage

from co_optimisation.algorithms import CoptPPO
from co_optimisation.modules import CoptActorCritic, CoptMorALActorCritic  # noqa: F401, eval below
from co_optimisation.runners.usd_generator import DesignGeneratorBase, Population
from co_optimisation.utils.analysis import log_prototype_and_instance_info
from co_optimisation.utils.env_state import capture_env_state, restore_env_state
from co_optimisation.utils.respawn import apply_actuator_params, respawn_robots
from co_optimisation.utils.update import (
    apply_link_length_params,
    apply_standing_height,
    assert_link_lengths_applied,
    refresh_rendered_geometry,
)
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner


def _population_to_dict(pop: Population) -> dict:
    """Serialise a :class:`Population` to plain Python lists for checkpointing.

    A population is fully described by its three parallel lists and the name of the
    emitted document's root body, so the snapshot is loss-less and rebuilds an
    identical population on resume.  The ``DesignSpec`` list is deliberately left
    out, being a convenience for logging that the generator can recompute.
    """
    return {
        "usd_files": pop.get_usd_files(),
        "actuator_params": pop.get_actuator_params(),
        "link_length_params": pop.get_link_length_params(),
        "root_body": pop.get_root_body(),
        "nominal": pop.get_nominal(),
    }


def _population_from_dict(state: dict) -> Population:
    """Reconstruct a :class:`Population` from :func:`_population_to_dict`.

    ``root_body`` defaults to the biped's when a checkpoint predates this field, that
    being the only family the URDF-era pipeline ever ran.
    """
    return Population(
        state["usd_files"],
        state["actuator_params"],
        state["link_length_params"],
        state.get("root_body", "base_Link"),
        nominal=state.get("nominal"),
    )


class CoptOnPolicyRunner(OnPolicyRunner):
    """On-policy runner with evolutionary design co-optimisation.

    Adds an outer EA loop on top of the standard PPO training loop.  Every
    ``ea_update_interval`` policy-update iterations the runner:

    1. Evaluates every design under one shared set of commands, terrain cells and initial states, see :meth:`_evaluate_population`.
    2. Calls ``design_generator.update_with_fitness()`` so the generator can
       improve future designs.
    3. Generates a new design population.
    4. Hot-swaps all robot articulations via the 10-step respawn sequence.
    5. Patches ``IdentifiedActuator`` tensor attributes for the new designs.
    6. Logs the generation, see :meth:`log_design_update`, and fetches fresh
       observations.

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
        # True reproduces every run to date.
        self._reinit_lr_each_generation: bool = copt_cfg.get(
            "reinit_lr_each_generation", True
        )
        self._randomise_before_late_start: bool = copt_cfg.get(
            "randomise_before_late_start", False
        )
        # Horizon of the controlled evaluation, 200 control steps at the default.
        self._eval_duration_s: float = copt_cfg.get("eval_duration_s", 4.0)
        self._eval_seed: int = copt_cfg.get("eval_seed", 0)
        # Writes designs/eval_trace_<g>.pt, the per step reward of every environment in every pass.
        self._eval_save_trace: bool = copt_cfg.get("eval_save_trace", False)
        self._eval_result: dict | None = None
        self._min_fitness_episode_length: int = copt_cfg.get(
            "min_fitness_episode_length", 0
        self._finetune_iterations: int | None = copt_cfg.get("finetune_iterations")
        # True from the generation boundary at which the search terminated, when the fine-tune begins.
        self._finetuning: bool = False
        # The last iteration of the fine-tune, set with _finetuning.
        self._finetune_stop_it: int | None = None
        self._cmaes_history: list[dict] = []
        self._design_generator = design_generator
        # Per-generation state
        self.generation: int = 0
        self.current_population: Population | None = None
        self.encoder_cfg = train_cfg["encoder"]
        self._copt_started = False

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        # Maps env_idx → individual_idx (round-robin)
        # The round robin is setup in co_optimisation.utils.respawn.respawn_robots
        self._env_to_individual: list[int] = self._assign_individuals_to_envs()
        self._eval_design: torch.Tensor = self._build_evaluation_design()
        # Every narrowed tracking width as configured, at which every generation is scored.
        self._eval_tracking_widths: dict[str, float] = self._configured_tracking_widths()
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
            "current_population": _population_to_dict(self.current_population),
            "design_generator": self._design_generator.get_state(),
            "cmaes_history": self._cmaes_history,
            "finetuning": self._finetuning,
            "finetune_stop_it": self._finetune_stop_it,
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
        # The fitness accumulators of earlier checkpoints are ignored, the fitness of
        # the restored population being measured afresh at its generation boundary.
        self._design_generator.load_state(c["design_generator"])
        self._cmaes_history = c.get("cmaes_history", [])
        self._finetuning = c.get("finetuning", False)
        self._finetune_stop_it = c.get("finetune_stop_it")

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
                # instancing prototype. This is done because the task configuration
                # spawns one asset, SOLEFOOT_IDENTIFIED_CFG, and therefore one
                # prototype, which the in-place editor cannot divide.
                self._reload_morphology()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

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
            late_start_toggled=False
            is_morph_update_time = (
                (it + 1) % self._ea_update_interval == 0
            ) and (is_late_start_toggle_time or self._randomise_before_late_start) and not self._finetuning
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

            if self.log_dir is not None and not self.disable_logs:
                boundary = (it + 1) % self.save_interval == 0
                # Before the evaluation and the swap, so that model_<it>.pt, design_parameters_<it>.csv
                # and cmaes_history row <it> all describe the population trained up to iteration it.
                if self.current_population is not None and (is_morph_update_time or (boundary and self._finetuning)):
                    pd.DataFrame(self.current_population.get_parameters()).to_csv(
                        os.path.join(self.log_dir, "designs", f"design_parameters_{it}.csv"),
                        index_label="individual",
                    )
                    self._log_population(it)
                if boundary:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

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
                    late_start_toggled = True
                with torch.inference_mode():
                    if self._copt_started and self.current_population is not None:
                        self._eval_result = self._evaluate_population()
                    self._update_morphology(it + 1, respawn=late_start_toggled)
                    if (self._finetune_iterations is not None and not self._finetuning
                            and getattr(self._design_generator, "_terminated", False)):
                        self._finetuning = True
                        self._finetune_stop_it = it + self._finetune_iterations
                        print(f"CMA-ES terminated at iteration {it}, fine-tuning on the final design "
                              f"until iteration {self._finetune_stop_it}")
                    self.env.episode_length_buf = torch.randint_like(
                        self.env.episode_length_buf, high=int(self.env.max_episode_length)
                    )
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
            # ---- end COPT injection -----------------------------------------

            if self.log_dir is not None and not self.disable_logs:
                # Log information
                self.log(locals())

            # Clear episode infos
            ep_infos.clear()

            if it == start_iter and not self.disable_logs:
                from rsl_rl.utils import store_code_state

                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

            if self._finetuning and it >= self._finetune_stop_it:
                print(f"Fine-tuning complete at iteration {it}")
                break

        if self.log_dir is not None and not self.disable_logs:
            self.save(
                os.path.join(
                    self.log_dir, f"model_{self.current_learning_iteration}.pt"
                )
            )
        # After the save, so the checkpoint holds the training state rather than the evaluation's.
        self.log_optimised_designs()

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
        actor_critic: CoptActorCritic | CoptMorALActorCritic = actor_critic_class(
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
        # spawn_multi_usd_file assigns designs in traversal order and fitness is
        # attributed by env_id % N, so the two agree only in numeric order.
        scene = unwrapped_env.scene
        matched = sim_utils.find_matching_prim_paths(scene.env_regex_ns)
        assert matched == list(scene.env_prim_paths), (
            "environment prims are traversed out of numeric order, so the design "
            f"assignment would not match the fitness attribution: {matched[:12]}"
        )
        unwrapped_env._suppress_terrain_curriculum = True
        unwrapped_env._suppress_curriculum = True
        try:
            self.respawn_robots(unwrapped_env, population.get_usd_files())
        finally:
            unwrapped_env._suppress_terrain_curriculum = False
            unwrapped_env._suppress_curriculum = False

    def _apply_nominal_pose(self, reset: bool) -> None:
        """Measure every environment's nominal pose, set its spawn height and publish the height.

        ``copt_nominal_base_height`` (num_envs,) is read by the nominal pose terms of
        mdp/rewards.py. *reset* re-places the robots where the caller's own reset has
        already run.
        """
        unwrapped_env = self.env.unwrapped
        family = self._design_generator.family
        standing = apply_standing_height(
            unwrapped_env,
            self.current_population.get_nominal(),
            family.foot_bodies,
            family.hip_bodies,
            family.pitch_joints,
        )
        unwrapped_env.copt_nominal_base_height = standing
        if reset:
            unwrapped_env._suppress_terrain_curriculum = True
            unwrapped_env._suppress_curriculum = True
            try:
                unwrapped_env.reset()
            finally:
                unwrapped_env._suppress_terrain_curriculum = False
                unwrapped_env._suppress_curriculum = False

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
            self._apply_nominal_pose(reset=True)

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
        advancing the design generator or incrementing the generation counter, both
        of which were already restored in :meth:`load`.  The usual
        ``unwrapped_env.reset()`` is replaced by
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
            unwrapped_env,
            self.current_population.get_link_length_params(),
            self.current_population.get_root_body(),
        )
        self._refresh_rendered_geometry()
        # restore_env_state's reset_to places the robots, so no reset here.
        self._apply_nominal_pose(reset=False)

        print("Restoring environment, curriculum, and RNG state")
        restore_env_state(self.env, self._restored_env_state)

        # The authoring API reports success whether or not physics ever reads the
        # attribute, so the realised body separations are read back out of the
        # simulation after the reset. See utils/update.py.
        assert_link_lengths_applied(
            unwrapped_env, self.current_population.get_link_length_params()
        )

        print("Applying checkpointed actuator parameters")
        apply_actuator_params(
            unwrapped_env, self.current_population.get_actuator_params()
        )

        # Consume the restore flags so a subsequent in-loop save/checkpoint does not
        # re-trigger the restore path.
        self._resumed_from_checkpoint = False
        self._restored_population = None
        self._restored_env_state = None

    def _refresh_rendered_geometry(self) -> None:
        """Redraw the in place designs, whose rendered geometry otherwise stays at the spawned one. See utils/update.py."""
        start = time.perf_counter()
        written = refresh_rendered_geometry(
            self.env.unwrapped,
            self.current_population.get_link_length_params(),
            self.current_population.get_root_body(),
        )
        if written:
            print(f"Refreshed {written} rendered geometry prims in {time.perf_counter() - start:.2f} s")

    def _update_morphology(self, it: int, respawn: bool = False) -> None:
        """In-place EA cycle: no delete/respawn, no manager re-binding.

        Args:
            it: Absolute iteration number used to gate the late-start transition.
            respawn: When True, apply the new population through the multi-USD
                :meth:`_respawn_population` pathway instead of the in-place
                :func:`apply_link_length_params` edit. Defaults to False, which
                reproduces the original in-place-only behaviour exactly. Must be
                set on any call where the new population's size differs from the
                size last given to :meth:`_respawn_population`, since the in-place
                edit only reaches as many environments as were given their own USD
                instancing prototype at that call, editing the first
                ``len(population)`` environments and relying on the rest sharing
                those same prototypes. A population-size change with the in-place
                path silently orphans every environment beyond the new, smaller
                size at whatever design it last had.
        """
        unwrapped_env = self.env.unwrapped
        sim = unwrapped_env.sim

        if sim.is_playing():  # 1. stop (Fabric off)
            sim._disable_app_control_on_stop_handle = True
            sim.stop()
            sim._disable_app_control_on_stop_handle = False

        evaluated = None
        if self._ea_late_start <= it:
            if not self._copt_started:
                self.alg.reinit_learning_rate()
                self._design_generator.sample_batch()
                self._copt_started = True
            elif self.current_population is not None:
                fitness = self._eval_result["fitness"].tolist()
                evaluated = self.current_population
                evaluated_generation = self.generation - 1
                print("Updating Design Population")
                update_start = time.time()
                self._design_generator.update_with_fitness(fitness)
                update_time = time.time() - update_start
        elif self._reinit_lr_each_generation:
            self.alg.reinit_learning_rate()
        population_start = time.time()
        previous_population = self.current_population
        self.current_population = self._design_generator.generate_population(
            self.generation
        )
        if self.current_population is None:
            # A terminated search proposes nothing, so the last population stays and is
            # re-applied below, which restarts the simulation stopped at step 1.
            self.current_population = previous_population
        self.generation += 1
        if self.current_population is not None:
            if respawn:
                print("Respawning population (prototype count is changing)")
                self._respawn_population(self.current_population)  # includes reset
                self._apply_nominal_pose(reset=True)
            else:
                print(
                    "Applying link length parameters in place"
                )  # 4. author + sim.reset() (inside)
                apply_link_length_params(
                    unwrapped_env,
                    self.current_population.get_link_length_params(),
                    self.current_population.get_root_body(),
                )
                self._refresh_rendered_geometry()

                # 4a. nominal pose and spawn height, before the reset places the robots
                self._apply_nominal_pose(reset=False)

                # 5. reset episodes — suppress terrain promotion/demotion because the
                # episodes are truncated by the morphology swap and the walked-distance
                # signal is meaningless at this boundary, and every other curriculum
                # term, which would read the same truncated sums (mdp/curriculums.py).
                unwrapped_env._suppress_terrain_curriculum = True
                unwrapped_env._suppress_curriculum = True
                unwrapped_env.reset()
                unwrapped_env._suppress_terrain_curriculum = False
                unwrapped_env._suppress_curriculum = False

                # 5a. read the realised separations back out of the simulation, the
                # only check that distinguishes an edit that was authored from an
                # edit that took effect. See utils/update.py.
                assert_link_lengths_applied(
                    unwrapped_env, self.current_population.get_link_length_params()
                )

            print("Applying Sampled Actuator Parameters")  # 6. actuators AFTER reset
            apply_actuator_params(
                unwrapped_env, self.current_population.get_actuator_params()
            )

        if evaluated is not None:
            population_time = time.time() - population_start
            self.log_design_update(locals())
        self._eval_result = None
    def _build_evaluation_design(self) -> torch.Tensor:
        """A Latin hypercube on the unit cube of (vx, vy, wz, yaw), one point per cell an evaluation visits.

        Returns (R, P, m, 4), point (r, q, j) the command and initial yaw of instance j in difficulty
        row r and column pass q, fixed for the run and mapped onto the command ranges at each
        evaluation. R and P are 1 on a plane.
        """
        m = self.env.num_envs // self._num_individuals
        origins = self.env.unwrapped.scene.terrain.terrain_origins
        rows, passes = (origins.shape[0], -(-origins.shape[1] // m)) if origins is not None else (1, 1)
        count = rows * passes * m
        generator = torch.Generator().manual_seed(self._eval_seed)
        axes = [
            (torch.randperm(count, generator=generator) + torch.rand(count, generator=generator)) / count
            for _ in range(4)
        ]
        return torch.stack(axes, dim=1).view(rows, passes, m, 4).to(self.env.device)

    def _evaluation_columns(self, cols: int) -> torch.Tensor:
        """Terrain column of each instance in each column pass, every column visited in every row.

        Returns (P, m) for P = ceil(cols / m) passes. Slot j * P + q is spread evenly over the
        columns, so the instances of one pass stand in distinct columns whenever m <= cols, every
        column is visited at least once, and the columns visited twice, when m does not divide
        cols, are spread evenly across the grid and so across the sub-terrains.
        """
        m = self.env.num_envs // self._num_individuals
        passes = -(-cols // m)
        q = torch.arange(passes, device=self.env.device).unsqueeze(1)
        j = torch.arange(m, device=self.env.device).unsqueeze(0)
        return (j * passes + q) * cols // (passes * m)

    @contextlib.contextmanager
    def _video_paused(self):
        """Pause every wrapper of the chain that carries a paused flag, for the duration of the block.

        Gymnasium 1.2.1 wrappers do not forward attribute lookups, so hasattr finds the flag on the
        recorder alone, PausableRecordVideo of scripts/rsl_rl/highest_terrain_camera_wrapper.py, and
        the chain is walked through the env attribute each wrapper and RslRlVecEnvWrapper hold.
        """
        recorders, layer, seen = [], self.env, set()
        while layer is not None and id(layer) not in seen:
            seen.add(id(layer))
            if hasattr(layer, "paused"):
                recorders.append(layer)
            layer = getattr(layer, "env", None)
        for recorder in recorders:
            recorder.paused = True
        try:
            yield
        finally:
            for recorder in recorders:
                recorder.paused = False

    def _evaluate_population(self) -> dict:
        """Score every design on one shared set of terrain cells, commands and initial states.

        Instance j of every design, environment j * N + k for design k, stands in column
        col[q, j] in column pass q, and every column pass is made in every difficulty row, so
        each cell of a pass holds one instance of every design and every design meets every cell
        of the grid at least once. Each visited cell carries its own command and initial yaw.
        Returns per design (N,) the fitness, the mean return over the R x P x m visited cells,
        the survival fraction and the mean linear and yaw rate tracking errors, and in addition
        the split-half rank correlation of the fitness over the even and odd columns, per row the
        mean return and survival, and the wall time. Every narrowed tracking kernel is held at its configured
        width. The widths, the terrain assignment, the step counter and the interval event timers
        are restored, and the reset of the design swap that follows overwrites everything else.
        """
        env = self.env.unwrapped
        robot = env.scene["robot"]
        terrain = env.scene.terrain
        command = env.command_manager.get_term("base_velocity")
        n = self._num_individuals
        m = env.num_envs // n
        rows, passes = self._eval_design.shape[:2]
        replica = torch.arange(env.num_envs, device=env.device) // n
        steps = int(round(self._eval_duration_s / env.step_dt))
        start = time.time()

        common_step_counter = env.common_step_counter
        interval_time_left = [t.clone() for t in env.event_manager._interval_term_time_left]
        grid = terrain.terrain_origins is not None
        if grid:
            assignment = (
                terrain.terrain_levels.clone(),
                terrain.terrain_types.clone(),
                terrain.env_origins.clone(),
            )
            col = self._evaluation_columns(terrain.terrain_origins.shape[1])
        else:
            col = torch.arange(m, device=env.device).unsqueeze(0)

        ranges = command.cfg.ranges
        lo = torch.tensor(
            [ranges.lin_vel_x[0], ranges.lin_vel_y[0], ranges.ang_vel_z[0], -math.pi], device=env.device
        )
        hi = torch.tensor(
            [ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_z[1], math.pi], device=env.device
        )
        design = lo + (hi - lo) * self._eval_design  # (R, P, m, 4)
        ret = torch.zeros(rows, passes, env.num_envs, device=self.device)
        survived = torch.zeros_like(ret)
        length = torch.zeros_like(ret)
        lin_err = torch.zeros_like(ret)
        yaw_err = torch.zeros_like(ret)
        trace = (
            torch.zeros(rows, passes, steps, env.num_envs, dtype=torch.float16, device=self.device)
            if self._eval_save_trace
            else None
        )
        widths = self._fix_tracking_widths(env)
        env._suppress_terrain_curriculum = True
        env._suppress_curriculum = True
        self.eval_mode()
        try:
            with self._video_paused():
                for r, q in itertools.product(range(rows), range(passes)):
                    if grid:
                        terrain.terrain_levels[:] = r
                        terrain.terrain_types[:] = col[q][replica]
                        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
                    condition = design[r, q][replica]
                    env.reset()
                    # The reset events draw a planar offset, a yaw, a velocity and joint offsets
                    # per environment, on which the designs would otherwise be ranked.
                    root = robot.data.default_root_state.clone()
                    root[:, :3] += env.scene.env_origins
                    zero = torch.zeros_like(condition[:, 3])
                    root[:, 3:7] = math_utils.quat_from_euler_xyz(zero, zero, condition[:, 3])
                    root[:, 7:] = 0.0
                    robot.write_root_state_to_sim(root)
                    robot.write_joint_state_to_sim(
                        robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
                    )
                    command.vel_command_b[:] = condition[:, :3]
                    command.is_heading_env[:] = False
                    command.is_standing_env[:] = False
                    command.time_left[:] = float("inf")
                    for t in env.event_manager._interval_term_time_left:
                        t[:] = float("inf")
                    env.observation_manager.reset()
                    obs = TensorDict(
                        env.observation_manager.compute(update_history=True), batch_size=[env.num_envs]
                    ).to(self.device)

                    target = condition[:, :3].to(self.device)
                    alive = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
                    for t in range(steps):
                        actions = self.alg.policy.act_inference(obs)
                        obs, rewards, dones, _ = self.env.step(actions.to(self.env.device))
                        obs, rewards = obs.to(self.device), rewards.to(self.device)
                        dones = dones.to(self.device).bool()
                        # step resets a terminated environment, so its state no longer belongs to the episode.
                        counted = alive & ~dones
                        lin_err[r, q] += counted * torch.norm(
                            target[:, :2] - robot.data.root_lin_vel_b[:, :2].to(self.device), dim=1
                        )
                        yaw_err[r, q] += counted * torch.abs(
                            target[:, 2] - robot.data.root_ang_vel_b[:, 2].to(self.device)
                        )
                        ret[r, q] += alive * rewards
                        if trace is not None:
                            trace[r, q, t] = alive * rewards
                        length[r, q] += counted
                        alive &= ~dones
                    survived[r, q] = alive.float()
        finally:
            for term, std in widths:
                term.params["std"] = std
            env._suppress_terrain_curriculum = False
            env._suppress_curriculum = False
            env.common_step_counter = common_step_counter
            for t, saved in zip(env.event_manager._interval_term_time_left, interval_time_left):
                t.copy_(saved)
            if grid:
                for t, saved in zip(
                    (terrain.terrain_levels, terrain.terrain_types, terrain.env_origins), assignment
                ):
                    t.copy_(saved)
            self.train_mode()

        shape = (rows, passes, m, n)
        ret, survived = ret.view(shape), survived.view(shape)
        steps_alive = length.view(shape).sum((0, 1, 2)).clamp(min=1)

        # The even and the odd columns, so the halves share no terrain sample and each holds every
        # row and every sub-terrain in equal share.
        even = (col % 2 == 0).to(self.device)  # (P, m)
        split = float("nan")
        if even.any() and (~even).any():
            halves = (ret[:, even].mean((0, 1)), ret[:, ~even].mean((0, 1)))
            ranks = torch.stack([h.argsort().argsort().float() for h in halves])
            split = torch.corrcoef(ranks)[0, 1].item()
        if trace is not None and self.log_dir is not None:
            torch.save(
                {"reward": trace.cpu(), "column": col.cpu(), "design": design.cpu(), "num_designs": n,
                 "step_dt": env.step_dt},
                os.path.join(self.log_dir, "designs", f"eval_trace_{self.generation:04d}.pt"),
            )
        return {
            "fitness": ret.mean((0, 1, 2)),
            "survival": survived.mean((0, 1, 2)),
            "lin_vel_error": lin_err.view(shape).sum((0, 1, 2)) / steps_alive,
            "yaw_rate_error": yaw_err.view(shape).sum((0, 1, 2)) / steps_alive,
            "split_half_rank_correlation": split,
            "row_return": {r: ret[r].mean().item() for r in range(rows)},
            "row_survival": {r: survived[r].mean().item() for r in range(rows)},
            "time": time.time() - start,
        }

    def log_design_update(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Write one CMA-ES generation to the CMAES head, the design logs and the terminal.

        Called by _update_morphology after update_with_fitness, with its locals, so the
        fitness is that of the evaluated population and the statistics those of the
        distribution that sampled the next.
        """
        it = self.current_learning_iteration
        fitness = torch.tensor(locs["fitness"], dtype=torch.float)
        evaluation = self._eval_result
        parameters = locs["evaluated"].get_parameters()
        if len(parameters) != len(fitness):
            # A population restored from a checkpoint that predates the parameter record.
            parameters = [{} for _ in range(len(fitness))]

        record = {
            "iteration": it,
            "generation": locs["evaluated_generation"],
            "fitness_mean": fitness.mean().item(),
            "fitness_std": fitness.std().item(),
            "fitness_max": fitness.max().item(),
            "fitness_min": fitness.min().item(),
            "fitness_median": fitness.median().item(),
            "split_half_rank_correlation": evaluation["split_half_rank_correlation"],
            "infeasible": sum(not p.get("feasible", True) for p in parameters),
        }
        for key in ("survival", "lin_vel_error", "yaw_rate_error"):
            record[f"eval/{key}"] = evaluation[key].mean().item()
        for r, value in evaluation["row_return"].items():
            record[f"eval/return_row_{r}"] = value
        for r, value in evaluation["row_survival"].items():
            record[f"eval/survival_row_{r}"] = value
        for key, value in (self._design_generator.search_statistics() or {}).items():
            if isinstance(value, dict):
                record.update({f"{key}/{name}": v for name, v in value.items()})
            else:
                record[key] = value
        best = parameters[int(fitness.argmax())]
        record.update({
            f"best/{name}": float(v)
            for name, v in best.items()
            if isinstance(v, float)
        })
        # The catalogue name of the best individual's actuator on every group, for cmaes_history.csv.
        record.update({
            f"best/{name}": v
            for name, v in best.items()
            if name.endswith("_actuator") and self._is_actuator_column(name)
        })
        # The same actuator's catalogue index, the "index" of catalogs/actuators.json that pick_index/<group> reports
        # for the distribution's mode, left out where the catalogue states none, as the quadruped's.
        record.update({
            f"best/{name}": int(v)
            for name, v in best.items()
            if name.endswith("_actuator_index") and self._is_actuator_column(name) and v is not None and v == v
        })
        record["eval_time"] = evaluation["time"]
        record["update_time"] = locs["update_time"]
        record["population_time"] = locs["population_time"]
        self._cmaes_history.append(record)

        if self.log_dir is not None and not self.disable_logs:
            for key, value in record.items():
                if key.startswith("pick_index/") or (
                    key.startswith("best/") and self._is_actuator_column(key[len("best/"):])
                ):
                    continue
                if key != "iteration" and isinstance(value, (int, float)):
                    # The distribution mean, CMA-ES's estimate of the optimum, under the searched name itself.
                    tag = key[len("mean/"):] if key.startswith("mean/") else key
                    self.writer.add_scalar(f"CMAES/{tag}", value, it)
            # The actuator progression of each optimised group, one table per group in the TEXT tab, whose
            # slider steps through the history as it stood at each generation.
            family = getattr(self._design_generator, "family", None)
            for group in getattr(family, "groups", ()):
                self.writer.add_text(f"CMAES/actuators/{group}", _actuator_table(self._cmaes_history, group), it)
            designs = os.path.join(self.log_dir, "designs")
            pd.DataFrame(self._cmaes_history).to_csv(
                os.path.join(designs, "cmaes_history.csv"), index=False
            )
            table = pd.DataFrame(parameters)
            table["fitness"] = fitness.tolist()
            for key in ("survival", "lin_vel_error", "yaw_rate_error"):
                table[f"eval_{key}"] = evaluation[key].tolist()
            table.to_csv(
                os.path.join(designs, f"cmaes_generation_{record['generation']:04d}.csv"),
                index_label="individual",
            )

        title = f" \033[1m CMA-ES generation {record['generation']} at learning iteration {it} \033[0m "
        log_string = f"""{"#" * width}\n{title.center(width, " ")}\n\n"""
        for key, value in record.items():
            if key in ("iteration", "generation"):
                continue
            text = f"{value:.4f}" if isinstance(value, float) else f"{value}"
            log_string += f"""{f"{key}:":>{pad}} {text}\n"""
        log_string += f"""{"-" * width}\n"""
        print(log_string, flush=True)
            else:
                fitness.append(0.0)
        return fitness
    def log_optimised_designs(self, width: int = 80, pad: int = 35) -> None:
        """Evaluate the population on the scene once more and write the final design to designs/optimised.

        The final design is the one the search recommends, optimised_design(), the distribution
        mean, and it is the design every later evaluation of the run uses. It heads
        final_results.txt, it is the single row of final_design.csv, and its MJCF is
        distribution_mean.xml. After a fine-tune every individual slot carries it, so the
        evaluation of the population on the scene is its evaluation under the fine-tuned policy,
        averaged over every environment. Without a fine-tune the population on the scene is the
        last generation sampled, which the evaluation ranks, and the final design is recorded
        unevaluated. optimised_designs.csv and individual_<k>.xml record the population on the
        scene in either case, individual k having run in environments k + N * j.
        """
        if not self._copt_started or self.current_population is None:
            return
        population = self.current_population
        with torch.inference_mode():
            evaluation = self._evaluate_population()
        fitness = evaluation["fitness"].cpu()
        order = fitness.argsort(descending=True).tolist()
        best = order[0]
        parameters = population.get_parameters()
        if len(parameters) != len(fitness):
            parameters = [{} for _ in range(len(fitness))]
        write = self.log_dir is not None and not self.disable_logs
        out = os.path.join(self.log_dir, "designs", "optimised") if write else None
        if write:
            os.makedirs(out, exist_ok=True)

        final = {}
        optimised = self._design_generator.optimised_design()
        if optimised is not None:
            values, picks = optimised
            feasible = (self._design_generator.emit_design(values, picks, os.path.join(out, "distribution_mean.xml"))
                        if write else None)
            # The terminating generation is the last history row once the fine-tune has begun.
            terminal = self._cmaes_history[-1] if self._finetuning and self._cmaes_history else {}
            on_scene = self._finetuning
            final = {
                "source": "distribution_mean",
                **{name: float(v) for name, v in values.items()},
                **(picks or {}),
                **self._design_generator.describe_design(values, picks),
                "feasible": feasible,
                "fine_tuned": on_scene,
                "fitness": fitness.mean().item() if on_scene else None,
                "fitness_std_over_slots": fitness.std().item() if on_scene else None,
                **{f"eval_{key}": (evaluation[key].mean().item() if on_scene else None)
                   for key in ("survival", "lin_vel_error", "yaw_rate_error")},
                "termination_iteration": terminal.get("iteration"),
                "termination_generation": terminal.get("generation"),
                "stop": terminal.get("stop"),
                "finetune_start_iteration": (self._finetune_stop_it - self._finetune_iterations
                                             if on_scene else None),
                "checkpoint_iteration": self.current_learning_iteration,
            }

        rows = [(f"final design/{key}", value) for key, value in final.items()]
        rows += [
            ("generation", self.generation - 1),
            ("learning iteration", self.current_learning_iteration),
            ("best individual", best),
            ("best fitness", fitness[best].item()),
        ]
        rows += [(f"best/{name}", value) for name, value in parameters[best].items()]
        rows += [
            ("fitness_mean", fitness.mean().item()),
            ("fitness_std", fitness.std().item()),
            ("split_half_rank_correlation", evaluation["split_half_rank_correlation"]),
        ]
        for key in ("survival", "lin_vel_error", "yaw_rate_error"):
            rows.append((f"eval/{key}", evaluation[key].mean().item()))
        for r, value in evaluation["row_return"].items():
            rows.append((f"eval/return_row_{r}", value))
        for key, value in (self._design_generator.search_statistics() or {}).items():
            if isinstance(value, dict):
                rows += [(f"{key}/{name}", v) for name, v in value.items()]
            else:
                rows.append((key, value))
        rows.append(("evaluation time", evaluation["time"]))

        title = " Optimised population "
        text = f"""{"#" * width}\n{title.center(width, " ")}\n\n"""
        for key, value in rows:
            shown = f"{value:.4f}" if isinstance(value, float) else f"{value}"
            text += f"""{f"{key}:":>{pad}} {shown}\n"""
        text += f"""\n{"Ranking, individual and fitness":>{pad}}\n"""
        for k in order:
            text += f"""{f"{k}:":>{pad}} {fitness[k].item():.4f}\n"""
        text += f"""{"-" * width}\n"""
        print(text, flush=True)
        if not write:
            return

        for k, path in enumerate(population.get_xml_files()):
            shutil.copy(path, os.path.join(out, f"individual_{k:04d}.xml"))
        table = pd.DataFrame(parameters)
        table["fitness"] = fitness.tolist()
        for key in ("survival", "lin_vel_error", "yaw_rate_error"):
            table[f"eval_{key}"] = evaluation[key].tolist()
        table["rank"] = fitness.argsort(descending=True).argsort().tolist()
        table.to_csv(os.path.join(out, "optimised_designs.csv"), index_label="individual")
        if final:
            pd.DataFrame([final]).to_csv(os.path.join(out, "final_design.csv"), index=False)
        with open(os.path.join(out, "final_results.txt"), "w") as fh:
            fh.write(text)

    def _assign_individuals_to_envs(self) -> list[int]:
        """Round-robin assignment: env_idx → individual_idx."""
        return [i % self._num_individuals for i in range(self.env.num_envs)]
