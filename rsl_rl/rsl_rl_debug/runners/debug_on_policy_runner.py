from __future__ import annotations

import os
import time
import torch
import warnings
from collections import deque

from rsl_rl_debug.modules import DebugActorCritic

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import store_code_state


class DebugOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner with per-phase timing instrumentation.

    The vendored runner times two aggregates per iteration, collection_time
    for the whole rollout and learn_time for compute_returns plus update
    together. This subclass splits collection_time into its three
    constituent per-step calls, act, env.step, and process_env_step, and
    splits learn_time into compute_returns and update, so a training run can
    be compared against another to see which phase, and within collection
    which call, accounts for a throughput difference. See
    context/go1_quadruped_throughput.md for the investigation this was built
    for.

    The timings are plain wall clock deltas around each call with no
    torch.cuda.synchronize between them. Isaac Lab's own env.step blocks on
    the simulation before returning, and these three calls execute strictly
    sequentially on a single host thread, so the boundaries are accurate
    for the work that matters here; only the asynchronous tail of a CUDA
    kernel queued by one call and still running when the next call's own
    kernels are queued could bleed across a boundary uncounted, which this
    trades away in exchange for adding no synchronization stalls of its own.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(env, train_cfg, log_dir, device)
        self._debug_timing: dict[str, float] = {}

    def _construct_algorithm(self, obs: TensorDict) -> PPO:
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
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
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

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        # Mirrors OnPolicyRunner.learn in /ws/rsl_rl/rsl_rl/runners/on_policy_runner.py,
        # with per-call timers added inside the collection loop and around
        # compute_returns/update. Keep this in step with that file if it changes.
        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )
        cur_episode_length = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.device
        )

        if self.alg.rnd:
            erewbuffer = deque(maxlen=100)
            irewbuffer = deque(maxlen=100)
            cur_ereward_sum = torch.zeros(
                self.env.num_envs, dtype=torch.float, device=self.device
            )
            cur_ireward_sum = torch.zeros(
                self.env.num_envs, dtype=torch.float, device=self.device
            )

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            act_times: list[float] = []
            env_step_times: list[float] = []
            process_step_times: list[float] = []

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    t0 = time.time()
                    actions = self.alg.act(obs)
                    t1 = time.time()

                    obs, rewards, dones, extras = self.env.step(
                        actions.to(self.env.device)
                    )
                    t2 = time.time()

                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    t3 = time.time()

                    act_times.append(t1 - t0)
                    env_step_times.append(t2 - t1)
                    process_step_times.append(t3 - t2)

                    intrinsic_rewards = (
                        self.alg.intrinsic_rewards if self.alg.rnd else None
                    )
                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        if self.alg.rnd:
                            cur_ereward_sum += rewards
                            cur_ireward_sum += intrinsic_rewards
                            cur_reward_sum += rewards + intrinsic_rewards
                        else:
                            cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
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

                t_returns0 = time.time()
                self.alg.compute_returns(obs)
                t_returns1 = time.time()

            loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            act_t = torch.tensor(act_times)
            env_t = torch.tensor(env_step_times)
            proc_t = torch.tensor(process_step_times)
            self._debug_timing = {
                "act_mean": act_t.mean().item(),
                "act_max": act_t.max().item(),
                "env_step_mean": env_t.mean().item(),
                "env_step_max": env_t.max().item(),
                "process_env_step_mean": proc_t.mean().item(),
                "compute_returns_time": t_returns1 - t_returns0,
                "update_time": stop - t_returns1,
            }

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs:
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

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        # Reuse the vendored logging and printing wholesale, then append the
        # extra per-phase timing as additional scalars and one extra line,
        # rather than duplicating that ~80 line method.
        super().log(locs, width=width, pad=pad)
        if self.writer is None or not self._debug_timing:
            return
        it = locs["it"]
        for key, value in self._debug_timing.items():
            self.writer.add_scalar(f"Perf/debug_{key}", value, it)
        d = self._debug_timing
        # print(
        #     f"""{"[debug]":>{pad}} act {d['act_mean'] * 1000:.2f}ms """
        #     f"""(max {d['act_max'] * 1000:.2f}ms), """
        #     f"""env.step {d['env_step_mean'] * 1000:.2f}ms """
        #     f"""(max {d['env_step_max'] * 1000:.2f}ms), """
        #     f"""process_env_step {d['process_env_step_mean'] * 1000:.2f}ms, """
        #     f"""compute_returns {d['compute_returns_time'] * 1000:.2f}ms, """
        #     f"""update {d['update_time'] * 1000:.2f}ms\n"""
        # )
