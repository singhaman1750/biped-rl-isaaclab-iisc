"""Record a trained policy in a single environment under one fixed velocity command.

The command (vx, vy, wz) is held constant for the whole recording, with heading control,
standing environments, resampling, curricula and pushes all disabled, so the video shows
the policy's response to that command alone. The default command is a spin in place.

The run's rewards, actuator gains and action transform are restored from its dumped
params/env.yaml exactly as play.py does, so the replay is of the robot the policy was
trained on. The terrain is replaced by a plane unless --keep_terrain is given.

Usage:
    python scripts/rsl_rl/record_command.py --task Isaac-Limx-SF-Identified-Blind-Rough-Play-Urdf-v0 \
        --checkpoint_path <run>/model_8000.pt --vx 0 --vy 0 --wz 1.0 --headless
"""

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--vx", type=float, default=0.0, help="Forward velocity command (m/s).")
parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s).")
parser.add_argument("--wz", type=float, default=1.0, help="Yaw rate command (rad/s).")
parser.add_argument("--video_length", type=int, default=500, help="Length of the video (in steps).")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Defaults to <run>/videos/command.")
parser.add_argument("--keep_terrain", action="store_true",
                    help="Keep the task's terrain instead of a plane.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import sys

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import environments  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))

import experiment_params  # noqa: E402


def configure(env_cfg, env_params):
    """Restore the run's parameters and fix the command."""
    if env_params is not None:
        rewards = experiment_params.apply_reward_cfg(env_cfg, env_params)
        gains = experiment_params.apply_actuator_cfg(env_cfg, env_params)
        acts = experiment_params.apply_action_cfg(env_cfg, env_params)
        print(f"[INFO] Restored from run params, {len(rewards['applied'])} rewards, "
              f"{len(gains['applied'])} actuator groups, {len(acts['applied'])} action terms.")
        for key in ("decimation", "episode_length_s"):
            if env_params.get(key) is not None:
                setattr(env_cfg, key, env_params[key])
        if (env_params.get("sim") or {}).get("dt") is not None:
            env_cfg.sim.dt = float(env_params["sim"]["dt"])
    else:
        print("[WARNING] No params/env.yaml beside the checkpoint, using the working tree.")

    cmd = env_cfg.commands.base_velocity
    cmd.heading_command = False
    cmd.rel_standing_envs = 0.0
    cmd.rel_heading_envs = 0.0
    cmd.resampling_time_range = (1.0e6, 1.0e6)
    cmd.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    cmd.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
    cmd.ranges.ang_vel_z = (args_cli.wz, args_cli.wz)

    # The command curricula rewrite the ranges above, so every curriculum goes.
    for name in list(vars(env_cfg.curriculum)):
        setattr(env_cfg.curriculum, name, None)
    for name, term in list(vars(env_cfg.events).items()):
        if getattr(term, "mode", None) == "interval":
            setattr(env_cfg.events, name, None)
    env_cfg.observations.policy.enable_corruption = False

    if not args_cli.keep_terrain:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (2.0, 2.0, 1.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed

    log_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint_path))
    configure(env_cfg, experiment_params.load_params(log_dir))

    tag = f"vx{args_cli.vx:+.2f}_vy{args_cli.vy:+.2f}_wz{args_cli.wz:+.2f}"
    ckpt = os.path.splitext(os.path.basename(args_cli.checkpoint_path))[0]
    output_dir = args_cli.output_dir or os.path.join(log_dir, "videos", "command")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=output_dir,
        name_prefix=f"{ckpt}_{tag}",
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    step_dt = env.unwrapped.step_dt
    settle = int(1.0 / step_dt)
    lin_b, yaw_rate, resets = [], [], 0
    start_xy = robot.data.root_pos_w[0, :2].clone()

    obs = env.get_observations()
    with torch.inference_mode():
        for i in range(args_cli.video_length):
            obs, _, dones, _ = env.step(policy(obs))
            resets += int(dones[0])
            if i >= settle:
                lin_b.append(robot.data.root_lin_vel_b[0, :2].clone())
                yaw_rate.append(robot.data.root_ang_vel_b[0, 2].clone())

    lin_b = torch.stack(lin_b)
    yaw_rate = torch.stack(yaw_rate)
    drift = (robot.data.root_pos_w[0, :2] - start_xy).norm()
    command = env.unwrapped.command_manager.get_command("base_velocity")[0].tolist()
    print(f"[RESULT] applied command {[round(c, 3) for c in command]}", flush=True)
    print(f"[RESULT] command vx {args_cli.vx:+.2f} vy {args_cli.vy:+.2f} wz {args_cli.wz:+.2f}", flush=True)
    print(f"[RESULT] after the first {settle * step_dt:.1f} s, "
          f"yaw rate mean {yaw_rate.mean():+.3f} std {yaw_rate.std():.3f} rad/s, "
          f"vx mean {lin_b[:, 0].mean():+.3f}, vy mean {lin_b[:, 1].mean():+.3f} m/s", flush=True)
    print(f"[RESULT] horizontal drift {drift:.3f} m over "
          f"{args_cli.video_length * step_dt:.1f} s, episode resets {resets}", flush=True)
    print(f"[RESULT] video written to {output_dir}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
