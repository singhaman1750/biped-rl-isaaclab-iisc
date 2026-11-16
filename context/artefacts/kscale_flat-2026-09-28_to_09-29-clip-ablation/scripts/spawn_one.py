"""Spawn ONE task and report its joint and body counts. One task per process.

Building two ManagerBasedRLEnv configurations in a single process exhausts the interpreter's
recursion limit inside configclass._validate at configclass.py:291, which is a property of
that validator against a deep configuration tree and not of any robot, so the probe is run
once per task and the task is named on the command line.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--probe_task", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import environments  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg(args.probe_task, device=args.device, num_envs=4)
env = gym.make(args.probe_task, cfg=cfg)
robot = env.unwrapped.scene["robot"]
jn, bn = list(robot.joint_names), list(robot.body_names)
print(f"PROBE {args.probe_task}")
print(f"PROBE joints {len(jn)} {jn}")
print(f"PROBE bodies {len(bn)}")
print(f"PROBE action_dim {env.unwrapped.action_manager.total_action_dim}")
print(f"PROBE mass {float(robot.data.default_mass[0].sum()):.5f}")
env.close()
simulation_app.close()
