"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import sys
import yaml

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=400,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=24000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Maximum number of iterations to train.",
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=None,
    help="The number of iterations between saves",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Relative path to checkpoint file.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from highest_terrain_camera_wrapper import HighestTerrainCameraWrapper
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# Import extensions to set up environment tasks
from bipedal_locomotion.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg
from co_optimisation.runners import CoptOnPolicyRunner
from co_optimisation.runners.usd_generator import RandomDesignGenerator
from himloco.runners import HIMOnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# Replacement functions for missing Isaac Lab utils
def dump_pickle(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def dump_yaml(filename, data, sort_keys=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


# @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # inject total steps into curriculum for progress calculation
    if hasattr(env_cfg.curriculum, "velocity_curriculum"):
        env_cfg.curriculum.velocity_curriculum.params["max_steps"] = (
            agent_cfg.max_iterations * agent_cfg.num_steps_per_env
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # track lowest-difficulty terrain env before each captured frame
        env = HighestTerrainCameraWrapper(env)

        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner_cls = OnPolicyRunner
    if args_cli.policy_type == "HIMPPO":
        runner_cls = HIMOnPolicyRunner
        agent_cfg.policy.class_name = "HIMActorCritic"
        agent_cfg.algorithm.class_name = "HIMPPO"
    runner = None
    if args_cli.policy_type == "COPT":

        _base_urdf = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/isaaclab/biped/exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/base_robot.urdf",
        )
        _num_individuals = 64
        design_generator = RandomDesignGenerator(
            base_urdf_path=_base_urdf,
            num_individuals=_num_individuals,
        )
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["copt"] = {
            "ea_update_interval": 1000,
            "num_individuals": _num_individuals,
        }
        runner = CoptOnPolicyRunner(
            env,
            design_generator,
            agent_cfg_dict,
            log_dir=log_dir,
            device=agent_cfg.device,
        )
    else:
        runner = runner_cls(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )

    # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        if args_cli.checkpoint_path is not None:
            resume_path = args_cli.checkpoint_path
        else:
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
