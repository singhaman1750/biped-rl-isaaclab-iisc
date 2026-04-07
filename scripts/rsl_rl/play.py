"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

import visualise
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
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=32, help="Number of environments to simulate."
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
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import copy
import gymnasium as gym
import numpy as np
import os
import torch

import pandas as pd
from highest_terrain_camera_wrapper import HighestTerrainCameraWrapper
from isaaclab.envs import (
    DirectMARLEnv,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

# Import extensions to set up environment tasks
import bipedal_locomotion  # noqa: F401
from bipedal_locomotion.utils.wrappers.rsl_rl import (
    RslRlPpoAlgorithmMlpCfg,
    export_mlp_as_onnx,
    export_policy_as_jit,
)
from himloco.runners import HIMOnPolicyRunner


class DataLogger:
    """A class to log and plot robot data from the simulation environment."""

    def __init__(self, log_dir: str, num_envs: int = 20, seed: int = 42):
        """Initialize the DataLogger.
        Args:
            log_dir: The directory where the plots will be saved.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.data = {
            "joint_velocities": [],
            "joint_torques": [],
            "joint_powers": [],
            "joint_positions": [],
            "joint_accelerations": [],
            "base_linear_velocity": [],
            "base_angular_velocity": [],
            "commanded_linear_velocity": [],
            "commanded_angular_velocity": [],
        }
        self.num_envs = num_envs
        self.seed = seed

    def log_link_properties(self, usd_path: str):
        """Log the mass and size of each link from the USD file.
        Args:
            usd_path: The path to the USD file of the robot.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        # Resolve the USD path if it contains environment variables or relative paths
        usd_path = os.path.abspath(usd_path)
        if not os.path.exists(usd_path):
            print(f"[WARNING] USD file not found at: {usd_path}")
            return

        stage = Usd.Stage.Open(usd_path)
        link_data = []

        for prim in stage.Traverse():
            # Check if prim is a rigid body
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                name = prim.GetName()

                # Get mass from MassAPI
                mass = 0.0
                if prim.HasAPI(UsdPhysics.MassAPI):
                    mass_api = UsdPhysics.MassAPI(prim)
                    mass_attr = mass_api.GetMassAttr().Get()
                    if mass_attr is not None:
                        mass = mass_attr

                # Get dimensions from bounding box
                geom = UsdGeom.Imageable(prim)
                # Compute the local bounding box
                res = geom.ComputeLocalBound(
                    Usd.TimeCode.Default(), UsdGeom.Tokens.default_
                )
                box = res.GetRange()
                size = box.GetSize()

                link_data.append(
                    {
                        "Link Name": name,
                        "Mass (kg)": mass,
                        "Size X (m)": size[0],
                        "Size Y (m)": size[1],
                        "Size Z (m)": size[2],
                    }
                )

        if link_data:
            df = pd.DataFrame(link_data)
            csv_path = os.path.join(self.log_dir, "link_properties.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Link properties saved to: {csv_path}")
        else:
            print(f"[WARNING] No links with RigidBodyAPI found in USD: {usd_path}")

    def log(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
        """Log data from the environment.
        Args:
            env: The simulation environment.
            asset_cfg: The configuration for the robot asset.
        """
        asset = env.scene[asset_cfg.name]

        # Detach tensors from the computation graph and move to CPU
        self.data["joint_velocities"].append(asset.data.joint_vel.cpu())
        self.data["joint_torques"].append(asset.data.applied_torque.cpu())
        joint_power = torch.mul(asset.data.applied_torque, asset.data.joint_vel)
        self.data["joint_powers"].append(joint_power.cpu())
        self.data["joint_positions"].append(asset.data.joint_pos.cpu())
        self.data["base_linear_velocity"].append(asset.data.root_lin_vel_w.cpu())
        self.data["base_angular_velocity"].append(asset.data.root_ang_vel_w.cpu())
        self.data["commanded_linear_velocity"].append(
            env.command_manager.get_command("base_velocity")[:, :2].cpu()
        )
        self.data["commanded_angular_velocity"].append(
            env.command_manager.get_command("base_velocity")[:, 2].cpu()
        )
        self.data["joint_accelerations"].append(asset.data.joint_acc.cpu())

    def plot(self):
        data = {}
        write_data = {}
        for key, item in self.data.items():
            if "commanded" not in key:
                data[key] = [self.data[key]]
            else:
                data[key] = self.data[key]
            write_data[key] = torch.stack(self.data[key]).numpy()
        visualise.visualise(data, self.log_dir, self.num_envs, self.seed)
        data_path = os.path.join(self.log_dir, "data", f"{self.seed}")
        os.makedirs(data_path, exist_ok=True)
        np.save(os.path.join(data_path, "dump.npy"), data)


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    agent_cfg: RslRlPpoAlgorithmMlpCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    env_cfg.seed = agent_cfg.seed

    # specify directory for logging experiments
    if args_cli.checkpoint_path is None:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    else:
        resume_path = args_cli.checkpoint_path
    log_dir = os.path.dirname(resume_path)

    # instantiate data logger
    data_logger = DataLogger(log_dir, args_cli.num_envs, agent_cfg.seed)
    # get asset cfg
    robot_cfg = env_cfg.scene.robot
    # log link properties
    if hasattr(robot_cfg.spawn, "usd_path"):
        data_logger.log_link_properties(robot_cfg.spawn.usd_path)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        env = HighestTerrainCameraWrapper(env)
        video_kwargs = {
            "video_folder": os.path.join(
                log_dir, "videos", "play", f"{agent_cfg.seed}"
            ),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    # load previously trained model
    print(
        f"[INFO]: Loading model checkpoint from: {resume_path} for policy type {args_cli.policy_type}"
    )
    ppo_runner = None
    if args_cli.policy_type == "HIMPPO":
        agent_cfg.policy.class_name = "HIMActorCritic"
        agent_cfg.algorithm.class_name = "HIMPPO"
        ppo_runner = HIMOnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    else:
        ppo_runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    # encoder = ppo_runner.get_inference_encoder(device=env.unwrapped.device)

    # export policy to onnx
    if EXPORT_POLICY:
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(ppo_runner.alg.policy, export_model_dir)
        print("Exported policy as jit script to: ", export_model_dir)
        export_mlp_as_onnx(
            ppo_runner.alg.actor_critic.actor,
            export_model_dir,
            "policy",
            ppo_runner.alg.actor_critic.num_actor_obs,
        )

    # reset environment
    obs = env.get_observations()
    print(obs)

    # simulate environment
    i = 0
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            # env stepping
            obs, _, _, infos = env.step(actions)

            # log data
            data_logger.log(env.unwrapped, SceneEntityCfg("robot"))
            i += 1
            if i > args_cli.video_length:
                break

    # plot data
    data_logger.plot()

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = False
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
