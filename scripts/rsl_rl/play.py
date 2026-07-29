"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from typing import Union

# import visualise
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

import isaaclab.utils.math as math_utils
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
from co_optimisation.algorithms import CoptPPO
from co_optimisation.runners import CoptOnPolicyRunner
from co_optimisation.runners.usd_generator import (
    DEFAULT_PARAM_RANGES,
    CMAESDesignGenerator,
    RandomDesignGenerator,
)
from himloco.runners import HIMOnPolicyRunner

# Regex pattern matching the feet link names of the robot being analysed.
# Update this according to the robot (e.g. "ankle_.*" for TRON1 solefoot).
FEET_LINK_NAMES = "Link6[LR]"

# Points on the sole, in the foot body frame, whose lowest world height is the true
# tilt invariant sole clearance, the same quantity foot_clearance_reward_v2 rewards.
# Defaults to None, under which only the body frame height is logged, which is the
# quantity the original foot_clearance_reward reads and the exact clearance only for
# a point foot or a foot held level. Set this alongside FEET_LINK_NAMES when analysing
# a sole footed robot, the SD_BRS1 table below is the one carried by brs_base_env_cfg.py.
SOLE_OFFSETS = None

# Support set of the SD_BRS1 Link6 sole, for use with FEET_LINK_NAMES = "Link6[LR]".
# The sole face lies at z -0.124 over x -0.1091..0.1521 and y +/-0.0970, with chamfered
# fore and aft edges rising to z -0.1144, and this table reproduces the collision mesh's
# lowest point to within 0.85 mm over pitch +/-50 deg and roll +/-30 deg. Same table
# serves both feet. Kept in step with the sole_offsets of rew_foot_clearance.
SD_BRS1_SOLE_OFFSETS = [
    [-0.1091, -0.0970, -0.1240],
    [-0.1091, 0.0970, -0.1240],
    [0.1521, -0.0970, -0.1240],
    [0.1521, 0.0970, -0.1240],
    [-0.1206, -0.0970, -0.1203],
    [-0.1206, 0.0970, -0.1203],
    [0.1636, -0.0970, -0.1203],
    [0.1636, 0.0970, -0.1203],
    [-0.1262, -0.0970, -0.1144],
    [-0.1262, 0.0970, -0.1144],
    [0.1692, -0.0970, -0.1144],
    [0.1692, 0.0970, -0.1144],
]


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
            "latent_space_output": [],
            "feet_contact_forces": [],
            "feet_velocities": [],
            "feet_frame_heights": [],
            "feet_sole_clearances": [],
            "feet_distance": [],
            "robot_mass": [],
            "robot_inertia": [],
            "robot_material_properties": [],
        }
        self.num_envs = num_envs
        self.seed = seed
        self.data_reward = {}

    def log_link_properties(self, usd_path: Union[list, str]):
        """Log the mass and size of each link from the USD file.
        Args:
            usd_path: The path to the USD file of the robot.
        """
        from pxr import Usd, UsdGeom, UsdPhysics

        # Resolve the USD path if it contains environment variables or relative paths
        if not isinstance(usd_path, list):
            usd_path = [usd_path]

        for path in usd_path:
            _usd_path = os.path.abspath(path)
            if not os.path.exists(_usd_path):
                print(f"[WARNING] USD file not found at: {_usd_path}")
                return

            stage = Usd.Stage.Open(_usd_path)
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

        # Detach tensors from the computation graph, move to CPU and convert to numpy
        # arrays, so self.data holds numpy arrays rather than torch tensors.
        self.data["joint_velocities"].append(asset.data.joint_vel.cpu().numpy())
        self.data["joint_torques"].append(asset.data.applied_torque.cpu().numpy())
        joint_power = torch.mul(asset.data.applied_torque, asset.data.joint_vel)
        self.data["joint_powers"].append(joint_power.cpu().numpy())
        self.data["joint_positions"].append(asset.data.joint_pos.cpu().numpy())
        self.data["base_linear_velocity"].append(asset.data.root_lin_vel_b.cpu().numpy())
        self.data["base_angular_velocity"].append(asset.data.root_ang_vel_b.cpu().numpy())
        self.data["commanded_linear_velocity"].append(
            env.command_manager.get_command("base_velocity")[:, :2].cpu().numpy()
        )
        self.data["commanded_angular_velocity"].append(
            env.command_manager.get_command("base_velocity")[:, 2].cpu().numpy()
        )
        self.data["joint_accelerations"].append(asset.data.joint_acc.cpu().numpy())

        # log feet contact forces and feet velocities
        feet_ids, _ = asset.find_bodies(FEET_LINK_NAMES)
        contact_sensor = env.scene.sensors["contact_forces"]
        sensor_feet_ids, _ = contact_sensor.find_bodies(FEET_LINK_NAMES)
        self.data["feet_contact_forces"].append(
            contact_sensor.data.net_forces_w[:, sensor_feet_ids].cpu().numpy()
        )
        self.data["feet_velocities"].append(
            asset.data.body_lin_vel_w[:, feet_ids].cpu().numpy()
        )

        # log feet heights, following the foot clearance reward. The body frame height is
        # the quantity foot_clearance_reward reads, and the sole clearance the quantity
        # foot_clearance_reward_v2 reads, so logging both exposes their difference, which
        # on a sole foot is the tilt the frame height cannot distinguish from a lift.
        self.data["feet_frame_heights"].append(
            asset.data.body_pos_w[:, feet_ids, 2].cpu().numpy()
        )
        if SD_BRS1_SOLE_OFFSETS is not None:
            self.data["feet_sole_clearances"].append(
                self._sole_clearance(asset, feet_ids).cpu().numpy()
            )

        # log signed per-axis distance between the two ankles, matching the quantity
        # penalised by feet_distance in mdp/rewards.py. body_pos_w[:, feet_ids] gives
        # shape (num_envs, 2, 3); index 0 is the first foot resolved by FEET_LINK_NAMES
        # (Link6R) and index 1 is the second (Link6L), so the difference is R minus L.
        feet_pos = asset.data.body_pos_w[:, feet_ids]  # (num_envs, 2, 3)
        self.data["feet_distance"].append(
            (feet_pos[:, 0, :] - feet_pos[:, 1, :]).cpu().numpy()
        )

        # log the morphology and material observations fed to the policy, matching the
        # extraction in mdp.robot_mass, mdp.robot_inertia and mdp.robot_material_properties.
        # These are per-morphology constants, logged every step to keep the plot() stacking
        # uniform with the time series above.
        self.data["robot_mass"].append(asset.data.default_mass.cpu().numpy())
        inertia = asset.data.default_inertia
        self.data["robot_inertia"].append(
            inertia.reshape(inertia.shape[0], -1).cpu().numpy()
        )
        material = asset.root_physx_view.get_material_properties()
        self.data["robot_material_properties"].append(
            material.reshape(material.shape[0], -1).cpu().numpy()
        )

    @staticmethod
    def _sole_clearance(asset, feet_ids: list) -> torch.Tensor:
        """Lowest world height of the sole points, as in mdp.foot_clearance_reward_v2.

        Args:
            asset: The robot articulation.
            feet_ids: Indices of the feet bodies.

        Returns:
            The per foot sole clearance, of shape (num_envs, num_feet).
        """
        foot_pos = asset.data.body_pos_w[:, feet_ids]  # (N, F, 3)
        foot_quat = asset.data.body_quat_w[:, feet_ids]  # (N, F, 4)
        num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]

        offsets = torch.as_tensor(
            SD_BRS1_SOLE_OFFSETS, dtype=foot_pos.dtype, device=foot_pos.device
        )  # (P, 3)
        num_pts = offsets.shape[0]
        # rotate every sole point by its foot's orientation, then offset by the foot position
        quat = foot_quat.unsqueeze(2).expand(num_envs, num_feet, num_pts, 4)
        pts = offsets.view(1, 1, num_pts, 3).expand(num_envs, num_feet, num_pts, 3)
        pts_w = math_utils.quat_apply(quat.reshape(-1, 4), pts.reshape(-1, 3)).view(
            num_envs, num_feet, num_pts, 3
        )
        pts_w = pts_w + foot_pos.unsqueeze(2)
        return pts_w[..., 2].min(dim=2)[0]  # (N, F)

    def log_reward(self, rewards, env: ManagerBasedRLEnv | None = None):
        """Log per-step rewards.

        Args:
            rewards: The total reward tensor of shape (num_envs,) returned by env.step().
            env:     Optional unwrapped ManagerBasedRLEnv. When provided, per-term rewards
                     are extracted from env.reward_manager._step_reward (shape num_envs x num_terms)
                     and stored under their term names. infos["log"] is not used because its
                     entries are 0-d episodic-mean scalars computed only on reset envs, not
                     per-env step values.
        """
        if "total_reward" not in self.data_reward:
            self.data_reward["total_reward"] = []
        self.data_reward["total_reward"].append(rewards.cpu().numpy())

        if env is not None and hasattr(env, "reward_manager"):
            reward_manager = env.reward_manager
            # _step_reward: (num_envs, num_terms), stores func(...) * weight for this step.
            step_reward = reward_manager._step_reward
            for term_idx, term_name in enumerate(reward_manager._term_names):
                if term_name not in self.data_reward:
                    self.data_reward[term_name] = []
                self.data_reward[term_name].append(step_reward[:, term_idx].cpu().numpy())

    def log_latent(self, latent):
        self.data["latent_space_output"].append(latent.cpu().numpy())

    def plot(self):
        data = {}
        write_data = {}
        for key, item in self.data.items():
            if "commanded" not in key:
                data[key] = [self.data[key]]
            else:
                data[key] = self.data[key]
            if len(self.data[key]) > 0:
                write_data[key] = np.stack(self.data[key])
        # visualise.visualise(data, self.log_dir, self.num_envs, self.seed)
        data_path = os.path.join(self.log_dir, "data", f"{self.seed}")
        os.makedirs(data_path, exist_ok=True)
        np.save(os.path.join(data_path, "dump.npy"), data)
        if self.data_reward:
            write_rewards = {k: np.stack(v) for k, v in self.data_reward.items() if len(v) > 0}
            np.save(os.path.join(data_path, "rewards.npy"), write_rewards)


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
    elif args_cli.policy_type == "COPT":

        _base_urdf = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/isaaclab/biped/exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/base_robot.urdf",
        )
        _num_individuals = 256
        param_ranges = {}
        params = ["thigh_length_scale", "shank_length_scale"]
        for param in params:
            param_ranges[param] = DEFAULT_PARAM_RANGES[param]
        design_generator = CMAESDesignGenerator(
            base_urdf_path=_base_urdf,
            num_individuals=_num_individuals,
            param_ranges=param_ranges,
            sigma0=0.1,
            seed=42,
            late_start=False,
        )
        agent_cfg.policy.class_name = "CoptActorCritic"
        agent_cfg.algorithm.class_name = "CoptPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["copt"] = {
            "ea_update_interval": 50,
            "ea_late_start": -1,
            "num_individuals": _num_individuals,
        }
        ppo_runner = CoptOnPolicyRunner(
            env,
            design_generator,
            agent_cfg_dict,
            log_dir=log_dir,
            device=agent_cfg.device,
        )
    elif args_cli.policy_type == "COPT-LEARNED":

        _base_urdf = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "/workspace/isaaclab/biped/exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/base_robot.urdf",
        )
        _num_individuals = 256
        param_ranges = {}
        params = ["thigh_length_scale", "shank_length_scale"]
        for param in params:
            param_ranges[param] = DEFAULT_PARAM_RANGES[param]
        design_generator = CMAESDesignGenerator(
            base_urdf_path=_base_urdf,
            num_individuals=_num_individuals,
            param_ranges=param_ranges,
            sigma0=0.1,
            seed=42,
            late_start=False,
        )
        agent_cfg.policy.class_name = "CoptLearnedModelActorCritic"
        agent_cfg.algorithm.class_name = "CoptLearnedModelPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["copt"] = {
            "ea_update_interval": 50,
            "ea_late_start": -1,
            "num_individuals": _num_individuals,
        }
        ppo_runner = CoptOnPolicyRunner(
            env,
            design_generator,
            agent_cfg_dict,
            log_dir=log_dir,
            device=agent_cfg.device,
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

    # simulate environment
    i = 0
    while simulation_app.is_running():
        # run everything in inference mode
        actions = None
        latent = None
        with torch.inference_mode():
            if args_cli.policy_type == "HIMPPO":
                actions, latent = policy(obs)
            else:
                actions = policy(obs)
            # env stepping
            obs, rewards, dones, infos = env.step(actions)

            # log data
            data_logger.log(env.unwrapped, SceneEntityCfg("robot"))
            data_logger.log_reward(rewards, env.unwrapped)
            if args_cli.policy_type == "HIMPPO":
                data_logger.log_latent(latent)
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
