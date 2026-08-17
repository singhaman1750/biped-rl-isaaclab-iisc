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
import sys
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

# The analysis package sits beside this one in the repository. Adding it to the path here
# rather than making it an installed dependency keeps play.py runnable from a checkout.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
)

import experiment_params
import stats
from stats import METADATA_KEYS

# Regex pattern matching the feet link names of the robot being analysed.
# Update this according to the robot (e.g. "ankle_.*" for TRON1 solefoot).
FEET_LINK_NAMES = "Link6[LR]"

# METADATA_KEYS, imported above, names the keys of the dump that carry index metadata or a
# constant of the run rather than a per-step series.

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

# The robot presently under analysis is SD_BRS1, matching FEET_LINK_NAMES above, so the
# table is selected here. Kept as an assignment rather than folded into the declaration so
# that the two are changed together when the robot changes.
SOLE_OFFSETS = SD_BRS1_SOLE_OFFSETS


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
            "base_com_position": [],
            "base_linear_velocity": [],
            "base_angular_velocity": [],
            "base_quaternion": [],
            "base_projected_gravity": [],
            "commanded_linear_velocity": [],
            "commanded_angular_velocity": [],
            "gait_command": [],
            "latent_space_output": [],
            "feet_contact_forces": [],
            "feet_velocities": [],
            "feet_quaternions": [],
            "feet_frame_heights": [],
            "feet_sole_clearances": [],
            "feet_distance": [],
            "robot_mass": [],
            "robot_inertia": [],
            "robot_material_properties": [],
            "episode_dones": [],
            "episode_terminated": [],
            "episode_time_outs": [],
            "joint_names": [],
            "body_names": [],
            "feet_names": [],
        }
        self.num_envs = num_envs
        self.seed = seed
        self.data_reward = {}
        self.reward_weights = {}

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

        self.data["joint_velocities"].append(asset.data.joint_vel.cpu().numpy())
        self.data["joint_torques"].append(asset.data.applied_torque.cpu().numpy())
        joint_power = torch.mul(asset.data.applied_torque, asset.data.joint_vel)
        self.data["joint_powers"].append(joint_power.cpu().numpy())
        self.data["joint_positions"].append(asset.data.joint_pos.cpu().numpy())
        self.data["base_com_position"].append(asset.data.root_com_pos_w.cpu().numpy())
        self.data["base_linear_velocity"].append(asset.data.root_lin_vel_b.cpu().numpy())
        self.data["base_angular_velocity"].append(asset.data.root_ang_vel_b.cpu().numpy())
        self.data["base_quaternion"].append(asset.data.root_quat_w.cpu().numpy())
        self.data["base_projected_gravity"].append(
            asset.data.projected_gravity_b.cpu().numpy()
        )
        self.data["commanded_linear_velocity"].append(
            env.command_manager.get_command("base_velocity")[:, :2].cpu().numpy()
        )
        self.data["commanded_angular_velocity"].append(
            env.command_manager.get_command("base_velocity")[:, 2].cpu().numpy()
        )
        if "gait_command" in env.command_manager.active_terms:
            self.data["gait_command"].append(
                env.command_manager.get_command("gait_command").cpu().numpy()
            )
        self.data["joint_accelerations"].append(asset.data.joint_acc.cpu().numpy())

        # log feet contact forces and feet velocities
        feet_ids, _ = asset.find_bodies(FEET_LINK_NAMES)
        contact_sensor = env.scene.sensors["contact_forces"]
        sensor_feet_ids, _ = contact_sensor.find_bodies(FEET_LINK_NAMES)

        # Record the ORDER in which every array here is stored, once, on the first call.
        # This cannot be inferred from the URDF. IsaacLab enumerates degrees of freedom and
        # bodies breadth-first over the articulation tree, but the within-depth tie-break is
        # not the URDF declaration order. SD_BRS.urdf declares the whole right leg chain
        # before the left, yet the articulation yields left before right at every depth, so
        # a consumer that reconstructs the order from the asset file gets it backwards. The
        # names are therefore captured from the live articulation. Purely additive, no
        # existing key changes, so dumps and consumers predating these keys are unaffected.
        if not self.data["joint_names"]:
            self.data["joint_names"] = list(asset.joint_names)
            self.data["body_names"] = list(asset.body_names)
            self.data["feet_names"] = [asset.body_names[i] for i in feet_ids]

            for key, attr in (
                ("joint_position_limits", "joint_pos_limits"),
                ("joint_soft_position_limits", "soft_joint_pos_limits"),
                ("default_joint_pos", "default_joint_pos"),
            ):
                value = getattr(asset.data, attr, None)
                if value is not None:
                    # Limits are per environment, and identical across environments
                    # unless a randomisation event has altered them, so environment
                    # zero is taken as the representative row.
                    # TODO look into environment specific logging of limits to take into account 
                    # randomisation. Though joint limit randomisation is unlikely, 
                    # changes during co-design are plausible.
                    self.data[key] = value[0].cpu().numpy().tolist()

            # The sim ceilings are retained under their own names rather than discarded,
            # since the two are genuinely different quantities and a torque clipped by
            # the actuator model is not the same event as one clipped by the solver.
            self._log_actuator_limits(asset)
            self.data["body_masses"] = asset.data.default_mass[0].cpu().numpy().tolist()
            self.data["step_dt"] = float(env.step_dt)
            self.data["contact_force_history_length"] = int(
                getattr(contact_sensor.cfg, "history_length", 0)
            )
            self.data["frame_convention"] = {
                "feet_distance": "base_yaw",
                "feet_velocities": "base_yaw",
                "feet_contact_forces": "base_yaw",
                "base_com_position": "world",
                "base_linear_velocity": "base",
                "base_angular_velocity": "base",
                "base_quaternion": "world",
                "feet_quaternions": "world",
            }

        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_feet_ids, :]
        peak_index = forces.norm(dim=-1).argmax(dim=1)  # (num_envs, num_feet)
        peak_forces = torch.gather(
            forces, 1, peak_index[:, None, :, None].expand(-1, 1, -1, 3)
        ).squeeze(1)  # (num_envs, num_feet, 3)

        yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w)
        self.data["feet_contact_forces"].append(
            self._to_base_frame(peak_forces, yaw_quat).cpu().numpy()
        )
        self.data["feet_velocities"].append(
            self._to_base_frame(
                asset.data.body_lin_vel_w[:, feet_ids], yaw_quat
            ).cpu().numpy()
        )

        self.data["feet_quaternions"].append(
            asset.data.body_quat_w[:, feet_ids].cpu().numpy()
        )

        self.data["feet_frame_heights"].append(
            asset.data.body_pos_w[:, feet_ids, 2].cpu().numpy()
        )
        if SOLE_OFFSETS is not None:
            self.data["feet_sole_clearances"].append(
                self._sole_clearance(asset, feet_ids).cpu().numpy()
            )

        feet_pos = asset.data.body_pos_w[:, feet_ids]  # (num_envs, 2, 3)
        separation_w = (feet_pos[:, 0, :] - feet_pos[:, 1, :]).unsqueeze(1)
        self.data["feet_distance"].append(
            self._to_base_frame(separation_w, yaw_quat).squeeze(1).cpu().numpy()
        )

        self.data["robot_mass"].append(asset.data.default_mass.cpu().numpy())
        inertia = asset.data.default_inertia
        self.data["robot_inertia"].append(
            inertia.reshape(inertia.shape[0], -1).cpu().numpy()
        )
        material = asset.root_physx_view.get_material_properties()
        self.data["robot_material_properties"].append(
            material.reshape(material.shape[0], -1).cpu().numpy()
        )

        termination_manager = getattr(env, "termination_manager", None)
        if termination_manager is not None:
            self.data["episode_dones"].append(
                termination_manager.dones.cpu().numpy()
            )
            self.data["episode_terminated"].append(
                termination_manager.terminated.cpu().numpy()
            )
            self.data["episode_time_outs"].append(
                termination_manager.time_outs.cpu().numpy()
            )

    @staticmethod
    def _to_base_frame(vectors: torch.Tensor, yaw_quat: torch.Tensor) -> torch.Tensor:
        """Rotate per body world frame vectors into the base frame by the base yaw.

        Args:
            vectors: World frame vectors of shape (num_envs, num_bodies, 3).
            yaw_quat: The yaw only base orientation, of shape (num_envs, 4), as returned
                by math_utils.yaw_quat.

        Returns:
            The same vectors expressed in the base frame, of the same shape.
        """
        num_envs, num_bodies = vectors.shape[0], vectors.shape[1]
        quat = yaw_quat.unsqueeze(1).expand(num_envs, num_bodies, 4)
        return math_utils.quat_apply_inverse(
            quat.reshape(-1, 4), vectors.reshape(-1, 3)
        ).view(num_envs, num_bodies, 3)

    def _log_actuator_limits(self, asset):
        """Record the effort and velocity ceilings the ACTUATOR MODELS enforce.

        Every saturation statistic is a comparison against one of these, so the ceiling
        must be the one the torque is actually clipped against. ActuatorBase.clip_effort
        clips to +/- self.effort_limit at actuator_base.py:381, and that attribute is the
        identified ceiling, whereas asset.data.joint_effort_limits carries the SIM ceiling
        which for an explicit actuator model is an unbounded default.
        For the current implementation explicit joints are used exclusively.

        Each actuator covers a subset of the joints, given by its joint_indices, which may
        be a slice where the actuator covers all of them. The per joint vector is assembled
        by scattering each actuator's limits into the joint indexed positions, and any
        position no actuator claims is left as NaN rather than as a plausible number, so a
        consumer sees an absent ceiling rather than a wrong one.

        Args:
            asset: The robot articulation.
        """
        for key, attr in (
            ("joint_effort_limits", "effort_limit"),
            ("joint_velocity_limits", "velocity_limit"),
        ):
            limits = torch.full(
                (asset.num_joints,), float("nan"), device=asset.device
            )
            found = False
            for actuator in getattr(asset, "actuators", {}).values():
                value = getattr(actuator, attr, None)
                if value is None:
                    continue
                # (num_envs, num_joints_in_actuator), environment zero being
                # representative for the same reason the position limits are.
                value = torch.as_tensor(value)
                row = value[0] if value.ndim > 1 else value
                limits[actuator.joint_indices] = row.to(limits.dtype).to(limits.device)
                found = True
            if found:
                self.data[key] = limits.cpu().numpy().tolist()
        # The solver's ceilings, kept beside the actuator's under their own names. These
        # are what the previous two keys used to hold, so a reader comparing an old dump
        # against a new one can see both figures and tell which convention it was on.
        for key, attr in (
            ("joint_effort_limits_sim", "joint_effort_limits"),
            ("joint_velocity_limits_sim", "joint_vel_limits"),
        ):
            value = getattr(asset.data, attr, None)
            if value is not None:
                self.data[key] = value[0].cpu().numpy().tolist()

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
            SOLE_OFFSETS, dtype=foot_pos.dtype, device=foot_pos.device
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
            # The configured weight of every term, captured once. rewards.npy stores
            # func × weight, a RATE per second, because RewardManager.compute divides by
            # dt at reward_manager.py:157 before storing into _step_reward. Recovering
            # the unweighted function value therefore requires the weight, and dividing
            # by a weight read from a configuration file is unsound because play.py
            # rebuilds its environment from the LIVE tree through parse_env_cfg, so the
            # tree's weight and the trained weight may differ.
            if not self.reward_weights:
                self.reward_weights = {
                    name: float(cfg.weight)
                    for name, cfg in zip(
                        reward_manager._term_names, reward_manager._term_cfgs
                    )
                }
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
        for key, item in self.data.items():
            if key in METADATA_KEYS:
                data[key] = item
                continue
            if len(item) > 0:
                data[key] = np.stack(item)
        # visualise.visualise(data, self.log_dir, self.num_envs, self.seed)
        data_path = os.path.join(self.log_dir, "data", f"{self.seed}")
        os.makedirs(data_path, exist_ok=True)
        np.save(os.path.join(data_path, "dump.npy"), data)

        write_rewards = {
            k: np.stack(v) for k, v in self.data_reward.items() if len(v) > 0
        }

        if self.reward_weights:
            write_rewards["_weights"] = self.reward_weights
        np.save(os.path.join(data_path, "rewards.npy"), write_rewards)
        try:
            np.save(
                os.path.join(data_path, "statistics.npy"),
                stats.compute_all(
                    stats.from_dump(data, write_rewards, dt=self.data.get("step_dt"))
                ),
            )
            print(f"[INFO] Statistics written to: {data_path}/statistics.npy")
        except Exception as error:  # noqa: BLE001
            print(f"[WARNING] Statistics computation failed, raw dumps are intact: {error}")


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

    # The configuration the policy was TRAINED under, taken from the artefact rather than
    # from the tree. Without this the replay applies whatever reward set the tree holds
    # today, which is how a policy comes to be evaluated against terms it never saw and
    # its budget read as though it had. Unconditional, since no evaluation wants the
    # tree's configuration in preference to the run's. Failure is reported and tolerated,
    # the replay falling back to the tree, since a checkpoint whose parameters were never
    # dumped must remain playable.
    env_params = experiment_params.load_params(log_dir)
    if env_params is None:
        print(f"[WARNING] No params/env.yaml under {log_dir}, using the working tree.")
    else:
        report = experiment_params.apply_reward_cfg(env_cfg, env_params)
        print(f"[INFO] Rewards from run params, {len(report['applied'])} applied, "
              f"{len(report['added'])} added, {len(report['removed'])} removed.")
        for name, reason in report["failed"].items():
            print(f"[WARNING] Reward term '{name}' could not be imported, {reason}")
        for key in ("decimation", "episode_length_s"):
            if env_params.get(key) is not None:
                setattr(env_cfg, key, env_params[key])
        if (env_params.get("sim") or {}).get("dt") is not None:
            env_cfg.sim.dt = float(env_params["sim"]["dt"])

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
