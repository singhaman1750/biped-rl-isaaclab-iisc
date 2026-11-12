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
parser.add_argument(
    "--keep_tree_actuators",
    action="store_true",
    help="Do not restore the run's actuator gains from params/env.yaml, using the "
    "working tree's instead. For deliberately replaying a policy on gains it was not "
    "trained under, which is a different experiment and not an evaluation of the run.",
)
parser.add_argument(
    "--keep_tree_actions",
    action="store_true",
    help="Do not restore the run's action scale and offset from params/env.yaml, using "
    "the working tree's instead. For deliberately replaying a policy under a different "
    "action mapping, which is a different experiment and not an evaluation of the run.",
)
parser.add_argument(
    "--copt_best_design",
    action="store_true",
    help="Replay the best/* design of the designs/cmaes_history.csv row whose iteration lies "
    "nearest the checkpoint's, spawned in every environment, and dump under "
    "best_design/model_<n>/ rather than beside the run. Off, the default, leaves the COPT "
    "replay as it was, with no searched design applied.",
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
import re
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
from isaaclab.sensors import RayCaster
from isaaclab.utils.dict import print_dict
from isaaclab.utils.warp import raycast_mesh
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import environments  # noqa: F401
from co_optimisation.algorithms import CoptPPO
from co_optimisation.runners import CoptOnPolicyRunner
from co_optimisation.runners.usd_generator import (
    CMAES_PARAM_KEYS,
    DEFAULT_PARAM_RANGES,
    CMAESDesignGenerator,
    FixedDesignGenerator,
    RandomDesignGenerator,
)
from co_optimisation.utils.respawn import respawn_robots
from environments.utils.wrappers.rsl_rl import (
    RslRlPpoAlgorithmMlpCfg,
    export_mlp_as_onnx,
    export_policy_as_jit,
)
from himloco.runners import HIMOnPolicyRunner
from rsl_rl.runners import OnPolicyRunner

# The analysis package sits beside this one in the repository. Adding it to the path here
# rather than making it an installed dependency keeps play.py runnable from a checkout.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis")
)

import experiment_params
import stats
from stats import METADATA_KEYS

# Regex pattern matching the feet link names of the robot being analysed.
FEET_LINK_NAMES = "foot_[LR]_Link"

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

# Support set of the kscale foot_6061 sole, for use with FEET_LINK_NAMES = "foot_6061.*".
# NOTE the axis permutation: this foot's link frame carries its vertical on +y and its
# fore-aft length on z, unlike SD_BRS1 where the vertical is z. The sole plane lies at
# y +0.0430, and this table reproduces the collision mesh's lowest point to within 0.836 mm
# over the ankle's full roll and pitch travel. Same table serves both feet. Kept in step with
# KSCALE_SOLE_OFFSETS in tasks/locomotion/cfg/SF/kscale_base_env_cfg.py, and measured by
# scripts/analysis/kscale_sole_analysis.py.
KSCALE_SOLE_OFFSETS = [
    [-0.0423, +0.0430, +0.0138],
    [-0.0362, +0.0430, -0.1572],
    [-0.0326, +0.0430, -0.1712],
    [-0.0258, +0.0430, -0.1805],
    [-0.0108, +0.0430, -0.1889],
    [+0.0051, +0.0430, -0.1899],
    [+0.0213, +0.0430, -0.1841],
    [+0.0326, +0.0430, -0.1712],
    [+0.0362, +0.0430, -0.1572],
    [+0.0423, +0.0430, +0.0138],
    [+0.0363, +0.0430, +0.0200],
    [-0.0363, +0.0430, +0.0200],
]

# SD_BRS1's sole face lies at z -0.1240 with the chamfers rising toward z -0.1144, so the
# face the machine stands on points along the link's MINUS z.
SD_BRS1_SOLE_NORMAL = (0.0, 0.0, -1.0)
# The kscale foot link is permuted, carrying its width on x, its fore and aft length on z
# and its vertical on y POINTING DOWNWARD, 
KSCALE_SOLE_NORMAL = (0.0, 1.0, 0.0)

# Support set of the TRON1 foot_[LR]_Link sole, for use with FEET_LINK_NAMES = "foot_[LR]_Link",
TRON1_SOLE_OFFSETS = [
    [-0.090, -0.035, -0.050],
    [-0.090, 0.035, -0.050],
    [0.110, -0.035, -0.050],
    [0.110, 0.035, -0.050],
    [-0.090, -0.035, -0.020],
    [-0.090, 0.035, -0.020],
    [0.110, -0.035, -0.020],
    [0.110, 0.035, -0.020],
]
# The TRON1 foot frame is unpermuted, x forward, y left and z up, and the generator places the
# sole level at the nominal pose, so the ground facing normal is the link's MINUS z.
TRON1_SOLE_NORMAL = (0.0, 0.0, -1.0)

# Radius of a spherical foot, used in place of a sole table where the profile carries none.
# The quadruped's feet are collision spheres of radius 0.022 m and the lowest point of a sphere sits
# one radius beneath its centre along the world vertical whatever the foot's orientation.
QUADRUPED_FOOT_RADIUS = 0.022

_ROBOT_PROFILES = (
    ("Link6[LR]", SD_BRS1_SOLE_OFFSETS, SD_BRS1_SOLE_NORMAL, None),
    ("foot_6061.*", KSCALE_SOLE_OFFSETS, KSCALE_SOLE_NORMAL, None),
    ("foot_[LR]_Link", TRON1_SOLE_OFFSETS, TRON1_SOLE_NORMAL, None),
    ("foot_.._Link", None, None, QUADRUPED_FOOT_RADIUS)
)

SOLE_OFFSETS = SD_BRS1_SOLE_OFFSETS
SOLE_NORMAL_L = SD_BRS1_SOLE_NORMAL
# Read only where SOLE_OFFSETS is None. Zero treats the foot frame origin as the contact point.
FOOT_RADIUS = 0.0

# The terrain mesh the height scanner casts against in every task here, whose warp mesh the
# clearance raycast reuses from RayCaster.meshes.
TERRAIN_MESH_PRIM_PATH = "/World/ground"

_PROFILE_RESOLVED = False


def _select_robot_profile(asset):
    """Set FEET_LINK_NAMES, SOLE_OFFSETS, SOLE_NORMAL_L and FOOT_RADIUS from the body names, once.

    A profile whose table is None is a point footed robot, its clearance being reported from
    the lowest point of a sphere of FOOT_RADIUS about the foot origin, rather than against
    another robot's sole geometry.
    The sole normal is governed by the same rule and for the same reason, a tilt measured
    against another robot's sole axis being not merely imprecise but a different quantity.
    """
    global FEET_LINK_NAMES, SOLE_OFFSETS, SOLE_NORMAL_L, FOOT_RADIUS, _PROFILE_RESOLVED
    if _PROFILE_RESOLVED:
        return
    _PROFILE_RESOLVED = True
    for pattern, table, normal, radius in _ROBOT_PROFILES:
        # find_bodies raises rather than returning empty when a key matches nothing, since
        # resolve_matching_names treats an unmatched key as an error, so a non-matching profile
        # arrives here as an exception and not as a falsy result.
        try:
            ids, names = asset.find_bodies(pattern)
        except (ValueError, KeyError):
            continue
        if ids:
            FEET_LINK_NAMES, SOLE_OFFSETS, SOLE_NORMAL_L = pattern, table, normal
            if radius is not None:
                FOOT_RADIUS = radius
            print(f"[play] resolved feet pattern {pattern!r} -> {names}")
            print(f"[play] sole table: {f'none, point foot of radius {FOOT_RADIUS} m' if table is None else str(len(table)) + ' points'}")
            print(f"[play] sole normal in the link frame: "
                  f"{'none, tilt logging disabled' if normal is None else normal}")
            return
    print(
        f"[play] WARNING no profile matched the articulation's bodies; falling back to "
        f"{FEET_LINK_NAMES!r}. Sole clearance and sole tilt will be logged against SD_BRS1's "
        f"geometry, which is meaningless for any other robot. Add a profile to _ROBOT_PROFILES."
    )


def _best_design(log_dir: str, resume_path: str) -> tuple[dict[str, float], dict]:
    """The best/* design of the cmaes_history.csv row whose iteration lies nearest the checkpoint's.

    Args:
        log_dir: The run directory, holding designs/cmaes_history.csv.
        resume_path: The checkpoint, whose basename model_<n>.pt gives the iteration.

    Returns:
        The searched parameters, and a record naming the row they were taken from.
    """
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(resume_path))
    if match is None:
        raise ValueError(f"cannot read an iteration from the checkpoint name {resume_path!r}")
    target = int(match.group(1))
    # A checkpoint of the fine-tune replays the final design, the fine-tuned distribution mean that
    # log_optimised_designs records. A checkpoint at or before the termination never trained on it.
    final = os.path.join(log_dir, "designs", "optimised", "final_design.csv")
    if os.path.exists(final):
        row = pd.read_csv(final).iloc[0]
        if bool(row.get("fine_tuned", False)) and target > int(row["finetune_start_iteration"]):
            parameters = {k: float(row[k]) for k in CMAES_PARAM_KEYS if k in row.index}
            return parameters, {
                "checkpoint_iteration": target,
                "history_iteration": int(row["termination_iteration"]),
                "generation": int(row["termination_generation"]),
                "source": "designs/optimised/final_design.csv, the fine-tuned distribution mean",
                "parameters": parameters,
            }
    path = os.path.join(log_dir, "designs", "cmaes_history.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no CMA-ES history at {path}")
    history = pd.read_csv(path)
    columns = [c for c in history.columns if c.startswith("best/")]
    if history.empty or not columns:
        raise ValueError(f"{path} carries no best/ columns")
    # idxmin takes the first minimum and the history is appended in iteration order, so a
    # checkpoint equidistant from two generations takes the earlier.
    row = history.loc[(history["iteration"] - target).abs().idxmin()]
    parameters = {c[len("best/"):]: float(row[c]) for c in columns}
    record = {
        "checkpoint_iteration": target,
        "history_iteration": int(row["iteration"]),
        "generation": int(row["generation"]),
        "parameters": parameters,
    }
    return parameters, record


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
            "joint_position_targets": [],
            "joint_action_raw": [],
            "joint_computed_torque": [],
            "joint_available_torque": [],
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
            "feet_sole_tilt": [],
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
        self._envelope = None
        self._action_slots = None

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
        computed = getattr(asset.data, "computed_torque", None)
        if computed is not None:
            self.data["joint_computed_torque"].append(computed.cpu().numpy())
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
        _select_robot_profile(asset)
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
            self._log_action_metadata(env, asset)
            self.data["body_masses"] = asset.data.default_mass[0].cpu().numpy().tolist()
            self.data["step_dt"] = float(env.step_dt)
            self.data["contact_force_history_length"] = int(
                getattr(contact_sensor.cfg, "history_length", 0)
            )
            if SOLE_NORMAL_L is not None:
                self.data["sole_normal_link"] = list(SOLE_NORMAL_L)
            self.data["frame_convention"] = {
                "feet_distance": "base_yaw",
                "feet_velocities": "base_yaw",
                "feet_contact_forces": "base_yaw",
                "base_com_position": "world",
                "base_linear_velocity": "base",
                "base_angular_velocity": "base",
                "base_quaternion": "world",
                "feet_quaternions": "world",
                "feet_sole_tilt": "base_yaw",
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
        height_scanner = env.scene.sensors.get("height_scanner")
        ray_hits_w = None if height_scanner is None else height_scanner.data.ray_hits_w
        if SOLE_OFFSETS is not None:
            clearance = self._sole_clearance(asset, feet_ids, ray_hits_w)
        else:
            clearance = self._sole_clearance_pf(asset, feet_ids, ray_hits_w)
        self.data["feet_sole_clearances"].append(clearance.cpu().numpy())
        if SOLE_NORMAL_L is not None:
            self.data["feet_sole_tilt"].append(
                self._sole_tilt(asset, feet_ids, yaw_quat).cpu().numpy()
            )

        # The commanded target and the raw action behind it, scattered by joint so that
        # every column aligns with joint_positions and joint_torques.
        targets, raw = self._commanded_targets(env, asset)
        if targets is not None:
            self.data["joint_position_targets"].append(targets.cpu().numpy())
        if raw is not None:
            self.data["joint_action_raw"].append(raw.cpu().numpy())

        available = self._available_torque(asset)
        if available is not None:
            self.data["joint_available_torque"].append(available.cpu().numpy())

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
            ("joint_saturation_effort", "_saturation_effort"),
        ):
            limits = torch.full(
                (asset.num_joints,), float("nan"), device=asset.device
            )
            found = False
            for actuator in getattr(asset, "actuators", {}).values():
                value = getattr(actuator, attr, None)
                if value is None and attr == "_saturation_effort":
                    value = getattr(getattr(actuator, "cfg", None),
                                    "saturation_effort", None)
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

    def _log_action_metadata(self, env, asset):
        """Record which action term drives which joint, and under what semantics.

        The commanded target is only interpretable beside the term that produced it. A
        JointPositionAction's processed action is an absolute joint angle in radians, a
        RelativeJointPositionAction's is an increment upon the current position, and a
        velocity term's is not a position at all, so the class name is dumped rather than
        assumed. This matters more than it did, since arms 7, 10 and 19 of
        plans/kscale_gait_shaping.md each remove a joint pair from the action term, after
        which the processed action is narrower than the articulation and a consumer that
        assumes the two are the same width reads the wrong column for every joint past the
        first removal.

        Args:
            env: The unwrapped ManagerBasedRLEnv.
            asset: The robot articulation.
        """
        manager = getattr(env, "action_manager", None)
        terms = getattr(manager, "_terms", None) if manager is not None else None
        if not terms:
            self._action_slots = []
            return
        slots, record = [], {}
        for name, term in terms.items():
            joint_ids = getattr(term, "_joint_ids", None)
            joint_names = getattr(term, "_joint_names", None)
            if joint_ids is None or joint_names is None:
                # A term that does not act in joint space, whose processed action cannot be
                # scattered by joint and is therefore recorded in the register alone.
                record[name] = {"class": type(term).__name__, "joints": None,
                                "action_dim": int(getattr(term, "action_dim", 0))}
                continue
            index = (torch.arange(asset.num_joints, device=asset.device)
                     if isinstance(joint_ids, slice)
                     else torch.as_tensor(list(joint_ids), device=asset.device))
            slots.append((name, index))
            record[name] = {"class": type(term).__name__,
                            "joints": list(joint_names),
                            "action_dim": int(getattr(term, "action_dim", len(index)))}
        self._action_slots = slots
        self.data["action_terms"] = record
        print(f"[play] action terms logged: "
              + ", ".join(f"{n} ({r['class']}, {r['action_dim']} dims)"
                          for n, r in record.items()))

    def _commanded_targets(self, env, asset):
        """The processed and the raw action of every joint space term, scattered by joint.

        Both are returned at the full articulation width with NaN in any column no term
        drives, so that they align column for column with joint_positions and a frozen
        joint is visibly absent rather than silently zero. Where two terms claim one joint,
        which no configuration here does, the later term in the manager's order wins and
        the register dumped by _log_action_metadata is what would reveal it.

        Args:
            env: The unwrapped ManagerBasedRLEnv.
            asset: The robot articulation.

        Returns:
            A pair of tensors of shape (num_envs, num_joints), or (None, None) where no
            term acts in joint space.
        """
        if self._action_slots is None:
            self._log_action_metadata(env, asset)
        if not self._action_slots:
            return None, None
        manager = env.action_manager
        shape = (asset.num_joints,)
        targets = torch.full((asset.data.joint_pos.shape[0],) + shape, float("nan"),
                             device=asset.device, dtype=asset.data.joint_pos.dtype)
        raw = torch.full_like(targets, float("nan"))
        for name, index in self._action_slots:
            term = manager._terms[name]
            processed = getattr(term, "processed_actions", None)
            if processed is not None and processed.shape[1] == index.numel():
                targets[:, index] = processed.to(targets.dtype)
            raw_actions = getattr(term, "raw_actions", None)
            if raw_actions is not None and raw_actions.shape[1] == index.numel():
                raw[:, index] = raw_actions.to(raw.dtype)
        return targets, raw

    def _available_torque(self, asset):
        """The velocity dependent torque bounds, as the actuator model itself computes them.

        This reproduces DCMotor._clip_effort at
        IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:295, which clamps the
        joint velocity to the speed at which the envelope meets the effort limit, takes the
        two sides of the torque speed line at that velocity, and bounds each by the flat
        effort limit. Both bounds are returned rather than one signed figure, since which
        of them binds depends on the sign of the torque being clipped and a single column
        forces the consumer to guess which convention was used. Saturation is then exactly
        the computed torque lying outside the pair, with no threshold and no dependence on
        the friction term that IdentifiedActuator subtracts after the clip.

        A joint whose actuator declares no saturation effort, meaning an ideal or implicit
        model, falls back to the flat effort limit on both sides, which is the correct
        envelope for such an actuator rather than an approximation of one.

        Args:
            asset: The robot articulation.

        Returns:
            A tensor of shape (num_envs, num_joints, 2) holding the lower and the upper
            bound, or None where no effort ceiling is known at all.
        """
        if self._envelope is None:
            effort = self.data.get("joint_effort_limits")
            if effort is None:
                self._envelope = ()
                return None
            device, dtype = asset.device, asset.data.joint_vel.dtype
            effort = torch.as_tensor(effort, device=device, dtype=dtype)
            saturation = torch.as_tensor(
                self.data.get("joint_saturation_effort", [float("nan")] * len(effort)),
                device=device, dtype=dtype)
            speed = torch.as_tensor(
                self.data.get("joint_velocity_limits", [float("nan")] * len(effort)),
                device=device, dtype=dtype)
            # A joint without a usable envelope is given the flat ceiling on both sides by
            # setting its saturation effort to its effort limit and its velocity limit to
            # infinity, which makes the torque speed line horizontal at the ceiling.
            flat = ~(torch.isfinite(saturation) & torch.isfinite(speed) & (speed > 0)
                     & (saturation > 0))
            saturation = torch.where(flat, effort, saturation)
            speed = torch.where(flat, torch.full_like(speed, float("inf")), speed)
            self._envelope = (saturation, effort, speed,
                              speed * (1.0 + effort / saturation))
        if not self._envelope:
            return None
        saturation, effort, speed, reach = self._envelope
        vel = torch.clamp(asset.data.joint_vel, min=-reach, max=reach)
        upper = torch.clamp(saturation * (1.0 - vel / speed), max=effort)
        lower = torch.clamp(saturation * (-1.0 - vel / speed), min=-effort)
        return torch.stack((lower, upper), dim=-1)

    @staticmethod
    def _sole_tilt(asset, feet_ids: list, yaw_quat: torch.Tensor) -> torch.Tensor:
        """The roll and pitch of each sole plane in the base yaw frame.

        The sole's ground facing normal is a fixed direction in the foot link's frame,
        declared per robot in _ROBOT_PROFILES because the axis differs between the two
        sole footed bipeds here and cannot be read off the sole point table. Rotating it
        into the world by the foot's orientation and then into the base yaw frame gives a
        unit vector that reads (0, 0, -1) for a sole lying flat on level ground, and the
        two angles below are the rotations about the forward and the lateral axis that
        carry it there. A positive roll is the sole's normal tipped toward the machine's
        left, which is the foot standing on its right edge.

        Args:
            asset: The robot articulation.
            feet_ids: Indices of the feet bodies.
            yaw_quat: The yaw only base orientation, of shape (num_envs, 4).

        Returns:
            The per foot (roll, pitch) in radians, of shape (num_envs, num_feet, 2).
        """
        foot_quat = asset.data.body_quat_w[:, feet_ids]  # (N, F, 4)
        num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]
        normal = torch.as_tensor(
            SOLE_NORMAL_L, dtype=foot_quat.dtype, device=foot_quat.device
        ).view(1, 1, 3).expand(num_envs, num_feet, 3)
        normal_w = math_utils.quat_apply(
            foot_quat.reshape(-1, 4), normal.reshape(-1, 3)
        ).view(num_envs, num_feet, 3)
        quat = yaw_quat.unsqueeze(1).expand(num_envs, num_feet, 4)
        normal_b = math_utils.quat_apply_inverse(
            quat.reshape(-1, 4), normal_w.reshape(-1, 3)
        ).view(num_envs, num_feet, 3)
        roll = torch.atan2(normal_b[..., 1], -normal_b[..., 2])
        pitch = torch.atan2(-normal_b[..., 0], -normal_b[..., 2])
        return torch.stack((roll, pitch), dim=-1)

    @staticmethod
    def _ground_height(points_w: torch.Tensor, ray_hits_w: torch.Tensor | None) -> torch.Tensor:
        """World height of the ground vertically beneath each point.

        One vertical ray is cast from 20 m above each point against the terrain mesh the
        height scanner has already loaded. This replaced, on 2026-10-21, a lookup of the
        highest of the four nearest scanner hits, which on the trunk mounted 0.1 m grid read
        the ground up to 28.5 mm high under a loaded foot on rough terrain and drifted by up
        to 5 mm under a planted one as the grid travelled with the base. A ray that misses
        the mesh, beyond the terrain border, reads the plane z = 0.

        Args:
            points_w: World frame query points, of shape (N, F, 3).
            ray_hits_w: The height scanner's ray hits, read only to establish that the task
                carries a scanner. None where it does not, in which case the ground is the
                plane z = 0.

        Returns:
            The ground height beneath each point, of shape (N, F).
        """
        if ray_hits_w is None:
            return torch.zeros_like(points_w[..., 2])
        starts = points_w.clone()
        starts[..., 2] += 20.0
        directions = torch.zeros_like(starts)
        directions[..., 2] = -1.0
        hits = raycast_mesh(starts, directions, mesh=RayCaster.meshes[TERRAIN_MESH_PRIM_PATH])[0]
        ground = hits.view_as(points_w)[..., 2].to(points_w.device)
        return torch.where(torch.isfinite(ground), ground, torch.zeros_like(ground))

    @staticmethod
    def _sole_clearance(asset, feet_ids: list, ray_hits_w: torch.Tensor | None = None) -> torch.Tensor:
        """Height of the lowest sole point above the ground beneath it, as in
        mdp.foot_clearance_reward_v4.

        Args:
            asset: The robot articulation.
            feet_ids: Indices of the feet bodies.
            ray_hits_w: The height scanner's ray hits, of shape (N, R, 3). None, the default,
                measures against the plane z = 0, which reproduces the world height the
                method returned before the scanner was read.

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
        lowest = pts_w[..., 2].argmin(dim=2, keepdim=True)  # (N, F, 1)
        # the ground is looked up beneath the lowest sole point, not beneath the ankle
        contact_w = torch.gather(pts_w, 2, lowest.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)
        return contact_w[..., 2] - DataLogger._ground_height(contact_w, ray_hits_w)  # (N, F)

    @staticmethod
    def _sole_clearance_pf(asset, feet_ids: list, ray_hits_w: torch.Tensor | None = None) -> torch.Tensor:
        """Height of the lowest point of a spherical foot above the ground beneath it.

        The lowest point of a sphere lies FOOT_RADIUS beneath its centre along the world
        vertical whatever the foot's orientation, so no rotation enters.

        Args:
            asset: The robot articulation.
            feet_ids: Indices of the feet bodies.
            ray_hits_w: The height scanner's ray hits, of shape (N, R, 3), or None to
                measure against the plane z = 0.

        Returns:
            The per foot clearance, of shape (num_envs, num_feet).
        """
        foot_pos = asset.data.body_pos_w[:, feet_ids]  # (N, F, 3)
        return foot_pos[..., 2] - FOOT_RADIUS - DataLogger._ground_height(foot_pos, ray_hits_w)

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
        # The actuator gains the policy was TRAINED under, for the same reason the
        # rewards are taken from the artefact. A policy replayed on a stiffness it never
        # saw is a different robot, and the mismatch is silent, the reward restoration
        # above succeeding normally while the gains disagree by any factor whatever.
        if args_cli.keep_tree_actuators:
            print("[WARNING] Actuator gains taken from the working tree by request, "
                  "the run's own gains are NOT applied.")
        else:
            gains = experiment_params.apply_actuator_cfg(env_cfg, env_params)
            if not gains["applied"] and not gains["missing"]:
                print("[WARNING] No actuator block in params/env.yaml, gains are the "
                      "working tree's and may differ from the run's.")
            else:
                print(f"[INFO] Actuators from run params, "
                      f"{len(gains['applied'])} groups applied.")
            for group, fields in gains["changed"].items():
                for field, (tree, run) in fields.items():
                    print(f"[INFO] Actuator '{group}' {field}, tree {tree} "
                          f"REPLACED BY run {run}")
            for group in gains["missing"]:
                print(f"[WARNING] Actuator group '{group}' is in the run and not in the "
                      f"tree, its gains could not be restored")
            for group in gains["extra"]:
                print(f"[WARNING] Actuator group '{group}' is in the tree and not in the "
                      f"run, it keeps the tree's gains")
            for group, fields in gains["structural"].items():
                for field, (tree, run) in fields.items():
                    print(f"[WARNING] Actuator '{group}' {field} differs, tree {tree} "
                          f"against run {run}, NOT changed")
        # The action scale the policy was TRAINED under. A replay at a different scale
        # commands a different joint offset for the same policy output, so the posture the
        # policy intends is not the posture the robot takes, and the failure is as silent
        # as the actuator one above.
        if args_cli.keep_tree_actions:
            print("[WARNING] Action transform taken from the working tree by request, "
                  "the run's own scale is NOT applied.")
        else:
            acts = experiment_params.apply_action_cfg(env_cfg, env_params)
            if not acts["applied"] and not acts["missing"]:
                print("[WARNING] No action block in params/env.yaml, the action scale is "
                      "the working tree's and may differ from the run's.")
            else:
                print(f"[INFO] Actions from run params, {len(acts['applied'])} terms applied.")
            for term, fields in acts["changed"].items():
                for field, (tree, run) in fields.items():
                    print(f"[INFO] Action '{term}' {field}, tree {tree} "
                          f"REPLACED BY run {run}")
            for term in acts["missing"]:
                print(f"[WARNING] Action term '{term}' is in the run and not in the tree, "
                      f"its transform could not be restored")
            for term in acts["extra"]:
                print(f"[WARNING] Action term '{term}' is in the tree and not in the run, "
                      f"it keeps the tree's transform")
            for term, fields in acts["structural"].items():
                for field, (tree, run) in fields.items():
                    print(f"[WARNING] Action '{term}' {field} differs, tree {tree} "
                          f"against run {run}, NOT changed")

    design_parameters, design_record = None, None
    dump_dir = log_dir
    if args_cli.copt_best_design:
        if args_cli.policy_type not in ("COPT", "COPT-MORAL"):
            raise ValueError("--copt_best_design applies only to the COPT and COPT-MORAL policy types")
        design_parameters, design_record = _best_design(log_dir, resume_path)
        dump_dir = os.path.join(log_dir, "best_design", f"model_{design_record['checkpoint_iteration']}")
        print(f"[INFO] Best design of generation {design_record['generation']}, history iteration "
              f"{design_record['history_iteration']}, for checkpoint iteration "
              f"{design_record['checkpoint_iteration']}: {design_parameters}")

    # instantiate data logger
    data_logger = DataLogger(dump_dir, args_cli.num_envs, agent_cfg.seed)
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
                dump_dir, "videos", "play", f"{agent_cfg.seed}"
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
    elif args_cli.policy_type in ("COPT", "COPT-MORAL"):

        _family = "quadruped" if "Isaac-Quadruped-Copt" in args_cli.task else "biped"
        if design_parameters is not None:
            _num_individuals = 1
            design_generator = FixedDesignGenerator(
                design_parameters,
                family=_family,
                output_dir=os.path.join(dump_dir, "copt_usds"),
            )
            # The builder geometry the run emitted its designs under, e.g. the quadruped's
            # shank_foot_gap of 0.001, recorded by train.py in params/copt_agent.yaml.
            run_copt = (experiment_params.load_params(log_dir, "copt_agent") or {}).get("copt", {})
            if run_copt.get("shank_foot_gap") is not None:
                design_generator.builder_kwargs = {"shank_foot_gap": run_copt["shank_foot_gap"]}
        else:
            _num_individuals = 256
            param_ranges = {}
            params = ["thigh_length_scale", "shank_length_scale"]
            for param in params:
                param_ranges[param] = DEFAULT_PARAM_RANGES[param]
            design_generator = CMAESDesignGenerator(
                family=_family,
                num_individuals=_num_individuals,
                param_ranges=param_ranges,
                sigma0=0.1,
                seed=42,
                late_start=False,
            )
        agent_cfg.policy.class_name = {"COPT": "CoptActorCritic", "COPT-MORAL": "CoptMorALActorCritic"}[
            args_cli.policy_type
        ]
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
    else:
        ppo_runner = OnPolicyRunner(
            env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
        )
    ppo_runner.load(resume_path)
    if design_parameters is not None:
        # Only learn() spawns a population, so the replay applies the design through the
        # runner's own generation zero pathway, respawn, nominal pose and reset, then the
        # actuator parameters. load() leaves current_population None, so sample_batch is
        # reached and is the base class's no-op.
        ppo_runner.respawn_robots = respawn_robots()
        with torch.inference_mode():
            ppo_runner._reload_morphology()

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
    if design_record is not None:
        with open(os.path.join(dump_dir, "data", f"{agent_cfg.seed}", "design.json"), "w") as handle:
            json.dump(design_record, handle, indent=2)

    # close the simulator
    env.close()


if __name__ == "__main__":
    EXPORT_POLICY = False
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
