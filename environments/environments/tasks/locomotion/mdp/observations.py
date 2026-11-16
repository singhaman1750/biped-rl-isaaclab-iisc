from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import (
    Camera,
    ContactSensor,
    Imu,
    RayCaster,
    RayCasterCamera,
    TiledCamera,
)
from isaaclab.sim.utils.stage import get_current_stage
from pxr import Usd, UsdPhysics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def robot_joint_torque(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """joint torque of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.applied_torque.to(device)


def robot_joint_acc(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """joint acc of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.joint_acc.to(device)


def robot_feet_contact_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg):
    """contact force of the robot feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    contact_force_tensor = contact_sensor.data.net_forces_w_history.to(device)
    return contact_force_tensor.view(contact_force_tensor.shape[0], -1)


def robot_mass(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # body_ids is slice(None) for a term that names no bodies.
    return asset.data.default_mass[:, asset_cfg.body_ids].to(device)


def robot_inertia(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """inertia of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    inertia_tensor = asset.data.default_inertia[:, asset_cfg.body_ids].to(device)
    return inertia_tensor.view(inertia_tensor.shape[0], -1)


def robot_joint_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """joint positions of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_pos.to(device)


def robot_joint_stiffness(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """joint stiffness of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_stiffness.to(device)


def robot_joint_damping(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """joint damping of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.default_joint_damping.to(device)


def robot_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """pose of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def robot_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """velocity of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_vel_w.to(device)


def robot_material_properties(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """material properties of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    material_tensor = asset.root_physx_view.get_material_properties().to(device)
    return material_tensor.view(material_tensor.shape[0], -1)


def robot_center_of_mass(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """center of mass of the robot"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    com_tensor = asset.root_physx_view.get_coms().clone().to(device)
    return com_tensor.view(com_tensor.shape[0], -1)


def robot_contact_force(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """The contact forces of the body."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    body_contact_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]

    return body_contact_force.reshape(body_contact_force.shape[0], -1)


def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as observation.

    The gait phase is represented by [sin(phase), cos(phase)] to ensure continuity.
    The phase is calculated based on the episode length and gait frequency.

    Returns:
        torch.Tensor: The gait phase observation. Shape: (num_envs, 2).
    """
    # check if episode_length_buf is available
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get the gait command from command manager
    command_term = env.command_manager.get_term("gait_command")
    # Calculate gait indices based on episode length
    gait_indices = torch.remainder(
        env.episode_length_buf * env.step_dt * command_term.command[:, 0], 1.0
    )
    # Reshape gait_indices to (num_envs, 1)
    gait_indices = gait_indices.unsqueeze(-1)
    # Convert to sin/cos representation
    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: The gait command parameters [frequency, offset, duration].
                     Shape: (num_envs, 3).
    """
    return env.command_manager.get_command(command_name)


def robot_base_pose(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """pose of the robot base"""
    asset: Articulation = env.scene[asset_cfg.name]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return asset.data.root_pos_w.to(device)


def feet_lin_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)


def joint_pos_rel_exclude_wheel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_joints_name: list[str] = ["wheel_[RL]_Joint"],
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)

    asset: Articulation = env.scene[asset_cfg.name]
    wheel_joints_idx = asset.find_joints(wheel_joints_name)[0]
    all_joints_idx = range(asset.num_joints)
    pos_idx_exclude_wheel = [i for i in all_joints_idx if i not in wheel_joints_idx]
    return (
        asset.data.joint_pos[:, pos_idx_exclude_wheel]
        - asset.data.default_joint_pos[:, pos_idx_exclude_wheel]
    )


class robot_link_lengths(ManagerTermBase):
    """Per-env link lengths derived from body-frame world positions.

    For each (parent, child) pair, returns ``||body_link_pos_w[child] - body_link_pos_w[parent]||``.
    The norm is invariant under joint angles for chains of revolute joints (every joint between
    a URDF parent link and its kinematic child is revolute or fixed in this robot), so the value
    is a per-morphology constant that can be cached.

    The cache is invalidated via ``reset()``, which is called by ``ObservationManager.reset()``
    on every per-env episode termination and after ``respawn_robots()`` (whose final ``env.reset()``
    routes through the same cascade). The next ``__call__`` then recomputes for all envs, reflecting
    the current morphology.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._cache: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # Cache is shared across all envs (one tensor of shape (num_envs, k));
        # nuke regardless of which envs reset. Recompute cost is one slice + norm.
        self._cache = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        parent_body_names: list[str],
        child_body_names: list[str],
    ) -> torch.Tensor:
        if self._cache is None:
            asset: Articulation = env.scene[asset_cfg.name]
            parent_ids, _ = asset.find_bodies(parent_body_names, preserve_order=True)
            child_ids, _ = asset.find_bodies(child_body_names, preserve_order=True)
            delta = (
                asset.data.body_link_pos_w[:, child_ids]
                - asset.data.body_link_pos_w[:, parent_ids]
            )
            self._cache = delta.norm(dim=-1).clone()
            torch.set_printoptions(precision=10)
        return self._cache


class robot_nominal_pose(ManagerTermBase):
    """The fold of each named body about +y against its parent at zero joint position, (num_envs, num_bodies).

    The co-optimisation documents fold the leg in the body frames (``pose_frames`` of
    both design_generator builders), so ``default_joint_pos`` is zero for every design
    and the nominal pose lives in the joint frames alone. Neither ``ArticulationData``
    nor the PhysX articulation view exposes a joint frame, so the fold is read on the
    stage from each body's incoming joint as ``localRot0 * localRot1^-1``, the
    importer authoring ``localRot0`` as the body quat composed with ``localRot1``
    (MjcfImporter.cpp:1341-1366). The read runs whenever the articulation has been
    re-initialised, which a respawn and an in place swap both do, since either may
    change the frames.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._view = None
        self._fold: torch.Tensor | None = None

    def __call__(self, env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        if self._view is not asset.root_physx_view:
            self._fold = self._read(env, asset, asset_cfg)
            self._view = asset.root_physx_view
        return self._fold

    @staticmethod
    def _read(env: ManagerBasedEnv, asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        stage = get_current_stage()
        ids = asset_cfg.body_ids
        bodies = asset.body_names[ids] if isinstance(ids, slice) else [asset.body_names[i] for i in ids]
        rel = asset.cfg.prim_path[len(env.scene.env_regex_ns):]
        root = stage.GetPrimAtPath(env.scene.env_prim_paths[0] + rel)
        joints = {}
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.Joint):
                targets = UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
                if targets and targets[0].name in bodies:
                    joints[targets[0].name] = prim.GetPath().MakeRelativePath(root.GetPath()).pathString
        missing = [b for b in bodies if b not in joints]
        if missing:
            raise ValueError(f"no joint under {root.GetPath()} has {missing} as its child body")
        fold = torch.empty(env.num_envs, len(bodies), dtype=torch.float64)
        for i, env_path in enumerate(env.scene.env_prim_paths):
            for k, body in enumerate(bodies):
                joint = stage.GetPrimAtPath(f"{env_path}{rel}/{joints[body]}")
                q = joint.GetAttribute("physics:localRot0").Get() * joint.GetAttribute("physics:localRot1").Get().GetInverse()
                x, y, z = q.GetImaginary()
                if max(abs(x), abs(z)) > 1e-4:
                    raise ValueError(f"{joint.GetPath()} folds about an axis other than y, {q}")
                fold[i, k] = math.remainder(2.0 * math.atan2(y, q.GetReal()), 2.0 * math.pi)
        return fold.to(device=env.device, dtype=torch.float32)
