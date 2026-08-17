"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)

def foot_landing_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        foot_radius: float,
        about_landing_threshold: float,
) -> torch.Tensor:
    """Penalize high foot landing velocities"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1

    foot_heights = torch.clip(
    asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain

    about_to_land = (foot_heights < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    reward = torch.sum(torch.square(landing_z_vels), dim=1)
    return reward

def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str, 
    sensor_cfg: SceneEntityCfg, 
    threshold_min: float,
    threshold_max: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # no reward for large steps
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.

    Each foot contributes a Gaussian kernel on its height error, exp(-(h - target)^2 / std^2),
    gated multiplicatively by a tanh of its horizontal speed. The reward is zero for a
    stationary foot, near zero for a foot far from the target height, and maximal for a fast
    swinging foot at the target clearance.

    The height is that of the body frame origin, which equals the sole clearance only for a
    point foot or a foot held level. On a sole foot the frame sits at the ankle, so tilting
    the foot about the ankle raises the frame while the sole edge stays on the ground, and
    the term cannot by itself distinguish a lifted foot from a tilted one. Passing
    ``sensor_cfg`` removes that ambiguity by paying nothing for a foot that is in contact.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset, resolving the feet bodies.
        target_height: Desired body frame height at swing apex (m).
        std: Width of the Gaussian height kernel (m).
        tanh_mult: Scaling applied to the horizontal foot speed inside the tanh gate.
        sensor_cfg: Optional contact sensor configuration. When given, a foot in contact
            with the ground earns nothing whatever its measured height, which prevents the
            term from being collected by rocking a grounded foot onto its edge. Its bodies
            must resolve in the same order as those of ``asset_cfg``. Defaults to None,
            preserving the original ungated behaviour.
        force_threshold: Contact force magnitude (N) above which a foot counts as grounded.
            Only used when ``sensor_cfg`` is given.

    Returns:
        The computed reward tensor, summed over the feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.exp(-foot_z_target_error / std**2) * foot_velocity_tanh
    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # history max, as in feet_slide, so contact chatter cannot flicker a grounded foot into earning
        in_contact = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
            > force_threshold
        )
        reward = reward * ~in_contact
    return torch.sum(reward, dim=1)

def foot_clearance_reward_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sole_offsets: list[list[float]],
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height, measured at the sole.

    Unlike :func:`foot_clearance_reward`, which uses the body frame origin height as a proxy
    for clearance, this term transforms a set of sole points through the foot's world
    orientation and takes their lowest world height. That quantity is the true clearance of
    the foot above the ground and is invariant to how the foot is tilted, which closes the
    exploit whereby a sole footed robot rocks a grounded foot onto its edge to raise the
    frame to the target while never lifting the foot. Consequently ``target_height`` here is
    the desired SOLE clearance, not a body frame height, and is a different quantity from the
    ``target_height`` of v1.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset, resolving the feet bodies.
        target_height: Desired sole clearance above the ground at swing apex (m).
        std: Width of the Gaussian clearance kernel (m).
        tanh_mult: Scaling applied to the horizontal foot speed inside the tanh gate.
        sole_offsets: Points on the sole, in the foot body frame, whose lowest world height
            defines the clearance. Supply the support set of the foot's contact geometry, it
            suffices to cover the points that can ever be the lowest under the reachable foot
            orientations. For SD_BRS1 the sole face lies at z -0.124 spanning x -0.1091 to
            0.1521 and y +/-0.0970, with chamfered fore and aft edges rising to z -0.1144 at
            x -0.1262 and 0.1692, and the twelve point table in the environment config
            reproduces the collision mesh to within 0.85 mm over the reachable range.
        sensor_cfg: Optional contact sensor configuration. When given, a foot in contact
            earns nothing. This is redundant with the sole measurement and is retained as
            defence in depth against geometry not covered by ``sole_offsets``. Its bodies
            must resolve in the same order as those of ``asset_cfg``. Defaults to None.
        force_threshold: Contact force magnitude (N) above which a foot counts as grounded.
            Only used when ``sensor_cfg`` is given.

    Returns:
        The computed reward tensor, summed over the feet.

    Note:
        The clearance is an absolute world height, which equals the height above the terrain
        only on flat ground. On generated terrain it must be referenced to the terrain height
        beneath the foot, as the TODO in :func:`feet_regulation` also records.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (N, F, 3)
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, F, 4)
    num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]

    offsets = torch.as_tensor(sole_offsets, dtype=foot_pos.dtype, device=foot_pos.device)  # (P, 3)
    num_pts = offsets.shape[0]
    # rotate every sole point by its foot's orientation, then offset by the foot position
    quat = foot_quat.unsqueeze(2).expand(num_envs, num_feet, num_pts, 4)
    pts = offsets.view(1, 1, num_pts, 3).expand(num_envs, num_feet, num_pts, 3)
    pts_w = math_utils.quat_apply(quat.reshape(-1, 4), pts.reshape(-1, 3)).view(num_envs, num_feet, num_pts, 3)
    pts_w = pts_w + foot_pos.unsqueeze(2)
    sole_clearance = pts_w[..., 2].min(dim=2)[0]  # (N, F)

    clearance_error = torch.square(sole_clearance - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.exp(-clearance_error / std**2) * foot_velocity_tanh
    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        # history max, as in feet_slide, so contact chatter cannot flicker a grounded foot into earning
        in_contact = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
            > force_threshold
        )
        reward = reward * ~in_contact
    return torch.sum(reward, dim=1)


def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)



def unbalance_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if the feet air time variance exceeds the balance threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    return torch.var(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], dim=-1)


def unbalance_feet_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the variance of feet maximum height using sensor positions."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_positions = contact_sensor.data.pos_w[:, sensor_cfg.body_ids]

    if feet_positions is None:
        return torch.zeros(env.num_envs)

    feet_heights = feet_positions[:, :, 2]
    max_feet_heights = torch.max(feet_heights, dim=-1)[0]
    height_variance = torch.var(max_feet_heights, dim=-1)
    return height_variance


# def feet_distance(
#     env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize if the distance between feet is below a minimum threshold."""

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions = asset.data.joint_pos[sensor_cfg.body_ids]

#     if feet_positions is None:
#         return torch.zeros(env.num_envs)

#     # feet distance on x-y plane
#     feet_distance = torch.norm(feet_positions[0, :2] - feet_positions[1, :2], dim=-1)

#     return torch.clamp(0.1 - feet_distance, min=0.0)


def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str]=["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,)-> torch.Tensor:
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward

def nominal_foot_position(env: ManagerBasedRLEnv, command_name: str,
                          base_height_target: float,
                           asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Compute the nominal foot position"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_center_b = torch.mean(feet_pos_b[:, :, :3], dim=1)
    base_height_error = torch.abs((feet_center_b[:, 2] - env._foot_radius + base_height_target))

    reward = torch.exp(-base_height_error / (std**2 + 1e-6))
    return reward

def leg_symmetry(env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / (std**2 + 1e-6))

def same_feet_x_position(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    # return torch.exp(-feet_x_distance / 0.2)
    return feet_x_distance

def no_fly(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    history_index: int = -1,
) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground.

    Args:
        env: The environment object.
        sensor_cfg: Configuration for the contact force sensor.
        threshold: Contact force magnitude (N) above which a foot counts as grounded.
        history_index: Which slot of the contact sensor's rolling history supplies the
            contact test. The sensor writes the NEWEST sample to index 0 and the oldest to
            the tail, as its own docstring states, so 0 is the current frame and the
            default of -1 is the OLDEST of the four buffered samples, roughly 15 ms stale.
            Defaults to -1, preserving the original behaviour for existing callers exactly.
            Pass 0 for the current frame.

    Returns:
        The computed reward tensor.

    Note:
        DEFECT left standing in the default. Reading the tail is a porting error, not a
        design. The Isaac Gym ancestor of this term has no history axis at all and tests
        contact instantaneously, no first-party IsaacLab code reads a trailing index, and
        this function itself read index 0 correctly until an unrelated commit regressed it.
        The default is retained only so that callers which have not opted in keep their
        behaviour bit for bit. Set ``history_index=0`` deliberately. Full provenance in
        /ws/NATURAL_GAIT_PLAN.md section 5.2.6.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]

    contacts = torch.norm(latest_contact_forces[:, history_index], dim = -1) > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1
    no_contact = torch.sum(contacts.float(), dim=1) == 0

    return 1.0 * single_contact - 5.0 * no_contact

def keep_ankle_pitch_zero_in_air(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["ankle_L_Joint", "ankle_R_Joint"]),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["ankle_[LR]_Link"]),
    force_threshold: float = 2.0,
    pitch_scale: float = 0.2,
    require_airborne: bool = False,
) -> torch.Tensor:
    """Reward for keeping ankle pitch angle close to zero when foot is in the air.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset. Must specify joint_names for the
            ankle pitch joints in the same order as sensor_cfg body_names (L before R).
        sensor_cfg: Configuration for the contact force sensor.
        force_threshold: Threshold value for contact detection (in Newtons).
        pitch_scale: Scaling factor for the exponential reward.
        require_airborne: If True, the reward is zero unless at least one foot is airborne.
            The pitch sum below runs only over airborne feet, so with every foot in contact
            it is empty and the exponential saturates at its maximum of 1, paying a standing
            bonus for keeping both feet planted. Defaults to False, preserving that original
            behaviour for existing callers.

    Returns:
        The computed reward tensor.

    """
    asset = env.scene[asset_cfg.name]
    contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_forces_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_forces_history[:, -2], dim=-1) > force_threshold
    contact_filt = torch.logical_or(current_contact, last_contact)
    # Use resolved joint_ids (shape: num_envs x num_ankle_joints) instead of hardcoded indices
    ankle_pitch = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids])  # (N, 2)
    weighted_ankle_pitch = torch.sum(ankle_pitch * ~contact_filt, dim=1)
    reward = torch.exp(-weighted_ankle_pitch / (pitch_scale + 1e-6))
    if require_airborne:
        reward = reward * torch.any(~contact_filt, dim=1).float()
    return reward

def knee_flexion_in_swing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target: float = 0.6,
    std: float = 0.2,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the swing leg's knee for approaching a flexed target while the foot is airborne.

    A straight stance knee transmits load efficiently and is left untouched, so this term is
    gated to swing and shapes only the airborne knee, supplying the swing flexion signal that
    no other reward provides. The knee joints resolved by ``asset_cfg`` must appear in the same
    order as the feet bodies resolved by ``sensor_cfg``, left before right, so each knee is
    paired with its own foot.

    Args:
        env: The environment object.
        asset_cfg: Robot asset, ``joint_names`` resolving the two knee pitch joints.
        sensor_cfg: Contact sensor, ``body_names`` resolving the two feet in the same order.
        target: Swing knee flexion target (rad), inside the knee range [0, 1.483].
        std: Width of the Gaussian flexion kernel (rad).
        force_threshold: Contact force magnitude (N) above which a foot counts as grounded.

    Returns:
        The reward tensor, summed over the two legs, in the interval [0, 2].

    Note:
        DEFECT, recorded for anyone tempted to reuse this form. Trained as run
        2026-07-23_11-31-57 with target 1.1, std 0.2 and weight 10, this term produced
        NOTHING. Its logged value never exceeded 1.05e-6 against a saturation of 10,
        because the policy's swing knee sits near 0.22 rad and the kernel placed the target
        0.88 rad away with a tolerance of 0.2, i.e. 4.4 tolerances into the tail, so the
        gradient at the operating point was 1.7e-6 per radian against 11.1 for a monotone
        reward of the same weight. A peaked kernel only instructs where it has a
        derivative, and this one had none anywhere the policy stood. Prefer
        :func:`knee_flexion_in_swing_v2`, whose gradient is constant over its whole active
        range. Full post-mortem in /ws/NATURAL_GAIT_PLAN.md section 2.6.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_history[:, -2], dim=-1) > force_threshold
    airborne = ~torch.logical_or(current_contact, last_contact)  # (N, F), True while swinging
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]          # (N, F)
    reward = torch.exp(-torch.square(knee - target) / std**2) * airborne.float()
    return torch.sum(reward, dim=1)


def knee_flexion_in_swing_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    nominal: float = 0.4814,
    cap: float = 0.9,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the swing leg's knee for flexing beyond its stance nominal, monotonically.

    A straight stance knee transmits load efficiently and is left untouched, so this term
    is gated to swing and shapes only the airborne knee. Unlike
    :func:`knee_flexion_in_swing`, which places a Gaussian at a fixed target and therefore
    vanishes wherever the policy is not already near it, this term is a monotone ramp from
    ``nominal`` to ``cap``. Its derivative is the constant ``1 / (cap - nominal)`` over that
    whole interval, so it carries usable gradient from the moment training begins, which is
    the property the v1 form lacked. It is a DIFFERENT quantity from v1, rewarding flexion
    RELATIVE to the stance pose rather than proximity to an absolute angle, which is why it
    is a new function rather than an optional argument.

    The knee joints resolved by ``asset_cfg`` must appear in the same order as the feet
    bodies resolved by ``sensor_cfg``, left before right, so each knee is paired with its
    own foot.

    Args:
        env: The environment object.
        asset_cfg: Robot asset, ``joint_names`` resolving the two knee pitch joints.
        sensor_cfg: Contact sensor, ``body_names`` resolving the two feet in the same order.
        nominal: Knee angle (rad) below which no reward is paid, normally the stance
            nominal carried by the default pose, so the term asks only for flexion beyond
            the posture the robot already holds.
        cap: Knee angle (rad) at which the ramp saturates, a natural swing flexion.
        force_threshold: Contact force magnitude (N) above which a foot counts as grounded.

    Returns:
        The reward tensor, summed over the two legs, in the interval [0, 2].

    Note:
        The airborne gate reads indices 0 and 1 of the contact history, the two NEWEST
        samples, giving the current-or-previous debounce that the Isaac Gym ancestor of
        this family expresses as ``contact_filt = contact OR last_contacts``. It takes no
        ``history_index`` argument because, unlike :func:`no_fly` and
        :func:`keep_ankle_pitch_zero_in_air`, this function is SD_BRS1 exclusive and has no
        prior run whose behaviour must be preserved, so the correct indexing is simply the
        only one it has ever had. Provenance in /ws/NATURAL_GAIT_PLAN.md section 5.2.6.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_history[:, 0], dim=-1) > force_threshold
    last_contact = torch.norm(contact_history[:, 1], dim=-1) > force_threshold
    airborne = ~torch.logical_or(current_contact, last_contact)  # (N, F), True while swinging
    knee = asset.data.joint_pos[:, asset_cfg.joint_ids]          # (N, F)
    ramp = torch.clamp((knee - nominal) / (cap - nominal), min=0.0, max=1.0)
    return torch.sum(ramp * airborne.float(), dim=1)


def no_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize if both feet are not in contact with the ground.
    """

    # Access the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the latest contact forces in the z direction (upward direction)
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # shape: (env_num, 2)

    # Determine if each foot is in contact
    contacts = latest_contact_forces > 1.0  # Returns a boolean tensor where True indicates contact

    return (torch.sum(contacts.float(), dim=1) == 0).float()


def stand_still(
    env, lin_threshold: float = 0.05, ang_threshold: float = 0.05, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    penalizing linear and angular motion when command velocities are near zero.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command("base_velocity")

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]

    reward_lin = torch.sum(
        torch.abs(base_lin_vel) * (torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold), dim=-1
    )

    reward_ang = torch.abs(base_ang_vel) * (torch.abs(ang_commands) < ang_threshold)

    total_reward = reward_lin + reward_ang
    return total_reward


# def feet_regulation(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     desired_body_height: float = 0.65,
# ) -> torch.Tensor:
#     """Penalize if the feet are not in contact with the ground.

#     Args:
#         env: The environment object.
#         sensor_cfg: The configuration of the contact sensor.
#         desired_body_height: The desired body height used for normalization.

#     Returns:
#         A tensor representing the feet regulation penalty for each environment.
#     """

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions_z = asset.data.joint_pos[sensor_cfg.body_ids, 2]

#     feet_vel_xy = asset.data.joint_vel[sensor_cfg.body_ids, :2]

#     vel_norms_xy = torch.norm(feet_vel_xy, dim=-1)

#     exp_term = torch.exp(-feet_positions_z / (0.025 * desired_body_height))

#     r_fr = torch.sum(vel_norms_xy**2 * exp_term, dim=-1)

#     return r_fr

def feet_regulation(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
    base_height_target: float,
    height_decay_scale: float | None = None,
) -> torch.Tensor:
    """Penalise horizontal foot speed, gated by an exponential in the foot's ground clearance.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset, resolving the feet bodies.
        foot_radius: Distance from the measured body frame origin down to the sole, used to
            convert the body height into a ground clearance. This is a per robot geometric
            constant, for SD_BRS1 the sole sits 0.124 m below the Link6 frame.
        base_height_target: Target base height, used as the height gate length scale when
            ``height_decay_scale`` is not given.
        height_decay_scale: Length scale (m) of the exponential height gate. Defaults to
            None, which reproduces the original behaviour of using ``base_height_target``,
            a scale an order of magnitude larger than the swing height band that leaves the
            gate near unity throughout and so penalises a swinging foot almost as heavily as
            a grounded one. A value near 0.025 * base_height_target, the ratio of the
            original formulation, confines the penalty to feet at ground level.

    Returns:
        The computed penalty tensor.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    if height_decay_scale is None:
        height_decay_scale = base_height_target
    height_scale = torch.exp(-feet_height / (height_decay_scale + 1e-6))
    reward = torch.sum(height_scale * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1)
    return reward


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height above the terrain from its target using L2 squared kernel.

    Uses a ray-caster sensor to measure the terrain surface elevation at multiple points
    beneath the robot. The robot's height is computed as its world-frame z-position minus
    the mean terrain surface height across all ray hits, giving the clearance above the
    local terrain. This correctly handles uneven terrain: a robot standing at the correct
    height on a slope or rough surface incurs zero penalty, whereas the same measurement
    using an absolute world-frame height would penalize it spuriously.

    Args:
        env: The RL environment instance.
        target_height: The desired clearance of the robot base above the terrain (meters).
        sensor_cfg: Configuration for the ray-caster sensor used to measure terrain height.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Per-environment L2 squared penalty: (mean_terrain_clearance - target_height)^2.

    Note:
        Ray hits at infinity (e.g. rays that miss all geometry) are replaced with
        ``target_height`` before averaging, so missed rays do not distort the penalty.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height above terrain at each ray sample: root_z - terrain_z (shape: N x num_rays)
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # replace non-finite values (rays missing geometry) with the target so they are neutral
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    # mean clearance over all ray samples, then squared deviation from target
    return torch.square(height.mean(dim=1) - target_height)


def base_com_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Combine rewards
        total_reward = force_reward + velocity_reward
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / (durations[stance_idxs] + 1e-6))
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs] + 1e-6)
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale


class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def reset(self, env_ids=None) -> None:
        """Reset action history for the specified environments.

        Called automatically by the RewardManager on episode reset.

        Args:
            env_ids: Indices of environments to reset. Defaults to None,
                in which case all environments are reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        if self.prev_action is not None:
            self.prev_action[env_ids] = 0.0
        if self.prev_prev_action is not None:
            self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty


class JointTorqueRatePenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in joint torques.
    This penalty encourages smoother actuation and reduces joint torque chattering.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.prev_torque = None

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        if self.prev_torque is not None:
            self.prev_torque[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        current_torque = self.asset.data.applied_torque[:, self.asset_cfg.joint_ids].clone()
        if self.prev_torque is None:
            self.prev_torque = current_torque
            return torch.zeros(current_torque.shape[0], device=current_torque.device)

        penalty = torch.sum(torch.square(current_torque - self.prev_torque), dim=1)
        self.prev_torque = current_torque

        startup_env_mask = env.episode_length_buf < 2
        penalty[startup_env_mask] = 0

        return penalty

