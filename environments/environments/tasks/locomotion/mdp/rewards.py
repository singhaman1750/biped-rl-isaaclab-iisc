"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import distributions, where
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
    """Penalize high foot landing velocities

    Note:
        THREE DEFECTS are left standing here for a sole foot robot. Use
        :func:`foot_landing_vel_v2` on a sole footed robot.

        1. The height gate is the FRAME PROXY ``body_pos_w[z] - foot_radius``, which assumes
           the sole sits a constant distance below the frame and is therefore exact only for
           a level foot. This is the same proxy that :func:`foot_clearance_reward` used and
           that :func:`foot_clearance_reward_v2` was written to replace.
        2. The penalised quantity is the FRAME vertical velocity, not the approach velocity
           of the sole. The two differ by the rotational term omega x r.
        3. The term is a TIME INTEGRAL of the squared velocity over a wide gate, so it is
           minimised by descending slowly through the upper part of the window rather than by
           arriving softly.
    """
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

def _sole_points_world(
    asset: Articulation,
    body_ids: list[int] | slice,
    sole_offsets: list[list[float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform a table of sole points into the world frame.

    Shared by :func:`foot_clearance_reward_v3` and :func:`foot_landing_vel_v2` so that the
    clearance the one rewards and the clearance the other gates on cannot drift apart.

    Args:
        asset: The articulation carrying the feet.
        body_ids: Resolved body indices of the feet.
        sole_offsets: Points on the sole in the foot body frame, as documented on
            :func:`foot_clearance_reward_v2`.

    Returns:
        A tuple ``(pts_w, r_w)`` where ``pts_w`` is (N, F, P, 3), the world positions of the
        sole points, and ``r_w`` is (N, F, P, 3), the world frame lever arms from each foot's
        frame origin to those points.
    """
    foot_pos = asset.data.body_pos_w[:, body_ids]                       # (N, F, 3)
    foot_quat = asset.data.body_quat_w[:, body_ids]                     # (N, F, 4)
    num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]

    offsets = torch.as_tensor(sole_offsets, dtype=foot_pos.dtype, device=foot_pos.device)
    num_pts = offsets.shape[0]
    quat = foot_quat.unsqueeze(2).expand(num_envs, num_feet, num_pts, 4)
    pts = offsets.view(1, 1, num_pts, 3).expand(num_envs, num_feet, num_pts, 3)
    r_w = math_utils.quat_apply(quat.reshape(-1, 4), pts.reshape(-1, 3)).view(
        num_envs, num_feet, num_pts, 3
    )
    return r_w + foot_pos.unsqueeze(2), r_w


def foot_landing_vel_v2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    sole_offsets: list[list[float]],
    about_landing_threshold: float,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalise the approach velocity of a sole that is about to land.

    Where :func:`foot_landing_vel` gates on a frame height proxy and penalises the frame's
    vertical velocity, this gates on the true sole clearance and penalises the vertical
    velocity OF THE LOWEST SOLE POINT, which is the quantity that governs the collision.

    The clearance is the lowest world height over ``sole_offsets``, exactly as
    :func:`foot_clearance_reward_v2` computes it. The penalised velocity is
    ``v_frame + omega x r`` evaluated at that lowest point, so a foot rotating its sole into
    the ground is charged for the rotation.

    The term remains a time integral and is therefore still, in principle, reducible by
    dawdling inside the gate. The remedy adopted is to size ``about_landing_threshold`` to
    the terminal approach rather than to the whole descent, since the free fall velocity from
    the threshold height bounds what an unpowered descent can deliver
    and a policy that wishes to arrive faster than that must pay to accelerate.

    Args:
        env: The environment object.
        asset_cfg: Robot asset configuration resolving the feet bodies.
        sensor_cfg: Contact sensor configuration resolving the same feet, in the same order.
        sole_offsets: Points on the sole in the foot body frame whose lowest world height is
            the clearance. Pass the same table as ``rew_foot_clearance``.
        about_landing_threshold: Sole clearance (m) below which a descending, unloaded foot
            is charged. This is a TRUE clearance and is therefore not comparable with the v1
            argument of the same name, which was a frame proxy standing 23.5 mm above it.
        force_threshold: Contact force magnitude (N) above which a foot counts as landed and
            is exempt. Defaults to 1.0, matching ``rew_foot_clearance``.

    Returns:
        The computed penalty tensor, summed over the feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pts_w, r_w = _sole_points_world(asset, asset_cfg.body_ids, sole_offsets)

    # lowest sole point per foot, and the lever arm that reaches it
    clearance, lowest = pts_w[..., 2].min(dim=2)                        # (N, F), (N, F)
    idx = lowest.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)
    r_low = torch.gather(r_w, 2, idx).squeeze(2)                        # (N, F, 3)

    lin_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]     # (N, F, 3)
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]     # (N, F, 3)
    # vertical component of v_point = v_link + omega x r
    approach_vel = lin_vel[..., 2] + (
        ang_vel[..., 0] * r_low[..., 1] - ang_vel[..., 1] * r_low[..., 0]
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > force_threshold
    )

    about_to_land = (clearance < about_landing_threshold) & (~in_contact) & (approach_vel < 0.0)
    landing_vel = torch.where(about_to_land, approach_vel, torch.zeros_like(approach_vel))
    return torch.sum(torch.square(landing_vel), dim=1)


def feet_impact_force(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
    clip: float = 2000.0,
) -> torch.Tensor:
    """Penalise per-foot contact force in excess of a threshold, inert below it.

    The form follows Humanoid-Gym (arXiv 2404.05695), whose large-contact-force term is
    max(F - F_thr, 0) clipped above. Being exactly zero over the range of forces a well
    behaved stance produces, it cannot distort the stance phase it is not meant to govern
    and acts only on the collision. The maximum over the contact history axis is taken so
    that a transient falling between two control steps is not missed.

    Args:
        env: The environment object.
        sensor_cfg: Contact sensor configuration resolving the feet bodies.
        force_threshold: Force (N) below which the term is exactly zero. For SD_BRS1 the
            body weight is 587 N and the steady single-support stance load is about that,
            so a threshold near 900 N admits normal stance and catches the collision.
        clip: Upper bound (N) on the per-foot excess, so that one pathological contact
            cannot dominate a batch. Defaults to 2000.0.

    Returns:
        The computed penalty tensor, summed over the feet.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    peak = forces.norm(dim=-1).max(dim=1)[0]                      # (N, F)
    return torch.sum(torch.clip(peak - force_threshold, 0.0, clip), dim=1)

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


def _swing_phase_from_gait_command(gait_params: torch.Tensor, episode_length_buf: torch.Tensor,
                                   dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the per foot swing phase from the gait command.

    This mirrors :meth:`GaitReward.compute_contact_targets` exactly in its treatment of the
    frequency, offset and duration parameters, so that a reward keyed to the swing phase and
    the clock that grades contact cannot disagree about when a foot is swinging.

    Args:
        gait_params: The gait command tensor, columns being frequency, offset, duration and
            swing height, as written by :class:`UniformGaitCommand`.
        episode_length_buf: The environment's episode step counter.
        dt: The control period (s).

    Returns:
        A tuple ``(in_swing, phi_swing)``, both (N, 2), the first a boolean mask of the feet
        the clock currently assigns to swing and the second the within swing phase, rising
        from 0 at lift off to 1 at touchdown. ``phi_swing`` is meaningless where ``in_swing``
        is False and is returned clamped to [0, 1] so it is safe to use unmasked.
    """
    num_envs = gait_params.shape[0]
    frequencies = gait_params[:, 0]
    offsets = gait_params[:, 1]
    durations = gait_params[:, 2].view(num_envs, 1).expand(num_envs, 2)

    gait_indices = torch.remainder(episode_length_buf * dt * frequencies, 1.0)
    foot_indices = torch.remainder(
        torch.cat(
            [gait_indices.view(num_envs, 1), (gait_indices + offsets + 1).view(num_envs, 1)],
            dim=1,
        ),
        1.0,
    )

    in_swing = foot_indices > durations
    phi_swing = torch.clamp((foot_indices - durations) / (1.0 - durations + 1e-6), 0.0, 1.0)
    return in_swing, phi_swing


def foot_clearance_reward_v3(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    std: float,
    sole_offsets: list[list[float]],
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward the swing foot for tracking a raised cosine clearance reference.

    Where :func:`foot_clearance_reward_v2` places a Gaussian at a single target clearance and
    multiplies by foot speed, this tracks a reference that is a function of the swing phase.
    A kernel on a set point is maximised by the trajectory that reaches the set point soonest
    and leaves it latest, which is a trapezoid. A reference penalises a foot
    that is high at the wrong moment exactly as one that is low at the wrong moment, so it
    specifies the whole path rather than its extremum, and the trapezoid stops being the
    maximiser rather than merely becoming expensive.

    The form follows Humanoid-Gym (arXiv:2404.05695), which drives its joints from a phase
    conditioned sinusoid rather than from an extremum, and the reference tracking foot height
    term of Seo et al. (arXiv:2512.01996). The reference is the RAISED COSINE 
    ``swing_height * sin^2(pi * phi)``, equivalently
    ``0.5 * swing_height * (1 - cos(2 pi phi))``.

    The reference amplitude is read from the gait command's ``swing_height``, and the phase
    from the same frequency, offset and duration parameterisation that :class:`GaitReward`
    uses, so the reward and the clock cannot drift apart. The SD_BRS1 configuration already
    declares ``swing_height`` at 0.08 m and no reward has ever read it.

    The term is evaluated only over the feet the clock assigns to swing. During commanded
    stance it returns zero rather than rewarding a clearance of zero, since the latter would
    pay a robot for standing still.

    Args:
        env: The environment object.
        asset_cfg: Robot asset configuration resolving the feet bodies.
        command_name: Name of the gait command term supplying frequency, offset, duration and
            swing height.
        std: Width of the Gaussian tracking kernel (m). Size it to the error the policy
            currently produces, not to the task range, per the kernel width lesson of
            ../context/literature.md cluster 11 and the dead Phase A knee reward of
            /ws/plans/NATURAL_GAIT_PLAN.md, whose kernel sat 4.4 tolerances from the operating
            point and delivered a gradient of order 1e-6. The measured swing clearance
            standard deviation about a half sinusoid fit is about 0.03 m, so 0.03 is the entry
            point and the term is alive across the whole of the present swing.
        sole_offsets: Sole support points in the foot body frame, as for v2.
        sensor_cfg: Optional contact gate, as for v2. Retained as defence in depth.
        force_threshold: Contact force threshold for that gate.

    Returns:
        The computed reward tensor, summed over the feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pts_w, _ = _sole_points_world(asset, asset_cfg.body_ids, sole_offsets)
    sole_clearance = pts_w[..., 2].min(dim=2)[0]                        # (N, F)

    gait_params = env.command_manager.get_command(command_name)
    in_swing, phi_swing = _swing_phase_from_gait_command(
        gait_params, env.episode_length_buf, env.step_dt
    )
    swing_height = gait_params[:, 3].unsqueeze(1)                       # (N, 1)
    # raised cosine, sin^2, NOT sin. See the docstring, this is the difference between a
    # reference that lands at zero vertical velocity and one that lands at 0.66 m/s.
    height_ref = swing_height * torch.square(torch.sin(torch.pi * phi_swing))

    reward = torch.exp(-torch.square(sole_clearance - height_ref) / std**2)
    reward = reward * in_swing.float()

    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
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
                  max_feet_distance: float = 1.0,
                  lateral_only: bool = False,)-> torch.Tensor:
    """Penalise the separation between the feet falling below a minimum or exceeding a maximum.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset.
        feet_links_name: Name patterns resolving the two feet bodies.
        min_feet_distance: Separation (m) below which the hinge becomes active.
        max_feet_distance: Separation (m) above which the upper hinge becomes active.
        lateral_only: When False, the default, the separation is the Euclidean norm of the
            planar difference between the two foot frames, which is the original behaviour and
            is preserved bit for bit for the TRON1 PF, WF and SF callers. When True the
            separation is the magnitude of the BASE FRAME lateral component alone, which is
            the stance width.

            The lateral component is taken in the base frame rather than the world frame,
            since a world frame y difference equals the stance width only when the robot's
            heading is zero, and the robot turns. The yaw of the root quaternion supplies the
            rotation, pitch and roll being deliberately discarded so that a leaning torso does
            not appear to narrow the stance.

    Returns:
        The computed penalty tensor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    if lateral_only:
        # rotate the planar separation into the base frame and keep the lateral component
        diff_w = feet_pos[:, 0, :3] - feet_pos[:, 1, :3]
        yaw_quat = math_utils.yaw_quat(asset.data.root_link_quat_w)
        diff_b = math_utils.quat_apply_inverse(yaw_quat, diff_w)
        feet_distance = torch.abs(diff_b[:, 1])
    else:
        # feet distance on x-y plane
        feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward


def feet_yaw_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalise the squared yaw of each foot relative to the base.

    The term follows the feet yaw reward of Booster Gym
    (arXiv:2506.15132), which carries it at twice the linear tracking weight.

    The difference is wrapped into the interval from minus pi to pi before squaring, so that a
    foot yawed just past the wrap point is charged for the small error it has rather than for
    the large one the raw difference would report.

    Args:
        env: The environment object.
        asset_cfg: Robot asset configuration resolving the feet bodies.

    Returns:
        The computed penalty tensor, summed over the feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]           # (N, F, 4)
    base_quat = asset.data.root_link_quat_w                             # (N, 4)

    foot_yaw = math_utils.euler_xyz_from_quat(foot_quat.reshape(-1, 4))[2].view(
        foot_quat.shape[0], foot_quat.shape[1]
    )
    base_yaw = math_utils.euler_xyz_from_quat(base_quat)[2].unsqueeze(1)

    yaw_error = math_utils.wrap_to_pi(foot_yaw - base_yaw)
    return torch.sum(torch.square(yaw_error), dim=1)


def joint_torque_tiredness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise the squared ratio of applied torque to each actuator's effort limit.

    The term follows the torque tiredness reward of Booster Gym (arXiv:2506.15132), which
    carries it one to two orders of magnitude above its plain torque penalty.
    Normalising by the limit makes approach to the ceiling expensive wherever it occurs, which
    is the property required, since a saturated actuator has no authority left for the
    correction that balance may next demand.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset, resolving the joints.

    Returns:
        The computed penalty tensor, summed over the joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    limit = asset.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(torque / (limit + 1e-6)), dim=1)

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
        The single contact test is instantaneous, so the interval during which a biped transfers
        its weight from one leg to the other, when both feet are necessarily loaded, is paid
        exactly nothing. The term therefore prices double support at zero and drives its
        duration toward zero with it.
    """

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]

    contacts = torch.norm(latest_contact_forces[:, history_index], dim = -1) > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1
    no_contact = torch.sum(contacts.float(), dim=1) == 0

    return 1.0 * single_contact - 5.0 * no_contact


class NoFlyWithGrace(ManagerTermBase):
    """Reward single support, but over a grace window, so that weight transfer is not taxed.

    This is :func:`no_fly` with the single contact branch widened in time. The free function
    is left exactly as it stands.

    The construction follows van Marum et al. (arXiv:2404.19173), whose single foot contact
    term returns unity if single contact occurred at least once in the preceding two tenths of
    a second. The purpose is precise. A biped must pass through double support to move its
    weight from one leg to the other, and an instantaneous single support test pays nothing
    for that interval, so a policy maximising it shortens the transfer until the transfer is
    an impact. Paying over a window makes a brief, genuine double support free while still
    refusing to pay a robot that simply stands on both feet, since standing exceeds the window.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.grace_steps = int(cfg.params.get("grace_steps", 0))
        # steps since single support was last seen. Initialised beyond the window so that an
        # environment which has not yet demonstrated single support earns nothing from the
        # grace branch, the robot beginning its episode on both feet.
        self._since_single = torch.full(
            (env.num_envs,), self.grace_steps + 1, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        """Reset the grace counter for the specified environments.

        Called automatically by the RewardManager on episode reset. Without this a policy
        would inherit the window across an episode boundary and be paid, for up to
        ``grace_steps``, for single support achieved in a previous episode.

        Args:
            env_ids: Indices of environments to reset. Defaults to None, in which case all
                environments are reset.
        """
        if env_ids is None:
            env_ids = slice(None)
        self._since_single[env_ids] = self.grace_steps + 1

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        threshold: float = 1.0,
        history_index: int = 0,
        grace_steps: int = 0,
    ) -> torch.Tensor:
        """Compute the reward.

        Args:
            env: The RL environment instance.
            sensor_cfg: Configuration for the contact force sensor.
            threshold: Contact force magnitude (N) above which a foot counts as grounded.
            history_index: Which slot of the contact sensor's rolling history supplies the
                contact test. The sensor writes the NEWEST sample to index 0. Unlike
                :func:`no_fly`, whose default of -1 is retained only for compatibility, this
                class defaults to 0, the current frame, which is the correct reading.
            grace_steps: Number of past control steps over which single support is allowed to
                have occurred for the single contact branch to pay. With 0 the test is
                instantaneous and the return is identical to :func:`no_fly` at the same
                ``history_index``, which makes this term a drop in replacement and lets an
                ablation isolate the window by changing one number. At this task's 0.01 s
                control period, 20 steps reproduces the 0.2 s window of van Marum et al.

        Returns:
            The computed reward tensor.
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]

        contacts = torch.norm(forces[:, history_index], dim=-1) > threshold
        num_contacts = torch.sum(contacts.float(), dim=1)
        single_contact = num_contacts == 1
        no_contact = num_contacts == 0

        # zero on the step single support is seen, incrementing otherwise, so the test
        # ``_since_single <= grace_steps`` is exactly "single support occurred within the
        # last grace_steps steps, inclusive of this one".
        self._since_single = torch.where(
            single_contact,
            torch.zeros_like(self._since_single),
            self._since_single + 1,
        )
        recently_single = self._since_single <= self.grace_steps

        return 1.0 * recently_single.float() - 5.0 * no_contact.float()

def keep_ankle_pitch_zero_in_air(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["ankle_L_Joint", "ankle_R_Joint"]),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["ankle_[LR]_Link"]),
    force_threshold: float = 2.0,
    pitch_scale: float = 0.2,
    require_airborne: bool = False,
    history_index: int = -1,
    use_default_offset: bool = False,
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
        history_index: Which slot of the contact sensor's rolling history supplies the
            current contact test, the preceding sample being taken from the slot adjacent
            to it in the direction of increasing age. The sensor writes the NEWEST sample
            to index 0 and the oldest to the tail, so 0 is the current frame and the 
            default of -1 is the OLDEST of the four buffered samples, roughly 15 ms stale, 
            paired against index -2. Defaults to -1, preserving the
            original behaviour for existing callers exactly. Pass 0 for the current frame,
            which then pairs against index 1.
        use_default_offset: Which posture the deviation is measured from. When False, the
            default, it is measured from the joint coordinate zero, which is the original
            behaviour and is preserved exactly. When True it is measured from the asset's
            default joint position, the posture the action term and the joint observations
            already work relative to, and which a startup randomisation may move per
            environment.

    Note:
            Where a robot's ankle default is not zero the two targets differ and the term
            charges a nominal pose for its own nominal offset, most of its range then being
            unreachable rather than merely unclaimed. The condition is aggravated
            by the fact that neither the actor nor the critic observes the absolute joint
            coordinate, both reading joint_pos_rel in the current policy configuration, so
            with the default the term names a target in a frame the policy cannot see.

    Returns:
        The computed reward tensor.

    """
    asset = env.scene[asset_cfg.name]
    contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    # the sample preceding history_index, which lies one slot further from index 0 in
    # whichever direction the caller's indexing runs
    previous_index = history_index - 1 if history_index < 0 else history_index + 1
    current_contact = torch.norm(contact_forces_history[:, history_index], dim=-1) > force_threshold
    last_contact = torch.norm(contact_forces_history[:, previous_index], dim=-1) > force_threshold
    contact_filt = torch.logical_or(current_contact, last_contact)
    # Use resolved joint_ids (shape: num_envs x num_ankle_joints) instead of hardcoded indices
    if use_default_offset:
        ankle_pitch = torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        )  # (N, 2)
    else:
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
        only one it has ever had.
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

