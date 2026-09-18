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
        FIVE DEFECTS are left standing here. Use :func:`foot_landing_vel_v2` on a sole
        footed robot on flat ground, and :func:`foot_landing_vel_v3` on any robot, soled or
        point footed, on terrain that is not flat. This function is retained unchanged for
        the callers that already read it.

        1. The height gate is the FRAME PROXY ``body_pos_w[z] - foot_radius``, which assumes
           the sole sits a constant distance below the frame and is therefore exact only for
           a level foot. This is the same proxy that :func:`foot_clearance_reward` used and
           that :func:`foot_clearance_reward_v2` was written to replace.
        2. The penalised quantity is the FRAME vertical velocity, not the approach velocity
           of the sole. The two differ by the rotational term omega x r.
        3. The term is a TIME INTEGRAL of the squared velocity over a wide gate, so it is
           minimised by descending slowly through the upper part of the window rather than by
           arriving softly.
        4. The height gate is an ABSOLUTE world height, correct only on flat terrain.
        5. On terrain that is not flat, the gate should be referenced to the LOCAL terrain
           beneath each foot, which :func:`foot_landing_vel_v3` does via
           :func:`_terrain_height_under_points`.
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


def foot_landing_vel_v3(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    about_landing_threshold: float,
    force_threshold: float = 1.0,
    sole_offsets: list[list[float]] | None = None,
    foot_radius: float = 0.0,
    num_neighbours: int = 4,
    max_horizontal_dist: float | None = 0.5,
    reduction: str = "max",
) -> torch.Tensor:
    """Penalise the approach velocity of a foot about to land on the LOCAL terrain.

    This is to :func:`foot_landing_vel_v2` what :func:`foot_clearance_reward_v4` is to
    :func:`foot_clearance_reward_v2`. Both earlier variants gate on an ABSOLUTE world
    height, correct only on flat terrain, so on generated terrain a foot descending toward a
    raised tread is judged against the plane the terrain was built upon rather than the
    surface it is about to strike, opening the gate too late on rising ground and too early
    on falling ground. Here the clearance is referenced to the terrain beneath each foot
    individually via :func:`_terrain_height_under_points`, the same estimator and defaults
    that :func:`foot_clearance_reward_v4` uses, so the clearance this term gates on and the
    clearance that term rewards cannot drift apart.

    The term serves a sole footed and a point footed robot from one implementation, and a
    biped and a quadruped alike, the foot count entering only as the axis the penalty is
    summed over. Which branch is taken is decided by ``sole_offsets``.

    On a sole foot the contact point is the lowest of the sole points, its clearance is that
    point's height above the terrain beneath ITSELF rather than beneath the ankle, and the
    penalised quantity is the vertical component of ``v_link + omega x r`` evaluated there,
    so a foot rotating its sole into the ground is charged for the rotation. On a point foot
    the contact point is ``foot_radius`` directly below the frame origin along the WORLD
    vertical, whatever the foot's orientation, because the lowest point of a sphere does not
    move in the body frame as the body turns. Its lever arm therefore has no horizontal
    component, ``omega_x r_y - omega_y r_x`` vanishes identically, and the approach velocity
    is the frame's vertical velocity exactly.

    This function reads ``body_link_lin_vel_w`` and ``body_link_pos_w`` throughout, as
    :func:`foot_landing_vel_v2` does, rather than the mismatched centre-of-mass velocity and
    link frame position that :func:`foot_landing_vel` pairs. A negative clearance, a foot
    that has penetrated the estimated terrain, is retained and charged rather than clipped.

    The term remains a TIME INTEGRAL of the squared approach velocity over the gate, and is
    therefore still reducible in principle by descending slowly through the window rather
    than by arriving softly. The remedy adopted is v2's, sizing ``about_landing_threshold``
    to the terminal approach rather than the whole descent, so the free fall velocity from
    the threshold height bounds what an unpowered descent can deliver.

    Args:
        env: The environment object.
        asset_cfg: Robot asset configuration resolving the feet bodies.
        sensor_cfg: Contact sensor configuration resolving the same feet, in the same order.
            A foot already carrying load is exempt.
        height_sensor_cfg: Configuration for the ray caster supplying the terrain heights,
            the same sensor :func:`foot_clearance_reward_v4` and :func:`base_height_rough_l2`
            read.
        about_landing_threshold: Clearance (m) above the LOCAL terrain below which a
            descending, unloaded foot is charged. Agrees with v2's argument of the same name
            on flat ground and diverges from it elsewhere, so a value tuned on a plane
            carries across unchanged while a value tuned against v1's frame proxy does not.
        force_threshold: Contact force magnitude (N) above which a foot counts as landed and
            is exempt. Defaults to 1.0, matching ``rew_foot_clearance``.
        sole_offsets: Points on the sole in the foot body frame whose lowest world height is
            the clearance. Leave as None on a point foot, where ``foot_radius`` supplies the
            offset instead.
        foot_radius: Radius (m) of the contact sphere of a point foot, read only when
            ``sole_offsets`` is None. Defaults to 0.0.
        num_neighbours: Ray hits reduced over per foot, passed through to the lookup.
        max_horizontal_dist: Radius (m) beyond which a ray hit is not accepted as a
            neighbour of a foot. Defaults to 0.5, half the scanner's 1.0 m width.
        reduction: ``"max"`` or ``"mean"``, passed through to the lookup.

    Returns:
        The computed penalty tensor, summed over the feet.

    Note:
        DEFECT LEFT STANDING. A foot for which the ray caster resolves no accepted
        neighbour is NOT charged, mirroring the choice :func:`foot_clearance_reward_v4`
        makes in refusing to pay such a foot. Refusing to pay is conservative for a reward;
        refusing to charge is permissive for a penalty, leaving a nominal exploit in which a
        policy escapes the term by placing a foot beyond half the scanner width from the
        base. On the configurations in this workspace the scanner spans 1.6 m by 1.0 m about
        the base and no reachable foot placement leaves it, so the exploit is not available
        in practice.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]

    # link frame throughout, so that the position and the velocity belong to the same point
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]        # (N, F, 3)
    lin_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids]     # (N, F, 3)

    if sole_offsets is None:
        # a sphere's lowest point sits foot_radius below the origin along the world vertical
        # whatever the foot's orientation, so the lever arm is purely vertical and the
        # rotational contribution to the vertical velocity is identically zero
        query_pts = foot_pos
        contact_z = foot_pos[..., 2] - foot_radius
        approach_vel = lin_vel[..., 2]
    else:
        pts_w, r_w = _sole_points_world(asset, asset_cfg.body_ids, sole_offsets)
        contact_z, lowest = pts_w[..., 2].min(dim=2)                    # (N, F), (N, F)
        idx = lowest.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)   # (N, F, 1, 3)
        r_low = torch.gather(r_w, 2, idx).squeeze(2)                    # (N, F, 3)
        # the ground is looked up beneath the lowest sole point, not beneath the ankle
        query_pts = torch.gather(pts_w, 2, idx).squeeze(2)              # (N, F, 3)
        ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids]
        approach_vel = lin_vel[..., 2] + (
            ang_vel[..., 0] * r_low[..., 1] - ang_vel[..., 1] * r_low[..., 0]
        )

    terrain_z, valid = _terrain_height_under_points(
        query_pts, sensor.data.ray_hits_w, num_neighbours, max_horizontal_dist, reduction
    )
    # world frame vertical difference, for the reason foot_clearance_reward_v4 records. A
    # negative value, the contact point below the estimated terrain, is retained rather than
    # clipped, so a foot that has penetrated is charged rather than hidden.
    clearance = contact_z - terrain_z                                   # (N, F)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # history max, as in feet_slide, so contact chatter cannot flicker a loaded foot back
    # into the gate between two control steps
    in_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > force_threshold
    )

    about_to_land = (
        (clearance < about_landing_threshold) & (~in_contact) & (approach_vel < 0.0) & valid
    )
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


def feet_air_time_v2(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel, penalising a step that overruns.

    Identical to :func:`feet_air_time` below the cap, since both reward
    ``last_air_time - threshold_min`` on first contact. Where that function clamps the
    reward at ``threshold_max - threshold_min`` so an overlong step earns no more but no
    less than the cap, this one continues past the cap with a NEGATIVE slope, so a step
    that ran on twice as long as intended is charged rather than merely capped.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # negative reward for small steps
    air_time = (last_air_time - threshold_min) * first_contact
    # negative penalty for large steps: reward grows up to the cap, then
    # falls below zero the further air_time exceeds threshold_max
    cap = threshold_max - threshold_min
    air_time = torch.where(air_time > cap, cap - air_time, air_time)
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

    Note:
        DEFECT LEFT STANDING. The clearance is an ABSOLUTE world height, correct only on
        flat terrain. Use :func:`foot_clearance_reward_v4` on terrain that is not flat,
        which references the clearance to the LOCAL terrain beneath each foot.
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

        Addendum. This is now closed by :func:`foot_clearance_reward_v4`, which references
        the clearance to the LOCAL terrain beneath each foot via
        :func:`_terrain_height_under_points`. Use it on terrain that is not flat.
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


def _terrain_height_under_points(
    points_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    num_neighbours: int = 4,
    max_horizontal_dist: float | None = None,
    reduction: str = "max",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the terrain height directly beneath a set of world frame points.

    For every query point this selects the ``num_neighbours`` ray hits nearest to it in the
    horizontal plane and reduces their world heights, which gives a LOCAL terrain reference
    rather than the single mean over all rays that :func:`base_height_rough_l2` takes. The
    distinction matters on generated terrain, where the four feet of a quadruped may stand
    on four different surfaces, a mean beneath the base being an average of all of them and
    therefore correct for none of them.

    The estimate is valid because the ray caster's directions are NOT rotated under a yaw
    alignment, only its start positions are, so every ray descends vertically in the world
    frame and ``ray_hits_w[..., 2]`` is the ground height vertically below that grid point.

    Args:
        points_w: World frame query points, shape (N, P, 3) or (N, P, 2). Only the first two
            components are read, the height of the point itself being irrelevant to where
            the ground beneath it lies.
        ray_hits_w: World frame ray hit positions from the ray caster, shape (N, R, 3).
        num_neighbours: How many nearest hits to reduce over. Clamped to the ray count. At
            the 0.1 m grid resolution this configuration uses, 4 brackets a point between
            the surrounding grid cells.
        max_horizontal_dist: Optional radius (m) beyond which a neighbour is rejected. A
            point with no accepted neighbour is reported invalid rather than silently
            resolved against a distant grid edge, which is the failure a query point outside
            the scanner's footprint would otherwise produce. Defaults to None, no limit.
        reduction: ``"max"``, the default, for the highest accepted neighbour, the
            conservative choice near a step edge where a foot must clear the tallest nearby
            ground, resolving a rising tread from the moment a foot's horizontal position
            clears its edge at the cost of a bias of about half a grid cell diagonal on a
            slope. ``"mean"`` for an inverse distance weighted mean instead, smooth in the
            point position and therefore well conditioned for a reward gradient, but wrong
            by a reversal of the clearance signal, not merely an offset, at a riser.

    Returns:
        A tuple of the estimated terrain height, shape (N, P), and a boolean validity mask
        of the same shape. The height at an invalid entry is zero and must not be read.
    """
    if reduction not in ("mean", "max"):
        raise ValueError(f"reduction must be 'mean' or 'max'. Received: '{reduction}'.")

    hits_xy = ray_hits_w[..., :2]  # (N, R, 2)
    hits_z = ray_hits_w[..., 2]  # (N, R)
    # A ray that struck no geometry reports a non finite hit. Such a ray is excluded from
    # SELECTION by an infinite distance, rather than having its height substituted as
    # base_height_rough_l2 does, since a substituted height adjacent to a foot would corrupt
    # that foot's local estimate in a way it cannot corrupt a mean over the whole grid.
    finite = torch.isfinite(ray_hits_w).all(dim=-1)  # (N, R)

    # squared horizontal distance from every query point to every ray hit, (N, P, R)
    delta = points_w[..., :2].unsqueeze(2) - hits_xy.unsqueeze(1)
    dist_sq = delta.pow(2).sum(dim=-1)
    dist_sq = torch.where(finite.unsqueeze(1), dist_sq, torch.inf)

    k = min(int(num_neighbours), hits_xy.shape[1])
    near_dist_sq, near_idx = torch.topk(dist_sq, k, dim=-1, largest=False)  # (N, P, k)
    near_z = torch.gather(hits_z.unsqueeze(1).expand(-1, points_w.shape[1], -1), 2, near_idx)

    accepted = torch.isfinite(near_dist_sq)
    if max_horizontal_dist is not None:
        accepted = accepted & (near_dist_sq <= max_horizontal_dist**2)
    valid = accepted.any(dim=-1)  # (N, P)

    if reduction == "max":
        # -inf so a rejected neighbour can never win the maximum
        height = torch.where(accepted, near_z, torch.full_like(near_z, -torch.inf)).max(dim=-1)[0]
    else:
        # inverse distance weighting, the epsilon bounding the weight of a point sitting
        # exactly on a grid node rather than letting it diverge
        weights = torch.where(accepted, 1.0 / (near_dist_sq + 1.0e-6), torch.zeros_like(near_dist_sq))
        total = weights.sum(dim=-1)
        height = (weights * near_z).sum(dim=-1) / total.clamp(min=1.0e-12)

    return torch.where(valid, height, torch.zeros_like(height)), valid


def foot_clearance_reward_v4(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
    sole_offsets: list[list[float]] | None = None,
    num_neighbours: int = 4,
    max_horizontal_dist: float | None = 0.5,
    reduction: str = "max",
) -> torch.Tensor:
    """Reward the swinging feet for clearing a height above the LOCAL terrain.

    This is the terrain referenced form of :func:`foot_clearance_reward`, and it exists to
    close the defect that variant and :func:`foot_clearance_reward_v2` both carry, that
    their clearance is an absolute world height and therefore equals the height above the
    ground only on flat terrain. Here each foot is referenced to the terrain beneath ITSELF,
    estimated from the ray caster hits nearest that foot, rather than to the mean terrain
    height beneath the base that :func:`base_height_rough_l2` uses, because on a generated
    terrain carrying stairs, slopes and waves the four feet may stand on four different
    surfaces and a single mean is correct for none of them.

    The clearance is a world frame vertical difference, foot height minus ground height,
    which is the physically correct quantity because the ray caster descends vertically in
    the world frame even under a yaw alignment. Projecting it into the body frame would
    scale it by the cosine of the trunk tilt and couple the reward to pitch and roll, so a
    robot on a slope would be charged for a clearance it in fact has.

    ``target_height`` here is a clearance ABOVE THE TERRAIN and is a different physical
    quantity from v1's absolute body frame height and v2's absolute sole height, so a value
    tuned against either of those must not be carried across without re-tuning.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset, resolving the feet bodies.
        height_sensor_cfg: Configuration for the ray caster supplying the terrain heights,
            the same sensor :func:`base_height_rough_l2` reads.
        target_height: Desired clearance of the foot above the local terrain (m).
        std: Width of the Gaussian clearance kernel (m).
        tanh_mult: Scaling applied to the horizontal foot speed inside the tanh gate.
        sensor_cfg: Optional contact sensor configuration. When given, a foot in contact
            earns nothing whatever its measured clearance. Defaults to None.
        force_threshold: Contact force magnitude (N) above which a foot counts as grounded.
            Only used when ``sensor_cfg`` is given.
        sole_offsets: Optional sole points in the foot frame, as v2 takes them. When given,
            the clearance is the lowest sole point above the terrain rather than the body
            frame origin above the terrain, closing the tilt exploit on a soled foot. Leave
            as None on a point foot, where the two agree and the machinery is inert.
        num_neighbours: Ray hits reduced over per foot, passed through to the lookup.
        max_horizontal_dist: Radius (m) beyond which a ray hit is not accepted as a
            neighbour of a foot. Defaults to 0.5, half the scanner's 1.0 m width.
        reduction: ``"mean"`` or ``"max"``, passed through to the lookup.

    Returns:
        The computed reward tensor, summed over the feet.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]

    foot_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]  # (N, F, 3)

    if sole_offsets is None:
        query_xy = foot_pos
        foot_z = foot_pos[..., 2]
    else:
        # lowest sole point, as v2 computes it, so the clearance is invariant to foot tilt
        foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, F, 4)
        num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]
        offsets = torch.as_tensor(sole_offsets, dtype=foot_pos.dtype, device=foot_pos.device)
        num_pts = offsets.shape[0]
        quat = foot_quat.unsqueeze(2).expand(num_envs, num_feet, num_pts, 4)
        pts = offsets.view(1, 1, num_pts, 3).expand(num_envs, num_feet, num_pts, 3)
        pts_w = math_utils.quat_apply(quat.reshape(-1, 4), pts.reshape(-1, 3)).view(num_envs, num_feet, num_pts, 3)
        pts_w = pts_w + foot_pos.unsqueeze(2)
        lowest = pts_w[..., 2].argmin(dim=2, keepdim=True)  # (N, F, 1)
        foot_z = torch.gather(pts_w[..., 2], 2, lowest).squeeze(2)
        # the ground is looked up beneath the lowest sole point, not beneath the ankle
        query_xy = torch.gather(pts_w, 2, lowest.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)

    terrain_z, valid = _terrain_height_under_points(
        query_xy, sensor.data.ray_hits_w, num_neighbours, max_horizontal_dist, reduction
    )

    clearance = foot_z - terrain_z  # (N, F), world frame vertical
    clearance_error = torch.square(clearance - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.exp(-clearance_error / std**2) * foot_velocity_tanh
    # a foot the scanner cannot resolve earns nothing rather than being scored against a
    # terrain height that was never measured beneath it
    reward = reward * valid

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

    Note:
        DEFECT LEFT STANDING. This measures the separation of the two foot LINK FRAME
        ORIGINS, which on a sole footed robot is the ankle and not the foot, and diverges
        from the true footprint separation the moment a foot yaws. Use
        :func:`feet_distance_v2` on a sole footed robot, which additionally penalises the
        footprint separation itself.
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


def _footprint_separation(pts_w: torch.Tensor) -> torch.Tensor:
    """Minimum horizontal distance between two footprints, vertex against edge, both directions.

    Args:
        pts_w: (N, 2, P, 3) world points of the two feet, the P points of each foot given in
            perimeter order so that consecutive entries bound an edge.

    Returns:
        (N,) the separation of the two convex footprints, which is exact while they are disjoint.

    For two disjoint convex polygons the closest approach is realised at a vertex of one against an
    edge of the other, so sweeping every such pair is exact rather than sampled. Comparing instead
    the P squared VERTEX pairs errs by up to 3.42 mm over the KScale pinned envelope, the closest
    approach falling on the interior of the sole's long edge, which
    scripts/analysis/kscale_feet_distance_analysis.py measures in its section 5b.

    The measure saturates at zero rather than going negative once the footprints interpenetrate,
    and that regime is deliberately left to `filtered_contacts` on the foot pair sensors, which
    reports the true contact this geometry could only approximate.
    """
    best = None
    for near, far in ((pts_w[:, 0, :, :2], pts_w[:, 1, :, :2]),
                      (pts_w[:, 1, :, :2], pts_w[:, 0, :, :2])):
        edge = torch.roll(far, -1, dims=1) - far                          # (N, Q, 2)
        delta = near.unsqueeze(2) - far.unsqueeze(1)                      # (N, P, Q, 2)
        scale = (delta * edge.unsqueeze(1)).sum(-1) / (edge * edge).sum(-1).clamp_min(1.0e-12).unsqueeze(1)
        foot = delta - scale.clamp(0.0, 1.0).unsqueeze(-1) * edge.unsqueeze(1)
        span = torch.linalg.vector_norm(foot, dim=-1).amin(dim=(1, 2))    # (N,)
        best = span if best is None else torch.minimum(best, span)
    return best


def _perimeter_order(sole_offsets: list[list[float]]) -> list[list[float]]:
    """The sole table sorted around its own perimeter, which an angular sort gives for a convex set.

    The configuration is not obliged to declare the points in traversal order, the SD_BRS1 table at
    cfg/SF/brs_base_env_cfg.py:644 being grouped in mirrored pairs, so the order is imposed here
    rather than required of the caller. The sole normal is taken as the axis of least spread, which
    needs no robot specific knowledge of which axis that is, and that matters because the KScale
    foot frame is a full axis permutation away from the SD_BRS1 convention.
    """
    offsets = np.asarray(sole_offsets, dtype=np.float64)
    plane = [i for i in range(3) if i != int(np.argmin(offsets.std(axis=0)))]
    planar = offsets[:, plane] - offsets[:, plane].mean(axis=0)
    return offsets[np.argsort(np.arctan2(planar[:, 1], planar[:, 0]))].tolist()


def feet_distance_v2(env: ManagerBasedRLEnv,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                     feet_links_name: list[str] = ["foot_[RL]_Link"],
                     min_feet_distance: float = 0.1,
                     max_feet_distance: float = 1.0,
                     lateral_only: bool = False,
                     sole_offsets: list[list[float]] | None = None,
                     min_sole_distance: float = 0.0,) -> torch.Tensor:
    """Penalise the separation of the feet, measured at the frame origins and at the soles.

    :func:`feet_distance` measures the separation of the two foot LINK FRAME ORIGINS, which on a
    sole footed robot is the ankle and not the foot. The quantity that decides whether the feet
    foul one another is the separation of the two SOLE POLYGONS projected onto the ground, and the
    two diverge the moment a foot yaws, the toe swinging laterally about an origin that need not
    move. On the KScale at its nominal pose the origin measure exceeds the true footprint
    separation by 84.6 mm, being the width of a sole the original cannot see, and with the ankles
    held at the configured threshold of 0.24 m the two footprints touch at 0.60 rad of symmetric
    toe in while the original reads a penalty of exactly zero throughout. Both figures are
    reproduced by scripts/analysis/kscale_feet_distance_analysis.py.

    This term therefore carries two hinges rather than one. The first is the frame origin hinge of
    :func:`feet_distance`, reproduced bit for bit, which regulates the stance width. The second
    charges the footprint separation falling below ``min_sole_distance``, and it is silent unless
    both ``sole_offsets`` and a positive floor are supplied, so that every existing caller of this
    signature is unaffected. Give the two hinges independent weights by registering the function
    twice, the second term setting ``min_feet_distance`` to zero so that only the sole hinge speaks.

    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset.
        feet_links_name: Name patterns resolving the two feet bodies.
        min_feet_distance: Frame origin separation (m) below which the lower hinge becomes active.
        max_feet_distance: Frame origin separation (m) above which the upper hinge becomes active.
        lateral_only: As :func:`feet_distance`, and it governs both hinges. When False the sole
            hinge measures the true planar distance between the footprints, which is a genuine
            clearance and stays silent for feet that are merely separated fore and aft. When True
            it measures the SIGNED lateral gap in the base yaw frame, which goes negative when the
            footprints overlap in the lateral band, and which is the stance width proper.
        sole_offsets: Points on the sole in the foot body frame, the same table
            :func:`foot_clearance_reward_v3` consumes, ordered around the perimeter internally.
            Leave unset to obtain :func:`feet_distance` exactly.
        min_sole_distance: Footprint separation (m) below which the sole hinge becomes active.
            Leave at zero to obtain :func:`feet_distance` exactly.

    Returns:
        The computed penalty tensor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:, feet_links_idx]
    yaw_quat = math_utils.yaw_quat(asset.data.root_link_quat_w) if lateral_only else None

    if lateral_only:
        # rotate the planar separation into the base frame and keep the lateral component
        diff_w = feet_pos[:, 0, :3] - feet_pos[:, 1, :3]
        diff_b = math_utils.quat_apply_inverse(yaw_quat, diff_w)
        feet_distance = torch.abs(diff_b[:, 1])
    else:
        # feet distance on x-y plane
        feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)

    if sole_offsets is None or min_sole_distance <= 0.0:
        return reward

    pts_w, _ = _sole_points_world(asset, feet_links_idx, _perimeter_order(sole_offsets))
    if lateral_only:
        # the signed lateral gap between the two footprints, negative where their bands overlap
        num_envs, num_feet, num_pts, _ = pts_w.shape
        rel = (pts_w - asset.data.root_link_pos_w[:, None, None, :]).reshape(-1, 3)
        quat = yaw_quat[:, None, None, :].expand(num_envs, num_feet, num_pts, 4).reshape(-1, 4)
        lateral = math_utils.quat_apply_inverse(quat, rel).view(num_envs, num_feet, num_pts, 3)[..., 1]
        sole_distance = torch.maximum(lateral[:, 0].amin(-1) - lateral[:, 1].amax(-1),
                                      lateral[:, 1].amin(-1) - lateral[:, 0].amax(-1))
    else:
        sole_distance = _footprint_separation(pts_w)
    reward += torch.clip(min_sole_distance - sole_distance, 0, 1)
    return reward


def feet_yaw_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    forward_axis: tuple[float, float, float] | None = None,
    common_mode: str = "sum",
    tolerance: float = 0.0,
    differential_scale: float = 0.0,
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
    history_index: int = 0,
    airborne_only: bool = False,
) -> torch.Tensor:
    """Penalise the yaw of each foot relative to the base.

    The term follows the feet yaw rewards of Booster Gym (arXiv:2506.15132). That work's
    Table II tabulates a single squared norm at minus 1.0, but its released implementation
    and configuration carry TWO terms priced separately and equally, `feet_yaw_diff` at
    minus 1.0 over the two feet against each other and `feet_yaw_mean` at minus 1.0 over the
    mean foot yaw against the base. Both are reproduced here. The earlier docstring of this
    function reported the weight as twice the LINEAR tracking weight; against Table II it
    equals a single linear component's 1.0 and is twice the YAW tracking weight of 0.5.

    Every difference is wrapped into the interval from minus pi to pi before squaring, so that
    a foot yawed just past the wrap point is charged for the small error it has rather than
    for the large one the raw difference would report.

    Args:
        env: The environment object.
        asset_cfg: Robot asset configuration resolving the feet bodies. Exactly two feet are
            required whenever differential_scale is non zero or common_mode is "mean".
        forward_axis: Which axis of the FOOT LINK frame points at the toe. When None, the
            default, the heading is taken as the yaw component of an Euler decomposition of
            the foot's world quaternion, which is the original behaviour and is preserved
            exactly. That path is correct only where the foot link's forward axis is its own
            x and its vertical is its own z, which is the convention of the SD_BRS1 and of
            Booster Gym's own T1, and is NOT the KScale convention, whose foot link carries
            its width on x, its vertical on y pointing downward and its fore and aft length
            on z. Pass (0.0, 0.0, -1.0) for the KScale. See context/KScale.md section 12 and
            plans/kscale_integration.md section 5.1.2, which measures the Euler path's error
            on that robot at 70 to 110 degrees, an error that would INVERT the term rather
            than blur it, driving the feet towards the very splay it is meant to remove.
        common_mode: How the per foot errors against the base are combined. "sum", the
            default, sums their squares, which is the original behaviour and is preserved
            exactly. "mean" squares the error of their circular MEAN against the base, which
            is Booster Gym's `_reward_feet_yaw_mean` and is the form that makes the common and
            differential modes ORTHOGONAL. Under "sum" a pure splay of plus and minus e
            registers 2 e squared on the common mode, so the two modes cannot be varied or
            ablated independently. Under "mean" it registers exactly zero. Prefer "mean"
            wherever differential_scale is non zero.
        tolerance: Half width of a dead band, in radians, applied to each mode's error before
            it is squared. Defaults to 0.0, which is Booster Gym's own behaviour and is
            preserved as the default. The human foot progression angle has a standard
            deviation of 5.6 degrees about a mean toe out of 3.3 degrees, so a tolerance near
            0.10 rad demands no more than natural walking does.
        differential_scale: Weight of the differential mode, being the squared wrapped
            difference between the two feet's headings, RELATIVE to the common mode. Defaults
            to 0.0, preserving the original behaviour. Pass 1.0 for Booster Gym's own equal
            pricing of the two modes.
        sensor_cfg: Contact sensor resolving the same feet, in the same order as asset_cfg.
            Required only when airborne_only is True. Defaults to None.
        force_threshold: Contact force, in newtons, above which a foot counts as planted.
        history_index: Which slot of the contact sensor's rolling history supplies the
            contact test. The sensor writes the NEWEST sample to index 0. Defaults to 0.
        airborne_only: When True the COMMON mode is evaluated only over airborne feet, the
            mean under common_mode "mean" being taken over those feet alone and the term
            being zero when none is airborne. The differential mode is never gated, both feet
            being required for it to be defined. Defaults to False, which is Booster Gym's
            own behaviour and is preserved as the default.

    Note:
            The gate exists to support the ablation of plans/kscale_integration.md section
            5.1.2 and is NOT recommended as a shipping default. Under common_mode "mean" a
            well executed turn moves both feet together and the mean tracks the base, so the
            common mode's time averaged value during a steady turn is zero to four decimal
            places at every commanded yaw rate in the KScale curriculum, and the gate has
            nothing to protect. It was necessary only under the summed form, whose mode mixing
            makes a turning stance foot register on the common mode.

    Returns:
        The computed penalty tensor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_yaw = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w)[2].unsqueeze(1)

    foot_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]           # (N, F, 4)
    if forward_axis is None:
        foot_yaw = math_utils.euler_xyz_from_quat(foot_quat.reshape(-1, 4))[2].view(
            foot_quat.shape[0], foot_quat.shape[1]
        )
    else:
        axis = torch.tensor(forward_axis, device=foot_quat.device, dtype=foot_quat.dtype)
        toe_w = math_utils.quat_apply(foot_quat, axis.expand_as(foot_quat[..., :3]))
        foot_yaw = torch.atan2(toe_w[..., 1], toe_w[..., 0])            # (N, F)

    def _band(err: torch.Tensor) -> torch.Tensor:
        if tolerance <= 0.0:
            return err
        return torch.sign(err) * torch.clamp(torch.abs(err) - tolerance, min=0.0)

    if airborne_only:
        forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history
        airborne = ~(
            torch.norm(forces[:, history_index, sensor_cfg.body_ids], dim=-1) > force_threshold
        )
    else:
        airborne = torch.ones_like(foot_yaw, dtype=torch.bool)

    if common_mode == "mean":
        # circular mean of the selected feet, taken on the unit circle so that no branch cut
        # repair is needed. Booster Gym adds pi to a linear mean where the two feet straddle
        # the wrap; resolving the mean as an angle is equivalent and has no special case.
        mask = airborne.to(foot_yaw.dtype)
        sin_m = torch.sum(torch.sin(foot_yaw) * mask, dim=1)
        cos_m = torch.sum(torch.cos(foot_yaw) * mask, dim=1)
        any_sel = torch.sum(mask, dim=1) > 0
        mean_yaw = torch.atan2(sin_m, cos_m)
        common = torch.where(
            any_sel, torch.square(_band(math_utils.wrap_to_pi(mean_yaw - base_yaw.squeeze(1)))),
            torch.zeros_like(sin_m),
        )
    elif common_mode == "sum":
        err = _band(math_utils.wrap_to_pi(foot_yaw - base_yaw))
        common = torch.sum(torch.square(err) * airborne.to(err.dtype), dim=1)
    else:
        raise ValueError(f"common_mode must be 'sum' or 'mean', got {common_mode!r}")

    if differential_scale <= 0.0:
        return common
    differential = _band(math_utils.wrap_to_pi(foot_yaw[:, 1] - foot_yaw[:, 0]))
    return common + differential_scale * torch.square(differential)

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


def filtered_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Penalise contacts between a sensor's own bodies and the bodies it filters against.

    This is the pairwise counterpart of :func:`isaaclab.envs.mdp.rewards.undesired_contacts`,
    which reads ``net_forces_w_history`` and therefore sees only the total force on a body,
    with no record of what that body touched. A foot is in periodic contact with the terrain,
    so the net force never distinguishes a foot resting on the ground from a foot struck by
    the other foot, and the term cannot be applied to the feet at all. Reading
    ``force_matrix_w_history`` instead resolves the force per sensor body and per filtered
    body, which separates the two cases.

    The sensor named by ``sensor_cfg`` must declare a non empty
    :attr:`~isaaclab.sensors.ContactSensorCfg.filter_prim_paths_expr`, and its ``prim_path``
    must resolve to exactly one prim per environment, PhysX supporting filtered reporting
    only as one to many. The filter axis is ordered by those expressions and is summed over
    in full, ``sensor_cfg`` selecting bodies along the sensor axis alone.

    Args:
        env: The environment object.
        sensor_cfg: Configuration resolving the filtered contact sensor and its bodies.
        threshold: Force norm (N) above which a pair counts as being in contact.

    Returns:
        The number of body and filter pairs in contact, per environment.

    Note:
        PhysX reports no contact between two links of one articulation unless
        ``enabled_self_collisions`` is set on its articulation root, so this term is
        identically zero for an intra robot filter while that flag is off.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # shape (N, T, B, M, 3), None whenever the sensor carries no filter expressions
    force_matrix = contact_sensor.data.force_matrix_w_history
    if force_matrix is None:
        raise ValueError(
            f"Contact sensor '{sensor_cfg.name}' reports no filtered contacts. Set"
            " 'filter_prim_paths_expr' on its ContactSensorCfg to enable pairwise reporting."
        )
    # max over the history window, matching the semantics of undesired_contacts
    is_contact = torch.max(torch.norm(force_matrix[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over the sensor body axis and the filter axis together
    return torch.sum(is_contact, dim=(1, 2))


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


def feet_hold_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    air_time_ceiling: float,
) -> torch.Tensor:
    """Penalise a foot held in the air beyond a ceiling.

    Unlike feet_air_time and feet_air_time_v2, which are evaluated only at touchdown and
    therefore pay a foot that never lands exactly zero, this term reads the RUNNING air
    time every step, so a retracted limb accrues without bound. Deliberately unclipped,
    since a clip would remove the gradient exactly where a runaway retraction should be
    punished more, which is the defect this term exists to avoid. Gated on the command
    norm so that a standing robot is not charged for holding still.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(current_air_time - air_time_ceiling, min=0.0)
    reward = torch.sum(excess, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

