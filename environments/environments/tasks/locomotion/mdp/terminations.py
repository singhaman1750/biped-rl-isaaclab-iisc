from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def root_height_below_minimum_rough(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height above terrain at each ray sample: root_z - terrain_z (shape: N x num_rays)
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # replace non-finite values (rays missing geometry) with the target so they are neutral
    height = torch.nan_to_num(
        height, nan=minimum_height, posinf=minimum_height, neginf=minimum_height
    )
    # terminate if mean clearance over all ray samples is below minimum
    return height.mean(dim=1) < minimum_height


def root_height_below_fraction_rough(
    env: ManagerBasedRLEnv,
    min_height_fraction: float,
    reference_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height_attr: str = "copt_nominal_base_height",
) -> torch.Tensor:
    """:func:`root_height_below_minimum_rough` with each environment's minimum a fraction of its design's standing height.

    The height is the one the co-optimisation runner writes. While it is unset the
    minimum is ``min_height_fraction * reference_height`` for every environment. A missed ray takes the
    environment's own minimum, so it stays neutral as it does there.

    Defect left standing in :func:`root_height_below_minimum_rough` is one minimum
    height asked of every design, which sits at a different fraction of each design's standing height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    nominal = getattr(env, target_height_attr, None)
    if nominal is None:
        nominal = torch.full((env.num_envs,), reference_height, device=env.device)
    minimum = (min_height_fraction * nominal).unsqueeze(1)
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    height = torch.where(torch.isfinite(height), height, minimum.expand_as(height))
    return height.mean(dim=1) < minimum.squeeze(1)
