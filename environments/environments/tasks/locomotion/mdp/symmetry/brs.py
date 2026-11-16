# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the SD_BRS1 biped.

Robot-specific module. Implements ``compute_symmetric_states`` in the signature the
rsl_rl PPO symmetry hook expects (``/ws/rsl_rl/rsl_rl/algorithms/ppo.py``) and referenced
through ``isaaclab_rl.rsl_rl.RslRlSymmetryCfg``. A biped has one sagittal mirror, so the
augmentation doubles the batch, originals in the first half, mirrored copies in the second.

Every observation group is mirrored, the critic included, so the value function is biased
toward the sagittal invariance the theory requires. The term layout, the history of each
group, the height scan pattern and the collision shapes are read at runtime, and the joint and
body maps are built from the runtime names, so the module is correct regardless of whether
Isaac Sim resolves the sibling links left-first or right-first. The machinery is in :mod:`.common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

# joints whose sign flips under the sagittal mirror, by name substring. HipPitch flips
# despite being a pitch joint because its left and right URDF axes are opposite.
_FLIP_JOINT_KINDS = ("HipRoll", "AnkleRoll", "HipPitch")

SPEC = common.MirrorSpec(
    joint_partner=common.suffix_side_partner,
    body_partner=common.suffix_side_partner,
    flip_joint_kinds=_FLIP_JOINT_KINDS,
    check_joint_limits=True,
)

# module-level cache, one entry per (device); env is a singleton
_CACHE: dict = {}


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: "torch.Tensor | None" = None,
):
    """Augment obs and actions with the left-right sagittal mirror, originals first."""
    return common.augment(env, obs, actions, SPEC, _CACHE)
