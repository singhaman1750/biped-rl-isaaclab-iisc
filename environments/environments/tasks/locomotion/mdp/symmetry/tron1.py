# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the TRON1 solefoot biped.

Robot-specific module over :mod:`.common`, for the asset emitted by the design generator at
``assets/mjcf/solefoot/tron1/base_robot.xml``, fifteen bodies and eight joints named
``<kind>_<side>_<suffix>`` with the side an R or an L between underscores.

The flip set is EMPTY. Composing each joint's body chain to the root gives the abduction axes
(-1, 0, 0) on the right and (1, 0, 0) on the left, so the left axis already equals the mirror
of the right, a -> -M a with M = diag(1, -1, 1), and the hip, knee and ankle pitch axes are
(0, -1, 0) on both sides, which is its own mirror. Every joint therefore keeps its sign, which
the identical left and right limits corroborate. The derivation is
``context/artefacts/tron1_quadruped-mjcf-mirror-audit-2026-10-20/``.

Every observation group of the plain, HIM and co-optimisation tasks is mirrored, so that an
estimator trained on the augmented batch learns a mirrored target from a mirrored input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

_SIDE = common.regex_side_partner(r"_(?P<side>[RL])_")

SPEC = common.MirrorSpec(
    joint_partner=_SIDE,
    body_partner=_SIDE,
    flip_joint_kinds=(),
    check_joint_limits=True,
)

_CACHE: dict = {}


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: "torch.Tensor | None" = None,
):
    """Augment obs and actions with the left-right sagittal mirror, originals first."""
    return common.augment(env, obs, actions, SPEC, _CACHE)
