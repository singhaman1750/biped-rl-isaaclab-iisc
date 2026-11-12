# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the design generator quadruped.

Robot-specific module over :mod:`.common`, for ``assets/mjcf/quadruped/quadruped.xml``,
twenty five bodies and twelve joints, each leg named by a fore or rear letter followed by a
side letter, ``FR``, ``FL``, ``RR`` and ``RL``. Only the side letter is exchanged, so the front
right leg pairs with the front left and the rear right with the rear left.

The group is the sagittal reflection alone. Every leg carries the same thigh and shank
rotations, so the knees of the front and the rear legs bend the same way and the robot has no
fore and aft mirror, unlike the ANYmal of Isaac Lab's reference module.

The abduction joints flip. Their axis is (1, 0, 0) on both sides, whose mirror is (-1, 0, 0),
so the left coordinate is the negation of the right. The hip and knee pitch axes are (0, 1, 0)
on both sides and keep their sign. The abduction limits are symmetric about zero and so cannot
corroborate the flip, the thigh and calf limits are identical across sides and corroborate the
keep. The derivation is ``context/artefacts/tron1_quadruped-mjcf-mirror-audit-2026-10-20/``.

The rule also serves the Unitree Go1 asset of the Go1 ablation task, whose legs carry the same
FR, FL, RR and RL prefixes and whose abduction axes follow the same convention.

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

_SIDE = common.regex_side_partner(r"(?:^|_)[FR](?P<side>[RL])_")

SPEC = common.MirrorSpec(
    joint_partner=_SIDE,
    body_partner=_SIDE,
    flip_joint_kinds=("_hip_joint",),
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
