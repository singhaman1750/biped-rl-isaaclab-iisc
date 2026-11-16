# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the kscale biped.

Robot-specific module, written closely against :mod:`.brs` so that the two are reviewable side
by side. Implements ``compute_symmetric_states`` in the signature the rsl_rl PPO symmetry hook
expects (``/ws/rsl_rl/rsl_rl/algorithms/ppo.py``) and referenced through
``isaaclab_rl.rsl_rl.RslRlSymmetryCfg``. A biped has one sagittal mirror, so the augmentation
doubles the batch, originals in the first half, mirrored copies in the second.

Three things differ from the SD_BRS1 module and each is a correctness matter rather than a
matter of taste.

The joint partner rule swaps the ``right_``/``left_`` PREFIX. SD_BRS1 swaps a trailing L for a
trailing R, and every kscale joint name ends in a digit, so that rule would match nothing and
degenerate silently to an identity permutation, producing an augmentation that teaches the
policy the robot is symmetric under doing nothing. The rule here raises instead of degrading.

The body partners are an explicit dictionary rather than any rule, because the right hip roll
link is named ``kd_d_201r_6061`` and its left partner is named ``rs03``, sharing no stem at all.
This is an Onshape export artefact: where a part was reused rather than mirrored in CAD, the
exporter names it after the part rather than after the side.

The sign flip set adds the hip yaw. It was established by composing each joint's origin
rotation chain to the root and comparing each left joint's mirrored world axis against its right
partner's actual axis, which settles the set without appeal to a naming convention or to a roll
against pitch rule. See ``scripts/analysis/kscale_physical_analysis.py`` and
``context/KScale.md`` section 7. Under a reflection a rotation axis transforms as a
pseudovector, ``a -> det(M) * M * a``, which is the treatment MorphoSymm establishes
(arXiv:2302.10433).

The resulting set is the physically expected pattern, the roll and yaw degrees of freedom
flipping and the pitch degrees of freedom not, with the single exception of the hip pitch, which
flips because its left and right URDF axes are genuinely anti-parallel. That exception is
corroborated independently by the joint limits: the hip pitch and hip roll are the only two
joints whose left and right limits are negations of one another, while the remaining four carry
identical limits on both sides. SD_BRS1 documents the same anomaly for itself.

This module is correct only against the CORRECTED root frame. The kscale URDF as exported
carried a root frame rotated ninety degrees about z, in which the mirror plane is y-z rather
than x-z and the flip set inverts to the physically implausible pattern of the pitch joints
flipping and the roll joints not. ``scripts/analysis/kscale_reframe_urdf.py`` performs the
correction and ``kscale_sole_analysis.py`` verifies it; that an incorrect mirror plane yields a
plausible looking table is exactly why the frame must be verified rather than assumed.

Every observation group is mirrored, the critic included, so the value function is biased toward
the sagittal invariance the theory requires. A value function that observes an unmirrored
privileged state cannot be invariant under the reflection. The term layout, the history of each
group, the height scan pattern and the collision shapes are read at runtime, and the joint and
body maps are built from the runtime names, so the module is correct regardless of whether Isaac
Sim resolves the sibling links left-first or right-first. The machinery is in :mod:`.common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

# joints whose sign flips under the sagittal mirror, by name substring. hip_pitch flips despite
# being a pitch joint because its left and right URDF axes are opposite. The hip yaw is here and
# absent from the SD_BRS1 set only because SD_BRS1 declares its hip yaw type="fixed", so that
# joint never enters robot.joint_names at all; kscale actuates twelve joints against its ten.
# None of these four substrings occurs in "knee_04" or in "foot_pitch_02", so the match is
# unambiguous and no negative guard is needed.
_FLIP_JOINT_KINDS = ("hip_pitch", "hip_roll", "hip_yaw", "foot_roll")

# body partners, as an explicit map. Every entry is given in both directions below, and the
# torso is its own partner, lying on the mirror plane.
_BODY_PARTNERS = {
    "assy_formfg___kd_b_102b_torso_btm": "assy_formfg___kd_b_102b_torso_btm",
    "kd_d_102r_6061": "kd_d_102l_6061",
    "kd_d_201r_6061": "rs03",
    "kd_d_301r_6061": "kd_d_301l_6061",
    "kd_d_401r_6061": "kd_d_401l_6061",
    "arb_uj111_cross_bearing": "arb_uj111_cross_bearing_2",
    "foot_6061": "foot_6061_2",
}
_BODY_PARTNERS.update({v: k for k, v in _BODY_PARTNERS.items()})

def _joint_partner(name: str) -> str:
    """Swap the side prefix. Raises rather than returning the name, so a miss is not silent."""
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    raise ValueError(
        f"joint {name!r} carries neither a 'right_' nor a 'left_' prefix, so its mirror partner "
        "cannot be determined. Every kscale joint is expected to carry one."
    )


def _body_partner(name: str) -> str:
    if name not in _BODY_PARTNERS:
        raise ValueError(
            f"body {name!r} has no entry in _BODY_PARTNERS. The kscale link names are not "
            "derivable by any rule, so the map must be extended by hand if the URDF changes."
        )
    return _BODY_PARTNERS[name]


SPEC = common.MirrorSpec(
    joint_partner=_joint_partner,
    body_partner=_body_partner,
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
