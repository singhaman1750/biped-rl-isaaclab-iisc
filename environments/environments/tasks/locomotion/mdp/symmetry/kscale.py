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

Both the policy and the critic observation groups are mirrored, so the value function is biased
toward the sagittal invariance the theory requires. A value function that observes an unmirrored
privileged state cannot be invariant under the reflection. The term layout is read at runtime
from the observation manager, and the joint, body, and collision-shape maps are built from the
runtime names, so the module is correct regardless of whether Isaac Sim resolves the sibling
links left-first or right-first.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

# history length shared by the policy and critic groups (flatten_history_dim=True)
_HISTORY = 10

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

# height-scan grid, lateral y rows by forward x columns (GridPatternCfg size (1.6, 1.0),
# resolution 0.1, ordering "xy" -> meshgrid shape (len(y), len(x)) = (11, 17)). These hold only
# because the root frame has been corrected; in the exported frame the 1.6 m axis lay ACROSS the
# robot and the two would be transposed.
_HEIGHT_ROWS = 11  # lateral y, the axis reversed by the mirror
_HEIGHT_COLS = 17  # forward x

# collision shapes per body in body order, for robot_material_properties. Leave None to skip
# mirroring that term (identity fallback). This cannot be determined statically; fill it after a
# one-time runtime print of the term width, which is three times the total shape count. Partner
# bodies must carry equal counts, which the builder below asserts.
_NUM_SHAPES_PER_BODY = None  # e.g. [1, 1,1, 1,1, 2,2, 2,2, 1,1, 2,2]

# fixed per-frame sign vectors
_VEC = (1.0, -1.0, 1.0)       # polar vector, flip lateral y
_PVEC = (-1.0, 1.0, -1.0)     # pseudovector, flip x and z
_CMD = (1.0, -1.0, -1.0)      # velocity command, flip lin_y and ang_z
# gait phase is a (sin, cos) pair of ONE shared clock, and the two feet are placed in antiphase
# by the command's offset of 0.5 rather than by the observation, so exchanging the feet is
# exactly a half-cycle shift of that clock and BOTH channels negate together. Treating the pair
# as an even and an odd channel and negating only the sine is wrong.
_GAIT = (-1.0, -1.0)
_INERTIA9 = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # flip y-coupling products

# per-term mirror kind, keyed by observation term name
_POLICY_KIND = {
    "base_lin_vel": "vec", "base_ang_vel": "pvec", "proj_gravity": "vec",
    "joint_pos": "joint", "joint_vel": "joint", "last_action": "joint",
    "velocity_commands": "cmd", "gait_phase": "gait",
}
_CRITIC_KIND = {
    "base_lin_vel": "vec", "base_ang_vel": "pvec", "proj_gravity": "vec",
    "joint_pos": "joint", "joint_vel": "joint", "last_action": "joint",
    "velocity_commands": "cmd", "heights": "heights",
    "robot_joint_torque": "joint", "robot_joint_acc": "joint",
    "feet_lin_vel": "identity", "robot_mass": "body_scalar",
    "robot_inertia": "body_inertia", "robot_joint_pos": "joint",
    "robot_joint_stiffness": "joint_nosign", "robot_joint_damping": "joint_nosign",
    "robot_pos": "identity", "robot_vel": "identity",
    "robot_material_properties": "material", "feet_contact_force": "identity",
    "gait_phase": "gait",
}

# module-level cache, one entry per (device); env is a singleton
_CACHE: dict = {}


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


def _build_joint_map(joint_names, device):
    """out[i] = sign[i] * x[perm[i]], partner is the same joint with its side prefix swapped."""
    n = len(joint_names)
    perm, sign = list(range(n)), [1.0] * n
    for i, name in enumerate(joint_names):
        partner = _joint_partner(name)
        if partner not in joint_names:
            raise ValueError(f"mirror partner {partner!r} of joint {name!r} is not in the articulation")
        perm[i] = joint_names.index(partner)
        if any(k in name for k in _FLIP_JOINT_KINDS):
            sign[i] = -1.0
    # The permutation must be an involution, which is what makes mirroring twice the identity.
    for i, p in enumerate(perm):
        assert perm[p] == i, f"joint permutation is not an involution at {joint_names[i]!r}"
    return (torch.tensor(perm, dtype=torch.long, device=device),
            torch.tensor(sign, dtype=torch.float, device=device), perm)


def _build_body_map(body_names):
    perm = list(range(len(body_names)))
    for i, name in enumerate(body_names):
        if name not in _BODY_PARTNERS:
            raise ValueError(
                f"body {name!r} has no entry in _BODY_PARTNERS. The kscale link names are not "
                "derivable by any rule, so the map must be extended by hand if the URDF changes."
            )
        partner = _BODY_PARTNERS[name]
        if partner not in body_names:
            raise ValueError(f"mirror partner {partner!r} of body {name!r} is not in the articulation")
        perm[i] = body_names.index(partner)
    for i, p in enumerate(perm):
        assert perm[p] == i, f"body permutation is not an involution at {body_names[i]!r}"
    return perm


def _build_material_perm(num_shapes_per_body, body_perm, device):
    if num_shapes_per_body is None:
        return None
    counts = list(num_shapes_per_body)
    for b, pb in enumerate(body_perm):
        assert counts[b] == counts[pb], "partner bodies must carry equal collision-shape counts"
    starts, acc = [], 0
    for c in counts:
        starts.append(acc)
        acc += c
    perm = []
    for b in range(len(counts)):
        src = body_perm[b]
        perm.extend(range(starts[src], starts[src] + counts[src]))
    return torch.tensor(perm, dtype=torch.long, device=device)


def _layout(env, group):
    """Return (term_names, block_widths) in concatenation order, widths already history-flattened."""
    om = env.unwrapped.observation_manager
    names = list(om.active_terms[group])
    dims = om.group_obs_term_dim[group]
    widths = [int(math.prod(d)) for d in dims]
    return names, widths


def _maps(env, device):
    if device in _CACHE:
        return _CACHE[device]
    robot = env.unwrapped.scene["robot"]
    jperm, jsign, _ = _build_joint_map(list(robot.joint_names), device)
    bperm_list = _build_body_map(list(robot.body_names))
    om = env.unwrapped.observation_manager
    m = {
        "jperm": jperm,
        "jsign": jsign,
        "bperm": torch.tensor(bperm_list, dtype=torch.long, device=device),
        "mat_perm": _build_material_perm(_NUM_SHAPES_PER_BODY, bperm_list, device),
        "vec": torch.tensor(_VEC, device=device),
        "pvec": torch.tensor(_PVEC, device=device),
        "cmd": torch.tensor(_CMD, device=device),
        "gait": torch.tensor(_GAIT, device=device),
        "inertia9": torch.tensor(_INERTIA9, device=device),
        "pol_layout": _layout(env, "policy"),
        "cri_layout": _layout(env, "critic") if "critic" in om.active_terms else None,
    }
    _CACHE[device] = m
    return m


def _apply(kind, blk, n, m):
    """blk is (n, block_width); return the mirrored block, same shape."""
    if kind == "identity":
        return blk
    sw = blk.shape[1] // _HISTORY
    x = blk.view(n, _HISTORY, sw)
    if kind in ("vec", "pvec", "cmd", "gait"):
        return (x * m[kind]).reshape(n, -1)
    if kind == "joint":
        return (x[..., m["jperm"]] * m["jsign"]).reshape(n, -1)
    if kind == "joint_nosign":
        return x[..., m["jperm"]].reshape(n, -1)
    if kind == "body_scalar":
        return x[..., m["bperm"]].reshape(n, -1)
    if kind == "body_inertia":
        xi = x.view(n, _HISTORY, sw // 9, 9)[..., m["bperm"], :] * m["inertia9"]
        return xi.reshape(n, -1)
    if kind == "heights":
        xh = torch.flip(x.view(n, _HISTORY, _HEIGHT_ROWS, _HEIGHT_COLS), dims=[-2])
        return xh.reshape(n, -1)
    if kind == "material":
        if m["mat_perm"] is None:
            return blk  # identity fallback until _NUM_SHAPES_PER_BODY is filled
        xm = x.view(n, _HISTORY, sw // 3, 3)[..., m["mat_perm"], :]
        return xm.reshape(n, -1)
    raise ValueError(f"unknown mirror kind {kind}")


def _mirror_group(flat, names, widths, kind_map, n, m):
    out = torch.empty_like(flat)
    start = 0
    for name, w in zip(names, widths):
        out[:, start:start + w] = _apply(kind_map.get(name, "identity"), flat[:, start:start + w], n, m)
        start += w
    assert start == flat.shape[1], f"width mismatch, mapped {start} of {flat.shape[1]}"
    return out


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: "torch.Tensor | None" = None,
):
    """Augment obs and actions with the left-right sagittal mirror, originals first."""
    device = obs["policy"].device if obs is not None else actions.device
    m = _maps(env, device)

    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)  # tiles every group, originals in [:n]
        pnames, pwidths = m["pol_layout"]
        obs_aug["policy"][n:] = _mirror_group(obs["policy"], pnames, pwidths, _POLICY_KIND, n, m)
        if m["cri_layout"] is not None and "critic" in obs.keys():
            cnames, cwidths = m["cri_layout"]
            obs_aug["critic"][n:] = _mirror_group(obs["critic"], cnames, cwidths, _CRITIC_KIND, n, m)
    else:
        obs_aug = None

    actions_aug = None if actions is None else torch.cat([actions, actions[:, m["jperm"]] * m["jsign"]], dim=0)
    return obs_aug, actions_aug
