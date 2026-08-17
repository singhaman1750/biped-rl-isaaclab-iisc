# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the SD_BRS1 biped.

Robot-specific module. Implements ``compute_symmetric_states`` in the signature the
rsl_rl PPO symmetry hook expects (``/ws/rsl_rl/rsl_rl/algorithms/ppo.py``) and referenced
through ``isaaclab_rl.rsl_rl.RslRlSymmetryCfg``. A biped has one sagittal mirror, so the
augmentation doubles the batch, originals in the first half, mirrored copies in the second.

Both the policy and the critic observation groups are mirrored, so the value function is
biased toward the sagittal invariance the theory requires. The term layout is read at
runtime from the observation manager, and the joint, body, and collision-shape maps are
built from the runtime names, so the module is correct regardless of whether Isaac Sim
resolves the sibling links left-first or right-first.
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

# joints whose sign flips under the sagittal mirror, by name substring. HipPitch flips
# despite being a pitch joint because its left and right URDF axes are opposite.
_FLIP_JOINT_KINDS = ("HipRoll", "AnkleRoll", "HipPitch")

# height-scan grid, lateral y rows by forward x columns (GridPatternCfg size (1.6, 1.0),
# resolution 0.1, ordering "xy" -> meshgrid shape (len(y), len(x)) = (11, 17)).
_HEIGHT_ROWS = 11  # lateral y, the axis reversed by the mirror
_HEIGHT_COLS = 17  # forward x

# collision shapes per body in body order, for robot_material_properties. Leave None to
# skip mirroring that term (identity fallback). Fill after the one-time runtime print in
# the plan, section 4.3. The observed total for SD_BRS1 is 19 shapes (width 57 = 19 * 3).
# Partner bodies must carry equal counts.
_NUM_SHAPES_PER_BODY = None  # e.g. [1, 1,1, 1,1, 2,2, 2,2, 1,1, 2,2]

# fixed per-frame sign vectors
_VEC = (1.0, -1.0, 1.0)       # polar vector, flip lateral y
_PVEC = (-1.0, 1.0, -1.0)     # pseudovector, flip x and z
_CMD = (1.0, -1.0, -1.0)      # velocity command, flip lin_y and ang_z
_GAIT = (-1.0, -1.0)          # gait phase, half-cycle shift under the foot swap
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


def _build_joint_map(joint_names, device):
    """out[i] = sign[i] * x[perm[i]], partner is the same joint with its side letter swapped."""
    n = len(joint_names)
    perm, sign = list(range(n)), [1.0] * n
    for i, name in enumerate(joint_names):
        partner = name[:-1] + "L" if name.endswith("R") else name[:-1] + "R" if name.endswith("L") else name
        perm[i] = joint_names.index(partner)
        if any(k in name for k in _FLIP_JOINT_KINDS):
            sign[i] = -1.0
    return (torch.tensor(perm, dtype=torch.long, device=device),
            torch.tensor(sign, dtype=torch.float, device=device), perm)


def _build_body_map(body_names):
    perm = list(range(len(body_names)))
    for i, name in enumerate(body_names):
        partner = name[:-1] + "L" if name.endswith("R") else name[:-1] + "R" if name.endswith("L") else name
        perm[i] = body_names.index(partner)
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
