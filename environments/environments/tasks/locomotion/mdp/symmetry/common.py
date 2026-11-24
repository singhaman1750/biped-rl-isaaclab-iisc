# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared machinery of the sagittal (left-right) symmetry data augmentation.

A robot module declares a :class:`MirrorSpec` and implements ``compute_symmetric_states`` as a
call to :func:`augment`. The function must stay a top level member of the robot module, since
``class_to_dict`` serialises it to a ``module:function`` string that the PPO constructor imports
again, and a closure or a lambda cannot survive that round trip.

Every mirror kind is a signed permutation of the coordinates of one observation frame, so each
observation group compiles once into an index and a sign vector over its last axis, and is
mirrored by one gather, ``x[..., index] * sign``. The reflection is an involution, so the compiled
pair satisfies ``index[index] == arange`` and ``sign * sign[index] == 1``, which the compiler
asserts. Every group the observation manager carries is mirrored, and a term absent from the
kind table raises, so no term is ever tiled unmirrored without notice.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

# fixed per-frame sign vectors
VEC = (1.0, -1.0, 1.0)       # polar vector, flip lateral y
PVEC = (-1.0, 1.0, -1.0)     # pseudovector, flip x and z
CMD = (1.0, -1.0, -1.0)      # velocity command, flip lin_y and ang_z
# gait phase is a (sin, cos) pair of ONE shared clock with the feet in antiphase at an offset of
# 0.5, so exchanging the feet is a half-cycle shift and both channels negate together
GAIT = (-1.0, -1.0)
INERTIA9 = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # flip y-coupling products
# world frame root velocity [v, w]. The mirror plane is vertical up to roll, so the yaw rate is
# the one rotational component whose image is known without the base yaw, and it negates
ROOT_VEL_W = (1.0, 1.0, 1.0, 1.0, 1.0, -1.0)

_CONSTANT_KINDS = {"vec": VEC, "pvec": PVEC, "cmd": CMD, "gait": GAIT, "root_vel_w": ROOT_VEL_W}

# mirror kind of every observation term of the SF, quadruped, SD_BRS1 and kscale tasks, whichever
# group carries it, for the plain, HIM and co-optimisation configurations alike. A world frame
# foot vector maps to M_w applied to the partner's, M_w the reflection about the vertical heading
# plane, so exchanging the feet is exact on the vertical and forward components and wrong only
# in the sign of the lateral one, which needs the base yaw no observation carries. The root
# position lies on the mirror plane, so identity is exact for it.
TERM_KINDS = {
    "base_lin_vel": "vec", "base_ang_vel": "pvec", "proj_gravity": "vec",
    "joint_pos": "joint", "joint_vel": "joint", "last_action": "joint",
    "velocity_commands": "cmd", "gait_phase": "gait", "heights": "heights",
    "robot_joint_torque": "joint", "robot_joint_acc": "joint", "robot_joint_pos": "joint",
    "robot_joint_stiffness": "joint_nosign", "robot_joint_damping": "joint_nosign",
    "actuator_effort_limit": "joint_nosign", "actuator_velocity_limit": "joint_nosign",
    "actuator_mass": "joint_nosign", "actuator_radius": "joint_nosign", "actuator_length": "joint_nosign",
    "robot_mass": "body_scalar", "robot_inertia": "body_inertia", "nominal_pose": "body_scalar",
    "link_lengths": "body_pair_scalar", "robot_material_properties": "material",
    "feet_lin_vel": "body_vec_w", "feet_contact_force": "body_vec_w",
    "robot_pos": "identity", "robot_vel": "root_vel_w",
}

_SWAP = {"L": "R", "R": "L", "l": "r", "r": "l", "left": "right", "right": "left"}


@dataclass(frozen=True)
class MirrorSpec:
    """The sagittal mirror of one robot.

    Args:
        joint_partner: Maps a joint name to its mirror partner, itself for a joint on the plane.
        body_partner: The same for body names.
        flip_joint_kinds: Name substrings of the joints whose coordinate negates under the mirror,
            those whose partner's world axis is the negation of the mirrored axis.
        term_kinds: The mirror kind of each observation term, by term name.
        check_joint_limits: Raise where the joint limits or the default pose contradict the derived sign.
    """

    joint_partner: Callable[[str], str]
    body_partner: Callable[[str], str]
    flip_joint_kinds: tuple[str, ...]
    term_kinds: dict[str, str] | None = None
    check_joint_limits: bool = False


def suffix_side_partner(name: str) -> str:
    """Swap a trailing L for R, the SD_BRS1 rule. A name ending in neither lies on the plane."""
    return name[:-1] + "L" if name.endswith("R") else name[:-1] + "R" if name.endswith("L") else name


def regex_side_partner(pattern: str) -> Callable[[str], str]:
    """A partner rule swapping the named group ``side`` of ``pattern``, the name itself where it does not match."""
    rx = re.compile(pattern)

    def partner(name: str) -> str:
        match = rx.search(name)
        if match is None:
            return name
        start, end = match.span("side")
        return name[:start] + _SWAP[name[start:end]] + name[end:]

    return partner


def _assert_involution(perm, names, what):
    for i, p in enumerate(perm):
        assert perm[p] == i, f"{what} permutation is not an involution at {names[i]!r}"


def build_perm(names, partner, what="body"):
    """perm[i] is the index of the partner of names[i]."""
    perm = []
    for name in names:
        p = partner(name)
        if p not in names:
            raise ValueError(f"mirror partner {p!r} of {what} {name!r} is not in the articulation")
        perm.append(names.index(p))
    _assert_involution(perm, names, what)
    return perm


def build_joint_map(names, partner, flip_kinds, device):
    """out[i] = sign[i] * x[perm[i]]."""
    perm = build_perm(names, partner, "joint")
    sign = [-1.0 if any(k in name for k in flip_kinds) else 1.0 for name in names]
    return (torch.tensor(perm, dtype=torch.long, device=device),
            torch.tensor(sign, dtype=torch.float, device=device))


def build_material_perm(num_shapes_per_body, body_perm):
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
    return perm


def count_shapes_per_body(robot) -> list[int]:
    """Collision shapes per body in body order, by the loop of randomize_rigid_body_material.__init__."""
    view = robot.root_physx_view
    counts = [robot._physics_sim_view.create_rigid_body_view(path).max_shapes for path in view.link_paths[0]]
    if sum(counts) != view.max_shapes:
        raise ValueError(f"parsed {sum(counts)} collision shapes against the {view.max_shapes} of the articulation")
    return counts


def ray_mirror_perm(sensor, tol: float = 1e-5) -> list[int]:
    """perm[i] is the ray at the lateral reflection of ray i, read from the pattern itself."""
    if sensor.cfg.ray_alignment == "world":
        raise ValueError("a world aligned ray pattern does not turn with the base, so it has no fixed mirror")
    xy = sensor.ray_starts[0, :, :2].detach().double().cpu()
    dist = torch.cdist(xy * xy.new_tensor([1.0, -1.0]), xy)
    err, perm = dist.min(dim=1)
    if err.max() > tol:
        raise ValueError(f"the ray pattern is not symmetric about its x axis, worst mismatch {err.max():.2e} m")
    return perm.tolist()


def check_joint_limits(robot, jperm, jsign, tol: float = 1e-4):
    """A kept joint shares its partner's limits and a flipped one their negation. Symmetric limits identify neither."""
    lim = robot.data.joint_pos_limits[0].detach().cpu()
    names = list(robot.joint_names)
    for i, (p, s) in enumerate(zip(jperm.tolist(), jsign.tolist())):
        lo, hi = lim[i].tolist()
        if abs(lo + hi) < tol:
            continue
        expect = (lo, hi) if s > 0 else (-hi, -lo)
        got = tuple(lim[p].tolist())
        if max(abs(got[0] - expect[0]), abs(got[1] - expect[1])) > tol:
            raise ValueError(
                f"joint {names[i]!r} limits {(lo, hi)} with sign {s:+.0f} predict {expect} for partner "
                f"{names[p]!r}, which has {got}, so the flip set is wrong for this asset"
            )


def check_default_pose(robot, jperm, jsign, tol: float = 1e-4):
    """The configured default pose of a symmetric robot is its own mirror, which decides a flip that symmetric
    limits cannot. The configuration is read rather than ``default_joint_pos``, which an event may randomise."""
    pose = robot.cfg.init_state.joint_pos
    default = torch.tensor([next((v for k, v in pose.items() if re.fullmatch(k, n)), 0.0) for n in robot.joint_names])
    err = (default[jperm.cpu()] * jsign.cpu() - default).abs()
    if err.max() > tol:
        i = int(err.argmax())
        raise ValueError(
            f"the default pose is not its own mirror at joint {list(robot.joint_names)[i]!r}, error {err[i]:.3e}, "
            "so the flip set is wrong for this asset or its default pose is asymmetric"
        )


def _term_bodies(env, robot, term_cfg):
    """The body names a term reports, in the order it reports them."""
    params = term_cfg.params or {}
    if "sensor_cfg" in params:
        cfg = params["sensor_cfg"]
        names = list(env.scene.sensors[cfg.name].body_names)
    else:
        cfg = params.get("asset_cfg")
        names = list(robot.body_names)
    ids = slice(None) if cfg is None else cfg.body_ids
    return [names[i] for i in range(len(names))[ids]] if isinstance(ids, slice) else [names[i] for i in ids]


def _frame_map(kind, w, term_cfg, ctx):
    """(index, sign) of length w for one frame of a term."""
    if kind == "identity":
        return list(range(w)), [1.0] * w
    if kind in _CONSTANT_KINDS:
        sign = list(_CONSTANT_KINDS[kind])
        assert w == len(sign), f"{kind} expects a frame of {len(sign)}, got {w}"
        return list(range(w)), sign
    if kind in ("joint", "joint_nosign"):
        jperm, jsign = ctx["jperm"].tolist(), ctx["jsign"].tolist()
        assert w == len(jperm), f"{kind} expects {len(jperm)} joints, got {w}"
        return jperm, jsign if kind == "joint" else [1.0] * w
    if kind in ("body_scalar", "body_inertia", "body_vec_w"):
        names = _term_bodies(ctx["env"], ctx["robot"], term_cfg)
        perm = build_perm(names, ctx["spec"].body_partner)
        block, sign = {"body_scalar": (1, (1.0,)), "body_inertia": (9, INERTIA9), "body_vec_w": (3, (1.0,) * 3)}[kind]
        assert w == block * len(names), f"{kind} expects {block} x {len(names)}, got {w}"
        return [block * p + k for p in perm for k in range(block)], list(sign) * len(names)
    if kind == "body_pair_scalar":
        partner = ctx["spec"].body_partner
        pairs = list(zip(term_cfg.params["parent_body_names"], term_cfg.params["child_body_names"]))
        assert w == len(pairs), f"{kind} expects {len(pairs)} pairs, got {w}"
        return [pairs.index((partner(a), partner(b))) for a, b in pairs], [1.0] * w
    if kind == "heights":
        perm = ray_mirror_perm(ctx["env"].scene.sensors[term_cfg.params["sensor_cfg"].name])
        assert w == len(perm), f"heights expects {len(perm)} rays, got {w}"
        return perm, [1.0] * w
    if kind == "material":
        if ctx["mat_perm"] is None:
            ctx["mat_perm"] = build_material_perm(count_shapes_per_body(ctx["robot"]), ctx["body_perm"])
        assert w == 3 * len(ctx["mat_perm"]), f"material expects 3 x {len(ctx['mat_perm'])}, got {w}"
        return [3 * p + k for p in ctx["mat_perm"] for k in range(3)], [1.0] * w
    raise ValueError(f"unknown mirror kind {kind}")


def _compile_group(om, group, kinds, ctx):
    """(index, sign) over the last axis of the group, its terms concatenated along that axis.

    A flattened term of h frames is h blocks of one frame each, time major. A term that keeps
    its history as a separate axis, (h, w), is mirrored frame by frame along the last axis alone.
    """
    names = list(om.active_terms[group])
    missing = [t for t in names if t not in kinds]
    if missing:
        raise KeyError(f"group {group!r} carries terms with no mirror kind, {missing}")
    if not om.group_obs_concatenate[group] or om._group_obs_concatenate_dim[group] != -1:
        raise ValueError(f"group {group!r} is not concatenated along its last axis")
    index, sign, start = [], [], 0
    for name, dim, cfg in zip(names, om.group_obs_term_dim[group], om._group_obs_term_cfgs[group]):
        width = int(dim[-1])
        frames = max(cfg.history_length, 1) if len(dim) == 1 else 1
        assert width % frames == 0, f"term {name!r} of width {width} is not {frames} frames"
        w = width // frames
        fidx, fsgn = _frame_map(kinds[name], w, cfg, ctx)
        index.extend(start + f * w + i for f in range(frames) for i in fidx)
        sign.extend(fsgn * frames)
        start += width
    _assert_involution(index, [f"{group}[{i}]" for i in range(len(index))], "observation")
    assert all(sign[i] * sign[p] == 1.0 for i, p in enumerate(index)), f"{group} signs are not an involution"
    return index, sign


def compile_maps(env: ManagerBasedRLEnv, spec: MirrorSpec, device) -> dict:
    """Build every permutation of the robot and compile every observation group into (index, sign)."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    om = unwrapped.observation_manager
    jperm, jsign = build_joint_map(list(robot.joint_names), spec.joint_partner, spec.flip_joint_kinds, device)
    if spec.check_joint_limits:
        check_joint_limits(robot, jperm, jsign)
        check_default_pose(robot, jperm, jsign)
    ctx = {
        "env": unwrapped, "robot": robot, "spec": spec, "jperm": jperm, "jsign": jsign,
        "body_perm": build_perm(list(robot.body_names), spec.body_partner), "mat_perm": None,
    }
    kinds = TERM_KINDS if spec.term_kinds is None else spec.term_kinds
    groups = {}
    for group in om.active_terms:
        index, sign = _compile_group(om, group, kinds, ctx)
        groups[group] = (torch.tensor(index, dtype=torch.long, device=device),
                         torch.tensor(sign, dtype=torch.float, device=device))
    print(f"[symmetry] mirrored groups {list(groups)}")
    return {"jperm": jperm, "jsign": jsign, "groups": groups}


@torch.no_grad()
def augment(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
    spec: MirrorSpec,
    cache: dict,
):
    """Augment obs and actions with the sagittal mirror, originals in the first half."""
    device = obs["policy"].device if obs is not None else actions.device
    if device not in cache:
        cache[device] = compile_maps(env, spec, device)
    m = cache[device]

    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)  # tiles every group, originals in [:n]
        for group in obs.keys():
            index, sign = m["groups"][group]
            obs_aug[group][n:] = obs[group][..., index] * sign
    else:
        obs_aug = None

    actions_aug = None if actions is None else torch.cat([actions, actions[:, m["jperm"]] * m["jsign"]], dim=0)
    return obs_aug, actions_aug
