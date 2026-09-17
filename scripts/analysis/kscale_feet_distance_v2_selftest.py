#!/usr/bin/env python3
"""Self test for `feet_distance_v2`, the footprint aware separation penalty.

Run this before any training run that repoints `pen_feet_distance` at the new term, in the manner
of scripts/analysis/kscale_feet_heading_selftest.py, which this file follows in structure. Three
families of check are performed and each answers a question the other two cannot.

The COMPATIBILITY checks establish that every new argument left at its default reproduces
`feet_distance` bit for bit, which is the condition CLAUDE.md imposes on any change to a shared
mdp module, the same function serving the TRON1 PF, WF and SF tasks and the SD_BRS1.

The INVARIANCE checks establish that the measure depends on the relative pose of the two feet and
on nothing else, being unchanged by a yaw of the whole robot and by the order in which
`find_bodies` happens to resolve the two feet.

The AGREEMENT checks compare the torch reward against the independent numpy geometry of
scripts/analysis/kscale_feet_distance_analysis.py at poses that script constructs, so that the
reward is validated against a second implementation rather than against a remembered number. The
same file supplies the exact convex polygon distance, which is the ground truth both measures
approximate.

Run with no arguments. Exits non zero on the first failure.
"""

from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kscale_feet_distance_analysis import (  # noqa: E402
    CFG_PATH,
    FOOT,
    MIN_FEET_DISTANCE,
    order_perimeter,
    pinned_frames,
    point_pair_distance,
    polygon_distance,
    segment_distance,
    sole_table,
    symmetric_pose,
)
from kscale_sole_analysis import convex_hull_2d, forward_kinematics, parse_urdf  # noqa: E402
from kscale_stance_analysis import DEFAULT_URDF  # noqa: E402

TOL = 1.0e-6

_failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:56s} {detail}")
    if not ok:
        _failures.append(name)


def mat_to_quat(mat: np.ndarray):
    """Rotation matrix to the (w, x, y, z) order Isaac Lab uses throughout."""
    import torch

    trace = mat.trace()
    w = math.sqrt(max(0.0, 1.0 + trace)) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + mat[0, 0] - mat[1, 1] - mat[2, 2])) / 2,
                      mat[2, 1] - mat[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - mat[0, 0] + mat[1, 1] - mat[2, 2])) / 2,
                      mat[0, 2] - mat[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - mat[0, 0] - mat[1, 1] + mat[2, 2])) / 2,
                      mat[1, 0] - mat[0, 1])
    return torch.tensor([w, x, y, z], dtype=torch.float32)


def stub_env(pos_w, quat_w, root_quat_w, root_pos_w):
    """The six fields the two rewards read, and nothing else.

    `body_pos_w` and `body_quat_w` are carried alongside their link frame twins because
    `_sole_points_world` reads the former and `feet_distance` the latter, the two being the same
    quantity under different names since IsaacLab/source/isaaclab/isaaclab/assets/articulation/
    articulation_data.py:1056.
    """
    data = types.SimpleNamespace(
        body_link_pos_w=pos_w, body_pos_w=pos_w,
        body_link_quat_w=quat_w, body_quat_w=quat_w,
        root_link_quat_w=root_quat_w, root_link_pos_w=root_pos_w,
    )
    asset = types.SimpleNamespace(data=data, find_bodies=lambda names: ([0, 1], names))
    return types.SimpleNamespace(scene={"robot": asset})


def yaw_quat_of(psi):
    import torch

    psi = torch.as_tensor(psi, dtype=torch.float32).reshape(-1)
    zero = torch.zeros_like(psi)
    return torch.stack([torch.cos(psi / 2), zero, zero, torch.sin(psi / 2)], dim=-1)


def _load_rewards(torch):
    """Return (feet_distance, feet_distance_v2), importing Isaac Lab where it is available.

    Inside the Isaac container the real module is imported, so the checks exercise exactly what
    training will run. Outside it neither Isaac Lab nor the task package imports, and rather than
    skip the checks the four function bodies are extracted from their source file and executed
    against a small shim supplying the three quaternion helpers they use. Both paths test the same
    source TEXT, so a regression in the reward is caught on either, and the shim is exercised
    against the real implementation whenever the container is available. The pattern is that of
    `_load_reward` in scripts/analysis/kscale_feet_heading_selftest.py.
    """
    try:
        from environments.tasks.locomotion.mdp.rewards import feet_distance, feet_distance_v2

        return feet_distance, feet_distance_v2
    except ImportError:
        pass

    import re
    import textwrap

    source_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "environments", "environments", "tasks", "locomotion", "mdp", "rewards.py",
    )
    with open(os.path.normpath(source_path)) as handle:
        source = handle.read()

    class _Shim:
        @staticmethod
        def quat_apply(quat, vec):
            shape = vec.shape
            quat, vec = quat.reshape(-1, 4), vec.reshape(-1, 3)
            xyz = quat[:, 1:]
            t = torch.cross(xyz, vec, dim=-1) * 2
            return (vec + quat[:, 0:1] * t + torch.cross(xyz, t, dim=-1)).view(shape)

        @staticmethod
        def quat_apply_inverse(quat, vec):
            conj = quat.reshape(-1, 4).clone()
            conj[:, 1:] *= -1.0
            return _Shim.quat_apply(conj, vec)

        @staticmethod
        def yaw_quat(quat):
            flat = quat.reshape(-1, 4)
            yaw = torch.atan2(
                2.0 * (flat[:, 0] * flat[:, 3] + flat[:, 1] * flat[:, 2]),
                1.0 - 2.0 * (flat[:, 2] ** 2 + flat[:, 3] ** 2),
            )
            out = torch.zeros_like(flat)
            out[:, 0], out[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
            return out.view(quat.shape)

    namespace = {
        "torch": torch,
        "np": np,
        "math_utils": _Shim,
        "ManagerBasedRLEnv": object,
        # the signature defaults `SceneEntityCfg("robot")`, which is evaluated when the extracted
        # def executes, so the stub must accept the call the real class does
        "SceneEntityCfg": type("SceneEntityCfg", (), {"__init__": lambda self, name="robot", **kw: setattr(self, "name", name)}),
        "Articulation": object,
    }
    for name in ("_sole_points_world", "feet_distance", "_footprint_separation",
                 "_perimeter_order", "feet_distance_v2"):
        match = re.search(rf"^def {name}\(.*?(?=^def |\Z)", source, re.S | re.M)
        if match is None:
            raise RuntimeError(f"{name} not found in rewards.py")
        exec(textwrap.dedent(match.group(0)), namespace)
    print("  (Isaac Lab unavailable, the reward source was loaded directly against a shim)\n")
    return namespace["feet_distance"], namespace["feet_distance_v2"]


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is unavailable, the reward cannot be exercised")
        return 1

    feet_distance, feet_distance_v2 = _load_rewards(torch)

    cfg = types.SimpleNamespace(name="robot", body_ids=[0, 1])
    table = sole_table()
    offsets = table.tolist()

    links, joints, root = parse_urdf(os.path.normpath(DEFAULT_URDF))
    rot_nom = {side: forward_kinematics(joints, root, symmetric_pose())[0][foot]
               for side, foot in FOOT.items()}

    def scene(separation: float, yaw: float, base_yaw: float = 0.0, swap: bool = False,
              fore_aft: float = 0.0):
        """One environment holding the pinned pose the analysis script defines, optionally turned."""
        frames = pinned_frames(rot_nom, separation, yaw)
        order = ["left", "right"] if swap else ["right", "left"]
        rz = np.array([[math.cos(base_yaw), -math.sin(base_yaw), 0.0],
                       [math.sin(base_yaw), math.cos(base_yaw), 0.0],
                       [0.0, 0.0, 1.0]])
        shift = {"right": np.array([+fore_aft / 2, 0.0, 0.0]),
                 "left": np.array([-fore_aft / 2, 0.0, 0.0])}
        quats = torch.stack([mat_to_quat(rz @ frames[s][0]) for s in order]).unsqueeze(0)
        poss = torch.tensor(
            np.stack([rz @ (frames[s][1] + shift[s]) for s in order])[None], dtype=torch.float32
        )
        return stub_env(poss, quats, yaw_quat_of(base_yaw), torch.zeros(1, 3))

    def v1(env, **kw):
        return float(feet_distance(env, cfg, ["foot"], MIN_FEET_DISTANCE, 1.0, **kw)[0])

    def v2(env, **kw):
        return float(feet_distance_v2(env, cfg, ["foot"], MIN_FEET_DISTANCE, 1.0, **kw)[0])

    print("Compatibility, the defaults must reproduce feet_distance exactly")
    worst = 0.0
    for sep in (0.10, 0.18, 0.24, 0.32):
        for yaw in (0.0, 0.3, 0.6):
            for lateral in (False, True):
                env = scene(sep, yaw)
                worst = max(worst, abs(v1(env, lateral_only=lateral) - v2(env, lateral_only=lateral)))
    check("every default path reproduces the original", worst < 1e-9, f"worst difference {worst:.3e}")
    env = scene(0.24, 0.5)
    check("an offsets table without a floor is still inert",
          abs(v2(env, sole_offsets=offsets, min_sole_distance=0.0)
              - v1(env)) < 1e-9, "zero as it must be")

    print("\nInvariance, the measure may depend on the relative pose and on nothing else")
    base = v2(scene(0.24, 0.4), sole_offsets=offsets, min_sole_distance=0.08)
    for psi in (0.0, 0.7, -2.1, math.pi):
        turned = v2(scene(0.24, 0.4, base_yaw=psi), sole_offsets=offsets, min_sole_distance=0.08)
        check(f"a whole robot yaw of {psi:+.2f} rad changes nothing", abs(turned - base) < 1e-5,
              f"{turned:.6f} against {base:.6f}")
    swapped = v2(scene(0.24, 0.4, swap=True), sole_offsets=offsets, min_sole_distance=0.08)
    check("the order find_bodies resolves the feet changes nothing", abs(swapped - base) < 1e-5,
          f"{swapped:.6f} against {base:.6f}")

    print("\nAgreement with the independent numpy geometry, over the pinned sweep")
    ordered = order_perimeter(table)
    worst_reward, worst_exact, fires, overlaps = 0.0, 0.0, [], []
    for sep in (0.20, 0.24, 0.28):
        for yaw in np.linspace(0.0, 0.7, 15):
            prints, hulls = {}, {}
            for side, (rot, pos) in pinned_frames(rot_nom, sep, float(yaw)).items():
                world = (ordered @ rot.T + pos)[:, :2]
                prints[side], hulls[side] = world, convex_hull_2d(world)
            exact = polygon_distance(hulls["right"], hulls["left"])
            got = v2(scene(sep, float(yaw)), sole_offsets=offsets, min_sole_distance=0.08)
            if exact <= 0.0:
                # the footprints interpenetrate, where a vertex against edge measure is positive by
                # construction and `filtered_contacts` on the foot pair sensors is the true detector
                overlaps.append(got - max(0.0, MIN_FEET_DISTANCE - sep))
                continue
            expected = max(0.0, MIN_FEET_DISTANCE - sep) + min(1.0, max(0.0, 0.08 - exact))
            worst_reward = max(worst_reward, abs(got - expected))
            worst_exact = max(worst_exact,
                              abs(segment_distance(prints["right"], prints["left"]) - exact))
            if sep == 0.24 and got > 0.0:
                fires.append(float(yaw))
    check("the reward matches the exact polygon distance while disjoint", worst_reward < 2e-4,
          f"worst departure {worst_reward * 1e3:.4f} mm")
    check("the numpy vertex against edge measure is itself exact", worst_exact < 1e-9,
          f"worst departure {worst_exact * 1e3:.6f} mm")
    check("in the overlap regime the sole hinge is near its own ceiling",
          bool(overlaps) and min(overlaps) > 0.9 * 0.08,
          f"least sole hinge over {len(overlaps)} overlapping poses "
          f"{min(overlaps) if overlaps else float('nan'):.5f} against the 0.08 floor")

    print("\nThe defect the new term repairs")
    pose = scene(MIN_FEET_DISTANCE, 0.6)
    old, new = v1(pose), v2(pose, sole_offsets=offsets, min_sole_distance=0.08)
    check("at the threshold separation with the feet turned in, the original is silent",
          abs(old) < 1e-9, f"{old:.6f}")
    check("and the new term is not", new > 0.0, f"{new:.6f}")
    check("the floor first binds at a plausible toe in",
          bool(fires) and 0.2 < min(fires) < 0.6, f"first firing at {min(fires):.3f} rad"
          if fires else "never fires")
    far = scene(MIN_FEET_DISTANCE, 0.0, fore_aft=0.45)
    check("a full step of fore and aft separation does not trip the floor",
          abs(v2(far, sole_offsets=offsets, min_sole_distance=0.08)) < 1e-9,
          "silent, the planar measure being a true distance and not a lateral one")

    print("\nThe lateral path, and the SD_BRS1 table whose points are not declared in order")
    for yaw in (0.0, 0.4):
        frames = pinned_frames(rot_nom, MIN_FEET_DISTANCE, yaw)
        lateral = {side: (ordered @ rot.T + pos)[:, 1] for side, (rot, pos) in frames.items()}
        expect_gap = max(lateral["right"].min() - lateral["left"].max(),
                         lateral["left"].min() - lateral["right"].max())
        env = scene(MIN_FEET_DISTANCE, yaw)
        got = v2(env, lateral_only=True, sole_offsets=offsets, min_sole_distance=0.20)
        expect = max(0.0, MIN_FEET_DISTANCE - MIN_FEET_DISTANCE) + min(1.0, max(0.0, 0.20 - expect_gap))
        check(f"the lateral gap at a toe in of {yaw:.1f} rad matches numpy", abs(got - expect) < 2e-4,
              f"{got:.6f} against {expect:.6f}, gap {expect_gap:.4f} m")
    crossed = v2(scene(0.02, 0.0), lateral_only=True, sole_offsets=offsets, min_sole_distance=0.05)
    check("a crossover drives the signed lateral gap negative", crossed > 0.05,
          f"{crossed:.5f} exceeds the floor, which an unsigned gap could not")

    brs = sole_table("SD_BRS1_SOLE_OFFSETS", CFG_PATH.replace("kscale_base", "brs_base"))
    brs_ordered = np.asarray(order_perimeter(brs))
    normal = int(np.argmin(brs.std(axis=0)))
    plane = [i for i in range(3) if i != normal]
    loop = brs_ordered[:, plane]
    edge = np.roll(loop, -1, axis=0) - loop
    turn = np.cross(edge, np.roll(edge, -1, axis=0))
    check("the BRS table, declared in mirrored pairs, orders into a convex traversal",
          bool(np.all(turn >= -1e-12) or np.all(turn <= 1e-12)),
          f"sole normal is axis {normal}, {len(brs_ordered)} points, every turn one way")

    print()
    if _failures:
        print(f"{len(_failures)} CHECK(S) FAILED")
        for name in _failures:
            print(f"  {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
