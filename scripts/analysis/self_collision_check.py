"""Penetration between the collision shapes of a robot document, at its nominal pose and at reset poses.

Loads each MJCF document with MuJoCo, poses the joints at zero (the nominal pose, the
fold being carried by the body frames) and at random offsets drawn as the reset event
reset_joints_by_offset draws them, and measures the signed distance between every pair
of collision geoms with mj_geomDistance. A pair is reported as adjacent when one body is
the other's parent, the pairs PhysX does not collide within an articulation, and as
active otherwise, the pairs that enabled_self_collisions lets PhysX collide.

Usage:
    python scripts/analysis/self_collision_check.py <mjcf or directory> [--samples 2000]
        [--offset 0.2] [--seed 0]

Runs in a plain interpreter with mujoco and numpy. See context/isaaclab_diagnostics.md.
"""

from __future__ import annotations

import argparse
import glob
import os

import mujoco
import numpy as np


def collision_geoms(model: mujoco.MjModel) -> list[int]:
    """The geoms that take part in contact, attached to a body other than the world."""
    return [
        g for g in range(model.ngeom)
        if model.geom_bodyid[g] != 0 and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]


def pair_table(model: mujoco.MjModel) -> list[tuple[int, int, bool]]:
    geoms = collision_geoms(model)
    pairs = []
    for i, a in enumerate(geoms):
        for b in geoms[i + 1:]:
            ba, bb = model.geom_bodyid[a], model.geom_bodyid[b]
            if ba == bb:
                continue
            adjacent = model.body_parentid[ba] == bb or model.body_parentid[bb] == ba
            pairs.append((a, b, adjacent))
    return pairs


def distances(model, data, pairs, distmax=0.1) -> np.ndarray:
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    fromto = np.zeros(6)
    return np.array([mujoco.mj_geomDistance(model, data, a, b, distmax, fromto) for a, b, _ in pairs])


def name(model, g) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])


def check(path: str, samples: int, offset: float, rng: np.random.Generator) -> dict:
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    pairs = pair_table(model)
    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    qadr = model.jnt_qposadr[hinge]
    lo, hi = model.jnt_range[hinge, 0], model.jnt_range[hinge, 1]

    data.qpos[:] = model.qpos0
    nominal = distances(model, data, pairs)
    worst_active = np.full(len(pairs), np.inf)
    hits = np.zeros(len(pairs), dtype=int)
    for _ in range(samples):
        data.qpos[:] = model.qpos0
        # reset_joints_by_offset adds a uniform offset to the default and clamps to the limits.
        data.qpos[qadr] = np.clip(rng.uniform(-offset, offset, len(hinge)), lo, hi)
        d = distances(model, data, pairs)
        worst_active = np.minimum(worst_active, d)
        hits += d < 0
    return {"model": model, "pairs": pairs, "nominal": nominal, "worst": worst_active, "hits": hits}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", help="An MJCF document or a directory of individual_*.xml")
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--offset", type=float, default=0.2, help="Half range of the reset joint offset (rad).")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    paths = sorted(glob.glob(os.path.join(a.path, "*.xml"))) if os.path.isdir(a.path) else [a.path]
    rng = np.random.default_rng(a.seed)

    summary: dict[tuple[str, str, bool], list] = {}
    for path in paths:
        r = check(path, a.samples, a.offset, rng)
        m = r["model"]
        for k, (g1, g2, adjacent) in enumerate(r["pairs"]):
            if r["nominal"][k] < 0 or r["worst"][k] < 0:
                key = (name(m, g1), name(m, g2), adjacent)
                summary.setdefault(key, []).append(
                    (os.path.basename(path), r["nominal"][k], r["worst"][k], r["hits"][k] / a.samples)
                )

    print(f"{len(paths)} documents, {a.samples} reset poses each, offset +-{a.offset} rad")
    print(f"{'body A':16s} {'body B':16s} {'kind':9s} {'docs':>5s} {'nominal mm':>11s} {'worst mm':>9s} {'reset frac':>10s}")
    for (b1, b2, adjacent), rows in sorted(summary.items(), key=lambda kv: min(r[2] for r in kv[1])):
        nominal = min(r[1] for r in rows) * 1e3
        worst = min(r[2] for r in rows) * 1e3
        frac = max(r[3] for r in rows)
        kind = "adjacent" if adjacent else "ACTIVE"
        print(f"{b1:16s} {b2:16s} {kind:9s} {len(rows):5d} {nominal:11.2f} {worst:9.2f} {frac:10.3f}")


if __name__ == "__main__":
    main()
