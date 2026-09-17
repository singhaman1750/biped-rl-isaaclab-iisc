"""Audit the KScale URDF for collision overlaps between links that PhysX does not auto filter.

Motivation. Setting ``enabled_self_collisions=True`` on the KScale articulation root collapsed the
mean episode length to about two steps, which points at a contact present in the reset pose rather
than at one the policy creates. PhysX filters only directly jointed parent and child pairs, so every
other pair of links is live from the first physics step, and any pair whose collision geometry
already interpenetrates at reset generates a depenetration impulse immediately.

Method. The geometry PhysX sees is not the STL. The IsaacLab URDF converter defaults to
``collider_type="convex_hull"`` at IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py:119,
and the KScale spawn configuration does not override it, so each ``<collision>`` element becomes one
convex hull. This script therefore hulls every collision part separately, places it by forward
kinematics, and tests hull against hull.

For two convex hulls expressed as halfspace sets ``A x <= a`` and ``B x <= b``, overlap is decided by
the Chebyshev centre linear program, which maximises ``r`` subject to ``n_i . x + r |n_i| <= c_i`` over
the constraints of both bodies. A feasible ``r > 0`` proves the intersection contains a ball of that
radius, so ``r`` is a scale invariant severity measure that does not depend on mesh tessellation. The
deepest vertex depth and the intersection volume are reported alongside it, the former being the
largest distance by which a vertex of one hull lies inside the other and the latter the volume of the
polytope formed by stacking both constraint sets.

Usage:
    python3 scripts/analysis/kscale_self_collision_audit.py [--pose nominal|zero] [--top N]
    python3 scripts/analysis/kscale_self_collision_audit.py --torso-sweep 20000
"""

from __future__ import annotations

import argparse
import itertools
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

URDF_PATH = (
    "/ws/tron1-rl-isaaclab-cozum/environments/environments/assets/urdf/solefoot/kscale/kscale.urdf"
)

# from init_state.joint_pos in environments/environments/assets/config/kscale_identified_cfg.py
NOMINAL_POSE = {
    "right_hip_pitch_04": -0.1,
    "left_hip_pitch_04": 0.1,
    "right_hip_roll_03": 0.0,
    "left_hip_roll_03": 0.0,
    "right_hip_yaw_03": 0.0,
    "left_hip_yaw_03": 0.0,
    "right_knee_04": 0.4,
    "left_knee_04": 0.4,
    "right_foot_pitch_02": -0.3,
    "left_foot_pitch_02": -0.3,
    "right_foot_roll_02": 0.0,
    "left_foot_roll_02": 0.0,
}


def rpy_to_matrix(rpy):
    """Fixed axis roll pitch yaw, the URDF convention, giving Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def transform(xyz, rpy):
    t = np.eye(4)
    t[:3, :3] = rpy_to_matrix(rpy)
    t[:3, 3] = xyz
    return t


def axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def parse_origin(elem):
    o = elem.find("origin")
    if o is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
    rpy = np.array([float(v) for v in o.get("rpy", "0 0 0").split()])
    return xyz, rpy


def parse_urdf(path):
    root = ET.parse(path).getroot()
    mesh_dir = os.path.dirname(path)

    links = {}
    for link in root.findall("link"):
        name = link.get("name")
        parts = []
        for i, col in enumerate(link.findall("collision")):
            mesh = col.find("geometry/mesh")
            if mesh is None:
                continue
            xyz, rpy = parse_origin(col)
            parts.append({
                "index": i,
                "file": os.path.join(mesh_dir, mesh.get("filename")),
                "T_link_part": transform(xyz, rpy),
            })
        links[name] = parts

    joints = []
    for joint in root.findall("joint"):
        xyz, rpy = parse_origin(joint)
        axis_elem = joint.find("axis")
        axis = (
            np.array([float(v) for v in axis_elem.get("xyz").split()])
            if axis_elem is not None
            else np.array([0.0, 0.0, 1.0])
        )
        joints.append({
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "child": joint.find("child").get("link"),
            "T_origin": transform(xyz, rpy),
            "axis": axis,
        })
    return links, joints


def forward_kinematics(links, joints, pose):
    """World pose of every link frame, root at identity."""
    children = {j["parent"]: [] for j in joints}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    all_children = {j["child"] for j in joints}
    roots = [n for n in links if n not in all_children]

    T = {}
    for r in roots:
        T[r] = np.eye(4)
        stack = [r]
        while stack:
            parent = stack.pop()
            for j in children.get(parent, []):
                local = j["T_origin"]
                if j["type"] in ("revolute", "continuous"):
                    q = pose.get(j["name"], 0.0)
                    rot = np.eye(4)
                    rot[:3, :3] = axis_angle_to_matrix(j["axis"], q)
                    local = local @ rot
                T[j["child"]] = T[parent] @ local
                stack.append(j["child"])
    return T


def build_hulls(links, joints, pose, cache):
    """Convex hull vertices of every collision part, in world coordinates."""
    T_world = forward_kinematics(links, joints, pose)
    hulls = []
    for link_name, parts in links.items():
        for part in parts:
            path = part["file"]
            if path not in cache:
                mesh = trimesh.load(path, force="mesh")
                cache[path] = np.asarray(mesh.vertices, dtype=float)
            v = cache[path]
            T = T_world[link_name] @ part["T_link_part"]
            w = v @ T[:3, :3].T + T[:3, 3]
            try:
                hull = ConvexHull(w)
            except Exception:
                continue
            hulls.append({
                "link": link_name,
                "part": os.path.basename(path).replace(".stl", "") + f"#{part['index']}",
                "vertices": w[hull.vertices],
                # ConvexHull.equations are [n, d] with n.x + d <= 0 inside, so n.x <= -d
                "normals": hull.equations[:, :3],
                "offsets": -hull.equations[:, 3],
                "lo": w.min(axis=0),
                "hi": w.max(axis=0),
            })
    return hulls


def chebyshev_overlap(h1, h2):
    """Radius of the largest ball inside the intersection, with its centre. Negative means disjoint."""
    n = np.vstack([h1["normals"], h2["normals"]])
    c = np.concatenate([h1["offsets"], h2["offsets"]])
    norms = np.linalg.norm(n, axis=1)
    # variables are [x, y, z, r], maximise r
    a_ub = np.hstack([n, norms[:, None]])
    res = linprog(
        c=np.array([0.0, 0.0, 0.0, -1.0]),
        A_ub=a_ub,
        b_ub=c,
        bounds=[(None, None)] * 3 + [(None, None)],
        method="highs",
    )
    if not res.success:
        return -np.inf, None
    return res.x[3], res.x[:3]


def joint_limits(scale=0.9):
    """Revolute joint ranges from the URDF, narrowed by ``soft_joint_pos_limit_factor``."""
    root = ET.parse(URDF_PATH).getroot()
    lims = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or joint.get("type") != "revolute":
            continue
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * scale
        lims[joint.get("name")] = (mid - half, mid + half)
    return lims


def local_hulls(links):
    """Convex hull of every collision part, once, in its link frame."""
    out = {}
    for link_name, parts in links.items():
        out[link_name] = []
        for part in parts:
            v = np.asarray(trimesh.load(part["file"], force="mesh").vertices, dtype=float)
            T = part["T_link_part"]
            v = v @ T[:3, :3].T + T[:3, 3]
            hull = ConvexHull(v)
            out[link_name].append((v[hull.vertices], hull.equations[:, :3], -hull.equations[:, 3]))
    return out


def place(local, link_name, T_world):
    """Rigidly transform precomputed link frame hulls into world coordinates."""
    R, t = T_world[link_name][:3, :3], T_world[link_name][:3, 3]
    placed = []
    for v, n, c in local[link_name]:
        w = v @ R.T + t
        n_w = n @ R.T
        placed.append({
            "vertices": w,
            "normals": n_w,
            "offsets": c + n_w @ t,
            "lo": w.min(axis=0),
            "hi": w.max(axis=0),
        })
    return placed


def torso_sweep(links, joints, torso, num_poses, seed=1):
    """Which links can reach the torso, over poses drawn from the joint ranges.

    PhysX filters only directly jointed pairs, so the links reported here are those that would be
    live against the torso once self collision is enabled. The result determines which pairs the
    ``SELF_COLLISION_FILTERED_PAIRS`` list of the asset configuration must carry in order that the
    torso report ground contact alone, which is what the ``base_contact`` termination and the torso
    entry of ``pen_undesired_contacts`` are calibrated against.
    """
    jointed = {frozenset((j["parent"], j["child"])) for j in joints}
    local = local_hulls(links)
    lims = joint_limits()
    names = sorted(lims)
    candidates = [
        n for n in links if n != torso and frozenset((n, torso)) not in jointed
    ]

    rng = np.random.default_rng(seed)
    result = {n: [0, 0.0] for n in candidates}
    for _ in range(num_poses):
        pose = {n: rng.uniform(*lims[n]) for n in names}
        T = forward_kinematics(links, joints, pose)
        torso_hulls = place(local, torso, T)
        for name in candidates:
            best = 0.0
            for h1 in torso_hulls:
                for h2 in place(local, name, T):
                    if np.any(h1["lo"] > h2["hi"]) or np.any(h2["lo"] > h1["hi"]):
                        continue
                    r, _ = chebyshev_overlap(h1, h2)
                    best = max(best, r)
            if best > 1e-5:
                result[name][0] += 1
                result[name][1] = max(result[name][1], best)
    return result, sorted(j["child"] for j in joints if j["parent"] == torso)


def deepest_vertex(h_from, h_into):
    """Largest distance by which a vertex of one hull lies inside the other."""
    slack = h_into["offsets"][None, :] - h_from["vertices"] @ h_into["normals"].T
    inside = slack.min(axis=1)
    return max(inside.max(), 0.0)


def intersection_volume(h1, h2, interior):
    n = np.vstack([h1["normals"], h2["normals"]])
    c = np.concatenate([h1["offsets"], h2["offsets"]])
    halfspaces = np.hstack([n, -c[:, None]])
    try:
        hs = HalfspaceIntersection(halfspaces, interior)
        return ConvexHull(hs.intersections).volume
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", choices=["nominal", "zero"], default="nominal")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--radius-threshold", type=float, default=1e-5)
    ap.add_argument(
        "--torso-sweep",
        type=int,
        default=0,
        metavar="N",
        help="instead of the single pose audit, report which links can reach the torso over N poses",
    )
    args = ap.parse_args()

    links, joints = parse_urdf(URDF_PATH)

    if args.torso_sweep:
        torso = "assy_formfg___kd_b_102b_torso_btm"
        result, auto = torso_sweep(links, joints, torso, args.torso_sweep)
        print(f"torso vs every non auto filtered link, N={args.torso_sweep} poses")
        print(f"{'link':<28} {'hit %':>8} {'max r mm':>9}")
        print("-" * 49)
        for name in sorted(result):
            count, r = result[name]
            print(f"{name:<28} {100 * count / args.torso_sweep:>7.2f}% {r * 1000:>9.2f}")
        print()
        print(f"auto filtered by PhysX, being directly jointed to the torso: {', '.join(auto)}")
        return
    pose = NOMINAL_POSE if args.pose == "nominal" else {}

    # PhysX filters directly jointed parent and child pairs only
    jointed = {frozenset((j["parent"], j["child"])) for j in joints}

    cache = {}
    hulls = build_hulls(links, joints, pose, cache)

    print(f"pose              : {args.pose}")
    print(f"links             : {len(links)}")
    print(f"collision parts   : {len(hulls)}")
    print(f"auto filtered pairs (directly jointed): {len(jointed)}")
    ext = np.vstack([h["hi"] - h["lo"] for h in hulls])
    print(f"part extent range : {ext.min():.4f} to {ext.max():.4f} m  (sanity check on mesh units)")
    print()

    findings = []
    checked = 0
    for h1, h2 in itertools.combinations(hulls, 2):
        if h1["link"] == h2["link"]:
            continue  # same rigid body, never collides with itself
        if frozenset((h1["link"], h2["link"])) in jointed:
            continue  # PhysX filters these
        if np.any(h1["lo"] > h2["hi"]) or np.any(h2["lo"] > h1["hi"]):
            continue  # cheap AABB reject
        checked += 1
        r, centre = chebyshev_overlap(h1, h2)
        if r <= args.radius_threshold:
            continue
        findings.append({
            "pair": (h1["link"], h2["link"]),
            "parts": (h1["part"], h2["part"]),
            "radius": r,
            "depth": max(deepest_vertex(h1, h2), deepest_vertex(h2, h1)),
            "volume": intersection_volume(h1, h2, centre),
        })

    print(f"part pairs surviving AABB reject: {checked}")
    print(f"interpenetrating part pairs     : {len(findings)}")
    print()

    findings.sort(key=lambda f: -f["radius"])
    link_pairs = {}
    for f in findings:
        key = tuple(sorted(f["pair"]))
        link_pairs[key] = max(link_pairs.get(key, 0.0), f["radius"])

    print(f"{'link A':<36} {'link B':<36} {'ball r mm':>10}")
    print("-" * 86)
    for (a, b), r in sorted(link_pairs.items(), key=lambda kv: -kv[1]):
        print(f"{a:<36} {b:<36} {r * 1000:>10.2f}")
    print()

    print("worst part pairs")
    print(
        f"{'part A':<30} {'part B':<30} {'r mm':>8} {'depth mm':>9} {'vol cm3':>9}"
    )
    print("-" * 90)
    for f in findings[: args.top]:
        vol = f["volume"] * 1e6
        print(
            f"{f['parts'][0]:<30} {f['parts'][1]:<30} "
            f"{f['radius'] * 1000:>8.2f} {f['depth'] * 1000:>9.2f} {vol:>9.3f}"
        )


if __name__ == "__main__":
    main()
