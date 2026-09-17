#!/usr/bin/env python3
"""Ground the thresholds of a footprint aware feet distance penalty, and price the approximations it makes.

`feet_distance` at environments/environments/tasks/locomotion/mdp/rewards.py:1048 measures the
planar separation of the two FOOT LINK FRAME ORIGINS, which on a sole footed robot is the ankle
and not the foot. The quantity that decides whether the feet collide is the separation of the two
SOLE POLYGONS projected onto the ground, and the two diverge the moment a foot yaws, because the
toe swings laterally about an origin that need not move. This script quantifies that divergence
from the URDF and the collision meshes, so that the threshold of a new term is chosen from
geometry rather than from taste.

Four measures are computed at every sampled pose.

    origin      the planar norm of the frame origin difference, which is what feet_distance reads
    points      the minimum over all pairs drawn from the two reduced sole tables, the cheap
                measure a reward can afford at every step for every environment
    polygon     the exact minimum distance between the two convex hulls of those same tables,
                which the `points` measure can only overestimate, never understate
    mesh        the exact minimum distance between the convex hulls of every collision vertex of
                the two foot links, projected down, which is the truth the reward approximates

The gap between `points` and `polygon` is the error the cheap reduction commits, and the gap
between `polygon` and `mesh` is the error the twelve point table commits. Both are reported as
worst cases over the sampled envelope, in millimetres, in the manner of the fidelity sweep of
kscale_sole_analysis.py.

Two sweeps are run because they answer different questions. The KINEMATIC sweep drives the real
joints from the nominal pose and shows what a policy can reach, and it establishes the result that
hip yaw alone does not close the feet, the same joint that turns the foot also carrying the ankle
outward. The PINNED sweep places both ankles at a chosen separation and rotates each foot about its
own origin, which isolates the orientation term exactly as the reward sees it once the frame origin
hinge of feet_distance holds the ankles at its own threshold.

Pure numpy, scipy and the standard library, with no dependency on Isaac Lab, on torch or on the
task package, so that it runs inside the simulation container and equally in a plain interpreter.

Usage
    python3 scripts/analysis/kscale_feet_distance_analysis.py
    python3 scripts/analysis/kscale_feet_distance_analysis.py --samples 13 --yaw 0.8
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import re
import sys

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kscale_sole_analysis import (  # noqa: E402
    convex_hull_2d,
    forward_kinematics,
    link_frame_vertices,
    parse_urdf,
)
from kscale_stance_analysis import DEFAULT_URDF, NOMINAL_POSE  # noqa: E402

FOOT = {"right": "foot_6061", "left": "foot_6061_2"}
HIP_YAW = {"right": "right_hip_yaw_03", "left": "left_hip_yaw_03"}
HIP_ROLL = {"right": "right_hip_roll_03", "left": "left_hip_roll_03"}
ANKLE_ROLL = {"right": "right_foot_roll_02", "left": "left_foot_roll_02"}
ANKLE_PITCH = {"right": "right_foot_pitch_02", "left": "left_foot_pitch_02"}

CFG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "environments", "environments", "tasks", "locomotion", "cfg", "SF",
    "kscale_base_env_cfg.py",
)

# The threshold pen_feet_distance carries at kscale_base_env_cfg.py:863. Every pinned figure below
# is quoted at this separation, so the script reports the margin the configured reward leaves
# rather than a bare distance.
MIN_FEET_DISTANCE = 0.24


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def sole_table(name: str = "KSCALE_SOLE_OFFSETS", path: str = CFG_PATH) -> np.ndarray:
    """The sole offsets AS THE CONFIGURATION DECLARES THEM, read from the source rather than copied.

    Duplicating the table here would let the check and the configuration drift apart silently,
    which is the failure this whole script exists to catch one level up.
    """
    with open(os.path.normpath(path)) as handle:
        source = handle.read()
    match = re.search(rf"^{name}\s*=\s*(\[.*?^\])", source, re.S | re.M)
    if match is None:
        raise RuntimeError(f"{name} not found in {path}")
    return np.array(ast.literal_eval(match.group(1)), dtype=np.float64)


def mesh_hulls(links) -> dict[str, np.ndarray]:
    """The three dimensional convex hull vertices of each foot's collision geometry, computed once.

    Every pose below projects this reduced set rather than re-reading the STL, which turns a sweep
    of thousands of poses from minutes into seconds and changes no result, the planar hull of a
    projected point set being determined by the three dimensional hull it lies on.
    """
    return {
        side: (lambda v: v[ConvexHull(v).vertices])(link_frame_vertices(links[foot]))
        for side, foot in FOOT.items()
    }


# ---------------------------------------------------------------------------------------------
# Planar convex geometry
# ---------------------------------------------------------------------------------------------


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-18 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _separated(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """Separating axis test over the edge normals of both convex polygons."""
    for poly in (poly_a, poly_b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12:
                continue
            axis = axis / norm
            pa, pb = poly_a @ axis, poly_b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return True
    return False


def polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Exact minimum distance between two convex polygons, zero when they intersect.

    For two disjoint convex polygons the closest pair is realised at a vertex against an edge or
    at two vertices, both of which the point to segment sweep below covers, so the result is exact
    and not a sample.
    """
    if not _separated(poly_a, poly_b):
        return 0.0
    best = math.inf
    for poly, other in ((poly_a, poly_b), (poly_b, poly_a)):
        for p in poly:
            for i in range(len(other)):
                best = min(best, _point_segment_distance(p, other[i], other[(i + 1) % len(other)]))
    return best


def order_perimeter(table: np.ndarray) -> np.ndarray:
    """The sole table sorted around its own perimeter, which an angular sort gives for a convex set.

    The vertex against edge measure below needs the points in perimeter order so that consecutive
    entries bound an edge, and the configuration is not obliged to declare them that way, the BRS
    table at cfg/SF/brs_base_env_cfg.py:644 being grouped in mirrored pairs rather than traversed.
    The sole normal is the table's axis of least spread, so the sort needs no robot specific
    knowledge of which axis that is, which matters because the KScale foot frame is a full axis
    permutation away from the BRS convention.
    """
    normal = int(np.argmin(table.std(axis=0)))
    plane = [i for i in range(3) if i != normal]
    planar = table[:, plane]
    angle = np.arctan2(*(planar - planar.mean(axis=0)).T[::-1])
    return table[np.argsort(angle)]


def segment_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Minimum over every vertex against every edge, both directions, the points in perimeter order.

    For two DISJOINT convex polygons the closest approach is realised at a vertex against an edge,
    so this is exact rather than sampled, and section 5 of the report demonstrates it numerically.
    It does not detect interpenetration, which is the regime `pen_foot_cross_contact_right` and
    `pen_foot_cross_contact_left` already own through the filtered contact sensors at
    cfg/SF/kscale_base_env_cfg.py:836 and :844.
    """
    best = math.inf
    for near, far in ((pts_a, pts_b), (pts_b, pts_a)):
        edge = np.roll(far, -1, axis=0) - far
        denom = np.maximum((edge * edge).sum(-1), 1e-18)
        delta = near[:, None, :] - far[None, :, :]
        t = np.clip((delta * edge[None]).sum(-1) / denom[None], 0.0, 1.0)
        best = min(best, float(np.linalg.norm(delta - t[..., None] * edge[None], axis=-1).min()))
    return best


def point_pair_distance(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """The measure a reward can afford, being the minimum over the cross product of two point sets."""
    return float(np.linalg.norm(pts_a[:, None, :] - pts_b[None, :, :], axis=-1).min())


def lateral_gap(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    """Signed clearance along the lateral axis, negative when the two footprints overlap in y.

    This is the measure a stance width term wants, the planar norm being unable to distinguish a
    wide stance from a narrow one that happens to be separated fore and aft.
    """
    return max(pts_a[:, 1].min() - pts_b[:, 1].max(), pts_b[:, 1].min() - pts_a[:, 1].max())


# ---------------------------------------------------------------------------------------------
# Pose evaluation
# ---------------------------------------------------------------------------------------------


def _measure(footprints: dict[str, np.ndarray], hulls: dict[str, np.ndarray],
             meshes: dict[str, np.ndarray], origin: float) -> dict[str, float]:
    return {
        "origin": origin,
        "points": point_pair_distance(footprints["right"], footprints["left"]),
        "polygon": polygon_distance(hulls["right"], hulls["left"]),
        "mesh": polygon_distance(meshes["right"], meshes["left"]),
        "lateral": lateral_gap(footprints["right"], footprints["left"]),
    }


def kinematic_measures(joints, root, table, local_hulls, pose) -> dict[str, float]:
    """Every measure of foot separation at one joint configuration, in the root frame.

    The root frame serves rather than the world because the two differ by the root pose alone,
    which cancels from a difference of two foot positions, and because the KScale root frame was
    reframed so that its x is forward and its z is up, which
    scripts/analysis/kscale_feet_heading_selftest.py asserts against the URDF.
    """
    rot, pos = forward_kinematics(joints, root, pose)
    prints, hulls, meshes = {}, {}, {}
    for side, foot in FOOT.items():
        world = table @ rot[foot].T + pos[foot]
        prints[side] = world[:, :2]
        hulls[side] = convex_hull_2d(world[:, :2])
        meshes[side] = convex_hull_2d((local_hulls[side] @ rot[foot].T + pos[foot])[:, :2])
    origin = float(np.linalg.norm((pos[FOOT["right"]] - pos[FOOT["left"]])[:2]))
    return _measure(prints, hulls, meshes, origin)


def pinned_frames(rot_nom, separation, yaw) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rotation and position of each foot with the ankles held apart and each foot yawed in place.

    This is the configuration the reward actually faces once the frame origin hinge is doing its
    work, that hinge fixing the ankles at its threshold and leaving the orientation free. Positive
    yaw is toe in on both feet. The self test feeds these same frames to the torch reward, so the
    two implementations are compared on one pose rather than on two descriptions of a pose.
    """
    frames = {}
    for side, sign in (("right", -1.0), ("left", +1.0)):
        psi = -sign * yaw
        rz = np.array([[math.cos(psi), -math.sin(psi), 0.0],
                       [math.sin(psi), math.cos(psi), 0.0],
                       [0.0, 0.0, 1.0]])
        frames[side] = (rz @ rot_nom[side], np.array([0.0, sign * separation / 2.0, 0.0]))
    return frames


def pinned_measures(rot_nom, table, local_hulls, separation, yaw) -> dict[str, float]:
    """Every measure at a pinned configuration, the frames coming from :func:`pinned_frames`."""
    prints, hulls, meshes = {}, {}, {}
    for side, (rot, pos) in pinned_frames(rot_nom, separation, yaw).items():
        world = table @ rot.T + pos
        prints[side] = world[:, :2]
        hulls[side] = convex_hull_2d(world[:, :2])
        meshes[side] = convex_hull_2d((local_hulls[side] @ rot.T + pos)[:, :2])
    return _measure(prints, hulls, meshes, separation)


def symmetric_pose(yaw: float = 0.0, roll: float = 0.0, pitch: float = 0.0,
                   hip_roll: float = 0.0) -> dict[str, float]:
    """The nominal pose with a mirrored perturbation, the sign convention fixed by the sweep below.

    The two hip yaw axes are co directed, which scripts/analysis/kscale_feet_heading_selftest.py
    asserts from the URDF, so a mirrored motion demands opposite signs at the two joints.
    """
    pose = dict(NOMINAL_POSE)
    pose[HIP_YAW["right"]] = +yaw
    pose[HIP_YAW["left"]] = -yaw
    pose[HIP_ROLL["right"]] = +hip_roll
    pose[HIP_ROLL["left"]] = -hip_roll
    pose[ANKLE_ROLL["right"]] = +roll
    pose[ANKLE_ROLL["left"]] = -roll
    pose[ANKLE_PITCH["right"]] = NOMINAL_POSE[ANKLE_PITCH["right"]] + pitch
    pose[ANKLE_PITCH["left"]] = NOMINAL_POSE[ANKLE_PITCH["left"]] + pitch
    return pose


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--yaw", type=float, default=0.6, help="foot yaw half range swept (rad)")
    parser.add_argument("--roll", type=float, default=0.3, help="ankle roll half range swept (rad)")
    parser.add_argument("--pitch", type=float, default=0.3, help="ankle pitch half range swept (rad)")
    parser.add_argument("--samples", type=int, default=9, help="samples per swept axis")
    args = parser.parse_args()

    path = os.path.normpath(args.urdf)
    links, joints, root = parse_urdf(path)
    table, local_hulls = sole_table(), mesh_hulls(links)
    rot_nom = {side: forward_kinematics(joints, root, symmetric_pose())[0][foot]
               for side, foot in FOOT.items()}

    print(f"URDF  {path}")
    print(f"table {len(table)} points, x {table[:, 0].min():+.4f} to {table[:, 0].max():+.4f}, "
          f"z {table[:, 2].min():+.4f} to {table[:, 2].max():+.4f}")

    nominal = kinematic_measures(joints, root, table, local_hulls, symmetric_pose())
    print("\n1. At the nominal pose of kscale_stance_analysis.NOMINAL_POSE")
    for key in ("origin", "points", "polygon", "mesh", "lateral"):
        print(f"     {key:8s} {nominal[key]:.4f} m")
    print(f"   the origin measure exceeds the true mesh separation by "
          f"{(nominal['origin'] - nominal['mesh']) * 1e3:.1f} mm, which is the sole the reward "
          f"cannot see")

    print("\n2. Hip yaw driven symmetrically from the nominal pose, the KINEMATIC sweep")
    print(f"     {'yaw':>6} {'origin':>9} {'points':>9} {'mesh':>9} {'v1 penalty':>11}")
    for yaw in np.linspace(0.0, args.yaw, 7):
        m = kinematic_measures(joints, root, table, local_hulls, symmetric_pose(yaw=float(yaw)))
        print(f"     {yaw:6.3f} {m['origin']:9.4f} {m['points']:9.4f} {m['mesh']:9.4f} "
              f"{max(0.0, MIN_FEET_DISTANCE - m['origin']):11.4f}")
    print("   hip yaw alone does NOT close the feet, the joint that turns the foot also carrying")
    print("   the ankle outward, so the footprint gap is very nearly flat while the origin grows")

    print("\n3. Hip roll driven symmetrically, adduction bringing the legs together")
    print(f"     {'hip roll':>8} {'origin':>9} {'points':>9} {'mesh':>9} {'v1 penalty':>11}")
    sign = None
    for probe in (+0.2, -0.2):
        m = kinematic_measures(joints, root, table, local_hulls, symmetric_pose(hip_roll=probe))
        if sign is None or m["mesh"] < sign[1]:
            sign = (probe / abs(probe), m["mesh"])
    for roll in np.linspace(0.0, 0.35, 8) * sign[0]:
        m = kinematic_measures(joints, root, table, local_hulls, symmetric_pose(hip_roll=float(roll)))
        print(f"     {roll:8.3f} {m['origin']:9.4f} {m['points']:9.4f} {m['mesh']:9.4f} "
              f"{max(0.0, MIN_FEET_DISTANCE - m['origin']):11.4f}")
    print("   this is the route by which the feet truly meet, and the v1 hinge does fire on it,")
    print("   the origin and the footprint falling together while the feet stay parallel")

    print(f"\n4. Ankles PINNED at the v1 threshold of {MIN_FEET_DISTANCE:.2f} m, foot yaw swept")
    print(f"     {'yaw':>6} {'points':>9} {'polygon':>9} {'mesh':>9} {'lateral':>9} {'v1 penalty':>11}")
    for yaw in np.linspace(0.0, args.yaw, 7):
        m = pinned_measures(rot_nom, table, local_hulls, MIN_FEET_DISTANCE, float(yaw))
        print(f"     {yaw:6.3f} {m['points']:9.4f} {m['polygon']:9.4f} {m['mesh']:9.4f} "
              f"{m['lateral']:9.4f} {max(0.0, MIN_FEET_DISTANCE - m['origin']):11.4f}")
    print("   here the orientation term stands alone, and the v1 penalty is identically zero down")
    print("   the whole column because the ankles never move, which is the defect in one table")

    worst_points, worst_table, at_points, at_table = 0.0, 0.0, None, None
    grid = np.linspace(-1.0, 1.0, args.samples)
    for y in grid * args.yaw:
        for r in grid * args.roll:
            for p in grid * args.pitch:
                m = kinematic_measures(joints, root, table, local_hulls,
                                       symmetric_pose(float(y), float(r), float(p)))
                if m["points"] - m["polygon"] > worst_points:
                    worst_points, at_points = m["points"] - m["polygon"], (y, r, p)
                if m["polygon"] - m["mesh"] > worst_table:
                    worst_table, at_table = m["polygon"] - m["mesh"], (y, r, p)
    for sep in np.linspace(0.10, MIN_FEET_DISTANCE, args.samples):
        for y in grid * args.yaw:
            m = pinned_measures(rot_nom, table, local_hulls, float(sep), float(y))
            if m["points"] - m["polygon"] > worst_points:
                worst_points, at_points = m["points"] - m["polygon"], (y, sep, 0.0)
            if m["polygon"] - m["mesh"] > worst_table:
                worst_table, at_table = m["polygon"] - m["mesh"], (y, sep, 0.0)

    print(f"\n5. Fidelity over the kinematic grid and the pinned grid together")
    print(f"     point pairs over the exact table polygon : {worst_points * 1e3:7.3f} mm at "
          f"{'%.3f, %.3f, %.3f' % at_points if at_points else 'nowhere'}")
    print(f"     table polygon over the collision mesh    : {worst_table * 1e3:7.3f} mm at "
          f"{'%.3f, %.3f, %.3f' % at_table if at_table else 'nowhere'}")
    print("   a positive figure is an OVERESTIMATE of the separation, so the reward reads the feet")
    print("   as further apart than they are and under penalises by that margin")

    print("\n5b. The two measures a reward can implement, against the exact polygon distance,")
    print("    over the pinned grid restricted to the DISJOINT regime, interpenetration being the")
    print("    regime pen_foot_cross_contact_right and _left already own through the filtered sensors")
    ordered = order_perimeter(table)
    worst_pp = worst_sg = 0.0
    at_pp = at_sg = None
    for sep in np.linspace(0.08, 0.30, 3 * args.samples):
        for yaw in np.linspace(0.0, 1.0, 6 * args.samples) * args.yaw / 0.6:
            prints, hulls = {}, {}
            for side, (rot, pos) in pinned_frames(rot_nom, float(sep), float(yaw)).items():
                world = (ordered @ rot.T + pos)[:, :2]
                prints[side], hulls[side] = world, convex_hull_2d(world)
            exact = polygon_distance(hulls["right"], hulls["left"])
            if exact <= 0.0:
                continue
            error_pp = point_pair_distance(prints["right"], prints["left"]) - exact
            error_sg = abs(segment_distance(prints["right"], prints["left"]) - exact)
            if error_pp > worst_pp:
                worst_pp, at_pp = error_pp, (sep, yaw)
            if error_sg > worst_sg:
                worst_sg, at_sg = error_sg, (sep, yaw)
    print(f"     vertex against vertex, P squared pairs : {worst_pp * 1e3:8.4f} mm at "
          f"{'sep %.3f yaw %.3f' % at_pp if at_pp else 'nowhere'}")
    print(f"     vertex against edge, twice that count  : {worst_sg * 1e3:8.4f} mm at "
          f"{'sep %.3f yaw %.3f' % at_sg if at_sg else 'nowhere'}")
    print("    the vertex against edge measure is exact for disjoint convex polygons and so errs")
    print("    only at machine precision, at twice the cost of the vertex pair measure")

    print("\n6. Threshold guidance, from the pinned sweep at the configured origin threshold")
    for floor in (0.02, 0.04, 0.06, 0.08):
        crossing = None
        for yaw in np.linspace(0.0, 1.2, 241):
            if pinned_measures(rot_nom, table, local_hulls, MIN_FEET_DISTANCE, float(yaw))["points"] < floor:
                crossing = yaw
                break
        where = f"{crossing:.3f} rad of symmetric toe in" if crossing is not None else "not reached below 1.2 rad"
        print(f"     a floor of {floor:.2f} m first binds at {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
