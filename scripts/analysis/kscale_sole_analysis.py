"""Sole geometry extraction for a URDF biped, from its collision meshes.

Produces the ``sole_offsets`` table that ``foot_clearance_reward_v3`` and
``foot_landing_vel_v2`` consume (environments/environments/tasks/locomotion/mdp/rewards.py),
being a short list of [x, y, z] points in the foot body's own link frame whose lowest world
height, after rotation by the foot's world quaternion, is the true tilt invariant sole
clearance. A scalar foot_radius cannot serve that purpose on a sole foot, because pure tilt
moves the link frame origin by an amount comparable to the whole intended swing clearance,
which is the defect recorded at /ws/context/brs_gait.md:111 and priced at /ws/plans/GAIT_STRATEGY.md:184.

Pure numpy and the standard library, with no dependency on Isaac Lab, on torch or on the task
package, in the manner of scripts/analysis/stats.py, so that it runs inside the simulation
container and equally in a plain interpreter against a checked out URDF.

The script makes no assumption that a foot link frame is axis aligned with world vertical. It
determines the downward axis by forward kinematics at a reference pose, which matters because
the kscale foot frame is a full axis permutation away from the SD_BRS1 convention, its local y
being the vertical and its local z the fore aft length. Reading the minimum of link frame z on
that robot returns 0.19, the foot's length, and not its depth, which is the misreading recorded
in the superseded comment at cfg/SF/kscale_base_env_cfg.py.

Two results are emitted beside the table. The first is a frame convention check, reporting which
root frame axis separates the two feet, since Isaac Lab evaluates the velocity command and the
base linear velocity observation in the root body frame and therefore requires forward to be the
root frame x. The second is a fidelity check, sweeping the foot through its own URDF roll and
pitch limits and reporting the worst height error the reduced table commits against the full
vertex set, which is the figure that justifies the chosen point count.

Usage
    python3 scripts/analysis/kscale_sole_analysis.py \
        --urdf environments/environments/assets/urdf/solefoot/kscale/kscale.urdf \
        --feet 'foot_6061.*' --points 12
"""

from __future__ import annotations

import argparse
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# Vertices are treated as lying on one plane when they fall within this distance of the extreme,
# in metres. One millimetre is loose enough to absorb the export's float32 quantisation and tight
# enough to exclude the recessed interior face that sits four millimetres above the kscale rim.
PLANE_TOLERANCE = 1.0e-3


# ---------------------------------------------------------------------------------------------
# URDF parsing and forward kinematics
# ---------------------------------------------------------------------------------------------


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Fixed axis extrinsic X-Y-Z rotation, the convention the URDF specification mandates."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    r_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    r_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return r_z @ r_y @ r_x


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation, used to apply a joint angle and to sweep the fidelity check."""
    a = axis / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray | None
    lower: float | None
    upper: float | None


@dataclass
class Geometry:
    """One <collision> or <visual> element, its mesh path and its placement in the link frame."""

    mesh_path: str
    origin_xyz: np.ndarray
    origin_rot: np.ndarray


@dataclass
class Link:
    name: str
    collisions: list[Geometry] = field(default_factory=list)
    visuals: list[Geometry] = field(default_factory=list)


def _vec(text: str | None, default: str) -> np.ndarray:
    return np.array([float(v) for v in (text if text else default).split()], dtype=np.float64)


def _geometries(link_elem: ET.Element, tag: str, urdf_dir: str) -> list[Geometry]:
    out: list[Geometry] = []
    for elem in link_elem.findall(tag):
        mesh = elem.find("geometry/mesh")
        if mesh is None:
            continue
        origin = elem.find("origin")
        xyz = _vec(origin.get("xyz") if origin is not None else None, "0 0 0")
        rpy = _vec(origin.get("rpy") if origin is not None else None, "0 0 0")
        filename = re.sub(r"^(package|file)://", "", mesh.get("filename", ""))
        out.append(Geometry(os.path.join(urdf_dir, filename), xyz, rpy_to_matrix(*rpy)))
    return out


def parse_urdf(path: str) -> tuple[dict[str, Link], dict[str, Joint], str]:
    """Return the links, the joints, and the name of the root link, being the one with no parent."""
    root = ET.parse(path).getroot()
    urdf_dir = os.path.dirname(os.path.abspath(path))

    links: dict[str, Link] = {}
    for elem in root.iter("link"):
        name = elem.get("name", "")
        links[name] = Link(name, _geometries(elem, "collision", urdf_dir), _geometries(elem, "visual", urdf_dir))

    joints: dict[str, Joint] = {}
    for elem in root.iter("joint"):
        origin = elem.find("origin")
        axis_elem = elem.find("axis")
        limit = elem.find("limit")
        joints[elem.get("name", "")] = Joint(
            name=elem.get("name", ""),
            type=elem.get("type", "fixed"),
            parent=elem.find("parent").get("link", ""),  # type: ignore[union-attr]
            child=elem.find("child").get("link", ""),  # type: ignore[union-attr]
            origin_xyz=_vec(origin.get("xyz") if origin is not None else None, "0 0 0"),
            origin_rot=rpy_to_matrix(*_vec(origin.get("rpy") if origin is not None else None, "0 0 0")),
            axis=_vec(axis_elem.get("xyz"), "1 0 0") if axis_elem is not None else None,
            lower=float(limit.get("lower")) if limit is not None and limit.get("lower") else None,
            upper=float(limit.get("upper")) if limit is not None and limit.get("upper") else None,
        )

    childed = {j.child for j in joints.values()}
    roots = [n for n in links if n not in childed]
    if len(roots) != 1:
        raise ValueError(f"expected exactly one root link, found {roots}")
    return links, joints, roots[0]


def forward_kinematics(
    joints: dict[str, Joint], root: str, joint_pos: dict[str, float] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Rotations and positions of every link frame in the root frame, at the given joint angles."""
    joint_pos = joint_pos or {}
    by_parent: dict[str, list[Joint]] = {}
    for j in joints.values():
        by_parent.setdefault(j.parent, []).append(j)

    rot = {root: np.eye(3)}
    pos = {root: np.zeros(3)}
    stack = [root]
    while stack:
        parent = stack.pop()
        for j in by_parent.get(parent, []):
            local = j.origin_rot
            if j.type in ("revolute", "continuous") and j.axis is not None:
                local = local @ axis_angle_to_matrix(j.axis, joint_pos.get(j.name, 0.0))
            rot[j.child] = rot[parent] @ local
            pos[j.child] = pos[parent] + rot[parent] @ j.origin_xyz
            stack.append(j.child)
    return rot, pos


# ---------------------------------------------------------------------------------------------
# STL parsing
# ---------------------------------------------------------------------------------------------


def load_stl(path: str) -> np.ndarray:
    """Vertices of a binary or ASCII STL, shape (3 * num_triangles, 3), in the mesh's own frame."""
    with open(path, "rb") as handle:
        blob = handle.read()
    if blob[:5].lstrip().lower().startswith(b"solid") and b"facet" in blob[:512].lower():
        verts = [
            [float(v) for v in line.split()[1:4]]
            for line in blob.decode("utf-8", "replace").splitlines()
            if line.strip().lower().startswith("vertex")
        ]
        return np.array(verts, dtype=np.float64)
    count = struct.unpack("<I", blob[80:84])[0]
    flat = np.array(
        [struct.unpack_from("<9f", blob, 84 + i * 50 + 12) for i in range(count)], dtype=np.float64
    )
    return flat.reshape(-1, 3)


def link_frame_vertices(link: Link, prefer_collision: bool = True) -> np.ndarray:
    """Every mesh vertex of a link, transformed through its <origin> into the link frame."""
    sources = (link.collisions or link.visuals) if prefer_collision else (link.visuals or link.collisions)
    if not sources:
        raise ValueError(f"link {link.name} carries no mesh geometry")
    return np.vstack([load_stl(g.mesh_path) @ g.origin_rot.T + g.origin_xyz for g in sources])


# ---------------------------------------------------------------------------------------------
# Sole extraction
# ---------------------------------------------------------------------------------------------


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain hull, counterclockwise, without the closing duplicate point."""
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def build(seq: np.ndarray) -> list[np.ndarray]:
        chain: list[np.ndarray] = []
        for p in seq:
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0:
                    break
                chain.pop()
            chain.append(p)
        return chain

    return np.array(build(pts)[:-1] + build(pts[::-1])[:-1])


def down_axis_in_link_frame(link_rot: np.ndarray) -> tuple[int, float]:
    """Index and sign of the link frame axis most nearly aligned with world down at this pose.

    Returns (index, sign) such that ``sign * vertices[:, index]`` increases downward, so the sole
    is the set maximising that quantity. This is what removes the assumption, false on kscale,
    that a foot link's own z is its vertical.
    """
    world_down = np.array([0.0, 0.0, -1.0])
    projections = link_rot.T @ world_down
    index = int(np.argmax(np.abs(projections)))
    return index, float(np.sign(projections[index]))


def extract_sole(
    vertices: np.ndarray, down_index: int, down_sign: float, tolerance: float = PLANE_TOLERANCE
) -> tuple[np.ndarray, float, np.ndarray]:
    """The sole plane depth, and the convex hull of the contact points lying on it.

    Returns (hull_points_3d, depth, plane_members). The depth is signed in the link frame along
    the down axis, so it is the coordinate the reward's offsets must carry on that axis.
    """
    depth_coord = down_sign * vertices[:, down_index]
    extreme = float(depth_coord.max())
    members = vertices[depth_coord > extreme - tolerance]

    lateral = [i for i in range(3) if i != down_index]
    hull_2d = convex_hull_2d(members[:, lateral])
    hull = np.zeros((len(hull_2d), 3))
    hull[:, lateral[0]] = hull_2d[:, 0]
    hull[:, lateral[1]] = hull_2d[:, 1]
    hull[:, down_index] = down_sign * extreme
    return hull, down_sign * extreme, members


def reduce_hull(hull: np.ndarray, down_index: int, target: int) -> np.ndarray:
    """Keep the ``target`` hull vertices that best preserve the polygon, by iterative area loss.

    A sole polygon exported from CAD carries a rounded toe of many near collinear vertices, and a
    naive truncation drops the widest part of the arc. Removing repeatedly the vertex whose
    deletion costs the least area keeps the extremes that dominate the tilted minimum.
    """
    lateral = [i for i in range(3) if i != down_index]
    keep = list(range(len(hull)))
    while len(keep) > target:
        pts = hull[keep][:, lateral]
        prev, nxt = np.roll(pts, 1, axis=0), np.roll(pts, -1, axis=0)
        area = np.abs(
            (nxt[:, 0] - prev[:, 0]) * (pts[:, 1] - prev[:, 1])
            - (pts[:, 0] - prev[:, 0]) * (nxt[:, 1] - prev[:, 1])
        )
        keep.pop(int(np.argmin(area)))
    return hull[keep]


def align_down_to_world(down_index: int, down_sign: float) -> np.ndarray:
    """Rotation carrying the link frame to a world frame in which the sole faces world down.

    The fidelity sweep must tilt the foot about its own lateral axes, not about the world axes of
    an arbitrarily permuted link frame, so the comparison is performed after this alignment.
    """
    down = np.zeros(3)
    down[down_index] = down_sign
    forward = np.zeros(3)
    forward[[i for i in range(3) if i != down_index][1]] = 1.0
    z_axis = -down
    x_axis = forward - np.dot(forward, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.vstack([x_axis, y_axis, z_axis])


def worst_tilt_error(
    vertices: np.ndarray,
    table: np.ndarray,
    down_index: int,
    down_sign: float,
    roll_limit: float,
    pitch_limit: float,
    samples: int = 41,
) -> tuple[float, tuple[float, float]]:
    """Largest height the reduced table misses against the full mesh, over the foot's own limits.

    Both sets are first aligned so the sole faces world down, then rotated identically through the
    ankle's reachable roll and pitch, and compared on their minimum world height, which is exactly
    the quantity ``_sole_points_world`` reduces. A table whose convex hull bounds the mesh under
    every reachable orientation commits zero error, so a nonzero result is the clearance the reward
    would overestimate at the worst attitude.
    """
    align = align_down_to_world(down_index, down_sign)
    mesh_w = vertices @ align.T
    table_w = table @ align.T

    worst, at = 0.0, (0.0, 0.0)
    for roll in np.linspace(-roll_limit, roll_limit, samples):
        r_roll = axis_angle_to_matrix(np.array([1.0, 0.0, 0.0]), roll)
        for pitch in np.linspace(-pitch_limit, pitch_limit, samples):
            rot = r_roll @ axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), pitch)
            error = float((table_w @ rot.T)[:, 2].min() - (mesh_w @ rot.T)[:, 2].min())
            if error > worst:
                worst, at = error, (float(roll), float(pitch))
    return worst, at


def ankle_limits(joints: dict[str, Joint], foot: str) -> tuple[float, float, str, str]:
    """Roll and pitch travel of the two joints immediately proximal to a foot link.

    The joint whose child is the foot is the ankle roll, and its own parent joint is the ankle
    pitch, which is the ordering every sole foot in this repository uses.
    """
    by_child = {j.child: j for j in joints.values()}
    roll = by_child.get(foot)
    if roll is None:
        return 0.35, 0.35, "default", "default"
    pitch = by_child.get(roll.parent)
    span = lambda j: max(abs(j.lower or 0.0), abs(j.upper or 0.0)) if j is not None else 0.35
    return span(roll), span(pitch), roll.name, pitch.name if pitch is not None else "default"


# ---------------------------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------------------------


def report_frame_convention(pos: dict[str, np.ndarray], feet: list[str]) -> None:
    """State which root frame axis separates the feet, which must be y for an Isaac Lab robot.

    Isaac Lab evaluates ``base_lin_vel`` and the ``UniformVelocityCommand`` in the root body frame,
    so a robot whose lateral axis is x has its forward velocity command pointing sideways.
    """
    if len(feet) != 2:
        print(f"\nFrame convention: skipped, expected two feet, found {len(feet)}")
        return
    separation = pos[feet[0]] - pos[feet[1]]
    lateral = int(np.argmax(np.abs(separation)))
    print("\nFrame convention check")
    print(f"  foot separation in the root frame : [{', '.join(f'{v:+.4f}' for v in separation)}]")
    print(f"  lateral axis                      : {'xyz'[lateral]}")
    if lateral == 1:
        print("  VERDICT: conventional, forward is the root frame x, as Isaac Lab expects.")
    else:
        print("  VERDICT: NON CONVENTIONAL. The lateral axis is not y, so the root frame is rotated")
        print("           about z relative to the Isaac Lab convention. The velocity command's")
        print("           lin_vel_x and the base_lin_vel observation are evaluated in this frame,")
        print("           so forward commands will drive lateral motion until the root is rotated.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", required=True, help="path to the robot URDF")
    parser.add_argument("--feet", required=True, help="regular expression matching the foot link names")
    parser.add_argument("--points", type=int, default=12, help="points to retain per foot (default 12)")
    parser.add_argument("--name", default=None, help="constant name to emit (default from the URDF stem)")
    parser.add_argument("--visual", action="store_true", help="use visual meshes where collision is absent")
    args = parser.parse_args()

    links, joints, root = parse_urdf(args.urdf)
    rot, pos = forward_kinematics(joints, root)
    pattern = re.compile(args.feet)
    feet = sorted(n for n in links if pattern.fullmatch(n))
    if not feet:
        raise SystemExit(f"no link matched {args.feet!r}; links are {sorted(links)}")

    print(f"URDF   : {args.urdf}")
    print(f"root   : {root}")
    print(f"feet   : {feet}")
    report_frame_convention(pos, feet)

    tables: dict[str, np.ndarray] = {}
    for foot in feet:
        vertices = link_frame_vertices(links[foot], prefer_collision=not args.visual)
        index, sign = down_axis_in_link_frame(rot[foot])
        hull, depth, members = extract_sole(vertices, index, sign)
        table = reduce_hull(hull, index, args.points)

        lateral = [i for i in range(3) if i != index]
        span = [members[:, i].max() - members[:, i].min() for i in lateral]
        print(f"\n{foot}")
        print(f"  vertices                    : {len(vertices)}")
        print(f"  link frame bounding box     : " + "  ".join(
            f"{'xyz'[i]}[{vertices[:, i].min():+.4f}, {vertices[:, i].max():+.4f}]" for i in range(3)))
        print(f"  downward link axis          : {'+-'[sign < 0]}{'xyz'[index]}  (from FK at the zero pose)")
        print(f"  sole plane                  : {'xyz'[index]} = {depth:+.4f}, i.e. {abs(depth):.4f} m below the link origin")
        print(f"  vertices on the sole plane  : {len(members)} within {PLANE_TOLERANCE * 1e3:.1f} mm")
        print(f"  sole extent {'xyz'[lateral[0]]}               : {span[0]:.4f} m")
        print(f"  sole extent {'xyz'[lateral[1]]}               : {span[1]:.4f} m")
        print(f"  convex hull vertices        : {len(hull)}, reduced to {len(table)}")

        roll_limit, pitch_limit, roll_name, pitch_name = ankle_limits(joints, foot)
        error, at = worst_tilt_error(vertices, table, index, sign, roll_limit, pitch_limit)
        print(f"  fidelity sweep              : roll +-{roll_limit:.4f} ({roll_name}), pitch +-{pitch_limit:.4f} ({pitch_name})")
        print(f"  worst height error vs mesh  : {error * 1e3:.3f} mm at roll {at[0]:+.3f} rad, pitch {at[1]:+.3f} rad")
        tables[foot] = table

    stem = args.name or os.path.splitext(os.path.basename(args.urdf))[0].upper()
    reference = tables[feet[0]]
    identical = all(
        tables[f].shape == reference.shape and np.allclose(np.sort(tables[f], axis=0), np.sort(reference, axis=0), atol=1e-6)
        for f in feet
    )
    print(f"\nThe feet {'share one table' if identical else 'DIFFER and need one table each'}.")
    print(f"\n{stem}_SOLE_OFFSETS = [")
    for x, y, z in (reference if identical else np.vstack([tables[f] for f in feet])):
        print(f"    [{x:+.4f}, {y:+.4f}, {z:+.4f}],")
    print("]")


if __name__ == "__main__":
    main()
