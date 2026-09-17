# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Physical parameterisation of a URDF biped, for reward tuning and actuator gain derivation.

This script answers, from the URDF and its meshes alone, the questions that the reward
configuration and the actuator configuration ask of a robot, and which are otherwise answered by
copying a figure from a different robot. It follows the method that ``context/BRS.md`` applies to
the SD_BRS1, generalised to full three dimensions because the KScale link inertial frames are not
in general aligned with the joint axes acting upon them, so the scalar projection that suffices
for an axis aligned robot understates the effective inertia here.

It reports, in order,

* the standing geometry at the nominal pose, being the torso height above the lowest sole point,
  the foot lateral separation and the sole clearance, from which the height and distance reward
  parameters follow,
* the effective rotational inertia at each joint of one leg by the parallel axis theorem over the
  distal subtree, with the natural frequency and damping ratio the configured gains imply,
* a proposed gain set achieving a target damping ratio at a target natural frequency chosen per
  joint role and bounded by the Nyquist frequency of the control loop,
* the symmetry flip set, by composing each joint's origin rotation chain to the root and
  comparing each left joint's mirrored world axis against its right partner's actual axis, which
  settles the set without appeal to any naming convention,
* an axis aligned bounding box audit of non adjacent link pairs, and of every link against the
  ground plane, at both the zero pose and the nominal pose.

Usage:
    python kscale_physical_analysis.py --urdf environments/environments/assets/urdf/solefoot/kscale/kscale.urdf
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kscale_sole_analysis import (  # noqa: E402
    Joint,
    forward_kinematics,
    link_frame_vertices,
    parse_urdf,
    rpy_to_matrix,
)

# The nominal standing pose, mirroring assets/config/kscale_identified_cfg.py. Kept here rather
# than imported because that module pulls in Isaac Lab, which this script deliberately avoids.
#
# Mirror it in FULL. This dictionary formerly carried the knees and the ankles alone while the
# asset configuration had already acquired the hip pitches, so every figure it reported was taken
# at a pose the robot never held, the standing height by 24 mm. Any joint the asset configuration
# sets away from zero belongs here.
#
# The pose closes the sagittal chain, hip_pitch + knee + ankle_pitch summing to zero about the
# world +y on both legs under the anti-parallel hip pitch axes, so the sole lies flat. The check
# is that the reported ankle origin height equals the 0.0430 m sole depth exactly.
# Updated 2026-09-16 to the published K-Scale K-Bot standing posture adopted by
# environments/environments/assets/config/kscale_identified_cfg.py on that date, per section 9.1
# of plans/kscale_actuator_limits_fix.md. It replaces the shallower knee 0.4 and ankle -0.3, and
# it closes the sagittal chain exactly as that pose did, the standing height falling from
# 0.77161 m to 0.73540 m. Mirror any further change here and in the asset configuration together.
NOMINAL_POSE = {
    "right_hip_pitch_04": -0.20369, "left_hip_pitch_04": 0.20369,
    "right_hip_roll_03": 0.042951, "left_hip_roll_03": -0.042951,
    "right_hip_yaw_03": 0.0, "left_hip_yaw_03": 0.0,
    "right_knee_04": 0.51239, "left_knee_04": 0.51239,
    "right_foot_pitch_02": -0.3087, "left_foot_pitch_02": -0.3087,
    "right_foot_roll_02": 0.041254, "left_foot_roll_02": -0.041254,
}

# The configured gains, likewise mirrored from the asset configuration, as stiffness, damping,
# armature and effort limit. The armature belongs here and not as an afterthought. Isaac Lab
# writes it onto the articulation's joint armature, so PhysX adds it to the mass matrix diagonal
# of the degree of freedom, and it is therefore part of the inertia the PD loop actually sees.
# At the distal joints of this robot the armature exceeds the link inertia several times over, so
# omitting it does not merely refine the answer, it changes which regime the joint is in.
CONFIGURED_GAINS = {
    # Refreshed 2026-09-17 from environments/environments/assets/config/kscale_identified_cfg.py.
    # (stiffness, damping, armature, effort_limit). The effort limits are the published K-Scale
    # robstride SOFT limits, the peak torque living in saturation_effort and not appearing here.
    # The figures this replaced, hip pitch 200/50, hip roll 150/45, hip yaw 40/5, knee 200/22,
    # ankle pitch 50/4 and ankle roll 20/4 at 60 and 17 Nm, had been stale since 2026-09-03.
    "hip_pitch": (200.0, 21.5, 0.010, 84.0),
    "hip_roll": (250.0, 28.0, 0.010, 42.0),
    "hip_yaw": (100.0, 3.5, 0.010, 42.0),
    "knee": (300.0, 12.0, 0.015, 84.0),
    "foot_pitch": (120.0, 5.0, 0.005, 11.9),
    "foot_roll": (120.0, 4.0, 0.005, 11.9),
}

GRAVITY = 9.80665

# The steady state tracking error tolerated at the gravitational load, in radians. The stiffness
# floor follows from it as K greater than tau over this, since a PD loop holding a constant load
# settles where the stiffness torque balances it.
POSITION_ERROR_BUDGET = 0.15

# Target natural frequencies by joint role, in rad/s. Proximal joints take a low bandwidth so the
# policy may shape smooth trajectories, distal joints a higher one for terrain adaptation, which
# is the split `context/BRS.md:108` establishes.
# The hip targets are the natural frequencies the configured stiffnesses already achieve, since
# both sit inside the 5 to 15 rad/s band that `context/BRS.md:108` prescribes for a proximal
# joint and neither therefore warrants a change. The remaining four sit above their band and are
# brought to its upper end, the yaw and knee toward 14 to 25 rad/s and the ankles to 30, the
# distal joints taking the higher bandwidth so that they may adapt to terrain within a step.
TARGET_OMEGA = {
    "hip_pitch": 13.0,
    "hip_roll": 12.1,
    "hip_yaw": 25.0,
    "knee": 14.3,
    "foot_pitch": 30.4,
    "foot_roll": 29.5,
}

TARGET_ZETA = 0.70

# The rough terrain task runs decimation 4 at sim.dt 0.005, so 50 Hz, which is the tighter of the
# two bounds the tasks impose and therefore the one to design against.
CONTROL_PERIOD = 0.02


def parse_inertials(path: str) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Mass, centre of mass in the link frame, and inertia tensor about that centre, per link."""
    out: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for elem in ET.parse(path).getroot().iter("link"):
        inertial = elem.find("inertial")
        if inertial is None:
            continue
        mass_elem, inertia_elem, origin = inertial.find("mass"), inertial.find("inertia"), inertial.find("origin")
        if mass_elem is None or inertia_elem is None:
            continue
        mass = float(mass_elem.get("value", "0"))
        xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()])
        g = inertia_elem.get
        tensor = np.array(
            [
                [float(g("ixx", "0")), float(g("ixy", "0")), float(g("ixz", "0"))],
                [float(g("ixy", "0")), float(g("iyy", "0")), float(g("iyz", "0"))],
                [float(g("ixz", "0")), float(g("iyz", "0")), float(g("izz", "0"))],
            ]
        )
        rot = rpy_to_matrix(*rpy)
        out[elem.get("name", "")] = (mass, xyz, rot @ tensor @ rot.T)
    return out


def descendants(joints: dict[str, Joint], link: str) -> list[str]:
    """Every link in the subtree rooted at the given link, the link itself included."""
    by_parent: dict[str, list[str]] = {}
    for j in joints.values():
        by_parent.setdefault(j.parent, []).append(j.child)
    out, stack = [], [link]
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(by_parent.get(node, []))
    return out


def joint_world_axis(joints: dict[str, Joint], rot: dict[str, np.ndarray], name: str) -> np.ndarray:
    """The joint's rotation axis expressed in the root frame, at the pose the rotations describe."""
    j = joints[name]
    axis = j.axis / np.linalg.norm(j.axis)
    return rot[j.parent] @ j.origin_rot @ axis


def effective_inertia(
    joints: dict[str, Joint],
    inertials: dict[str, tuple[float, np.ndarray, np.ndarray]],
    rot: dict[str, np.ndarray],
    pos: dict[str, np.ndarray],
    name: str,
) -> float:
    """Inertia of the distal subtree about the joint axis, by the parallel axis theorem in 3D.

    For each distal link, the inertia about the axis is the sum of the link's own inertia
    projected onto the axis direction and the point mass term m d squared, where d is the
    perpendicular distance from the centre of mass to the axis line. Computing the projection in
    three dimensions rather than reading a diagonal entry matters here because these inertial
    frames are not aligned with the axes acting upon them.
    """
    j = joints[name]
    axis = joint_world_axis(joints, rot, name)
    origin = pos[j.parent] + rot[j.parent] @ j.origin_xyz

    total = 0.0
    for link in descendants(joints, j.child):
        if link not in inertials:
            continue
        mass, com_local, tensor_local = inertials[link]
        tensor_world = rot[link] @ tensor_local @ rot[link].T
        com_world = pos[link] + rot[link] @ com_local
        offset = com_world - origin
        perpendicular = offset - np.dot(offset, axis) * axis
        total += float(axis @ tensor_world @ axis) + mass * float(perpendicular @ perpendicular)
    return total


def gravitational_torque(
    joints: dict[str, Joint],
    inertials: dict[str, tuple[float, np.ndarray, np.ndarray]],
    rot: dict[str, np.ndarray],
    pos: dict[str, np.ndarray],
    name: str,
) -> float:
    """Magnitude of the gravitational torque of the distal subtree about the joint axis.

    This is the load the stiffness term must hold in steady state, so it sets a floor beneath the
    stiffness that the bandwidth argument alone does not see. It is the swing phase load only. In
    stance the ankle carries the whole body weight through the centre of pressure, a far larger
    torque which the effort limit rather than the stiffness governs, and which is why the floor
    below is capped at the effort limit.
    """
    j = joints[name]
    axis = joint_world_axis(joints, rot, name)
    origin = pos[j.parent] + rot[j.parent] @ j.origin_xyz
    weight = np.array([0.0, 0.0, -GRAVITY])

    torque = np.zeros(3)
    for link in descendants(joints, j.child):
        if link not in inertials:
            continue
        mass, com_local, _ = inertials[link]
        com_world = pos[link] + rot[link] @ com_local
        torque += np.cross(com_world - origin, mass * weight)
    return abs(float(np.dot(torque, axis)))


def aabb(links, rot, pos, name) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis aligned bounding box of the link's collision meshes, in the root frame."""
    verts = link_frame_vertices(links[name])
    if verts.size == 0:
        return None
    world = verts @ rot[name].T + pos[name]
    return world.min(axis=0), world.max(axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--feet", default="foot_6061.*")
    parser.add_argument("--side", default="right_", help="joint name prefix of the leg to analyse")
    args = parser.parse_args()

    links, joints, root = parse_urdf(args.urdf)
    inertials = parse_inertials(args.urdf)
    import re as _re

    feet = sorted(n for n in links if _re.fullmatch(args.feet, n))

    print(f"URDF {args.urdf}\nroot {root}\n")

    total_mass = sum(m for m, _, _ in inertials.values())
    print(f"total mass {total_mass:.3f} kg over {len(inertials)} links, torso {inertials[root][0]:.3f} kg\n")

    # -- standing geometry ---------------------------------------------------------------------
    print("=" * 92)
    print("STANDING GEOMETRY")
    print("=" * 92)
    for label, pose in (("zero pose", {}), ("nominal pose", NOMINAL_POSE)):
        rot, pos = forward_kinematics(joints, root, pose)
        lowest = min(
            float((link_frame_vertices(links[n]) @ rot[n].T + pos[n])[:, 2].min())
            for n in links
            if link_frame_vertices(links[n]).size
        )
        sole_z = [float((link_frame_vertices(links[f]) @ rot[f].T + pos[f])[:, 2].min()) for f in feet]
        sep = pos[feet[0]] - pos[feet[1]]
        print(f"\n  {label}")
        print(f"    torso origin above the lowest point : {-lowest:.4f} m")
        print(f"    lowest point of each foot           : {', '.join(f'{v:.4f}' for v in sole_z)}")
        print(f"    foot separation vector              : [{sep[0]:+.4f}, {sep[1]:+.4f}, {sep[2]:+.4f}]")
        print(f"    foot separation, lateral component  : {abs(sep[1]):.4f} m")
        print(f"    ankle origin height                 : {pos[feet[0]][2] - lowest:.4f} m")

    rot_n, pos_n = forward_kinematics(joints, root, NOMINAL_POSE)
    hip = joints[f"{args.side}hip_pitch_04"]
    hip_origin = pos_n[root] + rot_n[root] @ hip.origin_xyz
    print(f"\n    hip pitch origin, root frame        : [{hip_origin[0]:+.4f}, {hip_origin[1]:+.4f}, {hip_origin[2]:+.4f}]")
    print(f"    hip lateral half separation         : {abs(hip_origin[1]):.4f} m")
    print(f"    hip to ankle span at the nominal pose: {abs(hip_origin[2] - pos_n[feet[0]][2]):.4f} m")

    # -- segment lengths -----------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("SEGMENT LENGTHS, from the joint origin offsets of one leg")
    print("=" * 92)
    chain = [n for n in joints if n.startswith(args.side)]
    order = ["hip_pitch_04", "hip_roll_03", "hip_yaw_03", "knee_04", "foot_pitch_02", "foot_roll_02"]
    chain = [args.side + s for s in order if args.side + s in joints]
    for name in chain:
        d = float(np.linalg.norm(joints[name].origin_xyz))
        print(f"  {name:<26} offset from its parent {d:.5f} m")

    # -- effective inertia and gains -----------------------------------------------------------
    print("\n" + "=" * 92)
    print("EFFECTIVE INERTIA AND ACTUATOR GAINS, at the nominal pose")
    print("=" * 92)
    nyquist = math.pi / CONTROL_PERIOD
    print(f"control period {CONTROL_PERIOD} s, Nyquist frequency {nyquist:.2f} rad/s, target zeta {TARGET_ZETA}")
    print(f"inertia includes the armature, which PhysX adds to the mass matrix diagonal\n")
    header = (
        f"  {'joint':<22}{'I_link':>9}{'arm':>7}{'I_eff':>9}{'tau_g':>8}"
        f"{'K':>7}{'D':>7}{'w_n':>8}{'zeta':>7}   {'K_min':>7}{'K*':>8}{'D*':>7}{'w_n*':>7}{'zeta*':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    proposals: dict[str, tuple[float, float]] = {}
    for name in chain:
        role = name[len(args.side) :].rsplit("_", 1)[0]
        stiffness, damping, armature, effort = CONFIGURED_GAINS[role]
        inertia_link = effective_inertia(joints, inertials, rot_n, pos_n, name)
        inertia = inertia_link + armature
        omega = math.sqrt(stiffness / inertia)
        zeta = damping / (2.0 * math.sqrt(stiffness * inertia))

        tau_g = gravitational_torque(joints, inertials, rot_n, pos_n, name)
        # The stiffness must at least hold the gravitational load within the error budget, and
        # must not demand more than the effort limit at that same deflection.
        k_floor = min(tau_g / POSITION_ERROR_BUDGET, effort / POSITION_ERROR_BUDGET)
        k_band = TARGET_OMEGA[role] ** 2 * inertia
        k_new = max(k_floor, k_band)
        omega_new = math.sqrt(k_new / inertia)
        d_new = 2.0 * TARGET_ZETA * math.sqrt(k_new * inertia)
        proposals[role] = (k_new, d_new)
        print(
            f"  {name:<22}{inertia_link:>9.5f}{armature:>7.3f}{inertia:>9.5f}{tau_g:>8.2f}"
            f"{stiffness:>7.1f}{damping:>7.1f}{omega:>8.2f}{zeta:>7.2f}   "
            f"{k_floor:>7.1f}{k_new:>8.1f}{d_new:>7.2f}{omega_new:>7.1f}{TARGET_ZETA:>7.2f}"
        )
    print("\n  I in kg m^2, tau_g in Nm, K in Nm/rad, D in Nms/rad, w_n in rad/s. Starred is proposed.")
    print("  tau_g is the gravitational torque of the distal subtree about the axis at this pose.")
    print("  K_min is the stiffness holding that load within the "
          f"{POSITION_ERROR_BUDGET} rad error budget, capped at the effort limit.")

    print("\n  Proposed configuration block:")
    for role, (k_new, d_new) in proposals.items():
        print(f"    {role:<12} stiffness {k_new:8.1f}   damping {d_new:7.2f}")

    # -- symmetry ------------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("SYMMETRY FLIP SET, reflection in the x and z plane, M = diag(1, -1, 1)")
    print("=" * 92)
    mirror = np.diag([1.0, -1.0, 1.0])
    det = float(np.linalg.det(mirror))
    rot_z, _ = forward_kinematics(joints, root, {})
    print(f"\n  {'joint':<20}{'left world axis':>28}{'mirrored':>28}{'right world axis':>28}  flip")
    for suffix in order:
        left, right = "left_" + suffix, "right_" + suffix
        if left not in joints or right not in joints:
            continue
        a_left = joint_world_axis(joints, rot_z, left)
        a_right = joint_world_axis(joints, rot_z, right)
        mirrored = det * (mirror @ a_left)
        flip = float(np.dot(mirrored, a_right)) < 0.0
        fmt = lambda v: "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"  # noqa: E731
        print(f"  {suffix:<20}{fmt(a_left):>28}{fmt(mirrored):>28}{fmt(a_right):>28}  {'YES' if flip else 'no'}")

    # -- collision audit -----------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("COLLISION AUDIT, axis aligned bounding boxes")
    print("=" * 92)
    adjacency = {frozenset((j.parent, j.child)) for j in joints.values()}
    for label, pose in (("zero pose", {}), ("nominal pose", NOMINAL_POSE)):
        rot, pos = forward_kinematics(joints, root, pose)
        boxes = {n: aabb(links, rot, pos, n) for n in links}
        boxes = {n: b for n, b in boxes.items() if b is not None}
        print(f"\n  {label}")
        ground = min(float(b[0][2]) for b in boxes.values())
        print(f"    lowest point over all links : {ground:+.4f} m in the root frame")
        overlaps = 0
        names = sorted(boxes)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if frozenset((a, b)) in adjacency:
                    continue
                lo = np.maximum(boxes[a][0], boxes[b][0])
                hi = np.minimum(boxes[a][1], boxes[b][1])
                if np.all(hi > lo):
                    overlaps += 1
                    ext = hi - lo
                    print(f"    overlap {a} / {b}, extent {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} m")
        if overlaps == 0:
            print("    no non adjacent bounding box overlap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
