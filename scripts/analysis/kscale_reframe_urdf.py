# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-express a URDF root link frame by a fixed rotation, in place.

Some CAD exports place the robot's root frame at an arbitrary yaw relative to the convention
Isaac Lab assumes, which is x forward, y to the left and z up. Isaac Lab evaluates the velocity
command and the base linear velocity observation in the root body frame, so a robot whose lateral
axis is not y has its forward command pointing sideways, and the defect is invisible in the
environment configuration because the configuration is correct and it is the asset that differs.

The KScale biped exported from Onshape carries exactly this defect, its feet separating along the
root frame x and its forward direction lying along minus y, so its frame is a rotation of ninety
degrees about z away from the convention.

This script repairs it by re-expressing the root frame in place rather than by inserting a fixed
joint above the root. In place re-expression preserves the body count exactly, which matters
because several critic observations iterate over bodies and an extra link would silently widen
them. Every quantity expressed in the root frame is premultiplied by the rotation, being the root
link's inertial, visual and collision origins, its inertia tensor, which transforms as R I R^T,
and the origins of the joints whose parent is the root. Nothing below the root changes, each
subtree being rigidly attached to its parent joint and moving with it.

The edit is textual and line oriented rather than a parse and re-serialise, so that the diff
remains reviewable and every untouched line survives byte for byte.

The URDF is a generated artefact, so a re-export from CAD reintroduces the original frame. This
script is therefore kept rather than discarded, so that the repair can be reapplied.

Verify the result with ``kscale_sole_analysis.py``, whose frame convention check must report the
lateral axis as y after the edit where it reports x before it.

Usage:
    python kscale_reframe_urdf.py --urdf path/to/robot.urdf --yaw 90
    python kscale_reframe_urdf.py --urdf path/to/robot.urdf --yaw 90 --dry-run
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys

import numpy as np

# Origin and inertia attribute patterns. The URDF writes floats in a mixture of plain and
# exponential notation, so the value pattern has to admit both.
_ORIGIN_RE = re.compile(r'(<origin\s+)xyz="([^"]*)"(\s+)rpy="([^"]*)"(\s*/?>)')
_INERTIA_RE = re.compile(
    r'(<inertia\s+)ixx="([^"]*)"(\s+)ixy="([^"]*)"(\s+)ixz="([^"]*)"'
    r'(\s+)iyy="([^"]*)"(\s+)iyz="([^"]*)"(\s+)izz="([^"]*)"(\s*/?>)'
)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Fixed axis extrinsic X then Y then Z, which is the URDF convention, so R = Rz Ry Rx."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def matrix_to_rpy(mat: np.ndarray) -> np.ndarray:
    """Inverse of ``rpy_to_matrix``, taking the branch with pitch in the closed interval.

    The gimbal lock branch is handled explicitly rather than left to produce a silently wrong
    triple, since a root frame rotation of exactly ninety degrees is precisely the case that
    lands on it for some of the origins involved.
    """
    sp = -float(mat[2, 0])
    sp = max(-1.0, min(1.0, sp))
    pitch = math.asin(sp)
    if abs(abs(sp) - 1.0) < 1.0e-9:
        # Gimbal lock, roll and yaw are no longer separable. Fold the whole rotation onto roll.
        roll = math.atan2(-float(mat[1, 2]), float(mat[1, 1]))
        yaw = 0.0
    else:
        roll = math.atan2(float(mat[2, 1]), float(mat[2, 2]))
        yaw = math.atan2(float(mat[1, 0]), float(mat[0, 0]))
    return np.array([roll, pitch, yaw])


def _fmt(value: float) -> str:
    """Render a float in the style the exporter uses, six significant figures, minus zero cleaned."""
    if abs(value) < 1.0e-12:
        return "0"
    text = f"{value:.6g}"
    return "0" if text in ("-0", "-0.0") else text


def _parse_triple(text: str) -> np.ndarray:
    parts = text.split()
    if len(parts) != 3:
        raise ValueError(f"expected three components, got {text!r}")
    return np.array([float(p) for p in parts])


def transform_origin(line: str, rot: np.ndarray) -> str:
    """Premultiply the xyz and the rpy of every origin on the line by ``rot``."""

    def repl(match: re.Match) -> str:
        xyz = rot @ _parse_triple(match.group(2))
        rpy = matrix_to_rpy(rot @ rpy_to_matrix(_parse_triple(match.group(4))))
        return (
            f'{match.group(1)}xyz="{" ".join(_fmt(v) for v in xyz)}"'
            f'{match.group(3)}rpy="{" ".join(_fmt(v) for v in rpy)}"{match.group(5)}'
        )

    return _ORIGIN_RE.sub(repl, line)


def transform_inertia(line: str, rot: np.ndarray) -> str:
    """Rotate the inertia tensor on the line, which transforms as R I R^T."""

    def repl(match: re.Match) -> str:
        ixx, ixy, ixz = float(match.group(2)), float(match.group(4)), float(match.group(6))
        iyy, iyz, izz = float(match.group(8)), float(match.group(10)), float(match.group(12))
        inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        out = rot @ inertia @ rot.T
        return (
            f'{match.group(1)}ixx="{_fmt(out[0, 0])}"{match.group(3)}ixy="{_fmt(out[0, 1])}"'
            f'{match.group(5)}ixz="{_fmt(out[0, 2])}"{match.group(7)}iyy="{_fmt(out[1, 1])}"'
            f'{match.group(9)}iyz="{_fmt(out[1, 2])}"{match.group(11)}izz="{_fmt(out[2, 2])}"'
            f'{match.group(12 + 1)}'
        )

    return _INERTIA_RE.sub(repl, line)


def find_root_link(lines: list[str]) -> str:
    """The root is the unique link that is never named as a joint child."""
    links = re.findall(r'<link\s+name="([^"]+)"', "".join(lines))
    children = set(re.findall(r'<child\s+link="([^"]+)"', "".join(lines)))
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise SystemExit(f"expected exactly one root link, found {roots}")
    return roots[0]


def link_block(lines: list[str], name: str) -> tuple[int, int]:
    """Half open line index range of the named link element."""
    start = None
    for index, line in enumerate(lines):
        if re.search(rf'<link\s+name="{re.escape(name)}"', line):
            start = index
            break
    if start is None:
        raise SystemExit(f"link {name!r} not found")
    for index in range(start, len(lines)):
        if "</link>" in lines[index]:
            return start, index + 1
    raise SystemExit(f"link {name!r} is not closed")


def joint_blocks_with_parent(lines: list[str], name: str) -> list[tuple[int, int]]:
    """Half open line index ranges of every joint element whose parent is the named link."""
    blocks: list[tuple[int, int]] = []
    start = None
    for index, line in enumerate(lines):
        if "<joint " in line:
            start = index
        elif "</joint>" in line and start is not None:
            body = "".join(lines[start : index + 1])
            if re.search(rf'<parent\s+link="{re.escape(name)}"', body):
                blocks.append((start, index + 1))
            start = None
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", required=True, help="path to the URDF to rewrite in place")
    parser.add_argument("--yaw", type=float, default=90.0, help="rotation about z in degrees, default 90")
    parser.add_argument("--dry-run", action="store_true", help="report the edit without writing")
    parser.add_argument("--no-backup", action="store_true", help="do not write a .bak beside the URDF")
    args = parser.parse_args()

    if not os.path.isfile(args.urdf):
        raise SystemExit(f"no such file, {args.urdf}")

    with open(args.urdf, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    angle = math.radians(args.yaw)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])

    root = find_root_link(lines)
    lo, hi = link_block(lines, root)
    joints = joint_blocks_with_parent(lines, root)

    print(f"root link          {root}")
    print(f"root link lines    {lo + 1} to {hi}")
    print(f"root child joints  {len(joints)} at lines {[b[0] + 1 for b in joints]}")
    print(f"rotation           {args.yaw} degrees about z")

    touched = 0
    spans = [(lo, hi)] + joints
    for start, end in spans:
        for index in range(start, end):
            before = lines[index]
            after = transform_inertia(transform_origin(before, rot), rot)
            if after != before:
                lines[index] = after
                touched += 1
                if touched <= 6:
                    print(f"  line {index + 1}\n    - {before.strip()}\n    + {after.strip()}")

    print(f"lines rewritten    {touched}")

    if args.dry_run:
        print("dry run, nothing written")
        return 0

    if not args.no_backup:
        shutil.copy2(args.urdf, args.urdf + ".bak")
        print(f"backup written     {args.urdf}.bak")

    with open(args.urdf, "w", encoding="utf-8") as handle:
        handle.writelines(lines)
    print(f"rewritten          {args.urdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
