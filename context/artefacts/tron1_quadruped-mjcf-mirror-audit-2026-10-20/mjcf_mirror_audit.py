"""Derive the sagittal mirror of an MJCF legged robot from its kinematic tree.

For every joint, the world axis at the zero configuration is composed from the body quaternion
chain. Under the reflection M = diag(1, -1, 1) a rotation axis maps as a pseudovector,
a -> det(M) M a = (-ax, ay, -az). A joint keeps its sign when its partner's world axis equals the
mirrored axis and flips when it equals the negation. Body positions and joint limits are checked
for consistency with the derived sign.
"""
import re
import xml.etree.ElementTree as ET

import numpy as np


def quat_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def walk(body, R, p, out_b, out_j):
    pos = np.array([float(v) for v in body.get("pos", "0 0 0").split()])
    q = np.array([float(v) for v in body.get("quat", "1 0 0 0").split()])
    Rw = R @ quat_to_mat(q)
    pw = p + R @ pos
    out_b[body.get("name")] = (pw, Rw)
    for j in body.findall("joint"):
        axis = np.array([float(v) for v in j.get("axis", "0 0 1").split()])
        rng = [float(v) for v in j.get("range").split()]
        out_j[j.get("name")] = (Rw @ axis, rng, body.get("name"))
    for child in body.findall("body"):
        walk(child, Rw, pw, out_b, out_j)


def main(path, partner_re, repl):
    root = ET.parse(path).getroot()
    bodies, joints = {}, {}
    for top in root.find("worldbody").findall("body"):
        walk(top, np.eye(3), np.zeros(3), bodies, joints)
    M = np.diag([1.0, -1.0, 1.0])

    def partner(n):
        return re.sub(partner_re, repl, n)

    print(f"== {path}")
    print("bodies:", len(bodies))
    worst = 0.0
    for b, (pw, _) in bodies.items():
        pb = partner(b)
        assert pb in bodies, (b, pb)
        worst = max(worst, np.abs(M @ pw - bodies[pb][0]).max())
    print(f"max body position mirror error: {worst:.2e} m")
    print(f"{'joint':<18}{'partner':<18}{'world axis':<24}{'sign':<6}limits consistent")
    for j, (a, rng, _) in joints.items():
        pj = partner(j)
        ap, rngp, _ = joints[pj]
        am = -(M @ a)
        if np.allclose(am, ap, atol=1e-6):
            s = +1
            lim_ok = np.allclose(rng, rngp)
        elif np.allclose(am, -ap, atol=1e-6):
            s = -1
            lim_ok = np.allclose(rng, [-rngp[1], -rngp[0]])
        else:
            s, lim_ok = 0, False
        sym = np.isclose(rng[0], -rng[1])
        print(f"{j:<18}{pj:<18}{np.array2string(np.round(a, 6)):<24}{s:+d}    {lim_ok}{' (limits symmetric, ambiguous)' if sym else ''}")


if __name__ == "__main__":
    base = "/ws/tron1-rl-isaaclab-cozum/environments/environments/assets/mjcf"
    main(f"{base}/solefoot/tron1/base_robot.xml", r"_([RL])_", lambda m: "_" + ("L" if m.group(1) == "R" else "R") + "_")
    main(f"{base}/quadruped/quadruped.xml", r"(?<=[FR])([RL])(?=_)", lambda m: "L" if m.group(1) == "R" else "R")
