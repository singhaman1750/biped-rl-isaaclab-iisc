#!/usr/bin/env python3
"""Self test for the KScale feet heading reward and the frame conventions it rests upon.

Step 2 of section 5.1.2 of plans/kscale_integration.md requires that the assumptions behind
`pen_feet_heading` be asserted numerically before any training run, since every one of them is
a frame convention that a regression in the asset would otherwise carry into training in
silence. Two families of check are performed.

The KINEMATIC checks read the URDF directly and need nothing but numpy. They establish that
the toe direction is the foot link's negative z, that both hip yaw axes are co directed so
that a positive command turns either foot the same way, and that the heading gain is the
0.995 which the 0.0998 rad axis cant implies.

The REWARD checks execute `feet_yaw_alignment` against stub objects supplying the two
quaternion fields it reads, and need torch but not Isaac Lab. They establish the orthogonality
of the mean form, the non orthogonality of the summed form, the dead band, the backwards
compatible default path, and that the forward_axis path recovers a heading the Euler path
misses by ninety degrees on this robot.

Run with no arguments. Exits non zero on the first failure.
"""

from __future__ import annotations

import math
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kscale_sole_analysis import forward_kinematics, parse_urdf  # noqa: E402
from kscale_stance_analysis import DEFAULT_URDF, NOMINAL_POSE  # noqa: E402

FOOT = {"right": "foot_6061", "left": "foot_6061_2"}
HIP_YAW = {"right": "right_hip_yaw_03", "left": "left_hip_yaw_03"}

# The heading gain AT THE NOMINAL POSE. It is not a constant of the robot. Section 2.3 of
# context/KScale.md records, in its correction of 2026-08-26, that the 0.0998 tilt of the hip
# yaw axis out of vertical is an artefact of the nominal hip pitch flexion of 0.1 rad
# propagating down the chain rather than a hardware cant, and the gain follows the same pose.
# It is 1.000 at the URDF zero pose, 0.995 here, and rises to 1.524 at a hip pitch of minus
# 1.0 rad. See test_gain_is_pose_dependent below, and section 5.1.2 of the plan for why this
# is the argument for regulating the heading in task space rather than at the joint.
EXPECTED_GAIN = 0.995
GAIN_TOL = 5.0e-3
ANGLE_TOL = 1.0e-3

_failures: list[str] = []


def check(name: str, ok: bool, detail: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")
    if not ok:
        _failures.append(name)


def toe_heading(rot: dict[str, np.ndarray], foot: str) -> float:
    """Heading of the toe in the root frame, the toe being the foot link's negative z."""
    toe = -rot[foot][:, 2]
    return math.atan2(toe[1], toe[0])


def kinematic_checks() -> None:
    print("Kinematic checks, read from the URDF")
    joints, root = parse_urdf(os.path.normpath(DEFAULT_URDF))[1:]
    rot, _ = forward_kinematics(joints, root, NOMINAL_POSE)

    for side, foot in FOOT.items():
        mapped = -rot[foot][:, 2]
        err = float(np.linalg.norm(mapped - np.array([1.0, 0.0, 0.0])))
        check(
            f"{side} toe (link -z) maps to root +x at nominal",
            err < ANGLE_TOL,
            f"maps to {np.round(mapped, 5)}, error {err:.2e}",
        )

    axes = {}
    for side, name in HIP_YAW.items():
        joint = joints[name]
        axes[side] = rot[joint.parent] @ joint.origin_rot @ np.asarray(joint.axis, dtype=float)
        check(
            f"{side} hip yaw axis points up in the root frame",
            axes[side][2] > 0.99,
            f"{np.round(axes[side], 4)}",
        )
    codirected = float(np.dot(axes["right"], axes["left"]))
    check(
        "hip yaw axes are CO DIRECTED",
        codirected > 0.99,
        f"dot product {codirected:.6f}, a negative value would invert the differential reading",
    )

    for side, foot in FOOT.items():
        for q in (0.1, 0.3, 0.8):
            pose = dict(NOMINAL_POSE)
            pose[HIP_YAW[side]] = q
            rot_q, _ = forward_kinematics(joints, root, pose)
            gain = toe_heading(rot_q, foot) / q
            check(
                f"{side} hip yaw {q:+.1f} rad gives nominal pose gain {EXPECTED_GAIN}",
                abs(gain - EXPECTED_GAIN) < GAIN_TOL,
                f"gain {gain:.5f}",
            )

    # The gain is a property of the POSE, not of the robot, and the reward must not be built
    # on an assumption that it is fixed. Unity at the URDF zero pose is the check that the
    # 0.0998 axis tilt is the hip pitch artefact section 2.3 of context/KScale.md records,
    # and the spread across the hip pitch travel is the check that quoting the gain without
    # its pose would be meaningless.
    zero_pose = {name: 0.0 for name in NOMINAL_POSE}
    zero_pose[HIP_YAW["right"]] = 0.3
    rot_z, _ = forward_kinematics(joints, root, zero_pose)
    check(
        "heading gain is unity at the URDF zero pose",
        abs(toe_heading(rot_z, FOOT["right"]) / 0.3 - 1.0) < 1e-4,
        f"gain {toe_heading(rot_z, FOOT['right']) / 0.3:.6f}",
    )

    gains = {}
    for hip_pitch in (0.0, -0.1, -0.5, -1.0):
        pose = dict(NOMINAL_POSE)
        pose["right_hip_pitch_04"] = hip_pitch
        pose[HIP_YAW["right"]] = 0.3
        rot_h, _ = forward_kinematics(joints, root, pose)
        gains[hip_pitch] = toe_heading(rot_h, FOOT["right"]) / 0.3
    spread = max(gains.values()) - min(gains.values())
    check(
        "heading gain is POSE DEPENDENT across the hip pitch travel",
        spread > 0.4,
        "gains " + ", ".join(f"{k:+.1f}->{v:.3f}" for k, v in gains.items()),
    )

    # The ankle roll axis lies along the toe direction, so it cannot yaw the foot at all.
    pose = dict(NOMINAL_POSE)
    pose["right_foot_roll_02"] = 0.2618
    rot_r, _ = forward_kinematics(joints, root, pose)
    check(
        "ankle roll at its limit contributes no toe heading",
        abs(toe_heading(rot_r, FOOT["right"])) < ANGLE_TOL,
        f"heading {math.degrees(toe_heading(rot_r, FOOT['right'])):+.4f} deg",
    )


def reward_checks() -> None:
    print("Reward checks, executing feet_yaw_alignment against stubs")
    try:
        import torch
    except ImportError:
        print("  [SKIP] torch is unavailable, the reward checks are not run")
        return

    math_utils, feet_yaw_alignment = _load_reward(torch)

    def quat_yaw(psi: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(psi)
        return torch.stack([torch.cos(psi / 2), zero, zero, torch.sin(psi / 2)], dim=-1)

    def env_for(quats: torch.Tensor, base: float = 0.0):
        num = quats.shape[0]
        data = types.SimpleNamespace(
            body_quat_w=quats,
            root_link_quat_w=quat_yaw(torch.full((num,), base)),
        )
        return types.SimpleNamespace(scene={"robot": types.SimpleNamespace(data=data)})

    cfg = types.SimpleNamespace(name="robot", body_ids=[0, 1])

    def value(e_right: float, e_left: float, **kwargs) -> float:
        env = env_for(quat_yaw(torch.tensor([[e_right, e_left]])))
        return float(feet_yaw_alignment(env, cfg, **kwargs)[0])

    splay_mean = value(0.3, -0.3, common_mode="mean")
    check("mean form: a pure splay gives a common mode of zero", abs(splay_mean) < 1e-6,
          f"{splay_mean:.8f}")
    splay_sum = value(0.3, -0.3, common_mode="sum")
    check("sum form: the same splay leaks onto the common mode", abs(splay_sum - 0.18) < 1e-5,
          f"{splay_sum:.8f}, expected 0.18")
    common_only = value(0.3, 0.3, common_mode="mean", differential_scale=1.0)
    check("mean form: a pure common rotation leaves the differential at zero",
          abs(common_only - 0.09) < 1e-5, f"{common_only:.8f}, expected 0.09")

    ratio_mean = value(0.3, -0.3, common_mode="mean", differential_scale=1.0) / common_only
    check("mean form splay to common ratio is 4", abs(ratio_mean - 4.0) < 1e-3,
          f"{ratio_mean:.4f}")
    ratio_sum = (value(0.3, -0.3, common_mode="sum", differential_scale=1.0)
                 / value(0.3, 0.3, common_mode="sum", differential_scale=1.0))
    check("sum form splay to common ratio is 3", abs(ratio_sum - 3.0) < 1e-3, f"{ratio_sum:.4f}")

    banded = value(0.3, -0.3, common_mode="mean", differential_scale=1.0, tolerance=0.10)
    check("dead band of 0.10 on a splay of +/-0.3 gives 0.25", abs(banded - 0.25) < 1e-5,
          f"{banded:.8f}")

    # Backwards compatibility: every new argument at its default must reproduce the summed
    # Euler form the function carried before section 5.1.2 extended it.
    env = env_for(quat_yaw(torch.tensor([[0.3, -0.2]])))
    original = float(
        torch.sum(torch.square(math_utils.wrap_to_pi(torch.tensor([[0.3, -0.2]]) - 0.0)), dim=1)[0]
    )
    default = float(feet_yaw_alignment(env, cfg)[0])
    check("defaults reproduce the original summed Euler form", abs(default - original) < 1e-9,
          f"{default:.10f} against {original:.10f}")

    # The permuted foot frame: forward_axis must recover the applied yaw where Euler cannot.
    joints, root = parse_urdf(os.path.normpath(DEFAULT_URDF))[1:]
    rot, _ = forward_kinematics(joints, root, NOMINAL_POSE)
    r_nom = rot[FOOT["right"]]

    def mat_to_quat(mat: np.ndarray) -> torch.Tensor:
        trace = mat.trace()
        w = math.sqrt(max(0.0, 1.0 + trace)) / 2
        x = math.copysign(math.sqrt(max(0.0, 1 + mat[0, 0] - mat[1, 1] - mat[2, 2])) / 2,
                          mat[2, 1] - mat[1, 2])
        y = math.copysign(math.sqrt(max(0.0, 1 - mat[0, 0] + mat[1, 1] - mat[2, 2])) / 2,
                          mat[0, 2] - mat[2, 0])
        z = math.copysign(math.sqrt(max(0.0, 1 - mat[0, 0] - mat[1, 1] + mat[2, 2])) / 2,
                          mat[1, 0] - mat[0, 1])
        return torch.tensor([w, x, y, z], dtype=torch.float32)

    for psi in (0.0, 0.2, 0.4):
        rz = np.array([[math.cos(psi), -math.sin(psi), 0.0],
                       [math.sin(psi), math.cos(psi), 0.0],
                       [0.0, 0.0, 1.0]])
        quat = mat_to_quat(rz @ r_nom)
        env = env_for(torch.stack([torch.stack([quat, quat])]))
        recovered = math.sqrt(
            float(feet_yaw_alignment(env, cfg, forward_axis=(0.0, 0.0, -1.0),
                                     common_mode="mean")[0])
        )
        euler = math.sqrt(float(feet_yaw_alignment(env, cfg, common_mode="mean")[0]))
        check(f"forward_axis recovers a toe yaw of {psi:.1f} rad", abs(recovered - psi) < 1e-3,
              f"recovered {recovered:.4f}")
        check(f"Euler path misses it by pi/2 at {psi:.1f} rad",
              abs(euler - (psi + math.pi / 2)) < 1e-3,
              f"reports {euler:.4f}, expected {psi + math.pi / 2:.4f}")


def _load_reward(torch):
    """Return (math_utils, feet_yaw_alignment), importing Isaac Lab where it is available.

    Inside the Isaac container the real modules are imported, so the checks exercise exactly
    what training will run. Outside it neither module imports, and rather than skip the checks
    the function body is extracted from its source file and executed against a small shim
    supplying the three math helpers it uses. Both paths test the same source TEXT, so a
    regression in the reward is caught on either, and the shim is exercised against the real
    implementation whenever the container is available.
    """
    try:
        import isaaclab.utils.math as math_utils

        from environments.tasks.locomotion.mdp.rewards import feet_yaw_alignment

        return math_utils, feet_yaw_alignment
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
    match = re.search(
        r"^def feet_yaw_alignment\(.*?(?=^def |\Z)", source, re.S | re.M
    )
    if match is None:
        raise RuntimeError("feet_yaw_alignment not found in rewards.py")

    class _Shim:
        @staticmethod
        def wrap_to_pi(angles):
            return (angles + math.pi) % (2 * math.pi) - math.pi

        @staticmethod
        def euler_xyz_from_quat(quat):
            w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
            roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            return roll, pitch, yaw

        @staticmethod
        def quat_apply(quat, vec):
            shape = vec.shape
            quat = quat.reshape(-1, 4)
            vec = vec.reshape(-1, 3)
            xyz = quat[:, 1:]
            t = torch.cross(xyz, vec, dim=-1) * 2
            return (vec + quat[:, 0:1] * t + torch.cross(xyz, t, dim=-1)).view(shape)

    namespace = {
        "torch": torch,
        "math_utils": _Shim,
        "ManagerBasedRLEnv": object,
        "SceneEntityCfg": object,
        "Articulation": object,
    }
    exec(textwrap.dedent(match.group(0)), namespace)
    print("  (Isaac Lab unavailable, the reward source was loaded directly against a shim)")
    return _Shim, namespace["feet_yaw_alignment"]


def main() -> int:
    print(f"URDF {os.path.normpath(DEFAULT_URDF)}\n")
    kinematic_checks()
    print()
    reward_checks()
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
