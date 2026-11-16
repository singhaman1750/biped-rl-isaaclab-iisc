"""The parameters of arm 7 of plans/kscale_gait_shaping.md, derived by forward kinematics.

Section 15.6 moves the nominal pose to the published K-Scale standing posture and must move the
two base height targets with it, the three values being one value rather than three.  This program
derives that one value from the URDF rather than quoting it, because the figure of 0.73540 m that
the plan and the addition of 2026-09-16 to context/KScale.md both carry was computed at a pose
whose hip roll and ankle roll nominals were zero, and the pose supplied on 2026-09-17 gave those
two joints the mirrored values the arm retains.  The roll costs half a millimetre of height and the
arm is asked to target the height its own posture produces, so the figure must be recomputed.

It also prices the settling buffer.  The workspace convention spawns the robot at its standing
height plus 0.02 m, and the buffer is here reduced to zero at the user's direction so that the
spawn height and the target height are the same number, which requires knowing what the buffer was
buying.  The answer is that it was buying nothing that the reset distribution does not overwhelm,
and sections C and D below are the measurement that establishes it.

Run it from this directory with no arguments.  It reads the URDF and no dump, so it survives the
deletion of the log tree, and it borrows the forward kinematics of scripts/analysis, which is the
same code every pose figure in context/KScale.md was computed with.
"""
import os
import sys

import numpy as np

ANALYSIS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", "..", "..", "scripts", "analysis"))
sys.path.insert(0, ANALYSIS)
from kscale_stance_analysis import Model, SOLE_NORMAL, stance_torques  # noqa: E402

URDF = os.path.normpath(os.path.join(ANALYSIS, "..", "..", "environments", "environments",
                                     "assets", "urdf", "solefoot", "kscale", "kscale.urdf"))

SOFT_FACTOR = 0.9        # kscale_identified_cfg.py:241
HIP_ROLL = 0.042951      # kscale_identified_cfg.py:203, retained by the arm
ANKLE_ROLL = 0.041254    # kscale_identified_cfg.py:211, retained by the arm
GRAVITY = 9.81

# The two poses, as the tree holds the first and as section 21.7 proposes the second.
PRESENT = (0.20369, 0.51239, -0.30870)
ARM5 = (0.34907, 0.87266, -0.52360)

# The startup perturbation of kscale_base_env_cfg.py:592 and the per joint reset perturbations of
# kscale_base_env_cfg.py:638 onward, both as half widths of a uniform draw.
STARTUP_OFFSET = 0.05
RESET_RANGE = {"hip_pitch": 0.24, "knee": 0.24, "foot_pitch": 0.15,
               "hip_roll": 0.1295, "hip_yaw": 0.1295, "foot_roll": 0.10}

# The six groups of the per joint action scale at kscale_base_env_cfg.py:193, with the stiffness
# and effort ceiling of kscale_identified_cfg.py, as the tree of 2026-09-23 holds them.  The two
# ankle entries stand at 0.08, the user having halved them from the 0.1619 that arm 4 derived.
TREE_SCALES = [
    ("hip_pitch", "right_hip_pitch_04", 200.0, 84.0, 0.4000),
    ("hip_yaw", "right_hip_yaw_03", 100.0, 42.0, 0.4000),
    ("knee", "right_knee_04", 300.0, 84.0, 0.2667),
    ("hip_roll", "right_hip_roll_03", 250.0, 42.0, 0.1600),
    ("foot_pitch", "right_foot_pitch_02", 100.0, 17.0, 0.0800),
    ("foot_roll", "right_foot_roll_02", 100.0, 17.0, 0.0800),
]
ANKLE_SCALE = 0.08                    # kscale_base_env_cfg.py:198 and :199 as of 2026-09-23


def pose(hip_pitch, knee, ankle_pitch, hip_roll=HIP_ROLL, ankle_roll=ANKLE_ROLL):
    """The twelve joint dictionary, the left leg mirroring the right at the two roll joints."""
    return {
        "right_hip_pitch_04": -hip_pitch, "left_hip_pitch_04": hip_pitch,
        "right_hip_roll_03": hip_roll, "left_hip_roll_03": -hip_roll,
        "right_hip_yaw_03": 0.0, "left_hip_yaw_03": 0.0,
        "right_knee_04": knee, "left_knee_04": knee,
        "right_foot_pitch_02": ankle_pitch, "left_foot_pitch_02": ankle_pitch,
        "right_foot_roll_02": ankle_roll, "left_foot_roll_02": -ankle_roll,
    }


def geometry(model, p):
    """Standing height, sole tilt, ankle roll origin height and ankle frame separation."""
    rot, pos, shift = model.kinematics(p)
    height = pos[model.root][2] + shift
    origin = pos[model.joints["right_foot_roll_02"].child][2] + shift
    down = rot[model.foot] @ np.array(SOLE_NORMAL)
    tilt = np.degrees(np.arccos(np.clip(-down[2], -1.0, 1.0)))
    right = pos[model.joints["right_foot_roll_02"].child]
    left = pos[model.joints["left_foot_roll_02"].child]
    return height, tilt, origin, abs(right[1] - left[1])


def soft_limits(model):
    """The soft travel of every leg joint at the factor of kscale_identified_cfg.py:241."""
    out = {}
    for name, joint in model.joints.items():
        if joint.lower is None or joint.upper is None:
            continue
        mid, half = 0.5 * (joint.lower + joint.upper), 0.5 * (joint.upper - joint.lower)
        out[name] = (joint.lower, joint.upper,
                     mid - SOFT_FACTOR * half, mid + SOFT_FACTOR * half)
    return out


def family(name):
    for key in RESET_RANGE:
        if key in name:
            return key
    return None


def main():
    model = Model(URDF)
    lims = soft_limits(model)

    print("=" * 98)
    print("A. The pose, its closure and the standing height it produces")
    print("=" * 98)
    print("  the closure identity is s*hip_pitch + knee + foot_pitch = 0, s being -1 left and +1 right,")
    print("  and a closed pose sets the ankle roll origin at exactly the 0.04300 m sole depth")
    print()
    print(f"  {'pose':38} {'closure':>9} {'height':>9} {'tilt':>8} {'origin':>9} {'ankle sep':>10}")
    cases = [
        ("present tree, rolls as configured", pose(*PRESENT)),
        ("present tree, rolls zeroed", pose(*PRESENT, 0.0, 0.0)),
        ("arm 7 posture, rolls as configured", pose(*ARM5)),
        ("arm 7 posture, rolls zeroed", pose(*ARM5, 0.0, 0.0)),
    ]
    results = {}
    for label, p in cases:
        closure = -p["left_hip_pitch_04"] + p["left_knee_04"] + p["left_foot_pitch_02"]
        height, tilt, origin, sep = geometry(model, p)
        results[label] = height
        print(f"  {label:38} {closure:+9.6f} {height:9.5f} {tilt:8.3f} {origin:9.5f} {sep:10.4f}")
    print()
    present = results["present tree, rolls as configured"]
    arm5 = results["arm 7 posture, rolls as configured"]
    zeroed = results["arm 7 posture, rolls zeroed"]
    print(f"  the arm lowers the machine by {1000 * (present - arm5):.1f} mm, from {present:.5f} m to {arm5:.5f} m")
    print(f"  the retained roll nominals cost {1000 * (zeroed - arm5):.2f} mm against the {zeroed:.5f} m of the zero roll")
    print(f"  pose, which is the figure section 21.7 and the addition of 2026-09-16 to context/KScale.md")
    print(f"  both carry, so the three values of the arm are {arm5:.5f} m and not {zeroed:.5f}")
    print()

    print("=" * 98)
    print("B. The settling buffer, and what zeroing it costs")
    print("=" * 98)
    print("  the workspace convention spawns at the standing height plus a margin, the arrival speed")
    print("  being that of a free fall through the clearance beneath the sole")
    print()
    print(f"  {'buffer':>8} {'spawn':>9} {'clearance':>11} {'arrival':>10} {'fall time':>11}")
    for buffer in (0.02, 0.01, 0.0):
        spawn = arm5 + buffer
        speed = float(np.sqrt(2.0 * GRAVITY * buffer))
        fall = float(np.sqrt(2.0 * buffer / GRAVITY)) if buffer > 0 else 0.0
        print(f"  {buffer:8.3f} {spawn:9.5f} {1000 * buffer:9.1f} mm {speed:8.3f} m/s {1000 * fall:9.1f} ms")
    print()

    print("=" * 98)
    print("C. The standing height the reset distribution actually produces")
    print("=" * 98)
    print("  the startup event of kscale_base_env_cfg.py:592 shifts every joint default by up to")
    print("  0.05 rad and the six reset events shift the drawn position again, each clamped to the")
    print("  soft limits by events.py:350, so the pose at the first step of an episode is not the")
    print("  nominal pose and the height it stands at is not the standing height")
    print()
    rng = np.random.default_rng(42)
    samples = 4000
    for label, angles in (("present tree", PRESENT), ("arm 7 posture", ARM5)):
        base = pose(*angles)
        heights = np.empty(samples)
        clamped = {k: 0 for k in RESET_RANGE}
        for i in range(samples):
            drawn = {}
            for name, value in base.items():
                fam = family(name)
                default = value + rng.uniform(-STARTUP_OFFSET, STARTUP_OFFSET)
                half = RESET_RANGE[fam]
                raw = default + rng.uniform(-half, half)
                lo, hi, slo, shi = lims[name]
                drawn[name] = float(np.clip(raw, slo, shi))
                if drawn[name] != raw:
                    clamped[fam] += 1
            heights[i] = geometry(model, drawn)[0]
        nominal = geometry(model, base)[0]
        print(f"  {label}, nominal {nominal:.5f} m")
        print(f"    height at reset   p01 {np.percentile(heights, 1):.5f}  p50 {np.median(heights):.5f}  "
              f"p99 {np.percentile(heights, 99):.5f}  max {heights.max():.5f}")
        for buffer in (0.02, 0.01, 0.0):
            spawn = nominal + buffer
            over = heights - spawn
            deep = over[over > 0.0]
            worst = 1000 * deep.max() if deep.size else 0.0
            median = 1000 * np.median(deep) if deep.size else 0.0
            print(f"    spawn at {spawn:.5f}, being the nominal plus {1000 * buffer:4.1f} mm, is below the drawn pose on "
                  f"{100 * (over > 0).mean():5.1f} pct of resets, those penetrating by a median "
                  f"{median:4.1f} mm and at worst {worst:4.1f} mm")
        print(f"    resets clamped to a soft limit, by family, per leg pair, out of {2 * samples}")
        print("      " + "  ".join(f"{k} {100 * clamped[k] / (2 * samples):5.2f} pct" for k in RESET_RANGE))
        print()

    print("=" * 98)
    print("D. The travel each nominal leaves, present against arm 7")
    print("=" * 98)
    print(f"  {'joint':22} {'present':>9} {'arm 7':>9} {'to lower':>10} {'to upper':>10} {'soft lower':>11} {'soft upper':>11}")
    now, new_pose = pose(*PRESENT), pose(*ARM5)
    for name in ("right_hip_pitch_04", "right_knee_04", "right_foot_pitch_02",
                 "right_hip_roll_03", "right_foot_roll_02"):
        lo, hi, slo, shi = lims[name]
        a, b = now[name], new_pose[name]
        print(f"  {name:22} {a:+9.5f} {b:+9.5f} {np.degrees(b - lo):9.1f}d {np.degrees(hi - b):9.1f}d "
              f"{np.degrees(b - slo):10.1f}d {np.degrees(shi - b):10.1f}d")
    print()
    print("  the same two rows for the present pose, in degrees to the nearer hard stop")
    for name in ("right_foot_pitch_02", "right_knee_04"):
        lo, hi, _, _ = lims[name]
        print(f"    {name:22} present {min(np.degrees(now[name] - lo), np.degrees(hi - now[name])):6.1f}d   "
              f"arm 7 {min(np.degrees(new_pose[name] - lo), np.degrees(hi - new_pose[name])):6.1f}d")
    print()

    print("=" * 98)
    print("G. The action scale table of kscale_base_env_cfg.py:164, re-derived at the arm 7 pose")
    print("=" * 98)
    print("  the action at saturation is (tau / K) / scale and is pose independent, the action to")
    print("  the nearer stop is the travel the nominal leaves divided by the scale and moves with")
    print("  the pose.  the scales are read from the tree of 2026-09-23, whose two ankle entries")
    print("  stand at 0.08 rather than the 0.1619 the sizing rule of section 21.4 derives")
    print()
    print(f"  {'group':12} {'K':>6} {'tau':>7} {'tau/K':>8} {'scale':>8} {'sat action':>11} "
          f"{'stop, present':>14} {'stop, arm 7':>12}")
    for label, joint, stiffness, ceiling, scale in TREE_SCALES:
        lo, hi, _, _ = lims[joint]
        ratio = ceiling / stiffness
        row = []
        for angles in (now, new_pose):
            nominal = angles[joint]
            row.append(min(nominal - lo, hi - nominal) / scale)
        print(f"  {label:12} {stiffness:6.0f} {ceiling:7.1f} {ratio:8.4f} {scale:8.4f} "
              f"{ratio / scale:11.3f} {row[0]:14.3f} {row[1]:12.3f}")
    print()

    print("=" * 98)
    print("F. The action magnitude the swing ankle pitch demands")
    print("=" * 98)
    print("  the action is a delta from the joint default, target = default + scale * action, so a")
    print("  deeper default is a smaller action for the same commanded angle.  section 12 measures")
    print("  the swing closure requiring -47.5 deg against the -50 deg stop of kscale.urdf:603, and")
    print(f"  the scale is the {ANKLE_SCALE} of the per joint dictionary at kscale_base_env_cfg.py:193")
    print()
    scale = ANKLE_SCALE
    required = np.radians(-47.5)
    stop = lims["right_foot_pitch_02"][0]
    print(f"  {'default':>26} {'to -47.5 deg':>14} {'action':>9} {'to the stop':>13} {'action':>9}")
    for label, angles in (("present, -17.7 deg", PRESENT), ("arm 7,   -30.0 deg", ARM5)):
        default = angles[2]
        print(f"  {label:>26} {required - default:+13.5f}  {(required - default) / scale:+9.3f} "
              f"{stop - default:+12.5f}  {(stop - default) / scale:+9.3f}")
    print()
    print(f"  the arm cuts the action the swing requirement costs by "
          f"{100 * (1 - (required - ARM5[2]) / (required - PRESENT[2])):.1f} per cent, and it does so by")
    print("  moving the default toward the stop rather than by moving the stop, so the headroom")
    print("  section D reports falls in the same proportion and the two must be read together")
    print()

    print("=" * 98)
    print("E. The single support stance load the fold moves")
    print("=" * 98)
    print("  the centre of pressure is placed beneath the ankle roll axis, which is the balanced case,")
    print("  so the hip yaw reads near zero at a zero roll pose and the 8.575 Nm of the addition of")
    print("  2026-09-16 to context/KScale.md is a different quantity, being section 10.6's hold torque")
    print("  at a laterally offset centre of mass.  the hip pitch and the knee reproduce that entry")
    print("  exactly at a zero roll pose, at 11.288 and 27.877 Nm, which is the check on this function")
    print()
    print(f"  {'joint':22} {'present':>10} {'arm 7':>10} {'change':>9}")
    a = stance_torques(model, pose(*PRESENT), 1.0)
    b = stance_torques(model, pose(*ARM5), 1.0)
    for name in sorted(set(a) & set(b)):
        if abs(a[name]) < 1e-9 and abs(b[name]) < 1e-9:
            continue
        pct = 100.0 * (abs(b[name]) - abs(a[name])) / abs(a[name]) if abs(a[name]) > 1e-9 else float("nan")
        print(f"  {name:22} {a[name]:+10.4f} {b[name]:+10.4f} {pct:+8.1f}%")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
