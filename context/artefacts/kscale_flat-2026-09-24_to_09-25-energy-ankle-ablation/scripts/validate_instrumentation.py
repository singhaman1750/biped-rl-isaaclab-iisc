"""Self test of the three instrumentation channels added to play.py for section 20.2.

Two of the three carry arithmetic that is easy to get silently wrong, and neither failure
would announce itself in a dump. A sole tilt taken about the wrong axis returns plausible
numbers of the wrong quantity, and a torque envelope assembled from the wrong parameter
returns a saturation fraction that is merely optimistic. Both are therefore checked here
against a construction whose answer is known in advance, rather than against a dump.

The third channel, the commanded target, needs no numerical check, being a tensor the
action term already holds. What it needs is the alignment check at the end, that a term
driving fewer joints than the articulation scatters into the right columns.

Nothing here imports isaaclab. Its math module reaches the USD runtime through
isaaclab/utils/mesh.py:14 and cannot be loaded without the simulator, and importing the
actuator class for the sake of one method would drag the same dependency in. The rotation
is therefore built as an explicit matrix from the very angles the check must recover, which
tests the quantity actually at risk, being the sign convention of the tilt extraction and
the per robot sole normal, and DCMotor._clip_effort is reproduced verbatim below from
IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:295 as the reference. The
quaternion helpers play.py calls are isaaclab's own and are exercised by the sole clearance
channel that has been dumped since 2026-08-07.

Runs anywhere torch is importable, which includes the GPU containers and excludes the
development container.
"""
import math
import sys

import torch

FAIL = []


def check(name, got, want, tol=1e-5):
    ok = abs(float(got) - float(want)) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:58s} got {float(got):+.6f} want {float(want):+.6f}")
    if not ok:
        FAIL.append(name)


# --- A. The sole tilt, against a plate rolled and pitched by known angles ---------------
#
# The check reproduces play.py's _sole_tilt on a single synthetic foot whose orientation is
# built from the very angles the function must recover. A sole lying flat on level ground
# has its ground facing normal at (0, 0, -1) in the base yaw frame, and the two angles are
# the rotations about the forward and the lateral axis that carry it there.
print("\nA. sole tilt, synthetic foot at known roll and pitch")


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float64)


def sole_tilt(foot_R, yaw_R, normal_l):
    """play.py DataLogger._sole_tilt, with the rotations given as matrices.

    The two lines that matter are the last two and they are play.py's verbatim. The
    rotation into the world and then into the base yaw frame is a matrix product here and
    a pair of quaternion applications there, which are the same map.
    """
    normal = torch.tensor(normal_l, dtype=torch.float64)
    normal_w = foot_R @ normal
    normal_b = yaw_R.T @ normal_w
    roll = torch.atan2(normal_b[1], -normal_b[2])
    pitch = torch.atan2(-normal_b[0], -normal_b[2])
    return float(roll), float(pitch)


identity = torch.eye(3, dtype=torch.float64)

# The SD_BRS1 convention, the sole facing the link's minus z, so the link frame at identity
# is already flat and the tilt must read zero.
r, p = sole_tilt(identity, identity, (0.0, 0.0, -1.0))
check("SD_BRS1 sole, foot at identity, roll", r, 0.0)
check("SD_BRS1 sole, foot at identity, pitch", p, 0.0)

for angle in (0.05, 0.15, -0.2618):
    r, p = sole_tilt(rot_x(angle), identity, (0.0, 0.0, -1.0))
    check(f"SD_BRS1 sole rolled {angle:+.4f} rad, roll recovered", r, angle)
    check(f"SD_BRS1 sole rolled {angle:+.4f} rad, pitch stays zero", p, 0.0)

for angle in (0.1, -0.3491):
    r, p = sole_tilt(rot_y(angle), identity, (0.0, 0.0, -1.0))
    check(f"SD_BRS1 sole pitched {angle:+.4f} rad, pitch recovered", p, angle)
    check(f"SD_BRS1 sole pitched {angle:+.4f} rad, roll stays zero", r, 0.0)

# The kscale convention, the sole facing the link's PLUS y. The link frame at identity is
# therefore NOT flat, the sole pointing along plus y rather than minus z, and the tilt must
# read minus a quarter turn of roll. A dump taken against SD_BRS1's normal on this robot
# would read that quarter turn as a ninety degree cant, which is the failure this row
# exists to catch, and it is the same class of failure as the foot yaw permutation that
# cluster 18 of /ws/context/literature.md records.
r, p = sole_tilt(identity, identity, (0.0, 1.0, 0.0))
check("kscale sole, foot link at identity, roll is a quarter turn", r, math.pi / 2)

# With the foot rolled by minus a quarter turn the kscale sole lies flat, which is the
# configuration the standing pose puts it in, and both angles must vanish.
r, p = sole_tilt(rot_x(-math.pi / 2), identity, (0.0, 1.0, 0.0))
check("kscale sole, foot rolled -pi/2, roll", r, 0.0)
check("kscale sole, foot rolled -pi/2, pitch", p, 0.0)

for angle in (0.05, -0.1745):
    r, p = sole_tilt(rot_x(-math.pi / 2 + angle), identity, (0.0, 1.0, 0.0))
    check(f"kscale sole rolled {angle:+.4f} from flat, roll recovered", r, angle)

# The base yaw must not leak into either angle, a sole flat on level ground reading zero
# whatever direction the machine faces, which is the property that licenses referencing the
# tilt to the base yaw rather than to the world.
for yaw in (0.3, 1.7, -2.4):
    r, p = sole_tilt(rot_z(yaw), rot_z(yaw), (0.0, 0.0, -1.0))
    check(f"flat sole at base yaw {yaw:+.2f}, roll", r, 0.0)
    check(f"flat sole at base yaw {yaw:+.2f}, pitch", p, 0.0)


# --- B. The torque envelope, against DCMotor._clip_effort itself ------------------------
#
# The envelope is not re-derived here, it is compared against the implementation that
# actually performs the clip, driven at the kscale ankle parameters and at the hip pitch
# parameters, across a velocity sweep that spans the sign change and the effort limit knee.
print("\nB. torque envelope, against DCMotor._clip_effort")


def envelope(vel, saturation, effort, speed):
    """play.py DataLogger._available_torque, for one joint."""
    reach = speed * (1.0 + effort / saturation)
    v = torch.clamp(vel, min=-reach, max=reach)
    upper = torch.clamp(saturation * (1.0 - v / speed), max=effort)
    lower = torch.clamp(saturation * (-1.0 - v / speed), min=-effort)
    return lower, upper


def reference_clip(effort_in, joint_vel, saturation, effort, speed):
    """DCMotor._clip_effort, reproduced verbatim from actuator_pd.py:295.

    Quoted rather than imported because importing the class reaches the USD runtime. The
    body below is the upstream body with the attribute names substituted for locals and
    nothing else changed, and _vel_at_effort_lim is its constructor's line 270.
    """
    vel_at_effort_lim = speed * (1 + effort / saturation)
    joint_vel = torch.clip(joint_vel, min=-vel_at_effort_lim, max=vel_at_effort_lim)
    torque_speed_top = saturation * (1.0 - joint_vel / speed)
    torque_speed_bottom = saturation * (-1.0 - joint_vel / speed)
    max_effort = torch.clip(torque_speed_top, max=effort)
    min_effort = torch.clip(torque_speed_bottom, min=-effort)
    return torch.clip(effort_in, min=min_effort, max=max_effort)


# (saturation_effort, effort_limit, velocity_limit), the two kscale groups as the runs of
# section 16 dumped them, the ankle carrying equal saturation and effort so that its corner
# velocity is zero, which is the condition section 10.2 records as unintended.
GROUPS = {
    "kscale ankle, 17.0 / 17 / 37.699": (17.0, 17.0, 37.699),
    "kscale hip pitch, 120 / 84 / 17.488": (120.0, 84.0, 17.488),
    "kscale hip yaw, 60 / 42 / 18.849": (60.0, 42.0, 18.849),
}
speeds = torch.tensor(
    [-60.0, -37.7, -25.0, -11.31, -1.17, -0.013, 0.0, 0.013, 1.17, 11.31, 25.0, 37.7, 60.0],
    dtype=torch.float64)

for label, (sat, eff, spd) in GROUPS.items():
    # The reference bounds are recovered by clipping a torque far outside the envelope on
    # each side, which returns the bound itself.
    big = torch.full_like(speeds, 1.0e6)
    ref_upper = reference_clip(big, speeds, sat, eff, spd)
    ref_lower = reference_clip(-big, speeds, sat, eff, spd)
    lower, upper = envelope(speeds, sat, eff, spd)
    worst_u = float((upper - ref_upper).abs().max())
    worst_l = float((lower - ref_lower).abs().max())
    check(f"{label}, upper bound over the sweep", worst_u, 0.0, tol=1e-9)
    check(f"{label}, lower bound over the sweep", worst_l, 0.0, tol=1e-9)

# The reading the envelope repairs, stated once at the ankle so that the size of the
# correction is on the record rather than asserted.
sat, eff, spd = GROUPS["kscale ankle, 17.0 / 17 / 37.699"]
for v in (0.013, 1.17, 11.31, 25.0):
    _, up = envelope(torch.tensor([v], dtype=torch.float64), sat, eff, spd)
    print(f"        ankle at {v:7.3f} rad/s, available {float(up):7.3f} N.m "
          f"against a flat ceiling of {eff:.1f}, a derating of "
          f"{100.0 * (1 - float(up) / eff):5.2f} pct")


# --- C. The commanded target, alignment under a reduced action set ----------------------
#
# The scatter is checked rather than the tensor, since the tensor is the action term's own.
# The property that matters is that a term driving a subset of the joints writes its
# columns at the ARTICULATION's joint indices and leaves the rest NaN, which is what arms
# 7, 10 and 19 of the plan will each produce and what a consumer assuming equal widths
# would read wrongly.
print("\nC. commanded target, scatter under a reduced action set")

JOINTS = ["left_hip_pitch_04", "right_hip_pitch_04", "left_hip_roll_03",
          "right_hip_roll_03", "left_hip_yaw_03", "right_hip_yaw_03",
          "left_knee_04", "right_knee_04", "left_foot_pitch_02",
          "right_foot_pitch_02", "left_foot_roll_02", "right_foot_roll_02"]
# Arm 16 freezes the hip yaw, leaving ten driven joints at articulation indices 0-3, 6-11.
driven = [i for i, n in enumerate(JOINTS) if "hip_yaw" not in n]
index = torch.tensor(driven)
processed = torch.arange(1.0, 1.0 + len(driven), dtype=torch.float64).view(1, -1)
targets = torch.full((1, len(JOINTS)), float("nan"), dtype=torch.float64)
targets[:, index] = processed
check("driven columns carry the processed action", float(targets[0, 6]), 5.0)
check("frozen hip yaw column is NaN, left", float(torch.isnan(targets[0, 4])), 1.0)
check("frozen hip yaw column is NaN, right", float(torch.isnan(targets[0, 5])), 1.0)
check("last driven column lands at the last joint", float(targets[0, 11]), 10.0)
check("no driven column is NaN", float(torch.isnan(targets[0, index]).any()), 0.0)



# --- D. The dashboard panel that reads the commanded target -----------------------------
#
# Rendered rather than inspected, because the failure this section exists to catch produced
# a figure that was complete in every visible respect and silently missing all sixty of its
# reference lines. plotly's add_hline carries exclude_empty_subplots=True by default, so a
# line added to a panel that holds no trace yet is dropped without an error, and the panel
# drew its limits before its traces. The check counts the shapes.
#
# Guarded rather than mandatory, this section needing plotly and a dump where the three
# above need neither, so the arithmetic is still checked where the visualisation stack is
# absent. It reconstructs the commanded target by inverting the control law, which is the
# very inference section 20.2 of the plan exists to retire, purely so that the panel can be
# exercised before a dump carrying the recorded channel exists.
def check_dashboard(dump_path):
    import os
    import numpy as np
    sys.path.insert(0, "/workspace/isaaclab/biped/scripts/analysis")
    import dashboard

    d = np.load(dump_path, allow_pickle=True).item()
    names = [str(n) for n in d["joint_names"]]
    KP = {"hip_yaw": 100.0, "hip_roll": 250.0, "hip_pitch": 200.0, "knee": 300.0,
          "foot_pitch": 100.0, "foot_roll": 100.0}
    KD = {"hip_yaw": 3.5, "hip_roll": 28.0, "hip_pitch": 21.5, "knee": 12.0,
          "foot_pitch": 1.25, "foot_roll": 1.05}
    kp = np.array([next(v for k, v in KP.items() if k in n) for n in names])
    kd = np.array([next(v for k, v in KD.items() if k in n) for n in names])
    pos, vel = np.asarray(d["joint_positions"]), np.asarray(d["joint_velocities"])
    d["joint_position_targets"] = (
        pos + (np.asarray(d["joint_torques"]) + kd * vel) / kp).astype(np.float32)

    fig = dashboard.create_tracking_plot({"A": d}, env_id=0, num_legs=2)
    check("one achieved and one commanded trace per joint", len(fig.data), 2 * len(names))
    check("two hard, two soft and one nominal line per joint",
          len(fig.layout.shapes), 5 * len(names))
    check("one subplot title per joint",
          sum(1 for a in fig.layout.annotations if "err<" in (a.text or "")), len(names))

    # A dump without the channel must say so rather than draw the achieved position alone,
    # which would read as perfect tracking on every joint.
    bare = {k: v for k, v in d.items() if k != "joint_position_targets"}
    empty = dashboard.create_tracking_plot({"A": bare}, env_id=0, num_legs=2)
    check("a dump lacking the channel draws nothing", len(empty.data), 0)
    check("and says why", float("play.py has recorded" in empty.layout.title.text), 1.0)


print("\nD. dashboard panel, rendered against a dump")
_DUMP = ("/workspace/isaaclab/logs/rsl_rl/kscale_flat_eval/"
         "2026-09-24_05-18-33_model_9000/data/42/dump.npy")
try:
    check_dashboard(_DUMP)
except Exception as error:  # noqa: BLE001
    print(f"  SKIP  {type(error).__name__}, {error}")
    print("        This section needs plotly and the dump above. The arithmetic of "
          "sections A through C is unaffected.")

print(f"\n{'ALL CHECKS PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
