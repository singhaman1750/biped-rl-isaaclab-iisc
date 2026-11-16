"""The commanded target, the hip yaw phase relation and the ankle roll mechanism.

Three questions the aggregate tables of ablation.py cannot answer are settled here.

Whether a joint's departure from its nominal is COMMANDED by the policy or is a tracking
failure of the servo, which is answered by inverting the control law. The KScale runs use
an explicit PD inside DCMotor, tau = clip(kp (q_des - q) - kd qdot), so wherever the
sample is unsaturated q_des = q + (tau + kd qdot) / kp exactly, with no approximation.
The hip yaw is unsaturated on every sample of every run, so its commanded target is
recovered in full.

Whether the two hip yaws move in phase with one another, which distinguishes a pelvis
twist from a per leg splay and decides whether a per leg penalty can reach it.

And whether the ankle roll deviation penalty reduced the deviation it prices or merely
drove the actuator against the ground, which is read from the sign of the torque against
the sign of the deviation.
"""
import os, re, sys
import numpy as np
sys.path.insert(0, "/workspace/isaaclab/biped/scripts/analysis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation import (EVAL, LOGS, TAG, FORCE, FRAC, envelope, available, rot, yaw_of,
                      wrap, edges)

JOBS = sys.argv[1:]

def load(job):
    d = np.load(os.path.join(EVAL, job, "data", "42", "dump.npy"),
                allow_pickle=True).item()
    return d

print("### Commanded target recovered by inverting the control law")
print("    q_des = q + (tau + kd qdot) / kp, exact on every unsaturated sample.")
print("    'commanded' is the mean absolute departure of q_des from the nominal,")
print("    'achieved' the same for q, and 'error' the mean absolute q_des - q.\n")
hdr = f"{'joint':24s} {'run':10s} {'valid%':>7s} {'commanded':>10s} {'achieved':>9s} {'error':>8s} {'ratio':>7s}"
print(hdr)
store = {}
for job in JOBS:
    d = load(job)
    exp = job.rsplit("_model_", 1)[0]
    names = [str(n) for n in d["joint_names"]]
    pos, vel, tau = (np.asarray(d[k]) for k in
                     ("joint_positions", "joint_velocities", "joint_torques"))
    nom = np.asarray(d["default_joint_pos"])
    sat, eff, vlim, kp, kd = envelope(d, os.path.join(LOGS, exp))
    store[job] = (d, names, pos, vel, tau, nom, sat, eff, vlim, kp, kd)

for j_name in ["left_hip_yaw_03", "right_hip_yaw_03", "left_hip_roll_03",
               "left_foot_roll_02", "right_foot_roll_02", "left_foot_pitch_02"]:
    for job in JOBS:
        d, names, pos, vel, tau, nom, sat, eff, vlim, kp, kd = store[job]
        j = names.index(j_name)
        av = available(tau[:, :, j], vel[:, :, j], sat[j], eff[j], vlim[j])
        ok = np.abs(tau[:, :, j]) < FRAC * av
        qdes = pos[:, :, j] + (tau[:, :, j] + kd[j] * vel[:, :, j]) / kp[j]
        if ok.sum() < 1000:
            print(f"{j_name:24s} {TAG[job.rsplit('_model_',1)[0]]+'@'+job.rsplit('_',1)[-1]:10s}"
                  f" {100.0*ok.mean():7.2f}   too few unsaturated samples")
            continue
        cmd = np.abs(qdes - nom[j])[ok].mean()
        ach = np.abs(pos[:, :, j] - nom[j])[ok].mean()
        err = np.abs(qdes - pos[:, :, j])[ok].mean()
        print(f"{j_name:24s} {TAG[job.rsplit('_model_',1)[0]]+'@'+job.rsplit('_',1)[-1]:10s}"
              f" {100.0*ok.mean():7.2f} {cmd:10.4f} {ach:9.4f} {err:8.4f} {cmd/max(ach,1e-9):7.3f}")
    print()

print("\n### Are the two hip yaws in phase with one another")
print("    A positive correlation of the two RAW angles is a pelvis twist, the legs")
print("    yawing together.  A negative correlation is a splay, the legs yawing apart.\n")
print(f"{'run':10s} {'corr(L,R) raw':>14s} {'corr mirrored':>14s} {'corr with base yaw rate':>24s} {'sum std':>9s} {'diff std':>9s}")
for job in JOBS:
    d, names, pos, vel, tau, nom, sat, eff, vlim, kp, kd = store[job]
    li, ri = names.index("left_hip_yaw_03"), names.index("right_hip_yaw_03")
    L = (pos[:, :, li] - nom[li]).ravel()
    R = (pos[:, :, ri] - nom[ri]).ravel()
    wz = np.asarray(d["base_angular_velocity"])[:, :, 2].ravel()
    c_raw = np.corrcoef(L, R)[0, 1]
    c_mir = np.corrcoef(-L, R)[0, 1]
    c_wz = np.corrcoef(L + R, wz)[0, 1]
    print(f"{TAG[job.rsplit('_model_',1)[0]]+'@'+job.rsplit('_',1)[-1]:10s} {c_raw:14.3f} {c_mir:14.3f}"
          f" {c_wz:24.3f} {np.std(L+R):9.4f} {np.std(L-R):9.4f}")

print("\n### Hip yaw against stance toe-in, with the left leg mirrored")
print("    toe-in already flips the left foot, so the left hip yaw must be flipped with it.\n")
print(f"{'run':10s} {'foot':8s} {'slope deg/deg':>14s} {'correlation':>12s} {'n':>10s}")
for job in JOBS:
    d, names, pos, vel, tau, nom, sat, eff, vlim, kp, kd = store[job]
    fq, bq = np.asarray(d["feet_quaternions"]), np.asarray(d["base_quaternion"])
    by = yaw_of(bq)
    contact = np.linalg.norm(np.asarray(d["feet_contact_forces"]), axis=-1) > FORCE
    feet = [str(n) for n in d["feet_names"]]
    for f in range(2):
        toe = rot(fq[:, :, f, :], (0.0, 0.0, -1.0))
        rel = wrap(np.arctan2(toe[..., 1], toe[..., 0]) - by)
        side = "left" if feet[f].startswith("left") else "right"
        sgn = -1.0 if side == "left" else 1.0
        toe_in = np.degrees(sgn * rel)
        j = names.index(f"{side}_hip_yaw_03")
        x = np.degrees(sgn * (pos[:, :, j] - nom[j]))[contact[:, :, f]]
        y = toe_in[contact[:, :, f]]
        sl = np.polyfit(x, y, 1)[0]
        cr = np.corrcoef(x, y)[0, 1]
        print(f"{TAG[job.rsplit('_model_',1)[0]]+'@'+job.rsplit('_',1)[-1]:10s} {side:8s}"
              f" {sl:14.3f} {cr:12.3f} {len(x):10d}")
    print()

print("\n### The ankle roll, what the deviation penalty did")
print("    'restoring' is the fraction of samples on which the torque opposes the")
print("    deviation from the nominal, which is the actuator pulling the joint home.")
print("    'against ground' is the same fraction restricted to stance.\n")
print(f"{'joint':22s} {'run':10s} {'dev':>8s} {'|tau|':>8s} {'restoring%':>11s} {'stance restoring%':>18s} {'stop%':>7s}")
for j_name in ["left_foot_roll_02", "right_foot_roll_02"]:
    for job in JOBS:
        d, names, pos, vel, tau, nom, sat, eff, vlim, kp, kd = store[job]
        j = names.index(j_name)
        contact = np.linalg.norm(np.asarray(d["feet_contact_forces"]), axis=-1) > FORCE
        feet = [str(n) for n in d["feet_names"]]
        side = "left" if j_name.startswith("left") else "right"
        f = 0 if feet[0].startswith(side) else 1
        dev = pos[:, :, j] - nom[j]
        t = tau[:, :, j]
        moving = np.abs(dev) > 1e-3
        rest = (np.sign(t) == -np.sign(dev)) & moving
        lim = np.asarray(d["joint_position_limits"])[j]
        stop = (pos[:, :, j] <= lim[0] + 0.01*(lim[1]-lim[0])) | \
               (pos[:, :, j] >= lim[1] - 0.01*(lim[1]-lim[0]))
        st = contact[:, :, f]
        print(f"{j_name:22s} {TAG[job.rsplit('_model_',1)[0]]+'@'+job.rsplit('_',1)[-1]:10s}"
              f" {np.abs(dev).mean():8.4f} {np.abs(t).mean():8.3f}"
              f" {100.0*rest[moving].mean():11.2f}"
              f" {100.0*rest[moving & st].mean():18.2f} {100.0*stop.mean():7.2f}")
    print()
