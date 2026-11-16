"""Raw dump analysis: joint limit residency, true toe-in angle, speed-binned crouch."""
import numpy as np

BASE = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"A 09-17_09-13": "2026-09-17_09-13-46", "B 09-17_11-25": "2026-09-17_11-25-59",
        "C 09-18_09-37": "2026-09-18_09-37-29", "D 09-18_10-03": "2026-09-18_10-03-48"}

D = {}
for label, run in RUNS.items():
    D[label] = np.load(f"{BASE}/{run}/data/42/dump.npy", allow_pickle=True).item()

d0 = D["A 09-17_09-13"]
JN = d0["joint_names"]
lim = np.array(d0["joint_position_limits"])
slim = np.array(d0["joint_soft_position_limits"])
dfl = np.array(d0["default_joint_pos"])
eff = np.array(d0["joint_effort_limits"])

print("JOINT REFERENCE TABLE (identical in all four runs, verified below)")
print(f"{'joint':<24}{'default':>9}{'hard lo':>9}{'hard hi':>9}{'soft lo':>9}{'soft hi':>9}{'effort':>9}")
for i, j in enumerate(JN):
    print(f"{j:<24}{dfl[i]:9.4f}{lim[i,0]:9.4f}{lim[i,1]:9.4f}{slim[i,0]:9.4f}{slim[i,1]:9.4f}{eff[i]:9.2f}")
for label in RUNS:
    same = (np.allclose(np.array(D[label]['default_joint_pos']), dfl)
            and np.allclose(np.array(D[label]['joint_position_limits']), lim)
            and np.allclose(np.array(D[label]['joint_effort_limits']), eff))
    print(f"  {label}: reference identical to A -> {same}")

print()
print("=" * 118)
print("SIGNED MEAN JOINT POSITION [rad], and residency at the soft and hard limits [% of samples]")
for stat, fn in [("signed mean", lambda q: q.mean()),
                 ("p5", lambda q: np.percentile(q, 5)),
                 ("p95", lambda q: np.percentile(q, 95))]:
    print(f"--- {stat}")
    print(f"  {'joint':<24}{'A':>11}{'B':>11}{'C':>11}{'D':>11}{'default':>10}")
    for i, j in enumerate(JN):
        cells = "".join(f"{fn(D[l]['joint_positions'][:, :, i]):11.4f}" for l in RUNS)
        print(f"  {j:<24}{cells}{dfl[i]:10.4f}")

print()
print("RESIDENCY OUTSIDE THE SOFT LIMIT [% of all samples]")
print(f"  {'joint':<24}{'A':>11}{'B':>11}{'C':>11}{'D':>11}")
for i, j in enumerate(JN):
    cells = []
    for l in RUNS:
        q = D[l]["joint_positions"][:, :, i]
        f = 100.0 * ((q < slim[i, 0]) | (q > slim[i, 1])).mean()
        cells.append(f"{f:11.3f}")
    print(f"  {j:<24}{''.join(cells)}")

print()
print("RESIDENCY WITHIN 1 PCT OF THE HARD LIMIT RANGE OF EITHER STOP [% of all samples]")
print(f"  {'joint':<24}{'A':>11}{'B':>11}{'C':>11}{'D':>11}")
for i, j in enumerate(JN):
    span = lim[i, 1] - lim[i, 0]
    cells = []
    for l in RUNS:
        q = D[l]["joint_positions"][:, :, i]
        f = 100.0 * ((q < lim[i, 0] + 0.01 * span) | (q > lim[i, 1] - 0.01 * span)).mean()
        cells.append(f"{f:11.3f}")
    print(f"  {j:<24}{''.join(cells)}")

print()
print("=" * 118)
print("TORQUE SIGN AT THE ANKLES, mean signed torque [Nm] (a constant sign means a joint pressed into a stop)")
print(f"  {'joint':<24}{'A':>11}{'B':>11}{'C':>11}{'D':>11}")
for i, j in enumerate(JN):
    if "foot" not in j and "hip_roll" not in j:
        continue
    cells = "".join(f"{D[l]['joint_torques'][:, :, i].mean():11.4f}" for l in RUNS)
    print(f"  {j:<24}{cells}")
print()
print("FRACTION OF SAMPLES WITH THE TORQUE OF ONE SIGN [%] (100 = always pushing the same way)")
print(f"  {'joint':<24}{'A':>11}{'B':>11}{'C':>11}{'D':>11}")
for i, j in enumerate(JN):
    cells = []
    for l in RUNS:
        t = D[l]["joint_torques"][:, :, i]
        cells.append(f"{100*max((t > 0).mean(), (t < 0).mean()):11.2f}")
    print(f"  {j:<24}{''.join(cells)}")
