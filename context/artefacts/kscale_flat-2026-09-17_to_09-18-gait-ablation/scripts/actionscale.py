"""Action scale against joint range and against the deflection that saturates each joint."""
import numpy as np

BASE = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"A 09-17_09-13": "2026-09-17_09-13-46", "B 09-17_11-25": "2026-09-17_11-25-59",
        "C 09-18_09-37": "2026-09-18_09-37-29", "D 09-18_10-03": "2026-09-18_10-03-48"}
D = {l: np.load(f"{BASE}/{r}/data/42/dump.npy", allow_pickle=True).item() for l, r in RUNS.items()}
d0 = D["A 09-17_09-13"]
JN = d0["joint_names"]
lim = np.array(d0["joint_position_limits"])
eff = np.array(d0["joint_effort_limits"])
SCALE = 0.4

GAINS = {"hip_yaw": (100.0, 3.5), "hip_roll": (250.0, 28.0), "hip_pitch": (200.0, 21.5),
         "knee": (300.0, 12.0), "foot_roll": (100.0, 1.05), "foot_pitch": (100.0, 1.25)}


def group(j):
    for g in GAINS:
        if g in j:
            return g
    raise KeyError(j)


print("ACTION AUTHORITY AUDIT, action scale = 0.4 rad per unit action, applied to every joint alike")
print()
print(f"{'joint':<22}{'K':>7}{'tau_max':>9}{'dq_sat':>9}{'range':>9}{'half':>8}{'a_sat':>8}{'a_range':>9}")
print(f"{'':22}{'Nm/rad':>7}{'Nm':>9}{'rad':>9}{'rad':>9}{'rad':>8}{'unit':>8}{'unit':>9}")
print("-" * 82)
for i, j in enumerate(JN):
    g = group(j)
    K, Dg = GAINS[g]
    dq_sat = eff[i] / K                      # deflection that puts the PD at its effort ceiling
    rng = lim[i, 1] - lim[i, 0]
    half = rng / 2
    a_sat = dq_sat / SCALE                   # action magnitude that saturates the joint
    a_rng = half / SCALE                     # action magnitude that spans half the joint range
    print(f"{j:<22}{K:7.0f}{eff[i]:9.2f}{dq_sat:9.4f}{rng:9.4f}{half:8.4f}{a_sat:8.3f}{a_rng:9.3f}")
print()
print("a_sat is the action magnitude at which the joint's PD hits its torque ceiling.")
print("a_range is the action magnitude that commands half the joint's mechanical range.")
print("A joint whose a_sat is small saturates on ordinary actions, one whose a_range is near or")
print("below 1 is commanded past its own end stops by an action of unit magnitude.")
print()

# the hip roll is asymmetric, so quote the adduction side separately
print("ASYMMETRIC RANGE, the hip roll adduction side alone")
for i, j in enumerate(JN):
    if "hip_roll" not in j:
        continue
    dflt = np.array(d0["default_joint_pos"])[i]
    add = min(abs(lim[i, 0] - dflt), abs(lim[i, 1] - dflt))
    abd = max(abs(lim[i, 0] - dflt), abs(lim[i, 1] - dflt))
    print(f"  {j:<22} travel to the near stop {add:6.4f} rad, to the far stop {abd:6.4f} rad,"
          f"  action to reach the near stop {add/SCALE:5.3f}")
print()

print("=" * 108)
print("IMPLIED COMMANDED POSITION, reconstructed as q_des = q + (tau + D qdot) / K")
print("valid only where the actuator is NOT clipped, so restricted to |tau| < 0.9 tau_max")
print(f"  {'joint':<22}{'A':>13}{'B':>13}{'C':>13}{'D':>13}   quantity")
for i, j in enumerate(JN):
    if not ("foot_roll" in j or "hip_roll" in j or "knee" in j):
        continue
    g = group(j)
    K, Dg = GAINS[g]
    cells = []
    for l in RUNS:
        d = D[l]
        q = d["joint_positions"][:, :, i]
        t = d["joint_torques"][:, :, i]
        qd = d["joint_velocities"][:, :, i]
        m = np.abs(t) < 0.9 * eff[i]
        if m.sum() < 100:
            cells.append(f"{'n<100':>13}")
            continue
        qdes = q[m] + (t[m] + Dg * qd[m]) / K
        out = 100.0 * ((qdes < lim[i, 0]) | (qdes > lim[i, 1])).mean()
        cells.append(f"{out:13.2f}")
    print(f"  {j:<22}{''.join(cells)}   pct of unclipped samples whose TARGET lies outside the stops")
print()

print("=" * 108)
print("REWARD BUDGET, per term mean rate per second, and the behavioural value after dividing by weight")
rw = {l: np.load(f"{BASE}/{r}/data/42/rewards.npy", allow_pickle=True).item() for l, r in RUNS.items()}
W = rw["A 09-17_09-13"]["_weights"]
terms = [k for k in rw["A 09-17_09-13"] if k not in ("_weights", "total_reward")]
print(f"  {'term':<36}{'weight':>10}{'A':>12}{'B':>12}{'C':>12}{'D':>12}")
print("  --- weighted rate per second")
tot = {l: 0.0 for l in RUNS}
for k in terms:
    cells = []
    for l in RUNS:
        v = rw[l][k].mean() if k in rw[l] else np.nan
        tot[l] += 0.0 if np.isnan(v) else v
        cells.append(f"{v:12.4f}" if not np.isnan(v) else f"{'--':>12}")
    wt = W.get(k, float("nan"))
    wts = f"{wt:10.5g}" if k in W else f"{'--':>10}"
    print(f"  {k:<36}{wts}{''.join(cells)}")
print(f"  {'SUM of the weighted rates':<36}{'':10}{''.join(f'{tot[l]:12.4f}' for l in RUNS)}")
print(f"  {'total_reward mean (per step)':<36}{'':10}"
      f"{''.join(f'{rw[l][chr(116)+chr(111)+chr(116)+chr(97)+chr(108)+chr(95)+chr(114)+chr(101)+chr(119)+chr(97)+chr(114)+chr(100)].mean():12.4f}' for l in RUNS)}")
print()
print("  --- behavioural value, the weighted rate divided by the weight")
for k in terms:
    if k not in W or W[k] == 0:
        continue
    cells = []
    for l in RUNS:
        v = (rw[l][k].mean() / rw[l]["_weights"][k]) if k in rw[l] else np.nan
        cells.append(f"{v:12.5f}" if not np.isnan(v) else f"{'--':>12}")
    print(f"  {k:<36}{W[k]:10.5g}{''.join(cells)}")
print()
print("  per run weights, to confirm each dump carries its own trained weights")
for k in ["termination_penalty", "pen_base_height", "pen_feet_heading", "pen_hip_yaw_deviation",
          "rew_keep_hip_yaw_zero_in_air"]:
    cells = []
    for l in RUNS:
        w = rw[l]["_weights"].get(k)
        cells.append(f"{w:12.5g}" if w is not None else f"{'absent':>12}")
    print(f"  {k:<36}{'':10}{''.join(cells)}")
