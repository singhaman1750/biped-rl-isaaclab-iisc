"""The evaluation reward budget of the three runs, term by term.

context/gait_metrics.md records that the per term series in rewards.npy are RATES PER SECOND
while total_reward is PER STEP, a distinction worth a factor of one hundred at the 0.01 s step,
so the per term figures below are quoted per second and the total is converted to match.
"""
import numpy as np, common
R = {r: common.load(r, "rewards") for r in common.RUNS}
W = R["A_scale0.4"]["_weights"]
terms = [k for k in R["A_scale0.4"] if k != "_weights"]
dt = 0.01
rows = []
for k in terms:
    v = [float(np.mean(R[r][k])) for r in common.RUNS]
    if k == "total_reward": v = [x/dt for x in v]
    rows.append((k, v, W.get(k, float("nan"))))
rows.sort(key=lambda r: -abs(r[1][0]))
print(f"{'term':36}{'weight':>10}" + "".join(f"{r:>13}" for r in common.RUNS) + "     B/A     C/A")
for k, v, w in rows:
    ba = v[1]/v[0] if abs(v[0]) > 1e-12 else float("nan")
    ca = v[2]/v[0] if abs(v[0]) > 1e-12 else float("nan")
    print(f"{k:36}{w:>10.5g}" + "".join(f"{x:>13.4f}" for x in v) + f"{ba:>8.2f}{ca:>8.2f}")
print("\npositive and negative budgets, per second")
for r in common.RUNS:
    pos = sum(float(np.mean(R[r][k])) for k in terms if k != "total_reward" and float(np.mean(R[r][k])) > 0)
    neg = sum(float(np.mean(R[r][k])) for k in terms if k != "total_reward" and float(np.mean(R[r][k])) < 0)
    tot = float(np.mean(R[r]["total_reward"]))/dt
    print(f"  {r:12} positive {pos:9.3f}  negative {neg:9.3f}  sum {pos+neg:9.3f}  total_reward {tot:9.3f}")
print("\nthe four energy terms as a share of the negative budget")
EN = ["pen_joint_torque","pen_joint_accel","pen_joint_vel_l2","pen_joint_torque_rate"]
for r in common.RUNS:
    neg = sum(float(np.mean(R[r][k])) for k in terms if k != "total_reward" and float(np.mean(R[r][k])) < 0)
    en = sum(float(np.mean(R[r][k])) for k in EN)
    print(f"  {r:12} energy {en:9.4f} of {neg:9.3f} negative, {100*en/neg:6.2f} per cent")
