"""Which joints pay pen_joint_pos_limits, reproduced from the dump against the live definition.

isaaclab/envs/mdp/rewards.py:189 sums the absolute excursion beyond each joint's soft limit over
all joints, so the term is a sum and its share is attributable joint by joint.  The question is
whether the term reports a fault or reports a joint doing its job, and the answer is the share.
"""
import numpy as np, common
print(f"{'joint':22}" + "".join(f"{r+' share%':>16}" for r in common.RUNS)
      + "".join(f"{r+' active%':>17}" for r in common.RUNS))
tot = {}
share = {}
for run in common.RUNS:
    d = common.load(run); q = d["joint_positions"]; soft = np.asarray(d["joint_soft_position_limits"])
    lo = np.clip(soft[None, None, :, 0] - q, 0, None)
    hi = np.clip(q - soft[None, None, :, 1], 0, None)
    oo = lo + hi                                   # (T,E,12)
    tot[run] = oo.sum(-1).mean()
    share[run] = oo.mean(axis=(0, 1))
J = common.load("A_scale0.4")["joint_names"]
for i, n in enumerate(J):
    row = "".join(f"{100*share[r][i]/share[r].sum():>15.1f}%" for r in common.RUNS)
    act = ""
    for r in common.RUNS:
        d = common.load(r); q = d["joint_positions"]; soft = np.asarray(d["joint_soft_position_limits"])
        act += f"{100*np.mean((q[:,:,i]<soft[i,0])|(q[:,:,i]>soft[i,1])):>16.1f}%"
    print(f"{n:22}{row}{act}")
print("\nterm value per step, unweighted, and the weighted rate per second at the configured -2.0")
for r in common.RUNS:
    print(f"  {r:12} sum of excursions {tot[r]:.5f} rad, weighted rate {-2.0*tot[r]/0.01*0.01:.4f} per second")
print("\nthe hip roll pair alone")
for r in common.RUNS:
    hipr = share[r][2] + share[r][3]
    print(f"  {r:12} {100*hipr/share[r].sum():5.1f} per cent of the term")
