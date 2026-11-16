"""The lateral weight shift the stance geometry demands, and the shift the policies produce.

Quasi-static single support requires the centre of pressure to lie under the centre of mass,
and the centre of pressure cannot leave the sole.  This gives a lower bound on the lateral
excursion the torso must make before single support is statically feasible at all, and an
upper bound at which the ankle roll moment vanishes.
"""
import numpy as np, common

SOLE_HALF_WIDTH = 0.0846 / 2          # kscale_base_env_cfg.py:32 contact offsets
G = 9.81
print(f"{'run':9}{'mass':>7}{'weight':>8}{'sep':>8}{'half':>7}{'inner edge':>11}"
      f"{'M_ankle no sway':>17}{'pct of 17Nm':>13}{'sway to zero it':>17}")
geo = {}
for run in common.RUNS:
    d = common.load(run)
    m = float(np.sum(d["body_masses"])); W = m*G
    sep = float(np.abs(d["feet_distance"][...,1]).mean()); s = sep/2
    inner = s - SOLE_HALF_WIDTH
    M = W*SOLE_HALF_WIDTH              # residual with the CoP driven to the sole's inner edge
    geo[run] = (W, s, inner)
    print(f"{run:9}{m:>7.3f}{W:>8.2f}{sep:>8.4f}{s:>7.4f}{inner:>11.4f}{M:>17.2f}{100*M/17:>12.1f}%{s:>17.4f}")

print(f"\nsole half width {SOLE_HALF_WIDTH:.4f} m, so the stance sole spans [s-{SOLE_HALF_WIDTH:.4f}, s+{SOLE_HALF_WIDTH:.4f}]")
print("the torso must reach the inner edge for single support to be statically feasible,")
print("and the ankle roll moment falls to zero only when it reaches the ankle at s.\n")

# what the branch's 0.03 m amplitude buys
A = 0.03
print(f"{'run':9}{'branch A=0.030':>16}{'reaches inner edge?':>22}{'residual M_ankle':>18}{'pct of 17Nm':>13}")
for run,(W,s,inner) in geo.items():
    p = min(A, s+SOLE_HALF_WIDTH); p = max(p, s-SOLE_HALF_WIDTH) if A>=inner else None
    ok = "yes" if A >= inner else f"NO, short by {inner-A:.4f} m"
    Mres = W*abs(s-A)
    print(f"{run:9}{A:>16.3f}{ok:>22}{Mres:>18.2f}{100*Mres/17:>12.1f}%")
