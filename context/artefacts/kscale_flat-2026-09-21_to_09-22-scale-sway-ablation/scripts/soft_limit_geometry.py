"""Where the soft joint limit falls relative to the nominal pose, for each joint.

soft_joint_pos_limit_factor at kscale_identified_cfg.py scales each joint's travel about the
MIDPOINT of its range.  On a joint whose range is symmetric that leaves equal clearance either
side of a central nominal, and on a grossly asymmetric range it does not, the soft limit landing
close to the nominal on the short side.  pen_joint_pos_limits charges the excursion beyond that
limit, so a joint whose nominal sits near its own soft limit is charged for standing still.
"""
import numpy as np, common
d = common.load("A_scale0.4")
J = d["joint_names"]
hard = np.asarray(d["joint_position_limits"]); soft = np.asarray(d["joint_soft_position_limits"])
dflt = np.asarray(d["default_joint_pos"])
print(f"{'joint':22}{'hard lo':>9}{'hard hi':>9}{'soft lo':>9}{'soft hi':>9}{'nominal':>9}"
      f"{'to soft lo':>11}{'to soft hi':>11}{'asymmetry':>11}")
for i, n in enumerate(J):
    lo, hi = hard[i]; slo, shi = soft[i]
    print(f"{n:22}{lo:>9.4f}{hi:>9.4f}{slo:>9.4f}{shi:>9.4f}{dflt[i]:>9.4f}"
          f"{dflt[i]-slo:>11.4f}{shi-dflt[i]:>11.4f}{max(abs(lo),abs(hi))/max(min(abs(lo),abs(hi)),1e-9):>11.2f}")
print("\nthe quantity that matters is the smaller of the two clearances, in radians and in degrees")
for i, n in enumerate(J):
    slo, shi = soft[i]
    c = min(dflt[i]-slo, shi-dflt[i])
    print(f"  {n:22}{c:>9.4f} rad {np.degrees(c):>8.2f} deg")
