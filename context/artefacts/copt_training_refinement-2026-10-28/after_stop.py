import sys, os
import numpy as np, pandas as pd
SPAN = {"thigh_length_scale": 0.5, "shank_length_scale": 0.5, "thigh_bend_offset": 0.2, "knee_bend_offset": 0.5}
BASE = {"copt_moral": (0.30, 0.35), "quadruped_copt": (0.213, 0.213)}
# CMA iteration of the first checkpoint verdict of verify_stop.py under the per-robot tolerances, mapped to its history row
STOP = {"copt_moral/2026-10-23_08-31-17": 25, "copt_moral/2026-10-26_04-55-55": 32,
        "quadruped_copt/2026-10-23_10-36-49": 25, "quadruped_copt/2026-10-26_04-55-55": 15}
for run, ci in STOP.items():
    h = pd.read_csv(os.path.join(run, "designs", "cmaes_history.csv"))
    lt, ls = BASE[run.split("/")[0]]
    row = h[h.cma_iteration == ci]
    if row.empty:
        row = h.tail(1)
    i = row.index[0]
    a, z = h.loc[i], h.iloc[-1]
    print(f"\n==== {run}: stop at cma_it {int(a.cma_iteration)} gen {int(a.generation)} it {int(a.iteration)}, last logged gen {int(z.generation)} it {int(z.iteration)}")
    for k in SPAN:
        print(f"  {k:20s} mean at stop {a['mean/' + k]:+.4f}  last {z['mean/' + k]:+.4f}  shift {abs(z['mean/' + k] - a['mean/' + k]) / SPAN[k]:.4f} of range")
    for k, L in (("thigh_length_scale", lt), ("shank_length_scale", ls)):
        print(f"  {k:20s} emitted length at stop {round(L * a['mean/' + k], 2):.2f} m, last {round(L * z['mean/' + k], 2):.2f} m")
    print(f"  best fitness of gen at stop {a.fitness_max:.2f}, at last {z.fitness_max:.2f}, mean {a.fitness_mean:.2f} -> {z.fitness_mean:.2f}")
    t = pd.read_csv(os.path.join(run, "designs", f"cmaes_generation_{int(a.generation):04d}.csv"))
    pairs = set(zip(np.round(lt * t.thigh_length_scale, 2), np.round(ls * t.shank_length_scale, 2)))
    print(f"  distinct (thigh, shank) length pairs in the stopping generation: {len(pairs)}")
