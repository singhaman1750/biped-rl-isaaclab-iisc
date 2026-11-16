"""Termination quantities of pycma 4.4.4 reconstructed from cmaes_history.csv and the per generation tables."""
import sys, os, glob
import numpy as np, pandas as pd
RUNS = sys.argv[1:]
SPAN = {"thigh_length_scale": 0.5, "shank_length_scale": 0.5, "thigh_bend_offset": 0.2, "knee_bend_offset": 0.5}
H = 11  # fit.hist length at N=4, lambda=80
for run in RUNS:
    h = pd.read_csv(os.path.join(run, "designs", "cmaes_history.csv"))
    names = [c[4:] for c in h.columns if c.startswith("std/")]
    nstd = np.stack([h[f"std/{k}"] / SPAN[k] for k in names], 1)  # sigma*sqrt(C_ii) in the unit box
    best = h["fitness_max"].to_numpy()
    cur = (h["fitness_max"] - h["fitness_min"]).to_numpy()
    hist = np.array([np.ptp(best[max(0, i - H + 1):i + 1]) if i >= H - 1 else np.nan for i in range(len(h))])
    r = h["split_half_rank_correlation"].to_numpy()
    rf = np.clip(2 * r / (1 + r), 0, 1)
    noise = np.sqrt(1 - rf) * h["fitness_std"].to_numpy()
    iqr = np.array([np.subtract(*np.percentile(pd.read_csv(f)["fitness"], [75, 25]))
                    for f in [os.path.join(run, "designs", f"cmaes_generation_{g:04d}.csv") for g in h["generation"]]])
    t = pd.DataFrame(dict(gen=h["generation"], it=h["iteration"], sigma=h["sigma"].round(3),
                          f_mean=h["fitness_mean"].round(2), f_max=best.round(2), f_std=h["fitness_std"].round(3),
                          cur_range=cur.round(2), iqr=iqr.round(3), hist11=np.round(hist, 2),
                          max_nstd=nstd.max(1).round(4), min_nstd=nstd.min(1).round(4),
                          r_split=np.round(r, 2), noise=np.round(noise, 3)))
    print(f"\n==== {run}  ({len(h)} generations, params {names})")
    print(t.to_string(index=False))
    last = t.tail(10)
    print("last 10 gens, medians:", {k: float(np.nanmedian(last[k])) for k in ["cur_range", "iqr", "hist11", "max_nstd", "noise", "f_std"]})
    d = np.diff(h["fitness_mean"].to_numpy())
    print("mean fitness change per gen, last 10: median %.3f, mean %.3f" % (np.median(d[-10:]), d[-10:].mean()))
