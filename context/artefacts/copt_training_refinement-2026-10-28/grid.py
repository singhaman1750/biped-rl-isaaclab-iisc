import sys, os
import numpy as np, pandas as pd
SPAN = {"thigh_length_scale": 0.5, "shank_length_scale": 0.5, "thigh_bend_offset": 0.2, "knee_bend_offset": 0.5}
BASE = {"copt_moral": (0.30, 0.35), "quadruped_copt": (0.213, 0.213)}
H = 11
for run in sys.argv[1:]:
    h = pd.read_csv(os.path.join(run, "designs", "cmaes_history.csv"))
    names = list(SPAN)
    nstd = pd.DataFrame({k: h[f"std/{k}"] / SPAN[k] for k in names})
    lt, ls = BASE[run.split("/")[0]]
    distinct = []
    for g in h["generation"]:
        t = pd.read_csv(os.path.join(run, "designs", f"cmaes_generation_{g:04d}.csv"))
        distinct.append((len(set(np.round(lt * t.thigh_length_scale, 2))), len(set(np.round(ls * t.shank_length_scale, 2))),
                         len(set(zip(np.round(lt * t.thigh_length_scale, 2), np.round(ls * t.shank_length_scale, 2))))))
    best = h["fitness_max"].to_numpy()
    hist = np.array([np.ptp(best[i - H + 1:i + 1]) if i >= H - 1 else np.inf for i in range(len(h))])
    print(f"\n==== {run}")
    late = nstd.tail(8)
    print("late per coordinate normalised std (median of last 8):", late.median().round(4).to_dict())
    print("binding coordinate per gen (last 8):", late.idxmax(axis=1).tolist())
    print("distinct (thigh, shank, pairs) lengths per gen, last 8:", distinct[-8:])
    print("distinct per gen, first 4:", distinct[:4])
    r = h["split_half_rank_correlation"].to_numpy(); rf = np.clip(2 * r / (1 + r), 0, 1)
    noise = float(np.median((np.sqrt(1 - rf) * h["fitness_std"])[-10:]))
    print(f"late evaluation noise std {noise:.3f}")
    mx = nstd.max(axis=1).to_numpy()
    for tx in (0.02, 0.03, 0.04, 0.05):
        i = np.argmax(mx < tx) if (mx < tx).any() else None
        print(f"  tolx {tx}: first gen {None if i is None else int(h.generation[i])} (it {None if i is None else int(h.iteration[i])})")
    for tf in (0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.25):
        i = np.argmax(hist < tf) if (hist < tf).any() else None
        print(f"  tolfunhist {tf}: first gen {None if i is None else int(h.generation[i])} (it {None if i is None else int(h.iteration[i])})")
