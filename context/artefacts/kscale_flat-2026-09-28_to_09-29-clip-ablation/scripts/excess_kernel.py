"""The l1 and l2 clip excess of run E, measured rather than inferred from the l2 sum."""
import os, re, sys
import numpy as np
sys.path.insert(0, "/workspace/isaaclab/biped/scripts/analysis")
import experiment_params as ep
EVAL = "/workspace/isaaclab/logs/rsl_rl/kscale_flat_eval"

def per_joint(params, key, names, default=np.nan):
    out = np.full(len(names), default, dtype=float)
    table = (params["actions"]["joint_pos"] or {}).get(key)
    if isinstance(table, dict):
        for expr, v in table.items():
            for j, n in enumerate(names):
                if re.fullmatch(expr, n):
                    out[j] = float(v)
    elif table is not None:
        out[:] = float(table)
    return out

def clip_bounds(params, names):
    lo = np.full(len(names), np.nan); hi = np.full(len(names), np.nan)
    table = (params["actions"]["joint_pos"] or {}).get("clip")
    if isinstance(table, dict):
        for expr, pair in table.items():
            for j, n in enumerate(names):
                if re.fullmatch(expr, n):
                    lo[j], hi[j] = float(pair[0]), float(pair[1])
    return lo, hi

for job in ["2026-09-29_06-15-58_model_8000", "2026-09-29_06-15-58_model_9000",
            "2026-09-29_06-15-58_model_10000"]:
    base = os.path.join(EVAL, job, "data", "42")
    d = np.load(os.path.join(base, "dump.npy"), allow_pickle=True).item()
    params = ep.load_params(os.path.join(EVAL, job))
    names = list(d["joint_names"])
    raw = np.asarray(d["joint_action_raw"])          # (T,E,J)
    cmd = np.asarray(d["joint_position_targets"])    # (T,E,J)
    scale = per_joint(params, "scale", names, 1.0)
    lo, hi = clip_bounds(params, names)
    # offset recovered from the samples the clip did NOT bind
    off = np.full(len(names), np.nan)
    for j in range(len(names)):
        if np.isnan(lo[j]):
            continue
        free = (cmd[:, :, j] > lo[j] + 1e-6) & (cmd[:, :, j] < hi[j] - 1e-6)
        if free.sum() < 100:
            continue
        off[j] = np.median((cmd[:, :, j] - raw[:, :, j] * scale[j])[free])
    unc = raw * scale + off
    exc = unc - cmd
    exc = np.where(np.isnan(exc), 0.0, exc)
    l2 = np.square(exc).sum(axis=-1)
    l1 = np.abs(exc).sum(axis=-1)
    tag = job.rsplit("_", 1)[-1]
    print(f"E@{tag}:  l2 sum {l2.mean():9.4f}   l1 sum {l1.mean():8.4f}   "
          f"ratio l2/l1 {l2.mean()/max(l1.mean(),1e-9):7.4f}")
    per2 = np.square(exc).mean(axis=(0, 1))
    per1 = np.abs(exc).mean(axis=(0, 1))
    if tag == "10000":
        for j, n in enumerate(names):
            if per1[j] > 1e-4:
                print(f"    {n:26s} l2 {per2[j]:9.4f}  l1 {per1[j]:8.4f}")
