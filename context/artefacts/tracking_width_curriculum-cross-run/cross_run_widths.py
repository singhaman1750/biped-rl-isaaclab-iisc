"""Cross run behaviour of reduce_tracking_rewards_std, for the survey of literature.md cluster 39.

For every run under <logs_root>/<experiment>/<run>/ whose params/env.yaml carries the curriculum
and whose event file is at least --min_mb, reads the configured weight, start width, floor,
threshold, interval and start of each tracking term, and from the event file the width, the
logged mean kernel (Episode_Reward/<term> divided by the weight, the quantity the gate compares
with its threshold) and the tracking error metric. Writes one row per run and term to
cross_run_widths.csv beside itself.

Usage:
    python cross_run_widths.py <logs_root> [--min_mb 20] [--workers 12]
"""

import argparse
import glob
import math
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util


class Loader(yaml.SafeLoader):
    pass


def _multi(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node) or suffix


Loader.add_multi_constructor("tag:yaml.org,2002:python/", _multi)

ERROR_TAG = {"lin": "Metrics/base_velocity/error_vel_xy", "ang": "Metrics/base_velocity/error_vel_yaw"}


def read_scalars(path, tags):
    out = {t: {} for t in tags}
    for event in EventFileLoader(path).Load():
        for v in event.summary.value:
            if v.tag in out:
                val = v.simple_value if v.HasField("simple_value") else float(tensor_util.make_ndarray(v.tensor))
                out[v.tag][event.step] = val
    return {t: pd.Series(d).sort_index() for t, d in out.items()}


def summarise_run(job):
    run, min_mb = job
    rows = []
    events = sorted(glob.glob(os.path.join(run, "events*")))
    params = os.path.join(run, "params", "env.yaml")
    if not events or not os.path.exists(params) or os.path.getsize(events[0]) < min_mb * 1e6:
        return rows
    cfg = yaml.load(open(params), Loader=Loader)
    terms = {
        name: t for name, t in (cfg.get("curriculum") or {}).items()
        if t and "reduce_tracking_rewards_std" in str(t.get("func", ""))
    }
    if not terms:
        return rows
    steps_per_it = None
    agent = os.path.join(run, "params", "agent.yaml")
    if os.path.exists(agent):
        steps_per_it = yaml.load(open(agent), Loader=Loader).get("num_steps_per_env")
    tags = [f"Curriculum/{n}" for n in terms] + [f"Episode_Reward/{t['params']['term_name']}" for t in terms.values()]
    tags += list(ERROR_TAG.values()) + ["Train/mean_episode_length"]
    s = read_scalars(events[0], tags)
    for name, t in terms.items():
        prm = t["params"]
        reward = cfg["rewards"][prm["term_name"]]
        weight, start_std = float(reward["weight"]), float(reward["params"]["std"])
        kind = "lin" if "lin" in prm["term_name"] else "ang"
        width, rew = s[f"Curriculum/{name}"], s[f"Episode_Reward/{prm['term_name']}"]
        if len(width) < 10 or len(rew) < 10:
            continue
        kernel = rew / weight
        err = s[ERROR_TAG[kind]]
        last = width.index.max()
        tail = lambda v: v[v.index > last - 500].mean() if len(v) else float("nan")
        drops = width.index[1:][np.diff(width.values) < -1e-6]
        below = [kernel[(kernel.index > it - 25) & (kernel.index <= it)].mean() < float(prm["update_threshold"]) for it in drops]
        floor = float(prm["minimum_std"])
        at_floor = width[width <= floor * 1.0001]
        e_mean = tail(err)
        rows.append({
            "experiment": os.path.basename(os.path.dirname(run)), "run": os.path.basename(run), "term": kind,
            "last_iteration": int(last), "steps_per_iteration": steps_per_it, "weight": weight,
            "start_width": start_std, "floor": floor, "threshold": float(prm["update_threshold"]),
            "rate": float(prm["update_rate"]), "interval_steps": int(prm["interval"]), "start_steps": int(prm.get("starting_step", 0)),
            "final_width": float(width.iloc[-1]), "narrowings": len(drops),
            "narrowings_with_mean_kernel_below_threshold": int(sum(below)),
            "first_narrowing_iteration": int(drops[0]) if len(drops) else None,
            "floor_reached_iteration": int(at_floor.index[0]) if len(at_floor) else None,
            "final_mean_kernel": tail(kernel), "final_mean_error": e_mean,
            "kungfubot_width_rms_estimate": e_mean * (math.sqrt(4 / math.pi) if kind == "lin" else math.sqrt(math.pi / 2)),
            "sensitivity_width_sqrt2_mean": e_mean * math.sqrt(2),
        })
    print(f"read {run}", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs_root")
    p.add_argument("--min_mb", type=float, default=20.0)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    runs = sorted(glob.glob(os.path.join(a.logs_root, "*", "*")))
    with Pool(a.workers) as pool:
        rows = [row for result in pool.map(summarise_run, [(r, a.min_mb) for r in runs]) for row in result]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cross_run_widths.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_rows", 500):
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
