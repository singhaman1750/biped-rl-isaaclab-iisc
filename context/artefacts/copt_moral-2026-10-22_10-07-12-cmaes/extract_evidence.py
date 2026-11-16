"""Per generation evidence for plans/copt_improvement.md section 2.1, from run copt_moral/2026-10-22_10-07-12.

Joins designs/cmaes_history.csv with the TensorBoard record, the tracking widths in force at
each evaluation, the kernel value at the evaluated mean error under those widths and under the
configured starting width of 0.4, and the measured tracking and remaining terms of the training
reward over the 400 iterations before each evaluation, rescaled to the 4 s horizon. Writes
evidence_by_generation.csv and timing_medians.txt beside itself. Run with the Isaac Sim python.

Usage:
    python extract_evidence.py <run_dir>
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
out = os.path.dirname(os.path.abspath(__file__))
ea = EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()


def series(tag):
    e = ea.Scalars(tag)
    return pd.Series([x.value for x in e], index=[x.step for x in e])


sl = series("Curriculum/modify_linear_tracking_reward_std")
sa = series("Curriculum/modify_angular_tracking_reward_std")
rl, ra = series("Episode_Reward/rew_lin_vel_xy"), series("Episode_Reward/rew_ang_vel_z")
mr, ml = series("Train/mean_reward"), series("Train/mean_episode_length")
start = (sl.iloc[0], sa.iloc[0])
rows = []
for _, r in pd.read_csv(os.path.join(run, "designs", "cmaes_history.csv")).iterrows():
    it = int(r.iteration)
    window = lambda v: v[(v.index > it - 400) & (v.index <= it)].mean()
    std_lin, std_ang = sl[sl.index <= it].iloc[-1], sa[sa.index <= it].iloc[-1]
    e_lin, e_yaw = r["eval/lin_vel_error"], r["eval/yaw_rate_error"]
    track = 4 * (window(rl) + window(ra))
    rows.append({
        "generation": int(r.generation), "iteration": it, "fitness_mean": r.fitness_mean,
        "survival": r["eval/survival"], "lin_vel_error": e_lin, "yaw_rate_error": e_yaw,
        "std_lin": std_lin, "std_ang": std_ang,
        "kernel_lin_current": math.exp(-e_lin**2 / std_lin**2), "kernel_lin_start": math.exp(-e_lin**2 / start[0] ** 2),
        "kernel_yaw_current": math.exp(-e_yaw**2 / std_ang**2), "kernel_yaw_start": math.exp(-e_yaw**2 / start[1] ** 2),
        "train_tracking_4s": track, "train_other_4s": window(mr) / window(ml) * 200 - track,
        "split_half": r.split_half_rank_correlation,
    })
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "evidence_by_generation.csv"), index=False)
with open(os.path.join(out, "timing_medians.txt"), "w") as f:
    for tag in ("Perf/collection time", "Perf/learning_time", "CMAES/eval_time", "CMAES/population_time"):
        v = np.array([x.value for x in ea.Scalars(tag) if x.step >= 6000])
        f.write(f"{tag}: median {np.median(v):.2f} s over {len(v)} records from iteration 6000\n")
    f.write(f"starting widths: linear {start[0]:.4f}, angular {start[1]:.4f}\n")
print(df.round(3).to_string(index=False))
