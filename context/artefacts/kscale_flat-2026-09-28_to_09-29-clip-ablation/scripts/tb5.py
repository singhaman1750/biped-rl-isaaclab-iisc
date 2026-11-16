"""Training progression of the five runs, at a matched iteration and a matched sample budget."""
import os, sys
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = "/workspace/isaaclab/logs/rsl_rl/kscale_flat"
RUNS = {"A": "2026-09-24_05-18-33", "B": "2026-09-25_08-00-34", "C": "2026-09-25_11-29-39",
        "D": "2026-09-28_10-25-57", "E": "2026-09-29_06-15-58",
        "F": "2026-09-30_06-43-38", "G": "2026-09-30_07-05-14"}
ENVS = {"A": 9000, "B": 9000, "C": 6000, "D": 9000, "E": 5000, "F": 5000, "G": 5000}
TAGS = ["Train/mean_reward", "Train/mean_episode_length",
        "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw",
        "Episode_Termination/base_contact", "Policy/mean_noise_std", "Loss/entropy",
        "Loss/symmetry", "Episode_Reward/rew_lin_vel_xy", "Episode_Reward/rew_ang_vel_z",
        "Episode_Reward/pen_action_excess", "Episode_Reward/pen_joint_power",
        "Episode_Reward/pen_joint_torque", "Episode_Reward/pen_joint_pos_limits",
        "Episode_Reward/rew_keep_hip_yaw_zero_in_air", "Episode_Reward/pen_feet_heading",
        "Episode_Reward/rew_keep_ankle_roll_zero_in_air",
        "Episode_Reward/rew_hip_lateral_sway", "Episode_Reward/pen_feet_distance",
        "Curriculum/modify_command_velocity_lin_x"]


def series(run):
    ea = event_accumulator.EventAccumulator(os.path.join(ROOT, RUNS[run]),
                                            size_guidance={"scalars": 0})
    ea.Reload()
    return {t: (np.array([e.step for e in ea.Scalars(t)]),
                np.array([e.value for e in ea.Scalars(t)]))
            for t in TAGS if t in ea.Tags()["scalars"]}


def at(s, tag, step, tol=60):
    if tag not in s:
        return None
    x, y = s[tag]
    i = int(np.argmin(np.abs(x - step)))
    return float(y[i]) if abs(x[i] - step) <= tol else None


S = {r: series(r) for r in RUNS}
print("### Recorded span")
for r in RUNS:
    st = S[r]["Train/mean_reward"][0]
    print(f"  {r} {RUNS[r]}  iterations {st.min()} to {st.max()}, {ENVS[r]} envs, "
          f"{ENVS[r]*24*(st.max()+1)/1e6:.0f} M env steps")

print("\n### At a matched ITERATION")
for tag in TAGS:
    if not any(tag in S[r] for r in RUNS):
        continue
    print(f"\n  {tag}")
    print("    " + f"{'iter':>6s} " + " ".join(f"{r:>10s}" for r in RUNS))
    for m in (1000, 2000, 4000, 6000, 8000, 9000, 10000, 11000):
        vals = [at(S[r], tag, m) for r in RUNS]
        if all(v is None for v in vals):
            continue
        print("    " + f"{m:>6d} " + " ".join(
            (f"{v:>10.4f}" if v is not None else f"{'-':>10s}") for v in vals))

print("\n### At a matched SAMPLE BUDGET [M env steps]")
for tag in ("Train/mean_reward", "Train/mean_episode_length",
            "Metrics/base_velocity/error_vel_xy", "Policy/mean_noise_std",
            "Loss/entropy", "Episode_Termination/base_contact"):
    if not any(tag in S[r] for r in RUNS):
        continue
    print(f"\n  {tag}")
    print("    " + f"{'M steps':>8s} " + " ".join(f"{r:>10s}" for r in RUNS))
    for budget in (250, 500, 1000, 1500, 2000):
        cells = [at(S[r], tag, budget * 1e6 / (ENVS[r] * 24), tol=200) for r in RUNS]
        if all(c is None for c in cells):
            continue
        print("    " + f"{budget:>8d} " + " ".join(
            (f"{c:>10.4f}" if c is not None else f"{'-':>10s}") for c in cells))
