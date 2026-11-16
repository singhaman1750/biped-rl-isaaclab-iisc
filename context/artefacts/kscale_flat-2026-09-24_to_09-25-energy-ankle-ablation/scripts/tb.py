"""Training progression of the three runs, at a matched iteration and a matched sample budget.

The population is not matched across the set, A and B collecting 9000 parallel environments
and C collecting 6000, so one iteration of C carries 0.667 of the experience one iteration of
A or B carries. Both readings are printed, since a comparison at a matched iteration flatters
C and one at a matched sample count flatters A and B, and the honest statement needs both.
"""
import os, sys
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

ROOT = "/workspace/isaaclab/logs/rsl_rl/kscale_flat"
RUNS = {"A": "2026-09-24_05-18-33", "B": "2026-09-25_08-00-34", "C": "2026-09-25_11-29-39"}
ENVS = {"A": 9000, "B": 9000, "C": 6000}
STEPS_PER_ITER = 24

TAGS = ["Train/mean_reward", "Train/mean_episode_length",
        "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw",
        "Episode_Termination/base_contact", "Episode_Termination/time_out",
        "Policy/mean_noise_std", "Loss/entropy", "Loss/value_function", "Loss/surrogate",
        "Loss/symmetry", "Curriculum/modify_command_velocity_lin_x",
        "Episode_Reward/rew_lin_vel_xy", "Episode_Reward/rew_ang_vel_z",
        "Episode_Reward/pen_joint_torque", "Episode_Reward/pen_joint_accel",
        "Episode_Reward/pen_joint_vel_l2", "Episode_Reward/pen_joint_torque_rate",
        "Episode_Reward/pen_joint_power", "Episode_Reward/pen_ankle_roll_deviation",
        "Episode_Reward/rew_keep_ankle_roll_zero_in_air",
        "Episode_Reward/rew_keep_hip_yaw_zero_in_air",
        "Episode_Reward/rew_keep_ankle_pitch_zero_in_air",
        "Episode_Reward/pen_feet_heading", "Episode_Reward/pen_hip_yaw_deviation",
        "Episode_Reward/pen_joint_pos_limits", "Episode_Reward/rew_gait",
        "Episode_Reward/rew_hip_lateral_sway", "Episode_Reward/pen_feet_distance",
        "Episode_Reward/termination_penalty", "Perf/total_fps"]


def series(run):
    ea = event_accumulator.EventAccumulator(os.path.join(ROOT, RUNS[run]),
                                            size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for t in TAGS:
        if t in ea.Tags()["scalars"]:
            ev = ea.Scalars(t)
            out[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def at(s, tag, step):
    if tag not in s:
        return None
    x, y = s[tag]
    i = int(np.argmin(np.abs(x - step)))
    return float(y[i]) if abs(x[i] - step) <= 60 else None


def main():
    S = {r: series(r) for r in RUNS}
    print("### Recorded span")
    for r in RUNS:
        st = S[r]["Train/mean_reward"][0]
        print(f"  {r}  {RUNS[r]}  iterations {st.min()} to {st.max()}, {len(st)} points, "
              f"{ENVS[r]} envs, {ENVS[r]*STEPS_PER_ITER*(st.max()+1)/1e6:.0f} M env steps")

    marks = [2000, 4000, 6000, 8000, 9000, 10000, 12000, 14000]
    print("\n### At a matched ITERATION")
    for tag in TAGS:
        if not any(tag in S[r] for r in RUNS):
            continue
        print(f"\n  {tag}")
        print("    " + f"{'iter':>7s} " + " ".join(f"{r:>10s}" for r in RUNS))
        for m in marks:
            vals = [at(S[r], tag, m) for r in RUNS]
            if all(v is None for v in vals):
                continue
            print("    " + f"{m:>7d} " + " ".join(
                (f"{v:>10.4f}" if v is not None else f"{'-':>10s}") for v in vals))

    print("\n### At a matched SAMPLE BUDGET, in millions of environment steps")
    for tag in ["Train/mean_reward", "Train/mean_episode_length",
                "Metrics/base_velocity/error_vel_xy", "Policy/mean_noise_std",
                "Loss/entropy", "Episode_Termination/base_contact"]:
        if not any(tag in S[r] for r in RUNS):
            continue
        print(f"\n  {tag}")
        print("    " + f"{'M steps':>8s} " + " ".join(f"{r:>10s}" for r in RUNS))
        for budget in (500, 1000, 1500, 1750, 2000, 2500, 3000):
            cells = []
            for r in RUNS:
                it = budget * 1e6 / (ENVS[r] * STEPS_PER_ITER)
                cells.append(at(S[r], tag, it))
            if all(c is None for c in cells):
                continue
            print("    " + f"{budget:>8d} " + " ".join(
                (f"{c:>10.4f}" if c is not None else f"{'-':>10s}") for c in cells))


if __name__ == "__main__":
    main()
