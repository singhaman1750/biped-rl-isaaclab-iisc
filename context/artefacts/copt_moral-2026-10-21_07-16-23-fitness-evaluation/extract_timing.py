"""Medians of the timing and training scalars of a run's event file, before and after the CMA-ES hand-over.

Usage: /isaac-sim/python.sh extract_timing.py <run_dir> <late_start> <out_csv>
"""
import csv, statistics, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run, late, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ea = EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()
tags = ["Perf/collection time", "Perf/learning_time", "Perf/total_fps", "Train/mean_episode_length",
        "Train/mean_reward", "Curriculum/terrain_levels", "Metrics/base_velocity/error_vel_xy",
        "Curriculum/modify_command_velocity_lin_x", "Curriculum/modify_command_velocity_lin_y",
        "Curriculum/modify_command_velocity_ang_z", "Curriculum/modify_push_force",
        "Curriculum/modify_linear_tracking_reward_std", "Curriculum/modify_angular_tracking_reward_std"]
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["tag", "iterations_logged", "last_iteration", "median_before_late_start", "median_from_late_start",
                "value_at_2000", "value_at_late_start"])
    for t in tags:
        s = ea.Scalars(t)
        w.writerow([t, len(s), s[-1].step, statistics.median(e.value for e in s if e.step < late),
                    statistics.median(e.value for e in s if e.step >= late),
                    next(e.value for e in s if e.step >= 2000), next(e.value for e in s if e.step >= late)])
