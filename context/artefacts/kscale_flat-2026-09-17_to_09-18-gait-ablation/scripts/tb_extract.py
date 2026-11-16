"""Extract TensorBoard scalars for the four KScale runs and tabulate at matched iterations."""
from tensorboard.backend.event_processing import event_accumulator
import glob, numpy as np

BASE = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {
    "A 09-17_09-13": "2026-09-17_09-13-46",
    "B 09-17_11-25": "2026-09-17_11-25-59",
    "C 09-18_09-37": "2026-09-18_09-37-29",
    "D 09-18_10-03": "2026-09-18_10-03-48",
}

data = {}
for label, run in RUNS.items():
    p = glob.glob(f"{BASE}/{run}/events.out.tfevents.*")[0]
    ea = event_accumulator.EventAccumulator(p, size_guidance={"scalars": 0})
    ea.Reload()
    d = {}
    for t in ea.Tags()["scalars"]:
        ev = ea.Scalars(t)
        d[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    data[label] = d
    steps = d["Train/mean_reward"][0]
    print(f"{label}: {len(steps)} points, steps {steps.min()}..{steps.max()}")


def at(label, tag, it):
    """Value of tag at iteration `it`, averaged over a +/-50 iteration window."""
    if tag not in data[label]:
        return None
    s, v = data[label][tag]
    m = (s >= it - 50) & (s <= it + 50)
    if not m.any():
        if it > s.max():
            return None
        m = np.argmin(abs(s - it))
        return float(v[m])
    return float(v[m].mean())


ITERS = [1000, 3000, 6000, 9000, 15000, 20000, 29900]
TAGS = [
    "Train/mean_reward", "Train/mean_episode_length",
    "Metrics/base_velocity/error_vel_xy", "Metrics/base_velocity/error_vel_yaw",
    "Episode_Termination/base_contact", "Episode_Termination/time_out",
    "Curriculum/modify_command_velocity_lin_x", "Curriculum/modify_command_velocity_lin_y",
    "Curriculum/modify_command_velocity_ang_z",
    "Curriculum/modify_linear_tracking_reward_std", "Curriculum/modify_angular_tracking_reward_std",
    "Curriculum/modify_push_force",
    "Episode_Reward/pen_base_height", "Episode_Reward/pen_feet_heading",
    "Episode_Reward/pen_hip_yaw_deviation", "Episode_Reward/rew_keep_hip_yaw_zero_in_air",
    "Episode_Reward/pen_hip_roll_deviation",
    "Episode_Reward/rew_lin_vel_xy", "Episode_Reward/rew_ang_vel_z",
    "Episode_Reward/rew_gait", "Episode_Reward/rew_foot_clearance",
    "Episode_Reward/feet_air_time", "Episode_Reward/rew_no_fly",
    "Episode_Reward/pen_feet_distance", "Episode_Reward/pen_flat_orientation",
    "Episode_Reward/termination_penalty", "Episode_Reward/pen_joint_pos_limits",
    "Episode_Reward/rew_keep_ankle_pitch_zero_in_air", "Episode_Reward/rew_keep_ankle_roll_zero_in_air",
    "Episode_Reward/pen_feet_regulation", "Episode_Reward/feet_slide",
    "Episode_Reward/pen_foot_landing_vel", "Episode_Reward/pen_feet_impact",
    "Policy/mean_noise_std", "Loss/entropy", "Loss/symmetry",
]

for tag in TAGS:
    print()
    print(f"### {tag}")
    hdr = "  ".join(f"{i:>10d}" for i in ITERS)
    print(f"{'run':<16}{hdr}")
    for label in RUNS:
        cells = []
        for it in ITERS:
            v = at(label, tag, it)
            cells.append("        --" if v is None else f"{v:10.4f}")
        print(f"{label:<16}{'  '.join(cells)}")
