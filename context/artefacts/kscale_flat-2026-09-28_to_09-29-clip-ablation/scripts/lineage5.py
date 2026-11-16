import sys, os
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
import experiment_params as ep
ROOT="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS={"A":"2026-09-24_05-18-33","B":"2026-09-25_08-00-34","C":"2026-09-25_11-29-39",
      "D":"2026-09-28_10-25-57","E":"2026-09-29_06-15-58"}
KEYS=["pen_joint_torque","pen_joint_accel","pen_joint_vel_l2","pen_joint_torque_rate",
      "pen_joint_power","pen_ankle_roll_deviation","rew_keep_ankle_roll_zero_in_air",
      "rew_keep_hip_yaw_zero_in_air","rew_keep_ankle_pitch_zero_in_air",
      "pen_feet_heading","pen_hip_yaw_deviation","pen_hip_roll_deviation",
      "pen_joint_pos_limits","pen_base_height","pen_action_rate","pen_action_smoothness",
      "pen_clip_excess","rew_hip_lateral_sway","pen_feet_distance"]
print(f"{'term':34s} " + "  ".join(f"{k:>12s}" for k in RUNS))
for k in KEYS:
    cells=[]
    for t,r in RUNS.items():
        p=ep.load_params(os.path.join(ROOT,r)); rw=p.get("rewards",{})
        term=rw.get(k)
        cells.append("-" if term is None else f"{term.get('weight'):g}")
    if any(c!="-" for c in cells):
        print(f"{k:34s} " + "  ".join(f"{c:>12s}" for c in cells))
print()
for t,r in RUNS.items():
    p=ep.load_params(os.path.join(ROOT,r))
    a=p["actions"]["joint_pos"]
    print(f"{t} {r}  envs={p['scene']['num_envs']}")
    print(f"    scale={a.get('scale')}")
    print(f"    clip ={a.get('clip')}")
