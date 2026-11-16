import sys, os
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
import experiment_params as ep
ROOT="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
KEYS=["pen_joint_torque","pen_joint_accel","pen_joint_vel_l2","pen_joint_torque_rate",
      "pen_joint_power","pen_ankle_roll_deviation","rew_keep_ankle_roll_zero_in_air",
      "rew_keep_hip_yaw_zero_in_air","pen_feet_heading","pen_hip_yaw_deviation",
      "pen_base_height","pen_action_rate","pen_action_smoothness"]
runs=sorted(os.listdir(ROOT))
hdr=f"{'run':22s} envs  "+"  ".join(f"{k[:11]:>11s}" for k in KEYS)
print(hdr)
for r in runs:
    if r < "2026-09-21": continue
    p=ep.load_params(os.path.join(ROOT,r))
    if not p: 
        print(f"{r:22s} NO PARAMS"); continue
    rw=p.get("rewards",{})
    n=p.get("scene",{}).get("num_envs")
    cells=[]
    for k in KEYS:
        t=rw.get(k)
        cells.append("-" if t is None else f"{t.get('weight'):g}")
    print(f"{r:22s} {str(n):5s} "+"  ".join(f"{c:>11s}" for c in cells))
