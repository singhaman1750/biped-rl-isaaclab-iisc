import numpy as np, os, sys, yaml
sys.path.insert(0,"/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
from experiment_params import load_params
ROOT="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
R={"A":"2026-09-21_06-45-40","B":"2026-09-22_07-18-05","C":"2026-09-22_07-40-43"}
D={k:np.load(os.path.join(ROOT,v,"data/42/dump.npy"),allow_pickle=True).item() for k,v in R.items()}
P={k:load_params(os.path.join(ROOT,v)) for k,v in R.items()}
J=D["A"]["joint_names"]
print("default_joint_pos as dumped, which carries the startup joint_offsets draw of +-0.05\n")
print(f"{'joint':24}"+ "".join(f"{k:>12}" for k in R))
for i,n in enumerate(J):
    print(f"{n:24}"+"".join(f"{D[k]['default_joint_pos'][i]:>12.5f}" for k in R))
print("\nheight related reward parameters")
for path in [("rewards","pen_base_height","params","target_height"),
             ("rewards","pen_feet_regulation","params","base_height_target"),
             ("rewards","pen_feet_regulation","params","height_decay_scale"),
             ("rewards","pen_base_height","weight"),
             ("rewards","pen_feet_regulation","weight"),
             ("rewards","pen_feet_distance","params","min_feet_distance"),
             ("rewards","pen_hip_roll_deviation","weight"),
             ("rewards","pen_joint_torque","weight"),
             ("rewards","pen_action_rate","weight"),
             ("rewards","pen_action_smoothness","weight"),
             ("rewards","pen_joint_torque_rate","weight"),
             ("rewards","pen_joint_accel","weight"),
             ("rewards","pen_joint_vel_l2","weight"),
             ("rewards","pen_joint_pos_limits","weight"),
             ]:
    vals=[]
    for k in R:
        o=P[k]
        try:
            for p in path: o=o[p]
        except Exception: o="<absent>"
        vals.append(o)
    print(f"{'.'.join(path):62}"+"".join(f"{str(v):>16}" for v in vals))
print("\nactuator ceilings")
for g in ("foot_roll","foot_pitch","hip_roll","hip_yaw","knee","hip_pitch"):
    row=[]
    for k in R:
        a=P[k]["scene"]["robot"]["actuators"][g]
        row.append(f"{a['effort_limit']}/{a['saturation_effort']}")
    print(f"{g:14}"+"".join(f"{v:>16}" for v in row))
print("\nclip on actions:", {k:P[k]["actions"]["joint_pos"].get("clip") for k in R})
print("episode length s:", {k:P[k]["episode_length_s"] for k in R})
print("decimation:", {k:P[k].get("decimation") for k in R})
