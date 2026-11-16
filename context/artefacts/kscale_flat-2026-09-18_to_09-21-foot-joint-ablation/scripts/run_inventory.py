import sys, os
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
from experiment_params import load_params

ROOT = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = ["2026-09-17_09-13-46","2026-09-17_11-25-59","2026-09-18_05-14-16","2026-09-18_09-24-17",
        "2026-09-18_09-37-29","2026-09-18_10-03-48","2026-09-21_06-45-40","2026-09-21_08-37-13",
        "2026-09-21_08-40-59"]

hdr = f"{'run':24} {'roll tau':>9} {'roll sat':>9} {'pitch tau':>10} {'pitch sat':>10} {'roll K':>7} {'roll D':>7} {'pitch K':>8} {'pitch D':>8} {'envs':>6} {'ascale':>7}"
print(hdr); print("-"*len(hdr))
for r in RUNS:
    p = load_params(os.path.join(ROOT, r))
    if p is None:
        print(f"{r:24} NO PARAMS"); continue
    a = p["scene"]["robot"]["actuators"]
    fr, fp = a["foot_roll"], a["foot_pitch"]
    act = p["actions"]["joint_pos"]
    print(f"{r:24} {fr['effort_limit']:>9} {fr['saturation_effort']:>9} {fp['effort_limit']:>10} {fp['saturation_effort']:>10} "
          f"{list(fr['stiffness'].values())[0]:>7} {list(fr['damping'].values())[0]:>7} "
          f"{list(fp['stiffness'].values())[0]:>8} {list(fp['damping'].values())[0]:>8} "
          f"{p['scene']['num_envs']:>6} {act['scale']:>7}")
