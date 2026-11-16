"""Does a URDF fixed joint keep its child link as a body, and does it leave joint_names?

The question decides the whole shape of the hip yaw fixed arm, and the SD_BRS1 answers it
without a simulator because that robot ALREADY declares both of its hip yaws type="fixed" at
SD_BRS1_Assembly2.urdf:89 and :431 while its configuration carries merge_fixed_joints=False at
sd_brs1_identified_cfg.py:161, which is the identical combination the arm proposes for the
KScale.  Its evaluation dump therefore records what the importer and PhysX actually produced.
"""
import os, sys
import numpy as np

LOGS = "/workspace/isaaclab/logs/rsl_rl"

for robot, run in (("sd_brs1_flat", "2026-08-17_04-26-44"),
                   ("kscale_flat", "2026-09-29_06-15-58")):
    path = os.path.join(LOGS, robot, run, "data", "42", "dump.npy")
    if not os.path.exists(path):
        print(f"{robot}/{run}: no dump")
        continue
    d = np.load(path, allow_pickle=True).item()
    jn = list(d.get("joint_names", []))
    mass = np.asarray(d.get("robot_mass")) if "robot_mass" in d else None
    print(f"=== {robot}/{run}")
    print(f"  joint_names  n={len(jn)}  {jn}")
    if mass is not None:
        print(f"  robot_mass shape {mass.shape}  -> bodies = {mass.shape[-1]}"
              f"  total {mass[0, 0].sum():.4f} kg")
    for k in ("feet_quaternions", "feet_contact_forces", "joint_positions"):
        if k in d:
            print(f"  {k:22s} {np.asarray(d[k]).shape}")
