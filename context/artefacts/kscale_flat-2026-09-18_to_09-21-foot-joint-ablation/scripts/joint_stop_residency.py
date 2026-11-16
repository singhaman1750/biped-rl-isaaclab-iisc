import numpy as np, common
JOINTS = ["left_hip_roll_03","right_hip_roll_03","left_foot_roll_02","right_foot_roll_02",
          "left_foot_pitch_02","right_foot_pitch_02","left_knee_04"]
for run in common.RUNS:
    d = common.load(run)
    jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    pos=d["joint_positions"]; hard=np.asarray(d["joint_position_limits"]); soft=np.asarray(d["joint_soft_position_limits"])
    dflt=np.asarray(d["default_joint_pos"])
    contact = np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    print(f"\n=== {run} ===")
    print(f"{'joint':20} {'hard lo':>8} {'hard hi':>8} {'default':>8} {'p1':>7} {'p50':>7} {'p99':>7} "
          f"{'at lo%':>7} {'at hi%':>7} {'stance p50':>10} {'swing p50':>9}")
    for n in JOINTS:
        j=ix[n]; q=pos[:,:,j]; lo,hi=hard[j]
        leg = 0 if n.startswith("left") else 1
        st = contact[:,:,leg]
        tol=0.01*(hi-lo)
        print(f"{n:20} {lo:8.4f} {hi:8.4f} {dflt[j]:8.4f} {np.percentile(q,1):7.4f} {np.percentile(q,50):7.4f} "
              f"{np.percentile(q,99):7.4f} {100*(q<lo+tol).mean():7.2f} {100*(q>hi-tol).mean():7.2f} "
              f"{np.median(q[st]):10.4f} {np.median(q[~st]):9.4f}")
