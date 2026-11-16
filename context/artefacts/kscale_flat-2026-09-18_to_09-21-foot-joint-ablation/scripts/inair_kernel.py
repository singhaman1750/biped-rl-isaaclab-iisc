import numpy as np, common
for run in common.RUNS:
    d=common.load(run); jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    pos=d["joint_positions"]; dflt=np.asarray(d["default_joint_pos"])
    contact=np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    print(f"\n=== {run} ===   kernel exp(-|dq|/0.2), airborne feet only")
    for grp,w in (("foot_pitch",1.0),("foot_roll",0.25),("hip_yaw",1.0)):
        jl=[ix[f"left_{grp}_02" if "foot" in grp else "left_hip_yaw_03"],
            ix[f"right_{grp}_02" if "foot" in grp else "right_hip_yaw_03"]]
        dq=np.abs(pos[:,:,jl]-dflt[jl])
        air=~contact
        s=(dq*air).sum(-1)
        rew=np.exp(-s/0.2)*np.any(air,axis=-1)
        # gradient of the weighted rate with respect to the deviation of one airborne joint
        grad=w*(-1/0.2)*np.exp(-s/0.2)*np.any(air,axis=-1)
        swing_dq = dq[:,:,0][air[:,:,0]]
        print(f"  {grp:11} |dq| swing p50 {np.median(swing_dq):6.3f} rad | term {rew.mean():6.4f} of max 1 | "
              f"weighted rate {w*rew.mean():6.4f}/s | d(rate)/d(dq) {np.abs(grad).mean():7.4f} per rad")
    print(f"   for scale, rew_lin_vel_xy is 46.7 to 48.7 per second")
