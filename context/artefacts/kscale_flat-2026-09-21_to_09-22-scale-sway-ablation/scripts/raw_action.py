"""The raw network output implied by the recovered joint target, against the action scale.

joint_actions.py applies target = default + scale * action, so a target recovered from the
torque divides back to the raw action the network emitted.  The question this answers is whether
the per joint action scale of arm 3 reduced the commanded target or whether the policy simply
widened its raw action to keep the same target, which is the premise arm 4b rests upon.
"""
import numpy as np, common
GAIN = {"hip_yaw":(100.,3.5),"hip_roll":(250.,28.),"hip_pitch":(200.,21.5),
        "knee":(300.,12.),"foot_roll":(100.,1.05),"foot_pitch":(100.,1.25)}
FRIC = {"hip_yaw":(0.2,0.1,0.02),"hip_roll":(0.2,0.1,0.02),"hip_pitch":(0.2,0.1,0.02),
        "knee":(0.2,0.1,0.02),"foot_roll":(0.1,0.1,0.02),"foot_pitch":(0.1,0.1,0.02)}
# actions.joint_pos.scale as each run dumped it to params/env.yaml
SCALE = {"A_scale0.4": {g:0.4 for g in GAIN},
         "B_arm3":     {"hip_pitch":0.4,"hip_yaw":0.4,"knee":0.2667,"hip_roll":0.16,
                        "foot_pitch":0.1619,"foot_roll":0.1619},
         "C_arm3+4":   {"hip_pitch":0.4,"hip_yaw":0.4,"knee":0.2667,"hip_roll":0.16,
                        "foot_pitch":0.1619,"foot_roll":0.1619}}
SAT = {"foot_roll":17.,"foot_pitch":17.,"hip_roll":60.,"hip_yaw":60.,"knee":120.,"hip_pitch":120.}
def group(n):
    for g in ("hip_yaw","hip_roll","hip_pitch","knee","foot_roll","foot_pitch"):
        if g in n: return g
JOINTS=["left_foot_roll_02","right_foot_roll_02","left_foot_pitch_02","right_foot_pitch_02",
        "left_hip_roll_03","right_hip_roll_03","left_hip_pitch_04","left_knee_04"]
print(f"{'run':12}{'joint':22}{'scale':>8}{'|a| p50':>9}{'|a| p99':>9}{'a p1':>9}{'a p99':>9}"
      f"{'tgt p50':>9}{'stop':>9}{'beyond%':>9}")
for run in common.RUNS:
    d=common.load(run); jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    tau=d["joint_torques"]; vel=d["joint_velocities"]; pos=d["joint_positions"]
    elim=np.asarray(d["joint_effort_limits"]); vlim=np.asarray(d["joint_velocity_limits"])
    hard=np.asarray(d["joint_position_limits"]); dflt=np.asarray(d["default_joint_pos"])
    for n in JOINTS:
        j=ix[n]; g=group(n); K,D=GAIN[g]; fs,av,fd=FRIC[g]; sc=SCALE[run][g]
        lo,hi = common.envelope(SAT[g], elim[j], vlim[j], vel[:,:,j])
        computed = tau[:,:,j] + fs*np.tanh(vel[:,:,j]/av)+fd*vel[:,:,j]
        un = (computed > lo*0.999) & (computed < hi*0.999)
        tgt = pos[:,:,j] + (computed + D*vel[:,:,j])/K
        act = (tgt - dflt[j])/sc
        qlo,qhi = hard[j]
        beyond = (tgt<qlo)|(tgt>qhi)
        a=act[un]; t=tgt[un]
        print(f"{run:12}{n:22}{sc:>8.4f}{np.percentile(np.abs(a),50):>9.3f}{np.percentile(np.abs(a),99):>9.3f}"
              f"{np.percentile(a,1):>9.3f}{np.percentile(a,99):>9.3f}{np.percentile(t,50):>9.4f}"
              f"{qlo if abs(np.percentile(t,50)-qlo)<abs(np.percentile(t,50)-qhi) else qhi:>9.4f}"
              f"{100*beyond[un].mean():>9.2f}")
    print()
