import numpy as np, common
# actuator gains and friction, kscale_identified_cfg.py:149-179 and :85-147
GAIN = {"hip_yaw":(100.,3.5,0.2,0.1,0.02),"hip_roll":(250.,28.,0.2,0.1,0.02),
        "hip_pitch":(200.,21.5,0.2,0.1,0.02),"knee":(300.,12.,0.2,0.1,0.02),
        "foot_roll":(100.,1.05,0.1,0.1,0.02),"foot_pitch":(100.,1.25,0.1,0.1,0.02)}
SAT={"tau11.9":17.,"tau17":17.,"tau34":34.}
def group(n):
    for g in ("hip_yaw","hip_roll","hip_pitch","knee","foot_roll","foot_pitch"):
        if g in n: return g
JOINTS=["left_foot_roll_02","right_foot_roll_02","left_foot_pitch_02","right_foot_pitch_02",
        "left_hip_roll_03","right_hip_roll_03"]
for run in common.RUNS:
    d=common.load(run); jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    tau=d["joint_torques"]; vel=d["joint_velocities"]; pos=d["joint_positions"]
    elim=np.asarray(d["joint_effort_limits"]); vlim=np.asarray(d["joint_velocity_limits"])
    hard=np.asarray(d["joint_position_limits"])
    contact=np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    print(f"\n=== {run} ===")
    print(f"{'joint':20} {'unclip%':>8} {'tgt p1':>8} {'tgt p50':>8} {'tgt p99':>8} {'beyond stop%':>12} "
          f"{'overshoot p50':>13} {'st beyond%':>10} {'sw beyond%':>10}")
    for n in JOINTS:
        j=ix[n]; g=group(n); K,D,fs,av,fd=GAIN[g]
        sat = SAT[run] if "foot" in n else (120. if ("hip_pitch" in n or "knee" in n) else 60.)
        lo,hi = common.envelope(sat, elim[j], vlim[j], vel[:,:,j])
        fr = fs*np.tanh(vel[:,:,j]/av)+fd*vel[:,:,j]
        computed = tau[:,:,j] + fr
        un = (computed > lo*0.999) & (computed < hi*0.999)      # strictly inside the envelope
        tgt = pos[:,:,j] + (computed + D*vel[:,:,j])/K
        qlo,qhi = hard[j]
        beyond = (tgt<qlo)|(tgt>qhi)
        over = np.where(tgt<qlo, qlo-tgt, np.where(tgt>qhi, tgt-qhi, 0.0))
        leg = 0 if n.startswith("left") else 1; st=contact[:,:,leg]
        if un.sum()==0: print(f"{n:20} {0.0:8.2f}  (never unsaturated)"); continue
        print(f"{n:20} {100*un.mean():8.2f} {np.percentile(tgt[un],1):8.4f} {np.percentile(tgt[un],50):8.4f} "
              f"{np.percentile(tgt[un],99):8.4f} {100*beyond[un].mean():12.2f} {np.median(over[un&beyond]) if (un&beyond).any() else 0:13.4f} "
              f"{100*beyond[un&st].mean() if (un&st).any() else float('nan'):10.2f} {100*beyond[un&~st].mean() if (un&~st).any() else float('nan'):10.2f}")
