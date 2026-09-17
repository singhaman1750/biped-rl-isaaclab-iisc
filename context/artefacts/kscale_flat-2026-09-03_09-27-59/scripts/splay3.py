import numpy as np
BASE="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS={"0828 calc 0.24/-100/-4":"2026-08-28_04-50-51",
      "0902a calc 0.14/-30/-2":"2026-09-02_06-59-31",
      "0902b calc 0.14 +yawlim":"2026-09-02_10-51-29",
      "arm0 ident as0.4":"2026-09-03_09-27-59",
      "n1015 ident as0.25":"2026-09-04_10-15-18"}
def R(q):
    w,x,y,z=[q[...,i] for i in range(4)]
    return np.stack([
      np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],-1),
      np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],-1),
      np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],-1)],-2)
def yawq(q):
    w,x,y,z=[q[...,i] for i in range(4)]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
rows={}
for name,r in RUNS.items():
    d=np.load(f"{BASE}/{r}/data/42/dump.npy",allow_pickle=True).item()
    jn=d['joint_names']; jp=d['joint_positions']
    fq=d['feet_quaternions']                     # (T,E,2,4) world, [0]=left foot_6061_2
    Rf=R(fq)                                     # (T,E,2,3,3)
    toe=-Rf[...,:,2]                             # foot -z in world  (T,E,2,3)
    toeyaw=np.arctan2(toe[...,1],toe[...,0])
    by=yawq(d['base_quaternion'])                # (T,E)
    sp=wrap(toeyaw-by[...,None])                 # (T,E,2) toe heading rel base
    fn=np.linalg.norm(d['feet_contact_forces'],axis=-1); stance=fn>1.0
    L,Rt=0,1
    # toe-out positive: left +yaw, right -yaw
    outL=sp[...,L]; outR=-sp[...,Rt]
    mL,mR=stance[...,L],stance[...,Rt]
    both=np.concatenate([outL[mL],outR[mR]])
    allb=np.concatenate([outL.ravel(),outR.ravel()])
    hyL=jp[:,:,jn.index('left_hip_yaw_03')]; hyR=jp[:,:,jn.index('right_hip_yaw_03')]
    dg=np.rad2deg
    rows[name]=dict(
      toeout_mean_deg=dg(both.mean()), toeout_med_deg=dg(np.median(both)),
      toeout_p5=dg(np.percentile(both,5)), toeout_p95=dg(np.percentile(both,95)),
      L_mean=dg(outL[mL].mean()), R_mean=dg(outR[mR].mean()),
      toein_frac=float((both<0).mean()), toein_gt5deg=float((both<-np.deg2rad(5)).mean()),
      toein_gt10deg=float((both<-np.deg2rad(10)).mean()),
      absmed_deg=dg(np.median(np.abs(both))),
      asym_deg=dg(abs(outL[mL].mean()+outR[mR].mean())),
      allphase_mean_deg=dg(allb.mean()),
      hipyawL_deg=dg(hyL.mean()), hipyawR_deg=dg(hyR.mean()),
      hipyaw_conv_deg=dg((hyR-hyL).mean()/2),
      stance_frac=float(stance.mean()),
      feet_y_med=float(np.median(np.abs(d['feet_distance'][...,1]))),
      term_frac=float(d['episode_terminated'].any(0).mean()),
    )
keys=list(next(iter(rows.values())).keys())
print(f"{'metric':20s}"+"".join(f"{n:>26s}" for n in rows))
for k in keys:
    print(f"{k:20s}"+"".join(f"{rows[n][k]:26.4f}" for n in rows))
