import numpy as np, common
from frames import quat_rot
def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
for run in common.RUNS:
    d=common.load(run); jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    pos=d["joint_positions"]; vel=d["joint_velocities"]; dt=float(d["step_dt"])
    contact=np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    fq=d["feet_quaternions"]; by=yaw_of(d["base_quaternion"]); c,s=np.cos(-by),np.sin(-by)
    T,E = pos.shape[0], pos.shape[1]
    print(f"\n=== {run} ===")
    for fi,leg in ((0,"left"),(1,"right")):
        ar=pos[:,:,ix[f'{leg}_foot_roll_02']]; ap=pos[:,:,ix[f'{leg}_foot_pitch_02']]
        vr=vel[:,:,ix[f'{leg}_foot_roll_02']]; st=contact[:,:,fi]
        side=np.where(ar>0.9*0.2618,1,np.where(ar<-0.9*0.2618,-1,0))
        flips=0
        for e in range(E):
            seq=side[:,e]; seq=seq[seq!=0]
            if seq.size: flips += int((seq[1:]!=seq[:-1]).sum())
        secs=T*E*dt
        # ankle pitch: regress in STANCE where the joint has variance
        up=-quat_rot(fq[:,:,fi,:],np.array([0.,1.,0.]))
        ux=c*up[...,0]-s*up[...,1]; uz=up[...,2]; spitch=np.arctan2(ux,uz)
        A=np.polyfit(ap[st],spitch[st],1)
        need_sw = ap[~st] - spitch[~st]/A[0]     # ankle pitch that would level the sole, swing
        print(f" {leg:5} roll  : stop-to-stop reversals {flips/secs:6.3f}/s  (cadence ~2/s) | "
              f"stance at LOW stop {100*(ar[st]<-0.9*0.2618).mean():5.1f}%  swing at HIGH stop {100*(ar[~st]>0.9*0.2618).mean():5.1f}%")
        print(f"        traverse |v| p99 {np.percentile(np.abs(vr),99):5.2f} rad/s, max {np.abs(vr).max():5.2f}; "
              f"damping torque at p99 = {1.05*np.percentile(np.abs(vr),99):5.2f} Nm")
        print(f" {leg:5} pitch : stance slope {A[0]:+.3f}; ankle pitch to level sole in SWING "
              f"p50 {np.degrees(np.median(need_sw)):+7.2f} p90 {np.degrees(np.percentile(need_sw,10)):+7.2f} deg, "
              f"beyond -50 deg stop {100*(need_sw<-0.872665).mean():5.1f}%")
