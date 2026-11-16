import numpy as np, common
from frames import quat_rot
def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
for run in common.RUNS:
    d=common.load(run); jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    fq=d["feet_quaternions"]; bq=d["base_quaternion"]; pos=d["joint_positions"]
    contact=np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    by=yaw_of(bq); c,s=np.cos(-by),np.sin(-by)
    print(f"\n=== {run} ===")
    for fi,leg in ((0,"left"),(1,"right")):
        up=-quat_rot(fq[:,:,fi,:],np.array([0.,1.,0.]))
        ux=c*up[...,0]-s*up[...,1]; uy=s*up[...,0]+c*up[...,1]; uz=up[...,2]
        sroll=np.arctan2(uy,uz); spitch=np.arctan2(ux,uz)
        st=contact[:,:,fi]
        ar=pos[:,:,ix[f"{leg}_foot_roll_02"]]; ap=pos[:,:,ix[f"{leg}_foot_pitch_02"]]
        # slope of sole roll against ankle roll, stance only
        A=np.polyfit(ar[st], sroll[st], 1)
        need = ar[st] - sroll[st]/A[0]            # ankle angle that would zero the sole roll
        print(f" {leg:5} stance: d(soleroll)/d(ankleroll) = {A[0]:+.3f}")
        print(f"        ankle roll NEEDED to flatten sole [deg]: p50 {np.degrees(np.median(np.abs(need))):5.2f}  "
              f"p90 {np.degrees(np.percentile(np.abs(need),90)):5.2f}  p99 {np.degrees(np.percentile(np.abs(need),99)):5.2f}  "
              f"(available 15.00)")
        print(f"        fraction of stance needing >15 deg: {100*(np.abs(need)>0.2618).mean():5.1f} pct")
        print(f"        swing: sole pitch p50 {np.degrees(np.median(spitch[~st])):+6.2f} deg, ankle pitch p50 {np.degrees(np.median(ap[~st])):+6.2f} deg")
