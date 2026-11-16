import numpy as np, common
from frames import quat_rot

def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))

for run in common.RUNS:
    d = common.load(run)
    jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    fq=d["feet_quaternions"]; bq=d["base_quaternion"]; pos=d["joint_positions"]
    contact=np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    by = yaw_of(bq)                                        # (T,E)
    print(f"\n=== {run} ===")
    for fi,leg in ((0,"left"),(1,"right")):
        up = -quat_rot(fq[:,:,fi,:], np.array([0.,1.,0.]))  # sole normal, world
        # express in base-yaw frame
        c,s = np.cos(-by), np.sin(-by)
        ux = c*up[...,0]-s*up[...,1]; uy = s*up[...,0]+c*up[...,1]; uz=up[...,2]
        sole_roll  = np.degrees(np.arctan2(uy, uz))        # lateral tilt, + = tips left
        sole_pitch = np.degrees(np.arctan2(ux, uz))        # fore-aft tilt
        st = contact[:,:,fi]
        ar = pos[:,:,ix[f"{leg}_foot_roll_02"]]
        ap = pos[:,:,ix[f"{leg}_foot_pitch_02"]]
        tilt = np.degrees(np.arccos(np.clip(uz,-1,1)))
        print(f" {leg:5} stance: sole tilt |{tilt[st].mean():5.2f}| deg, roll {sole_roll[st].mean():+6.2f}+-{sole_roll[st].std():4.2f}, "
              f"pitch {sole_pitch[st].mean():+6.2f}+-{sole_pitch[st].std():4.2f}")
        print(f"       ankle_roll(deg) stance {np.degrees(ar[st]).mean():+6.2f}  swing {np.degrees(ar[~st]).mean():+6.2f} | "
              f"ankle_pitch stance {np.degrees(ap[st]).mean():+6.2f} swing {np.degrees(ap[~st]).mean():+6.2f}")
        print(f"       corr(ankle_roll, sole_roll) in stance = {np.corrcoef(ar[st], sole_roll[st])[0,1]:+.3f}")
