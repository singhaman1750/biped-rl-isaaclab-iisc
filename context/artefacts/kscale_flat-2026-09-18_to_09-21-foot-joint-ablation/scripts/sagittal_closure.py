import numpy as np, common
from frames import quat_rot
def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
# verify the closure identity on the NOMINAL pose first
print("nominal pose check (kscale_identified_cfg.py:197):")
for leg,hp,kn,fp,s in (("left",0.20369,0.51239,-0.3087,-1),("right",-0.20369,0.51239,-0.3087,+1)):
    print(f"  {leg}: s*hip_pitch + knee + foot_pitch = {s*hp+kn+fp:+.6f}")
for run in common.RUNS:
    d=common.load(run); jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    pos=d["joint_positions"]; contact=np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    fq=d["feet_quaternions"]; by=yaw_of(d["base_quaternion"]); c,s_=np.cos(-by),np.sin(-by)
    print(f"\n=== {run} ===")
    for fi,leg,sg in ((0,"left",-1),(1,"right",+1)):
        hp=pos[:,:,ix[f"{leg}_hip_pitch_04"]]; kn=pos[:,:,ix[f"{leg}_knee_04"]]; fp=pos[:,:,ix[f"{leg}_foot_pitch_02"]]
        clo = sg*hp + kn + fp
        up=-quat_rot(fq[:,:,fi,:],np.array([0.,1.,0.]))
        ux=c*up[...,0]-s_*up[...,1]; uz=up[...,2]; spitch=np.arctan2(ux,uz)
        st=contact[:,:,fi]
        r=np.corrcoef(clo.ravel(), spitch.ravel())[0,1]
        A=np.polyfit(clo.ravel(), spitch.ravel(),1)
        # ankle pitch that would zero the closure, i.e. sole parallel to torso
        need = -(sg*hp + kn)
        print(f" {leg:5} corr(closure, sole pitch) = {r:+.3f}, slope {A[0]:+.3f}")
        print(f"       closure [deg]  stance p50 {np.degrees(np.median(clo[st])):+6.2f}  swing p50 {np.degrees(np.median(clo[~st])):+6.2f}")
        print(f"       ankle pitch REQUIRED for closure [deg]: stance p50 {np.degrees(np.median(need[st])):+7.2f}  "
              f"swing p50 {np.degrees(np.median(need[~st])):+7.2f}   (travel -50.0 .. +30.0)")
        print(f"       required beyond the -50 deg stop: stance {100*(need[st]<-0.872665).mean():5.1f}%  swing {100*(need[~st]<-0.872665).mean():5.1f}%")
