import numpy as np, common
from frames import quat_rot
def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
for run in common.RUNS:
    d=common.load(run); by=yaw_of(d["base_quaternion"])
    contact=np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    rels=[]
    for fi in (0,1):
        best=None
        for ax,lab in ((np.array([0.,0.,1.]),"+z"),(np.array([0.,0.,-1.]),"-z")):
            f=quat_rot(d["feet_quaternions"][:,:,fi,:],ax)
            fy=np.arctan2(f[...,1],f[...,0])
            rel=np.arctan2(np.sin(fy-by),np.cos(fy-by))
            if best is None or np.abs(rel).mean()<np.abs(best[1]).mean(): best=(lab,rel)
        rels.append(best); print(f"  {run} foot {fi}: forward axis {best[0]}, |rel| mean {np.degrees(np.abs(best[1]).mean()):.2f} deg")
    # toe-in positive = converging on midline. foot0=left: yaw<0 is inward. foot1=right: yaw>0 is inward.
    ti=np.stack([-rels[0][1], rels[1][1]],-1)
    tid=np.degrees(ti)
    st=np.stack([contact[:,:,0],contact[:,:,1]],-1)
    print(f"  => {run}: toe-in mean {tid.mean():6.2f} deg, p95 {np.percentile(tid,95):6.2f}, "
          f"stance L {tid[...,0][st[...,0]].mean():+6.2f} R {tid[...,1][st[...,1]].mean():+6.2f}\n")
