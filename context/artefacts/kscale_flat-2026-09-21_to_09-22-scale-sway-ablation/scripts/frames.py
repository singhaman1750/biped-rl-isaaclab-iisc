import numpy as np, common

def quat_rot(q, v):
    """Rotate v by WXYZ quaternion q. q (...,4), v (3,) -> (...,3)."""
    w,x,y,z = q[...,0],q[...,1],q[...,2],q[...,3]
    R = np.stack([
        1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y),
        2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1]+(3,3))
    return R @ v

d = common.load("A_scale0.4")
fq = d["feet_quaternions"]                       # (T,E,2,4) world
contact = np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
print("Which foot-link axis is vertical during stance (mean |dot| with world z):")
for ax,name in zip(np.eye(3), "xyz"):
    for fi,fn in enumerate(d["feet_names"]):
        w = quat_rot(fq[:,:,fi,:], ax)
        st = contact[:,:,fi]
        print(f"  axis {name} foot {fi}({fn}): mean dot_z = {w[...,2][st].mean():+.4f}  mean|dot_z| = {np.abs(w[...,2])[st].mean():.4f}")
