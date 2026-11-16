"""Which lateral edge of the sole is lower in stance, by transforming the sole corners.

The foot link frame carries the sole width on x, so the two lateral edges are at +-w/2 along
the link x axis.  Transforming both into the world and comparing their heights says which one
carries the contact, with no sign reasoning required.
"""
import numpy as np, common
from frames import quat_rot
W_HALF = 0.0846/2
def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
for run in common.RUNS:
    d = common.load(run)
    fq = d["feet_quaternions"]; by = yaw_of(d["base_quaternion"])
    contact = np.linalg.norm(d["feet_contact_forces"],axis=-1) > 1.0
    c,s = np.cos(-by), np.sin(-by)
    print(f"\n=== {run} ===")
    for fi, leg in ((0,"left"),(1,"right")):
        q = fq[:,:,fi,:]
        # the two lateral sole corners, expressed as offsets along the link x axis
        ex = quat_rot(q, np.array([1.0,0.0,0.0]))          # link x in world
        # lateral direction of link x in the base yaw frame, + is to the robot's left
        exy = s*ex[...,0] + c*ex[...,1]
        st = contact[:,:,fi]
        # height of the corner at +w/2 x relative to the corner at -w/2 x
        dz = 2*W_HALF*ex[...,2]
        # a corner is OUTER when its lateral offset points away from the midline.
        # left foot: outer is +y. right foot: outer is -y.
        sgn = +1.0 if leg == "left" else -1.0
        outer_is_plus_x = (sgn*exy > 0)
        # height of the outer corner minus the inner corner
        outer_minus_inner = np.where(outer_is_plus_x, dz, -dz)
        low = outer_minus_inner[st]
        print(f" {leg:5} stance: outer corner minus inner corner height = {low.mean()*1000:+6.2f} mm "
              f"({100*(low<0).mean():5.1f} pct of stance samples have the OUTER corner lower)")
