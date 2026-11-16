"""What the hip yaw does, and how much of the measured toe-in it accounts for.

The proposal to convert the hip yaw to a fixed joint needs to know two things the dump can answer,
whether the joint is in any distress and whether it is the joint that sets the foot yaw.  If the
toe-in survives a hip yaw held at zero then fixing the joint removes an actuator without removing
the fault, and the fault would then have to live in the hip roll and the ankle roll instead.
"""
import numpy as np, common
from frames import quat_rot
def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
print(f"{'run':12}{'leg':6}{'hy p50':>9}{'hy p1':>9}{'hy p99':>9}{'|hy| mean':>11}"
      f"{'toe-in stance':>15}{'corr(hy,toe)':>14}{'toe at hy=0':>13}{'slope':>8}")
for run in common.RUNS:
    d = common.load(run); J = list(d["joint_names"])
    by = yaw_of(d["base_quaternion"])
    contact = np.linalg.norm(d["feet_contact_forces"],axis=-1) > 1.0
    for fi, leg in ((0,"left"),(1,"right")):
        hy = d["joint_positions"][:,:,J.index(f"{leg}_hip_yaw_03")]
        f = quat_rot(d["feet_quaternions"][:,:,fi,:], np.array([0.,0.,-1.]))
        fy = np.arctan2(f[...,1], f[...,0])
        rel = np.arctan2(np.sin(fy-by), np.cos(fy-by))
        toe = -rel if leg == "left" else rel          # positive is toe-in
        st = contact[:,:,fi]
        a = np.polyfit(hy[st], toe[st], 1)
        r = np.corrcoef(hy[st], toe[st])[0,1]
        print(f"{run:12}{leg:6}{np.degrees(np.percentile(hy,50)):>9.2f}{np.degrees(np.percentile(hy,1)):>9.2f}"
              f"{np.degrees(np.percentile(hy,99)):>9.2f}{np.degrees(np.abs(hy).mean()):>11.2f}"
              f"{np.degrees(toe[st].mean()):>15.2f}{r:>14.3f}{np.degrees(a[1]):>13.2f}{a[0]:>8.3f}")
    print()
print("hip yaw torque and effort, and the deviation term that already prices it")
for run in common.RUNS:
    d = common.load(run); J = list(d["joint_names"])
    e = np.asarray(d["joint_effort_limits"])
    for leg in ("left","right"):
        i = J.index(f"{leg}_hip_yaw_03"); t = np.abs(d["joint_torques"][:,:,i])
        print(f"  {run:12}{leg:6} mean |tau| {t.mean():6.2f} Nm, p99 {np.percentile(t,99):6.2f}, "
              f"saturation {100*np.mean(t>=0.99*e[i]):5.2f} pct of a {e[i]:.0f} Nm ceiling")
