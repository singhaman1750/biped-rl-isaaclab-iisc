"""The three quantities the re-ranking of the ablation turns upon.

Whether the hip yaw sets the SWING toe-in as well as the stance toe-in, which decides
whether arm 16 reaches the fourteen degrees of swing splay or only the three of stance.

Whether the commanded ankle roll target in swing lies beyond the mechanical stop, which is
the condition arm 10's clip exists to remove and which cannot be read from the achieved
position. The joint is saturated on most swing samples, so the control law inverts to a
one sided bound rather than an equality, and the bound is what is reported.

And the value and the gradient of the three in air kernels at their own measured
deviations, which decides whether raising a weight can move a term or whether its width
must move first.
"""
import os, sys
import numpy as np
sys.path.insert(0, "/workspace/isaaclab/biped/scripts/analysis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation import (EVAL, LOGS, TAG, FORCE, FRAC, envelope, available, rot, yaw_of,
                      wrap, foot_index)

JOBS = sys.argv[1:]
def lab(j): return TAG[j.rsplit("_model_",1)[0]] + "@" + j.rsplit("_",1)[-1]

print("### The hip yaw against the toe-in, in stance and in swing separately")
print("    The left leg is mirrored on both axes, toe-in already flipping the left foot.\n")
print(f"{'run':10s} {'foot':6s} {'phase':7s} {'slope':>8s} {'corr':>7s} {'intercept deg':>14s} {'toe-in deg':>11s}")
for job in JOBS:
    d = np.load(os.path.join(EVAL, job, "data", "42", "dump.npy"), allow_pickle=True).item()
    names=[str(n) for n in d["joint_names"]]; pos=np.asarray(d["joint_positions"])
    nom=np.asarray(d["default_joint_pos"])
    fq,bq=np.asarray(d["feet_quaternions"]),np.asarray(d["base_quaternion"])
    by=yaw_of(bq)
    contact=np.linalg.norm(np.asarray(d["feet_contact_forces"]),axis=-1)>FORCE
    for side,f,sgn in (("left",0,-1.0),("right",1,1.0)):
        toe=rot(fq[:,:,f,:],(0.0,0.0,-1.0))
        ti=np.degrees(sgn*wrap(np.arctan2(toe[...,1],toe[...,0])-by))
        j=names.index(f"{side}_hip_yaw_03")
        x=np.degrees(sgn*(pos[:,:,j]-nom[j]))
        for ph,m in (("stance",contact[:,:,f]),("swing",~contact[:,:,f])):
            a,b=np.polyfit(x[m],ti[m],1)
            print(f"{lab(job):10s} {side:6s} {ph:7s} {a:8.3f} {np.corrcoef(x[m],ti[m])[0,1]:7.3f}"
                  f" {b:14.3f} {ti[m].mean():11.3f}")
    print()

print("\n### The commanded ankle roll target in swing against the mechanical stop")
print("    Where the sample is saturated the inversion gives a BOUND, the true target lying")
print("    at least as far out as the figure below, so the excess is understated.\n")
print(f"{'joint':20s} {'run':10s} {'stop':>8s} {'q_des mean':>11s} {'excess':>8s} {'beyond stop %':>14s}")
for jn in ["left_foot_roll_02","right_foot_roll_02","left_foot_pitch_02","right_foot_pitch_02"]:
    for job in JOBS:
        d=np.load(os.path.join(EVAL,job,"data","42","dump.npy"),allow_pickle=True).item()
        names=[str(n) for n in d["joint_names"]]; j=names.index(jn)
        pos,vel,tau=(np.asarray(d[k]) for k in ("joint_positions","joint_velocities","joint_torques"))
        lim=np.asarray(d["joint_position_limits"])[j]
        sat,eff,vlim,kp,kd=envelope(d, os.path.join(LOGS, job.rsplit("_model_",1)[0]))
        contact=np.linalg.norm(np.asarray(d["feet_contact_forces"]),axis=-1)>FORCE
        sw=~contact[:,:,foot_index(jn)]
        qdes=pos[:,:,j]+(tau[:,:,j]+kd[j]*vel[:,:,j])/kp[j]
        q=qdes[sw]
        near=lim[1] if pos[:,:,j][sw].mean()>0 else lim[0]
        beyond=(q>lim[1])|(q<lim[0])
        exc=np.where(q>lim[1],q-lim[1],np.where(q<lim[0],lim[0]-q,0.0))
        print(f"{jn:20s} {lab(job):10s} {near:8.4f} {q.mean():11.4f} {exc.mean():8.4f}"
              f" {100.0*beyond.mean():14.2f}")
    print()

print("\n### The three in air kernels at their own measured swing deviation")
print("    reward = exp(-sum of airborne |dev| / 0.2), so the gradient is value / 0.2 per")
print("    radian per unit weight. A term at a high value has a live gradient a weight can")
print("    multiply, a term at a low one has a vanished gradient a weight cannot reach.\n")
KERN={"rew_keep_ankle_pitch_zero_in_air":("foot_pitch_02",1.0),
      "rew_keep_ankle_roll_zero_in_air":("foot_roll_02",None),
      "rew_keep_hip_yaw_zero_in_air":("hip_yaw_03",1.0)}
print(f"{'term':36s} {'run':10s} {'swing dev':>10s} {'value':>8s} {'pct of max':>11s} {'d/d dev':>9s} {'paid/s':>8s}")
for term,(suffix,_) in KERN.items():
    for job in JOBS:
        d=np.load(os.path.join(EVAL,job,"data","42","dump.npy"),allow_pickle=True).item()
        rw=np.load(os.path.join(EVAL,job,"data","42","rewards.npy"),allow_pickle=True).item()
        if term not in rw: 
            print(f"{term:36s} {lab(job):10s}  absent"); continue
        names=[str(n) for n in d["joint_names"]]
        nom=np.asarray(d["default_joint_pos"]); pos=np.asarray(d["joint_positions"])
        contact=np.linalg.norm(np.asarray(d["feet_contact_forces"]),axis=-1)>FORCE
        devs=[]
        for jn in [n for n in names if n.endswith(suffix)]:
            j=names.index(jn); sw=~contact[:,:,foot_index(jn)]
            devs.append(np.abs(pos[:,:,j]-nom[j])[sw].mean())
        dev=float(np.mean(devs)); val=float(np.exp(-dev/0.2))
        print(f"{term:36s} {lab(job):10s} {dev:10.4f} {val:8.3f} {100*val:10.1f}%"
              f" {val/0.2:9.3f} {float(np.asarray(rw[term]).mean()):8.4f}")
    print()
