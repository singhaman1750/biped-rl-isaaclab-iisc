"""Where the swing ankle roll actually sits, and what the in air terms are paid for it."""
import os, sys
import numpy as np
sys.path.insert(0, "/workspace/isaaclab/biped/scripts/analysis")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ablation import EVAL, LOGS, TAG, FORCE, FRAC, envelope, available, foot_index

JOBS = sys.argv[1:]
def lab(j): return TAG[j.rsplit("_model_",1)[0]] + "@" + j.rsplit("_",1)[-1]

print("### The swing ankle, position against its stops")
print(f"{'joint':20s} {'run':10s} {'nominal':>8s} {'stops':>17s} {'q_sw mean':>10s} {'q_sw p5':>8s} {'q_sw p95':>9s}"
      f" {'@lo%':>6s} {'@hi%':>6s} {'|qdot|':>7s} {'sat%':>6s}")
for jn in ["left_foot_roll_02","right_foot_roll_02","left_foot_pitch_02","right_foot_pitch_02"]:
    for job in JOBS:
        d = np.load(os.path.join(EVAL, job, "data", "42", "dump.npy"), allow_pickle=True).item()
        names=[str(n) for n in d["joint_names"]]; j=names.index(jn)
        pos,vel,tau=(np.asarray(d[k]) for k in ("joint_positions","joint_velocities","joint_torques"))
        nom=np.asarray(d["default_joint_pos"]); lim=np.asarray(d["joint_position_limits"])[j]
        sat,eff,vlim,kp,kd=envelope(d, os.path.join(LOGS, job.rsplit("_model_",1)[0]))
        contact=np.linalg.norm(np.asarray(d["feet_contact_forces"]),axis=-1)>FORCE
        sw=~contact[:,:,foot_index(jn)]
        q=pos[:,:,j][sw]
        tol=0.01*(lim[1]-lim[0])
        av=available(tau[:,:,j],vel[:,:,j],sat[j],eff[j],vlim[j])
        hot=(np.abs(tau[:,:,j])>=np.maximum(FRAC*av,1e-9))[sw]
        print(f"{jn:20s} {lab(job):10s} {nom[j]:8.4f} [{lim[0]:7.4f},{lim[1]:7.4f}]"
              f" {q.mean():10.4f} {np.percentile(q,5):8.4f} {np.percentile(q,95):9.4f}"
              f" {100.0*np.mean(q<=lim[0]+tol):6.2f} {100.0*np.mean(q>=lim[1]-tol):6.2f}"
              f" {np.abs(vel[:,:,j][sw]).mean():7.4f} {100.0*hot.mean():6.2f}")
    print()

print("\n### The hip yaw, commanded against achieved, split by phase")
print("    The hip yaw is unsaturated everywhere, so q_des is exact on every sample.")
print(f"{'joint':20s} {'run':10s} {'phase':7s} {'cmd dev':>9s} {'achieved':>9s} {'|err|':>8s} {'cmd/ach':>8s} {'|tau|':>7s}")
for jn in ["left_hip_yaw_03","right_hip_yaw_03"]:
    for job in JOBS:
        d = np.load(os.path.join(EVAL, job, "data", "42", "dump.npy"), allow_pickle=True).item()
        names=[str(n) for n in d["joint_names"]]; j=names.index(jn)
        pos,vel,tau=(np.asarray(d[k]) for k in ("joint_positions","joint_velocities","joint_torques"))
        nom=np.asarray(d["default_joint_pos"])
        sat,eff,vlim,kp,kd=envelope(d, os.path.join(LOGS, job.rsplit("_model_",1)[0]))
        contact=np.linalg.norm(np.asarray(d["feet_contact_forces"]),axis=-1)>FORCE
        st=contact[:,:,foot_index(jn)]
        qdes=pos[:,:,j]+(tau[:,:,j]+kd[j]*vel[:,:,j])/kp[j]
        for label,m in (("stance",st),("swing",~st)):
            c=np.abs(qdes-nom[j])[m].mean(); a=np.abs(pos[:,:,j]-nom[j])[m].mean()
            e=np.abs(qdes-pos[:,:,j])[m].mean()
            print(f"{jn:20s} {lab(job):10s} {label:7s} {c:9.4f} {a:9.4f} {e:8.4f} {c/a:8.3f}"
                  f" {np.abs(tau[:,:,j])[m].mean():7.3f}")
    print()

print("\n### Reward budget, mean per step, and share of the negative total")
rows={}
for job in JOBS:
    rw=np.load(os.path.join(EVAL,job,"data","42","rewards.npy"),allow_pickle=True).item()
    t={k:float(np.asarray(v).mean()) for k,v in rw.items() if k not in ("_weights","total_reward")}
    t["TOTAL"]=float(np.asarray(rw["total_reward"]).mean())
    rows[job]=t
keys=sorted(set().union(*[set(v) for v in rows.values()]),
            key=lambda k:-abs(np.mean([rows[j].get(k,0.0) for j in JOBS])))
print(f"{'term':34s} " + " ".join(f"{lab(j):>12s}" for j in JOBS))
for k in keys:
    print(f"{k:34s} " + " ".join(
        ("{:>12.5f}".format(rows[j][k]) if k in rows[j] else f"{'-':>12s}") for j in JOBS))
print()
for job in JOBS:
    neg=sum(v for k,v in rows[job].items() if v<0 and k!="TOTAL")
    pos=sum(v for k,v in rows[job].items() if v>0 and k!="TOTAL")
    en=sum(rows[job].get(k,0.0) for k in ("pen_joint_torque","pen_joint_accel",
          "pen_joint_vel_l2","pen_joint_torque_rate","pen_joint_power"))
    print(f"{lab(job):10s} positive {pos:9.4f}  negative {neg:9.4f}  energy group {en:9.4f}"
          f"  energy share of negative {100.0*en/neg:6.2f} pct")
