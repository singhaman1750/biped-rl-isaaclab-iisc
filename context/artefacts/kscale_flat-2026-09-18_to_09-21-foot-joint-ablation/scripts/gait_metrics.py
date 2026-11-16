import numpy as np, common
from frames import quat_rot
def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
out={}
for run in common.RUNS:
    d=common.load(run); jn=list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    dt=float(d["step_dt"]); pos=d["joint_positions"]; vel=d["joint_velocities"]
    bq=d["base_quaternion"]; by=yaw_of(bq)
    contact=np.linalg.norm(d["feet_contact_forces"],axis=-1)>1.0
    mass=float(np.sum(d["body_masses"])); W=mass*9.81
    blv=d["base_linear_velocity"]; clv=d["commanded_linear_velocity"]
    err=np.linalg.norm(blv[...,:2]-clv,axis=-1)
    pw=np.abs(d["joint_powers"]).sum(-1)
    spd=np.linalg.norm(blv[...,:2],axis=-1)
    cot=pw.mean()/(W*spd.mean())
    jerk=np.diff(d["joint_accelerations"],axis=0)/dt
    # toe-in: foot link +z (fore-aft) projected, differenced against base heading
    toe=[]
    for fi,sgn in ((0,+1),(1,-1)):
        f=quat_rot(d["feet_quaternions"][:,:,fi,:],np.array([0.,0.,1.]))
        fyaw=np.arctan2(f[...,1],f[...,0])
        rel=np.arctan2(np.sin(fyaw-by),np.cos(fyaw-by))
        toe.append(rel)
    toe=np.stack(toe,-1)
    # sign so that positive = toes converging on the midline
    ti = np.degrees(np.stack([-toe[...,0], toe[...,1]],-1))
    ds=(contact[:,:,0]&contact[:,:,1]).mean()
    # cadence: touchdown rising edges
    td=((~contact[:-1])&contact[1:]).sum()/(contact.shape[0]*contact.shape[1]*dt)
    gr=-quat_rot(bq,np.array([0.,0.,1.]))   # projected gravity proxy
    tilt=np.degrees(np.arccos(np.clip(-gr[...,2],-1,1)))
    out[run]=dict(
        mass=mass, vel_err=err.mean(), cot=cot, power=pw.mean(),
        jerk=np.abs(jerk).mean(), cadence=td, double=100*ds,
        height=d["base_com_position"][...,2].mean(),
        knee=pos[:,:,[ix["left_knee_04"],ix["right_knee_04"]]].mean(),
        toein=ti.mean(), toein95=np.percentile(ti,95), tilt=tilt.mean(),
        term=100*d["episode_terminated"].mean()/dt/ (1/dt) ,
        term_rate=float(d["episode_terminated"].sum()),
        timeouts=float(d["episode_time_outs"].sum()),
        sep=np.abs(d["feet_distance"][...,1]).mean())
hdr=["mass","vel_err","cot","power","jerk","cadence","double","height","knee","toein","toein95","tilt","sep","term_rate","timeouts"]
print(f"{'quantity':12}"+"".join(f"{r:>14}" for r in common.RUNS))
for h in hdr:
    print(f"{h:12}"+"".join(f"{out[r][h]:>14.4f}" for r in common.RUNS))
