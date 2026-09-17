import numpy as np
B="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
G_CALC={'hip_yaw':(15.0,0.9),'hip_roll':(150.,17.3),'hip_pitch':(200.,22.4),'knee':(200.,10.),'foot_roll':(20.,0.5),'foot_pitch':(50.,1.7)}
G_IDENT={'hip_yaw':(500.,18.),'hip_roll':(250.,10.),'hip_pitch':(200.,8.),'knee':(300.,10.),'foot_roll':(170.,9.),'foot_pitch':(170.,9.)}
RUNS={"0828 calc as0.4":("2026-08-28_04-50-51",G_CALC,0.4),
      "arm0 ident as0.4":("2026-09-03_09-27-59",G_IDENT,0.4),
      "n1015 ident as0.25":("2026-09-04_10-15-18",G_IDENT,0.25)}
def grp(nm):
    for g in ['hip_yaw','hip_roll','hip_pitch','knee','foot_roll','foot_pitch']:
        if g in nm: return g
    raise KeyError(nm)
for n,(r,G,asc) in RUNS.items():
    d=np.load(f"{B}/{r}/data/42/dump.npy",allow_pickle=True).item()
    jn=d['joint_names']; jp=d['joint_positions']; jv=d['joint_velocities']; jt=d['joint_torques']
    dflt=np.array(d['default_joint_pos'])
    print(f"=== {n}  action_scale={asc}")
    print(f"  {'joint':24s}{'K':>7s}{'D':>7s}{'|e|rms rad':>12s}{'|e|p95':>9s}{'e/scale':>9s}{'|q-q0|rms':>11s}{'q-q0 p95':>10s}{'util':>7s}")
    for j,nm in enumerate(jn):
        K,D=G[grp(nm)]
        e=(jt[:,:,j]+D*jv[:,:,j])/K
        dev=jp[:,:,j]-dflt[j]
        qdes=jp[:,:,j]+e-dflt[j]     # commanded deviation from default = action*scale
        util=np.percentile(np.abs(qdes),95)/asc
        print(f"  {nm:24s}{K:7.0f}{D:7.1f}{np.sqrt((e**2).mean()):12.4f}{np.percentile(np.abs(e),95):9.4f}{np.percentile(np.abs(e),95)/asc:9.3f}{np.sqrt((dev**2).mean()):11.4f}{np.percentile(np.abs(dev),95):10.4f}{util:7.3f}")
