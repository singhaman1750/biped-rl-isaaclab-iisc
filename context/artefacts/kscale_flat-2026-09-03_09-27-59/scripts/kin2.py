import numpy as np
BASE="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS={"0828":"2026-08-28_04-50-51","0902a":"2026-09-02_06-59-31","0902b":"2026-09-02_10-51-29","arm0":"2026-09-03_09-27-59","n1015":"2026-09-04_10-15-18"}
rows={}
for n,r in RUNS.items():
    d=np.load(f"{BASE}/{r}/data/42/dump.npy",allow_pickle=True).item()
    jn=d['joint_names']; jp=d['joint_positions']; jv=d['joint_velocities']; jt=d['joint_torques']
    lim=np.array(d['joint_position_limits']); eff=np.array(d['joint_effort_limits'])
    dt=d['step_dt']
    o={}
    for j,nm in enumerate(jn):
        lo,hi=lim[j]
        atlim=((jp[:,:,j]<lo+0.02)|(jp[:,:,j]>hi-0.02)).mean()
        sat=(np.abs(jt[:,:,j])>=0.98*eff[j]).mean()
        o[f"atlim_{nm}"]=float(atlim); o[f"sat_{nm}"]=float(sat)
    # shaking: base angular velocity rms, joint vel rms, torque rate
    bav=d['base_angular_velocity']
    o['base_angvel_rms']=float(np.sqrt((bav**2).sum(-1).mean()))
    o['base_angvel_xy_rms']=float(np.sqrt((bav[...,:2]**2).sum(-1).mean()))
    o['jointvel_rms']=float(np.sqrt((jv**2).mean()))
    o['torque_rate_rms']=float(np.sqrt((np.diff(jt,axis=0)/dt)**2).mean())
    # dominant oscillation frequency of base roll rate
    x=bav[:,:,0]-bav[:,:,0].mean(0)
    P=np.abs(np.fft.rfft(x,axis=0))**2; f=np.fft.rfftfreq(x.shape[0],dt)
    Pm=P.mean(1); Pm[0]=0
    o['base_roll_peak_hz']=float(f[Pm.argmax()])
    o['base_roll_pow_gt5hz']=float(Pm[f>5].sum()/Pm.sum())
    # contact / gait
    fn=np.linalg.norm(d['feet_contact_forces'],axis=-1); st=fn>1.0
    o['double_support_frac']=float((st.sum(-1)==2).mean())
    o['flight_frac']=float((st.sum(-1)==0).mean())
    # step frequency: touchdowns per second per foot
    td=(~st[:-1])&(st[1:])
    o['step_freq_hz']=float(td.sum()/(st.shape[0]*dt*st.shape[1]*2)*2)
    o['sole_clear_med_air']=float(np.median(d['feet_sole_clearances'][~st]))
    rows[n]=o
keys=list(rows['0828'].keys())
print(f"{'metric':38s}"+"".join(f"{n:>12s}" for n in rows))
for k in keys:
    if k.startswith('atlim_') or k.startswith('sat_'):
        if max(rows[n][k] for n in rows)<0.02: continue
    print(f"{k:38s}"+"".join(f"{rows[n][k]:12.4f}" for n in rows))
