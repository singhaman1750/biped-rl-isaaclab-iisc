import numpy as np, common

SAT = {"tau11.9": 17.0, "tau17": 17.0, "tau34": 34.0}   # foot saturation_effort per run
rows = []
for run in common.RUNS:
    d = common.load(run)
    jn = list(d["joint_names"]); ix={n:i for i,n in enumerate(jn)}
    tau = d["joint_torques"]; vel = d["joint_velocities"]; pos = d["joint_positions"]
    elim = np.asarray(d["joint_effort_limits"]); vlim = np.asarray(d["joint_velocity_limits"])
    hard = np.asarray(d["joint_position_limits"]); soft = np.asarray(d["joint_soft_position_limits"])
    # per-joint saturation_effort: from the config, hips/knee keep sat>lim, feet per run
    sat = np.where(np.isin(np.arange(12), [ix[n] for n in jn if "foot" in n]), SAT[run],
                   np.where(np.array([("hip_pitch" in n or "knee" in n) for n in jn]), 120.0, 60.0))
    contact = np.linalg.norm(d["feet_contact_forces"], axis=-1) > 1.0   # (T,E,2)
    leg_stance = {"left": contact[:,:,0], "right": contact[:,:,1]}
    print(f"\n================ {run}  ({common.RUNS[run]}) ================")
    print(f"{'joint':22} {'|tau| mean':>10} {'envSat%':>8} {'limSat%':>8} {'stSat%':>7} {'swSat%':>7} "
          f"{'|v| p50':>8} {'|v| p99':>8} {'v>Vc%':>7} {'hardRes%':>8} {'softOut%':>8}")
    for j,n in enumerate(jn):
        lo,hi = common.envelope(sat[j], elim[j], vlim[j], vel[:,:,j])
        at_env = (tau[:,:,j] >= 0.98*hi) | (tau[:,:,j] <= 0.98*lo)
        at_lim = np.abs(tau[:,:,j]) >= 0.98*elim[j]
        vc = vlim[j]*(1.0 - elim[j]/sat[j])                      # corner velocity
        leg = "left" if n.startswith("left") else "right"
        st = leg_stance[leg]; sw = ~st
        near_hard = (np.abs(pos[:,:,j]-hard[j,0])<0.01*(hard[j,1]-hard[j,0])) | \
                    (np.abs(pos[:,:,j]-hard[j,1])<0.01*(hard[j,1]-hard[j,0]))
        out_soft = (pos[:,:,j]<soft[j,0]) | (pos[:,:,j]>soft[j,1])
        print(f"{n:22} {np.abs(tau[:,:,j]).mean():10.2f} {100*at_env.mean():8.2f} {100*at_lim.mean():8.2f} "
              f"{100*at_env[st].mean():7.2f} {100*at_env[sw].mean():7.2f} "
              f"{np.percentile(np.abs(vel[:,:,j]),50):8.2f} {np.percentile(np.abs(vel[:,:,j]),99):8.2f} "
              f"{100*(np.abs(vel[:,:,j])>vc).mean():7.2f} {100*near_hard.mean():8.2f} {100*out_soft.mean():8.2f}")
    print(f"  corner velocity, foot joints: {vlim[ix['left_foot_roll_02']]*(1-elim[ix['left_foot_roll_02']]/SAT[run]):.3f} rad/s")
