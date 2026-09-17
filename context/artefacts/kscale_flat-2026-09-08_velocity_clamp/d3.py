import numpy as np
B="/ws/IsaacLab/logs/rsl_rl/kscale_flat/"
R=[("REF0828","2026-08-28_04-50-51"),("A_0650","2026-09-08_06-50-00"),("B_0739","2026-09-08_07-39-29")]
D={n:np.load(B+p+"/data/42/dump.npy",allow_pickle=True).item() for n,p in R}
JN=D["REF0828"]["joint_names"]
for n,d in D.items():
    q=d['joint_positions']; qd=d['joint_velocities']; tau=d['joint_torques']
    lim=np.array(d['joint_position_limits']); soft=np.array(d['joint_soft_position_limits'])
    el=np.array(d['joint_effort_limits']); vl=np.array(d['joint_velocity_limits'])
    print(f"\n===== {n} =====")
    print(f"{'joint':20s}{'|tau|p95':>9s}{'tau_lim':>8s}{'%satur':>8s}{'|qd|p95':>9s}{'%|qd|>9.9':>10s}{'%out_hard':>10s}{'%out_soft':>10s}{'q_range_used':>14s}")
    for j in range(12):
        t=np.abs(tau[:,:,j]); v=np.abs(qd[:,:,j]); p=q[:,:,j]
        sat=(t>0.98*el[j]).mean()*100
        vsat=(v>9.9).mean()*100
        outh=((p<lim[j,0])|(p>lim[j,1])).mean()*100
        outs=((p<soft[j,0])|(p>soft[j,1])).mean()*100
        rng=lim[j,1]-lim[j,0]
        used=(p.max()-p.min())/rng
        print(f"{JN[j]:20s}{np.percentile(t,95):9.2f}{el[j]:8.1f}{sat:8.2f}{np.percentile(v,95):9.2f}{vsat:10.2f}{outh:10.2f}{outs:10.2f}{used:14.2f}")
