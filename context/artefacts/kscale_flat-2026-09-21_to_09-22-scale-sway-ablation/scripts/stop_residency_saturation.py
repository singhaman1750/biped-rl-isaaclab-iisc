import numpy as np, os
ROOT="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
R={"A":"2026-09-21_06-45-40","B":"2026-09-22_07-18-05","C":"2026-09-22_07-40-43"}
D={k:np.load(os.path.join(ROOT,v,"data/42/dump.npy"),allow_pickle=True).item() for k,v in R.items()}
J=D["A"]["joint_names"]
lim=np.array(D["A"]["joint_position_limits"])
soft=np.array(D["A"]["joint_soft_position_limits"])
print("HARD LIMIT RESIDENCY, fraction of samples within 1 percent of the travel of a hard stop\n")
print(f"{'joint':24}{'lo':>9}{'hi':>9}   " + "".join(f"{k+' %':>10}" for k in R))
for i,n in enumerate(J):
    lo,hi=lim[i]; tol=0.01*(hi-lo)
    row=[]
    for k in R:
        q=D[k]["joint_positions"][:,:,i]
        row.append(100*np.mean((q<lo+tol)|(q>hi-tol)))
    print(f"{n:24}{lo:>9.4f}{hi:>9.4f}   "+"".join(f"{v:>10.2f}" for v in row))
print("\nSOFT LIMIT EXCURSION, fraction of samples outside the soft limit, which is what pen_joint_pos_limits charges\n")
print(f"{'joint':24}{'slo':>9}{'shi':>9}   " + "".join(f"{k+' %':>10}" for k in R))
for i,n in enumerate(J):
    lo,hi=soft[i]
    row=[]
    for k in R:
        q=D[k]["joint_positions"][:,:,i]
        row.append(100*np.mean((q<lo)|(q>hi)))
    print(f"{n:24}{lo:>9.4f}{hi:>9.4f}   "+"".join(f"{v:>10.2f}" for v in row))
print("\nTORQUE SATURATION, fraction of samples at or beyond 99 percent of the dumped effort ceiling\n")
eff=np.array(D["A"]["joint_effort_limits"])
print(f"{'joint':24}{'tau_lim':>9}   " + "".join(f"{k+' %':>10}" for k in R) + "   " + "".join(f"{k+' p99':>10}" for k in R))
for i,n in enumerate(J):
    e=eff[i]
    row=[];p99=[]
    for k in R:
        t=np.abs(D[k]["joint_torques"][:,:,i])
        row.append(100*np.mean(t>=0.99*e)); p99.append(np.percentile(t,99))
    print(f"{n:24}{e:>9.1f}   "+"".join(f"{v:>10.2f}" for v in row)+"   "+"".join(f"{v:>10.2f}" for v in p99))
