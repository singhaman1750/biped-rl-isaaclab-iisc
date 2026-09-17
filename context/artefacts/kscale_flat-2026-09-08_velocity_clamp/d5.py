import numpy as np
B="/ws/IsaacLab/logs/rsl_rl/kscale_flat/"
R=[("REF0828","2026-08-28_04-50-51"),("A_0650","2026-09-08_06-50-00"),("B_0739","2026-09-08_07-39-29")]
D={n:np.load(B+p+"/data/42/dump.npy",allow_pickle=True).item() for n,p in R}
JN=D["REF0828"]["joint_names"]
print("AT THE 10 rad/s CLAMP: is the actuator driving INTO the clamp (tau*qd>0)?")
print(f"{'run':10s}{'joint':20s}{'%clamped':>9s}{'mean|tau|@clamp':>16s}{'%driving_in':>12s}{'mean P@clamp W':>15s}")
for n,d in D.items():
    qd=d['joint_velocities']; tau=d['joint_torques']
    for j in [0,4,6,10]:
        v=qd[:,:,j]; t=tau[:,:,j]
        m=np.abs(v)>9.9
        if m.sum()<50:
            print(f"{n:10s}{JN[j]:20s}{m.mean()*100:9.2f}{'--':>16s}{'--':>12s}{'--':>15s}"); continue
        drive=(t[m]*v[m])>0
        print(f"{n:10s}{JN[j]:20s}{m.mean()*100:9.2f}{np.abs(t[m]).mean():16.2f}{drive.mean()*100:12.1f}{(t[m]*v[m]).mean():15.1f}")
    print()
print("TORQUE PENALTY MAGNITUDES, per-step weighted contribution (from tensorboard at plateau)")
