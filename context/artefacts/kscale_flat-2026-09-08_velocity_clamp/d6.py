import numpy as np
B="/ws/IsaacLab/logs/rsl_rl/kscale_flat/"
R=[("REF0828","2026-08-28_04-50-51"),("A_0650","2026-09-08_06-50-00"),("B_0739","2026-09-08_07-39-29")]
D={n:np.load(B+p+"/data/42/dump.npy",allow_pickle=True).item() for n,p in R}
JN=D["REF0828"]["joint_names"]
def dwell(mask):
    out=[]
    for e in range(mask.shape[1]):
        m=mask[:,e].astype(np.int8); d=np.diff(np.concatenate(([0],m,[0])))
        s=np.flatnonzero(d==1); t=np.flatnonzero(d==-1)
        out.extend((t-s).tolist())
    return np.array(out) if out else np.array([0])
print("CONTINUOUS DWELL AT THE 10 rad/s CLAMP (control steps of 0.01 s)")
print(f"{'run':10s}{'joint':20s}{'n_events':>9s}{'mean':>7s}{'p95':>7s}{'max':>6s}{'mean_ms':>9s}")
for n,d in D.items():
    qd=np.abs(d['joint_velocities'])
    for j in [6,10,4]:
        w=dwell(qd[:,:,j]>9.9)
        print(f"{n:10s}{JN[j]:20s}{len(w):9d}{w.mean():7.2f}{np.percentile(w,95):7.1f}{w.max():6d}{w.mean()*10:9.1f}")
    print()
print("FEET SEPARATION, what pen_feet_distance actually measures (lateral_only=False -> planar norm)")
print(f"{'run':10s}{'min_thr':>8s}{'planar_norm p5':>15s}{'mean':>8s}{'|lateral| p5':>13s}{'mean':>8s}{'|foreaft| mean':>15s}{'%planar<thr':>12s}{'%lateral<thr':>13s}")
for (n,d),thr in zip(D.items(),[0.24,0.20,0.20]):
    fd=d['feet_distance']   # base_yaw frame, xyz difference
    planar=np.linalg.norm(fd[:,:,:2],axis=-1); lat=np.abs(fd[:,:,1]); fa=np.abs(fd[:,:,0])
    print(f"{n:10s}{thr:8.2f}{np.percentile(planar,5):15.3f}{planar.mean():8.3f}{np.percentile(lat,5):13.3f}{lat.mean():8.3f}{fa.mean():15.3f}{(planar<thr).mean()*100:12.2f}{(lat<thr).mean()*100:13.2f}")
