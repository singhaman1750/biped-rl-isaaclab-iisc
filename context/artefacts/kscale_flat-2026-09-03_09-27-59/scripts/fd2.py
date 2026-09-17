import numpy as np
BASE="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS={"0828":("2026-08-28_04-50-51",0.24,False,-100),
      "0902a":("2026-09-02_06-59-31",0.14,True,-30),
      "0902b":("2026-09-02_10-51-29",0.14,True,-30),
      "arm0":("2026-09-03_09-27-59",0.24,False,-100),
      "n1015":("2026-09-04_10-15-18",0.24,False,-100)}
def yawq(q):
    w,x,y,z=[q[...,i] for i in range(4)]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def R(q):
    w,x,y,z=[q[...,i] for i in range(4)]
    return np.stack([np.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],-1),
                     np.stack([2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],-1),
                     np.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],-1)],-2)
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
for n,(r,mn,lat,w) in RUNS.items():
    d=np.load(f"{BASE}/{r}/data/42/dump.npy",allow_pickle=True).item()
    fd=d['feet_distance']              # (T,E,3) base_yaw frame
    planar=np.linalg.norm(fd[...,:2],axis=-1)
    lateral=np.abs(fd[...,1])
    used = lateral if lat else planar
    hinge=np.clip(mn-used,0,1)
    jn=d['joint_names']; jp=d['joint_positions']
    hyL=jp[:,:,jn.index('left_hip_yaw_03')]; hyR=jp[:,:,jn.index('right_hip_yaw_03')]
    conv=(hyR-hyL)/2                    # + = converge (toe-in) empirically
    toe=-R(d['feet_quaternions'])[...,:,2]
    ty=np.arctan2(toe[...,1],toe[...,0]); by=yawq(d['base_quaternion'])
    sp=wrap(ty-by[...,None]); toein=-(sp[...,0]-sp[...,1])/2   # + = toe-in
    print(f"=== {n}  min={mn} lateral_only={lat} w={w}")
    print(f"   planar sep  med {np.median(planar):.4f}  p5 {np.percentile(planar,5):.4f}  p95 {np.percentile(planar,95):.4f}")
    print(f"   lateral sep med {np.median(lateral):.4f}  p5 {np.percentile(lateral,5):.4f}  p95 {np.percentile(lateral,95):.4f}")
    print(f"   hinge active frac {float((hinge>0).mean()):.4f}  mean hinge {hinge.mean():.5f}  cost/s {w*hinge.mean():.3f}")
    print(f"   corr(hinge, toein)  {np.corrcoef(hinge.ravel(),toein.ravel())[0,1]:+.4f}")
    print(f"   corr(hinge, conv)   {np.corrcoef(hinge.ravel(),conv.ravel())[0,1]:+.4f}")
    print(f"   corr(toein, planar) {np.corrcoef(toein.ravel(),planar.ravel())[0,1]:+.4f}")
    print(f"   corr(toein, lateral){np.corrcoef(toein.ravel(),lateral.ravel())[0,1]:+.4f}")
    print(f"   toein mean {np.rad2deg(toein.mean()):.3f} deg   conv mean {np.rad2deg(conv.mean()):.3f} deg")
    # binned: mean toein by hinge decile
    q=np.quantile(hinge.ravel(),[0,.25,.5,.75,.9,1.0])
    hb=hinge.ravel(); tb=np.rad2deg(toein.ravel()); lb=lateral.ravel()
    for i in range(len(q)-1):
        m=(hb>=q[i])&(hb<=q[i+1])
        if m.sum()>100: print(f"     hinge[{q[i]:.4f},{q[i+1]:.4f}] n={m.sum():7d} toein {tb[m].mean():7.3f} deg  lateral {lb[m].mean():.4f}")
