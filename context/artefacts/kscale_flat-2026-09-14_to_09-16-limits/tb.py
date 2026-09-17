from tensorboard.backend.event_processing import event_accumulator as ea
import glob, numpy as np
BASE='/ws/IsaacLab/logs/rsl_rl/kscale_flat/'
RUNS=[('A 09-14','2026-09-14_10-20-50'),('B 09-15','2026-09-15_08-07-56'),('C 09-16','2026-09-16_04-52-20')]
T={}
for k,v in RUNS:
    p=glob.glob(BASE+v+'/events*')[0]
    a=ea.EventAccumulator(p,size_guidance={ea.SCALARS:0}); a.Reload()
    T[k]={t:np.array([(e.step,e.value) for e in a.Scalars(t)]) for t in a.Tags()['scalars']}
def at(tag,iters):
    print(f"\n{tag}")
    print(f"{'iter':>8}"+"".join(f"{k:>14}" for k,_ in RUNS))
    for it in iters:
        row=[]
        for k,_ in RUNS:
            d=T[k].get(tag)
            if d is None or len(d)==0 or d[:,0].max()<it: row.append(None); continue
            i=np.argmin(np.abs(d[:,0]-it)); row.append(d[i,1])
        print(f"{it:>8}"+"".join(f"{v:>14.4g}" if v is not None else f"{'-':>14}" for v in row))
