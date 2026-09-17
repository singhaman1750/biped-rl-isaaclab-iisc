import numpy as np,sys
BASE='/ws/IsaacLab/logs/rsl_rl/kscale_flat/'
RUNS=[('A 09-14','2026-09-14_10-20-50'),('B 09-15','2026-09-15_08-07-56'),('C 09-16','2026-09-16_04-52-20')]
D={}
for k,v in RUNS:
    s=np.load(BASE+v+'/data/42/statistics.npy',allow_pickle=True).item()
    D[k]={(r['group'],r['quantity'],r['statistic']):(r['value'],r.get('unit','')) for r in s['records']}
def show(pairs,stat='mean'):
    print(f"{'quantity':<52}{'unit':<10}"+"".join(f"{k:>14}" for k,_ in RUNS))
    print("-"*(62+14*3))
    for g,q in pairs:
        row=[];unit=''
        for k,_ in RUNS:
            t=D[k].get((g,q,stat))
            if t is None: row.append(None)
            else: row.append(t[0]); unit=t[1]
        cells="".join(f"{v:>14.4g}" if isinstance(v,float) else f"{str(v):>14}" for v in row)
        print(f"{q[:51]:<52}{unit[:9]:<10}{cells}")
if __name__=='__main__':
    pass
