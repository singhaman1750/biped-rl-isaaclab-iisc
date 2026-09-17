import numpy as np, sys
sys.path.insert(0,'/ws/tron1-rl-isaaclab-cozum/scripts/analysis')
from experiment_params import load_params, reward_weights
D=np.load("/root/.claude/jobs/b40fca2a/tmp/tb2.npy",allow_pickle=True).item()
base="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
runs={"0828":"2026-08-28_04-50-51","arm0":"2026-09-03_09-27-59","base4":"2026-09-03_09-49-22","n1015":"2026-09-04_10-15-18","n1040":"2026-09-04_10-40-30"}
W={n:reward_weights(load_params(f"{base}/{d}/params/env.yaml")) for n,d in runs.items()}
def s(n,tag,i,w=100):
    a=D[n][tag]; m=(a[:,0]>=i-w)&(a[:,0]<=i+w); return float(a[m,1].mean()) if m.any() else float('nan')
for it in [2000,13000]:
    print(f"===== per-second reward RATE contribution (weight x raw), iteration {it}")
    print(f"{chr(39)}term{chr(39)}:36s"+"".join(f"{n:>10s}" for n in runs))
    tot={n:0.0 for n in runs}; pos={n:0.0 for n in runs}; neg={n:0.0 for n in runs}
    rows=[]
    for tag in sorted(t for t in D['base4'] if t.startswith('Episode_Reward/')):
        k=tag.split('/')[1]; r=[]
        for n in runs:
            L=s(n,'Train/mean_episode_length',it); v=s(n,tag,it)
            c=v/(L/2000)   # per-second rate contribution already weighted
            r.append(c); tot[n]+=c
            (pos if c>=0 else neg)[n]+=c
        rows.append((k,r))
    for k,r in sorted(rows,key=lambda x:-abs(x[1][2])):
        print(f"{k:36s} "+" ".join(f"{x:10.3f}" for x in r))
    print(f"{'TOTAL /s':36s} "+" ".join(f"{tot[n]:10.3f}" for n in runs))
    print(f"{'  positives':36s} "+" ".join(f"{pos[n]:10.3f}" for n in runs))
    print(f"{'  negatives':36s} "+" ".join(f"{neg[n]:10.3f}" for n in runs))
    print()
