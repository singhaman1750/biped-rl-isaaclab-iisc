import sys
sys.path.insert(0,'/ws/tron1-rl-isaaclab-cozum/scripts/analysis')
from experiment_params import load_params, reward_weights
B='/ws/IsaacLab/logs/rsl_rl/kscale_flat/'
runs={'0828':'2026-08-28_04-50-51','arm0':'2026-09-03_09-27-59','base4':'2026-09-03_09-49-22','n0915':'2026-09-04_10-15-18','n1040':'2026-09-04_10-40-30'}
P={k:load_params(B+v+'/params/env.yaml') for k,v in runs.items()}
W={k:reward_weights(p) for k,p in P.items()}
keys=sorted(set().union(*[set(w) for w in W.values()]))
print(f"{'term':38s}"+''.join(f"{k:>10s}" for k in runs))
for t in keys:
    row=[W[k].get(t) for k in runs]
    if len(set([str(r) for r in row]))>1:
        print(f"{t:38s}"+''.join(f"{('' if r is None else f'{r:.4g}'):>10s}" for r in row))
