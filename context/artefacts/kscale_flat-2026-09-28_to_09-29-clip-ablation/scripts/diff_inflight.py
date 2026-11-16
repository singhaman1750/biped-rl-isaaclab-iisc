import sys, os
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
import experiment_params as ep
ROOT="/ws/IsaacLab/logs/rsl_rl/kscale_flat"
def flat(o,p=""):
    out={}
    if isinstance(o,dict):
        for k,v in o.items(): out.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o,(list,tuple)):
        for i,v in enumerate(o): out.update(flat(v, f"{p}[{i}]"))
    else: out[p]=o
    return out
R={"E":"2026-09-29_06-15-58","F":"2026-09-30_06-43-38","G":"2026-09-30_07-05-14"}
F={t:flat(ep.load_params(os.path.join(ROOT,r))) for t,r in R.items()}
keys=set().union(*[set(v) for v in F.values()])
M=object()
n=0
for k in sorted(keys):
    v=[F[t].get(k,M) for t in ("E","F","G")]
    if not all(x==v[0] for x in v):
        n+=1
        print(f"{k}\n    E={'<absent>' if v[0] is M else v[0]!r}\n    F={'<absent>' if v[1] is M else v[1]!r}\n    G={'<absent>' if v[2] is M else v[2]!r}")
print(f"\n{n} differing leaf paths of {len(keys)}")
