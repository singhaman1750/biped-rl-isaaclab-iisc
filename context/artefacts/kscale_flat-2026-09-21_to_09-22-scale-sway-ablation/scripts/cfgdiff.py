import sys, os
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
from experiment_params import load_params

ROOT = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"A(09-21_06-45-40)":"2026-09-21_06-45-40",
        "B(09-22_07-18-05)":"2026-09-22_07-18-05",
        "C(09-22_07-40-43)":"2026-09-22_07-40-43"}

def flat(o, p="", out=None):
    if out is None: out={}
    if isinstance(o, dict):
        for k,v in o.items():
            if k=="__class__": continue
            flat(v, f"{p}.{k}" if p else str(k), out)
    elif isinstance(o,(list,tuple)):
        for i,v in enumerate(o): flat(v, f"{p}[{i}]", out)
    else:
        out[p]=o
    return out

F = {k: flat(load_params(os.path.join(ROOT,v))) for k,v in RUNS.items()}
keys = set()
for f in F.values(): keys |= set(f)
diff = [k for k in sorted(keys) if len({repr(f.get(k,'<absent>')) for f in F.values()})>1]
print(f"total leaf paths {len(keys)}, differing {len(diff)}\n")
names=list(RUNS)
print(f"{'path':68} " + " ".join(f"{n:>22}" for n in names))
print("-"*140)
for k in diff:
    print(f"{k:68} " + " ".join(f"{str(F[n].get(k,'<absent>'))[:22]:>22}" for n in names))
