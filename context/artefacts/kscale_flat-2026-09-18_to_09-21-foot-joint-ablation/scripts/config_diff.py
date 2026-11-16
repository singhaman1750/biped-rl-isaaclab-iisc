import sys, os
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
from experiment_params import load_params

ROOT = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"11.9(09-18_09-37-29)":"2026-09-18_09-37-29",
        "11.9stub(09-18_09-24-17)":"2026-09-18_09-24-17",
        "17(09-21_06-45-40)":"2026-09-21_06-45-40",
        "34(09-21_08-40-59)":"2026-09-21_08-40-59"}

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
print(f"{'path':62} " + " ".join(f"{n:>26}" for n in names))
print("-"*180)
for k in diff:
    print(f"{k:62} " + " ".join(f"{str(F[n].get(k,'<absent>'))[:26]:>26}" for n in names))
