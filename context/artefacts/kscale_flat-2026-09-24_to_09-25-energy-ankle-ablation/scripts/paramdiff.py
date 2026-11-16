import sys, os
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
import experiment_params as ep

RUNS = {
 "A": "/ws/IsaacLab/logs/rsl_rl/kscale_flat/2026-09-24_05-18-33",
 "B": "/ws/IsaacLab/logs/rsl_rl/kscale_flat/2026-09-25_08-00-34",
 "C": "/ws/IsaacLab/logs/rsl_rl/kscale_flat/2026-09-25_11-29-39",
}

def flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out

flat = {}
for tag, d in RUNS.items():
    p = ep.load_params(d)
    assert p is not None, d
    flat[tag] = flatten(p)

keys = set()
for t in flat: keys |= set(flat[t])
print(f"total leaf paths: A={len(flat['A'])} B={len(flat['B'])} C={len(flat['C'])} union={len(keys)}")
MISS = object()
diff = []
for k in sorted(keys):
    vals = [flat[t].get(k, MISS) for t in ("A","B","C")]
    if not all(v == vals[0] for v in vals):
        diff.append((k, vals))
print(f"differing leaf paths: {len(diff)}\n")
def s(v):
    return "<absent>" if v is MISS else repr(v)
for k, vals in diff:
    print(f"{k}\n    A={s(vals[0])}\n    B={s(vals[1])}\n    C={s(vals[2])}")
