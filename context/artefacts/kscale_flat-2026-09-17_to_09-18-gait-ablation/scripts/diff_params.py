"""Four way diff of the dumped env.yaml of the KScale ablation runs."""
import sys
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
from experiment_params import ParamsLoader
import yaml

BASE = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = [
    "2026-09-17_09-13-46",
    "2026-09-17_11-25-59",
    "2026-09-18_09-37-29",
    "2026-09-18_10-03-48",
]


def flatten(obj, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("__class__",):
                continue
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (list, tuple)):
        # keep short numeric sequences whole, they read better than per index rows
        if all(not isinstance(x, (dict, list, tuple)) for x in obj) and len(obj) <= 16:
            out[prefix] = repr(list(obj))
        else:
            for i, v in enumerate(obj):
                flatten(v, f"{prefix}[{i}]", out)
    else:
        out[prefix] = obj
    return out


docs = {}
for r in RUNS:
    with open(f"{BASE}/{r}/params/env.yaml") as f:
        docs[r] = flatten(yaml.load(f, Loader=ParamsLoader))

allkeys = set()
for r in RUNS:
    allkeys |= set(docs[r].keys())

MISSING = "<absent>"


def norm(v):
    if isinstance(v, float):
        return round(v, 10)
    return v


diffs = []
for k in sorted(allkeys):
    vals = [norm(docs[r].get(k, MISSING)) for r in RUNS]
    if len(set(map(repr, vals))) > 1:
        diffs.append((k, vals))

print(f"total leaf paths: {len(allkeys)}    differing paths: {len(diffs)}")
print("=" * 130)
print(f"{'path':<62} {'A 09-17_09-13':>16} {'B 09-17_11-25':>16} {'C 09-18_09-37':>16} {'D 09-18_10-03':>16}")
print("=" * 130)
for k, vals in diffs:
    cells = []
    for v in vals:
        s = str(v)
        if len(s) > 16:
            s = s[:15] + "~"
        cells.append(f"{s:>16}")
    print(f"{k:<62} {' '.join(cells)}")
print("=" * 130)
# long values printed in full underneath
print("\nFULL VALUES FOR TRUNCATED ROWS")
for k, vals in diffs:
    if any(len(str(v)) > 16 for v in vals):
        print(f"\n{k}")
        for r, v in zip(RUNS, vals):
            print(f"   {r}: {v}")
