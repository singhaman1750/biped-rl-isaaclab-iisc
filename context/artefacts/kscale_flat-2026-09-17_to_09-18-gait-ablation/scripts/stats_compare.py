"""Side by side comparison of the precomputed statistics records across the four runs."""
import numpy as np
from collections import defaultdict

BASE = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {
    "A 09-17_09-13": "2026-09-17_09-13-46",
    "B 09-17_11-25": "2026-09-17_11-25-59",
    "C 09-18_09-37": "2026-09-18_09-37-29",
    "D 09-18_10-03": "2026-09-18_10-03-48",
}

S = {}
for label, run in RUNS.items():
    s = np.load(f"{BASE}/{run}/data/42/statistics.npy", allow_pickle=True).item()
    idx = {}
    for rec in s["records"]:
        idx[(rec["group"], rec["quantity"], rec["statistic"])] = (rec["value"], rec.get("unit", ""))
    S[label] = idx

WANT = [
    # (group, quantity, statistic)
    ("posture", "base height", "mean"), ("posture", "base height", "std"),
    ("posture", "base height", "p5"), ("posture", "base height", "p50"), ("posture", "base height", "p95"),
    ("posture", "torso pitch", "mean"), ("posture", "torso roll", "mean"),
    ("posture", "torso tilt from vertical", "mean"),
    ("posture", "base height excursion per episode", "mean"),
    ("stance", "lateral separation", "mean"), ("stance", "lateral separation magnitude", "mean"),
    ("stance", "fore and aft separation magnitude", "mean"),
    ("stance", "planar separation", "mean"), ("stance", "total separation", "mean"),
    ("stance", "vertical separation magnitude", "mean"),
    ("feet", "foot yaw misalignment foot_6061", "mean"),
    ("feet", "foot yaw misalignment foot_6061_2", "mean"),
    ("feet", "foot roll foot_6061, in stance", "mean"),
    ("feet", "foot roll foot_6061_2, in stance", "mean"),
    ("feet", "foot pitch foot_6061", "mean"), ("feet", "foot pitch foot_6061_2", "mean"),
    ("feet", "sole clearance foot_6061, in swing", "mean"),
    ("feet", "sole clearance foot_6061_2, in swing", "mean"),
    ("feet", "forged contact foot_6061", "mean"), ("feet", "forged contact foot_6061_2", "mean"),
    ("temporal", "cadence", "mean"), ("temporal", "cycle period", "mean"),
    ("temporal", "double support", "mean"), ("temporal", "single support", "mean"),
    ("temporal", "flight", "mean"), ("temporal", "swing duration", "mean"),
    ("temporal", "stance duty foot_6061", "mean"), ("temporal", "stance duty foot_6061_2", "mean"),
    ("temporal", "commanded stance duty", "mean"), ("temporal", "commanded frequency", "mean"),
    ("swing", "swing apex clearance", "mean"), ("swing", "raised cosine correlation", "mean"),
    ("swing", "clearance peaks per swing", "mean"),
    ("impact", "peak contact force", "mean"), ("impact", "peak contact force", "p95"),
    ("impact", "loading rate", "mean"), ("impact", "touchdown sole approach speed", "mean"),
    ("impact", "stance impulse", "mean"),
    ("tracking", "planar velocity error", "mean"), ("tracking", "linear velocity x", "mean"),
    ("tracking", "linear velocity error x magnitude", "mean"),
    ("tracking", "lateral linear velocity magnitude", "mean"),
    ("tracking", "yaw rate error", "mean"),
    ("energetics", "cost of transport, absolute", "mean"),
    ("energetics", "cost of transport, positive", "mean"),
    ("energetics", "absolute joint power", "mean"), ("energetics", "mean forward speed", "mean"),
    ("energetics", "antagonism ratio", "mean"), ("energetics", "Froude number", "mean"),
    ("stability", "base of support", "mean"),
    ("stability", "centre of pressure offset, lateral", "mean"),
    ("stability", "extrapolated com offset, lateral", "mean"),
    ("stability", "pendulum eigenfrequency", "mean"),
    ("symmetry", "peak contact force symmetry index", "mean"),
    ("symmetry", "stance duty symmetry index", "mean"),
    ("symmetry", "swing duration symmetry index", "mean"),
    ("variability", "episode duration", "mean"), ("variability", "episode ends", "mean"),
    ("variability", "step width", "mean"),
    ("smoothness", "joint jerk, all joints", "mean"),
    ("smoothness", "spectral arc length of base speed", "mean"),
    ("odometry", "expected base position error", "mean"),
]

print(f"{'group / quantity / stat':<62}{'A':>13}{'B':>13}{'C':>13}{'D':>13}   unit")
print("=" * 125)
lastg = None
for g, q, st in WANT:
    if g != lastg:
        print(f"--- {g}")
        lastg = g
    cells, unit = [], ""
    for label in RUNS:
        v = S[label].get((g, q, st))
        if v is None:
            cells.append(f"{'--':>13}")
        else:
            unit = v[1]
            try:
                cells.append(f"{float(v[0]):13.5g}")
            except (TypeError, ValueError):
                cells.append(f"{str(v[0])[:13]:>13}")
    print(f"  {q+' ['+st+']':<60}{''.join(cells)}   {unit}")

# joints: per joint position, torque, velocity means
print()
print("=" * 125)
jn = np.load(f"{BASE}/{RUNS['A 09-17_09-13']}/data/42/statistics.npy", allow_pickle=True).item()["meta"]["joint_names"]
for fam, stt in [("joint position", "mean"), ("joint position", "std"),
                 ("joint torque", "mean"), ("joint velocity", "mean"),
                 ("joint power", "mean"), ("joint acceleration", "mean")]:
    print(f"--- {fam} [{stt}]")
    print(f"  {'joint':<36}{'A':>13}{'B':>13}{'C':>13}{'D':>13}")
    for j in jn:
        cells = []
        for label in RUNS:
            v = S[label].get(("joints", f"{fam} {j}", stt))
            cells.append(f"{float(v[0]):13.5g}" if v else f"{'--':>13}")
        print(f"  {j:<36}{''.join(cells)}")
