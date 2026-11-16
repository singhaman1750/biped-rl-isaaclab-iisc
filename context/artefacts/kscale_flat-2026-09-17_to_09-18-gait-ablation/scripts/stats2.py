"""Second pass, using the correct statistic names."""
import numpy as np

BASE = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"A 09-17_09-13": "2026-09-17_09-13-46", "B 09-17_11-25": "2026-09-17_11-25-59",
        "C 09-18_09-37": "2026-09-18_09-37-29", "D 09-18_10-03": "2026-09-18_10-03-48"}
S = {}
for label, run in RUNS.items():
    s = np.load(f"{BASE}/{run}/data/42/statistics.npy", allow_pickle=True).item()
    S[label] = {(r["group"], r["quantity"], r["statistic"]): r["value"] for r in s["records"]}
    jn = s["meta"]["joint_names"]


def row(g, q, st, name=None, scale=1.0):
    cells = []
    for label in RUNS:
        v = S[label].get((g, q, st))
        try:
            cells.append(f"{float(v)*scale:13.5g}")
        except (TypeError, ValueError):
            cells.append(f"{'--':>13}")
    print(f"  {(name or q):<58}{''.join(cells)}")


print(f"{'quantity':<60}{'A':>13}{'B':>13}{'C':>13}{'D':>13}")
print("=" * 112)
print("--- temporal (fractions of time)")
for q in ["double support", "single support", "flight"]:
    row("temporal", q, "fraction", q + " [fraction]")
for f in ["foot_6061", "foot_6061_2"]:
    row("temporal", f"stance duty {f}", "fraction", f"stance duty {f}")
row("temporal", "commanded stance duty", "fraction")
row("temporal", "cadence", "steps per second", "cadence [steps/s]")
row("temporal", "commanded frequency", "value", "commanded frequency [Hz]")

print("--- energetics")
for q in ["cost of transport, absolute", "cost of transport, positive", "cost of transport, net",
          "mean forward speed", "antagonism ratio", "Froude number"]:
    row("energetics", q, "value")

print("--- symmetry / stability / variability")
for q in ["peak contact force symmetry index", "stance duty symmetry index", "swing duration symmetry index"]:
    row("symmetry", q, "value")
row("stability", "base of support", "is a lower bound, sole extent unknown", "base of support [lower bound]")
row("stability", "pendulum eigenfrequency", "value", "pendulum eigenfrequency [rad/s]")
row("variability", "step width", "coefficient of variation", "step width [CoV]")
row("variability", "episode ends", "count", "episode ends [count]")
for f in ["foot_6061", "foot_6061_2"]:
    row("feet", f"forged contact {f}", "fraction", f"forged contact {f} [fraction]")

print()
print("=" * 112)
print("JOINT TORQUE SATURATION, fraction of samples above 98 pct of the effort ceiling")
print(f"  {'joint':<58}{'A':>13}{'B':>13}{'C':>13}{'D':>13}")
for j in jn:
    row("joints", f"joint torque {j}", "fraction above 98 pct of ceiling", j)
print()
print("JOINT TORQUE, p99 as a fraction of the ceiling")
for j in jn:
    row("joints", f"joint torque {j}", "p99 as fraction of ceiling", j)
print()
print("JOINT TORQUE, mean absolute [Nm]")
for j in jn:
    row("joints", f"joint torque {j}", "mean absolute", j)
print()
print("JOINT VELOCITY, fraction above 98 pct of ceiling")
for j in jn:
    row("joints", f"joint velocity {j}", "fraction above 98 pct of ceiling", j)
print()
print("JOINT VELOCITY, p99 as fraction of ceiling")
for j in jn:
    row("joints", f"joint velocity {j}", "p99 as fraction of ceiling", j)
print()
print("JOINT POSITION, mean absolute [rad]  (absolute angle, not deviation)")
for j in jn:
    row("joints", f"joint position {j}", "mean absolute", j)
print()
print("JOINT POWER, mean absolute [W]")
for j in jn:
    row("joints", f"joint power {j}", "mean absolute", j)
