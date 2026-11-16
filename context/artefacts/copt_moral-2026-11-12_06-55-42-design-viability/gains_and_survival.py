"""Servo stiffness of the hip and knee against evaluated survival, and the per window count of low gain designs.

    python3 -I gains_and_survival.py <run_dir> <eplen_file>
"""
import csv, glob, os, statistics as st, sys
run, eplen = sys.argv[1:3]
D = os.path.join(run, "designs")
kp = lambda r, g: float(r[g + "_actuator_stiffness"])
rows = [r for f in sorted(glob.glob(os.path.join(D, "cmaes_generation_*.csv"))) for r in csv.DictReader(open(f))]
print("CMA-ES generations, designs without the ODrive-D5065-DD hip, by min(hip, knee) stiffness")
groups = {}
for r in rows:
    if r["hip_actuator"] != "ODrive-D5065-DD":
        groups.setdefault(min(kp(r, "hip"), kp(r, "knee")), []).append(float(r["eval_survival"]))
for k in sorted(groups):
    v = groups[k]; print(f"  Kp {k:5.1f} Nm/rad  n={len(v):3d}  survival mean {st.mean(v):.3f}  min {min(v):.3f}")
odrive = [float(r["eval_survival"]) for r in rows if r["hip_actuator"] == "ODrive-D5065-DD"]
print(f"  ODrive-D5065-DD hip  n={len(odrive)}  survival {odrive}")
logged = dict(line.split() for line in open(eplen))
print("window      designs  hip_or_knee_Kp<=10  abad_Kp<=10  logged_len")
for f in sorted(glob.glob(os.path.join(D, "design_parameters_*.csv")), key=lambda p: int(p.split("_")[-1][:-4])):
    it = int(f.split("_")[-1][:-4]); R = list(csv.DictReader(open(f)))
    low = sum(min(kp(r, "hip"), kp(r, "knee")) <= 10 for r in R); la = sum(kp(r, "abad") <= 10 for r in R)
    print(f"{it-399:5d}-{it:<5d} {len(R):6d} {low:14d} {la:14d} {float(logged.get(str(it-399), 'nan')):11.1f}")
