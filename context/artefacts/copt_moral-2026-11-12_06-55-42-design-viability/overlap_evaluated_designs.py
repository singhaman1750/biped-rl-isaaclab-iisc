"""Overlap of the emitted MJCF of every evaluated design of the run, against its evaluated survival. Run with python3 -I <run_dir>, where mujoco and numpy are installed."""
import csv, glob, os, sys, re
import mujoco
run = sys.argv[1]
def total_mass(m): return float(m.body_mass.sum())
rows_all = []
for p in sorted(glob.glob(f"{run}/designs/cmaes_generation_*.csv")):
    g = int(p[-8:-4])
    for r in csv.DictReader(open(p)):
        rows_all.append((g, int(r["individual"]), r))
mismatch = 0; out = []
for g, k, r in rows_all:
    path = f"{run}/copt_usds/gen_{g:04d}/individual_{k:04d}.xml"
    if not os.path.exists(path):
        out.append((g, k, r, None, None)); continue
    m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m)
    if abs(total_mass(m) - float(r["total_mass"])) > 5e-3: mismatch += 1
    d.qpos[:] = 0
    if m.nq >= 7: d.qpos[2] = 2.0; d.qpos[3] = 1.0
    mujoco.mj_forward(m, d)
    pairs = sorted({tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name))) for c in d.contact[:d.ncon]})
    out.append((g, k, r, pairs, total_mass(m)))
missing = sum(1 for o in out if o[3] is None)
print("designs", len(out), "xml missing", missing, "mass mismatches vs CSV total_mass", mismatch)
ov = [o for o in out if o[3]]
print("designs with overlapping bodies at the nominal pose:", len(ov))
z = [o for o in out if o[3] is not None and float(o[2]["eval_survival"]) < 0.5]
print("designs with survival < 0.5:", len(z), "of which overlapping:", sum(1 for o in z if o[3]))
print("overlapping designs with survival >= 0.5:", sum(1 for o in ov if float(o[2]["eval_survival"]) >= 0.5))
print()
print("gen ind  abad | hip | knee | ankle                       survival fitness  pairs")
for g, k, r, pairs, _ in ov + [o for o in z if not o[3]]:
    tag = "" if pairs else "(no overlap) "
    print(f"{g:3d} {k:3d}  {r['abad']} | {r['hip']} | {r['knee']} | {r['ankle']:18s} {float(r['eval_survival']):.3f} {float(r['fitness']):8.2f}  {tag}{sorted({p for p in (pairs or [])})}")
