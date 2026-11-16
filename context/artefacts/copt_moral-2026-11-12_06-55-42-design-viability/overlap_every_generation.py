"""Overlap of the emitted MJCF of every design of every generation of the run, by hip actuator. Run with python3 -I <run_dir>, where mujoco and numpy are installed."""
import csv, os, sys, collections
import mujoco
run = sys.argv[1]
print("gen  iter   designs overlapping  hip actuators of the overlapping designs (count)")
tot = 0
for g in range(0, 51):
    it = 400 * (g + 1) - 1
    pf = f"{run}/designs/design_parameters_{it}.csv"
    if not os.path.exists(pf): continue
    rows = {int(r["individual"]): r for r in csv.DictReader(open(pf))}
    hips = collections.Counter(); n = 0
    for k, r in rows.items():
        path = f"{run}/copt_usds/gen_{g:04d}/individual_{k:04d}.xml"
        if not os.path.exists(path): continue
        m = mujoco.MjModel.from_xml_path(path); d = mujoco.MjData(m); d.qpos[:] = 0
        if m.nq >= 7: d.qpos[2] = 2.0; d.qpos[3] = 1.0
        mujoco.mj_forward(m, d)
        pairs = {tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name))) for c in d.contact[:d.ncon]}
        if pairs: n += 1; hips[r["hip_actuator"] + f" ({float(r['hip_actuator_diameter'])*1000:.0f} mm)"] += 1
    tot += n
    if g <= 14 or n: print(f"{g:3d} {it:6d} {len(rows):7d} {n:11d}  {dict(hips)}")
print("total overlapping designs over generations 0 to 50:", tot)
