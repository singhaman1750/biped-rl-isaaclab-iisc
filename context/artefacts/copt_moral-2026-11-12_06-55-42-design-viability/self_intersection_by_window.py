"""Count, per population window, the designs whose bodies overlap at the nominal pose.

Each row of designs/design_parameters_<it>.csv is rebuilt as the generator builds it, its Isaac Lab
MJCF loaded in MuJoCo with every joint at zero and the base two metres up, and the body pairs in
contact listed. The predicted logged episode length assumes an overlapping design ends its episodes
every step and every other environment after 850 steps. Run where mujoco and numpy are installed:
    python3 -I self_intersection_by_window.py <repo_root> <run_dir> <eplen_file>
<eplen_file> holds lines "<window_start> <logged Train/mean_episode_length>".
"""
import csv, glob, importlib.util, os, sys, types
import mujoco

root, run, eplen = sys.argv[1:4]
sys.path.insert(0, root)
for n in ("isaaclab", "isaaclab.sim", "cma", "cmaes"):
    sys.modules.setdefault(n, types.ModuleType(n))
conv = types.ModuleType("isaaclab.sim.converters")
conv.MjcfConverter = conv.MjcfConverterCfg = object
sys.modules["isaaclab.sim.converters"] = conv
spec = importlib.util.spec_from_file_location("U", os.path.join(root, "co_optimisation/co_optimisation/runners/usd_generator.py"))
U = importlib.util.module_from_spec(spec); sys.modules["U"] = U; spec.loader.exec_module(U)

fam = U.FAMILIES["biped"]; B = fam.module.builder; DS = fam.module.generate.DesignSpec
base = fam.baseline(); dx = fam.chain(base, 0.0, 0.0, 0.0)[0]
logged = dict(line.split() for line in open(eplen))

def spec_of(r):
    f = base.to_dict()
    f["thigh_length"] = round(f["thigh_length"] * float(r["thigh_length_scale"]), 2)
    f["shank_length"] = round(f["shank_length"] * float(r["shank_length_scale"]), 2)
    f["thigh_bend_offset"] += float(r["thigh_bend_offset"]); f["knee_bend_offset"] += float(r["knee_bend_offset"])
    for g, field in fam.spec_fields.items():
        f[field] = r[g + "_actuator"]
    return DS.from_dict({**f, **U.solve_nominal_pose(fam, DS.from_dict(f), dx)})

def overlaps(s):
    m = mujoco.MjModel.from_xml_string(B.build_xml(s, for_isaaclab=True)); d = mujoco.MjData(m)
    d.qpos[:] = 0; d.qpos[2] = 2.0; d.qpos[3] = 1.0; mujoco.mj_forward(m, d)
    return sorted({tuple(sorted((m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name)))
                   for c in d.contact[:d.ncon]})

print("window      designs  overlapping  predicted_len  logged_len  hip actuators of overlapping designs  pairs")
for f in sorted(glob.glob(os.path.join(run, "designs/design_parameters_*.csv")), key=lambda p: int(p.split("_")[-1][:-4])):
    it = int(f.split("_")[-1][:-4]); rows = list(csv.DictReader(open(f)))
    bad = [(r, overlaps(spec_of(r))) for r in rows]; bad = [(r, p) for r, p in bad if p]
    n, k = len(rows), len(bad); per = 6400 // n
    pred = 6400 / (k * per + (6400 - k * per) / 850.0)
    hips = sorted({r["hip_actuator"] for r, _ in bad}); pairs = sorted({q for _, p in bad for q in p})
    print(f"{it-399:5d}-{it:<5d} {n:6d} {k:10d} {pred:14.1f} {float(logged.get(str(it-399), 'nan')):11.1f}  {hips}  {pairs}")
