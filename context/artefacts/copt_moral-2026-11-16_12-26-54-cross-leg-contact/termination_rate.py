import sys, glob, re, mujoco, numpy as np
rng = np.random.default_rng(2)
CUR = re.compile(r"^(thigh_.*|shank_.*|.*_act)$"); NEW = re.compile(r"^(thigh_.*|shank_.*|(abad|hip|knee)_.*_act)$")
def side(n): return "R" if "_R_" in n else ("L" if "_L_" in n else "B")
any_c = cur = new = n = 0
for f in sorted(glob.glob(sys.argv[1] + "/individual_*.xml")):
    m = mujoco.MjModel.from_xml_path(f); d = mujoco.MjData(m)
    col = [g for g in range(m.ngeom) if m.geom_group[g] == 3]
    xl = [(a, b) for i, a in enumerate(col) for b in col[i+1:] if {side(m.body(m.geom_bodyid[a]).name), side(m.body(m.geom_bodyid[b]).name)} == {"R", "L"}]
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == 3]
    for _ in range(int(sys.argv[2])):
        d.qpos[:] = m.qpos0; d.qpos[m.jnt_qposadr[hinge]] = rng.uniform(-0.2, 0.2, len(hinge)); mujoco.mj_kinematics(m, d)
        hit = [(m.body(m.geom_bodyid[a]).name, m.body(m.geom_bodyid[b]).name) for a, b in xl if mujoco.mj_geomDistance(m, d, a, b, 0.01, None) < -1e-3]
        n += 1; any_c += bool(hit)
        cur += any(CUR.match(x) or CUR.match(y) for x, y in hit)
        new += any(NEW.match(x) or NEW.match(y) for x, y in hit)
print(f"resets in cross-leg contact {any_c/n:.3f}; ending under current leg_contact {cur/n:.3f}; under abad|hip|knee only {new/n:.3f}")
