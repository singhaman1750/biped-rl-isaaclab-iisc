import sys, glob, mujoco, numpy as np
rng = np.random.default_rng(1)
def side(n): return "R" if "_R_" in n else ("L" if "_L_" in n else "B")
def run(f, N=300):
    m = mujoco.MjModel.from_xml_path(f); d = mujoco.MjData(m)
    col = [g for g in range(m.ngeom) if m.geom_group[g] == 3]
    xl = [(a, b) for i, a in enumerate(col) for b in col[i+1:] if {side(m.body(m.geom_bodyid[a]).name), side(m.body(m.geom_bodyid[b]).name)} == {"R", "L"}]
    names = [m.joint(j).name for j in range(m.njnt)]
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == 3]
    def hit(q):
        d.qpos[:] = m.qpos0; d.qpos[m.jnt_qposadr[hinge]] = q; mujoco.mj_kinematics(m, d)
        return min(mujoco.mj_geomDistance(m, d, a, b, 1.0, None) for a, b in xl) < -1e-3
    abad = np.array(["abad" in m.joint(j).name for j in hinge])
    out = {}
    for label, mask in (("all", np.ones_like(abad)), ("abad only", abad), ("all but abad", ~abad)):
        out[label] = np.mean([hit(rng.uniform(-0.2, 0.2, len(hinge)) * mask) for _ in range(N)])
    # contact threshold with symmetric abad roll, others zero; find sign that moves legs together
    thr = {}
    for s in (+1, -1):
        for a in np.linspace(0, 0.4, 81):
            q = np.zeros(len(hinge)); q[abad] = s * a
            if hit(q): thr[s] = a; break
    return out, thr, [m.joint(j).name for j in hinge]
for f in sys.argv[1:]:
    out, thr, names = run(f)
    print(f.split("/")[-1], {k: round(v, 3) for k, v in out.items()}, "symmetric abad roll at first contact:", thr)
