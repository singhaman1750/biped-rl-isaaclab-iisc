import sys, glob, collections, mujoco, numpy as np
d0, N = sys.argv[1], int(sys.argv[2])
rng = np.random.default_rng(0)
cross = collections.Counter(); same = collections.Counter(); rows = []
def side(n): return "R" if "_R_" in n else ("L" if "_L_" in n else "B")
def norm(n): return n.replace("_R_", "_S_").replace("_L_", "_S_")
for f in sorted(glob.glob(d0 + "/individual_*.xml")):
    m = mujoco.MjModel.from_xml_path(f); d = mujoco.MjData(m); mujoco.mj_kinematics(m, d)
    col = [g for g in range(m.ngeom) if m.geom_group[g] == 3]
    cand = [(a, b) for i, a in enumerate(col) for b in col[i+1:]
            if m.geom_bodyid[a] != m.geom_bodyid[b] and m.body_parentid[m.geom_bodyid[a]] != m.geom_bodyid[b] and m.body_parentid[m.geom_bodyid[b]] != m.geom_bodyid[a]]
    # nominal min cross-leg distance
    xl = [(a, b) for a, b in cand if {side(m.body(m.geom_bodyid[a]).name), side(m.body(m.geom_bodyid[b]).name)} == {"R", "L"}]
    gap = min(mujoco.mj_geomDistance(m, d, a, b, 1.0, None) for a, b in xl)
    foot_y = d.xpos[m.body("foot_L_Link").id][1]
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] == 3]
    hits = 0
    for s in range(N):
        d.qpos[:] = m.qpos0; d.qpos[m.jnt_qposadr[hinge]] = rng.uniform(-0.2, 0.2, len(hinge)); mujoco.mj_kinematics(m, d)
        bad = False
        for a, b in cand:
            if mujoco.mj_geomDistance(m, d, a, b, 0.01, None) < -1e-3:
                na, nb = m.body(m.geom_bodyid[a]).name, m.body(m.geom_bodyid[b]).name
                k = tuple(sorted((norm(na), norm(nb))))
                (cross if side(na) != side(nb) else same)[k] += 1; bad = True
        hits += bad
    rows.append((f.split("/")[-1], foot_y, gap, hits / N))
r = np.array([(x[1], x[2], x[3]) for x in rows])
print(f"foot y (half track) mean {r[:,0].mean()*1000:.1f} mm, min {r[:,0].min()*1000:.1f}, max {r[:,0].max()*1000:.1f}")
print(f"nominal cross-leg clearance mean {r[:,1].mean()*1000:.1f} mm, min {r[:,1].min()*1000:.1f}")
print(f"reset samples in collision {r[:,2].mean():.3f}; corr(gap, collision) {np.corrcoef(r[:,1], r[:,2])[0,1]:.2f}")
print("CROSS-LEG:", cross.most_common(8)); print("SAME-LEG:", same.most_common(8))
