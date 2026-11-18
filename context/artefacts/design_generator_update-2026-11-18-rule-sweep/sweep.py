import sys, itertools, collections, numpy as np, mujoco, time
from multiprocessing import Pool
sys.path.insert(0, sys.argv[1]); fam_sel = sys.argv[2]
from dataclasses import replace
from design_generator.biped import generate as bg, builder as bb, actuator as ba
from design_generator.quadruped import generate as qg, builder as qb, actuator as qa
F = {'biped': (bg, bb, ba, ('abad_R_Joint','abad_L_Joint'), 'base_Link'),
     'quadruped': (qg, qb, qa, ('FR_hip_joint','FL_hip_joint'), 'trunk')}
gen, mod, act, joints, trunk = F[fam_sel]
base = gen.load_design_spec(); names = [a.name for a in act.ACTUATOR_CATALOG]
CASES = {'old': (False, False, False), 'R7': (True, False, False), 'R7+knee': (True, False, True), 'R7+R8': (True, True, False), 'R7+knee+R8': (True, True, True)}
TOL = -1e-4
lo, hi = mod.ABAD_RANGE; grid = np.unique(np.concatenate([np.linspace(lo, hi, 9), [0.0]]))
poses = [(a, b) for a in grid for b in grid]
def kind(n):
    if n == trunk: return 'trunk'
    for k in ('abad', 'hip', 'knee', 'thigh'):
        if n.startswith(k + '_'): return k
    return 'lower'
def side(n): return n.split('_')[1]
def work(combo):
    ab, hp, kn = combo; out = {}
    for cname, (fl, zr, kn_) in CASES.items():
        mod.HIP_FLUSH, mod.HIP_Z, mod.HIP_KNEE = fl, zr, kn_
        spec = replace(base, act_abduction=ab, act_hip=hp, act_knee=kn)
        m = mujoco.MjModel.from_xml_string(mod.build_xml(spec)); d = mujoco.MjData(m)
        bn = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or '' for i in range(m.nbody)]
        adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in joints]
        seen = collections.Counter()
        for a1, a2 in poses:
            d.qpos[:] = 0; d.qpos[adr[0]] = a1; d.qpos[adr[1]] = a2
            mujoco.mj_kinematics(m, d); mujoco.mj_collision(m, d)
            found = set()
            for c in d.contact[:d.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if b1 == 0 or b2 == 0 or c.dist >= TOL: continue
                n1, n2 = bn[b1], bn[b2]; k1, k2 = kind(n1), kind(n2)
                if k1 == 'trunk' or k2 == 'trunk':
                    found.add('trunk-' + (k2 if k1 == 'trunk' else k1))
                else:
                    found.add(('cross-' if side(n1) != side(n2) else 'same-') + '-'.join(sorted((k1, k2))))
            outw = (a1 >= 0 and a2 >= 0) if fam_sel == 'biped' else (a1 <= 0 and a2 >= 0)
            for f in found:
                seen[f] += 1
                if outw: seen[f + '|outward'] += 1
                if a1 == 0 and a2 == 0: seen[f + '|nominal'] += 1
        out[cname] = dict(seen)
    return out
if __name__ == '__main__':
    combos = list(itertools.product(names, repeat=3)); t0 = time.time()
    with Pool(32) as p: R = p.map(work, combos, chunksize=8)
    des = {c: collections.Counter() for c in CASES}; pos = {c: collections.Counter() for c in CASES}
    for r in R:
        for c, s in r.items():
            for k, v in s.items(): des[c][k] += 1; pos[c][k] += v
            if s: des[c]['ANY'] += 1
    print(fam_sel, 'designs', len(combos), 'poses/design', len(poses), 'seconds', round(time.time() - t0))
    keys = sorted({k for c in des.values() for k in c})
    print('designs with contact'); print('%-24s' % 'category', *['%7s' % c for c in CASES])
    for k in keys: print('%-24s' % k, *['%7d' % des[c][k] for c in CASES])
    print('design-pose contact records'); 
    for k in keys:
        if k != 'ANY': print('%-24s' % k, *['%7d' % pos[c][k] for c in CASES])
