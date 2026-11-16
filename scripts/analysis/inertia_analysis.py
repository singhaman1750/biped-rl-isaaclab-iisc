"""Effective inertia of every hinge of an MJCF about its own axis, as the pose varies.

Three inertias are computed for each hinge, all including its armature, and they are explained in
plans/w_eta_optimisation.md section 3.2.

    locked     M_jj, the hinge's distal subtree about its axis with every other hinge held, the
               parallel axis sum a.(R I R^T).a + m r_perp^2 of context/KScale.md section 4.
    free       1 / (M^-1)_jj, the inertia the hinge meets when every other degree of freedom,
               the floating base included, is free.
    reference  1 / max over poses of sum_k |(M^-1)_jk|, the largest inertia S_j for which
               lambda_max(S W) <= 1 at every pose by Gershgorin's theorem, W = (M^-1) over the hinges.

The locked inertia depends only on the hinges distal to j, so its minimum is swept on an exact grid
over them. The free and reference inertias depend on every hinge through the base, so they are swept
on a grid over each limb with the other limbs at zero, on random whole body poses, and refined by a
coordinate search from the worst poses found.

Usage:
    python3 scripts/analysis/inertia_analysis.py <model.xml> [--armature RE=VALUE ...] [--group NAME=RE ...]
        [--grid 13] [--samples 2000] [--dt 0.005] [--out DIR] [--check]

Writes inertia_dashboard.html, an interactive page with a slider per hinge, and inertia_sweep.json to
--out (default: beside the model). --armature overrides the MJCF armature of every hinge whose name
matches the regular expression, which is how the catalogue armature that apply_actuator_params writes
(co_optimisation/co_optimisation/utils/respawn.py:243) is supplied. --group aggregates hinges into
actuator groups for the summary table. --check compares the mass matrix with MuJoCo's mj_fullM when
mujoco is importable and the dashboard's JavaScript with this file when node is on the PATH.
"""

import argparse
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

import numpy as np


# --- model -----------------------------------------------------------------------------


def _vec(el, key, default):
    s = el.get(key)
    return [float(v) for v in s.split()] if s else list(default)


def _defaults(root):
    """class name -> joint attributes, inheriting along nested <default> elements."""
    out = {}

    def visit(el, inherited):
        attrs = dict(inherited)
        joint = el.find("joint")
        if joint is not None:
            attrs.update(joint.attrib)
        out[el.get("class", "main")] = attrs
        for child in el.findall("default"):
            visit(child, attrs)

    for d in root.findall("default"):
        visit(d, {})
    return out


def load_model(source, armature=()):
    """Kinematic tree of an MJCF, a path or the document's text, as plain lists, the form the dashboard's JavaScript reads."""
    root = ET.fromstring(source) if source.lstrip().startswith("<") else ET.parse(source).getroot()
    defaults = _defaults(root)
    bodies, floating = [], False

    def visit(el, parent):
        nonlocal floating
        hinges = []
        for j in el:
            if j.tag == "freejoint" or (j.tag == "joint" and j.get("type") == "free"):
                floating = floating or parent == -1
                continue
            if j.tag != "joint" or j.get("type", "hinge") != "hinge":
                continue
            attrs = dict(defaults.get(j.get("class", "main"), {}))
            attrs.update(j.attrib)
            name = attrs["name"]
            arm = float(attrs.get("armature", 0.0))
            for pattern, value in armature:
                if re.fullmatch(pattern, name):
                    arm = value
            axis = np.array(_vec(j, "axis", attrs.get("axis", "0 0 1").split()), float)
            hinges.append(dict(name=name, axis=list(axis / np.linalg.norm(axis)),
                               anchor=_vec(j, "pos", [0, 0, 0]),
                               range=[float(v) for v in attrs["range"].split()] if "range" in attrs else [-math.pi, math.pi],
                               armature=arm))
        inertial = el.find("inertial")
        mass, ipos, iquat, inertia = 0.0, [0, 0, 0], [1, 0, 0, 0], np.zeros((3, 3))
        if inertial is not None:
            mass, ipos, iquat = float(inertial.get("mass")), _vec(inertial, "pos", [0, 0, 0]), _vec(inertial, "quat", [1, 0, 0, 0])
            if inertial.get("fullinertia"):
                xx, yy, zz, xy, xz, yz = _vec(inertial, "fullinertia", [0] * 6)
                inertia = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            else:
                inertia = np.diag(_vec(inertial, "diaginertia", [0, 0, 0]))
        bodies.append(dict(name=el.get("name", f"body{len(bodies)}"), parent=parent, pos=_vec(el, "pos", [0, 0, 0]),
                           quat=_vec(el, "quat", [1, 0, 0, 0]), hinges=hinges, mass=mass, ipos=ipos, iquat=iquat,
                           inertia=inertia.tolist()))
        me = len(bodies) - 1
        for child in el.findall("body"):
            visit(child, me)

    for body in root.find("worldbody").findall("body"):
        visit(body, -1)
    names = [h["name"] for b in bodies for h in b["hinges"]]
    return dict(bodies=bodies, floating=floating, joints=names,
                ranges={h["name"]: h["range"] for b in bodies for h in b["hinges"]},
                armature={h["name"]: h["armature"] for b in bodies for h in b["hinges"]})


def _rot(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _axis_angle(axis, angle):
    a = np.asarray(axis, float)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * k @ k


# --- mechanics -------------------------------------------------------------------------


def kinematics(model, q):
    """World frame of every body and hinge at the pose *q* (hinge name -> angle, others zero)."""
    frames, hinges = [], []
    for b in model["bodies"]:
        R0, p0, chain = (np.eye(3), np.zeros(3), []) if b["parent"] < 0 else frames[b["parent"]][:3]
        p, R, chain = p0 + R0 @ np.array(b["pos"]), R0 @ _rot(b["quat"]), list(chain)
        for h in b["hinges"]:
            a, c = R @ np.array(h["axis"]), p + R @ np.array(h["anchor"])
            chain.append(len(hinges))
            hinges.append((h["name"], a, c))
            Rj = _axis_angle(h["axis"], q.get(h["name"], 0.0))
            p, R = p + R @ (np.array(h["anchor"]) - Rj @ np.array(h["anchor"])), R @ Rj
        Ri = R @ _rot(b["iquat"])
        frames.append((R, p, chain, p + R @ np.array(b["ipos"]), Ri @ np.array(b["inertia"]) @ Ri.T))
    return frames, hinges


def mass_matrix(model, q, floating=None):
    """Joint space mass matrix, base first when floating, armature on the hinge diagonal.

    The base velocity is the world linear velocity of the root origin and the world angular velocity,
    a choice that leaves the hinge block of the inverse unchanged (section 3.2 of the plan).
    """
    floating = model["floating"] if floating is None else floating
    frames, hinges = kinematics(model, q)
    nb, n = (6 if floating else 0), len(hinges)
    M = np.zeros((nb + n, nb + n))
    p_root = frames[0][1]
    for b, (R, p, chain, com, inertia) in zip(model["bodies"], frames):
        Jv, Jw = np.zeros((3, nb + n)), np.zeros((3, nb + n))
        if floating:
            r = com - p_root
            Jv[:, :3] = np.eye(3)
            Jv[:, 3:6] = -np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            Jw[:, 3:6] = np.eye(3)
        for k in chain:
            _, a, c = hinges[k]
            Jv[:, nb + k], Jw[:, nb + k] = np.cross(a, com - c), a
        M += b["mass"] * Jv.T @ Jv + Jw.T @ inertia @ Jw
    M[nb:, nb:] += np.diag([model["armature"][h[0]] for h in hinges])
    return M, nb


def inertias(model, q):
    """(locked, free, row sum of |W|, W) over the hinges at the pose *q*."""
    M, nb = mass_matrix(model, q)
    W = np.linalg.inv(M)[nb:, nb:]
    return np.diag(M)[nb:].copy(), 1.0 / np.diag(W), np.abs(W).sum(1), W


def locked_breakdown(model, q, joint):
    """Per distal body (name, own a.(R I R^T).a, m r_perp^2, contribution) about *joint*'s axis."""
    frames, hinges = kinematics(model, q)
    k = [h[0] for h in hinges].index(joint)
    _, a, c = hinges[k]
    rows = []
    for b, (R, p, chain, com, inertia) in zip(model["bodies"], frames):
        if k in chain and b["mass"] > 0:
            r = com - c
            r_perp = r - a * (a @ r)
            own, par = float(a @ inertia @ a), float(b["mass"] * r_perp @ r_perp)
            rows.append((b["name"], own, par, own + par))
    return rows


def distal(model, joint):
    """Hinges whose angle changes the locked inertia of *joint*, those strictly below it."""
    frames, hinges = kinematics(model, {})
    k = [h[0] for h in hinges].index(joint)
    return sorted({hinges[i][0] for f in frames for i in f[2] if k in f[2] and i > k}, key=model["joints"].index)


def limbs(model):
    """Hinge sets of the subtrees hanging from the root body."""
    out = []
    for i, b in enumerate(model["bodies"]):
        if b["parent"] == 0 or (b["parent"] == -1 and i > 0):
            names, stack = [], [i]
            while stack:
                j = stack.pop()
                names += [h["name"] for h in model["bodies"][j]["hinges"]]
                stack += [c for c, cb in enumerate(model["bodies"]) if cb["parent"] == j]
            if names:
                out.append(sorted(names, key=model["joints"].index))
    return out


# --- sweep -----------------------------------------------------------------------------


def _grid(model, names, n):
    return itertools.product(*[np.linspace(*model["ranges"][j], n) for j in names])


def sweep(model, grid=13, samples=2000, refine=4, seed=0):
    """Minimum locked and free inertia, and the reference inertia, of every hinge over the pose space."""
    joints, idx = model["joints"], {j: i for i, j in enumerate(model["joints"])}
    out = {j: dict(locked_min=math.inf, locked_max=0.0) for j in joints}

    # Locked: exact grid over the distal hinges of each hinge, the only angles it depends on.
    for j in joints:
        names = distal(model, j)
        for values in _grid(model, names, grid):
            q = dict(zip(names, values))
            v = mass_matrix(model, q, floating=False)[0][idx[j], idx[j]]
            if v < out[j]["locked_min"]:
                out[j]["locked_min"], out[j]["locked_pose"] = float(v), q
            out[j]["locked_max"] = max(out[j]["locked_max"], float(v))

    # Free and reference: limb grids, random whole body poses, then coordinate refinement.
    worst_free, worst_rows = {j: (math.inf, {}) for j in joints}, {j: (0.0, {}) for j in joints}

    def visit(q):
        _, free, rows, _ = inertias(model, q)
        for j in joints:
            if free[idx[j]] < worst_free[j][0]:
                worst_free[j] = (float(free[idx[j]]), dict(q))
            if rows[idx[j]] > worst_rows[j][0]:
                worst_rows[j] = (float(rows[idx[j]]), dict(q))

    coarse = max(3, (grid + 1) // 2)
    for limb in limbs(model):
        for values in _grid(model, limb, coarse):
            visit(dict(zip(limb, values)))
    rng = np.random.default_rng(seed)
    for _ in range(samples):
        visit({j: float(rng.uniform(*model["ranges"][j])) for j in joints})
    for j in joints:
        for target in ("free", "rows"):
            for _ in range(refine):
                q = dict((worst_free if target == "free" else worst_rows)[j][1])
                for k in joints:
                    for v in np.linspace(*model["ranges"][k], grid):
                        visit({**q, k: float(v)})

    for j in joints:
        out[j]["free_min"], out[j]["free_pose"] = worst_free[j]
        out[j]["reference"], out[j]["reference_pose"] = 1.0 / worst_rows[j][0], worst_rows[j][1]
    return out


def reference_estimate(model, joints, grid=3, passes=3, line=7):
    """Fast estimate of the reference inertia of *joints*, all on the first limb, for one design.

    A grid over the first limb with the others at zero, then coordinate passes over every hinge from
    the worst pose of each joint. Within 2 per cent of sweep() on both families except the biped hip,
    which it overstates by 15 per cent (plans/w_eta_optimisation.md section 3.2.2).
    """
    names, idx = model["joints"], {j: i for i, j in enumerate(model["joints"])}
    worst = {j: (0.0, {}) for j in joints}

    def visit(q):
        rows = inertias(model, q)[2]
        for j in joints:
            if rows[idx[j]] > worst[j][0]:
                worst[j] = (float(rows[idx[j]]), dict(q))

    limb = limbs(model)[0]
    for values in _grid(model, limb, grid):
        visit(dict(zip(limb, values)))
    for _ in range(passes):
        for j in joints:
            for k in names:
                q = dict(worst[j][1])
                for v in np.linspace(*model["ranges"][k], line):
                    visit({**q, k: float(v)})
    return {j: 1.0 / worst[j][0] for j in joints}


def coupling(model, result, key, samples=2000, seed=1):
    """max over sampled poses of lambda_max(S W) with S the diagonal of result[j][key]."""
    S = np.sqrt(np.array([result[j][key] for j in model["joints"]]))
    rng, worst = np.random.default_rng(seed), 0.0
    poses = [result[j][p] for j in model["joints"] for p in ("free_pose", "reference_pose", "locked_pose")]
    poses += [{j: float(rng.uniform(*model["ranges"][j])) for j in model["joints"]} for _ in range(samples)]
    for q in poses:
        W = inertias(model, q)[3]
        worst = max(worst, float(np.linalg.eigvalsh(S[:, None] * W * S[None, :]).max()))
    return worst


# --- checks ----------------------------------------------------------------------------


def check_mujoco(path, model, poses=20, seed=2):
    """Largest relative error against MuJoCo of the hinge block of M, of (M^-1), and of the locked inertia summed
    from the projection formula of the plan's section 3.2.1, or None without mujoco."""
    try:
        import mujoco
    except ImportError:
        return None
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    dofs = [m.jnt_dofadr[m.joint(j).id] for j in model["joints"]]
    for j, k in zip(model["joints"], dofs):
        m.dof_armature[k] = model["armature"][j]
    rng, worst = np.random.default_rng(seed), [0.0, 0.0, 0.0]
    for _ in range(poses):
        q = {j: float(rng.uniform(*model["ranges"][j])) for j in model["joints"]}
        d.qpos[:] = m.qpos0
        for j in model["joints"]:
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = q[j]
        mujoco.mj_forward(m, d)
        Mm = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, d, Mm)
        A, Wm = Mm[np.ix_(dofs, dofs)], np.linalg.inv(Mm)[np.ix_(dofs, dofs)]
        M, nb = mass_matrix(model, q)
        W = np.linalg.inv(M)[nb:, nb:]
        worst[0] = max(worst[0], np.abs(M[nb:, nb:] - A).max() / np.abs(A).max())
        worst[1] = max(worst[1], np.abs(W - Wm).max() / np.abs(Wm).max())
        for j, k in zip(model["joints"], dofs):
            # The projection formula of the plan's section 3.2.1, summed body by body.
            lock = model["armature"][j] + sum(row[3] for row in locked_breakdown(model, q, j))
            worst[2] = max(worst[2], abs(lock - Mm[k, k]) / Mm[k, k])
    return worst


def check_trig_law(model, points=25):
    """Residual of the trigonometric law of section 3.2.1 of the plan, per (hinge, distal hinge) pair.

    With one distal hinge k turned through theta and every other hinge at zero, the locked inertia of
    hinge j is c0 + c1 cos theta + s1 sin theta + c2 cos 2 theta + s2 sin 2 theta, and the second
    harmonic vanishes when the two axes are parallel. Returns rows (j, k, parallel, residual of the
    degree the law predicts, relative to the mean).
    """
    _, hinges = kinematics(model, {})
    axes = {h[0]: h[1] for h in hinges}
    idx = {j: i for i, j in enumerate(model["joints"])}
    rows = []
    for j in model["joints"]:
        for k in distal(model, j):
            t = np.linspace(-math.pi, math.pi, points)
            y = np.array([mass_matrix(model, {k: float(v)}, floating=False)[0][idx[j], idx[j]] for v in t])
            parallel = abs(abs(axes[j] @ axes[k]) - 1.0) < 1e-9
            degree = 1 if parallel else 2
            A = np.column_stack([np.ones_like(t)] + [f(n * t) for n in range(1, degree + 1) for f in (np.cos, np.sin)])
            fit = A @ np.linalg.lstsq(A, y, rcond=None)[0]
            rows.append((j, k, parallel, float(np.abs(fit - y).max() / y.mean())))
    return rows


def check_js(model, poses=5, seed=3):
    """Largest relative error of the dashboard's JavaScript locked and free inertias, or None."""
    node = shutil.which("node")
    if node is None:
        return None
    rng = np.random.default_rng(seed)
    qs = [{j: float(rng.uniform(*model["ranges"][j])) for j in model["joints"]} for _ in range(poses)]
    src = JS + f"\nconst MODEL={json.dumps(_model_json(model))};\nconst QS={json.dumps(qs)};\n" \
        "console.log(JSON.stringify(QS.map(q=>{const r=inertias(MODEL,q);return [r.locked,r.free];})));"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(src)
    try:
        js = json.loads(subprocess.run([node, f.name], capture_output=True, text=True, check=True).stdout)
    finally:
        os.unlink(f.name)
    worst = 0.0
    for q, (locked, free) in zip(qs, js):
        l, fr, _, _ = inertias(model, q)
        worst = max(worst, np.abs(np.array(locked) - l).max() / l.max(), np.abs(np.array(free) - fr).max() / fr.max())
    return worst


# --- dashboard -------------------------------------------------------------------------

JS = r"""
function rot(q){let n=Math.hypot(...q),[w,x,y,z]=q.map(v=>v/n);return [[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]];}
function mm(A,B){return A.map(r=>B[0].map((_,j)=>r.reduce((s,v,k)=>s+v*B[k][j],0)));}
function mv(A,v){return A.map(r=>r[0]*v[0]+r[1]*v[1]+r[2]*v[2]);}
function tr(A){return A[0].map((_,j)=>A.map(r=>r[j]));}
function add(a,b){return a.map((v,i)=>v+b[i]);} function sub(a,b){return a.map((v,i)=>v-b[i]);}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
function cross(a,b){return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];}
function aa(a,t){const s=Math.sin(t),c=1-Math.cos(t),K=[[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]],K2=mm(K,K);return [0,1,2].map(i=>[0,1,2].map(j=>(i==j?1:0)+s*K[i][j]+c*K2[i][j]));}
function kin(model,q){const F=[],H=[];for(const b of model.bodies){let R,p,ch;if(b.parent<0){R=[[1,0,0],[0,1,0],[0,0,1]];p=[0,0,0];ch=[];}else{const f=F[b.parent];R=f.R;p=f.p;ch=f.chain;}
 p=add(p,mv(R,b.pos));R=mm(R,rot(b.quat));ch=ch.slice();
 for(const h of b.hinges){const a=mv(R,h.axis),c=add(p,mv(R,h.anchor));ch.push(H.length);H.push({name:h.name,a,c});const Rj=aa(h.axis,q[h.name]||0);p=add(p,mv(R,sub(h.anchor,mv(Rj,h.anchor))));R=mm(R,Rj);}
 const Ri=mm(R,rot(b.iquat));F.push({R,p,chain:ch,com:add(p,mv(R,b.ipos)),I:mm(mm(Ri,b.inertia),tr(Ri))});}return {F,H};}
function massMatrix(model,q){const {F,H}=kin(model,q),nb=model.floating?6:0,n=H.length,N=nb+n,M=Array.from({length:N},()=>new Array(N).fill(0)),p0=F[0].p;
 model.bodies.forEach((b,i)=>{const f=F[i],Jv=[0,1,2].map(()=>new Array(N).fill(0)),Jw=[0,1,2].map(()=>new Array(N).fill(0));
  if(nb){const r=sub(f.com,p0);for(let k=0;k<3;k++){Jv[k][k]=1;Jw[k][3+k]=1;}const X=[[0,-r[2],r[1]],[r[2],0,-r[0]],[-r[1],r[0],0]];for(let k=0;k<3;k++)for(let l=0;l<3;l++)Jv[k][3+l]=-X[k][l];}
  for(const k of f.chain){const v=cross(H[k].a,sub(f.com,H[k].c));for(let r=0;r<3;r++){Jv[r][nb+k]=v[r];Jw[r][nb+k]=H[k].a[r];}}
  const IJ=mm(f.I,Jw);for(let r=0;r<N;r++)for(let c=0;c<N;c++){let s=0;for(let k=0;k<3;k++)s+=b.mass*Jv[k][r]*Jv[k][c]+Jw[k][r]*IJ[k][c];M[r][c]+=s;}});
 H.forEach((h,k)=>{M[nb+k][nb+k]+=model.armature[h.name];});return {M,nb,F,H};}
function inv(A){const n=A.length,M=A.map((r,i)=>r.concat(r.map((_,j)=>i==j?1:0)));for(let i=0;i<n;i++){let p=i;for(let r=i+1;r<n;r++)if(Math.abs(M[r][i])>Math.abs(M[p][i]))p=r;[M[i],M[p]]=[M[p],M[i]];const d=M[i][i];for(let j=0;j<2*n;j++)M[i][j]/=d;for(let r=0;r<n;r++)if(r!=i){const f=M[r][i];for(let j=0;j<2*n;j++)M[r][j]-=f*M[i][j];}}return M.map(r=>r.slice(n));}
function inertias(model,q){const {M,nb,F,H}=massMatrix(model,q),Mi=inv(M),n=H.length,W=[];for(let i=0;i<n;i++)W.push(Mi[nb+i].slice(nb));
 return {locked:H.map((_,k)=>M[nb+k][nb+k]),free:H.map((_,k)=>1/W[k][k]),W,F,H};}
function lamMax(A){let v=A.map(()=>1),l=0;for(let it=0;it<300;it++){const w=A.map(r=>r.reduce((s,x,j)=>s+x*v[j],0)),n=Math.hypot(...w);l=n/Math.hypot(...v);v=w.map(x=>x/n);}return l;}
"""

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Joint Inertia Explorer</title><style>
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6e6e73;--ln:#d2d2d7;--a:#2f6fdf;--b:#d9822b;--c:#2a9d6a;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#e8e8ea;--mut:#9a9aa0;--ln:#3a3a3e;--a:#6c9cf0;--b:#f0a35c;--c:#4cc38a;--bad:#ef6b5b}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 4px}.mut{color:var(--mut)}.wrap{display:flex;gap:20px;flex-wrap:wrap}
.side{flex:0 0 300px}.main{flex:1 1 560px;min-width:0}.lim{border:1px solid var(--ln);border-radius:8px;padding:8px;margin:8px 0}
label{display:block;font-size:12px;margin-top:6px}input[type=range]{width:100%}button,select,input[type=number]{font:inherit;color:var(--fg);background:transparent;border:1px solid var(--ln);border-radius:6px;padding:3px 8px;margin:2px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid var(--ln);padding:3px 6px;text-align:right}td:first-child,th:first-child{text-align:left}
svg text{fill:var(--fg);font-size:11px}.ok{color:var(--c)}.bad{color:var(--bad)}
</style></head><body>
<h1>Joint Inertia Explorer</h1><div class="mut" id="src"></div>
<div class="wrap"><div class="side"><div><button id="zero">All zero</button><select id="poses"></select></div><div id="sliders"></div></div>
<div class="main"><svg id="bars" width="100%" height="320"></svg>
<div class="mut">Bars, log scale, kg m². Locked: the hinge's distal subtree with every other hinge held (parallel axis theorem). Free: every other degree of freedom free, the floating base included. Ticks: sweep minimum of the locked inertia (|) and reference inertia (◆).</div>
<div style="margin:10px 0">Servo at the reference inertia, f<sub>n</sub> <input id="f" type="number" value="10" step="0.5" style="width:60px"> Hz, ζ <input id="z" type="number" value="0.85" step="0.05" style="width:60px">,
Δt <span id="dt"></span> s. <span id="stab"></span></div>
<div class="wrap"><svg id="view" width="420" height="260"></svg><div style="flex:1;min-width:260px"><div>Breakdown about <select id="joint"></select></div><table id="bd"></table></div></div></div></div>
<script>__JS__
const MODEL=__MODEL__,SWEEP=__SWEEP__,DT=__DT__;
const J=MODEL.joints,q={};J.forEach(j=>q[j]=0);
document.getElementById('src').textContent=MODEL.source+(MODEL.floating?', floating base':', fixed base');
document.getElementById('dt').textContent=DT;
const sl=document.getElementById('sliders');
MODEL.limbs.forEach((L,i)=>{const d=document.createElement('div');d.className='lim';d.innerHTML='<b>Limb '+(i+1)+'</b>';
 L.forEach(j=>{const [lo,hi]=MODEL.ranges[j],l=document.createElement('label');l.innerHTML=j+' <span id="v_'+j+'"></span><input type="range" min="'+lo+'" max="'+hi+'" step="0.005" value="0" id="s_'+j+'">';d.appendChild(l);});sl.appendChild(d);});
J.forEach(j=>document.getElementById('s_'+j).addEventListener('input',e=>{q[j]=+e.target.value;draw();}));
const sel=document.getElementById('joint'),ps=document.getElementById('poses');J.forEach(j=>sel.add(new Option(j,j)));sel.onchange=draw;
ps.add(new Option('Load a sweep pose…',''));J.forEach(j=>['locked_pose','free_pose','reference_pose'].forEach(k=>ps.add(new Option(j+', '+k.replace('_pose',''),j+'|'+k))));
ps.onchange=()=>{if(!ps.value)return;const [j,k]=ps.value.split('|');J.forEach(n=>q[n]=0);Object.assign(q,SWEEP[j][k]);sync();sel.value=j;draw();};
document.getElementById('zero').onclick=()=>{J.forEach(n=>q[n]=0);sync();draw();};
['f','z'].forEach(i=>document.getElementById(i).oninput=draw);
function sync(){J.forEach(j=>document.getElementById('s_'+j).value=q[j]);}
function fmt(x){return x<0.01?x.toExponential(2):x.toFixed(4);}
function draw(){const r=inertias(MODEL,q);J.forEach(j=>document.getElementById('v_'+j).textContent=q[j].toFixed(3));
 const svg=document.getElementById('bars'),W=svg.clientWidth||700,Hh=320,pad=150,all=r.locked.concat(r.free,J.map(j=>SWEEP[j].reference)),lo=Math.log10(Math.min(...all))-0.2,hi=Math.log10(Math.max(...all))+0.1,x=v=>pad+(W-pad-20)*(Math.log10(v)-lo)/(hi-lo),row=(Hh-30)/J.length;
 let s='';for(let d=Math.ceil(lo);d<=hi;d++){s+=`<line x1="${x(10**d)}" x2="${x(10**d)}" y1="0" y2="${Hh-20}" stroke="var(--ln)"/><text x="${x(10**d)}" y="${Hh-6}" text-anchor="middle">1e${d}</text>`;}
 J.forEach((j,k)=>{const y=k*row+4,h=row/2-3;s+=`<text x="${pad-6}" y="${y+row/2}" text-anchor="end">${j}</text>`;
  s+=`<rect x="${pad}" y="${y}" width="${Math.max(1,x(r.locked[k])-pad)}" height="${h}" fill="var(--a)"><title>locked ${fmt(r.locked[k])}</title></rect>`;
  s+=`<rect x="${pad}" y="${y+h+2}" width="${Math.max(1,x(r.free[k])-pad)}" height="${h}" fill="var(--b)"><title>free ${fmt(r.free[k])}</title></rect>`;
  s+=`<line x1="${x(SWEEP[j].locked_min)}" x2="${x(SWEEP[j].locked_min)}" y1="${y-1}" y2="${y+h+1}" stroke="var(--fg)" stroke-width="2"/>`;
  s+=`<text x="${x(SWEEP[j].reference)}" y="${y+2*h+2}" text-anchor="middle">◆</text>`;});
 s+=`<rect x="${W-170}" y="2" width="10" height="10" fill="var(--a)"/><text x="${W-156}" y="11">locked</text><rect x="${W-100}" y="2" width="10" height="10" fill="var(--b)"/><text x="${W-86}" y="11">free</text>`;svg.innerHTML=s;
 const w=2*Math.PI*(+document.getElementById('f').value),z=+document.getElementById('z').value,S=J.map(j=>SWEEP[j].reference),X=r.W.map((row,i)=>row.map((v,k)=>Math.sqrt(S[i])*v*Math.sqrt(S[k])));
 const mu=lamMax(X),cond=mu*((w*DT)**2+4*z*w*DT);document.getElementById('stab').innerHTML=`μ = λ<sub>max</sub>(S W) = ${mu.toFixed(3)}, μ(a + 2b) = ${cond.toFixed(3)} <span class="${cond<4?'ok':'bad'}">${cond<4?'< 4, stable':'≥ 4, not guaranteed'}</span>, μ b = ${(mu*2*z*w*DT).toFixed(3)}`;
 const v=document.getElementById('view');let t='';const P=[[0,2,'side, x–z',10],[1,2,'front, y–z',215]];const pts=r.F.map(f=>f.p),cs=r.F.map(f=>f.com);
 const ext=Math.max(...pts.concat(cs).map(p=>Math.max(Math.abs(p[0]),Math.abs(p[1]),Math.abs(p[2]))))*1.1;
 for(const [u,vv,lab,ox] of P){const X=p=>ox+97+97*p[u]/ext,Y=p=>125-115*p[vv]/ext;t+=`<text x="${ox+4}" y="12" class="mut">${lab}</text>`;
  MODEL.bodies.forEach((b,i)=>{if(b.parent>=0)t+=`<line x1="${X(pts[b.parent])}" y1="${Y(pts[b.parent])}" x2="${X(pts[i])}" y2="${Y(pts[i])}" stroke="var(--mut)" stroke-width="2"/>`;
   if(b.mass>0)t+=`<circle cx="${X(cs[i])}" cy="${Y(cs[i])}" r="${2+3*Math.sqrt(b.mass)}" fill="var(--a)" opacity="0.5"><title>${b.name} ${b.mass} kg</title></circle>`;});
  r.H.forEach(h=>{if(h.name==sel.value)t+=`<line x1="${X(h.c)}" y1="${Y(h.c)}" x2="${X(add(h.c,h.a.map(c=>c*ext*0.3)))}" y2="${Y(add(h.c,h.a.map(c=>c*ext*0.3)))}" stroke="var(--b)" stroke-width="3"/>`;});}
 v.innerHTML=t;
 const k=J.indexOf(sel.value),hk=r.H[k];let rows='<tr><th>body</th><th>a·(R I Rᵀ)·a</th><th>m r⊥²</th><th>sum</th></tr>',tot=0;
 MODEL.bodies.forEach((b,i)=>{const f=r.F[i];if(!f.chain.includes(k)||!b.mass)return;const rr=sub(f.com,hk.c),rp=sub(rr,hk.a.map(c=>c*dot(hk.a,rr))),own=dot(hk.a,mv(f.I,hk.a)),par=b.mass*dot(rp,rp);tot+=own+par;rows+=`<tr><td>${b.name}</td><td>${fmt(own)}</td><td>${fmt(par)}</td><td>${fmt(own+par)}</td></tr>`;});
 const arm=MODEL.armature[sel.value];rows+=`<tr><td>armature</td><td></td><td></td><td>${fmt(arm)}</td></tr><tr><td><b>locked</b></td><td></td><td></td><td><b>${fmt(tot+arm)}</b></td></tr><tr><td>free</td><td></td><td></td><td>${fmt(r.free[k])}</td></tr><tr><td>sweep min locked, reference</td><td></td><td></td><td>${fmt(SWEEP[sel.value].locked_min)}, ${fmt(SWEEP[sel.value].reference)}</td></tr>`;
 document.getElementById('bd').innerHTML=rows;}
window.onresize=draw;draw();
</script></body></html>"""


def _model_json(model):
    return {k: model[k] for k in ("bodies", "floating", "joints", "ranges", "armature")}


def write_dashboard(path, model, result, dt, source):
    m = dict(_model_json(model), limbs=limbs(model), source=source)
    page = PAGE.replace("__JS__", JS).replace("__MODEL__", json.dumps(m)).replace("__SWEEP__", json.dumps(result))
    with open(path, "w") as f:
        f.write(page.replace("__DT__", repr(dt)))


# --- main ------------------------------------------------------------------------------


def _pair(s, cast):
    k, v = s.rsplit("=", 1)
    return k, cast(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mjcf")
    p.add_argument("--armature", action="append", default=[], type=lambda s: _pair(s, float), metavar="RE=VALUE")
    p.add_argument("--group", action="append", default=[], type=lambda s: _pair(s, str), metavar="NAME=RE")
    p.add_argument("--grid", type=int, default=13)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--dt", type=float, default=0.005)
    p.add_argument("--out", default=None)
    p.add_argument("--check", action="store_true")
    args = p.parse_args()

    model = load_model(args.mjcf, args.armature)
    out = args.out or os.path.dirname(os.path.abspath(args.mjcf))
    os.makedirs(out, exist_ok=True)
    print(f"{args.mjcf}\n  {len(model['joints'])} hinges, {'floating' if model['floating'] else 'fixed'} base")
    if args.check:
        mj, js = check_mujoco(args.mjcf, model), check_js(model)
        print(f"  mj_fullM: {'mujoco absent' if mj is None else f'M {mj[0]:.1e}, M^-1 {mj[1]:.1e}, projection formula {mj[2]:.1e}'} relative error; "
              f"JavaScript: {'node absent' if js is None else f'{js:.1e}'} relative error")
        law = check_trig_law(model)
        for parallel in (True, False):
            sel = [r for r in law if r[2] == parallel]
            if sel:
                print(f"  trigonometric law, {'parallel axes, degree 1' if parallel else 'skew axes, degree 2'}: "
                      f"{len(sel)} pairs, largest relative residual {max(r[3] for r in sel):.1e}")

    result = sweep(model, args.grid, args.samples)
    for key in ("locked_min", "reference"):
        print(f"  lambda_max(S W) with S = {key}: {coupling(model, result, key, args.samples):.3f}")
    groups = args.group or [(j, re.escape(j)) for j in model["joints"]]
    first = [j for j in limbs(model)[0]]
    fast = reference_estimate(model, first)
    print("  fast reference estimate / sweep: " + ", ".join(f"{j} {fast[j] / result[j]['reference']:.3f}" for j in first))
    zero = dict(zip(model["joints"], inertias(model, {})[0]))
    print("  group | armature | locked nominal | locked min | locked max | free min | reference")
    for name, pattern in groups:
        js = [j for j in model["joints"] if re.fullmatch(pattern, j)]
        if js:
            g = lambda k: min(result[j][k] for j in js)
            print(f"  {name} | {min(model['armature'][j] for j in js):.4f} | {min(zero[j] for j in js):.4f} | {g('locked_min'):.4f} | "
                  f"{max(result[j]['locked_max'] for j in js):.4f} | {g('free_min'):.4f} | {g('reference'):.4f}")
    with open(os.path.join(out, "inertia_sweep.json"), "w") as f:
        json.dump(result, f, indent=1)
    write_dashboard(os.path.join(out, "inertia_dashboard.html"), model, result, args.dt, os.path.basename(args.mjcf))
    print(f"  wrote {out}/inertia_dashboard.html and inertia_sweep.json")


if __name__ == "__main__":
    main()
