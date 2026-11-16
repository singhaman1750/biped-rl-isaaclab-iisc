"""Coupled stability of the explicit servos over the pose space, the tables of plans/w_eta_optimisation.md sections 3.2 to 3.6.

Run on the host interpreter, which carries mujoco 3.16, from the repository root, as
    python3 -I context/artefacts/w_eta_optimisation-2026-10-27/validate_coupled.py > context/artefacts/w_eta_optimisation-2026-10-27/validate_coupled.txt
The sweeps biped/inertia_sweep.json and quadruped/inertia_sweep.json beside it were written by
scripts/analysis/inertia_analysis.py with --grid 13 --samples 2000 on the documents of validate_servo.py.
"""
import itertools
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "analysis"))
import inertia_analysis as ia  # noqa: E402

LOGS = Path("/ws/IsaacLab/logs/rsl_rl")
DT, ACTION_SCALE = 0.005, 0.25
# group -> (joint regex, armature, mu_s, mu_d, v_s, kp, kd, effort), catalogs/actuators.json as in validate_servo.py
FAMILIES = {
    "biped": (LOGS / "copt_moral/2026-10-26_04-55-55/copt_usds/gen_0008/individual_0000.xml", {
        "abad": ("abad_.*", 6.9e-5 * 81, 0.3, 0.02, 0.1, 55.0, 13.5, 60.0),
        "hip": ("hip_.*", 9.4e-5 * 81, 0.3, 0.02, 0.1, 80.0, 13.0, 60.0),
        "knee": ("knee_.*", 1.5e-4 * 81, 0.8, 0.02, 0.1, 60.0, 4.0, 90.0),
        "ankle": ("ankle_.*", 6.9e-5 * 81, 0.1, 0.02, 0.1, 10.0, 0.5, 40.0)}, "biped/inertia_sweep.json"),
    "quadruped": (LOGS / "quadruped_copt/2026-10-26_04-55-55/copt_usds/gen_0008/individual_0000.xml", {
        "abad": (".*_hip_joint", 0.01, 0.2, 0.02, 0.1, 40.0, 1.2, 23.622511),
        "hip": (".*_thigh_joint", 0.01, 0.2, 0.02, 0.1, 40.0, 1.1, 23.622511),
        "knee": (".*_calf_joint", 0.01, 0.2, 0.02, 0.1, 50.0, 0.9, 35.238)}, "quadruped/inertia_sweep.json"),
}
SCALES = {  # largest draw of randomize_actuator_gains, as in validate_servo.py
    "biped": {"abad": (1.091, 1.111), "hip": (1.125, 1.154), "knee": (1.167, 1.25), "ankle": (1.2, 1.2)},
    "quadruped": {"abad": (1.15, 1.2), "hip": (1.15, 1.2), "knee": (1.15, 1.2)},
}


def group_of(groups, joint):
    return next(g for g, v in groups.items() if __import__("re").fullmatch(v[0], joint))


def transition(W, K, D):
    n, I = len(K), np.eye(len(K))
    WK, WD = W * K[None, :], W * D[None, :]
    return np.block([[I - DT * DT * WK, DT * (I - DT * WD)], [-DT * WK, I - DT * WD]])


for family, (path, groups, sweep_file) in FAMILIES.items():
    arm = [(v[0], v[1]) for v in groups.values()]
    model = ia.load_model(str(path), arm)
    sweep = json.loads((HERE / sweep_file).read_text())
    joints = model["joints"]
    gj = [group_of(groups, j) for j in joints]
    ref_g = {g: min(sweep[j]["reference"] for j in joints if group_of(groups, j) == g) for g in groups}
    print(f"{family}: {path}")

    print("  fast reference estimate against the full sweep")
    for j, v in ia.reference_estimate(model, ia.limbs(model)[0]).items():
        print(f"    {j} | {v:.5f} | full {sweep[j]['reference']:.5f} | ratio {v / sweep[j]['reference']:.3f}")

    print("  group | reference | catalogue f_n Hz, zeta | 10 Hz, 0.85 kp, kd | kp*0.25/effort | authority f Hz")
    for g, (_, a, mu_s, mu_d, v_s, kp, kd, tau) in groups.items():
        S = ref_g[g]
        w = 2 * math.pi * 10.0
        print(f"    {g} | {S:.4f} | {math.sqrt(kp / S) / (2 * math.pi):.2f}, {kd / (2 * math.sqrt(kp * S)):.2f} | "
              f"{S * w * w:.1f}, {2 * 0.85 * S * w:.2f} | {S * w * w * ACTION_SCALE / tau:.2f} | "
              f"{math.sqrt(tau / (ACTION_SCALE * S)) / (2 * math.pi):.1f}")

    rng = np.random.default_rng(4)
    poses = [sweep[j][k] for j in joints for k in ("locked_pose", "free_pose", "reference_pose")]
    for limb in ia.limbs(model):
        poses += [dict(zip(limb, v)) for v in itertools.product(*[np.linspace(*model["ranges"][j], 5) for j in limb])]
    random_poses = [{j: float(rng.uniform(*model["ranges"][j])) for j in joints} for _ in range(3000)]
    Ws = [ia.inertias(model, q)[3] for q in poses + random_poses]
    nr = len(random_poses)
    Sv = np.array([ref_g[g] for g in gj])
    Sl = np.array([min(sweep[k]["locked_min"] for k in joints if group_of(groups, k) == g) for g in gj])
    cf = np.array([groups[g][2] / groups[g][4] + groups[g][3] for g in gj])
    mu = [max(np.linalg.eigvalsh(np.sqrt(s)[:, None] * W * np.sqrt(s)[None, :]).max() for W in Ws) for s in (Sv, Sl)]
    print(f"  max lambda_max(S W), S = reference {mu[0]:.3f}, S = locked minimum {mu[1]:.3f}, over {len(Ws)} poses")

    def gains(setting):
        if setting == "catalogue":
            return np.array([groups[g][5] for g in gj]), np.array([groups[g][6] for g in gj])
        if setting == "largest draw":
            return (np.array([groups[g][5] * SCALES[family][g][0] for g in gj]),
                    np.array([groups[g][6] * SCALES[family][g][1] for g in gj]))
        f, z, *locked = setting
        w, S = 2 * math.pi * f, (Sl if locked else Sv)
        return S * w * w, 2 * z * S * w

    print("  gains | Lambda servo | Lambda with friction | beta servo | beta with friction | max spectral radius | share of random poses > 1")
    for setting in ("catalogue", "largest draw", (10.0, 0.85), (14.0, 1.0), (10.0, 0.85, "against the locked minimum")):
        K, D = gains(setting)
        cells = []
        for Dt in (D, D + cf):
            cells.append(max(np.linalg.eigvals(W @ np.diag(DT * DT * K + 2 * DT * Dt)).real.max() for W in Ws))
        for Dt in (D, D + cf):
            cells.append(max(np.linalg.eigvals(DT * W @ np.diag(Dt)).real.max() for W in Ws))
        radii = [abs(np.linalg.eigvals(transition(W, K, D + cf))).max() for W in Ws]
        print(f"    {setting} | " + " | ".join(f"{c:.3f}" for c in cells) +
              f" | {max(radii):.3f} | {np.mean(np.array(radii[-nr:]) > 1):.3f}")
    print("  per group sufficient test against S_j with friction, a_j + 2 b_j, catalogue | largest draw | 10 Hz, 0.85")
    for g in groups:
        cells = []
        for setting in ("catalogue", "largest draw", (10.0, 0.85)):
            K, D = gains(setting)
            j = gj.index(g)
            cells.append(f"{(DT * DT * K[j] + 2 * DT * (D[j] + cf[j])) / Sv[j]:.2f}")
        print(f"    {g} | " + " | ".join(cells))

    # MuJoCo confirmation at the worst pose of the catalogue gains, zero gravity, no contact or passive terms.
    K, D = gains("catalogue")
    radii = [abs(np.linalg.eigvals(transition(W, K, D))).max() for W in Ws]
    q = (poses + random_poses)[int(np.argmax(radii))]
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    m.opt.gravity[:], m.opt.timestep, m.opt.integrator = 0.0, DT, mujoco.mjtIntegrator.mjINT_EULER
    m.dof_damping[:], m.dof_frictionloss[:], m.geom_contype[:], m.geom_conaffinity[:] = 0.0, 0.0, 0, 0
    m.jnt_limited[:] = 0
    dofs = [m.jnt_dofadr[m.joint(j).id] for j in joints]
    qadr = [m.jnt_qposadr[m.joint(j).id] for j in joints]
    for j, k in zip(joints, dofs):
        m.dof_armature[k] = model["armature"][j]
    for label, pose in (("worst", q), ("zero", {})):
        mujoco.mj_resetData(m, d)
        for j, a in zip(joints, qadr):
            d.qpos[a] = pose.get(j, 0.0)
        target = d.qpos[qadr].copy()
        d.qvel[dofs] = 1e-3 * np.random.default_rng(0).standard_normal(len(dofs))
        amp = []
        for _ in range(60):
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[dofs] = -K * (d.qpos[qadr] - target) - D * d.qvel[dofs]
            mujoco.mj_step(m, d)
            amp.append(float(np.abs(d.qvel[dofs]).max()))
        predicted = abs(np.linalg.eigvals(transition(ia.inertias(model, pose)[3], K, D))).max()
        print(f"  MuJoCo, catalogue gains at the {label} pose (predicted radius {predicted:.3f}): "
              f"max |qvel| after 1, 10, 30, 60 steps " + ", ".join(f"{amp[i]:.2e}" for i in (0, 9, 29, 59)))
    print(f"    worst pose {{{', '.join(f'{k}: {v:.2f}' for k, v in q.items() if abs(v) > 1e-9)}}}")


# The sufficient condition of section 3.3.7, lambda_max(W (dt^2 K + 2 dt D)) < 4 implies a spectral radius below one,
# tested on random positive definite W and diagonal K, D scaled towards the boundary, in units of dt = 1.
rng, violations, tested = np.random.default_rng(0), 0, 0
for _ in range(200000):
    n = int(rng.integers(2, 5))
    A = rng.standard_normal((n, n))
    W = A @ A.T + 1e-3 * np.eye(n)
    K = rng.exponential(1.0, n) * rng.choice([1e-3, 1e-1, 1.0])
    D = rng.exponential(1.0, n)
    lam = np.linalg.eigvals(W @ np.diag(K + 2 * D)).real.max()
    W *= rng.uniform(0.5, 0.9999) * 4.0 / lam
    I = np.eye(n)
    T = np.block([[I - W * K[None, :], I - W * D[None, :]], [-W * K[None, :], I - W * D[None, :]]])
    tested += 1
    violations += int(abs(np.linalg.eigvals(T)).max() > 1 + 1e-9)
print(f"sufficient condition: {tested} random cases near its boundary, {violations} with a spectral radius above one")


# The energy of case B, section 3.3.7 steps B4 to B7, on random coupled servos near the boundary at dt = 0.005.
worst, increases = 0.0, 0
for trial in range(200):
    n, dt = int(rng.integers(2, 5)), 0.005
    A = rng.standard_normal((n, n))
    W = A @ A.T + 0.1 * np.eye(n)
    w, V = np.linalg.eigh(W)
    Wh, Wmh = V @ np.diag(np.sqrt(w)) @ V.T, V @ np.diag(1 / np.sqrt(w)) @ V.T
    K, D = np.diag(rng.uniform(10, 100, n)), np.diag(rng.uniform(0.5, 3, n))
    scale = 1.01 * np.linalg.eigvalsh(dt * dt * Wh @ K @ Wh / 4 + dt * Wh @ D @ Wh / 2).max()
    K, D = K / scale, D / scale
    Kh, Dh, I = dt * dt * Wh @ K @ Wh, dt * Wh @ D @ Wh, np.eye(n)
    energy = lambda y, ym: (y - ym) @ (I - Dh / 2) @ (y - ym) + y @ Kh @ ym
    midpoint = lambda y, ym: (y - ym) @ (I - Kh / 4 - Dh / 2) @ (y - ym) + (y + ym) @ Kh @ (y + ym) / 4
    e, qd = rng.standard_normal(n), rng.standard_normal(n)
    for _ in range(50):
        y, ym = Wmh @ e, Wmh @ (e - dt * qd)
        worst = max(worst, abs((y - ym) @ (y - ym) - dt * dt * qd @ np.linalg.inv(W) @ qd),
                    abs(y @ Kh @ ym - dt * dt * e @ K @ (e - dt * qd)), abs(energy(y, ym) - midpoint(y, ym)))
        E0, d0 = energy(y, ym), y - ym
        qd = qd + dt * W @ (-K @ e - D @ qd)
        e = e + dt * qd
        y2, y1 = Wmh @ e, Wmh @ (e - dt * qd)
        s = d0 + (y2 - y1)
        worst = max(worst, abs(energy(y2, y1) - E0 + 0.5 * s @ Dh @ s))
        increases += int(energy(y2, y1) > E0 + 1e-12)
    L = np.block([[Wmh, 0 * I], [Wmh, -dt * Wmh]])
    Phi = np.block([[I - dt * dt * W @ K, dt * (I - dt * W @ D)], [-dt * W @ K, I - dt * W @ D]])
    Phi_y = np.block([[2 * I - Kh - Dh, Dh - I], [I, 0 * I]])
    worst = max(worst, np.abs(Phi_y - L @ Phi @ np.linalg.inv(L)).max())
print(f"energy of case B: largest identity error {worst:.1e} over 200 servos and 50 steps each, {increases} increases")
