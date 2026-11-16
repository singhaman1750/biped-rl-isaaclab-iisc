"""Stance load and actuator gain analysis for any biped, from its document and a shared configuration.

This script generalises ``kscale_stance_analysis.py`` in the way ``biped_physical_analysis.py``
generalises ``kscale_physical_analysis.py``, reading the model through MuJoCo and every robot
specific quantity from the shared JSON configuration of ``biped_configs/``. Where the companion
reports the inertia a leg presents while it swings, this script reports what the leg must do while
it stands, which is the question that decides whether a policy starting from the nominal pose is
held up by its actuators or folds onto its knees.

It reports, in order,

* the static stance torque at each joint of one leg, in double and in single support, obtained by
  balancing the distal free body of each joint against the ground reaction force at a centre of
  pressure beneath the centre of mass, with the sag a proportional loop settles at under that load
  and the action the policy must emit merely to cancel it,
* the stance inertia of each joint, being every body proximal to it about its axis, which is what
  the derivative term meets with the foot planted, beside the swing inertia,
* a proposed gain set, its stiffness floored by the single support load within a position error
  budget and its damping the smallest of the stance target, the swing critical value and the
  explicit integration ceiling,
* a zero action drop test simulated in MuJoCo with the configured actuator model, from the nominal
  pose and from the reset distribution, recording the settled base height and which links come to
  rest on the ground, for the configured gains and for the proposed ones,
* a sagittal bend sweep, re-posing the knee with the hip and the ankle solved to keep the ankle's
  fore-aft lead and the sole's pitch, which shows what a change of the bend baked into the link
  geometry would do to the standing height and to the stance load, and the equivalent bend
  offsets about the world lateral axis for the design generator.

Usage:
    python scripts/analysis/biped_stance_analysis.py --config scripts/analysis/biped_configs/tron1_sf.json
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from biped_physical_analysis import GRAVITY, Biped, load_config, load_spec  # noqa: E402

SAGITTAL = ("hip_pitch", "knee", "ankle_pitch")


def place_on_ground(bp: Biped, offsets: dict[str, float] | None = None) -> float:
    """Pose the robot with its lowest point on the plane z = 0 and return the root height."""
    bp.set_pose(offsets=offsets)
    floor, _ = bp.lowest(bp.collision)
    bp.set_pose(offsets=offsets, base_height=-floor)
    return -floor


def sole_centre(bp: Biped, side: str) -> np.ndarray:
    d = bp.data
    pts = np.array([d.geom_xpos[g] for g in bp.foot_geoms[side]])
    return pts.mean(axis=0)


def stance_torques(bp: Biped, side: str, supports: list[str]) -> dict[str, float]:
    """Actuator torque at each joint of ``side`` holding the current pose statically.

    The ground reaction on each supporting foot is the body weight shared equally, acting at a
    centre of pressure beneath the centre of mass in x and beneath the sole centre in y. For a
    joint, the free body is its distal subtree, so the actuator supplies the negative of the
    moment about the axis of the gravity on that subtree and of any reaction acting on it.
    """
    m, d = bp.model, bp.data
    weight = float(m.body_mass.sum()) * GRAVITY
    com = bp.com()
    out = {}
    for name in bp.legs[side]["joints"]:
        jid = bp.joint(name)["id"]
        p, a = bp.anchor(name), bp.axis(name)
        moment = np.zeros(3)
        for b in bp.subtree(m.jnt_bodyid[jid]):
            moment += np.cross(d.xipos[b] - p, np.array([0.0, 0.0, -m.body_mass[b] * GRAVITY]))
        if side in supports:
            cop = sole_centre(bp, side)
            cop[0], cop[2] = com[0], 0.0
            moment += np.cross(cop - p, np.array([0.0, 0.0, weight / len(supports)]))
        out[name] = -float(moment @ a)
    return out


def stance_inertia(bp: Biped, name: str) -> float:
    """Inertia about the joint axis of every body proximal to it, the foot being planted."""
    m = bp.model
    distal = bp.subtree(m.jnt_bodyid[bp.joint(name)["id"]])
    return bp.axis_inertia([b for b in range(1, m.nbody) if b not in distal], name)


def dc_motor_torque(act: dict, error: np.ndarray, vel: np.ndarray, gains: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The torque of ``environments/actuators/actuator_pd.py`` IdentifiedActuator for a fixed target."""
    K, D = gains
    tau = K * error - D * vel
    sat, lim, vlim = act["saturation_effort"], act["effort_limit"], act["velocity_limit"]
    hi = np.clip(sat * (1.0 - vel / vlim), 0.0, lim)
    lo = np.clip(sat * (-1.0 - vel / vlim), -lim, 0.0)
    tau = np.clip(tau, lo, hi)
    return tau - (act["friction_static"] * np.tanh(vel / act["activation_vel"]) + act["friction_dynamic"] * vel)


def drop_test(path: str, cfg: dict, gains: dict[str, tuple[float, float]], trials: int, seconds: float,
              seed: int = 0, gantry: bool = False) -> list[dict]:
    """Simulate zero action from the nominal pose (trial 0) and from the reset distribution.

    With ``gantry`` the root's free joint is replaced by a vertical slide, holding the trunk level
    and over its spawn point, so that only the vertical load reaches the legs. This
    separates the question the actuators answer, whether they hold the body weight at the nominal
    pose, from the question only a policy answers, whether the robot balances, a biped released
    with zero action toppling whatever its gains.

    The model's own joint damping, armature and friction are replaced by the configured armature
    and the actuator model, a ground plane of friction 1.0 is added, and the step is the configured
    physics step. A leg link is one the reward's undesired contact set names and a terminating body
    one its termination set names, so the ankle housing beside the sole counts as neither. MuJoCo's
    soft contact is not PhysX's, so the result is a qualitative check of
    whether the actuators hold the robot up, not a reproduction of the training simulator.
    """
    spec = load_spec(path)
    spec.worldbody.add_geom(name="analysis_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1], friction=[1.0, 0.005, 0.0001])
    if gantry:
        root = spec.worldbody.first_body()
        spec.delete(root.first_joint())
        root.add_joint(name="analysis_gantry", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
    bp = Biped(path, cfg, spec)
    m, d = bp.model, bp.data
    m.opt.timestep = cfg["control"]["physics_dt"]
    names = bp.joint_names()
    dofs = np.array([bp.joint(n)["dofadr"] for n in names])
    qadr = np.array([bp.joint(n)["qpos"] for n in names])
    for n in names:
        m.dof_armature[bp.joint(n)["dofadr"]] = bp.actuator(n)["armature"]
    acts = [bp.actuator(n) for n in names]
    K = np.array([gains[bp.joint(n)["group"]][0] for n in names])
    D = np.array([gains[bp.joint(n)["group"]][1] for n in names])
    stacked = {k: np.array([a[k] for a in acts]) for k in acts[0] if isinstance(acts[0][k], float)}
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "analysis_floor")
    undesired = {b for b in range(m.nbody) if re.fullmatch(cfg["rewards"]["undesired_contact_bodies"], m.body(b).name)}
    terminal = {b for b in range(m.nbody) if re.fullmatch(cfg["rewards"]["termination_contact_bodies"], m.body(b).name)}
    reset = cfg["reset"]
    rng = np.random.default_rng(seed)
    results = []
    for trial in range(trials):
        mujoco.mj_resetData(m, d)
        bp.set_pose(base_height=reset["spawn_height"])
        q0 = d.qpos[qadr].copy()
        if trial > 0:
            lo, hi = reset["joint_offset_range"]
            d.qpos[qadr] += rng.uniform(lo, hi, len(qadr))
            lo, hi = reset["joint_velocity_range"]
            d.qvel[dofs] = rng.uniform(lo, hi, len(dofs))
            v = reset["base_velocity_range"]
            if bp.free_root:
                d.qvel[0:6] = rng.uniform(-v, v, 6)
        steps = int(seconds / m.opt.timestep)
        heights, touched_leg, touched_trunk, first = [], set(), False, None
        for step in range(steps):
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[dofs] = dc_motor_torque(stacked, q0 - d.qpos[qadr], d.qvel[dofs], (K, D))
            mujoco.mj_step(m, d)
            heights.append(d.qpos[2] if bp.free_root else d.qpos[0])
            for c in d.contact[: d.ncon]:
                if floor not in (c.geom1, c.geom2):
                    continue
                body = m.geom_bodyid[c.geom2 if c.geom1 == floor else c.geom1]
                if body in terminal:
                    touched_trunk = True
                elif body in undesired:
                    touched_leg.add(m.body(body).name)
                if (body in terminal or body in undesired) and first is None:
                    first = (round(step * m.opt.timestep, 3), m.body(body).name)
        resting = sorted({m.body(m.geom_bodyid[c.geom2 if c.geom1 == floor else c.geom1]).name
                          for c in d.contact[: d.ncon] if floor in (c.geom1, c.geom2)})
        tail = np.array(heights[-int(0.5 / m.opt.timestep):])
        results.append({
            "settled_height": float(tail.mean()),
            "deflection": (d.qpos[qadr] - q0).tolist(),
            "resting_on": resting,
            "leg_link_touched": sorted(touched_leg),
            "trunk_touched": touched_trunk,
            "first_non_foot_contact": first,
            "trunk_bodies": [m.body(b).name for b in terminal],
        })
    return results


def solve_bend(bp: Biped, side: str, knee_offset: float, lead: float, pitch: float) -> tuple[float, float] | None:
    """Hip and ankle offsets restoring the ankle lead over the hip and the sole pitch, by Newton."""
    role = bp.legs[side]["role"]
    foot = bp.legs[side]["foot"]

    def residual(x):
        bp.set_pose(offsets={role["hip_pitch"]: x[0], role["knee"]: knee_offset, role["ankle_pitch"]: x[1]})
        r = bp.data.xmat[foot].reshape(3, 3)[:, 0]
        return np.array([bp.anchor(role["ankle_pitch"])[0] - bp.anchor(role["hip_pitch"])[0] - lead,
                         math.asin(np.clip(-r[2], -1, 1)) - pitch])

    x = np.array([-knee_offset / 2.0, -knee_offset / 2.0])
    for _ in range(50):
        f = residual(x)
        if np.linalg.norm(f) < 1e-9:
            return float(x[0]), float(x[1])
        jac = np.column_stack([(residual(x + e) - f) / 1e-6 for e in np.eye(2) * 1e-6])
        x = x - np.linalg.solve(jac, f)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="shared biped analysis configuration")
    parser.add_argument("--model", default=None, help="MJCF or URDF document, defaulting to the configuration's")
    parser.add_argument("--side", default=None, help="leg to report, defaulting to the first configured")
    parser.add_argument("--trials", type=int, default=32, help="drop test trials, the first from the nominal pose")
    parser.add_argument("--seconds", type=float, default=3.0, help="drop test duration")
    parser.add_argument("--sweep", type=float, nargs=3, default=(-0.6, 0.6, 0.1), metavar=("LO", "HI", "STEP"),
                        help="knee offsets of the bend sweep, radians")
    args = parser.parse_args()

    cfg = load_config(args.config)
    path = args.model or cfg["model"]
    side = args.side or next(iter(cfg["legs"]))
    sides = list(cfg["legs"])
    bp = Biped(path, cfg)
    m = bp.model
    scale = cfg["control"]["action_scale"]
    dt = cfg["control"]["physics_dt"]
    design = cfg["gain_design"]
    budget, zeta = design["position_error_budget"], design["target_zeta"]
    weight = float(m.body_mass.sum()) * GRAVITY

    print(f"MODEL {path}\nmass {m.body_mass.sum():.3f} kg, weight {weight:.1f} N, action scale {scale}\n")

    # -- stance torques ------------------------------------------------------------------------
    height = place_on_ground(bp)
    print("=" * 96)
    print(f"STATIC STANCE TORQUE at the nominal pose, root {height:.4f} m above the ground, leg {side}")
    print("=" * 96)
    double = stance_torques(bp, side, sides)
    single = stance_torques(bp, side, [side])
    swing = stance_torques(bp, side, [s for s in sides if s != side])
    com = bp.com()
    print(f"  centre of mass {com[0]:+.4f} m fore-aft, {com[2]:.4f} m high, centre of pressure placed beneath it\n")
    print(f"  {'joint':<16}{'double':>9}{'single':>9}{'swing':>8}{'limit':>8}{'K':>7}{'sag2':>8}{'sag1':>8}"
          f"{'act2':>7}{'act1':>7}{'use1':>7}")
    for name in bp.legs[side]["joints"]:
        act = bp.actuator(name)
        K, lim = act["stiffness"], act["effort_limit"]
        print(f"  {name:<16}{double[name]:>+9.2f}{single[name]:>+9.2f}{swing[name]:>+8.2f}{lim:>8.1f}{K:>7.1f}"
              f"{abs(double[name]) / K:>8.3f}{abs(single[name]) / K:>8.3f}{abs(double[name]) / K / scale:>7.2f}"
              f"{abs(single[name]) / K / scale:>7.2f}{abs(single[name]) / lim:>7.2f}")
    print("\n  torques in Nm, positive about the joint axis. sag in rad is torque over stiffness, the steady error")
    print("  a proportional loop settles at under that load, act is that sag in action units, the action the")
    print("  policy must add merely to hold the pose, and use1 the single support torque over the effort limit.")

    # -- inertia and proposed gains ------------------------------------------------------------
    print("\n" + "=" * 96)
    print(f"STANCE AND SWING INERTIA, AND THE PROPOSED GAINS, target zeta {zeta}, error budget {budget} rad")
    print("=" * 96)
    bp.set_pose()
    full = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, bp.data, full)
    margin = 1.1 / 0.95
    print(f"  {'joint':<16}{'I_swing':>9}{'I_stance':>10}{'K':>7}{'D':>7}{'zeta_sw':>8}{'zeta_st':>8}"
          f"{'K_min':>8}{'K*':>8}{'D*':>7}{'zsw*':>7}{'zst*':>7}{'sag1*':>7}")
    proposals: dict[str, tuple[float, float]] = {}
    for name in bp.legs[side]["joints"]:
        j, act = bp.joint(name), bp.actuator(name)
        i_swing = float(full[j["dofadr"], j["dofadr"]]) + act["armature"]
        i_stance = stance_inertia(bp, name) + act["armature"]
        K, D = act["stiffness"], act["damping"]
        k_min = min(abs(single[name]), act["effort_limit"]) / budget
        k_new = max(k_min, design["target_omega"][j["group"]] ** 2 * i_swing)
        d_new = min(2 * zeta * math.sqrt(k_new * i_stance), 2 * math.sqrt(k_new * i_swing), i_swing / (dt * margin))
        proposals[j["group"]] = (k_new, d_new)
        print(f"  {name:<16}{i_swing:>9.4f}{i_stance:>10.4f}{K:>7.1f}{D:>7.2f}"
              f"{D / (2 * math.sqrt(K * i_swing)):>8.2f}{D / (2 * math.sqrt(K * i_stance)):>8.2f}"
              f"{k_min:>8.1f}{k_new:>8.1f}{d_new:>7.2f}{d_new / (2 * math.sqrt(k_new * i_swing)):>7.2f}"
              f"{d_new / (2 * math.sqrt(k_new * i_stance)):>7.2f}{abs(single[name]) / k_new:>7.3f}")
    print("\n  I in kg m^2 with the configured armature. K_min holds the single support torque within the budget,")
    print("  capped at the effort limit. D* is the smallest of the stance target, the swing critical damping and")
    print(f"  the explicit integration ceiling I_swing / (dt x {margin:.3f}) of the randomised envelope.")
    print("\n  Proposed actuator block:")
    for group, (k_new, d_new) in proposals.items():
        print(f"    {group:<8} stiffness {k_new:7.1f}   damping {d_new:6.2f}")

    # -- drop test -----------------------------------------------------------------------------
    configured = {g: (a["stiffness"], a["damping"]) for g, a in cfg["actuators"].items()}
    cases = [(g, label, gains) for g in (True, False) for label, gains in (("configured", configured), ("proposed", proposals))]
    for gantry, label, gains in cases:
        res = drop_test(path, cfg, gains, args.trials, args.seconds, gantry=gantry)
        print("\n" + "=" * 96)
        print(f"ZERO ACTION DROP TEST, {'trunk held level on a gantry' if gantry else 'free'}, {label} gains, "
              f"{args.trials} trials of {args.seconds} s, spawn {cfg['reset']['spawn_height']} m")
        print("=" * 96)
        nominal = res[0]
        names = bp.joint_names()
        print(f"  from the nominal pose, settled root height {nominal['settled_height']:.4f} m against standing {height:.4f} m")
        print(f"    resting on {', '.join(nominal['resting_on'])}, leg links touched {nominal['leg_link_touched'] or 'none'}, "
              f"trunk touched {nominal['trunk_touched']}, first non foot contact {nominal['first_non_foot_contact']}")
        print("    deflection " + ", ".join(f"{n} {v:+.3f}" for n, v in zip(names, nominal["deflection"])))
        rest = res[1:]
        if rest:
            h = np.array([r["settled_height"] for r in rest])
            leg = np.mean([bool(r["leg_link_touched"]) for r in rest])
            trunk = np.mean([r["trunk_touched"] for r in rest])
            pattern = cfg["rewards"]["undesired_contact_bodies"]
            kneel = np.mean([any(re.fullmatch(pattern, b) and b not in r["trunk_bodies"] for b in r["resting_on"])
                             for r in rest])
            print(f"  from the reset distribution, settled root height {h.mean():.4f} m (min {h.min():.4f}, max {h.max():.4f}),")
            print(f"    {100 * leg:.0f} percent touched the ground with an undesired contact link, {100 * kneel:.0f} percent end "
                  f"resting on one, {100 * trunk:.0f} percent touched it with a terminating body")
            firsts = [r["first_non_foot_contact"] for r in rest if r["first_non_foot_contact"]]
            if firsts:
                bodies = sorted({f[1] for f in firsts})
                print(f"    first non foot contact after {np.median([f[0] for f in firsts]):.2f} s median, on "
                      f"{', '.join(f'{b} ({sum(f[1] == b for f in firsts)})' for b in bodies)}")

    # -- bend sweep ----------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("SAGITTAL BEND SWEEP, knee re-posed, hip and ankle solved to keep the ankle lead and the sole pitch")
    print("=" * 96)
    role = bp.legs[side]["role"]
    bp.set_pose()
    lead = bp.anchor(role["ankle_pitch"])[0] - bp.anchor(role["hip_pitch"])[0]
    r = bp.data.xmat[bp.legs[side]["foot"]].reshape(3, 3)[:, 0]
    pitch = math.asin(np.clip(-r[2], -1, 1))
    axes_y = {k: float(bp.axis(role[k])[1]) for k in SAGITTAL}
    knee_cfg = bp.actuator(role["knee"])
    hip_cfg = bp.actuator(role["hip_pitch"])
    ankle_cfg = bp.actuator(role["ankle_pitch"])
    print(f"  {'d_knee':>7}{'d_hip':>8}{'d_ankle':>8}{'thigh':>7}{'shank':>7}{'incl':>7}{'height':>8}"
          f"{'tk2':>7}{'tk1':>7}{'th1':>7}{'ta1':>7}{'act_k1':>7}{'bend_th':>8}{'bend_kn':>8}{'bend_an':>8}")
    for dk in np.arange(args.sweep[0], args.sweep[1] + 1e-9, args.sweep[2]):
        sol = solve_bend(bp, side, float(dk), lead, pitch)
        if sol is None:
            continue
        dh, da = sol
        signs = bp.sagittal_offsets({"hip_pitch": 1.0, "knee": 1.0, "ankle_pitch": 1.0}, side)
        offsets = {}
        for s in sides:
            rs = bp.legs[s]["role"]
            offsets.update({rs["hip_pitch"]: dh * signs[rs["hip_pitch"]], rs["knee"]: float(dk) * signs[rs["knee"]],
                            rs["ankle_pitch"]: da * signs[rs["ankle_pitch"]]})
        h = place_on_ground(bp, offsets)
        hip, knee, ankle = (bp.anchor(role[k]) for k in SAGITTAL)
        th, sh = knee - hip, ankle - knee
        incl = math.degrees(math.acos(np.clip(-th @ sh / np.linalg.norm(th) / np.linalg.norm(sh), -1, 1)))
        t2 = stance_torques(bp, side, sides)
        t1 = stance_torques(bp, side, [side])
        print(f"  {dk:>+7.2f}{dh:>+8.3f}{da:>+8.3f}{math.degrees(math.atan2(th[0], -th[2])):>+7.1f}"
              f"{math.degrees(math.atan2(sh[0], -sh[2])):>+7.1f}{incl:>7.1f}{h:>8.4f}"
              f"{t2[role['knee']]:>+7.1f}{t1[role['knee']]:>+7.1f}{t1[role['hip_pitch']]:>+7.1f}{t1[role['ankle_pitch']]:>+7.1f}"
              f"{abs(t1[role['knee']]) / knee_cfg['stiffness'] / scale:>7.2f}"
              f"{math.degrees(dh * axes_y['hip_pitch']):>+8.1f}{math.degrees(dk * axes_y['knee']):>+8.1f}"
              f"{math.degrees(da * axes_y['ankle_pitch']):>+8.1f}")
    print("\n  offsets in rad, angles in deg from the downward vertical, incl the knee included angle, height the root")
    print("  above the ground. tk2 and tk1 the knee torque in double and single support, th1 and ta1 the hip and the")
    print(f"  ankle in single support, Nm, against stiffnesses {knee_cfg['stiffness']}, {hip_cfg['stiffness']} and "
          f"{ankle_cfg['stiffness']}. act_k1 the knee action that cancels the")
    print("  single support sag. bend_* the equivalent rotation of each body about +y in degrees, the quantity the")
    print("  design generator's thigh, knee and ankle bend offsets add to the baked geometry.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
