"""Comparative analysis of the 2026-09-24 and 2026-09-25 energy and ankle ablation.

Reads the evaluation dumps compare.py staged under kscale_flat_eval, computes the
quantities the analysis brief of 2026-09-26 asks for, and prints one report. It is run
inside a GPU container, this development container carrying no numpy.

The swing and stance segmentation is by contact force on the foot link, which is the same
criterion stats.py uses, and every per phase figure below is conditioned on the phase of
the leg the joint belongs to rather than on a global gait clock, so that a quantity
attributed to swing is measured on the leg that is actually swinging.
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, "/workspace/isaaclab/biped/scripts/analysis")
import experiment_params as ep

EVAL = "/workspace/isaaclab/logs/rsl_rl/kscale_flat_eval"
LOGS = "/workspace/isaaclab/logs/rsl_rl/kscale_flat"
TAG = {"2026-09-24_05-18-33": "A", "2026-09-25_08-00-34": "B", "2026-09-25_11-29-39": "C"}
FORCE = 1.0
FRAC = 0.98
G = 9.81


def rot(quat, v):
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                 axis=-1).reshape(quat.shape[:-1] + (3, 3))
    return R @ np.asarray(v, dtype=R.dtype)


def yaw_of(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def foot_index(joint_name):
    """The foot a joint belongs to, as an index into the dump's foot arrays.

    The foot links of this robot are named foot_6061_2 and foot_6061 and carry no side
    marker, so the side cannot be read from the name and a startswith test resolves every
    joint to the same foot. Index 0 is the left. Three independent facts establish it, the
    articulation body order placing every left link before its right counterpart, the
    lateral separation of foot zero minus foot one standing positive in the base frame,
    and the right hip yaw correlating with foot one's toe-in at 0.95 against 0.18 for
    foot zero.
    """
    return 0 if joint_name.startswith("left") else 1


def envelope(dump, run_dir):
    """Per joint (saturation_effort, effort_limit, velocity_limit) from the run's params."""
    p = ep.load_params(run_dir)
    groups = ep.actuator_groups(p) if p else {}
    names = [str(n) for n in dump["joint_names"]]
    sat = np.full(len(names), np.nan)
    eff = np.full(len(names), np.nan)
    vel = np.full(len(names), np.nan)
    kp = np.full(len(names), np.nan)
    kd = np.full(len(names), np.nan)
    for g in groups.values():
        exprs = g.get("joint_names_expr") or []
        if isinstance(exprs, str):
            exprs = [exprs]
        for j, n in enumerate(names):
            if not any(re.fullmatch(e, n) for e in exprs):
                continue
            for arr, key in ((sat, "saturation_effort"), (eff, "effort_limit"),
                             (vel, "velocity_limit")):
                if g.get(key) is not None:
                    arr[j] = float(g[key])
            for arr, key in ((kp, "stiffness"), (kd, "damping")):
                v = g.get(key)
                if isinstance(v, dict):
                    v = list(v.values())[0]
                if v is not None:
                    arr[j] = float(v)
    return sat, eff, vel, kp, kd


def available(torque, vel_j, sat, eff, vlim):
    """The DC motor envelope, reproducing DCMotor._clip_effort at actuator_pd.py:295."""
    reach = vlim * (1.0 + eff / sat)
    q = np.clip(vel_j, -reach, reach)
    top = np.minimum(sat * (1.0 - q / vlim), eff)
    bot = np.maximum(sat * (-1.0 - q / vlim), -eff)
    return np.where(torque >= 0.0, top, -bot)


def action_scales(run_dir, names):
    p = ep.load_params(run_dir)
    sc = p["actions"]["joint_pos"].get("scale")
    out = np.full(len(names), np.nan)
    if isinstance(sc, dict):
        for expr, v in sc.items():
            for j, n in enumerate(names):
                if re.fullmatch(expr, n):
                    out[j] = float(v)
    elif sc is not None:
        out[:] = float(sc)
    return out


def edges(mask):
    """Indices where a boolean (steps, envs) series rises and falls."""
    prev = mask[:-1]
    cur = mask[1:]
    return (~prev) & cur, prev & (~cur)


def analyse(job):
    d = np.load(os.path.join(EVAL, job, "data", "42", "dump.npy"),
                allow_pickle=True).item()
    rw = np.load(os.path.join(EVAL, job, "data", "42", "rewards.npy"),
                 allow_pickle=True).item()
    exp = job.rsplit("_model_", 1)[0]
    run_dir = os.path.join(LOGS, exp)
    names = [str(n) for n in d["joint_names"]]
    dt = float(d["step_dt"])
    pos = np.asarray(d["joint_positions"])
    vel = np.asarray(d["joint_velocities"])
    tau = np.asarray(d["joint_torques"])
    nominal = np.asarray(d["default_joint_pos"])
    lim = np.asarray(d["joint_position_limits"])
    soft = np.asarray(d["joint_soft_position_limits"])
    sat, eff, vlim, kp, kd = envelope(d, run_dir)
    scale = action_scales(run_dir, names)
    contact = np.linalg.norm(np.asarray(d["feet_contact_forces"]), axis=-1) > FORCE
    mass = float(np.asarray(d["robot_mass"])[0, 0].sum())

    out = {"job": job, "tag": TAG.get(exp, "?"), "experiment": exp,
           "checkpoint": int(job.rsplit("_", 1)[-1]), "dt": dt, "mass": mass,
           "names": names}

    # --- saturation, per joint, against the envelope and against the flat ceiling ----
    satf, ceilf, hold = {}, {}, {}
    for j, n in enumerate(names):
        col = np.abs(tau[:, :, j])
        av = available(tau[:, :, j], vel[:, :, j], sat[j], eff[j], vlim[j])
        satf[n] = float(100.0 * np.mean(col >= np.maximum(FRAC * av, 1e-9)))
        ceilf[n] = float(100.0 * np.mean(col >= FRAC * eff[j]))
        hold[n] = float(np.mean(col))
    out["saturation_envelope"] = satf
    out["saturation_ceiling"] = ceilf
    out["mean_abs_torque"] = hold
    out["effort_limit"] = {n: float(eff[j]) for j, n in enumerate(names)}
    out["kp"] = {n: float(kp[j]) for j, n in enumerate(names)}
    out["kd"] = {n: float(kd[j]) for j, n in enumerate(names)}
    out["scale"] = {n: float(scale[j]) for j, n in enumerate(names)}

    # --- per joint position statistics against the nominal and the stops -------------
    stats = {}
    for j, n in enumerate(names):
        dev = pos[:, :, j] - nominal[j]
        near_lo = pos[:, :, j] <= lim[j, 0] + 0.01 * (lim[j, 1] - lim[j, 0])
        near_hi = pos[:, :, j] >= lim[j, 1] - 0.01 * (lim[j, 1] - lim[j, 0])
        outside = (pos[:, :, j] < soft[j, 0]) | (pos[:, :, j] > soft[j, 1])
        stats[n] = {
            "mean": float(pos[:, :, j].mean()), "std": float(pos[:, :, j].std()),
            "dev_abs_mean": float(np.abs(dev).mean()),
            "dev_rms": float(np.sqrt(np.mean(dev ** 2))),
            "dev_p95": float(np.percentile(np.abs(dev), 95)),
            "span": float(pos[:, :, j].max() - pos[:, :, j].min()),
            "speed_abs_mean": float(np.abs(vel[:, :, j]).mean()),
            "speed_p95": float(np.percentile(np.abs(vel[:, :, j]), 95)),
            "pct_at_stop": float(100.0 * np.mean(near_lo | near_hi)),
            "pct_outside_soft": float(100.0 * np.mean(outside)),
            "nominal": float(nominal[j]),
            "limits": [float(lim[j, 0]), float(lim[j, 1])],
        }
    out["joint"] = stats

    # --- phase conditioned behaviour of every joint, by the leg it belongs to --------
    phase = {}
    for j, n in enumerate(names):
        foot = foot_index(n)
        st = contact[:, :, foot]
        sw = ~st
        col = np.abs(tau[:, :, j])
        av = available(tau[:, :, j], vel[:, :, j], sat[j], eff[j], vlim[j])
        hot = col >= np.maximum(FRAC * av, 1e-9)
        dev = pos[:, :, j] - nominal[j]
        phase[n] = {
            "stance_frac": float(100.0 * st.mean()),
            "dev_abs_stance": float(np.abs(dev)[st].mean()),
            "dev_abs_swing": float(np.abs(dev)[sw].mean()),
            "sat_stance": float(100.0 * hot[st].mean()),
            "sat_swing": float(100.0 * hot[sw].mean()),
            "speed_stance": float(np.abs(vel[:, :, j])[st].mean()),
            "speed_swing": float(np.abs(vel[:, :, j])[sw].mean()),
            "torque_stance": float(col[st].mean()),
            "torque_swing": float(col[sw].mean()),
        }
    out["phase"] = phase

    # --- the transition transient, the brief's fifth observation ---------------------
    # A window is opened at every stance to swing edge and at every swing to stance edge
    # and the joint's departure from its nominal is averaged across events, so that a
    # transient is separated from a steady offset.
    W = 30  # steps, 0.30 s at the 0.01 s control period
    trans = {}
    for j, n in enumerate(names):
        foot = foot_index(n)
        st = contact[:, :, foot]
        lift, land = edges(st)[1], edges(st)[0]
        dev = pos[:, :, j] - nominal[j]
        col = np.abs(tau[:, :, j])
        av = available(tau[:, :, j], vel[:, :, j], sat[j], eff[j], vlim[j])
        hot = col >= np.maximum(FRAC * av, 1e-9)
        entry = {}
        for label, ev in (("liftoff", lift), ("touchdown", land)):
            s, e = np.nonzero(ev)
            s = s + 1
            keep = (s >= W) & (s < dev.shape[0] - W)
            s, e = s[keep], e[keep]
            if len(s) < 20:
                continue
            idx = s[:, None] + np.arange(-W, W + 1)[None, :]
            win_dev = dev[idx, e[:, None]]
            win_hot = hot[idx, e[:, None]]
            win_spd = np.abs(vel[:, :, j])[idx, e[:, None]]
            prof = np.abs(win_dev).mean(axis=0)
            entry[label] = {
                "events": int(len(s)),
                "profile_dev": [float(v) for v in prof],
                "profile_sat": [float(v) for v in 100.0 * win_hot.mean(axis=0)],
                "profile_speed": [float(v) for v in win_spd.mean(axis=0)],
                "pre": float(prof[:W].mean()), "at": float(prof[W]),
                "post": float(prof[W + 1:].mean()),
                "peak": float(prof.max()),
                "peak_offset_s": float((int(prof.argmax()) - W) * dt),
                "peak_event_dev": float(np.abs(win_dev).max(axis=1).mean()),
            }
        trans[n] = entry
    out["transition"] = trans

    # --- splay ----------------------------------------------------------------------
    fq = np.asarray(d["feet_quaternions"])
    bq = np.asarray(d["base_quaternion"])
    by = yaw_of(bq)
    rel = []
    for f in range(fq.shape[2]):
        toe = rot(fq[:, :, f, :], (0.0, 0.0, -1.0))
        rel.append(wrap(np.arctan2(toe[..., 1], toe[..., 0]) - by))
    toe_in = np.degrees(np.stack([-rel[0], rel[1]], axis=-1))
    out["splay"] = {
        "abs_mean": float(np.abs(toe_in).mean()),
        "signed_mean": float(toe_in.mean()),
        "p95": float(np.percentile(np.abs(toe_in), 95)),
        "differential": float(np.abs(np.degrees(wrap(rel[1] - rel[0]))).mean()),
        "stance_abs_mean": float(np.abs(toe_in)[contact].mean()),
        "stance_signed_mean": float(toe_in[contact].mean()),
        "swing_abs_mean": float(np.abs(toe_in)[~contact].mean()),
    }

    # --- the hip yaw against the stance splay, the slope arm 10 rests on -------------
    hy = [j for j, n in enumerate(names) if "hip_yaw" in n]
    if len(hy) == 2:
        li = hy[0] if names[hy[0]].startswith("left") else hy[1]
        ri = hy[1] if li == hy[0] else hy[0]
        pairs = []
        for foot, jj, sgn in ((0, li, -1.0), (1, ri, 1.0)):
            m = contact[:, :, foot]
            # toe_in already flips the left foot, so the left hip yaw is flipped with it.
            x = np.degrees(sgn * (pos[:, :, jj] - nominal[jj]))[m]
            y = toe_in[:, :, foot][m]
            if len(x) > 100:
                sl = float(np.polyfit(x, y, 1)[0])
                cr = float(np.corrcoef(x, y)[0, 1])
                pairs.append((sl, cr))
        if pairs:
            out["hip_yaw_vs_splay"] = {
                "slope_deg_per_deg": float(np.mean([p[0] for p in pairs])),
                "correlation": float(np.mean([p[1] for p in pairs])),
            }

    # --- gait and viability ---------------------------------------------------------
    knee = [j for j, n in enumerate(names) if "knee" in n]
    height = np.asarray(d["base_com_position"])[:, :, 2]
    lin = np.asarray(d["base_linear_velocity"])
    cmd = np.asarray(d["commanded_linear_velocity"])
    ang = np.asarray(d["base_angular_velocity"])
    cyaw = np.asarray(d["commanded_angular_velocity"])
    power = np.abs(np.asarray(d["joint_powers"])).sum(axis=-1)
    speed = np.linalg.norm(lin[:, :, :2], axis=-1)
    term = np.asarray(d["episode_terminated"])
    both = contact.all(axis=-1)
    none = (~contact).any(axis=-1)
    st0 = contact[:, :, 0]
    lift0 = edges(st0)[1]
    out["gait"] = {
        "base_height": float(height.mean()),
        "base_height_std": float(height.std()),
        "knee_mean": float(pos[:, :, knee].mean()),
        "planar_error": float(np.mean(np.linalg.norm(lin[:, :, :2] - cmd[:, :, :2],
                                                     axis=-1))),
        "yaw_error": float(np.mean(np.abs(ang[:, :, 2] - cyaw))),
        "abs_power": float(power.mean()),
        "cost_of_transport": float(power.mean() / (mass * G * max(speed.mean(), 1e-6))),
        "double_support_pct": float(100.0 * both.mean()),
        "flight_pct": float(100.0 * (~contact).all(axis=-1).mean()),
        "cadence": float(lift0.sum() / (contact.shape[1] * contact.shape[0] * dt) * 2),
        "termination_rate": float(term.sum() / (term.shape[1] * term.shape[0] * dt)),
        "terminations": int(term.sum()),
        "torso_roll_deg": float(np.degrees(np.abs(
            np.arcsin(np.clip(-np.asarray(d["base_projected_gravity"])[:, :, 1], -1, 1))
        )).mean()),
        "feet_lateral_sep": float(np.abs(np.asarray(d["feet_distance"])[:, :, 1]).mean()),
        "sole_clearance_stance_mm": float(
            1000.0 * np.asarray(d["feet_sole_clearances"])[contact].mean()),
    }

    # --- the reward budget ----------------------------------------------------------
    w = rw["_weights"].item() if hasattr(rw["_weights"], "item") else rw["_weights"]
    terms = {}
    for k, v in rw.items():
        if k in ("_weights", "total_reward"):
            continue
        terms[k] = float(np.asarray(v).mean())
    neg = sum(v for v in terms.values() if v < 0)
    out["reward"] = {
        "total_per_step": float(np.asarray(rw["total_reward"]).mean()),
        "terms": terms,
        "negative_total": neg,
        "weights": {k: float(x) for k, x in w.items()} if isinstance(w, dict) else {},
    }
    return out


def main():
    jobs = sys.argv[1:]
    recs = [analyse(j) for j in jobs]
    path = os.path.join(EVAL, "ablation_analysis.json")
    with open(path, "w") as h:
        json.dump(recs, h, indent=1)
    print(f"wrote {path} for {len(recs)} jobs")


if __name__ == "__main__":
    main()
