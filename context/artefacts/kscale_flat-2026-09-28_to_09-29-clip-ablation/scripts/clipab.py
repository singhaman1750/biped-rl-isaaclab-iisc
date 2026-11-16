"""The 2026-09-28 and 2026-09-29 ablation, read from the exact channels rather than by inference.

Every figure here is computed from the instrumentation play.py began recording on
2026-09-28 for section 20.2 of the plan, and the difference from the preceding round is
not one of precision but of availability. The commanded target is READ rather than
recovered by inverting the control law, which the previous round could do on 11.75 per
cent of run C's ankle roll samples and which therefore made every figure it quoted for
that joint a bound. Saturation is the computed torque lying outside the actuator's own
envelope, both being dumped, so it carries no threshold and no dependence on the friction
term subtracted after the clip. And the clip excess, which is the quantity arm 11 prices and which is in flight at two weights as this is written, is the difference
between the unclipped target and the clipped one and needs no reconstruction at all.
"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, "/workspace/isaaclab/biped/scripts/analysis")
import experiment_params as ep

EVAL = "/workspace/isaaclab/logs/rsl_rl/kscale_flat_eval"
LOGS = "/workspace/isaaclab/logs/rsl_rl/kscale_flat"
TAG = {"2026-09-24_05-18-33": "A", "2026-09-25_08-00-34": "B",
       "2026-09-25_11-29-39": "C", "2026-09-28_10-25-57": "D",
       "2026-09-29_06-15-58": "E"}
FORCE = 1.0
G = 9.81


def foot_index(joint_name):
    """The foot a joint belongs to. Index 0 is the left.

    The foot links carry no side marker, being foot_6061_2 and foot_6061, so the side
    cannot be read from the name. Three facts establish the mapping, the articulation body
    order placing every left link before its right counterpart, the lateral separation of
    foot zero minus foot one standing positive in the base frame, and the right hip yaw
    correlating with foot one's toe-in at 0.95 against 0.18 for foot zero.
    """
    return 0 if joint_name.startswith("left") else 1


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


def per_joint(params, key, names, default=np.nan):
    """A per joint vector from a regex keyed dictionary in the dumped action config."""
    out = np.full(len(names), default, dtype=float)
    table = (params["actions"]["joint_pos"] or {}).get(key)
    if isinstance(table, dict):
        for expr, v in table.items():
            for j, n in enumerate(names):
                if re.fullmatch(expr, n):
                    out[j] = float(v)
    elif table is not None:
        out[:] = float(table)
    return out


def clip_bounds(params, names):
    """The (lo, hi) clip of every joint, NaN where the term clips nothing."""
    lo = np.full(len(names), np.nan)
    hi = np.full(len(names), np.nan)
    table = (params["actions"]["joint_pos"] or {}).get("clip")
    if isinstance(table, dict):
        for expr, pair in table.items():
            for j, n in enumerate(names):
                if re.fullmatch(expr, n):
                    lo[j], hi[j] = float(pair[0]), float(pair[1])
    return lo, hi


def analyse(job):
    base = os.path.join(EVAL, job, "data", "42")
    d = np.load(os.path.join(base, "dump.npy"), allow_pickle=True).item()
    rw = np.load(os.path.join(base, "rewards.npy"), allow_pickle=True).item()
    exp = job.rsplit("_model_", 1)[0]
    params = ep.load_params(os.path.join(LOGS, exp))
    names = [str(n) for n in d["joint_names"]]
    dt = float(d["step_dt"])
    pos = np.asarray(d["joint_positions"])
    vel = np.asarray(d["joint_velocities"])
    tau = np.asarray(d["joint_torques"])
    nominal = np.asarray(d["default_joint_pos"])
    lim = np.asarray(d["joint_position_limits"])
    soft = np.asarray(d["joint_soft_position_limits"])
    contact = np.linalg.norm(np.asarray(d["feet_contact_forces"]), axis=-1) > FORCE
    mass = float(np.asarray(d["robot_mass"])[0, 0].sum())

    out = {"job": job, "tag": TAG.get(exp, "?"), "experiment": exp,
           "checkpoint": int(job.rsplit("_", 1)[-1]), "dt": dt, "mass": mass,
           "names": names, "envs": int(params["scene"]["num_envs"])}

    # --- saturation, EXACT, the computed torque against the actuator's own envelope ----
    computed = np.asarray(d.get("joint_computed_torque"))
    avail = np.asarray(d.get("joint_available_torque"))    # (T, E, J, 2), lower and upper
    exact = computed.size and avail.size
    sat_frac, hold, util = {}, {}, {}
    for j, n in enumerate(names):
        col = np.abs(tau[:, :, j])
        hold[n] = float(col.mean())
        if exact:
            hot = (computed[:, :, j] > avail[:, :, j, 1]) | \
                  (computed[:, :, j] < avail[:, :, j, 0])
            sat_frac[n] = float(100.0 * hot.mean())
            bound = np.where(tau[:, :, j] >= 0, avail[:, :, j, 1], -avail[:, :, j, 0])
            util[n] = float(100.0 * np.mean(col / np.maximum(bound, 1e-9)))
    out["saturation_exact"] = sat_frac
    out["mean_abs_torque"] = hold
    out["utilisation"] = util
    out["effort_limit"] = {n: float(np.asarray(d["joint_effort_limits"])[j])
                           for j, n in enumerate(names)}

    # --- the commanded target, READ rather than inferred -------------------------------
    cmd = np.asarray(d.get("joint_position_targets"))
    raw = np.asarray(d.get("joint_action_raw"))
    scale = per_joint(params, "scale", names)
    lo, hi = clip_bounds(params, names)
    out["clip_lo"] = {n: float(lo[j]) for j, n in enumerate(names)}
    out["clip_hi"] = {n: float(hi[j]) for j, n in enumerate(names)}
    out["scale"] = {n: float(scale[j]) for j, n in enumerate(names)}
    have_cmd = cmd.size > 0

    command = {}
    if have_cmd:
        # The UNCLIPPED target the policy asked for, which is what the clip removed and
        # what arm 11 prices.
        #
        # The offset must be RECOVERED per environment rather than taken from the dump's
        # default_joint_pos, which is environment zero's row alone. randomize_joint_default_pos
        # adds up to 0.05 rad per environment at reset, so environment zero's row is wrong
        # for the other thirty one by up to that much, and an excess computed against it is
        # never exactly zero even on a run that carries no clip at all. Since the offset is
        # constant in time within an episode and the clip binds on a minority of samples at
        # most joints, the median over time of the recorded target minus the scaled raw
        # action recovers it exactly on any joint the clip does not bind always, and to
        # within the clip's own bound where it does. This reproduces the term._offset that
        # action_clip_excess_l2 itself uses at rewards.py:2225.
        # The recovery must be restricted to the samples the clip did NOT bind, since on a
        # bound sample the recorded target is the clip bound rather than the policy's own
        # and the difference carries no information about the offset. On run E the hip roll
        # clip binds on more than ninety per cent of samples, and a median taken over all
        # of them recovers a figure 3.1 rad from the dumped row rather than the 0.048 the
        # randomisation can account for, which is how this condition was found.
        eps = 1e-6
        free = np.ones_like(cmd, dtype=bool)
        for j in range(len(names)):
            if np.isfinite(lo[j]):
                free[:, :, j] = (cmd[:, :, j] > lo[j] + eps) & (cmd[:, :, j] < hi[j] - eps)
        offset = np.empty(cmd.shape[1:])                       # (E, J)
        support = np.zeros(len(names))
        for j in range(len(names)):
            for e in range(cmd.shape[1]):
                m = free[:, e, j]
                offset[e, j] = (np.median((cmd[:, e, j] - raw[:, e, j] * scale[j])[m])
                                if m.any() else nominal[j])
            support[j] = float(100.0 * free[:, :, j].mean())
        out["offset_support_pct"] = {n: float(support[j]) for j, n in enumerate(names)}
        unclipped = raw * scale + offset[None, :, :]
        excess = unclipped - cmd
        for j, n in enumerate(names):
            dev = cmd[:, :, j] - nominal[j]
            beyond = (cmd[:, :, j] < lim[j, 0]) | (cmd[:, :, j] > lim[j, 1])
            unb = (unclipped[:, :, j] < lim[j, 0]) | (unclipped[:, :, j] > lim[j, 1])
            command[n] = {
                "cmd_dev_abs_mean": float(np.abs(dev).mean()),
                "cmd_dev_p95": float(np.percentile(np.abs(dev), 95)),
                "track_err_abs_mean": float(np.abs(cmd[:, :, j] - pos[:, :, j]).mean()),
                "cmd_beyond_stop_pct": float(100.0 * beyond.mean()),
                "unclipped_beyond_stop_pct": float(100.0 * unb.mean()),
                "excess_abs_mean": float(np.abs(excess[:, :, j]).mean()),
                "excess_sq_mean": float(np.mean(np.square(excess[:, :, j]))),
                "clipped_pct": float(100.0 * np.mean(np.abs(excess[:, :, j]) > 1e-6)),
                "raw_abs_mean": float(np.abs(raw[:, :, j]).mean()),
                "raw_p95": float(np.percentile(np.abs(raw[:, :, j]), 95)),
            }
        # The term itself, being the sum over joints of the squared excess per step,
        # which is exactly what action_clip_excess_l2 returns at rewards.py:2226.
        out["clip_excess_term"] = float(np.mean(np.sum(np.square(excess), axis=-1)))
        # How far the recovered offset stands from the dumped environment zero row, which
        # is the randomisation this recovery exists to remove and which should not exceed
        # the 0.05 rad that randomize_joint_default_pos adds.
        out["offset_recovery_max"] = float(np.abs(offset - nominal[None, :]).max())
    out["command"] = command

    # --- the saturating deflection, which is what the clip must respect to bind ---------
    # q_sat is the position error at which the servo reaches its effort ceiling at zero
    # speed. A clip wider than q_sat about the joint's own position cannot prevent
    # saturation, which is the arithmetic the ablation's third question turns upon.
    groups = ep.actuator_groups(params) or {}
    kp = np.full(len(names), np.nan)
    eff = np.full(len(names), np.nan)
    for g in groups.values():
        exprs = g.get("joint_names_expr") or []
        if isinstance(exprs, str):
            exprs = [exprs]
        for j, n in enumerate(names):
            if any(re.fullmatch(e, n) for e in exprs):
                s = g.get("stiffness")
                if isinstance(s, dict):
                    s = list(s.values())[0]
                if s is not None:
                    kp[j] = float(s)
                if g.get("effort_limit") is not None:
                    eff[j] = float(g["effort_limit"])
    out["q_sat"] = {n: float(eff[j] / kp[j]) if kp[j] else float("nan")
                    for j, n in enumerate(names)}
    out["kp"] = {n: float(kp[j]) for j, n in enumerate(names)}
    out["clip_width"] = {n: (float(hi[j] - lo[j]) if np.isfinite(lo[j]) else float("nan"))
                         for j, n in enumerate(names)}

    # --- joint position statistics and phase conditioning ------------------------------
    stats, phase = {}, {}
    for j, n in enumerate(names):
        dev = pos[:, :, j] - nominal[j]
        tol = 0.01 * (lim[j, 1] - lim[j, 0])
        at_stop = (pos[:, :, j] <= lim[j, 0] + tol) | (pos[:, :, j] >= lim[j, 1] - tol)
        stats[n] = {
            "dev_abs_mean": float(np.abs(dev).mean()),
            "pct_at_stop": float(100.0 * at_stop.mean()),
            "pct_outside_soft": float(100.0 * np.mean(
                (pos[:, :, j] < soft[j, 0]) | (pos[:, :, j] > soft[j, 1]))),
            "speed_abs_mean": float(np.abs(vel[:, :, j]).mean()),
            "nominal": float(nominal[j]),
            "limits": [float(lim[j, 0]), float(lim[j, 1])],
        }
        st = contact[:, :, foot_index(n)]
        entry = {"stance_frac": float(100.0 * st.mean()),
                 "dev_stance": float(np.abs(dev)[st].mean()),
                 "dev_swing": float(np.abs(dev)[~st].mean()),
                 "speed_stance": float(np.abs(vel[:, :, j])[st].mean()),
                 "speed_swing": float(np.abs(vel[:, :, j])[~st].mean()),
                 "stop_stance": float(100.0 * at_stop[st].mean()),
                 "stop_swing": float(100.0 * at_stop[~st].mean())}
        if exact:
            hot = (computed[:, :, j] > avail[:, :, j, 1]) | \
                  (computed[:, :, j] < avail[:, :, j, 0])
            entry["sat_stance"] = float(100.0 * hot[st].mean())
            entry["sat_swing"] = float(100.0 * hot[~st].mean())
        if have_cmd:
            entry["cmd_dev_stance"] = float(np.abs(cmd[:, :, j] - nominal[j])[st].mean())
            entry["cmd_dev_swing"] = float(np.abs(cmd[:, :, j] - nominal[j])[~st].mean())
        phase[n] = entry
    out["joint"] = stats
    out["phase"] = phase

    # --- the sole tilt, recorded rather than reconstructed ------------------------------
    tilt = np.asarray(d.get("feet_sole_tilt"))
    if tilt.size:
        out["sole"] = {
            "roll_abs_stance_deg": float(np.degrees(np.abs(tilt[:, :, :, 0]))[contact].mean()),
            "roll_signed_stance_deg": float(np.degrees(tilt[:, :, :, 0])[contact].mean()),
            "pitch_abs_stance_deg": float(np.degrees(np.abs(tilt[:, :, :, 1]))[contact].mean()),
            "roll_p95_stance_deg": float(np.percentile(
                np.degrees(np.abs(tilt[:, :, :, 0]))[contact], 95)),
        }

    # --- splay, and the hip yaw that sets it --------------------------------------------
    fq, bq = np.asarray(d["feet_quaternions"]), np.asarray(d["base_quaternion"])
    by = yaw_of(bq)
    rel = [wrap(np.arctan2(rot(fq[:, :, f, :], (0.0, 0.0, -1.0))[..., 1],
                           rot(fq[:, :, f, :], (0.0, 0.0, -1.0))[..., 0]) - by)
           for f in range(2)]
    toe_in = np.degrees(np.stack([-rel[0], rel[1]], axis=-1))
    out["splay"] = {
        "abs_mean": float(np.abs(toe_in).mean()),
        "signed_mean": float(toe_in.mean()),
        "p95": float(np.percentile(np.abs(toe_in), 95)),
        "stance_abs_mean": float(np.abs(toe_in)[contact].mean()),
        "swing_abs_mean": float(np.abs(toe_in)[~contact].mean()),
        "differential": float(np.abs(np.degrees(wrap(rel[1] - rel[0]))).mean()),
    }
    hy = {n: j for j, n in enumerate(names) if "hip_yaw" in n}
    pairs = []
    for side, f, sgn in (("left", 0, -1.0), ("right", 1, 1.0)):
        j = hy[f"{side}_hip_yaw_03"]
        for ph, m in (("stance", contact[:, :, f]), ("swing", ~contact[:, :, f])):
            x = np.degrees(sgn * (pos[:, :, j] - nominal[j]))[m]
            y = (sgn * toe_in[:, :, f] * sgn)[m] if False else toe_in[:, :, f][m]
            a, b = np.polyfit(x, y, 1)
            pairs.append({"side": side, "phase": ph, "slope": float(a),
                          "intercept": float(b),
                          "corr": float(np.corrcoef(x, y)[0, 1]),
                          "toe_in": float(y.mean())})
    out["hip_yaw_splay"] = pairs
    L = (pos[:, :, hy["left_hip_yaw_03"]] - nominal[hy["left_hip_yaw_03"]]).ravel()
    R = (pos[:, :, hy["right_hip_yaw_03"]] - nominal[hy["right_hip_yaw_03"]]).ravel()
    out["hip_yaw_mode"] = {"corr_raw": float(np.corrcoef(L, R)[0, 1]),
                           "sum_std": float(np.std(L + R)),
                           "diff_std": float(np.std(L - R))}

    # --- gait and viability --------------------------------------------------------------
    knee = [j for j, n in enumerate(names) if "knee" in n]
    height = np.asarray(d["base_com_position"])[:, :, 2]
    linv = np.asarray(d["base_linear_velocity"])
    cmdv = np.asarray(d["commanded_linear_velocity"])
    ang = np.asarray(d["base_angular_velocity"])
    cyaw = np.asarray(d["commanded_angular_velocity"])
    power = np.abs(np.asarray(d["joint_powers"])).sum(axis=-1)
    speed = np.linalg.norm(linv[:, :, :2], axis=-1)
    term = np.asarray(d["episode_terminated"])
    st0 = contact[:, :, 0]
    lift0 = (st0[:-1] & ~st0[1:])
    out["gait"] = {
        "base_height": float(height.mean()),
        "knee_mean": float(pos[:, :, knee].mean()),
        "planar_error": float(np.mean(np.linalg.norm(linv[:, :, :2] - cmdv[:, :, :2], axis=-1))),
        "yaw_error": float(np.mean(np.abs(ang[:, :, 2] - cyaw))),
        "abs_power": float(power.mean()),
        "cost_of_transport": float(power.mean() / (mass * G * max(speed.mean(), 1e-6))),
        "double_support_pct": float(100.0 * contact.all(axis=-1).mean()),
        "cadence": float(lift0.sum() / (contact.shape[1] * contact.shape[0] * dt) * 2),
        "termination_rate": float(term.sum() / (term.shape[1] * term.shape[0] * dt)),
        "torso_roll_deg": float(np.degrees(np.abs(np.arcsin(np.clip(
            -np.asarray(d["base_projected_gravity"])[:, :, 1], -1, 1)))).mean()),
        "feet_lateral_sep": float(np.abs(np.asarray(d["feet_distance"])[:, :, 1]).mean()),
    }

    # --- the reward budget ----------------------------------------------------------------
    terms = {k: float(np.asarray(v).mean()) for k, v in rw.items()
             if k not in ("_weights", "total_reward")}
    out["reward"] = {"total_per_step": float(np.asarray(rw["total_reward"]).mean()),
                     "terms": terms,
                     "negative_total": sum(v for v in terms.values() if v < 0),
                     "positive_total": sum(v for v in terms.values() if v > 0)}
    return out


def main():
    recs = [analyse(j) for j in sys.argv[1:]]
    path = os.path.join(EVAL, "clip_ablation_analysis.json")
    with open(path, "w") as h:
        json.dump(recs, h, indent=1)
    print(f"wrote {path} for {len(recs)} jobs")


if __name__ == "__main__":
    main()
