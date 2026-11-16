"""Whether the heading gate of arm 15 may be closed, measured rather than argued.

The arm proposes `airborne_only: True` on `pen_feet_heading`, which deletes every stance
sample from the common mode of `feet_yaw_alignment`.  Two questions decide whether that is
safe and neither was asked when the arm was written against runs A, B and C.

The first is how large the stance error now is, the gate having been justified by a stance
toe-in four to eighteen times smaller than the swing figure.

The second is whether a stance heading is INHERITED from the swing that planted it or
GENERATED during stance.  If inherited, a term that prices only swing still governs it and
the gate costs nothing.  If generated, the gate removes the only gradient that opposes it.
The test is the toe-in at the touchdown sample against the toe-in at the lift off sample of
the same contact interval, both measured on the same foot of the same environment.

The third quantity is the term's own arithmetic, being what the gate deletes from the value
`pen_feet_heading` reports, and what the term becomes in double support where no foot is
airborne at all and the gated common mode is identically zero by rewards.py:1330.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, "/workspace/isaaclab/biped/scripts/analysis")
import experiment_params as ep

EVAL = "/workspace/isaaclab/logs/rsl_rl/kscale_flat_eval"
TAG = {"2026-09-24_05-18-33": "A", "2026-09-25_08-00-34": "B",
       "2026-09-25_11-29-39": "C", "2026-09-28_10-25-57": "D",
       "2026-09-29_06-15-58": "E"}
FORCE = 1.0
TOL = 0.10


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


def band(e, tol=TOL):
    return np.sign(e) * np.clip(np.abs(e) - tol, 0.0, None)


def analyse(job):
    base = os.path.join(EVAL, job, "data", "42")
    d = np.load(os.path.join(base, "dump.npy"), allow_pickle=True).item()
    exp = job.rsplit("_model_", 1)[0]
    names = list(d["joint_names"])
    pos = np.asarray(d["joint_positions"])
    nominal = np.asarray(d["default_joint_pos"])
    contact = np.linalg.norm(np.asarray(d["feet_contact_forces"]), axis=-1) > FORCE   # (T,E,2)
    fq, bq = np.asarray(d["feet_quaternions"]), np.asarray(d["base_quaternion"])
    by = yaw_of(bq)

    # heading of each foot relative to the base, the KScale toe axis being -z of the link
    rel = np.stack([wrap(np.arctan2(rot(fq[:, :, f, :], (0.0, 0.0, -1.0))[..., 1],
                                    rot(fq[:, :, f, :], (0.0, 0.0, -1.0))[..., 0]) - by)
                    for f in range(2)], axis=-1)                                       # (T,E,2)
    toe_in = np.degrees(np.stack([-rel[..., 0], rel[..., 1]], axis=-1))                # (T,E,2)

    out = {"job": job, "tag": TAG.get(exp, "?"), "checkpoint": int(job.rsplit("_", 1)[-1])}

    # --- 1. the level of the error, by phase ---------------------------------------------
    out["level"] = {
        "stance_abs": float(np.abs(toe_in)[contact].mean()),
        "swing_abs": float(np.abs(toe_in)[~contact].mean()),
        "ratio_swing_over_stance": float(np.abs(toe_in)[~contact].mean()
                                         / max(np.abs(toe_in)[contact].mean(), 1e-9)),
        "stance_p95": float(np.percentile(np.abs(toe_in)[contact], 95)),
    }

    # --- 2. inherited or generated -------------------------------------------------------
    # A contact interval runs from the first sample of contact to the last before release.
    td_abs, lo_abs, mid_abs, growth, dur = [], [], [], [], []
    pre_abs = []
    for f in range(2):
        c = contact[:, :, f]
        td = (~c[:-1]) & c[1:]          # index t means touchdown at t+1
        lo = c[:-1] & (~c[1:])           # index t means last stance sample is t
        ti, ei = np.nonzero(td)
        for t, e in zip(ti, ei):
            t0 = t + 1
            nxt = np.nonzero(lo[t0:, e])[0]
            if nxt.size == 0:
                continue
            t1 = t0 + nxt[0]
            if t1 <= t0:
                continue
            seg = np.abs(toe_in[t0:t1 + 1, e, f])
            td_abs.append(abs(toe_in[t0, e, f]))
            lo_abs.append(abs(toe_in[t1, e, f]))
            mid_abs.append(seg.mean())
            growth.append(abs(toe_in[t1, e, f]) - abs(toe_in[t0, e, f]))
            dur.append(t1 - t0 + 1)
            pre_abs.append(abs(toe_in[t, e, f]))          # last swing sample before contact
    out["interval"] = {
        "n": len(td_abs),
        "last_swing_abs": float(np.mean(pre_abs)) if pre_abs else float("nan"),
        "touchdown_abs": float(np.mean(td_abs)) if td_abs else float("nan"),
        "stance_mean_abs": float(np.mean(mid_abs)) if mid_abs else float("nan"),
        "liftoff_abs": float(np.mean(lo_abs)) if lo_abs else float("nan"),
        "growth_liftoff_minus_touchdown": float(np.mean(growth)) if growth else float("nan"),
        "growth_p50": float(np.median(growth)) if growth else float("nan"),
        "frac_intervals_growing": float(np.mean(np.asarray(growth) > 0)) if growth else float("nan"),
        "mean_stance_samples": float(np.mean(dur)) if dur else float("nan"),
    }

    # --- 3. the term's own arithmetic ----------------------------------------------------
    airborne = ~contact
    sin_all = np.sin(rel).sum(axis=-1)
    cos_all = np.cos(rel).sum(axis=-1)
    common_ungated = np.square(band(wrap(np.arctan2(sin_all, cos_all))))
    m = airborne.astype(float)
    sin_g = (np.sin(rel) * m).sum(axis=-1)
    cos_g = (np.cos(rel) * m).sum(axis=-1)
    any_air = airborne.any(axis=-1)
    common_gated = np.where(any_air, np.square(band(wrap(np.arctan2(sin_g, cos_g)))), 0.0)
    differential = np.square(band(wrap(rel[..., 1] - rel[..., 0])))

    both_down = ~any_air
    out["term"] = {
        "common_ungated": float(common_ungated.mean()),
        "common_gated": float(common_gated.mean()),
        "gate_removes_pct": float(100.0 * (1.0 - common_gated.mean()
                                           / max(common_ungated.mean(), 1e-12))),
        "differential": float(differential.mean()),
        "double_support_pct": float(100.0 * both_down.mean()),
        "common_ungated_in_double_support": float(common_ungated[both_down].mean())
            if both_down.any() else 0.0,
        "share_of_ungated_common_from_double_support_pct": float(
            100.0 * common_ungated[both_down].sum() / max(common_ungated.sum(), 1e-12)),
        "total_ungated_ds1": float(common_ungated.mean() + differential.mean()),
        "total_gated_ds1": float(common_gated.mean() + differential.mean()),
        "total_gated_ds05": float(common_gated.mean() + 0.5 * differential.mean()),
        "weighted_ungated_ds1": float(-8.0 * (common_ungated.mean() + differential.mean())),
        "weighted_gated_ds05": float(-8.0 * (common_gated.mean() + 0.5 * differential.mean())),
    }

    # --- 4. the hip yaw behind it, by phase ----------------------------------------------
    hy = {n: j for j, n in enumerate(names) if "hip_yaw" in n}
    hydev = {}
    for side, f, sgn in (("left", 0, -1.0), ("right", 1, 1.0)):
        j = hy[f"{side}_hip_yaw_03"]
        dev = np.degrees(sgn * (pos[:, :, j] - nominal[j]))
        hydev[f"{side}_stance_signed"] = float(dev[contact[:, :, f]].mean())
        hydev[f"{side}_swing_signed"] = float(dev[~contact[:, :, f]].mean())
    out["hip_yaw_phase_deg"] = hydev
    return out


def main():
    jobs = sys.argv[1:] or [
        "2026-09-24_05-18-33_model_10000", "2026-09-25_08-00-34_model_10000",
        "2026-09-25_11-29-39_model_10000", "2026-09-28_10-25-57_model_10000",
        "2026-09-29_06-15-58_model_8000", "2026-09-29_06-15-58_model_9000",
        "2026-09-29_06-15-58_model_10000",
    ]
    res = [analyse(j) for j in jobs]
    with open(os.path.join(EVAL, "heading_gate_analysis.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for r in res:
        print(f"== {r['tag']}@{r['checkpoint']//1000}k ==")
        print(" level   ", json.dumps(r["level"]))
        print(" interval", json.dumps(r["interval"]))
        print(" term    ", json.dumps(r["term"]))
        print(" hipyaw  ", json.dumps(r["hip_yaw_phase_deg"]))


if __name__ == "__main__":
    main()
