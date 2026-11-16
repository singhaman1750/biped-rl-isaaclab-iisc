"""Render the tables of the 2026-09-30 analysis from clip_ablation_analysis.json."""
import json, sys
P = "/ws/IsaacLab/logs/rsl_rl/kscale_flat_eval/clip_ablation_analysis.json"
recs = json.load(open(P))
by = {r["job"]: r for r in recs}
sel = [by[s] for s in sys.argv[1:]] if len(sys.argv) > 1 else recs
def lab(r): return f"{r['tag']}@{r['checkpoint']//1000}k"
J = ["left_hip_yaw_03","right_hip_yaw_03","left_hip_roll_03","right_hip_roll_03",
     "left_hip_pitch_04","right_hip_pitch_04","left_knee_04","right_knee_04",
     "left_foot_pitch_02","right_foot_pitch_02","left_foot_roll_02","right_foot_roll_02"]

def tab(title, rows, w=36):
    print(f"\n### {title}")
    print(f"{'quantity':{w}s} " + " ".join(f"{lab(r):>9s}" for r in sel))
    print("-" * (w + 10 * len(sel)))
    for name, fn, fmt in rows:
        print(f"{name:{w}s} " + " ".join(f"{fn(r):>9{fmt}}" for r in sel))

def jtab(title, fn, fmt=">9.2f", w=26):
    print(f"\n### {title}")
    print(f"{'joint':{w}s} " + " ".join(f"{lab(r):>9s}" for r in sel))
    for j in J:
        print(f"{j:{w}s} " + " ".join(f"{fn(r, j):{fmt}}" for r in sel))

tab("Gait and viability", [
 ("base height [m]", lambda r: r["gait"]["base_height"], ".4f"),
 ("knee mean [rad]", lambda r: r["gait"]["knee_mean"], ".4f"),
 ("planar tracking error [m/s]", lambda r: r["gait"]["planar_error"], ".4f"),
 ("yaw tracking error [rad/s]", lambda r: r["gait"]["yaw_error"], ".4f"),
 ("absolute joint power [W]", lambda r: r["gait"]["abs_power"], ".2f"),
 ("cost of transport", lambda r: r["gait"]["cost_of_transport"], ".3f"),
 ("double support [pct]", lambda r: r["gait"]["double_support_pct"], ".2f"),
 ("cadence [steps/s]", lambda r: r["gait"]["cadence"], ".2f"),
 ("termination rate [1/s]", lambda r: r["gait"]["termination_rate"], ".4f"),
 ("torso roll [deg]", lambda r: r["gait"]["torso_roll_deg"], ".3f"),
 ("lateral foot separation [m]", lambda r: r["gait"]["feet_lateral_sep"], ".4f"),
 ("environments", lambda r: r["envs"], "d"),
])

tab("Foot splay, toe-in positive [deg]", [
 ("both feet, all samples", lambda r: r["splay"]["abs_mean"], ".3f"),
 ("both feet, signed", lambda r: r["splay"]["signed_mean"], ".3f"),
 ("ninety fifth percentile", lambda r: r["splay"]["p95"], ".3f"),
 ("in stance", lambda r: r["splay"]["stance_abs_mean"], ".3f"),
 ("in swing", lambda r: r["splay"]["swing_abs_mean"], ".3f"),
 ("differential", lambda r: r["splay"]["differential"], ".3f"),
 ("hip yaw common mode corr", lambda r: r["hip_yaw_mode"]["corr_raw"], ".3f"),
 ("hip yaw sum std / diff std", lambda r: r["hip_yaw_mode"]["sum_std"]/r["hip_yaw_mode"]["diff_std"], ".3f"),
])

tab("Sole tilt in stance [deg]", [
 ("roll, absolute", lambda r: r.get("sole",{}).get("roll_abs_stance_deg", float('nan')), ".3f"),
 ("roll, signed", lambda r: r.get("sole",{}).get("roll_signed_stance_deg", float('nan')), ".3f"),
 ("roll, ninety fifth percentile", lambda r: r.get("sole",{}).get("roll_p95_stance_deg", float('nan')), ".3f"),
 ("pitch, absolute", lambda r: r.get("sole",{}).get("pitch_abs_stance_deg", float('nan')), ".3f"),
])

jtab("EXACT saturation, computed torque outside the actuator envelope [pct]",
     lambda r, j: r["saturation_exact"].get(j, float('nan')))
jtab("Mean absolute torque [N.m]", lambda r, j: r["mean_abs_torque"][j], ">9.3f")
jtab("Time at a hard stop [pct]", lambda r, j: r["joint"][j]["pct_at_stop"])
jtab("COMMANDED target beyond the hard stop [pct]",
     lambda r, j: r["command"].get(j, {}).get("cmd_beyond_stop_pct", float('nan')))
jtab("UNCLIPPED target beyond the hard stop [pct]",
     lambda r, j: r["command"].get(j, {}).get("unclipped_beyond_stop_pct", float('nan')))
jtab("Samples the clip actually bound [pct]",
     lambda r, j: r["command"].get(j, {}).get("clipped_pct", float('nan')))
jtab("Mean |clip excess| [rad]",
     lambda r, j: r["command"].get(j, {}).get("excess_abs_mean", float('nan')), ">9.4f")
jtab("Commanded departure from nominal [rad]",
     lambda r, j: r["command"].get(j, {}).get("cmd_dev_abs_mean", float('nan')), ">9.4f")
jtab("Tracking error, commanded minus achieved [rad]",
     lambda r, j: r["command"].get(j, {}).get("track_err_abs_mean", float('nan')), ">9.4f")
jtab("Mean |raw action| [sigma]",
     lambda r, j: r["command"].get(j, {}).get("raw_abs_mean", float('nan')), ">9.3f")

print("\n### The clip against the saturating deflection, which decides whether it can bind")
print(f"{'joint':26s} {'q_sat=E/kp':>11s} {'clip width':>11s} {'ratio':>8s}")
r = sel[-1]
for j in J:
    q, w = r["q_sat"][j], r["clip_width"][j]
    print(f"{j:26s} {q:11.4f} {w:11.4f} {w/q:8.2f}" if w == w else
          f"{j:26s} {q:11.4f} {'unclipped':>11s} {'-':>8s}")

print("\n### The clip excess term, and what it would cost at each weight")
print(f"{'run':10s} {'sum sq excess/step':>19s} {'at -2':>9s} {'at -500':>9s} "
      f"{'neg budget':>11s} {'-2 share%':>10s} {'-500 share%':>12s}")
for r in sel:
    e = r.get("clip_excess_term")
    if e is None: continue
    neg = r["reward"]["negative_total"]
    print(f"{lab(r):10s} {e:19.6f} {2*e:9.4f} {500*e:9.3f} {neg:11.4f} "
          f"{100*2*e/abs(neg):10.2f} {100*500*e/abs(neg):12.2f}")

print("\n### Phase conditioned, stance then swing")
for title, key in (("EXACT saturation [pct]", ("sat_stance", "sat_swing")),
                   ("hard stop residency [pct]", ("stop_stance", "stop_swing")),
                   ("commanded departure [rad]", ("cmd_dev_stance", "cmd_dev_swing")),
                   ("joint speed [rad/s]", ("speed_stance", "speed_swing"))):
    print(f"\n  {title}")
    print(f"  {'joint':24s} " + " ".join(f"{lab(r):>19s}" for r in sel))
    for j in J:
        cells = []
        for r in sel:
            p = r["phase"][j]
            a, b = p.get(key[0], float('nan')), p.get(key[1], float('nan'))
            cells.append(f"{a:>8.2f}/{b:<8.2f}" if abs(a) >= 0.01 or abs(b) >= 0.01
                         else f"{a:>8.4f}/{b:<8.4f}")
        print(f"  {j:24s} " + " ".join(f"{c:>19s}" for c in cells))

print("\n### Hip yaw against toe-in, by phase, left leg mirrored")
print(f"{'run':10s} {'side':6s} {'phase':7s} {'slope':>8s} {'corr':>7s} {'intercept':>10s} {'toe-in':>8s}")
for r in sel:
    for p in r["hip_yaw_splay"]:
        print(f"{lab(r):10s} {p['side']:6s} {p['phase']:7s} {p['slope']:8.3f} "
              f"{p['corr']:7.3f} {p['intercept']:10.3f} {p['toe_in']:8.3f}")
