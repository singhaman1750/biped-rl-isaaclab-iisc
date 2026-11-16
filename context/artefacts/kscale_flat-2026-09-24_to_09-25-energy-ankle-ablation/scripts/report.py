"""Render the tables of the 2026-09-26 comparative analysis from ablation_analysis.json."""
import json, sys

P = "/ws/IsaacLab/logs/rsl_rl/kscale_flat_eval/ablation_analysis.json"
recs = json.load(open(P))
by = {r["job"]: r for r in recs}
SEL = sys.argv[1:] or [r["job"] for r in recs]
sel = [by[s] for s in SEL]
def lab(r): return f"{r['tag']}@{r['checkpoint']}"

def tab(title, rows, keyw=34):
    print(f"\n### {title}")
    print(f"{'quantity':{keyw}s} " + " ".join(f"{lab(r):>10s}" for r in sel))
    print("-" * (keyw + 11 * len(sel)))
    for name, fn, fmt in rows:
        print(f"{name:{keyw}s} " + " ".join(f"{fn(r):>10{fmt}}" for r in sel))

JOINTS = ["left_hip_yaw_03","right_hip_yaw_03","left_hip_roll_03","right_hip_roll_03",
          "left_hip_pitch_04","right_hip_pitch_04","left_knee_04","right_knee_04",
          "left_foot_pitch_02","right_foot_pitch_02","left_foot_roll_02","right_foot_roll_02"]

tab("Gait and viability", [
 ("base height [m]", lambda r: r["gait"]["base_height"], ".4f"),
 ("knee mean [rad]", lambda r: r["gait"]["knee_mean"], ".4f"),
 ("planar tracking error [m/s]", lambda r: r["gait"]["planar_error"], ".4f"),
 ("yaw tracking error [rad/s]", lambda r: r["gait"]["yaw_error"], ".4f"),
 ("absolute joint power [W]", lambda r: r["gait"]["abs_power"], ".2f"),
 ("cost of transport", lambda r: r["gait"]["cost_of_transport"], ".3f"),
 ("double support [pct]", lambda r: r["gait"]["double_support_pct"], ".2f"),
 ("flight phase [pct]", lambda r: r["gait"]["flight_pct"], ".2f"),
 ("cadence [steps/s]", lambda r: r["gait"]["cadence"], ".2f"),
 ("termination rate [1/s]", lambda r: r["gait"]["termination_rate"], ".4f"),
 ("torso roll [deg]", lambda r: r["gait"]["torso_roll_deg"], ".3f"),
 ("lateral foot separation [m]", lambda r: r["gait"]["feet_lateral_sep"], ".4f"),
 ("sole clearance in stance [mm]", lambda r: r["gait"]["sole_clearance_stance_mm"], ".2f"),
])

tab("Foot splay, toe-in positive", [
 ("both feet, all samples [deg]", lambda r: r["splay"]["abs_mean"], ".3f"),
 ("both feet, signed [deg]", lambda r: r["splay"]["signed_mean"], ".3f"),
 ("ninety fifth percentile [deg]", lambda r: r["splay"]["p95"], ".3f"),
 ("differential [deg]", lambda r: r["splay"]["differential"], ".3f"),
 ("in stance, absolute [deg]", lambda r: r["splay"]["stance_abs_mean"], ".3f"),
 ("in stance, signed [deg]", lambda r: r["splay"]["stance_signed_mean"], ".3f"),
 ("in swing, absolute [deg]", lambda r: r["splay"]["swing_abs_mean"], ".3f"),
 ("hip yaw to splay slope", lambda r: r.get("hip_yaw_vs_splay",{}).get("slope_deg_per_deg",float('nan')), ".3f"),
 ("hip yaw to splay correlation", lambda r: r.get("hip_yaw_vs_splay",{}).get("correlation",float('nan')), ".3f"),
])

print("\n### Envelope saturation by joint [pct of samples at or above 98 pct of available torque]")
print(f"{'joint':26s} " + " ".join(f"{lab(r):>10s}" for r in sel))
for j in JOINTS:
    print(f"{j:26s} " + " ".join(f"{r['saturation_envelope'][j]:>10.2f}" for r in sel))
print(f"{'-- flat ceiling reading --':26s}")
for j in JOINTS:
    print(f"{j:26s} " + " ".join(f"{r['saturation_ceiling'][j]:>10.2f}" for r in sel))

print("\n### Mean absolute torque [N.m], against the effort ceiling in the last column of each")
print(f"{'joint':26s} " + " ".join(f"{lab(r):>10s}" for r in sel) + "   ceiling")
for j in JOINTS:
    print(f"{j:26s} " + " ".join(f"{r['mean_abs_torque'][j]:>10.3f}" for r in sel)
          + f"   {sel[0]['effort_limit'][j]:.1f}")

print("\n### Joint excursion, absolute deviation from the nominal pose [rad]")
print(f"{'joint':26s} " + " ".join(f"{lab(r):>10s}" for r in sel))
for j in JOINTS:
    print(f"{j:26s} " + " ".join(f"{r['joint'][j]['dev_abs_mean']:>10.4f}" for r in sel))

print("\n### Time at a hard stop [pct of samples within one per cent of travel]")
print(f"{'joint':26s} " + " ".join(f"{lab(r):>10s}" for r in sel))
for j in JOINTS:
    print(f"{j:26s} " + " ".join(f"{r['joint'][j]['pct_at_stop']:>10.2f}" for r in sel))

print("\n### Outside the soft limit [pct of samples]")
print(f"{'joint':26s} " + " ".join(f"{lab(r):>10s}" for r in sel))
for j in JOINTS:
    print(f"{j:26s} " + " ".join(f"{r['joint'][j]['pct_outside_soft']:>10.2f}" for r in sel))

print("\n### Phase conditioned saturation, stance then swing [pct]")
print(f"{'joint':26s} " + " ".join(f"{lab(r):>21s}" for r in sel))
for j in JOINTS:
    cells = []
    for r in sel:
        p = r["phase"][j]
        cells.append(f"{p['sat_stance']:>9.2f}/{p['sat_swing']:<9.2f}")
    print(f"{j:26s} " + " ".join(f"{c:>21s}" for c in cells))

print("\n### Phase conditioned deviation from nominal, stance then swing [rad]")
print(f"{'joint':26s} " + " ".join(f"{lab(r):>21s}" for r in sel))
for j in JOINTS:
    cells = []
    for r in sel:
        p = r["phase"][j]
        cells.append(f"{p['dev_abs_stance']:>9.4f}/{p['dev_abs_swing']:<9.4f}")
    print(f"{j:26s} " + " ".join(f"{c:>21s}" for c in cells))

print("\n### Phase conditioned joint speed, stance then swing [rad/s]")
print(f"{'joint':26s} " + " ".join(f"{lab(r):>21s}" for r in sel))
for j in JOINTS:
    cells = []
    for r in sel:
        p = r["phase"][j]
        cells.append(f"{p['speed_stance']:>9.3f}/{p['speed_swing']:<9.3f}")
    print(f"{j:26s} " + " ".join(f"{c:>21s}" for c in cells))

print("\n### The liftoff transient, mean absolute deviation from the nominal [rad]")
print("    pre is the 0.30 s of stance before the event, post the 0.30 s of swing after")
print(f"{'joint / run':26s} {'pre':>9s} {'at':>9s} {'post':>9s} {'peak':>9s} {'t_peak[s]':>10s} {'events':>8s}")
for j in ["left_hip_yaw_03","right_hip_yaw_03","left_foot_roll_02","right_foot_roll_02",
          "left_hip_roll_03","right_hip_roll_03","left_foot_pitch_02","right_foot_pitch_02"]:
    for r in sel:
        t = r["transition"].get(j, {}).get("liftoff")
        if not t: continue
        print(f"{j[:18]+' '+lab(r):26s} {t['pre']:>9.4f} {t['at']:>9.4f} {t['post']:>9.4f}"
              f" {t['peak']:>9.4f} {t['peak_offset_s']:>10.3f} {t['events']:>8d}")
    print()
