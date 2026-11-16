"""Sagittal closure, the knee against the height, and the off axis excursion budget."""
import numpy as np

BASE = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"A 09-17_09-13": "2026-09-17_09-13-46", "B 09-17_11-25": "2026-09-17_11-25-59",
        "C 09-18_09-37": "2026-09-18_09-37-29", "D 09-18_10-03": "2026-09-18_10-03-48"}
D = {l: np.load(f"{BASE}/{r}/data/42/dump.npy", allow_pickle=True).item() for l, r in RUNS.items()}
JN = D["A 09-17_09-13"]["joint_names"]
IX = {j: i for i, j in enumerate(JN)}
dfl = np.array(D["A 09-17_09-13"]["default_joint_pos"])

print("SAGITTAL CLOSURE IDENTITY, the sole is flat exactly when the leg's sagittal sum vanishes")
print("right leg   s = q_hip_pitch + q_knee + q_ankle_pitch")
print("left  leg   s = -q_hip_pitch + q_knee + q_ankle_pitch   (the hip pitch axes are anti parallel)")
print()
print("check against the configured nominal pose, which context/KScale.md section 16.2 states is closed")
print(f"   right  {-0.20369 + 0.51239 - 0.3087:+.6f}")
print(f"   left   {-(+0.20369) + 0.51239 - 0.3087:+.6f}")
print()
print(f"{'quantity':<46}{'A':>12}{'B':>12}{'C':>12}{'D':>12}")
for side, sgn in [("right", +1.0), ("left", -1.0)]:
    hp = IX[f"{side}_hip_pitch_04"]
    kn = IX[f"{side}_knee_04"]
    ap = IX[f"{side}_foot_pitch_02"]
    cells = []
    for l in RUNS:
        q = D[l]["joint_positions"]
        s = sgn * q[:, :, hp] + q[:, :, kn] + q[:, :, ap]
        cells.append(f"{s.mean():12.4f}")
    print(f"  sagittal sum {side}, mean [rad]{'':17}{''.join(cells)}")
for side, sgn in [("right", +1.0), ("left", -1.0)]:
    hp, kn, ap = IX[f"{side}_hip_pitch_04"], IX[f"{side}_knee_04"], IX[f"{side}_foot_pitch_02"]
    cells = []
    for l in RUNS:
        q = D[l]["joint_positions"]
        s = sgn * q[:, :, hp] + q[:, :, kn] + q[:, :, ap]
        cells.append(f"{np.degrees(np.abs(s)).mean():12.3f}")
    print(f"  |sole tilt| {side}, mean [deg]{'':18}{''.join(cells)}")
print()
print("  the ankle pitch closure demand at the observed knee, and whether the joint can reach it")
print("  closure asks ankle_pitch = -(knee) + hip_pitch contribution, the hard lower stop is -0.87267 rad")
for l in RUNS:
    q = D[l]["joint_positions"]
    kn = q[:, :, IX["right_knee_04"]].mean()
    hp = q[:, :, IX["right_hip_pitch_04"]].mean()
    need = -(kn + hp)
    ap = q[:, :, IX["right_foot_pitch_02"]].mean()
    print(f"    {l}: knee {kn:6.4f}, hip pitch {hp:+7.4f} -> closure needs ankle {need:+7.4f},"
          f" actual {ap:+7.4f}, stop -0.87267, reachable {need >= -0.87267}")
print()
print("  the knee at which closure alone exhausts the ankle pitch travel, with the hip pitch at nominal")
print(f"    knee_max = 0.87267 + 0.20369 = {0.87267 - 0.20369:.5f} rad with the nominal hip pitch,"
      f" and {0.87267:.5f} rad with the hip pitch at zero")

print()
print("=" * 112)
print("KNEE AGAINST BASE HEIGHT, measured, against the swept table of context/KScale.md section 16.3")
SWEEP = [(0.00, 0.77015), (0.10, 0.77484), (0.20, 0.77666), (0.30, 0.77558), (0.40, 0.77161),
         (0.50, 0.76479), (0.60, 0.75519), (0.70, 0.74290), (0.80, 0.72804), (0.90, 0.71077),
         (0.97, 0.69734)]
print("  swept (knee, height) with the sole held flat:")
print("   ", "  ".join(f"{k:.2f}->{h:.3f}" for k, h in SWEEP))
print()
print(f"  {'run':<16}{'mean knee':>12}{'mean height':>14}{'swept height':>14}{'measured minus swept':>22}")
for l in RUNS:
    q = D[l]["joint_positions"]
    kn = 0.5 * (q[:, :, IX["left_knee_04"]] + q[:, :, IX["right_knee_04"]]).mean()
    h = D[l]["base_com_position"][:, :, 2].mean()
    ks = np.array([k for k, _ in SWEEP])
    hs = np.array([v for _, v in SWEEP])
    swept = np.interp(kn, ks, hs)
    print(f"  {l:<16}{kn:12.4f}{h:14.4f}{swept:14.4f}{h-swept:22.4f}")
print()
print("  the swept table stops at a knee of 0.97267, the closure cap, so a measured knee beyond it")
print("  describes a pose the flat sole condition cannot reach and the extrapolation is not quoted.")

print()
print("=" * 112)
print("OFF AXIS EXCURSION BUDGET, mean absolute deviation from the per environment default [rad]")
print("the addition of 2026-09-02 to context/KScale.md records this behaving as a budget the policy")
print("allocates rather than a quantity it minimises")
print(f"  {'joint family':<30}{'A':>13}{'B':>13}{'C':>13}{'D':>13}")
fam = {"hip yaw": ["left_hip_yaw_03", "right_hip_yaw_03"],
       "hip roll": ["left_hip_roll_03", "right_hip_roll_03"],
       "knee": ["left_knee_04", "right_knee_04"],
       "hip pitch": ["left_hip_pitch_04", "right_hip_pitch_04"],
       "ankle pitch": ["left_foot_pitch_02", "right_foot_pitch_02"],
       "ankle roll": ["left_foot_roll_02", "right_foot_roll_02"]}
tot = {l: 0.0 for l in RUNS}
offaxis = {l: 0.0 for l in RUNS}
for name, js in fam.items():
    cells = []
    for l in RUNS:
        q = D[l]["joint_positions"]
        v = np.mean([np.abs(q[:, :, IX[j]] - dfl[IX[j]]).mean() for j in js])
        cells.append(f"{v:13.4f}")
        tot[l] += v
        if name in ("hip yaw", "hip roll"):
            offaxis[l] += v
    print(f"  {name:<30}{''.join(cells)}")
print(f"  {'SUM over all six families':<30}{''.join(f'{tot[l]:13.4f}' for l in RUNS)}")
print(f"  {'SUM of hip yaw and hip roll':<30}{''.join(f'{offaxis[l]:13.4f}' for l in RUNS)}")

print()
print("=" * 112)
print("LATERAL BASE OF SUPPORT bought by the measured splay")
print("a rectangular sole of length L and width W yawed by t presents L|sin t| + W|cos t| laterally")
L, W = 0.2100, 0.0846
for l, ti in [("A 09-17_09-13", 9.687), ("B 09-17_11-25", 8.519), ("C 09-18_09-37", 4.594),
              ("D 09-18_10-03", 8.492)]:
    t = np.radians(ti)
    ext = L * np.sin(t) + W * np.cos(t)
    print(f"  {l}: toe-in {ti:5.2f} deg -> lateral extent {ext:.5f} m, a gain of"
          f" {100*(ext/W-1):5.1f} pct over the straight sole's {W:.4f} m")
for l, ti in [("A p95", 36.963), ("C p95", 21.394)]:
    t = np.radians(ti)
    ext = L * np.sin(t) + W * np.cos(t)
    print(f"  {l}: toe-in {ti:5.2f} deg -> lateral extent {ext:.5f} m, a gain of {100*(ext/W-1):5.1f} pct")

print()
print("ANKLE ROLL LATERAL MOMENT BALANCE in single support")
Wt = 20.2814 * 9.81
print(f"  body weight {Wt:.2f} N, ankle roll effort ceiling in every run 11.90 Nm")
for l in RUNS:
    sep = np.abs(D[l]["feet_distance"][:, :, 1]).mean()
    d = sep / 2.0
    tau_ankle = Wt * d
    relief_straight = Wt * (W / 2)
    print(f"  {l}: lateral separation {sep:.4f} m, half {d:.4f} m"
          f" -> moment about the ankle {tau_ankle:6.2f} Nm,"
          f" net after CoP to the straight sole edge {max(tau_ankle-relief_straight,0):5.2f} Nm")
print(f"  the relief a straight sole can offer is {Wt*(W/2):.2f} Nm and it is only reached with the")
print("  foot on the verge of rolling over its own edge, so the margin at that point is nil")
