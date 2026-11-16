"""True toe-in angle from raw quaternions, and the speed-binned crouch."""
import numpy as np

BASE = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"A 09-17_09-13": "2026-09-17_09-13-46", "B 09-17_11-25": "2026-09-17_11-25-59",
        "C 09-18_09-37": "2026-09-18_09-37-29", "D 09-18_10-03": "2026-09-18_10-03-48"}
D = {l: np.load(f"{BASE}/{r}/data/42/dump.npy", allow_pickle=True).item() for l, r in RUNS.items()}


def quat_rotate(q, v):
    """Rotate v by quaternion q in WXYZ convention. q (...,4), v (3,) -> (...,3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v
    # R @ v, expanded
    rx = (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y - w * z) * vy + 2 * (x * z + w * y) * vz
    ry = 2 * (x * y + w * z) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z - w * x) * vz
    rz = 2 * (x * z - w * y) * vx + 2 * (y * z + w * x) * vy + (1 - 2 * (x * x + y * y)) * vz
    return np.stack([rx, ry, rz], axis=-1)


q0 = D["A 09-17_09-13"]["base_quaternion"][0, 0]
print("base quaternion sample", q0, "norm", np.linalg.norm(q0))
print("  WXYZ assumed, |w| is the largest component ->", abs(q0[0]) == abs(q0).max())
print()

FOOT_FWD = np.array([0.0, 0.0, -1.0])   # the axis pen_feet_heading declares as forward
print("=" * 118)
print("TOE-IN ANGLE [deg], signed so that POSITIVE means the toe points toward the body midline")
print("computed as the yaw of the foot forward axis relative to the base heading")
print(f"  {'quantity':<44}{'A':>13}{'B':>13}{'C':>13}{'D':>13}")

res = {}
for label in RUNS:
    d = D[label]
    bq = d["base_quaternion"]                      # (T,E,4) world
    fq = d["feet_quaternions"]                     # (T,E,2,4) world
    base_fwd = quat_rotate(bq, np.array([1.0, 0.0, 0.0]))
    base_yaw = np.arctan2(base_fwd[..., 1], base_fwd[..., 0])          # (T,E)
    out = {}
    for k, name in enumerate(d["feet_names"]):
        ffwd = quat_rotate(fq[:, :, k, :], FOOT_FWD)
        fyaw = np.arctan2(ffwd[..., 1], ffwd[..., 0])
        rel = np.arctan2(np.sin(fyaw - base_yaw), np.cos(fyaw - base_yaw))
        out[name] = rel
    res[label] = out

FEET = D["A 09-17_09-13"]["feet_names"]   # ['foot_6061_2' (left), 'foot_6061' (right)]
# left foot toeing in means its forward axis rotates to the RIGHT of the base heading, i.e. negative yaw.
SIGN = {"foot_6061_2": -1.0, "foot_6061": +1.0}
for name in FEET:
    side = "left" if name.endswith("_2") else "right"
    cells = "".join(f"{np.degrees(res[l][name].mean()):13.3f}" for l in RUNS)
    print(f"  raw relative yaw {name} ({side}) [mean]     {cells}")
print()
for name in FEET:
    side = "left" if name.endswith("_2") else "right"
    cells = "".join(f"{np.degrees(SIGN[name]*res[l][name]).mean():13.3f}" for l in RUNS)
    print(f"  TOE-IN {name} ({side}) [mean]              {cells}")
print()
cells = []
for l in RUNS:
    ti = np.stack([SIGN[n] * res[l][n] for n in FEET])
    cells.append(f"{np.degrees(ti).mean():13.3f}")
print(f"  {'TOE-IN both feet [mean]':<44}{''.join(cells)}")
cells = []
for l in RUNS:
    ti = np.stack([SIGN[n] * res[l][n] for n in FEET])
    cells.append(f"{np.degrees(np.abs(ti)).mean():13.3f}")
print(f"  {'|relative yaw| both feet [mean]':<44}{''.join(cells)}")
cells = []
for l in RUNS:
    ti = np.stack([SIGN[n] * res[l][n] for n in FEET])
    cells.append(f"{np.degrees(np.percentile(ti, 95)):13.3f}")
print(f"  {'TOE-IN p95 [deg]':<44}{''.join(cells)}")

# restrict to stance, where the toe-in actually buys support
print()
print("  restricted to STANCE (vertical contact force above 20 N)")
for name in FEET:
    side = "left" if name.endswith("_2") else "right"
    cells = []
    for l in RUNS:
        d = D[l]
        k = d["feet_names"].index(name)
        fz = d["feet_contact_forces"][:, :, k, 2]
        m = fz > 20.0
        cells.append(f"{np.degrees(SIGN[name]*res[l][name][m]).mean():13.3f}")
    print(f"  TOE-IN in stance {name} ({side})        {''.join(cells)}")

print()
print("  effective lateral half extent of the stance foot implied by that toe-in [m]")
for l in RUNS:
    vals = []
    for name in FEET:
        d = D[l]
        k = d["feet_names"].index(name)
        fz = d["feet_contact_forces"][:, :, k, 2]
        m = fz > 20.0
        th = np.abs(SIGN[name] * res[l][name][m])
        vals.append((0.0423 * np.cos(th) + 0.105 * np.sin(th)).mean())
    print(f"    {l}: {np.mean(vals):.4f} m   (a straight foot gives 0.0423 m)")

print()
print("=" * 118)
print("BASE HEIGHT BINNED BY COMMANDED FORWARD SPEED [m]")
bins = [(-0.6, -0.1), (-0.1, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.1)]
print(f"  {'commanded vx bin':<24}{'A':>13}{'B':>13}{'C':>13}{'D':>13}{'n (A)':>10}")
for lo, hi in bins:
    cells, n = [], 0
    for l in RUNS:
        d = D[l]
        vx = d["commanded_linear_velocity"][:, :, 0]
        h = d["base_com_position"][:, :, 2]
        m = (vx >= lo) & (vx < hi)
        cells.append(f"{h[m].mean():13.4f}" if m.any() else f"{'--':>13}")
        if l.startswith("A"):
            n = int(m.sum())
    print(f"  [{lo:+.1f},{hi:+.1f})            {''.join(cells)}{n:10d}")

print()
print("KNEE ANGLE BINNED BY COMMANDED FORWARD SPEED [rad], mean of both knees")
ki = [D["A 09-17_09-13"]["joint_names"].index(j) for j in ["left_knee_04", "right_knee_04"]]
print(f"  {'commanded vx bin':<24}{'A':>13}{'B':>13}{'C':>13}{'D':>13}")
for lo, hi in bins:
    cells = []
    for l in RUNS:
        d = D[l]
        vx = d["commanded_linear_velocity"][:, :, 0]
        q = d["joint_positions"][:, :, ki].mean(axis=2)
        m = (vx >= lo) & (vx < hi)
        cells.append(f"{q[m].mean():13.4f}" if m.any() else f"{'--':>13}")
    print(f"  [{lo:+.1f},{hi:+.1f})            {''.join(cells)}")

print()
print("TOE-IN BINNED BY COMMANDED FORWARD SPEED [deg]")
print(f"  {'commanded vx bin':<24}{'A':>13}{'B':>13}{'C':>13}{'D':>13}")
for lo, hi in bins:
    cells = []
    for l in RUNS:
        d = D[l]
        vx = d["commanded_linear_velocity"][:, :, 0]
        ti = np.stack([SIGN[n] * res[l][n] for n in FEET])       # (2,T,E)
        m = (vx >= lo) & (vx < hi)
        cells.append(f"{np.degrees(ti[:, m]).mean():13.3f}" if m.any() else f"{'--':>13}")
    print(f"  [{lo:+.1f},{hi:+.1f})            {''.join(cells)}")
