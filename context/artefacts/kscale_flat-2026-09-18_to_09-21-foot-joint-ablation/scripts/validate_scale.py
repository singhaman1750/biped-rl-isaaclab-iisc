"""Check the scale dict actually written to the config against the robot's joint list.

Parses the literal out of kscale_base_env_cfg.py with ast so the test reads the real file, then
runs it through the same resolver joint_actions.py:86 uses.
"""
import ast, importlib.util, torch, numpy as np
_s = importlib.util.spec_from_file_location("il_string", "/ws/IsaacLab/source/isaaclab/isaaclab/utils/string.py")
su = importlib.util.module_from_spec(_s); _s.loader.exec_module(su)

CFG = "/ws/tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py"
tree = ast.parse(open(CFG).read())
scale = None
for node in ast.walk(tree):
    if isinstance(node, ast.keyword) and node.arg == "scale" and isinstance(node.value, ast.Dict):
        cand = ast.literal_eval(node.value)
        if any("foot_roll" in k for k in cand):
            scale = cand; break
assert scale is not None, "no scale dict found"
print("parsed from the config:"); [print(f"   {k:32} {v}") for k,v in scale.items()]

# the joint order the dump records, which is the order the articulation resolves
JOINTS = list(np.load("/ws/IsaacLab/logs/rsl_rl/kscale_flat/2026-09-21_06-45-40/data/42/dump.npy",
                      allow_pickle=True).item()["joint_names"])
s = torch.ones(1, len(JOINTS))
idx, names, vals = su.resolve_matching_names_values(scale, JOINTS)
s[:, idx] = torch.tensor(vals, dtype=torch.float32)
s = s[0].tolist()

K   = {"hip_pitch":200.,"hip_yaw":100.,"knee":300.,"hip_roll":250.,"foot_pitch":100.,"foot_roll":100.}
TAU = {"hip_pitch":84.,"hip_yaw":42.,"knee":84.,"hip_roll":42.,"foot_pitch":17.,"foot_roll":17.}
def grp(n):
    for g in ("hip_pitch","hip_yaw","hip_roll","knee","foot_pitch","foot_roll"):
        if g in n: return g

print(f"\n{'joint':22}{'scale':>8}{'covered':>9}{'act@sat':>9}")
bad = []
for n, v in zip(JOINTS, s):
    cov = abs(v-1.0) > 1e-9 or grp(n) in ("hip_pitch","hip_yaw")
    a = (TAU[grp(n)]/K[grp(n)])/v
    if not cov: bad.append(n)
    print(f"{n:22}{v:>8.4f}{'yes' if cov else 'NO':>9}{a:>9.3f}")
assert not bad, f"UNCOVERED, would silently run at 1.0: {bad}"
sats = [(TAU[grp(n)]/K[grp(n)])/v for n, v in zip(JOINTS, s)]
# The "before" spread depends on the ankle ceiling the scale departs from, and BOTH are printed
# because the two are easy to confuse.  Corrected 2026-09-23, this line having previously hard
# coded 3.54, which is the spread at the 11.9 N.m ankle of section 4.6 and not at the 17 N.m
# ankle the arm actually rests upon.
OLD = 0.4
for ceiling, label in ((17.0, "17 N.m, the arm's own baseline"), (11.9, "11.9 N.m, section 4.6")):
    tau = dict(TAU); tau["foot_pitch"] = tau["foot_roll"] = ceiling
    before = [(tau[grp(n)]/K[grp(n)])/OLD for n in JOINTS]
    print(f"spread at a single scale of {OLD} with a {label:32} {max(before)/min(before):.2f} to one")
print(f"\nall 12 joints covered, action at saturation spans {min(sats):.4f} to {max(sats):.4f}, "
      f"ratio {max(sats)/min(sats):.4f}")

# Arm 4 alone equalises that ratio at 1.0002.  A larger ratio is not a failure of this check, it
# is arm 6 of plans/kscale_gait_shaping.md at section 21.6, which halves the two ankle scales
# from the 0.1619 the sizing rule gives to 0.08 so that those two groups saturate LAST rather than
# with the rest.  The ratio is then the factor the ankle scales were cut by, and the plan quotes
# it as the factor a compensating policy would have to widen its raw ankle action by.
ankles = [s for n, s in zip(JOINTS, sats) if grp(n) in ("foot_pitch", "foot_roll")]
others = [s for n, s in zip(JOINTS, sats) if grp(n) not in ("foot_pitch", "foot_roll")]
if max(ankles) / min(others) > 1.01:
    print(f"arm 6 is ACTIVE, the ankles saturating at {max(ankles):.3f} against {max(others):.3f} "
          f"elsewhere, a factor of {max(ankles)/min(others):.4f}")
else:
    print("arm 6 is NOT active, the ankle scales standing at the value arm 4's sizing rule gives")
