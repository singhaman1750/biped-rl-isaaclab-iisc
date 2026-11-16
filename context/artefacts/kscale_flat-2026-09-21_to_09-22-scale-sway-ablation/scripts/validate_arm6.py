"""Check the tree as it stands after arm 6 stage one was applied on 2026-09-23.

Reads the two configuration files with ast rather than importing them, so the check needs no
simulator and reads the real file rather than a transcription of it.  Sections A and B assert
the arm 6 and arm 5 values.  Section C re-prices the four energy terms against the dumped
evaluation rewards of the three runs of this artefact directory, which is the only way to say
what a factor of ten actually buys, since the share the group reaches depends on the gait the
policy settles into and not on the weight alone.
"""
import ast
import re
import numpy as np
import common

CFG = "/ws/tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py"
POSE = "/ws/tron1-rl-isaaclab-cozum/environments/environments/assets/config/kscale_identified_cfg.py"

ARM6 = {"pen_joint_torque": -1.0e-4, "pen_joint_accel": -1.0e-6,
        "pen_joint_vel_l2": -1.0e-4, "pen_joint_torque_rate": -1.0e-4}
UNMOVED = {"pen_action_rate": -0.01, "pen_action_smoothness": -0.075}
ARM5_POSE = {"hip_pitch": 0.34907, "knee": 0.87266, "foot_pitch": -0.52360}
ARM5_HEIGHT, ARM5_SPAWN = 0.73492, 0.74492

tree = ast.parse(open(CFG).read())
rewards = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
weights = {}
for st in rewards.body:
    if isinstance(st, ast.Assign) and isinstance(st.targets[0], ast.Name) and isinstance(st.value, ast.Call):
        for kw in st.value.keywords:
            if kw.arg == "weight":
                weights[st.targets[0].id] = ast.literal_eval(kw.value)

print("A. Arm 6 stage one, the four energy weights raised by ten")
ok = True
for k, want in ARM6.items():
    got = weights.get(k)
    good = got is not None and abs(got - want) < 1e-12
    ok &= good
    print(f"   {k:24} {got!s:>10}   want {want:<10g} {'OK' if good else 'MISMATCH'}")
print("   the two that must NOT move, both quadratic in the raw action and already moved by arms 3 and 4b")
for k, want in UNMOVED.items():
    got = weights.get(k)
    good = got is not None and abs(got - want) < 1e-12
    ok &= good
    print(f"   {k:24} {got!s:>10}   want {want:<10g} {'OK' if good else 'MISMATCH'}")

print("\nB. Arm 5, the nominal pose and the two height targets it implies")
init = next(n for n in ast.walk(ast.parse(open(POSE).read()))
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "init_state")
kw = {k.arg: k.value for k in init.value.keywords}
spawn = ast.literal_eval(kw["pos"])
jp = ast.literal_eval(kw["joint_pos"])
for g, want in ARM5_POSE.items():
    got = jp[f"left_{g}_04" if g != "foot_pitch" else "left_foot_pitch_02"]
    got = abs(got) * (1 if want > 0 else -1)
    good = abs(got - want) < 1e-5
    ok &= good
    print(f"   left_{g:18} {got:>10.5f}   want {want:<10g} {'OK' if good else 'MISMATCH'}")
good = abs(spawn[2] - ARM5_SPAWN) < 1e-6
ok &= good
print(f"   spawn height           {spawn[2]:>10.5f}   want {ARM5_SPAWN:<10g} {'OK' if good else 'MISMATCH'}")
src = open(CFG).read()
for name in ("target_height", "base_height_target"):
    m = re.search(rf'^\s+"{name}": ([\d.]+),', src, re.M)
    got = float(m.group(1))
    good = abs(got - ARM5_HEIGHT) < 1e-6
    ok &= good
    print(f"   {name:22} {got:>10.5f}   want {ARM5_HEIGHT:<10g} {'OK' if good else 'MISMATCH'}")
for leg, s in (("left", -1.0), ("right", 1.0)):
    r = s * jp[f"{leg}_hip_pitch_04"] + jp[f"{leg}_knee_04"] + jp[f"{leg}_foot_pitch_02"]
    good = abs(r) < 1e-4
    ok &= good
    print(f"   {leg} sagittal closure residual {r:+.6f} rad, the soles lie flat when it is zero  "
          f"{'OK' if good else 'NOT CLOSED'}")

print("\nC. What the factor of ten buys, re-priced against the three dumped evaluation runs")
print("   the per term series of rewards.npy are RATES PER SECOND, per context/gait_metrics.md")
R = {r: common.load(r, "rewards") for r in common.RUNS}
FACTOR = 10.0
print(f"   {'run':14}{'energy now':>12}{'negative now':>14}{'share now':>11}"
      f"{'energy x10':>12}{'negative x10':>14}{'share x10':>11}")
for r in common.RUNS:
    terms = [k for k in R[r] if k != "_weights" and k != "total_reward"]
    neg = sum(float(np.mean(R[r][k])) for k in terms if float(np.mean(R[r][k])) < 0)
    en = sum(float(np.mean(R[r][k])) for k in ARM6)
    en2 = en * FACTOR
    neg2 = neg + en * (FACTOR - 1.0)
    print(f"   {r:14}{en:12.4f}{neg:14.3f}{100*en/neg:10.2f}%{en2:12.4f}{neg2:14.3f}{100*en2/neg2:10.2f}%")
print("   the share is a projection at the CURRENT behaviour and is an upper bound on what the")
print("   policy will actually pay, the arm existing precisely so that the behaviour moves.")

print("\n" + ("ALL CHECKS PASS" if ok else "ONE OR MORE CHECKS FAILED"))
