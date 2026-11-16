import sys, torch, importlib.util
_s=importlib.util.spec_from_file_location("il_string","/ws/IsaacLab/source/isaaclab/isaaclab/utils/string.py")
string_utils=importlib.util.module_from_spec(_s); _s.loader.exec_module(string_utils)

JOINTS = ['left_hip_pitch_04','right_hip_pitch_04','left_hip_roll_03','right_hip_roll_03',
          'left_hip_yaw_03','right_hip_yaw_03','left_knee_04','right_knee_04',
          'left_foot_pitch_02','right_foot_pitch_02','left_foot_roll_02','right_foot_roll_02']

def parse(scale_dict, joints=JOINTS):
    """Reproduces joint_actions.py:84-89 exactly."""
    s = torch.ones(1, len(joints))
    idx, names, vals = string_utils.resolve_matching_names_values(scale_dict, joints)
    s[:, idx] = torch.tensor(vals, dtype=torch.float32)
    return s[0]

print("A. complete dict, every joint covered")
full = {".*_hip_pitch_04": 0.40, ".*_hip_yaw_03": 0.40, ".*_knee_04": 0.267,
        ".*_hip_roll_03": 0.160, ".*_foot_pitch_02": 0.162, ".*_foot_roll_02": 0.162}
print("  ", [f"{n.split('_',1)[1]}={v:.3f}" for n,v in zip(JOINTS, parse(full).tolist())][:6], "...")
print("   all covered:", not any(abs(v-1.0)<1e-9 for v in parse(full).tolist()))

print("\nB. INCOMPLETE dict, ankles omitted -> what happens to them?")
partial = {".*_hip_pitch_04": 0.40, ".*_hip_yaw_03": 0.40, ".*_knee_04": 0.267, ".*_hip_roll_03": 0.160}
try:
    out = parse(partial)
    for n, v in zip(JOINTS, out.tolist()):
        if abs(v-1.0) < 1e-9: print(f"   {n:22} scale = {v:.3f}   <-- SILENTLY 1.0, was 0.4")
except Exception as e:
    print("   raised:", type(e).__name__, e)

print("\nC. overlapping patterns")
try:
    parse({".*_foot_.*": 0.16, "left_foot_roll_02": 0.12})
    print("   accepted")
except Exception as e:
    print("   raised:", type(e).__name__, str(e).split('!')[0]+"!")

print("\nD. a key matching nothing (typo)")
try:
    parse(dict(full, **{".*_ankle_roll_02": 0.1}))
    print("   accepted")
except Exception as e:
    print("   raised:", type(e).__name__, str(e).strip().splitlines()[0])
