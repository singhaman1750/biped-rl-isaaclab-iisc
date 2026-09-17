"""Standing geometry and stance load of the KScale biped at a candidate nominal pose.
Reads the URDF only. Reproduces the tables of sections 9.1, 9.1.1 and 9.1.2 of
../../../plans/kscale_actuator_limits_fix.md."""
import sys, io, contextlib
sys.path.insert(0, '/ws/tron1-rl-isaaclab-cozum/scripts/analysis')
URDF = '/ws/tron1-rl-isaaclab-cozum/environments/environments/assets/urdf/solefoot/kscale/kscale.urdf'
from kscale_sole_analysis import parse_urdf, forward_kinematics, link_frame_vertices
KBOT = {"right_hip_pitch_04": -0.34907, "left_hip_pitch_04": 0.34907,
        "right_hip_roll_03": 0.0, "left_hip_roll_03": 0.0,
        "right_hip_yaw_03": 0.0, "left_hip_yaw_03": 0.0,
        "right_knee_04": 0.87266, "left_knee_04": 0.87266,
        "right_foot_pitch_02": -0.52360, "left_foot_pitch_02": -0.52360,
        "right_foot_roll_02": 0.0, "left_foot_roll_02": 0.0}
links, joints, root = parse_urdf(URDF)
feet = sorted(n for n in links if n.startswith('foot_6061'))
def geom(hp, kn, an):
    pose = {"right_hip_pitch_04": -hp, "left_hip_pitch_04": hp,
            "right_knee_04": kn, "left_knee_04": kn,
            "right_foot_pitch_02": an, "left_foot_pitch_02": an}
    rot, pos = forward_kinematics(joints, root, pose)
    low = min(float((link_frame_vertices(links[n]) @ rot[n].T + pos[n])[:, 2].min())
              for n in links if link_frame_vertices(links[n]).size)
    return -low, pos[feet[0]][2] - low
print(f"{'pose':<30}{'height':>10}{'ankle origin':>14}")
for lbl, a in (("current nominal", (0.1, 0.4, -0.3)),
               ("K-Scale K-Bot posture", (0.34907, 0.87266, -0.52360)),
               ("intermediate knee 0.70", (0.28, 0.70, -0.42)),
               ("zero pose", (0.0, 0.0, 0.0))):
    h, k = geom(*a)
    print(f"{lbl:<30}{h:>10.5f}{k:>14.5f}")
sys.argv = ['x', '--urdf', URDF]
import kscale_stance_analysis as S
for lbl, pose in (("CURRENT", dict(S.NOMINAL_POSE)), ("K-BOT POSTURE", KBOT)):
    S.NOMINAL_POSE.clear(); S.NOMINAL_POSE.update(pose)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: S.main()
        except SystemExit: pass
    out = buf.getvalue().splitlines()
    for key, n in (('tau_2sup  tau_1sup', 8), ('M_prox  tau_hold', 8)):
        i = [j for j, l in enumerate(out) if key in l][0]
        print(f"\n### {lbl}"); print("\n".join(out[i:i + n]))
