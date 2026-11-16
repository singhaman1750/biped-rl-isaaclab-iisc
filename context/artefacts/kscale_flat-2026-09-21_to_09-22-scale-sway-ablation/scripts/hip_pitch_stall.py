"""Why run B holds the hip pitch at its torque ceiling although its action scale never moved.

The hip pitch scale is 0.4 in all three runs, so the 58.8 per cent effort saturation B records
against A's 0.06 is a behavioural difference and not a parameterisation one.  Two candidates are
separable from the dump, a position error the joint cannot close and a velocity that has carried
the actuator past its corner, and the DCMotor envelope of actuator_pd.py:295 distinguishes them.
"""
import numpy as np, common
d0 = common.load("A_scale0.4"); J = d0["joint_names"]
i = J.index("left_hip_pitch_04")
elim = np.asarray(d0["joint_effort_limits"])[i]; vlim = np.asarray(d0["joint_velocity_limits"])[i]
SAT = 120.0
corner = vlim * (1.0 - elim/SAT)
print(f"hip pitch, effort ceiling {elim} Nm, saturation effort {SAT} Nm, velocity limit {vlim:.3f} rad/s")
print(f"corner velocity {corner:.3f} rad/s, beyond which the available torque falls below the ceiling\n")
print(f"{'run':12}{'|tau| mean':>11}{'|tau| p99':>10}{'sat %':>8}{'|v| p50':>9}{'|v| p99':>9}"
      f"{'past corner %':>15}{'q p50':>9}{'q range':>10}{'knee p50':>10}{'height':>9}")
for run in common.RUNS:
    d = common.load(run)
    tau = d["joint_torques"][:,:,i]; v = d["joint_velocities"][:,:,i]; q = d["joint_positions"][:,:,i]
    k = d["joint_positions"][:,:,J.index("left_knee_04")]
    h = d["base_com_position"][:,:,2]
    print(f"{run:12}{np.abs(tau).mean():>11.2f}{np.percentile(np.abs(tau),99):>10.2f}"
          f"{100*np.mean(np.abs(tau)>=0.99*elim):>8.2f}{np.percentile(np.abs(v),50):>9.3f}"
          f"{np.percentile(np.abs(v),99):>9.3f}{100*np.mean(np.abs(v)>corner):>15.2f}"
          f"{np.percentile(q,50):>9.4f}{np.percentile(q,99)-np.percentile(q,1):>10.4f}"
          f"{np.percentile(k,50):>10.4f}{h.mean():>9.4f}")
print("\nthe position error the torque implies, tau = K(target - q) - D v with K 200 and D 21.5")
for run in common.RUNS:
    d = common.load(run)
    tau = d["joint_torques"][:,:,i]; v = d["joint_velocities"][:,:,i]
    err = (tau + 21.5*v)/200.0
    print(f"  {run:12} |error| p50 {np.percentile(np.abs(err),50):.4f} rad, p99 "
          f"{np.percentile(np.abs(err),99):.4f} rad, which at a scale of 0.4 is an action of "
          f"{np.percentile(np.abs(err),50)/0.4:.3f} and {np.percentile(np.abs(err),99)/0.4:.3f}")
print("\nknee, the joint whose scale DID move, 0.4 to 0.2667")
ik = J.index("left_knee_04")
print(f"{'run':12}{'|tau| mean':>11}{'q p1':>9}{'q p50':>9}{'at 0 stop %':>13}{'soft excursion %':>18}")
for run in common.RUNS:
    d = common.load(run); q = d["joint_positions"][:,:,ik]; t = d["joint_torques"][:,:,ik]
    soft = np.asarray(d["joint_soft_position_limits"])[ik]
    print(f"{run:12}{np.abs(t).mean():>11.2f}{np.percentile(q,1):>9.4f}{np.percentile(q,50):>9.4f}"
          f"{100*np.mean(q<0.0271):>13.2f}{100*np.mean((q<soft[0])|(q>soft[1])):>18.2f}")
