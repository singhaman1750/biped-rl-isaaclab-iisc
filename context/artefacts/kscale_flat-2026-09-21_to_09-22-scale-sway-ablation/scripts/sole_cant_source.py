"""Which joint sets the sole cant in stance, and what the cant would be with the ankle roll frozen.

Arm 24 freezes the ankle roll at its nominal, and the quantity that decides whether that flattens
the sole or merely moves the cant is the partial derivative of the sole roll with respect to each
of the two roll joints.  Regressing the stance sole roll on the hip roll, the ankle roll and the
torso roll together separates them, and evaluating the fit at the ankle roll's nominal states the
counterfactual without needing a simulator.
"""
import numpy as np, common
from frames import quat_rot
NOMINAL_ANKLE_ROLL = {"left": -0.041254, "right": +0.041254}
def yaw_of(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
print(f"{'run':12}{'leg':6}{'d sole/d ankle':>16}{'d sole/d hip':>14}{'d sole/d torso':>16}"
      f"{'R^2':>7}{'sole now':>10}{'sole frozen':>13}")
for run in common.RUNS:
    d = common.load(run); J = list(d["joint_names"])
    by = yaw_of(d["base_quaternion"]); c, s = np.cos(-by), np.sin(-by)
    contact = np.linalg.norm(d["feet_contact_forces"], axis=-1) > 1.0
    # torso roll in the base yaw frame
    g = d["base_projected_gravity"]
    torso_roll = np.arctan2(-g[...,1], -g[...,2])
    for fi, leg in ((0,"left"),(1,"right")):
        q = d["feet_quaternions"][:,:,fi,:]
        up = -quat_rot(q, np.array([0.,1.,0.]))
        uy = s*up[...,0] + c*up[...,1]
        sole = np.arctan2(uy, up[...,2])          # sole roll about the base yaw forward axis
        ar = d["joint_positions"][:,:,J.index(f"{leg}_foot_roll_02")]
        hr = d["joint_positions"][:,:,J.index(f"{leg}_hip_roll_03")]
        st = contact[:,:,fi]
        X = np.stack([ar[st], hr[st], torso_roll[st], np.ones(st.sum())], -1)
        b, *_ = np.linalg.lstsq(X, sole[st], rcond=None)
        pred = X @ b
        r2 = 1 - ((sole[st]-pred)**2).sum()/((sole[st]-sole[st].mean())**2).sum()
        frozen = b[0]*NOMINAL_ANKLE_ROLL[leg] + b[1]*hr[st].mean() + b[2]*torso_roll[st].mean() + b[3]
        print(f"{run:12}{leg:6}{b[0]:>16.3f}{b[1]:>14.3f}{b[2]:>16.3f}{r2:>7.3f}"
              f"{np.degrees(sole[st].mean()):>10.2f}{np.degrees(frozen):>13.2f}")
    print()
