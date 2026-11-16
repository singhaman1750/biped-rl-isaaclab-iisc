"""The numpy statics of co_optimisation/utils/filters.py against MuJoCo, on emitted documents.

Run where the mujoco package is installed, the development container having it and the training
containers not. At every stance the StandingFilter samples, the robot is placed in MuJoCo where the
statics place it, and the centre of mass, the base rows of qfrc_bias - J^T f, which equilibrium
makes zero, and the joint rows, which are the joint torques, are compared. See plans/design_filtering.md.

    python3 -I co_optimisation/tests/filters_mujoco_check.py co_optimisation/co_optimisation/utils/filters.py A.xml [B.xml ...]
"""
import importlib.util
import sys

import mujoco
import numpy as np

_spec = importlib.util.spec_from_file_location("filters", sys.argv[1])
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)

worst = 0.0
for xml in sys.argv[2:]:
    m = F.MjcfStatics(xml)
    q = F.StandingFilter("any").configurations([m])
    st = F.standing_statics([m], q)
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    assert abs(st["mass"][0] - model.body_subtreemass[1]) < 1e-9
    dof = [model.jnt_dofadr[model.joint(n).id] for n in m.joint_names]
    errors = []
    for s in range(q.shape[1]):
        G, origin = st["ground_rotation"][0, s], st["ground_origin"][0, s]
        data.qpos[:] = 0.0
        data.qpos[0:3] = -G @ origin
        mujoco.mju_mat2Quat(data.qpos[3:7], G.reshape(-1))
        for j, name in enumerate(m.joint_names):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = q[0, s, j]
        mujoco.mj_forward(model, data)
        tau = data.qfrc_bias.copy()
        for c, body in enumerate(m.contact_body):
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, None, st["contact"][0, s, c], model.body(m.body_names[body]).id)
            tau -= jacp.T @ np.array([0.0, 0.0, st["force"][0, s, c]])
        errors.append((np.abs(data.subtree_com[1] - st["com"][0, s]).max(), np.abs(tau[:6]).max(),
                       np.abs(tau[dof] - st["torque"][0, s]).max()))
    e = np.array(errors).max(axis=0)
    worst = max(worst, e.max())
    print(f"{xml}: |com| {e[0]:.1e} m, |base residual| {e[1]:.1e}, |torque| {e[2]:.1e} N.m over {q.shape[1]} stances")
assert worst < 1e-8, worst
print("statics agree with MuJoCo")
