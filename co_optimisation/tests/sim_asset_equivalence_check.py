"""Physical equivalence of a committed asset before and after its fold moved into the body frames.

Run with Isaac Sim, not collected by pytest. Both documents are converted as usd_generator
converts a design and spawned apart in free fall. At random joint configurations, the old
one at q and the new one at q plus nothing but the fold it no longer carries, every body's
position relative to the root, its centre of mass, its world inertia tensor and its mass
are compared, and the joint limits after the same shift. Both are then driven by the same
targets through 200 physics steps and their trajectories compared. The bodies whose inertia
differs in their own frame, which the robot_inertia observation reads, are listed.

    ./isaaclab.sh -p biped/co_optimisation/tests/sim_asset_equivalence_check.py FAMILY OLD.xml NEW.xml
"""
import os
import sys
import tempfile

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402
from isaaclab.utils.math import matrix_from_quat  # noqa: E402

from design_generator import biped, quadruped  # noqa: E402

family, old_xml, new_xml = sys.argv[1:4]
builder = {"biped": biped, "quadruped": quadruped}[family].builder
root_prim = {"quadruped": "/trunk/trunk", "biped": "/base_Link/base_Link"}[family]
fold = ({"thigh_joint": builder.THIGH_BEND_RAD, "calf_joint": builder.SHANK_BEND_RAD}
        if family == "quadruped" else {})
out = tempfile.mkdtemp(prefix="copt_equiv_")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
robots = {}
for i, (tag, xml) in enumerate((("old", old_xml), ("new", new_xml))):
    usd = MjcfConverter(MjcfConverterCfg(
        asset_path=xml, usd_dir=os.path.join(out, tag), usd_file_name=f"{tag}.usd", link_density=0.0,
        import_inertia_tensor=True, fix_base=False, self_collision=False, make_instanceable=True,
        force_usd_conversion=True)).usd_path
    sim_utils.UsdFileCfg(usd_path=usd).func(f"/World/{tag}/Robot", sim_utils.UsdFileCfg(usd_path=usd))
    nominal = {".*": 0.0}
    if tag == "old" and fold:
        nominal = {".*_hip_joint": 0.0, **{f".*_{k}": v for k, v in fold.items()}}
    robots[tag] = Articulation(ArticulationCfg(
        prim_path=f"/World/{tag}/Robot", spawn=None, articulation_root_prim_path=root_prim,
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=nominal),
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=40.0, damping=1.0)}))
sim.reset()
old, new = robots["old"], robots["new"]
assert old.joint_names == new.joint_names and old.body_names == new.body_names
shift = torch.tensor([[next((v for k, v in fold.items() if n.endswith(k)), 0.0) for n in new.joint_names]],
                     device=new.device)


def place(robot, q, x):
    pose = torch.tensor([[x, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0]], device=robot.device)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=robot.device))
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))


def relative(robot):
    d = robot.data
    rot = matrix_from_quat(d.body_link_quat_w[0])
    inertia = rot @ d.default_inertia[0].to(rot.device).view(-1, 3, 3) @ rot.transpose(1, 2)
    return (d.body_link_pos_w[0] - d.root_pos_w[0], d.body_com_pos_w[0] - d.root_pos_w[0], inertia)


worst = {"body": 0.0, "com": 0.0, "inertia": 0.0}
lo, hi = new.data.joint_pos_limits[0, :, 0], new.data.joint_pos_limits[0, :, 1]
limits = (old.data.joint_pos_limits[0] - (new.data.joint_pos_limits[0] + shift[0, :, None])).abs().max().item()
mass = (old.data.default_mass - new.data.default_mass).abs().max().item()
torch.manual_seed(0)
for _ in range(20):
    q_new = (lo + (hi - lo) * (0.1 + 0.8 * torch.rand_like(lo))).unsqueeze(0)
    place(old, q_new + shift, 0.0)
    place(new, q_new, 0.0)
    for key, a, b in zip(worst, relative(old), relative(new)):
        worst[key] = max(worst[key], (a - b).abs().max().item())
body_frame = (old.data.default_inertia[0] - new.data.default_inertia[0]).abs().amax(dim=-1)
changed = [n for n, d in zip(new.body_names, body_frame.tolist()) if d > 1e-7]
print(f"[{family}] inertia in the body frame, robot_inertia's channels, differs for {changed}")
print(f"[{family}] static, 20 configurations: body {worst['body']:.2e} m, com {worst['com']:.2e} m, "
      f"world inertia {worst['inertia']:.2e} kg m^2, mass {mass:.2e} kg, joint limits {limits:.2e} rad")

q_new = (lo + (hi - lo) * 0.5).unsqueeze(0)
target = (lo + (hi - lo) * (0.3 + 0.4 * torch.rand_like(lo))).unsqueeze(0)
place(old, q_new + shift, 0.0)
place(new, q_new, 5.0)
old.set_joint_position_target(target + shift)
new.set_joint_position_target(target)
start = {t: r.data.root_pos_w[0].clone() for t, r in robots.items()}
drift = 0.0
for _ in range(200):
    for r in robots.values():
        r.write_data_to_sim()
    sim.step(render=False)
    for r in robots.values():
        r.update(sim.get_physics_dt())
    a, b = relative(old)[0], relative(new)[0]
    fall = (old.data.root_pos_w[0] - start["old"]) - (new.data.root_pos_w[0] - start["new"])
    drift = max(drift, (a - b).abs().max().item(), fall.abs().max().item())
joint = (old.data.joint_pos - shift - new.data.joint_pos).abs().max().item()
print(f"[{family}] dynamic, 200 steps under the same targets: body and root {drift:.2e} m, joints {joint:.2e} rad")
app.close()
