"""Importer and physics check of the nominal pose, the in place editor and the nominal pose observation, run with Isaac Sim, not collected by pytest.

For each family it builds two designs A and B that differ in their lengths, their pose
and every actuator they pick, converts them with the MJCF importer exactly as
usd_generator does, and then

1. reads every pose joint's physics:localRot0 and localRot1 back from A's converted USD
   and checks localRot0 == body quat * localRot1, the rule update.py applies,
2. spawns A in environments 0 and 1 and B in environment 2, B as a respawn makes it,
   measures A's foot depth and foot lead over the hip pitch axis at zero joint angles
   against Family.nominal(A), and reads the nominal pose observation,
3. writes B in place through environment 0 with apply_link_length_params, and compares
   environments 0 and 1, the second never written, with environment 2 in every body's
   pose at zero joint angles, every body's mass, inertia and centre of mass as PhysX
   holds them and every collider's volume as PhysX cooked it, runs
   assert_link_lengths_applied, and measures the pose against Family.nominal(B),
4. reads the observation again, which must have re-read the frames, against
   pose_frames(B) and against the fold the bodies realise at zero joint angles,
5. times the observation's stage read over 1024 new references to A's file, whose layer
   the write of step 3 changed, so that they read B.

    ./isaaclab.sh -p biped/co_optimisation/tests/sim_nominal_pose_check.py [quadruped|biped]

plans/co_optimisation_audit_implementation.md section 9.2 records its output.
"""
import os
import sys
import tempfile
import time
import types

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omni.physx import get_physx_property_query_interface  # noqa: E402
from omni.physx.bindings._physx import PhysxPropertyQueryMode  # noqa: E402
from pxr import Gf, PhysicsSchemaTools, Usd, UsdUtils  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.managers import ObservationTermCfg, SceneEntityCfg  # noqa: E402
from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402

import environments.tasks.locomotion.mdp as mdp  # noqa: E402
from co_optimisation.runners import usd_generator as U  # noqa: E402
from co_optimisation.utils import update as UP  # noqa: E402

OUT = tempfile.mkdtemp(prefix="copt_pose_")
ROOT_PRIM = {"quadruped": "/trunk/trunk", "biped": "/base_Link/base_Link"}
PICKS_B = {"quadruped": {"abad": "A1-KNEE", "hip": "A1-KNEE", "knee": "A1-ABAD"},
           "biped": {"abad": "TRON1-ANKLE", "hip": "TRON1-KNEE", "knee": "TRON1-ANKLE", "ankle": "TRON1-HIP"}}
FOLDED = {"quadruped": ["thigh_.*_Link", "shank_.*_Link"],
          "biped": ["thigh_.*_Link", "shank_.*_Link", "ankle_.*_act"]}
PARENT = {"thigh": "knee_{}_act", "shank": "thigh_{}_Link", "ankle": "shank_{}_Link"}
results = []


def convert(fam, spec, tag):
    xml_path = os.path.join(OUT, f"{fam.name}_{tag}.xml")
    with open(xml_path, "w") as fh:
        fh.write(fam.build_xml(spec))
    cfg = MjcfConverterCfg(asset_path=xml_path, usd_dir=os.path.join(OUT, f"{fam.name}_{tag}_usd"),
                           usd_file_name=f"{fam.name}_{tag}.usd", link_density=0.0,
                           import_inertia_tensor=True, fix_base=False, self_collision=False,
                           make_instanceable=True, force_usd_conversion=True)
    return MjcfConverter(cfg).usd_path


def check_joint_frames(fam, spec, usd_path):
    stage = Usd.Stage.Open(usd_path)
    joints = {p.GetName(): p for p in stage.Traverse() if p.HasAttribute("physics:localRot0")}
    worst = 0.0
    for link, entry in fam.links(spec).items():
        for joint, q in entry.get("joint_rot", {}).items():
            prim = joints[joint]
            r0 = prim.GetAttribute("physics:localRot0").Get()
            r1 = prim.GetAttribute("physics:localRot1").Get()
            want = Gf.Quatf(*q) * r1
            d = min(max(abs(a - b) for a, b in zip((r0.GetReal(), *r0.GetImaginary()), (want.GetReal(), *want.GetImaginary()))),
                    max(abs(a + b) for a, b in zip((r0.GetReal(), *r0.GetImaginary()), (want.GetReal(), *want.GetImaginary()))))
            worst = max(worst, d)
    return worst


def zero_pose(robot):
    robot.write_joint_state_to_sim(torch.zeros_like(robot.data.default_joint_pos),
                                   torch.zeros_like(robot.data.default_joint_vel))
    pose = robot.data.default_root_state[:, :7].clone()
    pose[:, 0] = torch.arange(robot.num_instances, device=pose.device, dtype=pose.dtype) * 3.0
    pose[:, 1] = 0.0
    pose[:, 2] = 2.0
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=pose.device)
    robot.write_root_pose_to_sim(pose)
    return pose[:, :3]


def measure(robot, fam):
    zero_pose(robot)
    foot_ids, _ = robot.find_bodies(list(fam.foot_bodies), preserve_order=True)
    hip_ids, _ = robot.find_bodies(list(fam.hip_bodies), preserve_order=True)
    p = robot.data.body_link_pos_w
    return (2.0 - p[:, foot_ids, 2]).cpu().numpy(), (p[:, foot_ids, 0] - p[:, hip_ids, 0]).cpu().numpy()


def collider_volumes(stage, robot, root):
    stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
    out = np.zeros((robot.num_instances, robot.num_bodies))
    for i in range(robot.num_instances):
        for k, body in enumerate(robot.body_names):
            path = f"/World/envs/env_{i}/Robot/{root}/{body}"
            vol = []
            get_physx_property_query_interface().query_prim(
                stage_id=stage_id, prim_id=PhysicsSchemaTools.sdfPathToInt(path),
                query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS, timeout_ms=2000,
                rigid_body_fn=lambda r: None, collider_fn=lambda r: vol.append(r.volume), finished_fn=lambda: None)
            out[i, k] = sum(vol)
    return out


def physics_state(robot, stage, root):
    origin = zero_pose(robot)
    view = robot.root_physx_view
    return {
        "body position": (robot.data.body_link_pos_w - origin[:, None, :]).cpu().numpy(),
        "body orientation": robot.data.body_link_quat_w.cpu().numpy(),
        "mass": view.get_masses().cpu().numpy(),
        "inertia": view.get_inertias().cpu().numpy(),
        "centre of mass": view.get_coms().cpu().numpy()[..., :3],
        "collider volume": collider_volumes(stage, robot, root),
    }


class Scene(types.SimpleNamespace):
    def __getitem__(self, key):
        return self.articulations[key]


def stub_env(robot, paths, regex, sim=None):
    scene = Scene(env_prim_paths=paths, env_regex_ns=regex, articulations={"robot": robot})
    return types.SimpleNamespace(scene=scene, num_envs=len(paths), device="cuda:0", sim=sim)


def folds(fam, robot, spec_per_env):
    """The observation's expected value, pose_frames per environment's design, in articulation body order."""
    ids, names = robot.find_bodies(FOLDED[fam.name])
    want = [[fam.module.builder.pose_frames(s)[n.split("_")[0]] for n in names] for s in spec_per_env]
    return ids, names, torch.tensor(want)


def realised_folds(robot, names):
    zero_pose(robot)
    q = robot.data.body_link_quat_w.cpu()
    out = []
    for n in names:
        kind, side = n.split("_")[0], n.split("_")[1]
        c = q[:, robot.body_names.index(n)]
        p = q[:, robot.body_names.index(PARENT[kind].format(side))]
        w = p[:, 0] * c[:, 0] + (p[:, 1:] * c[:, 1:]).sum(-1)
        y = p[:, 0] * c[:, 2] - c[:, 0] * p[:, 2] - (p[:, 3] * c[:, 1] - p[:, 1] * c[:, 3])
        out.append(2.0 * torch.atan2(y, w))
    return torch.stack(out, 1)


FAMS = [a for a in sys.argv[1:] if a in ("quadruped", "biped")] or ["quadruped", "biped"]
for name in FAMS:
    gen = U.CMAESDesignGenerator(family=name, num_individuals=2, output_dir=OUT)
    fam = gen.family
    a = gen._spec_from_parameters({"thigh_length_scale": 1.2, "shank_length_scale": 0.85,
                                   "thigh_bend_offset": 0.06, "knee_bend_offset": -0.2})
    b = gen._spec_from_parameters({"thigh_length_scale": 0.8, "shank_length_scale": 1.15,
                                   "thigh_bend_offset": -0.08, "knee_bend_offset": 0.22}, PICKS_B[name])
    U.check_nominal_pose(fam, a, gen._dx_target, gen._lean_max())
    U.check_nominal_pose(fam, b, gen._dx_target, gen._lean_max())
    na, nb = fam.nominal(a), fam.nominal(b)
    print(f"\n==== {name}: A {a.actuators()[0].name}.. offsets {na['offsets']} height {na['height']:.4f}, "
          f"B picks {PICKS_B[name]} offsets {nb['offsets']} height {nb['height']:.4f}")
    usd_a, usd_b = convert(fam, a, "A"), convert(fam, b, "B")
    worst_frames = check_joint_frames(fam, a, usd_a)
    print(f" joint frames of A, worst localRot0 rule error {worst_frames:.2e}")

    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    for i, usd in enumerate((usd_a, usd_a, usd_b)):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform")
        sim_utils.UsdFileCfg(usd_path=usd).func(f"/World/envs/env_{i}/Robot", sim_utils.UsdFileCfg(usd_path=usd))
    robot = Articulation(ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot", spawn=None,
        articulation_root_prim_path=ROOT_PRIM[name],
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    ))
    sim.reset()
    stage = sim_utils.get_current_stage()
    root = fam.root_body
    paths = [f"/World/envs/env_{i}" for i in range(3)]
    env = stub_env(robot, paths, "/World/envs/env_.*", sim)

    depth, lead = measure(robot, fam)
    err_a = max(np.abs(depth[:2] - na["depth"]).max(), np.abs(lead[:2] - na["dx"]).max())
    print(f" A spawned: depth {depth[:2, 0]} want {na['depth']:.5f}, lead {lead[:2, 0]} want {na['dx']:.5f}, worst {err_a * 1e3:.3f} mm")

    ids, names, _ = folds(fam, robot, [a, a, b])
    asset_cfg = SceneEntityCfg("robot", body_names=FOLDED[name])
    asset_cfg.body_ids = ids
    term = mdp.robot_nominal_pose(ObservationTermCfg(func=mdp.robot_nominal_pose, params={"asset_cfg": asset_cfg}), env)
    obs = term(env, asset_cfg).cpu()
    obs_err_a = (obs - folds(fam, robot, [a, a, b])[2]).abs().max().item()
    print(f" observation over {names}: {np.round(obs.numpy(), 5).tolist()}, worst against pose_frames {obs_err_a:.2e} rad")

    before = physics_state(robot, stage, root)
    body_names, want_b = list(robot.body_names), folds(fam, robot, [b])[2]

    # B in place through environment 0 alone, the production path
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    sim._disable_app_control_on_stop_handle = False
    UP.apply_link_length_params(stub_env(robot, paths[:1], "/World/envs/env_.*", sim), [fam.links(b)], root)
    UP.assert_link_lengths_applied(env, [fam.links(b)])
    state = physics_state(robot, stage, root)

    def gap(value, rows):
        if value.ndim == 3 and value.shape[-1] == 4:
            return float((1.0 - np.abs((value[rows] * value[2:3]).sum(-1))).max())
        return float(np.abs(value[rows] - value[2:3]).max())

    diffs, moved = {}, {}
    print(" environments 0 (written) and 1 (never written) against environment 2 (B converted), A against B before the write:")
    for key in state:
        diffs[key], moved[key] = gap(state[key], slice(0, 2)), gap(before[key], slice(0, 1))
        print(f"   {key:18s} after {diffs[key]:.2e}   before {moved[key]:.2e}")
    depth, lead = measure(robot, fam)
    err_b = max(np.abs(depth - nb["depth"]).max(), np.abs(lead - nb["dx"]).max())
    print(f" B in place: depth {depth[:, 0]} want {nb['depth']:.5f}, lead {lead[:, 0]} want {nb['dx']:.5f}, worst {err_b * 1e3:.3f} mm")

    obs = term(env, asset_cfg).cpu()
    obs_err_b = (obs - want_b).abs().max().item()
    kin_err = (obs - realised_folds(robot, names)).abs().max().item()
    print(f" observation after the write: worst against pose_frames(B) {obs_err_b:.2e} rad, against the realised fold {kin_err:.2e} rad")

    # the stage read at scale, the cost every design swap pays once
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    sim._disable_app_control_on_stop_handle = False
    many = [f"/World/timing/env_{i}" for i in range(1024)]
    for p in many:
        sim_utils.UsdFileCfg(usd_path=usd_a).func(f"{p}/Robot", sim_utils.UsdFileCfg(usd_path=usd_a))
    asset = types.SimpleNamespace(body_names=body_names, cfg=types.SimpleNamespace(prim_path="/World/timing/env_.*/Robot"))
    start = time.perf_counter()
    big = mdp.robot_nominal_pose._read(stub_env(asset, many, "/World/timing/env_.*"), asset, asset_cfg)
    read_s = time.perf_counter() - start
    print(f" stage read over {len(many)} environments referencing A's file: {read_s:.3f} s, {big.shape}, "
          f"worst against pose_frames(B) {(big.cpu() - want_b).abs().max().item():.2e} rad")

    results.append((name, worst_frames, err_a, err_b, max(diffs.values()), min(moved.values()), obs_err_a, obs_err_b, kin_err, read_s))
    sim.clear_all_callbacks()
    sim.clear_instance()

print("\nSUMMARY family, localRot0 rule, A spawned (m), B in place (m), in place vs converted, smallest A to B change, "
      "observation A, observation B, observation vs realised, stage read of 1024 (s)")
for r in results:
    print("SUMMARY", tuple(round(v, 9) if isinstance(v, float) else v for v in r))
app.close()
