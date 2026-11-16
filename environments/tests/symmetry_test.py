"""Validation of the sagittal symmetry package, ``mdp/symmetry``, against stub environments.

The package is loaded as a free standing package by path rather than imported through
``environments``, whose ``__init__`` pulls in Isaac Lab, so the suite runs on torch and tensordict
alone. The stubs expose exactly the surfaces the package reads, the robot's names, limits, default
pose and PhysX views, the observation manager's layout and term configurations, and the sensors.

The SD_BRS1 and kscale modules are compared bitwise against their implementation at commit
23513d2, read with ``git show``, so the refactor onto ``common`` is shown to change the actions and
the policy group not at all, and the critic only in the four terms the old modules left unmirrored.

Run with ``python3 -m pytest environments/tests/symmetry_test.py``, or directly for a report.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import pathlib
import subprocess
import sys
import tempfile
import types

import torch
from tensordict import TensorDict

_REPO = pathlib.Path(__file__).resolve().parents[2]
_REL = "environments/environments/tasks/locomotion/mdp/symmetry"
_SYMMETRY = pathlib.Path(os.environ.get("SYMMETRY_DIR", _REPO / _REL))
_LEGACY_COMMIT = "23513d2"
N = 64


def _package(name: str, directory) -> types.ModuleType:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(directory)]
    sys.modules[name] = pkg
    return pkg


def _legacy_dir() -> pathlib.Path | None:
    """The legacy modules, from SYMMETRY_LEGACY_DIR or from git. Inside the Isaac Lab containers
    this repository is a submodule whose git directory is not mounted, so git cannot serve them."""
    if "SYMMETRY_LEGACY_DIR" in os.environ:
        return pathlib.Path(os.environ["SYMMETRY_LEGACY_DIR"])
    out = pathlib.Path(tempfile.mkdtemp(prefix="symmetry_legacy_"))
    for mod in ("brs", "kscale"):
        res = subprocess.run(["git", "-C", str(_REPO), "show", f"{_LEGACY_COMMIT}:{_REL}/{mod}.py"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            return None
        (out / f"{mod}.py").write_text(res.stdout)
    return out


_package("sym_new", _SYMMETRY)
_LEGACY = _legacy_dir()
if _LEGACY is not None:
    _package("sym_old", _LEGACY)
common = importlib.import_module("sym_new.common")


# ---------------------------------------------------------------- stubs

class _Entity:
    def __init__(self, name, body_ids: slice | list[int] = slice(None)):
        self.name, self.body_ids = name, body_ids


class _Term:
    def __init__(self, params=None, history_length=0):
        self.params, self.history_length = params or {}, history_length


class _View:
    def __init__(self, shapes):
        self.link_paths = [[f"/link{i}" for i in range(len(shapes))]]
        self.max_shapes = sum(shapes)


class _SimView:
    def __init__(self, shapes):
        self._shapes = shapes

    def create_rigid_body_view(self, path):
        return types.SimpleNamespace(max_shapes=self._shapes[int(path[5:])])


class _Robot:
    def __init__(self, joints, bodies, limits, shapes, default=None):
        self.joint_names, self.body_names = joints, bodies
        pose = {".*": 0.0} if default is None else dict(zip(joints, default))
        self.cfg = types.SimpleNamespace(init_state=types.SimpleNamespace(joint_pos=pose))
        self.data = types.SimpleNamespace(joint_pos_limits=torch.tensor(limits)[None].repeat(2, 1, 1))
        self.root_physx_view = _View(shapes)
        self._physics_sim_view = _SimView(shapes)


def _grid_rays(ordering="xy", size=(1.6, 1.0), res=0.1):
    x = torch.arange(-size[0] / 2, size[0] / 2 + 1e-9, res)
    y = torch.arange(-size[1] / 2, size[1] / 2 + 1e-9, res)
    gx, gy = torch.meshgrid(x, y, indexing="xy" if ordering == "xy" else "ij")
    starts = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(gx.numel())], dim=1) + torch.tensor([0.0, 0.0, 20.0])
    return types.SimpleNamespace(ray_starts=starts[None].repeat(2, 1, 1), cfg=types.SimpleNamespace(ray_alignment="yaw"))


class _ObsManager:
    """groups maps a group to (terms, flatten), a term being (name, width of one frame, cfg)."""

    def __init__(self, groups):
        self.active_terms, self.group_obs_term_dim, self._group_obs_term_cfgs = {}, {}, {}
        self.group_obs_concatenate, self._group_obs_concatenate_dim = {}, {}
        for g, (terms, flatten) in groups.items():
            self.add(g, terms, flatten)

    def add(self, g, terms, flatten=True):
        h = lambda t: max(t[2].history_length, 1)  # noqa: E731
        self.active_terms[g] = [t[0] for t in terms]
        self.group_obs_term_dim[g] = [(t[1] * h(t),) if flatten or t[2].history_length == 0 else (h(t), t[1])
                                      for t in terms]
        self._group_obs_term_cfgs[g] = [t[2] for t in terms]
        self.group_obs_concatenate[g], self._group_obs_concatenate_dim[g] = True, -1


class _Env:
    def __init__(self, robot, groups, sensors):
        self.scene = _Scene({"robot": robot}, sensors)
        self.observation_manager = _ObsManager(groups)

    @property
    def unwrapped(self):
        return self


class _Scene(dict):
    def __init__(self, entities, sensors):
        super().__init__(entities)
        self.sensors = sensors


def _proprio(nj, gait=False, lin_vel=True):
    terms = [("base_lin_vel", 3)] if lin_vel else []
    terms += [("base_ang_vel", 3), ("proj_gravity", 3), ("joint_pos", nj), ("joint_vel", nj), ("last_action", nj),
              ("velocity_commands", 3)]
    return terms + ([("gait_phase", 2)] if gait else [])


def _critic(nj, nb, feet_ids, num_shapes, h, gait=False):
    t = lambda **params: _Term(params, h)  # noqa: E731
    feet_asset = {"asset_cfg": _Entity("robot", feet_ids)}
    feet_sensor = {"sensor_cfg": _Entity("contact_forces", [nb - 1 - i for i in feet_ids])}
    return [(n, w, t()) for n, w in _proprio(nj)] + [
        ("robot_joint_torque", nj, t()), ("robot_joint_acc", nj, t()),
        ("feet_lin_vel", 3 * len(feet_ids), t(**feet_asset)),
        ("robot_mass", nb, t()), ("robot_inertia", 9 * nb, t()),
        ("robot_joint_pos", nj, t()), ("robot_joint_stiffness", nj, t()),
        ("robot_joint_damping", nj, t()), ("robot_pos", 3, t()), ("robot_vel", 6, t()),
        ("robot_material_properties", 3 * num_shapes, t()),
        ("feet_contact_force", 3 * len(feet_ids), t(**feet_sensor)),
        ("heights", 187, t(sensor_cfg=_Entity("height_scanner"))),
    ] + ([("gait_phase", 2, t())] if gait else [])


def _groups(nj, nb, feet_ids, num_shapes, h_policy=10, gait=False, him=False):
    commands = ([("velocity_commands", 3, _Term())], True)
    critic = (_critic(nj, nb, feet_ids, num_shapes, 10, gait), True)
    if not him:
        return {"policy": ([(n, w, _Term({}, h_policy)) for n, w in _proprio(nj, gait)], True),
                "critic": critic, "commands": commands}
    one = _proprio(nj, lin_vel=False)
    return {"policy": ([(n, w, _Term()) for n, w in one], True),
            "obsHistory": ([(n, w, _Term({}, 10)) for n, w in one], False),
            "critic": critic, "commands": commands,
            "estimatorGT": ([("base_lin_vel", 3, _Term())], True),
            "targetEnc": ([(n, w, _Term()) for n, w in one + [("base_lin_vel", 3)]], True)}


def _env(joints, bodies, limits, shapes, feet, default=None, **kw):
    robot = _Robot(joints, bodies, limits, shapes, default)
    feet_ids = [bodies.index(f) for f in feet]
    sensors = {"height_scanner": _grid_rays(kw.pop("ordering", "xy")),
               "contact_forces": types.SimpleNamespace(body_names=list(reversed(bodies)))}
    return _Env(robot, _groups(len(joints), len(bodies), feet_ids, sum(shapes), **kw), sensors)


def _obs(env, seed=0):
    g = torch.Generator().manual_seed(seed)
    om = env.observation_manager
    out = {}
    for k in om.active_terms:
        dims = om.group_obs_term_dim[k]
        lead = dims[0][:-1]
        out[k] = torch.randn(N, *lead, sum(d[-1] for d in dims), generator=g)
    return TensorDict(out, batch_size=[N])


def _starts(env, group):
    om, start, acc = env.observation_manager, {}, 0
    for n, d in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        start[n], acc = (acc, acc + d[-1]), acc + d[-1]
    return start


# ---------------------------------------------------------------- robots

BRS_JOINTS = [f"{k}{s}" for k in ("HipRoll", "HipPitch", "KneePitch", "AnkleRoll", "AnklePitch") for s in "LR"]
BRS_BODIES = ["Part_Torso"] + [f"Link{i}{s}" for i in range(1, 7) for s in "LR"]
KS_KINDS = ("hip_pitch_04", "hip_roll_03", "hip_yaw_03", "knee_04", "foot_pitch_02", "foot_roll_02")
KS_JOINTS = [f"{s}_{k}" for k in KS_KINDS for s in ("left", "right")]
KS_BODIES = ["assy_formfg___kd_b_102b_torso_btm", "kd_d_102l_6061", "kd_d_102r_6061", "rs03", "kd_d_201r_6061",
             "kd_d_301l_6061", "kd_d_301r_6061", "kd_d_401l_6061", "kd_d_401r_6061", "arb_uj111_cross_bearing_2",
             "arb_uj111_cross_bearing", "foot_6061_2", "foot_6061"]
TRON1_LIMITS = {"abad": (-0.383970, 1.396260), "hip": (-1.396260, 1.012290), "knee": (-1.213570, 1.172670),
                "ankle": (-0.791917, 1.308990)}
TRON1_JOINTS = [f"{k}_{s}_Joint" for k in TRON1_LIMITS for s in "LR"]
TRON1_BODIES = ["base_Link"] + [f"{k}_{s}_{x}" for k, x in (("abad", "act"), ("hip", "act"), ("knee", "act"),
                ("thigh", "Link"), ("shank", "Link"), ("ankle", "act"), ("foot", "Link")) for s in "LR"]
QUAD_LIMITS = {"hip": (-0.802851, 0.802851), "thigh": (-1.931537, 3.304453), "calf": (-0.927857, 0.852375)}
QUAD_LEGS = ("FL", "FR", "RL", "RR")
QUAD_JOINTS = [f"{leg}_{k}_joint" for k in QUAD_LIMITS for leg in QUAD_LEGS]
QUAD_BODIES = ["trunk"] + [f"{k}_{leg}_{x}" for k, x in (("abad", "act"), ("hip", "act"), ("knee", "act"),
               ("thigh", "Link"), ("shank", "Link"), ("foot", "Link")) for leg in QUAD_LEGS]
# the Go1 convention, abduction out on both sides at opposite signs
QUAD_DEFAULT = [{"L": 0.1, "R": -0.1}[j[1]] if "_hip_" in j else 0.8 for j in QUAD_JOINTS]


def _brs_env(**kw):
    return _env(BRS_JOINTS, BRS_BODIES, [(-1.0, 1.0)] * 10, [1] * 7 + [2] * 6, ["Link6L", "Link6R"], gait=True, **kw)


def _kscale_env(**kw):
    return _env(KS_JOINTS, KS_BODIES, [(-1.0, 1.0)] * 12, [1] * 13, ["foot_6061", "foot_6061_2"], **kw)


def _tron1_env(**kw):
    limits = [TRON1_LIMITS[j.split("_")[0]] for j in TRON1_JOINTS]
    return _env(TRON1_JOINTS, TRON1_BODIES, limits, [1] * 15, ["foot_L_Link", "foot_R_Link"], **kw)


def _quad_env(**kw):
    limits = [QUAD_LIMITS[j.split("_")[1]] for j in QUAD_JOINTS]
    return _env(QUAD_JOINTS, QUAD_BODIES, limits, [1] * 25, [f"foot_{leg}_Link" for leg in QUAD_LEGS],
                default=QUAD_DEFAULT, **kw)


def _fresh(pkg, mod):
    m = importlib.import_module(f"{pkg}.{mod}")
    m._CACHE.clear()
    return m


def _involution(mod, env):
    m = _fresh("sym_new", mod)
    obs, act = _obs(env), torch.randn(N, len(env.scene["robot"].joint_names))
    o1, a1 = m.compute_symmetric_states(env=env, obs=obs, actions=act)
    o2, a2 = m.compute_symmetric_states(env=env, obs=o1[N:], actions=a1[N:])
    assert set(o1.keys()) == set(obs.keys())
    for k in obs.keys():
        assert torch.equal(o1[k][:N], obs[k]), (mod, k, "originals first")
        assert torch.equal(o2[k][N:], obs[k]), (mod, k, "involution")
    assert torch.equal(a2[N:], act)
    return o1, a1, obs, act


# ---------------------------------------------------------------- tests

def test_legacy_actions_and_policy_unchanged():
    """Against commit 23513d2, the actions and the policy group are bitwise equal, the critic differs only in
    the four terms the old modules tiled, and the commands group, which they tiled, is now mirrored."""
    if _LEGACY is None:
        print("SKIP test_legacy_actions_and_policy_unchanged, no legacy source, set SYMMETRY_LEGACY_DIR")
        return
    corrected = ("feet_lin_vel", "robot_vel", "robot_material_properties", "feet_contact_force")
    for mod, make in (("brs", _brs_env), ("kscale", _kscale_env)):
        old, new = _fresh("sym_old", mod), _fresh("sym_new", mod)
        env = make()
        obs, act = _obs(env), torch.randn(N, len(env.scene["robot"].joint_names))
        keep = torch.ones(obs["critic"].shape[-1], dtype=torch.bool)
        for name, (a, b) in _starts(env, "critic").items():
            if name in corrected:
                keep[a:b] = False
        for kwargs in ({"obs": obs, "actions": act}, {"obs": obs, "actions": None}, {"obs": None, "actions": act}):
            o_old, a_old = old.compute_symmetric_states(env=env, **kwargs)
            o_new, a_new = new.compute_symmetric_states(env=env, **kwargs)
            assert (a_old is None) == (a_new is None) and (o_old is None) == (o_new is None)
            if a_old is not None:
                assert torch.equal(a_old, a_new), mod
            if o_old is not None:
                assert torch.equal(o_old["policy"], o_new["policy"]), (mod, "policy")
                assert torch.equal(o_old["critic"][:, keep], o_new["critic"][:, keep]), (mod, "critic")
                assert not torch.equal(o_old["critic"][N:, ~keep], o_new["critic"][N:, ~keep]), (mod, "corrected")
                assert torch.equal(o_new["commands"][N:], obs["commands"] * torch.tensor(common.CMD)), (mod, "commands")
        _involution(mod, make())


def test_tron1_maps():
    o1, a1, obs, act = _involution("tron1", _tron1_env())
    jperm = [TRON1_JOINTS.index(n.replace("_L_", "_X_").replace("_R_", "_L_").replace("_X_", "_R_")) for n in TRON1_JOINTS]
    assert torch.equal(a1[N:], act[:, jperm]), "TRON1 flips no joint"
    c, c1 = obs["critic"], o1["critic"][N:]
    assert torch.equal(c1[:, :30].view(N, 10, 3), c[:, :30].view(N, 10, 3) * torch.tensor(common.VEC))
    assert torch.equal(c1[:, 30:60].view(N, 10, 3), c[:, 30:60].view(N, 10, 3) * torch.tensor(common.PVEC))


def test_quadruped_maps():
    _, a1, _, act = _involution("quadruped", _quad_env())
    for j, name in enumerate(QUAD_JOINTS):
        leg, kind = name[:2], name.split("_")[1]
        p = QUAD_JOINTS.index(name.replace(leg, leg[0] + {"L": "R", "R": "L"}[leg[1]], 1))
        s = -1.0 if kind == "hip" else 1.0
        assert torch.equal(a1[N:, j], s * act[:, p]), name


def test_heights_from_rays_matches_grid_and_handles_yx():
    xy = common.ray_mirror_perm(_grid_rays("xy"))
    assert xy == torch.arange(187).view(11, 17).flip(0).flatten().tolist()
    yx = common.ray_mirror_perm(_grid_rays("yx"))
    assert yx == torch.arange(187).view(17, 11).flip(1).flatten().tolist()


def test_world_frame_kinds_exchange_the_feet():
    env = _tron1_env()
    o1, _, obs, _ = _involution("tron1", env)
    start = _starts(env, "critic")
    c, c1 = obs["critic"], o1["critic"][N:]
    rv = slice(*start["robot_vel"])
    assert torch.equal(c1[:, rv].view(N, 10, 6), c[:, rv].view(N, 10, 6) * torch.tensor(common.ROOT_VEL_W))
    for term in ("feet_lin_vel", "feet_contact_force"):
        fv = slice(*start[term])
        assert torch.equal(c1[:, fv].view(N, 10, 2, 3), c[:, fv].view(N, 10, 2, 3).flip(2)), (term, "feet exchanged")
    pos = slice(*start["robot_pos"])
    assert torch.equal(c1[:, pos], c[:, pos]), "root position on the plane"


def test_him_layout_mirrors_the_unflattened_history():
    """The HIM groups, a one frame policy and a (10, w) history, for the robot that has them all."""
    for mod, make in (("tron1", _tron1_env), ("quadruped", _quad_env), ("brs", _brs_env), ("kscale", _kscale_env)):
        env = make(him=True)
        o1, _, obs, _ = _involution(mod, env)
        assert obs["obsHistory"].dim() == 3
        h, h1, p, p1 = obs["obsHistory"], o1["obsHistory"][N:], obs["policy"], o1["policy"][N:]
        m = importlib.import_module(f"sym_new.{mod}")._CACHE[torch.device("cpu")]["groups"]
        pi, ps = m["policy"]
        assert torch.equal(h1, h[..., pi] * ps), (mod, "each history frame mirrors as one policy frame")
        assert torch.equal(p1, p[..., pi] * ps)


def test_copt_groups_of_the_quadruped():
    """The co-optimisation groups mirror, link lengths by pair and the morphology over its body subset."""
    env = _quad_env()
    legs = ("FR", "FL", "RR", "RL")
    parents = [f"thigh_{leg}_Link" for leg in legs] + [f"shank_{leg}_Link" for leg in legs]
    children = [f"shank_{leg}_Link" for leg in legs] + [f"foot_{leg}_Link" for leg in legs]
    moved = [i for i, b in enumerate(QUAD_BODIES) if b.split("_")[0] in ("abad", "hip", "knee", "thigh", "shank")]
    fold = [i for i, b in enumerate(QUAD_BODIES) if b.split("_")[0] in ("thigh", "shank")]
    lengths = {"parent_body_names": parents, "child_body_names": children, "asset_cfg": _Entity("robot")}
    om = env.observation_manager
    om.add("historyObs", [(n, w, _Term({}, 25)) for n, w in _proprio(12)]
           + [("heights", 187, _Term({"sensor_cfg": _Entity("height_scanner")}, 25))])
    om.add("morphologyObs", [("link_lengths", 8, _Term(lengths)),
                             ("robot_mass", len(moved), _Term({"asset_cfg": _Entity("robot", moved)})),
                             ("robot_inertia", 9 * len(moved), _Term({"asset_cfg": _Entity("robot", moved)})),
                             ("nominal_pose", len(fold), _Term({"asset_cfg": _Entity("robot", fold)}))])
    feet = [QUAD_BODIES.index(f"foot_{leg}_Link") for leg in QUAD_LEGS]
    om.add("privilegedDynamicsObs", [("robot_joint_torque", 12, _Term()), ("robot_joint_acc", 12, _Term()),
                                     ("feet_contact_force", 12, _Term({"sensor_cfg": _Entity("contact_forces", [24 - i for i in feet])})),
                                     ("feet_lin_vel", 12, _Term({"asset_cfg": _Entity("robot", feet)}))])
    om.add("estimatorGT", [("base_lin_vel", 3, _Term())])
    om.add("critic", [*zip(om.active_terms["critic"], [d[0] // 10 for d in om.group_obs_term_dim["critic"]],
                           om._group_obs_term_cfgs["critic"]), ("link_lengths", 8, _Term(lengths, 10)),
                      ("nominal_pose", len(fold), _Term({"asset_cfg": _Entity("robot", fold)}, 10))])
    o1, _, obs, _ = _involution("quadruped", env)
    ll, ll1 = obs["morphologyObs"][:, :8], o1["morphologyObs"][N:, :8]
    assert torch.equal(ll1, ll[:, [1, 0, 3, 2, 5, 4, 7, 6]]), "link lengths exchange FR with FL and RR with RL"
    mass, mass1 = obs["morphologyObs"][:, 8:8 + len(moved)], o1["morphologyObs"][N:, 8:8 + len(moved)]
    names = [QUAD_BODIES[i] for i in moved]
    swap = common.regex_side_partner(r"(?:^|_)[FR](?P<side>[RL])_")
    assert torch.equal(mass1, mass[:, [names.index(swap(n)) for n in names]])
    assert torch.equal(o1["estimatorGT"][N:], obs["estimatorGT"] * torch.tensor(common.VEC))


def test_joint_limits_reject_a_wrong_flip_set():
    wrong = dataclasses.replace(_fresh("sym_new", "tron1").SPEC, flip_joint_kinds=("abad",))
    try:
        common.compile_maps(_tron1_env(), wrong, "cpu")
    except ValueError as err:
        assert "abad" in str(err)
    else:
        raise AssertionError("a flipped TRON1 abduction must contradict its identical limits")


def test_default_pose_rejects_a_missing_flip():
    """Symmetric abduction limits cannot decide the flip, the default pose of opposite abductions can."""
    wrong = dataclasses.replace(_fresh("sym_new", "quadruped").SPEC, flip_joint_kinds=())
    try:
        common.compile_maps(_quad_env(), wrong, "cpu")
    except ValueError as err:
        assert "default pose" in str(err)
    else:
        raise AssertionError("an unflipped abduction must contradict a default pose of opposite abductions")


def test_an_unmapped_term_of_any_group_raises():
    for group in ("critic", "commands"):
        env = _tron1_env()
        om = env.observation_manager
        om.active_terms[group].append("unmapped")
        om.group_obs_term_dim[group].append((1,))
        om._group_obs_term_cfgs[group].append(_Term())
        try:
            common.compile_maps(env, _fresh("sym_new", "tron1").SPEC, "cpu")
        except KeyError:
            pass
        else:
            raise AssertionError(f"an unmapped term of {group} must raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
