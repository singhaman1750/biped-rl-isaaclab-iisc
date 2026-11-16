"""Invariance check of the nominal pose reward terms, run with Isaac Sim, not collected by pytest.

Each v2 is driven with a fake environment three ways, the attribute absent, the attribute
equal to the donor's height in every environment, and a per environment height, and is
compared with its original. The first two must equal the original exactly, the third
must equal the original evaluated environment by environment at that environment's
scaled parameter.
"""
import types

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import torch  # noqa: E402

import environments.tasks.locomotion.mdp as mdp  # noqa: E402

print("mdp from", mdp.__file__)
torch.manual_seed(0)
N, F, R = 6, 4, 50
dev = "cpu"


class Obj(types.SimpleNamespace):
    pass


def make_env():
    root = torch.rand(N, 3) + torch.tensor([0, 0, 0.2])
    hits = torch.rand(N, R, 3) * 0.05
    hits[0, 3, 2] = float("inf")
    hits[2, 7, 2] = float("nan")
    xy = (torch.rand(N, R, 2) - 0.5) * 1.0
    hits[..., :2] = xy + root[:, None, :2]
    robot = Obj(data=Obj(root_pos_w=root,
                         body_pos_w=torch.cat([(torch.rand(N, F, 2) - 0.5) * 0.8 + root[:, None, :2], torch.rand(N, F, 1) * 0.15], -1),
                         body_lin_vel_w=torch.randn(N, F, 3),
                         body_quat_w=torch.tensor([1.0, 0, 0, 0]).repeat(N, F, 1)))
    scanner = Obj(data=Obj(ray_hits_w=hits))
    first = (torch.rand(N, F) > 0.5).float()
    air = torch.rand(N, F) * 0.8
    forces = torch.rand(N, 3, F, 3) * 3.0
    contact = Obj(data=Obj(last_air_time=air, net_forces_w_history=forces),
                  compute_first_contact=lambda dt: first.bool())
    sensors = {"height_scanner": scanner, "contact_forces": contact}

    class Scene(dict):
        pass
    scene = Scene(robot=robot)
    scene.sensors = sensors
    cmd = torch.rand(N, 3) * 2 - 1
    env = Obj(scene=scene, step_dt=0.02, num_envs=N, device=dev,
              command_manager=Obj(get_command=lambda name: cmd))
    return env


class Cfg:
    def __init__(self, name, ids=slice(None)):
        self.name, self.body_ids = name, ids


env = make_env()
robot_cfg = Cfg("robot")
scan_cfg = Cfg("height_scanner")
cs_cfg = Cfg("contact_forces")
fails = []


def check(name, a, b, tol=0.0):
    err = (a - b).abs().max().item()
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name:60s} max |diff| {err:.3e}")
    if not ok:
        fails.append(name)


heights = torch.tensor([0.20, 0.25, 0.292, 0.33, 0.37, 0.40])

ref = mdp.base_height_rough_l2(env, 0.292, scan_cfg, robot_cfg)
check("base_height v2, attribute absent", mdp.base_height_rough_l2_v2(env, 0.292, scan_cfg, robot_cfg), ref)
env.copt_nominal_base_height = torch.full((N,), 0.292)
check("base_height v2, donor height everywhere", mdp.base_height_rough_l2_v2(env, 0.292, scan_cfg, robot_cfg), ref, 1e-7)
env.copt_nominal_base_height = heights
per = torch.stack([mdp.base_height_rough_l2(env, float(h), scan_cfg, robot_cfg)[i] for i, h in enumerate(heights)])
check("base_height v2, per environment", mdp.base_height_rough_l2_v2(env, 0.292, scan_cfg, robot_cfg), per, 1e-7)
del env.copt_nominal_base_height

fr_cfg = Cfg("robot")
ref = mdp.feet_regulation(env, fr_cfg, 0.05, 0.804)
check("feet_regulation v2, attribute absent", mdp.feet_regulation_v2(env, fr_cfg, 0.05, 0.804), ref)
env.copt_nominal_base_height = torch.full((N,), 0.804)
check("feet_regulation v2, donor height everywhere", mdp.feet_regulation_v2(env, fr_cfg, 0.05, 0.804), ref, 1e-6)
env.copt_nominal_base_height = heights * 2.75
per = torch.stack([mdp.feet_regulation(env, fr_cfg, 0.05, float(h))[i] for i, h in enumerate(heights * 2.75)])
check("feet_regulation v2, per environment", mdp.feet_regulation_v2(env, fr_cfg, 0.05, 0.804), per, 1e-6)
del env.copt_nominal_base_height

for orig, new, ref_h, tmin, tmax in ((mdp.feet_air_time_v2, mdp.feet_air_time_v3, 0.292, 0.15, 0.45),
                                     (mdp.feet_air_time, mdp.feet_air_time_nominal, 0.804, 0.2, 0.5)):
    ref = orig(env, "base_velocity", cs_cfg, tmin, tmax)
    check(f"{new.__name__}, attribute absent", new(env, "base_velocity", cs_cfg, tmin, tmax, ref_h), ref)
    env.copt_nominal_base_height = torch.full((N,), ref_h)
    check(f"{new.__name__}, donor height everywhere", new(env, "base_velocity", cs_cfg, tmin, tmax, ref_h), ref, 1e-6)
    hh = heights * ref_h / 0.292
    env.copt_nominal_base_height = hh
    k = torch.sqrt(hh / ref_h)
    per = torch.stack([orig(env, "base_velocity", cs_cfg, tmin * float(k[i]), tmax * float(k[i]))[i] for i in range(N)])
    check(f"{new.__name__}, per environment", new(env, "base_velocity", cs_cfg, tmin, tmax, ref_h), per, 1e-6)
    del env.copt_nominal_base_height

args = dict(asset_cfg=Cfg("robot"), height_sensor_cfg=scan_cfg, target_height=0.10, std=0.05, tanh_mult=2.0, sensor_cfg=cs_cfg)
ref = mdp.foot_clearance_reward_v4(env, **args)
check("foot_clearance v6, attribute absent", mdp.foot_clearance_reward_v6(env, **args, target_height_frac=0.10 / 0.292), ref)
env.copt_nominal_base_height = torch.full((N,), 0.292)
check("foot_clearance v6, donor height everywhere", mdp.foot_clearance_reward_v6(env, **args, target_height_frac=0.10 / 0.292), ref, 1e-6)
check("foot_clearance v6, fraction unset", mdp.foot_clearance_reward_v6(env, **args), ref)
env.copt_nominal_base_height = heights
per = torch.stack([mdp.foot_clearance_reward_v4(env, **{**args, "target_height": float(h) * 0.10 / 0.292})[i] for i, h in enumerate(heights)])
check("foot_clearance v6, per environment", mdp.foot_clearance_reward_v6(env, **args, target_height_frac=0.10 / 0.292), per, 1e-6)
del env.copt_nominal_base_height

print("RESULT", "PASS" if not fails else f"FAIL {fails}")
app.close()
