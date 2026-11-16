"""Spawn the ten joint KScale and read back what the simulator actually built.

ARM 14 of plans/kscale_gait_shaping.md section 21.14.  validate_arm14.py settles every
structural claim from the URDF and the configuration sources without a simulator, which is
what makes it cheap enough to run on every edit.  This program is the one check it cannot
make, being whether PhysX and the URDF importer agree with the prediction, and it is run once
after the arm is applied rather than routinely.

It asserts the two counts the arm turns upon, ten joints and THIRTEEN bodies, builds the
symmetry maps against the live articulation, and confirms the twelve joint task still builds
to twelve and thirteen in the same process.
"""
import argparse
import sys

# Building a SECOND ManagerBasedRLEnv configuration in one process exhausts the default
# recursion limit inside configclass._validate at configclass.py:291, which recurses once per
# nested configuration field and is a property of that validator against a deep tree rather
# than of any robot. The limit is raised rather than the probe being split, so that both
# robots are read in one process and the counts are directly comparable.
sys.setrecursionlimit(20000)

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Spawn probe for arm 14.")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
args.enable_cameras = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import environments  # noqa: E402, F401  registers the tasks
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from environments.tasks.locomotion.mdp.symmetry import kscale as sym  # noqa: E402

EXPECT = {
    "Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-v0": (10, 13, False),
    "Isaac-Limx-Kscale-Blind-Flat-v0": (12, 13, True),
}

fails = []


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' -- ' + detail) if not ok else ''}")
    if not ok:
        fails.append(label)


for task, (n_joints, n_bodies, has_yaw) in EXPECT.items():
    print(f"=== {task}")
    cfg = parse_env_cfg(task, device=args.device, num_envs=4)
    env = gym.make(task, cfg=cfg)
    robot = env.unwrapped.scene["robot"]
    jn, bn = list(robot.joint_names), list(robot.body_names)
    check(f"{n_joints} joints", len(jn) == n_joints, f"{len(jn)}: {jn}")
    check(f"{n_bodies} bodies", len(bn) == n_bodies, f"{len(bn)}: {bn}")
    check("hip yaw presence is as predicted",
          any("hip_yaw" in n for n in jn) == has_yaw, str(jn))
    check("both hip yaw CHILD links are present as bodies",
          {"kd_d_301l_6061", "kd_d_301r_6061"} <= set(bn),
          str(sorted(set(bn))))
    check("the action dimension equals the joint count",
          env.unwrapped.action_manager.total_action_dim == n_joints,
          str(env.unwrapped.action_manager.total_action_dim))
    masses = float(robot.data.default_mass[0].sum())
    print(f"       total mass {masses:.5f} kg over {len(bn)} bodies")

    # the symmetry maps, built against this live articulation
    sym._CACHE.clear()
    jperm, jsign, perm = sym._build_joint_map(jn, robot.device)
    check("the joint map is an involution", all(perm[p] == i for i, p in enumerate(perm)))
    bperm = sym._build_body_map(bn)
    check("the body map is an involution", all(bperm[p] == i for i, p in enumerate(bperm)))
    flipped = sorted(jn[i] for i, s in enumerate(jsign.tolist()) if s < 0)
    print(f"       sign flipped joints {flipped}")

    # one observation and one augmentation pass, which is where a width mismatch would raise
    obs, _ = env.reset()
    pol = obs["policy"]
    print(f"       policy observation width {tuple(pol.shape)}")
    # compute_symmetric_states reads obs.batch_size, so the groups must arrive as a
    # TensorDict exactly as rsl_rl hands them over, not as the plain dict gym returns.
    from tensordict import TensorDict
    td = TensorDict({k: v for k, v in obs.items()}, batch_size=[pol.shape[0]])
    aug_obs, aug_act = sym.compute_symmetric_states(
        env=env, obs=td, actions=torch.zeros(
            env.unwrapped.num_envs, n_joints, device=robot.device),
    )
    check("the augmentation doubles the batch",
          aug_obs["policy"].shape[0] == 2 * pol.shape[0],
          str(tuple(aug_obs["policy"].shape)))
    check("the mirrored action keeps its width",
          aug_act.shape[-1] == n_joints, str(tuple(aug_act.shape)))
    env.close()
    print()

print(f"{'ALL CHECKS PASSED' if not fails else str(len(fails)) + ' FAILED'}")
for f in fails:
    print(f"  FAILED: {f}")
simulation_app.close()
