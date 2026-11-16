"""Catch non-finite link velocities in a locomotion scene and record the state around each.

Steps a task's scene with Gaussian actions, no learning, and after every step checks the
link velocities PhysX returns, robot.data.body_lin_vel_w, the quantity behind the
feet_lin_vel observation. For every environment found non-finite it records whether the
environment was reset on that step, its design, its terrain cell, the bodies affected,
the root and joint state, the foot contact forces, and three re-reads that separate a
stale post-reset readback from a broken simulation: a fresh read of the same PhysX
buffer, a read after update_articulations_kinematic, and the next step's read.

A co-optimisation task (task id containing "Copt") is first given its generation zero
population through CoptOnPolicyRunner._reload_morphology, exactly as learn does, so the
scene holds one USD per design. Any other task runs on its configured single asset and
serves as the control.

Usage:
    python scripts/analysis/nan_reset_diagnostic.py --task Isaac-Quadruped-Copt-MoRAL-Rough-v0 \
        --num_envs 6400 --steps 20000 --out /tmp/nan_diag/copt --headless
    python scripts/analysis/nan_reset_diagnostic.py --task Isaac-Quadruped-Blind-Rough-v0 \
        --num_envs 6400 --steps 20000 --out /tmp/nan_diag/single --headless

See context/isaaclab_diagnostics.md.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--steps", type=int, default=20000, help="Environment steps to run.")
parser.add_argument("--action_std", type=float, default=1.0, help="Std of the Gaussian actions.")
parser.add_argument("--copt_num_individuals", type=int, default=80)
parser.add_argument("--max_events", type=int, default=50, help="Non-finite environments recorded in full.")
parser.add_argument("--out", type=str, required=True, help="Directory for events.pt and summary.txt.")
parser.add_argument("--pair_contacts", action="store_true",
                    help="Measure each foot's contact force against every shank and thigh.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import environments  # noqa: E402, F401
from isaaclab.sensors import ContactSensorCfg  # noqa: E402

LEGS = ("FR", "FL", "RR", "RL")


def add_pair_sensors(env_cfg) -> None:
    """One sensor per foot, filtered against every shank and thigh, the reporting being one to many."""
    for leg in LEGS:
        setattr(env_cfg.scene, f"pair_contact_{leg}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Robot/trunk/foot_{leg}_Link",
            # One path per body, PhysX not expanding a pattern that matches several.
            filter_prim_paths_expr=[
                f"{{ENV_REGEX_NS}}/Robot/trunk/{link}_{other}_Link"
                for link in ("shank", "thigh") for other in LEGS
            ],
            history_length=1,
            update_period=0.0,
        ))


class PairContacts:
    """Accumulates the force of each foot on each shank and thigh, own leg and other legs."""

    def __init__(self, scene):
        self.sensors = {leg: scene.sensors[f"pair_contact_{leg}"] for leg in LEGS}
        self.names = {leg: [p.rsplit("/", 1)[-1] for p in s.contact_physx_view.filter_paths[0]]
                      for leg, s in self.sensors.items()}
        self.max = {leg: torch.zeros(len(n)) for leg, n in self.names.items()}
        self.count = {leg: torch.zeros(len(n), dtype=torch.long) for leg, n in self.names.items()}

    def update(self):
        for leg, sensor in self.sensors.items():
            force = sensor.data.force_matrix_w[:, 0].norm(dim=-1)  # (num_envs, filters)
            self.max[leg] = torch.maximum(self.max[leg], force.max(0).values.cpu())
            self.count[leg] += (force > 1e-3).sum(0).cpu()

    def at(self, ids):
        return {leg: s.data.force_matrix_w[ids, 0].norm(dim=-1).cpu() for leg, s in self.sensors.items()}

    def lines(self):
        out = []
        for leg in LEGS:
            for k, body in enumerate(self.names[leg]):
                if self.count[leg][k]:
                    own = "own leg" if body.split("_")[1] == leg else "other leg"
                    out.append(f"foot_{leg} on {body} ({own}): {int(self.count[leg][k])} env-steps "
                               f"in contact, max {self.max[leg][k].item():.3g} N")
        return out or ["no foot touched any shank or thigh"]


def build_population(env, agent_cfg, out_dir):
    """Give a co-optimisation scene its generation zero population, as CoptOnPolicyRunner.learn does."""
    from co_optimisation.runners import CoptOnPolicyRunner
    from co_optimisation.runners.usd_generator import GrowingDesignDistCMAESDesignGenerator
    from co_optimisation.utils.respawn import respawn_robots

    family = "quadruped" if "Isaac-Quadruped-Copt" in args_cli.task else "biped"
    n = args_cli.copt_num_individuals
    ea_update_interval, ea_late_start = 400, 6000
    # The generator and the copt block of scripts/rsl_rl/train.py, design_search "length".
    generator = GrowingDesignDistCMAESDesignGenerator(
        family=family,
        num_individuals=n,
        param_ranges={
            "thigh_length_scale": ("scale", (0.75, 1.25)),
            "shank_length_scale": ("scale", (0.75, 1.25)),
        },
        output_dir=os.path.join(out_dir, "copt_usds"),
        sigma0=0.25,
        seed=42,
        late_start=True,
        late_start_it=int(ea_late_start / ea_update_interval),
        late_start_prior_pop_size=env.num_envs // 64,
        max_cma_iter=(agent_cfg.max_iterations - ea_late_start) / ea_update_interval,
    )
    agent_cfg.policy.class_name = {"COPT": "CoptActorCritic", "COPT-MORAL": "CoptMorALActorCritic"}.get(
        args_cli.policy_type, "CoptMorALActorCritic"
    )
    agent_cfg.algorithm.class_name = "CoptPPO"
    cfg = agent_cfg.to_dict()
    cfg["algorithm"]["estimator_loss_coef"] = 1.0
    cfg["copt"] = {
        "ea_update_interval": ea_update_interval,
        "ea_late_start": ea_late_start,
        "num_individuals": n,
        "randomise_before_late_start": True,
        "reinit_lr_each_generation": True,
        "eval_duration_s": 4.0,
        "eval_seed": 42,
    }
    runner = CoptOnPolicyRunner(env, generator, copy.deepcopy(cfg), log_dir=None, device=agent_cfg.device)
    runner.respawn_robots = respawn_robots()
    with torch.inference_mode():
        runner._reload_morphology()
    return n


def snapshot(env, robot, ids, dones, contact_ids, num_designs):
    """The state of the non-finite environments ids at the step they were caught."""
    unwrapped = env.unwrapped
    terrain = unwrapped.scene.terrain
    lin = robot.data.body_lin_vel_w[ids]
    record = {
        "env_ids": ids.cpu(),
        "design": (ids % num_designs).cpu() if num_designs else None,
        "just_reset": dones[ids].bool().cpu(),
        "episode_length": unwrapped.episode_length_buf[ids].cpu(),
        "bad_bodies": (~torch.isfinite(lin)).any(-1).cpu(),
        "body_lin_vel_w": lin.cpu(),
        "body_ang_vel_w": robot.data.body_ang_vel_w[ids].cpu(),
        "body_pos_w": robot.data.body_pos_w[ids].cpu(),
        "root_state_w": robot.data.root_state_w[ids].cpu(),
        "joint_pos": robot.data.joint_pos[ids].cpu(),
        "joint_vel": robot.data.joint_vel[ids].cpu(),
        "applied_torque": robot.data.applied_torque[ids].cpu(),
    }
    if terrain.terrain_origins is not None:
        record["terrain_level"] = terrain.terrain_levels[ids].cpu()
        record["terrain_type"] = terrain.terrain_types[ids].cpu()
    if contact_ids is not None:
        sensor = unwrapped.scene.sensors["contact_forces"]
        record["foot_contact_force_w"] = sensor.data.net_forces_w[ids][:, contact_ids].cpu()
    # A second read of the same PhysX buffer, then one after the forward kinematics
    # that sim.forward runs, neither advancing the simulation.
    fresh = robot.root_physx_view.get_link_velocities()[ids]
    record["fresh_read_finite"] = torch.isfinite(fresh).all(-1).all(-1).cpu()
    unwrapped.sim.physics_sim_view.update_articulations_kinematic()
    after = robot.root_physx_view.get_link_velocities()[ids]
    record["after_kinematic_finite"] = torch.isfinite(after).all(-1).all(-1).cpu()
    return record


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    os.makedirs(args_cli.out, exist_ok=True)
    if args_cli.pair_contacts:
        add_pair_sensors(env_cfg)
    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg))

    num_designs = build_population(env, agent_cfg, args_cli.out) if "Copt" in args_cli.task else 0
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    feet, feet_names = robot.find_bodies("foot_.*_Link")
    sensor = unwrapped.scene.sensors.get("contact_forces")
    contact_ids = sensor.find_bodies("foot_.*_Link")[0] if sensor is not None else None
    pairs = PairContacts(unwrapped.scene) if args_cli.pair_contacts else None
    if pairs is not None:
        print(f"[DIAG] pair contact filters {pairs.names}", flush=True)
    print(f"[DIAG] {args_cli.task}, {env.num_envs} envs, {num_designs or 1} designs, "
          f"{robot.num_bodies} bodies, feet {feet_names}", flush=True)

    events, pending = [], None
    resets = nan_steps = nan_envs = nan_at_reset = 0
    nan_by_body = torch.zeros(robot.num_bodies, dtype=torch.long)
    obs_nan_groups: dict[str, int] = {}
    start = time.time()
    with torch.inference_mode():
        obs = env.get_observations()
        for step in range(args_cli.steps):
            actions = args_cli.action_std * torch.randn(env.num_envs, env.num_actions, device=env.device)
            obs, _, dones, _ = env.step(actions)
            resets += int(dones.sum())
            if pairs is not None:
                pairs.update()
            if pending is not None:
                # The environments caught on the previous step, read after one physics step.
                pending["next_step_finite"] = torch.isfinite(
                    robot.data.body_lin_vel_w[pending["env_ids"].to(env.device)]
                ).all(-1).all(-1).cpu()
                pending = None

            bad = ~torch.isfinite(robot.data.body_lin_vel_w).all(-1)  # (num_envs, num_bodies)
            bad_env = bad.any(-1)
            if bad_env.any():
                ids = bad_env.nonzero().squeeze(-1)
                nan_steps += 1
                nan_envs += len(ids)
                nan_at_reset += int(dones[ids].sum())
                nan_by_body += bad[ids].sum(0).cpu()
                for group in obs.keys():
                    if not torch.isfinite(obs[group]).all():
                        obs_nan_groups[group] = obs_nan_groups.get(group, 0) + 1
                if len(events) < args_cli.max_events:
                    record = snapshot(env, robot, ids, dones, contact_ids, num_designs)
                    record["step"] = step
                    if pairs is not None:
                        record["pair_contact_force"] = pairs.at(ids)
                    events.append(record)
                    pending = record
                    names = [robot.body_names[b] for b in record["bad_bodies"].any(0).nonzero().squeeze(-1).tolist()]
                    print(f"[DIAG] step {step}: {len(ids)} env(s) {ids.tolist()[:8]} non-finite, "
                          f"just reset {record['just_reset'].tolist()[:8]}, "
                          f"episode length {record['episode_length'].tolist()[:8]}, bodies {names}, "
                          f"fresh read finite {record['fresh_read_finite'].tolist()[:8]}, "
                          f"after kinematic finite {record['after_kinematic_finite'].tolist()[:8]}",
                          flush=True)
            if step % 1000 == 0:
                print(f"[DIAG] step {step}, {resets} resets, {nan_envs} non-finite envs over "
                      f"{nan_steps} steps, {time.time() - start:.0f} s", flush=True)

    lines = [
        f"task {args_cli.task}",
        f"num_envs {env.num_envs}, designs {num_designs or 1}, steps {args_cli.steps}, action_std {args_cli.action_std}",
        f"resets {resets}",
        f"steps with a non-finite link velocity {nan_steps}",
        f"non-finite environment readings {nan_envs}, of which on the step of a reset {nan_at_reset}",
        "non-finite readings per body " + ", ".join(
            f"{robot.body_names[b]} {int(c)}" for b, c in enumerate(nan_by_body) if c
        ),
        f"observation groups non-finite on those steps {obs_nan_groups}",
        f"wall time {time.time() - start:.0f} s",
    ]
    if pairs is not None:
        lines += pairs.lines()
    if events and num_designs:
        designs = torch.cat([e["design"] for e in events])
        lines.append(f"designs of the recorded events {sorted(set(designs.tolist()))}")
    with open(os.path.join(args_cli.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    torch.save({"events": events, "body_names": robot.body_names}, os.path.join(args_cli.out, "events.pt"))
    print("[RESULT] " + "\n[RESULT] ".join(lines), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
