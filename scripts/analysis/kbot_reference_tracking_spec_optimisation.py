#!/usr/bin/env python3
"""CMA-ES search over KBot joint gains and torque limits against a reference joint trajectory.

Each parallel IsaacLab environment carries one CMA-ES candidate, so the population is evaluated in
a single synchronous rollout of the reference trajectory.  The candidate is a per actuator group
quadruple of stiffness, damping, effort limit and velocity limit, and the fitness is the joint
angle root mean square error accumulated over the rollout, plus a penalty for terminating on a
fall, and optionally minus the task reward at a weight the caller sets.

The velocity limit belongs in the search because it is not a bookkeeping quantity on this actuator
model.  DCMotor derates the available torque linearly in speed, actuator_pd.py _clip_effort, so the
limit sets the slope of the envelope rather than merely a cap, and at the configured 17.488 rad/s
the knee arrives at the reference's 5.215 rad/s peak with exactly its 84 Nm ceiling intact, which
is a crossover rather than a margin.

Note that the task reward already charges for falling through its own termination term, and that
RewardManager scales every term by the control step at reward_manager.py:150, which puts the per
step total on the order of unity.  Enabling the reward while leaving the fall penalty in place
therefore charges the fall twice.  The script reports the composition of the fitness at startup,
quoting the termination weight found in the loaded configuration, and leaves the choice to the
caller rather than resolving it silently.

All domain randomisation is disabled on the environment configuration object built here, which
leaves the shared task configuration untouched, so nothing this script does can alter the behaviour
of a training run that imports the same classes.

Usage:
    ./isaaclab.sh -p scripts/analysis/kbot_reference_tracking_spec_optimisation.py \
        --reference /ws/joint_data_0.73_0.05_0.5_transposed.csv \
        --popsize 64 --generations 120 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Optimise KBot joint gains and torque limits for reference trajectory tracking.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

# --- reference and task -------------------------------------------------------------------------
parser.add_argument(
    "--reference",
    type=str,
    default="./biped/joint_data_0.73_0.05_0.5_transposed.csv",
    help="Reference trajectory CSV. Column 0 is time in seconds, columns 1 to 12 are joint angles "
    "in radians in the order L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_foot_pitch, "
    "L_foot_roll, then the same six for the right leg. No header row, values may be quoted.",
)
parser.add_argument("--task", type=str, default="Isaac-Limx-Kscale-Blind-Flat-v0", help="Task id.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the environment and CMA-ES.")

# --- CMA-ES -------------------------------------------------------------------------------------
parser.add_argument(
    "--popsize",
    type=int,
    default=64,
    help="CMA-ES population size. One environment is spawned per individual, so this also sets "
    "the number of parallel environments.",
)
parser.add_argument("--generations", type=int, default=120, help="Number of CMA-ES generations.")
parser.add_argument(
    "--sigma0",
    type=float,
    default=0.25,
    help="Initial CMA-ES step size on the unit hypercube. pycma silently caps this at 0.333 of "
    "the box width, so values above that have no further effect.",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to a cma_state.pkl written by a previous run, to continue its search.",
)

# --- search space -------------------------------------------------------------------------------
parser.add_argument(
    "--gain-range",
    type=float,
    nargs=2,
    default=(0.25, 4.0),
    metavar=("LO", "HI"),
    help="Multiplicative bounds on stiffness and damping relative to the values in the live "
    "actuator configuration. Sampled log uniformly, so a geometrically symmetric pair such as "
    "0.25 4.0 places the configured value at the centre of the search box.",
)
parser.add_argument(
    "--effort-range",
    type=float,
    nargs=2,
    default=(0.25, 4.0),
    metavar=("LO", "HI"),
    help="Multiplicative bounds on the effort limit relative to the live configuration. The "
    "saturation effort is scaled by the same factor so the torque speed envelope stays coherent.",
)
parser.add_argument(
    "--velocity-range",
    type=float,
    nargs=2,
    default=(0.5, 2.0),
    metavar=("LO", "HI"),
    help="Multiplicative bounds on the motor velocity limit, the no load speed that sets the "
    "slope of the torque speed envelope. The simulation velocity limit is scaled by the same "
    "factor so the solver cap keeps its configured relation to it. The default is narrower than "
    "the gain and effort ranges because the informative region is narrow. At the configured "
    "17.488 rad/s the knee has exactly its 84 Nm ceiling available at the reference's 5.215 rad/s "
    "peak, so raising the limit buys nothing there, while 0.75 of it yields 72.3 Nm, 0.5 yields "
    "48.4 Nm and 0.25 yields nothing at all. Widen it if the optimum lands on a bound.",
)

# --- fitness ------------------------------------------------------------------------------------
parser.add_argument(
    "--fall-penalty",
    type=float,
    default=10.0,
    help="Flat cost added to the fitness of any individual that terminated on anything other than "
    "the episode time out. In the units of the RMSE term, which is radians.",
)
parser.add_argument(
    "--fall-time-weight",
    type=float,
    default=10.0,
    help="Additional cost multiplied by the fraction of the reference left unsimulated when the "
    "individual fell. Grades the penalty so an early fall is worse than a late one, which gives "
    "CMA-ES a gradient to climb out of the falling region instead of a flat plateau.",
)
parser.add_argument(
    "--fall-height",
    type=float,
    default=0.40,
    help="Base height above the environment origin, in metres, below which the individual is "
    "counted as fallen even if no termination term has fired. The task terminates on torso "
    "contact alone, which on this robot is a late signal.",
)
parser.add_argument(
    "--use-reward",
    action="store_true",
    default=False,
    help="Subtract the task reward from the fitness in addition to the RMSE and fall terms.",
)
parser.add_argument(
    "--reward-weight",
    type=float,
    default=1.0,
    help="Weight applied to the mean per step task reward before it is subtracted from the "
    "fitness. Set by hand and held fixed for the run. For reference, the per step total reward "
    "carries the manager's dt factor and so sits on the order of unity, roughly -1 to 1, while "
    "the RMSE term is on the order of 0.01 to 0.1 rad, so a weight near 0.05 puts the two at "
    "comparable magnitude and a weight of 1.0 lets the reward dominate. Choose it deliberately "
    "and keep it identical across the runs being compared, since a run whose weight differs is "
    "minimising a different objective.",
)

# --- rollout ------------------------------------------------------------------------------------
parser.add_argument(
    "--fixed-base",
    action="store_true",
    default=False,
    help="Pin the root link. Turns the problem into a pure servo tracking problem with no falls, "
    "which isolates the actuator specification from the balance of the gait. The fall penalty and "
    "the reward term are meaningless in this mode and are reported as zero.",
)
parser.add_argument(
    "--settle-steps",
    type=int,
    default=0,
    help="Control steps held at the first reference sample before the error accounting starts. "
    "The supplied reference already opens with about 2.5 s at the nominal pose, which serves the "
    "same purpose, so this defaults to off.",
)
parser.add_argument(
    "--trim-static-tail",
    action="store_true",
    default=False,
    help="Drop the trailing samples that are identical to the first, which on the supplied "
    "reference is about 3.5 s of dead time, to shorten each generation.",
)
parser.add_argument(
    "--command-vel-x", type=float, default=0.5, help="Pinned forward velocity command. Only "
    "affects the reward term. The supplied reference filename reads as 0.73, 0.05, 0.5, which is "
    "consistent with a base height, a swing height and a forward speed, so check this against the "
    "generator that produced the trajectory before trusting a reward weighted result.",
)
parser.add_argument("--command-vel-y", type=float, default=0.0, help="Pinned lateral velocity command.")
parser.add_argument("--command-ang-z", type=float, default=0.0, help="Pinned yaw rate command.")

# --- diagnostics --------------------------------------------------------------------------------
parser.add_argument(
    "--plot-envs",
    type=int,
    default=8,
    help="How many environments to write an individual reference against achieved figure for. "
    "-1 writes one per environment, 0 writes none, and the environment the video followed is "
    "always written whatever this is set to. The figures are rewritten every generation, so -1 "
    "at a population of 256 means 256 figures per generation, which costs more wall clock than "
    "the rollout it is documenting. The overlay carries every environment on one set of axes and "
    "is written either way, so no environment goes unplotted at the default.",
)
parser.add_argument(
    "--video",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record a video every generation, overwritten each time so one file remains at the end, "
    "showing the last generation. The camera follows the environment that held the previous "
    "generation's best candidate, and environment zero in the first generation, since a "
    "generation's winner is only known once it has been scored. Implies --enable_cameras, and "
    "costs a rendered frame every --video-stride steps of every generation.",
)
parser.add_argument(
    "--video-stride",
    type=int,
    default=2,
    help="Render every Nth control step. The control rate is 100 Hz, so the default of 2 yields "
    "a 50 fps video that plays at real time.",
)

# --- output -------------------------------------------------------------------------------------
parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Directory for the CMA-ES log, the state pickle and the per generation results. Defaults "
    "to logs/spec_opt/<timestamp> under the repository root.",
)
parser.add_argument("--num_envs", type=int, default=None, help=argparse.SUPPRESS)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# one environment per individual, which is what makes the population a single batched rollout
args_cli.num_envs = args_cli.popsize

# rendering a frame needs the camera pipeline up, exactly as scripts/rsl_rl/play.py:65 arranges it
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import json
import numpy as np
import os
import pickle
import torch
from datetime import datetime

try:
    import cma
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pycma is required. It is the same library co_optimisation/runners/usd_generator.py:70 "
        "uses for its design search. Install it with 'pip install cma'."
    ) from exc

from isaaclab.actuators import DCMotor
from isaaclab_tasks.utils import parse_env_cfg

import environments  # noqa: F401  registers the KScale tasks, as scripts/rsl_rl/play.py:98 does

# =================================================================================================
# Reference trajectory
# =================================================================================================

# CSV column order, as laid out in the supplied file, mapped onto the URDF joint names. The
# supplied file already carries the URDF's own sign convention, its first row reproducing the
# nominal pose of kscale_identified_cfg.py:197 on all twelve joints to within 1e-3 rad, including
# the anti parallel hip pitch axes and the mirrored roll signs, so no per joint sign flip applies.
#
# That agreement lapsed on 2026-09-23, when arm 5 of plans/kscale_gait_shaping.md moved the three
# sagittal nominals to the published K-Scale posture. The CSV is unchanged and still carries the
# pose supplied on 2026-09-17, so its first row now differs from the asset configuration by
# 0.14538 rad at the hip pitch, 0.36027 at the knee and 0.21490 at the ankle pitch. The sign
# claim above is unaffected, the conventions being what it asserts, but any figure this program
# reports that is read as a deviation from the nominal pose must be taken against the CSV's own
# first row and no longer against the asset configuration.
CSV_JOINT_ORDER = [
    "left_hip_pitch_04",
    "left_hip_roll_03",
    "left_hip_yaw_03",
    "left_knee_04",
    "left_foot_pitch_02",
    "left_foot_roll_02",
    "right_hip_pitch_04",
    "right_hip_roll_03",
    "right_hip_yaw_03",
    "right_knee_04",
    "right_foot_pitch_02",
    "right_foot_roll_02",
]


def load_reference(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read the reference CSV into a time vector and a (num_samples, 12) angle array.

    numpy.loadtxt cannot read this file because every field is double quoted, so the rows are
    parsed with the csv module and converted explicitly.
    """
    rows = []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            try:
                rows.append([float(v.strip().strip('"')) for v in row])
            except ValueError:
                # a header row, if one is ever added, simply fails to parse and is skipped
                continue
    data = np.asarray(rows, dtype=np.float64)
    if data.shape[1] != 1 + len(CSV_JOINT_ORDER):
        raise ValueError(
            f"expected {1 + len(CSV_JOINT_ORDER)} columns in {path}, found {data.shape[1]}"
        )
    return data[:, 0], data[:, 1:]


def resample_reference(
    t_src: np.ndarray, q_src: np.ndarray, dt_ctrl: float, trim_static_tail: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the reference onto the environment's control grid.

    The supplied reference is sampled at 1 kHz while the task runs its action manager at
    1 / (sim.dt * decimation), which for the KScale configuration is 100 Hz.  The action is held
    constant across the decimation loop, so the command the actuator sees is a zero order hold of
    this grid rather than the original signal.
    """
    if trim_static_tail:
        same = np.all(np.isclose(q_src, q_src[0], atol=1e-9), axis=1)
        moving = np.flatnonzero(~same)
        if moving.size:
            # keep a quarter second of settle beyond the last moving sample
            last = min(q_src.shape[0] - 1, moving[-1] + int(0.25 / (t_src[1] - t_src[0])))
            t_src, q_src = t_src[: last + 1], q_src[: last + 1]

    duration = float(t_src[-1] - t_src[0])
    num = int(np.floor(duration / dt_ctrl)) + 1
    t_ctrl = t_src[0] + dt_ctrl * np.arange(num)
    q_ctrl = np.stack([np.interp(t_ctrl, t_src, q_src[:, j]) for j in range(q_src.shape[1])], axis=1)
    return t_ctrl, q_ctrl


# =================================================================================================
# Environment construction
# =================================================================================================


def disable_randomisation(env_cfg) -> None:
    """Null every stochastic term on the configuration object this script owns.

    The startup events that scale masses, physics materials, actuator gains, joint offsets, joint
    friction and armature are the ones that would otherwise give each environment a different
    plant, which would make the population's fitnesses incomparable.  The reset events are kept but
    have their ranges collapsed to zero so that every reset returns the robot to exactly the
    configured initial state, and the interval push is removed outright.

    Mutating the instance rather than the class leaves the shared configuration at
    tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:475 untouched for every other caller.
    """
    ev = env_cfg.events

    # startup randomisation, removed entirely
    for term in (
        "add_base_mass",
        "add_link_mass",
        "robot_physics_material",
        "hip_joint_stiffness_and_damping",
        "knee_joint_stiffness_and_damping",
        "ankle_joint_stiffness_and_damping",
        "joint_offsets",
        "joint_friction",
        "scale_all_joint_armature",
    ):
        if hasattr(ev, term):
            setattr(ev, term, None)

    # interval disturbance, removed entirely
    if hasattr(ev, "push_robot"):
        ev.push_robot = None

    # reset events, kept so the reset is well defined, but with their ranges collapsed
    if getattr(ev, "reset_robot_base", None) is not None:
        ev.reset_robot_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        ev.reset_robot_base.params["velocity_range"] = {
            k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        }
    for name in dir(ev):
        if not name.startswith("reset_") or name == "reset_robot_base":
            continue
        term = getattr(ev, name)
        if term is None or not hasattr(term, "params"):
            continue
        if "position_range" in term.params:
            term.params["position_range"] = (0.0, 0.0)
        if "velocity_range" in term.params and isinstance(term.params["velocity_range"], tuple):
            term.params["velocity_range"] = (0.0, 0.0)

    # curricula, which move the terrain level and the push magnitude with the iteration count
    if env_cfg.curriculum is not None:
        for name in list(vars(env_cfg.curriculum).keys()):
            if not name.startswith("__"):
                setattr(env_cfg.curriculum, name, None)

    # observation noise, which does not enter the fitness but does enter the dumped observations
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False


def pin_commands(env_cfg, vx: float, vy: float, wz: float) -> None:
    """Collapse the velocity command distribution to a single commanded vector.

    Only the reward term reads the command, but it reads it every step, so a resampling command
    would make the reward weighted fitness a different objective for each individual.
    """
    bv = env_cfg.commands.base_velocity
    bv.heading_command = False
    bv.rel_standing_envs = 0.0
    bv.rel_heading_envs = 0.0
    bv.resampling_time_range = (1.0e6, 1.0e6)
    bv.ranges.lin_vel_x = (vx, vx)
    bv.ranges.lin_vel_y = (vy, vy)
    bv.ranges.ang_vel_z = (wz, wz)
    if hasattr(bv.ranges, "heading"):
        bv.ranges.heading = (0.0, 0.0)


def build_env(args):
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.popsize)
    env_cfg.seed = args.seed

    disable_randomisation(env_cfg)
    pin_commands(env_cfg, args.command_vel_x, args.command_vel_y, args.command_ang_z)

    if args.fixed_base:
        # ArticulationRootPropertiesCfg.fix_root_link, schemas_cfg.py:43
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    # flat ground only, the terrain generator would otherwise place each environment on a
    # different patch and the comparison across the population would confound terrain with gains
    if getattr(env_cfg.scene, "terrain", None) is not None:
        if getattr(env_cfg.scene.terrain, "terrain_type", None) == "generator":
            env_cfg.scene.terrain.terrain_type = "plane"
            env_cfg.scene.terrain.terrain_generator = None

    return env_cfg


# =================================================================================================
# Search space
# =================================================================================================

PARAM_KINDS = ("stiffness", "damping", "effort_limit", "velocity_limit")


class ActuatorSearchSpace:
    """Log uniform encoding of the per actuator group stiffness, damping, effort and velocity limit.

    The baseline is read from the live actuator objects rather than hard coded, so the search box
    follows the working tree's configuration as it moves.  The encoding is logarithmic because the
    quantities are multiplicative, and because a geometrically symmetric multiplier pair then puts
    the configured value at exactly 0.5 in the unit hypercube, which is where
    usd_generator.py:737 starts its mean.

    Two dependent quantities are carried along rather than searched independently, each holding the
    ratio it has in the loaded configuration.  The saturation effort follows the effort limit,
    since a search that raised the effort limit past a fixed saturation effort would find the
    torque speed envelope clipping at the saturation effort instead and would be silently bounded
    below its own upper bound.  The simulation velocity limit follows the motor velocity limit for
    the same reason, the solver cap being the quantity that actually stops the joint.
    """

    def __init__(self, robot, gain_range, effort_range, velocity_range):
        self.groups = list(robot.actuators.keys())
        self.ranges = {
            "stiffness": tuple(float(v) for v in gain_range),
            "damping": tuple(float(v) for v in gain_range),
            "effort_limit": tuple(float(v) for v in effort_range),
            "velocity_limit": tuple(float(v) for v in velocity_range),
        }
        self.baseline: dict[str, dict[str, float]] = {}
        self.saturation_ratio: dict[str, float] = {}
        self.velocity_sim_ratio: dict[str, float] = {}

        for name, act in robot.actuators.items():
            base = {
                "stiffness": float(act.stiffness[0].mean().item()),
                "damping": float(act.damping[0].mean().item()),
                "effort_limit": float(act.effort_limit[0].mean().item()),
                "velocity_limit": float(act.velocity_limit[0].mean().item()),
            }
            self.baseline[name] = base

            sat = getattr(act, "_saturation_effort", None)
            if sat is None:
                ratio = 1.0
            elif torch.is_tensor(sat):
                ratio = float(sat.mean().item()) / max(base["effort_limit"], 1e-9)
            else:
                ratio = float(sat) / max(base["effort_limit"], 1e-9)
            self.saturation_ratio[name] = ratio

            vsim = getattr(act, "velocity_limit_sim", None)
            if vsim is None:
                self.velocity_sim_ratio[name] = 1.0
            else:
                vsim_f = float(vsim.mean().item()) if torch.is_tensor(vsim) else float(vsim)
                self.velocity_sim_ratio[name] = vsim_f / max(base["velocity_limit"], 1e-9)

        self.keys = [(g, k) for g in self.groups for k in PARAM_KINDS]
        self.num_params = len(self.keys)

    def decode(self, x: np.ndarray) -> dict[str, dict[str, float]]:
        """Map one unit hypercube vector to a per group parameter dictionary."""
        out = {g: {} for g in self.groups}
        for i, (g, kind) in enumerate(self.keys):
            lo_m, hi_m = self.ranges[kind]
            xi = float(np.clip(x[i], 0.0, 1.0))
            out[g][kind] = self.baseline[g][kind] * lo_m * (hi_m / lo_m) ** xi
        for g in self.groups:
            out[g]["saturation_effort"] = out[g]["effort_limit"] * self.saturation_ratio[g]
            out[g]["velocity_limit_sim"] = out[g]["velocity_limit"] * self.velocity_sim_ratio[g]
        return out

    def describe(self) -> str:
        lines = [
            f"search space, {self.num_params} dimensions, log uniform on the multiplier,"
            f" {len(PARAM_KINDS)} parameters across {len(self.groups)} actuator groups"
        ]
        for g in self.groups:
            b = self.baseline[g]
            rk, re, rv = (self.ranges["stiffness"], self.ranges["effort_limit"],
                          self.ranges["velocity_limit"])
            lines.append(
                f"  {g:<11} K {b['stiffness']:8.2f} in [{b['stiffness']*rk[0]:8.2f},"
                f" {b['stiffness']*rk[1]:8.2f}]   D {b['damping']:7.3f} in"
                f" [{b['damping']*rk[0]:7.3f}, {b['damping']*rk[1]:7.3f}]   tau"
                f" {b['effort_limit']:7.2f} in [{b['effort_limit']*re[0]:7.2f},"
                f" {b['effort_limit']*re[1]:7.2f}]   vel {b['velocity_limit']:6.2f} in"
                f" [{b['velocity_limit']*rv[0]:6.2f}, {b['velocity_limit']*rv[1]:6.2f}]"
                f"   sat/tau {self.saturation_ratio[g]:.3f}"
                f"   velsim/vel {self.velocity_sim_ratio[g]:.3f}"
            )
        return "\n".join(lines)


def apply_population(robot, decoded: list[dict], device) -> None:
    """Write one decoded candidate into each environment's slice of the actuator tensors.

    Every actuator parameter is a (num_envs, num_joints) tensor, actuator_base.py:184, and the
    explicit PD law reads self.stiffness and self.damping straight out of those tensors at
    IsaacLab actuator_pd.py IdealPDActuator.compute, so writing them here is sufficient and no
    call into the simulator is needed.  Indeed articulation.py:652 records that
    write_joint_stiffness_to_sim does not reach the actuator models at all, and
    articulation.py:1768 has already zeroed the PhysX drive gains for every explicit actuator, so
    the Python side is the only side that matters.

    The two derived quantities DCMotor caches at construction have to be repaired by hand.
    _saturation_effort is stored as a scalar and is promoted to a tensor so it can vary per
    environment, and _vel_at_effort_lim is recomputed, since a stale value would clamp the velocity
    used in the torque speed curve against the old ceiling.  That recomputation reads both the
    effort limit and the velocity limit, so it has to come after both are written, which is why the
    ordering below is not incidental.

    The effort limit is not written to the simulator.  IdentifiedActuatorCfg leaves
    effort_limit_sim unset, so actuator_base.py:175 gives it _DEFAULT_MAX_EFFORT_SIM of 1e9 and
    PhysX imposes no ceiling of its own on an explicit actuator.  The velocity limit is different.
    articulation.py:1773 does push velocity_limit_sim into the solver at startup, where it becomes
    a hard cap that brakes the joint, so a candidate's velocity limit only takes effect once
    write_joint_velocity_limit_to_sim has carried it across, articulation.py:770.
    """
    for name, act in robot.actuators.items():
        col = lambda kind: torch.tensor(  # noqa: E731
            [d[name][kind] for d in decoded], dtype=torch.float32, device=device
        ).unsqueeze(1)

        act.stiffness[:] = col("stiffness")
        act.damping[:] = col("damping")
        act.effort_limit[:] = col("effort_limit")
        act.velocity_limit[:] = col("velocity_limit")

        # the solver side cap, which is a separate quantity from the motor's no load speed
        vel_sim = col("velocity_limit_sim").expand_as(act.velocity_limit).clone()
        if torch.is_tensor(getattr(act, "velocity_limit_sim", None)):
            act.velocity_limit_sim[:] = vel_sim
        else:
            act.velocity_limit_sim = vel_sim
        robot.write_joint_velocity_limit_to_sim(vel_sim, joint_ids=act.joint_indices)

        if isinstance(act, DCMotor):
            sat = col("saturation_effort").expand_as(act.effort_limit).clone()
            act._saturation_effort = sat
            act._vel_at_effort_lim = act.velocity_limit * (1.0 + act.effort_limit / sat)


# =================================================================================================
# Rollout and fitness
# =================================================================================================


def rollout(
    env,
    q_ref_t: torch.Tensor,
    args,
    record_buf: torch.Tensor | None = None,
    sink=None,
) -> dict[str, np.ndarray]:
    """Drive the reference through the action manager and return the raw fitness components.

    The action term is JointPositionActionCfg with use_default_offset, so the joint target it
    issues is raw * scale + default_joint_pos, kscale_base_env_cfg.py:191.  Inverting that gives
    the raw action that commands a chosen absolute angle, which is what the reference supplies.
    Since 2026-09-22 that scale is a per joint dictionary rather than a single float, so term._scale
    is a (num_envs, action_dim) tensor rather than a scalar.  The inversion below divides by it
    elementwise and is correct in either case, the tensor broadcasting against the target of the
    same shape, so nothing here needed changing when the configuration moved.

    Accounting is frozen per environment at the first step on which it terminates or drops below
    the fall height, because ManagerBasedRLEnv resets a terminated environment inside step and
    everything it reports afterwards belongs to a fresh episode rather than to this candidate.

    Args:
        record_buf: optional (num_steps, num_envs, num_joints) buffer that receives the achieved
            joint positions.  Passed in preallocated and overwritten in place each generation, so
            that when the search exits the buffer holds the last completed generation.
        sink: optional FrameSink that receives a rendered frame every video_stride steps.  One
            sink covers one generation and is closed by the caller.  Rendering needs the
            environment to have been created with an rgb_array render mode, which is the caller's
            responsibility.
    """
    device = env.device
    num_envs = env.num_envs
    num_steps = q_ref_t.shape[0]

    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    joint_ids = term._joint_ids

    # The action term stores its scale and its offset either as a scalar or as a per environment
    # (num_envs, action_dim) tensor, joint_actions.py:85-100, so the inversion has to be done
    # against the (num_envs, action_dim) shape rather than against the reference's own
    # (num_steps, action_dim).  Doing it one step at a time costs nothing and stays correct even
    # if a future configuration gives each environment a different default pose.
    scale = term._scale
    offset = term._offset

    def to_raw(q_target: torch.Tensor) -> torch.Tensor:
        """Map a (action_dim,) absolute joint target to the (num_envs, action_dim) raw action."""
        target = q_target.unsqueeze(0).expand(num_envs, -1)
        return (target - offset) / scale

    env.reset()

    zeros = torch.zeros(num_envs, device=device)
    active = torch.ones(num_envs, dtype=torch.bool, device=device)
    sq_sum = torch.zeros(num_envs, device=device)
    rew_sum = torch.zeros(num_envs, device=device)
    n_acc = torch.zeros(num_envs, device=device)
    fell = torch.zeros(num_envs, dtype=torch.bool, device=device)

    origin_z = env.scene.env_origins[:, 2]

    for _ in range(args.settle_steps):
        env.step(to_raw(q_ref_t[0]))

    for k in range(num_steps):
        _, rew, terminated, _, _ = env.step(to_raw(q_ref_t[k]))

        achieved = robot.data.joint_pos[:, joint_ids]
        if record_buf is not None:
            record_buf[k] = achieved

        if sink is not None and k % max(1, args.video_stride) == 0:
            rgb = env.render()
            if rgb is not None:
                sink.add(rgb)

        err = achieved - q_ref_t[k]
        sq = (err * err).mean(dim=-1)

        sq_sum += torch.where(active, sq, zeros)
        rew_sum += torch.where(active, rew.to(device).float(), zeros)
        n_acc += active.float()

        if not args.fixed_base:
            below = (robot.data.root_pos_w[:, 2] - origin_z) < args.fall_height
            down = terminated.to(device).bool() | below
            fell |= active & down
            active &= ~down
            if not bool(active.any()):
                break

    n_safe = n_acc.clamp(min=1.0)
    rmse = torch.sqrt(sq_sum / n_safe)
    mean_rew = rew_sum / n_safe
    frac_missing = 1.0 - n_acc / float(num_steps)

    return {
        "rmse": rmse.detach().cpu().numpy(),
        "mean_reward": mean_rew.detach().cpu().numpy(),
        "fell": fell.detach().cpu().numpy(),
        "frac_missing": frac_missing.detach().cpu().numpy(),
        "steps": n_acc.detach().cpu().numpy(),
    }


# =================================================================================================
# Diagnostics, written once per generation and overwritten so only the last generation survives
# =================================================================================================


def _import_pyplot():
    """Return matplotlib.pyplot, or None with one explanatory line if it is unavailable.

    Plotting is a convenience, not a result, so a container without matplotlib must still complete
    the search.  The underlying arrays are written to tracking.npz either way, which is the
    artefact a later session would actually reason about.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def write_tracking_plots(
    out_dir, t_ctrl, q_ref, achieved, steps, rmse, joint_names, args, always_include=None
) -> None:
    """Write the reference against achieved figures for the generation just evaluated.

    Two artefacts are produced.  The overlay carries every environment on one set of axes against
    the reference, which is where the spread of the population is read.  The per environment
    figures carry one candidate each, which is where a particular candidate's failure is read.
    Both are overwritten every generation, so what remains at the end belongs to the last one.

    Each environment's curve is truncated at the step it stopped being accounted for, so a fallen
    candidate ends where it fell rather than appearing to snap back to the nominal pose, which is
    what the buffer holds after ManagerBasedRLEnv resets it.
    """
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # uncompressed, because this is rewritten every generation and noisy float32 trajectories
    # compress by only a few per cent, which is not worth seconds of CPU per generation
    np.savez(
        os.path.join(plot_dir, "tracking.npz"),
        time=np.asarray(t_ctrl, dtype=np.float32),
        reference=np.asarray(q_ref, dtype=np.float32),
        achieved=np.asarray(achieved, dtype=np.float32),
        steps=steps,
        rmse=rmse,
        joint_names=np.array(joint_names),
    )

    plt = _import_pyplot()
    if plt is None:
        print("[spec-opt] matplotlib unavailable, wrote plots/tracking.npz only")
        return

    num_envs = achieved.shape[1]
    nj = len(joint_names)
    rows, cols = int(np.ceil(nj / 3)), 3

    def _panels(ax_iter, env_list, alpha, lw):
        for j, ax in zip(range(nj), ax_iter):
            for e in env_list:
                n = max(int(steps[e]), 1)
                ax.plot(t_ctrl[:n], achieved[:n, e, j], lw=lw, alpha=alpha, color="tab:blue")
            ax.plot(t_ctrl, q_ref[:, j], lw=1.4, color="k", ls="--", label="reference")
            ax.set_title(joint_names[j], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)

    # --- overlay across the whole population ----------------------------------------------------
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.3 * rows), sharex=True)
    flat = axes.ravel()
    _panels(flat, range(num_envs), alpha=min(0.9, 4.0 / max(num_envs, 1)), lw=0.6)
    for ax in flat[nj:]:
        ax.axis("off")
    fig.suptitle(
        f"reference against achieved, all {num_envs} candidates, "
        f"rmse min {np.nanmin(rmse):.4f} median {np.nanmedian(rmse):.4f} rad",
        fontsize=10,
    )
    fig.supxlabel("time [s]", fontsize=9)
    fig.supylabel("joint angle [rad]", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "tracking_all_envs.png"), dpi=110)
    plt.close(fig)

    # --- one figure per environment -------------------------------------------------------------
    n_indiv = num_envs if args.plot_envs < 0 else min(args.plot_envs, num_envs)
    wanted = list(range(n_indiv))
    if always_include is not None and always_include not in wanted:
        # the environment the video followed always gets a figure, however small --plot-envs is
        wanted.append(int(always_include))
    for e in wanted:
        fig, axes = plt.subplots(rows, cols, figsize=(13, 2.3 * rows), sharex=True)
        flat = axes.ravel()
        _panels(flat, [e], alpha=0.95, lw=1.2)
        for ax in flat[nj:]:
            ax.axis("off")
        n = int(steps[e])
        tag = "complete" if n >= len(t_ctrl) else f"stopped at step {n} of {len(t_ctrl)}"
        fig.suptitle(f"env {e:04d}, rmse {rmse[e]:.5f} rad, {tag}", fontsize=10)
        fig.supxlabel("time [s]", fontsize=9)
        fig.supylabel("joint angle [rad]", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"tracking_env{e:04d}.png"), dpi=100)
        plt.close(fig)

    print(
        f"[spec-opt] wrote plots/tracking_all_envs.png, {len(wanted)} per environment figures"
        f" and plots/tracking.npz"
    )


class FrameSink:
    """Stream rendered frames straight to the encoder instead of accumulating them.

    A generation renders about 750 frames at the viewport's resolution, which as a list of arrays
    is on the order of two gigabytes of host memory held for the duration of the rollout.  Sending
    each frame to the encoder as it arrives keeps the footprint at one frame, and costs nothing
    else, the encoder running in its own process.

    The encoder is opened lazily on the first frame, because the frame shape is not known until
    one has been rendered, and it must be even in both dimensions for most codecs, so an odd
    viewport is cropped by a row or a column rather than rejected.

    Failures here are reported and swallowed.  A video is a convenience and must never take the
    search down with it.
    """

    def __init__(self, path: str, fps: int):
        self.path = path
        self.fps = int(max(1, fps))
        self._writer = None
        self._shape = None
        self.count = 0
        self.failed = False

    def add(self, rgb) -> None:
        if self.failed:
            return
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return
        frame = np.ascontiguousarray(frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2, :3])
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if self._writer is None:
            try:
                import imageio.v2 as imageio

                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                # the .mp4 extension selects the ffmpeg plugin, whose binary ships inside
                # imageio_ffmpeg, this workspace having no ffmpeg on the system path
                self._writer = imageio.get_writer(self.path, fps=self.fps, macro_block_size=1)
                self._shape = frame.shape
            except Exception as exc:  # noqa: BLE001
                print(f"[spec-opt] could not open the video encoder, {exc}")
                self.failed = True
                return

        if frame.shape != self._shape:
            return
        try:
            self._writer.append_data(frame)
            self.count += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[spec-opt] video encoder failed at frame {self.count}, {exc}")
            self.failed = True

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
            self._writer = None


def finalise_video(out_dir, sink, env_index, gen, cost) -> None:
    """Close the encoder and record which candidate the video shows.

    The file name is fixed rather than stamped with the generation, so the run leaves exactly one
    video behind, the last generation's.  Which candidate it followed goes in the sidecar.
    """
    sink.close()
    if sink.count == 0:
        print("[spec-opt] no frames captured, no video written")
        return

    video_dir = os.path.join(out_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    with open(os.path.join(video_dir, "reference_tracking.json"), "w") as fh:
        json.dump(
            {
                "generation": gen,
                "env_index": env_index,
                "cost_of_that_env_this_generation": cost,
                "fps": sink.fps,
                "frames": sink.count,
                "note": "the camera follows the environment that held the best candidate of the "
                "previous generation, environment zero in the first generation, so this index "
                "carries a freshly sampled candidate rather than the winner it was chosen for",
            },
            fh,
            indent=2,
        )
    print(f"[spec-opt] wrote {sink.path}, {sink.count} frames at {sink.fps} fps")


def report_fitness_composition(env, dt_ctrl: float, args) -> None:
    """State the fitness the run is about to minimise, and flag the double counted fall.

    The task's own reward already charges for falling.  RewardManager scales every term by the
    control step, so a termination term of weight w contributes w * dt to the accumulated reward
    on the single step it fires, and w * dt / n to the mean over the n steps an individual
    survived.  Enabling the reward while leaving the script's fall penalty in place therefore
    charges the same event twice, once through the task's weight and once through --fall-penalty,
    which is not wrong so much as it is a choice that has to be made deliberately, since the two
    are on quite different scales and their ratio changes with how early the fall happens.
    """
    print(f"[spec-opt] fitness = rmse{'  -  reward_weight * mean_reward' if args.use_reward else ''}"
          f"{'' if args.fixed_base else '  +  fall terms'}")

    if args.fixed_base:
        print("[spec-opt]   fixed base, the fall terms and the reward term are both inactive")
        return

    print(
        f"[spec-opt]   fall terms, flat {args.fall_penalty:g} plus"
        f" {args.fall_time_weight:g} times the fraction of the reference left unsimulated"
    )

    if not args.use_reward:
        print("[spec-opt]   reward term disabled, pass --use-reward to enable it")
        return

    print(f"[spec-opt]   reward weight {args.reward_weight:g}, set by hand and held fixed")

    # read the task's own termination weight rather than assuming one, so the figure quoted
    # belongs to the configuration actually loaded
    term_weight = None
    for name in env.reward_manager.active_terms:
        if "termination" in name:
            try:
                term_weight = float(env.reward_manager.get_term_cfg(name).weight)
            except (ValueError, AttributeError):
                continue
            per_step = term_weight * dt_ctrl
            print(
                f"[spec-opt]   WARNING the reward already carries '{name}' at weight"
                f" {term_weight:g}, which is {per_step:+.4g} of accumulated reward on the step it"
                f" fires. Weighted at {args.reward_weight:g} that is"
                f" {-args.reward_weight * per_step:+.4g} of cost spread over the steps survived,"
                f" for instance {-args.reward_weight * per_step / 100.0:+.4g} for an individual"
                f" that fell at step 100 and"
                f" {-args.reward_weight * per_step / 1000.0:+.4g} for one that fell at step 1000."
            )
            break

    if term_weight is not None and (args.fall_penalty or args.fall_time_weight):
        print(
            "[spec-opt]   the fall is therefore charged twice, once by the task's own termination"
            " weight and once by the script's fall terms. Set --fall-penalty 0 --fall-time-weight"
            " 0 to leave the charge to the reward alone, or keep both deliberately and hold them"
            " identical across the runs being compared."
        )


def combine_fitness(parts: dict[str, np.ndarray], args) -> np.ndarray:
    """Assemble the scalar cost CMA-ES minimises from the raw rollout components.

    The reward weight is taken as given rather than calibrated from the population.  Calibrating
    it would be circular, because the quantity it would be calibrated against, the mean per step
    reward, is itself dominated by the task's own termination penalty whenever a part of the
    population falls, so the weight would track the fall rate of the first generation rather than
    the scale of the ongoing behaviour.  Holding it fixed also keeps the objective stationary,
    which CMA-ES's covariance adaptation assumes.
    """
    cost = parts["rmse"].astype(np.float64).copy()

    fell = parts["fell"].astype(bool)
    cost += fell * (args.fall_penalty + args.fall_time_weight * parts["frac_missing"])

    if args.use_reward and not args.fixed_base:
        cost -= args.reward_weight * parts["mean_reward"]

    return np.where(np.isfinite(cost), cost, 1.0e6)


# =================================================================================================
# Main
# =================================================================================================


def main():
    args = args_cli

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "logs",
        "spec_opt",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[spec-opt] output directory {out_dir}")

    # --- reference ------------------------------------------------------------------------------
    t_src, q_src = load_reference(args.reference)
    env_cfg = build_env(args)
    dt_ctrl = env_cfg.sim.dt * env_cfg.decimation
    t_ctrl, q_ctrl = resample_reference(t_src, q_src, dt_ctrl, args.trim_static_tail)
    duration = float(t_ctrl[-1] - t_ctrl[0])
    env_cfg.episode_length_s = duration + dt_ctrl * (args.settle_steps + 10)

    print(
        f"[spec-opt] reference {args.reference}\n"
        f"           {q_src.shape[0]} samples at {1.0/(t_src[1]-t_src[0]):.0f} Hz over"
        f" {t_src[-1]-t_src[0]:.3f} s, resampled to {q_ctrl.shape[0]} control steps at"
        f" {1.0/dt_ctrl:.0f} Hz\n"
        f"           episode length set to {env_cfg.episode_length_s:.3f} s so the time out never"
        f" fires inside the rollout"
    )

    # --- environment ----------------------------------------------------------------------------
    # The render mode has to be requested at construction, so it is decided by --video here rather
    # than at the point the frames are wanted.  Frames are pulled with env.render() directly rather
    # than through gym.wrappers.RecordVideo as scripts/rsl_rl/play.py:779 does, because the wrapper
    # triggers off its own step counter and this script needs a fresh video every generation.
    env = gym.make(
        args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None
    ).unwrapped
    robot = env.scene["robot"]

    # reorder the reference into the action term's own joint order, which need not match either
    # the CSV's per leg order or the articulation's order
    term = env.action_manager.get_term("joint_pos")
    csv_index = {n: i for i, n in enumerate(CSV_JOINT_ORDER)}
    missing = [n for n in term._joint_names if n not in csv_index]
    if missing:
        raise ValueError(f"the reference has no column for {missing}")
    perm = [csv_index[n] for n in term._joint_names]
    q_ref_t = torch.tensor(q_ctrl[:, perm], dtype=torch.float32, device=env.device)
    print(f"[spec-opt] action term joint order {term._joint_names}")

    # Report where the reference asks for a pose the joint cannot reach, and separately where it
    # merely leaves the soft band.  The two are not the same constraint and only the first is a
    # tracking obstacle.  The hard limits are the ones the solver enforces, written to PhysX by
    # articulation.py:710 write_joint_position_limit_to_sim, so a reference beyond them cannot be
    # tracked at any gain.  The soft limits are a derived buffer computed at articulation.py:767
    # from soft_joint_pos_limit_factor, kscale_identified_cfg.py:241, and are never written to the
    # solver, so an excursion past them is physically reachable and costs only a penalty in the
    # reward terms that read them.
    hard = robot.data.joint_pos_limits[0, term._joint_ids].detach().cpu().numpy()
    soft = robot.data.soft_joint_pos_limits[0, term._joint_ids].detach().cpu().numpy()
    ref_np = q_ctrl[:, perm]
    for j, name in enumerate(term._joint_names):
        under, over = float(ref_np[:, j].min()), float(ref_np[:, j].max())
        h_lo, h_hi = float(hard[j, 0]), float(hard[j, 1])
        s_lo, s_hi = float(soft[j, 0]), float(soft[j, 1])
        if under < h_lo - 1e-6 or over > h_hi + 1e-6:
            print(
                f"[spec-opt] INFEASIBLE {name} reference spans [{under:+.5f}, {over:+.5f}] against"
                f" a HARD limit of [{h_lo:+.5f}, {h_hi:+.5f}]. The solver clamps the joint there,"
                f" so no gain can track the excursion and the residual error is a floor."
            )
        elif under < s_lo - 1e-6 or over > s_hi + 1e-6:
            margin = min(under - h_lo, h_hi - over)
            print(
                f"[spec-opt] NOTE {name} reference spans [{under:+.5f}, {over:+.5f}], outside the"
                f" soft band [{s_lo:+.5f}, {s_hi:+.5f}] but inside the hard limit"
                f" [{h_lo:+.5f}, {h_hi:+.5f}] with {margin:+.5f} rad to spare. Trackable, but the"
                f" reward terms keyed on the soft band will charge for it."
            )

    # --- fitness composition, reported so the run records what it is actually minimising --------
    report_fitness_composition(env, dt_ctrl, args)

    # --- search space ---------------------------------------------------------------------------
    space = ActuatorSearchSpace(robot, args.gain_range, args.effort_range, args.velocity_range)
    print(space.describe())

    # --- CMA-ES ---------------------------------------------------------------------------------
    if args.resume:
        with open(args.resume, "rb") as fh:
            es = pickle.loads(fh.read())
        if es.N != space.num_params:
            raise ValueError(f"resumed state has dimension {es.N}, expected {space.num_params}")
        print(f"[spec-opt] resumed CMA-ES from {args.resume}")
    else:
        opts = cma.CMAOptions()
        opts.set(
            {
                "popsize": int(args.popsize),
                "bounds": [np.zeros(space.num_params), np.ones(space.num_params)],
                "seed": int(args.seed),
                "verb_disp": 1,
                "verb_filenameprefix": os.path.join(out_dir, "cma_log") + "/",
                "maxiter": int(args.generations),
                "BoundaryHandler": "BoundTransform",
            }
        )
        es = cma.CMAEvolutionStrategy(np.full(space.num_params, 0.5), float(args.sigma0), opts)

    history = []
    best_cost, best_x = np.inf, None

    # --- per generation diagnostics -------------------------------------------------------------
    # The achieved trajectories are written into one preallocated buffer that every generation
    # overwrites, so the arrays that survive the run belong to the last generation evaluated.  The
    # figures and the video are written under fixed names for the same reason.
    record_buf = torch.zeros(
        q_ref_t.shape[0], env.num_envs, q_ref_t.shape[1], device=env.device
    )
    video_fps = max(1, round(1.0 / (dt_ctrl * max(1, args.video_stride))))
    video_path = os.path.join(out_dir, "videos", "reference_tracking.mp4")

    # Which environment the camera follows.  The best candidate is only known once its generation
    # has been scored, so the camera is aimed at the previous generation's winner and the first
    # generation, having no predecessor, is recorded at environment zero.
    video_env_index = 0

    for gen in range(args.generations):
        solutions = es.ask()
        if len(solutions) != env.num_envs:
            raise RuntimeError(
                f"CMA-ES returned {len(solutions)} solutions for {env.num_envs} environments, "
                "the population and the environment count must stay one to one"
            )

        decoded = [space.decode(x) for x in solutions]
        apply_population(robot, decoded, env.device)

        sink = None
        if args.video:
            if env.viewport_camera_controller is not None:
                # viewport_camera_controller.py:105
                env.viewport_camera_controller.set_view_env_index(video_env_index)
            # a fresh sink each generation, writing over the previous generation's file
            sink = FrameSink(video_path, video_fps)

        parts = rollout(env, q_ref_t, args, record_buf=record_buf, sink=sink)

        costs = combine_fitness(parts, args)
        es.tell(solutions, list(costs))

        i = int(np.argmin(costs))
        if costs[i] < best_cost:
            best_cost, best_x = float(costs[i]), np.asarray(solutions[i]).copy()

        print(
            f"[spec-opt] gen {gen:4d}  cost min {costs.min():9.4f}  median"
            f" {np.median(costs):9.4f}   rmse min {np.nanmin(parts['rmse']):8.5f} median"
            f" {np.nanmedian(parts['rmse']):8.5f}   fell {int(parts['fell'].sum())}/{env.num_envs}"
            f"   reward median {np.nanmedian(parts['mean_reward']):8.4f}"
        )

        history.append(
            {
                "generation": gen,
                "solutions": np.asarray(solutions),
                "cost": costs,
                **{k: v for k, v in parts.items()},
            }
        )

        np.savez_compressed(
            os.path.join(out_dir, "history.npz"),
            **{
                f"gen{h['generation']:04d}_{k}": v
                for h in history
                for k, v in h.items()
                if k != "generation"
            },
        )
        with open(os.path.join(out_dir, "cma_state.pkl"), "wb") as fh:
            fh.write(es.pickle_dumps())
        dump_best(out_dir, space, best_x, best_cost, args)

        # overwritten every generation, so what remains at the end is the last one evaluated
        write_tracking_plots(
            out_dir, t_ctrl, q_ctrl[:, perm], record_buf.detach().cpu().numpy(),
            parts["steps"], parts["rmse"], term._joint_names, args,
            always_include=video_env_index,
        )
        if sink is not None:
            finalise_video(out_dir, sink, video_env_index, gen, float(costs[video_env_index]))

        # aim the camera at this generation's winner for the next rollout
        video_env_index = i

        if es.stop():
            print(f"[spec-opt] CMA-ES terminated, {es.stop()}")
            break

    if best_x is not None:
        print(f"[spec-opt] best cost {best_cost:.6f}")
        print(json.dumps(space.decode(best_x), indent=2))
        print(f"[spec-opt] written to {os.path.join(out_dir, 'best_params.json')}")

    env.close()


def dump_best(out_dir, space, best_x, best_cost, args) -> None:
    if best_x is None:
        return
    payload = {
        "cost": best_cost,
        "reward_weight": float(args.reward_weight) if args.use_reward else None,
        "x": best_x.tolist(),
        "parameters": space.decode(best_x),
        "baseline": space.baseline,
        "ranges": {k: list(v) for k, v in space.ranges.items()},
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None), list, tuple))},
    }
    with open(os.path.join(out_dir, "best_params.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
    simulation_app.close()
