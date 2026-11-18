"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import pickle
import sys
import yaml

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=400,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=24000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="Maximum number of iterations to train.",
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=None,
    help="The number of iterations between saves",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Relative path to checkpoint file.",
)
# The design search and the co-optimisation loop are configured by co_optimisation.config.CMAESCfg,
# one subclass per robot, rather than by arguments. See plans/copt_training_refinement.md.

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from highest_terrain_camera_wrapper import HighestTerrainCameraWrapper, PausableRecordVideo
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl_debug.runners import DebugOnPolicyRunner

from co_optimisation.algorithms import CoptPPO
from co_optimisation.config import CMAES_CFG_BY_FAMILY, CMAESCfg, without_termination
from co_optimisation.runners import CoptOnPolicyRunner
from co_optimisation.runners.usd_generator import (
    DEFAULT_PARAM_RANGES,
    GAIN_PARAM_KEYS,
    CatCMAESDesignGenerator,
    GrowingDesignDistCMAESDesignGenerator,
)

# Import extensions to set up environment tasks
from environments.utils.wrappers.rsl_rl import RslRlPpoAlgorithmMlpCfg
from himloco.runners import HIMOnPolicyRunner
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# Replacement functions for missing Isaac Lab utils
def dump_pickle(filename, data):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def dump_yaml(filename, data, sort_keys=False):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=sort_keys)


def _cmaes_cfg(family: str, agent_cfg, log_root_path: str) -> CMAESCfg:
    """The run's co-optimisation configuration, restored from params/cmaes.pkl of a resumed run.

    A resumed run begun before params/cmaes.pkl existed continues as it was launched, with the
    robot's configuration less the tolerances and the fine-tune, which those runs never had.
    """
    if agent_cfg.resume:
        resume_path = args_cli.checkpoint_path or get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        saved = os.path.join(os.path.dirname(resume_path), "params", "cmaes.pkl")
        if os.path.exists(saved):
            with open(saved, "rb") as f:
                return pickle.load(f)
        print(f"[WARNING] No params/cmaes.pkl beside {resume_path}, resuming without CMA-ES tolerances or fine-tune.")
        return without_termination(CMAES_CFG_BY_FAMILY[family]())
    return CMAES_CFG_BY_FAMILY[family]()


# @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main():
    """Train with RSL-RL agent."""
    # parse configuration
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
    )
    env_cfg.sim.log_dir = "/root/logs"
    env_cfg.sim.render.rendering_mode = "balanced"
    env_cfg.sim.render.dlss_mode = 0
    # env_cfg.viewer.resolution = (640, 480)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(
        args_cli.task, args_cli
    )

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.save_interval is not None:
        agent_cfg.save_interval = args_cli.save_interval

    # inject total steps into curriculum for progress calculation
    if hasattr(env_cfg.curriculum, "velocity_curriculum"):
        env_cfg.curriculum.velocity_curriculum.params["max_steps"] = (
            agent_cfg.max_iterations * agent_cfg.num_steps_per_env
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # track lowest-difficulty terrain env before each captured frame
        env = HighestTerrainCameraWrapper(env)

        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        # print_dict(video_kwargs, nesting=4)
        env = PausableRecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner_cls = OnPolicyRunner
    if args_cli.policy_type == "HIMPPO":
        runner_cls = HIMOnPolicyRunner
        agent_cfg.policy.class_name = "HIMActorCritic"
        agent_cfg.algorithm.class_name = "HIMPPO"
    elif args_cli.policy_type in ("quadruped-debug", "go1-debug", "go1-default-debug"):
        # Timing-instrumented runner, unchanged policy and algorithm class names, used to
        # attribute a throughput difference to the environment and MDP stack rather than
        # to the policy architecture. See context/go1_quadruped_throughput.md.
        runner_cls = DebugOnPolicyRunner
        agent_cfg.policy.class_name = "DebugActorCritic"
    runner = None
    _COPT_ACTOR_CRITIC = {"COPT": "CoptActorCritic", "COPT-MORAL": "CoptMorALActorCritic"}
    if args_cli.policy_type in _COPT_ACTOR_CRITIC:

        # The family selects which of the vendored generator's two packages supplies
        # the geometry, the mass model and the actuator catalogue. Its own
        # variable_params.json is the baseline, and it reproduces the donor robot.
        # An unknown task is refused here, its failure otherwise being an
        # unresolvable articulation root deep in the importer's output.
        _FAMILY_BY_TASK = {
            "Isaac-Quadruped-Copt": "quadruped",
            "Isaac-Limx-SF-Copt": "biped",
        }
        _family = next(
            (fam for key, fam in _FAMILY_BY_TASK.items() if key in args_cli.task), None
        )
        if _family is None:
            raise ValueError(
                f"Task {args_cli.task!r} maps to no design generator family. "
                f"Known keys: {sorted(_FAMILY_BY_TASK)}."
            )
        cmaes_cfg = _cmaes_cfg(_family, agent_cfg, log_root_path)
        _num_individuals = cmaes_cfg.num_individuals
        # ea_update_interval * num_steps_per_env (120 * 25 = 3000) should be more
        # than episode_duration / (dt * decimation) (20 / (0.005 * 4) = 1000)
        ea_update_interval = cmaes_cfg.ea_update_interval
        ea_late_start = cmaes_cfg.ea_late_start
        generator_kwargs = dict(
            family=_family,
            num_individuals=_num_individuals,
            param_ranges=dict(cmaes_cfg.param_ranges),
            output_dir=os.path.join(log_dir, "copt_usds"),
            filters=list(cmaes_cfg.filters),
            filter_params=dict(cmaes_cfg.filter_params),
        )
        late_start_pop_size = env.num_envs // cmaes_cfg.late_start_prior_divisor
        tolerances = dict(tolx=cmaes_cfg.tolx, tolfun=cmaes_cfg.tolfun, tolfunhist=cmaes_cfg.tolfunhist)
        # The co-optimisation type, see CMAESCfg.design_search.
        generator_cls = {
            "leg_geometry": GrowingDesignDistCMAESDesignGenerator,
            "actuator_leg_geometry": CatCMAESDesignGenerator,
            "actuator": CatCMAESDesignGenerator,
        }.get(cmaes_cfg.design_search)
        if generator_cls is None:
            raise ValueError(f"design_search {cmaes_cfg.design_search!r} is none of 'leg_geometry', "
                             "'actuator_leg_geometry' and 'actuator'")
        # The actuator search tunes the picks and their gains on the baseline's geometry, so its ranges
        # must name gain scales and nothing else.
        if cmaes_cfg.design_search == "actuator":
            names = set(cmaes_cfg.param_ranges)
            if not names or names - set(GAIN_PARAM_KEYS):
                raise ValueError(
                    f"design_search 'actuator' needs param_ranges of gain scales alone, at least one of "
                    f"{list(GAIN_PARAM_KEYS)}, got {sorted(names)}")
        # Every type stops on the tolerances, the categorical ones on CatCMAESDesignGenerator._stop_reasons.
        termination = dict(
            **tolerances,
            final_population="mean" if cmaes_cfg.finetune_iterations is not None else "last",
            tolcat=cmaes_cfg.tolcat,
        )
        design_generator = generator_cls(
            **generator_kwargs,
            sigma0=cmaes_cfg.sigma0,
            seed=cmaes_cfg.seed,
            late_start=True,
            late_start_it=int(ea_late_start / ea_update_interval),
            late_start_prior_pop_size=late_start_pop_size,
            max_cma_iter=(cmaes_cfg.max_cma_iter if cmaes_cfg.max_cma_iter is not None
                          else (agent_cfg.max_iterations - ea_late_start) / ea_update_interval),
            **termination,
        )
        population_sizes = (late_start_pop_size, _num_individuals)
        if cmaes_cfg.shank_foot_gap is not None:
            design_generator.builder_kwargs = {"shank_foot_gap": cmaes_cfg.shank_foot_gap}
        # Environment i runs design i % population size and fitness is averaged per
        # design, so a population that does not divide the scene is ranked on
        # unequal evidence.
        for n in population_sizes:
            if env.num_envs % n:
                raise ValueError(
                    f"--num_envs {env.num_envs} is not a multiple of the {n} designs a "
                    f"generation of design_search {cmaes_cfg.design_search} carries."
                )
        agent_cfg.policy.class_name = _COPT_ACTOR_CRITIC[args_cli.policy_type]
        agent_cfg.algorithm.class_name = "CoptPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        # 1.0 is the unweighted sum every run to date trained with.
        agent_cfg_dict["algorithm"]["estimator_loss_coef"] = 1.0
        agent_cfg_dict["copt"] = {
            "ea_update_interval": ea_update_interval,
            "ea_late_start": ea_late_start,
            "num_individuals": _num_individuals,
            "randomise_before_late_start": cmaes_cfg.randomise_before_late_start,
            "reinit_lr_each_generation": cmaes_cfg.reinit_lr_each_generation,
            "eval_duration_s": cmaes_cfg.eval_duration_s,
            "eval_seed": cmaes_cfg.eval_seed,
            # The tolerances are read by the generator, recorded here so that copt_agent.yaml names them.
            "tolx": cmaes_cfg.tolx,
            "tolfun": cmaes_cfg.tolfun,
            "tolfunhist": cmaes_cfg.tolfunhist,
            "tolcat": cmaes_cfg.tolcat,
            "finetune_iterations": cmaes_cfg.finetune_iterations,
            # The whole configuration, recorded so that copt_agent.yaml carries it beside params/cmaes.yaml.
            "cmaes": cmaes_cfg.to_dict(),
        }
        if cmaes_cfg.shank_foot_gap is not None:
            # Read by nothing in the runner, recorded so that copt_agent.yaml names the geometry.
            agent_cfg_dict["copt"]["shank_foot_gap"] = cmaes_cfg.shank_foot_gap
        runner = CoptOnPolicyRunner(
            env,
            design_generator,
            copy.deepcopy(agent_cfg_dict),
            log_dir=log_dir,
            device=agent_cfg.device,
        )
    else:
        runner = runner_cls(
            env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
        )

    # write git state to logs
    # runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        # get path to previous checkpoint
        if args_cli.checkpoint_path is not None:
            resume_path = args_cli.checkpoint_path
        else:
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # set seed of the environment
    env.seed(agent_cfg.seed)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    if args_cli.policy_type in _COPT_ACTOR_CRITIC:
        # agent.yaml serialises the configclass, without the copt block the runner receives.
        dump_yaml(os.path.join(log_dir, "params", "copt_agent.yaml"), agent_cfg_dict)
        dump_yaml(os.path.join(log_dir, "params", "cmaes.yaml"), cmaes_cfg)
        dump_pickle(os.path.join(log_dir, "params", "cmaes.pkl"), cmaes_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
