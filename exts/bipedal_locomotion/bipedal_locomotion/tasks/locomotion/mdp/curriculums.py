from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.terrains import TerrainImporter


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def compute_range(config, current_step, total_steps):
    start_step = config["start_frac"] * total_steps
    end_step = config["end_frac"] * total_steps

    if current_step < start_step:
        return tuple(config["min_range"]), 0.0
    elif current_step < end_step:
        alpha = (
            (current_step - start_step) / (end_step - start_step)
            if end_step > start_step
            else 1.0
        )
        start_range = config["min_range"]
        min_v = start_range[0] + alpha * (config["max_range"][0] - start_range[0])
        max_v = start_range[1] + alpha * (config["max_range"][1] - start_range[1])
        return (min_v, max_v), alpha
    else:
        return tuple(config["max_range"]), 1.0


def velocity_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    max_steps: int,
    x_config: dict[str, Any],
    y_config: dict[str, Any],
    z_config: dict[str, Any],
) -> torch.Tensor:
    """Curriculum that slowly increases the commanded velocity ranges based on training progress.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        command_name: The name of the command term.
        max_steps: Total number of steps for training (max_iterations * num_steps_per_env).
        x_config: Dictionary with 'start_frac', 'end_frac', 'min_range', 'max_range'.
        y_config: Dictionary with 'start_frac', 'end_frac', 'min_range', 'max_range'.
        z_config: Dictionary with 'start_frac', 'end_frac', 'min_range', 'max_range'.

    Returns:
        torch.Tensor: A tensor containing the current progress (0 to 1) for each component.
    """
    current_step = env.common_step_counter
    command_term = env.command_manager.get_term(command_name)

    # Linear velocity x
    range_x, progress_x = compute_range(x_config, current_step, max_steps)
    command_term.cfg.ranges.lin_vel_x = range_x

    # Linear velocity y
    range_y, progress_y = compute_range(y_config, current_step, max_steps)
    command_term.cfg.ranges.lin_vel_y = range_y

    # Angular velocity z
    range_z, progress_z = compute_range(z_config, current_step, max_steps)
    command_term.cfg.ranges.ang_vel_z = range_z

    progress = math.ceil(progress_x) or math.ceil(progress_y) or math.ceil(progress_z)

    if progress:
        return torch.ones(1)
    else:
        return torch.zeros(1)


def modify_push_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: float = 0.0,
):
    """Curriculum that modifies the maximum push (perturbation) velocity over some intervals.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        max_velocity: The maximum velocity of the push.
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """
    try:
        term_cfg = env.event_manager.get_term_cfg("push_robot")
    except:
        # print("No push_robot term found in the event manager")
        return 0.0
    curr_setting = term_cfg.params["velocity_range"]["x"][1]
    if env.common_step_counter < starting_step:
        return curr_setting
    if env.common_step_counter % interval == 0:
        if (
            torch.sum(env.termination_manager.get_term("base_contact"))
            < torch.sum(env.termination_manager.get_term("time_out")) * 2
        ):
            # obtain term settings
            term_cfg = env.event_manager.get_term_cfg("push_robot")
            # update term settings
            curr_setting = term_cfg.params["velocity_range"]["x"][1]
            curr_setting = torch.clamp(
                torch.tensor(curr_setting * 1.5), 0.0, max_velocity[0]
            ).item()
            term_cfg.params["velocity_range"]["x"] = (-curr_setting, curr_setting)
            curr_setting = term_cfg.params["velocity_range"]["y"][1]
            curr_setting = torch.clamp(
                torch.tensor(curr_setting * 1.5), 0.0, max_velocity[1]
            ).item()
            term_cfg.params["velocity_range"]["y"] = (-curr_setting, curr_setting)
            env.event_manager.set_term_cfg("push_robot", term_cfg)

        if (
            torch.sum(env.termination_manager.get_term("base_contact"))
            > torch.sum(env.termination_manager.get_term("time_out")) / 2
        ):
            # obtain term settings
            term_cfg = env.event_manager.get_term_cfg("push_robot")
            # update term settings
            curr_setting = term_cfg.params["velocity_range"]["x"][1]
            curr_setting = torch.clamp(
                torch.tensor(curr_setting - 0.2), 0.0, max_velocity[0]
            ).item()
            term_cfg.params["velocity_range"]["x"] = (-curr_setting, curr_setting)
            curr_setting = term_cfg.params["velocity_range"]["y"][1]
            curr_setting = torch.clamp(
                torch.tensor(curr_setting - 0.2), 0.0, max_velocity[1]
            ).item()
            term_cfg.params["velocity_range"]["y"] = (-curr_setting, curr_setting)
            env.event_manager.set_term_cfg("push_robot", term_cfg)

    return curr_setting


def reduce_tracking_rewards_std(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    minimum_std: float,
    interval: int,
    update_threshold: float = 0.8,
    update_rate: float = 0.995,
    starting_step: float = 0.0,
):
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if env.common_step_counter < starting_step:
        return term_cfg.params["std"]

    if env.common_step_counter % interval == 0:
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        if (
            torch.mean(rew) / env.max_episode_length
            > update_threshold * term_cfg.weight * env.step_dt
        ):
            if term_cfg.params["std"] > minimum_std:
                term_cfg.params["std"] = term_cfg.params["std"] * update_rate
    return term_cfg.params["std"]


def modify_command_velocity_y(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: float = 0.0,
    update_threshold: float = 0.8,
    update_rate: float = 0.5,
):
    """Curriculum that modifies the maximum command velocity over some intervals.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: Dictionary with 'lin_vel_x', 'lin_vel_y', 'ang_vel_z' max ranges.
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """

    command_cfg = env.command_manager.get_term("base_velocity").cfg

    if env.common_step_counter < starting_step:
        return command_cfg.ranges.lin_vel_y[1]

    if env.common_step_counter % interval == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        if (
            torch.mean(rew) / env.max_episode_length
            > update_threshold * term_cfg.weight * env.step_dt
        ):
            # Update lin_vel_y
            curr_lin_vel_y = command_cfg.ranges.lin_vel_y
            max_y = max_velocity
            curr_lin_vel_y = (
                torch.clamp(
                    torch.tensor(curr_lin_vel_y[0] - update_rate), max_y[0], 0.0
                ).item(),
                torch.clamp(
                    torch.tensor(curr_lin_vel_y[1] + update_rate), 0.0, max_y[1]
                ).item(),
            )
            command_cfg.ranges.lin_vel_y = curr_lin_vel_y

    return command_cfg.ranges.lin_vel_y[1]


def modify_command_velocity_angular(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    max_velocity: dict[str, Sequence[float]],
    interval: int,
    starting_step: float = 0.0,
    update_threshold: float = 0.8,
    update_rate: float = 0.5,
):
    """Curriculum that modifies the maximum command velocity over some intervals.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: Dictionary with 'lin_vel_x', 'lin_vel_y', 'ang_vel_z' max ranges.
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """

    command_cfg = env.command_manager.get_term("base_velocity").cfg

    if env.common_step_counter < starting_step:
        return command_cfg.ranges.ang_vel_z[1]

    if env.common_step_counter % interval == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        if (
            torch.mean(rew) / env.max_episode_length
            > update_threshold * term_cfg.weight * env.step_dt
        ):
            # Update ang_vel_z
            curr_ang_vel_z = command_cfg.ranges.ang_vel_z
            max_z = max_velocity
            curr_ang_vel_z = (
                torch.clamp(
                    torch.tensor(curr_ang_vel_z[0] - update_rate), max_z[0], 0.0
                ).item(),
                torch.clamp(
                    torch.tensor(curr_ang_vel_z[1] + update_rate), 0.0, max_z[1]
                ).item(),
            )
            command_cfg.ranges.ang_vel_z = curr_ang_vel_z

    return command_cfg.ranges.ang_vel_z[1]


def modify_command_velocity_x(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: float = 0.0,
    update_threshold: float = 0.8,
    update_rate: float = 0.5,
):
    """Curriculum that modifies the maximum command velocity over some intervals.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        max_velocity: The maximum velocity.
        interval: The number of steps after which the condition is checked again
        starting_step: The number of steps after which the curriculum is applied.
    """

    command_cfg = env.command_manager.get_term("base_velocity").cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x

    if env.common_step_counter < starting_step:
        return curr_lin_vel_x[1]

    if env.common_step_counter % interval == 0:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        if (
            torch.mean(rew) / env.max_episode_length
            > update_threshold * term_cfg.weight * env.step_dt
        ):
            curr_lin_vel_x = (
                torch.clamp(
                    torch.tensor(curr_lin_vel_x[0] - update_rate), max_velocity[0], 0.0
                ).item(),
                torch.clamp(
                    torch.tensor(curr_lin_vel_x[1] + update_rate), 0.0, max_velocity[1]
                ).item(),
            )
            command_cfg.ranges.lin_vel_x = curr_lin_vel_x

    return curr_lin_vel_x[1]


def terrain_levels_vel_delayed(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg,
    starting_step: float = 0.0,
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    if env.common_step_counter > starting_step:
        # extract the used quantities (to enable type-hinting)
        # compute the distance the robot walked
        distance = torch.norm(
            asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
            dim=1,
        )
        # robots that walked far enough progress to harder terrains
        move_up = distance > terrain.cfg.terrain_generator.size[0] * 0.75
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (
            distance
            < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.75
        )
        move_down *= ~move_up
        # update terrain levels
        terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
