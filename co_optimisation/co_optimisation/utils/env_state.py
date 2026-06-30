"""Capture and restore of IsaacLab environment and manager runtime state.

IsaacLab exposes no runtime-state checkpoint of its own.  ``InteractiveScene.get_state``
and ``ManagerBasedEnv.reset_to`` persist only the physical articulation state, while the
true curriculum progress, command ranges, push schedule, reward shaping, and per-environment
episodic buffers live as mutable attributes on the managers (catalogued in
``CONTEXT_isaaclab_env.md``).  This module reads those attributes directly at a checkpoint and
writes them back on resume, so a continued run begins from conditions identical to the run it
continues.

The capture returns a dictionary of detached CPU tensors and Python scalars.  The restore is
ordered deliberately.  ``reset_to`` runs the internal ``_reset_idx`` machinery, which advances
the curriculum, zeroes ``episode_length_buf`` and ``RewardManager._episode_sums``, and resamples
the command buffers (``manager_based_rl_env.py:349-394``).  The physical state is therefore
re-applied first, with the terrain curriculum suppressed so the saved ``env_origins`` that govern
robot placement are not mutated during the reset, and every buffer that ``_reset_idx`` perturbs
is overwritten with the saved value afterwards.  The global random number generators are restored
last, so the randomness consumed inside ``reset_to`` does not leave them advanced past the saved
position.
"""

from __future__ import annotations

import random

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Small per-field helpers
# ---------------------------------------------------------------------------


def _ranges_to_dict(ranges) -> dict[str, tuple[float, float]]:
    """Snapshot the three curriculum-mutated velocity-command ranges.

    The ``heading`` field is static for this task and is left untouched.
    """
    return {
        "lin_vel_x": tuple(ranges.lin_vel_x),
        "lin_vel_y": tuple(ranges.lin_vel_y),
        "ang_vel_z": tuple(ranges.ang_vel_z),
    }


def _restore_ranges(ranges, saved: dict[str, tuple[float, float]]) -> None:
    ranges.lin_vel_x = tuple(saved["lin_vel_x"])
    ranges.lin_vel_y = tuple(saved["lin_vel_y"])
    ranges.ang_vel_z = tuple(saved["ang_vel_z"])


def _capture_reward_std(reward_manager) -> dict[str, float]:
    """Snapshot the ``std`` parameter of every reward term that carries one.

    The tracking-reward std is sharpened in place by ``reduce_tracking_rewards_std``
    (``curriculums.py:305``) and exists in no checkpoint.
    """
    stds: dict[str, float] = {}
    for name in reward_manager.active_terms:
        params = reward_manager.get_term_cfg(name).params
        if "std" in params:
            stds[name] = float(params["std"])
    return stds


def _restore_reward_std(reward_manager, saved: dict[str, float]) -> None:
    for name, std in saved.items():
        params = reward_manager.get_term_cfg(name).params
        if "std" in params:
            params["std"] = std


def _capture_command_buffers(cmd) -> dict:
    """Snapshot the per-environment buffers of the velocity command term."""
    return {
        "vel_command_b": cmd.vel_command_b.detach().cpu().clone(),
        "heading_target": cmd.heading_target.detach().cpu().clone(),
        "is_heading_env": cmd.is_heading_env.detach().cpu().clone(),
        "is_standing_env": cmd.is_standing_env.detach().cpu().clone(),
        "time_left": cmd.time_left.detach().cpu().clone(),
        "command_counter": cmd.command_counter.detach().cpu().clone(),
        "metrics": {k: v.detach().cpu().clone() for k, v in cmd.metrics.items()},
    }


def _restore_command_buffers(cmd, saved: dict) -> None:
    device = cmd.vel_command_b.device
    cmd.vel_command_b.copy_(saved["vel_command_b"].to(device))
    cmd.heading_target.copy_(saved["heading_target"].to(device))
    cmd.is_heading_env.copy_(saved["is_heading_env"].to(device))
    cmd.is_standing_env.copy_(saved["is_standing_env"].to(device))
    cmd.time_left.copy_(saved["time_left"].to(device))
    cmd.command_counter.copy_(saved["command_counter"].to(device))
    for k, v in saved["metrics"].items():
        if k in cmd.metrics:
            cmd.metrics[k].copy_(v.to(device))


def _restore_rng(saved: dict) -> None:
    """Restore the global torch, cuda, numpy, and python RNG states."""
    torch.set_rng_state(saved["torch"])
    if torch.cuda.is_available() and saved["cuda"] is not None:
        torch.cuda.set_rng_state_all(saved["cuda"])
    np.random.set_state(saved["numpy"])
    random.setstate(saved["python"])


def _scene_state_to(scene_state: dict, device) -> dict:
    """Move a nested ``InteractiveScene.get_state`` dict onto ``device``."""
    moved: dict = {}
    for asset_type, entities in scene_state.items():
        moved[asset_type] = {}
        for name, components in entities.items():
            moved[asset_type][name] = {
                key: value.to(device) for key, value in components.items()
            }
    return moved


# ---------------------------------------------------------------------------
# Public capture / restore
# ---------------------------------------------------------------------------


def capture_env_state(env) -> dict:
    """Capture all evolving env and manager runtime state for an identical resume.

    Args:
        env: the wrapped vectorised environment (``RslRlVecEnvWrapper``).

    Returns:
        A dictionary of detached CPU tensors and Python scalars, suitable for
        ``torch.save``.
    """
    u = env.unwrapped
    cmd = u.command_manager.get_term("base_velocity")
    push = u.event_manager.get_term_cfg("push_robot")
    terrain = u.scene.terrain
    return {
        "common_step_counter": int(u.common_step_counter),
        "sim_step_counter": int(u._sim_step_counter),
        "episode_length_buf": u.episode_length_buf.detach().cpu().clone(),
        "terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
        "env_origins": terrain.env_origins.detach().cpu().clone(),
        "command_ranges": _ranges_to_dict(cmd.cfg.ranges),
        "push_velocity_range": {
            "x": tuple(push.params["velocity_range"]["x"]),
            "y": tuple(push.params["velocity_range"]["y"]),
        },
        "reward_std": _capture_reward_std(u.reward_manager),
        "last_episode_dones": u.termination_manager._last_episode_dones.detach()
        .cpu()
        .clone(),
        "episode_sums": {
            k: v.detach().cpu().clone()
            for k, v in u.reward_manager._episode_sums.items()
        },
        "command_buffers": _capture_command_buffers(cmd),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
        "scene_state": u.scene.get_state(is_relative=True),
    }


def restore_env_state(env, state: dict) -> None:
    """Write the captured state back in place for an identical continuation.

    The ordering is deliberate, see the module docstring: the physical state is
    re-applied first (terrain curriculum suppressed), then every buffer that the
    internal ``_reset_idx`` perturbs is overwritten, then the global RNGs are
    restored last.

    Args:
        env: the wrapped vectorised environment (``RslRlVecEnvWrapper``).
        state: the dictionary produced by :func:`capture_env_state`.
    """
    u = env.unwrapped
    device = u.device
    cmd = u.command_manager.get_term("base_velocity")
    terrain = u.scene.terrain

    # 1. Counters that reset() / reset_to() never touch (manager_based_rl_env.py:74,
    #    manager_based_env.py:133).
    u.common_step_counter = int(state["common_step_counter"])
    u._sim_step_counter = int(state["sim_step_counter"])

    # 2. Terrain progress, set before reset_to so the relative robot placement uses
    #    the saved origins (terrain_importer.py:347,353).
    terrain.terrain_levels.copy_(state["terrain_levels"].to(device))
    terrain.env_origins.copy_(state["env_origins"].to(device))

    # 3. Physical articulation state. Suppress the terrain curriculum so the internal
    #    _reset_idx (manager_based_rl_env.py:356) does not promote / demote levels or
    #    move env_origins during placement.
    u._suppress_terrain_curriculum = True
    try:
        u.reset_to(
            _scene_state_to(state["scene_state"], device),
            env_ids=None,
            is_relative=True,
        )
    finally:
        u._suppress_terrain_curriculum = False

    # 4. Overwrite everything _reset_idx perturbed with the saved values.
    _restore_ranges(cmd.cfg.ranges, state["command_ranges"])
    push = u.event_manager.get_term_cfg("push_robot")
    push.params["velocity_range"]["x"] = tuple(state["push_velocity_range"]["x"])
    push.params["velocity_range"]["y"] = tuple(state["push_velocity_range"]["y"])
    _restore_reward_std(u.reward_manager, state["reward_std"])

    u.episode_length_buf.copy_(state["episode_length_buf"].to(device))
    u.termination_manager._last_episode_dones.copy_(
        state["last_episode_dones"].to(device)
    )
    for k, v in state["episode_sums"].items():
        if k in u.reward_manager._episode_sums:
            u.reward_manager._episode_sums[k].copy_(v.to(device))
    _restore_command_buffers(cmd, state["command_buffers"])

    # 5. Global RNG last, so randomness consumed during reset_to does not leave the
    #    generators advanced past the saved position (CONTEXT_isaaclab_env.md section 1.4).
    _restore_rng(state["rng"])
