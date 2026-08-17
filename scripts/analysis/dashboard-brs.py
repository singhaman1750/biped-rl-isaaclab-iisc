import argparse
import glob
import os
import re

import dash
import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from plotly.subplots import make_subplots

# --- Configuration ---
# Map which axis index the angular command corresponds to.
# 0 = X (Roll), 1 = Y (Pitch), 2 = Z (Yaw).
CMD_ANG_AXIS = 2
DEFAULT_NUM_ENVS = 20  # Fallback if not detected

# Metric Definitions
JOINT_METRICS = {
    "joint_velocities": "Joint Velocities",
    "joint_torques": "Joint Torques",
    "joint_powers": "Joint Powers",
    "joint_positions": "Joint Positions",
    "joint_accelerations": "Joint Accelerations",
}

# Metric Units (Isaac Sim Conventions)
METRIC_UNITS = {
    "joint_velocities": "Velocity (rad/s)",
    "joint_torques": "Torque (Nm)",
    "joint_powers": "Power (W)",
    "joint_positions": "Position (rad)",
    "joint_accelerations": "Acceleration (rad/s²)",
}

# Fallback joint names for SD_BRS1, used only for dumps written before play.py began
# recording "joint_names". Prefer resolve_joint_names below, which reads the order the
# articulation actually reported rather than assuming one.
#
# The order is LEFT before RIGHT at every tree depth. IsaacLab enumerates degrees of
# freedom breadth-first, giving the depth sequence HipRoll, HipPitch, KneePitch,
# AnkleRoll, AnklePitch, but the within-depth tie-break is NOT the URDF declaration
# order. SD_BRS.urdf declares the whole right leg chain before the left, and the
# articulation still yields left first, so the ordering cannot be derived from the asset.
#
# CORRECTION, 2026-07-31. This list previously read right before left on exactly that
# faulty derivation, so every per-joint panel this dashboard has ever drawn carried a
# transposed side label. Proved three independent ways in the twentieth pass of
# /ws/context/brs_gait.md. Joint index 2 spans -1.2521 rad and only HipPitchL admits
# -1.25 while index 3 spans +1.2502 and only HipPitchR admits +1.25. Knee index 4 is
# near extension exactly when foot index 0 reports contact, on 97.2 percent of steps.
# And the asymmetric identified masses resolve every body uniquely, Link2L 3.860000
# against Link2R 3.860900. HipYaw joints are fixed and contribute no DOF.
JOINT_NAMES = [
    "HipRollL",
    "HipRollR",
    "HipPitchL",
    "HipPitchR",
    "KneePitchL",
    "KneePitchR",
    "AnkleRollL",
    "AnkleRollR",
    "AnklePitchL",
    "AnklePitchR",
]
# The joint count and grid geometry are derived per figure from the resolved name list
# rather than from the fallback, since a dump may carry a different articulation. The
# former module level NUM_JOINTS and GRID_ROWS were removed with that change.

# Fallback feet names, same provenance and same correction. find_bodies matches on BODY
# INDEX order rather than regex order, and the body order is likewise left before right,
# Link6L at index 11 and Link6R at 12, so "Link6[LR]" resolves as L then R.
FEET_NAMES = ["Link6L", "Link6R"]

# Reference levels drawn on the feet panel, matching brs_base_env_cfg.py rew_foot_clearance.
SOLE_CLEARANCE_TARGET = 0.08  # target_height of foot_clearance_reward_v2 (m)
CONTACT_FORCE_THRESHOLD = 1.0  # force_threshold above which a foot counts as grounded (N)

def pretty_joint_name(name):
    """Split CamelCase joint names into words, e.g. 'HipRollR' -> 'Hip Roll R'."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name)


def _names_from_dumps(experiments_dict, key, fallback):
    """Return the index order recorded in a dump, or the fallback for older dumps.

    play.py records "joint_names", "body_names" and "feet_names" from the live
    articulation, because the order cannot be derived from the URDF, the within-depth
    tie-break of the breadth-first enumeration not being the declaration order. Dumps
    written before that change carry none of these keys, so the hardcoded fallback stands
    in for them, and the first experiment that does carry the key supplies the order for
    the whole figure, every experiment in one dashboard being the same robot.
    """
    for data in experiments_dict.values():
        if not isinstance(data, dict):
            continue
        names = data.get(key)
        if names is None:
            continue
        names = list(np.asarray(names).ravel())
        if names:
            return [str(n) for n in names]
    return list(fallback)


def resolve_joint_names(experiments_dict):
    """Joint names in articulation order, preferring what the dump itself recorded."""
    return _names_from_dumps(experiments_dict, "joint_names", JOINT_NAMES)


def resolve_feet_names(experiments_dict):
    """Feet body names in the order the DataLogger resolved them."""
    return _names_from_dumps(experiments_dict, "feet_names", FEET_NAMES)


def load_statistics(seed_dir):
    """Return the statistics record set one play wrote, or None where it wrote none.

    Reads and nothing more. The record set is computed by play.py at the close of a play,
    from the arrays it holds in memory, so this function's only responsibility is to find
    the file and hand it on. A dashboard that recomputed would be a second implementation
    of every statistic, free to disagree with the first, and the point of the pipeline is
    that a figure has one origin.

    An absent file means the play predates the pipeline or its statistics block failed,
    and in both cases the remedy is to replay rather than to compute here.
    """
    path = os.path.join(seed_dir, "statistics.npy")
    if not os.path.exists(path):
        return None
    try:
        raw = np.load(path, allow_pickle=True)
        return raw.item() if raw.ndim == 0 else raw
    except Exception as error:
        print(f"[WARN] Failed to load statistics.npy at {path}: {error}")
        return None


def load_experiments(root_dir):
    """
    Traverses the directory structure to load experiment data.
    Structure: root -> exp_dir -> data -> seed_dir -> dump.npy
    Returns: dict[seed][exp_name] = data_dict
    """
    data_store = {}
    
    # Normalize path
    root_dir = os.path.normpath(root_dir)
    
    print(f"[INFO] Scanning for experiments in: {root_dir}")
    
    # Pattern: root/exp_name/data/seed/dump.npy
    search_pattern = os.path.join(root_dir, "**", "dump.npy")
    files = glob.glob(search_pattern, recursive=True)
    
    for file_path in files:
        try:
            # Extract hierarchy from path
            seed_dir = os.path.dirname(file_path)
            data_dir = os.path.dirname(seed_dir)
            exp_dir = os.path.dirname(data_dir)
            
            seed = os.path.basename(seed_dir)
            exp_name = os.path.basename(exp_dir)
            
            # Load numpy file
            try:
                raw_data = np.load(file_path, allow_pickle=True)
            except Exception as e:
                print(f"[ERROR] Failed to load npy {file_path}: {e}")
                continue

            # Handle case where npy contains a 0-d array wrapping a dict
            if raw_data.ndim == 0:
                data_dict = raw_data.item()
            else:
                data_dict = raw_data
                
            if seed not in data_store:
                data_store[seed] = {}
            
            rewards_path = os.path.join(seed_dir, "rewards.npy")
            if os.path.exists(rewards_path):
                try:
                    raw_rewards = np.load(rewards_path, allow_pickle=True)
                    reward_dict = raw_rewards.item() if raw_rewards.ndim == 0 else raw_rewards
                    data_dict["_rewards"] = reward_dict
                    print(f"[LOADED] Rewards keys: {list(reward_dict.keys())}")
                except Exception as e:
                    print(f"[WARN] Failed to load rewards.npy for {exp_name}/{seed}: {e}")

            data_dict["_statistics"] = load_statistics(seed_dir)

            data_store[seed][exp_name] = data_dict
            # print(data_dict.keys())
            # print(f"[LOADED] Seed: {seed} | Exp: {exp_name}")
            
        except Exception as e:
            print(f"[ERROR] Processing {file_path}: {e}")

    return data_store

def get_env_data(raw_data, env_id):
    """Extract one environment's series from a dump channel.

    Every per step channel is a stacked array of shape (Time, Env, ...), so this is a
    slice on the environment axis and nothing more. CHANGED, the previous form also
    accepted the list of arrays that DataLogger.plot used to write and the singleton
    wrap it put in front of most channels. Neither is written any longer, and a dump
    that still carries them predates the pipeline and must be replayed to be read here.

    Returns: (Time, ...) for the chosen environment, or an empty array where the channel
    is absent, empty, or has too few environments.
    """
    if raw_data is None:
        return np.array([])
    data_np = np.asarray(raw_data)
    if data_np.ndim < 2 or env_id >= data_np.shape[1]:
        return np.array([])
    return data_np[:, env_id]

def apply_smoothing(data, weight):
    """
    Applies Exponential Moving Average (EMA) smoothing to the data.
    """
    if weight <= 0.0 or data.size == 0:
        return data
    
    try:
        df = pd.DataFrame(data)
        smoothed = df.ewm(alpha=1.0-weight).mean().values
        return smoothed
    except Exception as e:
        print(f"[WARN] Smoothing failed: {e}")
        return data

def create_joint_plot(experiments_dict, metric_key, env_id, zoom_range=None, smoothing=0.0, hidden_experiments=None):
    """
    Creates a 5x2 grid for a SINGLE joint metric, comparing all experiments.
    """
    hidden_experiments = hidden_experiments or []

    joint_names = resolve_joint_names(experiments_dict)
    num_joints = len(joint_names)
    grid_rows = (num_joints + 1) // 2
    clean_titles = [pretty_joint_name(name) for name in joint_names]

    fig = make_subplots(
        rows=grid_rows, cols=2,
        subplot_titles=clean_titles,
        shared_xaxes="all",
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.D3
    exp_names = sorted(experiments_dict.keys())

    for exp_idx, exp_name in enumerate(exp_names):
        data = experiments_dict[exp_name]
        
        if metric_key not in data:
            continue
            
        raw_data = data[metric_key]
        metric_data = get_env_data(raw_data, env_id)
        
        if metric_data.size == 0:
            continue
            
        if metric_data.ndim < 2 or metric_data.shape[1] != num_joints:
            print(f"[WARN] Skipping {metric_key} for {exp_name}, shape mismatch: {metric_data.shape}")
            continue

        metric_data = apply_smoothing(metric_data, smoothing)
        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True

        for j in range(num_joints):
            row = (j // 2) + 1
            col = (j % 2) + 1
            show_leg = (j == 0)
            
            fig.add_trace(
                go.Scattergl(
                    y=metric_data[:, j],
                    mode='lines',
                    name=exp_name,
                    legendgroup=exp_name,
                    showlegend=show_leg, 
                    line=dict(color=color, width=1.5),
                    opacity=0.8,
                    visible=is_visible
                ),
                row=row, col=col
            )

    pretty_name = JOINT_METRICS.get(metric_key, metric_key.replace("_", " ").title())
    fig.update_layout(height=500 * grid_rows, title_text=f"{pretty_name} (Env {env_id})")
    
    # Axis Annotations
    unit_label = METRIC_UNITS.get(metric_key, "Value")
    fig.update_yaxes(title_text=unit_label)
    fig.update_xaxes(title_text="Time (steps)")
    
    if zoom_range:
        fig.update_xaxes(range=zoom_range)
        
    return fig

def create_base_plot(experiments_dict, env_id, zoom_range=None, smoothing=0.0, hidden_experiments=None):
    """
    Creates a 3x2 grid for Base and Commanded Velocities.
    """
    hidden_experiments = hidden_experiments or []
    
    base_lin_key = "base_linear_velocity"
    base_ang_key = "base_angular_velocity"
    cmd_lin_key = "commanded_linear_velocity"
    cmd_ang_key = "commanded_angular_velocity"

    
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Linear X", "Angular X", "Linear Y", "Angular Y", "Linear Z", "Angular Z"),
        shared_xaxes="all",
        vertical_spacing=0.08
    )

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.D3
    exp_names = sorted(experiments_dict.keys())
    
    # 1. Plot Measured Data
    for exp_idx, exp_name in enumerate(exp_names):
        data = experiments_dict[exp_name]
        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True
        
        # Linear
        if base_lin_key in data:
            lin_data = get_env_data(data[base_lin_key], env_id)
            if lin_data.size > 0 and lin_data.shape[1] >= 3:
                lin_data = apply_smoothing(lin_data, smoothing)
                for i in range(3):
                    fig.add_trace(
                        go.Scattergl(
                            y=lin_data[:, i], name=f"{exp_name}", line=dict(color=color), 
                            legendgroup=exp_name, showlegend=(i==0), visible=is_visible
                        ),
                        row=i+1, col=1
                    )

        # Angular
        if base_ang_key in data:
            ang_data = get_env_data(data[base_ang_key], env_id)
            if ang_data.size > 0 and ang_data.shape[1] >= 3:
                ang_data = apply_smoothing(ang_data, smoothing)
                for i in range(3):
                    fig.add_trace(
                        go.Scattergl(
                            y=ang_data[:, i], name=f"{exp_name}", line=dict(color=color), 
                            legendgroup=exp_name, showlegend=False, visible=is_visible
                        ),
                        row=i+1, col=2
                    )

    # 2. Plot Commanded Data
    cmd_plotted = False
    is_cmd_visible = 'legendonly' if "command" in hidden_experiments else True
    
    for exp_name in exp_names:
        if cmd_plotted: break
        data = experiments_dict[exp_name]
        cmd_lin_data = get_env_data(data.get(cmd_lin_key), env_id)
        cmd_ang_data = get_env_data(data.get(cmd_ang_key), env_id)
        
        has_lin = (cmd_lin_data.size > 0 and cmd_lin_data.shape[1] >= 2)
        has_ang = (cmd_ang_data.size > 0)
        
        if has_lin or has_ang:
            if has_lin:
                fig.add_trace(go.Scattergl(y=cmd_lin_data[:, 0], name="Cmd Lin X", line=dict(color='black', dash='dash', width=2), legendgroup="command", visible=is_cmd_visible), row=1, col=1)
                fig.add_trace(go.Scattergl(y=cmd_lin_data[:, 1], name="Cmd Lin Y", line=dict(color='black', dash='dash', width=2), legendgroup="command", visible=is_cmd_visible), row=2, col=1)
            
            if has_ang:
                y_data = cmd_ang_data if cmd_ang_data.ndim == 1 else cmd_ang_data[:, 0]
                fig.add_trace(go.Scattergl(y=y_data, name="Cmd Ang Z", line=dict(color='black', dash='dash', width=2), legendgroup="command", visible=is_cmd_visible), row=3, col=2)
            
            cmd_plotted = True

    # cmd_lin_data = None
    # cmd_ang_data = None
    # lin_data = None
    # ang_data = None
    # for exp_name in exp_names:
    #     data = experiments_dict[exp_name]
    #     cmd_lin_data = get_env_data(data.get(cmd_lin_key), env_id)
    #     cmd_ang_data = get_env_data(data.get(cmd_ang_key), env_id)
    #     if base_lin_key in data:
    #         lin_data = get_env_data(data[base_lin_key], env_id)
    #     if base_ang_key in data:
    #         ang_data = get_env_data(data[base_ang_key], env_id)
    #     if lin_data is not None:
    #         print('lin vel x mean error: ', np.sum(np.abs(lin_data[:, 0] - cmd_lin_data[:, 0]))/len(cmd_lin_data[:, 0]))
    #         print('lin vel y mean error: ', np.sum(np.abs(lin_data[:, 1] - cmd_lin_data[:, 1]))/len(cmd_lin_data[:, 0]))
    #     if ang_data is not None:
    #         y_data = cmd_ang_data if cmd_ang_data.ndim == 1 else cmd_ang_data[:, 0]
    #         print('ang vel z mean error: ', np.sum(np.abs(ang_data[:, 2] - y_data))/len(y_data))

    fig.update_layout(height=1800, title_text=f"Base & Commanded Velocities (Env {env_id})")
    
    # Axis Annotations
    # Col 1 (Linear)
    fig.update_yaxes(title_text="Linear Velocity (m/s)", col=1)
    # Col 2 (Angular)
    fig.update_yaxes(title_text="Angular Velocity (rad/s)", col=2)
    # X Axis
    fig.update_xaxes(title_text="Time (steps)")
    
    if zoom_range:
        fig.update_xaxes(range=zoom_range)
        
    return fig

def create_base_position_plot(experiments_dict, env_id, zoom_range=None, smoothing=0.0, hidden_experiments=None):
    """
    Plots the base centre of mass position in the simulation world frame, three rows
    (X, Y, Z) and one column, all experiments overlaid on each row. X and Y carry the
    environment origin offset, so they read as an absolute terrain position and are
    comparable across experiments only where the terrain layout is shared, whereas Z is
    the base height above the world plane and reads directly against the nominal stance.
    """
    hidden_experiments = hidden_experiments or []

    pos_key = "base_com_position"
    axis_titles = ["Base CoM X", "Base CoM Y", "Base CoM Z"]

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=axis_titles,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.D3
    exp_names = sorted(experiments_dict.keys())

    for exp_idx, exp_name in enumerate(exp_names):
        data = experiments_dict[exp_name]
        if pos_key not in data:
            continue

        pos_data = get_env_data(data[pos_key], env_id)
        if pos_data.size == 0 or pos_data.ndim < 2 or pos_data.shape[1] < 3:
            continue

        pos_data = apply_smoothing(pos_data, smoothing)
        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True

        for axis_idx in range(3):
            fig.add_trace(
                go.Scattergl(
                    y=pos_data[:, axis_idx],
                    mode='lines',
                    name=exp_name,
                    legendgroup=exp_name,
                    showlegend=(axis_idx == 0),
                    line=dict(color=color, width=1.5),
                    opacity=0.8,
                    visible=is_visible,
                ),
                row=axis_idx + 1, col=1,
            )

    fig.update_layout(height=900, title_text=f"Base CoM Position (Env {env_id})")
    fig.update_yaxes(title_text="Position (m)")
    fig.update_xaxes(title_text="Time (steps)", row=3)
    if zoom_range:
        fig.update_xaxes(range=zoom_range)
    return fig

def create_feet_plot(experiments_dict, env_id, zoom_range=None, smoothing=0.0, hidden_experiments=None):
    """
    Creates a 6xF grid gathering every per-foot channel into one window: contact force
    magnitude and vertical component, horizontal and vertical foot speed, sole clearance
    and body frame height. One column per foot, one row per quantity, so a single glance
    reads the joint state of a foot. Reading them together is the point: a foot reporting
    contact force while its sole clearance sits well above the ground is forged contact
    (self collision), and a frame height that rises while the sole clearance does not is
    tilt rather than a lift.
    """
    hidden_experiments = hidden_experiments or []

    # Channel definition: (row title, y axis label, extractor over the per-env array).
    # Each extractor takes the (Time, Body, ...) array and returns a (Time,) series.
    channels = [
        ("feet_contact_forces", "Contact Force |F|", "Force (N)", lambda a, f: np.linalg.norm(a[:, f, :], axis=-1)),
        ("feet_contact_forces", "Contact Force Fz", "Force (N)", lambda a, f: a[:, f, 2]),
        ("feet_velocities", "Horizontal Speed |v_xy|", "Velocity (m/s)", lambda a, f: np.linalg.norm(a[:, f, :2], axis=-1)),
        ("feet_velocities", "Vertical Velocity v_z", "Velocity (m/s)", lambda a, f: a[:, f, 2]),
        ("feet_sole_clearances", "Sole Clearance", "Height (m)", lambda a, f: a[:, f]),
        ("feet_frame_heights", "Body Frame Height", "Height (m)", lambda a, f: a[:, f]),
    ]

    feet_names = resolve_feet_names(experiments_dict)
    num_feet = len(feet_names)
    num_rows = len(channels)

    titles = []
    for _, row_title, _, _ in channels:
        for foot_name in feet_names:
            titles.append(f"{row_title} — {foot_name}")

    fig = make_subplots(
        rows=num_rows, cols=num_feet,
        subplot_titles=titles,
        shared_xaxes="all",
        vertical_spacing=0.05,
        horizontal_spacing=0.08,
    )

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.D3
    exp_names = sorted(experiments_dict.keys())

    for exp_idx, exp_name in enumerate(exp_names):
        data = experiments_dict[exp_name]
        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True
        first_trace = True

        for row_idx, (metric_key, _, _, extract) in enumerate(channels):
            if metric_key not in data:
                continue

            metric_data = get_env_data(data[metric_key], env_id)
            if metric_data.size == 0 or metric_data.ndim < 2:
                continue

            for foot_idx in range(min(num_feet, metric_data.shape[1])):
                try:
                    series = extract(metric_data, foot_idx)
                except IndexError:
                    print(f"[WARN] Skipping {metric_key} for {exp_name}, shape {metric_data.shape}")
                    break

                series = np.asarray(apply_smoothing(series, smoothing)).reshape(-1)

                fig.add_trace(
                    go.Scattergl(
                        y=series,
                        mode='lines',
                        name=exp_name,
                        legendgroup=exp_name,
                        showlegend=first_trace,
                        line=dict(color=color, width=1.5),
                        opacity=0.8,
                        visible=is_visible,
                    ),
                    row=row_idx + 1, col=foot_idx + 1
                )
                first_trace = False

    # Reference levels: the contact threshold the gait terms key on, and the clearance target.
    for foot_idx in range(num_feet):
        fig.add_hline(
            y=CONTACT_FORCE_THRESHOLD, line=dict(color='grey', dash='dot', width=1),
            row=1, col=foot_idx + 1
        )
        fig.add_hline(
            y=SOLE_CLEARANCE_TARGET, line=dict(color='grey', dash='dot', width=1),
            row=5, col=foot_idx + 1
        )

    fig.update_layout(height=350 * num_rows, title_text=f"Feet Forces, Velocities & Heights (Env {env_id})")

    for row_idx, (_, _, unit_label, _) in enumerate(channels):
        fig.update_yaxes(title_text=unit_label, row=row_idx + 1)
    fig.update_xaxes(title_text="Time (steps)", row=num_rows)

    if zoom_range:
        fig.update_xaxes(range=zoom_range)

    return fig

def create_torque_velocity_plot(experiments_dict, env_id, smoothing=0.0, hidden_experiments=None):
    """
    Creates a 5x2 grid Scatter plot: Joint Torque (Y) vs Joint Velocity (X).
    """
    hidden_experiments = hidden_experiments or []

    joint_names = resolve_joint_names(experiments_dict)
    num_joints = len(joint_names)
    grid_rows = (num_joints + 1) // 2
    clean_titles = [pretty_joint_name(name) for name in joint_names]

    fig = make_subplots(
        rows=grid_rows, cols=2,
        subplot_titles=clean_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.D3
    exp_names = sorted(experiments_dict.keys())

    for exp_idx, exp_name in enumerate(exp_names):
        data = experiments_dict[exp_name]
        
        if "joint_torques" not in data or "joint_velocities" not in data:
            continue
            
        raw_torques = data["joint_torques"]
        raw_velocities = data["joint_velocities"]
        
        torques = get_env_data(raw_torques, env_id)
        velocities = get_env_data(raw_velocities, env_id)
        
        if torques.size == 0 or velocities.size == 0:
            continue

        if torques.ndim < 2 or torques.shape[1] != num_joints or velocities.shape[1] != num_joints:
            print(f"[WARN] Skipping torque-velocity for {exp_name}, shape mismatch: {torques.shape} vs {velocities.shape}")
            continue

        # Apply Smoothing (to both axes to see filtered trend)
        torques = apply_smoothing(torques, smoothing)
        velocities = apply_smoothing(velocities, smoothing)

        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True

        for j in range(num_joints):
            row = (j // 2) + 1
            col = (j % 2) + 1
            show_leg = (j == 0)

            fig.add_trace(
                go.Scattergl(
                    x=velocities[:, j],
                    y=torques[:, j],
                    mode='markers',
                    marker=dict(size=3, color=color, opacity=0.5), 
                    name=exp_name,
                    legendgroup=exp_name,
                    showlegend=show_leg, 
                    visible=is_visible
                ),
                row=row, col=col
            )

    fig.update_layout(
        height=600 * grid_rows,
        title_text=f"Joint Torque vs Velocity (Env {env_id})",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Velocity (rad/s)")
    fig.update_yaxes(title_text="Torque (Nm)")
    
    return fig

def create_rewards_plot(experiments_dict, env_id, zoom_range=None, smoothing=0.0, hidden_experiments=None):
    """
    Plots all reward terms stored in rewards.npy.
    One subplot per reward key, two columns, all experiments overlaid on each subplot.
    Subplot titles are derived from the reward term keys.
    """
    hidden_experiments = hidden_experiments or []

    # Collect the ordered union of reward keys across all experiments.
    all_reward_keys = []
    for data in experiments_dict.values():
        for k in data.get("_rewards", {}):
            if k not in all_reward_keys:
                all_reward_keys.append(k)

    if not all_reward_keys:
        return go.Figure().update_layout(title="No reward data found (rewards.npy missing or empty)")

    num_cols = 2
    num_rows = (len(all_reward_keys) + num_cols - 1) // num_cols
    subplot_titles = [k.replace("_", " ").title() for k in all_reward_keys]

    # Scale vertical_spacing down with row count so gaps do not consume all the
    # normalised height. With N rows there are N-1 gaps; keeping total gap fraction
    # below 0.4 leaves at least 60% of the figure for the actual plot areas.
    vertical_spacing = min(0.05, 0.4 / max(num_rows - 1, 1))

    fig = make_subplots(
        rows=num_rows, cols=num_cols,
        subplot_titles=subplot_titles,
        shared_xaxes="all",
        vertical_spacing=vertical_spacing,
        horizontal_spacing=0.08,
    )

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.D3
    exp_names = sorted(experiments_dict.keys())

    for exp_idx, exp_name in enumerate(exp_names):
        data = experiments_dict[exp_name]
        reward_dict = data.get("_rewards", {})
        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True
        first_trace = True

        for key_idx, key in enumerate(all_reward_keys):
            if key not in reward_dict:
                continue
            series = get_env_data(reward_dict[key], env_id)
            if series.size == 0:
                continue
            series = np.asarray(apply_smoothing(series, smoothing)).reshape(-1)

            row = (key_idx // num_cols) + 1
            col = (key_idx % num_cols) + 1

            fig.add_trace(
                go.Scattergl(
                    y=series,
                    mode='lines',
                    name=exp_name,
                    legendgroup=exp_name,
                    showlegend=first_trace,
                    line=dict(color=color, width=1.5),
                    opacity=0.8,
                    visible=is_visible,
                ),
                row=row, col=col,
            )
            first_trace = False

    fig.update_layout(height=400 * num_rows, title_text=f"Reward Terms (Env {env_id})")
    fig.update_yaxes(title_text="Reward")
    fig.update_xaxes(title_text="Time (steps)")
    if zoom_range:
        fig.update_xaxes(range=zoom_range)
    return fig


def create_feet_plot_2(experiments_dict, env_id, zoom_range=None, smoothing=0.0, hidden_experiments=None):
    """
    Plots the signed per-axis distance between the two ankles (Link6R minus Link6L)
    over time. Three rows (X, Y, Z), one column. The X axis reflects fore-aft
    separation, Y reflects lateral separation, and Z reflects height difference.
    This mirrors the scalar quantity penalised by feet_distance in mdp/rewards.py,
    but retains the sign and splits the three axes so asymmetries are visible.
    """
    hidden_experiments = hidden_experiments or []

    axis_labels = ["X (fore-aft, m)", "Y (lateral, m)", "Z (height diff, m)", "Total Distance XY", "Total Distance"]

    fig = make_subplots(
        rows=5, cols=1,
        subplot_titles=axis_labels,
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.D3
    exp_names = sorted(experiments_dict.keys())

    for exp_idx, exp_name in enumerate(exp_names):
        data = experiments_dict[exp_name]
        if "feet_distance" not in data:
            continue

        dist_data = get_env_data(data["feet_distance"], env_id)
        if dist_data.size == 0 or dist_data.ndim < 2 or dist_data.shape[1] < 3:
            continue

        dist_data = apply_smoothing(dist_data, smoothing)
        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True

        for axis_idx in range(3):
            fig.add_trace(
                go.Scattergl(
                    y=dist_data[:, axis_idx],
                    mode='lines',
                    name=exp_name,
                    legendgroup=exp_name,
                    showlegend=(axis_idx == 0),
                    line=dict(color=color, width=1.5),
                    opacity=0.8,
                    visible=is_visible,
                ),
                row=axis_idx + 1, col=1,
            )
        fig.add_trace(
            go.Scattergl(
                y=np.sqrt(np.sum(np.square(dist_data[:, :2]), axis = -1)),
                mode='lines',
                name=exp_name,
                legendgroup=exp_name,
                showlegend=(axis_idx == 0),
                line=dict(color=color, width=1.5),
                opacity=0.8,
                visible=is_visible,
            ),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scattergl(
                y=np.sqrt(np.sum(np.square(dist_data), axis = -1)),
                mode='lines',
                name=exp_name,
                legendgroup=exp_name,
                showlegend=(axis_idx == 0),
                line=dict(color=color, width=1.5),
                opacity=0.8,
                visible=is_visible,
            ),
            row=5, col=1,
        )

    fig.update_layout(height=900, title_text=f"Ankle-to-Ankle Distance by Axis (Env {env_id})")
    fig.update_yaxes(title_text="Distance (m)")
    fig.update_xaxes(title_text="Time (steps)", row=3)
    if zoom_range:
        fig.update_xaxes(range=zoom_range)
    return fig


# --- Main Application ---

def create_statistics_table(experiments_dict, group_filter=None, search=""):
    """Two index comparison table, quantity by statistic, one column per experiment.

    The row order is the order in which the module emitted the records for the first
    experiment that carries them, which groups related statistics together and puts the
    families in the order of the module's own assembly rather than alphabetically,
    because a reader scanning for a defect scans by family.
    """
    exp_names = sorted(experiments_dict.keys())
    rows, order = {}, []
    for exp_name in exp_names:
        record_set = experiments_dict[exp_name].get("_statistics")
        if not record_set:
            continue
        for entry in record_set.get("records", []):
            if group_filter and entry.get("group") != group_filter:
                continue
            key = (entry["group"], entry["quantity"], entry["statistic"])
            if key not in rows:
                rows[key] = {
                    "Group": entry["group"],
                    "Quantity": entry["quantity"],
                    "Statistic": entry["statistic"],
                    "Unit": entry.get("unit", ""),
                }
                order.append(key)
            value = entry["value"]
            rows[key][exp_name] = (
                "" if value is None or (isinstance(value, float) and np.isnan(value))
                else f"{value:.4g}"
            )
    if not order:
        return html.Div("No statistics.npy under any experiment for this seed. The "
                        "record set is written by play.py, so replay these runs.")
    records = [rows[key] for key in order]
    if search:
        needle = search.lower()
        records = [
            r for r in records
            if needle in r["Quantity"].lower() or needle in r["Statistic"].lower()
        ]
    columns = [
        {"name": "Quantity", "id": "Quantity"},
        {"name": "Statistic", "id": "Statistic"},
        {"name": "Unit", "id": "Unit"},
    ] + [{"name": e, "id": e} for e in exp_names]
    return dash_table.DataTable(
        data=records,
        columns=columns,
        # The two index columns are frozen so that they remain visible while the
        # experiment columns scroll, which is the whole ergonomic point of the window
        # once more than three experiments are under comparison.
        fixed_columns={"headers": True, "data": 2},
        fixed_rows={"headers": True},
        style_table={"overflowX": "auto", "overflowY": "auto",
                     "maxHeight": "78vh", "minWidth": "100%"},
        style_cell={"fontFamily": "monospace", "fontSize": "12px",
                    "textAlign": "center", "padding": "4px",
                    "minWidth": "150px", "maxWidth": "500px"},
        style_cell_conditional=[
            {"if": {"column_id": c}, "textAlign": "left"}
            for c in ("Quantity", "Statistic", "Unit")
        ],
        style_data_conditional=[
            # Alternate the shading by quantity rather than by row, so that the block of
            # statistics belonging to one quantity reads as one object.
            {"if": {"row_index": "odd"}, "backgroundColor": "#f6f8fa"},
        ],
        style_header={"backgroundColor": "#e8f4f8", "fontWeight": "bold"},
        sort_action="native",
        filter_action="native",
        page_size=200,
    )


def main():
    parser = argparse.ArgumentParser(description="RSL-RL Experiment Dashboard (SD_BRS1)")
    parser.add_argument("log_dir", type=str, help="Path to the directory containing all experiments")
    args = parser.parse_args()

    data_store = load_experiments(args.log_dir)
    
    if not data_store:
        print("[ERROR] No valid data found. Exiting.")
        return

    seeds = sorted(list(data_store.keys()))

    app = dash.Dash(__name__)

    # Build Metric Tabs including new Torque vs Velocity
    metric_tabs_list = [dcc.Tab(label="Base & Commanded", value='base')]
    metric_tabs_list.append(dcc.Tab(label="Base Position", value='base_position'))
    for k, v in JOINT_METRICS.items():
        metric_tabs_list.append(dcc.Tab(label=v, value=k))
    metric_tabs_list.append(dcc.Tab(label="Feet", value='feet'))
    metric_tabs_list.append(dcc.Tab(label="Torque vs Velocity", value='torque_velocity'))
    metric_tabs_list.append(dcc.Tab(label="Rewards", value='rewards'))
    metric_tabs_list.append(dcc.Tab(label="Feet Distance", value='feet_distance'))
    metric_tabs_list.append(dcc.Tab(label="Statistics", value='statistics'))

    # Layout
    app.layout = html.Div([
        dcc.Store(id='zoom-store', storage_type='memory'),
        dcc.Store(id='legend-store', storage_type='memory', data=[]),

        html.H1("RSL-RL Experiment Dashboard — SD_BRS1", style={'textAlign': 'center'}),
        
        html.Div([
            # Row 1: Tabs
            html.Div([
                html.Label("1. Select Seed:", style={'fontWeight': 'bold'}),
                dcc.Tabs(id="seed-tabs", value=seeds[0], children=[
                    dcc.Tab(label=f"Seed {s}", value=s) for s in seeds
                ]),
            ], style={'marginBottom': '15px'}),

            html.Div([
                html.Label("2. Select Metric View:", style={'fontWeight': 'bold'}),
                dcc.Tabs(id="metric-tabs", value='base', children=metric_tabs_list),
            ], style={'marginBottom': '15px'}),
            
            # Row 2: Environment Slider
            html.Div([
                html.Label("3. Select Environment ID:", style={'fontWeight': 'bold'}),
                html.Div([
                    dcc.Slider(
                        id='env-slider',
                        min=0, max=DEFAULT_NUM_ENVS - 1, step=1, value=0,
                        marks={i: str(i) for i in range(0, DEFAULT_NUM_ENVS, 5)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'padding': '0px 20px 20px 20px'})
            ], style={'backgroundColor': '#e8f4f8', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '10px'}),
            
            # Row 3: Smoothing Slider
            html.Div([
                html.Label("4. Select Smoothing (EMA):", style={'fontWeight': 'bold'}),
                html.Div([
                    dcc.Slider(
                        id='smoothing-slider',
                        min=0.0, max=0.99, step=0.01, value=0.0,
                        marks={0: '0 (Raw)', 0.5: '0.5', 0.99: '0.99 (Smooth)'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'padding': '0px 20px 20px 20px'})
            ], style={'backgroundColor': '#e8f4f8', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'})

        ], style={'padding': '10px', 'backgroundColor': '#f9f9f9', 'margin': '10px'}),

        html.Div([
            dcc.Loading(id="loading-plot", type="default", children=dcc.Graph(id="main-graph", style={'height': '85vh'}))
        ]),

        html.Div([
            dcc.Input(id="stat-search", type="text", debounce=True, value="",
                      placeholder="filter by quantity or statistic",
                      style={'width': '40%', 'marginBottom': '8px'}),
            dcc.Dropdown(id="stat-group", value=None, placeholder="all groups",
                         options=[{"label": g, "value": g} for g in (
                             "tracking", "temporal", "impact", "swing", "feet", "stance",
                             "joints", "energetics", "posture", "stability", "symmetry",
                             "smoothness", "variability", "odometry", "rewards")],
                         style={'width': '40%', 'marginBottom': '8px'}),
            html.Div(id="statistics-table"),
        ], id="statistics-panel", style={'display': 'none', 'padding': '10px'}),
    ])

    # Callbacks
    @app.callback(
        Output('zoom-store', 'data'),
        Input('main-graph', 'relayoutData'),
        State('zoom-store', 'data'),
        prevent_initial_call=True
    )
    def store_zoom(relayout_data, current_zoom):
        if not relayout_data: return dash.no_update
        for key, value in relayout_data.items():
            if 'xaxis' in key and key.endswith('.range[0]'):
                prefix = key.rsplit('.', 1)[0]
                upper_key = f"{prefix}.range[1]"
                if upper_key in relayout_data: return [value, relayout_data[upper_key]]
        for key, value in relayout_data.items():
            if 'xaxis' in key and key.endswith('.range'): return value
        for key in relayout_data.keys():
            if 'xaxis' in key and 'autorange' in key: return None
        return dash.no_update

    @app.callback(
        Output('legend-store', 'data'),
        Input('main-graph', 'restyleData'),
        [State('main-graph', 'figure'), State('legend-store', 'data')],
        prevent_initial_call=True
    )
    def update_legend_store(restyle_data, current_figure, hidden_list):
        if not restyle_data or not current_figure: return dash.no_update
        hidden_set = set(hidden_list or [])
        update_dict, indices = restyle_data[0], restyle_data[1]
        if 'visible' not in update_dict: return dash.no_update
        vis_value = update_dict['visible']
        vis_values = [vis_value] * len(indices) if not isinstance(vis_value, list) else vis_value
        for i, idx in enumerate(indices):
            if idx >= len(current_figure['data']): continue
            group_name = current_figure['data'][idx].get('legendgroup')
            if not group_name: continue
            if vis_values[i] == 'legendonly': hidden_set.add(group_name)
            elif vis_values[i] is True:
                if group_name in hidden_set: hidden_set.remove(group_name)
        return list(hidden_set)

    @app.callback(
        Output("main-graph", "figure"),
        [Input("seed-tabs", "value"),
         Input("metric-tabs", "value"),
         Input("env-slider", "value"),
         Input("smoothing-slider", "value")],
        [State("zoom-store", "data"),
         State("legend-store", "data")]
    )
    def update_graph(seed, metric_type, env_id, smoothing, zoom_range, hidden_experiments):
        if not seed: return go.Figure().update_layout(title="Select a Seed")
        try:
            experiments_dict = data_store[seed]
        except KeyError:
            return go.Figure().update_layout(title="Data not found for selection")

        if metric_type == 'base':
            return create_base_plot(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments)
        elif metric_type == 'base_position':
            return create_base_position_plot(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments)
        elif metric_type == 'feet':
            return create_feet_plot(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments)
        elif metric_type == 'torque_velocity':
            # Do not pass Time-based zoom range to Velocity axis
            return create_torque_velocity_plot(experiments_dict, env_id, smoothing, hidden_experiments)
        elif metric_type == 'rewards':
            return create_rewards_plot(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments)
        elif metric_type == 'feet_distance':
            return create_feet_plot_2(experiments_dict, env_id, zoom_range, smoothing, hidden_experiments)
        elif metric_type in JOINT_METRICS:
            return create_joint_plot(experiments_dict, metric_type, env_id, zoom_range, smoothing, hidden_experiments)
        else:
            return go.Figure().update_layout(title="Unknown Metric Selected")

    # The statistics view returns an HTML component rather than a figure, so it is given
    # its own callback and its own output area rather than being folded into update_graph.
    # Every branch of that callback constructs a figure, so returning a table from one of
    # them would change the output type of the whole callback and therefore every branch,
    # which is a larger blast radius than the feature warrants.
    @app.callback(
        [Output("statistics-table", "children"),
         Output("statistics-panel", "style"),
         Output("main-graph", "style")],
        [Input("seed-tabs", "value"), Input("metric-tabs", "value"),
         Input("stat-search", "value"), Input("stat-group", "value")],
    )
    def update_statistics(seed, metric_type, search, group):
        hidden, shown = {'display': 'none'}, {'display': 'block'}
        if metric_type != 'statistics' or not seed:
            return dash.no_update, hidden, {'height': '85vh'}
        return (create_statistics_table(data_store[seed], group, search or ""),
                {'display': 'block', 'padding': '10px'}, hidden)

    print("[INFO] Starting Dash Server...")
    app.run(debug=True, host='0.0.0.0', port=8051)

if __name__ == "__main__":
    main()
