import argparse
import glob
import numpy as np
import os

import dash
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
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

# Joint Names based on Robot Config (Interleaved L/R)
JOINT_NAMES = [
    "abad_L_Joint",
    "abad_R_Joint",
    "hip_L_Joint",
    "hip_R_Joint",
    "knee_L_Joint",
    "knee_R_Joint",
    "ankle_L_Joint",
    "ankle_R_Joint",
]

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
            
            data_store[seed][exp_name] = data_dict
            print(f"[LOADED] Seed: {seed} | Exp: {exp_name}")
            
        except Exception as e:
            print(f"[ERROR] Processing {file_path}: {e}")

    return data_store

def get_env_data(raw_data, env_id):
    """Extract one environment's series from a dump channel.

    Every per step channel is a stacked array of shape (Time, Env, ...), so this is a
    slice on the environment axis and nothing more. CHANGED, the previous form also
    accepted the list of tensors that DataLogger.plot used to write and the singleton
    wrap it put in front of most channels. Neither is written any longer, and a dump
    that still carries them predates the pipeline and must be replayed to be read here.
    The torch dependency went with that branch, the dump having held numpy arrays since
    long before this change.

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
    Creates a 4x2 grid for a SINGLE joint metric, comparing all experiments.
    """
    hidden_experiments = hidden_experiments or []
    
    clean_titles = [name.replace("_", " ").title() for name in JOINT_NAMES]

    fig = make_subplots(
        rows=4, cols=2,
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
            
        if metric_data.ndim < 2 or metric_data.shape[1] != 8:
            print(f"[WARN] Skipping {metric_key} for {exp_name}, shape mismatch: {metric_data.shape}")
            continue
        
        metric_data = apply_smoothing(metric_data, smoothing)
        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True
        
        for j in range(8):
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
    fig.update_layout(height=2000, title_text=f"{pretty_name} (Env {env_id})")
    
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

def create_torque_velocity_plot(experiments_dict, env_id, smoothing=0.0, hidden_experiments=None):
    """
    Creates a 4x2 grid Scatter plot: Joint Torque (Y) vs Joint Velocity (X).
    """
    hidden_experiments = hidden_experiments or []
    
    clean_titles = [name.replace("_", " ").title() for name in JOINT_NAMES]

    fig = make_subplots(
        rows=4, cols=2,
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
            
        # Apply Smoothing (to both axes to see filtered trend)
        torques = apply_smoothing(torques, smoothing)
        velocities = apply_smoothing(velocities, smoothing)

        color = colors[exp_idx % len(colors)]
        is_visible = 'legendonly' if exp_name in hidden_experiments else True
        
        for j in range(8):
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
        height=2400, 
        title_text=f"Joint Torque vs Velocity (Env {env_id})",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Velocity (rad/s)")
    fig.update_yaxes(title_text="Torque (Nm)")
    
    return fig

# --- Main Application ---

def main():
    parser = argparse.ArgumentParser(description="RSL-RL Experiment Dashboard")
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
    for k, v in JOINT_METRICS.items():
        metric_tabs_list.append(dcc.Tab(label=v, value=k))
    metric_tabs_list.append(dcc.Tab(label="Torque vs Velocity", value='torque_velocity'))

    # Layout
    app.layout = html.Div([
        dcc.Store(id='zoom-store', storage_type='memory'),
        dcc.Store(id='legend-store', storage_type='memory', data=[]),

        html.H1("RSL-RL Experiment Dashboard", style={'textAlign': 'center'}),
        
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
        ])
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
        elif metric_type == 'torque_velocity':
            # Do not pass Time-based zoom range to Velocity axis
            return create_torque_velocity_plot(experiments_dict, env_id, smoothing, hidden_experiments)
        elif metric_type in JOINT_METRICS:
            return create_joint_plot(experiments_dict, metric_type, env_id, zoom_range, smoothing, hidden_experiments)
        else:
            return go.Figure().update_layout(title="Unknown Metric Selected")

    print("[INFO] Starting Dash Server...")
    app.run(debug=True, host='0.0.0.0', port=8051)

if __name__ == "__main__":
    main()
