# Cost of Transport Energy Regularisation for the Quadruped, and a Foot Contact Schedule Panel

Status, IMPLEMENTED 2026-10-21. The reward function, the `RewardsCfg` term and the dashboard panel landed verbatim as section 4 gives them, and the verification of section 6 was discharged except for the visual inspection of the rendered panel, recorded in section 7. The training run the change exists for has not yet been launched.

## 1. Context

Run `quadruped_flat/2026-10-20_10-59-00` trained `Isaac-Quadruped-Blind-Flat-v0` for 8000 iterations with symmetry on, and it learned a bouncing trot. Its play dump `data/42/dump.npy` holds 3001 steps of 32 environments. Taking contact as a peak force magnitude above 1 N, which is `stats.CONTACT_FORCE_THRESHOLD`, the dump measures as follows.

| Quantity | Value |
|---|---|
| All four feet airborne | 34.5 % of steps |
| Exactly two feet loaded | 57.5 % |
| Duty factor FR, FL, RR, RL | 0.39, 0.39, 0.26, 0.27 |
| Mean commanded and achieved forward speed | 0.64 and 0.60 m/s |
| Mean joint power, positive and absolute | 61 W and 127 W |
| CoT above 0.2 m/s, positive and absolute convention | 0.69 and 1.43 |

Liang et al. [1], read from the arXiv HTML rendering, combine their terms as `R = (R_motion + α_en R_en) · exp(-R_aux)` in Eq. 2. The motion reward is `R_motion = R_lin + 0.5 R_ang`. The energy reward of Eq. 4 is `R_en = exp(-Σ|τ_i||q̇_i| / (σ_en,x |v_x| + σ_en,z |ω_z|))`, with σ_en,x = 1000 and σ_en,z = 500, and v_x and ω_z are the measured motion of the robot. They choose α_en = 1.0. At 0.0 and 0.5 the regularisation is minor and the robot bounces over the whole velocity range (Sec. IV-B), and at 1.5 tracking accuracy is lost (Fig. 3).

## 2. Evaluation of the current formulation

The figures are the run's `rewards.npy`, a rate per second for each term.

- The positive budget is 14.67 per second. Linear tracking holds 61 % of it, angular tracking 33 % and `rew_foot_clearance` 6 %.
- The negative budget is 1.85 per second, 13 % of the positive.
- The energy proxies are `pen_joint_torque` at 6.6 % of the negative budget, `pen_joint_accel` at 9.8 % and `pen_joint_vel_l2` at 0.6 %. Together they come to 0.31 per second, 2 % of the positive budget. This is the regime of α_en near zero, which the paper reports produces exactly this bouncing gait [1], and it agrees with the finding of Fu et al. that a recognisable gait emerges only under an energy term [2].
- In the paper's product form the auxiliary factor scales motion and energy alike. The ratio of α_en to the motion reward therefore sets the energy preference, and an additive form preserves that ratio. The additive formulation is kept, as the user directed.
- `rew_foot_clearance`, at 0.87 per second, pays every airborne moving foot. A flight phase collects it on all four feet at once. This flight positive incentive is recorded and left standing.

### 2.1 The weight

- The tracking weights here are 10 for linear and 5 for angular velocity. That is the paper's 1 to 0.5 scaled by ten, so the paper's α_en of 1.0 maps to a weight of 10.0.
- The paper's bracket maps the same way. Weights up to 5 bounced, 10 was chosen, and 15 degraded tracking.
- At the run's gait the exponent is near 127 / (1000 × 0.6) ≈ 0.21, giving R_en ≈ 0.81 and a contribution of about 8 per second. Halving the power gains about 1 per second, which is more than the whole present energy budget of 0.31 per second. The energy preference therefore becomes real without displacing tracking.
- The sigmas stay at the paper's 1000 and 500. The paper does not state the Go1 mass against which they were set, so no mass rescale is drawn from recall.

## 3. Callers of the changed configuration

Rule 3 of CLAUDE.md requires the callers of the shared class to be listed. `RewardsCfg` in `cfg/quadruped/base_env_cfg.py:1088` is read by the following.

- `QuadrupedPFEnvCfg` (line 1516) serves the blind flat and rough tasks and the Go1 asset ablation `QuadrupedPFGo1AssetBaseEnvCfg`.
- `QuadrupedPFHIMEnvCfg` (line 1547) serves the HIM flat and rough tasks.
- `CoptRewardsCfg(RewardsCfg)` (line 1231) serves the three co-optimisation tasks.

All of them gain the term, as the user directed. `Go1NativeMatchedEnvCfg` is IsaacLab's own configuration and is unaffected. The new term reads only root velocity and all joints, so it resolves on the Go1 asset too. Historical runs are protected at replay, because `experiment_params.apply_reward_cfg` (`scripts/analysis/experiment_params.py:208`) sets to None every term the tree carries that the run's `params/env.yaml` does not. A replay of `2026-10-20_10-59-00` therefore keeps its trained reward set. The defect left standing is that a fresh run of any of these tasks now trains under the energy term, and `params/env.yaml` records that it did.

## 4. Changes

### 4.1 The reward function

In `environments/environments/tasks/locomotion/mdp/rewards.py`, insert the following immediately after `joint_powers_l1` (line 1156), which it reuses. It is exported through the existing `from .rewards import *` in `mdp/__init__.py`, and no existing function changes.

```python
def cost_of_transport_exp(
    env: ManagerBasedRLEnv,
    sigma_lin: float = 1000.0,
    sigma_ang: float = 500.0,
    min_denominator: float = 1.0e-6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocity normalised energy reward, Eq. 4 of Liang et al., arXiv 2403.20001.

    The absolute joint power over a weighted sum of the measured forward and yaw speeds,
    a cost of transport whose denominator also credits turning in place, mapped through
    a negative exponential into (0, 1].

    The denominator is the MEASURED speed, as the paper takes it, so the reward vanishes
    at standstill and a standing command earns nothing from it. The power is summed over
    every joint, joint_powers_l1 ignoring asset_cfg.joint_ids.

    Args:
        env: The environment object.
        sigma_lin: Scale of the forward speed in the denominator (W s/m).
        sigma_ang: Scale of the yaw rate in the denominator (W s/rad).
        min_denominator: Floor on the denominator, guarding the division at standstill.
        asset_cfg: Configuration for the robot asset.

    Returns:
        The computed reward tensor.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    power = joint_powers_l1(env, asset_cfg)
    speed = (
        sigma_lin * torch.abs(asset.data.root_lin_vel_b[:, 0])
        + sigma_ang * torch.abs(asset.data.root_ang_vel_b[:, 2])
    )
    return torch.exp(-power / torch.clamp(speed, min=min_denominator))
```

### 4.2 The reward configuration

In `cfg/quadruped/base_env_cfg.py`, inside `RewardsCfg`, insert the following immediately after `rew_ang_vel_z` (line 1109). No other class is edited and no task is registered.

```python
    # Velocity normalised energy reward, Eq. 4 of Liang et al., arXiv 2403.20001. Their
    # alpha_en of 1.0 is carried by the factor of ten that maps their 1 to 0.5 linear to
    # angular tracking weights onto the 10 to 5 above. Their 0.5 still bounced and their
    # 1.5 lost tracking. Run 2026-10-20_10-59-00 spent 2 % of its positive budget on
    # energy and flew for 34.5 % of its steps.
    rew_energy_cot = RewTerm(
        func=mdp.cost_of_transport_exp,
        weight=10.0,
        params={"sigma_lin": 1000.0, "sigma_ang": 500.0},
    )
```

The next experiment then trains with the existing ids, `Isaac-Quadruped-Blind-Flat-v0`, and plays with `Isaac-Quadruped-Blind-Flat-Play-v0` through `scripts/rsl_rl/play.py`, which needs no change. The run lands under `logs/rsl_rl/quadruped_flat` beside the baseline.

### 4.3 The foot contact schedule

In `scripts/analysis/dashboard.py`, insert the following immediately before `create_feet_plot` (line 744).

```python
# Contact bar colours by foot index, after the red, olive, blue and green of the contact
# schedule figure of Liang et al., arXiv 2403.20001. A biped takes the first two.
CONTACT_SCHEDULE_COLORS = ("#d62728", "#bcbd22", "#1f77b4", "#2ca02c")


def _contact_runs(contact):
    """Start and length, in steps, of every run of True in a one dimensional series."""
    edges = np.diff(np.concatenate(([0], contact.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    return starts, np.flatnonzero(edges == -1) - starts


def create_contact_schedule_plot(experiments_dict, env_id, zoom_range=None,
                                 hidden_experiments=None, num_legs=2, profile=DEFAULT_PROFILE):
    """The foot contact schedule over the whole dump, one row per foot, one panel per run.

    A foot is in contact where the logged peak contact force exceeds the profile's
    threshold, the test stats.contact_state applies, so the schedule agrees with the duty
    factor and flight fraction of the statistics table. Each panel title carries the duty
    factor of every foot and the flight fraction, the share of steps on which no foot is
    loaded, and dotted verticals mark the episode resets. The last panel is the commanded
    against the measured forward speed, so that a change of gait is read against the speed
    at which it occurs, as the velocity axis of the paper's figure is.
    """
    hidden_experiments = hidden_experiments or []

    feet_names = resolve_feet_names(experiments_dict, profile)
    if not feet_names:
        width = channel_width(experiments_dict, "feet_contact_forces") or num_legs
        feet_names = [f"foot {i}" for i in range(width)]
    exp_names = [n for n in sorted(experiments_dict)
                 if n not in hidden_experiments and "feet_contact_forces" in experiments_dict[n]]
    if not exp_names:
        return go.Figure().update_layout(title="No feet contact forces in any visible dump for this seed")

    titles, schedules = [], []
    for exp_name in exp_names:
        forces = get_env_data(experiments_dict[exp_name]["feet_contact_forces"], env_id)
        if forces.ndim != 3:
            schedules.append(None)
            titles.append(f"{exp_name} · no contact forces for env {env_id}")
            continue
        contact = np.linalg.norm(forces, axis=-1) > profile.contact_force_threshold  # (T, F)
        schedules.append(contact)
        duty = ", ".join(f"{n} {d:.2f}" for n, d in zip(feet_names, contact.mean(axis=0)))
        flight = 100.0 * float((~contact.any(axis=1)).mean())
        titles.append(f"{exp_name} · duty {duty} · flight {flight:.1f}%")
    titles.append("Forward Velocity")

    num_rows = len(exp_names) + 1
    fig = make_subplots(
        rows=num_rows, cols=1,
        subplot_titles=titles,
        shared_xaxes=True,
        vertical_spacing=min(0.06, 0.3 / num_rows),
        row_heights=[len(feet_names)] * len(exp_names) + [3],
    )

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.D3

    for row, (exp_name, contact) in enumerate(zip(exp_names, schedules), start=1):
        data = experiments_dict[exp_name]
        if contact is not None:
            steps = contact.shape[0]
            for foot_idx, foot_name in enumerate(feet_names[:contact.shape[1]]):
                color = CONTACT_SCHEDULE_COLORS[foot_idx % len(CONTACT_SCHEDULE_COLORS)]
                # The light band is the whole dump, so a gap in it reads as the foot aloft.
                fig.add_trace(
                    go.Bar(x=[steps], y=[foot_name], base=[0], orientation="h", width=0.6,
                           marker_color=color, opacity=0.15, showlegend=False, hoverinfo="skip"),
                    row=row, col=1,
                )
                starts, lengths = _contact_runs(contact[:, foot_idx])
                fig.add_trace(
                    go.Bar(x=lengths, y=[foot_name] * len(starts), base=starts, orientation="h",
                           width=0.6, marker_color=color, showlegend=False, name=foot_name,
                           hovertemplate="from step %{base} for %{x} steps<extra>%{y}</extra>"),
                    row=row, col=1,
                )
            fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(feet_names)),
                             row=row, col=1)
            # Drawn after the bars, add_vline dropping a line on a panel that holds no trace.
            for step in np.flatnonzero(get_env_data(data.get("episode_dones"), env_id)):
                fig.add_vline(x=float(step), line=dict(color="grey", dash="dot", width=1),
                              row=row, col=1)

        color = colors[(row - 1) % len(colors)]
        measured = get_env_data(data.get("base_linear_velocity"), env_id)
        if measured.ndim == 2:
            fig.add_trace(
                go.Scattergl(y=measured[:, 0], mode="lines", name=f"{exp_name} measured",
                             legendgroup=exp_name, line=dict(color=color, width=1.5)),
                row=num_rows, col=1,
            )
        commanded = get_env_data(data.get("commanded_linear_velocity"), env_id)
        if commanded.ndim == 2:
            fig.add_trace(
                go.Scattergl(y=commanded[:, 0], mode="lines", name=f"{exp_name} commanded",
                             legendgroup=exp_name, line=dict(color=color, width=1.2, dash="dash")),
                row=num_rows, col=1,
            )

    fig.update_layout(
        barmode="overlay",
        height=45 * len(feet_names) * len(exp_names) + 120 * num_rows + 200,
        title_text=f"Foot Contact Schedule (Env {env_id})",
    )
    fig.update_yaxes(title_text="Velocity (m/s)", row=num_rows, col=1)
    fig.update_xaxes(title_text="Time (steps)", row=num_rows, col=1)
    if zoom_range:
        fig.update_xaxes(range=zoom_range)
    return fig
```

In `main` of the same file, insert the tab immediately after the "Feet" tab (line 1219).

```python
    metric_tabs_list.append(dcc.Tab(label="Contact Schedule", value='contact_schedule'))
```

In `update_graph`, insert the branch immediately after the `'feet'` branch (line 1360).

```python
        elif metric_type == 'contact_schedule':
            return create_contact_schedule_plot(experiments_dict, env_id, zoom_range,
                                                hidden_experiments, num_legs=profile.num_legs,
                                                profile=profile)
```

## 5. Records required by CLAUDE.md

No code is involved in these.

- Add this document's status row to `plans/README.md`.
- Add a dated section 13 to `context/quadruped.md` with the measured flight fraction, the budget shares, the weight derivation and the caller list above. Add `rew_energy_cot` to its term table in section 3.2 as a dated addition. Record that the tree's `pen_base_height` of -15 at 0.35 m differs from this run's -5 at 0.37 m, so the next run is not a single variable comparison against `2026-10-20_10-59-00`.
- Add the Liang et al. entry to cluster 19 of `/ws/context/literature.md`, beside the Fu et al. energy entry. Give the author list as retrieved and the venue as arXiv only.
- Move `/ws/foot_contact.png` to `../../context/artefacts/liang2024-energy-regularisation/foot_contact.png`, register it in `/ws/context/artefacts/README.md` as the paper's contact schedule figure supplied by the user, and cite it from the context section.
- Add a line to the dashboard consumer section of `/ws/context/gait_metrics.md`.

## 6. Verification

1. Check the function against a stub environment of `SimpleNamespace` tensors in the dev container. Values should lie in (0, 1], zero velocity should give 0 without NaN, and 127 W at 0.6 m/s should give about 0.81.
2. Smoke test training through `docker exec` into `isaac-lab-base-shreyas-gpu0`, from a cwd holding no `environments/` folder, with `python scripts/rsl_rl/train.py --task Isaac-Quadruped-Blind-Flat-v0 --headless --num_envs 64 --max_iterations 3`. Confirm that `Episode_Reward/rew_energy_cot` is logged and that `params/env.yaml` records the term with weight 10 and both sigmas, then delete the smoke run.
3. Smoke test the replay of `2026-10-20_10-59-00` with `play.py --task Isaac-Quadruped-Blind-Flat-Play-v0 --checkpoint_path .../model_8000.pt --video_length 50`. The report should list `rew_energy_cot` as removed, which confirms that the historical run replays under its own reward set. Write this to a scratch log directory rather than over the run's `data/42`.
4. In the dev container, import `dashboard`, load `/ws/IsaacLab/logs/rsl_rl/quadruped_flat/2026-10-20_10-59-00` and one biped dump, call `create_contact_schedule_plot` and write the HTML to the scratchpad. Confirm four rows for the quadruped and two for the biped, and a pooled flight fraction near 34.5 %. Then launch `dashboard.py <dir> --robot quadruped` and inspect the tab.

## 7. Outcome, 2026-10-21

The three code changes landed exactly as section 4 states them, and nothing diverged from the plan. One point of the plan was revised before implementation at the user's direction. The first draft added a separate reward class and a separately registered task, and the term now lives in the shared `RewardsCfg` instead, so that the next run trains under the existing identifiers.

The verification of section 6 was discharged as follows.

1. The function body was extracted from the live file and evaluated with stub tensors in `isaac-lab-base-shreyas-gpu0`, the Isaac Lab import requiring a running application. It returned 0.809 at 127 W and 0.6 m/s, 0.0 at standstill with no NaN, 0.905 for 50 W turning at 1 rad/s, and 1.0 at zero power.
2. A 64 environment, three iteration run of `Isaac-Quadruped-Blind-Flat-v0` in `isaac-lab-base-shreyas-gpu1` listed `rew_energy_cot` at weight 10.0 in the reward manager table. It logged `Episode_Reward/rew_energy_cot` and dumped the term to `params/env.yaml` with both sigmas. The smoke run was then deleted.
3. `model_8000.pt` and `params/` of `2026-10-20_10-59-00` were replayed from a copy in the smoke directory, so that the run's own `data/42` was not overwritten. The replay reported 22 terms applied and 1 removed, and its `rewards.npy` carried no `rew_energy_cot`, which confirms that a historical replay keeps its trained reward set.
4. `create_contact_schedule_plot` rendered from the quadruped dump with four rows, and from two TRON1 dumps of `sf_tron_1a_flat` with two rows each. Environment 0 of the quadruped run reads duty factors of 0.38, 0.40, 0.32 and 0.36 and a flight fraction of 33.4 %, against the 34.5 % pooled over all 32 environments. The rendered HTML was not inspected by eye, since neither container carries an image exporter, and the user should confirm the layout on first use of the tab.

## 8. Bibliography

1. Liang, B., Sun, L., Zhu, X., Zhang, B., Xiong, Z., Wang, Y., Li, C., Sreenath, K., Tomizuka, M. Adaptive Energy Regularization for Autonomous Gait Transition and Energy-Efficient Quadruped Locomotion. arXiv preprint, revised 5 March 2025. arXiv:2403.20001. No peer reviewed venue was stated by the retrieved source.
2. Fu, Kumar, Malik, Pathak. Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots. Conference on Robot Learning, 2021. arXiv:2111.01674. Taken from cluster 19 of `../../context/literature.md`, which gives the surnames only, so the initials are omitted.
