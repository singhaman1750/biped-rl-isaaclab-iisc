# ARCHITECTURE.md: Isaac Lab & RSL-RL Task Integration

This project utilizes a modular, inheritance-based configuration system to define bipedal locomotion tasks, bridging **Isaac Lab** (simulation/MDP) with **RSL-RL** (RL algorithms/runners).

## 1. Directory Roles & Responsibilities

| Directory/File | Responsibility |
| :--- | :--- |
| [`scripts/rsl_rl/train.py`](scripts/rsl_rl/train.py) | **Entry Point:** Parses `--task`, initializes the simulator, and invokes the RL Runner. |
| [`environments/environments/tasks/locomotion/mdp/`](environments/environments/tasks/locomotion/mdp/) | **MDP Logic:** Definition of Reward functions, Observation terms, and Events. |
| [`environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`](environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py) | **MDP Templates:** Defines the robot-specific observation, reward, and action spaces. |
| [`environments/environments/tasks/locomotion/robots/limx_solefoot_env_cfg.py`](environments/environments/tasks/locomotion/robots/limx_solefoot_env_cfg.py) | **Scenario & Asset Config:** Defines USD asset paths, joint positions, and terrain scenarios (Flat, Rough, etc.). |
| [`environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`](environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py) | **RL Hyperparameters:** Configures PPO algorithm settings and Actor-Critic network dimensions. |
| [`environments/environments/tasks/locomotion/robots/__init__.py`](environments/environments/tasks/locomotion/robots/__init__.py) | **The Registry:** Maps Task IDs to environment and agent configuration classes. |
| [`scripts/rsl_rl/play.py`](scripts/rsl_rl/play.py) | **Evaluation:** Script to load a trained checkpoint and visualize the policy. |
| [`scripts/rsl_rl/cli_args.py`](scripts/rsl_rl/cli_args.py) | **Arguments:** Centralized definition of command-line arguments for training and playback. |

## 2. Class Hierarchy & Relationships

The configuration follows a hierarchical structure using Isaac Lab's `@configclass` decorator.

### Environment Configuration Chain (Sole-Foot)
1.  **`ManagerBasedRLEnvCfg`** (external: `isaaclab.envs`): Foundation class.
2.  **`SFEnvCfg`** ([`limx_base_env_cfg.py`](environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py)): Robot-specific MDP template (Scene, Rewards, Observations, Actions).
3.  **`SFBaseEnvCfg`** ([`limx_solefoot_env_cfg.py`](environments/environments/tasks/locomotion/robots/limx_solefoot_env_cfg.py)): Foundation for scenarios. Defines `SOLEFOOT_CFG` asset and default joint positions.
4.  **`SFBlindFlatEnvCfg`** ([`limx_solefoot_env_cfg.py`](environments/environments/tasks/locomotion/robots/limx_solefoot_env_cfg.py)): Final leaf class. Overrides parent to disable height scanners and set "plane" terrain.

### Environment Configuration Chain (Quadruped)

The quadruped mirrors the chain above at two levels beneath each base class rather than four, having no USD variant and therefore needing no layer to switch between URDF and USD. Every class carries a `Quadruped` prefix, because the names the chain would otherwise take are already held by the LimX TRON1 pointfoot biped and `cfg/__init__.py` flattens every family's public names into one namespace.

1.  `ManagerBasedRLEnvCfg` (external, `isaaclab.envs`), the foundation class.
2.  `QuadrupedPFEnvCfg`, `QuadrupedPFHIMEnvCfg` and `QuadrupedPFCoptEnvCfg` ([`base_env_cfg.py`](environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py)), one MDP template per runner, differing only in their observation class.
3.  `QuadrupedPFBaseEnvCfg` and its two siblings ([`quadruped_pointfoot_env_cfg.py`](environments/environments/tasks/locomotion/robots/quadruped_pointfoot_env_cfg.py)), which attach `QUADRUPED_IDENTIFIED_CFG`, the nominal crouch, and a viewer framing scaled to a 0.292 metre robot.
4.  `QuadrupedPFBlindFlatEnvCfg` and `QuadrupedPFBlindRoughEnvCfg`, with their `_PLAY` counterparts, the leaf classes that select the terrain.

The star import in `cfg/__init__.py` places `quadruped` first rather than last. Nine classes in each family module carry generic names that already collide three ways among the biped families, and placing the quadruped first leaves every existing resolution exactly as it stood, adding six names and displacing none.

The physical parameterisation, the derived actuator gains and the reward configuration are recorded in [context/quadruped.md](context/quadruped.md), [context/joint_control_analysis_quadruped.md](context/joint_control_analysis_quadruped.md) and [context/quadruped_xml_to_urdf_conversion.md](context/quadruped_xml_to_urdf_conversion.md).

## 3. Reward Definition and Formulation

The reward functions are defined in `environments/environments/tasks/locomotion/mdp/rewards.py`. The total reward is a weighted sum of several terms, designed to encourage stable and efficient locomotion.

### Key Reward Terms:

-   **Velocity Tracking:**
    -   `rew_lin_vel_xy`: Rewards matching the target linear velocity in the x-y plane.
    -   `rew_ang_vel_z`: Rewards matching the target angular velocity around the z-axis.
-   **Penalties for undesired behavior:**
    -   `pen_lin_vel_z`: Penalizes vertical velocity.
    -   `pen_ang_vel_xy`: Penalizes angular velocity in the x-y plane.
    -   `pen_action_rate`: Penalizes large changes in actions between consecutive timesteps.
    -   `pen_flat_orientation`: Penalizes deviation from a flat orientation.
    -   `pen_undesired_contacts`: Penalizes contacts with parts of the robot other than the feet.
    -   `joint_powers_l1`: Penalizes high joint power consumption.
-   **Gait and Foot Placement:**
    -   `GaitReward`: A custom reward class that encourages a specific foot contact pattern based on a given gait command (frequency, offset, duration). It uses a von Mises distribution to create a smooth reward signal for being in the correct phase of the gait.
    -   `foot_landing_vel`: Penalizes high foot velocities upon landing.
    -   `feet_distance`: Penalizes if the distance between feet is too small or too large.
    -   `nominal_foot_position`: Rewards keeping the feet at a nominal position relative to the base.
-   **Stability and balance:**
    -   `unbalance_feet_air_time`: Penalizes large variance in the air time of the feet.
    -   `base_height_rough_l2`: Penalizes deviation from a target height, even on rough terrain.
    -   `stay_alive`: A constant reward for not terminating the episode.

The weights for these reward terms are specified in the environment configuration files (e.g., `PFBlindStairEnvCfg` in `limx_pointfoot_env_cfg.py`).

## 4. Data Flow & Environment Management

The project uses a "Manager-Based" architecture within the `ManagerBasedRLEnv` to orchestrate the simulation loop.

-   **`ObservationManager`**: Computes the policy and critic observations based on the `ObservationsCfg`.
-   **`RewardManager`**: Calculates individual reward terms and their weighted sum as defined in `RewardsCfg`.
-   **`ActionManager`**: Processes the actions output by the policy and applies them to the robot's actuators.
-   **`RslRlVecEnvWrapper`**: A critical shim that wraps the Isaac Lab environment to make it compatible with the expected input/output format of the RSL-RL library.

### Training Loop
1.  `train.py` creates the environment via `gym.make(task_id)`.
2.  The environment is wrapped by `RslRlVecEnvWrapper`.
3.  An `OnPolicyRunner` (or `HIMOnPolicyRunner`) is instantiated.
4.  `runner.learn()` is called, which handles data collection (rollouts) and PPO updates.

## 5. Environment Registration

Tasks are registered in [`robots/__init__.py`](environments/environments/tasks/locomotion/robots/__init__.py) using `gym.register`.

```python
gym.register(
    id="Isaac-Limx-SF-HIM-v0",             # CLI Task Name
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": SFHIMBlindFlatEnvCfg,  # From limx_solefoot_env_cfg.py
        "rsl_rl_cfg_entry_point": SF_TRON1AFlatPPORunnerCfg(), # From limx_rsl_rl_ppo_cfg.py
    },
)
```

### The Quadruped Task Registry

Twelve identifiers, written as explicit `gym.register` blocks in the same style as the biped tasks.

| Identifier | Configuration class | Runner |
|---|---|---|
| `Isaac-Quadruped-Blind-Flat-v0`, `-Play-v0` | `QuadrupedPFBlindFlatEnvCfg` | vanilla |
| `Isaac-Quadruped-Blind-Rough-v0`, `-Play-v0` | `QuadrupedPFBlindRoughEnvCfg` | vanilla |
| `Isaac-Quadruped-HIM-Blind-Flat-v0`, `-Play-v0` | `QuadrupedPFHIMBlindFlatEnvCfg` | `HIMOnPolicyRunner` |
| `Isaac-Quadruped-HIM-Blind-Rough-v0`, `-Play-v0` | `QuadrupedPFHIMBlindRoughEnvCfg` | `HIMOnPolicyRunner` |
| `Isaac-Quadruped-Copt-Flat-v0` | `QuadrupedPFCoptBlindFlatEnvCfg` | `CoptOnPolicyRunner` |
| `Isaac-Quadruped-Copt-Rough-v0`, `-Play-v0` | `QuadrupedPFCoptBlindRoughEnvCfg` | `CoptOnPolicyRunner` |
| `Isaac-Quadruped-Copt-Learned-Rough-v0` | `QuadrupedPFCoptBlindRoughEnvCfg` | `CoptOnPolicyRunner`, learned model |

The four co-optimisation identifiers are registered but must not yet be launched, the design generator not yet accepting this robot and `scripts/rsl_rl/train.py` hardcoding the biped URDF at lines 198 to 210. The vanilla and hybrid internal model identifiers need no change to `train.py` whatever.

## 6. Steps to Create a New Task

1.  **Define MDP Logic:** If unique observations or rewards are needed, update robot `limx_base_env_cfg.py`.
2.  **Create Scenario Class:** In `limx_solefoot_env_cfg.py`, define a class inheriting from `SFBaseEnvCfg`. Use `__post_init__` for overrides.
3.  **Configure Agent:** Define a `RunnerCfg` in `limx_rsl_rl_ppo_cfg.py` for PPO hyperparameters.
4.  **Register Task:** Add a `gym.register` block in `robots/__init__.py`.
5.  **Select Runner:** Update `scripts/rsl_rl/train.py` if a custom runner (e.g., [`HIMOnPolicyRunner`](himloco/himloco/runners/him_on_policy_runner.py)) is required.

## 6. HIM Architecture Summary
*   **Observations:** Requires `policy` (1-step) and `obsHistory` (e.g., 25-step) groups.
*   **Format:** `flatten_history_dim` must be `True`.
*   **Network:** Uses [`HIMActorCritic`](himloco/himloco/modules/him_actor_critic.py) with a dedicated estimator.
*   **Toggle:** Switch via `--task Isaac-Limx-SF-HIM-v0 --policy_type HIMPPO`.
