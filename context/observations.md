# Quadruped Observation Layout

This document registers every observation group of the quadruped environment configuration, the size of each term, its index range in the flattened tensor the networks receive, and the file that implements it. It was written on 2026-11-19 to resolve a NaN reported by the co-optimisation run, and it is the reference for reading any `NaN in <name> at indices` report printed by `report_nan` in `co_optimisation/co_optimisation/modules/copt_actor_critic.py:39`.

## 1. Conventions

The quadruped has 12 actuated joints, 25 bodies and 4 feet, the joint count being the width of the action vector and of the `mean` and `std` NaN reports, and the body count being fixed by the 258 channel morphology width recorded in `context/quadruped.md` section 7. The eight link lengths are the four thighs and four shanks, as set by `parent_body_names` and `child_body_names` at `environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:436`.

A group that sets `history_length` has that length applied to every term by `ObservationManager`, which overrides the term's own value (`../IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:550`). Each term then keeps its own circular buffer, ordered oldest to newest, and is flattened on its own (`observation_manager.py:409`). The history layout is therefore term major, the block of one term holding its ten steps back to back before the next term begins, and it is not step major. The flattened block of a term of width `d` occupies `10 d` consecutive indices, and step `k` of the term, `k = 0` being the oldest, occupies the `d` indices starting at the term's first index plus `k d`.

Terms are concatenated in the order they are declared in the group class (`observation_manager.py:433`). The tables below list each term in that order, and the index column is the inclusive range in the concatenated tensor of the group. Every size below is read from the term function's return shape, and the one size that is known only at runtime, the collision shape count of `robot_material_properties`, is written `S` and carried symbolically, since `get_material_properties` returns three values per collision shape per environment (`environments/environments/tasks/locomotion/mdp/observations.py:119`). It can be read with `env.unwrapped.observation_manager.group_obs_dim["critic"]`, and every critic range after the material term is quoted as an offset in `S`.

The `mdp` namespace of the configuration is built by `environments/environments/tasks/locomotion/mdp/__init__.py:1`, which first re-exports the Isaac Lab velocity task `mdp`, itself re-exporting `isaaclab.envs.mdp`, and then the local modules. A name defined in both resolves to the local one, which affects only `generated_commands`, defined at `environments/environments/tasks/locomotion/mdp/observations.py:207` with the same behaviour as the Isaac Lab version at `../IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py:675`.

The file paths in the tables are abbreviated as follows. `IL` is `../IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py` and `LOC` is `environments/environments/tasks/locomotion/mdp/observations.py`.

## 2. The reported NaN

The report in the failing run names five tensors, and each is read against the layout of section 3 below.

| Report | Indices | Reading |
|---|---|---|
| `encoderOut` | 0 to 215 | The estimator output, 216 channels, is NaN in every channel. |
| normalised actor input | 480 to 695 | The last 216 of 696 channels, which are the `encoderOut` block appended to the 480 channel `policy` group. The first 480 channels are finite. |
| `mean`, `std` | 0 to 11 | The 12 action channels, NaN because the actor input is NaN. |

The actor input is 696 wide because `CoptMorALActorCritic` appends `encoderOut` to `obs_groups["policy"]` (`co_optimisation/co_optimisation/modules/copt_actor_critic.py:92`), and the runner that gives an estimator with a 216 channel output is `PFQuadrupedCoptMorALPPORunnerCfg`, whose `obs_groups` are `policy`, `critic`, `encoderIn = ["policy"]` and `gtEncoderOut = ["morphologyObs"]` (`environments/environments/tasks/locomotion/agents/quadruped_rsl_rl_ppo_cfg.py:92`). The estimator input is the raw, unnormalised `policy` group (`copt_actor_critic.py:134`) and the output width is that of `morphologyObs`, 216, which is the width counted in section 5.

The normalised actor input is NaN only in 480 to 695, and `act` runs `_report_nan` over the raw groups plus `encoderOut` (`copt_actor_critic.py:175`), which printed only `encoderOut`, while the normalised actor input, `mean` and `std` are reported separately by `_update_distribution` (`copt_actor_critic.py:158`). The observation terms of the policy group are therefore finite, and no observation term is the source. The NaN is produced inside the estimator, either by NaN parameters left by an earlier optimiser step or by an overflow in its forward pass, and it enters the actor only through the appended block. The `std` failure in the traceback is a consequence, since a NaN `std` fails the `std >= 0.0` check of `torch.distributions.Normal.sample`.

Were the same indices met under `PFQuadrupedCoptPPORunnerCfg` (`quadruped_rsl_rl_ppo_cfg.py:70`), whose `policy` set is `["policy", "morphologyObs"]` with no estimator, indices 480 to 695 would be the morphology group itself, and the table of section 5 gives the term at each of them. That variant prints no `encoderOut` report, so it is not the run in question.

For the `MorAL` variant, the index of `encoderOut` is also the index of the regression target, since the estimator is regressed onto `morphologyObs` (`copt_actor_critic.py:137`). Output channel `i` therefore corresponds to the morphology channel at the same index, listed in section 5.

Correction of 2026-11-20. The reading of this section is superseded. A later crash of the same task, printed with the weight checks of `copt_actor_critic.py`, reported `feet_lin_vel` NaN in `critic` at 720 to 839 and in `privilegedDynamicsObs` at 36 to 47 on the step of a reset, a NaN `value`, and then every parameter of the actor, the estimator and the critic NaN after one optimiser step, `clip_grad_norm_` spreading the NaN total norm to every gradient. The `encoderOut` report of 2026-11-19 was therefore the end of that chain and not its source, an observation term being the source after all. The analysis is [isaaclab_diagnostics.md](isaaclab_diagnostics.md) section 7a.

## 3. ObservationsCfg

Defined at `environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:143` and used by `QuadrupedPFEnvCfg` (`base_env_cfg.py:1524`). It has three groups.

### 3.1 policy

`PolicyCfg`, `base_env_cfg.py:147`. History 10, flattened, corruption enabled (`base_env_cfg.py:199`). Width 48 per step and 480 in total.

| Term | Size per step | Size with history | Indices | Implementation |
|---|---|---|---|---|
| `base_lin_vel` | 3 | 30 | 0 to 29 | `IL:54` |
| `base_ang_vel` | 3 | 30 | 30 to 59 | `IL:64` |
| `proj_gravity` | 3 | 30 | 60 to 89 | `IL:74` |
| `joint_pos` | 12 | 120 | 90 to 209 | `IL:212` |
| `joint_vel` | 12 | 120 | 210 to 329 | `IL:257` |
| `last_action` | 12 | 120 | 330 to 449 | `IL:657` |
| `velocity_commands` | 3 | 30 | 450 to 479 | `LOC:207` |

### 3.2 critic

`CriticCfg`, `base_env_cfg.py:266`. History 10, flattened, corruption disabled (`base_env_cfg.py:341`). Width `578 + 3S` per step, the sum of the twenty terms below, and `5780 + 30S` in total. The eight link lengths and eight nominal pose channels of the co-optimisation critic are absent, so this critic is the 48 channel policy state, the privileged terms and the height scan.

| Term | Size per step | Size with history | Indices | Implementation |
|---|---|---|---|---|
| `base_lin_vel` | 3 | 30 | 0 to 29 | `IL:54` |
| `base_ang_vel` | 3 | 30 | 30 to 59 | `IL:64` |
| `proj_gravity` | 3 | 30 | 60 to 89 | `IL:74` |
| `joint_pos` | 12 | 120 | 90 to 209 | `IL:212` |
| `joint_vel` | 12 | 120 | 210 to 329 | `IL:257` |
| `last_action` | 12 | 120 | 330 to 449 | `IL:657` |
| `velocity_commands` | 3 | 30 | 450 to 479 | `LOC:207` |
| `robot_joint_torque` | 12 | 120 | 480 to 599 | `LOC:28` |
| `robot_joint_acc` | 12 | 120 | 600 to 719 | `LOC:37` |
| `feet_lin_vel` | 12 | 120 | 720 to 839 | `LOC:198` |
| `robot_mass` | 25 | 250 | 840 to 1089 | `LOC:54` |
| `robot_inertia` | 225 | 2250 | 1090 to 3339 | `LOC:64` |
| `robot_joint_pos` | 12 | 120 | 3340 to 3459 | `LOC:74` |
| `robot_joint_stiffness` | 12 | 120 | 3460 to 3579 | `LOC:83` |
| `robot_joint_damping` | 12 | 120 | 3580 to 3699 | `LOC:92` |
| `robot_pos` | 3 | 30 | 3700 to 3729 | `LOC:101` |
| `robot_vel` | 6 | 60 | 3730 to 3789 | `LOC:110` |
| `robot_material_properties` | 3S | 30S | 3790 to 3789 + 30S | `LOC:119` |
| `feet_contact_force` | 12 | 120 | 3790 + 30S to 3909 + 30S | `LOC:139` |
| `heights` | 187 | 1870 | 3910 + 30S to 5779 + 30S | `IL:292` |

The height scan is the 1.6 m by 1.0 m grid at 0.1 m resolution of `height_scanner` (`base_env_cfg.py:75`), which is 17 by 11, or 187, rays.

### 3.3 commands

`CommandsObsCfg`, `base_env_cfg.py:349`. No history. Width 3.

| Term | Size | Indices | Implementation |
|---|---|---|---|
| `velocity_commands` | 3 | 0 to 2 | `LOC:207` |

## 4. HIMObservationsCfg

Defined at `base_env_cfg.py:680` and used by `QuadrupedPFHIMEnvCfg` (`base_env_cfg.py:1555`). It has six groups, and the base linear velocity is withheld from the actor and given to the estimator as a target.

### 4.1 policy

`PolicyCfg`, `base_env_cfg.py:684`. No history, corruption enabled. Width 45.

| Term | Size | Indices | Implementation |
|---|---|---|---|
| `base_ang_vel` | 3 | 0 to 2 | `IL:64` |
| `proj_gravity` | 3 | 3 to 5 | `IL:74` |
| `joint_pos` | 12 | 6 to 17 | `IL:212` |
| `joint_vel` | 12 | 18 to 29 | `IL:257` |
| `last_action` | 12 | 30 to 41 | `IL:657` |
| `velocity_commands` | 3 | 42 to 44 | `LOC:207` |

### 4.2 obsHistory

`HistoryObsCfg`, `base_env_cfg.py:737`. History 10, not flattened (`base_env_cfg.py:790`), so the group tensor is shaped `(num_envs, 10, 45)` and is indexed on the last axis exactly as the policy group above. Each step carries the same six terms with the same ranges as section 4.1, and the group has `10 x 45 = 450` values per environment.

### 4.3 critic

`CriticCfg`, `base_env_cfg.py:843`. History 10, flattened, corruption disabled. Width `578 + 3S` per step, the height scan being placed immediately after the command, and `5780 + 30S` in total.

| Term | Size per step | Size with history | Indices | Implementation |
|---|---|---|---|---|
| `base_lin_vel` | 3 | 30 | 0 to 29 | `IL:54` |
| `base_ang_vel` | 3 | 30 | 30 to 59 | `IL:64` |
| `proj_gravity` | 3 | 30 | 60 to 89 | `IL:74` |
| `joint_pos` | 12 | 120 | 90 to 209 | `IL:212` |
| `joint_vel` | 12 | 120 | 210 to 329 | `IL:257` |
| `last_action` | 12 | 120 | 330 to 449 | `IL:657` |
| `velocity_commands` | 3 | 30 | 450 to 479 | `LOC:207` |
| `heights` | 187 | 1870 | 480 to 2349 | `IL:292` |
| `robot_joint_torque` | 12 | 120 | 2350 to 2469 | `LOC:28` |
| `robot_joint_acc` | 12 | 120 | 2470 to 2589 | `LOC:37` |
| `feet_lin_vel` | 12 | 120 | 2590 to 2709 | `LOC:198` |
| `robot_mass` | 25 | 250 | 2710 to 2959 | `LOC:54` |
| `robot_inertia` | 225 | 2250 | 2960 to 5209 | `LOC:64` |
| `robot_joint_pos` | 12 | 120 | 5210 to 5329 | `LOC:74` |
| `robot_joint_stiffness` | 12 | 120 | 5330 to 5449 | `LOC:83` |
| `robot_joint_damping` | 12 | 120 | 5450 to 5569 | `LOC:92` |
| `robot_pos` | 3 | 30 | 5570 to 5599 | `LOC:101` |
| `robot_vel` | 6 | 60 | 5600 to 5659 | `LOC:110` |
| `robot_material_properties` | 3S | 30S | 5660 to 5659 + 30S | `LOC:119` |
| `feet_contact_force` | 12 | 120 | 5660 + 30S to 5779 + 30S | `LOC:139` |

### 4.4 commands

`CommandsObsCfg`, `base_env_cfg.py:921`. No history. Width 3, a single `velocity_commands` term at indices 0 to 2 (`LOC:207`).

### 4.5 estimatorGT

`EstimatorGTCfg`, `base_env_cfg.py:927`. No history, and it sets no `__post_init__`. Width 3.

| Term | Size | Indices | Implementation |
|---|---|---|---|
| `base_lin_vel` | 3 | 0 to 2 | `IL:54` |

### 4.6 targetEnc

`TargetEncCfg`, `base_env_cfg.py:794`. No history, corruption enabled but with no noise configured, so the values are clean. Width 48.

| Term | Size | Indices | Implementation |
|---|---|---|---|
| `base_ang_vel` | 3 | 0 to 2 | `IL:64` |
| `proj_gravity` | 3 | 3 to 5 | `IL:74` |
| `joint_pos` | 12 | 6 to 17 | `IL:212` |
| `joint_vel` | 12 | 18 to 29 | `IL:257` |
| `last_action` | 12 | 30 to 41 | `IL:657` |
| `velocity_commands` | 3 | 42 to 44 | `LOC:207` |
| `base_lin_vel` | 3 | 45 to 47 | `IL:54` |

## 5. CoptObservationsCfg

Defined at `base_env_cfg.py:371` and used by `QuadrupedPFCoptEnvCfg` (`base_env_cfg.py:1587`). It declares seven group classes and attaches six of them. `HistoryObsCfg` is defined but its attribute is commented out at `base_env_cfg.py:676`, so it is not built.

### 5.1 policy

`PolicyCfg`, `base_env_cfg.py:375`. Identical in terms, order, scales and noise to section 3.1, with history 10, flattened. Width 48 per step and 480 in total, with the ranges of the table in section 3.1. This is the `policy` group of the failing run, and it is also the estimator input of `PFQuadrupedCoptMorALPPORunnerCfg`.

### 5.2 morphologyObs

`MorphologyCfg`, `base_env_cfg.py:430`. No history, corruption enabled but no noise configured. Width 216. The three moved body terms select the twenty bodies named by `_COPT_MOVED_BODIES` (`base_env_cfg.py:359`), which are `abad_.*_act`, `hip_.*_act`, `knee_.*_act`, `thigh_.*_Link` and `shank_.*_Link`, four of each.

| Term | Size | Indices | Implementation |
|---|---|---|---|
| `link_lengths` | 8 | 0 to 7 | `LOC:233` |
| `robot_mass` | 20 | 8 to 27 | `LOC:54` |
| `robot_inertia` | 180 | 28 to 207 | `LOC:64` |
| `nominal_pose` | 8 | 208 to 215 | `LOC:276` |

The inertia size is 20 bodies of 9 components. This table is also the meaning of each `encoderOut` index, and of indices 480 to 695 of the actor input under `PFQuadrupedCoptPPORunnerCfg`, where the morphology block starts at 480 and the index in the table is the actor index minus 480. Actor indices 480 to 487 are then the link lengths, 488 to 507 the masses, 508 to 687 the inertias and 688 to 695 the nominal pose.

### 5.3 privilegedDynamicsObs

`PrivilegedDynamicsCfg`, `base_env_cfg.py:479`. No history, corruption disabled. Width 48. Under the runner configurations of `quadruped_rsl_rl_ppo_cfg.py` no `obs_groups` entry names it, so it is built and computed but read by nothing.

| Term | Size | Indices | Implementation |
|---|---|---|---|
| `robot_joint_torque` | 12 | 0 to 11 | `LOC:28` |
| `robot_joint_acc` | 12 | 12 to 23 | `LOC:37` |
| `feet_contact_force` | 12 | 24 to 35 | `LOC:139` |
| `feet_lin_vel` | 12 | 36 to 47 | `LOC:198` |

### 5.4 critic

`CriticCfg`, `base_env_cfg.py:500`. History 10, flattened, corruption disabled. It is the critic of section 3.2 with `link_lengths` and `nominal_pose` appended after `heights`. Width `594 + 3S` per step and `5940 + 30S` in total. The terms up to and including `feet_contact_force` carry the ranges of the table in section 3.2, and the remaining three are as follows.

| Term | Size per step | Size with history | Indices | Implementation |
|---|---|---|---|---|
| `heights` | 187 | 1870 | 3910 + 30S to 5779 + 30S | `IL:292` |
| `link_lengths` | 8 | 80 | 5780 + 30S to 5859 + 30S | `LOC:233` |
| `nominal_pose` | 8 | 80 | 5860 + 30S to 5939 + 30S | `LOC:276` |

In this critic `robot_mass` and `robot_inertia` take no `asset_cfg`, so they cover all 25 bodies, 25 and 225 channels, unlike the 20 body selection of `morphologyObs`.

### 5.5 commands

`CommandsObsCfg`, `base_env_cfg.py:666`. No history. Width 3, a single `velocity_commands` term at indices 0 to 2 (`LOC:207`).

### 5.6 historyObs, defined and not attached

`HistoryObsCfg`, `base_env_cfg.py:611`, with its attribute commented out at `base_env_cfg.py:676`. It is recorded because attaching it is the evident use of the `encoderIn` set. History 25, flattened, corruption enabled. Width 235 per step and 5875 in total, which is the estimator input width in `context/co_optimisation_audit.md` section 12.

| Term | Size per step | Size with history | Indices | Implementation |
|---|---|---|---|---|
| `base_lin_vel` | 3 | 75 | 0 to 74 | `IL:54` |
| `base_ang_vel` | 3 | 75 | 75 to 149 | `IL:64` |
| `proj_gravity` | 3 | 75 | 150 to 224 | `IL:74` |
| `joint_pos` | 12 | 300 | 225 to 524 | `IL:212` |
| `joint_vel` | 12 | 300 | 525 to 824 | `IL:257` |
| `last_action` | 12 | 300 | 825 to 1124 | `IL:657` |
| `velocity_commands` | 3 | 75 | 1125 to 1199 | `LOC:207` |
| `heights` | 187 | 4675 | 1200 to 5874 | `IL:292` |

## 6. Final network inputs per runner

The table gives the tensor each network reads for each runner configuration of the quadruped, as assembled by `CoptMorALActorCritic` and `ActorCritic` from `obs_groups`.

| Runner | Network | Groups concatenated | Width | Index map |
|---|---|---|---|---|
| `PFQuadrupedCoptMorALPPORunnerCfg` | estimator input | `policy` | 480 | section 5.1 |
| `PFQuadrupedCoptMorALPPORunnerCfg` | estimator target | `morphologyObs` | 216 | section 5.2 |
| `PFQuadrupedCoptMorALPPORunnerCfg` | actor input | `policy`, `encoderOut` | 696 | 0 to 479 section 5.1, 480 to 695 the estimator output, one to one with section 5.2 |
| `PFQuadrupedCoptMorALPPORunnerCfg` | critic input | `critic` | `5940 + 30S` | section 5.4 |
| `PFQuadrupedCoptPPORunnerCfg` | actor input | `policy`, `morphologyObs` | 696 | 0 to 479 section 5.1, 480 to 695 section 5.2 |
| `PFQuadrupedCoptPPORunnerCfg` | critic input | `critic` | `5940 + 30S` | section 5.4 |

The actor and critic observation normalisers are applied to the whole concatenated vector (`copt_actor_critic.py:169`), so a NaN in a single channel of a normalised tensor is a NaN in that channel of the raw tensor or in the normaliser statistics of that channel.

## 7. Open points

The collision shape count `S` is read at runtime and is not recorded in this repository for the quadruped, so the critic ranges after `robot_material_properties` are stated as offsets. The `S` of the generated MJCF is the number to add to this document once a run prints `group_obs_dim["critic"]`, after which the offsets can be evaluated.

The cause of the NaN in the estimator is not established here. This document establishes only that no observation term is the source. The two candidates that the layout leaves open are NaN estimator parameters after an optimiser step, the estimator mean squared error being added to the PPO loss at `copt_ppo.py:260` with `estimator_loss_coef` of 1.0 by default (`copt_ppo.py:64`), the audit entry D2 having recorded that loss as unscaled, and an overflow in the 216 channel regression target, whose inertia channels span several orders of magnitude, and which `clip=(0.0, 100.0)` bounds only for `link_lengths` and `robot_mass` (`base_env_cfg.py:458`).
