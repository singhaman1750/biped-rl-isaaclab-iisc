# KScale Biped Integration, Implementation Plan

> Status, MIXED. Chapters 1 to 4 are IMPLEMENTED 2026-08-24, every section of chapter 4 having been carried out, and the outcome together with seven divergences from what this document proposed is recorded in section 7 at the foot of the page. Chapter 5, KBot Specific Reward Tuning, is IMPLEMENTED 2026-08-26 and has now been RUN AND MEASURED THREE TIMES, in `2026-08-26_08-08-13`, `2026-08-31_04-57-06` and `2026-09-01_06-53-24`, whose outcomes are recorded in sections 7.9, 7.10 and 7.11. The sequence establishes that pricing the foot splay does not remove the off axis excursion but relocates it, the hip yaw excursion having fallen by a third between the two later runs while the hip roll excursion rose by nearly three quarters and their sum stood still, and that the best gait of the four was produced by a two group reweighting that promoted the posture, style and regularisation terms by a factor near 3.3 relative to the task and gait shaping terms. A fourth pass is OUTSTANDING and BLOCKED, the working tree failing to import at `cfg/SF/kscale_base_env_cfg.py:795`, where a keyword argument is written inside a dictionary literal. The third literature pass is appended to section 5.1.1, the revised terms, rationale, budget, instructions and ablation sequence to the existing subsections of section 5.1.2, and the central proposal is now to express the stance width requirement at the outcome rather than at the joint and to prefer a barrier form over a flat price. The physical parameterisation this document establishes has been promoted to [../context/KScale.md](../context/KScale.md), which supersedes section 3 wherever the two disagree. Chapter 6, added 2026-09-09, is OUTSTANDING and orders the experiments that follow the two runs of 2026-09-08, whose failure is traced in the correction of 2026-09-09 to [../context/KScale.md](../context/KScale.md) to a mismatch between the actuator's `velocity_limit` and the `velocity_limit_sim` of 10 rad per second the solver enforces, inherited unset from a URDF placeholder. See [README.md](README.md) for the register.

This document is the implementation brief for bringing the KScale biped to parity with the SD_BRS1 biped, which this repository refers to throughout as the BRS. The KScale configuration entered the tree as a direct copy of the BRS configuration taken at an earlier point in its development, retargeted onto a different set of link and joint names, and it has since fallen behind the BRS on every axis the gait work stream advanced, the sole aware clearance and landing rewards, the impact penalty, the graced single support term, the symmetry augmentation, and the location of the sources themselves. The object of this plan is that the KScale environment and agent configuration mimic the BRS, differing only where the two robots genuinely differ, and that the BRS, the TRON1 SoleFoot, PointFoot and WheelFoot tasks, and the quadruped are left bit for bit unaltered.

The plan carries its whole codebase investigation as validation material, so that the implementing agent needs no further exploration, and it states the design rationale for every proposed value rather than only the value. Its conclusions rest on four parallel investigations recorded during the writing of this document, on direct measurement of the KScale meshes, and on the accumulated record of the BRS gait work stream in `../../context/brs_gait.md` and the workspace plans it indexes.

## 1. Introduction

The two robots are both sole footed bipeds of six actuated degrees of freedom per leg, arranged in the same order from the torso outward, a hip pitch, a hip roll, a hip yaw, a knee, an ankle pitch and an ankle roll, terminating in a flat plate that meets the ground over an area rather than at a point. That shared topology is what makes the BRS configuration a legitimate starting point for the KScale, and it is why the original port succeeded in producing a configuration that at least reads as coherent. The resemblance is close enough that a reader comparing the two environment configuration files side by side finds them structurally identical, term for term, differing only in the regular expressions that name the bodies and the joints.

The resemblance is nonetheless superficial in three respects that this plan exists to address, and each of the three defeats a different assumption the original port made.

The first is scale. The KScale masses 20.281 kg against the BRS at 59.85 kg, a ratio near one third, and its hip to ankle leg length is 0.505 m against 0.87 m. Every reward parameter carrying units of length, force or height is therefore wrong by roughly that ratio when copied across unchanged, and several were copied across unchanged. The consequences are not uniform, since a Gaussian kernel width and a contact force threshold scale differently, so the correction has to be made term by term against the physics of each rather than by applying one factor throughout.

The second is geometry. The BRS carries a near rectangular sole plate whose lowest surface sits 0.124 m below the ankle frame, and the reward set that the gait work stream converged upon depends on knowing the shape of that plate, not merely its depth. The KScale sole is a tapered blade of a different shape at a different depth, and, decisively, its foot link frame is a full axis permutation away from the BRS convention, its own y axis pointing downward where the BRS uses z. The original port read the minimum of the foot mesh's link frame z coordinate, obtained 0.19, compared it against the BRS figure of 0.124, found the discrepancy inexplicable, and recorded its own suspicion in a comment at the point of use. That suspicion was correct. The figure 0.19 is the foot's length and not its depth, and the four reward terms that depend on the sole were dropped rather than shipped against a number nobody trusted, which is the immediate reason the KScale reward set is four terms short of the BRS.

The third, and the most consequential, is the frame convention of the root link itself, which no comment in the tree records and which the original port did not detect. The KScale root frame is rotated by ninety degrees about its vertical axis relative to the convention Isaac Lab assumes, so the axis the configuration commands as forward velocity is in fact the robot's lateral axis. This is established in section 3.2 and its remedy is section 4.2. It is stated here, at the head of the document, because it governs the interpretation of everything that follows, a velocity command, a height scanner footprint, a symmetry mirror plane and a foot separation measurement all being expressed in that frame.

The document proceeds by establishing the differences between the two robots in detail, then by reporting the measurement of the KScale URDF and its meshes together with the script that performs it, and finally by setting out the changes, each with the rationale that justifies it.

## 2. BRS and KScale Biped Comparison

### 2.1 Morphological Differences

The KScale is the smaller robot throughout, and the ratios are not uniform across its segments, which matters because a uniform scale factor would otherwise be a legitimate shortcut for the reward parameters.

| Quantity | BRS | KScale | Ratio | Source |
|---|---|---|---|---|
| Total mass | 59.85 kg | 20.281 kg | 0.339 | `../context/BRS.md:33`, `assets/urdf/solefoot/kscale/kscale.urdf` inertial blocks |
| Torso mass | not restated here | 4.319 kg | | `kscale.urdf:7-11` |
| Thigh length, hip yaw to knee | see `../context/BRS.md:18` | 0.21417 m | | `kscale.urdf` joint origins |
| Shank length, knee to ankle | see `../context/BRS.md:19` | 0.29051 m | | `kscale.urdf` joint origins |
| Hip to ankle leg length | 0.87 m | 0.505 m | 0.580 | `../context/BRS.md:18-19` |
| Nominal standing height | 1.15 m | 0.7953 m | 0.692 | `cfg/SF/brs_base_env_cfg.py:750`, measured at the KScale nominal pose |
| Sole depth below the ankle frame | 0.124 m | 0.0430 m | 0.347 | `../../context/brs_gait.md:89`, section 3.3 below |
| Sole length | 0.2612 m | 0.2100 m | 0.804 | `cfg/SF/brs_base_env_cfg.py:644-657`, section 3.3 below |
| Sole width | 0.194 m | 0.0846 m | 0.436 | as above |
| Foot lateral separation at the nominal pose | 0.259 m | 0.252 m | 0.973 | `../../context/literature.md` cluster 13 correction, section 3.2 below |

Three observations follow from the table and each bears on a later section.

The sole depth scales with the mass ratio, 0.347 against 0.339, whereas the sole length scales far more weakly, 0.804, and the standing height weakly again, 0.692. The KScale therefore stands on a foot that is nearly as long as the BRS foot but less than half as wide and sits at a third of the height beneath the ankle. A parameter derived from the depth may be scaled by the mass ratio with some confidence, and a parameter derived from the footprint may not.

The foot lateral separation is very nearly the same on the two robots, 0.252 m against 0.259 m, despite the threefold mass difference. The KScale is a narrow robot in the fore and aft sense and a comparatively broad one across the hips, which means the stance width parameters transfer across almost unchanged where the height and force parameters do not.

The hip yaw joint is live on the KScale and mechanically absent on the BRS. The BRS declares both hip yaw joints with `type="fixed"` in its URDF, so they never enter `robot.joint_names` at all, and the KScale declares both as revolute over the full range of plus and minus 1.5708 rad at `kscale.urdf:614` and `:1094`. The KScale consequently actuates twelve joints where the BRS actuates ten, which propagates into the observation width, the action width and the symmetry permutation alike. A note in the tree describes the BRS hip yaw as disabled by a zero width limit rather than as a fixed joint, and that description is inaccurate, the correction being recorded in section 4.9.

The joint limits of the KScale URDF carry a uniform effort of 10 Nm and a uniform velocity of 10 rad/s on every joint without exception, from the hip pitch that carries the whole leg to the ankle roll that carries only the foot. This is the signature of an unedited export default rather than a measured hardware limit, and it is consistent with the docstring at `assets/config/kscale_identified_cfg.py:11-20` recording that the robot arrived with no actuator specification. The limits that actually bind in simulation come from the per joint actuator configuration rather than from the URDF, which is fortunate, but the URDF figures should not be mistaken for data.

### 2.2 Reward Functions

The BRS reward set carries twenty six live terms and the KScale twenty three, and the difference is not a simple subset relation. The KScale is missing four BRS terms and carries one term the BRS deliberately removed.

The four missing terms are `rew_keep_ankle_roll_zero_in_air`, `rew_foot_clearance`, `pen_foot_landing_vel` and `pen_feet_impact`. Three of the four depend on the sole geometry, which is why they were dropped together, and the comment at `cfg/SF/kscale_base_env_cfg.py:809-816` states as much, recording that a per vertex table of the true lowest points of the sole mesh had not been measured and that the term would be restored once it had. That comment is an accurate account of why the terms are absent, and section 3 supplies the missing measurement.

The one extra term is `pen_ankle_deviation`, a `joint_deviation_l1` over the four ankle joints at weight minus 0.1. The BRS carries the same term commented out at `cfg/SF/brs_base_env_cfg.py:738-746` at weight minus 0.2. The history is that the term was introduced on the BRS as an experiment against the defect of the ankles resting at their mechanical stops, recorded at `../../plans/GAIT_EFFICIENCY_PLAN.md:9`, and was subsequently removed for producing no observable effect on policy performance. The KScale revives it at half the BRS weight, which is to say it revives an experiment the BRS abandoned, and section 4.5 removes it.

Two further differences are behavioural rather than structural.

The first is that `rew_no_fly` calls the free function `no_fly` on the KScale and the stateful class `NoFlyWithGrace` on the BRS. The distinction is the grace window. A single support reward without a window prices every instant of double support at zero, and therefore drives the number of support transitions downward, which is the opposite of what a walking gait requires, since the double support interval is where weight transfer occurs and occupies roughly 24 percent of the human gait cycle [10]. Van Marum and colleagues define their single foot contact term as awarding credit if single contact occurred at least once in the preceding 0.2 s, so that a brief double support during transfer is not punished [6]. That window is twenty control steps at this task's 0.01 s period, which is deeper than the four sample contact sensor history, and is the reason the term had to become a stateful class rather than acquire an argument. The migration is nonetheless free of risk, because `NoFlyWithGrace` with `grace_steps` set to zero is bit for bit identical to `no_fly` at the same history index, a fact recorded at `../../context/brs_gait.md:593`.

The second is that `rew_keep_ankle_pitch_zero_in_air` on the KScale omits four of the six parameters the BRS passes. The BRS sets `history_index` to 0, `force_threshold` to 1.0, `pitch_scale` to 0.2 and `use_default_offset` to False, and the KScale sets only `require_airborne`. The `history_index` omission is the material one. The contact sensor writes its newest sample to index 0, so the function's default of minus 1 reads the oldest of the four buffered samples, roughly 15 ms stale, and the comment at `cfg/SF/brs_base_env_cfg.py:681-684` records exactly this. The KScale therefore grades its ankle posture against a contact state three control steps out of date, and `rew_no_fly` does so as well, its `history_index` of 0 being the one parameter of the BRS term the KScale did carry across.

The remaining terms are common to both configurations and differ only in their parameters, which section 2.5 tabulates.

### 2.3 Agent Configuration

`KscaleFlatPPORunnerCfg` at `tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py:296-331` and `SD_BRS1FlatPPORunnerCfg` at `:241-282` agree on every hyperparameter of the algorithm and on the shape of every network. The value loss coefficient, the clipping parameter, the entropy coefficient at 0.005, the five learning epochs, the four minibatches, the learning rate, the adaptive schedule, the discount and trace decay factors, the desired divergence and the gradient norm ceiling are identical, as are the actor and critic hidden dimensions of 512, 256 and 128, the ELU activation, the two observation normalisation flags and the encoder geometry.

They differ in exactly two places.

The first is `max_iterations`, 15000 on the KScale against 30000 on the BRS. Nothing in the tree justifies the halving and the KScale is the harder learning problem of the two by virtue of carrying two additional actuated degrees of freedom, so the figure should be restored.

The second is `symmetry_cfg`, which the BRS sets and the KScale omits. The BRS passes an `RslRlSymmetryCfg` enabling data augmentation and mirror loss with a mirror loss coefficient of 0.0, referring the augmentation function to `tasks/locomotion/mdp/symmetry/brs.py`. The comment at `limx_rsl_rl_ppo_cfg.py:286-294` explains the omission, that the BRS augmentation matches joint names by a trailing L and R suffix and would silently mismatch against the KScale naming convention, and it is right to have declined rather than to have risked a silent bad augmentation. The omission is nonetheless costly, the symmetry augmentation being the single change that first produced a coordinated alternating gait on the BRS, recorded in the status banner of `../../plans/SYMMETRY_PLAN.md`, and section 4.7 supplies the KScale specific module.

One detail of the BRS configuration should not be copied without deliberation. Setting `mirror_loss_coeff` to 0.0 while leaving `use_mirror_loss` true causes the mirror loss to be computed on every minibatch and then multiplied by zero, so the additional forward pass is executed and its gradient contribution discarded. The loss in question is the squared discrepancy between the policy's action at a state and the mirrored action at the mirrored state, introduced by Yu, Turk and Liu [5] and adopted as the standard form thereafter [2], and it is exactly what the Isaac Lab symmetry configuration computes when the flag is set. The data augmentation, which is the mechanism that carries the benefit, is gated independently by `use_data_augmentation` and is genuinely active. The literature supports that division, both the primary symmetry paper and the Isaac Lab symmetry note reporting that augmentation outperforms the loss, the former excluding the loss from its comparison altogether on the strength of the latter's finding [1][4]. The recommendation for the KScale is therefore to enable augmentation and to set `use_mirror_loss` to False rather than to carry a coefficient of zero, which obtains the same learning behaviour without the wasted computation.

### 2.4 Other Environment Configurations

The observations, the actions and the events were, as the task brief supposed, ported faithfully, and this was verified term by term rather than assumed. Every noise standard deviation, every clip, every scale and every `__post_init__` flag of `ObservationsCfg` and `HIMObservationsCfg` matches the BRS, across all of the policy, critic, history, target encoder, commands and estimator ground truth groups. Every event range and distribution parameter matches. The only edits are the necessary retargeting of body and joint name patterns. Four exceptions were found and each is small but real.

The first is that `add_link_mass` addresses `_LEG_LINKS_NO_FOOT` on the KScale and therefore excludes the feet from mass randomisation, where the BRS includes them.

The second is that `gait_command.durations` is 0.6 on the KScale against 0.62 on the BRS, while the comment above the term at `cfg/SF/kscale_base_env_cfg.py:143-146` states that the block was ported unchanged. The comment misstates the value it claims to have copied. The BRS raised its own figure from 0.6 to 0.62 as part of the double support work recorded at `../../context/brs_gait.md:522`.

The third is that both `_PLAY` variants drop three of the four command and episode overrides the BRS play configurations carry, retaining only `lin_vel_x`, and set that to the range the BRS uses for its HIM play task even on the non HIM task.

The fourth is that the KScale defines five leaf environment configuration classes against the BRS exemplar of eight, lacking `KscaleHIMBlindRoughEnvCfg` and its play sibling, so the combination of the hybrid internal model with rough terrain cannot be launched for this robot at all.

Two structural matters lie outside the configuration files and are more serious than any of the four above.

The task registry at `tasks/locomotion/robots/__init__.py` carried an unclosed brace in the `Isaac-Limx-Kscale-Blind-Flat-Play-v0` registration, which is a syntax error in the module that registers every task of every robot in this repository, so that the BRS, the three TRON1 variants and the quadruped were all unloadable on this branch alongside the KScale. This was repaired in the pass that wrote this document and is recorded in section 4.1.

The launcher at `/ws/djinn` carries no KScale branch in either its training dispatch at `djinn:118-154` or its play dispatch at `djinn:192-206`. Both are chains of `elif` clauses over a task variable that has already been assigned a default of `Isaac-Limx-SF-Identified-Blind-Flat-v0`, so a request for the KScale does not fail, it falls through every clause and launches the TRON1 SoleFoot instead. A user would obtain a plausible looking run of the wrong robot with no diagnostic of any kind. This is the most dangerous defect in the integration precisely because it is silent, and section 4.8 repairs it.

A defect unrelated to the KScale was found during the validation pass and is recorded here rather than repaired, in the manner rule 6 of `../../CLAUDE.md` requires. `assets/__init__.py:2` executes `from .usd import *`, and `assets/usd/` contains only payload directories and no `__init__.py`, so the import cannot resolve. It dates from the quadruped integration commit `bdc40ee` and is latent only because nothing imports `environments.assets` directly, every call site reaching past it to `environments.assets.config` and below. Repairing it would change what `from environments.assets import *` exports, which is a decision for the quadruped work stream rather than for this plan.

### 2.5 Summary of the discussed differences

The table gathers every difference established above, together with the consequence of leaving it standing and the section that proposes its remedy. Values marked as derived are computed in section 3 and justified in section 4.

| Aspect | BRS | KScale as it stands | Same | Consequence if left | Remedy |
|---|---|---|---|---|---|
| Root frame convention | lateral axis y, forward x | lateral axis x, forward minus y | No | Forward velocity commands drive lateral motion | 4.2 |
| Source location | `environments/environments/` | was `exts/bipedal_locomotion/` | No | Unimportable, no package `__init__.py` anywhere in the chain | 4.1, done |
| Task registry syntax | valid | unclosed brace at the play registration | No | Every robot's tasks unloadable | 4.1, done |
| `djinn` dispatch | `brs`, `brs-simplified` clauses | no clause | No | Silently launches the TRON1 SoleFoot | 4.8 |
| Leaf env cfg classes | 8 | 5 | No | HIM with rough terrain unlaunchable | 4.6 |
| `sole_offsets` table | 12 points, `brs_base_env_cfg.py:644` | absent | No | Four reward terms dropped | 3.3, 4.3 |
| `rew_foot_clearance` | `foot_clearance_reward_v3`, weight 10.0 | absent | No | No swing shaping, the plateau defect returns | 4.4 |
| `pen_foot_landing_vel` | `foot_landing_vel_v2`, weight minus 30.0 | absent | No | Impact unpriced on the descent side | 4.4 |
| `pen_feet_impact` | `feet_impact_force`, minus 3.0e-2 at 850 N | absent | No | Impact unpriced on the force side | 4.4 |
| `rew_keep_ankle_roll_zero_in_air` | weight 0.25 | absent | No | Ankle roll unregulated in swing | 4.4 |
| `pen_ankle_deviation` | removed after producing no effect | live at minus 0.1 | No | Revives an abandoned experiment | 4.5 |
| `rew_no_fly` function | `NoFlyWithGrace`, `grace_steps` 20 | `no_fly` | No | Double support priced at zero, transitions suppressed | 4.4 |
| `rew_keep_ankle_pitch_zero_in_air` params | 6 parameters set | 1 parameter set | No | Graded against a 15 ms stale contact state | 4.4 |
| `pen_feet_regulation` `foot_radius` | 0.124, measured | 0.19, an admitted misread | No | A grounded foot reports 0.147 m of false clearance | 4.3 |
| `pen_base_height` `target_height` | 1.15 | 1.0, a guess | No | Height penalty centred 0.2 m above the true stance | 4.3 |
| `pen_feet_regulation` `base_height_target` | 1.15 | 1.0, tracking the guess | No | As above | 4.3 |
| `low_height` `minimum_height` | 0.4 | 0.35, scaled from the guess | No | Termination threshold at 44 percent of stance, not 35 | 4.3 |
| `pen_feet_distance` `min_feet_distance` | 0.25 | 0.21 | No | Stance floor set below the BRS ratio | 4.3 |
| `gait_command.durations` | 0.62 | 0.6, with a comment claiming parity | No | Diverges from the BRS double support setting | 4.3 |
| `gait_command.swing_height` | 0.08 | 0.08 | Yes | Proportionally a much larger swing on a shorter leg | 4.3 |
| `add_link_mass` bodies | includes the feet | excludes the feet | No | Feet escape mass randomisation | 4.3 |
| `_PLAY` overrides | 4 | 1 | No | Play runs at training command ranges | 4.6 |
| Hip yaw | fixed in the URDF | revolute, plus and minus 1.5708 | No | Twelve actuated joints against ten, no gain randomisation, no reset perturbation | 4.3, 4.7 |
| `enabled_self_collisions` | see 3.4 | False, while `self_collision` is True | No | Contradictory, and the BRS forged contact exploit is undetectable | 4.3 |
| Actuator damping ratios | targeted near 0.7 | 1.64 to 16.4, every joint | No | Near rigid joints, two ankles at or above Nyquist | 4.3 |
| `max_iterations` | 30000 | 15000 | No | Halved training budget on the harder problem | 4.7 |
| `symmetry_cfg` | set, augmentation active | absent | No | The change that first produced a walking BRS gait is missing | 4.7 |
| Observations, all groups | | identical | Yes | | none |
| Events, all terms | | identical but for `add_link_mass` | Yes | | none |
| PPO hyperparameters and network shapes | | identical | Yes | | none |
| Curriculum, all terms | | identical | Yes | | none |

## 3. KScale URDF And STL Analysis

The URDF alone does not answer the questions the reward set asks of it. It gives the joint origins and the link inertias, and it names a mesh file for each collision shape, but the shape of the sole, the depth of the sole below the ankle frame and the orientation of the foot link frame are properties of the mesh and must be measured from it. This section reports that measurement, states the script that performs it, and records two findings the measurement produced that were not being sought.

### 3.1 The script

The script is `scripts/analysis/kscale_sole_analysis.py`, written in the manner of `scripts/analysis/stats.py`, pure numpy and the standard library with no dependency on Isaac Lab, on torch or on the task package, so that it runs inside the simulation container and equally in a plain interpreter against a checked out URDF. It is robot agnostic, taking the URDF path and a regular expression naming the foot links, so that it serves the BRS and any future robot as readily as the KScale.

It proceeds in five steps. It parses the URDF into links and joints and identifies the root as the unique link that is never a joint child. It computes forward kinematics at the zero pose to obtain the world rotation of every link frame. It uses that rotation to determine which axis of the foot link frame points downward, rather than assuming that the foot's own z is its vertical, which is the assumption that produced the erroneous 0.19 figure. It transforms every collision mesh vertex through the collision origin into the link frame, isolates the vertices lying within one millimetre of the extreme along the downward axis, and takes the convex hull of that set in the two remaining axes. It then reduces the hull to a requested number of points by repeatedly deleting the vertex whose removal costs the least polygon area, which preserves the extremes that dominate the tilted minimum where a naive truncation would discard the widest part of a rounded toe.

Two checks are emitted beside the table and both earn their place.

The first is a frame convention check, reporting which root frame axis separates the two feet. Isaac Lab evaluates the velocity command and the base linear velocity observation in the root body frame, so a robot whose lateral axis is not y has its forward command pointing sideways, and this check is what detected the defect of section 3.2.

The second is a fidelity sweep, rotating both the full vertex set and the reduced table through the ankle's own roll and pitch limits, read from the two joints immediately proximal to the foot, and reporting the largest height by which the reduced table overestimates the true clearance. The comparison is performed after aligning the sole to face world down, so that the sweep exercises the foot's real roll and pitch axes rather than the arbitrary axes of a permuted link frame. This distinction is not pedantic. Sweeping in the unaligned frame reported an error of 8.399 mm for the twelve point table, and the same table measured correctly commits 0.836 mm.

The sweep also settles the point count, which would otherwise be arbitrary.

| Points retained | Worst height error over the ankle's travel |
|---|---|
| 4 | 25.116 mm |
| 8 | 2.530 mm |
| 12 | 0.836 mm |
| 16 | 0.657 mm |
| 24 | 0.389 mm |

Twelve points is the knee of that curve, improving on eight by a factor of three and improved upon by sixteen by less than a quarter. It also matches the cardinality of the BRS table and lands within a twentieth of a millimetre of the 0.85 mm fidelity the BRS table achieves over its own travel, which is the closest thing to an independent calibration available. Four points, which an earlier analysis proposed on the reasoning that a straight sided trapezoid's corners bound its hull, commits 25 mm of error, because the KScale toe is a rounded arc rather than a straight taper and a chord drawn across that arc falls some two centimetres inside the true boundary.

### 3.2 The root frame convention

The two feet of the KScale are separated in the root frame by the vector 0.2520, 0.0001, 0.0000 in metres, so the lateral axis is x. The two feet of the BRS are separated by 0.000, minus 0.259, 0.000, so its lateral axis is y, which is the Isaac Lab convention. The KScale root frame is rotated by ninety degrees about the vertical relative to that convention.

Three independent lines of evidence agree and the finding does not rest on the separation vector alone. The two hip pitch joint origins at `kscale.urdf:631` and its left mirror differ only in x, at plus and minus 0.055, and are identical in y and z. The foot's long axis of 0.21 m lies along the root frame y. The toe extends toward minus y from the ankle while the heel sits 0.02 m to plus y, so forward is minus y. The joint naming is nonetheless correct throughout, the joints named as pitch rotating about the lateral axis and those named as roll about the fore and aft axis, so nothing in the URDF is mislabelled. It is the frame convention alone that differs.

The consequences reach four places in the configuration. The velocity command's `lin_vel_x` range of minus 0.3 to 0.8 is evaluated in the root body frame and therefore commands a leftward sidestep of up to 0.8 m/s, while `lin_vel_y` at plus and minus 0.01 pins the true fore and aft velocity near zero. The height scanner's `GridPatternCfg` of size 1.6 by 1.0 lays its long axis across the robot rather than along its path. The symmetry mirror plane is the y and z plane rather than the x and z plane, which inverts the flip set as section 3.5 records. And any per axis foot separation statistic reads the stride where it intends to read the stance width.

### 3.3 The sole

The foot link frame is a full axis permutation away from the BRS convention. Its x is the sole's width, its y is the vertical with the positive sense pointing downward, and its z is the fore and aft length. Both feet use the same mesh under the same collision origin, at `kscale.urdf:582-587`, the visual origin being identical to the collision origin, so the two feet share one table and the mirroring is carried entirely by the link rotations.

The sole is a genuinely flat manufactured surface. Of the 51228 vertices, 462 lie within one millimetre of the extreme, and those 462 span a range of one micron, which is a real machined face and not a numerical accident. A second, shallower flat band exists four millimetres above it, an interior recessed face bordered by the deeper rim, and the outer rim is the physically correct contact surface since it is what first touches the ground under any orientation.

| Quantity | Value |
|---|---|
| Sole plane in the link frame | y equal to plus 0.0430 |
| Sole depth below the link origin | 0.0430 m |
| Sole width, the x extent | 0.0846 m |
| Sole length, the z extent | 0.2100 m |
| Convex hull vertices on the sole plane | 78 |
| Shape | full width at the heel, tapering through a rounded toe |

The figure 0.19 that the superseded comment reported is arithmetically correct as a minimum of the link frame z, and it is the foot's length rather than its depth. The comment's arithmetic was sound and its choice of which axis to call the lowest point was not, because it carried the BRS convention that a foot link's z is its vertical onto a robot for which that is false. The comparable figure is 0.0430 m against the BRS 0.124 m, a ratio of 0.347 which tracks the mass ratio of 0.339 closely, and is therefore evidence that the foot is a normally proportioned sole plate rather than the tall bracket the comment feared.

The table the script produces is reproduced in section 4.3.

### 3.4 Self collision and link penetration

No left leg link overlaps any right leg link at either the zero pose or the nominal pose, the two chains remaining separated throughout by the 0.126 m per side lateral offset of the hip. No link penetrates the ground plane at the configured spawn height of 1.0 m, the lowest point of the robot clearing the ground by 0.2047 m at the nominal pose, which is a large margin and suggests the spawn height is set well above the true stance.

Two non adjacent axis aligned bounding box overlaps exist at both poses. The torso overlaps each hip roll link across a region of 0.003 by 0.116 by 0.089 m, two joints removed. Each shank overlaps its own foot, one hop around the ankle bearing, over a region growing from 0.085 by 0.096 by 0.035 m at the zero pose to 0.085 by 0.202 by 0.055 m at the nominal pose. Bounding box overlap is a conservative test and neither pair is necessarily a true mesh intersection, but both warrant a visual check.

The configuration is internally contradictory on the matter. `assets/config/kscale_identified_cfg.py:113` sets `enabled_self_collisions` to False on the articulation properties while `:160` sets `self_collision` to True on the spawn configuration. This is the same flag combination that `../../context/brs_gait.md:151` and `:171` record as central to a confirmed BRS exploit, in which a trained policy pressed its legs together to forge a contact signal and thereby defeated every contact keyed reward term at once. The KScale carries the same exposure with the same geometry and, unlike the BRS, carries no instrumentation that would reveal it, no equivalent of the sole clearance logging the BRS work stream built. Resolving the contradiction and adding the forged contact check, a non zero contact force on a foot whose true sole clearance is well above zero, should precede any trust in a KScale contact keyed reward.

### 3.5 The symmetry mirror

Under a reflection, a rotation about an axis maps to a rotation about the image of that axis carrying the determinant's sign, so for the reflection matrix the axis transforms as a pseudovector. Composing each joint's origin rotation chain to the root and comparing each left joint's mirrored world axis against its right partner's actual axis settles the flip set without any appeal to a naming convention or to a roll against pitch rule, which is the method the MorphoSymm framework establishes [3] and the reason the BRS hip pitch flips despite being a pitch joint.

Every KScale joint declares its axis as the local 0, 0, 1, with the whole orientation carried in the origin rotation, so the flip set cannot be read off the axis vectors and must be composed. Doing so about the correct y and z mirror plane gives the following.

| Joint | Left world axis | Right world axis | Flips |
|---|---|---|---|
| `hip_pitch_04` | minus x | plus x | Yes |
| `hip_roll_03` | minus y | minus y | Yes |
| `hip_yaw_03` | plus z | plus z | Yes |
| `knee_04` | plus x | plus x | No |
| `foot_pitch_02` | plus x | plus x | No |
| `foot_roll_02` | plus y | plus y | Yes |

This is the physically expected pattern, the roll and yaw degrees of freedom flipping and the pitch degrees of freedom not, with the single exception of the hip pitch, which flips because its left and right URDF axes are genuinely anti parallel. That exception is corroborated independently by the joint limits, the hip pitch and hip roll being the only two joints whose left and right limits are negations of one another, at minus 1.0472 to 2.21657 against minus 2.21657 to 1.0472 and at minus 0.20944 to 2.26893 against minus 2.26893 to 0.20944, while the remaining four carry identical limits on both sides. It is also exactly the anomaly the BRS symmetry module documents for itself at `../../plans/SYMMETRY_PLAN.md:220`. The two robots are therefore structurally identical under the mirror once the frame is corrected, the KScale simply adding the live hip yaw to the flip set where the BRS hip yaw is a fixed joint.

Reflecting about the wrong plane, which is what a reader assuming the Isaac Lab convention would do, yields the flip set of hip yaw, knee and foot pitch, which is the physically implausible pattern of the pitch joints flipping and the roll joints not. That an incorrect mirror plane produces a plausible looking table is the reason section 4.2 corrects the frame before section 4.7 writes the mirror, rather than writing a mirror against the frame as it stands.

### 3.6 Actuator parameterisation

Following the parallel axis method of `../context/BRS.md:112-231`, and computing in full three dimensions because the KScale link inertial frames are not generally aligned with the joint axes acting on them, the effective inertia at each joint of the right leg at the nominal pose is as follows, with the natural frequency and damping ratio implied by the placeholder gains at `assets/config/kscale_identified_cfg.py:22-105`.

| Joint | Stiffness | Damping | Effective inertia | Natural frequency | Damping ratio |
|---|---|---|---|---|---|
| `hip_yaw` | 40 | 5 | 0.0133 | 54.9 | 3.43 |
| `hip_roll` | 150 | 45 | 1.0162 | 12.2 | 1.82 |
| `hip_pitch` | 200 | 50 | 1.1657 | 13.1 | 1.64 |
| `knee` | 200 | 22 | 0.1074 | 43.2 | 2.37 |
| `foot_roll` | 20 | 4 | 0.00074 | 164.3 | 16.42 |
| `foot_pitch` | 50 | 4 | 0.00260 | 138.6 | 5.54 |

Units are newton metres per radian, newton metre seconds per radian, kilogramme metres squared, and radians per second, the damping ratio being dimensionless.

Every joint is overdamped, the ratios running from 1.64 to 16.4 against the 0.7 that `../context/BRS.md` targets, and the two ankle joints carry a natural frequency at or above the Nyquist bound of 157.08 rad/s imposed by the 50 Hz control loop, the foot roll exceeding it outright. This is the direct consequence of copying the BRS gains by joint role onto effective inertias roughly a sixth as large, and it is the opposite failure mode from the one the BRS itself suffered, where the same class of copied gain was too soft and left the proximal joints ringing at ratios between 0.07 and 0.16. The KScale joints as configured will resist a commanded change in position almost as though position controlled open loop, and the ankles operate at the edge of what the discrete loop can represent without aliasing.

A further caution applies at the distal joints. The armature values of 0.005 to 0.015 kg m squared are of the same order as the effective inertias of the hip yaw, ankle pitch and ankle roll joints themselves, so the reflected rotor inertia is not the safely ignorable correction there that it is at the hips, where the effective inertia exceeds 1 kg m squared.

The three actuator housing meshes, `rs02.stl`, `rs03.stl` and `rs04.stl`, are placed largest at the hip pitch and smallest at the shank, with mesh volumes of 514.71, 181.16 and 60.63 cubic centimetres respectively. The naming and the size ordering are consistent with the Robstride RS02, RS03 and RS04 quasi direct drive actuator line, which would be a far better source of gains than any scaling argument. This identification is plausible speculation and nothing more, no manufacturer name, part number or specification reference appearing anywhere in the URDF or the surrounding configuration, and it is recorded so that it may be confirmed or refuted rather than relied upon.

## 4. Proposed Changes

The changes are ordered so that each rests only on those before it. Section 4.2 must precede section 4.7, because a mirror written against the uncorrected frame would carry the wrong flip set, and section 4.3 must precede any training run, because the parameters it corrects govern terms that are already live.

Throughout, the governing constraint of `../../CLAUDE.md` is that no existing caller may change behaviour. That constraint turns out to be easy to satisfy here, for a reason worth stating plainly. No reward function in `tasks/locomotion/mdp/rewards.py` hard codes a link name, a body count or a sole geometry. Every robot specific quantity is already an explicit configuration parameter, `sole_offsets`, `foot_radius`, `min_feet_distance`, `force_threshold`, `base_height_target` and the rest, and every body set arrives through a `SceneEntityCfg` pattern. The reward terms this plan restores are therefore a data problem and not a code problem, and no function in the shared `mdp` package need be edited, no optional argument added and no version two created. The BRS, the three TRON1 variants and the quadruped are untouched by every change below, and the only shared file this plan modifies at all is the task registry, where it adds registrations without altering existing ones.

### 4.1 Relocation of the sources, completed

Carried out in the pass that wrote this document, and recorded here because the plan's file references depend upon it.

The assets moved from `exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/kscale/` to `environments/environments/assets/urdf/solefoot/kscale/`, being the URDF and its 35 STL meshes. The environment configuration moved from `exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/cfg/kscale/kscale_base_env_cfg.py` to `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py`, the KScale being a sole footed biped and therefore belonging with the BRS and the TRON1 SoleFoot rather than in a directory of its own. The now empty `exts/` tree was deleted in its entirety.

Five import statements were rewritten, two in `kscale_base_env_cfg.py`, one in `assets/config/kscale_identified_cfg.py` and three in `tasks/locomotion/robots/kscale_solefoot_env_cfg.py`, each changing the package root from `bipedal_locomotion` to `environments`. In `tasks/locomotion/robots/__init__.py` the dead `from ..cfg.kscale import kscale_base_env_cfg` was folded into the existing `from ..cfg.SF import ...` group, and the unclosed brace in the `Isaac-Limx-Kscale-Blind-Flat-Play-v0` registration was closed.

The relocation is a repair rather than a tidying. The `exts/` tree contained no `__init__.py` at any level, so `bipedal_locomotion` was never an importable package and every import referring to it was unresolvable, and the unclosed brace was a syntax error in the module that registers every task of every robot. The KScale had never been launchable and, on this branch, neither had anything else.

The relocation was validated four ways. Byte compilation succeeds across `environments/`, `scripts/` and `co_optimisation/`. No reference to `bipedal_locomotion` survives in any Python source. All 114 mesh references in the URDF resolve relative to the URDF's own directory, so the relative paths survived the move. And a static walk of the import graph over all 60 modules resolves every intra package module and every imported symbol, the single exception being the pre existing and unrelated `assets/__init__.py` defect recorded at the end of section 2.4.

### 4.2 Correct the root frame convention

This change must be made before any training run and before the symmetry module of section 4.7.

The object is that the root link frame carry the Isaac Lab convention, x forward, y to the left and z up. The frame is presently rotated by ninety degrees about z from that, so the correction is to re-express the root frame by the rotation that carries the robot's forward direction of minus y onto plus x, which is a rotation of plus ninety degrees about z.

Two mechanisms are available and the second is preferred.

The first is to insert a massless `base_link` above the torso joined by a fixed joint carrying `rpy="0 0 1.5708"`. It is the more conventional fix and the less invasive to the existing link definitions, but it adds a body to the articulation, which changes the width of every critic observation that iterates over bodies, `robot_mass`, `robot_inertia` and `robot_material_properties` among them, and it introduces a root link that is not the torso, so the height scanner prim path and the `_TORSO_LINK` references would each need review.

The second, and the one this plan adopts, is to re-express the existing root frame in place, which preserves the body count exactly and therefore changes no observation width. Every quantity expressed in the root frame is premultiplied by the rotation, being the root link's own inertial, visual and collision origins, the origins of the two joints whose parent is the root, and the root's inertia tensor, which transforms as `R I R^T`. Nothing below the root changes, since each subtree is rigidly attached to its parent joint and moves with it.

The transformation is mechanical and should be performed by a script rather than by hand, and it is self checking. Re-running `scripts/analysis/kscale_sole_analysis.py` after the edit must report the lateral axis as y and the verdict as conventional, where it presently reports x and non conventional. The foot separation vector must become approximately 0.000, 0.252, 0.000. The composed flip set of section 3.5, recomputed about the x and z plane in the corrected frame, must reproduce the same table, since the flip set is a physical property of the robot and not of the frame it is expressed in.

Two configuration values become correct as a consequence and neither should be edited before the frame is fixed, on pain of correcting the same defect twice in opposite directions. The velocity command ranges at `cfg/SF/kscale_base_env_cfg.py:135-140` then mean what they say, `lin_vel_x` being forward. The height scanner pattern at `:105` then lays its 1.6 m axis along the direction of travel.

Should the frame correction be deferred, which this plan does not recommend, the only coherent alternative is to swap the command ranges and the scanner dimensions and to write the symmetry mirror about the y and z plane, and to record loudly at each site that the robot's x is lateral. That path is worse in every respect except immediacy, because it distributes one defect across four files instead of repairing it in one.

### 4.3 Correct the parameters that are already live

These terms exist in the KScale configuration and carry values that were copied or guessed rather than derived. Each row states the derivation.

The sole table is the prerequisite for section 4.4 and is placed here because it belongs beside the geometry it summarises. It is emitted by the script of section 3.1 and should be inserted into `cfg/SF/kscale_base_env_cfg.py` as a module constant beside the `RewardsCfg` class, in the manner of `SD_BRS1_SOLE_OFFSETS` at `cfg/SF/brs_base_env_cfg.py:644`, so that the clearance reward and the landing gate cannot drift apart.

```python
# Twelve points on the sole rim in the foot link frame, measured from the collision mesh by
# scripts/analysis/kscale_sole_analysis.py. NOTE the axis permutation, this foot's link frame
# carries its vertical on +y and its fore-aft length on z, unlike SD_BRS1 where the vertical is
# z. The sole plane is y = +0.0430, i.e. 0.0430 m below the ankle. Reproduces the true lowest
# point of the mesh to within 0.836 mm over the ankle's full roll and pitch travel.
KSCALE_SOLE_OFFSETS = [
    [-0.0423, +0.0430, +0.0138],
    [-0.0362, +0.0430, -0.1572],
    [-0.0326, +0.0430, -0.1712],
    [-0.0258, +0.0430, -0.1805],
    [-0.0108, +0.0430, -0.1889],
    [+0.0051, +0.0430, -0.1899],
    [+0.0213, +0.0430, -0.1841],
    [+0.0326, +0.0430, -0.1712],
    [+0.0362, +0.0430, -0.1572],
    [+0.0423, +0.0430, +0.0138],
    [+0.0363, +0.0430, +0.0200],
    [-0.0363, +0.0430, +0.0200],
]
```

| Parameter | Present | Proposed | Derivation |
|---|---|---|---|
| `pen_base_height.target_height` | 1.0 | 0.795 | Measured torso height above the lowest mesh point at the nominal pose |
| `pen_feet_regulation.base_height_target` | 1.0 | 0.795 | Kept in step with the above, as the BRS keeps its own pair in step |
| `pen_feet_regulation.foot_radius` | 0.19 | 0.043 | The measured sole depth of section 3.3, against a figure that was the foot's length |
| `low_height.minimum_height` | 0.35 | 0.28 | The BRS ratio of 0.4 against 1.15 is 0.348, applied to 0.795 |
| `pen_feet_distance.min_feet_distance` | 0.21 | 0.24 | The BRS ratio of 0.25 against a 0.259 m separation is 0.965, applied to 0.252 |
| `gait_command.durations` | 0.6 | 0.62 | Parity with the BRS, which raised its own figure for the double support work |
| `gait_command.swing_height` | 0.08 | 0.05 | The BRS swing is 9.2 percent of its 0.87 m leg, applied to the 0.505 m leg |
| `init_state.pos` z | 1.0 | 0.85 | Approximately 0.05 m above the corrected 0.795 m stance, so the robot settles rather than drops |
| `add_link_mass` bodies | legs without feet | legs with feet | Parity with the BRS, which randomises foot mass |
| `enabled_self_collisions` | False, against `self_collision` True | resolve to one value | Section 3.4, the contradiction and the forged contact exposure |

The `foot_radius` correction is the largest single error in the live configuration and its magnitude deserves stating. The term computes a ground clearance as the body frame height less `foot_radius`, so at 0.19 against a true 0.043 a foot resting flat on the ground reports a clearance of minus 0.147 m. Paired with the `height_decay_scale` of 0.03, the gate `exp(-(z - r)/s)` evaluates to `exp(4.9)`, so the penalty is amplified by a factor near 134 rather than being focused near the ground, which inverts the term's intent entirely. This is the same defect family the BRS suffered in the opposite direction, where an inherited point foot radius of 0.03 against a true 0.124 made a grounded foot report 0.094 m of false clearance and retained only four percent of the configured weight, recorded at `../../plans/GAIT_STRATEGY.md:184` and `../../context/brs_gait.md:89`.

The `min_feet_distance` figure is carried across by the BRS ratio rather than derived afresh, and the biomechanical standard corroborates the result rather than the method, a stance width of 1.0 to 1.3 times hip width [11] placing the KScale floor of 0.24 m against a hip separation of 0.252 m within that band. It should be re-examined against the corrected frame, since a scalar separation term measures the planar norm and therefore reads the stride where it intends to read the stance width.

The `swing_height` reduction warrants its own note because it is the one row above that changes a command rather than a penalty. The gait command implements the periodic contact schedule of Siekmann and colleagues [7] in the form Walk These Ways gives it [8], and it declares a swing height that, until the BRS Phase 3 work, no reward read. It is now read by `foot_clearance_reward_v3` as the amplitude of the raised cosine reference, so it has become a physical setpoint rather than an unused declaration, and a swing of 0.08 m on a 0.505 m leg is proportionally half again what the BRS asks of its own leg.

Two matters in this section are directional rather than settled and are marked as such. The actuator gains of section 3.6 require a retuning pass in the manner of `../context/BRS.md` section 9, targeting a damping ratio near 0.7 and a natural frequency comfortably below the 157.08 rad/s Nyquist bound, which points toward hip stiffnesses on the order of a third of the present values and ankle stiffnesses an order of magnitude below them. That derivation is deliberately not attempted here, both because it deserves the full treatment `../context/BRS.md` gives the BRS and because confirming the Robstride identification of section 3.6 would supersede any scaling argument with measured data. The nominal standing pose likewise remains the placeholder that `assets/config/kscale_identified_cfg.py:118-125` admits it to be, and the 0.795 m figure above is measured against that placeholder, so it must be re-measured once the pose is settled.

Finally, the hip yaw joint is presently actuated and observed while receiving neither actuator gain randomisation nor a reset perturbation, where every other joint receives both. Two event terms should be extended to cover it, `hip_joint_stiffness_and_damping` at `cfg/SF/kscale_base_env_cfg.py:513` acquiring `(right|left)_hip_yaw_03` in its joint list, and a `reset_hip_yaw_joints` term added beside the existing reset terms with a position range comparable to the hip roll's.

### 4.4 Restore the four missing reward terms and repair two more

Each block below is the BRS term retargeted onto the KScale names and parameters. No reward function changes.

```python
    # Repointed from the stateless no_fly. NoFlyWithGrace with grace_steps 0 is bit for bit
    # identical to no_fly, so this is a strict extension. 20 steps is van Marum's 0.2 s window
    # at this task's 0.01 s control period.
    rew_no_fly = RewTerm(
        func=mdp.NoFlyWithGrace,
        weight=15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "threshold": 1.0,
            "history_index": 0,
            "grace_steps": 20,
        },
    )

    # The four parameters the port omitted are restored. history_index 0 reads the CURRENT
    # contact frame, the default of -1 reading the oldest of the four buffered samples.
    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(right|left)_foot_pitch_02"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "use_default_offset": False,
        },
    )

    # New. The BRS counterpart, absent from the port. Same function, quarter weight, roll joints.
    rew_keep_ankle_roll_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(right|left)_foot_roll_02"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
        },
    )

    # New. Tracks a raised cosine reference read from the gait clock, so the whole swing path is
    # determined rather than only its extremum. std scaled with swing_height, 0.03 * 0.05 / 0.08.
    rew_foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward_v3,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "command_name": "gait_command",
            "std": 0.02,
            "sole_offsets": KSCALE_SOLE_OFFSETS,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "force_threshold": 1.0,
        },
    )

    # New. Charges the vertical velocity of the LOWEST SOLE POINT, gated on the true sole
    # clearance, so the gate cannot be defeated by tilting the foot. Threshold 0.75 of the
    # swing height, matching the BRS ratio of 0.06 against 0.08.
    pen_foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel_v2,
        weight=-30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "sole_offsets": KSCALE_SOLE_OFFSETS,
            "about_landing_threshold": 0.04,
            "force_threshold": 1.0,
        },
    )

    # New. Prices the impact from the force side. 850 N is 1.448 body weights on the 59.85 kg
    # BRS, and 1.448 body weights on the 20.281 kg KScale is 288 N.
    pen_feet_impact = RewTerm(
        func=mdp.feet_impact_force,
        weight=-3.0e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "force_threshold": 290.0,
        },
    )
```

The weights transfer unchanged and this is deliberate. A weight multiplies a kernel whose argument has already been normalised by a parameter carrying the units, so the scaling belongs in the parameter and not in the weight. The two impact terms are the exception worth watching, since `feet_impact_force` is a hinge on an absolute force rather than a normalised kernel, and its threshold has been scaled while its weight has not, which preserves the price per newton of excess. Whether the price per newton should itself scale with the robot is a question the BRS record does not answer, and the recommendation is to launch with the weight unchanged and to read the term's logged value against the BRS baseline before adjusting it.

The clearance reward should be introduced with attention to the failure it is meant to prevent. A Gaussian on instantaneous foot height multiplied by a tanh of foot speed, which is the form of the superseded version two, has an integrand depending on the instantaneous height alone, so its maximiser over a swing of fixed duration is the trajectory that reaches the target soonest, holds longest and leaves latest, which is a plateau and not an arc. A reward on an extremum determines only that extremum, whereas a reward on a reference determines the whole path, which is the reasoning behind the phase conditioned tracking form that Humanoid-Gym adopts [9] and that version three implements here. The human swing profile has an interior minimum near mid swing rather than a plateau [10], so the reference form is also the biomechanically faithful one.

### 4.5 Remove `pen_ankle_deviation`

Delete the term at `cfg/SF/kscale_base_env_cfg.py:712-720`. It is a `joint_deviation_l1` over the four ankle joints at weight minus 0.1, and it revives on the KScale an experiment the BRS ran and abandoned. The BRS introduced the same term at weight minus 0.2 against the defect of the ankles resting at their mechanical stops, recorded at `../../plans/GAIT_EFFICIENCY_PLAN.md:9`, and removed it after it produced no observable effect on policy performance, leaving it commented at `cfg/SF/brs_base_env_cfg.py:738-746` as the record of the attempt.

Two considerations argue for removal beyond mere parity. The term opposes `rew_keep_ankle_pitch_zero_in_air` and `rew_keep_ankle_roll_zero_in_air`, which section 4.4 restores and which already regulate the ankle posture, but does so unconditionally rather than only when the foot is airborne, so it penalises the ankle articulation that stance requires. And it is being carried at half the weight the BRS found ineffective, so the KScale would be running a weaker version of a null result.

`pen_hip_deviation` should be retained as it stands, including its coverage of the hip yaw joints, which the BRS cannot penalise because its own hip yaw is a fixed joint. That coverage follows the Isaac Lab G1 recipe of penalising deviation on the hip roll and yaw while leaving the hip pitch and the knee to swing freely, and it is the appropriate treatment of a degree of freedom the KScale has and the BRS does not.

### 4.6 Complete the environment class hierarchy and the registrations

Add `KscaleHIMBlindRoughEnvCfg` and `KscaleHIMBlindRoughEnvCfg_PLAY` to `tasks/locomotion/robots/kscale_solefoot_env_cfg.py`, following `brs_solefoot_env_cfg.py:247-265` as the template, so that the eight leaf classes of the BRS exemplar are matched. Register both in `tasks/locomotion/robots/__init__.py` with `HIMManagerBasedRLEnv` as the entry point, in the manner of the existing HIM registrations.

Restore the three dropped overrides to both `_PLAY` variants, being `lin_vel_y`, `ang_vel_z` and `episode_length_s`, and set `lin_vel_x` on the non HIM play task to the range the BRS non HIM play task uses rather than to its HIM range.

The six existing registrations are otherwise correct, each pointing at a class that exists, with the HIM tasks correctly using `HIMManagerBasedRLEnv` and the remainder the standard manager based environment.

### 4.7 Symmetry augmentation and the agent configuration

The new module is `tasks/locomotion/mdp/symmetry/kscale.py`, written closely against `symmetry/brs.py` so that the two are reviewable side by side. It must not be a copy with substituted strings, for a reason that is easy to miss. The BRS module discovers each joint's mirror partner by swapping a trailing L for a trailing R, at `symmetry/brs.py:82-91`. Every KScale joint name ends in a digit, so that rule matches nothing and degenerates silently to an identity permutation, raising no error and producing an augmentation that teaches the policy the robot is symmetric under doing nothing. The partner rule must instead swap the `right_` and `left_` prefixes. The body names need an explicit dictionary rather than any rule at all, since the right hip roll link is named `kd_d_201r_6061` and its left partner is named `rs03`, sharing no stem whatever.

Four constants change from the BRS module and each is established above. The joint partner rule becomes the prefix swap. The body partner map becomes an explicit dictionary. The sign flip set becomes the table of section 3.5, being the hip pitch, the hip roll, the hip yaw and the ankle roll, which differs from the BRS set by the addition of the hip yaw. And the height scan grid shape must be recomputed from whatever `GridPatternCfg` the corrected configuration carries after section 4.2.

Two elements transfer unchanged and should be understood rather than merely copied. The mirror of the observation groups follows the reflection physics, a polar vector such as the base linear velocity or the projected gravity flipping only its lateral component, a pseudovector such as the base angular velocity flipping its roll and yaw components and keeping its pitch, and the velocity command flipping its lateral and yaw components [3]. The gait phase, which the KScale policy group observes and which is a sine and cosine pair of one shared clock, negates both channels together, because the two feet are placed in antiphase by the command's offset of 0.5 rather than by the observation, so exchanging the feet is exactly a half cycle shift of the shared clock [1]. A naive treatment of the pair as an even and an odd channel would negate only one and is wrong.

The critic group must be mirrored as well as the policy group. A value function that observes an unmirrored privileged state cannot be invariant under the reflection, and this was raised late in the BRS work and is recorded as established fact in `../../context/rsl_rl.md`.

One quantity cannot be determined statically. The number of collision shapes per body, which the mirror of `robot_material_properties` requires, is known only at runtime, and the BRS module carries it as a constant obtained from a one time print. The KScale module should follow the same procedure, leaving the constant `None` until the count has been read from a running environment, under which the term falls back to an identity mirror rather than to a wrong one.

In `agents/limx_rsl_rl_ppo_cfg.py`, `KscaleFlatPPORunnerCfg` acquires the symmetry configuration and restores its iteration budget.

```python
from environments.tasks.locomotion.mdp.symmetry.kscale import (
    compute_symmetric_states as kscale_compute_symmetric_states,
)

    max_iterations = 30000
    ...
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            # False rather than the BRS's True with a zero coefficient, which computes the loss
            # every minibatch and multiplies it by zero. Augmentation is the mechanism the
            # literature finds effective, the loss being consistently outperformed by it.
            use_mirror_loss=False,
            data_augmentation_func=kscale_compute_symmetric_states,
            mirror_loss_coeff=0.0,
        ),
```

The choice of augmentation over the mirror loss is grounded rather than inherited. The catalogue of four mechanisms distinguishes duplication of transitions through the mirror, an auxiliary equivariance penalty, phase replay and a hard equivariant network, and reports that no single method dominates across robots [2]. The primary symmetry paper excludes the loss from its comparison on the strength of the Isaac Lab finding that augmentation converges faster and behaves better [1][4], and it recommends augmentation specifically for intrinsic motion symmetry on a real biped, where actuator and mass asymmetries break the perfect symmetry assumption and a hard constraint becomes brittle under distribution shift. The KScale, like the BRS, carries an asymmetric inertial model, so augmentation is the grounded first choice.

A caution from the same literature bears on the reset events. A strictly symmetric policy cannot leave a symmetric neutral pose, which is the neutral state problem, so training must begin from a noised non neutral posture [2]. The KScale reset events already perturb every joint, so this is satisfied, and it is recorded so that the perturbations are not removed as an economy.

A self test should precede any launch. Mirroring twice must return the original tensors to within floating point tolerance, the permutation must be a genuine involution over the twelve joints, and the augmented batch must be exactly twice the original in its leading dimension.

### 4.8 Register the KScale with the launcher

Add a `kscale` clause and a `kscale-him` clause to both dispatch chains of `/ws/djinn`, at `djinn:118-154` for training and `djinn:192-206` for play, pointing at the identifiers the registry already declares and setting `policy_type` to `HIMPPO` on the HIM clause in the manner of the existing `him` clause.

The change is small and its warrant is not. Both chains assign a default before testing any clause, so an unrecognised argument does not fail, it silently selects the TRON1 SoleFoot. A user asking for the KScale today receives a complete, plausible, converging run of a different robot, with the wrong task identifier appearing only in the run's own dumped parameters where nobody looks until something is already wrong. A defensive improvement worth making in the same pass, though strictly beyond this plan's scope, is a final `else` arm that fails loudly on an unrecognised robot argument, which would have made this defect self reporting.

This section is the one place where the plan reaches outside the repository, `djinn` being workspace level tooling. It is recorded here rather than split into a second document because it is two clauses and because a plan that left the robot unlaunchable would not be complete.

### 4.9 Record the findings

Add `../context/KScale.md`, recording the physical parameterisation established in section 3, in the manner `../context/BRS.md` records the BRS and `../context/quadruped.md` records the quadruped. It should carry the link and joint inventories, the segment lengths, the sole geometry with its measurement method, the effective inertias with the natural frequencies and damping ratios they imply, the self collision audit, and the frame convention finding. Register it in `../context/README.md`, whose document register and summaries section both require an entry.

Register this plan in `README.md`, which presently states that the directory is empty and that no plan is specific to this repository alone. That statement becomes false with this document and the surrounding paragraph needs rewriting rather than merely extending, since its argument is that every plan so far has spanned the workspace.

Three corrections belong in the context record and are listed so that they are not lost. The BRS hip yaw joints are declared `type="fixed"` in the URDF and are absent from the joint list entirely, rather than being present with a zero width limit, which several comments in the tree assert. The figure of roughly 0.13 m by which the KScale foot was said to stand off to the side of the hip is measured from the torso centreline and is correct as such, being the sum of two fixed bracket offsets, but the foot sits directly beneath the hip roll axis to within four microns, so the claim as phrased, that the zero pose is anomalous, is refuted. And `scripts/rsl_rl/play.py` applies the BRS sole table to whatever robot is played, a defect recorded at `../../context/brs_gait.md:698` and deliberately left standing, which now acquires a second affected robot and should be revisited before any KScale dump is read.

## 5. KBot Specific Reward Tuning

> Status, IMPLEMENTED 2026-08-26. Every proposal of this chapter was carried out as specified and the outcome, together with the two divergences it produced, is recorded in section 7.8. The chapter is retained in its proposing voice rather than rewritten into the past tense, so that what was predicted before the run may be read against what the run reports.

The integration of chapter 4 was judged a success on a narrow and deliberate criterion, that the KScale reward set should match the SD_BRS1 reward set term for term, and it met that criterion exactly, twenty seven terms carrying identical functions and identical weights with every differing parameter a derived robot specific quantity. That criterion was the right one for a first pass, because a reward set whose every departure from a working exemplar is deliberate is a reward set whose failures can be attributed, whereas a set retuned in the same pass in which it is ported confounds the two sources of error beyond recovery. The criterion has now served its purpose and it must be retired, because the run of 2026-08-25 produced a walking policy and therefore produced, for the first time, behaviour to judge.

The reason a ported reward set cannot remain a copy indefinitely is that a reward set is not a specification of behaviour in the abstract. It is a specification of behaviour over a particular mechanism, and a term is silent about every degree of freedom the exemplar mechanism does not possess. Where two robots differ in degree, in mass or in leg length or in sole width, a ported term specifies the same intent and the derived parameters carry the difference, which is what chapter 4 achieved. Where they differ in kind, in the presence or absence of a joint, a ported term specifies nothing whatever about the joint that is present in one and absent in the other, and no amount of care in transcribing parameters will supply the missing specification. The KScale differs from the SD_BRS1 in kind at exactly one place, and the defect this section addresses is located there.

The physical differences established across section 2 of this document and sections 2, 7 and 12 of [../context/KScale.md](../context/KScale.md) are gathered below, with the last column stating whether the difference is one of degree, which a derived parameter absorbs, or one of kind, which it cannot.

| Property | SD_BRS1 | KScale | Ratio | Character |
|---|---|---|---|---|
| Total mass | 59.85 kg | 20.2814 kg | 0.339 | Degree |
| Standing height | 1.15 m | 0.77161 m | 0.671 | Degree |
| Leg length, hip to ankle | 0.87 m | 0.505 m | 0.580 | Degree |
| Sole length | 0.2612 m | 0.2100 m | 0.804 | Degree |
| Sole width | 0.194 m | 0.0846 m | 0.436 | Degree |
| Sole depth below the ankle | 0.124 m | 0.0430 m | 0.347 | Degree |
| Ankle effort limit | 131 to 420 Nm | 17 Nm | 0.130 at best | Degree |
| Ankle roll travel | wider | plus and minus 0.26180 rad | | Degree |
| Active revolute joints | 10 | 12 | | Kind |
| Hip yaw joint | `type="fixed"` | `type="revolute"`, plus and minus 1.57080 rad | | Kind |
| Hip yaw axis in the root frame | absent | -0.0998, 0.0000, 0.9950 | | Kind |

Every difference of degree was addressed in chapter 4 and again in the gain derivation recorded in section 7.7, and none of them is at issue here. The single difference of kind is the hip yaw, and it is the widest range of travel on the robot, plus and minus ninety degrees, exceeding the ankle roll's travel by a factor of six. The SD_BRS1 declares both hip yaw joints with `type="fixed"` at `environments/environments/assets/urdf/solefoot/SD_BRS1/SD_BRS.urdf:31` and `:36`, so they never enter `robot.joint_names` at all and no reward term of that robot's set was ever required to say anything about them. The KScale declares both as revolute at `environments/environments/assets/urdf/solefoot/kscale/kscale.urdf:614` and `:1094`. The KScale therefore inherits a reward specification with a blind spot whose width is exactly the two joints the exemplar lacks, and it inherits it silently, because a term set that omits a joint looks no different from a term set that regulates it well.

Section 2.5 of this document anticipated the consequence in a single clause, noting that the KScale trains one more degree of freedom per leg than the SD_BRS1 and would need its own `joint_deviation_l1` treatment if the extra axis wandered off in training. It has wandered off. What follows establishes by what margin, why the one term that nominally covers the axis does not restrain it, and what should be added.

### 5.1 Feet Heading Direction

The behaviour is visible in the training video at `IsaacLab/logs/rsl_rl/kscale_flat/2026-08-25_10-44-21/videos/train/rl-video-step-400000.mp4`, which records the policy at 400000 environment steps, near the end of a run of 17217 iterations. The robot walks, tracks the commanded forward velocity, and survives 1425 steps of the 2000 step episode, so the gross failure recorded in section 7.7 is repaired and the policy has learned a gait. Within that gait the feet do not point where the robot is going. Across the sampled stride the swing foot is placed with its long axis rotated markedly out of the plane of travel, the two feet frequently point in appreciably different directions within the same double support interval, and the legs cross at the shank so that the stance leg is twisted beneath a torso facing forward. The commanded heading over the sampled interval is constant and forward, indicated by the command arrow the debug visualiser draws, so none of the rotation is attributable to a turn.

The logs establish the magnitude, and they establish it more precisely than the video can, because the one term that reads the axis reports a number. `pen_hip_deviation` at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:791` is a `joint_deviation_l1` over four joints, being both hip rolls and both hip yaws, at weight minus 0.1. Isaac Lab logs an episode reward as the episode sum divided by the nominal episode length in seconds, so dividing the logged value by the weight and by the realised fraction of the nominal episode recovers the time averaged sum of absolute deviations across the four joints.

| Iteration | `pen_hip_deviation` | Mean episode length | Recovered L1 sum, four joints | Mean per joint |
|---|---|---|---|---|
| 2000 | -0.0385 | 1290.65 | 0.596 rad | 0.149 rad, 8.5 degrees |
| 8000 | -0.2739 | 1888.92 | 2.900 rad | 0.725 rad, 41.5 degrees |
| 17150 | -0.1342 | 1425.23 | 1.884 rad | 0.471 rad, 27.0 degrees |

The comparison that settles whether 0.471 rad is large is the SD_BRS1 itself, whose own `pen_hip_deviation` at `cfg/SF/brs_base_env_cfg.py:733` carries the identical function and the identical weight of minus 0.1 over its two hip roll joints, those being the only off axis leg joints it has. The mature SD_BRS1 run at `IsaacLab/logs/rsl_rl/sd_brs1_flat/2026-08-17_04-26-44`, at iteration 29999 and a mean episode length of 1950.7, logs minus 0.0217, which recovers to 0.2225 rad across two joints and therefore 0.111 rad, being 6.4 degrees, per joint. The KScale runs its off axis hip joints at 4.2 times the per joint excursion of the robot it was copied from, and it does so while surviving a shorter fraction of its episode.

That figure is an average across the roll and the yaw axes and the logged term cannot separate them, so the yaw excursion alone is bracketed rather than measured. If the KScale hip roll behaves as the SD_BRS1 hip roll does, at 0.111 rad, the two hip yaws carry the balance and average 0.83 rad, being 47.6 degrees. If instead the roll and the yaw share the excursion equally, each averages 0.471 rad, being 27.0 degrees. The lower end of that bracket already exceeds the SD_BRS1 figure by a factor of four, so the conclusion does not turn on which end is nearer the truth.

The mechanism by which the excursion is bought is visible in the correlation structure of the run. Across the 17009 logged iterations from iteration 200 onward, the magnitude of `pen_hip_deviation` correlates with `rew_ang_vel_z` at 0.966 and with `rew_lin_vel_xy` at 0.852. Both are high, because in a converging run most quantities improve together and a raw correlation across training is therefore a weak instrument. The partial correlations discriminate. Controlling for linear velocity tracking, the correlation of the hip deviation with yaw tracking is 0.927, whereas controlling for yaw tracking, its correlation with linear tracking falls to 0.648. The off axis hip excursion tracks the yaw tracking reward specifically and not merely the general improvement of the run, which is what one expects if the policy is generating yaw by rotating its legs about the vertical rather than by placing its feet.

That the policy should discover this is not surprising, and section 5.1.1 records the physics. What matters here is that nothing in the reward set prices it. The full episode reward breakdown at iteration 17217 places `pen_hip_deviation` at minus 0.1464 per second against a total of plus 26.0431 and a linear tracking reward of plus 27.0038, so the entire price charged for every degree of off axis hip excursion on both legs together is 0.54 per cent of the tracking reward it purchases. Seventeen of the twenty seven terms are larger in magnitude. A policy trading 47 degrees of leg twist for a measurable improvement in yaw tracking is not defeating the reward set, it is obeying it.

The three terms a reader might expect to catch the behaviour do not. `pen_feet_distance` at `cfg/SF/kscale_base_env_cfg.py:858` hinges on the separation of the two foot frames and is blind to their orientation, a splayed pair of feet at the correct separation costing exactly nothing. The two `rew_keep_ankle_*_zero_in_air` terms regulate the ankle pitch and the ankle roll joint coordinates and say nothing about the vertical axis, which those joints do not turn about. And `pen_joint_pos_limits` reads minus 0.4084, which is real but is a limit penalty rather than a posture penalty and is in any case diffuse across twelve joints. There is no term anywhere in the KScale set that reads the direction a foot points.

One kinematic result determines how the missing term should be built and is established here because both of the following sections depend on it. Composing the origin rotations of the KScale URDF through to the foot link at the nominal pose, and taking the toe direction as the foot link's negative z axis, which section 12 of [../context/KScale.md](../context/KScale.md) establishes as the fore and aft axis of that permuted frame, the heading of the toe in the root frame is 0.995 times the hip yaw joint coordinate across the whole of that joint's travel. The ankle roll contributes nothing, its axis lying along the toe direction itself and moving the heading by 0.0001 degrees at its limit, and the three sagittal joints cannot yaw the foot at all. The direction a KScale foot points is the hip yaw angle and nothing else.

The coefficient of 0.995, however, is a property of the POSE and not of the robot, and the distinction is the more important half of the result. Section 2.3 of that document records, in its correction of 2026-08-26, that the 0.0998 tilt of the hip yaw axis out of vertical is not a hardware cant but an artefact of the nominal hip pitch flexion of 0.1 rad propagating down a chain in which the hip yaw is rigidly downstream of the hip pitch. The heading gain follows the same pose. It is 1.000 at the URDF's own zero pose, 0.995 at the nominal stance, and 1.073 and 1.524 at hip pitch flexions of 0.5 and 1.0 rad, so it varies by more than half across the travel of a joint that swings through most of that range on every step.

| Hip pitch | Heading gain per radian of hip yaw |
|---|---|
| 0.0 rad, the URDF zero pose | 1.000 |
| -0.1 rad, the nominal stance | 0.995 |
| -0.5 rad | 1.073 |
| -1.0 rad | 1.524 |

The consequence bears directly on the choice of instrument. A given hip yaw excursion turns the foot furthest at the extremes of hip pitch flexion, which is to say during swing, which is exactly when the foot's direction is being decided, so a penalty written against the joint coordinate systematically under prices the heading error at the moment it matters most. A penalty written against the toe direction prices what the foot does regardless of the pose the leg is in. This is the strongest of the three arguments section 5.1.2 gives for the task space form and it was not available when that section was first drafted, the gain having then been believed constant.

That result cuts both ways and both halves are load bearing. It means a foot heading reward and a hip yaw deviation penalty are, on this robot, very nearly the same instrument, so the case for the former over simply raising the weight of the latter must be made rather than assumed, and section 5.1.2 makes it. It also means the bracket of 0.471 to 0.83 rad computed above transfers to the foot heading error to within the pose dependence just described, which is the quantity the new term will read, and therefore that the term's magnitude at present behaviour can be predicted before it is ever run, the gain's departure from unity making that prediction a lower bound rather than an estimate wherever the leg is deeply flexed.

#### 5.1.1 Literature Survey and Related Work

The regulation of foot orientation about the vertical is a recent and thinly reported concern in the learned locomotion literature, for a reason that section 5.1 has already supplied. The exemplars from which this repository's reward vocabulary descends are quadrupeds and point footed or fixed hip yaw bipeds, and a machine that cannot rotate its foot about the vertical needs no reward that says it should not. The concern appears in the literature at precisely the point at which humanoids with a full six degree of freedom leg become the common subject, and the treatments divide into three families.

The first family regulates the joint. The Isaac Lab reference configurations penalise the summed absolute deviation of the off axis hip joints from their default posture, applying `joint_deviation_l1` to the hip yaw and hip roll joints together at weight minus 0.1 for the Unitree G1 at `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py:57-61`, and deliberately not to the hip pitch or the knee, which must swing. The comment there names the principle, that the term covers the joints which are not essential for locomotion. The Digit configuration, which is the closest true biped analogue in that tree, separates the two axes and prices them differently, carrying `joint_deviation_hip_roll` at minus 0.1 and `joint_deviation_hip_yaw` at minus 0.2 at `.../config/digit/rough_env_cfg.py:97-105`, so the vertical axis is charged at twice the rate of the fore and aft axis. This is the family the KScale already belongs to, and the comparison is unflattering, the KScale lumping both axes into a single term at the roll's rate rather than the yaw's, which is to say at half the price the nearest published analogue assigns.

The second family regulates the foot in task space, and Booster Gym is its most complete published reference [12]. Its Table II carries a feet yaw term of the form given below at weight minus 1.0, beside a feet roll term at minus 0.1, a feet slip term at minus 0.1 and a feet distance term at minus 1.0, against velocity tracking weights of 1.0 for each linear axis and 0.5 for the yaw rate.

```
Feet yaw    ||psi_feet - psi_base||^2    -1.0
Feet roll   ||phi_feet||^2               -0.1
```

The single tabulated row is not what the released work computes, and the difference is the most useful thing this survey recovers. The implementation at `envs/t1.py` carries two distinct methods. `_reward_feet_yaw_diff` squares the wrapped difference between the two feet's yaw angles and is entirely independent of the base. `_reward_feet_yaw_mean` squares the wrapped difference between the base yaw and the MEAN of the two feet's yaws, with a branch cut repair of pi added to that mean so that averaging two angles either side of the wrap does not produce a value lying between them. The configuration at `envs/T1.yaml` prices the two SEPARATELY AND EQUALLY, carrying `feet_yaw_diff` at minus 1.0 and `feet_yaw_mean` at minus 1.0, so the tabulated single weight of minus 1.0 is the weight of each of a pair rather than of one term.

The decomposition is the substance rather than a presentational detail, and the reason is that the two modes are orthogonal coordinates under the mean form and are not orthogonal under the summed form. Writing the two per foot errors against the base as e_1 and e_2, a pure differential fault, being one foot toed out by e and the other toed in by e, gives a mean form common mode of exactly zero and a differential of 4e squared, so the fault is attributed entirely to the mode that names it. The same fault under the summed form of e_1 squared plus e_2 squared gives 2e squared, so a splay registers on the common mode as well and the two terms cannot be varied independently. A pure common fault, both feet rotated together by e, gives e squared under the mean form and 2e squared under the summed. The consequence for this plan is that a design wishing to give each mode its own physical meaning, and to ablate one against the other, must adopt the mean form for the common mode, and section 5.1.2 does so.

The two modes describe physically distinct faults and both are present in the KScale run. The common mode is a stance rotated bodily away from the direction of travel, which a turning robot exhibits legitimately and which a walking one should not. The differential mode is one foot toed in against the other toed out, the pigeon toed and duck footed postures, which nothing legitimises at any commanded velocity, since a coordinated turn moves both feet the same way and therefore leaves the differential at zero by construction. The behaviour recorded in section 5.1 is predominantly differential, the feet pointing towards one another and away from one another within the same double support interval, which is the mode Booster Gym prices and which the summed form would have blurred into the common mode.

The existing `feet_yaw_alignment` in this repository at `environments/environments/tasks/locomotion/mdp/rewards.py:579` implements the summed form and therefore carries the two modes at a fixed and unadjustable ratio, pricing a splay at three times a common rotation of the same per foot magnitude rather than at the four to one the mean form gives. Its docstring states that Booster Gym carries the term at twice the linear tracking weight. Against Table II and the released configuration, each of the two weights is minus 1.0, equal to a single linear tracking component's 1.0 and twice the yaw tracking weight of 0.5, so the docstring names the wrong reference quantity. The correction is recorded here rather than in the function, which has no caller and is examined again in section 5.1.2.

The third family regulates the foot's orientation as part of a broader posture objective without isolating the vertical axis. Van Marum and colleagues carry a feet orientation term at weight 0.05 in a table whose largest entry is a sparse touchdown triggered air time reward at 1.0 [6], so foot orientation is present but is among the least consequential terms in that design, and their reported failure mode under velocity tracking alone is two footed hopping rather than foot misalignment. Humanoid-Gym shapes the swing trajectory and the contact schedule without a foot yaw term at all [9]. The absence is informative. Where a design obtains its turning from foot placement and its swing from a clock, the foot yaw axis is not the cheapest route to yaw and the policy does not exploit it, so no term is needed. The KScale's situation differs, and section 5.1 has measured the difference.

Two properties of the Booster Gym treatment bear directly on whether it may be ported, and both were checked against that work's own robot description rather than assumed. Neither term carries any contact gate whatever, the two methods being evaluated at every step irrespective of whether a foot is planted or swinging, and neither carries a dead band, the squared error being charged from the first milliradian. The plan proposed in section 5.1.2 follows the first of these and departs from the second, and section 5.1.2 quantifies both decisions rather than resting them on the precedent.

The physics of why the axis is exploited is established from the disturbance side rather than the control side. A swinging leg generates a yaw moment about the vertical which the stance foot must absorb through friction, and the effect is large enough that mechanisms have been designed for the express purpose of cancelling it [13]. Popovic, Hofmann and Herr established that whole body angular momentum is regulated to a small range throughout human walking, with segment to segment cancellation accounting for some eighty per cent of the horizontal component [14], which is the biomechanical statement that a walking machine both generates and must cancel yaw momentum continuously. A machine whose hip yaw joints are fixed, as the SD_BRS1's are, can obtain a commanded yaw only from the ground, as a friction moment beneath the sole or as the reaction to a change in whole body angular momentum, and this repository's own survey records that constraint. A machine whose hip yaw joints are free has a third and much cheaper route, which is to counter rotate the legs about the vertical and let the resulting reaction turn the torso. That route requires no friction margin, no change in foot placement and no reorganisation of the gait, so a policy rewarded for yaw rate will find it early, which is what the partial correlation of 0.927 in section 5.1 reports. The KScale's narrow sole aggravates the matter from the other side, its 0.0846 m width against the SD_BRS1's 0.194 m giving it a proportionally smaller friction moment about the vertical for the same coefficient and load, so the ground route is dearer on this robot exactly where the joint route is cheaper.

The counter rotation argument also explains why the observed fault is differential rather than common. A torque applied between the pelvis and one thigh about the vertical acts equally and oppositely upon the other, so a policy generating yaw internally does so by rotating the two legs in OPPOSITE senses about their own hip yaw axes, which is the differential mode exactly. A common mode rotation of both legs the same way carries no reaction against the torso at all and therefore buys no yaw, so a policy exploiting the joint route has no reason to produce one. The prediction is that the KScale fault should be predominantly differential, and the video of section 5.1 bears it out.

The biomechanical literature supplies the tolerance rather than the term. The foot progression angle, defined as the angle between the long axis of the foot and the line of progression, is the direct human analogue of the quantity to be regulated. Cibulka and colleagues measured it in sixty healthy adults and report a mean of 3.3 degrees of toe out with a standard deviation of 5.6 degrees and a range from 9.7 degrees of toe in to 14.3 degrees of toe out [15]. Human walking therefore does not hold the foot exactly along the line of progression, it holds it within roughly one tenth of a radian of it, and a reward that demands exact alignment demands something more than natural. One standard deviation of that distribution, 5.6 degrees or 0.098 rad, is the natural scale for a dead band on the common mode, and it is an order of magnitude below the 0.471 to 0.83 rad this policy exhibits, so the tolerance and the defect are not in danger of being confused. The same data bear on the differential mode differently. The difference of two independent draws from that distribution has a standard deviation of 5.6 times the square root of two, being 7.9 degrees or 0.138 rad, so a common tolerance applied to both modes holds the differential to a stricter standard in per foot terms than it holds the common mode. That asymmetry is deliberate and is defended in section 5.1.2, the mean toe out of 3.3 degrees being a COMMON mode which human walking does exhibit, against which no comparable systematic differential exists.

The survey leaves one question open and it must be settled by argument rather than by citation, because no surveyed source addresses it. Every published term references the foot to the robot's own base and none references it to the commanded heading, whereas the request that occasions this section is framed in terms of the direction of movement. The two coincide when the base tracks its heading command, and this configuration commands heading directly, `CommandsCfg.base_velocity` at `cfg/SF/kscale_base_env_cfg.py:156-170` setting `heading_command=True` with `rel_heading_envs=1.0` and a heading control stiffness of 0.5, so the yaw rate the robot is asked for is itself computed from the heading error. Referencing the feet to the base is therefore the correct choice on three grounds. It measures the quantity actually at fault, which is the foot against the body and not the body against the world. It avoids charging the same error twice, the base's own heading error being already priced by `rew_ang_vel_z` at weight 15. And it remains well defined when the commanded velocity is zero, where a heading referenced term would be regulating the feet against a direction of movement that does not exist. The base referenced form satisfies the requirement that the feet face the direction of movement precisely because the base is what faces the direction of movement, and it satisfies the requirement that the feet twist only to turn without any special case, since during a turn the base yaws and a foot that follows it incurs no error.

The survey above was written before the term existed and it asked which reward would express the fault. The run of 2026-08-26 has since answered a different and more consequential question, whether a reward is the right kind of instrument at all, and the literature bears on that question in a way the original survey had no occasion to consult. What follows is the second pass, added 2026-08-28, and it is organised around the failure the run actually produced rather than around the term the run was built to test.

The general result is that a shaped penalty inside a summed objective is a price rather than a prohibition, and a policy is free to pay it. Skalse and colleagues make this formal, proving that a proxy reward is unhackable with respect to a true reward only under conditions that a weighted sum of competing terms does not satisfy, so that nothing in a squared heading error distinguishes a splay bought by counter rotating the hips from a splay avoided by stepping to turn, over the interval the term integrates [16]. Pan, Bhatia and Steinhardt show that the severity of such misspecification grows rather than shrinks with policy capability, since a more competent optimiser finds the cheap corner of the objective more reliably [17]. Reda, Tao and van de Panne supply the closest structural analogue in legged locomotion, demonstrating that a survival bonus sized wrongly produces a policy that balances and never steps, which is the same pathology as the present one, a term intended to shape gait instead purchasing a degenerate equilibrium [18]. These three establish that the observed outcome, a penalty absorbed as a cost rather than obeyed as a constraint, is the expected behaviour of the instrument rather than a defect of its tuning, and they are the reason the second pass looks past the weight.

The constraint family is the one that answers this directly, and within it the decisive distinction is between a constraint expressed as a subtracted cost and a constraint expressed as a termination. Kim and colleagues reformulate many hand tuned kernel terms as constraints that are zero inside a range read from the URDF and grow outside it, which repairs the vanishing gradient of a peaked reward but leaves the cost inside an unconstrained sum the policy may still elect to pay [19]. Chane-Sane, Leziart, Flayols, Stasse, Souères and Mansard depart from that on exactly the axis this investigation needs, converting a constraint violation into a probability of terminating the episode rather than into a subtracted reward, so that violating the constraint threatens the entire remaining return rather than discounting the instant in which it occurs [20]. The consequence is that there ceases to be a fixed exchange rate between the style requirement and the tracking objective, because one of the two can now end the accumulation of the other, and it is precisely a fixed exchange rate that the measurements of this run show the policy exploiting. The method is reported evaluated on obstacle crossing with the Solo quadruped and the retrieved source does not establish that it has been applied to a foot orientation or hip deviation constraint on any biped, nor does the arXiv record state the venue, so the transfer to the present fault is an extrapolation from the mechanism rather than a reported result, and it is recorded here as the most directly responsive published idea rather than as a validated recipe.

The imitation family would also suppress the fault but inherits the objection rather than escaping it. Peng and colleagues extend the adversarial motion prior into a latent conditioned family of reusable skills trained against a large unstructured dataset, so that a downstream reward selects among learned skills rather than shaping one from nothing [21]. Wu, Wang, Ye and Xing report applying such a prior selectively, retaining it for periodic stability critical gaits where it suppresses erratic behaviour and omitting it deliberately for highly dynamic gaits on the stated ground that its regularisation would over constrain motion the reference dataset does not cover [22]. That second finding is the material one for this plan, since it establishes from published practice that a style prior strong enough to forbid one behaviour can also forbid behaviours the task legitimately needs, which is the same failure mode in reverse that the ungated heading term risks against turning. Neither retrieved source states that its discriminator prices foot yaw or hip counter rotation specifically, so the claim that an adversarial prior would suppress this fault is an inference from the mechanism, and in any case a discriminator reward remains a reward inside a sum, so the family costs a reference dataset this configuration does not have and does not remove the exchange rate that the run has shown to be the problem.

The explicit placement family is conceptually the cleanest and is the least well served by released implementations. Singh, Benallegue, Morisawa, Cisneros and Kanehiro condition the policy on the upcoming planned footsteps rather than on a velocity command alone and report that this suffices for omnidirectional walking, though the retrieved source does not establish whether each planned footstep carries a target heading in addition to its position [23]. The structural observation is negative and it is worth stating plainly, every source this document has retrieved that scores a foot's orientation, Booster Gym's paired yaw terms [12], van Marum's feet orientation term [6] and the Isaac Lab joint deviation configurations, expresses that orientation as a reward to be traded against other rewards, and none of the footstep planning literature retrieved here folds heading into the planner's output as a hard target the controller must track. The field's dominant practice is therefore to price foot orientation rather than to command it, which is why this plan's original decision to price it was conventional, and why the run's outcome is evidence about the convention rather than about this implementation of it. The clinical literature offers the human counterpart, a pilot randomised trial reporting that adults can be retrained to a prescribed foot progression angle and sustain it under real time feedback rather than under mechanical constraint [24], which is a style target achieved by changing behaviour under feedback, in contrast to the present policy's choice to retain the behaviour and absorb the feedback as a cost.

Two families that appear promising must be recorded as dead ends for this particular fault, and the reasons are structural rather than empirical. The symmetry family, surveyed for this workspace in the context registry and comprising the mirror loss of Yu, Turk and Liu [5], the taxonomy of Abdolhosseini and colleagues [2] and the augmentation and equivariance comparisons that follow them [1] [4], cannot see the fault at all. Writing the two base relative foot headings as e1 and e2, the left right mirror maps e1 to minus e2 and e2 to minus e1, under which the common mode is anti invariant and the differential mode is invariant, so a pure splay of plus and minus 0.30 rad has a differential of minus 0.600 both before and after mirroring. A perfectly mirror symmetric policy is therefore entirely free to turn both feet inward or both outward, and a mirror loss would constrain only the common mode, which section 5.1.2 records as already measured at essentially zero. Enabling the mirror loss would regularise the half of the decomposition that is not broken. The energy family is a dead end for a different reason, since it prices the torque the counter rotation costs rather than the heading it buys, and would have to be sized large enough to make that specific torque expensive without also flattening the hip extension a normal step requires, a trade off no surveyed source quantifies for this joint.

The two remaining directions are supported by argument rather than by precedent, and the survey records the absence of precedent as a finding in its own right. For swing gating a joint level regulariser, the dominant published practice for off axis hip regulation is continuous and ungated, the Isaac Lab G1 and Digit configurations pricing `joint_deviation_l1` on hip roll and hip yaw at every instant, so the swing gate contemplated here is a departure from the established convention rather than a documented refinement of it. The nearest published analogue encodes a per leg phase and adds a swing phase contact penalty [25], but that penalises ground contact during scheduled swing rather than an off axis joint angle during swing, the retrieved source does not confirm whether the penalty is strictly zero outside the swing window, and no ablation against an always on version was retrieved. For locking a degree of freedom and releasing it later, no retrieved source reports training a biped with its hip yaw locked and subsequently unlocking it. The nearest published mechanism grows the action space over training through a tanh transform whose scale increases, reported to reduce gradient variance early, but it restricts action magnitude across every joint simultaneously rather than any joint selectively [26], so it is evidence for curricula on action space breadth in general and not for staged release of one off axis joint. Both directions are therefore proposed below as experiments rather than as reproductions, and the plan should not lean on a citation where it in fact leans on an argument.


The two passes above were each written against a question the preceding run had raised, and the runs of 2026-08-31 and 2026-09-01 have displaced the question a third time. What follows is the third pass, added 2026-09-02, and it is organised around a phenomenon that neither earlier pass anticipated, namely that the fault was not removed by pricing it but relocated to an axis no term reads.

The measurement that occasions this pass is set out in full in section 5.1.2 and is stated here only so far as it directs the search. Between the run of 2026-08-31 and the run of 2026-09-01 the hip yaw excursion fell by 33 per cent while the hip roll excursion rose by 72 per cent, and the sum of the two stood still, 0.807 against 0.790 at matched iterations, a difference of two per cent on a quantity whose components moved by a third and by three quarters. The policy did not abandon its off axis expenditure, it reallocated it from the axis this plan had learned to charge for onto the axis it had not.

Wang and Huang establish that this is the expected behaviour of an optimised agent rather than an accident of these particular weights, proving under five stated axioms that an agent will systematically under invest in any quality dimension its evaluation does not cover, and deriving from that an index predicting the direction and the severity of the resulting distortion before deployment [27]. The material claim for this plan is their account of substitution, which they present as a structural equilibrium property of finite evaluation rather than as a contingent empirical observation, so that patching one exploited dimension is expected to produce a new exploitation along an unmonitored one. Their domain is alignment theory and their examples are drawn from language model training, so the transfer to a locomotion reward is by mechanism and not by reported result, and it is recorded here on that footing. The entry belongs beside Skalse and colleagues [16] and Pan and colleagues [17], which the second pass cited for the proposition that a proxy inside a weighted sum may be paid rather than obeyed, and it extends them in the one direction the second pass did not consider, that the payment may be made in a currency the sum does not itemise.

The remedy the literature offers for a fault of this shape is not another term but a different arrangement of the terms that exist, and the most complete published instance is the barrier based style framework of Kim, Lee and Park [28]. Two features of it bear on this plan and they are separable. The first is the form of the style reward itself, a relaxed logarithmic barrier which behaves as a logarithm while the constrained quantity remains inside its bound and transitions to a quadratic once the bound is violated, so that the gradient remains finite and informative on the wrong side of the constraint where an unrelaxed logarithm would be undefined. This is an instrument intermediate between the price this plan has been levying and the termination its second pass proposed, in that the penalty grows without limit as the bound is approached from within and therefore cannot be absorbed at a fixed exchange rate, while the episode is never ended and the credit assignment never becomes sparse. The second feature is architectural, the framework carrying independent critics which process the barrier rewards and the standard task and regularisation rewards exclusively, with the advantages normalised before the surrogate is formed so that the relative magnitudes of the two groups are preserved rather than being set by their raw weights. The authors present this as reducing the sensitivity of the outcome to the particular weights and functions chosen, and they report the framework carrying biped, tripod and quadruped locomotion on a 45 kg machine without exteroceptive input. They apply the barrier form to gait timing, foot clearance, joint position, body height, target velocity, base motion and joint velocity, so foot orientation is not among the quantities they constrain and the application to it is an extrapolation.

The architectural half of that framework is the published form of what the run of 2026-09-01 performed by hand and without recognising it as such, and the recognition is the most useful thing this pass recovers. That run divided the reward set into a group whose weights were reduced by a factor near a third and a group whose weights were left untouched, and the division falls almost exactly along the boundary between task and gait shaping terms on the one hand and posture, style and regularisation terms on the other. The consequence, developed in section 5.1.2 and confirmed by the group aggregation of section 7.12, is that the run was not a reduction of the tracking reward, which is how it was conceived, but a promotion of the style and regularisation group by a factor of roughly 3.3 relative to everything it competes with, which is the same operation the multi critic performs continuously and automatically.

Li, Wu, Liu, Guo and Xue offer the temporal rather than the architectural separation, training first on flat ground under gait related rewards to acquire what they describe as natural and robust movement, and only then learning difficult terrain by adversarial imitation of the experience the first stage generated [29]. The separation of concerns is the same as the multi critic's and the mechanism is different, the groups being visited in sequence rather than weighted in parallel, and the relevance here is that this plan already carries a curriculum instrument capable of expressing it, `modify_reward_weight` at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/curriculums.py:24`. The retrieved abstract reports the outcome qualitatively as natural gait patterns on a physical quadruped and supplies no quantitative measure of naturalness, so the entry grounds the staging as an established practice and does not license a prediction of its magnitude.

Two entries are added for the narrow stance that the run of 2026-09-01 produced, and both are recorded with their limitations because neither answers the question directly. Xie, Bai, Shi, Yang, Ge, Zhang and Li train a humanoid to walk on extremely narrow terrain by extending the zero moment point into a reward and pairing it with task rewards under a whole body actor critic, reporting balance maintained under external disturbance from proprioception alone [30]. The retrieved source establishes that a narrow support is a trainable regime and does not report the trade off this plan needs, namely what a policy gains in lateral stability by drawing its feet toward the midline and what it forfeits in disturbance rejection, so the entry marks the regime as studied rather than supplying its economics. The anthropometric target comes from Hollman, McDade and Petersen, who measured spatiotemporal gait in 294 adults aged seventy and above and report a step width between 7.0 and 9.9 cm against a step length between 54 and 69 cm and a walking speed of 110 plus or minus 19 cm per second [31]. Two cautions attach to the use of that figure. The cohort is elderly, and older adults are widely held to walk with a wider base than young adults, so the value is if anything an upper bound on the human proportion rather than a central estimate of it. And step width is measured between successive foot placements rather than between the two feet at an instant, so it is the correct analogue of a stance width parameter and not of an instantaneous separation between two foot frames, which bears directly on the defect in the present parameterisation that section 5.1.2 records.

The pass closes with the negative results, which are recorded because they cost search effort and because their absence is itself informative. No retrieved source reports measuring the redistribution of an off axis excursion from one joint axis to another under a task space penalty, so the substitution documented here is offered as an observation of this robot rather than as a reproduction, and the general result that predicts it [27] is drawn from a different field. No retrieved source prices a biped stance width against a lateral disturbance rejection margin, so the question of how narrow is too narrow must be settled on this robot by the push curriculum rather than by citation. And no retrieved source reports a barrier form applied to foot orientation or to hip yaw, so the proposal below to adopt one is an extrapolation from the instrument to a quantity its authors did not constrain, on the same footing as the second pass's extrapolation of the termination mechanism [20].

#### 5.1.2 Implementation Plan

This section was rewritten on 2026-09-03 against the evaluation of three policies. Everything it previously carried by way of derivation now lives in section 5.1.1, in section 3 and in section 7, and is not repeated here. What remains is the state of the implementation, the record of what has been run, the measurements those runs produced, and the configuration proposed next.

##### What is implemented

`feet_yaw_alignment` at `environments/environments/tasks/locomotion/mdp/rewards.py:579` carries the six optional arguments below, every default reproducing the original behaviour, and it has one caller, the KScale configuration. The body is not reproduced here, the shipped code being the authority and a copy in a plan being free to drift from it.

```python
def feet_yaw_alignment(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    forward_axis: tuple[float, float, float] | None = None,
    common_mode: str = "sum",
    tolerance: float = 0.0,
    differential_scale: float = 0.0,
    sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 1.0,
    history_index: int = 0,
    airborne_only: bool = False,
) -> torch.Tensor:
```

Two properties of it govern every arm below. The heading is taken by rotating a named link axis into the world rather than by an Euler decomposition, because the KScale foot link frame is a full axis permutation from the convention Booster Gym's own robot uses [12] and the Euler path reads a heading 70 to 110 degrees from the true one on this robot. And `common_mode="mean"` is required rather than preferred, being the only form under which the common and the differential modes are orthogonal, so that the ablation can separate them. Twenty five checks in `scripts/analysis/kscale_feet_heading_selftest.py` establish both, and section 7.8 records the pass.

`feet_distance` at `rewards.py:564` carries `lateral_only` at `:569`, defaulting to False, whose else branch at `:602` preserves the planar Euclidean form for the four TRON1 and SD_BRS1 callers bit for bit. The KScale configuration is the only caller that sets it. The lateral form is correct on this robot only because the root frame correction of section 4.2 landed, the branch keeping the component of index one, which is the lateral axis under the Isaac Lab convention and was the fore and aft axis as exported.

The reward set carries thirty terms. `pen_hip_deviation` is split into `pen_hip_roll_deviation` and `pen_hip_yaw_deviation`, which is exact for an L1 sum and moves no incentive, and `pen_ankle_deviation` is removed.

##### The experiments, and what each was for

| Run | Change from its predecessor | Rationale | Outcome |
|---|---|---|---|
| `2026-08-26` | `pen_feet_heading` at minus 2.0, hip deviation split | Price the foot heading in the task space, no term reading the hip yaw coordinate | Off axis excursion down 37 per cent, yaw tracking down 47 per cent |
| `2026-08-28_04-50-51` | `pen_feet_heading` raised to minus 4.0, `keep_balance` lowered to 0.05, run to 30000 iterations | Establish the reference gait | Best gait of the sequence, worst splay, style surrendered monotonically after iteration 4000 |
| `2026-08-31_04-57-06` | add `rew_keep_hip_yaw_zero_in_air` at 1.0, restore `keep_balance` to 0.5 | Arm G | Stiff legged gait, sections 7.10 and 7.14 |
| `2026-09-01_06-53-24` | eleven task and gait terms scaled by 0.10 to 0.50, nineteen left alone | Let a natural gait outweigh velocity tracking | Stiffness removed, stance narrowed, splay persisted, section 7.11 |
| `2026-09-02_06-59-31` | Baseline 3, being the above plus `pen_feet_distance` at minus 30 over a 0.14 m lateral threshold | Arrest the substitution at the outcome rather than at the joint | Splay bias cut 70 per cent, stance width floor honoured, the robot stopped walking |
| `2026-09-02_10-51-29` | Baseline 3 with the hip yaw travel reduced to plus and minus 0.22 rad | Test whether the range of motion is the enabling condition | Best training survival of the sequence, yaw tracking restored fourfold, gait further degraded |

The last of these carries a caveat that governs its use. Its policy trained against hip yaw stops at plus and minus 0.22 rad and was evaluated against plus and minus 1.5708, the URDF change having been reverted before the evaluation was run, and the dumped `joint_position_limits` in all three evaluation dumps read plus and minus 1.5708 identically while the observed hip yaw in that policy's own evaluation reaches 1.586 rad. Its training record is admissible and its evaluation dump is not, which is why its mean episode length reaches the full 2000 steps in training and 89.3 per cent of its evaluation episodes terminate early. The two `params/env.yaml` files of the 09-02 series are byte identical, so no dumped artefact records the change at all, and the lesson is that a change made in the asset rather than in the environment configuration leaves no provenance.

##### What the runs measure

Raw term rates at the matched window common to all three, iterations 9400 to 9600, recovered as `Episode_Reward` divided by the weight and by the episode length in units of 2000 steps, so that they are comparable across runs whose weights differ. Higher is better for a reward and lower for a penalty.

| Quantity | 0828 | 0902a | 0902b | Reading |
|---|---|---|---|---|
| `feet_air_time` | 0.0735 | 0.0161 | 0.0109 | Swing all but abolished |
| `rew_foot_clearance` | 0.302 | 0.116 | 0.078 | Foot held at the ground |
| `feet_slide` | 0.242 | 0.515 | 0.556 | Dragging, up 113 per cent |
| `rew_gait` | -0.229 | -0.447 | -0.409 | Phase match halved |
| `pen_feet_heading` | 0.0891 | 0.0274 | 0.0074 | The heading term works |
| `pen_hip_yaw_deviation` | 0.368 | 0.292 | 0.141 | Yaw excursion down |
| `pen_hip_roll_deviation` | 0.181 | 0.203 | 0.181 | Roll excursion held |
| off axis total, yaw plus roll | 0.549 | 0.495 | 0.322 | The substitution is arrested, not merely relocated |
| `pen_feet_distance` | 0.00895 | 0.00363 | 0.00566 | Stance width floor honoured |
| `rew_keep_hip_yaw_zero_in_air` | absent | 0.352 | 0.345 | Held under a threefold change in relative weight |
| `rew_lin_vel_xy` | 0.809 | 0.843 | 0.879 | Tracking improved under a threefold weight cut |
| `rew_ang_vel_z` | 0.102 | 0.037 | 0.293 | Yaw tracking collapses, and the hip yaw bound restores it |
| `error_vel_yaw` | 3.79 | 7.52 | 1.78 | The same finding in the metric |
| `pen_ang_vel_xy` | 0.779 | 0.283 | 0.255 | Lateral disturbance down |
| `pen_flat_orientation` | 0.0108 | 0.0025 | 0.0031 | Posture better |
| `pen_action_smoothness` | 65.0 | 19.5 | 24.2 | Action motion down 70 per cent |
| `pen_undesired_contacts` | 0.00047 | 0.00036 | 0.00339 | Not being ignored at 0902a |
| `base_contact` termination | 0.188 | 0.059 | 0.052 | Falls down 69 per cent |
| `mean_episode_length`, of 2000 | 1551 | 1840 | 1855 | Survival improved |
| `Policy/mean_noise_std` | 0.646 | 0.373 | 0.378 | Converged rather than exploring |

Gait statistics from the evaluation, 32 environments and 3001 steps each, computed by `scripts/analysis/stats.py`. The 0902b column is quoted only where the quantity is a property of the policy rather than of the mismatched stops, and is parenthesised where it is not.

| Quantity | 0828 | 0902a | 0902b | Unit |
|---|---|---|---|---|
| double support | 41.1 | 51.1 | (64.8) | pct |
| single support | 56.1 | 47.7 | (33.3) | pct |
| stance duty, per foot | 69.8, 68.6 | 74.9, 75.0 | (81.2, 81.7) | pct |
| swing apex clearance | 34.8 | 27.3 | 29.7 | mm |
| steps with sole clearance above 20 mm | 9.4, 11.8 | 4.6, 4.8 | (2.3, 1.7) | pct |
| fore and aft separation, mean magnitude | 152.3 | 79.5 | (67.0) | mm |
| lateral separation, mean | 242.6 | 203.1 | 218.2 | mm |
| lateral separation, 5th percentile | 108.3 | 143.1 | 160.7 | mm |
| stance slip per stance step | 4.2 | 5.0 | 5.5 | mm |
| foot speed in stance, mean | 0.420 | 0.503 | 0.545 | m/s |
| forged contact | 0.159 | 0.588 | 0.228 | pct |
| torso tilt from vertical | 6.60 | 3.47 | 5.55 | deg |
| centre of pressure offset, lateral rms | 103.6 | 77.3 | 74.0 | mm |
| peak contact force, mean | 2.37 | 1.84 | 1.64 | body weights |
| `foot_roll_02` within 0.02 rad of a stop | 80.5 | 36.4 | (86.4) | pct |
| `hip_roll_03` within 0.02 rad of a stop | 11.7 | 1.14 | (0.70) | pct |
| `knee_04` within 0.02 rad of a stop | 2.11 | 12.6 | (4.95) | pct |
| episodes terminated early | 81.3 | 62.2 | (89.3) | pct |

The splay, measured signed for the first time in this investigation, by rotating each foot link's negative z axis into the world, projecting it onto the horizontal and differencing against the base heading. `foot_6061_2` is the left foot and `foot_6061` the right, established from the URDF at `kscale.urdf:1070` and `:590` rather than from the name.

| Quantity | 0828 | 0902a | 0902b | Unit |
|---|---|---|---|---|
| left foot heading, mean | -7.70 | -2.61 | -2.94 | deg |
| right foot heading, mean | +7.46 | +1.97 | +2.34 | deg |
| symmetric inward splay, mean | 7.58 | 2.29 | 2.64 | deg |
| per foot magnitude, mean | 11.2, 11.0 | 9.12, 8.84 | 9.27, 9.83 | deg |

Both toes converge on the midline in all three runs. The gait is pigeon toed, which section 5.1.1 could not establish and which the previous two passes of this section left open. Baseline 3 removes 70 per cent of the bias while removing only 19 per cent of the magnitude, so what remains is symmetric jitter about zero rather than a postural bias, which is the outcome the term was added to produce.

The reward budget by group, taken from the evaluation as the absolute weighted rate summed within each group, which is what the policy gradient sees after the advantages are normalised at `rsl_rl/rsl_rl/algorithms/ppo.py:191`.

| Group | 0828 | 0902a | Ratio against the gait block, 0828 | Ratio against the gait block, 0902a | Change |
|---|---|---|---|---|---|
| task | 38.40 | 12.43 | 1.043 | 0.969 | 0.93 |
| gait shaping | 36.84 | 12.83 | 1.000 | 1.000 | 1.00 |
| style | 0.885 | 0.835 | 0.024 | 0.065 | 2.71 |
| stability demand | 14.64 | 5.60 | 0.397 | 0.436 | 1.10 |
| regularisation | 4.57 | 3.08 | 0.124 | 0.240 | 1.94 |
| survival | 0.050 | 0.500 | 0.0014 | 0.0390 | 28.7 |

##### What the measurements establish

The reweighting of 2026-09-01 achieved what it was for. The splay bias fell 70 per cent, the stance width floor was honoured for the first time, the off axis excursion fell 10 per cent in sum rather than merely moving between axes as it had in every previous pass, falls fell 69 per cent, posture and lateral disturbance improved by more than half, and velocity tracking improved rather than degraded under a threefold cut in its own weight. None of these is small and none should be surrendered.

It paid for them by ceasing to walk. Air time fell 78 per cent, clearance 62 per cent, step length by half, and the fraction of steps in which a sole stood more than 20 mm off the ground fell from 10.6 to 4.7 per cent, while double support rose from 41 to 51 per cent against a commanded 24 and slide rose 113 per cent. The human evidence is that these covary as one pattern rather than as separate faults, a raised double support fraction being accompanied by reduced hip flexion, reduced knee flexion and reduced swing foot height at a fixed walking speed [32], which is exactly the joint of measurements above and which makes the double support fraction the correct single number summary of the defect.

The cause is a change of ratio that the two group description of that run concealed. The task and the gait blocks were cut together and their ratio barely moved, 1.043 to 0.969, so the gait block was not demoted against the task. It was demoted against everything else. Style rose 2.71 fold against it, regularisation 1.94 fold and the survival bonus 28.7 fold. Those three blocks share one property, that each is cheaper to satisfy when the robot does not leave the ground, and the gait block is the only block that pays for leaving it. A policy facing that budget has no reason to swing, and the trend confirms that it did not merely fail to learn to, every gait quantity of the 2026-09-01 configuration reaching its plateau by iteration 6000 and not moving over the following fourteen thousand, against a reference run in which every one of them improved monotonically for thirty thousand.

That velocity tracking did not suffer is the load bearing negative result. A shuffle tracks 0.86 m/s as well as a walk does, so the task term cannot distinguish them, which is the formal position the specification gaming literature takes and which this workspace has now confirmed on two robots [16] [17] [27]. Only the gait block distinguishes them, and it was the block that was cut.

Two mechanical defects are visible in the same evaluation and are not reward faults at all. The ankle roll joint stood within 0.02 rad of a mechanical stop for 80.5 per cent of steps in the reference run and 86.4 per cent in the last, on a joint whose entire travel is plus and minus 0.2618 rad, at a stiffness of 20 Nm per radian. Section 7.7 records that same joint pinned at its stop at a stiffness of 5 and raised it to 20, and the measurement says 20 was not enough either. And the hip yaw carried a stiffness of 15 against 150 to 200 at every other joint, giving the highest mean joint speed in the robot at 7.3 rad per second and an acceleration near 1300 rad per second squared at an amplitude of only 0.15 rad, which is a joint buzzing rather than tracking. The consequence is measured, the base yaw rate reaching a magnitude of 4.5 to 6.5 rad per second against commands near 0.6 with a command correlation of zero, and the one run that pinned the joint mechanically restored the yaw tracking eightfold. The published position is that a low proportional gain leaves a joint behaving as a torque source with large tracking errors while an excessive one destabilises training [33], and this robot has been trained four times on the first of those.

The splay has a geometric cause specific to this robot and it explains why every instrument so far has relocated it rather than removed it. The sole is 0.2100 m long and 0.0846 m wide, a slenderness of 2.48, so yawing a foot converts its length into lateral extent. At the 7.7 degrees measured in the reference run each foot gains 31.5 per cent of lateral base of support, 84.6 mm becoming 111.3 mm, at no cost in joint effort and no cost in any term that reads a joint coordinate. Splay is the cheapest lateral support this robot can buy, which is why pricing the hip yaw moved the excursion to the hip roll, and why the lateral demand of `pen_flat_orientation` at minus 50 and `pen_ang_vel_xy` at minus 5, identical in all four runs, keeps re-purchasing it. The same arithmetic bounds the remedy, the two toes converging by 27.4 mm of inner gap at that angle against a fifth percentile gap of 23.7 mm in the reference run, so at the narrow tail of that run's stance distribution the toes crossed.

The splay is a standing postural bias rather than a reflex, and this matters for the choice of instrument. Its within run correlation against instantaneous lateral velocity, roll rate and lateral projected gravity lies between minus 0.26 and plus 0.17 across the three runs, which is no coupling, so it is not a response to disturbance as it arises. It is a posture adopted once and held, which is why a price removes it and did.

##### The configuration proposed next

The design follows from the two findings that survive. The gait block must be restored, because it was the only block cut and the shuffle is what the cut bought. The style block must not be returned to its former relative strength, because at 0.024 against the gait block the reference run splayed at 7.58 degrees and at 0.065 it did not. The construction that satisfies both is to return to the reference weight set and carry forward only the changes the measurements support, which places the style ratio near 0.03 by arithmetic rather than by choice. The swing gate is not among those changes. Section 7.14 shows that the run which introduced it is the only single variable comparison implicating it and that the comparison exonerating it spans a reweighting of the whole reward set, so its effect is unknown rather than established, and it is given its own arms rather than being folded into a baseline whose other changes would then be confounded with it.

Three code changes are required and are stated in full. The first is the separation term, whose parameters are unchanged from Baseline 3 and whose weight is unchanged, this being the change the measurements most clearly endorse.

```python
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-30,
        params={
            # Ported from the SD_BRS1 on the inner-edge clearance between the soles, not on
            # the fraction of the nominal separation. The BRS demands 0.056 m of clearance
            # (0.25 against a 0.194 m sole), and the same clearance on this robot's 0.0846 m
            # sole gives 0.141. The fraction port, which produced the previous 0.24/0.26, is
            # wrong here because the two nominal separations differ by 3% while the sole
            # widths differ by 2.29x. Measured outcome at 0.14: the 5th percentile stance
            # width rose from 108.3 mm to 143.1 mm. See plans/kscale_integration.md 5.1.2.
            "min_feet_distance": 0.14,
            "feet_links_name": [_FOOT_LINKS],
            # Base-frame lateral component only, so the term measures stance width rather
            # than the planar norm, which conflates it with step length. Correct on this
            # robot only because the root frame correction of section 4.2 landed.
            "lateral_only": True,
        },
    )
```

The second is the swing gated hip yaw regulariser, which arms 0G and 4G add and which the baselines omit. It is given at the weight it was run at and not at the 0.35 the working tree once carried, so that the arms repeat the intervention rather than a weaker version of it.

```python
    rew_keep_hip_yaw_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(right|left)_hip_yaw_03"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
        },
    )
```

The third is the termination that expresses the hip yaw requirement as a constraint rather than as a price, which has still never been run and which the constrained reinforcement learning literature argues for over a price [20]. It is not part of the baseline and belongs to arm H.

```python
    hip_yaw_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(right|left)_hip_yaw_03"]),
            "bounds": (-0.35, 0.35),
        },
    )
```

The weight set proposed as Baseline 4 is the 2026-08-28 set with a single change, and the restraint is deliberate. The three preceding sections have established that the pigeon toed splay is bought by a geometric demand rather than paid for by a style price, so the instrument that removes it is the parameterisation of the term that makes the demand, and every other weight is left where the one run that walked well had it. The Baseline 3 column is retained so that the reversal of the earlier reweighting is visible.

| Term | Group | 2026-08-28 | Baseline 3 | Baseline 4 | Why |
|---|---|---|---|---|---|
| `keep_balance` | survival | 0.05 | 0.5 | 0.05 | The 28.7 fold promotion against the gait block pays for standing and requires nothing |
| `rew_lin_vel_xy` | tracking | 50 | 15 | 50 | Held at the reference, this being the only large positive in the budget and the term whose reduction is measured below to extinguish exploration |
| `rew_ang_vel_z` | tracking | 15 | 6 | 15 | Restored, the cut coinciding with a fall of raw yaw tracking from 0.102 to 0.037 |
| `rew_no_fly` | gait shaping | 15 | 6 | 15 | Restored with its block |
| `feet_air_time` | gait shaping | 12.5 | 4.5 | 12.5 | Restored, the term whose measurement fell furthest |
| `rew_foot_clearance` | gait shaping | 10 | 3 | 10 | Restored with its block |
| `pen_foot_landing_vel` | gait shaping | -30 | -6 | -30 | Restored with its block |
| `feet_slide` | gait shaping | -5 | -1.5 | -5 | Restored, slide having risen 113 per cent under the cut |
| `rew_gait` | gait shaping | 40 | 7.5 | 40 | Restored with its block |
| `pen_base_height` | gait shaping | -30 | -10 | -30 | Restored with its block |
| `pen_feet_impact` | gait shaping | -0.03 | -0.03 | -0.03 | Unchanged in every run |
| `pen_feet_regulation` | gait shaping | -0.2 | -0.2 | -0.2 | Unchanged in every run |
| `pen_feet_distance` | style, outcome | -100 | -30 | -100 | Weight held at the reference, `min_feet_distance` moved to 0.14 and `lateral_only` set True, this being the whole of the change and its ground being given below |
| `pen_feet_heading` | style, outcome | -4 | -2 | -4 | Held at the reference, the heading price being shown below not to be the operative instrument on this quantity |
| `pen_hip_yaw_deviation` | style, joint | -0.1 | -0.3 | -0.1 | Held at the reference, for the same reason |
| `pen_hip_roll_deviation` | style, joint | -0.1 | -0.1 | -0.1 | Held, deliberately not raised, for the reason below |
| `rew_keep_hip_yaw_zero_in_air` | style, joint | absent | 1 | absent | Withdrawn from the baseline, section 7.14 having re-indicted it, and supplied instead by arms 0G and 4G so that its effect is read at two reward scales |
| `rew_keep_ankle_pitch_zero_in_air` | style, joint | 1 | 1 | 1 | Unchanged in every run |
| `rew_keep_ankle_roll_zero_in_air` | style, joint | 0.25 | 0.25 | 0.25 | Unchanged in every run |
| `pen_flat_orientation` | stability demand | -50 | -50 | -50 | Unchanged in every run, and the source of the lateral demand the splay answers |
| `pen_ang_vel_xy` | stability demand | -5 | -5 | -5 | Unchanged in every run |
| `pen_lin_vel_z` | stability demand | -0.5 | -0.5 | -0.5 | Unchanged in every run |
| the eight regularisation terms | regularisation | as run | as run | as run | Unchanged, their 1.94 fold promotion being corrected by restoring the gait block rather than by cutting them |

Beside the weights, Baseline 4 fixes two quantities that are not weights at all and that the sequence has since measured to matter more than any of them. The actuator gains are the identified set at `environments/environments/assets/config/kscale_identified_cfg.py`, hip yaw 500 and 18, hip roll 250 and 10, hip pitch 200 and 8, knee 300 and 10, ankle roll and ankle pitch 170 and 9. The action scale at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:148` is 0.25 rather than the 0.4 every run before 2026-09-04 carried. Both are load bearing, and the evidence for each is set out below in its own place.

##### The stance width demand is the cause of the splay, and its parameterisation is the remedy

The splay has a geometric cause this document has already recorded. The sole measures 0.2100 m by 0.0846 m, a slenderness of 2.48, so the lateral base of support of a sole yawed through an angle is the length times the absolute sine plus the width times the absolute cosine, and 7.7 degrees of yaw raises it by 31.5 per cent at no cost in any term that reads a joint coordinate. Splay is the cheapest lateral support this robot can buy. What the sequence adds is the identity of the demand that makes it necessary, and the measurement is unambiguous.

`pen_feet_distance` at `environments/environments/tasks/locomotion/mdp/rewards.py:564` measures, under its default `lateral_only=False`, the planar Euclidean norm between the two foot link origins, and penalises the shortfall against `min_feet_distance` through a unit clipped hinge. At 0.24 m that hinge is active on 47.7 per cent of samples in the reference run and on 49.4 and 55.0 per cent in the two runs on the identified gains, so the demand is not an occasional correction but a standing pressure. The robot answers it in the only way its travel permits. Outward abduction at the hip roll is capped near 0.209 rad by asymmetric joint limits, left at `[-0.2094, 2.2689]` and right at `[-2.2689, 0.2094]`, and in the reference run the hip roll therefore stands within 0.02 rad of its mechanical stop on 78.9 and 77.8 per cent of samples. With abduction exhausted the remaining width must come from somewhere, and the correlation says where. The correlation between toe-in and the base frame lateral separation is plus 0.869 in the reference run, plus 0.524 in arm 0 and plus 0.733 in arm 0S, while the correlation between toe-in and the planar separation the term actually reads is negative in all three. Toe-in is the instrument by which the stance is widened once the hip roll can widen it no further.

The one experiment that relaxed the demand settles the matter. Baseline 3, the run `2026-09-02_06-59-31`, moved `min_feet_distance` to 0.14 and set `lateral_only` True at unchanged gains, and the hinge fell from 47.7 to 4.2 per cent active, the hip roll stop residency from 78.9 to 1.1 per cent, the mean stance toe-in from 13.07 to 2.97 degrees, the hip yaw convergence from 18.66 to 2.41 degrees, the fraction of stance samples beyond ten degrees of toe-in from 0.690 to 0.307 and the left to right asymmetry from 26.16 to 5.94 degrees. That is a 4.4 fold reduction in the splay and a 7.7 fold reduction in the joint coordinate that produces it, obtained from a parameter and not from a price.

The same comparison acquits the two style prices, and this is why Baseline 4 returns both to the reference. Baseline 3 HALVED `pen_feet_heading` from minus 4 to minus 2 and TRIPLED `pen_hip_yaw_deviation` from minus 0.1 to minus 0.3 in the very run in which the splay fell by a factor of four. A term whose price was cut while the quantity it prices improved fourfold is not the instrument that produced the improvement, and a term whose price was tripled contributed at most 0.186 per second of a budget whose positives run near sixty. `pen_feet_distance` was causing the fault that `pen_feet_heading` was being asked to price, and the correct treatment of a fault is at its cause. The heading term is retained at its reference weight as an outcome level observer of the quantity, not promoted to an instrument it has never functioned as.

`pen_hip_roll_deviation` is deliberately not raised, notwithstanding that the roll axis is where the excursion went under the earlier reweightings. Raising it treats the axis the fault moved to rather than the outcome it produces, and the general result holds that a fault so treated migrates again to whichever channel remains unpriced [27]. The separation term is the outcome level instrument for the same fault and it now measurably works.

The published practice supports treating this as a placement geometry rather than as a posture price. The nearest retrieved work shapes the foothold's yaw angle directly and prices the foothold's lateral offset through a Gaussian rather than a hinge, within a two stage framework whose first stage learns to track a template planner's footholds [36], and the alternative that dispenses with a width parameter altogether rewards the zero moment point's position within the support polygon, which is the quantity a stance width parameter is a proxy for [30]. Neither is adopted here, the first requiring a foothold planner this configuration does not carry and the second requiring privileged contact information in the critic, but both indicate that the lateral demand belongs on the outcome and not on a joint.

##### The tracking weight is held at the reference, and the reason is measured

One arm of the sequence tested the opposite policy and is recorded here because its failure is the strongest single constraint on the design of any successor. Arm R, the run `2026-09-03_09-49-22`, carried the identified gains at an action scale of 0.4 and rebalanced four weights at once, `rew_lin_vel_xy` from 50 to 25, `pen_feet_distance` from minus 100 to minus 30 with the relaxed parameters, `pen_feet_heading` from minus 4 to minus 2 and `pen_hip_yaw_deviation` from minus 0.1 to minus 0.3. Its `agent.yaml` is byte identical to that of the reference run and of arm 0, so nothing on the algorithm side distinguishes it. It never walked. Its mean episode length stood between 25 and 30 steps at every checkpoint from iteration 250 to 13444, its `base_contact` termination rate was 1.0000 throughout, and the twelve frames spanning its training video at three hundred and twenty thousand steps show every robot prone on the ground, recorded at `../context/artefacts/kscale_flat-2026-09-03_09-49-22/train_320000_episode_span.png`.

The failure is an entropy collapse, and the trajectory separates it cleanly from arm 0 which shared its gains. Arm R's policy entropy fell from 17.06 to 4.86 by iteration 2000, to 0.145 by 5000 and through zero to minus 1.38 by 13000, its action noise standard deviation settling at 0.326, whereas arm 0's entropy never fell below 7.72 and its noise standard deviation never below 0.61, and arm 0 escaped. Once the noise has collapsed the walk cannot be rediscovered, and the remaining eight thousand iterations changed the episode length by less than two steps.

The budget account gives the cause. Decomposing the logged rewards into weighted rates per second at iteration 2000, when all three runs were still fallen, the reachable positive rate was 59.65 in the reference run, 26.43 in arm 0 and 13.81 in arm R, against penalty floors of 25.63, 52.19 and 53.66 respectively. Arm R and arm 0 stood on the same floor and arm R could reach barely half the positive return, because `rew_lin_vel_xy` is the only large positive in this objective and its weight had been halved. With the reachable positive cut below the floor, the locally optimal behaviour is to minimise action variance, and the measurements show the policy learned exactly that, its raw `pen_action_smoothness` falling from 46.2 in arm 0 to 15.9, its `pen_action_rate` from 48.6 to 28.0 and its `pen_joint_torque_rate` from 8729 to 5339. Two aggravating factors are recorded with it. `rew_gait` carries a weight of plus 40 but its raw value is NEGATIVE while the robot is fallen, so it contributed minus 22.78 per second and was the largest single penalty in the fallen regime, and `keep_balance` at 0.05 contributes 0.05 per second, so no survival bonus of any consequence bridges the interval in which the tracking reward is out of reach. Cutting the tracking weight on this objective is therefore not a free simplification, it is the removal of the only ladder out of the initial basin, and the practice of staging such terms in time rather than scaling them down is what the curriculum literature recommends in its place [29] [35].

##### The action scale is 0.25, and it removes the delay the identified gains impose

Arm 0 and arm 0S differ in exactly one line of their dumped configurations, the action scale, 0.4 against 0.25, every one of the twenty nine reward weights, both parameterisations, the gains and the agent configuration being identical. The pair is therefore a clean single variable test, and it answers two questions at once.

The first is the delay. The identified servos are stance servos and not control servos, and the cost of that is paid at the beginning of training. The reference run on the calculated gains reached a mean episode length of 235 steps by iteration 1000 and 1288 by 2000. Arm 0, on the identified gains at the same action scale, stood at 31 steps at iteration 1000, 34 at 2000, 40 at 4000 and 48 at 5000, and did not escape until somewhere between iterations 5000 and 8000, reaching 949 steps at 8000 and 1432 at 10000. That is a delay of roughly a factor of six in the emergence of the gait, and it is the expected consequence of a servo set chosen to hold a posture rather than to move through one. Arm 0S removes it entirely. At an action scale of 0.25 the same gains and the same rewards reached 317 steps by iteration 1000, 685 by 2000 and 1350 by 3000, which reproduces the reference schedule within the noise of a single seed. The delay is therefore not a property of the identified gains as such, it is a property of the identified gains driven at an action scale sized for the calculated ones.

The second is the gait, and arm 0S is the best walk the sequence has produced. Replayed at seed 42 it steps at 2.48 Hz with a double support fraction of 0.128 against a commanded 0.24, a median airborne sole clearance of 29.0 mm, and not one of its thirty two environments terminated, against 0.125 in arm 0 and 0.000 in the reference run. Its mean stance toe-in is 4.56 degrees against 7.81 in arm 0 and 13.07 in the reference. Its base angular velocity root mean square is 5.63 rad per second against 6.52 in arm 0, so the shaking the identified gains introduce is reduced by fourteen per cent and not removed. One cycle of its gait at three hundred and twenty thousand steps, the step at which every run in the sequence is sampled, is at `../context/artefacts/kscale_flat-2026-09-04_10-15-18/train_320000_one_cycle.png`, and one at seven hundred and ten thousand steps at `../context/artefacts/kscale_flat-2026-09-04_10-15-18/train_710000_one_cycle.png`, both showing an upright torso and a clean swing.

Two defects survive into it and both are recorded rather than repaired by this baseline. The first is yaw. Arm 0S carries a yaw velocity tracking error of 5.43 rad per second against 1.49 in the reference run and a raw `rew_ang_vel_z` of 0.388 against 1.753, while its linear velocity tracking is unimpaired at a raw 22.37 against 20.89. A hip yaw at 500 Nm per radian holds the axis but cannot steer with it, and since hip yaw effort saturation is only 0.081 the failure is not an actuator limit. The second is that the reduction in action scale did NOT touch the joint coordinate that produces the splay. The hip yaw convergence is 10.25 degrees in arm 0 and 9.38 in arm 0S, a change of nine per cent, against the 87 per cent the stance width relaxation achieved at unchanged gains, and the reason is visible in the reconstructed commands. Inverting the ideal proportional derivative law over the dumps gives a ninety fifth percentile commanded hip yaw deviation of 0.91 rad in arm 0 and 0.95 rad in arm 0S, the same absolute excursion at two different scales, the policy having simply raised its action magnitude to compensate. The action term carries `clip: null`, so the scale is a reparameterisation of an unbounded Gaussian output and not a bound upon it, which is the bounded action space bias the policy gradient literature identifies and proposes a bounded distribution to remove [37]. The action scale governs the resolution of the policy's control and the saturation of the actuators, and it does not govern the reachable joint set at all. That is why Baseline 4 takes the stance width parameter and not the action scale as its instrument on the splay, and why the bound, if one is wanted, must be a clip or a termination and belongs to arms Y and H.

##### The calculated gains, and why the run carrying the swing derived distal set never stood

Arm CS, the run `2026-09-04_10-40-30`, is the falsification test of the calculated hip yaw stiffness and it failed without qualification. It carried the calculated set with the revised hip yaw of 120 and 2.5 at an action scale of 0.25, differing from the reference run in those two respects alone. Its mean episode length rose to 55.8 at initialisation, fell to 16.3 by iteration 500 and stood at 23.4 at iteration 29000, its `base_contact` termination rate was 1.0000 at every checkpoint across the full thirty thousand iterations, and its mean reward never exceeded minus 2.4. Its failure is not the failure of arm R. Its entropy never collapsed, holding between 5.0 and 6.9 from iteration 10000 onward with a noise standard deviation near 0.53, so the policy explored for thirty thousand iterations and found nothing. An optimiser that keeps exploring and never improves is reporting that the environment is not survivable, not that the objective is badly shaped. The twelve frames spanning its video at three hundred and twenty thousand steps, at `../context/artefacts/kscale_flat-2026-09-04_10-40-30/train_320000_episode_span.png`, show every robot buckling into a crouch and going down, and twelve consecutive frames at seven hundred and ten thousand steps, at `../context/artefacts/kscale_flat-2026-09-04_10-40-30/train_710000_one_cycle.png`, show the same collapse after the full thirty thousand iterations, so nothing changed across the run.

Three readings follow, and each is the empirical form of a statement the gain derivation at `../context/KScale.md` section 11 makes on analytical grounds alone.

The first concerns the practice of sizing a stiffness from a logged torque, which is how the 120 Nm per radian hip yaw this run carried was arrived at, the ninety ninth percentile of the measured torque being divided by an allowed deflection. The practice presumes the measured torque is a disturbance the joint meets from outside. It is not. At a stiffness of 15 a torque of 23.72 Nm is the spring's own response to a commanded deflection of 1.58 rad, and inverting the control law over that run's dump gives a ninety fifth percentile commanded hip yaw deviation of 1.28 rad against an actual deviation of 0.77 rad, so the joint was tracking sixty per cent of a very large command rather than resisting an external moment. Raising the stiffness by a factor of eight against a command the policy is free to raise in turn does not deliver the intended deflection, it delivers the intended deflection times eight in torque, and 0.91 rad at 120 Nm per radian demands 110 Nm against a limit of 60. The practice is sound only where the recorded torque is exogenous, and at a position controlled joint driven by an unbounded policy output it is not, which is why `../context/KScale.md` section 11.1 excludes it and sizes the hip yaw by matching its stance bandwidth to the other proximal joints instead, at 100 Nm per radian and 11.0 Nm s per radian.

The second concerns the stance reflected inertia, which `../context/KScale.md` section 4.9 takes as the proximal assembly's inertia about the joint axis and which the run's own damping was not sized against. A reflection through the vertical lift alone, `M (dz_com/dq)^2`, would assign the hip yaw a stance inertia equal to its swing inertia of 0.02763 kg m squared, on the correct observation that a vertical axis does not raise the centre of mass to first order. The vertical reflection is not the only one. With a foot planted, rotating the hip yaw rotates the entire robot about a vertical axis through that foot, and the inertia the joint accelerates is the whole machine's yaw inertia about that axis. Computed properly over the proximal assembly, the figure is 0.61603 kg m squared, 22.3 times the swing value, and the two ankles stand at 4.86 and 5.38 kg m squared against swing values near 0.006, ratios of 639 and 937. At a stiffness of 120 and a damping of 2.5 the hip yaw damping ratio against 0.61603 is 0.16, an almost undamped mode driven by every footfall, and the derived 100 and 11.0 gives 0.70. The identified hip yaw at 500 and 18 gives 0.51, better than the run's but still short. The gap is invisible to any derivation that damps a stance joint against its swing inertia, which is what the ankle figures of 20 and 0.5 and 50 and 1.7 that this run carried also are, and `../context/KScale.md` section 10.6 gives the stiffness floor those two violate, a gravitational stiffness of 84.90 Nm per radian below which the servo cannot hold the machine upright about a planted sole at any deflection.

The third concerns separability, and it is the one that explains the tumble. A gain set is not a list of independent joints when several of them serve one function. The reference run's lateral support came from the splay, worth 31.5 per cent of lateral base at 7.7 degrees, because the calculated ankle roll at 20 Nm per radian supplied none, standing within 0.02 rad of a stop on 73.9 per cent of samples with 57.8 per cent effort saturation. Arm CS raises the hip yaw eightfold, which makes the splay eight times more expensive in torque, while leaving the ankle roll at 20 and the hip roll at 150. It removes the robot's only source of lateral support and provides no replacement, and the reward record shows precisely that failure and no other. At iteration 2000 arm CS paid minus 26.21 per second on `pen_ang_vel_xy` and minus 21.79 on `pen_flat_orientation`, against minus 3.45 and minus 0.45 in the reference run, the two lateral stability terms an order of magnitude worse than in any other run in the sequence while its tracking, its regularisation and its contact terms sat within the ordinary range. The identified set raises the hip yaw to 500 and simultaneously raises the ankle roll to 170 and the hip roll to 250, so the lateral duty transfers from the foot yaw to the ankle and the hip rather than vanishing, and the splay accordingly halves without a fall. The lesson generalises beyond this joint. Where a mechanical property is doing the work of a controller, a gain change that removes it must be paired with the change that replaces it, and the practice of preserving the mechanical substrate faithfully through training rather than treating it as a nuisance parameter is what the recent curriculum work on parallel actuated humanoids argues for on independent grounds [38].

A hip yaw of 120 and 2.5 on an ankle roll of 20 is therefore not a configuration this robot can walk on, and the ablation sequence carries no arm proposing one. The derived calculated set of `../context/KScale.md` section 12 raises the hip yaw, the hip roll, the knee and both ankles together and has never been run in full, which is the one gain experiment the evidence now asks for and which arm Z below is the cheapest partial form of.

##### The ablation sequence

The sequence below supersedes every earlier one in this document. Arms L and E are recorded as run and answered and are folded into the reference weights. Arm G is reopened, section 7.14 having withdrawn the confounding that its earlier verdict rested on, and it returns as the pair of arms 0G and 4G. Arm M, testing `pen_hip_roll_deviation` raised in place of the stance width change, is withdrawn, the stance width change having been run and having worked. Arm P, raising `min_feet_distance` to 0.18 m, is withdrawn, the fifth percentile stance width having risen to 143.1 mm without it.

| Arm | Change | Question it answers | Status |
|---|---|---|---|
| 0 | the 2026-08-28 weight set unchanged, no swing gate, identified gains, action scale 0.4 | How much of the splay, the stop residency and the yaw failure is mechanical rather than a matter of reward | RUN as `2026-09-03_09-27-59`, answered below |
| 0S | arm 0 with the action scale at 0.25, one line different | Does the action scale account for the delayed emergence, and does it act on the splay | RUN as `2026-09-04_10-15-18`, answered below |
| CS | the calculated gains with the revised hip yaw at the reduced action scale | Does the disturbance derived hip yaw stiffness hold on the machine that produced its measurement | RUN as `2026-09-04_10-40-30`, falsified, answered above |
| R | the reference weights with the tracking weight halved and the style prices rebalanced, identified gains, action scale 0.4 | Can the style block be strengthened and the tracking weight relaxed together | RUN as `2026-09-03_09-49-22`, falsified, answered above |
| Baseline 4 | arm 0S with `min_feet_distance` at 0.14 and `lateral_only` True | Does relaxing the stance width demand remove the splay once the gait is already sound | to be run first |
| 0G | arm 0S plus `rew_keep_hip_yaw_zero_in_air` at 1.0 | Does the swing gate produce the stiff legged gait at the reference reward scale | pending the pairing check |
| 4G | Baseline 4 plus `rew_keep_hip_yaw_zero_in_air` at 1.0 | Does the swing gate produce the same defect once the stance width demand no longer competes with it | pending the pairing check |
| Y | Baseline 4 with the action term clipped, the hip yaw channel bounded so that the commanded deviation cannot exceed 0.2 rad | Does a bound on the action deliver what the scale demonstrably does not [37] | new, and the cheapest test of the yaw failure |
| H | add `hip_yaw_out_of_bounds` at plus and minus 0.35 rad | Does a constraint remove the exchange rate a price only re-prices [20] | never run, the largest gap in the evidence |
| Z | Baseline 4 on the derived calculated gain set of `../context/KScale.md` section 12, hip pitch 200 and 20, hip roll 250 and 28, hip yaw 100 and 11, knee 300 and 10, both ankles 120 and 1.8 | Does a set damped against the stance inertia recover the yaw tracking the identified set loses, without the ringing it carries | new, and the only arm that varies a gain |
| Q | `keep_balance` and the eight regularisation terms scaled by one third against Baseline 4 | Is the residual shuffle bought by the survival and regularisation blocks rather than by the style block | |
| N | `pen_feet_heading` replaced by a relaxed logarithmic barrier over the same quantity | Does an instrument whose price rises without bound near the limit outperform both the flat price and the termination [28] | |
| K | `modify_reward_weight` engaging `pen_feet_heading` at 8000 iterations from an initial zero | Can gait and style be learned in sequence where they are surrendered when learned at once [29] [35] | |
| I | Baseline 4 with H | Are the constraint and the outcome level term complementary or redundant | |
| J | H with the bound tightened to 0.25 rad | Where does the bound begin to shape the gait rather than forbid the pathology | |
| B | `differential_scale=0.0` | How much of the correction is the differential mode alone responsible for | |
| C | `common_mode="sum"` | Does the mode mixing of the summed form measurably degrade the correction | |
| D | `tolerance=0.0` | Is the dead band earning its place, this arm being exact Booster Gym parity | |

The four arms already run determine the order of the rest. Arm 0S is the reference against which Baseline 4 is read, since the two differ in the stance width parameterisation alone, and arm 0 is retained as the record of what the action scale costs. Arm Y is placed immediately after Baseline 4 because the yaw failure is the largest surviving defect of the best run in the sequence and the action reconstruction has identified its cause precisely, an unbounded command at an over stiff joint. Arm Z is its mechanical alternative and the two should be read together, a bound that succeeds where a gain change also succeeds being a refinement while a bound that succeeds where a gain change fails being a finding. Arm Z carries an independent motive beyond the yaw, the derived set of `../context/KScale.md` section 12 never having been run in full while every run to date has carried either the swing derived distal gains that fall below the gravitational stiffness floor or the identified set that section 8 of the same document shows is underdamped at every joint in stance, at ratios between 0.149 and 0.513 against a target of 0.70. Arms 0G and 4G form a two by two with arm 0S and Baseline 4, the stance width parameterisation on one factor and the swing gate on the other, and the gate's effect is the difference within each pair. Arm 0G deserves particular note, being the run `2026-08-31_04-57-06` performed properly, since that run added the gate and simultaneously restored `keep_balance` from 0.05 to 0.5, so the stiffness it produced has two candidate causes and section 7.14 can separate neither. Arm H remains the untested claim of the second pass. Arm Q follows, being the cheapest test of the budget account given above, the two blocks it scales being the ones the group table shows were promoted without anybody intending it. Arms N and K refine the instrument and should follow H rather than precede it, since a barrier that succeeds where a termination also succeeds is a refinement while a barrier that succeeds where a termination fails is a finding. Arms B through D refine the heading term's parameters and are deferred to the end, the adequacy of the instrument being what the earlier group tests. The separation of a dense locomotion objective from a sparse style objective by giving each its own critic is the alternative to all of this and remains unattempted here [28] [34], and it should be reconsidered if arms Q, N and K all fail.

##### Preconditions and falsification conditions

Two preconditions must be discharged before the arms that depend on them are trusted. The joint to foot pairing of `rew_keep_hip_yaw_zero_in_air` remains unverified after three training runs, the reward pairing joint i with foot i positionally while both selections resolve through `SceneEntityCfg` with `preserve_order` false at `IsaacLab/source/isaaclab/isaaclab/managers/scene_entity_cfg.py:102`, so a transposition would charge the right hip yaw against the left foot's flight window and no training curve would reveal it. The check is two lines and must precede arms 0G and 4G.

```python
    import re
    print([n for n in robot.joint_names if re.fullmatch(r"(right|left)_hip_yaw_03", n)])
    print([n for n in scene["contact_forces"].body_names if re.fullmatch(r"foot_6061.*", n)])
```

The second is aliasing at the identified distal gains. Section 3.6 gives the hip yaw an effective inertia of 0.0133 kg m squared and the ankle roll 0.00074, against armatures of 0.01 and 0.005 which are of the same order and therefore not negligible. Taking the sum of each pair, a stiffness of 500 places the hip yaw near 146 rad per second and a stiffness of 170 places the ankle roll near 172, against a Nyquist bound of 157.08 rad per second imposed by the 50 Hz control loop. The ankle roll therefore sits above that bound and the hip yaw approaches it. Arm 0S is now the evidence on this question and it is reassuring rather than conclusive, that run having trained to a stable walk over thirty thousand iterations with a base angular velocity root mean square of 5.63 rad per second, elevated against the reference run's 1.60 but not divergent, and with 53.4 per cent of its base roll rate power above 5 Hz against the reference run's 51.1. Aliasing, if present, is not preventing the policy from training, and the question is now one of transfer rather than of stability.

Four falsification conditions govern the sequence. For Baseline 4, if the mean stance toe-in fails to fall below five degrees, the stance width parameterisation is not the operative cause on the identified gains as it demonstrably was on the calculated ones, and the remaining candidate is the sole slenderness itself, which no reward can reach. For Baseline 4 equally, if the toe-in falls but the step frequency rises above four hertz or the double support fraction above thirty per cent, the relaxation has reproduced the tremble that Baseline 3 produced on the calculated gains, in which case the gait block is not in fact holding the walk up and arm Q's budget test must precede any further style change. For arm H, if the episode length collapses or the termination rate rises materially above the twenty six per cent arm 0S recorded, the bound is shaping the gait rather than forbidding the pathology and must be loosened before its effect on the feet is read. For arms 0G and 4G, the gate is convicted only if the air time and clearance fall in BOTH pairs, since a defect appearing at one reward scale and not the other is a property of the budget rather than of the term.

One falsification condition set for arm 0 is discharged and its verdict recorded, since the interpretation of section 5 turns on it. The condition was that if the splay fell to the 2.3 degrees Baseline 3 achieved with no reward change at all, then the style block of this plan is answering a fault the actuator caused and the whole of section 5 should be reopened. It fell to 7.81 degrees at an action scale of 0.4 and to 4.56 at 0.25, against 13.07 in the reference run, so the mechanical substrate accounts for forty to sixty five per cent of the splay and the stance width demand for the remainder. Section 5 is therefore revised rather than reopened, and the revision is exactly the one made above, the style prices returned to the reference and the geometric demand made the instrument.

Two measurements should be added to every arm and are currently absent from all of them. The double support fraction is the single number that separates a walk from a shuffle on this robot, it moved from 11.9 per cent in the reference run to 51.1 in Baseline 3 to 12.8 in arm 0S while every other gait quantity moved with it [32], and it is computed at evaluation but never logged during training. The signed splay should be logged with it, both feet separately, since the unsigned terms in the reward set cannot distinguish a bias from jitter and that distinction is what separates a fault from noise.

##### What is deliberately not proposed

Six adjacent changes are declined, listed so that they are not mistaken for omissions, each with the ground that decides it.

A feet roll term, carried by Booster Gym at minus 0.1 [12] and by van Marum as feet orientation at 0.05 [6], is declined because `rew_keep_ankle_roll_zero_in_air` already regulates that axis and no run implicates it. Referencing the heading term to the commanded heading rather than to the base is declined for the reasons in section 5.1.1, and should be revisited only if a configuration ever drops `heading_command=True`.

The mirror loss is declined although it is the cheapest change in the repository, requiring only that `use_mirror_loss` be set true and `mirror_loss_coeff` be made non zero at `agents/limx_rsl_rl_ppo_cfg.py:317`, because the differential mode is invariant under the left right mirror while the common mode is anti invariant, so the loss would constrain the component measured near zero and leave the measured fault untouched. That the SD_BRS1 runner sets the flag true with a zero coefficient at `:272` is an inconsistency worth correcting on its own account and not a precedent here.

An adversarial motion prior is declined because it needs a reference dataset this configuration does not have, because the nearest published practice reports that such a prior over constrains motion outside the style its dataset covers [22], and because a discriminator reward remains a reward inside a summed objective and inherits the exchange rate that has already been shown to be exploited.

Locking the hip yaw and releasing it later is declined for now, no retrieved source reporting the staged procedure for a biped and the nearest mechanism restricting action magnitude uniformly rather than selectively [26], so nothing establishes whether a policy that learns to turn by stepping retains that solution once the joint reopens. The question should follow arms Y and H rather than pre-empt them, both of which bound the same joint by cheaper means.

Reducing the action scale further, below 0.25, is declined and the ground is now measured rather than argued. The scale does not bound the commanded joint position, the ninety fifth percentile hip yaw command standing at 0.91 rad at a scale of 0.4 and 0.95 rad at 0.25, so a further reduction buys resolution and saturation headroom at the price of driving the policy further into the tail of its own output distribution, which is where the Gaussian parameterisation is least well behaved [37]. Where a bound is wanted it should be imposed as a bound, which is arm Y.

The multi critic architecture is declined on jurisdictional rather than technical grounds, being the published instrument that most exactly matches what the reweighting performed by hand [28] [34], but altering the agent rather than the environment and reaching a module the SD_BRS1 and three TRON1 variants share at `agents/limx_rsl_rl_ppo_cfg.py`, together with the vendored `rsl_rl` on which every historical run depends. It is recorded as the correct long term destination, and arms Q, K and N are the environment level approximations the sequence tests in the meantime.

---

## 6. Conclusion of the Integration Phase

The KScale arrived as a faithful copy of the BRS configuration and the fidelity of that copy is the reason the defects in it are subtle. Nothing is obviously wrong on reading the file. The observations, the events, the curriculum and the network architecture are correct term for term, and the reward set is coherent, internally consistent and, taken on its own terms, reasonable. The defects are all of one kind, which is that a quantity carrying physical units was carried across from a robot three times the mass, or that a convention true of the source robot was assumed true of the target.

Three findings account for most of the work this plan prescribes and none of the three is visible in the configuration file that contains it.

The root frame is rotated ninety degrees from the Isaac Lab convention, so the velocity command that reads as forward is lateral. This is invisible in the configuration because the configuration is correct, it is the asset that differs, and it would have produced a policy that trained, converged and crab walked.

The foot link frame carries its vertical on y rather than z, so the sole depth is 0.043 m and not the 0.19 m a reader assuming the BRS convention obtains. This one was very nearly caught, the original author having recorded the suspicion that the number was a misread, and the four reward terms that depend on the sole were withheld rather than shipped against it. That was the right call, and this plan supplies the measurement that was missing rather than overturning a judgement.

The launcher has no branch for the robot and fails open rather than closed, so a request for the KScale trains the TRON1. This is the only defect of the three that would waste a training budget without leaving any evidence in the training run itself.

Against those, the reward work proves lighter than the task brief anticipated. Because every robot specific quantity in the reward package is already a configuration parameter and no reward function hard codes a name, a count or a geometry, the four missing terms are restored by wiring alone. No shared function is edited, no optional argument is introduced, no version two is created, and the BRS, the three TRON1 variants and the quadruped are provably untouched, which satisfies the backwards compatibility rule of `../../CLAUDE.md` not by careful handling but by construction.

The order of work is determined by the dependencies. The frame correction of section 4.2 comes first, because the symmetry mirror and two configuration blocks are wrong until it is done and would otherwise be corrected twice in opposite directions. The parameter corrections of section 4.3 come next, since they govern terms that are already live and are therefore already doing harm. The reward restoration of section 4.4 and the removal of section 4.5 follow, then the class hierarchy and registrations of section 4.6, then the symmetry module and agent configuration of section 4.7. The launcher clause of section 4.8 may be done at any point and should be done early, since without it none of the rest can be exercised.

Two matters are deliberately left open and should not be mistaken for oversights. The actuator gains require a retuning pass of their own, the present values leaving every joint overdamped and two ankle joints at or above the Nyquist bound of the control loop, and that derivation deserves the treatment `../context/BRS.md` gives the BRS rather than a scaling argument appended here, particularly since confirming the Robstride identification would replace any such argument with measured data. And the nominal standing pose remains the placeholder its own docstring admits it to be, so the standing height of 0.795 m that four parameters in section 4.3 derive from must be re-measured once that pose is settled.


## 7. Outcome and divergences from the integration plan

Every proposal of chapter 4 was implemented in the pass of 2026-08-24. The KScale reward set now matches the BRS term for term, twenty seven terms carrying identical functions and identical weights, and every parameter that differs between the two configurations is a derived robot specific quantity rather than an inherited one. No function in the shared `mdp` package was edited, no optional argument was added and no version two was created, so the BRS, the three TRON1 variants and the quadruped are untouched by construction rather than by careful handling, as section 4 predicted. The three shared files that were modified, the task registry, the agent configuration module and the symmetry package initialiser, were each audited line by line to confirm that no line outside KScale scope changed.

Six things diverge from what this document proposed and each is recorded here rather than silently absorbed.

The effective inertias of section 3.6 omitted the armature, and they should not have. Isaac Lab writes armature onto the PhysX joint, so it enters the mass matrix diagonal of the degree of freedom and is part of the inertia the proportional derivative loop actually sees, and at this robot's distal joints it exceeds the link inertia several times over, the ankle roll's 0.005 kg m squared against a link inertia of 0.00074. Including it moves the ankle roll from a natural frequency of 164.3 rad/s to 59.0 and the ankle pitch from 138.6 to 81.1, so the claim in section 3.6 that two ankle joints sit at or above the Nyquist bound of 157.08 rad/s is withdrawn. No joint does. The overdamping finding stands unchanged and is if anything the more clearly the dominant defect once the bandwidth alarm is removed. The corrected figures are in [../context/KScale.md](../context/KScale.md) section 6.

The actuator retuning that section 4.3 deliberately declined to attempt was carried out, at the explicit direction of the user. It follows the method of `../context/BRS.md` section 9 rather than a scaling argument, and its result is independently corroborated at the ankles and the hip yaw by the mass scaled BRS recommendation. The Robstride identification remains unconfirmed and would still supersede it.

The contradiction between `enabled_self_collisions` and `self_collision` that section 4.3 proposed resolving to one value was left as it stands. The reason emerged only on inspection of the BRS asset configuration, which carries the identical pairing, so resolving it on the KScale alone would have made the KScale differ from the exemplar it is meant to mimic. In effect the articulation root property governs at runtime and self collision is off for both robots, which is also why the standing bounding box overlaps do not produce persistent interpenetration forces. The forged contact exposure the section describes is real and is recorded in the context document rather than repaired here.

The defect in `scripts/rsl_rl/play.py` that section 4.9 listed as deliberately left standing was repaired instead, because section 4.8 makes KScale evaluation a one word command and a dump silently taken against the BRS sole table would have been the first thing that command produced. The foot pattern and the sole table are now resolved at runtime against the articulation's own body names, the BRS path resolving to exactly the values it carried before.

The flip set of section 3.5 was stated about the y and z plane, that being the correct mirror plane in the frame as exported. After the frame correction of section 4.2 the mirror plane is x and z, and recomputing the set about it reproduces the same four joints, which is the confirmation the section predicted, the flip set being a physical property of the robot rather than of the frame it is expressed in.

The launcher gained four clauses per chain rather than the two section 4.8 proposed, the rough terrain variants being registered and therefore worth reaching, and it gained the defensive final arm that section recommended in passing. That arm is the more valuable half of the change. Both dispatch chains assigned a default before testing any clause, so an unrecognised argument did not fail, it silently selected the TRON1 SoleFoot, and a user asking for a robot with no clause received a complete, plausible, converging run of a different robot with the wrong task identifier appearing only in that run's dumped parameters.

One matter this document left open remains open. The nominal standing pose is still the unverified knee bend guess its own docstring admits it to be, and the four reward parameters derived from the 0.7953 m standing height must be re-measured once it is settled by visual inspection in the simulator.

### 7.7 The gain derivation was wrong, and the first training run failed because of it

Recorded 2026-08-25, after the run this plan authorised was launched and failed.

Section 3.6 of this plan called for actuator gains derived from the effective inertia of each joint's distal subtree, and section 4.3 carried that call into the configuration. The derivation was performed as specified and the specification was wrong. The distal subtree inertia is the inertia a leg presents while swinging freely in the air, and it says nothing about whether a joint can hold the robot up, which is a question answered by the static ground reaction torque and never asked anywhere in this plan.

The consequence was a knee at 25 Nm/rad against a static double support load of 8.109 Nm and an ankle pitch at 7 Nm/rad against 3.150 Nm. Minimising the gravitational plus spring potential energy with the sole planted puts the settled stance at a base height of 0.4362 m against a nominal 0.77161 m, so the robot collapsed the instant it was set down. The training run at `IsaacLab/logs/rsl_rl/kscale_flat/2026-08-24_11-22-46` shows an episode length pinned at 40 steps from iteration 1000 to iteration 18000, with every termination attributed to `low_height` and none whatever to `base_contact`, which is the signature of legs folding under a body rather than a body toppling over its feet.

Two further defects of the same derivation are recorded with it. The ankle roll at 5 Nm/rad deflected 0.841 rad under a centre of pressure held at the edge of the sole, being 3.2 times that joint's entire travel of plus and minus 0.2618 rad, so it sat pinned at its stop whenever the robot leaned. And the spawn height of 0.85 m, which this plan left unexamined, stood 0.0784 m above a standing height that had moved twice since it was chosen, so the robot was dropped rather than set down.

The remedy raised the knee to 200 Nm/rad, the ankle pitch to 50 and the ankle roll to 20, keeping the corrected damping which was the sound part of the derivation and sizing it against the stance reflected inertia rather than the swing inertia. The spawn height was lowered to 0.79 m, being the standing height plus the 0.02 m settling margin the SD_BRS1 already uses. `scripts/analysis/kscale_stance_analysis.py` was written so that the omitted calculation is reproducible rather than remembered, and [../context/KScale.md](../context/KScale.md) was restructured to the analytical order of `context/BRS.md` so that both inertia regimes and the stance load are computed for every robot added to this workspace hereafter.

The lesson this plan should have carried, and which any successor plan for a new robot must, is that a stiffness has two independent requirements. It must place the joint's bandwidth somewhere sensible, and it must hold the robot's weight. The second is the binding one and it is the cheaper to check.

### 7.8 The feet heading reward, implemented 2026-08-26

Every proposal of chapter 5 was implemented in a single pass on 2026-08-26 and the chapter's specification was followed without amendment. `feet_yaw_alignment` at `environments/environments/tasks/locomotion/mdp/rewards.py:579` gained the six optional arguments, `pen_feet_heading` was added to `RewardsCfg` at `cfg/SF/kscale_base_env_cfg.py:874` at weight minus 2.0 with the mean form, a differential scale of 1.0, a tolerance of 0.10 rad and no contact gate, and `pen_hip_deviation` was replaced by `pen_hip_roll_deviation` and `pen_hip_yaw_deviation`, both at the minus 0.1 the single term carried.

The verification the chapter's step 2 called for was written as `scripts/analysis/kscale_feet_heading_selftest.py` and passes twenty five checks. Twelve are kinematic and read the URDF directly, establishing that the negative z axis of each foot link maps to the root frame positive x at the nominal pose to within 1.1e-05, that both hip yaw axes read minus 0.0998, 0.0000, plus 0.9950 in the root frame with a mutual dot product of 1.000000, that a positive command at either hip produces a toe heading gain between 0.99416 and 0.99497 of the same sign on both legs, and that the ankle roll at its limit moves the toe heading by 0.0001 degrees. Thirteen are behavioural and execute the shipped reward, establishing the orthogonality of the mean form against the summed form's failure of it, the four to one and three to one splay ratios, the dead band, the backwards compatible default path to ten decimal places, and the ninety degree offset of the Euler path at three foot yaw angles.

The inertness of the hip deviation split was confirmed by resolving both joint name patterns against the URDF rather than by argument. The single term and the pair resolve to the identical four joint set, `left_hip_roll_03`, `left_hip_yaw_03`, `right_hip_roll_03` and `right_hip_yaw_03`, and the two new patterns are disjoint, so no joint is counted twice or dropped. With `joint_deviation_l1` a plain summation at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:186` and both weights equal, the total reward is unchanged exactly.

The blast radius is as chapter 5 predicted. Every hunk of the diff to the shared `mdp` package falls inside `feet_yaw_alignment`, whose only caller in the tree is the KScale term added in the same pass, so the SD_BRS1, the three TRON1 variants and the quadruped are untouched by construction. The KScale symmetry module needed no change, the new term reading body poses rather than the observation or action vectors, and it was confirmed to reference no reward term at all.

Two divergences from the chapter are recorded.

The self test was written to run outside the Isaac container as well as inside it, which the chapter did not ask for. Neither `isaaclab` nor the environments package imports on a bare interpreter, so a test that merely imported the reward would have been unrunnable in exactly the situation in which a frame regression is most likely to be introduced, which is an editing session rather than a training session. The script therefore imports the real modules where they are available and otherwise extracts the function body from `rewards.py` and executes it against a shim supplying the three mathematics helpers it uses. Both paths exercise the same source text, so the checks are meaningful either way, and the shim is validated against the real implementation whenever the container is available.

The module docstring of `cfg/SF/kscale_base_env_cfg.py` was corrected in the same pass, which the chapter did not list. It described the SD_BRS1 hip yaw joints as disabled by a zero width limit, which section 4.9 of this plan had already established to be false, they being declared `type="fixed"`, and it directed the reader to a `pen_hip_deviation` that no longer exists under that name. Leaving either standing would have pointed the next reader at a term that is absent and a mechanism that is imagined.

The chapter's remaining steps are not yet discharged and are the whole of what is outstanding. No training run has been launched, so the four quantities of step 6 have no values, the 0.0000 turning tax and the minus 1.40 to minus 4.82 fault magnitudes remain predictions from statics and geometry rather than observations, and the ablation table of section 5.1.2 has no arm completed, not even its baseline. The falsification condition should be read first. If `rew_ang_vel_z` falls materially below the plus 2.7620 of the reference run, the term has corrected the feet by suppressing the turn and the weight is too high.



### 7.9 The feet heading reward, run and measured 2026-08-28

The term implemented under section 7.8 was trained in run `2026-08-26_08-08-13` and compared against `2026-08-25_10-44-21`, which differs from it in exactly the three terms section 5 introduced and whose angular curriculum stood at the identical setting throughout, so the pairing is controlled. The outcome is recorded in full under the reward budgeting subsection of section 5.1.2 and is summarised here as the plan's own account of what it predicted and what occurred.

Two predictions were confirmed. The decomposition behaved as designed, the differential mode carrying roughly 78 per cent of the measured penalty against a prediction derived from the hip yaw excursion alone, which vindicates the choice of the mean form and the orthogonality it buys. The joint space instrument, deliberately left at its previous total weight so that it would remain a measurement rather than a second intervention, duly recorded a 37 per cent fall in off axis excursion that no term reads directly, which is the attribution the design was built to permit.

One prediction failed, and it was the one the section recorded a falsification condition against. Yaw tracking fell 47 per cent and its error nearly doubled, which section 7.8 stated in advance would mean that the term had corrected the feet by suppressing the turn and that the weight was too high. The subsequent analysis establishes that the diagnosis was half right and half wrong. The weight is indeed too high to be paid without harm, but raising or lowering it does not reach the cause, because the splay is a purchase of lateral stability worth roughly a 1.874 fold margin over three quarters of the gait cycle on a sole whose aspect ratio is 2.481, against which a penalty costing 4.80 per cent of return is not competitive at any weight the tracking terms would survive. The instrument rather than its magnitude is what the evidence indicts, and section 5.1.2 accordingly proposes a constraint expressed as a termination in place of a price.

Two divergences from the plan's expectations are recorded. The first is that the plan reasoned about the swing gate as a protection for a turning robot and retired it on the measurement that the turn tax was zero, whereas the gate returns in the second pass for an entirely different reason, as a means of forcing the foot to land straight so that the existing slide and landing penalties may tax its rotation under load, which is an argument the first pass did not contain. The second is that the plan assumed throughout that the fault's sign was known, and it is not, since the penalty is quadratic and the joint deviation absolute so neither distinguishes toes turned inward from toes turned outward, while the recorded video is taken from a fixed oblique overview in which a foot occupies some ten to twenty five pixels and cannot settle it either. Establishing the sign is the first item of the revised sequence and should have been instrumented from the beginning.
### 7.10 The swing gate and the doubled heading weight, run and measured 2026-09-02

The run `2026-08-31_04-57-06` added `rew_keep_hip_yaw_zero_in_air` at weight 1.0 and simultaneously raised `pen_feet_heading` from minus 2.0 to minus 4.0, which is arm G of the revised sequence confounded with the arm the same revision had withdrawn. Its dumped `params/agent.yaml` is byte identical to that of the preceding run, so the outcome is attributable to those two weights alone.

The intervention succeeded on the quantity it named and destroyed the gait that carried it. The heading penalty fell from a raw rate of 0.61392 to 0.11071, a reduction of 82 per cent, and the hip yaw excursion from 0.79740 to 0.57152, a reduction of 28 per cent, which is the largest single improvement in foot heading this sequence has recorded. Against that, the foot clearance reward fell 59 per cent, the air time reward rose 93 per cent, the gait phase reward degraded by a factor of 2.13 and the foot slide penalty rose 83 per cent, which is the stiff legged gait the user observed in the video and which the frames confirm, the swing leg extending as a rigid pendulum with the foot plantarflexed and trailing. The yaw tracking error reached 3.48958, the worst of the four runs and 2.7 times the original baseline. The policy noise standard deviation rose monotonically from 0.62 at iteration 3000 to 1.23 at the end, so the run never converged.

Two things are recorded that the plan did not anticipate. The first is that the mechanism reached its target and then lost it, the hip yaw excursion touching 0.2252 at iteration 7000, being 0.113 rad per joint against the SD_BRS1 reference of 0.111, before climbing monotonically back to 0.5838 over the following twelve thousand iterations. A style objective that is achieved and then surrendered is a different fault from one that is never achieved, and the plan had been reasoning about the second. The second is that the stiffness is not attributable to the swing gate, which the succeeding run carries at identical weight and at a threefold greater relative weight without producing it. The gate is exonerated and the doubled heading weight is indicted, which is the verdict the revised sequence had already reached on the budgeting evidence when it withdrew arm F, and the run is therefore a confirmation of that withdrawal obtained at the cost of performing the experiment anyway.

### 7.11 The two group reweighting, run and measured 2026-09-02

The run `2026-09-01_06-53-24` was undertaken to remove the stiffness by reducing the tracking rewards so that a natural gait should outweigh the velocity command. It removed the stiffness, it is the best run of the four by a wide margin, and it was not the change its author intended to make.

What the dumped weights record is a two group reweighting. Eleven terms were scaled by factors between 0.100 and 0.500 with a median near 0.30, comprising the two tracking terms, the whole of the gait shaping apparatus and the foot separation term, and nineteen terms were left untouched, comprising every regulariser, the three joints held at zero in the air, both hip deviation penalties and the orientation and body angular velocity terms. The operation was therefore a promotion of the posture, style and regularisation group by a factor near 3.3 relative to everything it competes with, and not a reduction of the tracking reward as such. It cannot have acted through the overall magnitude of the reward, since the surrogate objective is scale free under the advantage normalisation at `rsl_rl/rsl_rl/algorithms/ppo.py:191`, so it acted through the ratios between the groups. This is the environment level approximation of the multi critic architecture of Kim, Lee and Park [28], arrived at by hand and at a ratio chosen without a principle.

The measured outcome is set out in full in the measurement tables of section 5.1.2 and is summarised here. Against the original baseline of 2026-08-25 the run improves the yaw tracking error by 48 per cent, the flat orientation penalty by 49 per cent, the body angular velocity penalty by 54 per cent and the action smoothness penalty by 38 per cent, at a cost of 2.2 per cent of the linear tracking reward and 12.7 per cent of the gait phase reward, with the episode length and the termination rate materially unchanged and the policy noise standard deviation the lowest of the four. Against the run of 2026-08-26 it reduces the heading penalty by 76 per cent and the hip yaw excursion by 52 per cent. The stiffness is absent and the foot clearance reward recovers to 0.48215 from 0.22551.

The divergence the plan must absorb is the substitution. The hip yaw excursion fell 33 per cent and the hip roll excursion rose 72 per cent while their sum stood still, 0.80731 against 0.79017, and the video shows what that trade purchased, the two feet drawn toward the midline until they pass within a few centimetres of one another with the swing foot crossing beside the stance foot rather than outside it. The foot separation raw rate rose by a factor of twenty five, and although its weight was cut tenfold in the same change the policy's actual outlay on the term rose from 0.515 to 1.287 per second, so the narrow stance was bought at an increased cost rather than acquired by inattention. The expectation that the undesired contact penalty had also been abandoned is not supported, its raw rate of 0.00056 being the lowest of the four runs and 23 per cent below the original baseline, and the visual impression of contact arises from shanks passing close without touching.

The plan's own account of this is that it was measuring the splay and not the excursion, and that the two are not the same thing. Every instrument section 5 introduced names the vertical axis, and the policy has moved its off axis expenditure to the fore and aft axis, where the only charge is a joint deviation penalty at minus 0.1 and a foot separation term whose threshold of 0.24 m against a nominal separation of 0.252 m could never be satisfied and therefore never informed anything. Both defects were repaired in Baseline 3, whose outcome section 7.12 records.


### 7.12 Baseline 3, run and measured 2026-09-03

The run `2026-09-02_06-59-31` implemented Baseline 3 exactly as section 5.1.2 specified it, its dumped parameters carrying `min_feet_distance` of 0.14 with `lateral_only` true at weight minus 30 and every other weight as tabulated. It answered its question in the affirmative and revealed a second fault that no previous run had isolated.

The substitution is arrested. The off axis excursion falls in sum rather than moving between axes, 0.549 to 0.495 at the matched window, where the two preceding passes had conserved it to within two per cent. The foot separation term is satisfied for the first time in the sequence, its raw rate falling 59 per cent and the fifth percentile of the measured stance width rising from 108.3 to 143.1 mm, which is the honest measure of a floor since the mean fell in the same change. And the sign of the splay is established at last by rotating each foot link's negative z axis into the world rather than by any term in the reward. The left foot stands at minus 7.70 degrees and the right at plus 7.46 in the reference run, both toes converging on the midline, so the gait is pigeon toed and not duck footed. Baseline 3 removes 70 per cent of that bias while removing only 19 per cent of the per foot magnitude, so what survives is symmetric jitter rather than a posture.

The cost is that the robot stopped walking. Air time fell 78 per cent, clearance 62 per cent and step length by half, the fraction of steps with a sole more than 20 mm off the ground fell from 10.6 to 4.7 per cent, and the double support fraction rose from 41.1 to 51.1 against a commanded 24. The human evidence is that these move as one coordinated pattern rather than as separate faults [32], which is why the double support fraction is the right single number to carry forward and why no separate clearance target need be posited to explain the clearance loss.

The cause is a change of ratio the two group description of the preceding run concealed, and it is visible only when the budget is aggregated by group from the evaluation rather than read off the weight list. The gait block was not demoted against the task block, their ratio moving only from 1.043 to 0.969. It was demoted against everything else, the style block rising 2.71 fold against it, the regularisation block 1.94 fold and the survival bonus 28.7 fold. Each of those three is cheaper to satisfy when the robot does not leave the ground and the gait block is the only one that pays for leaving it. That velocity tracking improved under a threefold cut to its own weight, from a raw 0.809 to 0.843, is the load bearing negative result, since it establishes that the task term never distinguished a walk from a shuffle and that only the cut block did.

Two mechanical defects are recorded from the same evaluation and neither is a reward fault. The ankle roll stood within 0.02 rad of a mechanical stop for 80.5 per cent of steps in the reference run on a joint whose whole travel is plus and minus 0.2618 rad, which is the condition section 7.7 found at a stiffness of 5 and believed it had cured by raising the stiffness to 20. And the hip yaw, at a stiffness of 15 against 150 to 200 elsewhere, showed the highest mean joint speed in the robot at 7.3 rad per second at an amplitude of 0.15 rad, with the base yaw rate reaching 4.5 to 6.5 rad per second against commands near 0.6 at a command correlation of zero. The published position is that a low proportional gain leaves a joint behaving as a torque source with large tracking errors [33], and four runs were trained in that condition.

The splay itself has a geometric cause specific to this robot, which explains why three successive instruments relocated it rather than removing it. The sole is 0.2100 m long and 0.0846 m wide, a slenderness of 2.48, so a yaw of 7.7 degrees converts length into lateral extent and raises the lateral base of support of each foot by 31.5 per cent at no cost in any term that reads a joint coordinate. It is a standing postural bias and not a reflex, its within run correlation against instantaneous lateral velocity, roll rate and lateral projected gravity lying between minus 0.26 and plus 0.17, which is why a price removes it and did.

### 7.13 The hip yaw range restriction, run 2026-09-02 and its evaluation set aside

The run `2026-09-02_10-51-29` repeated Baseline 3 with the hip yaw travel reduced to plus and minus 0.22 rad, and its training record is the best of the sequence while its evaluation dump must be set aside. The restriction was made in the URDF and reverted before the evaluation was run, so the policy trained against stops at 0.22 rad and was replayed against stops at 1.5708. All three evaluation dumps carry identical `joint_position_limits` and the observed hip yaw in this policy's own replay reaches 1.586 rad, which is seven times the bound it was trained under. Its mean episode length reaches the full 2000 steps of 2000 in training and 89.3 per cent of its evaluation episodes terminate early, and the gap between those two figures is the measure of the mismatch rather than of the policy.

The training record is nonetheless the most informative datum in the sequence on the hip yaw. Against Baseline 3 at the matched window it halves the hip yaw excursion, 0.292 to 0.141, reduces the heading penalty by 73 per cent, and raises the yaw tracking reward from a raw 0.037 to 0.293 while the yaw rate error falls from 7.52 to 1.78 rad per second. That is a restoration of yaw tracking by a factor near eight, obtained by removing a joint's freedom rather than by pricing its use, and it identifies the free hip yaw as the mechanism by which the base yaw was escaping control. The new identified gains address that mechanism directly and at its cause, raising the hip yaw stiffness from 15 to 500 Nm per radian and its damping from 0.9 to 18, which is why the range restriction was reverted and why it should not be reinstated before the gain change has been measured on its own.

Two procedural lessons are recorded with it. A change made in the asset rather than in the environment configuration leaves no provenance, the two `params/env.yaml` files of the 09-02 series being byte identical, so nothing in the dumped record distinguishes these two runs at all. And an evaluation must be run against the asset the policy trained on, which requires that an asset change be either committed or recorded alongside the checkpoint, neither of which happened here.

### 7.14 A correction to section 7.10, and the withdrawal of the charge against the doubled heading weight

Recorded 2026-09-03, on reading the dumped weights of every KScale run in sequence rather than the two the earlier pass compared.

Section 7.10 states that the run `2026-08-31_04-57-06` raised `pen_feet_heading` from minus 2.0 to minus 4.0 and that it is therefore arm G confounded with the withdrawn arm F. That is wrong. The doubling happened one run earlier. `2026-08-26_08-08-13` carries minus 2.0, `2026-08-28_04-50-51` carries minus 4.0, and `2026-08-31_04-57-06` carries minus 4.0 unchanged. The three runs differ as follows and in no other term.

| Term | 2026-08-26 | 2026-08-28 | 2026-08-31 |
|---|---|---|---|
| `keep_balance` | 0.5 | 0.05 | 0.5 |
| `pen_feet_heading` | -2.0 | -4.0 | -4.0 |
| `rew_keep_hip_yaw_zero_in_air` | absent | absent | 1.0 |

Three consequences follow and each reverses a finding of the earlier pass.

The doubled heading weight is exonerated. It was present at minus 4.0 throughout the run that produced the best gait of the whole sequence and produced no stiffness there, so it cannot be what produced the stiffness one run later. The withdrawal of arm F under reward budgeting was argued on the budget and reached the right conclusion for reasons that did not include this, and the conclusion now rests on a direct observation instead.

The swing gate is correspondingly re-indicted, and the user's original reading, that the stiffness appeared when `rew_keep_hip_yaw_zero_in_air` was introduced, is the better supported one. Section 7.10 exonerated the gate on the ground that the succeeding run carries it at a threefold greater relative weight without producing stiffness, and that argument still stands as far as it goes, but it is a comparison across a reweighting of the entire reward set while `2026-08-28` against `2026-08-31` is a comparison across two terms. Neither is decisive. The gate is one of two candidates and its status is open.

The second candidate is `keep_balance`, which moved from 0.05 back to 0.5 in the same run. That is a tenfold promotion of a flat survival bonus, and section 7.12 establishes by group aggregation that this is one of the three promotions that make standing cheaper than swinging. The stiff legged gait of 2026-08-31 and the shuffling gait of 2026-09-02 may therefore be the same fault at two different strengths, arrived at by two different routes, which would be a more economical account than treating them as separate defects. Nothing measured so far distinguishes the two candidates.

The runs already in flight settle it. `2026-09-03_05-49-21` and `2026-09-03_06-01-23` carry identical weights and differ only in the presence of `rew_keep_hip_yaw_zero_in_air`, both on the new identified gains, which is the single variable comparison the gate has never had. Whatever they return, the gate's contribution is read directly and neither the plan nor this section needs to argue it.

A procedural note is recorded with the correction. The error arose from reading the plan's own narrative of what each run changed rather than the dumped weights of each run, and the narrative had been written from the intention rather than from the artefact. Every weight in section 5.1.2 is now taken from a `params/env.yaml`, and any future comparison should be built the same way, since the dump is the only record of what actually trained.

---

### 7.15 The gain and action scale sequence, run and measured 2026-09-04

Four runs completed between 2026-09-03 and 2026-09-04 and together they resolve the attribution that every earlier section could only defer, the mechanical substrate having changed at the same moment as the reward set. Their dumped configurations differ from one another in a small and enumerable number of lines, which is what makes them readable as a design rather than as four separate experiments.

| Run | Arm | Gains | Action scale | Reward set | Outcome |
|---|---|---|---|---|---|
| `2026-09-03_09-27-59` | 0 | identified | 0.40 | 2026-08-28, all twenty nine terms | walks, gait emerging between iterations 5000 and 8000 |
| `2026-09-03_09-49-22` | R | identified | 0.40 | four weights rebalanced | never walked, entropy collapsed |
| `2026-09-04_10-15-18` | 0S | identified | 0.25 | 2026-08-28, all twenty nine terms | walks, gait emerging by iteration 1000, best run of the sequence |
| `2026-09-04_10-40-30` | CS | calculated, hip yaw 120 and 2.5 | 0.25 | 2026-08-28, all twenty nine terms | never stood, falling within 0.23 s at every iteration to 30000 |

Arm 0S and arm 0 differ in exactly one line of `params/env.yaml`, the action scale, and arm CS differs from the reference run `2026-08-28_04-50-51` in three, the hip yaw stiffness, the hip yaw damping and the action scale, the third being an explicit restatement of the default `lateral_only: false`. The `agent.yaml` of all four is byte identical to the reference run's.

Five findings are recorded, and two of them correct entries elsewhere in this document.

The identified gains delay the emergence of the gait by roughly a factor of six at the action scale the calculated gains were trained at, and the delay is removed entirely by the reduction of that scale to 0.25. The reference run reached a mean episode length of 235 steps by iteration 1000, arm 0 reached 31 and did not exceed 100 until after iteration 5000, and arm 0S reached 317. This is the expected cost of a servo set chosen for stance rather than for control, and the remedy is a scale sized for the servos rather than a change to the servos. One cycle of arm 0's gait at three hundred and twenty thousand steps is at `../context/artefacts/kscale_flat-2026-09-03_09-27-59/train_320000_one_cycle.png` and twelve frames spanning the same recording at `../context/artefacts/kscale_flat-2026-09-03_09-27-59/train_320000_episode_span.png`, the first showing a recognisable walking cycle with an upright torso and the second that the walk holds across the whole clip.

The identified gains halve the pigeon toed splay with no reward change whatever. Replayed at seed 42 the mean stance toe-in is 13.07 degrees in the reference run, 7.81 in arm 0 and 4.56 in arm 0S, and the hip yaw convergence 18.66, 10.25 and 9.38 degrees. The falsification condition set for arm 0 in section 5 is therefore discharged with a divided verdict, the mechanical substrate accounting for between forty and sixty five per cent of the splay and the stance width demand for the remainder.

The reduction in action scale does not act on the splay and cannot, because the action scale is not a bound. Inverting the ideal proportional derivative law over the two dumps gives a ninety fifth percentile commanded hip yaw deviation of 0.91 rad at a scale of 0.4 and 0.95 rad at 0.25, the policy having raised its action magnitude to hold the same absolute excursion. The action term carries `clip: null`, so the scale reparameterises an unbounded Gaussian output rather than constraining it. Any future statement that lowering the scale restricts what the policy may command is false on this configuration and must be withdrawn wherever it appears.

The identified gains trade yaw control for stance robustness, and the trade is measured. Arm 0S carries a yaw velocity tracking error of 5.43 rad per second against the reference run's 1.49 and a raw `rew_ang_vel_z` of 0.388 against 1.753, while its linear velocity tracking is unimpaired and its termination rate is lower, 0.262 against 0.352. Hip yaw effort saturation stands at 0.081, so the failure is not an actuator limit but a joint too stiff to steer with.

This CORRECTS section 7.13's expectation that the identified ankle roll stiffness of 170 would lift that joint off its mechanical stop. It did not. Ankle roll stop residency is 0.739 and 0.734 at a stiffness of 20 in the reference run, 0.669 and 0.667 at 170 in arm 0, and 0.765 and 0.774 at 170 in arm 0S, so an eight and a half fold change in the gain moves the residency by at most seven points and the reduced action scale moves it back. The joint is saturating rather than being held by its spring, which is the condition section 11.10 of `../context/KScale.md` describes, and the operative limit is the 17 Nm effort ceiling rather than the stiffness.

The failure of arm CS is a failure of the calculated hip yaw stiffness and its analysis is given in section 5 rather than here, together with the three corrections it forces upon the derivation in `../context/KScale.md`, the misattribution of a commanded deflection as an exogenous disturbance, the omission of the horizontal rotational term from the stance reflected inertia at the lateral and vertical axes, and the non separability of a gain set whose joints share one function.

---

## Bibliography
1. Su et al., Leveraging Symmetry in RL-based Legged Locomotion Control, IROS, 2024, arXiv:2403.17320.
2. Abdolhosseini et al., On Learning Symmetric Locomotion, Motion in Games, 2019, DOI 10.1145/3359566.3360070.
3. Ordonez Apraez et al., On discrete symmetries of robotics systems, a group theoretic and data driven analysis, RSS 2023 and IJRR 2024, arXiv:2302.10433.
4. Mittal et al., Symmetry Considerations for Learning Task Symmetric Robot Policies, 2024, arXiv:2403.04359.
5. Yu, Turk and Liu, Learning Symmetric and Low-Energy Locomotion, 2018, arXiv:1801.08093.
6. Van Marum et al., Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking, 2024, arXiv:2404.19173.
7. Siekmann et al., Learning Memory-Based Control for Human-Scale Bipedal Locomotion, 2020, arXiv:2011.01387. Cited for the periodic contact schedule construction the gait clock implements.
8. Walk These Ways, Tuning Robot Control for Generalization with Multiplicity of Behavior, 2022, arXiv:2212.03238. Authors not recorded by the retrieved source, the short form is given.
9. Humanoid-Gym, Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer, 2024, arXiv:2404.05695. Authors not recorded by the retrieved source, the short form is given.
10. Nilsson and Thorstensson, Ground reaction forces at different speeds of human walking and running, Acta Physiologica Scandinavica, 136(2), 1989. Cited for the ground reaction force range and the double support fraction of the gait cycle.
11. Perry, Gait Analysis, Normal and Pathological Function, 1992. Cited for stance width as a multiple of hip width.
12. Wang, Chen, Han, Wu and Zhao, Booster Gym, An End-to-End Reinforcement Learning Framework for Humanoid Robot Locomotion, 2025, arXiv:2506.15132. Cited for the feet yaw, feet roll and torque tiredness terms of its Table II, and for the common and differential decomposition its released implementation carries.
13. Bevel-geared mechanical foot, a bioinspired robotic foot compensating yaw moment of bipedal walking, Advanced Robotics, DOI 10.1080/01691864.2021.2017343. Cited for the yaw moment a swinging leg imposes upon the stance foot. Authors, venue year and page range are not recorded here, the publisher page having returned a 403 to retrieval, and the short form is given in accordance with the citation rule of `/ws/CLAUDE.md`.
14. Popovic, Hofmann and Herr, Angular momentum regulation during human walking, biomechanics and control, ICRA, 2004. Cited for the regulation of whole body angular momentum and the segment to segment cancellation of its horizontal component.
15. Cibulka, Winters, Kampwerth, McAfee, Payne, Roeckenhaus and Ross, Predicting foot progression angle during gait using two clinical measures in healthy adults, a preliminary study, International Journal of Sports Physical Therapy, 11(3), 2016, pages 400 to 408. Cited for the foot progression angle of 3.3 degrees plus or minus 5.6 degrees measured over sixty healthy adults.
16. Skalse, Howe, Krasheninnikov and Krueger, Defining and Characterizing Reward Hacking, NeurIPS, 2022, arXiv:2209.13085.
17. Pan, Bhatia and Steinhardt, The Effects of Reward Misspecification, Mapping and Mitigating Misaligned Models, ICLR, 2022, arXiv:2201.03544.
18. Reda, Tao and van de Panne, Learning to Locomote, Understanding How Environment Design Matters for Deep Reinforcement Learning, Motion, Interaction and Games, 2020, arXiv:2010.04304.
19. Kim, Oh, Lee, Choi, Ji, Jung, Youm and Hwangbo, Not Only Rewards But Also Constraints, Applications on Legged Robot Locomotion, IEEE Transactions on Robotics, 2024, arXiv:2308.12517.
20. Chane-Sane, Leziart, Flayols, Stasse, Souères and Mansard, CaT, Constraints as Terminations for Legged Locomotion Reinforcement Learning, 2024, arXiv:2403.18765. The arXiv record does not state a venue and none is asserted here.
21. Peng, Guo, Halper, Levine and Fidler, ASE, Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters, ACM Transactions on Graphics 41(4), Article 144, SIGGRAPH, 2022, arXiv:2205.01906.
22. Wu, Wang, Ye and Xing, Multi-Gait Learning for Humanoid Robots Using Reinforcement Learning with Selective Adversarial Motion Prior, 2026, arXiv:2604.19102. Venue beyond the arXiv listing not established by the retrieved source.
23. Singh, Benallegue, Morisawa, Cisneros and Kanehiro, Learning Bipedal Walking On Planned Footsteps For Humanoid Robots, 2022, arXiv:2207.12644. Venue beyond the arXiv submission not established by the retrieved source.
24. Toe-in and toe-out gait retraining interventions for individuals with knee osteoarthritis, a pilot randomised clinical trial, Clinical Biomechanics, 2024, PMID 39566359. Author list, volume and pages not established, the publisher page having refused the retrieval.
25. Ntagkas, Kiourt and Chatzilygeroudis, PGTT, Phase-Guided Terrain Traversal for Perceptive Legged Locomotion, 2026, arXiv:2510.18348. Venue recorded as IROS in the arXiv listing and not independently confirmed.
26. Liao, Li, Yang, Chang, Fan, Wang, Shi, Cao, Wu and Sartoretti, GPO, Growing Policy Optimization for Legged Robot Locomotion and Whole-Body Control, 2026, arXiv:2601.20668. Venue beyond the arXiv listing not established by the retrieved source.

27. Wang and Huang, Reward Hacking as Equilibrium under Finite Evaluation, 2026, arXiv:2603.28063. Cited for the formal result that an optimised agent under invests in quality dimensions its evaluation does not cover, and that patching one exploited dimension produces substitution along an unmonitored one as an equilibrium property. The domain is alignment theory rather than locomotion and the transfer is by mechanism. Venue beyond the arXiv listing not established by the retrieved source.
28. Kim, Lee and Park, A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards, IEEE International Conference on Robotics and Automation, 2025, arXiv:2409.15780. Cited for the relaxed logarithmic barrier style reward, for the multi critic separating barrier rewards from task and regularisation rewards with advantages normalised before the surrogate, and for the seven quantities to which the barrier form is applied.
29. Li, Wu, Liu, Guo and Xue, Experience-Learning Inspired Two-Step Reward Method for Efficient Legged Locomotion Learning Towards Natural and Robust Gaits, 2024, arXiv:2401.12389. Cited for the temporal separation of gait related rewards from terrain difficulty. The retrieved source reports naturalness qualitatively and supplies no quantitative measure. Venue beyond the arXiv listing not established by the retrieved source.
30. Xie, Bai, Shi, Yang, Ge, Zhang and Li, Humanoid Whole-Body Locomotion on Narrow Terrain via Dynamic Balance and Reinforcement Learning, 2025, arXiv:2502.17219. Cited for the zero moment point driven reward on narrow support. The retrieved source does not report the trade off between stance width and lateral disturbance rejection. Venue beyond the arXiv listing not established by the retrieved source.
31. Hollman, McDade and Petersen, Normative Spatiotemporal Gait Parameters in Older Adults, Gait and Posture, 34(1), 2011, pages 111 to 118, PMID 21531139. Cited for a step width between 7.0 and 9.9 cm against a step length between 54 and 69 cm over 294 adults aged seventy and above.
32. Williams and Martin, Gait modification when decreasing double support percentage, Journal of Biomechanics, 92, 2019, pages 76 to 83, DOI 10.1016/j.jbiomech.2019.05.028. Cited for the covariation of the double support fraction with hip flexion, knee flexion and swing foot height at a fixed walking speed.
33. Spoljaric, Yashuai and Lee, Variable Stiffness for Robust Locomotion through Reinforcement Learning, 16th IFAC joint symposia of mechatronics and robotics, 2025, arXiv:2502.09436. Cited for the reported effect of proportional and derivative gain magnitude on training stability and on tracking error, the paper attributing the statement to prior work it surveys.
34. Wang, Wang, Ren, Ben, Huang, Zhang and Pang, BeamDojo, Learning Agile Humanoid Locomotion on Sparse Footholds, Robotics, Science and Systems, 2025, arXiv:2502.10363. Cited for the dual critic separation of a dense locomotion objective from a sparse foothold objective.
35. Peng, Bao and Zhou, Gait-Conditioned Reinforcement Learning with Multi-Phase Curriculum for Humanoid Locomotion, 2025, arXiv:2505.20619. Venue beyond the arXiv listing not established by the retrieved source. Cited for the progressive introduction of gait complexity ahead of command space expansion.
36. Huang, Xu, Wang, Gao and Zhang, Traversing Narrow Paths, A Two-Stage Reinforcement Learning Framework for Robust and Safe Humanoid Walking, 2025, arXiv:2508.20661. Cited for the shaping of the foothold yaw angle and the Gaussian penalty on the foothold lateral offset within a two stage framework whose first stage tracks a template planner's footholds. The mathematical form of the two terms is not stated by the retrieved abstract and is not asserted here. Venue beyond the arXiv listing not established by the retrieved source.
37. Chou, Maturana and Scherer, Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution, International Conference on Machine Learning, Proceedings of Machine Learning Research volume 70, 2017. Cited for the bounded action space bias a Gaussian policy incurs where the physically admissible actions are bounded, and for the bounded distribution proposed to remove it.
38. Tanaka, Zhu, Wang, Liu and Hong, Mechanical Intelligence-Aware Curriculum Reinforcement Learning for Humanoids with Parallel Actuation, IEEE-RAS International Conference on Humanoid Robots, 2025, arXiv:2507.00273. Cited for the argument that the mechanical properties a mechanism embeds must be preserved through training rather than approximated away, and for the reported gain in transfer that preserving them delivers.

## 6. The velocity clamp, and the experiment ladder that follows from it

This chapter is written after the two runs of 2026-09-08 and supersedes the closing recommendation of [../context/KScale.md](../context/KScale.md) as it stood before the correction of 2026-09-09, which that correction now carries. Its purpose is to order the outstanding experiments so that each answers one question, and to record which of the questions the data already answers so that no run is spent on them.

### 6.1 What the data has already settled

Three propositions are established by the runs in hand and need no further experiment.

The action scale is not the discriminator. The runs `2026-09-08_06-50-00` and `2026-09-08_07-39-29` differ in the action scale alone, at 0.25 and 0.4, over 1670 configuration leaves, and they fail identically, at 21.1 and 18.5 steps of mean episode length against the reference's 1555.4 and at minus 1.28 and minus 1.31 of per step return against the reference's plus 0.17. A quantity that takes two values and produces one outcome is not the cause of that outcome.

The separation floor is not the discriminator. At the plateau `pen_feet_distance` contributes minus 0.0014 per step of a floor of minus 1.3, and the measured planar separation falls below its threshold on 4.23 per cent of samples where the reference, at a higher threshold, falls below on 47.72 per cent and walks regardless. Restoring the floor to 0.24 cannot recover one thousandth of what must be recovered.

The effort terms are not what the policy is avoiding. `pen_joint_torque` contributes minus 0.0053 per step and `pen_joint_accel` minus 0.0186, together one and nine tenths of one per cent of the floor, while the failing policies pass 2957.5 W and 4039.3 W of mean joint power against the reference's 302.2 W. The reading that the policy has learned that applying no torque is cheapest is not supported, the policy applying thirteen times the reference's power and receiving no authority for it.

What remains is the actuator parameterisation, and within it the single quantity the correction of 2026-09-09 identifies, the mismatch between the `velocity_limit` that shapes the torque speed curve and the `velocity_limit_sim` of 10 rad per second that the solver actually enforces, inherited unset from a URDF placeholder.

### 6.2 The change the ladder is built on

`environments/environments/assets/config/kscale_identified_cfg.py` sets `velocity_limit` on all six actuator groups and `velocity_limit_sim` on none. The repair is to set it, and the choice of value is the experiment. Two values are defensible and the ladder tests both, since each isolates the mechanism from a different side and agreement between them would establish it beyond the reach of a single confound.

The change touches one file, adds one keyword to each of six configuration objects and alters no shared module, so the constraint of `../../CLAUDE.md` that no existing caller may change behaviour is satisfied without an optional argument or a version two. No other robot reads this file. The TRON1 configuration already sets the same keyword at `environments/environments/assets/config/solefoot_cfg.py:50`, `61`, `111` and `122`, so the change brings the KScale into line with the pattern the working configuration already follows rather than introducing a new one.

### 6.3 The ladder

Arm V1 raises the clamp to meet the curve. Take the gains and effort limits of `2026-09-08_07-39-29` exactly as they stand and add `velocity_limit_sim` equal to `velocity_limit` on each group, being 40.0 at the hip yaw, hip roll, hip pitch and knee and 20.0 at the two ankles. Restore `min_feet_distance` to 0.24 and the action scale to 0.4, which are the reference values, so that the actuator parameterisation is the only difference from `2026-08-28_04-50-51`. The prediction is specific and falsifiable. Clamp occupancy at the knee falls from 51 per cent toward the reference's fraction of one per cent, mean joint power falls from the order of 3 kW toward the order of 300 W, and mean episode length passes 100 steps before iteration 1500 rather than falling to 17 at iteration 400.

Arm V2 lowers the curve to meet the clamp. Take the same gains and effort limits and instead revert `velocity_limit` to the reference values of 20.0 and 10.0, leaving `velocity_limit_sim` unset so that the solver clamp remains at 10. The torque available at the clamp returns to 30 Nm at the hips and knee and to zero at the ankles, which is the reference condition, while the doubled stall torque and the new gains are retained. Same reward set, same separation floor, same action scale as V1. The prediction is that this arm also survives, and the two arms together separate the mechanism from any residual effect of the gains, since they change the two limits in opposite directions and agree only in removing the actuator's ability to drive into the clamp.

Arm V3 is the arm the user proposed and it is retained unchanged, being cheap and informative. Take the reference gains, limits and action scale of `2026-08-28_04-50-51` and reduce `min_feet_distance` to 0.20 alone. This isolates the separation floor on a substrate known to walk, which no run in the series has yet done, and it is the only way to learn whether the floor helps or harms without the answer being contaminated by an actuator failure. It should be run with `lateral_only` set to True, since at False the term measures the planar norm and a stride of ordinary length supplies a third of the threshold on its own, which is the defect section 5 records at line 320 of this document. Where the intent is to compare against the reference on identical terms, run it at False and accept that it tests the planar norm rather than the stance width, but do not describe the result as a stance width finding.

Arm V4 is deferred until one of V1 or V2 produces a gait, and it is the arm the user proposed as the second of the pair, being the new gains at a separation floor of 0.24. Section 6.1 establishes that this arm run today would fail for the reason the 2026-09-08 pair failed, the separation floor accounting for one thousandth of the deficit, so running it before the clamp is repaired spends a run on a question already answered. Run it after, as the composition of a working actuator set with the floor that V3 will by then have priced.

Arm V5 is the per joint action scale, deferred likewise. The ankle roll's travel is 0.5236 rad and an action scale of 0.4 commands 76 per cent of it for a unit action, while the same scale commands 12 per cent of the hip pitch's 3.2637 rad. The measured range used by the ankle roll exceeds its full travel in all three runs, at 1.05, 1.02 and 1.15, which is the signature of a command spending much of its time outside the reachable set. `JointPositionActionCfg.scale` accepts a dictionary keyed by joint name expression at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py:35`, resolved at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:86`, so the change is a configuration change in the calling file alone. A scale proportional to travel, holding the hip pitch at its present 0.4 and setting the ankle roll near 0.064 and the ankle pitch near 0.171, is the natural first form.

### 6.4 The solver iteration counts, and when to revisit them

`solver_position_iteration_count` and `solver_velocity_iteration_count` stand at 2 at `environments/environments/assets/config/kscale_identified_cfg.py:113` and `114`. They are identical across all three runs and are therefore excluded as the differential cause, and they are not to be changed in V1 through V3, since changing them would confound the arm that carries the change.

They are nonetheless implicated and the measurement records how. The 10 rad per second clamp is overshot to 14.37 rad per second in the reference and to 23.68 in the run at action scale 0.4, a constraint violated by 44 per cent under 30 Nm of driving torque and by 137 per cent under 90 Nm. Two velocity iterations do not converge the projection and the residual scales with the force behind it. If V1 succeeds, the clamp moves to 40 and 20 rad per second and the joints no longer press against it, so the question retires of its own accord. If V1 fails while V2 succeeds, the clamp is still being pressed and an arm V6 raising both counts to 4 becomes the next question, to be run against V1 so that the iteration count is the only difference. Position iterations are the less likely culprit of the two, the ninety ninth percentile penetration beyond a hard joint limit standing at 0.0046 rad at worst across all three runs, so the limits themselves are being held.

### 6.5 What to log, so that the next failure is diagnosed from the training run

Three quantities would have identified this failure at iteration 400 rather than after two runs of twelve thousand iterations each, and none of them is presently logged during training. The fraction of timesteps at which any joint stands within one per cent of `velocity_limit_sim`, which would have read near zero for the reference and above a half for the failing pair. The summed absolute joint power, which separates the two by an order of magnitude. And the fraction of clamped samples at which the actuator torque is directed into the clamp, which is the signature proper, standing near a fifth for the reference and at unity for the failing pair. The first two are one line each against `articulation.data`, and the third is the product of two quantities already held.

### 6.6 Arm V1a as configured on 2026-09-09

The working tree was set on 2026-09-09 to an arm that is not V1 as section 6.3 specifies it, and the difference is recorded here so that the run is read against what it actually tests. V1 holds the derived gains of 2026-09-08 and repairs the clamp beneath them, isolating the clamp. The arm configured instead restores the gains of `2026-08-28_04-50-51` in full, being hip yaw 15 and 0.9, hip roll 150 and 17.3, hip pitch 200 and 22.4, knee 200 and 10.0, ankle roll 20 and 0.5 and ankle pitch 50 and 1.7, together with that run's action scale of 0.4 and its separation floor of 0.24, while retaining the doubled effort limits of 120 and 34 Nm, the raised `velocity_limit` of 40 and 20 rad per second and a `velocity_limit_sim` of 40 on every group. It therefore isolates the actuator limits rather than the clamp, the gains being held at the only set this robot has walked on.

The arm is well posed and its result will not be ambiguous, but it compounds two changes that push in opposite directions and it cannot separate them. Raising `velocity_limit_sim` from the URDF placeholder of 10 to 40 removes the dead zone entirely, no joint of the reference run having exceeded 14.37 rad per second, and that is expected to help. Doubling the effort limit removes a governor the reference was relying upon without having been designed to, since at an action scale of 0.4 the product of stiffness and scale is 80 Nm at the hip pitch and the knee, 60 at the hip roll and 20 at the ankle pitch, each of which the old limits of 60 and 17 Nm clipped and the new limits of 120 and 34 Nm do not. The action authority at those four joints therefore roughly doubles at the same gains, and that is expected to hurt. Should the arm fail, the two effects are separable by a follow up holding `velocity_limit_sim` at 40 and returning `effort_limit` and `saturation_effort` to 60 and 17, which restores the governor while keeping the dead zone repaired.

The ankle groups carry `velocity_limit` at 20 against `velocity_limit_sim` at 40, and the inequality is the correct direction rather than an oversight. The torque speed curve reaches zero at 20 rad per second and the solver clamp stands at 40, so the curve governs and the clamp is a backstop the joint never reaches. The reference carried the inverse, a clamp of 10 beneath a curve reaching zero at 20 at the hips, which is the pathology section 6.2 identifies.
