# The Quadruped, Its Environment Configuration as Built

This document records the quadruped's environment configuration as it stands in the tree, in the manner [tron1.md](tron1.md) records the TRON1's physical parameterisation. It is descriptive rather than prescriptive. Where a value required a derivation, that derivation lives in [joint_control_analysis_quadruped.md](joint_control_analysis_quadruped.md) for the actuators and in [quadruped_xml_to_urdf_conversion.md](quadruped_xml_to_urdf_conversion.md) for the asset, and this document states the value and points there rather than repeating the argument. What it does supply in full is the reasoning behind the reward, event and curriculum choices, since those were settled by a survey of the published quadruped configurations and are recorded nowhere else.

## 1. The robot

A twelve degree of freedom quadruped of 15.837077 kilogrammes, standing at 0.292 metres on legs of 0.448 metres from abduction axis to sole, with a lateral stance of 0.264 metres between the front feet. Each leg carries an abduction joint about the roll axis, a hip joint about the pitch axis and a knee joint about the pitch axis, in that order from the trunk, together with three actuator housings, a thigh of 0.213 metres, a calf of 0.213 metres and a spherical foot of radius 0.022 metres. The four legs carry 95.7 percent of the machine's mass, which is why the abduction joint sees the largest effective inertia of the three and why the leg's own weight materially offsets the ground reaction in the stance load.

The asset is `environments/environments/assets/urdf/quadruped/quadruped.urdf`, emitted by the generator beside it from the corrected MuJoCo model kept in the same directory. Its naming follows TRON1, `[abad/hip/knee]_[FR/FL/RR/RL]_[thigh/actuator]_[Link/joint]`, with the twelve actuated joints taking a capitalised `_Joint` suffix and the fixed joints a lowercase `_joint`, and the four feet named `foot_<LEG>_Link` so that a feet selecting expression is disjoint from the three joint group expressions.

The abduction axes are mirrored, `-1 0 0` on the two right legs and `1 0 0` on the two left, so that a positive command means outward on both sides as it does on TRON1.

## 2. The actuators

Three `IdentifiedActuatorCfg` groups in `environments/environments/assets/config/quadruped_identified_cfg.py`, selected by `abad_.._Joint`, `hip_.._Joint` and `knee_.._Joint`. The two wildcard form is deliberate and must not be relaxed, `hip_.*_Joint` additionally matching the fixed `hip_FR_thigh_joint`.

| Group | Kp | Kd | Effort limit | Velocity limit | Saturation effort | Armature |
|---|---|---|---|---|---|---|
| `abad` | 40.0 | 1.2 | 23.622511 | 30.0 | 158.0 | 0.01 |
| `hip` | 40.0 | 1.1 | 23.622511 | 30.0 | 158.0 | 0.01 |
| `knee` | 50.0 | 0.9 | 35.238000 | 20.0 | 236.0 | 0.01 |

Every stiffness and damping figure is derived and none is taken from the source model, whose position actuator gain of 100 is rejected for the reasons given in the companion document. Friction is the biped's, static 0.2, dynamic 0.02, activation velocity 0.1. The articulation raises its solver position iteration count from the biped's 4 to 8, following the recommendation of section 7.2 of [joint_control_analysis.md](joint_control_analysis.md) for underdamped joints undergoing contact transitions on rough terrain.

The articulation spawns at 0.32 metres, 28 millimetres above the standing height, so that a rough terrain tile cannot capture a foot at reset, with joint positions at the MuJoCo keyframe, abduction 0.0, hip 0.884337 and knee -1.768673 on all four legs. The pose is symmetric across the legs, where Isaac Lab's A1 and Go2 configurations instead use an abduction offset of 0.1 radians and different front and rear hip angles at `/ws/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:76` and `:159`. The symmetric pose is retained so that the URDF and the MuJoCo model describe the same standing robot, and the asymmetric alternative is recorded as a tuning option.

Self collision is enabled on both the spawn and the articulation root properties, because the abduction range permits a leg to reach its neighbour.

## 3. The reward configuration, and the survey that settled it

Seventeen terms. The observation set, the action interface, the command sampler, the terminations and the domain randomisation of the biped configuration are all robot agnostic in construction and needed only body names changed. The rewards are not, because the biped set contains terms whose entire purpose is to shape a two legged gait and because the weights of the terms that do transfer were tuned against a biped whose failure modes differ.

### 3.1 A note on scale, stated once

Every weight quoted from the literature below is on the scale of `legged_gym`, in which the linear velocity tracking reward carries a scale of 1.0 [1]. This repository carries the same term at 25, and both frameworks multiply by the control period before accumulating, so the two scales differ by a uniform factor of twenty five and a literature weight of `w` corresponds to `25w` here. Quoting a canonical weight without that factor invites a hundredfold error.

### 3.2 The terms and their weights

| Term | Function | Weight | Reasoning |
|---|---|---|---|
| `keep_balance` | `mdp.stay_alive` | 0.05 | Absent from every reference [1][2][3][6][7], which achieve the same protection through `only_positive_rewards`. This framework has no such clip, so the alive bonus is the mechanism available |
| `rew_lin_vel_xy` | `track_lin_vel_xy_exp` | 25, std sqrt(0.16) | Defines the scale |
| `rew_ang_vel_z` | `track_ang_vel_z_exp` | 12.5, std sqrt(0.16) | Restores the 2 to 1 linear to angular ratio universal across the references [1][2][5], against the biped's 3.33 to 1 |
| `pen_base_height` | `base_height_rough_l2` | -5.0, target 0.292 | The standing height, being the 0.270 kinematic drop plus the 0.022 foot radius |
| `pen_lin_vel_z` | `lin_vel_z_l2` | -2.0 | Canonical is -2.0 [1][2][5]. The biped's -0.5 diverges a hundredfold from canon at this scale, and the move is partial, see section 3.3 |
| `pen_ang_vel_xy` | `ang_vel_xy_l2` | -0.5 | Canonical is -0.05 invariantly [1][2][3][5], a tenfold move, likewise partial |
| `pen_joint_torque` | `joint_torques_l2` | -2.0e-4 | The Go2 value in two independent ports [2][5], this robot sharing Go2's 0.213 metre segments and near identical torque ceilings |
| `pen_joint_accel` | `joint_acc_l2` | -2.5e-7 | Adopted outright, the single most stable constant in the survey [1][2][3][7] |
| `pen_action_rate` | `action_rate_l2` | -0.05 | Between the -0.01 of [1][2][5] and the -0.1 of [7], the quadruped's higher joint count raising the aggregate |
| `pen_joint_pos_limits` | `joint_pos_limits` | -2.0 | Retained, the soft limit factor of 0.9 doing most of the work |
| `pen_joint_vel_l2` | `joint_vel_l2` | -5.0e-5 | Retained. The references fold this into a power term instead [3][4] |
| `pen_action_smoothness` | `ActionSmoothnessPenalty` | -0.075 | Retained unchanged, no canonical value exists to prefer [6] |
| `pen_flat_orientation` | `flat_orientation_l2` | -1.0 | The rough terrain band is -0.2 to -1.0 [3][7] and the flat value is far higher [6], so the single value is a compromise a flat variant may raise |
| `pen_undesired_contacts` | `undesired_contacts` | -2.5, threshold 10.0 | Raised from the biped's -0.5, a quadruped's thighs and shanks meeting terrain far more often. Selects `base_Link`, `abad_.*`, `hip_.*`, `knee_.*`, which is every body but the four feet |
| `pen_abad_deviation` | `joint_deviation_l1` | -2.0 | Added on the survey's recommendation, following the `hip_pos` term of Extreme Parkour [7]. The abduction joints are the degrees of freedom a policy widens first, to buy a larger support polygon, and this robot's 0.264 metre stance is narrow against its 0.448 metre leg |
| `feet_air_time` | `feet_air_time` | 2.0, min 0.25, max 0.40 | Threshold retuned, see section 3.4 |
| `feet_slide` | `feet_slide` | -0.25 | Retained unchanged [2][6] |

### 3.3 Why the penalty rebalancing is partial

Adopting the canonical penalties outright, which at this repository's scale would mean -50 for the vertical velocity and -1.25 for the horizontal angular velocity, would be defensible in `legged_gym` and is not defensible here. Every reference carrying those weights also sets `only_positive_rewards`, which clips the summed reward at zero and so prevents a policy from discovering that termination is cheaper than locomotion [1][2][7]. This framework has no such clip, and the alive bonus of 0.05 is the whole of the protection. A hundredfold increase in the dominant penalty against an unclipped return is a change whose failure mode is a policy that lies down, and it must not be made blind.

Each divergent penalty therefore moves part of the way, by a factor of four for the vertical velocity and ten for the horizontal angular velocity, and the canonical target is recorded so that a later pass may close the gap deliberately once a first policy has trained. Should the trained gait bounce vertically, the vertical velocity penalty is the first weight to raise, and the canonical -50 is the ceiling to raise it toward.

### 3.4 The air time threshold, and why zero is the wrong value

With `threshold_min` at zero the term reduces to the air time itself gated on first contact, which is non negative and monotonically increasing, so every additional millisecond aloft is paid for and nothing whatever penalises a foot that hangs. The positive threshold is precisely the mechanism that supplies the lower branch, a foot returning sooner than the threshold earning a negative contribution and thereby being pushed toward lift off, and removing it leaves a term that on a light robot is maximised by hopping [1].

The canonical 0.5 seconds is not arbitrary either. It is half the stride period of ANYmal's roughly one hertz trot, so it is the swing duration of the gait the term is meant to elicit at a duty factor of one half [1]. The same derivation at the two hertz trot this quadruped's scale suggests gives a stride period of 0.5 seconds and a swing duration of 0.25 seconds, which is the value adopted. This repository's `feet_air_time` additionally clamps the excess at `threshold_max - threshold_min`, a refinement the canonical form lacks, and 0.40 seconds is the ceiling, so the term saturates at 1.6 times the target swing rather than paying without limit.

### 3.5 Terms deliberately absent

Foot clearance during swing is present and weighted in HIMLoco, DreamWaQ and Walk These Ways [3][4][6] and is the standard mechanism against toe stubbing, which air time does not police because air time rewards duration and says nothing about trajectory. This repository implements it at `environments/environments/tasks/locomotion/mdp/rewards.py:234`, and it is nonetheless omitted from every variant. The implementation measures `body_pos_w[..., 2]`, a world frame height, against a fixed target. On flat ground that is exactly the sole clearance for a spherical foot, the frame origin being the sphere's centre, and the term is correct. On generated terrain the ground is not at zero, so a foot lifted correctly over a raised tile reads as far above target and earns nothing, while a foot dragged along the floor of a pit reads as at target and earns fully. A term right on one variant and wrong on the other would have to be enabled on the flat tasks and zeroed on the rough ones, which makes the two families no longer comparable and adds a tuning surface before the configuration has produced a single gait. The remedy is a variant measuring against the height scanner already mounted on `base_Link`, which would also repair the term for the biped configurations that carry it today, and that is new work on a shared module.

A gait or phase symmetry term requires a phase clock, a desired contact schedule and per command gait parameters, and is machinery rather than a reward line [6][8]. None of the plain tracking baselines carries it and the air time term already supplies a soft periodicity signal.

A base acceleration penalty is a biped motivated term addressing double support hopping and appears in none of the quadruped configurations retrieved [9]. A stumble penalty sits at a scale of zero in the base `legged_gym` function set and is revived only for parkour [1][7], and it largely detects the same event the slide and contact penalties already price. A power penalty is an alternative parameterisation of the torque and joint velocity terms already carried [3][4]. A termination penalty is switched off in every reference, the clipping mechanism serving in its place [1].

### 3.6 Rebalancing is safe with respect to the curriculum

One objection to changing the tracking weights would be that the curriculum keys off them. It does not, in the sense that matters. Every curriculum term that reads a reward compares the episode mean against `update_threshold * term_cfg.weight * env.step_dt`, at `environments/environments/tasks/locomotion/mdp/curriculums.py:302` and `:436` and the corresponding lines of the other two, so the comparison is against a fraction of the maximum the term can attain and is invariant under a change of weight. Raising `rew_ang_vel_z` from 7.5 to 12.5 therefore leaves the angular command and tracking standard deviation curricula behaving exactly as before.

## 4. The events

Twelve terms, of which two of the biped's are dropped and one is added. `prepare_quantity_for_tron` exists solely to populate `env._foot_radius` for `feet_regulation` and `nominal_foot_position`, neither of which the quadruped uses, and the ankle gain randomisation addresses a joint that does not exist.

The mass randomisation required a change and the reason is a finding rather than a preference. The biped's `add_base_mass` scales `base_Link` by 0.65 to 1.35, which on a biped whose trunk is 5 kilogrammes of a 9.6 kilogramme robot perturbs the total by roughly a fifth. This quadruped's trunk is 0.674397 kilogrammes of 15.837077, 4.3 percent of the machine, so the same scale perturbs the total by under two percent and the randomisation is very nearly inert.

The obvious repair, an absolute addition of plus or minus 1.5 kilogrammes, is wrong and must not be made. `randomize_rigid_body_mass` validates a scale range for positivity but performs no such check on an additive one, at `/ws/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:321`, so a sample of minus 1.5 would give the trunk a mass of minus 0.826 kilogrammes. What is configured instead is a small asymmetric payload on the trunk of -0.3 to 0.6 kilogrammes guarded by `min_mass=0.2`, together with a second term `add_link_mass` scaling every body by 0.90 to 1.10, which delivers the plus or minus ten percent of total mass the references randomise over. The biped sketched exactly such a term and left it commented out.

The three gain randomisations are re-expressed over the quadruped's three joint groups with uniform ranges, 0.85 to 1.15 on stiffness and 0.80 to 1.20 on damping, the biped's asymmetric per group figures having been derived from absolute ranges that do not apply here. Joint offsets, joint friction and armature scaling are carried across unaltered, as are both reset terms.

The push magnitude is halved to plus or minus 0.25 metres per second. Isaac Lab's own Go2 rough configuration disables the push entirely, on the reasoning that a light robot on randomised terrain is already perturbed enough [2]. The push is retained at half magnitude rather than removed, because the push curriculum is one of the mechanisms the co-optimisation experiments rely upon and removing it would make the quadruped runs non comparable with the biped runs.

## 5. The terrain

A quadruped specific generator at `environments/environments/tasks/locomotion/cfg/quadruped/terrains_cfg.py`, rather than the biped's, because this robot stands at 0.292 metres where the biped stands at 0.75 and reusing the biped's would put 0.10 metre stairs and 0.20 metre wave amplitudes under a robot a third its height. Isaac Lab's own Go2 port scales its terrain difficulty down explicitly for the same reason [2].

The sub terrain proportions are the biped's exactly, so that the curriculum presents the same mixture to both robots, and only the amplitudes differ. Stair heights run to 0.08 metres against the biped's 0.10, wave amplitudes to 0.12, random rough noise to 0.05. The horizontal scale is halved from 0.1 to 0.05 metres, since a 0.1 metre height field cell is a third of this robot's foot to foot stance and would quantise the terrain into features the foot cannot resolve, and the stair width is narrowed from 0.3 to 0.25 metres for the same reason.

## 6. The class hierarchy and the task registry

Every quadruped class carries a `Quadruped` prefix. The specification originally named the classes `PFSceneCfg`, `PFEnvCfg` and so on, every one of which is already taken by the LimX TRON1 pointfoot biped, and `cfg/__init__.py` flattens every family's public names into one namespace, so a quadruped module joining that star import under the original names would shadow the biped's classes and the failure would surface as a biped task spawning a quadruped rather than as an import error. The prefix removes the hazard rather than containing it.

The `cfg/__init__.py` star import places the quadruped first rather than last, which is the one detail of that file that is not arbitrary. Nine further classes in each family module carry generic names, `CommandsCfg`, `ActionsCfg`, `ObservationsCfg`, `EventsCfg`, `RewardsCfg`, `TerminationsCfg`, `CurriculumCfg` and the two co-optimisation and hybrid internal model observation classes, and those already collide three ways among the biped families. Placing the quadruped first leaves every existing resolution exactly as it stood, so the fourth star import adds six names and displaces none, which was verified by simulating the flattening over the parsed module trees.

Correction and outcome, 2026-08-24. A full codebase review conducted after implementation, `grep`ing every `.py` file under `environments/` for an import of the bare `tasks.locomotion.cfg` package or of `..cfg`/`.cfg` from within it, found none, every robot module imports its own family's submodule directly instead, for instance `robots/limx_pointfoot_env_cfg.py:6` reads `from environments.tasks.locomotion.cfg.PF.limx_base_env_cfg import PFEnvCfg` rather than going through the flattened package, and the same holds for SF, WF and the quadruped itself at `robots/quadruped_pointfoot_env_cfg.py:4`. The flattened namespace that `cfg/__init__.py` builds is consequently never read by anything in the tree, Python still executes the file's four star imports as a side effect of importing any submodule beneath the package, but the names they bind are inert. The ordering analysis above remains a correct static property of the file and the prefixing decision it justifies stands, but the shadowing hazard it guards against is not, at present, one any caller could actually trigger.

Three base environment classes in `cfg/quadruped/base_env_cfg.py`, one per runner, and eighteen classes in `robots/quadruped_pointfoot_env_cfg.py` giving each runner a base, a play, a blind flat, a blind flat play, a blind rough and a blind rough play variant. The hierarchy is two levels beneath each base class rather than the biped's four, because the quadruped has no USD variant and therefore needs no layer to switch between URDF and USD.

Twelve task identifiers are registered in `robots/__init__.py` as explicit `gym.register` blocks in the biped's style, no loop.

| Identifier | Configuration | Runner |
|---|---|---|
| `Isaac-Quadruped-Blind-Flat-v0` and `-Play-v0` | `QuadrupedPFBlindFlatEnvCfg` | vanilla |
| `Isaac-Quadruped-Blind-Rough-v0` and `-Play-v0` | `QuadrupedPFBlindRoughEnvCfg` | vanilla |
| `Isaac-Quadruped-HIM-Blind-Flat-v0` and `-Play-v0` | `QuadrupedPFHIMBlindFlatEnvCfg` | hybrid internal model |
| `Isaac-Quadruped-HIM-Blind-Rough-v0` and `-Play-v0` | `QuadrupedPFHIMBlindRoughEnvCfg` | hybrid internal model |
| `Isaac-Quadruped-Copt-Flat-v0` | `QuadrupedPFCoptBlindFlatEnvCfg` | co-optimisation |
| `Isaac-Quadruped-Copt-Rough-v0` and `-Play-v0` | `QuadrupedPFCoptBlindRoughEnvCfg` | co-optimisation |
| `Isaac-Quadruped-Copt-Learned-Rough-v0` | `QuadrupedPFCoptBlindRoughEnvCfg` | co-optimisation, learned model |

The four co-optimisation identifiers must not be launched yet. The environment side is complete, `CoptOnPolicyRunner` taking its observation group mapping from `PFQuadrupedCoptPPORunnerCfg.obs_groups` at `co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:483` and reading no body name, joint name or asset path of its own, but the design generator does not yet accept this robot and `scripts/rsl_rl/train.py` hardcodes the biped URDF at lines 198 to 210. Registering them costs nothing, `gym.register` constructing no environment, and keeps the task matrix complete against the day that work lands.

The container tooling gains four training modes and four evaluation modes, `quadruped`, `quadruped-flat`, `quadruped-him` and `quadruped-copt`, in `../../djinn`.

## 7. Observation dimensions, useful at first run

The velocity command term returns three values, the heading being consumed internally by the command generator. The policy group of `ObservationsCfg` is therefore 3 plus 3 plus 3 plus 12 plus 12 plus 12 plus 3, which is 48 per step and 480 over the ten step history. The `HIMObservationsCfg` policy group omits the base linear velocity and is 45 per step. The morphology group is eight link lengths plus twenty five body masses plus two hundred and twenty five inertia components, which is 258, the eight link lengths being the four thighs and four shanks against the biped's four.

None of these is asserted anywhere in the configuration, every network being sized from the observation at construction, so a mismatch surfaces as a shape error at the first forward pass rather than as silent misbehaviour.

## 8. Bibliography

1. Rudin, N., Hoeller, D., Reist, P., Hutter, M. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Conference on Robot Learning, 2021. arXiv:2109.11978. Reward scales, the terrain curriculum and the command curriculum were read from the `leggedrobotics/legged_gym` repository, files `legged_gym/envs/base/legged_robot_config.py`, `legged_gym/envs/base/legged_robot.py` and `legged_gym/envs/anymal_c/mixed_terrains/anymal_c_rough_config.py`, retrieved 2026-08-19.
2. Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., and others. Orbit, A Unified Simulation Framework for Interactive Robot Learning Environments. IEEE Robotics and Automation Letters 8(6), 3740 to 3747, 2023. arXiv:2301.04195. The author list was not verified beyond the six named and the short form is used. Configuration values were read from the `isaac-sim/IsaacLab` repository, files `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`, `mdp/rewards.py`, `mdp/curriculums.py`, `config/anymal_c/rough_env_cfg.py`, `config/go2/rough_env_cfg.py` and `config/go2/flat_env_cfg.py`, retrieved 2026-08-19, and from the local checkout at `/ws/IsaacLab`.
3. Long, J., Wang, Z., Li, Q., Cao, L., Gao, J., Pang, J. Hybrid Internal Model, Learning Agile Legged Locomotion with Simulated Robot Response. International Conference on Learning Representations, 2024. arXiv:2312.11460. Reward weights cross verified against the `InternRobotics/HIMLoco` repository, files `legged_gym/legged_gym/envs/a1/a1_config.py` and `envs/base/legged_robot_config.py`, retrieved 2026-08-19.
4. Nahrendra, I. M. A., Yu, B., Myung, H. DreamWaQ, Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination via Deep Reinforcement Learning. IEEE International Conference on Robotics and Automation, 2023. arXiv:2301.10602. The reward table was retrieved from the article's HTML rendering and no official repository was located to corroborate it, so its weights carry the caution due a single source.
5. Unitree Robotics. `unitreerobotics/unitree_rl_gym` repository, files `legged_gym/envs/go2/go2_config.py`, `legged_gym/envs/g1/g1_config.py` and `legged_gym/envs/base/legged_robot_config.py`, retrieved 2026-08-19. A software artefact with no accompanying article identified.
6. Margolis, G. B., Agrawal, P. Walk These Ways, Tuning Robot Control for Generalization with Multiplicity of Behavior. Conference on Robot Learning, 2022. arXiv:2212.03238. Reward functions read from the `Improbable-AI/walk-these-ways` repository, files `go1_gym/envs/go1/go1_config.py` and `go1_gym/envs/rewards/corl_rewards.py`, retrieved 2026-08-19. The retrieved source did not confirm whether the author list extends beyond the two named.
7. Cheng, X., Shi, K., Agarwal, A., Pathak, D. Extreme Parkour with Legged Robots. IEEE International Conference on Robotics and Automation, 2024. arXiv:2309.14341. Reward scales read from the `chengxuxin/extreme-parkour` repository, files `legged_gym/legged_gym/envs/a1/a1_parkour_config.py`, `envs/base/legged_robot.py` and `envs/base/legged_robot_config.py`, retrieved 2026-08-19.
8. Siekmann, J., Green, K., Warila, J., Fern, A., Hurst, J. Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition. IEEE International Conference on Robotics and Automation, 2021. arXiv:2011.01387.
9. van Marum, B., and others. Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking. 2024. arXiv:2404.19173. The author list was not verified in this pass and is reused from the existing entry in `../../context/literature.md` cluster 7.
