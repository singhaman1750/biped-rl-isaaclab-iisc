# MorAL.md — Software Architecture and Implementation of the Vendored MorAL Clone

This document records the architecture and the implementation of the `MorAL/` tree vendored at the root of this repository, read in full against its sources on 2026-08-10. It exists so that a later session may judge what the implementation offers the design and controller co-optimisation work without re-reading seven thousand lines of Isaac Gym era code. Every claim about behaviour carries a `path:line` citation resolved against this repository's root, and every claim drawn from the literature carries a numbered marker resolved against the bibliography in the closing section.

## 1. Provenance and what the clone actually is

MorAL is published as a morphologically adaptive locomotion controller for quadrupedal robots on challenging terrain, whose contribution is the concurrent training of a control policy alongside an adaptive module that reads temporal robot states, so that the policy identifies the platform it inhabits online and estimates its own body velocity rather than being told either [1]. The published work evaluates four specific quadruped configurations and compares against proprioception only baselines [1].

The tree vendored here is not the authors' release. Its remote is `git@github.com:StochLab/MorAL.git`, its head is `0ac74da8` dated 2026-02-19, and its `README.md` documents installation of HIMLoco rather than of MorAL, instructing the reader to clone `OpenRobotLab/HIMLoco` and retaining the `projects/himloco` and `projects/h_infinity` directories of that repository intact. The codebase is therefore the hybrid internal model implementation [2] with a morphology adaptation layer grafted onto it, and the commit history states the evolution explicitly. Commit `17c22b8e` fed privileged morphology directly to the policy, `3b59e21e` moved to estimating it from observation history through the HIM estimator, `de6125a2` introduced a dedicated morphology network together with the adaptive framework and the phase separation, `8552cd5a` removed that network again, `5c9b97cb` restored it and trained it under both the regression and the policy gradient, and `6dcf2fdb` declares that phase one now reproduces the MorAL paper including its rewards, that a wider variety of URDFs is produced by `generate_urdf_variants.py`, and that phases two and three improve on phase one over the same variant set.

Two consequences follow for any reader. The first is that this is a working reimplementation carrying its author's own extensions, so a discrepancy against the published method is expected rather than anomalous, and the document below describes the code rather than the paper. The second is that the morphology randomisation strategy, procedurally generating a population of simulated robots so that one controller spans them all, is the GenLoco construction [3], and the configuration flag that switches it on is literally named `GenLoco` at `MorAL/legged_gym/legged_gym/envs/base/legged_robot_config.py:35`.

## 2. Repository layout and package boundaries

The clone carries two installable packages and a resource tree, following the legged_gym convention [5].

| Path | Role |
|---|---|
| `MorAL/legged_gym/legged_gym/envs/base/legged_robot.py` | The single environment class, 1439 lines, holding the simulation, the observation assembly, the morphology loading and every reward |
| `MorAL/legged_gym/legged_gym/envs/base/legged_robot_config.py` | The base configuration, and the location of the `GenLoco` switch and the phase selector |
| `MorAL/legged_gym/legged_gym/envs/{a1,go1,aliengo,mule,b2,minicheetah,stoch3}/` | Per robot configuration overrides |
| `MorAL/legged_gym/legged_gym/utils/` | Task registry, terrain generation, argument parsing, policy export |
| `MorAL/legged_gym/legged_gym/scripts/` | Training, playback, URDF generation and population analysis entry points |
| `MorAL/legged_gym/resources/robots/` | Stock robot assets, the `All_robots` evaluation set, and the `Generated` training population |
| `MorAL/rsl_rl/rsl_rl/` | The vendored learning library, carrying both the stock `PPO` path and the `HIM` path that MorAL uses |

Only four tasks are registered, at `MorAL/legged_gym/legged_gym/envs/__init__.py:43-46`, namely `a1`, `go1`, `aliengo` and `mule`, all four bound to the same `LeggedRobot` class and differing only in configuration. The `b2`, `minicheetah` and `stoch3` configurations exist but are never registered, and the `mule` configuration points at a URDF path, `resources/robots/mule_v2_description/urdf/urdf_fixed.urdf`, that is absent from the resource tree. In practice the sole exercised task is `go1`. It is the only configuration carrying `GenLoco` branches at `MorAL/legged_gym/legged_gym/envs/go1/go1_config.py:36, :111, :130`, it is the task the `README.md` instructs the reader to train, and `MorAL/legged_gym/legged_gym/scripts/train.py:64-65` copies `go1_config.py` into the log directory of every run whatever the task. The `GenLoco` path in any case replaces the configured asset with the generated population, so a task selection survives only in the control gains, the nominal pose and the reward weights.

## 3. Simulation architecture

The substrate is Isaac Gym Preview 4 driven through the legged_gym abstractions [5]. `BaseTask` at `MorAL/legged_gym/legged_gym/envs/base/base_task.py:38` acquires the gym handle, allocates the observation, reward, reset and timeout buffers, and calls `create_sim`, which builds the terrain and then the environments at `MorAL/legged_gym/legged_gym/envs/base/legged_robot.py:366-382`.

Timing is fixed by three configured quantities. The physics step is 0.005 seconds at `legged_robot_config.py:239`, the control decimation is four at `:116`, and the resulting control period is 0.02 seconds, computed at `legged_robot.py:1124`. The episode is twenty seconds at `legged_robot_config.py:57`, hence one thousand control steps, and the rollout horizon is 120 steps per environment under `GenLoco` at `:303-304`, so a policy update spans rather less than an eighth of an episode.

The step function at `legged_robot.py:84-120` clips the action, optionally staggers it across the decimation window to emulate an actuation delay at `:93-97`, and then runs the four physics substeps, recomputing the torque from the possibly delayed action at each one and refreshing only the degree of freedom state tensor inside the loop at `:103-111`. The root, contact and rigid body tensors are refreshed once afterwards at `:128-130`. Termination, reward and reset follow at `:147-151`, and the observation is assembled after the reset so that a reset environment reports its new state at `:153`.

The return signature of `step` is widened relative to stock legged_gym, returning seven values rather than five at `:120`, the two additions being the indices of the environments that terminated and the privileged observation captured immediately before their reset. This exists so that the learning loop can form a correct next state target for the estimator at an episode boundary, and it is consumed at `MorAL/rsl_rl/rsl_rl/runners/him_on_policy_runner.py:119-131`. The abstract `VecEnv` interface at `MorAL/rsl_rl/rsl_rl/env/vec_env.py:50` still declares the five value signature, so the contract is documented nowhere but in the two call sites.

Termination is narrower than the legged_gym default. At `legged_robot.py:166-181` an environment resets on contact force above one newton at any body named in `terminate_after_contacts_on`, on timeout, or when the base height falls below four centimetres. The roll and pitch termination that the file once carried is commented out at `:169-173`, so a robot may walk in any attitude that keeps its base above four centimetres and its trunk off the ground.

Terrain is a triangle mesh built from a height field at `MorAL/legged_gym/legged_gym/utils/terrain.py:58-71`, laid out as a ten by twenty grid of eight metre square patches, the row index carrying difficulty and the column index carrying type. The configured type proportions are one tenth smooth slope, one fifth rough slope, three tenths stairs up, three tenths stairs down and one tenth discrete obstacles at `legged_robot_config.py:81`.

## 4. How morphological variation is realised

This is the centre of the implementation and the part most relevant to co-design, so it is treated in its own right.

### 4.1 The population is generated offline

`MorAL/legged_gym/legged_gym/scripts/generate_urdf_variants.py` is a standalone script, not a module, executing its work at import at `:521-545`. It writes 3600 URDF files into `resources/robots/Generated/urdf` and 3600 matching JSON parameter files into `resources/robots/Generated/params`, and both directories are present in the clone at those counts, occupying 85 megabytes.

Every variant is produced by parametric surgery on one donor file, `go1_new.urdf`, at `:187-354`. The script recomputes the inertia tensor of each link from its primitive shape, a box for the trunk, thigh and calf, a cylinder for the hip and a sphere for the foot at `:104-163`, then rewrites the trunk collision box, the hip joint origins from the trunk half extents scaled by sampled insets at `:176-184`, the thigh joint lateral offset from the hip length, the calf joint offset from the thigh length, and the foot offset from the calf length at `:231-262`. The joint graph is never touched, so every variant carries the identical twelve joints under the identical names, which is precisely the constraint that the workspace architecture document records for the Isaac Lab `MultiUsdFileCfg` path in `../../CO_OPTIMISATION.md` section 4.7.2.

Sampling is at `:366-457`. One overall size scale is drawn from a uniform distribution and the limbs are scaled by that factor raised to the power seven tenths, an allometric convention at `:371-374`. Masses are not sampled directly. Volumes are computed from the sampled dimensions, target masses are drawn, the implied densities are clipped into physically plausible bands at `:431-435`, and the masses are then recomputed from the clipped densities at `:438-442`, which guarantees that no variant is made of an impossible material. A final guard rescales any population member above eighty kilogrammes at `:449-455`. Four donor parameter sets seed the sampler by index, Go1 below five hundred, AlienGo from five hundred to one thousand, X30 from one thousand to two thousand and B2 above two thousand at `:521-532`.

The resulting population, measured directly from the 3600 JSON files, spans 4.96 to 79.99 kilogrammes with a mean of 39.21, thigh lengths from 0.182 to 0.425 metres and trunk lengths from 0.296 to 0.985 metres. Rather more than half the population lies between twenty and fifty kilogrammes.

### 4.2 One asset per environment, assigned once

Under `GenLoco` the environment count is forced to 3600 at `legged_robot_config.py:259`, matching the population exactly. At `legged_robot.py:930-951` the directory listing is sorted and one Isaac Gym asset is loaded per environment, wrapping modulo the file count if the two ever disagree, and at `:1053-1056` each environment's actor is created from its own asset. The asset list is discarded and the garbage collector invoked immediately afterwards at `:1079-1081`, since 3600 loaded assets are otherwise resident for the life of the run.

The morphology vector accompanying each environment is read from the matching JSON at `:960-986`, selecting ten of the fourteen sampled fields, namely trunk length, trunk width, trunk height, trunk mass, hip mass, thigh length, thigh mass, calf length, calf mass and total mass. Two entries are rescaled on load, the trunk mass divided by ten at `:984` and the total mass divided by twelve at `:985`, the latter being the same ratio used as the actuator gain scale. The pairing of a URDF to its parameter file is positional, both directories being sorted independently and indexed by environment number, so it is correct only because the two naming schemes happen to sort congruently.

Two derived per environment quantities follow. The nominal base height is computed from the leg geometry at `:997-999` as the thigh length times the cosine of one radian plus the calf length times the cosine of minus half a radian, which over this population ranges from 0.243 to 0.639 metres with a mean of 0.431. The total mass is retained unscaled at `:1005` and used at `:569-574` to scale both actuator gains by the ratio of the robot's mass to twelve kilogrammes, a factor spanning 0.41 to 6.67 across the population.

### 4.3 What this mechanism does not do

The morphology of an environment is fixed at construction and never altered. There is no reload path, no per environment respawn, and no design optimiser anywhere in the tree. The population is drawn once by an offline script and the training run consumes it as a fixed distribution. MorAL is therefore a generalist controller over a design distribution rather than a co-design method, and it contributes nothing directly to the between generation morphology reload that `CoptOnPolicyRunner` performs, whose implementation is recorded in `../../CO_OPTIMISATION.md` section 5.1.

## 5. Observation and action space architecture

The action is twelve dimensional at `legged_robot_config.py:38`, interpreted as a joint position offset from the nominal pose, scaled by 0.25 for the Go1 configuration at `MorAL/legged_gym/legged_gym/envs/go1/go1_config.py:61` and applied through a proportional derivative law at `legged_robot.py:578-580`. Actions are clipped to plus or minus one hundred at `legged_robot_config.py:219`, which is no constraint in practice.

The single step observation is 45 dimensional at `legged_robot_config.py:37` and is assembled at `legged_robot.py:266-272`. The single step privileged observation extends it to 264 at `:49` and is assembled at `:282-296`. The layout of the latter, which is the layout the estimator targets index into, is as follows.

| Index range | Content | Scale | Source |
|---|---|---|---|
| 0 to 3 | Commanded planar and yaw velocity | 2.0, 2.0, 0.25 | `legged_robot.py:266` |
| 3 to 6 | Base angular velocity in base frame | 0.25 | `:267` |
| 6 to 9 | Projected gravity in base frame | 1.0 | `:268` |
| 9 to 21 | Joint position minus nominal | 1.0 | `:269` |
| 21 to 33 | Joint velocity | 0.05 | `:270` |
| 33 to 45 | Previous action | 1.0 | `:271` |
| 45 to 48 | Base linear velocity in base frame | 2.0 | `:283` |
| 48 to 58 | The ten dimensional morphology vector | as loaded | `:284` |
| 58 to 61 | Applied disturbance force | 1.0 | `:285` |
| 61 to 65 | Foot contact indicators | boolean | `:287, :290` |
| 65 to 77 | Foot positions | 1.0 | `:288, :291` |
| 77 to 264 | Height scan, eleven by seventeen samples | 5.0 | `:293-296` |

Noise is added to the first 45 entries only, from the vector built at `:681-712`, and separately to the height scan at `:295`. The command entries receive no noise at `:701` and the previous action entries none at `:707`.

The actor observation is a stacked history of the first 45 entries, six deep, giving 270 dimensions at `legged_robot_config.py:48` with `num_obs_hist` set to five at `:40`. The buffer is a shift register with the newest frame at index zero at `legged_robot.py:298`. The privileged buffer is one frame deep at `legged_robot_config.py:51` and its shift at `legged_robot.py:299` degenerates to a straight overwrite.

Two configuration fields adjacent to these, `num_adapt_obs_hist` at `:41` and `error_input_to_adapt_policy` at `:43`, are declared and are read by nothing in the tree.

Foot positions enter the critic in world coordinates, taken directly from the rigid body state tensor at `legged_robot.py:141`, so their magnitude grows with the robot's absolute position on the terrain until the observation clip at plus or minus one hundred at `legged_robot_config.py:218` truncates them. Every other spatial quantity in the observation is expressed in the base frame.

## 6. Policy architecture

`HIMActorCritic` at `MorAL/rsl_rl/rsl_rl/modules/him_actor_critic.py:74` holds five networks, of which four are trainable by the policy optimiser and one keeps its own.

The morphology encoder at `:108-116` is a three hidden layer perceptron of widths 128, 64 and 32, mapping the full 270 dimensional observation history to thirteen outputs. Its first three outputs are read as a base velocity estimate and its remaining ten as the morphology estimate, the latter passed through a rectifier at `:197` so that predictions of lengths and masses cannot go negative.

The hybrid internal model estimator at `:105` is `HIMEstimator` from `MorAL/rsl_rl/rsl_rl/modules/him_estimator.py:8`, unchanged in structure from the published architecture [2]. Its encoder maps the same 270 dimensional history to nineteen outputs at `:33-39`, three read as a velocity estimate and sixteen as a latent that is projected onto the unit sphere at `:70`. A target network maps a single frame to the same sixteen dimensional sphere at `:42-48`, and a bank of thirty two prototypes at `:51` mediates the swapped assignment objective of Caron et al. [4] between the two, its Sinkhorn normalisation implemented at `:116-130` and its loss at `:104`. The account of that objective already held by this repository, in `HIM.md`, applies to this file unchanged.

The nominal actor at `:118-128` is a perceptron of widths 512, 256 and 128 consuming 58 inputs at `:99`, namely the newest observation frame concatenated with the encoder's velocity and morphology estimates. It does not see the observation history directly, only the newest frame and the two estimates distilled from the history.

The adaptive actor at `:131-139` is a second perceptron of the same widths consuming 86 inputs at `:98`, namely the newest observation frame, the nominal actor's twelve output actions, the estimator's three velocity outputs, the ten morphology estimates and the estimator's sixteen dimensional latent. Its output is a residual added to the nominal action rather than a replacement for it.

The critic at `:142-151` consumes the full 264 dimensional privileged observation, which carries the true morphology vector in clear at indices 48 to 58. The value function is therefore morphology aware by construction, which is the same property that `../../context/copt.md` establishes for the co-optimisation critic in this workspace.

A running mean and variance normaliser is defined at `:39-72` and instantiated nowhere, so observations reach every network unnormalised beyond the fixed configured scales.

## 7. The three phase training architecture

The phase is a single integer at `legged_robot_config.py:305` read by the runner at `him_on_policy_runner.py:67` and passed to both the policy and the algorithm. It selects among three branches of `update_distribution` at `him_actor_critic.py:200-243` and among three loss compositions at `MorAL/rsl_rl/rsl_rl/algorithms/him_ppo.py:158-199`. The phases are not a schedule within a run. Each is a separate run resuming from the previous one's checkpoint.

Phase one, at `him_actor_critic.py:201-213`, trains the morphology encoder, the nominal actor and the critic. The action distribution is centred on the nominal actor's output alone, so the adaptive actor is evaluated under no gradient and its output discarded. Because the actor consumes the encoder's outputs without detaching them, the encoder receives gradient from the policy surrogate as well as from its own regression, which is the arrangement commit `5c9b97cb` describes.

Phase two, at `:215-227`, freezes the morphology encoder, the nominal actor and the estimator behind a no gradient block, and trains only the adaptive actor and the critic. The action distribution is centred on the sum of the nominal action and the residual.

Phase three, at `:229-241`, trains everything except the estimator under the policy optimiser, the estimator remaining behind a no gradient block in the forward path while continuing to train under its own optimiser. This is the configured default at `legged_robot_config.py:305`.

The loss composition follows the same partition. In phases one and three the morphology encoder is additionally supervised at `him_ppo.py:161-169`, regressing its velocity output onto privileged indices 45 to 48 and its morphology output onto indices 48 to 58, the two mean squared errors summed. That sum is combined with the policy loss at `:198-199` as one half of each, so the policy gradient is halved whenever the regression is active. In phases two and three the estimator updates itself at `:171-174`, once per minibatch, its learning rate slaved to the adaptive learning rate that the Kullback Leibler schedule computes at `:142-156`. With five epochs over four minibatches at `legged_robot_config.py:290-291`, the estimator therefore takes twenty gradient steps per policy iteration.

One further term belongs to the phase structure and lives in the runner rather than in the environment. At `him_on_policy_runner.py:121-123`, in phases two and three, the reward is augmented by one hundredth of the negative exponential of the norm of the difference between the sampled action and the nominal mean, which prices the residual and the exploration noise together and pays the policy for leaving the nominal action alone. It is added after the environment returns and before the transition is stored, so it enters the return without appearing in any logged episode sum.

The remainder of the algorithm is unmodified proximal policy optimisation [6] with generalised advantage estimation at `MorAL/rsl_rl/rsl_rl/storage/him_rollout_storage.py:104-118`, advantages normalised over the whole batch at `:118`, and timeout bootstrapping at `him_ppo.py:109-110`. The storage carries one field beyond the stock layout, the next privileged observation at `:65`, which the runner overwrites at terminating indices with the pre reset observation at `him_on_policy_runner.py:130-131` so that the regression targets are never contaminated by a reset state.

Hyperparameters are at `legged_robot_config.py:284-297`, namely a clip parameter of 0.2, an entropy coefficient of 0.01, five epochs, four minibatches, an initial learning rate of one thousandth under the adaptive schedule, a discount of 0.99, a trace decay of 0.95 and a target divergence of 0.01. With 3600 environments and a 120 step horizon the batch is 432000 transitions and each minibatch is 108000.

## 8. Reward architecture

Rewards are discovered by name at `legged_robot.py:815-838`, any term of zero weight being removed before the lookup and every surviving weight multiplied by the control period, so the configured numbers are rates per second and the applied numbers are per step. This is the same convention that `../../context/gait_metrics.md` records for the Isaac Lab pipeline, and the same trap applies to any budget computed from them.

The effective weights under `GenLoco`, after the overrides at `go1_config.py:111-122` are applied to the base block at `:89-110`, are as follows.

| Term | Weight | Implementation |
|---|---|---|
| `tracking_lin_vel` | 1.0 | `legged_robot.py:1314-1317` |
| `tracking_ang_vel` | 0.5 | `:1319-1322` |
| `feet_air_time` | 1.0 | `:1403-1414` |
| `base_height` | -2.0 | `:1344-1347` |
| `lin_vel_z` | -2.0 | `:1324-1326` |
| `collision` | -0.5 | `:1380-1382` |
| `orientation` | -0.2 | `:1332-1334` |
| `foot_clearance` | -0.1 | `:1349-1362` |
| `ang_vel_xy` | -0.05 | `:1328-1330` |
| `action_rate` | -0.01 | `:1364-1366` |
| `joint_power` | -2e-5 | `:1340-1342` |
| `smoothness` | -1e-4 | `:1368-1370` |
| `dof_acc` | -1.25e-7 | `:1336-1338` |

Negative totals are not clipped, `only_positive_rewards` being false at `go1_config.py:124`, so the policy may be paid a net penalty. Terms present in the configuration at zero weight, and therefore inert, are `termination`, `feet_stumble`, `stand_still`, `torques`, `dof_vel`, `dof_pos_limits`, `dof_vel_limits`, `torque_limits`, `thigh_positions` and `hip_positions`.

Two terms are morphology dependent, and they are the reason this section matters beyond MorAL. The base height penalty at `:1344-1347` is quadratic in the deviation from the per environment nominal height derived in section 4.2, so a four kilogramme robot and an eighty kilogramme robot are each held to their own kinematically consistent stance rather than to one shared constant. The configured constant `base_height_target` of 0.33 at `go1_config.py:131` is consequently dead under `GenLoco`. The foot clearance penalty at `:1349-1362` likewise derives its setpoint from that nominal height, taking one third of it, and multiplies the squared height error by the lateral foot speed so that the penalty falls away when the foot is not travelling.

The clearance term carries a sign defect. Foot position is expressed in the base frame at `:1355`, where a supporting foot sits roughly one nominal height below the origin and the coordinate is therefore negative, while the setpoint at `:1358` is a positive third of the nominal height, between 0.081 and 0.213 metres over this population. The line it replaced, retained as a comment at `:1360`, used the configured `clearance_height_target` of minus 0.20 at `go1_config.py:133`, which carries the correct sign for that frame. As written the term prices the foot against a target above the hip, which no gait can reach, so what survives is a foot speed penalty with a large and nearly constant multiplier rather than a clearance shaping term.

## 9. Curricula

Three curricula operate, of which only two are curricula in the usual sense.

The terrain curriculum at `legged_robot.py:643-663` is the game inspired promotion rule of legged_gym [5]. An environment that travels more than half a terrain length in an episode is promoted one difficulty row, one that travels less than half the distance its command implied is demoted, and one that clears the top row is scattered uniformly rather than held there. The initial level is drawn from the first six rows at `legged_robot_config.py:75`.

The command curriculum at `:665-678` widens the forward velocity range by two tenths in each direction whenever the mean tracking reward exceeds four fifths of its maximum, up to a ceiling of two metres per second at `go1_config.py:68`. The gate is evaluated separately over the first fifth of the environments and the remainder, both of which must pass, because command resampling at `legged_robot.py:542-548` reserves that first fifth for the widened range and holds the remainder at plus or minus one metre per second. Commands below a magnitude of two tenths are zeroed at `:551`, and the yaw command is derived from a heading error at `:517-520`.

The phase progression of section 7 is the third, and it is manual. Nothing in the tree advances the phase, promotes a checkpoint or verifies that the run being resumed was trained at the preceding phase.

## 10. Domain randomisation

The randomisations active under the committed configuration at `legged_robot_config.py:144-180` are the ground friction over the band 0.2 to 1.25, the base centre of mass displacement over plus or minus five centimetres, the proportional and derivative gain factors over 0.9 to 1.1, an external disturbance force over plus or minus thirty newtons applied every eight steps, and a base velocity push every sixteen seconds. Friction and restitution are resampled on every environment reset at `legged_robot.py:424-437`, gain factors likewise at `:220-225`.

Payload mass, link mass, restitution and the actuation delay are configured off. One further entry is configured on and has no effect. The motor strength factor is allocated at `:794`, sampled at `:803-804` and resampled at `:224-225`, and is read by nothing, the torque computation at `:578-580` applying only the gain factors and the mass scale. Motor strength randomisation is therefore silently inactive.

## 11. Entry points, persistence and export

Training runs through `MorAL/legged_gym/legged_gym/scripts/train.py:43-67`, which builds the environment and the runner from the registry, writes the invoking command line and the pickled environment and training configurations into the log directory, and copies `legged_robot.py`, `legged_robot_config.py` and `go1_config.py` alongside them at `:60-65`, so a completed run is self describing in the same spirit as the dumped `params/env.yaml` of the Isaac Lab pipeline.

Checkpoints are written every hundred iterations at `him_on_policy_runner.py:161-162` and hold the policy state, the policy optimiser state, the estimator optimiser state and the iteration counter at `:247-253`. The prototype bank and the target network are inside the policy state and therefore persist. Nothing else does, so the observation history buffers, the terrain levels, the command range reached by the curriculum and the reward episode sums are all lost across a resume, and a phase two run therefore restarts its terrain and command curricula from the beginning.

Two defects sit on the resume path. The load root is hard coded to `logs/rough_<task>` at `MorAL/legged_gym/legged_gym/utils/task_registry.py:154`, ignoring the experiment name under which the run was written, so the sequence the `README.md` prescribes, training into `phase1_genloco_exp1` and then loading that name, resolves to `logs/rough_go1/phase1_genloco_exp1` and fails unless the directory is moved by hand. And the TorchScript exporter at `MorAL/legged_gym/legged_gym/utils/helpers.py:238-248` is inherited unmodified from HIMLoco, feeding the actor the newest observation frame with the estimator's velocity and sixteen dimensional latent, sixty four inputs in all, whereas the MorAL actor was built at `him_actor_critic.py:99` to accept fifty eight, namely the frame with the morphology encoder's velocity and ten dimensional morphology estimate. The exporter also omits the adaptive actor entirely. Export is enabled by default in `play.py` at `:184`, so it will be exercised, and it cannot produce a usable artefact.

Playback runs through `MorAL/legged_gym/legged_gym/scripts/play.py:44-181`. It caps the environment count at fifty at `:68`, forces `GenLoco` on at `:69`, and repoints the population directories at `resources/robots/All_robots` at `:71-72`, which holds four real robots, the A1, the AlienGo, the Go1 and the MULE 2, so evaluation cycles over genuine hardware descriptions rather than over the training population. It hard codes phase three at `:107` and adds the residual to the nominal action at `:140-141`. The sibling `play_batch.py` is stale, calling the inference policy as though it returned actions alone at `:248`, whereas it returns five tensors at `him_actor_critic.py:262`.

## 12. Register of defects and divergences established by this reading

The following are recorded so that a later reader neither rediscovers them nor mistakes them for design. Each is a statement about the code as committed at `0ac74da8`.

1. Every environment receives the degree of freedom properties of the first generated asset. `dof_props_asset` is fetched from `assets[0]` at `legged_robot.py:955` and passed unchanged into every actor at `:1061-1063`. The consequence is that the mass scaled effort limits the generator computes at `generate_urdf_variants.py:165-174` and writes into each URDF at `:344-351` are discarded. Measured directly, the first variant declares 54.55 newton metres and the last declares 250.41, and both are simulated and clipped at 54.55. Since the actuator gains are simultaneously scaled by up to 6.67 at `legged_robot.py:569-574`, the heaviest robots in the population are commanded large torques and then truncated at a thirteen kilogramme robot's ceiling.
2. In the same path, `self.dof_pos_limits` and `self.dof_vel_limits` are written without an environment guard at `:453-455`, so the last environment's values survive for all. This is presently harmless, the generator not varying joint limits, and becomes a defect the moment it does.
3. The foot clearance setpoint carries the wrong sign for the frame it is evaluated in, as set out in section 8.
4. Motor strength randomisation is sampled and never applied, as set out in section 10.
5. The TorchScript exporter cannot produce a runnable policy, as set out in section 11.
6. The resume path ignores the experiment name, as set out in section 11.
7. Estimation loss is accumulated twice per minibatch in phase three, both conditional blocks at `him_ppo.py:210-215` firing, so the logged estimation loss is double the true value in the default phase.
8. The morphology loss is accumulated as a tensor rather than a scalar at `him_ppo.py:211`, retaining autograd references across the epoch loop.
9. Two stray imports are present, `from sqlite3 import adapt` at `him_actor_critic.py:30` and `from mpmath import phase` at `him_on_policy_runner.py:36`, the latter a hard dependency on a package absent from `requirements.txt`.
10. The `feet_stumble` reward name in the configuration at `go1_config.py:104` has no matching `_reward_feet_stumble` method, the implementation being named `_reward_stumble` at `legged_robot.py:1416`. It is latent only because the weight is zero and the term is removed before the lookup.
11. Foot positions enter the critic in world coordinates, as set out in section 5.

## 13. Assessment against the co-design work of this workspace

Four elements of this implementation bear directly on the design and controller co-optimisation pipeline recorded in `../../CO_OPTIMISATION.md`, and it is worth separating them from the rest.

The first is the supervised morphology head. The pipeline's learned model extension, described in `../../CO_OPTIMISATION.md` section 5.3 and grounded in `../../context/literature.md`, gives the actor a latent from an estimator trained on dynamic quantities. MorAL supplies the complementary construction, a head regressing the design vector itself from proprioceptive history, so that a policy identifies which body it is driving rather than inferring the consequences of that body. The two are compatible, and the target is trivially available in the co-optimisation setting because the design generator produced it. The relevant caveat is that MorAL regresses ten of the fourteen sampled design parameters, the hip length and radius, the limb cross sections, the foot radius and the hip insets never appearing in the target, so the residual four act as unobserved variation the policy must absorb through the estimator latent instead.

The second is the morphology dependent reward setpoint. Deriving the nominal base height from the leg link lengths at `legged_robot.py:997-999` rather than fixing it in configuration is the correct treatment of a target that a changing design invalidates, and it addresses precisely the failure that `../../context/brs_gait.md` records for SD_BRS1, where the configured base height target proved kinematically unreachable. Any co-optimisation over link lengths inherits that problem by construction, since the design the optimiser proposes changes the height the robot can hold, and MorAL's derivation is the shortest available remedy. The same argument applies to the gain scaling by mass at `:569-574`, which keeps the closed loop bandwidth comparable across a population whose inertia spans a factor of sixteen, and which should be read alongside the inertia based gain derivation in `BRS.md`.

The third is the residual phase structure. Freezing a nominal policy and training a residual on top of it, then optionally unfreezing both, offers a way to hold a comparison fixed while a specialisation is learned, which speaks to the recurring difficulty in this workspace of comparing runs whose reward weights have moved. It is not free, since the small residual bonus at `him_on_policy_runner.py:121-123` is an unlogged term in the return, and since nothing in the tree enforces that a resumed checkpoint was trained at the preceding phase.

The fourth is what does not transfer. The morphology mechanism is a startup time assignment of one asset per environment with no reload path, which is the Isaac Gym analogue of `MultiUsdFileCfg` and answers a strictly weaker requirement than the between generation reload that `CoptOnPolicyRunner` already implements. There is no design optimiser, no fitness accumulation and no notion of a generation anywhere in the tree, so nothing here informs the evolutionary half of the loop. The environment is Isaac Gym rather than Isaac Lab, so no code ports without rewriting, and the whole reward and observation specification is quadruped shaped, twelve actuated joints in four identical legs with point feet, which the bipedal tasks of this repository do not resemble. What transfers is the three architectural ideas above, not the code that carries them.

## 14. Bibliography

1. Luo, Z., Dong, Y., Li, X., Huang, R., Shu, Z., Xiao, E., Lu, P. (2024). MorAL: Learning Morphologically Adaptive Locomotion Controller for Quadrupedal Robots on Challenging Terrains. IEEE Robotics and Automation Letters 9, 4019-4026. DOI 10.1109/LRA.2024.3375086. The issue number was not stated by the retrieved sources.
2. Long, J., Wang, Z., Li, Q., Gao, J., Cao, L., Pang, J. (2024). Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response. ICLR 2024, arXiv:2312.11460.
3. Feng, G., Zhang, H., Li, Z., Peng, X. B., Basireddy, B., Yue, L., Song, Z., Yang, L., Liu, Y., Sreenath, K., Levine, S. (2022). GenLoco: Generalized Locomotion Controllers for Quadrupedal Robots. CoRL 2022, arXiv:2209.05309.
4. Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A. (2020). Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. NeurIPS 2020, arXiv:2006.09882.
5. Rudin, N., Hoeller, D., Reist, P., Hutter, M. (2022). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. CoRL 2021, PMLR 164, 91-100, arXiv:2109.11978.
6. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
