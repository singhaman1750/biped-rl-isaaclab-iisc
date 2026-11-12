# Sagittal Symmetry for the TRON1 and the Quadruped, Implementation Plan

> Status, IMPLEMENTED on 2026-10-20, after being revised and validated in a sandbox the same day. The six symmetry files, the test suite and the three diffs were applied to the working tree verbatim, `himloco` was reinstalled in both containers, and the suite and the smoke test of section 9.3 passed on the live tree, section 13 recording the application and its two departures from the text. The training runs of section 9.4 have not been launched. Written against commit `23513d2` and the uncommitted working tree of that date. See [README.md](README.md) for the register.

> Revision of 2026-10-20. The first pass of this plan registered new symmetry tasks and runner subclasses for the two robots and left the SD_BRS1 and KScale modules bitwise unchanged. Review asked for four changes, each adopted here. The co-optimisation policy group carries the plain task's ten frame history, which the first pass denied (section 3.3, item 1). The old robots receive the corrected world frame mirror as well as the new (item 4). Every term of every group of every configuration, plain, HIM and co-optimisation, is mapped, and a term without a mapping raises rather than passing unmirrored (items 5 and 6, section 5.1). And the symmetry configuration is set on the two existing parent runners, so that every task they serve inherits it and no task is registered (section 6). The first pass's six task identifiers and four runner subclasses are withdrawn. The HIM and co-optimisation paths, which the first pass left out of scope, are brought into it, and each needed one line outside the symmetry package (sections 8.8 and 8.9).

## 1. Introduction

Two robots of this repository have learned gaits that a left right mirror of the robot would not reproduce. The TRON1 solefoot biped walked on its knees for roughly its first twelve thousand iterations, and the replay of its iteration 6000 checkpoint found that the knee mode is not a kneel on both shins but an asymmetric gait on the right foot and the left shin, the right foot loaded in 80.8 per cent of environments and the left shank in 20.5 per cent against 2.7 for the right (`context/tron1.md:311`, `plans/tron1_locomotion_fine_tuning.md:298`). The quadruped of the design generator has repeatedly retracted a limb, run A of the degenerate gait investigation holding the front right and rear left pair aloft (`/ws/plans/QUADRUPED_DEGENERATE_GAIT.md:198`) and a later run ceasing to land its rear left foot, whose landings fell from about 110 per episode to 1.3 (`/ws/plans/QUADRUPED_DEGENERATE_GAIT.md:453`).

Both faults share a structure. Each is a support pattern that singles out one side, and its mirror image, the left foot and the right shin, or the rear right foot retracted, is an equally good solution that the policy never visits. A legged robot that is symmetric about its sagittal plane carries a reflection symmetry in its Markov decision process, and when the dynamics, the initial state distribution and the reward are invariant under that reflection the optimal policy is equivariant and the optimal value invariant [1]. Unconstrained learning ignores this structure. The workspace already exploits it for the SD_BRS1 and the KScale biped through the symmetry hook of the vendored rsl_rl, and the run that first enabled it produced the first SD_BRS1 policy to walk with a coordinated alternating gait (`/ws/plans/SYMMETRY_PLAN.md:3`).

This plan extends that machinery to the TRON1 and to the quadruped, for every task their runners serve, the plain locomotion tasks on flat and rough terrain, the HIM tasks and the co-optimisation tasks alike, and enables the mirror loss on both. It first moves the code the two existing modules share into `mdp/symmetry/common.py`, where one table maps every observation term the four robots' tasks carry, and it corrects the existing modules where they leave a mirrored quantity unmirrored, while keeping their mirrored actions and policy observations bitwise identical. The document proceeds from the theory and the literature (section 2), through the current implementation and its defects (section 3), the robots (section 4), their environments and tasks (section 5) and their agents (section 6), to the architecture of the reorganised module (section 7), the implementation itself (section 8), its application (section 9), its backwards compatibility (section 10), the record of its validation (section 11) and its risks (section 12).

One limit must be stated at the outset. The mirror removes a side bias, it does not forbid a gait. An equivariant policy may still retract a foot, provided it retracts the mirrored foot in the mirrored state, and a fault that is itself invariant under the reflection, a kneel on both shins or a quadruped that lifts both rear feet, is untouched by it. The workspace recorded the latter caution for the quadruped before this plan (`/ws/context/literature.md`, cluster 19, the synthesis on symmetry). What the mirror changes is that neither side may be preferred, so a fault that persists only because one side was discovered first loses its support, and the doubled data reaches the mirrored half of the state space that a side biased policy never samples.

## 2. Theory and literature

### 2.1 The group and what it implies

The sagittal reflection is the matrix M equal to diag(1, -1, 1) acting on the base frame, an element g of order two, so the group is C2 equal to {e, g} [1][5]. Its action on each observed quantity follows from that quantity's geometric type. A polar vector, the base linear velocity or the projected gravity, maps to M v and so negates its lateral component. A pseudovector, the base angular velocity, maps to det(M) M w, which negates its roll and yaw components and keeps the pitch. The velocity command negates its lateral velocity and its yaw rate. A joint coordinate maps to the coordinate of its partner joint, with a sign set by the joint axes, since a rotation about the axis a reflects into a rotation by the same angle about -M a, so a joint keeps its sign when its partner's axis equals -M a and negates when it equals M a [5]. A per body scalar such as a mass is permuted between partner bodies, and an inertia tensor is permuted and conjugated, which negates exactly the products of inertia that couple the lateral axis (`/ws/plans/SYMMETRY_PLAN.md:163`).

Every one of these maps is a signed permutation of the coordinates. This is the observation on which the reorganised module of section 7 rests, because it means that the mirror of an entire observation group, of any width and any history, is one index vector and one sign vector over its last axis.

### 2.2 The four mechanisms, and augmentation against the loss

The literature offers four ways to impose the symmetry, the duplication of transitions through the mirror, an auxiliary penalty on the departure from equivariance, the replay of half a gait cycle, and a network that is equivariant by construction [2]. The duplication is the data augmentation of the rsl_rl hook, and the penalty is the mirror loss, whose form is the squared distance between the action at a state and the mirror of the action at the mirrored state [3][2]. No single method won everywhere in the foundational comparison, the loss being the most consistent soft method and the network the only exact one [2].

The comparison that bears most directly on this plan is the one made on the very interface the workspace uses. On ANYmal, augmentation obtained the highest return and the fastest convergence with a low symmetry loss, the loss alone induced symmetry at a cost in performance and convergence, and the two combined did not necessarily improve either, the augmentation not benefiting from the additional gradients of the loss [4]. The same work showed that a mirrored sample must be scored against the old action probability of the original transition rather than its own, since the latter can be arbitrarily small for a policy that is not yet symmetric, and the vendored rsl_rl satisfies this by tiling the old log probabilities from the originals (`/ws/rsl_rl/rsl_rl/algorithms/ppo.py:243`) [4]. A humanoid study adds that a soft loss reduced its tracking error by 15.3 per cent against an unconstrained baseline but left a spatial asymmetry the equivariant policy did not, because a soft loss can be violated at inference [6]. The primary quadruped study found the strictly equivariant policy the most sample efficient in simulation, and the soft augmentation the more robust where real actuator and mass asymmetries break the assumption [1].

The plan therefore implements both mechanisms in the existing runners, as the request asks, with the baselines in flight as the control. The comparison of augmentation alone against the combination, which [4] found not necessarily better, lies one field of the configuration away and is left as an ablation (section 6.2).

### 2.3 The critic

A value function that observes an unmirrored privileged state cannot learn the invariance the theory requires, since each mirrored row would carry the original privileged state and the replicated return of that same state, so the value loss on the mirrored half would merely repeat the original half (`/ws/context/rsl_rl.md`, the section on the critic group). The SymmLoco implementation transforms the critic observations with their own representation, and the primary study states that augmenting the critic biases the learned value toward invariance [1]. The most direct measurement located is the humanoid study's ablation, in which replacing its invariant critic with a plain one raised the velocity tracking error from 9.85 to 11.06 [6]. The two existing modules mirror the critic, and the reorganised module mirrors it and every other group.

### 2.4 Earlier quadruped evidence

Symmetry augmentation was applied to the quadruped of the DeepMind control suite before any of the works above, with faster learning in a data limited regime [7]. The ANYmal reference module of Isaac Lab augments with four transformations, the identity, both reflections and their composition (`/ws/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/symmetry/anymal.py:47-63`) [4], and SymmLoco's CyberDog2 uses the sagittal reflection alone, swapping the left and right legs and negating the abduction of each (`/ws/plans/SYMMETRY_PLAN.md:71`). Section 4.2 establishes that this repository's quadruped admits only the sagittal reflection.

### 2.5 Two cautions carried forward

A strictly symmetric policy cannot leave a symmetric neutral pose, so training must begin from a perturbed posture [2]. Both robots reset joints by a uniform offset of 0.2 rad and the base with random velocities (`environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:843-867`, `environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:1050-1072`), which breaks the symmetry of every initial state. And symmetry enforcement gives no reliable learning speedup and may slow learning [2], so the success criterion of section 9.4 is gait symmetry and tracking, not iterations to convergence.

## 3. The current implementation

### 3.1 The consumers

The rsl_rl PPO stores the symmetry configuration at construction, resolving a `module:function` string to the callable (`/ws/rsl_rl/rsl_rl/algorithms/ppo.py:76-97`), and the runner injects the environment as `_env` into the algorithm dictionary it was handed, in place (`/ws/rsl_rl/rsl_rl/runners/on_policy_runner.py:28`, `on_policy_runner.py:401`, `/ws/rsl_rl/rsl_rl/modules/symmetry.py:23-24`). The function is called at three sites by keyword, with both inputs for the augmentation (`ppo.py:231-246`), with observations only for the mirror loss when augmentation is off (`ppo.py:319-323`), and with actions only to mirror the mean actions (`ppo.py:332-335`). It returns the augmented pair with the originals in the first rows, which the entropy, the KL and the mirror loss rely upon (`ppo.py:254-256`, `ppo.py:339-341`), and the augmentation factor is read from the returned batch (`ppo.py:241`). The containers train with an installed copy of the library, which was confirmed identical to `/ws/rsl_rl/rsl_rl/algorithms/ppo.py` on 2026-10-20.

Two further algorithms consume the same configuration, and both were confirmed byte identical to their installed copies on the same date. The co-optimisation algorithm repeats both branches verbatim (`co_optimisation/co_optimisation/algorithms/copt_ppo.py:179-186`, `copt_ppo.py:268-290`) and adds an estimator loss computed on the augmented batch, against a target drawn from the same batch (`copt_ppo.py:256-259`), so the estimator's input and target groups must be mirrored with the rest. HIMPPO repeats the augmentation branch (`himloco/himloco/algorithms/him_ppo.py:169-186`) but stores each augmented batch beside its unaugmented successor batch for the estimator (`him_ppo.py:244`), whose update encodes the 2n states and regresses them onto the n successors' targets (`him_ppo.py:359-361`, `himloco/himloco/modules/him_estimator.py:84-127`), a mismatch of 2n rows against n that the velocity loss and the swap loss cannot broadcast (`him_estimator.py:96`, `him_estimator.py:120-121`).

```text
 rollout (env.step)                      update (per mini-batch)
 ------------------                      -----------------------------------------------
 obs TensorDict --> RolloutStorage -->   obs_batch, actions_batch [, next_obs_batch for HIM]
 {policy, critic, commands,                    |
  obsHistory, targetEnc, estimatorGT,          v
  morphologyObs, privilegedDynamicsObs, data_augmentation_func(env, obs, actions)   ppo.py:235
  historyObs}                                  |
                       +-----------------------+------------------------+
                       | obs_aug = obs.repeat(2)                         |
                       | obs_aug[g][n:] = obs[g][..., index_g] * sign_g  |
                       |     for EVERY group g, unmapped term raises     |
                       | act_aug = [a ; a[:, jperm] * jsign]             |
                       +-----------------------+------------------------+
                                               |
          HIM only: next_obs_batch mirrored alike, so the estimator     him_ppo.py:169-186
          pairs 2n states with 2n successors                            (section 8.8)
                                               |
          old log prob, values, advantages, returns tiled x2            ppo.py:243-246
                                               |
          surrogate on all 2n rows, value loss on all 2n rows (critic mirrored)
          copt only: estimator loss on all 2n rows (input and target mirrored)
          entropy and KL on the first n rows only                       ppo.py:254-256
                                               |
          mirror loss  ||pi(M_s s) - M_a pi(s)||^2 on rows n..2n        ppo.py:316-346
          added x mirror_loss_coeff when use_mirror_loss
```

### 3.2 The two modules and what they share

`mdp/symmetry/brs.py` and `mdp/symmetry/kscale.py` were written side by side, and a line by line reading shows that they differ in exactly four places, the joint partner rule, the body partner rule, the flip set, and the strictness of their builders. The table records the split.

| Element | `brs.py` | `kscale.py` | Shared |
|---|---|---|---|
| Joint partner | trailing L or R swapped, a name with neither its own partner (`brs.py:83`) | `right_` or `left_` prefix swapped, raising otherwise (`kscale.py:138-147`) | no |
| Body partner | the same trailing rule (`brs.py:94`) | explicit dictionary, the CAD names sharing no stem (`kscale.py:80-89`) | no |
| Flip set | HipRoll, AnkleRoll, HipPitch (`brs.py:35`) | hip_pitch, hip_roll, hip_yaw, foot_roll (`kscale.py:76`) | no |
| Missing partner, involution | `index` raises, no involution check | explicit raise and involution assertion (`kscale.py:156-163`, `kscale.py:180-181`) | no |
| History, grid, shape counts | 10, (11, 17), None (`brs.py:31`, `brs.py:39-46`) | identical (`kscale.py:68`, `kscale.py:95-102`) | yes |
| Sign vectors | `brs.py:49-53` | `kscale.py:105-113`, identical | yes |
| Term kind tables | `brs.py:56-72` | `kscale.py:116-132`, identical | yes |
| Material permutation, layout, cache, apply, group mirror, entry point | `brs.py:99-209` | `kscale.py:185-295`, identical | yes |

Everything below the partner rules is duplicated, some 120 lines in each file. The modules are reached by the two runner configurations (`agents/limx_rsl_rl_ppo_cfg.py:315-320`, `agents/limx_rsl_rl_ppo_cfg.py:360-365`), by the package's aliases (`mdp/symmetry/__init__.py:5-10`), and by two analysis scripts preserved as artefacts (`context/artefacts/kscale_flat-2026-09-28_to_09-29-clip-ablation/scripts/spawn_probe_arm14.py:39`, `validate_arm14.py:191`). No module of `mdp` imports the package (`mdp/__init__.py:1-8`), so it is loaded only by the agent configurations.

### 3.3 Defects and hazards in the current code

Nine properties of the existing code and its consumers bear on the extension. The plan corrects every one of them, for the old robots as for the new, except the residual of item 4, which no observation can remove.

1. The history length is a constant of 10 (`brs.py:31`, `kscale.py:68`), applied to every term of the two groups mirrored (`brs.py:153`, `kscale.py:239`). The groups of the tasks these runners serve carry four layouts. The policy and critic of the plain and co-optimisation tasks hold ten frames flattened time major (`cfg/SF/limx_base_env_cfg.py:237-239`, `limx_base_env_cfg.py:322-324`, `cfg/quadruped/base_env_cfg.py:202-204`, `base_env_cfg.py:344-346`), the co-optimisation policy included, since the TRON1's now inherits the plain group and the quadruped's sets the same ten frames (`base_env_cfg.py:426-427`), and since both robots' estimators of the morphology now read it (`agents/limx_rsl_rl_ppo_cfg.py:164`, `agents/quadruped_rsl_rl_ppo_cfg.py:62`). `historyObs` holds twenty five frames flattened (`limx_base_env_cfg.py:418-419`, `base_env_cfg.py:662-663`), and is to be retired in favour of the policy group, being read meanwhile only by the TRON1's other three experiments (`limx_rsl_rl_ppo_cfg.py:177`, `limx_rsl_rl_ppo_cfg.py:190`, `limx_rsl_rl_ppo_cfg.py:203`). The HIM policy, target and estimator groups hold one frame, and the HIM history holds ten frames kept as a separate axis (`limx_base_env_cfg.py:554-556`, `base_env_cfg.py:788-790`), the census of section 5.1 recording every one. A constant serves the first layout alone. On the single frame HIM policy group it divides a three wide term into ten frames of zero width, and the reshape raises, so the SD_BRS1 and KScale HIM tasks cannot complete an update under the symmetry configuration their own runners carry, which section 11 confirmed on the live tree. The reorganised module reads each term's `history_length` from the observation manager, which writes the group's history into every term (`/ws/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:550-552`) and flattens it time major (`observation_manager.py:423-424`, `observation_manager.py:638-647`), and a term whose history is kept as its own axis is mirrored along its last axis frame by frame. The first pass of this plan cited an override at `limx_base_env_cfg.py:1428` that set the co-optimisation policy history to zero, an override the working tree removed on 2026-10-20, and drew from it the conclusion that the co-optimisation policy carries no history, which is withdrawn.
2. The height grid is a constant of 11 rows by 17 columns, valid only for the "xy" ordering of a 1.6 by 1.0 m pattern at 0.1 m (`brs.py:37-40`). The reorganised module derives the mirror from the ray pattern itself, pairing each ray with the ray at its lateral reflection, so any pattern and either ordering is handled and a pattern that is not symmetric raises.
3. The material term is left identity until a hand filled list of shape counts is supplied (`brs.py:42-46`), and neither module has had it filled. The reorganised module counts the shapes per body from the PhysX views by the loop Isaac Lab itself uses (`/ws/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:205-221`), which runs only for a body subset there, the full body list resolving to `slice(None)`, for all four robots.
4. The four world frame critic terms are identity (`brs.py:66-70`). `feet_lin_vel` reports each foot's velocity in the world frame (`mdp/observations.py:198-204`) and `feet_contact_force` each foot's net world frame contact force (`observations.py:139-148`), so identity assigns each foot its own vector in the mirrored state rather than its partner's, a mirror of the robot in which the feet were not exchanged. Section 5.1 shows that exchanging the feet is exact on the vertical and forward components and wrong only in the sign of the lateral one, that the root position (`observations.py:101-107`) is exactly invariant, and that the world yaw rate of the root velocity (`observations.py:110-116`) negates exactly. The correction is adopted for all four robots. The residual, the lateral sign, requires the base yaw, which no stored observation carries, and is recorded as standing in section 12.
5. A term with no entry in the kind table is silently identity (`brs.py:181`). A term added to a group, the link lengths of the co-optimisation critic for instance (`limx_base_env_cfg.py:1424`), would be tiled unmirrored without notice. The reorganised module raises on any term of any group that the table does not map.
6. Only the `policy` and `critic` groups are mirrored, and every other group is tiled unmirrored (`brs.py:200-204`). The co-optimisation estimator reads its input and its target from the augmented batch (`copt_ppo.py:256-259`, `co_optimisation/co_optimisation/modules/copt_actor_critic.py:126-133`), and the HIM estimator reads the history, the target encoding and the base velocity target, so tiling those groups would teach either estimator that a mirrored input predicts an unmirrored target. The reorganised module mirrors every group the observation manager carries, the `commands` group included.
7. HIMPPO pairs the augmented batch of 2n states with the unaugmented batch of n successors (section 3.1). Mirroring the successor batch alike is one call, added under the same condition as the augmentation, so an algorithm without symmetry is untouched (section 8.8).
8. The co-optimisation branch of `train.py` hands its runner the dictionary it later dumps to `params/copt_agent.yaml` (`scripts/rsl_rl/train.py:305`, `train.py:316-322`, `train.py:353`). Once a symmetry configuration is set, the runner writes the environment into that dictionary, and the dump then fails, the YAML representer being unable to pickle the environment's USD stage, which section 11 recorded. The plain and HIM branches pass a fresh `to_dict()` to their runners and dump the configuration class (`train.py:325`, `train.py:348`), and are unaffected. The runner receives a copy (section 8.9).
9. The comment of the package `__init__.py` states that `compute_symmetric_states` is kept bound to SD_BRS1, but the name is not bound, only the prefixed aliases (`mdp/symmetry/__init__.py:2-10`). The comment is corrected and no binding changes.

## 4. The robots

### 4.1 The TRON1

The TRON1 asset is the MuJoCo document emitted by the design generator, `assets/mjcf/solefoot/tron1/base_robot.xml`, spawned through `MjcfFileCfg` by `SOLEFOOT_IDENTIFIED_CFG` (`environments/environments/assets/config/solefoot_identified_cfg.py:9`, `solefoot_identified_cfg.py:100-120`), and every SF task the parent runner serves spawns it, the plain, HIM and co-optimisation tasks alike (the census of section 5.1). It carries fifteen bodies and eight hinge joints, a base box and per leg an abduction housing welded to the base, the abduction joint in the hip housing, the hip pitch joint in the knee housing, a thigh welded to it, the knee joint in the shank, an ankle housing welded to the shank and the ankle joint in the foot (`base_robot.xml:44-91`). Names follow `<kind>_<side>_<suffix>` with the side an R or an L between underscores.

The sign table was derived rather than read by eye. A script composes every body quaternion to the root, expresses each joint axis in the base frame, and compares the mirrored axis of each joint with its partner's (`context/artefacts/tron1_quadruped-mjcf-mirror-audit-2026-10-20/mjcf_mirror_audit.py`, output in `mjcf_mirror_audit.txt` of the same directory).

| Joint pair | Right axis | Left axis | Mirrored right axis | Sign | Limits |
|---|---|---|---|---|---|
| abad | (-1, 0, 0) | (1, 0, 0) | (1, 0, 0) | keep | identical, (-0.384, 1.396) |
| hip | (0, -1, 0) | (0, -1, 0) | (0, -1, 0) | keep | identical, (-1.396, 1.012) |
| knee | (0, -1, 0) | (0, -1, 0) | (0, -1, 0) | keep | identical, (-1.214, 1.173) |
| ankle | (0, -1, 0) | (0, -1, 0) | (0, -1, 0) | keep | identical, (-0.792, 1.309) |

The flip set of the TRON1 is empty. The textbook rule that roll joints flip would negate the abduction pair, and it would be wrong here, because the emitter already authors the two abduction axes as mirror images, so that a positive coordinate abducts either leg outward. The limits corroborate the result independently, since a kept joint must share its partner's limits and a flipped one their negation, and the abduction limits are asymmetric about zero, so the corroboration is decisive (`base_robot.xml:52`, `base_robot.xml:101`). The same script finds the body positions mirrored to zero error, and the masses and inertias of each pair are identical in the document, so the TRON1 is modelled exactly symmetric, unlike the identified SD_BRS1 (`/ws/plans/SYMMETRY_PLAN.md:27`). Its four actuator groups name both sides in one configuration and carry no side dependent term, the friction model being odd in velocity (`solefoot_identified_cfg.py:14-67`, `environments/environments/actuators/actuator_pd.py:27-39`).

### 4.2 The quadruped

The quadruped asset is `assets/mjcf/quadruped/quadruped.xml`, spawned by `QUADRUPED_IDENTIFIED_CFG` (`environments/environments/assets/config/quadruped_identified_cfg.py:9`). It carries twenty five bodies and twelve joints, a trunk and per leg an abduction housing, a hip housing holding the abduction joint, a knee housing holding the hip pitch joint, a thigh, a shank holding the knee joint, and a spherical foot (`quadruped.xml:40-206`). Each leg is named by a fore or rear letter and a side letter, FR, FL, RR and RL, so the partner rule must exchange only the second letter, and a rule that swapped every R would pair the rear right leg with the front left.

| Joint kind | Right axis | Left axis | Mirrored right axis | Sign | Limits |
|---|---|---|---|---|---|
| hip (abduction) | (1, 0, 0) | (1, 0, 0) | (-1, 0, 0) | flip | symmetric about zero, ±0.803, not identifying |
| thigh (hip pitch) | (0, 1, 0) | (0, 1, 0) | (0, 1, 0) | keep | identical, (-1.932, 3.304) |
| calf (knee) | (0, 1, 0) | (0, 1, 0) | (0, 1, 0) | keep | identical, (-0.928, 0.852) |

The quadruped flips its four abduction joints and keeps the rest, the pattern SymmLoco records for the CyberDog2 and Isaac Lab for ANYmal (`anymal.py:244-245`). The abduction limits are symmetric and so cannot corroborate the flip, which rests on the axes alone. The nominal pose sets the abduction to zero on every leg (`environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:1149-1156` relies on it, and `context/quadruped.md:25` records it), so the flip is consistent with the default pose but not decided by it.

The quadruped has no fore and aft mirror. Every leg carries the same thigh and shank rotations (`quadruped.xml:59`, `quadruped.xml:143`), so the front and rear knees bend the same way, and a fore and aft reflection would map a forward bent knee onto a backward bent one that the robot does not possess. Isaac Lab's ANYmal module applies a front back mirror because ANYmal's legs are arranged for it (`anymal.py:250-259`), and porting that transformation here would teach the policy a robot that does not exist. The group is C2, the augmentation factor two.

### 4.3 The other assets the runners reach

The quadruped's parent runner also serves `Isaac-Quadruped-Go1Asset-Blind-Rough-v0`, which swaps the asset for Isaac Lab's Unitree Go1 (`robots/__init__.py:577-585`). Its legs carry the same FR, FL, RR and RL prefixes, so the quadruped module's partner rule serves it unchanged, and Isaac Lab's Go1 configuration sets the abductions of the two sides to opposite signs, so the default pose check of section 7.2 decides its flip at the first update. That task fails during environment construction, in code this plan does not change, in the live tree as in the sandbox (section 11), so its mirror has not been exercised at runtime and is recorded as unvalidated in section 12.

The SD_BRS1 and KScale modules keep their partner rules and flip sets, which their own documents derived (`/ws/plans/SYMMETRY_PLAN.md:27`, `/ws/plans/SYMMETRY_PLAN.md:96`, `context/KScale.md` section 7). The reorganised module subjects both to the joint limit and default pose checks for the first time, and both pass (section 11). The SD_BRS1's default pose sets its hip pitch to 0.2379 on the right and -0.2379 on the left with every other pair equal, which decides its anomalous hip pitch flip independently of the axes (`context/BRS.md`, addition of 2026-10-20).

### 4.4 The runtime names and orders

The articulation's order is set by the importer, not by the document, and the augmentation reads it at runtime so that it is correct for any order. The sandbox runs of section 11 recorded the orders, which the plan states so that a later reader can check a dump against them. The quadruped resolves its joints depth major and right first, `FR_hip_joint, FL_hip_joint, RR_hip_joint, RL_hip_joint` then the four thighs and the four calves, giving the joint permutation [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10] with the first four signs negative, its bodies likewise depth major with the trunk first, and exactly one collision shape per body, twenty five in all. Its contact sensor lists the same bodies leg major instead, trunk then the six bodies of each leg in turn, which is why the module resolves the bodies of every sensor term from the sensor's own names. The TRON1 resolves its joints in the same depth major, right first order, `abad_R_Joint, abad_L_Joint` then the hip, knee and ankle pairs, giving the permutation [1, 0, 3, 2, 5, 4, 7, 6] with every sign positive, its fifteen bodies depth major with the base first, and one collision shape per body (`context/artefacts/quadruped_symmetry-2026-10-20_06-54-05-sandbox-smoke/smoke_probe.txt`, `context/artefacts/sf_tron_1a_symmetry-2026-10-20_06-56-00-sandbox-smoke/smoke_probe.txt`, both reproduced by the revised sandbox). The SD_BRS1 resolves its joints left first, `HipRollL, HipRollR` then the pitch, knee and ankle pairs, giving the permutation [1, 0, 3, 2, 5, 4, 7, 6, 9, 8] with the hip roll, hip pitch and ankle roll negative, and thirteen bodies of one collision shape each, which corrects the nineteen shapes its module's comment records (`brs.py:42-46`). The KScale resolves its twelve joints left first, `left_hip_pitch_04, right_hip_pitch_04` and then the roll, yaw, knee, foot pitch and foot roll pairs, giving the permutation [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10] with the hip pitch, hip roll, hip yaw and foot roll negative, and thirteen bodies carrying fifty seven collision shapes, nine on the torso and from one to eight elsewhere, equal within every pair, so that its material term is the first of the four robots' to be mirrored by shape blocks of unequal length. Its default pose decides three of its four flips, the hip pitch, the hip roll and the foot roll standing at opposite values across the sides (`context/KScale.md`, addition of 2026-10-20). The order is therefore a property of each asset and its importer, which is why it is read rather than assumed.

## 5. The environments and tasks under the mirror

### 5.1 The observation groups of every task

The request is that every term of every configuration the module may meet be mapped, so that a term is never tiled unmirrored by default. The terms were therefore enumerated from the configurations rather than by reading them, by instantiating the environment configuration of every training task of the four robot families, thirty three in all, and walking its observation groups (`context/artefacts/sf_quadruped_brs_kscale-obs-term-census-2026-10-20/dump_obs_terms.py`, output `obs_terms.json` of the same directory). The tasks fall into three layouts, which the table gives with the widths the sandbox recorded for the two robots of this plan.

| Layout | Groups, history | TRON1 widths | Quadruped widths |
|---|---|---|---|
| Plain (`ObservationsCfg`) | policy 10 flat, critic 10 flat, commands 1 | 360, 4790, 3 | 480, 6530, 3 |
| HIM (`HIMObservationsCfg`) | policy 1, obsHistory 10 kept as an axis, critic 10 flat, commands 1, estimatorGT 1, targetEnc 1 | 33, (10, 33), 4790, 3, 3, 36 | 45, (10, 45), 6530, 3, 3, 48 |
| Co-optimisation | policy 10 flat, critic 10 flat with link lengths and nominal pose, commands 1, morphologyObs 1, privilegedDynamicsObs 1, historyObs 25 flat, and for the TRON1 estimatorGT 1 | 360, 4890, 3, 160, 28, 5575, 3 | 480, 6690, 3, 216, 48, 5875 |

The thirty three tasks carry twenty three distinct term names among the four robots, and one table maps them all, whichever group carries them. The per frame widths are the TRON1's and the quadruped's.

| Term | Function, frame | Groups | TRON1 | Quadruped | Kind | Mirror |
|---|---|---|---|---|---|---|
| base_lin_vel | `base_lin_vel`, base | policy, critic, historyObs, obsHistory, targetEnc, estimatorGT | 3 | 3 | vec | multiply (1, -1, 1) |
| base_ang_vel | `base_ang_vel`, base | policy, critic, historyObs, obsHistory, targetEnc | 3 | 3 | pvec | multiply (-1, 1, -1) |
| proj_gravity | `projected_gravity`, base | the same | 3 | 3 | vec | multiply (1, -1, 1) |
| joint_pos, joint_vel, last_action | `joint_pos_rel`, `joint_vel_rel`, `last_action` | the same | 8 each | 12 each | joint | partner joint, signed by section 4 |
| velocity_commands | `generated_commands` | the same, and commands | 3 | 3 | cmd | multiply (1, -1, -1) |
| gait_phase | `get_gait_phase` | SD_BRS1 policy and critic | | | gait | negate both channels |
| heights | `height_scan` | critic, historyObs | 187 | 187 | heights | lateral reflection of the ray pattern |
| robot_joint_torque, robot_joint_acc | `robot_joint_torque`, `robot_joint_acc` | critic, privilegedDynamicsObs | 8 each | 12 each | joint | partner joint, signed |
| robot_joint_pos | `robot_joint_pos`, the default positions | critic | 8 | 12 | joint | partner joint, signed |
| robot_joint_stiffness, robot_joint_damping | gains | critic | 8 each | 12 each | joint_nosign | partner joint |
| robot_mass | `robot_mass` | critic, morphologyObs over its body subset | 15 | 25 | body_scalar | partner body |
| robot_inertia | `robot_inertia` | the same | 135 | 225 | body_inertia | partner body, products coupling y negated |
| nominal_pose | `robot_nominal_pose`, the fold about y | co-optimisation critic, morphologyObs | 6 | 8 | body_scalar | partner body |
| link_lengths | `robot_link_lengths` | co-optimisation critic, morphologyObs | 4 | 8 | body_pair_scalar | partner pair |
| robot_material_properties | `robot_material_properties` | critic | 45 | 75 | material | partner body's shapes |
| feet_lin_vel | `feet_lin_vel`, world | critic, privilegedDynamicsObs | 6 | 12 | body_vec_w | exchange the feet |
| feet_contact_force | `robot_contact_force`, world | the same | 6 | 12 | body_vec_w | exchange the feet |
| robot_pos | `robot_pos`, world | critic | 3 | 3 | identity | exact, the root lies on the mirror plane |
| robot_vel | `robot_vel`, world | critic | 6 | 6 | root_vel_w | world yaw rate negated |

Two tasks fall outside the four families' runners and are not mapped, the TRON1's Berkeley imitation tasks, which keep their own runner (`agents/limx_rsl_rl_ppo_cfg.py:248-282`), and the native Go1 task, which keeps Isaac Lab's (`robots/__init__.py:55`). Their terms named `actions`, `projected_gravity` and `height_scan` therefore never reach the table, and a later runner that wires the module to them meets a `KeyError` naming the term rather than a silent identity.

The world frame terms deserve their derivation, since section 3.3 changes their treatment. A world frame vector v of the foot k maps in the mirrored state to M_w v, the reflection about the vertical plane through the base heading, carried by the partner foot. That plane is vertical up to the base roll, so M_w preserves the vertical component and the component along the heading and negates the lateral one, and only the last requires the base yaw, which no stored observation provides (`/ws/plans/SYMMETRY_PLAN.md:165`). Exchanging the feet therefore reproduces two of the three components exactly, among them the vertical component of the contact force that dominates it, where the identity of the existing modules reproduces none, since it gives each foot its own vector. The root position lies on the mirror plane and is invariant exactly. The world angular velocity maps to -M_w w, so its vertical component, the yaw rate, negates exactly, and the two horizontal components cannot be resolved. The kind `body_vec_w` exchanges the bodies of a world frame term without changing a sign, and `root_vel_w` negates the world yaw rate alone.

The bodies of every per body term are resolved from the term's own configuration rather than from the robot, the articulation's names for an `asset_cfg` and the contact sensor's names for a `sensor_cfg`. This matters because the contact sensor orders its bodies leg major while the articulation orders them depth major (section 4.4), so a permutation built from the robot's order would exchange the wrong forces, and because the co-optimisation morphology reads the mass and inertia of a body subset rather than of every body.

### 5.2 The rough terrain

Both height scanners are a 1.6 by 1.0 m grid at 0.1 m, aligned with the base yaw and cast from twenty metres above the base (`cfg/SF/limx_base_env_cfg.py:75-82`, `cfg/quadruped/base_env_cfg.py:75-82`), which the sandbox confirmed as 187 rays in the "xy" ordering. The observation is the base height above each ray's hit, less an offset, per ray in pattern order (`/ws/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py:292-300`). Because the pattern turns with the base yaw, the mirrored scan is the scan of the reflected terrain seen from the reflected base, and it is the same set of rays reindexed, the ray at (x, y) taking the value of the ray at (x, -y). The grid's lateral coordinates are symmetric about zero by construction (`/ws/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns.py:45-47`), so the reindexing is exact. The module computes it from `ray_starts` (`/ws/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py:218-226`) and raises for a world aligned pattern or an asymmetric one.

The augmentation is valid only if the mirrored terrain is as probable as the original. Both generators compose a flat share, inverted and upright pyramid slopes and stairs, waves and uniform random roughness (`cfg/SF/terrains_cfg.py:205-255`, `cfg/quadruped/terrains_cfg.py:16-69`). The pyramids are symmetric about both axes, the waves are a symmetric sinusoid field, and the random roughness is independent per cell, so each tile's distribution is reflection invariant, and since the base yaw is reset uniformly over the full circle (`cfg/SF/limx_base_env_cfg.py:847`, `cfg/quadruped/base_env_cfg.py:1054`) so is the distribution of terrain seen in the base frame. The terrain level curriculum promotes on distance walked and is invariant (`cfg/SF/limx_base_env_cfg.py:1111-1117`, `cfg/quadruped/base_env_cfg.py:1298-1304`).

Neither actor observes the terrain. Its only terrain signal is through proprioception and the critic, so the height scan mirror acts on the value function alone, which for the quadruped's blind actor on stairs is where the asymmetric information of the terrain enters the advantage (`/ws/plans/QUADRUPED_DEGENERATE_GAIT.md:182`).

### 5.3 Invariance of the MDP

The augmentation is valid only for the parts of the MDP that are invariant, and the audit found no term that is not.

| Component | TRON1 | Quadruped | Invariance |
|---|---|---|---|
| Velocity command | lin y ±0.5, yaw ±0.75, heading ±π (`limx_base_env_cfg.py:113-127`) | lin y ±0.1, yaw ±0.75 (`base_env_cfg.py:113-127`) | every range symmetric about zero |
| Reset | joint offsets ±0.2, base velocities symmetric | the same | symmetric ranges |
| Randomisation | base mass, gains per joint group, default offsets per joint, friction per body | the same, plus every body's mass | independent per joint or body with a shared range, so invariant in distribution, the per environment asymmetry being mirrored in the critic |
| Push | x and y velocity symmetric (`limx_base_env_cfg.py:884-891`) | symmetric (`base_env_cfg.py:1077-1084`) | symmetric |
| Tracking rewards | `track_lin_vel_xy_exp`, `track_ang_vel_z_exp` | the same | errors of mirrored quantities have equal norm |
| Regularisers | torque, acceleration, rate, limits, velocity, smoothness, orientation, base height | the same, plus abduction deviation | sums over joints, norms of mirrored vectors, mean height of the scan, the abduction deviation an absolute value about a zero default |
| Foot terms | air time, slide, no fly, ankle pitch, feet distance, feet regulation | air time v2, hold, slide, clearance v4, landing v3, impact | sums or norms over both feet, feet distance the norm of their difference (`mdp/rewards.py:1207-1257`) |
| Terminations | base, thigh and shank contact | trunk, thigh and shank contact, and a termination penalty | body sets closed under the partner rule, the penalty a constant |
| Actuators | identical per side, friction odd in velocity | the same | symmetric |
| Co-optimisation designs | thigh and shank length scales shared by both legs (`scripts/rsl_rl/train.py:261-264`) | the same | every design symmetric |

The default offsets per joint are drawn independently for each joint (`cfg/SF/limx_base_env_cfg.py:807-808`, `mdp/events.py:247-286`), so one environment's defaults are asymmetric while their distribution is not. This is why the module mirrors the per environment `robot_joint_pos` term as a joint quantity, and why its own default pose check of section 7.2 reads the configured pose rather than the randomised buffer, a distinction the sandbox forced when the buffer of environment 0 failed the check by 0.048 rad on the TRON1's hip.

### 5.4 The co-optimisation groups

The co-optimisation tasks enable four further groups, the morphology, the privileged dynamics, the twenty five frame history with the height scan, and on the TRON1 the base velocity target, and the critic adds the link lengths and the nominal pose (`cfg/SF/limx_base_env_cfg.py:1418-1425`, `cfg/quadruped/base_env_cfg.py:671-676`). The estimator of both robots' first experiment now reads the policy group, with its ten frame history, and regresses onto the morphology (`agents/limx_rsl_rl_ppo_cfg.py:161-166`, `agents/quadruped_rsl_rl_ppo_cfg.py:59-64`), the TRON1's other three experiments reading `historyObs` with the privileged dynamics or the morphology (`limx_rsl_rl_ppo_cfg.py:169-205`). Since CoptPPO computes the estimator loss on the augmented batch against a target drawn from the same batch (`copt_ppo.py:256-259`), every one of these groups must be mirrored, and the module mirrors them, the link lengths by exchanging the pairs whose parent and child are partners, the morphology over its own body subset, and the nominal pose, a fold about the lateral axis that a reflection preserves, by exchanging partner bodies. `historyObs` is to be retired, and is mirrored for as long as a configuration carries it. Because every design varies the two legs alike (section 5.3), the mirrored morphology is the morphology of the same design, and the estimator is taught nothing a design could contradict.

## 6. The agents and the mirror loss

### 6.1 The two parent runners carry it

The request is that every task of the two robots inherit the symmetry, plain, HIM and co-optimisation alike, without a new task or a new runner. The two parent runner classes reach every one of them. `SF_TRON1AFlatPPORunnerCfg` (`agents/limx_rsl_rl_ppo_cfg.py:96-138`) is instantiated twice, as `limx_sf_blind_flat_runner_cfg`, which serves eighteen registrations, flat, rough and identified, the identified HIM tasks among them, and as `limx_sf_him_blind_flat_runner_cfg`, which serves four HIM registrations (`robots/__init__.py:41-43`), and its subclass `SFCoptBaseRunnerCfg` and the four experiment classes below it inherit its algorithm (`limx_rsl_rl_ppo_cfg.py:142-205`), the first instantiated for three co-optimisation registrations (`robots/__init__.py:47`). `PFQuadrupedPPORunnerCfg` (`agents/quadruped_rsl_rl_ppo_cfg.py:14-48`) is instantiated as `quadruped_runner_cfg` for the flat, rough and Go1 asset tasks and as `quadruped_him_runner_cfg` for the HIM tasks, and its subclass `PFQuadrupedCoptPPORunnerCfg` serves the three co-optimisation registrations (`robots/__init__.py:49-53`, `quadruped_rsl_rl_ppo_cfg.py:55-71`). Setting `symmetry_cfg` in the algorithm of each parent therefore reaches twenty five TRON1 and twelve quadruped registrations through inheritance alone, since a configuration class subclass inherits the parent's default and `configclass` deep copies it per instance (`/ws/IsaacLab/source/isaaclab/isaaclab/utils/configclass.py:380-381`, `configclass.py:384-402`). No other runner is touched, so the point foot and wheel foot TRON1, the Berkeley imitation, the native Go1, the SD_BRS1 and the KScale runners keep exactly their present configuration. The field defaults to None in Isaac Lab (`/ws/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:128`), which is why neither parent has carried it until now.

### 6.2 The coefficient and the control

Both parents set the augmentation and the mirror loss, which the request names, and log the loss, which rsl_rl computes whenever a symmetry configuration is set (`ppo.py:316-346`). No published source states a mirror loss weight on this interface [4], and the weights of the foundational works were set against different action scalings [3][2], so the coefficient is calibrated against this workspace's own measurement. The SD_BRS1 logged a symmetry loss of 0.061 under augmentation against a surrogate of 0.010 (`/ws/context/brs_gait.md:341`), so a weight of 0.5 makes the loss term about three times the surrogate, large enough to act and an order below the value loss of 7.94 in the same record. The quadruped's sandbox run logged 0.019 at its first iteration and the TRON1's 0.006 (section 11), so the term starts small and grows only as an asymmetry develops. The same weight is used on both robots, so that they differ in the robot alone.

The control is the pair of baselines in flight on 2026-10-20, `sf_tron_1a_flat/2026-10-20_05-24-20` and `quadruped_flat/2026-10-20_05-09-02`, which were launched before the change and train without symmetry. The ablation of augmentation alone against the combination [4] requires no new task either. Setting `use_mirror_loss=False` in a parent for one launch keeps the loss logged and out of the gradient, and the run's own `params/agent.yaml` records the choice, so the comparison is made by the dumped configuration rather than by the task identifier.

### 6.3 The policy types

`--policy-type` selects the algorithm and the runner for the same task (`scripts/rsl_rl/cli_args.py:60-68`, `scripts/rsl_rl/train.py:225-246`). The default PPO builds the stock runner and algorithm, and the timing instrumented runner of the debug types subclasses it (`rsl_rl/rsl_rl_debug/runners/debug_on_policy_runner.py:23`). HIMPPO resolves the configuration through its own runner (`himloco/himloco/runners/him_on_policy_runner.py:69`) and needs the successor batch mirrored (section 3.3, item 7). COPT builds `CoptPPO` through `CoptOnPolicyRunner`, which resolves the configuration (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:502`) and needs the dump of `train.py` to receive an untouched dictionary (section 3.3, item 8). `play.py` builds the same three runners (`scripts/rsl_rl/play.py:1107-1147`) and never calls the function, since it never updates, and the configuration adds no parameter to a checkpoint, so a checkpoint trained before the change loads after it (section 11).

### 6.4 The cost

The augmentation doubles every mini-batch in the update, not the rollout. With four mini-batches the TRON1's 4096 environments over 25 steps give 25600 rows, augmented to 51200 of 5153 floats, about 1.06 GB, and the quadruped's 6000 environments give 75000 augmented rows of 7013 floats, about 2.1 GB, each with transient copies of similar size for the repeat and the mirrored half. The in flight baselines hold 37.0 and 18.9 GB of 49.1 GB on their cards, so both fit. The HIM layout adds the ten frame history and the successor batch, and the co-optimisation layout its four groups, both smaller than the critic. The learning phase should take roughly twice as long, and the collection phase is unchanged. Neither parent normalises its observations, so the question of running statistics computed on unmirrored data does not arise for the TRON1 or the quadruped.

## 7. The architecture of the reorganised module

### 7.1 The structure

```text
 mdp/symmetry/
 |
 +-- common.py   MirrorSpec            partner rules, flip set, optional kind table, limit checks
 |               partner rules         suffix_side_partner, regex_side_partner
 |               TERM_KINDS            one table of every term of the four robots' tasks
 |               builders              build_perm, build_joint_map, build_material_perm,
 |                                     count_shapes_per_body, ray_mirror_perm,
 |                                     check_joint_limits, check_default_pose
 |               compile_maps          every group: term layout x kind x frame map -> (index, sign)
 |               augment               obs.repeat(2), obs[g][n:] = obs[g][..., index] * sign, actions
 |
 +-- brs.py        SPEC (suffix rule, HipRoll AnkleRoll HipPitch)            -> augment
 +-- kscale.py     SPEC (prefix rule, body dictionary, four flips)           -> augment
 +-- tron1.py      SPEC (regex _R_ / _L_, no flips)                          -> augment
 +-- quadruped.py  SPEC (regex FR/FL RR/RL, abduction flips)                 -> augment
 +-- __init__.py   prefixed aliases of every compute_symmetric_states

 compile, once per device, at the first update:
   robot names --partner rule--> joint perm, sign, checked against limits and default pose
   observation manager --> every group [(term, dims, history, cfg)], unmapped term raises
   for each term:  kind --> frame map (index_f, sign_f)
                   flattened (h*w,): tile over h frames     kept axis (h, w): one frame on the last axis
   concatenate blocks --> (index, sign) over the group's last axis, involution asserted
```

### 7.2 The design

A robot is described by a frozen `MirrorSpec`, its two partner rules, its flip set and a switch for the joint checks, with an optional kind table that defaults to `TERM_KINDS`. The robot module holds the spec and a one line `compute_symmetric_states`, which must stay a top level function so that it survives the serialisation of the agent configuration to a `module:function` string and back (`/ws/context/rsl_rl.md`, the consumer contract, item 2), and the sandbox runs confirmed that every function was dumped in that form. Nothing the first pass's switches held fixed survives as a constant, the history, the height grid and the shape counts all being read from the simulation, so none can fall out of date with an asset or a configuration.

Every kind compiles to a signed permutation of one frame (section 2.1), so a group compiles to one index and one sign vector over its last axis, and its mirror is one gather and one multiplication, `x[..., index] * sign`, which serves a flattened group and a group that keeps its history as an axis alike. The compiler asserts that the result is an involution, which a wrong partner rule, a missing partner or a sign set that disagrees between partners all violate, and it raises on any term of any group without a kind. Two checks guard the flip set. A kept joint must share its partner's limits and a flipped one their negation, which decides the TRON1's empty set, whose abduction limits are asymmetric. And the configured default pose must be its own mirror, which decides a flip where the limits are symmetric, as for the SD_BRS1's hip pitch and the Go1's abduction. The pose is read from the articulation's configuration rather than from `default_joint_pos`, which the randomisation of section 5.3 perturbs per environment.

### 7.3 Extending it to another robot

A new robot requires a partner rule and a flip set. The partner rule for a hexapod named LF, LM, LH, RF, RM and RH is `regex_side_partner(r"(?:^|_)(?P<side>[LR])[FMH]_")`, and a humanoid's arms and waist need no new machinery, a waist yaw or roll joint lying on the plane being its own partner and flipping its sign, a waist pitch keeping it. The flip set follows from the joint axes by the rule of section 2.1, which the audit script of section 4 computes for any MuJoCo document, and the two checks confirm it at the first update. A new observation term needs an entry in `TERM_KINDS`, without which the first update raises naming it, and a term whose geometric type is none of the existing ones needs a new frame map in `_frame_map`, which is the only place a kind is interpreted.

One extension is deliberately not built. The augmentation factor is two, the sagittal reflection alone. A robot with a second symmetry, the fore and aft mirror of an ANYmal or the rotations of a hexapod, would need the spec to carry a list of group elements and `augment` to tile by their number, which rsl_rl supports since it reads the factor from the returned batch (`ppo.py:241`).

### 7.4 Considerations that governed the reorganisation

The refactor keeps the SD_BRS1 and KScale outputs where their correctness was not in question, and changes them where the review found them wrong. Their mirrored actions and policy observations are bitwise identical to commit `23513d2` in every call form, which the test suite asserts, since every floating point operation of the old modules is a multiplication by plus or minus one or a gather and the compiled form performs the same multiplications. Their critic differs in exactly the four terms of section 3.3, items 3 and 4, and their `commands` group, formerly tiled, is now mirrored, which no network of their tasks reads. Each keeps its docstring, its flip set and its partner rules, the SD_BRS1's unchecked rule now checked, which it passes. The module reads three private members of Isaac Lab, the observation manager's term configurations and concatenation axis, the articulation's simulation view and its link paths, the last two exactly as Isaac Lab's own material randomisation does (`events.py:205-221`), and a change of version that renames them would raise at the first update rather than corrupt the batch.

## 8. Implementation

The edits are ten, applied in the order of section 9.1. Six files of `environments/environments/tasks/locomotion/mdp/symmetry/` are written in full, the shared module, the two rebuilt modules, the two new ones, and the package initialiser. The two agent configurations, HIMPPO and `train.py` receive the diffs of sections 8.7 to 8.9, and a test suite is added at `environments/tests/symmetry_test.py`. No environment configuration and no task registration changes. Every block below is the file or the diff that was tested in the sandbox, verbatim, save for the probe of section 11, which was a sandbox addition and is not part of the plan.

### 8.1 `mdp/symmetry/common.py`, new

The shared machinery, every element of which section 7 describes, with the single kind table `TERM_KINDS` of section 5.1.

```python
# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Shared machinery of the sagittal (left-right) symmetry data augmentation.

A robot module declares a :class:`MirrorSpec` and implements ``compute_symmetric_states`` as a
call to :func:`augment`. The function must stay a top level member of the robot module, since
``class_to_dict`` serialises it to a ``module:function`` string that the PPO constructor imports
again, and a closure or a lambda cannot survive that round trip.

Every mirror kind is a signed permutation of the coordinates of one observation frame, so each
observation group compiles once into an index and a sign vector over its last axis, and is
mirrored by one gather, ``x[..., index] * sign``. The reflection is an involution, so the compiled
pair satisfies ``index[index] == arange`` and ``sign * sign[index] == 1``, which the compiler
asserts. Every group the observation manager carries is mirrored, and a term absent from the
kind table raises, so no term is ever tiled unmirrored without notice.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

# fixed per-frame sign vectors
VEC = (1.0, -1.0, 1.0)       # polar vector, flip lateral y
PVEC = (-1.0, 1.0, -1.0)     # pseudovector, flip x and z
CMD = (1.0, -1.0, -1.0)      # velocity command, flip lin_y and ang_z
# gait phase is a (sin, cos) pair of ONE shared clock with the feet in antiphase at an offset of
# 0.5, so exchanging the feet is a half-cycle shift and both channels negate together
GAIT = (-1.0, -1.0)
INERTIA9 = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0)  # flip y-coupling products
# world frame root velocity [v, w]. The mirror plane is vertical up to roll, so the yaw rate is
# the one rotational component whose image is known without the base yaw, and it negates
ROOT_VEL_W = (1.0, 1.0, 1.0, 1.0, 1.0, -1.0)

_CONSTANT_KINDS = {"vec": VEC, "pvec": PVEC, "cmd": CMD, "gait": GAIT, "root_vel_w": ROOT_VEL_W}

# mirror kind of every observation term of the SF, quadruped, SD_BRS1 and kscale tasks, whichever
# group carries it, for the plain, HIM and co-optimisation configurations alike. A world frame
# foot vector maps to M_w applied to the partner's, M_w the reflection about the vertical heading
# plane, so exchanging the feet is exact on the vertical and forward components and wrong only
# in the sign of the lateral one, which needs the base yaw no observation carries. The root
# position lies on the mirror plane, so identity is exact for it.
TERM_KINDS = {
    "base_lin_vel": "vec", "base_ang_vel": "pvec", "proj_gravity": "vec",
    "joint_pos": "joint", "joint_vel": "joint", "last_action": "joint",
    "velocity_commands": "cmd", "gait_phase": "gait", "heights": "heights",
    "robot_joint_torque": "joint", "robot_joint_acc": "joint", "robot_joint_pos": "joint",
    "robot_joint_stiffness": "joint_nosign", "robot_joint_damping": "joint_nosign",
    "robot_mass": "body_scalar", "robot_inertia": "body_inertia", "nominal_pose": "body_scalar",
    "link_lengths": "body_pair_scalar", "robot_material_properties": "material",
    "feet_lin_vel": "body_vec_w", "feet_contact_force": "body_vec_w",
    "robot_pos": "identity", "robot_vel": "root_vel_w",
}

_SWAP = {"L": "R", "R": "L", "l": "r", "r": "l", "left": "right", "right": "left"}


@dataclass(frozen=True)
class MirrorSpec:
    """The sagittal mirror of one robot.

    Args:
        joint_partner: Maps a joint name to its mirror partner, itself for a joint on the plane.
        body_partner: The same for body names.
        flip_joint_kinds: Name substrings of the joints whose coordinate negates under the mirror,
            those whose partner's world axis is the negation of the mirrored axis.
        term_kinds: The mirror kind of each observation term, by term name.
        check_joint_limits: Raise where the joint limits or the default pose contradict the derived sign.
    """

    joint_partner: Callable[[str], str]
    body_partner: Callable[[str], str]
    flip_joint_kinds: tuple[str, ...]
    term_kinds: dict[str, str] | None = None
    check_joint_limits: bool = False


def suffix_side_partner(name: str) -> str:
    """Swap a trailing L for R, the SD_BRS1 rule. A name ending in neither lies on the plane."""
    return name[:-1] + "L" if name.endswith("R") else name[:-1] + "R" if name.endswith("L") else name


def regex_side_partner(pattern: str) -> Callable[[str], str]:
    """A partner rule swapping the named group ``side`` of ``pattern``, the name itself where it does not match."""
    rx = re.compile(pattern)

    def partner(name: str) -> str:
        match = rx.search(name)
        if match is None:
            return name
        start, end = match.span("side")
        return name[:start] + _SWAP[name[start:end]] + name[end:]

    return partner


def _assert_involution(perm, names, what):
    for i, p in enumerate(perm):
        assert perm[p] == i, f"{what} permutation is not an involution at {names[i]!r}"


def build_perm(names, partner, what="body"):
    """perm[i] is the index of the partner of names[i]."""
    perm = []
    for name in names:
        p = partner(name)
        if p not in names:
            raise ValueError(f"mirror partner {p!r} of {what} {name!r} is not in the articulation")
        perm.append(names.index(p))
    _assert_involution(perm, names, what)
    return perm


def build_joint_map(names, partner, flip_kinds, device):
    """out[i] = sign[i] * x[perm[i]]."""
    perm = build_perm(names, partner, "joint")
    sign = [-1.0 if any(k in name for k in flip_kinds) else 1.0 for name in names]
    return (torch.tensor(perm, dtype=torch.long, device=device),
            torch.tensor(sign, dtype=torch.float, device=device))


def build_material_perm(num_shapes_per_body, body_perm):
    counts = list(num_shapes_per_body)
    for b, pb in enumerate(body_perm):
        assert counts[b] == counts[pb], "partner bodies must carry equal collision-shape counts"
    starts, acc = [], 0
    for c in counts:
        starts.append(acc)
        acc += c
    perm = []
    for b in range(len(counts)):
        src = body_perm[b]
        perm.extend(range(starts[src], starts[src] + counts[src]))
    return perm


def count_shapes_per_body(robot) -> list[int]:
    """Collision shapes per body in body order, by the loop of randomize_rigid_body_material.__init__."""
    view = robot.root_physx_view
    counts = [robot._physics_sim_view.create_rigid_body_view(path).max_shapes for path in view.link_paths[0]]
    if sum(counts) != view.max_shapes:
        raise ValueError(f"parsed {sum(counts)} collision shapes against the {view.max_shapes} of the articulation")
    return counts


def ray_mirror_perm(sensor, tol: float = 1e-5) -> list[int]:
    """perm[i] is the ray at the lateral reflection of ray i, read from the pattern itself."""
    if sensor.cfg.ray_alignment == "world":
        raise ValueError("a world aligned ray pattern does not turn with the base, so it has no fixed mirror")
    xy = sensor.ray_starts[0, :, :2].detach().double().cpu()
    dist = torch.cdist(xy * xy.new_tensor([1.0, -1.0]), xy)
    err, perm = dist.min(dim=1)
    if err.max() > tol:
        raise ValueError(f"the ray pattern is not symmetric about its x axis, worst mismatch {err.max():.2e} m")
    return perm.tolist()


def check_joint_limits(robot, jperm, jsign, tol: float = 1e-4):
    """A kept joint shares its partner's limits and a flipped one their negation. Symmetric limits identify neither."""
    lim = robot.data.joint_pos_limits[0].detach().cpu()
    names = list(robot.joint_names)
    for i, (p, s) in enumerate(zip(jperm.tolist(), jsign.tolist())):
        lo, hi = lim[i].tolist()
        if abs(lo + hi) < tol:
            continue
        expect = (lo, hi) if s > 0 else (-hi, -lo)
        got = tuple(lim[p].tolist())
        if max(abs(got[0] - expect[0]), abs(got[1] - expect[1])) > tol:
            raise ValueError(
                f"joint {names[i]!r} limits {(lo, hi)} with sign {s:+.0f} predict {expect} for partner "
                f"{names[p]!r}, which has {got}, so the flip set is wrong for this asset"
            )


def check_default_pose(robot, jperm, jsign, tol: float = 1e-4):
    """The configured default pose of a symmetric robot is its own mirror, which decides a flip that symmetric
    limits cannot. The configuration is read rather than ``default_joint_pos``, which an event may randomise."""
    pose = robot.cfg.init_state.joint_pos
    default = torch.tensor([next((v for k, v in pose.items() if re.fullmatch(k, n)), 0.0) for n in robot.joint_names])
    err = (default[jperm.cpu()] * jsign.cpu() - default).abs()
    if err.max() > tol:
        i = int(err.argmax())
        raise ValueError(
            f"the default pose is not its own mirror at joint {list(robot.joint_names)[i]!r}, error {err[i]:.3e}, "
            "so the flip set is wrong for this asset or its default pose is asymmetric"
        )


def _term_bodies(env, robot, term_cfg):
    """The body names a term reports, in the order it reports them."""
    params = term_cfg.params or {}
    if "sensor_cfg" in params:
        cfg = params["sensor_cfg"]
        names = list(env.scene.sensors[cfg.name].body_names)
    else:
        cfg = params.get("asset_cfg")
        names = list(robot.body_names)
    ids = slice(None) if cfg is None else cfg.body_ids
    return [names[i] for i in range(len(names))[ids]] if isinstance(ids, slice) else [names[i] for i in ids]


def _frame_map(kind, w, term_cfg, ctx):
    """(index, sign) of length w for one frame of a term."""
    if kind == "identity":
        return list(range(w)), [1.0] * w
    if kind in _CONSTANT_KINDS:
        sign = list(_CONSTANT_KINDS[kind])
        assert w == len(sign), f"{kind} expects a frame of {len(sign)}, got {w}"
        return list(range(w)), sign
    if kind in ("joint", "joint_nosign"):
        jperm, jsign = ctx["jperm"].tolist(), ctx["jsign"].tolist()
        assert w == len(jperm), f"{kind} expects {len(jperm)} joints, got {w}"
        return jperm, jsign if kind == "joint" else [1.0] * w
    if kind in ("body_scalar", "body_inertia", "body_vec_w"):
        names = _term_bodies(ctx["env"], ctx["robot"], term_cfg)
        perm = build_perm(names, ctx["spec"].body_partner)
        block, sign = {"body_scalar": (1, (1.0,)), "body_inertia": (9, INERTIA9), "body_vec_w": (3, (1.0,) * 3)}[kind]
        assert w == block * len(names), f"{kind} expects {block} x {len(names)}, got {w}"
        return [block * p + k for p in perm for k in range(block)], list(sign) * len(names)
    if kind == "body_pair_scalar":
        partner = ctx["spec"].body_partner
        pairs = list(zip(term_cfg.params["parent_body_names"], term_cfg.params["child_body_names"]))
        assert w == len(pairs), f"{kind} expects {len(pairs)} pairs, got {w}"
        return [pairs.index((partner(a), partner(b))) for a, b in pairs], [1.0] * w
    if kind == "heights":
        perm = ray_mirror_perm(ctx["env"].scene.sensors[term_cfg.params["sensor_cfg"].name])
        assert w == len(perm), f"heights expects {len(perm)} rays, got {w}"
        return perm, [1.0] * w
    if kind == "material":
        if ctx["mat_perm"] is None:
            ctx["mat_perm"] = build_material_perm(count_shapes_per_body(ctx["robot"]), ctx["body_perm"])
        assert w == 3 * len(ctx["mat_perm"]), f"material expects 3 x {len(ctx['mat_perm'])}, got {w}"
        return [3 * p + k for p in ctx["mat_perm"] for k in range(3)], [1.0] * w
    raise ValueError(f"unknown mirror kind {kind}")


def _compile_group(om, group, kinds, ctx):
    """(index, sign) over the last axis of the group, its terms concatenated along that axis.

    A flattened term of h frames is h blocks of one frame each, time major. A term that keeps
    its history as a separate axis, (h, w), is mirrored frame by frame along the last axis alone.
    """
    names = list(om.active_terms[group])
    missing = [t for t in names if t not in kinds]
    if missing:
        raise KeyError(f"group {group!r} carries terms with no mirror kind, {missing}")
    if not om.group_obs_concatenate[group] or om._group_obs_concatenate_dim[group] != -1:
        raise ValueError(f"group {group!r} is not concatenated along its last axis")
    index, sign, start = [], [], 0
    for name, dim, cfg in zip(names, om.group_obs_term_dim[group], om._group_obs_term_cfgs[group]):
        width = int(dim[-1])
        frames = max(cfg.history_length, 1) if len(dim) == 1 else 1
        assert width % frames == 0, f"term {name!r} of width {width} is not {frames} frames"
        w = width // frames
        fidx, fsgn = _frame_map(kinds[name], w, cfg, ctx)
        index.extend(start + f * w + i for f in range(frames) for i in fidx)
        sign.extend(fsgn * frames)
        start += width
    _assert_involution(index, [f"{group}[{i}]" for i in range(len(index))], "observation")
    assert all(sign[i] * sign[p] == 1.0 for i, p in enumerate(index)), f"{group} signs are not an involution"
    return index, sign


def compile_maps(env: ManagerBasedRLEnv, spec: MirrorSpec, device) -> dict:
    """Build every permutation of the robot and compile every observation group into (index, sign)."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    om = unwrapped.observation_manager
    jperm, jsign = build_joint_map(list(robot.joint_names), spec.joint_partner, spec.flip_joint_kinds, device)
    if spec.check_joint_limits:
        check_joint_limits(robot, jperm, jsign)
        check_default_pose(robot, jperm, jsign)
    ctx = {
        "env": unwrapped, "robot": robot, "spec": spec, "jperm": jperm, "jsign": jsign,
        "body_perm": build_perm(list(robot.body_names), spec.body_partner), "mat_perm": None,
    }
    kinds = TERM_KINDS if spec.term_kinds is None else spec.term_kinds
    groups = {}
    for group in om.active_terms:
        index, sign = _compile_group(om, group, kinds, ctx)
        groups[group] = (torch.tensor(index, dtype=torch.long, device=device),
                         torch.tensor(sign, dtype=torch.float, device=device))
    print(f"[symmetry] mirrored groups {list(groups)}")
    return {"jperm": jperm, "jsign": jsign, "groups": groups}


@torch.no_grad()
def augment(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
    spec: MirrorSpec,
    cache: dict,
):
    """Augment obs and actions with the sagittal mirror, originals in the first half."""
    device = obs["policy"].device if obs is not None else actions.device
    if device not in cache:
        cache[device] = compile_maps(env, spec, device)
    m = cache[device]

    if obs is not None:
        n = obs.batch_size[0]
        obs_aug = obs.repeat(2)  # tiles every group, originals in [:n]
        for group in obs.keys():
            index, sign = m["groups"][group]
            obs_aug[group][n:] = obs[group][..., index] * sign
    else:
        obs_aug = None

    actions_aug = None if actions is None else torch.cat([actions, actions[:, m["jperm"]] * m["jsign"]], dim=0)
    return obs_aug, actions_aug
```

### 8.2 `mdp/symmetry/brs.py`, rewritten

The SD_BRS1 module keeps its flip set and its partner rule, `suffix_side_partner`, which is the expression of `brs.py:83` and `brs.py:94` unchanged, and its docstring save for the paragraph on what is mirrored. The history, the grid and the shape count constants are removed, all three now being read from the simulation, and the joint checks are enabled.

```python
# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the SD_BRS1 biped.

Robot-specific module. Implements ``compute_symmetric_states`` in the signature the
rsl_rl PPO symmetry hook expects (``/ws/rsl_rl/rsl_rl/algorithms/ppo.py``) and referenced
through ``isaaclab_rl.rsl_rl.RslRlSymmetryCfg``. A biped has one sagittal mirror, so the
augmentation doubles the batch, originals in the first half, mirrored copies in the second.

Every observation group is mirrored, the critic included, so the value function is biased
toward the sagittal invariance the theory requires. The term layout, the history of each
group, the height scan pattern and the collision shapes are read at runtime, and the joint and
body maps are built from the runtime names, so the module is correct regardless of whether
Isaac Sim resolves the sibling links left-first or right-first. The machinery is in :mod:`.common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

# joints whose sign flips under the sagittal mirror, by name substring. HipPitch flips
# despite being a pitch joint because its left and right URDF axes are opposite.
_FLIP_JOINT_KINDS = ("HipRoll", "AnkleRoll", "HipPitch")

SPEC = common.MirrorSpec(
    joint_partner=common.suffix_side_partner,
    body_partner=common.suffix_side_partner,
    flip_joint_kinds=_FLIP_JOINT_KINDS,
    check_joint_limits=True,
)

# module-level cache, one entry per (device); env is a singleton
_CACHE: dict = {}


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: "torch.Tensor | None" = None,
):
    """Augment obs and actions with the left-right sagittal mirror, originals first."""
    return common.augment(env, obs, actions, SPEC, _CACHE)
```

### 8.3 `mdp/symmetry/kscale.py`, rewritten

The KScale module keeps its docstring save for the paragraph on what is mirrored, its flip set, its body dictionary and its two raising partner rules, the second now a function over the dictionary rather than a loop, with the error messages of `kscale.py:144-147` and `kscale.py:171-175` unchanged. The history, the grid and the shape count constants are removed, and the joint checks are enabled.

```python
# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the kscale biped.

Robot-specific module, written closely against :mod:`.brs` so that the two are reviewable side
by side. Implements ``compute_symmetric_states`` in the signature the rsl_rl PPO symmetry hook
expects (``/ws/rsl_rl/rsl_rl/algorithms/ppo.py``) and referenced through
``isaaclab_rl.rsl_rl.RslRlSymmetryCfg``. A biped has one sagittal mirror, so the augmentation
doubles the batch, originals in the first half, mirrored copies in the second.

Three things differ from the SD_BRS1 module and each is a correctness matter rather than a
matter of taste.

The joint partner rule swaps the ``right_``/``left_`` PREFIX. SD_BRS1 swaps a trailing L for a
trailing R, and every kscale joint name ends in a digit, so that rule would match nothing and
degenerate silently to an identity permutation, producing an augmentation that teaches the
policy the robot is symmetric under doing nothing. The rule here raises instead of degrading.

The body partners are an explicit dictionary rather than any rule, because the right hip roll
link is named ``kd_d_201r_6061`` and its left partner is named ``rs03``, sharing no stem at all.
This is an Onshape export artefact: where a part was reused rather than mirrored in CAD, the
exporter names it after the part rather than after the side.

The sign flip set adds the hip yaw. It was established by composing each joint's origin
rotation chain to the root and comparing each left joint's mirrored world axis against its right
partner's actual axis, which settles the set without appeal to a naming convention or to a roll
against pitch rule. See ``scripts/analysis/kscale_physical_analysis.py`` and
``context/KScale.md`` section 7. Under a reflection a rotation axis transforms as a
pseudovector, ``a -> det(M) * M * a``, which is the treatment MorphoSymm establishes
(arXiv:2302.10433).

The resulting set is the physically expected pattern, the roll and yaw degrees of freedom
flipping and the pitch degrees of freedom not, with the single exception of the hip pitch, which
flips because its left and right URDF axes are genuinely anti-parallel. That exception is
corroborated independently by the joint limits: the hip pitch and hip roll are the only two
joints whose left and right limits are negations of one another, while the remaining four carry
identical limits on both sides. SD_BRS1 documents the same anomaly for itself.

This module is correct only against the CORRECTED root frame. The kscale URDF as exported
carried a root frame rotated ninety degrees about z, in which the mirror plane is y-z rather
than x-z and the flip set inverts to the physically implausible pattern of the pitch joints
flipping and the roll joints not. ``scripts/analysis/kscale_reframe_urdf.py`` performs the
correction and ``kscale_sole_analysis.py`` verifies it; that an incorrect mirror plane yields a
plausible looking table is exactly why the frame must be verified rather than assumed.

Every observation group is mirrored, the critic included, so the value function is biased toward
the sagittal invariance the theory requires. A value function that observes an unmirrored
privileged state cannot be invariant under the reflection. The term layout, the history of each
group, the height scan pattern and the collision shapes are read at runtime, and the joint and
body maps are built from the runtime names, so the module is correct regardless of whether Isaac
Sim resolves the sibling links left-first or right-first. The machinery is in :mod:`.common`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

# joints whose sign flips under the sagittal mirror, by name substring. hip_pitch flips despite
# being a pitch joint because its left and right URDF axes are opposite. The hip yaw is here and
# absent from the SD_BRS1 set only because SD_BRS1 declares its hip yaw type="fixed", so that
# joint never enters robot.joint_names at all; kscale actuates twelve joints against its ten.
# None of these four substrings occurs in "knee_04" or in "foot_pitch_02", so the match is
# unambiguous and no negative guard is needed.
_FLIP_JOINT_KINDS = ("hip_pitch", "hip_roll", "hip_yaw", "foot_roll")

# body partners, as an explicit map. Every entry is given in both directions below, and the
# torso is its own partner, lying on the mirror plane.
_BODY_PARTNERS = {
    "assy_formfg___kd_b_102b_torso_btm": "assy_formfg___kd_b_102b_torso_btm",
    "kd_d_102r_6061": "kd_d_102l_6061",
    "kd_d_201r_6061": "rs03",
    "kd_d_301r_6061": "kd_d_301l_6061",
    "kd_d_401r_6061": "kd_d_401l_6061",
    "arb_uj111_cross_bearing": "arb_uj111_cross_bearing_2",
    "foot_6061": "foot_6061_2",
}
_BODY_PARTNERS.update({v: k for k, v in _BODY_PARTNERS.items()})

def _joint_partner(name: str) -> str:
    """Swap the side prefix. Raises rather than returning the name, so a miss is not silent."""
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    raise ValueError(
        f"joint {name!r} carries neither a 'right_' nor a 'left_' prefix, so its mirror partner "
        "cannot be determined. Every kscale joint is expected to carry one."
    )


def _body_partner(name: str) -> str:
    if name not in _BODY_PARTNERS:
        raise ValueError(
            f"body {name!r} has no entry in _BODY_PARTNERS. The kscale link names are not "
            "derivable by any rule, so the map must be extended by hand if the URDF changes."
        )
    return _BODY_PARTNERS[name]


SPEC = common.MirrorSpec(
    joint_partner=_joint_partner,
    body_partner=_body_partner,
    flip_joint_kinds=_FLIP_JOINT_KINDS,
    check_joint_limits=True,
)

# module-level cache, one entry per (device); env is a singleton
_CACHE: dict = {}


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: "torch.Tensor | None" = None,
):
    """Augment obs and actions with the left-right sagittal mirror, originals first."""
    return common.augment(env, obs, actions, SPEC, _CACHE)
```

### 8.4 `mdp/symmetry/tron1.py`, new

```python
# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the TRON1 solefoot biped.

Robot-specific module over :mod:`.common`, for the asset emitted by the design generator at
``assets/mjcf/solefoot/tron1/base_robot.xml``, fifteen bodies and eight joints named
``<kind>_<side>_<suffix>`` with the side an R or an L between underscores.

The flip set is EMPTY. Composing each joint's body chain to the root gives the abduction axes
(-1, 0, 0) on the right and (1, 0, 0) on the left, so the left axis already equals the mirror
of the right, a -> -M a with M = diag(1, -1, 1), and the hip, knee and ankle pitch axes are
(0, -1, 0) on both sides, which is its own mirror. Every joint therefore keeps its sign, which
the identical left and right limits corroborate. The derivation is
``context/artefacts/tron1_quadruped-mjcf-mirror-audit-2026-10-20/``.

Every observation group of the plain, HIM and co-optimisation tasks is mirrored, so that an
estimator trained on the augmented batch learns a mirrored target from a mirrored input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

_SIDE = common.regex_side_partner(r"_(?P<side>[RL])_")

SPEC = common.MirrorSpec(
    joint_partner=_SIDE,
    body_partner=_SIDE,
    flip_joint_kinds=(),
    check_joint_limits=True,
)

_CACHE: dict = {}


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: "torch.Tensor | None" = None,
):
    """Augment obs and actions with the left-right sagittal mirror, originals first."""
    return common.augment(env, obs, actions, SPEC, _CACHE)
```

### 8.5 `mdp/symmetry/quadruped.py`, new

```python
# Copyright (c) 2024, the SD_BRS1 project.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal-plane) symmetry data augmentation for the design generator quadruped.

Robot-specific module over :mod:`.common`, for ``assets/mjcf/quadruped/quadruped.xml``,
twenty five bodies and twelve joints, each leg named by a fore or rear letter followed by a
side letter, ``FR``, ``FL``, ``RR`` and ``RL``. Only the side letter is exchanged, so the front
right leg pairs with the front left and the rear right with the rear left.

The group is the sagittal reflection alone. Every leg carries the same thigh and shank
rotations, so the knees of the front and the rear legs bend the same way and the robot has no
fore and aft mirror, unlike the ANYmal of Isaac Lab's reference module.

The abduction joints flip. Their axis is (1, 0, 0) on both sides, whose mirror is (-1, 0, 0),
so the left coordinate is the negation of the right. The hip and knee pitch axes are (0, 1, 0)
on both sides and keep their sign. The abduction limits are symmetric about zero and so cannot
corroborate the flip, the thigh and calf limits are identical across sides and corroborate the
keep. The derivation is ``context/artefacts/tron1_quadruped-mjcf-mirror-audit-2026-10-20/``.

The rule also serves the Unitree Go1 asset of the Go1 ablation task, whose legs carry the same
FR, FL, RR and RL prefixes and whose abduction axes follow the same convention.

Every observation group of the plain, HIM and co-optimisation tasks is mirrored, so that an
estimator trained on the augmented batch learns a mirrored target from a mirrored input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import common

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from tensordict import TensorDict

__all__ = ["compute_symmetric_states"]

_SIDE = common.regex_side_partner(r"(?:^|_)[FR](?P<side>[RL])_")

SPEC = common.MirrorSpec(
    joint_partner=_SIDE,
    body_partner=_SIDE,
    flip_joint_kinds=("_hip_joint",),
    check_joint_limits=True,
)

_CACHE: dict = {}


@torch.no_grad()
def compute_symmetric_states(
    env: "ManagerBasedRLEnv",
    obs: "TensorDict | None" = None,
    actions: "torch.Tensor | None" = None,
):
    """Augment obs and actions with the left-right sagittal mirror, originals first."""
    return common.augment(env, obs, actions, SPEC, _CACHE)
```

### 8.6 `mdp/symmetry/__init__.py`, rewritten

The two existing aliases are kept, the two new ones added, and the stale comment of section 3.3, item 9, corrected.

```python
# Every robot-specific module exports the same symbol, so none may be re-exported bare without
# shadowing the others. Each is reached through its own module or the prefixed alias below, and
# the shared machinery lives in .common.
from .brs import compute_symmetric_states as brs_compute_symmetric_states  # noqa: F401
from .kscale import (
    compute_symmetric_states as kscale_compute_symmetric_states,
)  # noqa: F401
from .quadruped import (
    compute_symmetric_states as quadruped_compute_symmetric_states,
)  # noqa: F401
from .tron1 import (
    compute_symmetric_states as tron1_compute_symmetric_states,
)  # noqa: F401

__all__ = [
    "brs_compute_symmetric_states",
    "kscale_compute_symmetric_states",
    "quadruped_compute_symmetric_states",
    "tron1_compute_symmetric_states",
]
```

### 8.7 The agent configurations

One field in each parent's algorithm, and the import of its function. Every subclass and every instance of section 6.1 inherits it, and nothing else of either file changes. The diff is against the working tree.

```diff
--- a/environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py
+++ b/environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py
@@ -12,6 +12,9 @@
 from environments.tasks.locomotion.mdp.symmetry.kscale import (
     compute_symmetric_states as kscale_compute_symmetric_states,
 )
+from environments.tasks.locomotion.mdp.symmetry.tron1 import (
+    compute_symmetric_states as tron1_compute_symmetric_states,
+)
 from environments.utils.wrappers.rsl_rl.rl_mlp_cfg import (
     EncoderCfg,
     RslRlPpoAlgorithmMlpCfg,
@@ -121,6 +124,14 @@
         desired_kl=0.01,
         max_grad_norm=1.0,
         # obs_history_len=10,
+        # Inherited by every SF task of this runner, plain, HIM and co-optimisation alike.
+        # See plans/symmetry_tron1_quadruped.md section 6.2 for the coefficient.
+        symmetry_cfg=RslRlSymmetryCfg(
+            use_data_augmentation=True,
+            use_mirror_loss=True,
+            data_augmentation_func=tron1_compute_symmetric_states,
+            mirror_loss_coeff=0.5,
+        ),
     )
     encoder = EncoderCfg(
         output_detach=True,
--- a/environments/environments/tasks/locomotion/agents/quadruped_rsl_rl_ppo_cfg.py
+++ b/environments/environments/tasks/locomotion/agents/quadruped_rsl_rl_ppo_cfg.py
@@ -1,6 +1,9 @@
 from isaaclab.utils import configclass
-from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg
+from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg
 
+from environments.tasks.locomotion.mdp.symmetry.quadruped import (
+    compute_symmetric_states as quadruped_compute_symmetric_states,
+)
 from environments.utils.wrappers.rsl_rl.rl_mlp_cfg import (
     EncoderCfg,
     RslRlPpoAlgorithmMlpCfg,
@@ -38,6 +41,14 @@
         lam=0.95,
         desired_kl=0.01,
         max_grad_norm=1.0,
+        # Inherited by every quadruped task of this runner, plain, HIM and co-optimisation alike.
+        # See plans/symmetry_tron1_quadruped.md section 6.2 for the coefficient.
+        symmetry_cfg=RslRlSymmetryCfg(
+            use_data_augmentation=True,
+            use_mirror_loss=True,
+            data_augmentation_func=quadruped_compute_symmetric_states,
+            mirror_loss_coeff=0.5,
+        ),
     )
     encoder = EncoderCfg(
         output_detach=True,
```

### 8.8 HIMPPO

The successor batch is mirrored under the condition that mirrors the batch itself, so that HIMPPO without a symmetry configuration, or with the loss alone, runs exactly as before. The containers train with an installed copy of `himloco` (section 9.1).

```diff
--- a/himloco/himloco/algorithms/him_ppo.py
+++ b/himloco/himloco/algorithms/him_ppo.py
@@ -175,6 +175,10 @@
                     actions=actions_batch,
                     env=self.symmetry["_env"],
                 )
+                # The estimator pairs each state with its successor, so the successor is mirrored alike
+                next_obs_batch, _ = data_augmentation_func(
+                    obs=next_obs_batch, actions=None, env=self.symmetry["_env"]
+                )
                 # Compute number of augmentations per sample
                 num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                 # Repeat the rest of the batch
```

### 8.9 `scripts/rsl_rl/train.py`

The co-optimisation runner receives a deep copy of the dictionary that `train.py` later dumps, so the environment the runner writes into its own copy never reaches the dump. The dictionary holds configuration values and the augmentation function, which a deep copy shares rather than duplicates, so the runner receives the same content as before.

```diff
--- a/scripts/rsl_rl/train.py
+++ b/scripts/rsl_rl/train.py
@@ -3,6 +3,7 @@
 """Launch Isaac Sim Simulator first."""
 
 import argparse
+import copy
 import pickle
 import sys
 import yaml
@@ -313,10 +314,12 @@
             "min_fitness_episode_length": 25,
             "reinit_lr_each_generation": True,
         }
+        # The runner writes the environment into its symmetry configuration, which the
+        # copt_agent.yaml dump below could not serialise, so it receives a copy.
         runner = CoptOnPolicyRunner(
             env,
             design_generator,
-            agent_cfg_dict,
+            copy.deepcopy(agent_cfg_dict),
             log_dir=log_dir,
             device=agent_cfg.device,
         )
```

### 8.10 The test suite, `environments/tests/symmetry_test.py`, new

The suite stubs exactly the surfaces the package reads and loads the package by path, in the manner of the existing reward suites (`environments/tests/rewards_feet_hold_penalty_test.py:1-16`), so it runs on torch and tensordict alone. Its first test compares the rebuilt SD_BRS1 and KScale modules with commit `23513d2` in all three call forms of section 3.1, the actions and the policy group bitwise, the critic bitwise outside the four corrected terms and different within them, and the `commands` group mirrored. The others assert that the originals occupy the first half and the mirror is an involution for every group, the TRON1's unsigned and the quadruped's signed joint maps, the ray derived height mirror against the fixed grid and for the "yx" ordering, the world frame kinds with the feet exchanged in both the velocity and the contact force, the HIM layout for all four robots with its history kept as an axis, the co-optimisation groups, and that a wrong flip set, a missing flip under an asymmetric default pose and an unmapped term in any group are each refused.

```python
"""Validation of the sagittal symmetry package, ``mdp/symmetry``, against stub environments.

The package is loaded as a free standing package by path rather than imported through
``environments``, whose ``__init__`` pulls in Isaac Lab, so the suite runs on torch and tensordict
alone. The stubs expose exactly the surfaces the package reads, the robot's names, limits, default
pose and PhysX views, the observation manager's layout and term configurations, and the sensors.

The SD_BRS1 and kscale modules are compared bitwise against their implementation at commit
23513d2, read with ``git show``, so the refactor onto ``common`` is shown to change the actions and
the policy group not at all, and the critic only in the four terms the old modules left unmirrored.

Run with ``python3 -m pytest environments/tests/symmetry_test.py``, or directly for a report.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import pathlib
import subprocess
import sys
import tempfile
import types

import torch
from tensordict import TensorDict

_REPO = pathlib.Path(__file__).resolve().parents[2]
_REL = "environments/environments/tasks/locomotion/mdp/symmetry"
_SYMMETRY = pathlib.Path(os.environ.get("SYMMETRY_DIR", _REPO / _REL))
_LEGACY_COMMIT = "23513d2"
N = 64


def _package(name: str, directory) -> types.ModuleType:
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(directory)]
    sys.modules[name] = pkg
    return pkg


def _legacy_dir() -> pathlib.Path | None:
    """The legacy modules, from SYMMETRY_LEGACY_DIR or from git. Inside the Isaac Lab containers
    this repository is a submodule whose git directory is not mounted, so git cannot serve them."""
    if "SYMMETRY_LEGACY_DIR" in os.environ:
        return pathlib.Path(os.environ["SYMMETRY_LEGACY_DIR"])
    out = pathlib.Path(tempfile.mkdtemp(prefix="symmetry_legacy_"))
    for mod in ("brs", "kscale"):
        res = subprocess.run(["git", "-C", str(_REPO), "show", f"{_LEGACY_COMMIT}:{_REL}/{mod}.py"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            return None
        (out / f"{mod}.py").write_text(res.stdout)
    return out


_package("sym_new", _SYMMETRY)
_LEGACY = _legacy_dir()
if _LEGACY is not None:
    _package("sym_old", _LEGACY)
common = importlib.import_module("sym_new.common")


# ---------------------------------------------------------------- stubs

class _Entity:
    def __init__(self, name, body_ids: slice | list[int] = slice(None)):
        self.name, self.body_ids = name, body_ids


class _Term:
    def __init__(self, params=None, history_length=0):
        self.params, self.history_length = params or {}, history_length


class _View:
    def __init__(self, shapes):
        self.link_paths = [[f"/link{i}" for i in range(len(shapes))]]
        self.max_shapes = sum(shapes)


class _SimView:
    def __init__(self, shapes):
        self._shapes = shapes

    def create_rigid_body_view(self, path):
        return types.SimpleNamespace(max_shapes=self._shapes[int(path[5:])])


class _Robot:
    def __init__(self, joints, bodies, limits, shapes, default=None):
        self.joint_names, self.body_names = joints, bodies
        pose = {".*": 0.0} if default is None else dict(zip(joints, default))
        self.cfg = types.SimpleNamespace(init_state=types.SimpleNamespace(joint_pos=pose))
        self.data = types.SimpleNamespace(joint_pos_limits=torch.tensor(limits)[None].repeat(2, 1, 1))
        self.root_physx_view = _View(shapes)
        self._physics_sim_view = _SimView(shapes)


def _grid_rays(ordering="xy", size=(1.6, 1.0), res=0.1):
    x = torch.arange(-size[0] / 2, size[0] / 2 + 1e-9, res)
    y = torch.arange(-size[1] / 2, size[1] / 2 + 1e-9, res)
    gx, gy = torch.meshgrid(x, y, indexing="xy" if ordering == "xy" else "ij")
    starts = torch.stack([gx.flatten(), gy.flatten(), torch.zeros(gx.numel())], dim=1) + torch.tensor([0.0, 0.0, 20.0])
    return types.SimpleNamespace(ray_starts=starts[None].repeat(2, 1, 1), cfg=types.SimpleNamespace(ray_alignment="yaw"))


class _ObsManager:
    """groups maps a group to (terms, flatten), a term being (name, width of one frame, cfg)."""

    def __init__(self, groups):
        self.active_terms, self.group_obs_term_dim, self._group_obs_term_cfgs = {}, {}, {}
        self.group_obs_concatenate, self._group_obs_concatenate_dim = {}, {}
        for g, (terms, flatten) in groups.items():
            self.add(g, terms, flatten)

    def add(self, g, terms, flatten=True):
        h = lambda t: max(t[2].history_length, 1)  # noqa: E731
        self.active_terms[g] = [t[0] for t in terms]
        self.group_obs_term_dim[g] = [(t[1] * h(t),) if flatten or t[2].history_length == 0 else (h(t), t[1])
                                      for t in terms]
        self._group_obs_term_cfgs[g] = [t[2] for t in terms]
        self.group_obs_concatenate[g], self._group_obs_concatenate_dim[g] = True, -1


class _Env:
    def __init__(self, robot, groups, sensors):
        self.scene = _Scene({"robot": robot}, sensors)
        self.observation_manager = _ObsManager(groups)

    @property
    def unwrapped(self):
        return self


class _Scene(dict):
    def __init__(self, entities, sensors):
        super().__init__(entities)
        self.sensors = sensors


def _proprio(nj, gait=False, lin_vel=True):
    terms = [("base_lin_vel", 3)] if lin_vel else []
    terms += [("base_ang_vel", 3), ("proj_gravity", 3), ("joint_pos", nj), ("joint_vel", nj), ("last_action", nj),
              ("velocity_commands", 3)]
    return terms + ([("gait_phase", 2)] if gait else [])


def _critic(nj, nb, feet_ids, num_shapes, h, gait=False):
    t = lambda **params: _Term(params, h)  # noqa: E731
    feet_asset = {"asset_cfg": _Entity("robot", feet_ids)}
    feet_sensor = {"sensor_cfg": _Entity("contact_forces", [nb - 1 - i for i in feet_ids])}
    return [(n, w, t()) for n, w in _proprio(nj)] + [
        ("robot_joint_torque", nj, t()), ("robot_joint_acc", nj, t()),
        ("feet_lin_vel", 3 * len(feet_ids), t(**feet_asset)),
        ("robot_mass", nb, t()), ("robot_inertia", 9 * nb, t()),
        ("robot_joint_pos", nj, t()), ("robot_joint_stiffness", nj, t()),
        ("robot_joint_damping", nj, t()), ("robot_pos", 3, t()), ("robot_vel", 6, t()),
        ("robot_material_properties", 3 * num_shapes, t()),
        ("feet_contact_force", 3 * len(feet_ids), t(**feet_sensor)),
        ("heights", 187, t(sensor_cfg=_Entity("height_scanner"))),
    ] + ([("gait_phase", 2, t())] if gait else [])


def _groups(nj, nb, feet_ids, num_shapes, h_policy=10, gait=False, him=False):
    commands = ([("velocity_commands", 3, _Term())], True)
    critic = (_critic(nj, nb, feet_ids, num_shapes, 10, gait), True)
    if not him:
        return {"policy": ([(n, w, _Term({}, h_policy)) for n, w in _proprio(nj, gait)], True),
                "critic": critic, "commands": commands}
    one = _proprio(nj, lin_vel=False)
    return {"policy": ([(n, w, _Term()) for n, w in one], True),
            "obsHistory": ([(n, w, _Term({}, 10)) for n, w in one], False),
            "critic": critic, "commands": commands,
            "estimatorGT": ([("base_lin_vel", 3, _Term())], True),
            "targetEnc": ([(n, w, _Term()) for n, w in one + [("base_lin_vel", 3)]], True)}


def _env(joints, bodies, limits, shapes, feet, default=None, **kw):
    robot = _Robot(joints, bodies, limits, shapes, default)
    feet_ids = [bodies.index(f) for f in feet]
    sensors = {"height_scanner": _grid_rays(kw.pop("ordering", "xy")),
               "contact_forces": types.SimpleNamespace(body_names=list(reversed(bodies)))}
    return _Env(robot, _groups(len(joints), len(bodies), feet_ids, sum(shapes), **kw), sensors)


def _obs(env, seed=0):
    g = torch.Generator().manual_seed(seed)
    om = env.observation_manager
    out = {}
    for k in om.active_terms:
        dims = om.group_obs_term_dim[k]
        lead = dims[0][:-1]
        out[k] = torch.randn(N, *lead, sum(d[-1] for d in dims), generator=g)
    return TensorDict(out, batch_size=[N])


def _starts(env, group):
    om, start, acc = env.observation_manager, {}, 0
    for n, d in zip(om.active_terms[group], om.group_obs_term_dim[group]):
        start[n], acc = (acc, acc + d[-1]), acc + d[-1]
    return start


# ---------------------------------------------------------------- robots

BRS_JOINTS = [f"{k}{s}" for k in ("HipRoll", "HipPitch", "KneePitch", "AnkleRoll", "AnklePitch") for s in "LR"]
BRS_BODIES = ["Part_Torso"] + [f"Link{i}{s}" for i in range(1, 7) for s in "LR"]
KS_KINDS = ("hip_pitch_04", "hip_roll_03", "hip_yaw_03", "knee_04", "foot_pitch_02", "foot_roll_02")
KS_JOINTS = [f"{s}_{k}" for k in KS_KINDS for s in ("left", "right")]
KS_BODIES = ["assy_formfg___kd_b_102b_torso_btm", "kd_d_102l_6061", "kd_d_102r_6061", "rs03", "kd_d_201r_6061",
             "kd_d_301l_6061", "kd_d_301r_6061", "kd_d_401l_6061", "kd_d_401r_6061", "arb_uj111_cross_bearing_2",
             "arb_uj111_cross_bearing", "foot_6061_2", "foot_6061"]
TRON1_LIMITS = {"abad": (-0.383970, 1.396260), "hip": (-1.396260, 1.012290), "knee": (-1.213570, 1.172670),
                "ankle": (-0.791917, 1.308990)}
TRON1_JOINTS = [f"{k}_{s}_Joint" for k in TRON1_LIMITS for s in "LR"]
TRON1_BODIES = ["base_Link"] + [f"{k}_{s}_{x}" for k, x in (("abad", "act"), ("hip", "act"), ("knee", "act"),
                ("thigh", "Link"), ("shank", "Link"), ("ankle", "act"), ("foot", "Link")) for s in "LR"]
QUAD_LIMITS = {"hip": (-0.802851, 0.802851), "thigh": (-1.931537, 3.304453), "calf": (-0.927857, 0.852375)}
QUAD_LEGS = ("FL", "FR", "RL", "RR")
QUAD_JOINTS = [f"{leg}_{k}_joint" for k in QUAD_LIMITS for leg in QUAD_LEGS]
QUAD_BODIES = ["trunk"] + [f"{k}_{leg}_{x}" for k, x in (("abad", "act"), ("hip", "act"), ("knee", "act"),
               ("thigh", "Link"), ("shank", "Link"), ("foot", "Link")) for leg in QUAD_LEGS]
# the Go1 convention, abduction out on both sides at opposite signs
QUAD_DEFAULT = [{"L": 0.1, "R": -0.1}[j[1]] if "_hip_" in j else 0.8 for j in QUAD_JOINTS]


def _brs_env(**kw):
    return _env(BRS_JOINTS, BRS_BODIES, [(-1.0, 1.0)] * 10, [1] * 7 + [2] * 6, ["Link6L", "Link6R"], gait=True, **kw)


def _kscale_env(**kw):
    return _env(KS_JOINTS, KS_BODIES, [(-1.0, 1.0)] * 12, [1] * 13, ["foot_6061", "foot_6061_2"], **kw)


def _tron1_env(**kw):
    limits = [TRON1_LIMITS[j.split("_")[0]] for j in TRON1_JOINTS]
    return _env(TRON1_JOINTS, TRON1_BODIES, limits, [1] * 15, ["foot_L_Link", "foot_R_Link"], **kw)


def _quad_env(**kw):
    limits = [QUAD_LIMITS[j.split("_")[1]] for j in QUAD_JOINTS]
    return _env(QUAD_JOINTS, QUAD_BODIES, limits, [1] * 25, [f"foot_{leg}_Link" for leg in QUAD_LEGS],
                default=QUAD_DEFAULT, **kw)


def _fresh(pkg, mod):
    m = importlib.import_module(f"{pkg}.{mod}")
    m._CACHE.clear()
    return m


def _involution(mod, env):
    m = _fresh("sym_new", mod)
    obs, act = _obs(env), torch.randn(N, len(env.scene["robot"].joint_names))
    o1, a1 = m.compute_symmetric_states(env=env, obs=obs, actions=act)
    o2, a2 = m.compute_symmetric_states(env=env, obs=o1[N:], actions=a1[N:])
    assert set(o1.keys()) == set(obs.keys())
    for k in obs.keys():
        assert torch.equal(o1[k][:N], obs[k]), (mod, k, "originals first")
        assert torch.equal(o2[k][N:], obs[k]), (mod, k, "involution")
    assert torch.equal(a2[N:], act)
    return o1, a1, obs, act


# ---------------------------------------------------------------- tests

def test_legacy_actions_and_policy_unchanged():
    """Against commit 23513d2, the actions and the policy group are bitwise equal, the critic differs only in
    the four terms the old modules tiled, and the commands group, which they tiled, is now mirrored."""
    if _LEGACY is None:
        print("SKIP test_legacy_actions_and_policy_unchanged, no legacy source, set SYMMETRY_LEGACY_DIR")
        return
    corrected = ("feet_lin_vel", "robot_vel", "robot_material_properties", "feet_contact_force")
    for mod, make in (("brs", _brs_env), ("kscale", _kscale_env)):
        old, new = _fresh("sym_old", mod), _fresh("sym_new", mod)
        env = make()
        obs, act = _obs(env), torch.randn(N, len(env.scene["robot"].joint_names))
        keep = torch.ones(obs["critic"].shape[-1], dtype=torch.bool)
        for name, (a, b) in _starts(env, "critic").items():
            if name in corrected:
                keep[a:b] = False
        for kwargs in ({"obs": obs, "actions": act}, {"obs": obs, "actions": None}, {"obs": None, "actions": act}):
            o_old, a_old = old.compute_symmetric_states(env=env, **kwargs)
            o_new, a_new = new.compute_symmetric_states(env=env, **kwargs)
            assert (a_old is None) == (a_new is None) and (o_old is None) == (o_new is None)
            if a_old is not None:
                assert torch.equal(a_old, a_new), mod
            if o_old is not None:
                assert torch.equal(o_old["policy"], o_new["policy"]), (mod, "policy")
                assert torch.equal(o_old["critic"][:, keep], o_new["critic"][:, keep]), (mod, "critic")
                assert not torch.equal(o_old["critic"][N:, ~keep], o_new["critic"][N:, ~keep]), (mod, "corrected")
                assert torch.equal(o_new["commands"][N:], obs["commands"] * torch.tensor(common.CMD)), (mod, "commands")
        _involution(mod, make())


def test_tron1_maps():
    o1, a1, obs, act = _involution("tron1", _tron1_env())
    jperm = [TRON1_JOINTS.index(n.replace("_L_", "_X_").replace("_R_", "_L_").replace("_X_", "_R_")) for n in TRON1_JOINTS]
    assert torch.equal(a1[N:], act[:, jperm]), "TRON1 flips no joint"
    c, c1 = obs["critic"], o1["critic"][N:]
    assert torch.equal(c1[:, :30].view(N, 10, 3), c[:, :30].view(N, 10, 3) * torch.tensor(common.VEC))
    assert torch.equal(c1[:, 30:60].view(N, 10, 3), c[:, 30:60].view(N, 10, 3) * torch.tensor(common.PVEC))


def test_quadruped_maps():
    _, a1, _, act = _involution("quadruped", _quad_env())
    for j, name in enumerate(QUAD_JOINTS):
        leg, kind = name[:2], name.split("_")[1]
        p = QUAD_JOINTS.index(name.replace(leg, leg[0] + {"L": "R", "R": "L"}[leg[1]], 1))
        s = -1.0 if kind == "hip" else 1.0
        assert torch.equal(a1[N:, j], s * act[:, p]), name


def test_heights_from_rays_matches_grid_and_handles_yx():
    xy = common.ray_mirror_perm(_grid_rays("xy"))
    assert xy == torch.arange(187).view(11, 17).flip(0).flatten().tolist()
    yx = common.ray_mirror_perm(_grid_rays("yx"))
    assert yx == torch.arange(187).view(17, 11).flip(1).flatten().tolist()


def test_world_frame_kinds_exchange_the_feet():
    env = _tron1_env()
    o1, _, obs, _ = _involution("tron1", env)
    start = _starts(env, "critic")
    c, c1 = obs["critic"], o1["critic"][N:]
    rv = slice(*start["robot_vel"])
    assert torch.equal(c1[:, rv].view(N, 10, 6), c[:, rv].view(N, 10, 6) * torch.tensor(common.ROOT_VEL_W))
    for term in ("feet_lin_vel", "feet_contact_force"):
        fv = slice(*start[term])
        assert torch.equal(c1[:, fv].view(N, 10, 2, 3), c[:, fv].view(N, 10, 2, 3).flip(2)), (term, "feet exchanged")
    pos = slice(*start["robot_pos"])
    assert torch.equal(c1[:, pos], c[:, pos]), "root position on the plane"


def test_him_layout_mirrors_the_unflattened_history():
    """The HIM groups, a one frame policy and a (10, w) history, for the robot that has them all."""
    for mod, make in (("tron1", _tron1_env), ("quadruped", _quad_env), ("brs", _brs_env), ("kscale", _kscale_env)):
        env = make(him=True)
        o1, _, obs, _ = _involution(mod, env)
        assert obs["obsHistory"].dim() == 3
        h, h1, p, p1 = obs["obsHistory"], o1["obsHistory"][N:], obs["policy"], o1["policy"][N:]
        m = importlib.import_module(f"sym_new.{mod}")._CACHE[torch.device("cpu")]["groups"]
        pi, ps = m["policy"]
        assert torch.equal(h1, h[..., pi] * ps), (mod, "each history frame mirrors as one policy frame")
        assert torch.equal(p1, p[..., pi] * ps)


def test_copt_groups_of_the_quadruped():
    """The co-optimisation groups mirror, link lengths by pair and the morphology over its body subset."""
    env = _quad_env()
    legs = ("FR", "FL", "RR", "RL")
    parents = [f"thigh_{leg}_Link" for leg in legs] + [f"shank_{leg}_Link" for leg in legs]
    children = [f"shank_{leg}_Link" for leg in legs] + [f"foot_{leg}_Link" for leg in legs]
    moved = [i for i, b in enumerate(QUAD_BODIES) if b.split("_")[0] in ("abad", "hip", "knee", "thigh", "shank")]
    fold = [i for i, b in enumerate(QUAD_BODIES) if b.split("_")[0] in ("thigh", "shank")]
    lengths = {"parent_body_names": parents, "child_body_names": children, "asset_cfg": _Entity("robot")}
    om = env.observation_manager
    om.add("historyObs", [(n, w, _Term({}, 25)) for n, w in _proprio(12)]
           + [("heights", 187, _Term({"sensor_cfg": _Entity("height_scanner")}, 25))])
    om.add("morphologyObs", [("link_lengths", 8, _Term(lengths)),
                             ("robot_mass", len(moved), _Term({"asset_cfg": _Entity("robot", moved)})),
                             ("robot_inertia", 9 * len(moved), _Term({"asset_cfg": _Entity("robot", moved)})),
                             ("nominal_pose", len(fold), _Term({"asset_cfg": _Entity("robot", fold)}))])
    feet = [QUAD_BODIES.index(f"foot_{leg}_Link") for leg in QUAD_LEGS]
    om.add("privilegedDynamicsObs", [("robot_joint_torque", 12, _Term()), ("robot_joint_acc", 12, _Term()),
                                     ("feet_contact_force", 12, _Term({"sensor_cfg": _Entity("contact_forces", [24 - i for i in feet])})),
                                     ("feet_lin_vel", 12, _Term({"asset_cfg": _Entity("robot", feet)}))])
    om.add("estimatorGT", [("base_lin_vel", 3, _Term())])
    om.add("critic", [*zip(om.active_terms["critic"], [d[0] // 10 for d in om.group_obs_term_dim["critic"]],
                           om._group_obs_term_cfgs["critic"]), ("link_lengths", 8, _Term(lengths, 10)),
                      ("nominal_pose", len(fold), _Term({"asset_cfg": _Entity("robot", fold)}, 10))])
    o1, _, obs, _ = _involution("quadruped", env)
    ll, ll1 = obs["morphologyObs"][:, :8], o1["morphologyObs"][N:, :8]
    assert torch.equal(ll1, ll[:, [1, 0, 3, 2, 5, 4, 7, 6]]), "link lengths exchange FR with FL and RR with RL"
    mass, mass1 = obs["morphologyObs"][:, 8:8 + len(moved)], o1["morphologyObs"][N:, 8:8 + len(moved)]
    names = [QUAD_BODIES[i] for i in moved]
    swap = common.regex_side_partner(r"(?:^|_)[FR](?P<side>[RL])_")
    assert torch.equal(mass1, mass[:, [names.index(swap(n)) for n in names]])
    assert torch.equal(o1["estimatorGT"][N:], obs["estimatorGT"] * torch.tensor(common.VEC))


def test_joint_limits_reject_a_wrong_flip_set():
    wrong = dataclasses.replace(_fresh("sym_new", "tron1").SPEC, flip_joint_kinds=("abad",))
    try:
        common.compile_maps(_tron1_env(), wrong, "cpu")
    except ValueError as err:
        assert "abad" in str(err)
    else:
        raise AssertionError("a flipped TRON1 abduction must contradict its identical limits")


def test_default_pose_rejects_a_missing_flip():
    """Symmetric abduction limits cannot decide the flip, the default pose of opposite abductions can."""
    wrong = dataclasses.replace(_fresh("sym_new", "quadruped").SPEC, flip_joint_kinds=())
    try:
        common.compile_maps(_quad_env(), wrong, "cpu")
    except ValueError as err:
        assert "default pose" in str(err)
    else:
        raise AssertionError("an unflipped abduction must contradict a default pose of opposite abductions")


def test_an_unmapped_term_of_any_group_raises():
    for group in ("critic", "commands"):
        env = _tron1_env()
        om = env.observation_manager
        om.active_terms[group].append("unmapped")
        om.group_obs_term_dim[group].append((1,))
        om._group_obs_term_cfgs[group].append(_Term())
        try:
            common.compile_maps(env, _fresh("sym_new", "tron1").SPEC, "cpu")
        except KeyError:
            pass
        else:
            raise AssertionError(f"an unmapped term of {group} must raise")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
```

## 9. Applying the plan

### 9.1 The order

1. Write the six symmetry files of sections 8.1 to 8.6 and the suite of section 8.10.
2. Run the suite, with the legacy source supplied as section 9.2 describes, and stop on any failure.
3. Apply the HIMPPO diff of section 8.8. Both containers import `himloco` from an installed copy under `/workspace/isaaclab/_isaac_sim/kit/python/lib/python3.11/site-packages/himloco/` rather than from the tree, so reinstall it in each, `cd /workspace/isaaclab/biped && /workspace/isaaclab/isaaclab.sh -p -m pip install --no-deps --force-reinstall ./himloco`, and confirm with `cmp` that the installed `algorithms/him_ppo.py` equals the tree's. A run in flight keeps the module it imported.
4. Apply the `train.py` diff of section 8.9.
5. Apply the agent diffs of section 8.7, after confirming with `git diff` that neither file has changed since 2026-10-20, and applying the hunks by symbol should they have. The repository is mounted live into both containers, so from this step every SF and quadruped run launched trains with the symmetry configuration, and a run in flight, which imported the configuration when it started, does not.
6. Run the smoke test of section 9.3 and confirm each printed line of mirrored groups.
7. Launch the runs of section 9.4.
8. Make the documentation edits of section 9.5 in the same pass.

### 9.2 The test suite

The development container has no torch, and inside the Isaac Lab containers git cannot read this repository, which is a submodule whose git directory is not mounted there. So the legacy modules are extracted in the development container, copied into a production container, and the suite is run there on its processor, which leaves the training jobs' cards alone.

```bash
mkdir -p /ws/scratch/symmetry_legacy
for m in brs kscale; do
  git -C /ws/tron1-rl-isaaclab-cozum show 23513d2:environments/environments/tasks/locomotion/mdp/symmetry/$m.py \
    > /ws/scratch/symmetry_legacy/$m.py
done
docker cp /ws/scratch/symmetry_legacy isaac-lab-base-shreyas-gpu1:/tmp/symmetry_legacy
docker exec isaac-lab-base-shreyas-gpu1 bash -lc 'cd /workspace/isaaclab/biped && \
  CUDA_VISIBLE_DEVICES="" SYMMETRY_LEGACY_DIR=/tmp/symmetry_legacy \
  /isaac-sim/python.sh environments/tests/symmetry_test.py'
```

Without `SYMMETRY_LEGACY_DIR` and without a working git the equivalence test reports itself skipped rather than passing, so its output must be read.

### 9.3 The smoke test

Three iterations of one task per layout and robot on 64 environments. The working directory must hold no folder named `environments`, since a namespace package of that name shadows the installed one and leaves every task unregistered, which the sandbox met. The co-optimisation tasks take a random design search of four individuals, so that 64 environments divide every population. Each run must print its line of mirrored groups, `['policy', 'critic', 'commands']` for a plain task, `['policy', 'obsHistory', 'critic', 'commands', 'estimatorGT', 'targetEnc']` for a HIM task, and for a co-optimisation task the plain groups with `morphologyObs`, `privilegedDynamicsObs` and `historyObs`, and on the TRON1 `estimatorGT`, and any exception from the involution, kind, joint limit or default pose checks is a defect to be resolved before training.

```bash
docker exec isaac-lab-base-shreyas-gpu1 bash -lc 'mkdir -p /tmp/symmetry_smoke && cd /tmp/symmetry_smoke && \
  T=/workspace/isaaclab/biped/scripts/rsl_rl/train.py; A="--num_envs 64 --max_iterations 3 --headless --device cuda:0"; \
  C="--policy-type COPT --design_search random --copt_num_individuals 4"; \
  for t in "Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0" "Isaac-Quadruped-Blind-Rough-v0" \
           "Isaac-Limx-SF-HIM-Identified-Blind-Rough-Urdf-v0 --policy-type HIMPPO" \
           "Isaac-Quadruped-HIM-Blind-Rough-v0 --policy-type HIMPPO" \
           "Isaac-Limx-SF-Copt-MoRAL-Rough-v0 $C" "Isaac-Quadruped-Copt-Rough-v0 $C" \
           "Isaac-Limx-SDBRS1-Blind-Rough-v0" "Isaac-Limx-Kscale-Blind-Rough-v0"; do \
    /workspace/isaaclab/isaaclab.sh -p $T --task $t $A 2>&1 | grep -a "\[symmetry\]\|symmetry loss\|Error"; \
  done'
```

### 9.4 The training runs and their measurement

Each robot is trained on its rough task with its baseline's command line otherwise unchanged, the baseline in flight on 2026-10-20 being the control (section 6.2).

```bash
cd /workspace/isaaclab
./isaaclab.sh -p biped/scripts/rsl_rl/train.py --task Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0 \
  --run_name symmetry --video --video_length 2000 --video_interval 10000 --save_interval 1000 --seed 42 --headless
./isaaclab.sh -p biped/scripts/rsl_rl/train.py --task Isaac-Quadruped-Blind-Rough-v0 \
  --run_name symmetry --video --video_length 2000 --video_interval 10000 --save_interval 1000 --seed 42 --headless
```

The measurements follow the faults of section 1. For the TRON1, the per side share of environments loading each shank above 10 N and each foot, read by the contact probe of the fine tuning plan's arm 0 (`plans/tron1_locomotion_fine_tuning.md:298-302`), whose iteration 6000 figures of 20.5 against 2.7 per cent on the shanks and 80.8 per cent on the right foot are the baseline. For the quadruped, the per foot stance duty and landings per episode, read as the degenerate gait investigation reads them (`/ws/plans/QUADRUPED_DEGENERATE_GAIT.md:198`, `/ws/plans/QUADRUPED_DEGENERATE_GAIT.md:453`). For both, the Robinson symmetry index of each pair's stance duty [8], formed per environment and averaged as a magnitude, never pooled across environments (`/ws/context/gait_metrics.md:121-125`), together with `Loss/symmetry`, the two tracking rewards, and the share of terminations. The success criterion is a stance duty index well below the baseline's at an unchanged tracking reward. The statistics script pairs joints by a trailing L or R (`scripts/analysis/stats.py:1143-1161`) and so reports no joint pair for either robot, and its foot statistics are written for two feet (`stats.py:1207`), so these indices are to be computed from the evaluation dump directly until that script is extended.

### 9.5 Documentation in the same pass

The plans index registers this document. On applying it, `ARCHITECTURE.md` of this repository and of the workspace record, where they describe the SF and quadruped runners and their training modes, that both parents carry the symmetry configuration and so every plain, HIM and co-optimisation task of the two robots trains with it, and `CO_OPTIMISATION.md` records the same for the co-optimisation configuration together with the `train.py` copy. `context/tron1.md`, `context/quadruped.md` and `context/KScale.md` record each robot's mirror and its checks, `/ws/context/rsl_rl.md` records the in place injection of `_env`, the HIMPPO successor batch and its correction, and the standing HIM configuration defects of section 12 are recorded where those tasks are documented. The status banner above and the register move to IMPLEMENTED, with any divergence recorded.

## 10. Backwards compatibility

The request changes the behaviour of a set of existing callers deliberately, and the change is confined to that set.

1. Every task served by `SF_TRON1AFlatPPORunnerCfg` and `PFQuadrupedPPORunnerCfg` and their subclasses, twenty five TRON1 and twelve quadruped registrations (section 6.1), trains with the augmentation and the mirror loss from the moment the agent diff is applied. This is the change the request asks for. A run in flight is unaffected, having imported its configuration at launch. A run launched afterwards records `symmetry_cfg` in its `params/agent.yaml`, and a run of the same task launched before carries none, so the dumped configuration tells the two apart where the task identifier no longer does, and a comparison against history must read it.
2. No other runner changes. The point foot and wheel foot TRON1, the Berkeley imitation and the native Go1 runners carry no symmetry configuration and are untouched, and the SD_BRS1 and KScale runners keep theirs unchanged, coefficient and flags included.
3. The SD_BRS1 and KScale augmentations change where the review found them wrong and nowhere else. Their mirrored actions and policy groups are bitwise identical to commit `23513d2` in every call form, and their critic differs only in `feet_lin_vel`, `feet_contact_force`, `robot_vel` and `robot_material_properties`, which the suite asserts. Their `commands` group, which no network of their tasks reads, is now mirrored. A run of either robot launched afterwards therefore trains a critic against a corrected mirror, and its actor sees the same augmented data as before.
4. The SD_BRS1 and KScale HIM tasks could not complete an update under their own runners' symmetry configuration before the change (section 3.3, item 1). After it, the mirror compiles and the augmented update runs for both, and they still cannot train, for the standing configuration defects of section 12, which no part of this plan causes or touches.
5. No environment configuration and no task registration changes, so every task builds exactly the environment it built before, which the sandbox's byte identical configuration files establish.
6. No shared `mdp` function changes. The package is imported only by the agent configurations (section 3.2), and its two existing aliases keep their names and their targets. The two analysis scripts preserved as artefacts import `mdp.symmetry.kscale` and call its `compute_symmetric_states`, which keeps its signature, and whose output changes only as item 3 states.
7. The HIMPPO edit runs only when a symmetry configuration with augmentation is set, so HIMPPO without one is unchanged, and the `train.py` edit hands the co-optimisation runner a copy with the same content, so a co-optimisation run without symmetry is unchanged and its `copt_agent.yaml` identical.
8. A checkpoint trained before the change loads and plays after it, since the configuration adds no parameter, which section 11 confirmed on both baselines' checkpoints.

The defects left standing are recorded where they live, the lateral sign of the world frame foot vectors at the definition of `TERM_KINDS`, and the three task failures of section 12 in the context files of their robots.

## 11. Validation record

All validation was performed on 2026-10-20 without touching the working tree.

The test suite of section 8.10 passed all ten of its tests inside `isaac-lab-base-shreyas-gpu1` on its own torch 2.7.0 and tensordict 0.12.4, run on the processor, the equivalence test comparing the rebuilt modules with commit `23513d2` extracted by `git show` in the development container and copied in. The joint sign tables of section 4 were derived by the audit script from the two MuJoCo documents, md5 `808a09a927219e8d707cc27a4a83770b` for the TRON1 and `81a3b2a88f3a9d222952c3f2b0ec04d7` for the quadruped (`context/artefacts/tron1_quadruped-mjcf-mirror-audit-2026-10-20/`). The census of section 5.1 instantiated the environment configuration of all thirty three training tasks of the four families (`context/artefacts/sf_quadruped_brs_kscale-obs-term-census-2026-10-20/`).

Every file of section 8 was applied to a fresh copy of the working tree inside the same container, `/tmp/symsandbox_v2_20261020`, whose environment configuration files were confirmed byte identical to the live tree's, with the sandbox's `environments` and `himloco` placed ahead of the installed packages on the Python path and a probe in `compile_maps` printing the runtime names, permutations, shape counts, limits, configured default pose and layouts. The md5 of every sandbox file was confirmed equal to the file this plan carries, the probe aside. Each task then trained three iterations on 64 environments, with the following outcome (`context/artefacts/symmetry_revised-2026-10-20-sandbox-smoke/`).

| Task | Policy type | Groups mirrored | Symmetry loss | Outcome |
|---|---|---|---|---|
| `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0` | PPO | 3 | 0.0063, 0.0045, 0.0036 | passed, `model_2.pt` written |
| `Isaac-Quadruped-Blind-Rough-v0` | PPO | 3 | 0.0193, 0.0154 | passed, `model_2.pt` written |
| `Isaac-Limx-SF-HIM-Identified-Blind-Rough-Urdf-v0` | HIMPPO | 6 | 0.0035 | passed, `model_2.pt` written |
| `Isaac-Quadruped-HIM-Blind-Rough-v0` | HIMPPO | 6 | 0.0052 | passed, `model_2.pt` written |
| `Isaac-Limx-SF-Copt-MoRAL-Rough-v0` | COPT | 7 | 0.0053 | passed after the `train.py` copy, `model_2.pt` written |
| `Isaac-Quadruped-Copt-Rough-v0` | COPT | 6 | 0.0114 | passed after the `train.py` copy, `model_2.pt` written |
| `Isaac-Limx-SDBRS1-Blind-Rough-v0` | PPO | 3 | 0.0159, 0.0114 | passed, `model_2.pt` written |
| `Isaac-Limx-Kscale-Blind-Rough-v0` | PPO | 3 | 0.0159, 0.0133 | passed, `model_2.pt` written |
| `Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-v0` | PPO | 3 | 0.0182, 0.0136 | passed, `model_2.pt` written |
| `Isaac-Limx-SDBRS1-HIM-Blind-Rough-v0` | HIMPPO | 6 | | mirror and augmented update passed, the HIM estimator then failed on its target width, 42 against 45 |
| `Isaac-Limx-Kscale-HIM-Blind-Rough-v0` | HIMPPO | | | failed during scene construction, before any agent code |
| `Isaac-Quadruped-Go1Asset-Blind-Rough-v0` | PPO | | | failed during environment construction, before any agent code |

The symmetry loss is given for each iteration the runner logged, the quadruped's and the HIM and co-optimisation runners' logging fewer than three, and every passing run wrote the checkpoint of its last iteration. The TRON1 and quadruped plain runs reproduced the first pass's symmetry losses to four places under the same seed, the corrected critic having left the actor's first iterations unmoved. Every run's dumped configuration names its function as an importable member, `!!python/name:` in `agent.yaml` and a `module:function` string in `copt_agent.yaml`.

Four observations of the sandbox shaped the plan. The first plain TRON1 run raised in the default pose check, the per environment offsets of section 5.3 having perturbed the buffer it then read, and the check was moved to the configured pose. The first two co-optimisation runs failed in the dump of `copt_agent.yaml`, which section 3.3, item 8, explains and section 8.9 repairs, and both passed on rerun. The live tree, run unmodified from a clean directory, reproduced the SD_BRS1 HIM failure of section 3.3, item 1, at `brs.py:154`, `shape '[384, 10, 0]' is invalid for input of size 1152`, and the Go1 asset failure, `Sensor at path '/World/envs/env_.*/Robot/trunk/.*' could not find any bodies with contact reporter API`, so neither is caused by this plan. Finally `play.py` of the sandbox loaded copies of the two baselines' checkpoints, `model_2000.pt` of `sf_tron_1a_flat/2026-10-20_05-24-20` and `model_1000.pt` of `quadruped_flat/2026-10-20_05-09-02`, both trained without symmetry, under runners carrying it, played sixty steps and wrote their evaluation dumps, which section 10, item 8, relies upon.

Not validated. No run has been trained beyond three iterations, so nothing is yet known of the effect on either fault. The Go1 asset task's mirror has not been exercised, since the task cannot be built (section 4.3), and the SD_BRS1 and KScale HIM tasks have not completed an update, for the defects of section 12.

## 12. Risks, open questions and what is out of scope

1. The mirror may not remove the faults. Section 1 explains that an equivariant policy may still retract a foot provided it does so on the mirrored side, and that faults invariant under the reflection are untouched. If the quadruped exchanges a single retracted foot for a retracted rear pair, the cure lies with the support pattern terms of the degenerate gait plan, not here.
2. The mirror loss may slow learning [4]. The ablation of section 6.2 measures this without a new task.
3. The lateral sign of the world frame foot vectors remains wrong in the mirrored critic sample (section 5.1), now for all four robots, against every component wrong under the identity it replaces. A base frame variant of those privileged terms would remove it, at the price of a change to the environment and the loss of comparability with every earlier run, and is not proposed.
4. Every SF and quadruped task now trains with the symmetry, the co-optimisation tasks included. A co-optimisation design search that came to vary the legs independently would leave the mirrored morphology physically valid but outside the population, which the estimator would then be taught to predict. The present searches vary both legs alike (section 5.3).
5. Three task failures stand outside this plan and are recorded rather than repaired. The SD_BRS1 and KScale HIM policy groups keep `base_lin_vel`, so their target encoding is as wide as the policy, while the HIM estimator sizes its target network for the policy width plus three (`himloco/himloco/modules/him_estimator.py:50`, `himloco/himloco/modules/him_actor_critic.py:93`), and the SD_BRS1 HIM update fails on that mismatch, 42 against 45, with or without the mirror. The KScale HIM task fails earlier, during scene construction, on the type of `gpuMaxRigidContactCount`. And the Go1 asset task fails during environment construction, its contact sensor finding no body with the contact reporter under `trunk`. Each fails in code this plan does not change.
6. Hard equivariance is out of scope. Both robots are modelled exactly symmetric, which removes the reason cluster 9 gave for preferring augmentation on the SD_BRS1, so an equivariant actor and an invariant critic [1][6] are admissible in simulation, but they are a change of network rather than of data.
7. The statistics script reports no joint pairs for these robots (section 9.4). Extending its pairing to the partner rules of `common` is a small change to an analysis script, deferred so that this plan touches no analysis code.
8. The module reads private members of Isaac Lab (section 7.4). An upgrade of Isaac Lab must rerun the suite and the smoke test.

## 13. Implementation record

Applied on 2026-10-20 in the order of section 9.1. The six files of sections 8.1 to 8.6 and the suite of section 8.10 were written from the tested copies, `git diff` confirming the symmetry package unchanged since `23513d2` beforehand, and the suite passed all ten tests on the live tree in `isaac-lab-base-shreyas-gpu1` with the legacy source supplied as section 9.2 directs. The three diffs of sections 8.7 to 8.9 applied cleanly, and each resulting file was confirmed byte identical to the file the sandbox ran. `himloco` was reinstalled in both containers by the command of section 9.1, step 3, and the installed package was confirmed byte identical to the tree, file by file, in each.

The smoke test of section 9.3 then ran on the live tree, and every one of its eight tasks trained three iterations and wrote `model_2.pt`, the symmetry loss read from each run's event file being 0.0063, 0.0045 and 0.0036 on the TRON1 plain task and 0.0193, 0.0154 and 0.0095 on the quadruped's, equal to the sandbox's, 0.0035, 0.0015 and 0.0011 and 0.0052, 0.0020 and 0.0015 on the two HIM tasks, 0.0053, 0.0031 and 0.0031 and 0.0114, 0.0092 and 0.0067 on the two co-optimisation tasks, and 0.0138, 0.0093 and 0.0114 and 0.0159, 0.0133 and 0.0121 on the SD_BRS1 and KScale (`../context/artefacts/symmetry_applied-2026-10-20-live-smoke/`).

Two departures from the text are recorded. First, the printed line of mirrored groups that section 9.3 makes the criterion is not reliable. The two plain tasks printed neither that line nor any iteration, while their runs completed, because the simulator closes without flushing a standard output redirected to a file, so the criterion is the final checkpoint and the `Loss/symmetry` scalar of the run's event file, which every run carried. Second, a diagnostic run of the plain quadruped task launched from `/workspace/isaaclab` wrote a three iteration directory, `quadruped_flat/2026-10-20_10-21-23`, into the shared log tree, which was deleted the same hour once read, and section 9.3's instruction to run from a scratch directory is the guard against it. The `~imloco` directories found in the site-packages of `isaac-lab-base-shreyas-gpu0`, dated 2026-06-30, are the residue of an earlier reinstallation, which pip reports and ignores, and were left in place.

## References

[1] Su and colleagues. Leveraging Symmetry in RL-based Legged Locomotion Control. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2024. arXiv:2403.17320. Full author list not retrieved by the survey.

[2] Abdolhosseini and colleagues. On Learning Symmetric Locomotion. Motion in Games, 2019. DOI 10.1145/3359566.3360070. Full author list not retrieved by the survey.

[3] Yu, W., Turk, G., Liu, C. K. Learning Symmetric and Low-energy Locomotion. ACM Transactions on Graphics 37(4), Article 144, SIGGRAPH, 2018. DOI 10.1145/3197517.3201397, arXiv:1801.08093.

[4] Mittal, M., Rudin, N., Klemm, V., Allshire, A., Hutter, M. Symmetry Considerations for Learning Task Symmetric Robot Policies. IEEE International Conference on Robotics and Automation (ICRA), 2024. arXiv:2403.04359.

[5] Ordoñez-Apraez and colleagues. On discrete symmetries of robotics systems, a group theoretic and data driven analysis. Robotics, Science and Systems (RSS), 2023, and International Journal of Robotics Research, 2024. arXiv:2302.10433. Full author list not retrieved by the survey.

[6] Nie, B., Zhang, Y., Jin, R., Cao, Z., Lin, H., Yang, X., Gao, Y. Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy. AAAI Conference on Artificial Intelligence, 2026. DOI 10.1609/aaai.v40i22.38918, arXiv:2508.01247.

[7] Mishra, S., Abdolmaleki, A., Guez, A., Trochim, P., Precup, D. Augmenting learning using symmetry in a biologically-inspired domain. arXiv preprint, 2019, venue not stated by the arXiv record. arXiv:1910.00528.

[8] Robinson, Herzog and Nigg. Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. Journal of Manipulative and Physiological Therapeutics, 1987. Initials and identifier not recorded by the survey.
