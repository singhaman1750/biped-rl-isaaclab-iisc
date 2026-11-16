# TRON1 Locomotion Fine Tuning, the Late Emergence of a Bipedal Gait and the Ablation that Would Explain It

> Status, NOT IMPLEMENTED, a living document. Written 2026-10-19 against run `sf_tron_1a_flat/2026-10-09_06-04-18`, its dumped `params/env.yaml` and `params/agent.yaml`, its TensorBoard record, its forty four training videos, and the two analysis scripts this investigation added, `scripts/analysis/biped_physical_analysis.py` and `scripts/analysis/biped_stance_analysis.py`. Arm 0, the contact measurement, was run on 2026-10-19 and its results stand in section 6.2.1, falsifying F5 and clearing arms 1 to 7 to proceed with the 10 N threshold unchanged. No training arm has yet been run. The working tree carries one uncommitted edit that belongs to arm 1, the weight of `pen_base_height` raised from -5.0 to -10.0 at `environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:952`, which section 6.1 relocates before any arm is launched. On the same day `scripts/rsl_rl/play.py` was reviewed and made ready to replay the TRON1 for arm 0, its robot profile table gaining the TRON1 sole table and normal it had lacked, section 6.2.2 recording the review, and the arm 0 contact probe, run the same day, still stands in that script pending its removal. Each arm closes with a results sub-section that is to be filled in as the arm is run, and the status line above is to be amended with it.

## 1. What this document is, and what it rests upon

The TRON1 sole footed biped of this repository learns, under the configuration of the run named above, to walk upon its knees for roughly twelve thousand PPO iterations, three hundred million environment steps, before a gait upon its feet appears. This document establishes why, from four bodies of evidence gathered for the purpose, and proposes the ablation that would confirm the cause and remove it.

The evidence is taken in an order that builds from the robot to the learner. Section 3.1 measures the nominal pose baked into the link geometry and compares it with the URDF it replaced and with the original LimX document. Section 3.2 measures what that pose asks of the actuators while standing. Section 3.3 reads the reward and termination configuration against those measurements. Sections 3.4 and 3.5 read the videos and the time series of the run. Section 4 assembles the failure modes and ranks them, section 5 surveys the literature that bears on each, and section 6 sets out the ablation with every edit specified to the line, so that an arm may be executed from this document alone.

Every claim about the workspace carries its `file:line` citation, resolved against `tron1-rl-isaaclab-cozum/` unless the path says otherwise, and every claim from the literature carries a numbered marker resolved in section 8. The artefacts the document reasons about are registered in `../context/artefacts/README.md` under `sf_tron_1a_flat-2026-10-09_06-04-18/` and `tron1_sf-model-analysis-2026-10-19/`.

## 2. The run under study

The run is not the flat task its directory name suggests. Its dumped configuration names `SFIdentifiedBlindRoughEnvUrdfCfg` on its first line, the class at `environments/environments/tasks/locomotion/robots/limx_solefoot_env_cfg.py:1025-1038`, which `environments/environments/tasks/locomotion/robots/__init__.py:325-333` registers as `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0`. The experiment name `sf_tron_1a_flat` comes from the shared runner configuration at `environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py:101`. Despite the word Urdf, the class spawns the generated MJCF `environments/environments/assets/mjcf/solefoot/tron1/base_robot.xml` through `SOLEFOOT_IDENTIFIED_CFG` at `environments/environments/assets/config/solefoot_identified_cfg.py:100-120`, and it trains on `BERKELEY_MIMIC_TERRAINS_CFG` at `environments/environments/tasks/locomotion/cfg/SF/terrains_cfg.py:205-216`, a grid of six rows and sixty four columns of which only 0.3 is flat, the remainder being slopes, stairs, waves and rough patches, with a terrain level curriculum active from iteration 1920.

| Quantity | Value | Source |
|---|---|---|
| Environments, steps per iteration | 4096, 25 | `params/env.yaml:87`, `params/agent.yaml` |
| Physics step, decimation, control period | 0.005 s, 4, 0.02 s | `limx_base_env_cfg.py:1323-1327` |
| Episode length | 20 s, 1000 control steps | `limx_base_env_cfg.py:1324` |
| Actions | joint position targets, scale 0.25, about the default pose | `limx_base_env_cfg.py:130-139` |
| Default joint pose | every joint at zero, the pose being baked into the geometry | `limx_solefoot_env_cfg.py:1031-1038` |
| Spawn | root at 0.8 m, joints offset uniformly by plus or minus 0.2 rad, base velocities plus or minus 0.5 | `solefoot_identified_cfg.py:85-91`, `limx_base_env_cfg.py:843-867` |
| Actuators, stiffness and damping | abad 55 and 13.5, hip 80 and 13.0, knee 60 and 4.0, ankle 10 and 0.5 | `solefoot_identified_cfg.py:14-67` |
| Failure termination | `base_Link` contact above 1 N, nothing else | `limx_base_env_cfg.py:1080-1091` |
| PPO | learning rate 1e-3 adaptive at desired KL 0.01, entropy 0.005, initial noise 1.0 | `limx_rsl_rl_ppo_cfg.py:97-123` |
| Iterations completed | about 17200, checkpoints to `model_17000.pt` | the run directory |

The videos are numbered by the recorder's step counter, which counts vectorised `env.step` calls (`scripts/rsl_rl/train.py:206-219`), so a video at step N shows iteration N/25 and the step 310000 at which the user saw feet walking is iteration 12400. The curricula count iterations of 24 steps although the runner takes 25, so a curriculum written to start at 6000 iterations starts at 5760 (`limx_base_env_cfg.py:1106-1176`).

## 3. Evidence

### 3.1 The nominal pose baked into the link geometry

The generator places the thigh frame at `THIGH_BEND_DEG + 180` about +y, the shank at `SHANK_BEND_DEG` absolute and the sole at `SHANK_BEND_DEG + ANKLE_BEND_DEG` (`design_generator/biped/builder.py:154-168`), which with the constants -145.000014, -40.000189 and 40.003913 of `design_generator/biped/constant_params.json:23-25` gives a thigh rotated 35 degrees, a shank rotated -40 degrees and a level sole. The emitted MJCF carries these as body quaternions (`base_robot.xml:63`, `:68`, `:74`). `biped_physical_analysis.py` measures the result between joint axes, and the table compares it with the simplified URDF `environments/environments/assets/urdf/solefoot/tron1/base_robot.urdf` and with the original LimX document `base_robot_tron1.urdf` in the same directory (artefact `tron1_sf-model-analysis-2026-10-19/physical_analysis.txt`).

| Quantity | `base_robot.xml` | `base_robot.urdf` | `base_robot_tron1.urdf` |
|---|---|---|---|
| Total mass, kg | 21.328 | 21.328 | 20.813 |
| Trunk with welded housings, kg | 6.910 | 6.910 | 9.595 |
| Mass below the knee, kg | 2.790 | 2.790 | 1.840 |
| Share of mass in the legs | 0.676 | 0.676 | 0.539 |
| Thigh, hip axis to knee axis | 0.3487 m at -35.0 deg | 0.3487 m at -35.0 deg | 0.3007 m at -30.0 deg |
| Shank, knee axis to ankle axis | 0.3535 m at +37.2 deg | 0.3535 m at +37.2 deg | 0.3000 m at +30.0 deg |
| Knee included angle | 107.8 deg | 107.8 deg | 119.9 deg |
| Extension ratio, hip to ankle over thigh plus shank | 0.808 | 0.808 | 0.866 |
| Ankle ahead of the hip | 14.0 mm | 14.0 mm | 0.0 mm |
| Standing height, root above the sole | 0.8044 m | 0.8044 m | 0.8397 m |
| Centre of mass height | 0.5046 m | 0.5119 m | 0.5291 m |

Three conclusions follow. The MJCF and the simplified URDF carry the same nominal pose, the URDF writing it into joint origin rotations of 0.610865, -1.3090 and 0.6982 rad (`base_robot.urdf:154`, `:257`, `:292`), so the generator reproduced the URDF it replaced and did not deepen it. Against the original LimX geometry, both bend the knee twelve degrees further, 107.8 against 119.9, and stand 35 mm lower. And the larger departure is not the angle but the mass distribution, the generated trunk being 2.69 kg lighter and the mass below the knee 0.95 kg heavier, so that two thirds of the robot is leg. The thigh's own frame is rotated by 35.0 degrees as `context/tron1.md` records in its correction of 2026-10-07.

The analysis also exposes four small defects that bear on what follows. The spawn height of 0.8 m sits 4.4 mm below the standing height, so the soles begin every episode embedded in the ground, MuJoCo reporting a contact depth of 4.35 mm at spawn. The base height target of 0.75 m (`limx_base_env_cfg.py:949`) sits 54.4 mm below the standing height. The simplified URDF gives its left knee the range -0.873 to 1.361 rad and its right knee -1.214 to 1.173 rad (`base_robot.urdf:262`, `:528`), which the MJCF does not inherit. And the comments at `base_robot.urdf:250-254` claiming the original knee position no longer describe the file.

The joint limits leave the policy ample travel about the nominal pose, the knee reaching its soft limits 1.094 rad below and 1.053 rad above zero, 4.4 and 4.2 action units at the scale of 0.25. The same travel admits kneeling. A symmetric sweep of the sagittal joints within their limits, trunk level, finds that 10.3 per cent of postures rest on a leg link rather than a sole. The highest such posture holds the root 0.5418 m above the ground on the shanks, at offsets of 0.611, 0.477 and 1.134 rad on the hip, knee and ankle, and the lowest 0.2434 m on the ankle housings. The original LimX geometry kneels at 0.5749 m, so the kneeling posture is a property of a backward knee biped of this size and not of the generator.

### 3.2 What the pose asks of the actuators

`biped_stance_analysis.py` balances the distal free body of each joint against a ground reaction beneath the centre of mass and reports the torque each actuator must hold, the sag a proportional loop settles at under it, and the action the policy must emit merely to cancel that sag (artefacts `stance_analysis_mjcf.txt` and `stance_analysis_tron1_urdf.txt`).

| Joint, right leg | Double support torque, Nm | Single support torque, Nm | Stiffness | Single support sag, rad | Action to cancel it | Original TRON1 single support, Nm |
|---|---|---|---|---|---|---|
| abad | -0.63 | -1.49 | 55 | 0.027 | 0.11 | 0.08 |
| hip | -1.09 | +0.25 | 80 | 0.003 | 0.01 | -5.52 |
| knee | -14.85 | -34.40 | 60 | 0.573 | 2.29 | 32.79 |
| ankle | +2.87 | +5.68 | 10 | 0.568 | 2.27 | -4.45 |

The knee and the ankle are the soft joints. In single support the knee folds by 0.57 rad and the ankle by 0.57 rad under body weight alone, and the policy must add 2.3 action units to each, about 2.3 standard deviations of the initial Gaussian, merely to stand on one leg in the nominal pose. Every torque stays well inside its effort limit, the knee using 0.38 of its 90 Nm, so the deficit is one of stiffness and not of strength. The original LimX geometry asks almost the same of its knee, 32.79 Nm against 34.40, its heavier trunk offsetting its straighter leg, so the deeper bend accounts for about five per cent of the knee load and not for the softness.

The proposed gains follow from flooring the stiffness by the single support torque within a 0.15 rad budget and sizing the damping as the least of the stance target at a damping ratio of 0.7, the swing critical value and the explicit integration ceiling, the method of `context/KScale.md`. They are abad 115.7 and 17.07, hip 121.2 and 16.07, knee 229.4 and 15.82, ankle 37.9 and 1.14. The stance inertia, everything proximal to the joint with the foot planted, is 2.776 kg m^2 at the knee against a swing inertia of 0.273, and 6.066 at the ankle against 0.0085, so the configured damping leaves the stance knee at a damping ratio of 0.15 and the stance ankle at 0.03, the swing and stance regimes again disagreeing by an order of magnitude as they did on the KScale.

A zero action drop test in MuJoCo with the actuator model of `environments/actuators/actuator_pd.py:18-41` separates holding from balancing. With the trunk on a vertical slide, the configured gains settle the root at 0.7546 m, the knee deflected 0.223 rad, against a standing height of 0.8044, and over thirty one reset draws at 0.7476 m with no leg link touching the ground. The run's height target of 0.75 m is therefore the height the configured gains sag to, and not the height of the pose. The proposed gains hold 0.7912 m. Released free, the robot topples under either gain set, its first contact on a link other than a sole coming a median 0.78 s after release, and in 68 per cent of draws a shank, a thigh or a housing touches the ground at some point, the trunk following in every draw, so the passive outcome of a fall is a body resting on its lower legs, and a policy that has not yet learned to balance meets the kneeling posture before any other.

The bend sweep re-poses the knee with the hip and the ankle solved to keep the ankle lead and the sole level, which emulates a change of the baked bend. Straightening the knee by 0.2 rad, the hip and ankle following by 0.099 and 0.101 rad, restores the original LimX knee angle of 119.2 degrees, raises the standing height to 0.8428 m and lowers the single support knee torque from 34.4 to 29.8 Nm, fourteen per cent. Straightening by 0.3 rad lowers it to 27.3 Nm. Deepening by 0.3 rad raises it to 40.8 Nm. The bend therefore moves the knee load by tens of per cent while the stiffness would have to move it by a factor of nearly four, which is why section 4 ranks the pose as a contributor and not as the cause.

### 3.3 The reward and termination configuration against those measurements

The dumped weights are those of `RewardsCfg` at `limx_base_env_cfg.py:906-1036` with `pen_base_height` at -5.0, the -10.0 of the working tree postdating the run. Isaac Lab multiplies every term by its weight and the control period, so a term of weight w and value f contributes w times f per second, and the comparison below is made in that unit (`IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:150-153`).

The base height penalty `base_height_rough_l2` (`environments/environments/tasks/locomotion/mdp/rewards.py:2080-2116`) prices the squared deviation of the root above the mean of the height scan from 0.75 m. At the highest kneeling height of 0.5418 m it costs 0.217 per second, at the lowest 1.283, and at the run's measured mean squared error of 0.0757 at iteration 2000 it cost 0.38. Against tracking rewards that pay up to 25 and 7.5 per second, of which the knee walker collected 13 to 19 and 2 to 4, the penalty is one to two per cent of the return, the measured share being 1.2 to 1.7 per cent (section 3.5). Doubling it to -10 doubles a price that is already negligible. The L2 kernel compounds this, its gradient vanishing at the target and growing only linearly away from it, and on rough terrain the reference is the mean of a 1.6 by 1.0 m grid of ray hits rather than the surface the feet stand on (`limx_base_env_cfg.py:75-82`).

The undesired contact penalty counts the bodies among `abad_.*`, `hip_.*`, `knee_.*`, `thigh_.*`, `shank_.*` and `base_Link` whose contact force exceeds 10 N (`limx_base_env_cfg.py:961-973`, `IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:267-275`) at a weight of -0.5, so kneeling on both shanks costs 1.0 per second, four per cent of the linear tracking reward. In the MJCF the bodies named `knee_*_act` are the knee actuator housings at the hip, and the knee joint itself sits at the top of `shank_*_Link`, so the pattern covers the kneeling contact only through `shank_.*` and `thigh_.*`. The ankle housings `ankle_*_act`, which sit 50 mm above the sole and graze the ground whenever a foot pitches, are rightly excluded.

The termination set holds `base_Link` contact alone (`limx_base_env_cfg.py:1085-1091`, narrowed again at `limx_solefoot_env_cfg.py:54`). A kneeling robot therefore lives the full episode and collects its tracking reward throughout, which is the structural fact on which the rest of this document turns.

Three further terms tilt the balance toward a low posture without causing it. `no_fly` pays 1.5 per second for exactly one loaded foot and charges 7.5 per second for none (`mdp/rewards.py:1598-1634`), and a kneeling robot whose feet rest at the ends of its shanks keeps a foot loaded. `keep_ankle_pitch_zero_in_air` pays 1.0 per second whenever both feet are in contact, its exponent summing over airborne feet only, as its own docstring records (`mdp/rewards.py:1731-1801`). And `feet_air_time` with a lower threshold of 0.2 s pays a negative reward for every step shorter than 0.2 s and nothing for a foot that never lifts (`mdp/rewards.py:338-365`), so it charges the short, tentative steps by which an upright gait would be discovered. `feet_regulation` scales its height gate by the 0.75 m target rather than by a swing height, which its docstring records as a defect left standing (`mdp/rewards.py:2019-2055`), so it charges a swinging foot almost as much as a sliding one.

The LimX lineage from which this configuration descends shows how far the contact price sits from published practice. The VMTS pipeline for the LimX P1 biped carries the same feet distance weight of -100 and a base height weight of -10, but a collision weight of -30, sixty times this configuration's -0.5 [1].

### 3.4 What the videos show

The forty four training videos were sampled into contact sheets of eight frames each, with sixteen frame sheets for the ten videos from step 250000 to 340000 (artefacts `sf_tron_1a_flat-2026-10-09_06-04-18/frames/`). The camera follows environment zero only and does not turn with it, so only side views of that one robot can be read for support.

| Steps | Iterations | Support observed in environment zero |
|---|---|---|
| 0 to 40000 | 0 to 1600 | not readable, every environment piled near the origin |
| 50000 to 110000 | 2000 to 4400 | mostly upright, kneeling in a minority of frames from 60000 |
| 120000 to 350000 | 4800 to 14000 | kneeling dominates whenever the robot moves, 13 of 16 frames at 250000, 300000 and 340000, upright stretches at 290000, 310000 and 320000 that do not persist |
| 380000 to 430000 | 15200 to 17200 | feet increasingly win, first clearly at 390000, in a deep crouch with the thighs near horizontal |

The kneeling posture is unambiguous in the side views (`frames/dense_step_300000.jpg`). The thigh descends steeply from the hip to the knee, the knee rests on the ground, the shank lies flat along it, and the foot rests at its far end, as a person kneels upon the shins. The late gait upon the feet (`frames/sheet_step_430000.jpg`) is the mirror of it, the thigh near horizontal and the shank steep, so both hold the trunk at roughly one link length, which is why a height penalty barely distinguishes them. Environment zero also reaches the edge of the terrain repeatedly and falls from it at 260000.

The visual onset is later than the user's reading of step 310000, which corresponds to one of the upright stretches that did not persist, and later than the time series of section 3.5. The three readings are reconciled there.

### 3.5 What the time series shows

The TensorBoard record was extracted in full and every value below is a centred fifty iteration mean (artefact `tensorboard/scalars_every50.csv.gz`, script `tensorboard/extract.py`, plots `tensorboard/plot_*.svg`). Each episode reward series is the episode sum of the weighted term divided by 20 s, confirmed by the summed terms reproducing `Train/mean_reward` to within one per cent at iterations 2000 and 13000.

| Iteration | Episode length | Base contact share | Linear tracking per s | Base height per s | Implied height error RMS, m | Undesired contacts per step | Noise std | Learning rate | Mean return |
|---|---|---|---|---|---|---|---|---|---|
| 250 | 510 | 0.604 | 3.56 | | | | 0.541 | 3.3e-4 | 79 |
| 2000 | 743 | 0.289 | 13.26 | -0.28 | 0.275 | 0.519 | 0.520 | 8.5e-5 | 266 |
| 6000 | 919 | 0.113 | 18.89 | -0.29 | 0.253 | 0.275 | 0.537 | 4.7e-5 | 399 |
| 10000 | 912 | 0.144 | 17.35 | -0.27 | 0.245 | 0.192 | 0.712 | 3.2e-5 | 331 |
| 12400 | 915 | | | | 0.218 | 0.059 | 0.92 | | |
| 13000 | 894 | 0.184 | 16.59 | -0.20 | 0.211 | 0.037 | 0.982 | 1.9e-5 | 254 |
| 16000 | 882 | 0.180 | 15.25 | -0.16 | 0.193 | 0.027 | 1.255 | 1.2e-5 | 169 |

The undesired contact count falls first, from a global change point at 9350 with a logistic midpoint at 11789 and a ten to ninety per cent interval of 10816 to 12761. The height penalty follows with a midpoint at 12192, `feet_air_time` at 12502, and the episode length and the base contact share lag at about 13950. The population therefore leaves its knees over iterations 10800 to 12800, which agrees with the user's 12400, while environment zero, a single sample, persists until about 15600. The implied height error never falls below 0.19 m, so even the late gait stands at most 0.56 m above the terrain mean against a target of 0.75, consistent with the deep crouch of the videos.

Four further observations bound the interpretation. The knee phase is a stable optimum and not a transient, the return rising to 399 at iteration 6000 and the base contact share falling to 0.11 while the robot knelt. The escape is not an improvement of that optimum, the return falling from 399 to 161 and the value loss rising from 10 to 141 across it. The escape coincides with three changes the run made to itself, the command curricula widening the linear range from 1.0 to 1.27 m/s and the yaw range to 1.35 rad/s from iteration 5760, the push curriculum raising the push velocity from 0.5 to 1.9 m/s by 11000, and the terrain level rising from 0.17 to 0.58. And it coincides with the exploration noise, which fell from 1.0 to 0.52 in the first hundred iterations and stayed there until 8000, rising thereafter to 0.92 at 12400 and 1.31 at 17000 while the adaptive schedule held the learning rate between 1e-5 and 4e-5.

One measurement does not fit. Kneeling on both shanks should register two undesired contacts in every step, yet the implied count was 0.52 per step at iteration 2000 and 0.19 at 10000. Either the population knelt far less than environment zero, or it knelt with its weight shared so that the shank force often fell below 10 N, or the contact sensor does not report the shank loads at all. The last would also blind every contact based remedy of section 6, which is why arm 0 measures it before any arm is run. Resolved 2026-10-19 by arm 0, section 6.2.1. The sensor reports the shank loads, and a replay of the iteration 6000 checkpoint counts 0.270 undesired contacts per environment step against the 0.275 implied here, because the knee mode at that iteration is an asymmetric gait on the right foot and the intermittently loaded left shin rather than a kneel on both shins. The first explanation offered above, that the population knelt less than environment zero, is the correct one.

## 4. Failure modes

The evidence supports five failure modes. They are ranked by the weight of evidence, and the first three are mechanisms while the last two qualify how the first is to be remedied and measured.

F1, the knee mode is admitted by the termination set and unpriced by the reward. A kneeling robot is never terminated, keeps a foot loaded for `no_fly`, and pays at most 2 per cent of its return in height and 4 per cent in contact penalties while collecting most of the tracking reward (sections 3.3 and 3.5). The velocity objective cannot itself prefer the feet, since a knee walker satisfies it [2], and the published account of exactly this failure, a character executing the task motion while lying on the ground, names early termination on the offending contact as its cure [3]. The configuration inherits the quadruped convention of legged_gym, which penalises knee and shank contact and terminates only on the base [4], and which is adequate where knee contact is incidental but not where it is a support mode. This is the primary mechanism.

F2, the actuators make standing expensive and kneeling cheap. The knee and ankle stiffnesses of 60 and 10 Nm/rad sag 0.57 rad each under single support, so standing on one foot requires 2.3 action units on two joints, while kneeling is reached by letting gravity fold the leg, which the free drop test shows to be the passive outcome of a fall (section 3.2). A joint position action space about a fixed default with low torque authority is the documented recipe for a static or low effort local minimum [5], and gains are known to shape learnability as much as performance [6], [7]. This is a contributing mechanism of moderate weight, and it is the only one that the nominal pose touches.

F3, the nominal pose is a minor contributor, not the root cause. The baked bend is twelve degrees deeper than the original LimX geometry and raises the knee load by about five per cent over it, and straightening by 0.2 rad lowers the load fourteen per cent, against a factor of nearly four by which the stiffness falls short (section 3.2). The original geometry kneels at a similar height, so the bend does not create the kneeling posture. The pose does matter in two secondary ways, the height target having been set to the sag of this pose under these gains, and the deep crouch of the late gait inheriting it.

F4, the escape was forced by the curricula and enabled by rising exploration noise. The knee mode was a stable optimum at a return of 399, the escape cost return, and it began when command ranges, pushes and terrain had grown and the noise had risen by eighty per cent (section 3.5). Higher entropy smooths the landscape and connects local optima [8], the initial noise level shapes which modes are sampled [9], and the kneel to stand transition is documented as an exploration bottleneck under Gaussian action noise [10]. The evidence is circumstantial, a single seed, and this failure mode explains the timing of the escape rather than the entry into the knee mode.

F5, the contact measurement may be blind. The implied contact count is a quarter of what the videos predict (section 3.5). If the sensor misses shank loads, F1's penalty was never applied at its nominal price and a contact termination would never fire, so this is a precondition on every remedy and not a cause in its own right. FALSIFIED 2026-10-19 by arm 0, section 6.2.1, the sensor reporting the left shank above 10 N in every sample and the replayed contact count matching the training record to within two per cent.

The causal chain, stated once. The robot spawns, cannot yet balance, and falls toward its shanks, its soft knees offering no resistance. Nothing terminates the kneeling episode, so the critic learns that kneeling states are worth the full tracking stream, and the policy learns to crawl, since the crawl tracks the command and costs almost nothing. The upright alternative needs 2.3 action units of sustained effort on two joints and coordinated balance before it pays more, so it is not found by noise of 0.52. It is found only when the curricula make the crawl inadequate and the noise has grown.

## 5. Literature survey

The survey is targeted at the five failure modes, and it moves from the general account of degenerate optima, through the instruments that exclude them, to the biped pipelines that represent the present practice. Every work is registered in cluster 34 of `/ws/context/literature.md` or cross referenced there to an earlier cluster.

### 5.1 Degenerate optima under a task reward

Heess and colleagues trained humanoids on a velocity reward with termination on falling, defined by the head to feet distance falling below 0.9 m or the head height falling below 1.1 m, and report that the humanoid is prone to exploit redundancies in the task specification and to settle in local optima with visually unsatisfactory gaits, the result depending on the reward, the termination condition and the seed [11]. Both of their termination conditions are geometric floors that a kneeling posture would violate. Reda, Tao and van de Panne isolate the environment design choices responsible, finding that a PD target action space is more prone than torque control to a local minimum because a constant set of targets holding a static posture is easily learned and paid by the survival bonus, that low torque limits make the agent likelier to stick, and that time limit terminations must be bootstrapped [5]. Skalse and colleagues give the formal reason the tracking terms cannot prefer the feet, a proxy being unhackable only if increasing it can never decrease the true return [2]. Fu and colleagues show that gaits emerge only under an energy term [12], a result this document treats as a caution, since a short lever arm kneeling posture may be energetically cheap here. The nearest report of a small biped converging to ground contact locomotion is Haarnoja and colleagues' miniature OP3, whose end to end training converged either to rolling on the ground or to standing without scoring, and which was rescued by a separately trained get up skill and termination on falling [13]. Together these establish that the TRON1 outcome is an expected failure of this design and not an anomaly.

### 5.2 Termination and constraints as the instruments that exclude a mode

DeepMimic terminates an episode when links such as the torso or head touch the ground, describing this both as reward shaping that discourages undesirable behaviour and as a curation of the data, and states that early termination eliminates local optima in which the character falls and mimes the motion while lying on the ground [3]. Its ablation shows the effect is large for acrobatic skills, removing it dropping the backflip from 0.791 to 0.379, and small for walking, 0.980 against 0.974 [3], an asymmetry this document reads as saying that termination matters where a ground contact mode competes with the task. Rudin and colleagues penalise knee and shank contacts and reset only on base contact, and bootstrap time out terminations with a gain of ten to twenty per cent in total reward [4]. Chane-Sane and colleagues recast constraints as stochastic terminations, knee or base collision being a hard constraint at a termination probability of 1.0, and report that a naive binary termination of every constraint, and an ablation making all of them hard, both fail to learn at all [14]. Kim and colleagues formulate a body contact constraint that is zero only for foot contact and report that a robot trained on task rewards alone shows undesirable shank contact which the constraint removes [15]. In biped practice Li and colleagues terminate Cassie on a pelvis height below 0.55 m and on tarsus contact with the ground, and state that reward design alone may be insufficient because the robot can maintain a pose that collects part of the reward [16], Xie and colleagues terminate below a pelvis height of 0.4 m or when the step reward falls below 0.3 [17], and Hwangbo and colleagues tune every cost against a terminal cost of 1 [18]. Pardo and colleagues require that a time limit termination remain bootstrapped while a failure termination is terminal [19], which rsl_rl implements (`/ws/rsl_rl/rsl_rl/algorithms/ppo.py:177-179`). The Isaac Lab biped tasks pair a torso or pelvis contact termination with a terminal penalty of -200 (`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/h1/rough_env_cfg.py:23`, `.../g1/rough_env_cfg.py:23`). This group grounds arms 2 to 5.

### 5.3 Initial state, curricula and exploration

Florensa and colleagues grow the start distribution backward from the goal [20], which clarifies that TRON1's problem is not reaching the upright region, where it already starts, but leaving it. Yu, Turk and Liu apply a virtual assistant force that is withdrawn as skill grows, without which their policies fail to move or balance [21], Tidd and colleagues use guiding forces on a biped that are reduced to zero before perturbations are added, framing the task as the progression from crawling to walking upright [22], and Huang and colleagues identify the transition from falling to stable kneeling and rising as the exploration bottleneck under random action noise, remedying it with an upward base force starting at 200 N and reduced by 20 N steps [10]. Hwangbo and colleagues scale every cost by a curriculum factor starting at 0.3 so that the objective is found before regularisation bites [18]. On exploration, Andrychowicz and colleagues find on-policy performance highly sensitive to the initial action standard deviation [9], Ahmed and colleagues show that higher entropy smooths the optimisation landscape and connects local optima [8], and the survey of Ha and colleagues recommends starting constraints soft and tightening them [23]. This group grounds the timing account of F4 and the fallback arms of section 6.10.

### 5.4 Nominal pose, action space and actuation

Peng and van de Panne show that PD target actions learn faster than torques across motions and morphologies, in a setting with a reference and a good default posture [24]. Siekmann and colleagues add the policy output to the offsets of a neutral standing pose, so that zero action stands [25], which is the convention here. Liao and colleagues' Berkeley Humanoid, the closest published platform to TRON1 at 16 kg and 0.85 m in Isaac Lab, regularises the hip and knee toward their nominal positions and randomises the default joint position by only plus or minus 0.05 rad, a quarter of TRON1's reset offset, and states that mid scale humanoids are harder to control for their low centre of gravity [26]. Bronars, Park and Agrawal argue that gains should be chosen for learnability under the algorithm in use [6], and Spoljaric and colleagues report that a large proportional gain destabilises training while a low one leaves the joint a torque source [7]. This group grounds F2, F3 and arms 6 and 7.

### 5.5 Biped and humanoid pipelines, the state of the art

The present practice for learned biped locomotion reaches the feet by one of three routes, a reference or clock that a knee walker cannot satisfy, a termination or constraint that removes it, or a posture term that prices it heavily. On TRON1 itself, Su and colleagues replace the height penalty by a LIPM constraint plane whose intercept is the standing height [27], and the LimX P1 pipeline of Chen and colleagues adds a phase referenced contact term with a 0.5 s period to its collision weight of -30 [1]. The HIM estimator from which this codebase descends kept the quadruped convention of base contact termination [28], and its humanoid successor measures body height relative to the feet and pays a single foot contact term [29]. Humanoid-Gym tracks base height through an exponential kernel relative to the stance feet and adds a sinusoidal reference with a contact pattern reward [30]. Siekmann and colleagues compose periodic phase gated rewards for every common bipedal gait [25]. Van Marum and colleagues show that a velocity only humanoid finds a degenerate hopping mode and list five individually sufficient exits from it, a base height reward, clock rewards and inputs, tuning the exploration noise, a feet contact transition reward and a single foot contact reward, the last being the most reliable [31]. Li and colleagues track a gait library in which pelvis height is a commanded reference [32], Rodriguez and Behnke pay the alive reward only above a height floor and within roll and pitch bounds on a 19 kg biped [33], and Radosavovic and colleagues train a full size humanoid with a transformer policy and a biomechanically inspired reward [34]. The production pipeline nearest this lineage, unitree_rl_gym's G1, combines a base height weight of -10, a roll and pitch termination and a phase contact reward of period 0.8 s (`/ws/external/unitree_rl_gym/legged_gym/envs/g1/g1_config.py:5-19,63-96`). No pipeline in this survey relies on a light L2 height penalty and base termination alone, which is what TRON1 does.

The position the survey supports is that the knee mode should be excluded by a termination on lower leg contact as the primary instrument, with a terminal price and a height floor as its companions, that the actuator authority should be made adequate to the pose so that the upright mode is cheap to find, and that a clock or a feet relative height term is the route to gait quality once the mode is excluded. Section 6 orders the arms accordingly.

## 6. The ablation

### 6.1 Protocol, preliminary edits and measurement

Every arm is a subclass of `SFIdentifiedBlindRoughEnvUrdfCfg`, the run's own class, in a new module, and changes only what its `__post_init__` sets, so that no existing task changes behaviour and each run's dumped `params/env.yaml` records its arm, as `CLAUDE.md` rules 2 to 4 require. Each arm except arm 0 is a single change relative to arm 1, so that it is compared with arm 1 directly, and arm 8 alone combines the arms that succeed. Every arm runs the run's own seed of 42, its 4096 environments and its terrain, for 6000 iterations, a budget at which the baseline had knelt for 4000 iterations and before the command and push curricula begin at 5760, so that the curricula cannot confound the onset.

Preliminary edit P1 restores the shared configuration. The -10.0 at `environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:952` changes every task that inherits `RewardsCfg` through `SFEnvCfg` and `SFHIMEnvCfg` (`limx_base_env_cfg.py:1316`, `:1347`), which is every SF task except the co-optimisation task, whose `CoptRewardsCfg` overrides the term (`limx_base_env_cfg.py:1050-1057`). Change line 952 back to `weight=-5.0,`, reproducing the run, and let arm 1 carry the -10.0.

Preliminary edit P2 creates `environments/environments/tasks/locomotion/robots/limx_solefoot_ablation_env_cfg.py` with the header below, the arm classes of sections 6.2 to 6.9 being appended to it in order.

```python
"""Ablation arms of plans/tron1_locomotion_fine_tuning.md, the late emergence of a bipedal gait.

Every arm subclasses SFIdentifiedBlindRoughEnvUrdfCfg, the configuration of run
sf_tron_1a_flat/2026-10-09_06-04-18, and sets only what its own __post_init__ sets, so that no
existing task changes and each run's dumped params/env.yaml records its arm.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from environments.tasks.locomotion import mdp

from .limx_solefoot_env_cfg import SFIdentifiedBlindRoughEnvUrdfCfg

# The links whose ground contact is kneeling. The knee joint sits at the top of shank_*_Link and
# the thigh box ends 25 mm short of it (base_robot.xml:63-72), so a kneeling robot loads one or
# both. knee_*_act is the knee housing at the hip and ankle_*_act grazes the ground whenever a
# foot pitches, so neither is listed.
LOWER_LEG_LINKS = ["thigh_.*_Link", "shank_.*_Link"]
```

Preliminary edit P3 registers the arms. In `environments/environments/tasks/locomotion/robots/__init__.py`, add `limx_solefoot_ablation_env_cfg,` to the import list at lines 23 to 31, after `limx_solefoot_env_cfg,`, and append the loop below after line 795.

```python
############################
# SF ablation of plans/tron1_locomotion_fine_tuning.md
############################
for _arm in ("Arm1", "Arm2", "Arm3", "Arm4", "Arm5", "Arm6", "Arm7"):
    gym.register(
        id=f"Isaac-Limx-SF-Ablation-{_arm}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": getattr(limx_solefoot_ablation_env_cfg, f"SFAblation{_arm}EnvCfg"),
            "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
        },
    )
```

Every arm is launched from `tron1-rl-isaaclab-cozum/` with the command below, the experiment name keeping the arms apart from the run's own directory.

```bash
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Ablation-Arm<k>-v0 --headless --seed 42 \
    --max_iterations 6000 --video --video_interval 10000 --video_length 2000 \
    --experiment_name sf_tron_1a_rough_ablation --run_name arm<k>
```

The measurement reuses `context/artefacts/sf_tron_1a_flat-2026-10-09_06-04-18/tensorboard/extract.py` with its `RUN`, `EV` and `WEIGHTS` constants set to the arm's run, event file and dumped weights. Four quantities are read from it and one from the videos.

| Metric | Definition | Baseline |
|---|---|---|
| M1, the onset | first iteration at which the implied undesired contacts per step, `Episode_Reward/pen_undesired_contacts` divided by the weight and by `Train/mean_episode_length / 1000`, stays below 0.05 for 500 iterations | about 12500 |
| M2, the height | implied RMS height error, `Episode_Reward/pen_base_height` divided by the weight and the alive fraction, square rooted, at 4000 and 6000 | 0.253 m at 6000 |
| M3, the tracking | `Episode_Reward/rew_lin_vel_xy` at 4000 and 6000 | 18.89 per s at 6000 |
| M4, the cost of the remedy | `Episode_Termination/<term>` share and `Train/mean_episode_length` at 1000, 2000 and 6000 | base contact 0.29 at 2000 |
| M5, the visual onset | first video in which environment zero walks on its feet in the majority of side view frames | step 390000 |

An arm that terminates on contact makes M1 a biased estimate, since a kneeling robot no longer survives to be counted, so for such arms M1 is read together with M4, the termination share falling below 0.05 marking the onset instead.

### 6.2 Arm 0, the contact measurement that every contact remedy presupposes

Hypothesis. The contact sensor reports the shank load of a kneeling robot above 10 N, so that F5 is false and the contact based arms can act. Its role is a precondition, not a remedy, and it costs no training.

Change. None to the configuration. The probe below stands in `scripts/rsl_rl/play.py` immediately after `obs, rewards, dones, infos = env.step(actions)` at line 1181, inserted on 2026-10-19, and is to be removed once the arm is read. It prints, every fifty steps, the share of environments in which each body's contact force exceeds 10 N.

```python
            # Arm 0 of plans/tron1_locomotion_fine_tuning.md, the share of environments in
            # which each body's contact force exceeds 10 N. Remove after use.
            if env.unwrapped.common_step_counter % 50 == 0:
                cs = env.unwrapped.scene.sensors["contact_forces"]
                peak = cs.data.net_forces_w_history.norm(dim=-1).amax(dim=1)  # (envs, bodies)
                share = (peak > 10.0).float().mean(dim=0)
                print({n: round(float(s), 3) for n, s in zip(cs.body_names, share)})
```

The script was made ready for the TRON1 on 2026-10-19 by three findings, section 6.2.2 recording the review in full. Its robot profile table matched no TRON1 body, so a TRON1 replay had logged its sole clearance and tilt against the SD_BRS1 sole, and a TRON1 row now carries the eight vertex sole table and the minus z normal. The play loop runs for `--video_length` steps whether or not a video is recorded (`scripts/rsl_rl/play.py:1195-1197`), so the default of 200 must be raised to 500 explicitly. And the rough play task spawns at random terrain levels including the hardest, which the kneeling policy never met, so the flat play task of the same robot is the primary measurement and the rough one the confirmation. The run's rewards, gains and action scale are restored from its `params/env.yaml` by `experiment_params`, which reads this run's dump correctly, its `pen_base_height` restored at -5.0 over the tree's -10.0.

```bash
python scripts/rsl_rl/play.py --task Isaac-Limx-SF-Identified-Blind-Flat-Play-Urdf-v0 --num_envs 1024 --headless \
    --video_length 500 --checkpoint_path /ws/IsaacLab/logs/rsl_rl/sf_tron_1a_flat/2026-10-09_06-04-18/model_6000.pt
python scripts/rsl_rl/play.py --task Isaac-Limx-SF-Identified-Blind-Rough-Play-Urdf-v0 --num_envs 1024 --headless \
    --video_length 500 --checkpoint_path /ws/IsaacLab/logs/rsl_rl/sf_tron_1a_flat/2026-10-09_06-04-18/model_6000.pt
```

A video of the same replay, which the decision below needs, is taken by repeating the flat command with `--num_envs 32 --video`, which enables the cameras itself (`scripts/rsl_rl/play.py:65-66`). Each command writes its dump to `data/42/` of the run directory, overwriting the previous one, so the dump of the flat replay is to be copied out before the rough one is run.

Decision. The shank bodies are known to be in the sensor's body list, since the run's `pen_undesired_contacts` resolved the pattern `shank_.*` against that sensor, and Isaac Lab refuses an unmatched pattern when it resolves one (`IsaacLab/source/isaaclab/isaaclab/utils/string.py:178`), so the question is only whether their force is reported. If the shank shares stay near zero while the play video shows kneeling, the sensor's force reporting on the MJCF bodies at `limx_base_env_cfg.py:85-90` is the first defect to repair and arms 2 and 3 must wait for it. If the shares are substantial, F5 is false, the low implied count of section 3.5 is explained by the population's mix, and the arms proceed. If they are small but non zero while the robot kneels, the shank load sits below 10 N and the threshold of arms 2 and 3 should be lowered to 1 N, the value the base termination uses.

#### 6.2.1 Results, run 2026-10-19

Arm 0 was run on 2026-10-19 against `model_6000.pt` of the baseline run, a checkpoint from the middle of the knee phase. The replay departed from the command of section 6.2 in three respects, and each is recorded so that the figures below are read correctly. It used the rough play task `Isaac-Limx-SF-Identified-Blind-Rough-Play-Urdf-v0`, whose configuration the log names as `SFIdentifiedBlindRoughEnvUrdfCfg_PLAY`, rather than the flat one. It ran 32 environments rather than 1024, so every share is a multiple of 1/32, 0.031. And it ran with the video on and a length of 3000 control steps, so the probe printed sixty samples at fifty step intervals rather than ten. The log confirmed the integration signatures of section 6.2.2 in full. Twenty reward terms, four actuator groups and one action term were restored from the run's parameters, `pen_base_height` stood at -5.0, the actor accepted the 360 wide observation, the feet resolved as `'foot_[LR]_Link' -> ['foot_R_Link', 'foot_L_Link']` with a sole table of 8 points and the normal (0.0, 0.0, -1.0), and the statistics were written to `data/42/statistics.npy` of the run directory. The probe output is preserved as `../context/artefacts/sf_tron_1a_flat-2026-10-09_06-04-18/arm0/contact_probe_model_6000.txt`.

The table gives, for each body, the share of the 32 environments whose peak contact force over the sensor's four sample history exceeded 10 N, summarised over the sixty samples.

| Body | Mean | Median | Min | Max | Share of samples non zero | In the undesired contact set |
|---|---|---|---|---|---|---|
| `base_Link` | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | yes, and terminating |
| `abad_R_act` | 0.027 | 0.031 | 0.000 | 0.062 | 0.63 | yes |
| `abad_L_act` | 0.006 | 0.000 | 0.000 | 0.031 | 0.20 | yes |
| `hip_R_act`, `hip_L_act` | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | yes |
| `knee_R_act`, `knee_L_act` | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | yes |
| `thigh_R_Link` | 0.001 | 0.000 | 0.000 | 0.031 | 0.02 | yes |
| `thigh_L_Link` | 0.004 | 0.000 | 0.000 | 0.031 | 0.13 | yes |
| `shank_R_Link` | 0.027 | 0.031 | 0.000 | 0.125 | 0.58 | yes |
| `shank_L_Link` | 0.205 | 0.219 | 0.094 | 0.344 | 1.00 | yes |
| `ankle_R_act` | 0.036 | 0.031 | 0.000 | 0.094 | 0.73 | no |
| `ankle_L_act` | 0.007 | 0.000 | 0.000 | 0.062 | 0.22 | no |
| `foot_R_Link` | 0.808 | 0.796 | 0.625 | 0.969 | 1.00 | no |
| `foot_L_Link` | 0.557 | 0.546 | 0.312 | 0.781 | 1.00 | no |

| Aggregate per environment step | Arm 0 replay | Training, iteration 6000, implied (section 3.5) |
|---|---|---|
| Bodies of the undesired contact set above 10 N | 0.270, range 0.125 to 0.500 | 0.275 |
| Thighs and shanks above 10 N | 0.237 | |
| Feet above 10 N | 1.365 | |

Three findings follow. First, the sensor reports the lower leg loads, every sample carrying a non zero left shank share and the thighs registering occasionally, so F5 is false and the contact based arms can act on the existing sensor. Second, the replayed count of undesired contacts, 0.270 per environment step, agrees with the 0.275 the training record implies at the same iteration to within two per cent, so the low count of section 3.5 was a true property of the population and not an artefact of measurement. Third, the knee mode at iteration 6000 is not a kneel on both shins but an asymmetric gait upon the right foot and the left shin. The left shank is loaded in a mean of 20.5 per cent of environments against 2.7 per cent for the right, the right foot in 80.8 per cent against 55.7 for the left, and the pattern holds across the replay, the left shank's share being 0.200 over the first thirty samples and 0.210 over the last thirty. Since the shank is loaded at any instant in about a fifth of the population, kneeling in this policy is an intermittent support taken in stride rather than a posture held throughout, which is also why environment zero, the only robot the training videos follow, looked more persistently knelt than the population.

The ankle housings register as predicted in section 3.3, the right one in 3.6 per cent of environments, and they stay outside both the undesired contact set and the termination set of arm 2. The abduction housings register in 2.7 and 0.6 per cent, and since `abad_.*` lies in the undesired contact set, arm 4's price applies to them while arm 2's termination, which names only the thighs and shanks, does not.

Decision. The arms proceed, with the 10 N threshold of arms 2 and 4 unchanged, since the shank load clears it in every sample. The single knee asymmetry predicts that arm 2 will terminate on the left shin first, so the arm 2 results should report its termination share per side, which the contact probe gives directly if it is retained for that replay. The two departures in terrain and population leave the conclusion standing, the rough play task spawning across all terrain levels and without pushes while still reproducing the training count, but the flat replay at 1024 environments specified in section 6.2 remains available should a finer resolution of the per side shares be wanted. The probe still stands in `scripts/rsl_rl/play.py` after `env.step` and is to be removed, or retained deliberately for the arm 2 replay, before the next evaluation.

#### 6.2.2 The review of play.py for the TRON1, 2026-10-19

The replay path was read end to end against this run before arm 0 was launched, and every finding is recorded here with its disposition.

| Item | Finding | Disposition |
|---|---|---|
| Foot resolution | The profile row commented as the TRON1's, `foot_.._Link`, requires two characters between `foot_` and `_Link` and so matches the quadruped's `foot_FL_Link` to `foot_RR_Link` and not the TRON1's `foot_R_Link` and `foot_L_Link`. A TRON1 replay therefore fell through to the fallback, resolving its feet only because the fallback pattern happens to be `foot_[LR]_Link`, and logged sole clearance and tilt against the SD_BRS1 table, a sole 74 mm too deep | A row `foot_[LR]_Link` with the TRON1 table and normal added before the quadruped row, which is left unchanged so that quadruped dumps are unaffected (`scripts/rsl_rl/play.py:236`). The resolution was simulated against the body names of all four robots, each reaching its own row |
| Sole table | The foot collision is the unrotated box `foot_[RL]_collision` of half extents (0.100, 0.035, 0.015) at (0.010, 0, -0.035) in the foot frame (`base_robot.xml:83`, `:132`) | `TRON1_SOLE_OFFSETS` holds its eight vertices, the sole face at z -0.050 over x -0.090 to 0.110 and y plus and minus 0.035, the upper face at z -0.020 (`scripts/rsl_rl/play.py:213-222`). It reproduces the box's lowest point to 2.2e-16 m in MuJoCo over the full range of all four joints of both legs, where the four sole corners alone err by 9.9 mm once a foot pitches past ninety degrees |
| Sole normal | The foot frame is unpermuted and the sole level at the nominal pose, MuJoCo reading the link's minus z as (-6.4e-05, 0, -1) in the world | `TRON1_SOLE_NORMAL = (0.0, 0.0, -1.0)` (`scripts/rsl_rl/play.py:227`) |
| Sole depth | 0.050 m below the foot body origin, which `biped_physical_analysis.py` reports independently | Consistent with the 0.050 the co-optimisation `pen_feet_regulation` carries (`limx_base_env_cfg.py:1064`), and not with the 0.03 `foot_radius` of the plain one (`limx_base_env_cfg.py:989`), recorded as defect 10 of section 7 |
| Imports | `highest_terrain_camera_wrapper`, `co_optimisation.runners` with `CMAESDesignGenerator`, `RandomDesignGenerator` and `DEFAULT_PARAM_RANGES`, `himloco`, `experiment_params` and `stats` are all present in the tree | None needed |
| Runner | The default `--policy-type PPO` builds `OnPolicyRunner` as training did (`scripts/rsl_rl/train.py:224-226`, `scripts/rsl_rl/cli_args.py:61-66`) | None needed |
| Run parameters | `experiment_params.load_params` reads this run's dump, twenty reward terms with `pen_base_height` at -5.0, the four actuator groups at 55 and 13.5, 80 and 13.0, 60 and 4.0, 10 and 0.5, and the action scale 0.25 about the default offset | None needed, the replay restores the trained configuration over the tree's |
| Observations | The play classes switch off corruption, pushes and the base mass event and widen the linear command to 1.35 m/s, and change no observation term, so the trained actor's input width is preserved (`limx_solefoot_env_cfg.py:208-228`) | None needed |
| Loop length | The loop breaks after `--video_length` steps, default 200, with or without `--video` (`scripts/rsl_rl/play.py:1195-1197`) | Pass `--video_length 500` |
| Terrain | The rough play task spawns at random levels with the terrain curriculum off, and `_sole_clearance` measures against world z, so on stairs and slopes the clearance is not a clearance | The flat play task `Isaac-Limx-SF-Identified-Blind-Flat-Play-Urdf-v0` is the primary measurement, the rough task the confirmation |
| Downstream | `stats.py` treats a missing sole channel as the frame height and a present one as the sole, and its docstring now records that TRON1 dumps written before 2026-10-19 carry the SD_BRS1 table's figure and must be read as the frame height | Docstring amended, no logic changed |

Nothing in this review could be executed in the analysis container, which has no Isaac Sim, so the first launch is itself the integration test. Its expected signatures are the line `[play] resolved feet pattern 'foot_[LR]_Link' -> ['foot_L_Link', 'foot_R_Link']` in whichever order the articulation yields them, a sole table of 8 points, and on the flat task a `feet_sole_clearances` channel reading about zero for a planted foot.

### 6.3 Arm 1, the base height penalty doubled

Hypothesis. The height price is too low, and doubling it moves the onset. This is the step the working tree already carries, and the arithmetic of section 3.3 predicts that it will not suffice, since it doubles a price of one to two per cent of the return to two to four. Its value is that it tests the cheapest remedy and falsifies or confirms that prediction, and it serves as the comparator of every later arm. Its literature is the negative result that no surveyed pipeline relies on a light L2 height term alone (section 5.5), van Marum and colleagues listing a base height reward among the exits from a degenerate mode only in an exponential form [31].

Change. Append to the module of P2.

```python
@configclass
class SFAblationArm1EnvCfg(SFIdentifiedBlindRoughEnvUrdfCfg):
    """Arm 1, pen_base_height at -10.0 instead of the run's -5.0."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pen_base_height.weight = -10.0
```

Prediction and decision. M1 within 2000 iterations of the baseline's onset would confirm the prediction that the price is not the lever. An onset before 4000 would falsify it, and the later arms would then be read as competing with a price that works.

#### 6.3.1 Results

Pending.

### 6.4 Arm 2, termination on lower leg contact

Hypothesis. F1, the knee mode persists because it is never terminated. Terminating on a thigh or shank contact removes the kneeling states from the data and gives them zero future value, which is the instrument DeepMimic names for this failure [3], the termination Li and colleagues use on Cassie's tarsus [16], and a hard constraint in the sense of CaT [14] and of Kim and colleagues [15]. Only this one constraint is made hard, after CaT's finding that making every constraint hard prevents learning [14]. The time out stays bootstrapped [19].

Change. Append to the module.

```python
@configclass
class SFAblationArm2EnvCfg(SFAblationArm1EnvCfg):
    """Arm 2, arm 1 plus termination on any thigh or shank contact above 10 N."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.lower_leg_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=LOWER_LEG_LINKS),
                "threshold": 10.0,
            },
        )
```

Design rationale. `mdp.illegal_contact` is Isaac Lab's own term (`IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py:154-162`), so no new code is written. The 10 N threshold matches the undesired contact term, tolerating a graze in swing while firing on any load bearing contact, and arm 0 decides whether it should be lowered. `time_out` is left as the only bootstrapped termination, the new term defaulting to `time_out=False`. The thigh is listed as well as the shank because the sprawled variant of the knee mode seen at steps 410000 and 420000 rests on the thighs.

Prediction and decision. The termination share should rise sharply in the first thousand iterations and then fall below 0.05 as the policy learns to stay on its feet, with M5 before step 150000. If the share stays high and the return collapses, the policy is accepting termination because its expected future reward is not positive, and arm 3 adds the terminal price that CaT assumes and the Isaac Lab bipeds supply.

#### 6.4.1 Results

Pending.

### 6.5 Arm 3, a terminal price on arm 2

Hypothesis. Termination deters only while the expected future reward is positive [14], and the run's return at iteration 250 was 79 over an episode, about 0.15 per step, so a policy might still prefer an early exit to an uncertain upright episode. A terminal penalty makes the exit itself cost.

Change. Append to the module.

```python
@configclass
class SFAblationArm3EnvCfg(SFAblationArm2EnvCfg):
    """Arm 3, arm 2 plus a terminal penalty of -200 on every non time out termination."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pen_termination = RewTerm(func=mdp.is_terminated, weight=-200.0)
```

Design rationale. `mdp.is_terminated` returns one on a failure termination and zero on a time out (`IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:37`), and -200 is the weight of the Isaac Lab H1, G1 and Cassie tasks cited in section 5.2, so that it is in the same per second units, -4 per termination after the control period. Arm 3 is run only if arm 2's termination share fails to fall below 0.10 by iteration 2000.

#### 6.5.1 Results

Pending.

### 6.6 Arm 4, pricing the contact instead of terminating it

Hypothesis. A contact price of the magnitude the LimX P1 pipeline uses [1] removes the knee mode without truncating episodes. This arm is the alternative to arm 2, not an addition to it, and it tests the constraint against the price, which Kim and colleagues and CaT argue the constraint wins [15], [14].

Change. Append to the module.

```python
@configclass
class SFAblationArm4EnvCfg(SFAblationArm1EnvCfg):
    """Arm 4, arm 1 with pen_undesired_contacts at -30.0 instead of -0.5."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pen_undesired_contacts.weight = -30.0
```

Design rationale. At -30, kneeling on both shanks costs 60 per second, more than the entire tracking reward of 32.5, so the knee mode loses money at any tracking quality. The term also counts `abad_.*`, `hip_.*` and `knee_.*` housings and `base_Link`, whose contacts are rarer, and the price applies to them equally, which is deliberate.

Prediction and decision. If M1 matches arm 2, the price suffices and is preferred for not truncating data. If M1 is markedly later, the constraint is the instrument, as the literature predicts.

#### 6.6.1 Results

Pending.

### 6.7 Arm 5, a height floor termination

Hypothesis. A floor beneath the standing height but above every kneeling height excludes the knee mode without depending on the contact sensor, the instrument of Li and colleagues, Xie and colleagues and DeepWalk [16], [17], [33]. It is the fallback should arm 0 find the sensor blind.

Change. Append to the module.

```python
@configclass
class SFAblationArm5EnvCfg(SFAblationArm1EnvCfg):
    """Arm 5, arm 1 plus termination when the root falls below 0.55 m above the terrain mean."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.root_height_floor = DoneTerm(
            func=mdp.root_height_below_minimum_rough,
            params={"minimum_height": 0.55, "sensor_cfg": SceneEntityCfg("height_scanner")},
        )
```

Design rationale. `root_height_below_minimum_rough` already exists and measures against the height scan as the height penalty does (`environments/environments/tasks/locomotion/mdp/terminations.py:15-37`), so no code is written. The floor of 0.55 m sits 8 mm above the highest kneeling posture of 0.5418 m and 0.25 m below the standing height, and the late crouched gait of the run stood at least 0.56 m above the terrain mean (section 3.5). The margin above the kneeling height is thin, a pitched trunk or uneven terrain could lift a kneeling root above it, which is why arm 5 is the fallback and arm 2 the primary.

#### 6.7.1 Results

Pending.

### 6.8 Arm 6, actuator gains sized to the stance load

Hypothesis. F2, the soft knee and ankle make standing expensive and kneeling cheap. Gains that hold the single support load within 0.15 rad make zero action stand at 0.79 m instead of 0.75, and reduce the action needed to stand on one leg from 2.3 units to 0.6. This arm runs without any termination change so that it isolates F2, and its literature is the learnability argument for gains [6], [7] and the observation that low torque authority favours static minima [5].

Change. Append to the module.

```python
@configclass
class SFAblationArm6EnvCfg(SFAblationArm1EnvCfg):
    """Arm 6, arm 1 with the gains of biped_stance_analysis.py sized to the single support load."""

    def __post_init__(self):
        super().__post_init__()
        gains = {"abad": (115.7, 17.07), "hip": (121.2, 16.07), "knee": (229.4, 15.82), "ankle": (37.9, 1.14)}
        # A fresh dictionary of fresh actuator configurations, so that the module level
        # SOLEFOOT_IDENTIFIED_CFG of solefoot_identified_cfg.py:100 is not mutated.
        self.scene.robot.actuators = {
            name: cfg.replace(stiffness={".*": gains[name][0]}, damping={".*": gains[name][1]})
            for name, cfg in self.scene.robot.actuators.items()
        }
```

Design rationale. The figures are the proposed block of `stance_analysis_mjcf.txt`, every damping the least of the stance target, the swing critical value and the integration ceiling, so no joint is overdamped in swing or unstable under the randomised envelope. The startup gain randomisation scales the configured value (`limx_base_env_cfg.py:756-805`), so it follows the new gains without edit. These gains depart from the identified hardware set at `solefoot_identified_cfg.py:14-67`, the knee by a factor of 3.8, so arm 6 is a test of F2 and not a proposal to deploy, and a positive result would be followed by an intermediate set, knee 120 and ankle 25, before any hardware use.

Prediction and decision. If F2 is a contributor, M1 is earlier than arm 1's without any termination, and M2 is smaller at equal iteration. If arm 6 matches arm 1, actuator authority is not on the causal path and F1 stands alone.

#### 6.8.1 Results

Pending.

### 6.9 Arm 7, the nominal pose straightened to the original LimX knee angle

Hypothesis. F3, the deeper baked bend contributes a little. Straightening the knee by 0.2 rad through the default joint position, the hip and ankle following by 0.099 and 0.101 rad, restores the original 119.2 degree knee angle and lowers the knee load fourteen per cent. The prediction is a small effect, which would close the user's question about the pose. The default is changed in the task rather than in the generator, because `base_robot.xml` is shared by every SF task and the co-optimisation runner, as `CLAUDE.md` rule 3 requires. The design generator's `thigh_bend_offset`, `knee_bend_offset` and `ankle_bend_offset` (`design_generator/biped/generate.py:146-148`) are the route for a later geometric change, the equivalent rotations about +y being -5.7, +11.5 and -5.8 degrees as the bend sweep reports.

Change. Append to the module.

```python
@configclass
class SFAblationArm7EnvCfg(SFAblationArm1EnvCfg):
    """Arm 7, arm 1 with the knee default straightened 0.2 rad and the targets moved with it."""

    def __post_init__(self):
        super().__post_init__()
        pose = {"abad_[LR]_Joint": 0.0, "hip_[LR]_Joint": 0.099, "knee_[LR]_Joint": -0.200, "ankle_[LR]_Joint": 0.101}
        self.scene.robot.init_state = self.scene.robot.init_state.replace(pos=(0.0, 0.0, 0.86), joint_pos=pose)
        self.rewards.pen_base_height.params["target_height"] = 0.80
        self.rewards.pen_feet_regulation.params["base_height_target"] = 0.80
        self.rewards.rew_keep_ankle_pitch_zero_in_air.params["use_default_offset"] = True
```

Design rationale. The offsets are the row d_knee of -0.20 of the bend sweep, and `scripts/analysis/biped_configs/tron1_sf_arm7_nominal.json` reproduces its analysis (artefact `stance_analysis_mjcf_arm7_nominal.txt`). The pose stands at 0.8428 m and sags to 0.8057 m under the configured gains on the gantry, so the height target moves from 0.75 to 0.80 to keep the run's convention of targeting the sagged height, and the spawn moves to 0.86 m, the standing height plus a settling margin, which also removes the 4.4 mm embedding of section 3.1. The ankle reward is given its existing optional argument `use_default_offset`, whose default reproduces the old behaviour (`mdp/rewards.py:1731-1781`), because a non zero ankle default would otherwise be charged as a deviation. The joint offsets keep every joint inside its soft limits, the knee retaining 0.894 rad of flexion headroom.

Prediction and decision. M1 earlier than arm 1's by less than a quarter would confirm F3 as minor. A large effect would reopen the geometry and justify regenerating the MJCF with the generator's bend offsets.

#### 6.9.1 Results

Pending.

### 6.10 Arm 8 and the fallbacks, combining what works and shaping the gait

Arm 8 is written only after arms 2 to 7 are read. It combines the termination instrument that succeeded, arm 2 or 4 or 5, with arm 6 or arm 7 if either shortened the onset, and it is run for the full 30000 iterations with the curricula active, its purpose being the speed and quality of the final gait rather than the onset.

Two further directions are held in reserve and are not to be run unless arms 2 to 7 leave the onset later than 2000 iterations. The first is a phase clock, enabling the gait command and `GaitReward` that the configuration already carries commented out (`limx_base_env_cfg.py:102-111`, `:231-232`, `:1002-1016`), the route of every LimX and Unitree biped pipeline surveyed [1], [25], [30]. The second is an assistive upward base force withdrawn by a curriculum, in the manner of Yu, Turk and Liu, of Tidd and colleagues and of HoST [21], [22], [10], which would need a new event term and would be specified in a later revision. A third, an exploration change, raising the floor of the action noise or fixing the learning rate against the adaptive schedule's collapse to 1e-5, is suggested by F4 [8], [9] but rests on circumstantial evidence and would be specified only if the onset of arm 2 still waits for the noise to rise.

#### 6.10.1 Results

Pending.

## 7. Defects recorded and left standing

The following are recorded so that other implementations may opt in deliberately, as `CLAUDE.md` rule 6 requires. None is changed by this plan outside the ablation classes.

1. The shared `RewardsCfg` carried the uncommitted -10.0 at `limx_base_env_cfg.py:952`, which P1 restores to -5.0.
2. The spawn height of 0.8 m at `solefoot_identified_cfg.py:86` embeds the soles 4.4 mm, arm 7 alone raising it.
3. The height target of 0.75 m is the sag of the nominal pose under the identified gains, 54.4 mm below the standing height, and is shared by every SF task.
4. `feet_regulation` scales its height gate by the base height target, a defect its own docstring records.
5. `keep_ankle_pitch_zero_in_air` pays a standing bonus whenever both feet are loaded, its `require_airborne` argument defaulting to the old behaviour.
6. `feet_air_time` with a 0.2 s lower threshold charges every short step and pays nothing for a foot that never lifts, which charges the discovery of a gait.
7. The simplified URDF's left knee range disagrees with its right knee and with the MJCF (`base_robot.urdf:262`, `:528`), and its comments at `:250-254` are stale.
8. The run is logged under `sf_tron_1a_flat` although it trains on rough terrain, its runner configuration being shared with the flat tasks (`robots/__init__.py:331`).
9. The curricula count 24 steps to an iteration where the runner takes 25, which `CoptCurriculumCfg` corrects for the co-optimisation task alone (`limx_base_env_cfg.py:1216-1297`).
10. The plain `pen_feet_regulation` gives the TRON1 a `foot_radius` of 0.03 m (`limx_base_env_cfg.py:989`) against a sole depth of 0.050 m below the foot body origin, so it reads a planted foot as 20 mm airborne, the co-optimisation variant alone carrying 0.050 (`limx_base_env_cfg.py:1064`).

## 8. References

Entries are ordered by first appearance in the text. Software is cited in the text by `file:line` and carries no entry here.

[1] Chen, F., Wan, R., Liu, P., Zheng, N., Zhou, B. VMTS, Vision-Assisted Teacher-Student Reinforcement Learning for Multi-Terrain Locomotion in Bipedal Robots. arXiv preprint, 2025, venue not stated by the retrieved record. arXiv:2503.07049.

[2] Skalse, J., Howe, N. H. R., Krasheninnikov, D., Krueger, D. Defining and Characterizing Reward Hacking. Advances in Neural Information Processing Systems (NeurIPS), 2022. arXiv:2209.13085.

[3] Peng, X. B., Abbeel, P., Levine, S., van de Panne, M. DeepMimic, Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills. ACM Transactions on Graphics 37(4), SIGGRAPH, 2018. DOI 10.1145/3197517.3201311, arXiv:1804.02717.

[4] Rudin, N., Hoeller, D., Reist, P., Hutter, M. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Conference on Robot Learning (CoRL), PMLR 164, 2021. arXiv:2109.11978.

[5] Reda, D., Tao, T., van de Panne, M. Learning to Locomote, Understanding How Environment Design Matters for Deep Reinforcement Learning. ACM SIGGRAPH Conference on Motion, Interaction and Games (MIG), 2020. DOI 10.1145/3424636.3426907, arXiv:2010.04304.

[6] Bronars, Park, Agrawal. Tune to Learn, How Controller Gains Shape Robot Policy Learning. arXiv preprint, 2026, author initials and venue not stated by the retrieved record. arXiv:2604.02523.

[7] Spoljaric, Yashuai, Lee. Variable Stiffness for Robust Locomotion through Reinforcement Learning. 16th IFAC Joint Symposia of Mechatronics and Robotics, 2025, author initials not stated by the retrieved record. arXiv:2502.09436.

[8] Ahmed, Z., Le Roux, N., Norouzi, M., Schuurmans, D. Understanding the Impact of Entropy on Policy Optimization. International Conference on Machine Learning (ICML), PMLR 97, pages 151 to 160, 2019. arXiv:1811.11214.

[9] Andrychowicz, M., Raichuk, A., Stańczyk, P., Orsini, M., Girgin, S., Marinier, R., Hussenot, L., Geist, M., Pietquin, O., Michalski, M., Gelly, S., Bachem, O. What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study. arXiv preprint, 2020, venue not stated by the retrieved record. arXiv:2006.05990.

[10] Huang, T., Ren, J., Wang, H., Wang, Z., Ben, Q., Wen, M., Chen, X., Li, J., Pang, J. Learning Humanoid Standing-up Control across Diverse Postures. Robotics, Science and Systems (RSS), 2025. DOI 10.15607/RSS.2025.XXI.064, arXiv:2502.08378.

[11] Heess, N., TB, D., Sriram, S., Lemmon, J., Merel, J., Wayne, G., Tassa, Y., Erez, T., Wang, Z., Eslami, S. M. A., Riedmiller, M., Silver, D. Emergence of Locomotion Behaviours in Rich Environments. arXiv preprint, 2017, venue not stated by the retrieved record. arXiv:1707.02286.

[12] Fu, Z., Kumar, A., Malik, J., Pathak, D. Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots. Conference on Robot Learning (CoRL), 2021. arXiv:2111.01674.

[13] Haarnoja, T., Moran, B., Lever, G., Huang, S. H., Tirumala, D., Humplik, J., Wulfmeier, M., Tunyasuvunakool, S., Siegel, N. Y., Hafner, R., Bloesch, M., Hartikainen, K., Byravan, A., Hasenclever, L., Tassa, Y., Sadeghi, F., Batchelor, N., Casarini, F., Saliceti, S., Game, C., Sreendra, N., Patel, K., Gwira, M., Huber, A., Hurley, N., Nori, F., Hadsell, R., Heess, N. Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning. Science Robotics 9(89), 2024. DOI 10.1126/scirobotics.adi8022, arXiv:2304.13653.

[14] Chane-Sane, E., Leziart, P.-A., Flayols, T., Stasse, O., Souères, P., Mansard, N. CaT, Constraints as Terminations for Legged Locomotion Reinforcement Learning. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 13303 to 13310, 2024. DOI 10.1109/IROS58592.2024.10802334, arXiv:2403.18765.

[15] Kim, Y., Oh, H., Lee, J., Choi, J., Ji, G., Jung, M., Youm, D., Hwangbo, J. Not Only Rewards But Also Constraints, Applications on Legged Robot Locomotion. IEEE Transactions on Robotics 40, pages 2984 to 3003, 2024. DOI 10.1109/TRO.2024.3400935, arXiv:2308.12517.

[16] Li, Z., Peng, X. B., Abbeel, P., Levine, S., Berseth, G., Sreenath, K. Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control. International Journal of Robotics Research 44(5), pages 840 to 888, 2024. DOI 10.1177/02783649241285161, arXiv:2401.16889.

[17] Xie, Z., Clary, P., Dao, J., Morais, P., Hurst, J., van de Panne, M. Iterative Reinforcement Learning Based Design of Dynamic Locomotion Skills for Cassie. arXiv preprint, 2019, its Conference on Robot Learning version appearing as Learning Locomotion Skills for Cassie, Iterative Design and Sim-to-Real, PMLR 100, pages 317 to 329. arXiv:1903.09537.

[18] Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., Hutter, M. Learning Agile and Dynamic Motor Skills for Legged Robots. Science Robotics 4(26), eaau5872, 2019. DOI 10.1126/scirobotics.aau5872, arXiv:1901.08652.

[19] Pardo, F., Tavakoli, A., Levdik, V., Kormushev, P. Time Limits in Reinforcement Learning. International Conference on Machine Learning (ICML), PMLR 80, pages 4045 to 4054, 2018. arXiv:1712.00378.

[20] Florensa, C., Held, D., Wulfmeier, M., Zhang, M., Abbeel, P. Reverse Curriculum Generation for Reinforcement Learning. Conference on Robot Learning (CoRL), PMLR 78, pages 482 to 495, 2017. arXiv:1707.05300.

[21] Yu, W., Turk, G., Liu, C. K. Learning Symmetric and Low-energy Locomotion. ACM Transactions on Graphics 37(4), Article 144, SIGGRAPH, 2018. DOI 10.1145/3197517.3201397, arXiv:1801.08093.

[22] Tidd, B., Hudson, N., Cosgun, A. Guided Curriculum Learning for Walking Over Complex Terrain. Australasian Conference on Robotics and Automation (ACRA), 2020, per the arXiv record. arXiv:2010.03848.

[23] Ha, S., Lee, J., van de Panne, M., Xie, Z., Yu, W., Khadiv, M. Learning-based Legged Locomotion, State of the Art and Future Perspectives. International Journal of Robotics Research 44(8), pages 1396 to 1427, 2025. DOI 10.1177/02783649241312698, arXiv:2406.01152.

[24] Peng, X. B., van de Panne, M. Learning Locomotion Skills Using DeepRL, Does the Choice of Action Space Matter? ACM SIGGRAPH / Eurographics Symposium on Computer Animation (SCA), 2017. DOI 10.1145/3099564.3099567, arXiv:1611.01055.

[25] Siekmann, J., Godse, Y., Fern, A., Hurst, J. Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition. IEEE International Conference on Robotics and Automation (ICRA), pages 7309 to 7315, 2021. DOI 10.1109/ICRA48506.2021.9561814, arXiv:2011.01387.

[26] Liao, Q., Zhang, B., Huang, X., Huang, X., Li, Z., Sreenath, K. Berkeley Humanoid, A Research Platform for Learning-Based Control. IEEE International Conference on Robotics and Automation (ICRA), pages 2897 to 2904, 2025. DOI 10.1109/ICRA55743.2025.11127524, arXiv:2407.21781.

[27] Su, H., Luo, H., Yang, S., Jiang, K., Zhang, W., Chen, H. LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots. arXiv preprint, 2025. arXiv:2509.09106.

[28] Long, J., Wang, Z., Li, Q., Cao, L., Gao, J., Pang, J. Hybrid Internal Model, Learning Agile Legged Locomotion with Simulated Robot Response. International Conference on Learning Representations (ICLR), 2024. arXiv:2312.11460.

[29] Long, J., Ren, J., Shi, M., Wang, Z., Huang, T., Luo, P., Pang, J. Learning Humanoid Locomotion with Perceptive Internal Model. IEEE International Conference on Robotics and Automation (ICRA), pages 9997 to 10003, 2025. DOI 10.1109/ICRA55743.2025.11128333, arXiv:2411.14386.

[30] Gu, X., Wang, Y.-J., Chen, J. Humanoid-Gym, Reinforcement Learning for Humanoid Robot with Zero-Shot Sim2Real Transfer. ICRA 2024 Workshop on Agile Robotics, per the arXiv record, 2024. arXiv:2404.05695.

[31] van Marum, B., Shrestha, A., Duan, H., Dugar, P., Dao, J., Fern, A. Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 11256 to 11263, 2024. DOI 10.1109/IROS58592.2024.10802680, arXiv:2404.19173.

[32] Li, Z., Cheng, X., Peng, X. B., Abbeel, P., Levine, S., Berseth, G., Sreenath, K. Reinforcement Learning for Robust Parameterized Locomotion Control of Bipedal Robots. IEEE International Conference on Robotics and Automation (ICRA), pages 2811 to 2817, 2021. DOI 10.1109/ICRA48506.2021.9560769, arXiv:2103.14295.

[33] Rodriguez, D., Behnke, S. DeepWalk, Omnidirectional Bipedal Gait by Deep Reinforcement Learning. IEEE International Conference on Robotics and Automation (ICRA), pages 3033 to 3039, 2021. DOI 10.1109/ICRA48506.2021.9561717, arXiv:2106.00534.

[34] Radosavovic, I., Xiao, T., Zhang, B., Darrell, T., Malik, J., Sreenath, K. Real-World Humanoid Locomotion with Reinforcement Learning. Science Robotics 9(89), 2024. DOI 10.1126/scirobotics.adi9579, arXiv:2303.03381.
