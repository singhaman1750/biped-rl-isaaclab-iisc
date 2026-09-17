# Making the KScale Biped Walk Under the Published K-Scale Actuator Limits

> Status, written 2026-09-16 against the three runs `kscale_flat/2026-09-14_10-20-50`, `2026-09-15_08-07-56` and `2026-09-16_04-52-20`. NOT IMPLEMENTED. No edit proposed below has been applied to the tree. The premise fixed by the requester is that the actuator effort and velocity ceilings of `2026-09-16_04-52-20` are hardware quantities and are to be retained, so every remedy proposed here acts on the configuration around them and none proposes to relax them. The ablation of section 9 opens with the nominal pose moved to the published K-Scale posture, applied as a preparatory change rather than as an arm, and then re-draws the contrast between the original and the published ceilings upon it.

## 1. The question

Three runs share one reward set, one gain set, one action interface and one curriculum, and differ from one another in nine numbers, all of them actuator effort or velocity ceilings. The first walks. The second, which raises one ceiling at one joint group, does not. The third, which carries the published K-Scale ceilings in full, does not either. The question is why an actuator made stronger and faster should destroy a gait that a weaker and slower one supports, and the answer is not that the robot lacks authority, since authority is the thing that was added.

The account below proceeds from the configuration delta to the measured outcome, from the outcome to the mechanism, and from the mechanism to a root cause that is stated as a single sentence and then tested against every observation the three runs produced. It closes with an ablation whose arms are ordered by the strength of the evidence behind them rather than by the ease of applying them.

## 2. The configuration delta

Every difference between the three dumped configurations is reproduced below, read from `params/env.yaml` of each run. The gains, the armatures, the frictions, the twenty nine reward terms and their weights, the action scale of 0.4, the decimation of 2, the physics timestep of 0.005 s and the curriculum are identical across all three, verified by direct diff of the three files.

| Group | Joints | `2026-09-14` | `2026-09-15` | `2026-09-16` |
|---|---|---|---|---|
| hip_yaw | `(left\|right)_hip_yaw_03` | 60 Nm, 20 rad/s | 60 Nm, 20 rad/s | 42 Nm, 20 rad/s |
| hip_roll | `(left\|right)_hip_roll_03` | 60 Nm, 20 rad/s | 60 Nm, 20 rad/s | 42 Nm, 20 rad/s |
| hip_pitch | `(left\|right)_hip_pitch_04` | 60 Nm, 20 rad/s | 84 Nm, 20 rad/s | 84 Nm, 15 rad/s |
| knee | `(left\|right)_knee_04` | 60 Nm, 20 rad/s | 60 Nm, 20 rad/s | 84 Nm, 15 rad/s |
| foot_pitch | `(left\|right)_foot_pitch_02` | 17 Nm, 10 rad/s | 17 Nm, 10 rad/s | 17 Nm, 44 rad/s |
| foot_roll | `(left\|right)_foot_roll_02` | 17 Nm, 10 rad/s | 17 Nm, 10 rad/s | 17 Nm, 44 rad/s |

In every case `saturation_effort` equals `effort_limit` and `velocity_limit_sim` equals `velocity_limit`, so each group is described by one torque and one speed. The environment count differs, 6400 in the first run and 4096 in the other two, which changes the sample count per iteration and is treated in section 8 as a caveat rather than as a candidate cause.

Two properties of the third column deserve statement before anything is inferred from it. The mapping of the published K-Scale figures onto this robot is not uniform, the four proximal groups taking the derated soft limit of their RobStride class, 42 Nm for the robstride_03 and 84 for the robstride_04, while the two ankles retain the robstride_02 peak of 17 Nm rather than its soft limit of 11.9 [see `/ws/context/kscale-opensource.md` section 6]. And the change at the ankles is not a torque change at all but a speed change, the velocity ceiling rising from 10 rad/s to the robstride_02 firmware limit of 44. Section 5.2 shows that this last is the single most consequential number in the table, and section 5.4 shows that it is nonetheless not the root cause.

The delta from the first run to the second is one group and one quantity, the hip pitch torque ceiling rising from 60 Nm to 84. That isolation is what makes the sequence diagnostic rather than merely suggestive.

## 3. What the three policies do

### 3.1 The visual record

Frames were cut from `videos/play/42/rl-video-step-0.mp4` of each run at one hertz over the first twelve seconds and are held at `../context/artefacts/kscale_flat-2026-09-14_to_09-16-limits/`. The first run stands upright with a vertical torso and alternating legs and traverses the grid, which is the gait the requester describes as pigeon footed and which section 3.3 confirms by joint measurement. The second run is folded into a deep crouch with the torso pitched forward and the knees sharply flexed, and it shuffles without advancing. The third run collapses repeatedly, one frame showing the machine flat on the ground, and its posture between falls is the same forward pitched fold.

### 3.2 The rollout measurement

Every figure below is computed from `data/42/statistics.npy` and `data/42/dump.npy` of each run, being thirty two environments over three thousand and one control steps at a step of 0.01 s.

| Quantity | `2026-09-14` | `2026-09-15` | `2026-09-16` |
|---|---|---|---|
| Episode duration | 14.77 s | 4.48 s | 1.71 s |
| Episodes terminated early | 5.9 pct | 96.3 pct | 99.9 pct |
| Forward velocity command correlation | 0.962 | 0.054 | 0.134 |
| Base height | 0.722 m | 0.560 m | 0.628 m |
| Torso tilt from vertical | 4.4 deg | 10.8 deg | 14.0 deg |
| Cadence | 2.08 /s | 6.06 /s | 11.11 /s |
| Double support fraction | 14.8 pct | 62.6 pct | 74.8 pct |
| Swing duration | 0.312 s | 0.124 s | 0.076 s |
| Reward per step | +0.529 | minus 0.009 | minus 0.320 |

The first run walks and the other two do not, and the manner of the failure is uniform. A cadence of six and eleven steps per second against the first run's two, paired with a double support fraction of 63 and 75 per cent against fifteen, is not a fast gait but a foot chattering against the ground while the machine stands still, and a command correlation of 0.05 and 0.13 against 0.96 states that neither policy is tracking the commanded velocity at all.

### 3.3 The training record

Read from the event files of each run. The first reached 20865 iterations, the second 10429 and the third 2004.

| Iteration | Episode length, 09-14 | 09-15 | 09-16 | Reward, 09-14 | 09-15 | 09-16 | Noise std, 09-14 | 09-15 | 09-16 |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | 41 | 107 | 81 | minus 20.4 | minus 40.8 | minus 26.1 | 0.732 | 0.866 | 0.878 |
| 2000 | 40 | 317 | 482 | minus 11.6 | minus 5.8 | plus 48.9 | 0.625 | 0.839 | 1.035 |
| 3000 | 292 | 534 | n/a | plus 40.7 | plus 22.6 | n/a | 0.622 | 0.924 | n/a |
| 5000 | 1749 | 493 | n/a | plus 798.7 | plus 27.7 | n/a | 0.524 | 1.089 | n/a |
| 10000 | 1513 | 603 | n/a | plus 573.8 | plus 70.9 | n/a | 0.658 | 1.541 | n/a |

Three readings follow and the third is the one that matters.

The first run is the slower learner for the first three thousand iterations and by a wide margin. Its `base_contact` termination rate stands at 1.000 from iteration 50 to iteration 3000, meaning every episode ends in a fall, and its mean episode length sits near forty steps while the second run has already reached three hundred and seventeen. It then breaks through between iteration 3000 and 5000, episode length rising from 292 to 1749 and reward from 40.7 to 798.7, after which its policy noise standard deviation falls to 0.524 and its entropy from 17.8 to 6.5. That is a policy discovering a basin and committing to it.

The second run never breaks through. Its episode length plateaus near six hundred steps from iteration 3000 onward, its `base_contact` rate stands at 0.710 at iteration 10000, so seven episodes in ten still end in a fall after ten thousand iterations, and its reward plateaus near seventy against the first run's eight hundred. Its policy noise standard deviation RISES from 0.819 at iteration 1500 to 1.541 at iteration 10000 and its entropy from 13.1 to 20.3, while its surrogate loss rises from 0.013 to 0.080. A policy whose entropy grows across seven thousand iterations is not converging on anything, it is being pushed back toward its prior because the advantage signal it receives does not favour any direction.

The third run's entropy turns upward earlier than either of the others, and this is the one respect in which its own training record does condemn it. Taking a fifty one iteration moving average and locating its minimum gives a turning point at iteration 6614 for the first run, 1579 for the second and 1175 for the third, at minima of 5.398, 13.104 and 14.167 nats. Fitted over iterations 1000 to 1500 the entropy slope is minus 3.143 and minus 3.449 per thousand iterations in the first two runs and PLUS 1.021 in the third, so at a point where both other runs are still sharpening the third has already begun to disperse, and it does so four hundred iterations before the run that is known to fail. Both the timing of the turn and the depth of the minimum order the three runs exactly as their outcomes do.

Two properties of that measurement should be stated before it is relied upon. The logged entropy is not a restatement of the logged noise standard deviation, the first being the sum of the per joint logarithms and the second their arithmetic mean, and the gap between them measures how unequally the policy has sharpened its twelve dimensions. That gap grows through training, reaching 3.828 nats in the first run by iteration 20000, but at iteration 2000 it stands at 1.575, 1.627 and 1.567 in the three runs, which is to say it is the same in all of them, so the entropy difference at that point is a genuine difference in exploration and not an artefact of some joints sharpening faster than others. And the three `params/agent.yaml` files are byte identical, carrying an entropy coefficient of 0.005 and an adaptive learning rate schedule against a desired divergence of 0.01, so none of this is a hyperparameter difference.

The third run cannot be declared failed on its episode length and this is recorded plainly rather than glossed. At iteration 2000, the only point at which all three may be compared, it leads both others, its episode length of 482 standing against 317 and 40 and its reward of plus 48.9 against minus 5.8 and minus 11.6. The run that eventually walked was, at that iteration, far the worst of the three. What is established about the third run is therefore its rollout at 2000 iterations, which section 3.2 records, and its posture, which section 3.4 records, and both are consistent with the second run rather than with the first. Its failure is therefore established by the entropy turn above and by the posture of section 3.4, and not by the return it had reached when it stopped, and section 9 nonetheless places it to twenty thousand iterations because the entropy screen is one sided in the manner section 9.4 describes.

### 3.4 The posture, and the joints that are not in control

The median joint angle of each policy, with the fraction of steps spent within two per cent of the span of either mechanical stop.

| Joint | Hard limits | Median, 09-14 | Median, 09-15 | Median, 09-16 | Stop residency, 09-14 | 09-15 | 09-16 |
|---|---|---|---|---|---|---|---|
| left_hip_roll_03 | minus 0.209 to 2.269 | minus 0.158 | minus 0.210 | minus 0.209 | 49.4 pct | 95.0 pct | 84.8 pct |
| right_hip_roll_03 | minus 2.269 to 0.209 | plus 0.150 | plus 0.210 | plus 0.210 | 45.7 pct | 95.3 pct | 93.1 pct |
| left_knee_04 | 0.000 to 2.705 | 0.577 | 1.345 | 1.479 | 5.5 pct | 1.3 pct | 1.3 pct |
| right_knee_04 | 0.000 to 2.705 | 0.549 | 1.555 | 0.720 | 5.4 pct | 0.3 pct | 3.4 pct |
| left_foot_pitch_02 | minus 0.873 to 0.524 | minus 0.449 | minus 0.877 | minus 0.293 | 37.4 pct | 95.7 pct | 4.7 pct |
| right_foot_pitch_02 | minus 0.873 to 0.524 | minus 0.401 | minus 0.865 | minus 0.873 | 35.0 pct | 64.6 pct | 79.7 pct |
| left_foot_roll_02 | minus 0.262 to 0.262 | minus 0.163 | minus 0.262 | plus 0.262 | 56.1 pct | 97.8 pct | 87.8 pct |
| right_foot_roll_02 | minus 0.262 to 0.262 | plus 0.169 | minus 0.259 | plus 0.155 | 53.1 pct | 94.5 pct | 44.8 pct |

Two findings are carried by this table and they are of unequal weight.

The knee flexion separates the three policies absolutely. The walking policy reaches a maximum knee angle whose ninety ninth percentile is 1.064 rad and spends 0.0 per cent of its steps beyond 1.2 rad. The second policy has a median maximum knee of 1.559 rad and spends 92.0 per cent of its steps beyond 1.2, and the third 1.480 rad and 82.5 per cent. The two failing policies are folded into a crouch that the walking policy never once enters.

The stop residency is severe in all three runs and it is severe in the run that works. The ankle roll of the walking policy stands at a mechanical stop for 53 to 56 per cent of steps and its hip roll for 46 to 49 per cent, on joints whose whole travel is plus and minus 0.262 rad and 0.209 rad of abduction respectively. This corroborates rather than discovers, the addition of 2026-09-03 to `../context/KScale.md` having recorded an ankle roll stop residency of 80.5 per cent in the earlier reference run, and it establishes that the working gait is not a healthy one. It walks on jammed joints. Section 6.2 takes this up as a contributing cause that the ceilings did not create and did not cure.

## 4. The actuator model, and what a ceiling actually does

The KScale actuators are `IdentifiedActuator` at `environments/environments/actuators/actuator_pd.py:18`, which derives from Isaac Lab's `DCMotor` and adds a friction model. The torque is computed in Python and handed to the solver as an effort, so the servo is integrated explicitly and the whole of the analysis in section 11.1.1 of `../context/KScale.md` applies to it. The model is an ideal torque source behind a clipped proportional derivative law, which is the approximation an actuator network exists to replace and which no system identification of the RobStride hardware is available to improve upon, in this workspace or in the published K-Scale tree [1]. The clipping is at `/ws/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:296-306` and takes the four quadrant form

```
tau_max(qdot) = min( saturation_effort * (1 - qdot / velocity_limit),  effort_limit )
tau_min(qdot) = max( saturation_effort * (-1 - qdot / velocity_limit), -effort_limit )
tau_applied   = clip( kp * (q_des - q) - kd * qdot,  tau_min,  tau_max )
```

with the joint velocity pre clipped to `velocity_limit * (1 + effort_limit / saturation_effort)` at line 297. Three consequences govern everything that follows.

Where `saturation_effort` equals `effort_limit`, as it does at every group of every run here, the flat ceiling binds only at zero speed and the curve is a plain descending line reaching exactly zero available torque at `velocity_limit`. The two parameters that are meant to describe a stall torque and a continuous ceiling have been collapsed onto one number, so the model carries no torque speed curve distinct from its clip.

Where the demanded torque exceeds the envelope, the applied torque ceases to depend on the commanded position at all and becomes a function of joint velocity alone. Its derivative with respect to velocity is `minus saturation_effort / velocity_limit`, which is to say that a saturated joint behaves as a viscous damper of exactly that coefficient. This term is not written in any configuration file and does not appear in any gain table, and section 5.2 shows it to be the dominant source of damping at the ankles of this robot.

The friction is subtracted after the clip at `environments/environments/actuators/actuator_pd.py:34-35`, so the applied torque exceeds `effort_limit` by up to `friction_static` plus `friction_dynamic` times speed. The measured ninety fifth percentile hip pitch torque of 60.35 Nm against a 60 Nm ceiling is this effect and not an error. It is recorded as a defect left standing, being of no consequence at the magnitudes involved and not worth an edit to a shared module.

## 5. The mechanism

### 5.1 The policy does not control this robot through its servo

The action is a joint position target at `JointPositionActionCfg` with `scale` 0.4 and `clip` unset, read from `params/env.yaml` of every run, which is the position target interface the massively parallel legged learning literature has settled on [2]. The position error at which a joint saturates is `effort_limit / kp`, and expressed as a multiple of the action scale it gives the normalised action magnitude beyond which the commanded position no longer reaches the joint.

| Joint | kp | Effort, 09-14 | Saturating action, 09-14 | Effort, 09-16 | Saturating action, 09-16 | Measured envelope residency, 09-14 |
|---|---|---|---|---|---|---|
| Hip pitch | 200 | 60 | 0.75 | 84 | 1.05 | 0.977 |
| Hip roll | 150 | 60 | 1.00 | 42 | 0.70 | 0.115 |
| Hip yaw | 15 | 60 | 10.00 | 42 | 7.00 | 0.000 |
| Knee | 200 | 60 | 0.75 | 84 | 1.05 | 0.003 |
| Ankle pitch | 50 | 17 | 0.85 | 17 | 0.85 | 0.711 |
| Ankle roll | 20 | 17 | 2.12 | 17 | 2.12 | 0.683 |

The policy noise standard deviation of the walking run stands between 0.52 and 1.05 across training, so a one sigma action saturates the hip pitch, the knee and the ankle pitch outright. The measured residency confirms it, the hip pitch of the walking policy sitting on its envelope for 97.7 per cent of steps and the two ankles for 68 to 71 per cent. The stiffnesses of those joints are not the operative control parameter for most of an episode and the effort ceiling is, which corroborates by independent measurement the finding already recorded at section 11.10 of `../context/KScale.md`.

This is the sense in which the ceilings are not a safety margin on the robot. They are its control surface. That an action scale cannot substitute for a bound is established for this workspace already, the reduction from 0.4 to 0.25 having RAISED the ankle roll stop residency from 0.669 to 0.765, and the reason is that a Gaussian policy over an unbounded support holds its commanded excursion invariant under a reparameterisation of the scale that multiplies it [3].

### 5.2 The ankle velocity ceiling was the ankle's damping

The two ankles carry stance reflected inertias of 4.85781 and 5.37942 kg m squared against swing inertias of 0.00760 and 0.00574, ratios of 639 and 937, so a damping chosen to be adequate in the air is negligible on the ground [`../context/KScale.md` section 4.9]. At the configured damping of 1.7 and 0.5 the stance damping ratios are 0.055 and 0.024, which is to say that the ankles of this robot have effectively no damping whatever while carrying weight, against a band of 0.7 to 1.0 that the Isaac Sim gain guidance recommends and that this workspace targets [4].

The envelope term of section 4 is what supplied it. Its coefficient is `saturation_effort / velocity_limit`, which at the ankles of the first two runs is 17 divided by 10, being 1.70 Nm s per radian, and at the ankles of the third is 17 divided by 44, being 0.386. Taken together with the configured damping and evaluated against the stance inertia, the ankle damping ratio while saturated is as follows.

| Joint | kd | Envelope term, 09-14 | Stance damping ratio, 09-14 | Envelope term, 09-16 | Stance damping ratio, 09-16 |
|---|---|---|---|---|---|
| Ankle pitch | 1.7 | 1.700 | 0.109 | 0.386 | 0.067 |
| Ankle roll | 0.5 | 1.700 | 0.106 | 0.386 | 0.043 |

At the ankle roll the envelope was supplying 77 per cent of the joint's total damping and the published ceiling removes three quarters of that contribution. The measurement confirms that the joints were living on the boundary, the ninety fifth percentile ankle speed of the walking run standing at 9.65 and 9.80 rad per second against a ceiling of 10, which is 96.5 and 97.9 per cent of it, and the torque available at that speed being 0.59 and 0.35 Nm out of 17. Under the published ceiling the same percentile falls to 0.14 and 0.17 of the limit and the available torque rises to 14.0 Nm, a factor of twenty four.

The plain statement is that the 10 rad/s ceiling was not limiting the ankle, it was damping it, and the published ceiling removes that damping from a joint that has almost none of its own.

### 5.3 The hip pitch ceiling was the floor under the base height

This is the finding the sequence turns on and it is the one the second run isolates. Conditioning the median hip pitch torque on the base height gives the following, with the occupancy of each band beside it.

| Base height band | 09-14 torque | 09-14 occupancy | 09-15 torque | 09-15 occupancy |
|---|---|---|---|---|
| 0.25 to 0.45 m | no samples | 0.0 pct | 82.3 Nm | 2.8 pct |
| 0.45 to 0.55 m | no samples | 0.0 pct | 83.5 Nm | 36.9 pct |
| 0.55 to 0.62 m | no samples | 0.0 pct | 83.5 Nm | 50.9 pct |
| 0.62 to 0.68 m | 57.8 Nm | 5.3 pct | 83.7 Nm | 4.3 pct |
| 0.68 to 0.74 m | 58.9 Nm | 94.2 pct | 83.4 Nm | 2.0 pct |
| 0.74 to 0.85 m | 59.2 Nm | 0.5 pct | 84.1 Nm | 0.8 pct |

The hip pitch is pinned at its ceiling in both runs and at every height, at 58 to 59 Nm against a limit of 60 in the first and at 82 to 84 against a limit of 84 in the second. It is therefore delivering all the torque it has, always, and the height the machine settles at is the height at which that torque balances the load. At 60 Nm no band below 0.62 m collects two hundred samples out of ninety six thousand. At 84 Nm the machine spends 90.6 per cent of its steps below 0.62 m and descends to 0.45.

The ceiling is a floor. A hip pitch limited to 60 Nm cannot hold this machine in a crouch deeper than about 0.62 m, so a policy that folds past that point falls, terminates on `base_contact` and is charged for it. A hip pitch permitted 84 Nm holds the fold, and the fold survives.

### 5.4 The root cause

The torque ceiling of 60 Nm at the hip pitch was making a degenerate solution mechanically unreachable, and raising it to the published 84 Nm makes that solution reachable, whereupon the policy finds it long before it would have found walking and never leaves it.

The degenerate solution is the deep crouch. It collects the `keep_balance` survival bonus, it avoids the `base_contact` termination which is the only penalised terminal, and it requires no coordination whatever. The training record in section 3.3 is the signature of a policy that found it early, the second run reaching an episode length of 317 steps at iteration 2000 where the first run, denied the crouch, still stands at 40 and is falling in every episode. It is also the signature of a policy that cannot leave it, the entropy and the noise standard deviation rising monotonically from iteration 1500 onward because no local perturbation of a crouch improves a return that is already better than falling.

This is a known shape of failure rather than a novel one. Reda, Tao and van de Panne ablate the survival bonus over three magnitudes and state outright that values too small or too large lead to local minima corresponding to falling forward and standing still respectively, the large bonus case being exploited by a character that balances and never steps [5]. What the present sequence adds is that the same local minimum may be opened or closed by an actuator ceiling at a fixed survival bonus, so the reachability of the exploit is a property of the plant and not only of the reward. The formal reason the reward cannot exclude it unaided is that a return built substantially from commanded velocity error and a survival bonus does not distinguish among the postures that collect them [6].

The account is consistent with every observation the three runs produced. It explains why the second run fails on a single ceiling at a single joint group, since that is the ceiling that sets the height floor. It explains why the failure is a crouch rather than a topple, since the crouch is the solution that became available. It explains why both failing runs show high double support, low command correlation and a cadence of six and eleven per second, since a machine standing in a fold and chattering its feet is what the statistics of a crouch look like. It explains why the reward per step is near zero rather than strongly negative, since the crouch is genuinely worth more than falling. And it explains the entropy signature, which no account resting on numerical instability would produce, since a diverging servo degrades the return monotonically rather than holding it at a plateau for seven thousand iterations.

### 5.5 Two candidate causes that the evidence does not support

The explicit servo stability bound is not violated by any of the three runs. The criterion is `kd * dt_phys / I_swing` below two against the swing inertia [`../context/KScale.md` section 11.1.1], and at the configured damping it stands at 0.096 for the hip pitch, 0.085 for the hip roll, 0.163 for the hip yaw, 0.414 for the knee, 1.118 for the ankle pitch and 0.436 for the ankle roll. The damping was not changed by any of the three configurations, so this number is identical in all of them and cannot distinguish between a run that walks and one that does not. The ankle pitch figure is nonetheless recorded in section 6.3 as a standing defect, since it exceeds the worst corner ceiling the randomisation envelope imposes.

A deficit of authority is excluded by the direction of the change. The single support hold torques of this robot are 13.651 Nm at the hip pitch and 16.468 at the knee [`../context/KScale.md` section 10], so 60 Nm already carried a margin of four and the failing configurations carry more. Nothing failed for want of torque.

## 6. Three contributing causes the ceilings unmasked

These did not begin with the published ceilings and are not cured by reverting them. They are recorded because the first run tolerates them and a corrected configuration should not have to.

### 6.1 The nominal pose is kinematically almost straight

The pose sets the knees to 0.4 rad and the ankle pitches to minus 0.3, and section 16.3 of `../context/KScale.md` establishes by a swept table that this lowers the machine 3.5 mm against the zero pose, the sagittal chain being nearly collinear across that whole range. A leg at that pose has very little knee moment arm with which to modulate vertical force, and the consequence is visible in the measurement, the hip pitch standing at its ceiling for 97.7 per cent of steps in the run that works. The policy is generating its vertical support by saturating the largest joint on the robot rather than by using a bent leg.

### 6.2 Two joints have no travel left to command with

The ankle roll carries a stiffness of 20 Nm per radian against a single support hold torque of 25.07 Nm, so the hold alone would deflect it 1.25 rad against a total travel of 0.262, and section 11.8 of `../context/KScale.md` derives an authority floor of 100.3 Nm per radian for that joint. The hip roll carries 150 against a derived floor of 238, which is the stiffness at which the 24.95 Nm hold consumes no more than half the 0.209 rad of abduction available. A third shortfall is more severe than either and was found by running `scripts/analysis/kscale_stance_analysis.py` in the course of this work. The gravitational stiffness of both ankles in single support is plus 84.90 Nm per radian, a destabilising sign, and a joint whose stiffness does not exceed it cannot hold the assembly above it upright at any deflection whatever. The configured ankle pitch stiffness is 50 and the ankle roll 20. Both ankles of this robot are therefore statically unstable in single support at the gains every run in its history has carried, which is not a matter of insufficient margin but of the wrong sign, and it is the mechanical reason the machine must reach a stop rather than settle at a deflection. The measured stop residencies of 53 to 56 per cent and 46 to 49 per cent in the run that walks are the direct expression of all three shortfalls. This robot has no ankle roll authority and very little hip roll authority, and the pigeon footed splay is its remaining lateral strategy rather than a stylistic fault.

### 6.3 The ankle pitch damping already stands past its bound

The worst corner of the randomisation envelope requires `kd` below `0.864 * I_swing / dt_phys`, which is 1.31 Nm s per radian at the ankle pitch [`../context/KScale.md` section 11.1.1]. The configured 1.7 stands at 1.30 times that bound. This is why section 9 does not propose to answer the lost envelope damping of section 5.2 by raising `kd` at that joint, there being no headroom to raise it into.

## 7. What the remedy may and may not be

The ceilings are fixed by the requester and are treated here as a hardware constraint. The 60 Nm hip pitch limit that closed the crouch is therefore unavailable, and its function must be supplied by something else. Three families of substitute exist and only one of them is supported by the evidence in this workspace.

Reducing the action scale is excluded. It has been tried on this robot and measured, and it raised the ankle roll stop residency from 0.669 to 0.765 rather than lowering it, for the reason that a Gaussian over an unbounded support is not constrained by the scale that multiplies it [3]. This is recorded in cluster 25 of `/ws/context/literature.md` and must not be proposed again.

Pricing the crouch more heavily is excluded by measurement. The configuration already carries `pen_base_height` at a weight of minus 30 against a target of 0.772 m, and the second run pays minus 1.17 per second on that term against the first run's minus 0.0127, a factor of ninety two. The price is being paid rather than avoided, which is the exact signature of a requirement that belongs as a constraint rather than as a price, and cluster 20 of `/ws/context/literature.md` records the same conclusion reached on this robot for a different term.

Constraining the crouch directly is what remains. The constraint the 60 Nm ceiling expressed was a lower bound on base height, and the configuration already contains the instrument, commented out, at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:957-960`, where a `low_height` termination stands disabled at a threshold of 0.27 m.

One objection to that instrument must be answered rather than assumed away, since the same term is live on the sibling biped at `environments/environments/tasks/locomotion/cfg/SF/brs_base_env_cfg.py:913-916` for an unrelated reason, that robot having base contact behaviour this one does not share. The justification offered here is independent of that one and rests on section 5.3. The quantity being restored is not a proxy for a contact the sensor misses, it is the height floor that the 60 Nm ceiling supplied mechanically and that the published ceiling removes. A threshold of 0.27 m would not supply it, being far below the 0.45 to 0.62 m band the crouch occupies, so the disabled term is not merely to be re-enabled but to be re-parameterised near the measured floor of the run that walks, 0.62 m being that floor at the pose those runs carried and section 9.1 moving the pose beneath it.

## 8. Caveats on the evidence

The environment count differs, 6400 against 4096, so at a matched iteration the two later runs have seen 0.64 times the samples. This is a throughput difference rather than a difference in the attainable optimum, and it does not account for the observation, since the second run held an episode length near six hundred steps for seven thousand iterations while its entropy rose. A run starved of samples climbs slowly and does not plateau with rising entropy.

The three rollouts are taken at 20000, 10000 and 2000 iterations respectively and are therefore not matched. Section 3.3 states what follows from this for the third run and section 9 opens with the arm that repairs it.

The stance reflected inertias quoted throughout treat the planted sole as welded and every other joint as locked, which overstates the constrained inertia in double support [`../context/KScale.md` section 4.9]. The conclusions drawn from them are comparative between configurations that share the assumption, so the overstatement does not move them.

Section 11 of `../context/KScale.md` computes its Nyquist ceilings at a control period of 0.02 s, being a decimation of 4, while the three runs here carry a decimation of 2 and a control period of 0.01 s. Every ceiling in that section is therefore conservative by a factor of two as applied to these runs, and the damping bounds of section 6.3, which depend on the physics timestep rather than the control period, are unaffected.

## 9. The ablation

Every change is to `environments/environments/assets/config/kscale_identified_cfg.py` or to `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py`, both configuration files private to this robot. Nothing below edits the shared `environments` mdp package, so nothing below can alter the behaviour of the SD_BRS1 or of either TRON1 task, and no in flight or historical comparison is invalidated.

### 9.1 The preparatory change, applied before any arm is run

The nominal pose is moved to the published K-Scale posture scaled onto this robot's closure condition, being hip pitch plus and minus 0.349 rad, knee 0.873 and ankle pitch minus 0.524, replacing the present plus and minus 0.1, 0.4 and minus 0.3 at `environments/environments/assets/config/kscale_identified_cfg.py`. Those three angles satisfy this robot's sagittal closure identity exactly, their signed sum vanishing, so the sole remains flat and the ankle roll origin remains at the 0.043 m sole depth, which section 16.2 of `../context/KScale.md` names as the cheapest available check that a pose is closed. The standing height falls to 0.73540 m from 0.77161, computed by forward kinematics against the URDF and confirmed by `scripts/analysis/kscale_stance_analysis.py` at the new pose. It must not be read off the sweep of section 16.3 of `../context/KScale.md`, which holds the hip pitch at plus and minus 0.1 throughout and therefore describes a different family of poses, and which would give 0.712 at this knee angle.

This is a preparatory change and not an arm because it is not being tested against the present pose. It is adopted on the ground that the published K-Scale posture is the one that stack validated on hardware carrying these actuators, and on the independent ground of section 6.1, that the present pose is kinematically almost straight and therefore obliges the hip pitch to saturate in order to generate vertical support. Arm 0 below is what establishes that the change is safe, by re-running on it the one configuration known to walk.

Two consequences must be measured rather than assumed. The ankle pitch origin moves from minus 0.3 to minus 0.524, leaving 0.349 rad to the minus 0.873 stop where the present pose leaves 0.573, on a joint already at that stop for 35 to 37 per cent of steps in the run that walks, so the ankle pitch stop residency is the quantity that decides whether the pose is retained. And the deeper fold places the knee nearer the 1.2 rad region that section 3.4 identifies with the degenerate crouch, so the height floor of section 5.3 and the crouch must be re-measured against the new standing height rather than against 0.62 m, which was measured at the old one.

### 9.1.1 The height parameters that move with the pose

Three parameters carry the standing height and all three must move together, none of them being derivable from the others by the reader of a configuration file.

| Parameter | Location | Present value | New value | Basis |
|---|---|---|---|---|
| `pen_base_height` target | `cfg/SF/kscale_base_env_cfg.py:804` | 0.772 | 0.735 | The standing height itself, the present target tracking the present 0.77161 to within 0.4 mm |
| `pen_feet_regulation` base height target | `cfg/SF/kscale_base_env_cfg.py:873` | 0.772 | 0.735 | The same quantity, carried by a second term |
| Spawn height | `assets/config/kscale_identified_cfg.py:259` | 0.79 | 0.755 | The standing height plus the 0.02 m settling margin this workspace uses |

The sole depth of 0.043 m and the decay scale of 0.03 m in `pen_feet_regulation` do not move, the pose remaining closed and the ankle roll origin still standing at exactly 0.04300 m above the ground, which section 16.2 of `../context/KScale.md` names as the check that a pose is closed and which was verified at the new angles.

### 9.1.2 What the deeper pose costs, computed rather than assumed

Running the stance script at both poses gives the following, the single support column being the quantity that governs.

| Joint | tau_1sup, current | tau_1sup, K-Bot pose | Configured K | Sag at K, current | Sag at K, K-Bot pose |
|---|---|---|---|---|---|
| Hip pitch | 13.651 | 11.288 | 200 | 0.068 | 0.056 |
| Knee | 16.468 | 27.877 | 200 | 0.082 | 0.139 |
| Ankle pitch | 6.133 | 6.133 | 50 | 0.123 | 0.123 |
| Hip yaw hold | 2.503 | 8.575 | 15 | 0.167 | 0.572 |
| Ankle roll hold | 25.067 | 25.067 | 20 | at a stop | at a stop |

The hip pitch load FALLS, by 17 per cent, which is a result worth stating because the pose was adopted for a different reason and this is a second argument for it. The knee load rises by 69 per cent to 27.877 Nm, which stands at 33 per cent of the published 84 Nm ceiling and is therefore comfortably held, but which deflects the joint 0.139 rad at the configured stiffness of 200 against 0.082 at present. The ankle pitch and ankle roll loads do not move at all, both being determined by the sole geometry and the lateral lean rather than by the sagittal fold.

The hip yaw is the joint the pose hurts and the figure is severe. Its single support hold torque rises from 2.503 to 8.575 Nm, and at the configured stiffness of 15 that deflects it 0.572 rad. This is not a new defect, section 11.6 of `../context/KScale.md` deriving an authority floor of 100 Nm per radian for that joint against the configured 15, but the pose multiplies the consequence by 3.4 and the hip yaw is the joint whose excursion this robot's splay is made of. Arm 0 must therefore report hip yaw excursion as well as the ankle pitch stop residency, and a hip yaw stiffness raise belongs in arm 4 if either degrades.

### 9.2 The arms

| Arm | Ceilings | Additional change | Purpose | Falsifies |
|---|---|---|---|---|
| 0 | Original, as `2026-09-14` | none | Establishes that the pose change alone does not break the configuration known to walk | The preparatory change of section 9.1 |
| 1 | Published K-Scale, as `2026-09-16` | none | The decisive re-test, whether the corrected pose alone makes the published ceilings workable | That the pose is sufficient |
| 2 | Published K-Scale | `low_height` termination re-enabled, threshold set from arm 0 | Restores the height floor of section 5.3 as a constraint rather than a price | The root cause of section 5.4 |
| 3 | Published K-Scale | Ankle roll `damping` 0.5 to 0.95 | Recovers part of the envelope damping of section 5.2 within the bound of section 6.3 | That the lost ankle damping matters |
| 4 | Published K-Scale | Ankle roll `stiffness` 20 to 100, ankle pitch 50 to 100, hip roll 150 to 250 | Raises both ankles above the gravitational stiffness of 84.90 they presently sit beneath, section 6.2 | That the stop residency matters |
| 5 | Published K-Scale | `clip` set on the action term at plus and minus 1.0 | Supplies the bound the effort ceiling was supplying, section 5.1 | That the unbounded action space matters |

Arms 0 and 1 are run first and in that order, and they are the pair that carries the design. Arm 0 is the control, reproducing on the new pose the only configuration this robot has ever walked under, and a failure there condemns the pose rather than the ceilings and sends the work back to section 9.1. Arm 1 repeats the contrast the three runs of this document drew, with the pose corrected, and it is the arm that may make every arm below it unnecessary. Both must be run to at least twenty thousand iterations, since section 3.3 establishes that the run which eventually walked was the worst of the three at iteration 2000 and that no shorter run is diagnostic.

Arms 2 through 5 are run only if arm 1 fails, and arm 2 is run alone before any other, since section 5.4 identifies it as the remedy that addresses the root cause directly and it is a two line configuration change. Its threshold is not fixed here at 0.62 m, that figure having been measured at the old pose, and is instead to be taken as the fifth percentile base height of arm 0, which is the same quantity measured where it now applies.

Arms 3 and 4 are run together on top of arm 2 should that arm prove insufficient, since both address the distal authority deficit of section 6.2 and neither is expected to suffice alone. Arm 5 follows last, being the only arm that perturbs the action interface itself, which every run in this robot's history has shared.

### 9.3 Three notes on the arms

Arm 3 raises the ankle roll damping to 0.95 and no further, the worst corner bound of section 6.3 standing at 0.992 for that joint. The ankle pitch is deliberately not raised, standing already at 1.30 times its own bound, and that defect is recorded rather than repaired here. The full remedy for the envelope damping lost in section 5.2 is a physics timestep of 0.0025 s, which doubles every bound in section 6.3 and costs a factor of two in throughput, and it is named as the fallback should arm 3 prove insufficient rather than proposed now.

Arm 4 raises both ankles to 100 Nm per radian rather than the ankle roll alone, since section 6.2 establishes that the gravitational stiffness of 84.90 exceeds the configured stiffness at both. The Nyquist ceiling against the swing inertia admits it at the control period these runs carry, being `I_swing * omega_nyq^2` at 314.16 rad per second, which is 750 Nm per radian at the ankle pitch and 566 at the ankle roll. Section 11.8 of `../context/KScale.md` quotes a ceiling of 141.7 computed at a control period of 0.02 s, and the discrepancy of section 8 must be resolved in that document before either figure is quoted elsewhere, but neither reading forbids 100.

Arm 5 is proposed in place of an action scale reduction and not alongside one, section 7 recording that the reduction has been measured on this robot and moves the stop residency the wrong way.

### 9.4 The measurement that decides every arm

An arm has recovered the gait when its mean episode length exceeds 1400 steps, its `base_contact` termination rate falls below 0.3, its forward velocity command correlation exceeds 0.9 and its base height holds within 0.04 m of the standing height of the new pose, the first three being the values the run of 2026-09-14 attains and the fourth its equivalent restated against a pose it did not carry.

A cheap early screen is available and should be applied before any arm is run to completion. The entropy slope fitted over iterations 1000 to 1500 is negative in both runs that were still improving and positive in the run that had already turned, and the turning point itself orders the three runs exactly as their outcomes do, at iteration 6614, 1579 and 1175. An arm whose entropy has turned upward by iteration 1500 may be abandoned at that point. The screen is one sided and must be used as such, since the run that walked did not reach its own turning point until iteration 6614, so the absence of a turn is evidence of health only in proportion to the iteration reached and never a confirmation.

Two diagnostics are to be logged additionally and neither is at present, being the fraction of steps at which each joint sits within two per cent of the span of a mechanical stop, and the fraction at which each joint sits on its torque envelope. Sections 3.4 and 5.1 establish that these are the quantities which actually describe this robot's control, and no run in its history has recorded either during training.

## 10. Bibliography

1. Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., Hutter, M. (2019). Learning agile and dynamic motor skills for legged robots. Science Robotics 4(26), eaau5872. arXiv:1901.08652, DOI 10.1126/scirobotics.aau5872.
2. Rudin, N., Hoeller, D., Reist, P., Hutter, M. (2022). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Proceedings of Machine Learning Research 164, 91 to 100. arXiv:2109.11978.
3. Chou, P., Maturana, D., Scherer, S. (2017). Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution. International Conference on Machine Learning, Proceedings of Machine Learning Research 70.
4. NVIDIA. Tutorial 6, Joint Gains Tuning. Isaac Sim OpenUSD Tuning Tutorials, `docs.isaacsim.omniverse.nvidia.com`.
5. Reda, D., Tao, T., van de Panne, M. (2020). Learning to Locomote, Understanding How Environment Design Matters for Deep Reinforcement Learning. Motion, Interaction and Games 2020. arXiv:2010.04304.
6. Skalse, J., Howe, N., Krasheninnikov, D., Krueger, D. (2022). Defining and Characterizing Reward Hacking. Advances in Neural Information Processing Systems 35. arXiv:2209.13085.
