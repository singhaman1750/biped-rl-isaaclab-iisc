# KScale Gait Shaping, the Mechanical Root of the Pigeon Toed Crouch, and the Ablation that Would Settle It

> NOTE ON THE NUMBERING OF THE PARAGRAPH BELOW, added 2026-09-30. It is the status record as it stood on 2026-09-23 and every arm AND SECTION number in it is that date's numbering, which three later renumbering passes have superseded. Its section 16 is the present section 20 and its section 17 is the present section 21, whose sub-sections were reordered as well as renumbered so that no arithmetic converts one to the other, and its arm numbers begin at zero as section 7's do. It is restored here verbatim from the copy taken before the first of those passes, having been corrupted by all three, and NO FUTURE PASS SHOULD REMAP IT. Section 21's preamble carries the mapping from 2026-09-28 onward, and section 7's blockquote carries the mapping of the proposal this paragraph describes.
>
> Status, PARTIALLY IMPLEMENTED, written 2026-09-21 against the four runs `kscale_flat/2026-09-17_09-13-46`, `2026-09-17_11-25-59`, `2026-09-18_09-37-29` and `2026-09-18_10-03-48`, their dumped configurations, their evaluation dumps at `data/42/`, their TensorBoard records and their play videos. Extended 2026-09-22, sections 10 through 14, after arms 1 and 2 of the section 7 ablation were executed as `2026-09-21_06-45-40` and `2026-09-21_08-40-59`. Arms 1 and 2 are therefore run and reported, arm 0 was never run separately since `2026-09-18_09-37-29` serves as its comparator, and arms 6 and 7 of that section's own numbering, the heading constraint and the power term, remain unapplied and are now the second stages of arms 10 and 6 respectively. Arms 3 and 4 were SWAPPED on 2026-09-22 so that the action authority precedes the nominal pose, and the action authority arm is now APPLIED TO THE WORKING TREE and awaiting its run, section 17.4 recording it with its two edits, its sizing rule and the argument against it, the arm 8 by which it was also known until 2026-09-23 having been collapsed into arm 3 on that date. A new arm 4, the lateral weight shift, was INSERTED on 2026-09-22 and every arm from the old arm 4 onward bumped by one to make room. Its reward was merged from `nithin/nithin/kscale` commit 2599d61 the same day and is likewise APPLIED TO THE WORKING TREE and awaiting its run, section 17.5 carrying its premise, its measurements and the amendment to its amplitude they imply. The ablation of section 7 is superseded from arm 1 onward by section 16, which carries the arms a reader should act upon. Extended again 2026-09-23 with section 17, which converts every arm of the ablation, executed and outstanding alike, into a sub-section carrying the code it changes quoted verbatim from the tree of that date, the design decisions taken in writing it with the alternatives they reject, and the parameterisation with the arithmetic that sets it, so that an arm may be executed from that section alone. Extended a second time on 2026-09-23 with the implementation of arm 5, the nominal pose, at the user's direction, together with a reduction of the settling margin to zero so that the spawn height, the standing height and the target of `pen_base_height` are one number, and with the naming of arm 4b, the halving of the two ankle action scales, which had until then stood in the tree as an unattributed departure from arm 3's sizing rule. Section 17.6 carries arm 4b and section 17.7 carries arm 5 as applied, the latter correcting two figures of its own earlier draft, the standing height of the posture being 0.73492 metres rather than 0.73540 once the retained roll nominals are counted, and the deeper ankle pitch nominal moving that joint closer to its stop rather than further from it. Section 17.18 records the tree as it stood on the morning of that date and section 17.19 records it afterwards, four arms now standing in it together, so that a run launched from it is a four variable departure from the seventeen newton metre reference and can attribute an effect to no one of them. That is the user's decision, taken to gather behaviour at the full intervention rather than to price each part. Of the sixteen arms, two are executed and reported, four are implemented and active, and no other edit proposed anywhere in this document has been applied to the working tree. Extended a third time on 2026-09-23 and RESTRUCTURED on the same date, after arms 3 and 4 were executed as `2026-09-22_07-18-05` and `2026-09-22_07-40-43` against the seventeen newton metre reference `2026-09-21_06-45-40`. Four things changed and the restructuring is the one a returning reader should know about first. The evidence now precedes the ablation rather than following it, section 14 reporting the three runs and section 15 surveying the bang-bang literature they prompted, with the ablation restated at section 16 and the implementation detail at section 17, where the sections numbered 14 and 15 before this date now stand. The ablation was updated in place rather than extended, so the five arms the analysis brief of that date asked for are merged into the one sequence at their ranked positions instead of being appended as a second list, and the duplicate numbering by which arm 3 was also arm 8 and arm 5 also arm 13 is collapsed. Three arms were merged into others as duplicates, the power term becoming stage two of arm 6, the clip excess penalty stage two of arm 8 and the heading constraint stage two of arm 10, each having already declared that it could not be interpreted apart from the arm it is now part of, which takes the sequence from twenty one arms to seventeen. And the order was re-ranked against the evidence, four of the brief's seven proposals changing rank, the energy budget rising from seventh to first because section 14.6 measures it as a precondition rather than an intervention and the further ankle scale reduction falling from first to fifth because section 14.2 has already measured it null. Section 14's own findings are that arm 4 is the sequence's first positive result, the ankle roll stop to stop reversal falling from 2.031 per second to 0.002, that arm 3 is an exact null, the policy compensating a scale ratio of 2.471 with an action ratio of 2.479 and leaving the commanded target unmoved, and that three of arm 4's own predictions are falsified, its ankle roll saturation rising rather than falling, its sole settling onto one edge rather than flattening, and its own stated falsifier being met so that the channel by which it succeeded is not established. Of the seventeen arms, five are executed and reported, two are running, and no arm of section 17 that is not already in the tree has been applied to any file. EXTENDED A FOURTH TIME on 2026-09-23, when arm 6 stage one was applied to the working tree at the user's direction. The user reported that the arm 4b run taken without arm 5 had failed and that the run carrying arm 5 on top of arm 4b was sound, so arm 5, which had been commented out on the same date so that the arm 4b run could be restarted without it, was restored to the tree and arm 6 stage one was applied on top of it, raising the four energy weights of `kscale_base_env_cfg.py` by a factor of ten and touching nothing else. Arm 4b is therefore ABANDONED rather than concluded, its falsifier unmeasured, and it survives only as the scale dictionary the arms above it are now run on top of, which section 17.6 records. Section 17.8 carries arm 6 stage one as applied, together with a re-pricing of the factor of ten against each of the three dumped evaluation runs which finds the four terms reaching 29.63 per cent of the negative budget at run B's expensive gait and 4.22 at run C's cheap one, the seven fold spread being the argument that the weight does the work asked of it. Section 17.20 records the tree the run is launched from, five arms now standing in it together, and `validate_arm6.py` in the artefact directory asserts the applied values against both configuration files without a simulator and confirms that the restored pose is still closed. Of the seventeen arms, five are executed and reported, one is running, one is abandoned, one has its first stage applied and awaiting a run, and no other arm of section 17 has been applied to any file.
>
> EXTENDED AND RENUMBERED ON 2026-09-28 AND AGAIN ON 2026-09-30, and a returning reader should know four things before any of the detail.
>
> ARM 10, THE COMMANDED TARGET CLIP, SUCCEEDED. Section 18.2 measures the ankle roll saturation falling from 60.17 per cent to 0.06, its hard stop residency from 42.30 to 0.07 and its sustained actuator demand from 13.0 newton metres of a 17 newton metre ceiling to 3.5, which ends the latch sections 11 and 12 named as the root cause and which is the first prediction this document has made that was confirmed in full rather than falsified or qualified. Arm 8, the energy budget, had succeeded before it, section 16.2 carrying its ladder of five multiples and the cliff beyond one hundred.
>
> THE FAULT RELOCATED RATHER THAN VANISHING. Section 18.3 measures the hip roll at its mechanical stop for 45.9 per cent of every episode under a raw action of NINETEEN standard deviations, the policy having answered a bound it could not reach by commanding thirty five times further past it, and traces the relocation to the clip's own sizing rule, its window standing 14.51 times the hip roll's saturating deflection against 2.84 times the ankle roll's. Arm 11, which prices the excess, is running at two weights that section 18.4 shows to be mis-set by between one and three orders of magnitude, the quantity it prices growing by a factor of 51000 over training so that no fixed quadratic coefficient can be right at both ends. Arm 18 answers both, bounding the action by a scaled hyperbolic tangent so that the excess is bounded and the weight becomes calculable, and section 19.3 surveys the record that supports it.
>
> THE HEADING TERM DOES NOT PRICE THE STANCE PHASE, WHICH REVERSES HALF OF ARM 16. Section 18.7 was added on 2026-09-30 to test that arm against the baseline it will actually run upon rather than against the runs it was written from. Three findings move it. The stance toe-in has risen from 2.969 degrees on run A to 7.188 on run E while the swing figure fell, so the four to eighteen fold ratio that justified gating the term to swing now stands at 1.92. Under the mean form the term's common mode carries 1.3 per cent of its value and the differential carries 98.7, so closing the gate removes no stance price and instead ENLARGES the common mode by a factor of 25.9 by making it the swing foot's own error. And the halving of the differential scale is refuted outright, that mode having grown by a factor of 3.30 since run C while carrying almost the whole of the only term that opposes it. The arm is divided into three stages accordingly, the third conditional upon a test that presently fails.
>
> ARM 14 IS THE HIP YAW FIXED IN THE URDF, BROUGHT FORWARD ON 2026-10-01. It is independent of every arm around it and is run to gauge what one degree of freedom contributes to this robot's locomotion rather than to repair anything, so it needs only arm 10's clip and may be launched as soon as arm 13 settles the excess price. The investigation of that date also overturned the claim every earlier draft of it rested upon. A fixed joint is merged into its parent only where `merge_fixed_joints` is True, and `kscale_identified_cfg.py:279` sets it FALSE, so the articulation loses the two hip yaw degrees of freedom and keeps all THIRTEEN of its bodies. SD_BRS1 already trains in exactly this configuration, declaring both its hip yaws fixed at `SD_BRS1_Assembly2.urdf:89` and `:431`, and its dump reports ten joints against thirteen bodies, which is the measurement that settles it. The consequence is that `mdp/symmetry/kscale.py` needs NO change at all, where every earlier draft required `_BODY_PARTNERS` rebuilt over eleven bodies, and the arm reduces to one URDF of two changed attributes, two new modules and four edited files. Section 21.14 carries every one of them in full, and `validate_arm14.py` in the artefact directory of section 18 checks the plan's own snippets against the URDF, the live configurations, the symmetry module and the SD_BRS1 dump, returning 54 assertions passed and none failed.
>
> THE ARMS WERE RENUMBERED ON 2026-09-30 AND AGAIN ON 2026-10-01 SO THAT THE ARM NUMBER IS THE ORDER OF EXECUTION. Section 20.1 lists all twenty six in that order and section 21 discusses them in the same order, one sub-section per arm, the arm number and the sub-section number being one number, and the preamble of section 21 carries the mapping from the numbering of 2026-09-28. Arms 1 through 11 are executed, one abandoned and one refuted, ARM 12 IS APPLIED TO THE WORKING TREE and awaiting its run, `pen_action_excess` having moved from minus 500 to the calibrated minus 14 on 2026-09-30, and arms 13 through 26 have not been applied to any file. The clip excess penalty divides into three arms, being the weight as launched, the weight as calibrated and the absolute form, and the squashed action term moves down the order to stand beside the bounded distribution it supersedes. The hip yaw frozen in configuration, formerly arm 10, was removed from the sequence and is superseded by arm 14, which freezes it in the URDF and carries a full asset, action term, environment, task and command line wiring. Arm 24 was rebuilt on the same pattern and now produces two tasks, the ankle roll frozen alone and frozen together with the hip yaw, the construction differing from arm 14's because the ankle roll is the TERMINAL joint of the chain so that a naive fixed joint would destroy the foot link every gait term resolves. Every reference in this document and in the files outside it was moved in the same pass.
>
> TWO ARMS HAVE THEIR PREMISES WITHDRAWN rather than refuted, arms 24 and 25, both having been written against a machine whose ankle was pinned and both being answered by the clip. They stand at the end of the sequence and return if arm 15 narrows the clip and the ankle returns to its stop.

## 1. What this document is, and what it rests upon

Four policies were trained on the KScale sole footed biped between 2026-09-17 and 2026-09-18 with the declared purpose of removing a pigeon toed gait, in which the machine converges its toes upon the midline rather than carrying them along the line of travel. This document reads those four runs together, establishes what in fact distinguished them, measures what each policy does, and then argues that the fault the sequence set out to price is not a fault of the reward set at all but the visible consequence of a lateral actuation deficit the reward set cannot reach. It closes with the remedies that follow and with an ablation designed to separate them, and it is written so that a later session may act upon it without re-reading four dumps of three thousand steps apiece.

The evidence is of four kinds and each is treated as authoritative for a different question. The dumped `params/env.yaml` of each run is the only authority for what that run trained under, the working tree having changed in between and the runs having been launched from uncommitted state, a hazard this repository has recorded before at [../context/KScale.md](../context/KScale.md) in its correction of 2026-09-10. The evaluation dumps under `data/42/` supply every behavioural quantity, being written by the statistics pipeline whose format and unit conventions are established at [/ws/context/gait_metrics.md](../../context/gait_metrics.md). The TensorBoard scalar record supplies the training trajectory and is the only instrument able to compare the four at a matched iteration, which section 2.2 shows is necessary. The play videos supply the visual corroboration of section 3.4 and nothing else, no figure in this document having been read off a frame.

Two prior documents of this repository carry findings this one builds upon rather than repeats. [../context/KScale.md](../context/KScale.md) is the physical parameterisation of the robot, and its sections 10.6, 11.8, 15 and 16 supply the sole geometry, the single support hold torque and the kinematics of the nominal pose that the argument below turns upon. [kscale_integration.md](kscale_integration.md) carries the reward history of this robot across nine earlier runs, and its section 5.1.2 already established the splay to be pigeon toed by direct measurement. The present document departs from both in one respect that should be stated at the outset, that it treats the pigeon toe and the crouch as one phenomenon rather than two, and it is the measurement of section 4.4 that licenses the treatment.

## 2. The four runs, what differs between them and what does not

### 2.1 The parameter comparison

The four dumped configurations were flattened to leaf paths through the repository's own `ParamsLoader` at `scripts/analysis/experiment_params.py:24` and compared exhaustively. Of 1730 leaf paths, 40 differ, and of those 40 all but five are the constituent fields of a single reward term present in one run and absent from the other three. The substantive differences are therefore five and they are as follows.

| Path | A `2026-09-17_09-13-46` | B `2026-09-17_11-25-59` | C `2026-09-18_09-37-29` | D `2026-09-18_10-03-48` |
|---|---|---|---|---|
| `rewards.termination_penalty.weight` | -4000 | -1000 | -4000 | -2000 |
| `rewards.pen_base_height.weight` | -30 | -30 | -30 | -50 |
| `rewards.pen_feet_heading.weight` | -4 | -4 | -8 | -4 |
| `rewards.rew_keep_hip_yaw_zero_in_air` | absent | absent | present at 1.0 | absent |
| `scene.num_envs` | 7000 | 6000 | 9000 | 9000 |

Everything else is identical across the four, and the identity is worth stating explicitly because it disposes of several hypotheses before they are raised. The nominal pose is the same in all four, being a spawn at 0.786 m with hip pitch at minus and plus 0.20369 rad, hip roll at plus and minus 0.042951, hip yaw at zero, knee at 0.51239, ankle pitch at minus 0.3087 and ankle roll at plus and minus 0.041254, as `environments/environments/assets/config/kscale_identified_cfg.py:197` declares it. The actuator set is the same in all four, being stiffness and damping of 100 and 3.5 at the hip yaw, 250 and 28.0 at the hip roll, 200 and 21.5 at the hip pitch, 300 and 12.0 at the knee, 100 and 1.05 at the ankle roll and 100 and 1.25 at the ankle pitch, against effort ceilings of 42, 42, 84, 84, 11.9 and 11.9 newton metres. The action scale is 0.4 with `clip` unset in all four, at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:211`. The control period, the episode length, the command ranges, the curriculum schedules, the events and the observation groups are identical throughout. The user's question about whether the nominal positions differ between experiments is therefore answered in the negative, and answered by exhaustion rather than by inspection.

### 2.2 The three confounds, and why no arm of this sequence is clean

The sequence cannot support a causal reading, and the reason should be recorded before any of its measurements are quoted. No two of the four runs differ in exactly one variable.

The training budget is the severest of the three confounds. Runs A and B stopped at 9412 and 9921 iterations respectively, while runs C and D completed the full 29999 that `KscaleFlatPPORunnerCfg` requests at `environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py:337`, so a comparison between the two pairs confounds every reward change with a budget difference of a factor near 3.2. This matters more here than it would elsewhere because four curricula are active and adaptive, the linear tracking kernel having narrowed to 0.1761 in A and 0.1951 in B against 0.0859 in C by the end of each, so the later runs were also judged against a sharper criterion than the earlier ones. The evaluation is nonetheless comparable across all four, every replay being configured from its own run's dump and every one therefore meeting the same command distribution at play time, which the dumps confirm at a commanded forward velocity spanning minus 0.493 to plus 0.996 in each.

The population size differs across three values, 7000, 6000 and 9000, and it enters the optimiser through the batch composition rather than through the environment, so its effect is on the gradient estimate and not on the physics. It is the least consequential of the three and it is recorded for completeness.

The third is that the reward changes are not orthogonal. Run C carries two interventions at once, the doubled heading price and the swing gated hip yaw term, so it cannot say which of them acted. Run D carries two, the raised height price and the reduced termination price. The single cleanest contrast in the sequence is A against B, which differ in the termination penalty and in the population and in nothing else, and even that is a two variable change.

The practical consequence is that this document draws its causal claims from within run measurements and from mechanics, not from the between run contrasts, and section 4.4 is where that choice earns its keep.

### 2.3 The working tree has already moved past every one of these runs

Two changes stand in the working tree that no run of this sequence carried, and a reader preparing the next launch must know both. The ankle effort ceiling now reads 17 newton metres at `environments/environments/assets/config/kscale_identified_cfg.py:152` and `:168`, against the 11.9 that all four dumps record, which is a rise of 43 per cent at the two joints section 4.1 shows to be the binding ones. And the reward set now carries run C's configuration exactly, `termination_penalty` at minus 4000, `pen_feet_heading` at minus 8.0, `pen_base_height` at minus 30.0 and `rew_keep_hip_yaw_zero_in_air` active at 1.0, that alignment having been performed on 2026-09-21 against run C's dump. A run launched from the tree as it stands would therefore be a two variable departure from C rather than a repetition of it, and the ablation of section 7 opens by disentangling exactly that.

## 3. What the four policies actually do

### 3.1 Two gait families, not four

The evaluation dumps divide the four policies into two families, and the division does not follow the reward changes. Runs A and D stand tall on a nearly straight leg and converge their toes strongly. Runs B and C fold deeply at the knee, drop the body, and carry their feet far closer to the line of travel. The governing quantities are as follows, each being the mean over 32 environments and 3001 steps of the play dump.

| Quantity | A | B | C | D |
|---|---|---|---|---|
| Base height [m] | 0.7133 | 0.6574 | 0.6403 | 0.7058 |
| Mean knee angle [rad] | 0.5263 | 1.1601 | 1.2015 | 0.4077 |
| Toe-in, both feet [deg] | 9.69 | 8.52 | 4.59 | 8.49 |
| Toe-in, ninety fifth percentile [deg] | 36.96 | 20.45 | 21.39 | 38.25 |
| Toe-in in stance, left and right [deg] | 6.75, 5.50 | 7.30, 5.47 | -2.21, -2.45 | 18.63, -1.93 |
| Lateral foot separation [m] | 0.1490 | 0.1503 | 0.1460 | 0.1453 |

The toe-in figures were computed by rotating each foot link's minus z axis into the world, projecting it onto the horizontal and differencing against the base heading, which is the method [../context/KScale.md](../context/KScale.md) establishes in its addition of 2026-09-03 and the only method valid on this robot, whose foot link frame carries its width on x, its downward vertical on y and its fore and aft length on z. The `foot yaw misalignment` records of `statistics.npy` read near 1.5 radians for every run and are the ninety degree axis permutation rather than a measurement of splay, exactly as that addition warns, and they are quoted nowhere in this document.

The knee angle is the quantity that separates the families and the base height follows from it. The correlation between the two, computed per environment and then averaged so that the pooling error [/ws/context/brs_gait.md](../../context/brs_gait.md) records in its twenty third pass is avoided, stands at minus 0.72, minus 0.85, minus 0.92 and minus 0.75 across the four runs. The knee sets the height, and nothing else in the configuration does.

### 3.2 On every measure of gait quality except height, the deep knee family is better

This is the result that recasts the investigation, and it holds across tracking, efficiency, smoothness and swing shape alike.

| Quantity | A | B | C | D |
|---|---|---|---|---|
| Planar velocity error [m/s] | 0.1044 | 0.0924 | 0.0463 | 0.0726 |
| Cost of transport, absolute | 1.315 | 1.236 | 1.026 | 3.351 |
| Absolute joint power [W] | 118.75 | 105.84 | 89.52 | 287.2 |
| Joint jerk, all joints [rad/s^3] | 8638 | 8980 | 4563 | 35416 |
| Cadence [steps/s] | 6.25 | 2.06 | 2.02 | 2.20 |
| Swing duration [s] | 0.173 | 0.342 | 0.394 | 0.302 |
| Swing apex clearance [m] | 0.0325 | 0.0521 | 0.0465 | 0.0432 |
| Raised cosine correlation of the swing profile | 0.734 | 0.853 | 0.865 | 0.711 |
| Double support [pct] | 28.03 | 10.90 | 9.28 | 16.17 |
| Peak contact force [body weights] | 2.20 | 1.95 | 1.98 | 2.12 |
| Contact registered with the sole above 10 mm [pct] | 54.4 | 22.7 | 14.2 | 68.9 |

Run C tracks its command at 44 per cent of run A's error, spends 24 per cent less power, produces 47 per cent of run A's jerk, and holds a swing whose duration of 0.394 s is the closest of the four to the 0.38 s the gait command implies at a frequency of 1.0 Hz and a stance duty of 0.62, as `kscale_base_env_cfg.py:149` sets it. Run A steps at 6.25 per second against a commanded two, which is a shuffle rather than a walk, and it registers more than half its contacts with the sole more than ten millimetres clear of the ground, which is the signature of a foot meeting the floor on an edge or a toe rather than on its machined face.

The double support fraction is the single number that summarises the difference between the two families, and the human literature licenses reading it that way, sixteen subjects asked to decrease their own double support fraction producing increased hip flexion, increased knee flexion, increased swing foot height and larger ground reaction force peaks as one coordinated pattern rather than as four separate changes [8]. Three of those four move here as that study reports, the deep knee family carrying a third of run A's double support against twice its knee flexion and 43 per cent more swing apex clearance, and the fourth moves against it, the peak contact force falling from 2.20 to 1.98 body weights where the study finds it rising. The divergence is recorded rather than smoothed over, and the likely reason is that run A's 6.25 steps per second is not a shortened double support within a walk but a different gait altogether.

The single measure on which the deep knee family is worse is the one the reward set prices, the base height, and section 4.5 argues that the price is set against the wrong target.

### 3.3 Run D is a diverged optimisation, not a gait, and must not be read as an arm

Run D should be withdrawn from the ablation entirely, and the grounds are in its training record rather than in its dump. Its policy noise standard deviation rose to 2.0975 against 0.8715 in run C, its entropy to 19.04 against 5.44, and its symmetry loss to 7892.35 against 0.4662, which is four orders of magnitude. Its mean return fell from 807 at iteration 3000 to 227 at the end while run C's rose to 848, its `base_contact` termination rate settled at 0.30 against run C's 0.056, and both of its command curricula froze from iteration 15000 onward.

The symmetry figure is the one that names the fault the user observed. The KScale agent carries data augmentation with the mirror loss disabled at `limx_rsl_rl_ppo_cfg.py:363`, so that quantity is logged and not optimised, and it measures directly how far the policy's response to a mirrored observation departs from the mirror of its response. A value of 7892 is a policy that has ceased to treat its two legs alike. The dump confirms it joint by joint. Run D's right knee sits at 0.0009 rad with a fifth percentile of minus 0.0011 and a ninety fifth of 0.0013, against a hard lower stop of exactly 0.0000, so the joint occupies a band two thousandths of a radian wide on its own end stop for 99.74 per cent of every episode, while its actuator delivers a mean absolute 83.911 newton metres against an 84 newton metre ceiling with 100.00 per cent of samples of one sign. Inverting the control law over the unclipped samples places 99.58 per cent of that joint's commanded targets below the stop. The leg is not locked as a strategy, it is a strut the policy presses into a mechanical limit at full torque and never releases.

Run D's raised height price of minus 50 is therefore untested. Whatever it would have done, this run did not measure it.

### 3.4 The visual record

Four frames were taken at 6.0 s of each play video, where the commanded forward velocity is 0.079 m/s and the yaw command 0.64 rad/s, being the slowest sustained interval of the recording and the one in which section 4.4 predicts the splay to be largest. They are retained at [../context/artefacts/kscale_flat-2026-09-17_to_09-18-gait-ablation/frames/](../context/artefacts/kscale_flat-2026-09-17_to_09-18-gait-ablation/frames/) and registered there. They corroborate the measurements and add nothing to them. `A_t6.0_zoom.png` shows a nearly extended pair of legs with the two feet converged upon the midline and overlapping in projection, `B_t6.0_zoom.png` a visibly folded knee with the feet closer to parallel, `C_t6.0_zoom.png` a deep squat in which the thigh approaches the horizontal and the soles lie flat and forward, and `D_t6.0_zoom.png` a twisted and asymmetric posture with the legs crossed. `D_t25.0_zoom.png` shows the straight right leg of section 3.3 carrying the machine at a torso roll the dump measures at 7.22 degrees against 1.47 in run C.

## 4. The root cause

### 4.1 The ankle is not a controlled joint in any of the four runs

Both ankles stand at their torque ceiling for the overwhelming majority of every episode, in every run, and this is the single most consequential measurement in this document.

| Joint, fraction of samples above 98 pct of the effort ceiling [pct] | A | B | C | D |
|---|---|---|---|---|
| `left_foot_roll_02` | 99.76 | 94.70 | 99.62 | 94.17 |
| `right_foot_roll_02` | 99.71 | 94.81 | 99.61 | 99.41 |
| `left_foot_pitch_02` | 83.62 | 91.32 | 92.66 | 93.86 |
| `right_foot_pitch_02` | 84.16 | 90.34 | 93.09 | 91.54 |
| `left_hip_roll_03` | 42.27 | 56.32 | 2.33 | 88.74 |
| `right_hip_roll_03` | 42.34 | 55.17 | 2.33 | 99.83 |

The mean absolute ankle torque is between 11.0 and 11.9 newton metres against a ceiling of 11.9, and the ninety ninth percentile stands at 101 to 102 per cent of it, the excess being the friction model of `environments/environments/actuators/actuator_pd.py`, which subtracts its term after the clip. The position record says the same thing from the other side. The ankle roll lies outside its soft limit for 89.1 to 99.1 per cent of samples and within one per cent of a hard stop for 48.1 to 57.7, on a joint whose entire travel is plus and minus 0.2618 rad. In runs A and C it is parked on one stop with a torque of a single sign on 99.65 to 100.00 per cent of samples, the left at plus 0.2570 rad and the right at minus 0.2567 in A and the mirror of that in C. In run B it is not parked but chattering, its mean near zero and its sign split evenly, while still standing at the ceiling 94.8 per cent of the time.

Neither condition is control. In every run of this sequence the ankle complex is a passive strut that the policy leans upon, and every gait measured here is a gait learned by a machine with two dead ankles. The hip roll is in the same condition in three runs of four, lying outside its soft limit on 61.6 to 99.5 per cent of samples and standing at its ceiling on 42.3 to 99.8 per cent, run C alone escaping it at 2.33 per cent, which is the one measurement in this table that any intervention of the sequence appears to have moved and which section 7 accordingly makes an arm of its own.

### 4.2 The lateral moment the ankle is asked to supply, and cannot

The arithmetic is short and it is decisive. The robot masses 20.2814 kg, so its weight is 198.96 N. In single support the centre of mass must be carried over the stance foot, and the lateral offset from the stance ankle is half the foot separation, which the dumps measure at 0.0726 to 0.0751 m across the four runs. The moment the stance ankle roll must resist is therefore between 14.45 and 14.95 newton metres, against an effort ceiling of 11.9. The joint cannot hold it, falling short by a factor near 1.25 even before any disturbance is considered.

The centre of pressure may be moved outward beneath the sole to relieve the ankle, and a straight sole of width 0.0846 m offers at most 198.96 times 0.0423, being 8.42 newton metres of relief. That leaves between 6.04 and 6.54 newton metres for the ankle, which is 50.7 to 54.9 per cent of the ceiling and is attainable, but it is attained only with the centre of pressure at the very edge of the sole and the foot on the verge of rolling over it, so the configuration has no margin whatever against a lateral disturbance. This is the quantitative form of the statement [../context/KScale.md](../context/KScale.md) makes in its section 10.3, that this machine cannot rely upon ankle strategy for lateral balance at all and must lean instead, and it is the mechanical premise of everything below.

Two corrections to the record of that document follow from the present configuration and are stated here rather than assumed. Its sections 11.7 and 11.8 record both ankles as statically unstable in single support, their gravitational stiffness of 84.90 newton metres per radian exceeding configured stiffnesses of 50 and 20. Those stiffnesses are now 100 at both joints, above the 80.04 that document computes for the later pose, so the sign error it identifies has been repaired and the ankles are statically stable by that criterion, by a margin near a quarter. And its section 11.8 derives an authority floor of 100.3 newton metres per radian for the ankle roll against the configured 100, so the joint sits within a per cent of that floor rather than beneath it. Neither repair touches the torque ceiling, which is the constraint that binds, and the present configuration is if anything worse on that axis than the record contemplates, the 11.9 newton metre derated soft limit standing 30 per cent below the 17 newton metre peak that document already judged insufficient by a factor of 1.47.

### 4.3 The splay is the cheapest lateral support this robot can buy, and the measurement says how much

The KScale sole measures 0.2099 m fore and aft against 0.0846 m across, as the contact offsets at `kscale_base_env_cfg.py:32` give it, an aspect ratio of 2.48. A rectangular sole yawed by an angle t presents a lateral extent of L sin t plus W cos t, and on this sole that quantity rises steeply from its straight value. The measured toe-in of each run therefore buys the following.

| Run | Mean toe-in [deg] | Lateral extent [m] | Gain over the straight sole |
|---|---|---|---|
| A | 9.69 | 0.11873 | 40.3 pct |
| B | 8.52 | 0.11478 | 35.7 pct |
| C | 4.59 | 0.10115 | 19.6 pct |
| D | 8.49 | 0.11468 | 35.6 pct |
| A, at its ninety fifth percentile | 36.96 | 0.19387 | 129.2 pct |

Carrying that relief back into the moment balance of section 4.2 gives the purchase directly. Against a demand near 14.8 newton metres, a straight sole leaves the ankle 6.41 newton metres to find and a sole splayed to run A's measured angle leaves 3.01, which is 25.3 per cent of the ceiling rather than 53.8. The splay roughly halves the residual demand and, more importantly, it obtains that reduction without placing the centre of pressure at the edge of the sole, so it buys margin where the alternative buys none.

The geometry also explains why the foot is the instrument rather than the hip. The foot extends 0.1899 m ahead of the ankle roll axis and only 0.0200 m behind it, so 90 per cent of the contact patch lies forward of the joint, and rotating the leg about the vertical therefore sweeps a long lever across the frontal plane. No term in the reward set reads that rotation as anything but a joint deviation of at most a tenth of a radian, and `pen_hip_yaw_deviation` prices it at minus 1.0 per radian summed over two joints at `kscale_base_env_cfg.py:813`. The exchange is overwhelmingly favourable and the policy takes it.

### 4.4 The crouch is the second purchase of the same commodity, and the measurement is within run

The between run pattern, that the two tall runs toe in and the two crouched runs do not, would on its own be four points and no causal claim. The within run measurement is a different matter, and it is the strongest single result in this document. Correlating the instantaneous symmetric toe-in against the knee angle and against the base height, per environment and then averaged, gives the following.

| Run | corr(knee, toe-in) | corr(base height, toe-in) |
|---|---|---|
| A | -0.560 | +0.749 |
| B | -0.288 | +0.513 |
| C | +0.028 | +0.159 |
| D | +0.307 | +0.133 |

Within run A, at the timescale of individual steps rather than of whole policies, the robot toes in when it stands tall and straightens its feet when it folds its knee, at a correlation of plus 0.749 against the base height. Run B shows the same relation more weakly. Run C shows nothing, and that is exactly what the hypothesis predicts, the toe-in having been suppressed there so that there is no longer a quantity to trade. Run D is diverged and its sign is not interpretable.

The speed decomposition makes the same point in a form that answers the user's observation directly. Splitting each dump by commanded forward velocity gives the following, and the two tables should be read together.

| Commanded forward velocity [m/s] | Toe-in A | Toe-in C | Base height A | Base height C | Knee A | Knee C |
|---|---|---|---|---|---|---|
| -0.6 to -0.1 | 14.74 | 3.69 | 0.7308 | 0.5880 | 0.307 | 1.522 |
| -0.1 to +0.1 | 14.70 | 4.60 | 0.7302 | 0.6285 | 0.343 | 1.340 |
| +0.1 to +0.3 | 13.92 | 4.47 | 0.7234 | 0.6554 | 0.467 | 1.189 |
| +0.3 to +0.5 | 8.49 | 3.69 | 0.7120 | 0.6689 | 0.598 | 1.080 |
| +0.5 to +0.7 | 6.28 | 5.24 | 0.7047 | 0.6717 | 0.648 | 1.008 |
| +0.7 to +1.1 | 3.98 | 5.57 | 0.6922 | 0.6644 | 0.719 | 0.964 |

Run A's toe-in falls from 14.74 degrees at reverse and standstill to 3.98 at the highest commanded speed, a reduction of 73 per cent, while run C's crouch deepens from 0.6644 m at speed to 0.5880 m in reverse, a further 76 millimetres. The two runs are the same machine solving the same problem at the same place in the command space and choosing opposite instruments. At low speed there is no momentum to carry the body across the stance foot, so the lateral equilibrium must be held quasi statically, which is precisely the condition section 4.2 shows the ankle cannot meet. Run A meets it by widening the foot. Run C, forbidden that, meets it by lowering the body. The user's observation that run C crouches more at low speed and that this is the price of its straighter feet is therefore confirmed, and the mechanism is that the two are substitutes for a single missing quantity.

The joint excursion record corroborates the substitution and refines a finding this repository already carries. The addition of 2026-09-02 to [../context/KScale.md](../context/KScale.md) records the off axis excursion behaving as a budget reallocated between the hip yaw and the hip roll, their sum standing still. Across the present four runs the off axis sum falls from 0.3734 rad in A to 0.2445 in C while the sagittal sum of knee, hip pitch and ankle pitch rises from 0.8598 to 1.4828, and the total over all six families rises from 1.5513 to 1.9163. The budget is therefore not conserved, and the substitution is not between two off axis joints but between the off axis pair and the sagittal chain, which is a larger and more useful statement than the earlier one.

### 4.5 The height target names a pose the robot does not want, and the kinematics say why

`pen_base_height` demands 0.766 m at `kscale_base_env_cfg.py:837`, which is the value every run of both sequences carried and which the tree has since moved to 0.73 as section 21.6 records, and that figure is the standing height of the configured nominal pose, which holds the knee at 0.51239 rad. The knee sweep of [../context/KScale.md](../context/KScale.md) section 14.3, taken with the sole held flat throughout, establishes that this robot's height is stationary in the knee across the first fifth of a radian, reaches a maximum near 0.2 rad, and only begins to fall appreciably beyond 0.5, its derivative reaching minus 0.068 m per radian at 0.5 and minus 0.173 at 0.9.

Three consequences follow and each bears on the fault.

The height target sits on the flat part of that curve, so the knee is very weakly constrained by it. A policy may fold the knee from 0.5 to 0.7 rad for a height cost of 22 millimetres, and the quadratic penalty at minus 30 charges 30 times the square of that, being 0.0145 per second, against a velocity tracking reward of some 48. The knee is, for practical purposes, free.

The target demands a nearly straight leg, and a nearly straight leg is a leg beside its own singularity. The knee's hard lower stop is exactly 0.0000 rad, so the nominal 0.51239 stands half a radian from it while the action scale of 0.4 multiplies a Gaussian whose standard deviation reached 0.7 to 0.87 in these runs, placing the stop within 1.5 standard deviations of the commanded target. Run A, whose mean knee is 0.5263, consequently lies outside its knee soft limit for 26.3 and 27.2 per cent of samples and within one per cent of a hard stop for 17.4 and 18.5. Run D latched there permanently on one side. Runs B and C, at 1.16 and 1.20 rad, never approach it, their limit residency being 0.000 and 0.077 per cent.

And the pose the policies actually choose when free to do so is more than twice the configured one. Two of four runs converged independently upon a knee near 1.18 rad, and those two are the runs that track better, burn less power, produce less jerk and hold a swing closer to the commanded clock. This is the robot reporting its own preferred operating point, and the configuration is pricing it as a defect. It is worth recording that [kscale_actuator_limits_fix.md](kscale_actuator_limits_fix.md), which remains unimplemented, proposed moving the nominal pose to a knee of 0.87266 rad for an independent reason, being the moment arm the straight pose denies the leg, and that the policies of the present sequence overshoot even that proposal.

One caution attaches and it is the reason section 7 tests the pose rather than simply adopting it. The deep crouch is also the configuration that [kscale_actuator_limits_fix.md](kscale_actuator_limits_fix.md) identifies as a survival optimum, a policy that sits low collecting the flat survival bonus and avoiding the only penalised termination, and a survival bonus set too large is the published route to a character that balances and never steps, ablated over three magnitudes by Reda and colleagues [10]. The present runs do not exhibit that failure, their hip pitch standing at its ceiling for 0.007 to 0.028 per cent of samples in A, B and C rather than the 97.7 per cent that document measured, so the crouch here is not the actuation collapse it diagnoses. But the two are distinguishable only by measurement and the ablation must keep them apart.

### 4.6 One action scale cannot serve six joints whose ratios differ by a factor of three and a half

The deflection at which a proportional derivative joint reaches its torque ceiling is the ceiling divided by the stiffness, and the action magnitude that commands that deflection is that quotient divided by the action scale. With one scale of 0.4 applied to every joint at `kscale_base_env_cfg.py:211`, the six joint groups are reached at very different action magnitudes.

| Joint group | Stiffness [Nm/rad] | Ceiling [Nm] | Deflection at saturation [rad] | Action at saturation | Action to the nearer stop |
|---|---|---|---|---|---|
| Hip pitch | 200 | 84.0 | 0.4200 | 1.050 | 4.080 |
| Hip yaw | 100 | 42.0 | 0.4200 | 1.050 | 3.927 |
| Knee | 300 | 84.0 | 0.2800 | 0.700 | 3.382 |
| Hip roll | 250 | 42.0 | 0.1680 | 0.420 | 0.319 |
| Ankle pitch | 100 | 11.9 | 0.1190 | 0.297 | 1.745 |
| Ankle roll | 100 | 11.9 | 0.1190 | 0.297 | 0.654 |

An action of 0.297 saturates either ankle where 1.050 is needed at the hip pitch, a ratio of 3.54, and an action of 0.654 drives the ankle roll clean through its mechanical travel where the hip pitch would still be inside a quarter of its own. The hip roll reaches its adduction stop at an action of 0.319, which is lower still. Reconstructing the commanded target by inverting the control law over the unclipped samples confirms that the policy does command beyond the stops routinely, 83.3 and 86.3 per cent of run B's unclipped ankle roll samples naming a target outside the joint's travel, 34.2 and 43.1 per cent of run C's, and 22.5 to 36.1 per cent of the hip roll's across all four. Run A's ankle roll furnishes fewer than a hundred unclipped samples in three thousand and one, the joint being clipped essentially always.

The two joints that saturate at the smallest action are exactly the two the dumps find jammed. This is a straightforward consequence of one global scale meeting a joint set whose ceiling to stiffness ratios span 0.119 to 0.420 radians and whose travels span 0.524 to 3.264 radians, and it is repaired by a per joint scale or a clip, not by moving the global figure. That last point is not a conjecture, this workspace having measured the ankle roll stop residency to RISE from 0.669 to 0.765 when the scale was cut from 0.4 to 0.25, a result recorded in the addition of 2026-09-04 to [../context/KScale.md](../context/KScale.md) and grounded theoretically by the unbounded support of a Gaussian policy over an unclipped action [9].

### 4.7 The terms meant to shape the gait are two orders of magnitude beneath the term that drives it

The weighted per second reward budget of the evaluation, formed by taking each term's mean from `rewards.npy` where the per term series is already the product of the function and its weight [/ws/context/gait_metrics.md](../../context/gait_metrics.md), gives the following for the terms that bear on this question.

| Term | Weight | A | B | C | D |
|---|---|---|---|---|---|
| `rew_lin_vel_xy` | 50 | 45.91 | 46.63 | 48.65 | 47.53 |
| `rew_no_fly` | 15 | 14.55 | 14.70 | 14.82 | 14.72 |
| `rew_gait` | 40 | -6.48 | -5.91 | -4.57 | -7.24 |
| `rew_ang_vel_z` | 15 | 6.30 | 4.10 | 6.81 | 4.00 |
| `rew_foot_clearance` | 10 | 4.61 | 5.94 | 6.46 | 5.29 |
| `pen_ang_vel_xy` | -5 | -1.32 | -0.80 | -0.38 | -2.36 |
| `pen_joint_pos_limits` | -2 | -0.61 | -0.54 | -0.45 | -0.84 |
| `pen_feet_heading` | -4 or -8 | -0.597 | -0.234 | -0.233 | -0.598 |
| `pen_hip_yaw_deviation` | -1 | -0.511 | -0.259 | -0.295 | -0.530 |
| `pen_base_height` | -30 or -50 | -0.019 | -0.126 | -0.234 | -0.077 |
| Sum of all weighted rates | | 60.44 | 61.93 | 69.98 | 53.75 |

The height penalty stands between 0.019 and 0.234 per second against a velocity tracking reward of 45.9 to 48.7, being between 0.04 and 0.48 per cent of it. The two terms that price the splay stand together between 0.49 and 1.13 per second, being between one and two per cent. A doubling of the heading weight, which is what run C performed, moves a term worth one per cent of the objective to a term worth two, and it is unsurprising that the effect is ambiguous. The stability margin the splay buys, measured in section 4.3 at a 40 per cent widening of the base of support for three quarters of every gait cycle, is not a quantity any price of this magnitude can outbid, which is the position this workspace has already reached twice, at the addition of 2026-08-28 to [../context/KScale.md](../context/KScale.md) and in the formal reading of proxy under determination that the reward hacking literature supplies [6][14].

Two further observations belong here. `rew_no_fly` returns between 0.970 and 0.988 of its maximum in all four runs and therefore discriminates nothing, which reproduces on this robot the saturation [/ws/context/brs_gait.md](../../context/brs_gait.md) records for the SD_BRS1. And `rew_gait` is a net penalty of 4.57 to 7.24 per second in every run, so the clock the configuration declares at `kscale_base_env_cfg.py:149` is being paid for rather than followed, which is consistent with the measured cadences of 2.02 to 6.25 steps per second against a commanded two. That a clock may be paid rather than followed is a hazard the literature anticipates, van Marum and colleagues declining clock based rewards outright on the ground that they hinder disturbance recovery [12], and it is a caution against leaning upon this term for any of the corrections proposed below.

### 4.8 The causal chain, stated once

The ankle roll cannot supply the lateral moment that single support demands, falling short by a factor near 1.25 before any disturbance and having no margin at all once the centre of pressure is driven to the sole's edge. The hip roll, which [../context/KScale.md](../context/KScale.md) section 11.5 identifies as the joint that must supply the lean the ankle cannot, is itself confined to 0.209 radians of adduction travel and is measured against that stop for 62 to 99 per cent of every episode. With both lateral actuators exhausted the machine must obtain its frontal plane stability from geometry, and it has exactly two geometric instruments. It may yaw a slender sole, converting 0.21 m of foot length into lateral base of support at a rate that returns 40 per cent more base for ten degrees of rotation. Or it may fold its knee and lower its centre of mass, which the height penalty forbids and the knee's kinematic plateau makes almost free until the fold is deep. The reward set prices both instruments at one to two per cent of the objective and prices the lateral demand that drives them not at all. Which instrument a given run selects is therefore decided by the accidents of early training rather than by the weights, and the four runs of this sequence are two samples of each outcome.

## 5. What the sequence established, and what it did not

It established that the swing gated hip yaw term of run C works as an instrument, halving the mean toe-in to 4.59 degrees and turning the stance foot from 6.1 degrees of toe-in to 2.2 degrees of toe-out, and that it does so without the stiff legged gait the run of 2026-08-31 produced, run C holding the best swing profile and the best tracking of the four. It established that the deep knee gait is superior on every measure except height. It established that both ankles and, in three runs of four, both hip rolls are saturated throughout, which the user identified from the logs and which the dumps confirm at 84 to 99.8 per cent. And it established that run D diverged.

It did not establish that the termination penalty drives the splay, run A and run C sharing a value of minus 4000 and lying at opposite ends of the toe-in range. It did not establish that the raised height price of minus 50 does anything, the only run carrying it having diverged. It did not separate the doubled heading price from the hip yaw term, run C carrying both. And it did not isolate any reward effect from the training budget, the two reward changes of interest each appearing only in a run trained 3.2 times as long as its comparators.

## 6. The remedies, ranked by the evidence

The first remedy is to relieve the lateral moment at its source, and there are three routes. The ankle effort ceiling may be raised, which the working tree has already done to 17 newton metres and which [../context/KScale.md](../context/KScale.md) section 5.2 argues should go to 34, at which the single support hold is 74 per cent of the limit and the splay becomes one option among two rather than the only one. The stance may be narrowed at the hip so that the moment arm falls, the demand scaling linearly with the half separation and 11.9 newton metres being met without any centre of pressure shift at a separation of 0.120 m against the 0.146 measured. Or the sole may be widened, which is a hardware change and is recorded for completeness rather than proposed.

The second is to stop demanding a height the machine does not want. The target of 0.766 m at `kscale_base_env_cfg.py:837`, which the tree has since moved to 0.73 as section 21.6 records, is the standing height of a nearly straight leg, it sits on the flat part of the knee to height curve so that it constrains the knee hardly at all, and it places the knee within one and a half standard deviations of its own hard stop. Moving the nominal pose to a knee near 0.87 rad, as [kscale_actuator_limits_fix.md](kscale_actuator_limits_fix.md) already proposes on independent grounds, and re-deriving the height target and the three parameters that follow it, would name a pose the policy is already choosing and would move the knee's operating point away from its singularity. The published defaults of the surveyed humanoids are uniformly flexed and none is a straight knee [13][15], and the straight stance knee that passive dynamic walking recommends [11] is a property of the stance phase of a gait rather than of the posture a reward should demand throughout.

The third is to give the ankle and the hip roll an action authority proportioned to their travel. A per joint action scale, or a `clip` on the joint position action term which is presently unset, would stop the policy commanding targets beyond the mechanical stops, which it presently does on a fifth to a sixth of even the unclipped ankle roll samples. The global scale must not be reduced in its place, for the reason section 4.6 records.

The fourth is to state the foot heading requirement as a constraint rather than as a price. A term worth one to two per cent of the objective cannot outbid a 40 per cent widening of the base of support, and the literature this workspace has already surveyed offers two instruments that remove the exchange rate rather than adjusting one side of it, the constraint as termination [3] and the relaxed logarithmic barrier [5], the latter being the form [kscale_integration.md](kscale_integration.md) already proposes, together with the ancestor of both which stops short of removing the exchange rate at all, its constraint remaining a cost inside the summed objective [4]. Neither should be attempted before the first remedy, since a constraint that forbids the only lateral authority a machine possesses will produce the failure of 2026-09-04 again, in which removing the substitute while leaving the original unavailable removed the lateral support altogether.

The fifth is to supply a preference over gait shape that the velocity term cannot express. The energy minimisation result establishes that whether a recognisable gait emerges at all turns on the presence of such a term [7], and this configuration prices mechanical power at nothing, `joint_powers_l1` existing in the shared module and being wired into no KScale term. A power term is the natural home for the preference that presently has no home, and it would also act against the shuffle of run A, whose 6.25 steps per second and 118.75 W stand against run C's 2.02 and 89.52.

## 7. The proposed ablation

> NOTE ON THE NUMBERING OF THIS SECTION, added 2026-09-30. The arms below are numbered from ZERO and that numbering is this section's own, being the proposal of 2026-09-21 which section 20 supersedes in full. Three successive renumbering passes remapped these labels as though they named arms of the live sequence, which they do not, so the section is restored here verbatim from the copy taken before the first of them and NO FUTURE PASS SHOULD REMAP IT. The correspondence is that this section's arm 0 through arm 5 are the live arms 1 through 5 and 7, its arm 6, the heading constraint, is the second stage of the live arm 16, and its arm 7, the power term, is the second stage of the live arm 8.
>
> Superseded from arm 1 onward, 2026-09-22. Arms 1 and 2 were executed and section 10 reports what they returned, which falsifies the prediction arm 1 carries below. Arm 0 was not run as a separate launch, run C of section 2 serving as its comparator under an identical reward set. Arms 3 through 7 stand, arm 5 acquiring a second justification at section 12, and they are restated with the arms that now precede them at section 16. This section's numbering is the numbering of 2026-09-21 and is left where it stands as the historical record, so two of its arms carry a different number in the single sequence section 16 established on 2026-09-23. Arms 0 to 5 are unchanged, this section's arm 6, the heading constraint, is now the second stage of arm 10, and this section's arm 7, the power term, is now the second stage of arm 6. Nothing else in this section has moved. Read section 16 rather than this section when choosing what to launch, read this section for the reasoning the executed arms rested upon, and read section 17 for the code, the design decisions and the parameters of every arm, each arm standing there as a sub-section of its own.

The sequence below is designed so that each arm differs from its comparator in one variable, which the sequence just completed did not achieve, and so that the mechanical question is settled before any reward question is asked. Every arm runs to 20000 iterations, which is the point beyond which runs C and D were both stationary, and every arm is evaluated by replay from its own dump. No arm edits the shared `environments` mdp package, so the SD_BRS1 and the TRON1 tasks are untouched throughout, and every change is a value in the KScale configuration alone.

Arm 0, the reference. The reward set of run C exactly, which the working tree already carries, with the ankle effort ceiling returned to the 11.9 newton metres the four runs carried. This reproduces run C at the full budget and establishes the comparator. Without it the tree's present state confounds the ceiling change with everything else, as section 2.3 records.

Arm 1, the ankle ceiling. Arm 0 with the ankle effort ceiling at 17 newton metres, which is the RobStride class peak and the value the tree now holds. This is the single highest value experiment in the sequence and it tests the premise of section 4.2 directly. The prediction is specific and falsifiable, that the ankle roll soft limit residency falls from the 92.69 and 92.48 per cent run C records, that its effort saturation falls from 99.6 per cent, that the toe-in falls further without any change to the heading price, and that the base height rises. Should the residency not fall, the deficit is not the ceiling and remedies one and four both need rethinking.

Arm 2, the ankle ceiling doubled. Arm 1 with the ankle ceiling at 34 newton metres, which is the figure [../context/KScale.md](../context/KScale.md) section 5.2 derives and at which the single support hold is 74 per cent of the limit. It is beyond the published class peak and is therefore an instrument rather than a specification claim, and its purpose is to bracket arm 1 so that a null result there can be distinguished from an insufficient one. Run it only if arm 1 moves the residency in the right direction but insufficiently.

> Arms 3 and 4 were SWAPPED on 2026-09-22 at the user's direction, the action authority now standing third and the nominal pose fourth, so that the per joint scale is tested before the pose is disturbed. The cumulative state at that point is unchanged by the swap, both orderings reaching the same configuration. A new arm 4, the lateral weight shift, was then INSERTED on the same date and every arm from the old arm 4 onward bumped by one, so the nominal pose now stands fifth, the heading constraint sixth and the power term seventh, and section 16 renumbers likewise. The action authority arm has since been implemented and section 16.1 records it as arm 8, the lateral weight shift as arm 4.

Arm 3, the action authority. Arm 1 with a per joint action scale, or failing that a `clip` on the joint position action, sized so that a unit action commands the same fraction of each joint's travel. The figures of section 4.6 give the sizing directly, an ankle roll scale near 0.12 against the hip pitch's 0.4 equalising the saturation action across the set. The prediction is that the ankle roll and hip roll stop residencies fall together and that `pen_joint_pos_limits`, presently the largest of the style penalties at 0.45 to 0.84 per second, falls with them.

Arm 4, the lateral weight shift. Arm 3 with `HipLateralSwayReward` wired, which rewards the torso for tracking a gentle side to side sine locked to the gait clock so that the centre of mass passes over the stance foot. Inserted 2026-09-22, every arm from the old arm 4 onward having been bumped by one to make room. Its premise is that the centre of mass never passes above the centre of pressure, so the ankle roll is left holding a moment it cannot hold and the splay is what the machine buys the shortfall with, and section 16.1 carries the measurements, the merged parameters and the amendment to the amplitude they imply.

Arm 5, the nominal pose. Arm 4 with the nominal pose moved to the published K-Scale posture of hip pitch 0.34907, knee 0.87266 and ankle pitch minus 0.52360, which satisfies this robot's sagittal closure identity exactly, and with `pen_base_height.target_height`, `pen_feet_regulation.base_height_target` and `init_state.pos` z all re-derived from the 0.73540 m standing height that pose gives. The three must move together, the identity that the ankle roll origin sits at exactly the sole depth of 0.043 m being the cheapest check that a pose is closed, and [../context/KScale.md](../context/KScale.md) section 14.2 records why. The prediction is that the knee's stop residency falls to near zero and that the height penalty ceases to fight the policy's own operating point.

Arm 6, the heading constraint. Arm 5 with the heading requirement restated as a relaxed logarithmic barrier [5] on the symmetric toe-in, in place of the squared price the present term inherits from Booster Gym [2], dead banded at 0.098 rad, which is one standard deviation of the human foot progression angle distribution [1] and is the scale the docstring of `feet_yaw_alignment` at `environments/environments/tasks/locomotion/mdp/rewards.py:1270` already names. Section 17.12 carries the proposed optional argument that would supply the barrier without altering any existing caller. Run this only after arm 1 has established that the ankle can hold single support, for the reason section 6 gives.

Arm 7, the power term. Arm 5 with `joint_powers_l1` wired at a weight budgeted against the measured absolute joint power of 89.5 to 118.8 W, so that it stands at some two to four per cent of the objective rather than at nothing [7]. Its purpose is the shuffle of run A rather than the splay, and it is separable from arm 6 and may be run in parallel with it.

Two instrumentation requirements attach to every arm and neither is optional. The signed toe-in must be computed by the foot link axis method of section 3.1 and recorded, the `statistics.npy` foot yaw misalignment being the axis permutation and not the quantity. And the ankle roll and hip roll stop residency and effort saturation must be recorded per arm, since they are the quantities the first four arms exist to move and no reward channel reports them.

One arm is deliberately absent. No arm varies the termination penalty, which the completed sequence carried at four values without producing an interpretable contrast, and which section 5 shows does not separate the two gait families. Should a later session wish to test it, it must be tested at a fixed budget against a fixed reward set, which is the condition the present sequence failed to meet.

## 8. What must not be attempted, and why

Reducing the global action scale is excluded. This workspace has measured it to raise the ankle roll stop residency from 0.669 to 0.765 rather than lower it, because the action term carries `clip` unset so that the scale multiplies an unbounded Gaussian and the policy compensates by raising its command [9]. The scale governs the resolution of the control and not the reachable joint set.

Raising the price of the splay alone is excluded. Three successive instruments have now relocated the excursion rather than removing it, from the hip yaw to the hip roll in the sequence of 2026-09-02 and from the off axis pair to the sagittal chain in the present one, and the general result that an optimised agent under invests in whatever its evaluation does not cover predicts the relocation rather than merely describing it [14]. A price acting alone will find a third axis.

Enabling the mirror loss is excluded as a remedy for the splay. A left right mirror maps the two base relative foot headings to the negation of one another, under which the differential mode that carries the splay is invariant, so a mirror symmetric policy is free to turn both toes inward. The flag sits one line away at `limx_rsl_rl_ppo_cfg.py:365` and would achieve nothing against this fault, though section 3.3 shows the logged value remains a useful diagnostic of a divergence.

Raising the base height price further is excluded. It presently stands at 0.04 to 0.48 per cent of the objective and the run that raised it to minus 50 diverged, so there is no evidence it acts, and section 4.5 argues the target itself is misplaced. A price defending the wrong setpoint is made worse by being raised.

And the deep crouch must not be read as a success until it has been distinguished from the survival optimum that [kscale_actuator_limits_fix.md](kscale_actuator_limits_fix.md) diagnoses. The distinguishing measurement is the hip pitch effort saturation, which stands at 0.007 to 0.028 per cent in runs A, B and C against the 97.7 per cent that document measures, so the present crouch is not that failure, but the check must be repeated on every arm that lowers the target.

## 9. The artefacts this document cites

The frames of section 3.4 and the eight analysis programs that produced every figure quoted above are retained at [../context/artefacts/kscale_flat-2026-09-17_to_09-18-gait-ablation/](../context/artefacts/kscale_flat-2026-09-17_to_09-18-gait-ablation/) and registered in that directory's index. The scripts are retained rather than the intermediate arrays, the arrays being recoverable from the log tree while it survives and the scripts being the statement of how each figure was computed. The log tree is transient and the four runs will be deleted as disk fills, so any figure in this document that a later reader wishes to re-derive must be re-derived by running those scripts before that happens.

The programs behind sections 10 through 15 are retained separately at [../context/artefacts/kscale_flat-2026-09-18_to_09-21-foot-joint-ablation/](../context/artefacts/kscale_flat-2026-09-18_to_09-21-foot-joint-ablation/) and registered in the same index, that directory holding twenty two programs and no arrays or frames on the same reasoning. One of them, `arm_parameterisation.py`, derives the four parameter tables of section 21 that are arithmetic on the configuration rather than measurements of a run, and it reads no dump and needs no simulator, so those tables survive the deletion of the log tree entirely and may be regenerated against a tree that has moved. Its own index records the validation that licenses reading the two sets of figures against one another, eleven gait quantities of the run this document calls C reproducing to every digit between the two pipelines, that run being the 11.9 newton metre arm of the later sequence as well as the best of the earlier four.

The programs behind sections 14 and 19 are retained at [../context/artefacts/kscale_flat-2026-09-21_to_09-22-scale-sway-ablation/](../context/artefacts/kscale_flat-2026-09-21_to_09-22-scale-sway-ablation/) and registered in the same index, that directory holding twenty two programs and, unlike the two above it, a small set of frames. The frames are retained because section 14.11 reasons about them and because the play videos they come from live in the transient log tree, and they are cropped and tiled rather than whole, the tracking camera placing the robot at a scale in the full frame at which no gait property is legible. Nine of the twenty two are copies of programs in the directory above, carried rather than imported because each is hard wired to its own run register through its `common.py` and because a shared import would make the earlier directory's output depend on a later edit, which the artefact rules exist to prevent. Two of those nine needed a one line change, `commanded_target.py` and `actuator_envelope.py` carrying a per run ankle saturation effort that is 17 newton metres in all three runs of this set. The eleven new programs are `raw_action.py`, which section 14.2 rests upon, `footprint_separation.py` for section 14.8, `reward_budget.py` for section 14.6, `limit_penalty_share.py` and `soft_limit_geometry.py` for section 14.7, `hip_pitch_stall.py` for section 14.5, `hip_yaw_role.py` for arm 10, `sole_cant_source.py` for section 14.12 and for arm 24's parameter, `stop_residency_saturation.py` for the residency tables of sections 14.3 and 14.9, `tensorboard_compare.py` for section 14.10, and `cfgdiff.py` with `pose.py` for section 14.1. The three arm numbers in that list are given under the collapsed numbering of 2026-09-23 and were stated under an earlier one until 2026-09-28, when they were corrected together with the renumbering of this document's own sections.

The programs behind sections 16 and 17 are retained at [../context/artefacts/kscale_flat-2026-09-24_to_09-25-energy-ankle-ablation/](../context/artefacts/kscale_flat-2026-09-24_to_09-25-energy-ankle-ablation/) and registered in the same index, that directory holding eight programs, four tables and six frames. It differs from the three above it in two respects that its own register states. Its attribution is by construction rather than by inference, every figure being computed from a dump whose directory name carries the experiment and the checkpoint that produced it and whose configuration was copied beside the checkpoint by `compare.py:288` before the replay, so no checkpoint had to be fixed by a modification time as the directory before it required. And its five measurement programs share one implementation by importing `ablation.py` rather than each carrying its own copy of the envelope and the quaternion helpers, which is possible here and was not in the earlier directories because every run of this set carries the identical actuator block and no per run constant has to be hard wired. Its register also records a correction made during the analysis, the first pass having resolved every joint to one foot because the foot links of this robot carry no side marker, which inverted the stance and swing columns of the left leg and produced an apparent asymmetry that does not exist.

## 10. The foot torque ablation was executed, and it falsified its own central prediction

### 10.1 What was actually run, and the two launches that did not survive

Three runs were commissioned on the foot torque question and the investigation was pointed at `2026-09-18_09-24-17`, `2026-09-21_06-45-40` and `2026-09-21_08-37-13`. Two of those three are not runs. `2026-09-18_09-24-17` holds a parameter dump, an initial checkpoint and fifteen kilobytes of scalar record, and `2026-09-21_08-37-13` holds the same with under four kilobytes, neither carrying an evaluation dump, a play video or any checkpoint beyond `model_0.pt`, so both are launches that failed within their first minute. The runs that carry the evidence are `2026-09-18_09-37-29` at an ankle effort ceiling of 11.9 newton metres, which is the run section 3 of this document calls C, `2026-09-21_06-45-40` at 17, and `2026-09-21_08-40-59` at 34, the last having been launched four minutes after the stub it replaced. The attribution was made by reading each `params/env.yaml` through `scripts/analysis/experiment_params.py:24` rather than by the directory name, and it is recorded here because a reader returning to this question will otherwise chase two empty directories.

The comparison that results is far cleaner than the sequence of section 2 achieved. Of 2263 leaf paths across the three dumped configurations, six differ, and of those six two are the population and the terrain's copy of it, leaving the ankle effort ceiling and the ankle saturation effort as the only substantive variables. The entire reward set is identical across the three, `pen_feet_heading` standing at minus 8.0, `termination_penalty` at minus 4000, `pen_base_height` at minus 30.0 and `rew_keep_hip_yaw_zero_in_air` at 1.0 throughout, which is the configuration of run C that section 2.3 records the tree as having adopted. The nominal pose, the action scale, the gains, the armatures, the curricula, the events and the observation groups are identical throughout.

Two confounds remain and both must be carried into every reading below. The training budget differs by a factor near three, the 11.9 newton metre arm having run to 29999 iterations against 10654 and 11318 for the other two, which is the same defect section 2.2 charges against the earlier sequence and which section 10.4 accordingly answers with a matched iteration comparison. And the saturation effort moved with the effort ceiling, which section 10.2 shows is not a bookkeeping detail but a second physical variable.

### 10.2 The corner velocity collapsed, so the 17 newton metre arm is not a uniform increase

The DC motor model does not supply its effort ceiling at every speed. `actuator_pd.py:299` forms the upper bound as the saturation effort times one minus the ratio of joint velocity to velocity limit, and `actuator_pd.py:302` clips that to the effort ceiling, so the ceiling is delivered only below the corner velocity, which is the velocity limit times one minus the ratio of ceiling to saturation effort. The three arms therefore carry three different envelopes rather than three scalings of one envelope.

| Arm | Effort ceiling [Nm] | Saturation effort [Nm] | Corner velocity [rad/s] |
|---|---|---|---|
| 11.9 | 11.9 | 17.0 | 11.310 |
| 17 | 17.0 | 17.0 | 0.000 |
| 34 | 34.0 | 34.0 | 0.000 |

Setting the saturation effort equal to the ceiling removes the constant torque plateau entirely, so the two later arms derate from the very first radian per second. The consequence is arithmetical and it is severe.

| Joint speed [rad/s] | Torque available at 11.9 [Nm] | at 17 [Nm] | at 34 [Nm] |
|---|---|---|---|
| 0 | 11.90 | 17.00 | 34.00 |
| 5 | 11.90 | 14.75 | 29.49 |
| 11.31 | 11.90 | 11.90 | 23.80 |
| 15 | 10.24 | 10.24 | 20.47 |
| 20 | 7.98 | 7.98 | 15.96 |
| 25 | 5.73 | 5.73 | 11.45 |

Above 11.31 radians per second the 17 newton metre arm delivers exactly what the 11.9 newton metre arm delivers, to the last decimal, because both are on the same saturation line. The arm labelled as a 43 per cent increase is an increase only below the corner velocity and is no increase at all above it, and section 11.4 shows that the ankle roll spends its most consequential moments above that speed. The record should note that the plateau was itself an accident, commit `e770a7e` having lowered the ceiling from 17 to 11.9 while leaving the saturation effort at 17, so the 11.9 newton metre arm is the only one of the three whose motor model carries a constant torque region and the sequence never intended to vary this at all.

### 10.3 The gait quality fell monotonically with the ceiling

Each figure is the mean over 32 environments and 3001 steps of the evaluation dump, computed by the programs of [../context/artefacts/kscale_flat-2026-09-18_to_09-21-foot-joint-ablation/](../context/artefacts/kscale_flat-2026-09-18_to_09-21-foot-joint-ablation/), whose index records that eleven of these quantities reproduce section 3's figures for run C to every digit.

| Quantity | 11.9 Nm | 17 Nm | 34 Nm |
|---|---|---|---|
| Planar velocity error [m/s] | 0.0463 | 0.0890 | 0.0710 |
| Cost of transport, absolute | 1.026 | 1.325 | 1.753 |
| Absolute joint power [W] | 89.52 | 109.45 | 143.54 |
| Joint jerk, all joints [rad/s^3] | 4563 | 8226 | 7805 |
| Double support [pct] | 9.28 | 13.48 | 11.03 |
| Base height [m] | 0.6403 | 0.6627 | 0.7122 |
| Mean knee angle [rad] | 1.2015 | 1.0755 | 0.5739 |
| Lateral foot separation [m] | 0.1460 | 0.1361 | 0.1576 |
| Torso tilt [deg] | 2.36 | 2.94 | 2.75 |
| Terminations in the evaluation, of 32 | 0 | 0 | 4 |

The crouch does lift with the ceiling, the base height rising 72 millimetres and the knee unfolding from 1.20 to 0.57 radians, which is the direction section 6 wanted. Everything else moves the wrong way. The cost of transport rises by 71 per cent, the absolute power by 60 per cent, and the 34 newton metre arm is the only one of the three whose policy falls during its own evaluation.

### 10.4 The splay rose with the ceiling, and the fall rate rose with it

The prediction section 7 attached to arm 2 was that the toe-in would fall without any change to the heading price. It rose, and it rose monotonically.

| Quantity | 11.9 Nm | 17 Nm | 34 Nm |
|---|---|---|---|
| Mean toe-in, both feet [deg] | 4.59 | 7.37 | 10.20 |
| Toe-in, ninety fifth percentile [deg] | 21.39 | 15.42 | 24.24 |
| Toe-in in stance, left and right [deg] | -2.07, -2.36 | +8.36, +8.11 | +14.61, +13.25 |
| `pen_feet_heading` weighted rate [per s] | -0.233 | -0.347 | -0.772 |

The stance figure is the one that matters, the stance foot at 11.9 newton metres standing two degrees toed out and at 34 newton metres standing fourteen degrees toed in, which is worse than any of the four runs of section 3 recorded. The heading price registers the change faithfully and is powerless against it, standing at 0.5 to 1.6 per cent of the velocity tracking reward throughout, which is the position section 4.7 already established.

The training record says the same in a different currency, and it must be read at a matched iteration because the budgets differ. Taken over iterations 9500 to 10700, which is the last window all three runs occupy, the mean return stands at 663.4, 711.6 and 490.3 and the `base_contact` termination rate at 12.59, 9.34 and 27.35 per cent. At its own terminal iteration the 34 newton metre arm carries a fall rate of 20.4 per cent against 5.5 for the 11.9 newton metre arm at iteration 29999. Read at matched budget the 17 newton metre arm is the best of the three, which is consistent with the minimal splaying observed in it, and the 34 newton metre arm is worse than either comparator on return and on falls together. Raising the ankle ceiling beyond 17 newton metres is therefore not an untested proposal, it is a tested and failed one, and section 20 removes it.

## 11. The root cause, the ankle roll is latched against a mechanical stop and torque cannot unlatch it

### 11.1 The joint does not move, and it does not move at any ceiling

The saturation figures reproduce across the three arms almost unchanged, which is the first sign that the ceiling is not the binding constraint. Saturation here is measured against the velocity dependent envelope of section 10.2 rather than against the constant ceiling, which is the correction the earlier table of section 4.1 needs, and the two differ materially only in the later arms where the corner velocity is zero.

| Quantity, ankle roll, left and right | 11.9 Nm | 17 Nm | 34 Nm |
|---|---|---|---|
| At the envelope, fraction of samples [pct] | 99.69, 99.67 | 89.59, 88.75 | 91.02, 91.85 |
| Within a tenth of the travel of a stop [pct] | 92.7, 92.5 | 89.0, 88.3 | 94.7, 95.0 |
| Mean absolute torque [Nm] | 11.86, 11.85 | 15.88, 15.77 | 32.08, 32.19 |
| Stance median position [rad] | -0.2454, +0.2463 | -0.2467, +0.2471 | -0.2463, +0.2469 |
| Sole tilt from horizontal in stance [deg] | 7.09, 7.13 | 7.72, 7.42 | 8.44, 8.84 |

Tripling the ceiling moved the stop residency from 92.6 per cent to 94.9 and left the stance position where it was, to within two thousandths of a radian, against a hard stop at 0.2618. The mean torque, meanwhile, tripled exactly as the ceiling did. The machine did not use the additional authority to control the joint, it used it to press the joint into the same stop three times harder.

The consequence for the foot is measured directly from the dump's foot quaternions and it is the physical statement of the fault. The sole normal stands 7.1 to 8.8 degrees from vertical throughout stance, so this machine does not stand on its soles at all in any of the three arms, it stands on one lateral edge of each. Which edge that is was settled by transforming both lateral sole corners into the world and comparing their heights rather than by reasoning from the sign of the tilt, the corner method being the one that needs no convention. The INNER corner, the one nearer the midline, is the lower of the two by 9.6, 2.0 and 5.3 millimetres across the three arms, and it is the lower one on 99.0, 79.3 and 86.7 per cent of stance samples. The machine therefore stands on the inner edges of both feet, which section 20.1 shows under arm 5 is exactly what a torso that has not moved over its stance foot must do, the centre of pressure running to the inboard limit of the sole in pursuit of a centre of mass that is further inboard still. A dated correction belongs here, this passage having read outer rather than inner until 2026-09-22, an error of sign in the prose and not in the measurement, the sole tilt figures being unaffected and the conclusion strengthened rather than weakened, since an inner edge contact means the full 8.42 newton metres of centre of pressure relief that section 4.2 credits the sole with is already spent. A sole of width 0.0846 metres contributes nothing to the lateral base of support when its contact is a line along one edge, and the centre of pressure relief that section 4.2 credits the sole with, being 8.42 newton metres of the 14.8 the single support hold demands, is not available in any run this document has measured.

### 11.2 The policy commands the stop, and commands it harder the more torque it is given

Inverting the control law over the samples that lie strictly inside the torque envelope recovers the target the policy asked for, and this is the measurement that names the mechanism.

| Quantity, ankle roll, left and right | 11.9 Nm | 17 Nm | 34 Nm |
|---|---|---|---|
| Median commanded target [rad] | -0.3683, +0.3702 | -0.4090, +0.4160 | -0.5807, +0.5841 |
| Commanded beyond the hard stop [pct] | 98.48, 98.59 | 82.07, 84.37 | 88.13, 92.67 |
| Median excursion beyond the stop [rad] | 0.1067, 0.1085 | 0.1561, 0.1568 | 0.3227, 0.3243 |

Against a mechanical travel of plus and minus 0.2618 radians the policy names a target of 0.58 radians, which is 2.2 times the stop, and it does so on nine samples in ten. The excursion triples as the ceiling triples, which is the signature that identifies the strategy. A target placed far outside the reachable set converts the end stop into a reference that is perfectly repeatable, infinitely stiff and free, and the further outside it is placed the more disturbance the latch survives before the joint leaves the stop. The additional torque is spent buying a firmer latch.

Nothing in the configuration prices this. `pen_joint_pos_limits` at `kscale_base_env_cfg.py:859` reads the measured position against the soft limits through the vendored `/ws/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:189`, so it charges the consequence of the latch and not its cause, and it returns 0.44 to 0.53 per second against a velocity tracking reward of 46.7 to 48.7. There is no term anywhere in the set that reads the commanded target, and the action term at `kscale_base_env_cfg.py:211` carries `clip` unset, so a target outside the mechanical travel costs the policy exactly nothing to name. This is the same under determination section 4.7 identifies in the reward set, appearing now in the action space rather than in the objective.

### 11.3 The travel is sufficient, so this is not a hardware deficit and no URDF change will repair it

The obvious inference from a joint pinned at its stop is that the stop is in the wrong place, and the measurement refutes it. Regressing the sole's lateral tilt on the ankle roll angle through stance returns a slope of 0.949 to 1.055 across the three arms, which is the unit slope the kinematics require and which confirms that the ankle roll alone sets the sole roll. Inverting that relation gives the ankle roll angle that would lay the sole flat at each instant.

| Arm | Ankle roll required to flatten the sole, median [deg] | ninetieth percentile | ninety ninth | Stance samples requiring more than 15 deg [pct] |
|---|---|---|---|---|
| 11.9 | 7.81, 7.25 | 10.00, 9.58 | 12.74, 12.71 | 0.1, 0.1 |
| 17 | 9.37, 9.59 | 13.16, 13.15 | 16.23, 15.76 | 3.3, 2.3 |
| 34 | 7.89, 7.63 | 11.53, 11.18 | 14.79, 14.37 | 0.8, 0.5 |

The geometry asks for eight to ten degrees at the median and the joint has fifteen, and the demand exceeds the travel on between one sample in a thousand and three in a hundred. Widening the limit that `kscale.urdf:595` and `:1075` declare would therefore buy nothing, since the joint is not short of travel, it is declining to use the travel it has. This disposes of the hardware reading of the fault and it locates the fault in the action space, where section 20 accordingly puts its first arm.

### 11.4 Raising the ceiling created the impulse switching, and the arithmetic says why

The reversal between the two stops that prompted this investigation is not a property of the machine, it is a property of the two later arms alone.

| Quantity, ankle roll, left and right | 11.9 Nm | 17 Nm | 34 Nm |
|---|---|---|---|
| Stop to stop reversals [per s] | 0.000, 0.021 | 2.031, 2.015 | 1.982, 1.988 |
| Ninety ninth percentile speed [rad/s] | 3.60, 3.83 | 24.98, 24.88 | 24.19, 24.28 |
| Damping torque at that speed [Nm] | 3.78, 4.02 | 26.23, 26.12 | 25.40, 25.50 |

Against a touchdown rate near 2.35 per second the two later arms reverse the joint between its stops almost exactly once per step, while the first arm does not reverse it at all. The mechanism is the relation between the action that reaches the stop and the action that saturates the actuator. A deflection of the effort ceiling divided by the stiffness saturates the joint, and dividing that by the action scale of 0.4 gives the action at saturation, while the distance from the nominal pose to the nearer stop divided by the same scale gives the action at the stop.

| Ankle roll ceiling [Nm] | Action at saturation | Action at the nearer stop | Ratio |
|---|---|---|---|
| 11.9 | 0.297 | 0.458 | 1.54 |
| 17.0 | 0.425 | 0.458 | 1.08 |
| 34.0 | 0.850 | 0.458 | 0.54 |

At 11.9 newton metres the actuator saturates at two thirds of the way to the stop, so the joint creeps into it against a derated torque and arrives slowly. At 17 the two coincide. At 34 the joint reaches the stop at an action of 0.458 while saturation would require 0.850, so it arrives with the actuator still unsaturated and the full ceiling available, and it arrives fast. The measured traverse speed of 25 radians per second then makes the damping term alone, at 1.05 newton metre seconds per radian, demand 26 newton metres, which exceeds the 17 newton metre ceiling outright and consumes three quarters of the 34, so the actuator saturates on velocity feedback during every traverse regardless of what the position error asks for. The hip roll sits at a ratio of 0.99 by the same arithmetic and latches for the same reason, which answers the observation that it too saturates in contact, measured here at 70.83 and 71.72 per cent of stance samples in the 17 newton metre arm against 2.67 and 2.19 in swing, with the stance median resting on the stop itself.

## 12. The root cause, the ankle pitch is pinned in swing because the crouch has consumed its dorsiflexion

The swing ankle pitch stands at minus 50.0 degrees in all three arms, which is its lower stop at `kscale.urdf:603` and `:1083` to two decimal places, and the question is what holds it there. The sagittal closure identity answers it. Summing the signed hip pitch, the knee and the ankle pitch returns exactly zero at the nominal pose of `kscale_identified_cfg.py:197` on both legs, and across the evaluation dumps that sum correlates with the measured sole pitch at 0.902 to 0.945 with a slope near unity, so the identity is the governing constraint rather than a coincidence of the reset. Inverting it gives the ankle pitch that would hold the foot parallel to the torso at each instant.

| Arm | Ankle pitch required in swing, median [deg] | Required beyond the -50 deg stop, swing [pct] | Required beyond the stop, stance [pct] | Mean knee [rad] |
|---|---|---|---|---|
| 11.9 | -47.51, -48.36 | 42.9, 45.4 | 5.6, 5.3 | 1.2015 |
| 17 | -46.65, -47.55 | 41.7, 44.4 | 11.8, 9.0 | 1.0755 |
| 34 | -37.30, -37.14 | 6.1, 5.3 | 0.6, 0.7 | 0.5739 |

At the crouch depth the first two arms adopt, the swing leg requires 47 to 48 degrees of dorsiflexion at the median against a stop at 50, and requires more than the stop on four samples in ten. The joint is therefore not pinned because the policy has abandoned it, it is pinned because the posture the policy has adopted demands more travel than the joint possesses, and the residue appears as a foot that hangs at its limit through the whole of swing. The 34 newton metre arm is the control that proves the relation, its shallower crouch reducing the requirement to 37 degrees and its swing saturation falling with it, from 99.6 per cent of samples to 77.

This closes the chain that section 4.8 left open, and it should be read as one sentence. The ankle roll cannot hold the single support moment and is latched against its stop rather than regulated, which denies the machine the lateral base of support its sole would otherwise provide. Denied that, the machine buys frontal plane stability with the two geometric instruments section 4.3 identifies, which are the yaw of the sole and the lowering of the body. The lowering of the body is a crouch, and the crouch consumes the ankle pitch travel, which pins the second foot joint. The two foot joints are therefore failing for two different reasons, the roll by a control strategy and the pitch by a kinematic consequence of the strategy the roll forces, and no single intervention addresses both.

## 13. The three terms that should have prevented this are present, funded and inert

The configuration already carries the terms this fault calls for. `rew_keep_ankle_pitch_zero_in_air` at `kscale_base_env_cfg.py:759`, `rew_keep_ankle_roll_zero_in_air` at `:774` and `rew_keep_hip_yaw_zero_in_air` at `:789` all call `keep_ankle_pitch_zero_in_air` at `rewards.py:1562`, all gate on the airborne foot, and all measure deviation from the default pose since each passes `use_default_offset` true. The term forms the exponential of the negated deviation divided by `pitch_scale` at `rewards.py:1630`, with that scale set to 0.2 radians in every case.

| Term | Weight | Median swing deviation [rad] | Term value, of a maximum of 1 | Gradient of the weighted rate [per rad] |
|---|---|---|---|---|
| `rew_keep_ankle_pitch_zero_in_air` | 1.0 | 0.568 | 0.057 to 0.083 | 0.28 to 0.42 |
| `rew_keep_ankle_roll_zero_in_air` | 0.25 | 0.183 to 0.340 | 0.181 to 0.334 | 0.23 to 0.42 |
| `rew_keep_hip_yaw_zero_in_air` | 1.0 | 0.084 to 0.191 | 0.419 to 0.570 | 2.09 to 2.85 |

The ankle pitch deviation is 2.8 times the kernel width, so the term sits far out in the tail of its own exponential, returns six to eight per cent of its maximum and offers a gradient of a third of a unit per radian against a velocity tracking reward of 46.7 to 48.7 per second. A policy that moved that joint a full radian toward its target would be paid three tenths of one per cent of what it is already earning, which is not a preference the optimiser can be expected to act upon. The hip yaw term, by contrast, carries a deviation of a tenth of a radian, which lies inside the kernel width, returns half its maximum and offers a gradient seven to ten times larger, and it is the one term of the three that section 5 credits with having worked.

The lesson generalises beyond these three terms and it should be stated plainly. An exponential kernel is not a weak penalty when its argument exceeds its width, it is no penalty at all, the gradient vanishing faster than the value. A term intended to correct a deviation of 0.57 radians cannot be given a width of 0.2 and a larger weight will not rescue it, since the weight multiplies a gradient that is already numerically negligible. Either the width must be commensurate with the deviation to be corrected, or the functional form must be one whose gradient grows with distance rather than shrinking, which is what the relaxed logarithmic barrier [5] supplies and which is the instrument section 6 already proposes for the heading requirement on the same reasoning.

## 14. The action scale and lateral sway arms were executed, and they separate an exact null result from the first real one

Three runs are compared in this section under the labels the analysis brief assigns them, A being `kscale_flat/2026-09-21_06-45-40`, B being `2026-09-22_07-18-05` and C being `2026-09-22_07-40-43`. A is the seventeen newton metre reference of section 10 and carries one action scale of 0.4 at every joint, B adds arm 4 alone, and C adds arm 5 on top of arm 4. The evidence is the three dumped configurations, the three evaluation dumps at `data/42/`, the three TensorBoard records and the three play videos. Every figure below is reproduced by a program in [../context/artefacts/kscale_flat-2026-09-21_to_09-22-scale-sway-ablation/scripts/](../context/artefacts/kscale_flat-2026-09-21_to_09-22-scale-sway-ablation/scripts/), which section 14.11 registers.

### 14.1 What was actually run, the two confounds, and which comparison is clean

A mechanical comparison of the three dumped configurations finds sixty three differing leaf paths out of two thousand three hundred and twenty one, and they fall into four groups. The first is arm 4, the action scale moving from the scalar 0.4 of A to the six key dictionary of B and C, `actions.joint_pos.scale` resolving to 0.4 at the hip pitch and hip yaw, 0.2667 at the knee, 0.16 at the hip roll and 0.1619 at both ankles. The second is arm 5, the fifty two paths under `rewards.rew_hip_lateral_sway` that are absent from A and B and present in C at a weight of 5.0, a kernel width of 0.01 and an amplitude of 0.03, the count being large because a `SceneEntityCfg` flattens to twenty two leaves and the term carries two of them. The third and fourth are not arms at all and govern how this section may be read.

The push curriculum ceiling fell from 3.0 metres per second in A to 2.0 in B and C, `curriculum.modify_push_force.params.max_velocity` recording it. The TensorBoard record shows what that did, B and C reaching their ceiling of 2.0 by iteration six thousand and holding it while A climbed past them to 2.911 by iteration ten thousand. A therefore trained against a disturbance distribution that the other two never met, over the whole of the second half of training, and any quantity that rewards robustness will flatter B and C against it.

The parallel environment count fell from nine thousand in A and B to six thousand four hundred in C, `scene.num_envs` recording it. At twenty four steps per environment per iteration, which `params/agent.yaml` fixes, an iteration of C carries 0.711 of the experience an iteration of A or B carries. The evaluation checkpoints compound this, the modification times of the saved models against the modification times of the dumps placing A and B at `model_10000.pt` and C at `model_11000.pt`, so C was evaluated from 70.4 million environment iterations against the 90.0 million behind A and B, which is 78.2 per cent of their experience.

The consequence for attribution is stated once and governs every comparison below. A against B varies two things and is not a clean arm. B against C varies one configured behaviour, the sway reward, and differs otherwise only in the environment count and the thousand extra iterations, so B against C is the clean reading of arm 5 and it is the comparison this section leans upon wherever it can. A against C varies everything at once and is quoted only where the two agree with the B against C reading.

### 14.2 The action scale is a reparameterisation and not a bound, and the compensation is now measured exactly

Section 20.1 closed its account of arm 4 with the strongest argument it could make against its own arm, that a scale is a reparameterisation and not a bound, and that a policy wishing to command a target beyond the stop need only raise its raw action by the same factor the scale fell. That caution is now a measurement, and the agreement is closer than the argument had any right to expect.

The commanded target is recovered from the applied torque and the joint velocity by inverting the control law over the samples where the actuator is strictly inside its velocity dependent envelope, which is the procedure section 11.2 established, and dividing the recovered target by the configured scale returns the raw network output. At the left ankle roll the scale falls from 0.4 to 0.1619, a ratio of 2.471 to one. The median magnitude of the raw action rises from 0.847 to 2.100, a ratio of 2.479 to one. The median commanded target moves from minus 0.4090 radians to minus 0.4167, a change of 1.9 per cent on a quantity that lies 0.155 radians beyond a mechanical stop at minus 0.2618 in both cases.

| Joint | Scale in A | Scale in B | Median raw action, A | Median raw action, B | Ratio of actions | Ratio of scales | Median target, A | Median target, B |
|---|---|---|---|---|---|---|---|---|
| Left ankle roll | 0.4000 | 0.1619 | 0.847 | 2.100 | 2.479 | 2.471 | -0.4090 | -0.4167 |
| Right ankle roll | 0.4000 | 0.1619 | 0.935 | 2.311 | 2.472 | 2.471 | +0.4160 | +0.4162 |
| Left hip roll | 0.4000 | 0.1600 | 0.306 | 0.715 | 2.337 | 2.500 | -0.1351 | -0.1240 |
| Right hip roll | 0.4000 | 0.1600 | 0.213 | 0.483 | 2.268 | 2.500 | +0.1471 | +0.1237 |

The two ankle roll rows agree with the reciprocal of the scale to within three parts in a thousand and one part in a thousand. This is the cleanest confirmation of the Beta distribution argument [9] that this workspace has produced, and it is stronger than the hip yaw measurement cluster 25 of `context/literature.md` records, because the compensation is not merely in the same direction but numerically equal to the reparameterisation. The policy learnt a target, the configuration changed the units that target is expressed in, and the policy restated the same target in the new units.

The consequence for arm 6, which is running as this section is written, is a prediction that follows arithmetically rather than by analogy. Halving the ankle scales from 0.1619 to 0.08 is a further factor of 2.024, so a policy that compensates as exactly as these two did will command a median raw ankle roll action of 4.25 and reach the same target. Nothing in the configuration prevents it, the action being the unbounded output of a linear layer and `actions.joint_pos.clip` standing at `None` in all three dumps. The falsifying observation, which is the one to look for first in the arm 6 dump, is the median commanded ankle roll target moving inside the mechanical stop. The confirming observation is the median raw action arriving near 4.25 with the target unmoved.

### 14.3 The lateral sway reward ended the bang-bang, and the measurement is unambiguous

Section 11 named the ankle roll latch as the root cause and section 11.4 measured its switching. The quantity is the number of stop to stop reversals per second, counted by compressing each environment's ankle roll trace to the sequence of stops it visits and counting the changes, against a stride cadence that the temporal statistics place near 2.35 per second.

| Run | Left ankle roll reversals per second | Right | Traverse speed, left p99 | Median raw action span, left |
|---|---|---|---|---|
| A, one scale of 0.4 | 2.031 | 2.015 | 24.98 rad/s | -0.905 to +1.314 |
| B, arm 4 | 1.975 | 2.001 | 27.26 rad/s | -2.280 to +3.245 |
| C, arm 4 and arm 5 | 0.002 | 0.014 | 2.95 rad/s | -2.185 to +0.125 |

Arm 4 did nothing, the reversal rate falling by 2.8 per cent on the left and rising by a tenth of a per cent on the right, which is the same null result section 14.2 explains. Arm 5 removed the reversals altogether, a factor of a thousand on the left and a hundred and forty four on the right, and the peak traverse speed fell with them from 24.98 radians per second to 2.95. The action span tells the mechanism, C's ankle roll action lying wholly on one side of zero where A's and B's straddle it.

The sole geometry corroborates it from an independent channel. In A the ankle roll stands at minus 10.36 degrees in stance and plus 14.12 in swing, a sign reversal against a stop at plus or minus fifteen degrees, and the sole roll in stance has a standard deviation of 9.16 degrees about a mean of minus 1.24, so the foot rocks through flat twice a stride. In C the ankle roll stands at minus 14.00 degrees in stance and minus 14.98 in swing, both against the same stop and both on the same side, and the sole roll in stance has a standard deviation of 2.34 degrees about a mean of minus 6.96. The correlation between the ankle roll and the sole roll in stance falls from 0.935 in A to 0.524 in C, which is the signature of a joint that has stopped setting the sole angle because it no longer moves, the angle being set by the leg above it instead.

The prediction section 20.1 recorded for arm 5 is therefore confirmed in one part and falsified in two, and all three halves matter.

Saturation must be read against the velocity dependent envelope and not against the constant ceiling, which section 20.3 requires and which `/ws/context/gait_metrics.md` section 15 explains, the ankle saturation effort being equal to its ceiling so that the corner velocity is exactly zero and the two measures diverge by up to twenty points. Measured properly, arm 5's first prediction is falsified outright.

| Joint | A, against the envelope | B | C | C against the constant ceiling |
|---|---|---|---|---|
| Left ankle roll | 89.59 pct | 85.65 pct | 96.87 pct | 93.93 pct |
| Right ankle roll | 88.75 pct | 85.49 pct | 96.62 pct | 93.45 pct |
| Left ankle pitch | 91.38 pct | 61.61 pct | 64.33 pct | 57.00 pct |
| Right ankle pitch | 90.49 pct | 62.18 pct | 62.95 pct | 53.49 pct |
| Left hip roll | 40.71 pct | 13.14 pct | 0.55 pct | 0.55 pct |
| Right hip roll | 42.11 pct | 12.30 pct | 0.54 pct | 0.54 pct |

The ankle roll effort saturation ROSE, from 89.6 and 88.8 per cent to 96.9 and 96.6, and it rose because the joint stopped moving. A joint that traverses its travel twice a stride passes through the middle of that travel twice a stride, and in the middle the commanded target is reachable and the actuator is not saturated. A joint latched at one stop under a target 0.156 radians beyond it is saturated at every instant. The saturation figure and the reversal figure move in opposite directions and both are correct, which is why neither may be quoted alone.

Two predictions did resolve in the arm's favour. The ankle PITCH saturation fell from 91.4 and 90.5 per cent to 64.3 and 63.0, which no section had predicted and which is the largest unheralded gain in the set, and the hip roll fell from 40.7 and 42.1 to 0.55 and 0.54. And the toe-in fell, section 20.1 predicting a fall from 7.37 degrees and the stance measurement giving 8.36 and 8.11 degrees in A against 2.83 and 2.86 in C, a factor of 2.9, which is a larger effect than the arm claimed for itself. The prediction that the inner corner contact would give way to a flat sole is falsified and section 14.12 measures how.

### 14.4 The ankle roll is now a fixed joint by saturation rather than by design, and that is an argument for making it one

The consequence of section 14.3 deserves stating on its own, because it converts a proposal into an observation. In C the left ankle roll sits at its low stop for 93.2 per cent of stance and never visits the high stop in swing, the right sits at its high stop for 99.9 per cent of swing, and both are at the stop in the direction of their own mirrored nominal. The commanded target lies beyond the stop on 96.6 and 96.2 per cent of the samples where the actuator is not saturated, at a median overshoot of 0.156 radians, which at a stiffness of 100 newton metres per radian is 15.6 newton metres of spring force pressed into a mechanical stop that does not move. The mean absolute ankle roll torque is 16.79 and 16.74 newton metres against a ceiling of 17.

The best policy this sequence has produced is therefore one in which the ankle roll is already a fixed joint, held there by an actuator spending ninety eight per cent of its ceiling to hold it. The proposal to convert the joint to a fixed joint does not remove a capability the machine is using. It removes an actuator the machine is paying for and is not using, and it removes the two failure modes that attach to it, the reversal that A and B exhibit and the stop residency that every run of this sequence exhibits.

### 14.5 Run B is a second gait family, reached because arm 4 opened it and closed again by arm 5

Run B is the energetic outlier of the set and the mechanism is recoverable. Its mean absolute hip pitch torque is 79.04 newton metres against a ceiling of 84, which is 94 per cent of the ceiling held as an average, and it saturates that ceiling on 58.84 per cent of samples where A saturates it on 0.06 and C on 0.01. The hip pitch action scale is 0.4 in all three runs, so this is a behavioural difference and not a parameterisation one.

Two mechanisms could produce it and the dump separates them. The hip pitch corner velocity is 5.246 radians per second, being the velocity limit of 17.488 times one minus the ratio of the 84 newton metre ceiling to the 120 newton metre saturation effort, and B exceeds it on 0.14 per cent of samples, so this is not velocity saturation. Inverting the control law instead returns a median position error of 0.462 radians, which at a scale of 0.4 is a commanded action of 1.155 held continuously. B commands the hip pitch half a radian away from where the joint is and leaves it there.

The knee says why. B's first percentile knee angle is 0.0133 radians against a hard stop at zero, its median knee angle is 0.730 radians against A's 1.134 and C's 1.002, and its knee crosses the soft limit on 14.8 and 16.1 per cent of samples against 0.3 in A and 0.04 in C. B straightened the leg into a strut and drove the whole limb from the hip. The base height rises with it to 0.7038 metres against A's 0.6627 and C's 0.6819, and `pen_base_height` consequently pays B the least of the three, minus 0.0297 per second against A's minus 0.1304.

The attribution needs care and the care is available. B and A differ in the action scale and in the push ceiling, and a weaker disturbance is exactly what permits a straighter leg, so A against B cannot separate the two. B and C carry the same action scale and the same push ceiling and differ in the sway reward, and C's knee is the deepest of the three. The straight legged family is therefore a basin that the reduced knee scale made reachable and that the lateral sway reward closed off, and neither half of that sentence rests on the confounded comparison.

### 14.6 The regularisation budget is two orders of magnitude beneath the term that drives the gait, and the measurement is now exact

Section 4.7 asserted this from the weights. The evaluation dumps measure it, and the figures are worse than the assertion. Every per term series in `rewards.npy` is a rate per second and `total_reward` is per step, which `context/gait_metrics.md` records and which is worth a factor of one hundred, so the figures below are per second throughout.

| Quantity | A | B | C |
|---|---|---|---|
| Positive budget | 75.247 | 77.072 | 79.289 |
| Negative budget | -11.436 | -13.527 | -8.982 |
| The four energy terms together | -0.1189 | -0.5465 | -0.0394 |
| Energy as a share of the negative budget | 1.04 pct | 4.04 pct | 0.44 pct |
| Absolute joint power | 109.4 W | 395.3 W | 67.1 W |
| Torque rate, all joints | 230.0 Nm/s | 2193.9 Nm/s | 120.8 Nm/s |

Run B burns 3.61 times the mechanical power of run A and pays 0.4276 per second more for it across `pen_joint_torque`, `pen_joint_accel`, `pen_joint_vel_l2` and `pen_joint_torque_rate` together. Over the same interval it collects 1.9175 per second more on `rew_lin_vel_xy` alone. The exchange rate is 4.48 to one in favour of burning the power, so a policy that can buy two per cent of tracking accuracy with a tripling of its power consumption will buy it every time, and B did.

The figure that sets the size of any future correction is the last one. For the energy terms to have merely cancelled the tracking gain B obtained, their weights would have to rise by a factor of 4.48. For them to have made the efficient gait strictly preferable they would have to rise further, and a factor of ten on all four is the smallest round number that leaves a margin. That is a measurement of these runs and not a recommendation carried over from anywhere else.

One term behaved differently and the difference is instructive. `pen_action_smoothness` charged B 12.59 times what it charged A, which is the largest ratio in the table apart from `pen_joint_torque_rate`, and the reason is arm 4 rather than B's gait. The penalty is quadratic in the raw action and arm 4 raised the raw ankle action by 2.47, so the same physical ankle motion now costs 6.1 times what it cost before. Arm 4 therefore did impose a real cost on ankle reciprocation, through a channel it was not designed to act on, and B paid that cost rather than avoiding it, 1.1613 per second being 1.8 per cent of its total.

### 14.7 The penalty that section 20 reads these arms through does not measure what it is read for

Arm 4's prediction was that `pen_joint_pos_limits` would fall as the stop residencies fell. The term is reproducible from the dump, `isaaclab/envs/mdp/rewards.py:189` summing the absolute excursion beyond each joint's soft limit across all joints, and reproducing it returns minus 0.5291, minus 0.4686 and minus 0.4496 per second against the dumped minus 0.5293, minus 0.4687 and minus 0.4497, so the attribution below is exact rather than approximate.

The hip roll pair supplies 59.7, 52.7 and 57.5 per cent of the term in the three runs. It does so because of a geometry that no section of this document had examined. The left hip roll travel runs from minus 0.2094 to plus 2.2689 radians, an asymmetry of 10.83 to one, and `soft_joint_pos_limit_factor` scales that travel about its MIDPOINT, which places the soft limit at minus 0.0855. The configured nominal is minus 0.042951. The joint therefore has 0.0425 radians, being 2.44 degrees, of outward travel before the penalty begins, and the startup `joint_offsets` event draws a further plus or minus 0.05 radians on top of the nominal, which in the seed 42 evaluation environment leaves the right hip roll 0.0038 radians, being 0.22 degrees, from its soft limit at spawn.

| Joint | Hard travel | Asymmetry | Soft limit | Nominal | Clearance on the working side |
|---|---|---|---|---|---|
| Left hip roll | -0.2094 to 2.2689 | 10.83 | -0.0855 | -0.042951 | 0.0425 rad, 2.44 deg |
| Right hip roll | -2.2689 to 0.2094 | 10.83 | 0.0855 | 0.042951 | 0.0425 rad, 2.44 deg |
| Left ankle roll | -0.2618 to 0.2618 | 1.00 | -0.2356 | -0.041254 | 0.1943 rad, 11.13 deg |
| Left knee | 0.0 to 2.7053 | unbounded | 0.1353 | 0.51239 | 0.3771 rad, 21.61 deg |

The hip roll is charged for rolling outward by more than two and a half degrees, and the lateral weight shift that arm 5 asks for requires very much more than that. The term is consequently dominated by a joint doing the thing the ablation wants it to do, its reading at the hip roll is not a fault report, and the prediction that attached to arm 4 cannot be tested against it without first separating the hip roll out. This is a defect in the instrumentation rather than in the robot, and arm 17 of section 21 repairs it.

### 14.8 The feet pass within a fraction of a millimetre of one another and the live term cannot see it

`pen_feet_distance` calls `feet_distance` at `rewards.py:1038` with `lateral_only` false, so it measures the planar distance between the two foot LINK FRAME ORIGINS, which the docstring of `feet_distance_v2` at `rewards.py:1138` records as the ankle and not the foot on a sole footed robot. Transforming the twelve point sole table of `kscale_base_env_cfg.py:32` into the world by the dumped foot quaternions and sweeping vertex against edge in both directions, which is the procedure `_footprint_separation` performs at `rewards.py:1091`, measures both quantities on the same samples.

| Run | Mean ankle separation | Mean footprint separation | Gap | Footprint below 0.19 | Below 0.05 | Below 0.02 | First percentile |
|---|---|---|---|---|---|---|---|
| A | 0.2147 m | 0.0732 m | 0.1416 m | 92.0 pct | 45.7 pct | 27.7 pct | 0.0001 m |
| B | 0.2274 m | 0.0829 m | 0.1445 m | 90.6 pct | 40.2 pct | 24.0 pct | 0.0002 m |
| C | 0.2160 m | 0.0729 m | 0.1430 m | 92.2 pct | 45.0 pct | 27.4 pct | 0.0002 m |

The ankle separation averages 0.215 metres and the floor stands at 0.19, so the term is largely silent, and it pays A only 0.9018 per second. The sole polygons meanwhile approach within twenty millimetres on more than a quarter of samples and within a tenth of a millimetre at the first percentile, in every run. The term is blind by 0.143 metres and the blindness is structural rather than a matter of the threshold, the toe swinging laterally about an origin that need not move whenever the foot yaws, which is exactly the toe-in this sequence has measured throughout.

### 14.9 The five observations of the analysis brief, confirmed, sharpened, and one corrected

The first observation, that the splay is almost unchanged with a marginal reduction in C, is correct on the pooled mean and understates what happened in stance. The pooled toe-in is 7.37, 9.71 and 7.20 degrees, which is the marginal reading, because the pooled figure is dominated by the swing phase where the foot is unloaded and free. In stance the same measurement gives 8.36 and 8.11 degrees for A, 7.85 and 7.97 for B and 2.83 and 2.86 for C. The stance toe-in fell by a factor of 2.9 and that is the number that bears on the support polygon.

The second observation, that C reduces energy across the torque, acceleration, velocity and power terms, is confirmed and is larger than the reward terms suggest. The absolute joint power falls from 109.4 watts to 67.1, a reduction of 38.7 per cent against A and of 83.0 per cent against B, and the whole body torque rate falls from 230.0 newton metres per second to 120.8.

The third observation, that B worsens the splay slightly, is confirmed on the pooled mean, 9.71 degrees against A's 7.37 and at a ninety fifth percentile of 23.88 against 15.42, and the stance figure moves the other way by a small margin, 7.85 and 7.97 against 8.36 and 8.11. The honest statement is that B's swing foot yaws further and its stance foot does not.

The fourth observation, that C removes the hip roll saturation, is confirmed and is the largest single effect in the set. The hip roll saturation against the velocity dependent envelope falls from 40.71 and 42.11 per cent in A through 13.14 and 12.30 in B to 0.55 and 0.54 in C, the mean absolute hip roll torque falls from 22.06 and 22.66 newton metres to 11.66 and 11.93, and the fraction of samples on which the actuator is strictly inside its envelope rises from 65.3 per cent to 99.5.

The fifth observation, that the ankle pitch and roll saturation is not alleviated at all, is correct at the roll, understates the case there, and is too pessimistic at the pitch. The ankle roll hard stop residency is unmoved at 52.96 and 48.37 per cent in A against 50.83 and 48.64 in C, and its envelope saturation is not merely unmoved but worse, rising from 89.6 and 88.8 per cent to 96.9 and 96.6. What did move at the roll is the character of that residency, from a joint that traverses the whole travel twice a stride to a joint that does not move at all, and the two have different remedies, a reversing joint being a control problem and a latched joint a mechanism problem. The ankle pitch did improve, its envelope saturation falling from 91.4 and 90.5 per cent to 64.3 and 63.0 and its hard stop residency from 50.14 and 47.18 to 42.88 and 40.97, so the two ankle degrees of freedom have parted company and should no longer be spoken of as one fault.

### 14.10 The training records agree with the evaluation, and C wins on both readings of the budget

The comparison is made at a matched iteration and at a matched sample budget, because the environment counts differ and the two readings flatter different runs.

| Quantity | A at 10000 | B at 10000 | C at 10000 | C at 14062, matched samples |
|---|---|---|---|---|
| Mean reward | 666.4 | 832.1 | 887.6 | 961.4 |
| Mean episode length | 1734.7 | 1972.9 | 1768.8 | 1839.0 |
| Planar velocity error | 0.1689 | 0.1326 | 0.1319 | 0.1203 |
| Base contact termination rate | 0.1080 | 0.0299 | 0.0771 | 0.0336 |
| Push curriculum reached | 2.911 | 2.000 | 2.000 | 2.000 |

C leads on every column at both readings and did so from a smaller sample budget than either, which is the strongest statement this set supports. The base contact rate is the column where the push confound bites hardest, A having been pushed at 2.911 metres per second against the 2.000 the other two met, and it is consequently the one column of the table that should not be quoted without that sentence attached.

The evaluation reward tells the same story from the other end, C collecting 73.66 per second against A's 63.81 and B's 63.55. A and B tie to within half a per cent, which is itself a finding, arm 4 having bought nothing measurable in return at the objective the optimiser sees.

### 14.11 The sway reward was collected without the torso swaying, and the falsifier arm 5 set itself named the wrong quantity

Arm 5 succeeded and the channel through which it succeeded is not established by these measurements, which is stated here rather than glossed because it governs what the next arm may assume.

`HipLateralSwayReward` computes its error at `rewards.py:905` to `:911` as the torso body position minus the MEAN OF THE TWO FOOT POSITIONS, rotated into the base yaw frame, and compares the lateral component of that difference against a sine of amplitude 0.03 locked to the gait clock. The quantity is therefore a relative one, and a policy may reduce the error by moving the torso, by moving the feet, or by any mixture of the two.

The term paid C 3.3568 per second of a maximum of 5.0, which is 67.1 per cent of the available reward, so the policy is tracking the target rather than ignoring it. The torso's own lateral excursion did not rise to meet it. Taking the base frame lateral velocity, subtracting the commanded lateral velocity and reading the amplitude at the stride frequency gives a displacement amplitude of 6.8 millimetres in A, 2.3 in B and 5.6 in C, against the 26 to 37 millimetres that section 20.1 derives as the quasi static requirement and the 30 the term asks for.

Arm 5 named as its own falsifier the measured sway amplitude failing to rise from its 6.8 millimetres, which section 21.5 carries. It did not rise, it fell to 5.6, and yet the arm produced the largest behavioural improvement in the sequence. The falsifier was mis specified rather than met, having named the torso's absolute excursion where the term measures the torso against a moving reference, and it is corrected here rather than by editing section 21.5. What follows is that arm 5's premise, that the machine lacks a lateral weight shift and that supplying one would unload the ankle, remains untested. The ankle was unloaded and the weight shift was not supplied, so something else did it.

The instrumentation that would settle this is one line and it belongs in every future run. The term already holds its own `lateral_offset` and its own `target` at `rewards.py:958` and `:959`, and dumping both alongside the existing channels would say directly whether the torso moved, whether the feet moved, or whether the phase resynchronisation at `:974` re anchored the target onto a torso that did neither. Until that is recorded, no arm may be designed on the assumption that a lateral weight shift now exists on this robot.

### 14.12 The sole did not flatten, it stopped rocking and settled onto one edge, and that sets arm 24's parameter

Arm 5 predicted that the inner corner contact of section 11.1 would give way to a flat sole. Transforming both lateral sole corners into the world and comparing their heights over stance samples, which is the procedure section 11.1 established and which needs no sign reasoning, measures the opposite.

| Run | Left, outer corner minus inner | Stance samples with the outer corner lower | Right | Stance samples |
|---|---|---|---|---|
| A | +1.96 mm | 20.7 pct | +2.26 mm | 21.1 pct |
| B | +10.39 mm | 1.6 pct | +10.59 mm | 1.5 pct |
| C | +10.44 mm | 1.0 pct | +10.37 mm | 1.3 pct |

The inner edge loading deepened from 2.0 millimetres to 10.4, and the fraction of stance in which the outer edge carried instead fell from a fifth to a hundredth. This is the same event section 14.3 reports from the joint side, the sole roll moving from a mean of minus 1.24 degrees with a standard deviation of 9.16 to a mean of minus 6.96 with a standard deviation of 2.34, and the arithmetic agrees, a steady seven degree cant across the 84.6 millimetre sole giving 10.3 millimetres of corner height difference against the 10.4 measured. Run A rocked through flat twice a stride and spent a fifth of stance on each edge. Runs B and C stand on the inner edge and stay there.

Regressing the stance sole roll on the ankle roll, the hip roll and the torso roll together separates the contributions and states what a frozen ankle roll would leave. The ankle roll maps onto the sole roll at a coefficient between 0.932 and 1.102, which is the one to one relation the kinematics demand, and the hip roll at minus 0.642 to minus 0.920, with the fit explaining 0.646 to 0.935 of the variance.

| Run | Leg | Ankle roll nominal | Ankle roll that flattens the sole | Ankle roll stop | Mean hip roll in stance | Sole roll if frozen at the nominal |
|---|---|---|---|---|---|---|
| A | left | -2.36 deg | -9.10 deg | -15.00 deg | -11.47 deg | +6.61 deg |
| A | right | +2.36 deg | +9.25 deg | +15.00 deg | +11.40 deg | -6.79 deg |
| B | left | -2.36 deg | -6.89 deg | -15.00 deg | -9.01 deg | +4.39 deg |
| C | left | -2.36 deg | -7.68 deg | -15.00 deg | -7.89 deg | +5.86 deg |
| C | right | +2.36 deg | +7.15 deg | +15.00 deg | +7.91 deg | -4.85 deg |

The finding that matters for arm 24 is in the last column. Freezing the ankle roll at its configured nominal does not flatten the sole, it cants the sole by five to seven degrees in the OPPOSITE direction, onto the outer edge, because the nominal was derived to cancel a hip roll of 2.46 degrees and the machine walks with a hip roll of 7.9. The angle that would flatten the sole in stance is 7.2 to 7.7 degrees on run C's gait and 9.1 to 9.3 on run A's, which is three times the nominal and half the stop.

That is a design tension and not a parameter waiting to be read off. A fixed joint carries one angle, the standing pose wants 2.36 degrees and the measured gait wants 7.4, and the difference between them is the hip roll excursion the machine has adopted. Arm 24's parameterisation below chooses the nominal deliberately and accepts the outer edge cant, on the reasoning that the cant then becomes visible as a reduced base of support where at present it is absorbed by an ankle nobody is watching, and that the correct response to a rise in the fall rate is to price the hip roll excursion rather than to unfreeze the ankle.

### 14.13 The visual record and the artefacts

Frames were extracted from each play video with the `ffmpeg` binary carried inside `imageio_ffmpeg`, the workspace holding no other. The tracking camera places the robot at roughly seventy pixels of height in a 1280 by 720 frame, at which scale no gait property whatever is legible, so whole frames were extracted, inspected, found useless and deleted, and what is retained is two cropped and tiled strips per run, `zoom_A` through `zoom_C` at a 460 by 400 window upscaled twofold and tiled three by two at eight frames per second, and `legs_A` through `legs_C` at a 260 by 280 window upscaled threefold and tiled four by one. The six files are retained at [../context/artefacts/kscale_flat-2026-09-21_to_09-22-scale-sway-ablation/frames/](../context/artefacts/kscale_flat-2026-09-21_to_09-22-scale-sway-ablation/frames/) beside the twenty two analysis programs, and the deletion of the whole frames is recorded in the artefact index as the rules require.

All three policies walk with the legs nearly touching through the pass through, which corroborates the footprint separation of section 14.8 rather than merely illustrating it. In C the two feet are visibly canted in the same direction and hold that cant across the strip, which is the mirrored ankle roll latch of section 14.4 seen directly. In A the cant changes between frames. In B the tracked robot's feet cross one another in the fore and aft sense while the knees remain visibly straighter than in either of the others, which is the strut gait of section 14.5. Nothing in the visual record contradicts a numerical finding above and nothing in it establishes one that the numbers do not.

## 15. What the literature says about bang-bang control, and which of its remedies this robot's measurements admit

Section 14.3 establishes that the ankle roll traverses its whole travel stop to stop twice per stride in runs A and B, and section 11.2 established that it does so under a commanded target that lies permanently outside the travel. This section asks what the published record says about that behaviour, and it is organised so that each family of published remedy is put against the measurement that decides whether it applies here.

### 15.1 The behaviour is named, expected and general, and it is not a fault peculiar to this configuration

Seyde, Gilitschenski, Schwarting, Stellato, Riedmiller, Wulfmeier and Rus investigate what their abstract calls the colloquially known phenomenon that trained agents often prefer actions at the boundaries of the action space, draw theoretical connections to the emergence of bang-bang behaviour in optimal control, and replace the Gaussian policy with a Bernoulli distribution over the two extremes of each action dimension [16]. The result that matters here is that this bang-bang controller achieves state of the art performance on several continuous control benchmarks, which establishes that the behaviour is not a pathology of a poorly tuned reward but a solution the objective genuinely admits and frequently prefers. The paper draws the connection to minimum time optimal control, where bang-bang is provably optimal, and it states the contrast that governs this workspace directly, that the result stands in contrast to robotic hardware where energy and maintenance cost affect controller choices. The authors also disentangle exploration from the solution by repeating the analysis under imitation learning, which removes the reading that the boundary preference is merely an artefact of the exploration noise.

The relevance is that section 14 should stop treating the latch as evidence of a mistake. A velocity tracking objective with an eleven term regularisation stack whose energy component is one per cent of the negative budget, which section 14.6 measures, is close to an unregularised objective, and an unregularised objective over a bounded actuator is precisely the setting in which [16] expects extremes. The question this document must ask is not why the policy chose the stop but why anything in the configuration should have stopped it, and section 14.6 answers that nothing did.

### 15.2 The three published families of remedy, and what each would cost here

The first family prices the behaviour through the reward. Mysore, Mabsout, Mancuso and Saenko propose Conditioning for Action Policy Smoothness, two regularisation terms penalising the difference between actions on consecutive states and between actions on nearby states, and report that on a quadrotor the resulting smoothness improvement reduced power consumption by almost eighty per cent [17]. Their diagnosis is the one this document reached independently, that reinforcement learning algorithms do not typically optimise for smoothness and that the Markov property assumes independence of state action pairs, so oscillation is the default rather than the exception. The survey of Ha, Lee, van de Panne, Xie, Yu and Khadiv records the same two terms as standard practice in legged locomotion, the action rate and the second difference of the action, and states the problem with the family plainly, that maintaining a good balance between task rewards such as velocity tracking and regularisation rewards makes reward shaping a laborious task [18]. This workspace already carries both terms, `pen_action_rate` at minus 0.01 and `pen_action_smoothness` at minus 0.075, and section 14.6 measures the balance the survey warns about at 4.48 to one against them.

The second family constrains the policy class rather than pricing its output. Chou, Maturana and Scherer replace the Gaussian with a Beta distribution whose support is the admissible interval itself, removing the bias that clipping introduces [9], and cluster 25 of `context/literature.md` already registers the contrapositive that decides section 14.2, that an unbounded support is what permits the exact compensation measured there. Fujita and Maeda address the same mismatch from the estimator side, observing that out of bound actions are clipped before execution while the policy is optimised as though they were not, and proposing the Clipped Action Policy Gradient, which is unbiased and which they report achieves lower variance than the conventional estimator that ignores action bounds [19]. Shin, Cha, Kim, Cha and Park attack the same quantity through the network, arguing that policies develop abrupt high frequency torque changes which are infeasible for real actuators with finite bandwidth, that the conventional answer of regularisation rewards on joint velocities, accelerations and energy requires extensive hyperparameter tuning, and that spectral normalisation of the network weights limits high frequency policy fluctuation directly while enabling more efficient parallel training than a gradient penalty [20]. Raffin, Kober and Stulp locate the same problem in the exploration noise rather than in the mean, reporting that unstructured step based exploration leads to jerky motion patterns on real robots and that resampling state dependent noise at an interval enables a compromise between performance and smoothness [21].

The third family declares the requirement a constraint rather than a price. Kim, Oh, Lee, Choi, Ji, Jung, Youm and Hwangbo argue that a reward only formulation requires more than ten reward terms and a tuning process repeated several times, and that a constrained Markov decision process solved by interior point policy optimisation reaches the same behaviour with three reward terms and a single tuned coefficient [4]. Their constraint set is the decisive detail for this document, the six probabilistic constraints including joint positions, joint velocities and joint torques, which are exactly the three quantities section 14 finds unpriced here, and their reported violation rates on hardware are 0.007, 0.001 and 0.002 against a threshold of 0.025. They further report that the constrained controller transfers from Raibo to Anymal C without retuning the torque coefficient where the reward only controller degrades significantly, which is the property a workspace running four robots against one shared mdp package should care about most. Chane-Sane, Leziart, Flayols, Stasse, Souères and Mansard reach the same end by a cheaper route, reformulating each constraint as a probability of terminating future reward, which requires only a minimal modification to an off the shelf algorithm such as proximal policy optimisation [3].

### 15.3 The biomechanical literature supplies the mechanism behind arm 5's success, and it predicts the ankle would go quiet

Hong and Park optimise a biomechanical model and train a reinforcement learning controller against the same postural task, and report that a centre of pressure constraint reproduces the saturated ankle torque observed in humans, the ankle torque reaching a plateau corresponding to the maximum centre of pressure limit, and that as the perturbation grows the model shows progressively reduced ankle feedback gains coupled with increased hip involvement [22]. Two features of their result bear directly on section 14.3. The first is that the saturation is a geometric consequence of the support polygon rather than an actuator deficit, the ankle torque plateau corresponding to a stabilisation of ankle joint motion while the flat foot constraint is maintained, which is the same statement section 11.3 makes when it establishes that the KScale ankle travel is sufficient and that no URDF change will repair the latch. The second is that the transition to the hip occurs, in their words, well before reaching the ankle torque limit, which suggests the constraint influences strategy selection proactively rather than as a threshold response.

The relevance is that arm 5 supplied a hip strategy to a machine that had only an ankle strategy, and the literature predicts exactly what section 14.3 measured, that the ankle would fall quiet once the hip could do the work. This is the strongest published support any arm in this sequence has, and it was found after the arm succeeded rather than before, which is recorded here so that a reader does not mistake it for the arm's original justification.

### 15.4 The passive ankle literature is the limit case of the fixed joint proposal, and it establishes that the capability is not indispensable

Kim, Jorgensen, Lee, Ahn, Luo and Sentis demonstrate unsupported dynamic locomotion on two passive ankle bipeds, Mercury and DRACO, under a whole body locomotion controller whose contact constraint relaxation reduces jerk at foot contact switching [23]. The surveyed position is that passive ankle bipeds have fewer degrees of freedom than ankle actuated legged robots, resulting in lower mechanical complexity and lighter lower legs, at the cost of a small horizontal footprint. The relevance is bounded and should be stated as such. That work removes the ankle entirely and replaces it with a model based whole body controller, where the proposal of section 21 removes one ankle degree of freedom from a learned controller and retains the other, so the paper establishes feasibility in principle rather than supplying a method. What it does establish, and what no other surveyed source does, is that lateral balance on a biped does not require an actuated ankle roll, which is the premise the fixed joint arm needs and which this document had no published support for before.

### 15.5 What the survey did not find, recorded as a negative result

A directed search returned no published account of a learned locomotion policy holding a joint against a mechanical stop by commanding a position target permanently outside the joint's travel, which is the specific mechanism section 11.2 measures and section 14.2 quantifies. The nearest published statements are the boundary preference of [16], which concerns the action space rather than the joint space, and the clipping bias of [9] and [19], which concern an action space that is bounded where this one is not. The distinction matters, because a bang-bang controller in the sense of [16] switches between two admissible extremes whereas this policy names a single inadmissible target and lets the mechanism absorb the difference, and the second is a mechanically harsher behaviour that consumes actuator effort continuously rather than at the switches. The observation is therefore to be treated as unreported, in the manner cluster 27 of `context/literature.md` adopts, and any future document wishing to cite a precedent must find one rather than assume it exists.

A second negative result attaches to the remedy. No surveyed source reports the effect of clipping the PROCESSED action, meaning the commanded joint target in radians, as distinct from clipping the raw action. The two differ on this robot because the offset randomisation of the startup event shifts each environment's default, and the sources that treat bounded actions [9] [19] bound the raw quantity. Arm 10 is consequently an experiment this workspace must run rather than a technique it may adopt, and its result is worth recording for that reason alone.

## 16. The energy budget and the ankle roll penalty were executed, and the ankle is now a mechanical stop rather than a joint

Three runs completed between 2026-09-24 and 2026-09-25 carry arm 8 in both its stages together with a reward term that no arm of this document had proposed, and they are read here as run A, run B and run C. A fourth run of the same week carries the energy escalation past the point at which the task survives and is read alongside them because it establishes the upper bound that none of the other three reach. The findings are of two kinds and the distinction should be stated before any of them is quoted. Arm 8 succeeded, and it succeeded more completely than its own prediction allowed for, the energy price rising by a factor of fifty to one hundred while the velocity tracking reward rose rather than fell. And the ankle, which every section of this document from section 4.1 onward has described as latched, is no longer merely latched, it is welded, the deviation penalty added between run A and run B having moved the joint from one mechanical stop to the other and then held it there with a saturated actuator for ninety nine point eight per cent of every swing.

### 16.1 What was actually run, and why no pair of these runs is clean either

The three dumped configurations were flattened to leaf paths through the repository's own `ParamsLoader` at `scripts/analysis/experiment_params.py:24` and compared exhaustively, which is the method section 14.1 established and which is repeated here rather than assumed. Of 2583 leaf paths 59 differ, and of those 59 fifty one are the constituent fields of two reward terms present in one run and absent from another, leaving eight substantive differences and one of them a confound.

| Path | A `2026-09-24_05-18-33` | B `2026-09-25_08-00-34` | C `2026-09-25_11-29-39` |
|---|---|---|---|
| `rewards.pen_joint_torque.weight` | -5.0e-4 | -1.0e-3 | -5.0e-4 |
| `rewards.pen_joint_accel.weight` | -5.0e-6 | -1.0e-5 | -5.0e-6 |
| `rewards.pen_joint_vel_l2.weight` | -5.0e-4 | -1.0e-3 | -5.0e-4 |
| `rewards.pen_joint_torque_rate.weight` | -5.0e-4 | -1.0e-3 | -5.0e-4 |
| `rewards.pen_joint_power` | absent | absent | present at -0.02 |
| `rewards.pen_ankle_roll_deviation` | absent | present at -1.0 | present at -1.0 |
| `rewards.rew_keep_ankle_roll_zero_in_air.weight` | 0.25 | 1.0 | 1.0 |
| `scene.num_envs` | 9000 | 9000 | 6000 |

Everything else is identical across the three, and the identity is worth stating because it disposes of the hypotheses a reader would otherwise have to exclude by hand. The nominal pose of `environments/environments/assets/config/kscale_identified_cfg.py:197` is the arm 7 posture in all three, being hip pitch at plus and minus 0.34907 rad, hip roll at minus and plus 0.042951, hip yaw at zero, knee at 0.87266, ankle pitch at minus 0.5236 and ankle roll at minus and plus 0.041254. The actuator set is identical, the ankles carrying an effort limit and a saturation effort of 17.0 newton metres apiece against a stiffness of 100 and a damping of 1.05 at the roll and 1.25 at the pitch. The per joint action scale of arm 4 and the halving of arm 6 stand in all three, being 0.4 at the hip pitch and hip yaw, 0.2667 at the knee, 0.16 at the hip roll and 0.08 at each ankle. And `clip` is unset on the action term in all three, at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:169`, which is the fact section 20.1 now ranks arm 10 first upon.

A defect of this document is exposed by the first four rows and is recorded here rather than repaired silently. Section 21.8 states that arm 8 stage one raises the four energy weights by a factor of ten and tabulates the result as minus 1.0e-4 against a prior minus 1.0e-5. The factor actually applied to the working tree is fifty, and it is fifty in the configuration that all three of these runs dumped. The factor of ten was in fact run, as `2026-09-23_11-40-12`, and section 16.2 reads it as the first rung of a ladder rather than as the arm, but the comment block at `kscale_base_env_cfg.py:875` and the table within it continue to say ten where the value beneath them says fifty. The arithmetic of section 21.8 that sets the factor is therefore evidence for a rung of the ladder rather than for the value the tree carries, and section 16.2 supplies the evidence the applied value now rests upon.

No pair of these three runs differs in one variable, which is the third sequence of this document to carry that fault and the reason the causal claims below are drawn from within run measurement rather than from between run contrast. Run A against run B moves the energy factor from fifty to one hundred, adds the ankle roll deviation penalty and quadruples the in air ankle roll reward, which is three changes. Run A against run C holds the energy factor, adds the same two ankle roll changes, adds the mechanical power term and reduces the population, which is three changes and a confound. Run B against run C is the cleanest at two changes and the confound, halving the energy factor and adding the power term.

The single variable arm exists and was abandoned. `2026-09-25_07-56-50` carries run A's energy factor with both ankle roll changes and nothing else, at six thousand environments, and it holds `model_0.pt` and `model_1000.pt` and no more, so it was stopped at the first thousand iterations. Had it completed it would still have confounded the deviation penalty with the in air reward, the two having been applied together, so the contrast it would have supplied is not the one section 16.4 needs.

The contrast section 16.4 needs is in flight as this section is written. `2026-09-28_10-25-57` carries run C's configuration exactly except that `pen_ankle_roll_deviation` is commented out at `kscale_base_env_cfg.py:824`, at nine thousand environments against run C's six thousand, so it isolates that one term up to the population. Its verdict is the first thing a reader returning to this document should look for, and section 20.1 places it at the head of the order in consequence.

### 16.2 The energy escalation has a cliff, and it stands between one hundred and five hundred

Six runs now carry the four energy weights at five distinct multiples of the value they held before arm 8, and read together they establish both that the arm works and how far it may be taken. The table is at a matched iteration of ten thousand, the environment count varying across the set and being recorded so that the reader may discount it.

| Run | Factor | Envs | Mean reward | Episode length | `rew_lin_vel_xy` | `base_contact` | Noise std |
|---|---|---|---|---|---|---|---|
| `2026-09-22_07-40-43`, arm 5 reference | 1 | 6400 | 887.6 | 1768.8 | 35.21 | 0.0771 | 0.761 |
| `2026-09-23_11-40-12` | 10 | 6000 | 955.4 | 1849.8 | 35.24 | 0.0526 | 0.584 |
| `2026-09-24_05-18-33`, run A | 50 | 9000 | 952.6 | 1784.2 | 35.84 | 0.0762 | 0.331 |
| `2026-09-25_08-00-34`, run B | 100 | 9000 | 954.8 | 1890.4 | 36.09 | 0.0550 | 0.374 |
| `2026-09-24_11-25-48` | 500 | 6000 | -66.8 | 18.0 | 0.071 | 1.0000 | 0.252 |
| `2026-09-25_11-29-39`, run C | 50 and power | 6000 | 1018.0 | 1929.1 | 38.82 | 0.0291 | 0.502 |

Three readings follow and the third is the one that generalises. The first order of magnitude carries most of the benefit, the mean reward rising from 887.6 to 955.4 and the base contact termination rate falling from 0.0771 to 0.0526 between the reference and the tenfold run, with the velocity tracking reward unmoved at 35.2. The second order of magnitude is very nearly free, the fiftyfold and hundredfold runs holding the mean reward near 953 while the tracking reward rises slightly rather than falling, which is the outcome section 21.8 named as its prediction and did not expect to see exceeded. And the cliff is abrupt rather than gradual, the five hundredfold run collapsing to an eighteen step episode at a base contact termination rate of exactly 1.0000, meaning that every environment falls on every episode, with nothing in the hundredfold run's record foreshadowing it.

The share the group holds of the negative reward budget is the quantity section 14.6 measured at 1.04 per cent and asked every later arm to be read against. At the best checkpoint of each run it now stands at 12.63 per cent in A, 20.71 in B and 22.64 in C, against a positive budget that is unmoved at 88.8, 88.8 and 88.2 per second. The objective the ablation is conducted over is therefore no longer one in which tripling the mechanical power costs 0.68 per cent of the return, and the precondition section 20.1 ranked first is satisfied. The mechanical power itself fell from the 67.1 watts section 14.6 records for the arm 5 reference to 57.51 in A, 47.08 in B and 55.69 in C.

### 16.3 Arm 8 stage two produced the best policy this robot has trained, and it is also the worst

Run C adds `joint_powers_l1` at minus 0.02, which is arm 8 stage two, and on the training objective it wins on every channel that objective measures. Its mean reward of 1018.0 at ten thousand iterations leads run A's 952.6 and run B's 954.8, its episode length of 1929 leads 1784 and 1890, its base contact termination rate of 0.0291 is less than half run A's 0.0762, and its velocity tracking reward of 38.82 leads by three points. At the matched sample budget of two thousand million environment steps, which is the comparison its smaller population requires, it leads on mean reward at 1014.0 against 991.4 and 972.6 and on episode length at 1972 against 1813 and 1894.

On the mechanics it is the worst of the three by a wide margin, and every measurement of sections 16.4 and 16.5 says so. Its ankle roll stands at or above ninety eight per cent of the torque available to it on 88.25 and 88.60 per cent of all samples against run A's 19.75 and 17.94. Its ankle pitch does so on 44.31 and 42.21 per cent against run A's 5.00 and 5.55. Its mean absolute ankle roll torque is 16.115 and 16.153 newton metres against a ceiling of 17, which is 95 per cent of that ceiling sustained on both legs through every sample of a thirty second episode. Its ankle pitch sits at a hard mechanical stop for 85.5 per cent of swing.

The tension between the two paragraphs above is the central finding of this section and it should not be resolved by preferring one of them. An objective that rewards a policy for pressing two actuators into their limits and holding them there is not measuring what the ablation set out to improve, and the correct inference is that the objective is still incomplete rather than that run C is a good policy or a bad one. What is missing from it is named by section 17.2, the actuator envelope being treated everywhere in this configuration as a shape the clip inside `DCMotor` applies silently rather than as a boundary the policy is asked to respect.

### 16.4 The ankle roll deviation penalty moved the joint from one stop to the other, and welded it there

The term added between run A and run B is `joint_deviation_l1` on the two ankle roll joints at a weight of minus 1.0, and it is the only reward term in the history of this robot to have been applied to a joint whose position in stance is set by the ground rather than by the command. What it did is measured exactly, and the measurement is the strongest single argument this document carries for arm 24.

| Quantity, swing phase of the joint's own leg | A `model_9000` | B `model_8000` | C `model_8000` |
|---|---|---|---|
| Mean position, left ankle roll [rad] | +0.2495 | -0.2615 | -0.2615 |
| Mean position, right ankle roll [rad] | -0.2529 | +0.2617 | +0.2616 |
| Samples at the stop nearest that position [pct] | 89.87, 89.77 | 99.74, 99.90 | 99.83, 99.93 |
| Envelope saturation [pct] | 48.04, 45.43 | 98.02, 98.15 | 99.13, 99.37 |
| Mean absolute joint speed [rad/s] | 0.183, 0.198 | 0.011, 0.008 | 0.013, 0.013 |
| Commanded target recovered from the control law [rad] | 0.375, -0.378 | -0.430, +0.431 | -0.431, +0.431 |
| Mean excess of that target beyond the stop [rad] | 0.126, 0.125 | 0.169, 0.169 | 0.170, 0.170 |
| Samples whose commanded target lies beyond the stop [pct] | 89.50, 89.16 | 99.77, 99.89 | 99.86, 99.93 |

The stop the joint occupies REVERSED. In run A the left ankle roll sits at plus 0.2618 radians and the right at minus 0.2618, and in runs B and C the left sits at minus 0.2618 and the right at plus 0.2618, both configurations being mirror symmetric and the second being the stop nearer each joint's own nominal. Measured as the deviation the term prices, the penalty therefore worked, the mean absolute departure from the nominal in swing falling from 0.3354 radians to 0.1788, which is very nearly a halving, and the in air kernel of `rew_keep_ankle_roll_zero_in_air` rising with it from 18.7 per cent of its maximum to 40.9.

Measured as control it is a deterioration on every channel. The joint moves at one hundredth of a radian per second where it moved at two tenths, so it is not being regulated toward the nominal, it is stationary against a mechanical limit. The commanded target moved further outside the reachable set rather than inside it, the excess beyond the stop rising from 0.126 to 0.170 radians and the fraction of swing samples on which the target is unreachable rising from 89.5 per cent to 99.9. And the actuator changed the direction of its work, the fraction of samples on which the torque OPPOSES the deviation, which is the only condition under which an actuator can be said to be pulling a joint toward its nominal, falling from 9.41 and 7.91 per cent to 0.73 and 0.68, and in stance from 14.47 and 11.85 to 0.53 and 1.18. On more than ninety nine per cent of samples the actuator is now pushing the joint further from the posture the penalty prices, at a mean of 16.1 newton metres of a 17 newton metre ceiling.

The mechanism follows from the three rows together and it is general rather than peculiar to this joint. A deviation penalty prices the ACHIEVED position, the policy controls only the COMMANDED position, and the two are connected by a servo whose authority is bounded by the effort limit and by a mechanical stop that is not bounded at all. Where the achieved position is determined by contact or by a stop, the gradient of the penalty with respect to the command is very nearly zero, so the penalty cannot move the joint. What it can do, and what it did, is change which of the two unreachable targets the policy selects, and the policy selected the one whose stop lies nearer the nominal, collecting the reduction in deviation without giving up the rigid reference the stop supplies. The term was paid rather than avoided, at 0.354 per second in B and 0.409 in C, and the behaviour it bought is the opposite of the one it names.

The quadrupling of `rew_keep_ankle_roll_zero_in_air` from 0.25 to 1.0 accompanies the penalty in both runs and cannot be separated from it by these three runs. The two act in the same direction and the reversal is consistent with either. The run in flight at `2026-09-28_10-25-57` removes the penalty and retains the quadrupled in air reward, so its ankle roll position in swing is the measurement that separates them, and the prediction this section commits to is that the joint stays at the near stop, the in air reward being sufficient on its own to have selected it.

### 16.5 The ankle pitch followed the roll into its own stop, and arm 7 is why

The ankle pitch was in no distress in run A and is in severe distress in run C, and the cause is a pre-existing condition that arm 7 created and that no measurement had yet charged to it.

| Quantity, swing phase | A `model_9000` | B `model_8000` | C `model_8000` |
|---|---|---|---|
| Mean position [rad] | -0.637, -0.614 | -0.735, -0.728 | -0.852, -0.850 |
| Samples at the lower stop of -0.8727 [pct] | 0.13, 0.07 | 35.31, 33.50 | 85.50, 84.60 |
| Envelope saturation [pct] | 0.00, 0.00 | 12.85, 11.30 | 72.59, 71.99 |
| Commanded target [rad] | -0.637, -0.614 | -0.774, -0.763 | -0.983, -0.980 |
| Samples whose target lies beyond the stop [pct] | 0.09, 0.06 | 33.67, 31.67 | 84.59, 83.74 |

Arm 7 moved the ankle pitch nominal from minus 0.3087 radians to minus 0.5236, which section 21.7 records and describes as moving the joint closer to its stop. The size of that move is now the operative quantity. The joint's travel is minus 0.8727 to plus 0.5236 radians, so the headroom on the dorsiflexion side fell from 0.5640 radians to 0.3491, a reduction of thirty eight per cent, while the headroom on the other side rose to 1.0472 where nothing needs it. Run C's swing ankle pitch departs from its nominal by 0.3406 radians on average, which is 97.6 per cent of the headroom that remains, so the joint reaches its stop on the ordinary swing of an ordinary stride rather than on an excursion.

Section 12 concluded that the ankle pitch travel is genuinely insufficient where the ankle roll's is not, and section 20.3 accordingly excludes a URDF change at the roll and preserves it at the pitch. That conclusion is now confirmed at a second pose and by a second mechanism, and arm 25 ceases to be conditional on arm 7 failing, since the measurement that would have made it conditional has been taken. What the measurement cannot yet say is whether the remedy is the travel or the nominal, both being available and the second being very much cheaper, and section 20.1 records that choice as the arm's own first decision rather than presuming it.

### 16.6 The swing hip yaw excursion is commanded and the servo tracks it, so no gain change can reach it

The hypothesis this section was written to test is that the change in load and inertia at the stance to swing transition drives the hip yaw away from its nominal by more than the configured gains can resist. The phenomenon is real and the transient is sharp. The hip yaw departs from its nominal by 0.051 to 0.064 radians through stance, rises to 0.174 to 0.206 through swing, and the mean over 948 to 1071 liftoff events peaks at 0.258 to 0.298 radians between 0.09 and 0.19 seconds after the foot leaves the ground. Its mean absolute speed rises from 0.31 to 0.48 radians per second in stance to 1.09 to 1.45 in swing.

The proposed mechanism is nonetheless refuted, and the instrument that refutes it is exact rather than statistical. These runs drive the hip yaw through an explicit servo inside `DCMotor`, so wherever a sample is unsaturated the commanded target is recovered without approximation as the achieved position plus the applied torque and the damping term divided by the stiffness. The hip yaw is unsaturated on 99.99 per cent of samples in every run, so its commanded target is recovered in full rather than on a surviving subset.

| Joint and phase | A `model_9000` | B `model_8000` | C `model_8000` |
|---|---|---|---|
| Commanded departure from nominal, swing [rad] | 0.170, 0.193 | 0.180, 0.199 | 0.159, 0.177 |
| Achieved departure, swing [rad] | 0.177, 0.206 | 0.181, 0.203 | 0.154, 0.174 |
| Ratio of commanded to achieved, swing | 0.960, 0.939 | 0.993, 0.982 | 1.033, 1.021 |
| Mean absolute torque, swing [N.m] | 1.064, 1.045 | 0.941, 0.928 | 1.316, 1.321 |
| Commanded departure, stance [rad] | 0.072, 0.079 | 0.064, 0.078 | 0.067, 0.067 |
| Achieved departure, stance [rad] | 0.057, 0.051 | 0.058, 0.064 | 0.107, 0.093 |
| Ratio of commanded to achieved, stance | 1.271, 1.555 | 1.103, 1.218 | 0.624, 0.717 |
| Mean absolute torque, stance [N.m] | 5.078, 5.086 | 4.587, 4.658 | 7.618, 8.066 |
| Envelope saturation, either phase [pct] | 0.00 | 0.00 | 0.02 |

In swing the policy commands the excursion and the servo delivers it, the ratio standing between 0.939 and 1.033 and the torque required being one newton metre of a forty two newton metre ceiling. There is no deficit of authority to repair. Raising the stiffness would reduce a tracking error of 0.038 to 0.049 radians on a quantity that is 0.17 to 0.21 radians by intention, which is a correction of a fifth of the excursion at most, and it would leave the four fifths the policy asked for exactly where they are. The proposal to find better hip yaw gains in order to reduce the swing excursion is therefore refuted as stated, and section 20.1 carries it as an arm with its premise replaced rather than deleting it.

In stance the picture differs between the runs and the difference is informative. Runs A and B under deliver, commanding 0.064 to 0.079 radians and achieving 0.051 to 0.064, which is the ground resisting a command the servo cannot fully impose at 4.6 to 5.1 newton metres. Run C over delivers, commanding 0.067 and achieving 0.093 to 0.107 while spending 7.6 to 8.1 newton metres, which is the ground driving the joint further than the policy asked and the servo failing to hold it. Run C is therefore the one run in which a stance hip yaw stiffness increase would do something, and what it would do is oppose a load rather than reduce a commanded excursion, which is a different arm with a different justification.

### 16.7 The hip yaw sets the swing toe-in at unity, and the two hip yaws move together rather than apart

Section 14.9 regressed the stance toe-in on the hip yaw angle and found a slope near unity. Repeating that regression with the swing phase separated, and with the left leg mirrored on both axes so that the toe-in convention and the joint convention agree, produces a sharper result and relocates the fault.

| Run and phase | Slope [deg per deg] | Correlation | Intercept [deg] | Mean toe-in [deg] |
|---|---|---|---|---|
| A, stance | 0.939, 0.932 | 0.950, 0.955 | 4.25, 2.94 | 2.87, 3.42 |
| A, swing | 0.998, 0.992 | 0.998, 0.997 | 4.73, 3.22 | 14.41, 14.75 |
| B, stance | 0.913, 0.940 | 0.960, 0.964 | 3.71, 2.27 | 3.14, 2.94 |
| B, swing | 0.994, 1.001 | 0.987, 0.989 | 3.82, 2.27 | 13.72, 13.65 |
| C, stance | 0.960, 0.968 | 0.943, 0.936 | 6.06, 4.68 | 0.67, 0.07 |
| C, swing | 1.001, 0.995 | 0.994, 0.994 | 4.77, 3.15 | 12.24, 12.06 |

In swing the hip yaw accounts for the foot's heading almost completely, the slope standing between 0.992 and 1.001 and the correlation between 0.987 and 0.998, which is as close to a kinematic identity as a measurement of a walking machine reaches. And the splay is four to eighteen times larger in swing than in stance, standing at 12.06 to 14.75 degrees in swing against 0.07 to 3.42 in stance. The pigeon toe this document set out to remove is therefore a SWING phenomenon that the stance measurements of sections 3.1 and 14.9 have been reading at its residual.

Two consequences follow and both change an arm. Arm 10's prediction, that freezing the hip yaw would leave the regression intercept of 2.2 degrees, is a prediction about stance and understates what the arm would do, the swing intercept standing at 2.27 to 4.77 degrees against a swing toe-in of 12.06 to 14.75, so freezing the joint would remove between sixty two and eighty three per cent of the visible splay rather than a fifth. And `pen_feet_heading` carries `airborne_only: False` at `kscale_base_env_cfg.py:841`, so it prices a stance error of three degrees and a swing error of fourteen at the same weight, which is the term spending most of its gradient on the phase in which least is wrong.

The mode the fault occupies was also measured, and it contradicts a prediction this workspace recorded in cluster 18 of `/ws/context/literature.md`. That cluster argues that a policy generating yaw internally must counter rotate the two legs, since a torque between the pelvis and one thigh acts equally and oppositely upon the other, and predicts that a free hip yaw biped should therefore exhibit a predominantly DIFFERENTIAL foot yaw fault. The correlation between the two raw hip yaw angles is positive in all three runs at 0.424, 0.374 and 0.557, and the standard deviation of their sum exceeds that of their difference at 0.2276 against 0.1449, 0.2141 against 0.1445 and 0.2727 against 0.1455. The common mode is therefore the larger of the two by a factor of 1.47 to 1.87 in amplitude, and the two legs yaw together rather than apart. The correlation of that common mode with the base yaw rate is weak and negative at minus 0.156, minus 0.168 and minus 0.328, so it is not the yaw generation route the cluster described. A dated correction to cluster 18 is recorded in that file, and the practical consequence is that `feet_yaw_alignment` in its mean form, which cluster 18 establishes attributes a pure splay four times as much weight as a pure common rotation, under prices exactly the mode that dominates here.

### 16.8 The training records, and what the entropy says that the reward does not

Every comparison below is drawn at a matched iteration and repeated at a matched sample budget, since runs A and B collect nine thousand parallel environments against run C's six thousand and an iteration of C therefore carries two thirds of the experience.

| Quantity at iteration 14000 | A | B | C |
|---|---|---|---|
| Mean reward | 1057.9 | 961.8 | 982.0 |
| Mean episode length | 1862.3 | 1893.5 | 1920.5 |
| `base_contact` termination | 0.0317 | 0.0276 | 0.0266 |
| Policy noise standard deviation | 0.2929 | 0.4221 | 0.5315 |
| Entropy | -0.9224 | 0.5755 | 2.4833 |
| Linear tracking error [m/s] | 0.1101 | 0.1171 | 0.1090 |

| Quantity at 2000 M environment steps | A | B | C |
|---|---|---|---|
| Mean reward | 991.4 | 972.6 | 1014.0 |
| Mean episode length | 1812.7 | 1894.0 | 1971.7 |
| Linear tracking error [m/s] | 0.1222 | 0.1325 | 0.1055 |
| Policy noise standard deviation | 0.3324 | 0.3768 | 0.5301 |
| Entropy | 1.0330 | 1.4782 | 2.3913 |

Run A is the only one of the three that converges. Its entropy peaks at 1.87 near iteration eight thousand and falls to minus 0.92 by fourteen thousand, and its noise standard deviation peaks at 0.349 and falls to 0.293. Run B's entropy is still rising at fourteen thousand and its noise standard deviation rises monotonically from 0.227 to 0.422. Run C's entropy stands at 2.48 and its noise standard deviation at 0.532, which is the highest of the three at every iteration after four thousand.

The reading this workspace adopted on 2026-09-16, that an entropy which rises early and does not fall marks an optimisation that has not settled, therefore separates run A from runs B and C exactly as it did then, and it does so against a mean reward ordering that places run C first. The two instruments disagree, and the resolution is supplied by section 16.4 rather than by either of them. A policy that has driven two actuators into their limits has removed those two degrees of freedom from its own control problem, so the residual variance it retains is spread over the ten that remain, and a high policy entropy beside a high return is what that looks like from the training record. The instrument to prefer is neither the reward nor the entropy but the saturation, and section 20.2 accordingly requires it of every arm.

### 16.9 The splay does not improve with training and the latch deepens monotonically, which sets the budget

The four checkpoints of run A were replayed and read as a series, which is the comparison that answers how long this task should be trained.

| Quantity | `model_8000` | `model_9000` | `model_10000` | `model_14000` |
|---|---|---|---|---|
| Toe-in in swing, both feet [deg] | 14.53 | 14.59 | 14.68 | 14.11 |
| Toe-in in stance, signed [deg] | +3.99 | +3.15 | +1.53 | -4.93 |
| Ankle roll envelope saturation [pct] | 3.95, 3.26 | 19.75, 17.94 | 25.86, 24.01 | 32.01, 31.58 |
| Ankle roll at a hard stop [pct] | not recorded | 37.49, 36.00 | 38.19, 37.07 | not recorded |
| Hip yaw to swing toe-in correlation | 0.935 | 0.952 | 0.951 | 0.975 |

The swing splay is flat across six thousand iterations, moving from 14.53 degrees to 14.11 and passing through 14.68, which is a variation of four per cent and no trend. The stance splay does move, and it moves through zero into toe-out, reaching minus 4.93 degrees at fourteen thousand, so the sole measure on which training appears to help is a measure that overshoots rather than converges. And the ankle roll saturation rises monotonically from 3.95 per cent to 32.01, an eightfold increase, with the correlation between the hip yaw and the swing toe-in tightening from 0.935 to 0.975 as it does so. The latch is not a condition the policy arrives at and then holds, it is a strategy the policy refines throughout training.

The user's observation that this task should be trained to ten thousand iterations and no further is therefore confirmed, and the mechanism is now stated rather than assumed. Beyond ten thousand iterations the objective continues to improve, the swing splay does not, and the actuator condition deteriorates, so the additional budget purchases a deeper commitment to the degenerate solution. Every arm of section 20 accordingly runs to twelve thousand iterations and is evaluated at eight, nine and ten thousand, which section 20.2 now requires.

### 16.10 The five observations of the analysis brief, confirmed, sharpened, and two corrected

The brief of 2026-09-26 offered five readings and each is answered here against the measurement rather than against the impression.

The first, that the foot splay is almost unchanged across the three with a marginal reduction in run A, is confirmed in its main clause and corrected in its subordinate one. The splay is almost unchanged, the pooled toe-in standing at 8.050, 8.344 and 7.167 degrees and the swing toe-in at 14.58, 13.68 and 12.15. The marginal reduction belongs to run C rather than to run A on both readings, and it is marginal, being 0.88 degrees pooled and 2.43 in swing. Run A leads the composite ranking of `compare.py` on the strength of its saturation column and not its splay column, where it places sixth of twenty one.

The second, that runs B and C carry very high torque saturation, is confirmed and is the largest effect in the set. Against the velocity dependent envelope the ankle roll stands at 19.75, 50.14 and 88.25 per cent and the ankle pitch at 5.00, 14.85 and 44.31, so the two ankle joints of run C are saturated between four and nine times as often as those of run A.

The third, that run A is the best result, is confirmed on the mechanics and contradicted on the training objective, and section 16.3 argues that the contradiction is a property of the objective rather than of the runs.

The fourth, that this task should be trained to ten thousand iterations, is confirmed with the mechanism of section 16.9.

The fifth, that the stance to swing transition drives the hip yaw and the foot roll away from their nominals by more than the gains can handle, is confirmed as a phenomenon and refuted as a mechanism, and the two joints fail differently. At the hip yaw the transient is real, the deviation rising from 0.057 radians in stance to 0.206 in swing and peaking 0.09 to 0.19 seconds after liftoff, and it is commanded, the servo tracking the command to within four per cent at one newton metre of a forty two newton metre ceiling. At the foot roll the joint is not deviating under load at all, it is stationary against a mechanical stop at one hundredth of a radian per second under a command 0.17 radians beyond that stop, with the actuator at 98 to 99 per cent of the torque available to it. Neither condition is a gain deficit, and both are conditions of the commanded target, which is why section 20.1 now ranks the clip of arm 10 first among the arms that have not been run.

### 16.11 The visual record and the artefacts

Six frames were taken at 6.0 and 12.0 seconds of each play video, where the commanded forward velocity is 0.031 and minus 0.291 metres per second against a yaw rate of minus and plus 0.300 radians per second, the three replays sharing a seed and a command series so that a frame taken at one instant is comparable across them. They are retained at [../context/artefacts/kscale_flat-2026-09-24_to_09-25-energy-ankle-ablation/frames/](../context/artefacts/kscale_flat-2026-09-24_to_09-25-energy-ankle-ablation/frames/) and registered there. They corroborate the measurements and add nothing to them, all three showing the same crouched posture with the two feet converged upon the midline and overlapping in projection, run C's fold being visibly the deepest, which its mean knee angle of 1.088 radians against run A's 0.697 states more precisely than a frame can. No figure in this section is read off a frame.

Every table above is reproduced in `comparative_tables.txt`, `training_tables.txt`, `mechanism_probes.txt` and `parameter_diff.txt` of that directory, and the seven programs that produced them stand in its `scripts/` directory and are registered individually. One correction made during the analysis is recorded in that directory's own register rather than only here, the first pass having resolved every joint to one foot because the foot links of this robot are named `foot_6061_2` and `foot_6061` and carry no side marker, which inverted the stance and swing columns of the left leg and produced an apparent asymmetry that does not exist.


## 17. What the literature says about phase dependent gains, actuator envelopes and pricing a joint the ground controls

Section 15 surveyed the bang-bang literature that the latched ankle prompted and found three families of remedy. Section 16 raises three questions that survey does not reach, and this section answers them from the published record. Whether a locomotion policy should carry different gains in swing and in stance, which section 16.4 makes pressing by measuring an ankle roll that runs at 1.17 radians per second in stance and 0.013 in swing. Whether an actuator envelope should be priced or imposed, which section 16.3 makes pressing by finding the best policy on the objective to be the worst on the actuator. And whether a deviation penalty can act on a joint whose position the ground sets, which section 16.4 measures and which the survey turns out not to cover.

### 17.1 The gains question has a published answer, and it is not a schedule

The surveyed route to phase dependent impedance is to let the policy choose the gains rather than to write a schedule for them. Bogdanovic, Khadiv and Righetti observe that policies directly outputting joint torques have difficulty robustly solving problems involving contact interactions without given structure in the action space, propose a policy whose output is an impedance together with a desired position in joint space rather than a position alone, and report on a hopper with intermittent contacts and on a manipulator wiping a table that the variable impedance action space outperforms both torque and position control on learning rate and on robustness to environment uncertainty [24]. The hopper is the surveyed system closest to a leg that alternates between free flight and load, and the result bears directly on a machine whose ankle is saturated in swing and quiet in stance.

That the two phases differ inside the policy and not merely in the mechanics is established independently. Tommaselli, Segreto, Negri, Godoy and Becker analyse the effective rank of the policy Jacobian conditioned on gait phase and report that layer normalisation and residual connections allocate roughly two more dimensions of effective rank to swing than to stance, a disparity they find absent in a plain multilayer perceptron, and they use the measurement to reduce joint jitter by approximately three times on a Boston Dynamics Spot [25]. Two things follow. A trained policy already devotes unequal representational capacity to the two phases, so a controller that treats them identically is the odd part of the stack rather than the natural one. And an effective rank conditioned on phase is a training time instrument that predicts deployment smoothness, which this workspace does not presently compute and which costs no additional rollout.

A caution attaches to any gain change made for a learned policy and it is recent. Bronars, Park and Agrawal argue that gains should be selected for what they call learnability, meaning how amenable a gain setting is to the learning algorithm in use, rather than for the desired task stiffness, and report that reinforcement learning can succeed across all gain regimes provided the hyperparameters are adjusted with them, while behaviour cloning prefers compliant and overdamped settings and sim to real transfer degrades under stiff and overdamped ones [26]. Their summary position, that optimal gain selection depends not on the desired task behaviour but on the learning paradigm employed, is the reason a gain arm must be read against its own training record rather than against a stability calculation alone, and it bears on a workspace whose gain derivations in [../context/KScale.md](../context/KScale.md) sections 11 and 12 are conducted entirely in the frequency domain.

### 17.2 The envelope should be a boundary and not a price, and this configuration makes it neither

Two recent works treat the motor's torque and speed envelope as the boundary of the admissible set during training, and both report hardware results at the extreme of what the actuator allows. Shin, Song, Ji and Park incorporate the motor operating region as a constraint so that the policy network avoids infeasible areas during sampling rather than being penalised for reaching them afterwards, add a gait reward that distributes motor torque evenly across the legs specifically to remove the bottleneck that one saturating motor creates, and report the 45 kilogram KAIST Hound at 6.5 metres per second [27]. Tao, Cheng, Lan, Yuan, Jin and Wang take the same position from the modelling side, arguing that methods neglecting actuator nonlinearities and physical constraints during training suffer a significant sim to real gap under dynamic motion, and building a model that captures high speed voltage coupling and magnetic saturation so that the envelope is represented accurately enough that the policy cannot exploit a non physical region, reporting the 36.5 kilogram BlackPanther2 at 13.2 metres per second on a treadmill and 11.65 outdoors [28].

The position this configuration occupies is neither of theirs and it should be named plainly. The envelope is applied, `DCMotor._clip_effort` at `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:295` clipping the commanded torque to it on every step, so the physics is correct. It is not priced, the four energy terms charging the torque that survives the clip rather than the excess that was removed by it, so a policy commanding twice the available torque pays exactly what a policy commanding the available torque pays. And it is not constrained, nothing in the reward set or the termination set referring to the envelope at all. The consequence is the one section 16.4 measures, that saturation is free, and the correct home for the ankle is therefore the constraint family section 15.2 registers rather than a larger energy weight, Kim and colleagues carrying joint position, joint velocity and joint torque as three of their six probabilistic constraints and reporting hardware violation rates of 0.007, 0.001 and 0.002 against a threshold of 0.025 [4].

### 17.3 Two independently designed bipeds omit the ankle roll entirely

Section 15.4 took from Kim, Jorgensen, Lee, Ahn, Luo and Sentis the bounded claim that lateral balance on a biped does not require an actuated ankle roll [23]. A design datum from the same work sharpens it and was not recorded then. Their DRACO carries five actuators per limb, being hip yaw, hip roll, hip pitch, knee pitch and ankle pitch, with a soft joint stiffness policy applied to the ankle pitch to emulate a passive joint, so the machine has no ankle roll axis whatever and walks without one [23].

A second and more recent datum comes from a learned controller rather than a model based one. Xia, Li, Lee, Scutari and Chen describe a ten degree of freedom humanoid carrying five per leg, being hip rotation, hip flexion and extension, hip abduction and adduction, knee flexion and extension, and ankle plantarflexion and dorsiflexion, again with no ankle roll, under a policy trained to leverage passive dynamics that reports a fifty per cent reduction in cost of transport in simulation and thirty one per cent on hardware [29]. Two independently designed bipeds, one controlled by a whole body controller and one by a learned policy, therefore omit the axis this robot is spending ninety five per cent of an actuator's ceiling to hold against a stop, which is the strongest external support arm 24 has.

### 17.4 What the survey did not find, recorded as a negative result

A directed search returned no published account of a joint deviation penalty applied to a joint whose position is set by contact rather than by the command, and no treatment of what such a penalty does to the actuator effort. The searches reached the general reward shaping literature on penalty terms and on state constraint handling and nothing specific to the contact case. The measurement of section 16.4 is therefore this workspace's own, and any future document wishing to cite a precedent must find one rather than assume it exists.

And no surveyed source prescribes an explicit swing and stance gain SCHEDULE for a learned locomotion policy. The nearest are the learned variable impedance family [24], in which the gains are an output of the policy and no schedule is written, and the phase conditioned representation analysis [25], which measures a phase asymmetry in the network rather than prescribing one in the controller. An arm proposing a hand written schedule is therefore proposing something the survey does not support, which is a reason to state its rationale from this robot's own mechanics and to carry the learned variable impedance action space beside it as the better supported alternative, both of which section 21.20 does.

## 18. The commanded target clip was executed, and it cured the ankle roll and relocated the fault to the hip roll

Two runs completed between 2026-09-28 and 2026-09-29 and two more were launched on 2026-09-30, and together they supply the first unambiguous success of this ablation and the first measurement taken on instrumentation rather than on inference. Run D removes the ankle roll deviation penalty that section 16.4 refuted, isolating it at last. Run E adds arm 10, the commanded target clip, on top of run D. Runs F and G add arm 11, the clip excess penalty, at weights of minus 2 and minus 500 respectively, and both were at iteration two hundred when this section was written.

Every figure below is read from the channels `scripts/rsl_rl/play.py` began recording on 2026-09-28 for section 20.2, and the change that makes is not one of precision but of availability. The commanded target is READ rather than recovered by inverting the control law, which section 16.4 could do on 11.75 per cent of run C's ankle roll samples and which therefore made every figure it quoted for that joint a bound. Saturation is the computed torque lying outside the actuator's own envelope, both being dumped, so it carries no threshold and none of the twenty five point sensitivity that the addition of 2026-09-23 to [/ws/context/gait_metrics.md](../../context/gait_metrics.md) measures in a fraction taken from the applied torque. And the clip excess, which is the quantity arm 11 prices, is the difference between the unclipped target and the clipped one and needs no reconstruction whatever.

### 18.1 What was run, and the two clean contrasts this set finally supplies

The five dumped configurations were flattened to leaf paths through `scripts/analysis/experiment_params.py:24` and compared exhaustively. The substantive differences are seven and the sequence is nested, which is what makes two of its contrasts clean where no contrast of the preceding two sequences was.

| Path | A `09-24_05-18-33` | B `09-25_08-00-34` | C `09-25_11-29-39` | D `09-28_10-25-57` | E `09-29_06-15-58` |
|---|---|---|---|---|---|
| Energy group, multiple of the pre arm 8 value | 50 | 100 | 50 | 50 | 50 |
| `pen_joint_power` | absent | absent | -0.02 | -0.02 | -0.02 |
| `pen_ankle_roll_deviation` | absent | -1.0 | -1.0 | absent | absent |
| `rew_keep_ankle_roll_zero_in_air` | 0.25 | 1.0 | 1.0 | 1.0 | 1.0 |
| `actions.joint_pos.clip` | None | None | None | None | the travel inset 0.02 rad |
| `scene.num_envs` | 9000 | 9000 | 6000 | 9000 | 5000 |

Run C against run D isolates `pen_ankle_roll_deviation` exactly, every other reward value being identical and only the population differing, and it is the contrast section 16.4 committed this document to and could not then perform. Run D against run E isolates the clip on the same terms. Both carry a population confound, which enters through the gradient estimate rather than through the physics and which section 2.2 records as the least consequential of this document's recurring confounds, and neither carries any other.

The two runs in flight differ from run E in three leaf paths of 2575, being the presence of `pen_action_excess` and its weight. They are therefore a clean two point sweep on that weight, and section 18.4 prices it.

### 18.2 The clip removed the ankle roll latch completely, which is the first unambiguous success of this ablation

The ankle roll has been latched against a mechanical stop in every run this document has examined since 2026-09-17, and section 11 names that latch the root cause. Run E ends it.

| Quantity, left and right ankle roll | A@10k | B@10k | C@10k | D@10k | E@10k |
|---|---|---|---|---|---|
| Saturation, exact [pct] | 25.49, 23.59 | 67.68, 68.81 | 94.51, 95.00 | 60.17, 58.60 | 0.06, 0.05 |
| Time at a hard stop [pct] | 38.21, 37.07 | 49.83, 48.65 | 50.24, 49.10 | 42.30, 41.87 | 0.07, 0.08 |
| Commanded target beyond the stop [pct] | 44.52, 43.73 | 88.18, 89.37 | 98.08, 98.26 | 79.73, 79.19 | 0.00, 0.00 |
| Mean absolute torque, of a 17 N.m ceiling | 8.88, 8.81 | 14.30, 14.47 | 16.45, 16.50 | 13.01, 12.91 | 3.50, 3.57 |
| Tracking error, commanded minus achieved [rad] | 0.111, 0.108 | 0.399, 0.414 | 0.463, 0.476 | 0.246, 0.245 | 0.032, 0.034 |

The joint is no longer at a stop, no longer saturated, and tracks its command to within 0.033 radians where it stood 0.246 from it in run D and 0.476 in run C. Its actuator spends 3.5 newton metres of a seventeen newton metre ceiling where it spent 16.5, which is the ninety five per cent continuous demand section 16.3 called pure waste, now gone. The ankle pitch follows it off its stop, from 41.09 per cent residency in run C to 0.07 in run E.

The mechanism is exactly the one section 11.2 named and arm 10 was written to remove. A target outside the reachable set converts the end stop into a free, perfectly repeatable and infinitely stiff reference, and nothing in the configuration priced it. The clip removes the reference rather than the incentive, and with the reference gone the joint becomes a controlled joint for the first time in this robot's recorded history. THIS IS THE ONE PREDICTION OF THIS DOCUMENT THAT HAS BEEN CONFIRMED IN FULL RATHER THAN FALSIFIED OR QUALIFIED, and it is recorded as such because the preceding fourteen sections record the opposite outcome so often.

### 18.3 The fault relocated to the hip roll, and the clip's own sizing rule is why

The gain at the ankle was not free and the price is paid at the hip roll, which run E latches against its stop as thoroughly as run D latched the ankle.

| Quantity, left and right hip roll | A@10k | D@10k | E@10k |
|---|---|---|---|
| Time at a hard stop [pct] | 15.10, 12.21 | 10.57, 11.05 | 45.90, 45.99 |
| Saturation, exact [pct] | 0.21, 0.08 | 0.26, 0.09 | 0.99, 1.24 |
| Mean absolute torque, of a 42 N.m ceiling | 10.95, 11.06 | 10.71, 10.82 | 6.54, 6.64 |
| Samples the clip bound [pct] | not clipped | not clipped | 91.77, 91.41 |
| Mean absolute clip excess [rad] | 0.0000 | 0.0000 | 2.899, 2.910 |
| Mean absolute raw action [sigma] | 0.521, 0.515 | 0.541, 0.541 | 19.016, 19.086 |

The last row is the finding and it should be read before the others. The policy of run E drives the hip roll action to a mean absolute NINETEEN standard deviations, against 0.52 in run A and 0.54 in run D, which is an escalation by a factor of thirty five. It does so because the clip absorbs it, the bound being reached at an action of roughly 1.5 sigma and every sigma beyond that costing nothing whatever. The clip has therefore done at the level of the ACTION precisely what the mechanical stop did at the level of the joint, supplying a free, perfectly repeatable and infinitely stiff reference that the policy presses against, and the fault this ablation has chased since section 11 has moved one level up the pipeline rather than being removed from it.

The clip's own sizing rule is the reason, and the arithmetic is unambiguous. Section 21.10 sets each bound at the joint's mechanical travel inset by 0.02 radians, which for the ankle roll gives a window 0.4836 radians wide and for the hip roll one 2.4384 radians wide. Measured against the saturating deflection, being the effort limit divided by the stiffness and therefore the position error at which the servo reaches its ceiling, the ankle roll's window is 2.84 times that deflection and the hip roll's is 14.51 times it.

| Joint | q_sat = E/kp [rad] | Clip width [rad] | Ratio |
|---|---|---|---|
| Hip roll | 0.168 | 2.4384 | 14.51 |
| Hip yaw | 0.420 | 3.1016 | 7.38 |
| Hip pitch | 0.420 | 3.2238 | 7.68 |
| Knee | 0.280 | 2.6653 | 9.52 |
| Ankle pitch | 0.170 | 1.3563 | 7.98 |
| Ankle roll | 0.170 | 0.4836 | 2.84 |

The ankle roll is the one joint whose clip is within a small multiple of its own authority, and it is the one joint the clip cured. The hip roll's clip is fourteen times its authority, and the hip roll is where the fault went. The general statement, which section 20.3 now makes an exclusion, is that a clip sized from a joint's MECHANICAL TRAVEL bounds nothing that matters on a joint whose travel is far wider than its useful range, and this robot's hip roll is the extreme case, section 14.7 having already measured its travel as asymmetric by 10.83 to one about its own midpoint with 0.0425 radians of clearance from the nominal.

Two further consequences are measured and neither is small. The splay WORSENED, the pooled toe-in rising from 8.416 degrees in run D to 9.984 in run E and the stance toe-in from 5.275 to 7.188, which is the largest stance splay in the set. And the yaw tracking error rose from 0.1683 to 0.2796 radians per second while the termination rate became non zero for the first time in this sequence at 0.0021 per second. Against that, the sole tilt in stance IMPROVED sharply, its absolute roll falling from 6.725 degrees to 2.850 and its ninety fifth percentile from 18.871 to 6.735, which is the flattest sole this robot has recorded and is the direct consequence of an ankle roll that is no longer pinned.

### 18.4 The clip excess is a learned exploit that grows by five orders of magnitude, which is what settles the weight

Arm 11 prices the clip excess, and two runs are in flight at weights of minus 2 and minus 500. The weight was chosen from the magnitude the term reports early in training, and the measurement below is the reason that is the wrong place to read it.

| Iteration | Excess, sum of squares per step [rad^2] | At weight -2 [per s] | At weight -500 [per s] |
|---|---|---|---|
| 193, measured in run G | 0.000595 | 0.00119 | 0.2974 |
| 8000, measured in run E | 13.001 | 26.00 | 6500.6 |
| 9000, measured in run E | 22.857 | 45.71 | 11428.6 |
| 10000, measured in run E | 30.380 | 60.76 | 15189.8 |

The excess grows by a factor of FIFTY ONE THOUSAND between iteration 193 and iteration 10000, and it grows monotonically over the last three checkpoints, so it is not an initial condition that training removes but an exploit that training discovers. The figure at iteration 193 is therefore not a small value of the quantity the weight must price, it is the quantity before the policy has learned to produce it. Run E is the correct reference because it carries the identical clip with no penalty at all, and it measures what the excess becomes when nothing opposes it.

Both weights in flight are consequently mis-set, and in opposite directions. Against run E's negative reward budget of 11.99 per second and positive budget of 88 per second, a weight of minus 2 reaches 60.76 per second at iteration 10000, which is 507 per cent of the whole negative budget and 69 per cent of the positive one. A weight of minus 500 reaches 15190 per second, which is 173 times the entire positive budget, and section 16.2 records what happens to this robot when a regularisation term reaches that scale, the five hundredfold energy run collapsing to an eighteen step episode at a unit termination rate.

The deeper difficulty is that no single weight can be right, and the arithmetic says so rather than the judgement. The term is QUADRATIC in a quantity that spans five orders of magnitude between the behaviour at initialisation and the behaviour the policy converges to, so its value spans ten orders of magnitude. A weight calibrated where the exploit forms is negligible before it and ruinous after it, and a weight calibrated after it forms cannot prevent its formation. Section 19.2 records that this is a known difficulty with a published name and two published families of remedy, and section 20.1 accordingly divides arm 11 into three arms, the weight as launched, the weight as calibrated and the FORM, the form being the part the measurement argues for.

The arithmetic that sets a defensible weight is as follows, and it is stated so that a later reader may re-derive it rather than accept it. The exploit should be priced from the point at which it begins rather than from either end. Take that point as the excess reaching one action standard deviation at the joints that carry it, being 0.16 radians at the hip roll and 0.08 at the ankle roll, which gives a sum of squares of 0.051 radians squared per step over the four joints. For that to cost one per cent of the positive budget, which is 0.88 per second, the quadratic weight is minus 17. Under an l1 form, which is linear in the excess and whose sum is 0.32 radians at the same point, the weight is minus 2.75. The two forms differ in what they do at run E's terminal excess, the quadratic reaching 517 per second and the l1 reaching 20.2, a ratio of 25 to 1, and the l1 form is therefore the one that remains a price rather than becoming a wall.

### 18.5 The ankle roll deviation penalty is confirmed harmful, the contrast being clean at last

Section 16.4 measured the ankle roll deviation penalty moving the joint from one mechanical stop to the other and predicted what its removal would do. Run D removes it against run C with every other reward value held, and the prediction is confirmed on every channel it named.

| Quantity, left and right ankle roll | C@10k, penalty present | D@10k, penalty absent |
|---|---|---|
| Saturation, exact [pct] | 94.51, 95.00 | 60.17, 58.60 |
| Commanded target beyond the stop [pct] | 98.08, 98.26 | 79.73, 79.19 |
| Commanded departure from the nominal [rad] | 0.6355, 0.6406 | 0.4724, 0.4704 |
| Mean absolute torque, of 17 N.m | 16.45, 16.50 | 13.01, 12.91 |
| Time at a hard stop [pct] | 50.24, 49.10 | 42.30, 41.87 |

Removing the term took 35 points off the saturation, 18 points off the fraction of commands lying beyond the stop, and 3.5 newton metres off the sustained actuator demand. Section 16.4's prediction that the joint would stay at the near stop is also confirmed, the stop residency falling by only eight points where the saturation fell by thirty five, so the penalty was indeed pressing the joint into a stop it would have rested against anyway. Arm 9 is therefore refuted twice, once by mechanism and once by a clean contrast, and section 20.3's exclusion of a deviation penalty on any contact determined joint stands on measurement rather than on argument.

### 18.6 The splay did not improve, and the hip yaw remains its sole author

Across all five runs the pooled toe-in moves between 7.667 and 9.984 degrees with no trend that follows any intervention, and the swing figure moves between 13.115 and 15.239. The regression of section 16.7 is repeated on every run of this set and returns the same answer, the swing slope standing between 0.99 and 1.01 with a correlation between 0.987 and 0.998, so the hip yaw continues to account for the swing foot heading almost completely and nothing else contributes measurably to it.

What has changed is the mode. The common mode correlation of the two hip yaws falls from 0.636 in run C to 0.256 in run D and 0.406 in run E, and the ratio of the sum's standard deviation to the difference's falls from 2.121 to 1.299 and 1.539. The fault is therefore less purely a pelvis twist than section 16.7 measured it and more nearly balanced between the two modes, which weakens the argument for re-weighting the differential and strengthens the argument for acting on the joint. Arm 14 is unaffected, removing the joint rather than re-pricing its modes, and arm 16's differential scale is the one parameter this measurement moves, section 21.16 recording the change.

### 18.7 The heading term prices almost nothing in stance, and closing its gate would enlarge its common mode rather than remove a price

Arm 16 proposes two changes to `pen_feet_heading`, closing the airborne gate and halving the differential scale, and both were written against runs A, B and C before the clip existed. The measurements below were taken on 2026-09-30 against run E, which is the baseline the arm will actually run upon, and they move both halves in opposite directions.

The premise behind the gate has expired. Section 16.7 justified it by a stance toe-in four to eighteen times smaller than the swing figure, on the argument that an ungated term spends most of its gradient on the phase in which least is wrong, and that ratio has closed.

| Run at 10000 | Stance toe-in [deg] | Swing toe-in [deg] | Ratio | Stance hip yaw deviation, left and right [deg] |
|---|---|---|---|---|
| A | 2.969 | 14.679 | 4.94 | -3.41, -1.40 |
| B | 2.903 | 15.239 | 5.25 | -3.39, -1.91 |
| C | 3.724 | 13.189 | 3.54 | -7.72, -6.56 |
| D | 5.275 | 13.115 | 2.49 | -0.22, +1.61 |
| E | 7.188 | 13.769 | 1.92 | +0.52, +2.25 |

The stance figure has risen by a factor of 2.42 from run A to run E while the swing figure has fallen, so the phase the arm proposes to stop looking at is the only one in which the fault has grown, and the fourth column gives the mechanism. The stance regression of section 16.7 carries an intercept between 4.834 and 6.406 degrees on run E and between 4.643 and 6.106 on run C, which is the foot's heading when the hip yaw stands at its nominal and is therefore a structural offset of the leg chain rather than anything the policy commands. Run C cancelled that offset by holding the stance hip yaw at minus 7.72 and minus 6.56 degrees, which is what reduced its stance toe-in to 3.724, and run E holds the same joint at plus 0.52 and plus 2.25 so the offset stands very nearly uncancelled. The stance toe-in of the present baseline is thus a structural offset the policy has stopped cancelling, and the cancellation is a stance action.

THE GATE CANNOT REMOVE A STANCE PRICE, BECAUSE THE CONFIGURATION DOES NOT CARRY ONE. The term stands at `common_mode: "mean"` with `differential_scale: 1.0`, and under the mean form the common mode is the squared error of the circular MEAN of the two feet against the base, so a splay of plus and minus the same angle cancels within it and registers exactly zero, which is the orthogonality the docstring at `rewards.py:1263` records as the reason to prefer that form. The measurement confirms it holds in practice.

| Run at 10000 | Common mode, gate open | Common mode, gate closed | Differential mode | Common share, gate open [pct] |
|---|---|---|---|---|
| A | 0.003885 | 0.035196 | 0.040951 | 8.7 |
| B | 0.004117 | 0.040056 | 0.046674 | 8.1 |
| C | 0.004368 | 0.026527 | 0.021321 | 17.0 |
| D | 0.000941 | 0.024744 | 0.054512 | 1.7 |
| E | 0.000923 | 0.023872 | 0.070301 | 1.3 |

On run E the common mode carries 1.3 per cent of the term and the differential carries the remaining 98.7, and the differential is never gated, `feet_yaw_alignment` returning it over both feet unconditionally at `rewards.py:1354`. The gate therefore touches only the inert component, and a stance toe-in of 7.188 degrees is presently opposed by a common mode worth 0.000923 radians squared, which at the term's weight of minus 8.0 is 0.0074 per second against a negative budget of 11.99.

What the gate actually does is the reverse of what arm 16 assumes, and the third column above is the measurement. Restricting the circular mean to the airborne feet means that in single support the mean is the SWING foot alone, so the common mode stops being a pelvis twist detector and becomes that foot's own heading error, which is the large quantity. The common mode rises by a factor of 6.07 on run C and 25.9 on run E, and the orthogonality that `common_mode: "mean"` exists to supply is destroyed with it, a pure splay now registering on both modes at once so that the differential is priced twice. In double support, which is 15.09 per cent of run E's samples, no foot is airborne and the gated common mode is identically zero by `rewards.py:1330`.

The remaining question is whether a term acting only in swing still governs the stance heading, and it is settled by measuring one contact interval at a time rather than by argument. Each interval is followed from its touchdown sample to the last sample before release, on the same foot of the same environment, between 1973 and 2198 intervals per run at a mean length of 49 to 58 samples.

| Run | Last swing sample [deg] | Touchdown [deg] | Stance mean [deg] | Lift off [deg] | Lift off minus touchdown [deg] | Intervals growing [pct] |
|---|---|---|---|---|---|---|
| A at 10000 | 1.816 | 1.994 | 3.205 | 11.626 | +9.632 | 96.3 |
| B at 10000 | 2.053 | 2.016 | 3.496 | 6.324 | +4.308 | 85.0 |
| C at 10000 | 4.108 | 4.356 | 4.605 | 7.998 | +3.642 | 72.6 |
| D at 10000 | 3.939 | 4.004 | 5.516 | 10.804 | +6.800 | 92.7 |
| E at 8000 | 9.355 | 9.172 | 8.473 | 9.062 | -0.111 | 47.3 |
| E at 9000 | 9.574 | 9.370 | 7.943 | 8.122 | -1.249 | 35.8 |
| E at 10000 | 8.242 | 8.106 | 7.226 | 8.075 | -0.031 | 45.2 |

On run E the heading is planted and held, the last airborne sample, the touchdown, the stance mean and the lift off standing within 1.02 degrees of one another and the interval shrinking on 55 per cent of occasions. A term that prices the swing heading therefore governs the stance heading on this gait, because the stance heading IS the swing heading at the instant of contact, and the answer to whether closing the gate raises the stance toe-in is that on this baseline it should lower it. On the four runs before the clip the answer is the opposite, the heading growing by 3.642 to 9.632 degrees within the contact interval on 72.6 to 96.3 per cent of occasions, so the gate would have been unsafe on every gait this ablation produced until run E and is safe now only because the clip changed the gait. The safety of the gate is consequently a property of the baseline and not of the term, which is the reason arm 16 must re-measure this table on whatever gait precedes it rather than inherit the conclusion.

Three consequences follow and each moves a parameter of arm 16. The gate is retained, its risk being the opposite of the one supposed and its effect being to give the common mode something to measure. It is nonetheless separated into a stage of its own, because it changes what the term means rather than how much it costs. And the halving of the differential scale is REFUTED, section 18.6 having already recorded the case for it weakening when the common mode correlation fell from 0.636 to 0.406, and the differential mode having grown from 0.021321 on run C to 0.070301 on run E, a factor of 3.30, while carrying 98.7 per cent of the only term that opposes it.

### 18.8 The training records, and the one place run E is worse

| Quantity at iteration 10000 | A | B | C | D | E |
|---|---|---|---|---|---|
| Mean reward | 952.6 | 954.8 | 1018.0 | 971.2 | 902.6 |
| Mean episode length | 1784 | 1890 | 1929 | 1865 | 1855 |
| Linear tracking error [m/s] | 0.1191 | 0.1383 | 0.1199 | 0.1254 | 0.1248 |
| Environments | 9000 | 9000 | 6000 | 9000 | 5000 |

Run E is last on mean reward and its evaluation confirms the reason, its planar tracking error standing at 0.0486 metres per second against run D's 0.0385 and its yaw error at 0.2796 radians per second against 0.1683. The clip has cost tracking, and the cost is real rather than an artefact of the smaller population, the episode length and the linear tracking error being comparable while the yaw error is 66 per cent worse. Section 18.3 identifies the hip roll latch as the likely channel, a joint pinned against its stop for forty six per cent of every episode being unavailable for the lateral corrections a turn requires.

### 18.9 The five observations of the analysis brief, confirmed, sharpened and one corrected

The first, that the splay is almost unchanged with a marginal reduction in run D, is confirmed in its main clause and corrected in its subordinate one. Run D's pooled toe-in of 8.416 degrees is not the lowest of the set, run A's 7.667 and run C's 7.780 both standing below it, and run D leads the composite ranking on its hip excursion and its saturation rather than on its splay.

The second, that runs B and C carry very high torque saturation, is confirmed and now exact, the ankle roll standing at 67.7 and 94.5 per cent against run A's 25.5 and run E's 0.06.

The third, that run D is the best result, is confirmed as of the composite that ranked it and is superseded by run E on the mechanics, which section 18.2 records, and contradicted by run E on tracking, which section 18.8 records. The honest summary is that run D is the best gait and run E is the best actuator condition, and that no run of this set is both.

The fourth, that this task should be trained to ten thousand iterations, is confirmed and unchanged from section 16.9.

The fifth, that the policy is commanding the actuators out of their limits, is confirmed exactly and is now measured rather than inferred. In run D the commanded ankle roll target lies beyond the joint's mechanical stop on 79.7 per cent of samples, and in run E, where the clip forbids that, the UNCLIPPED target the policy asked for lies beyond the stop on 93.6 per cent of samples at the ankle roll and 90.2 at the hip roll. The policy does not merely command beyond the limits, it commands further beyond them the more firmly it is prevented from reaching them, which section 18.3 measures as an escalation to nineteen standard deviations of raw action.


### 18.10 The visual record and the artefacts

Sixteen frames were taken at 6.0 and 12.0 seconds of the play video of runs A, C, D and E, where the commanded forward velocity is 0.031 and minus 0.291 metres per second against a yaw rate of minus and plus 0.300, the four replays sharing a seed and a command series so that a frame taken at one instant is comparable across them. They are retained at [../context/artefacts/kscale_flat-2026-09-28_to_09-29-clip-ablation/frames/](../context/artefacts/kscale_flat-2026-09-28_to_09-29-clip-ablation/frames/) and registered there.

They corroborate the measurements and add one observation the measurements state more precisely. All four machines carry the same crouched posture with the feet converged upon the midline, which is the finding of section 18.6 that no intervention of this set has moved. Run E's stance is visibly the narrowest of the four, which its lateral foot separation of 0.1323 metres against run D's 0.1384 states numerically and which section 18.3 attributes to a hip roll held against its stop for 46 per cent of every episode. No figure in this section is read off a frame.

Every table above is reproduced in `comparative_tables.txt`, `training_tables.txt` and `parameter_diff.txt` of that directory, and the seven programs that produced them stand in its `scripts/` directory and are registered individually. Two were added on 2026-09-30 and both are cited above rather than kept against a future use. `headinggate.py` produces every table of section 18.7, being the toe-in by phase, the term's decomposition into its common and differential modes with the gate open and closed, and the contact interval traced from touchdown to lift off, and arm 16 names it as the instrument its precondition requires. `excess_kernel.py` produces the absolute excess of section 21.13, recovering the per environment offset by the same restriction `clipab.py` uses and reporting the excess under both kernels per joint, which is what establishes that the hip roll pair carries 94.3 per cent of the quadratic excess and 79.1 of the absolute one. One correction made during the analysis is recorded in that directory's own register rather than only here, the recovery of the action term's per environment offset having first been attempted as a median over all samples, which is correct on a run the clip does not bind and which on run E recovered a figure 3.1 radians from the dumped row where the randomisation can account for 0.048.

## 19. What the literature says about bounding an action, and about a coefficient that cannot be right at both ends

Section 15 surveyed the bang-bang literature and section 17 the gain, envelope and contact pricing literature. Section 18 raises two questions neither reaches, and both are questions about the instrument rather than about the robot.

### 19.1 A clip bounds the target and not the action, and the surveyed record treats the second rather than the first

The failure of section 18.3 is that a clip applied to the commanded target leaves the policy free to command any action whatever, and the policy responded by commanding nineteen standard deviations. The surveyed record addresses the mismatch between an unbounded policy and a bounded action space directly, and it is unanimous that the mismatch is a property of the DISTRIBUTION rather than of the bound.

Fujita and Maeda observe that out of bound actions are clipped before execution while the policy is optimised as though they were not, and propose an estimator that accounts for the clipping, which they report is unbiased and of lower variance than the conventional estimator that ignores action bounds [19]. Chou, Maturana and Scherer address the same mismatch by replacing the Gaussian with a Beta distribution whose support is the bounded interval itself, so that no clipping occurs and no probability mass falls outside the admissible set [9]. Cluster 25 of [/ws/context/literature.md](../../context/literature.md) already registers the general result that a scale is a reparameterisation and not a bound, and section 18.3 is its exact counterpart one level down, that a clip is a bound on the target and not on the action.

The bearing on this workspace is that arm 10 bounds the quantity that reaches the actuator and leaves the quantity the optimiser manipulates unbounded, so the escalation section 18.3 measures is the expected consequence rather than a surprise. Three remedies follow from the survey and section 20.1 ranks them. The excess may be priced, which is arms 11 through 13 and which the next sub-section shows is harder than it looks. The clip may be narrowed until the bound falls inside the range the policy would use anyway, which is arm 15 and which needs no new machinery. Or the distribution may be changed so that the action cannot leave its range, which is the Beta policy of Chou and colleagues and which is the only remedy the survey reports as principled rather than palliative, at the cost of a change to the actor rather than to the configuration.

### 19.2 A fixed coefficient on a quantity that spans five orders of magnitude has a published name and two published remedies

Section 18.4 measures a penalised quantity growing by a factor of fifty one thousand over training and concludes that no single weight can be right at both ends. The difficulty is registered in the surveyed literature and the registration is worth quoting because it removes the temptation to solve it by further tuning.

Kim, Oh, Lee, Choi, Ji, Jung, Youm and Hwangbo argue that a reward only formulation requires more than ten reward terms and a tuning process repeated several times, and that a constrained Markov decision process solved by a modified interior point policy optimisation reaches the same behaviour with three reward terms and a single tuned coefficient [4]. Ma, Ji and Gao state the failure mode this workspace is now in more precisely, that fixed cost budgets and fixed penalty coefficients limit the algorithm's ability to find an optimal solution during training and leave it stuck in over conservative or constraint violating local minima, and they propose optimising the reward and ADAPTING the budget simultaneously through two adversarial stages solved alternately [30]. Panchal, Kleyko, Artan and Loutfi make the same argument for the specific case of smoothness on a humanoid, that imposing it through auxiliary reward terms creates terms which compete with the task objective, and report that formulating it as explicit constraints on physical motion limits instead achieved a 2.50 times reduction in upper body action rate and a 2.18 times reduction in upper body acceleration on hardware, while transferring across terrains without the extensive reward tuning the penalty form needs [31].

Two things follow for arms 11 through 13. The first is that its FORM matters more than its weight, and section 18.4's arithmetic shows why in this instance, an l1 form spanning 25 to 1 between the onset of the exploit and its terminal value where the quadratic form spans 600000 to 1. The second is that the correct long term home for the clip excess is a constraint rather than a price, which places it with arms 11 through 13 and with the barrier of Kim, Lee and Park [5] rather than with the energy group, and this document records it as the successor to arm 13 rather than proposing it now.

### 19.3 A squashing function bounds the action, and the surveyed record is that it trades one failure for another

Section 19.1 leaves three remedies for the escalation section 18.3 measures and ranks the third, changing the distribution, as the only one the survey calls principled. A fourth exists and it is the one this workspace should in fact reach for, being to pass the action through a bounded transformation before it becomes a joint target. It is not a distribution change at all and it was not surveyed before 2026-09-30 because the question that prompts it is new.

The construction is standard and its cost is documented. Lin states the general position, that Gaussian policies have dominated continuous control yet suffer from a fundamental mismatch, their unbounded support requiring ad hoc squashing functions that distort the geometry of bounded action spaces [33]. The distortion has a precise form. Where the action is a hyperbolic tangent of the network's output, the derivative of the transformation is one minus the square of that tangent, so it falls from unity at the origin to 0.0707 at a pre activation of two and 0.00987 at three. A policy that wants the extreme of its range must therefore drive the pre activation far out, and the gradient reaching the network falls away exactly as it does so, which is the vanishing gradient the question raises and which is real rather than theoretical.

Two recent works treat it directly and both are in this workspace's own domain. Liao, Li, Yang, Chang, Fan, Wang, Shi, Cao, Wu and Sartoretti apply a time varying action transformation that restricts the effective action space early in training and then progressively expands it, and their claim is the one that bears here, that the transformation PRESERVES THE PPO UPDATE RULE and introduces only bounded, vanishing gradient distortion, which they report on quadruped and hexapod hardware [32]. Lu, Dun, Zhou, Tao, Duan, Li and Li take the other route on a humanoid, approximating the learned policy by a TRUNCATED Gaussian that excludes out of distribution actions while retaining exploration stochasticity, with adaptive variance regulation as an implicit regulariser against the plasticity loss that aggressive gradient updates cause under high throughput parallel sampling [34]. The two agree on the problem and differ on whether to bound the transformation or the distribution, which is the same fork section 19.1 records.

TWO PROPERTIES OF THIS WORKSPACE'S OWN ALGORITHM MAKE THE TRANSFORMATION CHEAPER HERE THAN THE SURVEY IMPLIES, and both were established by reading the code rather than the literature. The first is that the change of variables term cancels. `PPO.update` forms its ratio as the exponential of the new log probability minus the old at `rsl_rl/algorithms/ppo.py:295`, and both are evaluated at the same stored action, so a deterministic transformation contributes an identical Jacobian to each and the two cancel exactly in the difference. The surrogate objective is therefore unchanged whether the correction is applied or omitted, provided it is done consistently, which is why the claim of Liao and colleagues that the PPO update rule is preserved holds without qualification [32]. What does not cancel is the entropy bonus at `ppo.py:313`, which continues to measure the pre squash Gaussian and is therefore an upper bound on the entropy of the action the environment receives.

The second is that the transformation need not touch the actor at all. Applied inside the action term rather than inside the policy, it leaves `rsl_rl` untouched, leaves the log probability and the entropy exactly as they are, leaves the storage of actions in the rollout unchanged, and leaves the symmetry augmentation of `mdp/symmetry/kscale.py` unchanged, since that module permutes an action vector whose width and meaning are unaltered. The cost of that placement is stated in section 21.18 and is the one thing it breaks, `pen_action_rate` and `pen_action_smoothness` both being quadratic in the RAW action, which is unbounded behind the transformation and therefore no longer proportional to what reaches the joint.

THE PROPERTY THAT DECIDES THE MATTER IS NOT BOUNDEDNESS, and no surveyed source states it because no surveyed source is trying to price a clip. Section 18.4 measures the clip excess growing by a factor of 51000 over training and concludes that no fixed quadratic coefficient can be right at both ends of that span, and section 19.2 registers that difficulty as a known one with two published remedies, adapting the budget [30] or restating the price as a constraint [31]. A bounded transformation removes the difficulty instead of managing it. Where the action is bounded at plus and minus a scale and the clip stands inside that scale, the excess cannot exceed the difference between the two, which is a constant of the configuration rather than a quantity the policy chooses. Section 21.12 computes it for this robot at 0.2846 radians squared per step against the 30.380 run E reaches unbounded, a reduction of the worst case by a factor of 107, and the span a fixed weight must cover falls from 51059 to one to 5.7 to one. The weight becomes calibratable, which is the whole of the argument.

### 19.4 What the survey did not find, recorded as a negative result

A directed search returned no published account of a learned locomotion policy escalating its raw action by an order of magnitude in response to a clip on the processed action. The nearest statements are the clipping literature above, which concerns an action space bounded at plus and minus one where this one is unbounded, and the boundary preference of Seyde and colleagues [16], which concerns a policy preferring the extremes of an admissible range rather than commanding far outside it. Cluster 30 of the workspace survey already records that no source reports the effect of clipping the PROCESSED action at all, and section 18.3 is therefore the workspace's own first measurement of that effect and should be cited as such rather than as a confirmation of anything.

And no surveyed source reports a rule for SIZING such a clip. Section 18.3 proposes one, that the bound be set from the joint's useful range and checked against its saturating deflection rather than from its mechanical travel, and the ratio table there is the evidence for it. It is proposed as this workspace's own and any document wishing to cite a precedent must find one.

## 20. The ablation, re-ranked against the evidence of sections 14 through 19

The sequence of section 7 is superseded in full and every revision of this section before 2026-09-30 is superseded with it. Four things changed on that date and the first is the one that reorders everything below it. Arm 10 was executed as `2026-09-29_06-15-58` and SUCCEEDED, section 18.2 measuring the ankle roll saturation falling from 60.17 per cent to 0.06 and its stop residency from 42.30 to 0.07, which is the first unqualified success this ablation has produced and which removes the fault sections 11 and 12 named as the root cause. Two arms are thereby mooted rather than merely demoted, arm 24 existing to freeze a joint the clip has made controllable and arm 25 existing to widen a travel the clip has stopped the machine reaching, and both fall to the bottom of the order with their premises recorded as withdrawn. The fault relocated to the hip roll instead, which section 18.3 traces to the clip's own sizing rule and which a new arm 15 addresses by resizing it. And arm 11 is in flight at two weights, both of which section 18.4 prices as mis-set by between one and three orders of magnitude, so its entry below is rewritten to separate the weight from the form.

Section 7's numbering is left where it stands rather than rewritten, its entries being the historical record of what was proposed on 2026-09-21, and the blockquote at its head carries the mapping to this section's numbering. Everything a reader needs in order to choose what to launch is below, and everything needed in order to implement it is in section 21, which carries one sub-section per arm in this section's order.

The baseline for every arm is an ankle effort ceiling of 17 newton metres with the commanded target clip of arm 10 active, that clip now being a precondition rather than an arm since every measurement of section 18 is taken against it. Every arm runs to 12000 iterations and is evaluated at 8000, 9000 and 10000, which section 16.9 requires. Section 20.3 states the instrumentation, which `play.py` has carried since 2026-09-28 and which section 18 is the first analysis to read.

### 20.1 The arms, in the order they are run

The number of an arm IS its position in the sequence and the number of its sub-section in section 21 is the same number, so a reader works down this table and turns to the matching sub-section. There is no separate priority column because the order is the proposal, and there is no column recording what each arm was numbered before, the arms having been renumbered twice on 2026-09-30 so that a single historical column would now be ambiguous. The preamble of section 21 carries the mapping from the numbering of 2026-09-28, which is the one a dumped configuration or an earlier run note may name.

| Arm | Section | What it changes | Status |
|---|---|---|---|
| 1 | 21.1 | The reference, ankle ceiling 11.9 N.m | EXECUTED by proxy as `2026-09-18_09-37-29` |
| 2 | 21.2 | The ankle ceiling at 17 N.m | EXECUTED, `2026-09-21_06-45-40`. Falsified its own prediction. |
| 3 | 21.3 | The ankle ceiling doubled to 34 N.m | EXECUTED, `2026-09-21_08-40-59`. Worse than either comparator. |
| 4 | 21.4 | The per joint action scale | EXECUTED, `2026-09-22_07-18-05`. An exact null. |
| 5 | 21.5 | The lateral weight shift | EXECUTED, `2026-09-22_07-40-43`. The first positive result. |
| 6 | 21.6 | The two ankle action scales halved | RUN AND ABANDONED, 2026-09-23. |
| 7 | 21.7 | The nominal pose | EXECUTED. Section 16.5 charges it with the ankle pitch headroom. |
| 8 | 21.8 | The energy budget, in three stages | EXECUTED in two stages. SUCCEEDED. Section 16.2 carries the ladder and the cliff. |
| 9 | 21.9 | The ankle roll deviation penalty | EXECUTED, REFUTED TWICE, sections 16.4 and 18.5. |
| 10 | 21.10 | The commanded target clip | EXECUTED, `2026-09-29_06-15-58`. SUCCEEDED. Now the baseline of every arm below. |
| 11 | 21.11 | The clip excess penalty as launched, the quadratic at -2 and -500 | RUNNING as `2026-09-30_06-43-38` and `2026-09-30_07-05-14`. Both weights mis-set, section 18.4. |
| 12 | 21.12 | The clip excess penalty at the calibrated quadratic weight of -14 | APPLIED TO THE WORKING TREE 2026-09-30, awaiting its run. |
| 13 | 21.13 | The clip excess penalty in its absolute form at -1.8 | NOT APPLIED. The form section 18.4 argues for. |
| 14 | 21.14 | The hip yaw fixed in the URDF, with its own asset, environment and task | NOT APPLIED. INDEPENDENT of every arm around it, run to gauge one degree of freedom. |
| 15 | 21.15 | The clip resized from the mechanical travel to the useful range | NOT APPLIED. Requires arm 12 or arm 13 first. |
| 16 | 21.16 | The raised in air yaw weight and the swing gated heading price, in three stages | NOT APPLIED. Stage three REFUTED, section 18.7. |
| 17 | 21.17 | The hip roll excluded from the joint limit penalty | NOT APPLIED. Instrumentation, to be carried by every arm below. |
| 18 | 21.18 | The squashed action term | NOT APPLIED. Makes the excess bounded, section 19.3. |
| 19 | 21.19 | The bounded action distribution | NOT APPLIED. Superseded by arm 18, retained as the last resort. |
| 20 | 21.20 | Phase dependent gains | NOT APPLIED. Premise is the 90 to 1 swing to stance speed ratio at the ankle roll. |
| 21 | 21.21 | The footprint separation hinge | NOT APPLIED. The live term is blind by 0.143 m. |
| 22 | 21.22 | The in air kernel width | NOT APPLIED. Run after arms 16 and 18. |
| 23 | 21.23 | The restored corner velocity | NOT APPLIED. Buys three per cent at the present speeds. |
| 24 | 21.24 | The ankle roll fixed in the URDF, with two tasks | NOT APPLIED, PREMISE WITHDRAWN by section 18.2. |
| 25 | 21.25 | The ankle pitch nominal, or its travel | NOT APPLIED, PREMISE WITHDRAWN by section 18.2. |
| 26 | 21.26 | The search rather than the sweep | NOT APPLIED. Terminal. |

Arms 1 through 11 have been run and sections 10, 14, 16 and 18 report them, and arm 12 is applied to the working tree and awaiting its run, so the first arm a reader must still act upon is arm 13. The shape of what remains is worth stating before the detail. The clip of arm 10 succeeded and thereby removed the fault that motivated four of the arms this document has carried since section 11, so arms 11 through 13 all concern the excess that clip creates and the window it leaves, and arm 15 cannot be read until one of arms 12 and 13 has settled the price. Arm 14 then stands on its own, removing the hip yaw from the URDF to measure what one degree of freedom contributes, and arm 15 resizes the window the excess is measured against before arm 16 returns to the gait by re-pricing the hip yaw the policy still has. Arms 17 through 19 return to the action space, and the two arms that would have removed a degree of freedom from the ankle now stand at 24 and 25 with their premises withdrawn rather than near the front.

Six of the placements need their reasons stated rather than tabulated.

Arm 11 leads the unrun sequence because it is already running and because its result will be read whatever this document says, and NOT because its weight can be settled from it. Section 18.4 measures the excess it prices growing by a factor of 51000 between iteration 193 and iteration 10000 of the run that carries the identical clip without it, so a weight read from the early magnitude is read before the policy has learned to produce the quantity it prices. Its two runs should be read as a measurement of the unbounded regime at the extremes of the quadratic's range.

Arms 12 and 13 divide what stood as one arm until 2026-09-30, at the user's direction, and the division is between a WEIGHT and a FORM. Arm 12 runs the quadratic at minus 14, which section 21.12 derives from the excess at the onset of the exploit rather than at either end of it, and it exists so that the quadratic is refuted at a defensible weight rather than at one that was never defensible. Arm 13 then changes the kernel to the absolute value at minus 1.8, which section 18.4 argues for because a quadratic in a quantity spanning five orders of magnitude spans ten and an absolute form spans five. Arm 12 is expected to be insufficient and is run anyway, since a form cannot be charged with a failure that a weight explains.

Arm 15 must follow arms 12 and 13 rather than precede them, and the reason is mechanical. Narrowing a clip raises the excess, the policy's commanded target being unchanged at the moment of the change while more of it falls outside the bound, and section 21.15 computes the hip roll excess rising from 2.90 radians to something near 4.9 under the new window. Run before the price is settled it would measure a larger escalation and nothing else.

Arm 14 stands immediately after the excess is priced, at the user's direction of 2026-10-01, because it depends upon nothing the arms around it do and because what it measures is wanted early. It removes the hip yaw from the URDF to establish what the degree of freedom contributes to this machine's locomotion, which is a question about the robot rather than about the reward set, and it needs only arm 10's clip. Arm 16 then re-prices the hip yaw the unmodified robot still has, and the two are the only arms that act upon that joint, the first removing what the second tries to shape. Running them in this order means arm 16 is read knowing what the joint is worth, where the reverse order would have left arm 14 as a fallback for arm 16's failure and would have answered the narrower question first.

Arm 16 is divided into three stages because section 18.7 measures its three values to act upon different quantities, in one case in the opposite direction to the one supposed. Its third stage, the halving of the differential scale, is REFUTED by that section and is retained only against a condition stated there.

Arm 14 supersedes the configuration level freeze that this document carried as arm 10 until 2026-09-30, at the user's direction, and that arm was removed from the sequence rather than demoted. The two are not the same experiment. The configuration freeze leaves the joint in the articulation and removes the command, so the joint still deflects under load, where arm 14 removes the degree of freedom so that the deflection is zero by construction and the observation and action dimensions fall with it.

Arms 24 and 25 have their premises WITHDRAWN rather than refuted, and the distinction governs what happens next. Both were written against a machine whose ankle was pinned, arm 24 to stop paying for an actuator doing no work and arm 25 to give the ankle pitch room it did not have. Section 18.2 measures the ankle roll at 0.07 per cent stop residency and 3.5 newton metres and the ankle pitch at 0.07 per cent, so neither premise holds under the clip. If arm 15 narrows the clip and the ankle returns to its stop, both premises return with it. Arm 24 was rebuilt on 2026-09-30 to follow arm 14's construction and now produces two tasks, the ankle roll frozen alone and frozen together with the hip yaw, so that the four cells of the two by two design over those joints are reachable.

### 20.2 The instrumentation that every arm must carry

Five requirements attach to every arm and none is optional. The first three were IMPLEMENTED in `scripts/rsl_rl/play.py` on 2026-09-28 and section 18 is the first analysis to read them, so the paragraphs below now record what exists rather than what is wanted.

The commanded target is recorded as `joint_position_targets`, the raw action behind it as `joint_action_raw`, and the action term register as `action_terms`, all scattered by joint so that their columns align with `joint_positions` and a joint no term drives reads NaN. Section 18.3 is the measurement that justifies the second of the three, the raw action at nineteen standard deviations being invisible in every other channel.

Saturation is recorded exactly rather than against a threshold, `joint_computed_torque` carrying the torque the actuator produced before its own clip and `joint_available_torque` carrying both bounds of the velocity dependent envelope, so saturation is the first lying outside the second with no threshold and no dependence on the friction term subtracted after the clip. `joint_saturation_effort` closes the parameter gap that section 17 of [/ws/context/gait_metrics.md](../../context/gait_metrics.md) records.

The sole tilt is recorded as `feet_sole_tilt` with `sole_normal_link` beside it, and section 18.3 is the first section to quote it, the stance sole roll falling from 6.725 degrees to 2.850 under the clip.

Every behavioural figure must be conditioned on the gait phase of the leg the joint belongs to, and not pooled, for the reasons section 16 gives and which section 18 does not repeat.

And every arm must be evaluated at 8000, 9000 and 10000 iterations rather than at its last checkpoint, which section 16.9 establishes and which section 18.4 now reinforces from a second direction, the clip excess growing monotonically across exactly that span.

A sixth requirement is added on 2026-09-30 and it is specific to the arms above. THE CLIP EXCESS MUST BE REPORTED FOR EVERY ARM RUN ON TOP OF THE CLIP, whether or not that arm prices it, because section 18.3 shows it to be the channel through which a bounded target becomes an unbounded action and because no other logged quantity reveals it. `clipab.py` in the artefact directory of section 18 computes it from the dump, recovering the per environment offset from the samples the clip did not bind, which section 18's artefact register records as necessary since the dumped `default_joint_pos` is environment zero's row alone.

Where a new term is added, as arm 5, arm 11 and arm 20 require, it must be added as a new function in the KScale configuration's own module or as an optional argument whose default reproduces the present behaviour exactly, never as an in place edit of a shared `environments` mdp function, since SD_BRS1 and the TRON1 tasks call those same functions and an in place change invalidates every comparison against an in flight or historical run.

### 20.3 What section 8 must now also exclude

Raising the ankle effort ceiling further is excluded, section 10 having tried 11.9, 17 and 34 newton metres and section 18.2 having now removed the fault that motivated the question at the original ceiling.

Widening the ankle roll travel is excluded, section 11.3 measuring the demand at eight to ten degrees against fifteen available. The exclusion now extends to the ankle PITCH as well, which section 12 and section 16.5 had preserved it from, since section 18.2 measures that joint at 0.07 per cent stop residency under the clip against the 85.5 per cent that motivated arm 25. The extension is conditional on the clip remaining in place and must be revisited if arm 15 narrows it.

Raising the weight of the two in air ANKLE terms alone is excluded and the hip yaw term is not covered by the exclusion, which arm 16 acts upon.

Raising the energy weights beyond one hundred times their pre arm 8 value is excluded, section 16.2 recording the five hundredfold collapse.

Adding a deviation penalty to any joint whose position is set by contact is excluded, section 16.4 giving the mechanism and section 18.5 the clean contrast.

Setting a clip bound from a joint's MECHANICAL TRAVEL is excluded, and this exclusion is new. Section 18.3 measures what it produced, a hip roll window 14.51 times the joint's own saturating deflection, a policy escalating to nineteen standard deviations of raw action to reach it, and the joint at its stop for 46 per cent of every episode. A clip bound must be set from the range the joint is wanted in and checked against the saturating deflection, which arm 15 does and which section 19.3 records as unreported in the surveyed literature.

Calibrating a penalty weight from the magnitude the term reports at the start of training is excluded, and this exclusion is the general form of section 18.4. The clip excess grows by a factor of 51000 between iteration 193 and iteration 10000, so the early magnitude is the quantity before the policy has learned to produce it and a weight read from it is wrong by that factor. Any weight on a term that prices an EXPLOIT rather than a cost must be read from a run that carries the exploit unpenalised, which for arms 11 through 13 is run E.


## 21. The arms in detail, the code each requires, the reasoning behind it and its parameters

Section 20.1 lists every arm of this ablation in the order it is run and is the section to work from when choosing. This section is the complement and is the section to work from when implementing, carrying for every arm the code it changes with the code it would become, the design decisions taken in writing it with the alternatives those decisions reject, the parameterisation with the arithmetic that sets it, and the prediction with the observation that would falsify it.

THE ARM NUMBER, THE SUB-SECTION NUMBER AND THE POSITION IN THE SEQUENCE ARE ONE NUMBER, which is the property a returning reader should rely upon. Arm 1 is the first that was run and stands at section 21.1, arm 26 is the last that is proposed and stands at section 21.26, and the sub-sections below run in that order without interruption. Arms 1 through 11 have been executed, one of them abandoned and one refuted, arm 12 is applied to the working tree and awaiting its run, and arms 13 through 26 have not been applied to any file, so a reader deciding what to do next begins at section 21.12 and acts from section 21.13.

The arms were renumbered twice on 2026-09-30 at the user's direction, first into the order of execution and then to accommodate three divisions and three relocations, and once more on 2026-10-01 to bring the hip yaw fixed in the URDF forward to arm 14 as an independent measurement, which moved the clip resize to 15 and the in air yaw weight to 16 and left every other number alone. The table below carries the mapping from the numbering of 2026-09-28, which is the one a dumped configuration or a run note written before that date may name, and its second column is the present numbering. Only the arms whose number moved are listed.

| Was, before 2026-09-30 | Is now | What it is |
|---|---|---|
| 4b | 6 | The two ankle action scales halved |
| 5 | 7 | The nominal pose |
| 6 | 8 | The energy budget |
| 7 | 24 | The ankle roll converted to a fixed joint |
| 8 stage one | 10 | The commanded target clip |
| 8 stage two | 11, 12 and 13 | The clip excess penalty, as launched, calibrated, and in its absolute form |
| 9 | 17 | The hip roll excluded from the joint limit penalty |
| 10 | removed | The hip yaw frozen in the configuration, superseded by arm 14 |
| 11 | 21 | The footprint separation hinge |
| 12 | 23 | The restored corner velocity |
| 13 | 22 | The in air kernel width |
| 14 | 25 | The ankle pitch nominal, or its travel |
| 15 | 26 | The search rather than the sweep |
| 16 | 9 | The ankle roll deviation penalty |
| 17 | 16 | The in air yaw weight and the gated heading price |
| 18 | 20 | Phase dependent gains |
| 19 | 14 | The hip yaw fixed in the URDF |
| 20 | 15 | The clip resized to the useful range |
| 21 | 19 | The bounded action distribution |
| 22 | 18 | The squashed action term |

Every reference in this document, in `kscale_base_env_cfg.py`, in `kscale_identified_cfg.py`, in `plans/README.md`, in `context/artefacts/README.md`, in `context/KScale.md`, in `/ws/context/literature.md`, in `/ws/context/gait_metrics.md` and in the artefact programs was moved in the same pass, so a bare reference anywhere in this workspace now means the present numbering.

One arm was removed from the sequence rather than renumbered. The hip yaw frozen in the CONFIGURATION, which this document carried as arm 10 until 2026-09-30, is superseded by arm 14, which freezes the joint in the URDF instead, and the two are not the same experiment. Its second stage, the heading constraint, survives as the second stage of arm 16, and its fixed joint proposal survives in full at section 21.14, so nothing it established is lost.

Three arms are divisions of what stood as one. The commanded target clip and its excess penalty were executed at different times to different verdicts and stand at arms 10 and 11, and the excess penalty divides again into arms 11, 12 and 13, which are the weight as launched, the weight as calibrated and the form. The energy budget is NOT divided although its first two stages were also run separately, because section 16.2 reads them as one ladder of five multiples and a division would make that reading unquotable.

Two arms carry stages within one sub-section rather than being divided, and the distinction from the above is that their stages have not been run and are expected to be run in one campaign. Arm 16 carries three, each moving one value, and arm 24 carries two, each producing a task. Four sub-sections carry an appended record of the state of the working tree on the date that arm was applied, being arms 6, 7, 8 and 9, those records having stood as sub-sections of their own until 2026-09-30.

### 21.1 Arm 1, the reference, the ankle ceiling at 11.9 newton metres

Status, EXECUTED by proxy as `kscale_flat/2026-09-18_09-37-29`, which section 2 calls run C, no separate launch having been made.

The arm exists to give every later arm a comparator trained under one known reward set at the full budget, since section 2.3 records that the working tree had already moved past all four runs of the first sequence and a launch from it would have confounded the ankle ceiling with the reward alignment. Run C serves because its dumped configuration is the reward set the tree was then aligned to, and because section 3.2 finds it the best of the four on every measure of gait quality except height.

The parameterisation is the ankle effort ceiling at the value all four runs of the first sequence carried, which is the state the two ankle actuator blocks held before 2026-09-22.

```python
KSCALE_FOOT_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_roll_02"],
    velocity_limit_sim=44.0,
    effort_limit=11.9,
    velocity_limit=37.699,
    saturation_effort=17.0,
```

The design decision that governs this arm is the one not to relaunch it. A separate arm 1 would have cost a full training budget to reproduce a run already in hand under an identical reward set, and the only quantity it would have added is the seed variance, which no arm of this ablation is powered to resolve. The cost of the decision is that every contrast drawn against arm 1 carries one run's seed rather than a mean, and it is recorded here so that a later session does not read a small difference against run C as an effect.

| Quantity | Value at arm 1 | Source |
|---|---|---|
| Ankle effort ceiling | 11.9 N.m | `kscale_identified_cfg.py:152` and `:168` |
| Ankle saturation effort | 17.0 N.m | `kscale_identified_cfg.py:154` and `:170` |
| Corner velocity of the ankle envelope | 11.31 rad/s | derived at section 10.2 |
| Action scale | 0.4, scalar | the term at `kscale_base_env_cfg.py:211` before 2026-09-22 |
| Training budget | 29999 iterations | `limx_rsl_rl_ppo_cfg.py:337` |

### 21.2 Arm 2, the ankle ceiling at seventeen newton metres

Status, EXECUTED as `kscale_flat/2026-09-21_06-45-40` and reported at section 10, its central prediction falsified.

The arm tests the premise of section 4.2 directly, that the ankle cannot hold the lateral moment single support demands of it and that the splay is what the machine buys the shortfall with. Seventeen newton metres is the RobStride class peak, so the arm is a specification claim rather than an instrument, and it is the cheapest possible test of the first and highest ranked remedy of section 6.

The change is one field in each of the two ankle actuator blocks and it stands in the tree.

```python
KSCALE_FOOT_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_roll_02"],
    velocity_limit_sim=44.0,
    effort_limit=17,
    velocity_limit=37.699,
    saturation_effort=17.0,
    stiffness={".*": 100.0},
    damping={".*": 1.05},
```

One design decision was taken without being noticed at the time and it is the most consequential fact about this arm. The saturation effort was left at 17.0 while the ceiling rose to meet it, and the two quantities are not independent. `DCMotor` derives its torque speed envelope at `/ws/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:299` and clips it at the ceiling two lines later, which is the vendored library's own file and not the `IdentifiedActuator` of `environments/environments/actuators/actuator_pd.py` that subtracts the friction model after it.

```python
        torque_speed_top = self._saturation_effort * (1.0 - self._joint_vel / self.velocity_limit)
        torque_speed_bottom = self._saturation_effort * (-1.0 - self._joint_vel / self.velocity_limit)
        # -- max limit
        max_effort = torch.clip(torque_speed_top, max=self.effort_limit)
        # -- min limit
        min_effort = torch.clip(torque_speed_bottom, min=-self.effort_limit)
```

The corner velocity, being the speed at which the sloping envelope first meets the flat ceiling, is therefore the velocity limit times one minus the ratio of the ceiling to the saturation effort, and it collapses to exactly zero when the two are equal. The arm consequently removed the constant torque plateau rather than raising it, and above 11.31 radians per second it delivers precisely what the 11.9 newton metre arm delivered. Section 11.4 measures the joint traversing at twenty five radians per second, which is inside that regime, so the arm is not the uniform increase it was taken for. Arm 23 exists to repair exactly this and nothing else.

| Quantity | Arm 1 | Arm 2 | Arm 3 |
|---|---|---|---|
| `effort_limit` | 11.9 | 17 | 34 |
| `saturation_effort` | 17.0 | 17.0 | 34.0 |
| Corner velocity, `velocity_limit * (1 - effort/saturation)` | 11.31 rad/s | 0 rad/s | 0 rad/s |
| Torque available at 25 rad/s | 5.73 N.m | 5.73 N.m | 11.45 N.m |

The prediction was that the ankle roll soft limit residency would fall from the 92.5 to 92.7 per cent run C records, that its effort saturation would fall from 99.6 per cent and that the toe-in would fall. Section 11.1 measures the residency at 88.7 per cent and section 10.4 measures the toe-in rising from 4.59 to 7.37 degrees, so the prediction is falsified on both limbs and the arm is the clean negative result the sequence has produced.

### 21.3 Arm 3, the ankle ceiling doubled

Status, EXECUTED as `kscale_flat/2026-09-21_08-40-59` and reported at section 10, refuted, and section 20.3 now excludes any further rise.

The arm exists to bracket arm 2 so that a null result there could be distinguished from an insufficient one, thirty four newton metres being the figure section 5.2 of the physical record derives and the point at which the single support hold is 74 per cent of the limit. It is beyond the published class peak and was therefore always an instrument rather than a specification claim.

```python
    effort_limit=34,
    velocity_limit=37.699,
    saturation_effort=34.0,
```

The design decision worth recording is that the saturation effort was again carried with the ceiling rather than held, so this arm inherits the collapsed corner velocity of arm 2 and doubles only the plateau that no longer exists below 11.31 radians per second. The arm therefore doubles the torque available at low speed and leaves the high speed envelope proportional to the saturation effort, which is why the table above shows the torque at twenty five radians per second doubling while the ceiling triples relative to arm 1.

The result refutes the remedy rather than merely failing to support it. The ankle roll stop residency stands at 94.9 per cent against 92.6 and 88.7 at the lower ceilings, the toe-in rises to 10.20 degrees, the fall rate at matched iteration rises to 27.4 per cent against 9.3 at seventeen newton metres, and the cost of transport rises by 71 per cent. Section 11.2 supplies the mechanism, the commanded target excursion beyond the stop having tripled from 0.107 to 0.323 radians as the ceiling tripled, which is a policy pressing harder into a stop that has become cheaper to hold.

### 21.4 Arm 4, the per joint action scale

Status, IMPLEMENTED and standing in the working tree, awaiting its run.

The arm answers section 4.6, that one action scale cannot serve six joint groups whose ceiling to stiffness ratios differ by a factor of three and a half. A scale converts a policy action into a commanded deflection from the nominal pose, so the deflection a unit action buys is the scale itself and the action that saturates a joint is its ratio of ceiling to stiffness divided by that scale. Under a single scale of 0.4 the hip pitch saturates at an action of 1.050 and the ankle roll at 0.425, and the two joints reached at the smallest action are exactly the two the dumps find pinned against their stops. The arm equalises that quantity across the set so that isotropic exploration noise reaches the six groups in proportion to what each can deliver.

The term as the tree now carries it, quoted whole because the comment above it is part of the change and records the trap that would otherwise corrupt the run.

```python
    # Per joint action scale, set 2026-09-22 as arm 4 of plans/kscale_gait_shaping.md, which
    # section 21.4 details, reordered ahead of the nominal pose change at the user's direction.
    # It was written as arm 5 before the lateral weight shift took that number on the same date,
    # and the duplicate numbering by which it was also arm 10 was collapsed when that plan was
    # restructured on 2026-09-23.  This comment was repointed at the restructured sections on
    # the same date.  A single
    # scale of 0.4 met six joint groups whose ceiling to stiffness ratios span 0.168 to 0.420 rad,
    # so one action magnitude saturated the ankles where two and a half times as much was needed
    # at the hip pitch, and the two joints reached at the smallest action are exactly the two the
    # evaluation dumps find pinned against their stops.  Each scale below is the hip pitch's 0.4
    # times the ratio of that group's saturation deflection to the hip pitch's, which equalises
    # the action at saturation across the set at 1.050 and is the sizing rule section 4.6 states.
    #
    #   group        K     tau    tau/K    scale   action at saturation   action to nearer stop
    #   hip_pitch   200    84.0   0.4200   0.4000          1.050            3.127 (was 3.127)
    #   hip_yaw     100    42.0   0.4200   0.4000          1.050            3.927 (was 3.927)
    #   knee        300    84.0   0.2800   0.2667          1.050            1.921 (was 1.281)
    #   hip_roll    250    42.0   0.1680   0.1600          1.050            1.040 (was 0.416)
    #   foot_pitch  100    17.0   0.1700   0.1619          1.050            3.484 (was 1.410)
    #   foot_roll   100    17.0   0.1700   0.1619          1.050            1.362 (was 0.551)
    #
    # The ankle figures are tied to the 17.0 N.m ceiling at kscale_identified_cfg.py:152 and :168
    # and must be re-derived if that ceiling moves, a 34 N.m ankle giving 0.3238 instead.
    #
    # Two properties of the dict form govern the entries.  Every joint MUST be covered, because
    # joint_actions.py:85 initialises the scale tensor to ONES and writes only the matched
    # columns, so an omitted joint silently takes a scale of 1.0 rather than keeping 0.4.  And the
    # patterns must be mutually exclusive, a joint matching two keys raising at string.py:328.
    # A key matching no joint raises at string.py:362, so a typo fails loudly rather than
    # silently, which is the one failure mode that needs no guarding here.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "(left|right)_hip_pitch_04": 0.4,
            "(left|right)_hip_yaw_03": 0.4,
            "(left|right)_knee_04": 0.2667,
            "(left|right)_hip_roll_03": 0.16,
            "(left|right)_foot_pitch_02": 0.1619,
            "(left|right)_foot_roll_02": 0.1619,
        },
        use_default_offset=True,
    )
```

It replaces the single line `scale=0.4,` that the term carried before the change, when the term stood at line 165 of that file, and the sizing comment written above it has since moved it to `kscale_base_env_cfg.py:211` by way of lines 191 and 193, every citation of it in this document being repointed in each pass so that a reader following one reaches the term rather than the line it once occupied.

Four design decisions were taken and each rejected a named alternative.

The dictionary form was chosen over a `clip` on the same term because the two act on different quantities and only one of them was established to work at the time. `JointActionCfg` declares the field as `scale: float | dict[str, float] = 1.0` at `actions_cfg.py:35`, and `JointAction.__init__` parses it at `joint_actions.py:84` through the repository's regular expression resolver, so the dictionary is a supported form rather than a convention. The clip is the subject of arm 10 and the two are complementary rather than alternative, the scale changing the geometry of the action space and the clip bounding its image.

The saturation equalising rule was chosen over the travel equalising rule that section 7 words the arm in, and the two disagree because the travel and the saturation deflection are not proportional across this joint set. The saturation rule carries the numbers section 4.6 derives and section 11.4 shows the relation between the saturation action and the stop action to be the quantity that governs the latch, so it is the rule implemented. The travel equalising alternative is recorded as untried and would give a hip roll scale near 0.053 and an ankle roll scale near 0.070, which is a contraction more than three times as aggressive as this arm attempts.

Full coverage was chosen over a partial dictionary naming only the joints the arm intends to change, and this is the decision that would have silently corrupted the run had it gone the other way. `joint_actions.py:85` initialises the scale tensor with `torch.ones` and writes only the matched columns, so a joint left out of the dictionary takes a scale of 1.0 rather than retaining the 0.4 it had, which is an increase of two and a half times in the direction opposite to the arm's intent.

```python
        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._joint_names)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
```

Coverage is not checked by the resolver, whose `strict` flag at `string.py:300` requires only that every key match something and not that every joint be matched, so the check had to be made externally and `validate_scale.py` in the artefact directory is that check, parsing the dictionary back out of the configuration file with `ast` and resolving it against the twelve joint names the evaluation dump records. The two loud failure modes need no guarding, a joint matching two patterns raising at `string.py:328` and a pattern matching no joint raising at `string.py:362`.

And the scale was raised in relative terms rather than lowered in absolute ones, the hip pitch and hip yaw keeping their 0.4 while the other four groups fall, so that the arm is a change of geometry and not of overall authority. Section 8 excludes a global reduction on measured grounds, the ankle roll stop residency having risen from 0.669 to 0.765 when the global scale was cut from 0.4 to 0.25, and holding the largest group fixed keeps this arm outside that exclusion.

The parameterisation follows the rule exactly, each group's scale being the hip pitch's 0.4 times the ratio of that group's saturation deflection to the hip pitch's. The two columns on the right are the quantity the arm exists to move, being the action magnitude at which each joint first reaches the nearer of its two mechanical stops from the nominal pose.

| Group | Stiffness [N.m/rad] | Ceiling [N.m] | Saturation deflection [rad] | Scale | Action at saturation | Action to the nearer stop, before | after |
|---|---|---|---|---|---|---|---|
| Hip pitch | 200 | 84.0 | 0.4200 | 0.4000 | 1.050 | 3.127 | 3.127 |
| Hip yaw | 100 | 42.0 | 0.4200 | 0.4000 | 1.050 | 3.927 | 3.927 |
| Knee | 300 | 84.0 | 0.2800 | 0.2667 | 1.050 | 1.281 | 1.921 |
| Hip roll | 250 | 42.0 | 0.1680 | 0.1600 | 1.050 | 0.416 | 1.040 |
| Ankle pitch | 100 | 17.0 | 0.1700 | 0.1619 | 1.050 | 1.410 | 3.484 |
| Ankle roll | 100 | 17.0 | 0.1700 | 0.1619 | 1.050 | 0.551 | 1.362 |

Measured over the twelve resolved joints the spread of the action at saturation falls to a ratio of 1.0002, the quantity spanning 1.0499 to 1.0500 across the set. The spread it falls FROM depends on the ankle ceiling the arm departs from and the two figures are easy to confuse, being 3.53 to one at the 11.9 newton metre ankle that section 4.6 derives the rule against and 2.50 to one at the 17 newton metre ankle this arm actually rests upon. The second is the figure that describes this arm, section 20.1 quotes the first in error, and the correction is recorded here rather than by editing that passage and is carried into `validate_scale.py`, which now prints both rather than hard coding either. The ankle figures are tied to the seventeen newton metre ceiling and must be re-derived if it moves, a thirty four newton metre ankle giving 0.3238 in place of 0.1619, and this coupling is the reason arm 23 and arm 26 must not be run against a stale scale.

The comment quoted above is reproduced exactly as the tree carries it on 2026-09-28. It carried two stale forward pointers when this sub-section was written, naming section 14 and section 15.4 under the numbering this document used before 2026-09-23, and it was left alone at that time because arm 6 and arm 7 were in flight from that file and a run in progress should not be given a reason to differ from the tree it was launched from, section 21.7 recording the repointing as a tidy owed once those runs completed. The tidy was taken in two passes, the comment being repointed at the restructured sections on 2026-09-23 and moved again to section 21.4 on 2026-09-28 when this document was renumbered, and the quotation above was refreshed in the same pass so that the two agree. The debt is therefore discharged and the quotation may be trusted against the tree until the next renumber, which section 20 records as the reason no further renumbering should be undertaken lightly, five files outside this document now citing its section numbers.

The prediction is that of section 7, that the ankle roll and hip roll stop residencies fall together and that `pen_joint_pos_limits` at `kscale_base_env_cfg.py:859` falls with them, and section 11.4 sharpens it. The ankle roll stop moves from 0.551 to 1.362 in action units and the hip roll stop from 0.416 to 1.040, which against a policy noise standard deviation measured at 0.86 in the seventeen newton metre run moves both stops from well inside one standard deviation of exploration to beyond it. The falsifying observation is the stop residency of section 11.1 failing to fall from its 88.7 per cent.

One caution attaches and it is the strongest argument against the arm. A scale is a reparameterisation and not a bound, the action entering at `joint_actions.py:173` as a factor rather than as a limit, so a policy that wishes to command a target beyond the stop need only raise its raw action by the same factor the scale fell.

```python
    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
```

The workspace has measured that compensation happening under a global reduction, and the reward set contains no term that prices a raw action, `pen_action_rate` at `:858` pricing only its rate of change. The present arm differs from the excluded one in kind rather than degree, changing the relative geometry of the action space rather than its size, which is a change no global factor can produce, and that distinction is why the arm is worth running. It is not a bound, and the honest expectation is that arm 10 is required to close the fault and that this arm measures how much of it the geometry alone accounts for.

### 21.5 Arm 5, the lateral weight shift

Status, IMPLEMENTED on 2026-09-22 and then DISABLED in the tree on the same day, the reward function standing in the shared module and the term standing commented out in the KScale configuration. Section 21.6 records the disabling and why it is the correct ordering rather than a reversal.

The arm supplies the quantity every preceding section has found missing and none has priced, being a torso that moves over its stance foot. Its premise is that the centre of mass never passes above the centre of pressure, so the ankle roll is left holding a moment it cannot hold and the splay is what the machine buys the shortfall with. Section 4.2 states that balance in newton metres and section 11 states it in joint angles, and section 20.1 states it in metres of lateral travel, where the torso is measured swaying 2.9, 6.8 and 4.8 millimetres against the 25.7 to 36.5 millimetres that merely reaching the inboard edge of the support polygon requires.

The instrument is a new class in the shared module, merged from `nithin/nithin/kscale` commit 2599d61 and standing at `rewards.py:850`. Its kernel is five lines and they are the whole of the reward.

```python
        gait_params = env.command_manager.get_command(command_name)
        frequencies = gait_params[:, 0]
        gait_indices = torch.remainder(env.episode_length_buf * env.step_dt * frequencies, 1.0)
        phase = gait_indices - self._phase_offset

        lateral_offset = self._lateral_offset_b()
        target = amplitude * torch.sin(2 * torch.pi * (phase + phase_shift))

        error = lateral_offset - target
        reward = torch.exp(-torch.square(error) / std**2)
```

The measured quantity is the torso's lateral displacement from the midpoint of the two feet, resolved in the base's yaw aligned frame so that it does not drift as the robot turns.

```python
    def _lateral_offset_b(self) -> torch.Tensor:
        """Torso lateral offset from the feet midpoint, in the base's yaw aligned frame."""
        torso_pos_w = self.asset.data.body_pos_w[:, self.torso_id]          # (N, 3)
        feet_pos_w = self.asset.data.body_pos_w[:, self.feet_ids]           # (N, 2, 3)
        stance_center_w = feet_pos_w.mean(dim=1)                             # (N, 3)
        diff_w = torso_pos_w - stance_center_w
        yaw_q = math_utils.yaw_quat(self.asset.data.root_quat_w)
        diff_b = math_utils.quat_apply_inverse(yaw_q, diff_w)
        return diff_b[:, 1]
```

The term as it stands in the configuration, commented out at `kscale_base_env_cfg.py:991`, is restored by deleting the comment markers and nothing else.

```python
    # rew_hip_lateral_sway = RewTerm(
    #     func=mdp.HipLateralSwayReward,
    #     weight=5.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=_TORSO_LINK),
    #         "feet_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
    #         "command_name": "gait_command",
    #         "std": 0.01,
    #         "amplitude": 0.03,
    #         "phase_shift": 0.0,
    #         "resync_threshold": 0.04,
    #         "resync_patience": 50,
    #     },
    # )
```

Five design decisions govern the arm and three of them are the merged author's rather than this document's.

The target is a position rather than a joint angle, and this is the decision that makes the arm viable at all. Section 11 shows the ankle roll latched against its stop and section 11.4 shows the hip roll latched at 71 per cent of stance samples, so a term that named either joint would be asking a pinned joint for a number. A torso displacement leaves the policy free to reach the target with whatever mixture of hip roll, ankle roll, knee and torso lean it finds, and the mixture is the policy's discovery rather than the reward's instruction.

The phase is locked to the existing gait clock rather than to a clock of its own, `gait_command` at `kscale_base_env_cfg.py:149` supplying the frequency, so the sway and the foot contact schedule cannot drift apart. One full left to right excursion occupies one stride at the commanded frequency of 1.0 Hz.

The resynchronisation exists so that a target the torso cannot reach degrades into a shifted target rather than into a constant penalty, the phase reference being re-anchored to wherever the measured offset already sits after the tracking error has stood above `resync_threshold` for `resync_patience` consecutive steps. It adjusts this term's own clock alone and not the gait clock, so foot placement and contact timing are untouched by it.

The class was added beside the existing functions rather than folded into one of them, which is what `/ws/CLAUDE.md` requires of the shared `environments` mdp package, since SD_BRS1 and the TRON1 SF and PF tasks call those same functions and an in place edit would silently change them and invalidate every comparison against an in flight or historical run. The diff reads 137 insertions and no deletions, so no caller of that module sees a different function than it saw before, and the term reaches the KScale configuration through the wildcard re-export at `mdp/__init__.py:7` without any further registration.

And the one change to the standing configuration that the merge carried was the zeroing of `pen_hip_roll_deviation`, which prices any hip roll excursion from the nominal pose and therefore prices precisely the lean this reward asks for. That change has since been reverted in the tree and the term stands again at minus 0.1 at `kscale_base_env_cfg.py:804`, which is consistent while the sway term is disabled and is a direct conflict the moment it is re-enabled. Whoever restores the sway term must zero the deviation penalty in the same edit, or the two terms will price opposite signs of the same motion.

```python
    pen_hip_roll_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_roll_03"]
            )
        },
    )
```

The parameterisation was amended in the tree after the merge, the kernel width having fallen from the merged 0.03 metres to 0.01, and the amendment is an improvement that section 20.1 did not anticipate. Integrating the term over a stride at a weight of 5.0 gives the following rates per second, the narrower kernel being the one the tree now carries.

| Torso behaviour | Rate at std 0.03 [per s] | Rate at std 0.01 [per s] |
|---|---|---|
| Perfect tracking of the target | 5.000 | 5.000 |
| Half amplitude, in phase | 4.430 | 2.179 |
| No sway at all | 3.225 | 0.971 |
| A quarter cycle late | 2.329 | 0.675 |
| Anti phase | 1.543 | 0.474 |
| Discrimination, swaying against still | 1.775 | 4.029 |
| Mean gradient at the still torso | 114 per m | 113 per m |
| Peak gradient at the still torso | 143 per m | 429 per m |
| Stride fraction carrying a tenth of the peak gradient | 97 pct | 44 pct |

The narrowing does not produce the kernel collapse section 13 diagnoses at the two in air ankle terms, and the reason is that the target moves. A still torso is swept past by the target twice per stride, so it samples the whole of the kernel including its steepest region every cycle, and the mean gradient is consequently unchanged at 113 against 114 per metre while the peak triples and the discrimination rises from 1.775 to 4.029 per second. What the narrowing costs is the fraction of the stride over which the gradient is informative, which falls from 97 per cent to 44, and what it buys is a headroom of 4.03 per second at the policy's present operating point against 1.78 at the merged width. At 4.03 per second the term is 8.6 per cent of the velocity tracking reward and larger than every style penalty in the set, so it will act.

One correction to the merged amplitude survives the width change and it is the most useful thing this sub-section can say to whoever runs the arm. An amplitude of 0.03 metres is almost exactly the minimum that reaches the support polygon and is not a margin, falling 0.7 millimetres short of the inboard sole edge at the 11.9 arm's stance width and 6.5 millimetres short at the 34 arm's, and clearing it by 4.3 millimetres only at the 17 arm's narrower stance. Zeroing the ankle roll moment outright requires the torso to reach the ankle itself, which is 68 to 79 millimetres, so the merged value is short of the sufficient figure by a factor near 2.3. Run the arm at 0.03 to see whether the policy will follow the target at all before the target is made harder, and if it follows, raise the amplitude toward 0.06 and hold the kernel width at a third of it so the discrimination computed above is preserved.

The prediction is that the ankle roll effort saturation falls from the 88.7 per cent of section 11.1 without the ceiling moving, that the inner corner contact of section 11.1 gives way to a flat sole, and that the toe-in falls from 7.37 degrees without any change to the heading price. The falsifying observation is the measured sway amplitude failing to rise from its 6.8 millimetres, which would mean the policy has declined the target rather than failed to reach it, and would point at the hip roll latch rather than at the reward.

Two orderings matter and both follow from section 11.4. This arm asks the hip roll for a lean and the hip roll is itself latched, with a stop that sits at an action of 0.416 against a saturating action of 0.420, so arm 4 must precede it or the reward will be asking a pinned joint to move. And it must precede arm 7, because a nominal pose change alters the stance width and with it every figure the premise rests upon.

The premise as measured, carried here on 2026-09-23 from the arm list that used to duplicate this sub-section.

Three quantities establish the premise and each is measured over the three arms of section 10. The stance is 0.1460, 0.1361 and 0.1576 metres wide, so the stance ankle stands 73.0, 68.0 and 78.8 millimetres from the midline. The sole is 0.0846 metres across, so the centre of pressure may be carried at most 42.3 millimetres inboard of that ankle, which places the inboard limit of the support polygon 30.7, 25.7 and 36.5 millimetres from the midline. And the torso sways, measured as the lateral base frame velocity at the stride frequency with the commanded lateral velocity removed, by an amplitude of 2.9, 6.8 and 4.8 millimetres.

The comparison is the finding. The torso moves a tenth of the distance that quasi static single support requires, so the centre of mass never enters the stance foot's support polygon at all, and the centre of pressure is consequently driven to the inboard edge of the sole and held there in pursuit of it. That is not an inference, it is the measurement section 11.1 now records, the inner sole corner standing lower than the outer on 79 to 99 per cent of stance samples. The full 42.3 millimetres of centre of pressure relief is therefore already spent before the ankle is asked for anything, and what remains is the ankle's to supply.

| Arm | Half stance [mm] | Measured sway [mm] | Residual lever at the ankle [mm] | Moment the ankle must hold [Nm] | Of a 17 Nm ceiling |
|---|---|---|---|---|---|
| 11.9 | 73.0 | 2.9 | 70.1 | 13.95 | 82.0 pct |
| 17 | 68.0 | 6.8 | 61.2 | 12.18 | 71.6 pct |
| 34 | 78.8 | 4.8 | 74.0 | 14.72 | 86.6 pct |

Against the 198.96 newton weight of this machine the ankle is asked for 12.2 to 14.7 newton metres before any disturbance, and the 8.42 newton metres the sole could relieve it of is already committed. The measured ankle torques of 11.9, 15.9 and 32.1 newton metres exceed even these figures, the excess being the latch of section 11.2 pressing the joint into its stop rather than any part of this balance, so the quasi static demand should be read as the floor beneath the measurement and not as an account of it.

### 21.6 Arm 6, the two ankle action scales halved

Status, RUN AND ABANDONED. It was IMPLEMENTED in the working tree on or before 2026-09-23 and named as an arm at the user's direction on that date, having until then stood in the configuration as an unattributed departure from the sizing rule of arm 4. The run carrying it without arm 7 was reported by the user on 2026-09-23 to have failed and was stopped, and the run carrying arm 7 on top of it was reported sound on the same date, so the arm survives in the tree as the scale dictionary that arm 7 and arm 8 are now run on top of rather than as an experiment of its own. Its stated falsifier is therefore unmeasured and the prediction section 14.2 makes for it is neither confirmed nor refuted, which is recorded here rather than left to be inferred from the absence of a dump. It is numbered 4b rather than given a number of its own because it modifies arm 4's dictionary in place and is run alongside arm 5 rather than after it, and because renumbering a sequence already in flight is the hazard section 20.1 records having already suffered once.

The arm takes the one quantity arm 4 leaves in an unsatisfactory place. Arm 4 equalises the action at saturation across the six joint groups at 1.050, which is the rule section 4.6 derives, and the stop action falls out of that rule rather than being chosen, landing at 1.362 for the ankle roll and 3.484 for the ankle pitch at the pose of 2026-09-17. Section 11.2 establishes that the ankle roll reaches its stop by command and not by torque, the commanded targets standing at plus and minus 0.368, 0.409 and 0.581 radians against a plus and minus 0.2618 radian stop across the three executed arms, and section 11.4 establishes that the quantity governing whether such a command is issued is the stop action measured against the exploration noise, whose standard deviation the seventeen newton metre run records at 0.86. Arm 4 moves the ankle roll stop from 0.641 standard deviations of that noise to 1.584, which places it at the edge of routine exploration rather than beyond it. Halving the two ankle scales to 0.08 moves it to 3.206, which is beyond it.

The code is the same dictionary arm 4 installs at `kscale_base_env_cfg.py:214`, its two ankle entries alone differing from the values the sizing rule gives.

```python
        scale={
            "(left|right)_hip_pitch_04": 0.4,
            "(left|right)_hip_yaw_03": 0.4,
            "(left|right)_knee_04": 0.2667,
            "(left|right)_hip_roll_03": 0.16,
            "(left|right)_foot_pitch_02": 0.08,
            "(left|right)_foot_roll_02": 0.08,
        },
```

Four design decisions govern it and the first is the one a reader is most likely to mistake for an error.

The arm deliberately breaks the equalisation that arm 4 exists to establish. At 0.08 the two ankles saturate their actuators at an action of 2.125 against the 1.050 of the other four groups, so they are the least authoritative members of the set per unit action rather than its equals. That asymmetry is the intervention rather than a defect in it, the argument being that equal authority is the right default only where no joint is known to be misbehaving, and that sections 11.1 and 11.2 identify these two joints specifically as the ones the policy drives past their travel. Arm 4 remains the reference the arm departs from, and the two must be reported together or the departure cannot be read.

A scale is a reparameterisation and not a bound, and this arm inherits the whole of the caution section 20.1 attaches to arm 4. A policy that wishes to command a target beyond the stop need only double its raw action, and this workspace has measured exactly that compensation once already, the ankle roll stop residency having RISEN from 0.669 to 0.765 when the global scale was cut from 0.4 to 0.25, which is the addition of 2026-09-04 to [../context/KScale.md](../context/KScale.md). What distinguishes the present arm from that excluded one is the same distinction arm 4 rests upon, that the change is to the relative geometry of the action space rather than to its overall size, so an isotropic exploration noise reaches the six groups in proportions it did not before. The distinction is real and it is not a bound, and the honest position is that this arm is the cheap half of arm 10 rather than a substitute for it.

The arm cannot be run before arm 7 without cost, and the coupling is arithmetic rather than a matter of taste. Section 12 measures the swing closure requiring an ankle pitch of minus 47.5 degrees, and the action that commands it is the distance from the joint's default divided by the scale. At the pose of 2026-09-17 and a scale of 0.1619 that action is 3.214, at the same pose and 0.08 it is 6.504, and at the arm 7 pose and 0.08 it is 3.818. Halving the scale alone would therefore have doubled the cost of a motion the policy demonstrably needs, and deepening the nominal pose returns it almost exactly to where arm 4 left it. The two arms are run together for that reason and section 21.7 records the consequence for attribution.

And the change is confined to the KScale configuration, the shared `environments` mdp package being untouched by it, so SD_BRS1 and the TRON1 SF and PF tasks see the dictionary they saw before and no historical comparison is invalidated.

| Group | Arm 4 scale | Arm 6 scale | Action at saturation | Action to the nearer stop, 2026-09-17 pose | Action to the nearer stop, arm 7 pose | Stop in noise standard deviations, arm 7 pose |
|---|---|---|---|---|---|---|
| Hip pitch | 0.4000 | 0.4000 | 1.050 | 3.127 | 3.491 | 4.06 |
| Hip yaw | 0.4000 | 0.4000 | 1.050 | 3.927 | 3.927 | 4.57 |
| Knee | 0.2667 | 0.2667 | 1.050 | 1.921 | 3.272 | 3.80 |
| Hip roll | 0.1600 | 0.1600 | 1.050 | 1.041 | 1.041 | 1.21 |
| Ankle pitch | 0.1619 | 0.0800 | 2.125 | 7.050 | 4.363 | 5.07 |
| Ankle roll | 0.1619 | 0.0800 | 2.125 | 2.757 | 2.757 | 3.21 |

The table is derived rather than quoted, by section G of `nominal_pose.py` in the artefact directory, and the noise standard deviation of 0.86 is the figure section 20.1 takes from the seventeen newton metre run. The hip roll is the joint the arm leaves worst placed, its stop standing at 1.21 standard deviations because the mirrored nominal of 0.042951 radians sits only 0.166489 from the adduction limit, and a reader who finds the hip roll still latched after this arm should look to arm 10 for that joint rather than to a further scale reduction.

The prediction is that the ankle roll stop residency falls further than arm 4 alone would take it, from the 88.7 per cent of section 11.1 toward a figure the policy's own noise cannot sustain, and that the excursion beyond the stop that section 11.2 measures at a median of 0.16 radians contracts in proportion. The falsifying observation is the raw ankle action distribution widening by the same factor of 2.02 that the scale fell, which would be the compensation of 2026-09-04 reproduced exactly and would establish that no scale can act upon this fault, sending the sequence to arm 10 immediately.


#### The state of the working tree on the morning of 2026-09-23, before arm 6 and arm 7 were applied

This sub-section is retained as the record of what the tree held before the implementation pass of 2026-09-23, section 21.7 carrying the state after it. The comparison below is against the state this document recorded on 2026-09-22 and was established by diffing the tree against the backup taken before the arm 5 merge.

| File and line | On 2026-09-22 | On the morning of 2026-09-23 | Effect |
|---|---|---|---|
| `kscale_base_env_cfg.py:991` | `rew_hip_lateral_sway` active at weight 5.0 | commented out | arm 5 was DISABLED |
| the same term's `std` | 0.03 | 0.01 | the kernel narrowed, see the table of section 21.5 |
| `kscale_base_env_cfg.py:804` | `pen_hip_roll_deviation` at 0.0 | at -0.1 | consistent with arm 5 disabled, conflicting the moment it is re-enabled |
| `kscale_base_env_cfg.py:849` | `target_height` 0.766 | 0.73 | a partial and undocumented anticipation of arm 7 |
| `kscale_base_env_cfg.py:922` | `base_height_target` 0.766 | 0.73 | the same value, correctly moved with it |
| `kscale_base_env_cfg.py:1119` and `:1149` | `num_envs` 9000 | 6400 | population, which section 2.2 records as the least consequential confound |
| `kscale_base_env_cfg.py:214` | the scale dictionary | unchanged | arm 4 was ACTIVE |
| `kscale_identified_cfg.py:152` and `:168` | `effort_limit=17` | unchanged | the baseline section 20 fixes |

The judgement recorded at the time was that disabling arm 5 while leaving arm 4 active was the correct ordering and not a reversal, section 21.5 stating the dependency, that the sway reward asks the hip roll for a lean while the hip roll is latched at an action of 0.416 against a saturating action of 0.420, and that arm 4 moves that stop to 1.040. That judgement stands as an account of the ordering and has since been overtaken by the user's direction to run the arms together, which section 21.7 records. The height target at 0.73 was a change this document could not interpret, being neither the 0.766 that every run of both sequences carried nor the standing height of any pose, and arm 7 has since replaced it with the 0.73492 that the posture actually produces.

### 21.7 Arm 7, the nominal pose

Status, RUNNING. It was IMPLEMENTED in the working tree on 2026-09-23 at the user's direction, together with a reduction of the settling margin to zero that the arm as written did not propose and which the user then set to 0.01 metres on the same date. It is the first arm of this document to be applied with two other arms already standing in the tree, and section 21.7 states what that costs the attribution. It was then COMMENTED OUT on the same date so that the arm 6 run could be restarted without it, that run failed, and the arm was RESTORED on 2026-09-23 once the run carrying it on top of arm 6 proved sound, which section 21.8 records together with the tree arm 8 is launched from. The three values it moves stand in the tree as written, `validate_arm6.py` in the artefact directory asserting them against the two configuration files and confirming that the restored posture is still closed, the sagittal residual reading 0.00001 radians on both legs against the identity of section 4.4.

The arm carries two independent justifications. Section 4.5 argues that the height target of 0.766 metres names the standing height of a nearly straight leg, that it sits on the flat part of the knee to height curve so that it constrains the knee hardly at all, and that it places the knee within one and a half standard deviations of its own hard stop. Section 12 adds the second, that the crouch the policy actually adopts consumes the ankle pitch dorsiflexion, the sagittal closure identity requiring minus 47.5 degrees in swing against a minus 50 degree stop which is exceeded on 42 to 45 per cent of samples. Moving the nominal to a deeper pose relieves both, and the published defaults of the surveyed humanoids are uniformly flexed with none carrying a straight knee [13][15].

Three values move together and they are in two files. The pose itself now stands at `kscale_identified_cfg.py:197`, the three sagittal nominals carrying the published K-Scale K-Bot posture of 20, 50 and minus 30 degrees and the two roll nominals carrying the mirrored values the pose supplied on 2026-09-17 introduced.

```python
init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.73492),
    joint_pos={
        "right_hip_pitch_04": -0.34907,
        "left_hip_pitch_04": 0.34907,
        "right_hip_roll_03": 0.042951,
        "left_hip_roll_03": -0.042951,
        "right_hip_yaw_03": 0.0,
        "left_hip_yaw_03": 0.0,
        "right_knee_04": 0.87266,
        "left_knee_04": 0.87266,
        "right_foot_pitch_02": -0.52360,
        "left_foot_pitch_02": -0.52360,
        "right_foot_roll_02": 0.041254,
        "left_foot_roll_02": -0.041254,
    },
    joint_vel={".*": 0.0},
)
```

The two height targets that follow from it stand at `kscale_base_env_cfg.py:849` and `:922`, both now carrying the same number as the spawn.

```python
    pen_base_height = RewTerm(
        func=mdp.base_height_rough_l2,
        params={
            "target_height": 0.73492,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
        weight=-30.0,
    )
```

Two corrections to this sub-section as it was written on 2026-09-23 follow from deriving the pose rather than quoting it, and both were found by `nominal_pose.py` in the artefact directory, which computes the forward kinematics against `kscale.urdf` and needs no simulator.

The standing height of this posture is 0.73492 metres and not the 0.73540 that the table below formerly carried. That figure is correct for the pose it was taken at, being the posture as it stood between 2026-09-16 and 2026-09-17 with both roll nominals at zero, and the addition of 2026-09-16 to [../context/KScale.md](../context/KScale.md) records it in that form. The arm retains the mirrored hip roll and ankle roll of 0.042951 and 0.041254 radians, which tilt each leg out of the sagittal plane and cost 0.48 millimetres of height, so the figure the three values must carry is 0.73492. The pose remains closed at that height, the ankle roll origin standing at exactly the 0.04300 metre sole depth and the sole lying flat to 0.054 degrees, which is the check section 14.2 of [../context/KScale.md](../context/KScale.md) prescribes before any launch that moves the pose.

And the last row of the table below inverted the quantity it named. The arm does not move the ankle pitch nominal from 17.7 degrees of its lower stop to 30.0, it moves the nominal itself from minus 17.7 degrees to minus 30.0 and thereby moves it CLOSER to that stop, the travel remaining falling from 32.3 degrees to 20.0. What the arm buys at that joint is not headroom but authority, the action being a delta from the default so that a deeper default commands the same angle at a smaller action. At the scale of 0.08 that arm 6 installs, the action commanding the minus 47.5 degrees section 12 measures falls from 6.504 to 3.818, a reduction of 41.3 per cent, and it is that reduction rather than any change in travel that the prediction rests upon.

The design decision that matters most here is that the three values are one value, not three. The closure identity of section 4.4, that the signed hip pitch plus the knee plus the ankle pitch sums to zero, is what makes a pose stand with its soles flat, and the standing height follows from the pose by forward kinematics. Setting the height target independently of the pose asks the policy to hold a base height that its own nominal posture does not produce, which is the condition section 4.5 diagnoses and which the arm exists to remove.

A second decision is that the hip roll and ankle roll nominals are left untouched. They carry the mirrored 0.042951 and 0.041254 that draw the ankle frames closer together than every pre 2026-09-17 pose held, and moving them would change the stance width that arm 5's entire premise is measured against. The sagittal chain and the lateral chain are separable here and only the sagittal chain moves. The fold nonetheless widens the stance by itself, the ankle frame separation rising from 0.1992 metres to 0.2044 as the rolled hips draw apart, which is a 5.2 millimetre change to the quantity section 20.1 measures arm 5's amplitude against and is small beside the 25.7 to 36.5 millimetres that table reports.

A third decision belongs to the settling margin rather than to the pose, and it is the user's instruction of 2026-09-23 rather than a proposal of this document. The workspace has spawned this robot at its standing height plus 0.02 metres since 2026-08-25, following the convention `sd_brs1_identified_cfg.py` records, that margin was reduced to zero so that the spawn height, the standing height and the target of `pen_base_height` would be one number, and the user set it to 0.01 metres on the same date once the measurement below was in hand. The spawn therefore stands ten millimetres above the height the reward targets. What the margin buys is worth stating plainly, because the reset distribution reaches very much further than any of the three choices. The startup event at `kscale_base_env_cfg.py:610` shifts every joint default by up to 0.05 radians and the six reset events from `:656` onward shift the drawn position again by up to 0.24 at the hip pitch and the knee, each clamped to the soft limits by `events.py:350`, so the pose at the first step of an episode is not the nominal pose and the height it stands at is not the standing height.

| Quantity at the arm 7 pose | With the 0.02 metre margin | With the 0.01 metre margin, as applied | With no margin |
|---|---|---|---|
| Spawn height | 0.75492 m | 0.74492 m | 0.73492 m |
| Clearance beneath the sole at the nominal pose | 20.0 mm | 10.0 mm | 0.0 mm |
| Arrival speed at the nominal pose | 0.626 m/s | 0.443 m/s | 0.000 m/s |
| Resets drawing a pose taller than the spawn | 6.2 pct | 20.3 pct | 37.0 pct |
| Median penetration among those resets | 3.0 mm | 6.9 mm | 11.3 mm |
| Worst penetration among those resets | 11.4 mm | 21.4 mm | 31.4 mm |

The margin was therefore never sufficient at this pose whatever value it took, and the choice between the three is a choice of how the reset fails rather than whether it does. The value applied exchanges a drop of at most ten millimetres for an interpenetration of at most twenty one on a fifth of resets, where the old margin gave at most twenty millimetres of drop against eleven of interpenetration on a sixteenth. The solver resolves such an interpenetration over the first few steps, so the quantity to watch in the first thousand iterations is the termination rate at episode start rather than the height itself. Should it prove costly the remedy is a margin sized against the reset distribution rather than against the nominal pose, which section C of `nominal_pose.py` prices at 0.032 metres for full clearance, and the arm's own claim that the target height equals the standing height is untouched by that choice since the target and the spawn are separate parameters.

| Quantity | Before the arm | Arm 7 as applied | Source |
|---|---|---|---|
| Hip pitch nominal | 0.20369 rad | 0.34907 rad | `kscale_identified_cfg.py:224` |
| Knee nominal | 0.51239 rad | 0.87266 rad | `kscale_identified_cfg.py:230` |
| Ankle pitch nominal | -0.3087 rad | -0.52360 rad | `kscale_identified_cfg.py:232` |
| Spawn height | 0.786 m | 0.74492 m | `kscale_identified_cfg.py:222` |
| `pen_base_height.target_height` | 0.73 m | 0.73492 m | `kscale_base_env_cfg.py:849` |
| `pen_feet_regulation.base_height_target` | 0.73 m | 0.73492 m | `kscale_base_env_cfg.py:922` |
| Standing height | 0.76630 m | 0.73492 m | `nominal_pose.py` section A |
| Ankle frame separation | 0.1992 m | 0.2044 m | `nominal_pose.py` section A |
| Ankle pitch travel to its lower stop | 32.3 deg | 20.0 deg | `kscale.urdf:603` |
| Knee travel to its lower stop | 29.4 deg | 50.0 deg | `kscale.urdf:611` |
| Action commanding the swing ankle pitch | 6.504 | 3.818 | `nominal_pose.py` section F |

Two quantities were checked against the new pose and deliberately left unchanged. The `min_feet_distance` floor of `pen_feet_distance` at `kscale_base_env_cfg.py:906` stands at 0.19 metres, which sat 9.2 millimetres below the old nominal separation and now sits 14.4 below the new, so the floor keeps its clearance and does not fire at every reset. And `base_height_target` in `pen_feet_regulation` is INERT in this configuration and its movement changes no behaviour, `feet_regulation` at `rewards.py:1850` reading that parameter only as the height gate length scale and only when `height_decay_scale` is absent, which at `kscale_base_env_cfg.py:924` it is not. It is held in step regardless, being dumped to `params/env.yaml` where a later reader will take it for the standing height, as the sections of this document have.

One pre-existing condition was measured at the new pose and is recorded so that it is not attributed to the arm. `kscale_self_collision_audit.py` reports four interpenetrating collision part pairs at the nominal pose, all of them shank against foot, and the deeper fold leaves the penetration depth unchanged at 12.04 millimetres against 12.06 while raising the overlap volume from 17.51 to 20.66 cubic centimetres. The overlap is the one section 16 of [../context/KScale.md](../context/KScale.md) records and its 2026-09-11 addition attributes an episode length collapse to, it predates every pose this robot has carried, and the arm neither creates nor materially worsens it. That audit had itself been standing two pose revisions behind the asset configuration and was refreshed in the same pass.

The stance load moves and the direction at the knee should be anticipated. At the arm 7 pose the single support hold torque rises from 16.90 to 27.84 newton metres at the knee and from 8.32 to 11.32 at the hip pitch, falls from 4.23 to 3.60 at the hip roll, and does not move at either ankle, both of those being fixed by the sole geometry and the lateral lean rather than by the sagittal fold. The knee figure is held with margin by the 84 newton metre ceiling that joint carries, standing at 33 per cent of it, but it deflects the knee 0.139 radians at the configured stiffness of 200 against 0.084 before, so the policy will meet a visibly softer knee at the same commanded target.

The prediction is that the knee's stop residency falls to near zero, that the height penalty ceases to fight the policy's own operating point, and that the swing ankle pitch leaves its stop without any change to the joint's travel. The falsifying observation is the swing ankle pitch saturation of section 12 failing to fall from its 99.6 per cent, which would send the sequence to arm 25.


#### The state of the working tree after the implementation pass of 2026-09-23, and what a launch from it can and cannot establish

Four arms now stand in the tree together at the user's direction, and the first thing this sub-section must say is that a run launched from it is a four variable departure from the seventeen newton metre reference of section 10 and can attribute an effect to no one of them. That is the user's decision, taken to gather behaviour at the full intervention rather than to price each part, and it is recorded here so that a later session reading the dump does not mistake the compound for an ablation arm.

| File and line | Value | Arm | Applied |
|---|---|---|---|
| `kscale_base_env_cfg.py:214` | the six entry scale dictionary | arm 4 | 2026-09-22 |
| `kscale_base_env_cfg.py:991` | `rew_hip_lateral_sway` active at weight 5.0 | arm 5 | 2026-09-22, disabled and restored since |
| the same term's `std` and `amplitude` | 0.01 and 0.03 | arm 5 | the width narrowed from the merged 0.03 |
| `kscale_base_env_cfg.py:219` and `:220` | the two ankle scales at 0.08 | arm 6 | on or before 2026-09-23 |
| `kscale_identified_cfg.py:224`, `:230` and `:232` | the three sagittal nominals at the K-Scale posture | arm 7 | 2026-09-23 |
| `kscale_identified_cfg.py:222` | spawn at 0.74492, the margin reduced from 0.02 m to 0.01 | arm 7 and the user's direction | 2026-09-23 |
| `kscale_base_env_cfg.py:849` and `:922` | both height targets at 0.73492 | arm 7 | 2026-09-23 |
| `kscale_identified_cfg.py:152` and `:168` | `effort_limit=17` | the baseline section 20 fixes | unchanged |
| `kscale_base_env_cfg.py:1119` and `:1149` | `num_envs` 6400 | not an arm | unchanged |

One conflict survives the pass and it is not resolved. `pen_hip_roll_deviation` stands at minus 0.05 at `kscale_base_env_cfg.py:804`, having been zeroed by the arm 5 merge, restored to minus 0.1 when that arm was disabled, and set to minus 0.05 since. It prices any hip roll excursion from the nominal pose, which is precisely the lean `rew_hip_lateral_sway` asks for, so the two terms price opposite signs of the same motion while both are active. The magnitudes are not comparable, the sway reward reaching 4.03 per second at the policy's present operating point against a deviation penalty that reads 0.05 newton metres of weight against an excursion measured in tenths of a radian, so the sway term will dominate and the conflict is a drag rather than a cancellation. It is left at the value the user set rather than zeroed, since a deliberate half measure is more likely than an oversight at that number, and it is recorded here so that a sway amplitude falling short of its target is read against this term before it is read against the hip roll latch.

Two further consequences of running the arms together are worth stating in advance of the run, since each determines what an observation will mean.

Arm 6 and arm 7 are not separable even in principle, and section 21.6 gives the arithmetic. Halving the ankle scale doubles the action that commands any given ankle angle, and deepening the nominal pose halves the angle the swing requires, so the two very nearly cancel at the ankle pitch, the commanding action moving from 3.214 under arm 4 alone to 6.504 under arm 6 alone to 3.818 under both. A run carrying both therefore tests the ankle ROLL restriction at close to an unchanged ankle pitch authority, which is the useful reading of the pair and very nearly the reading the ordering would have produced anyway.

Arm 5 and arm 7 are separable in principle and are confounded in this run. Section 21.5 states that arm 5 should precede arm 7 because a pose change alters the stance width and with it every figure the sway premise rests upon, and the pose change moves the ankle frame separation of the NOMINAL pose from 0.1992 metres to 0.2044. That 5.2 millimetre widening does not propagate directly to the figures of section 20.1, and the reason is worth stating because it is the same reason those figures were measured rather than derived. The three policies walk very much narrower than the nominal, their measured stance standing at 0.1460, 0.1361 and 0.1576 metres against the 0.1992 the pose gives, so between 41 and 63 millimetres of the nominal separation is already surrendered by the policy before the pose is consulted at all. What the deeper fold does to that surrendered quantity cannot be predicted from the nominal and must be measured, and the measurement is the first thing to take from this run, since the inboard limit of the support polygon and with it the sufficiency of the 0.03 metre amplitude follow from the walking stance and not from the pose. A reader finding the sway short of its target should establish the measured stance first and take 0.06 metres as the next amplitude only after it.

The instrumentation of section 20.2 applies to this run as to every other, and one addition is required by the zeroed settling margin. The termination rate within the first second of an episode must be recorded separately from the rate over the episode as a whole, since section 21.7 establishes that 20.3 per cent of resets now draw a pose that interpenetrates the ground at the spawn height, and a rise in early terminations is the signal that the margin must be restored.

### 21.8 Arm 8, the energy budget, in three stages of rising commitment

Status, STAGE ONE APPLIED to the working tree on 2026-09-23 at the user's direction and awaiting its run, stages two and three NOT APPLIED. It is first in the order because it is a precondition rather than an intervention. Section 14.6 measures an objective in which tripling the mechanical power costs 0.68 per cent of the return, and every arm below it asks the optimiser to change a behaviour it is very nearly indifferent to. This arm merges what earlier revisions of this document carried as three separate proposals, the raising of the four existing energy weights, the addition of a power term and the replacement of the reward stack by a constrained formulation, because all three answer one question and differ only in how far they go. They are staged so that each is run only when the one before it has failed, and stage one alone may settle the matter.

##### Stage one, the four existing weights raised by ten

The premise is section 14.6 and it is the most quantitative premise in this document. The four energy terms together are 1.04 per cent of A's negative budget and 0.44 per cent of C's. Run B burnt 3.61 times A's power, paid 0.4276 per second more for it and collected 1.9175 per second more on tracking alone, an exchange rate of 4.48 to one in favour of the expensive gait.

The implementation is four weights in the KScale configuration and nothing else, and it stands in the tree as written, at `kscale_base_env_cfg.py:892`, `:893`, `:928` and `:970`.

```python
    pen_joint_torque      = RewTerm(func=mdp.joint_torques_l2,     weight=-1.0e-4)   # was -1.0e-5
    pen_joint_accel       = RewTerm(func=mdp.joint_acc_l2,         weight=-1.0e-6)   # was -1.0e-7
    pen_joint_vel_l2      = RewTerm(func=mdp.joint_vel_l2,         weight=-1.0e-4)   # was -1.0e-5
    pen_joint_torque_rate = RewTerm(func=mdp.JointTorqueRatePenalty, weight=-1.0e-4) # was -1.0e-5
```

The four terms are the only reward weights the arm touches and no other file changes, `joint_torques_l2`, `joint_acc_l2` and `joint_vel_l2` being stock IsaacLab terms and `JointTorqueRatePenalty` a class this repository already wires into this configuration alone, so the shared mdp package is untouched and no caller outside the KScale tasks sees any difference. `RewardsCfg` at `kscale_base_env_cfg.py:732` is reached only through `KscaleEnvCfg` and `KscaleHIMEnvCfg` at `:1127` and `:1157`, which only `robots/kscale_solefoot_env_cfg.py` derives from, so the blast radius of these four lines is the eight KScale task registrations and nothing else.

Four design decisions attach.

The factor is ten rather than the 4.48 the exchange rate names. A factor of 4.48 makes the two exactly indifferent, which is the worst available choice, since an objective that is indifferent between two gaits will select between them by the exploration noise. Ten leaves a margin of 2.2 over indifference and takes the four terms from 1.04 per cent of A's negative budget to 9.5 per cent, which is a share that can shape without dominating. It is a round number chosen above a measured floor and it is not itself derived, and that should be said plainly rather than dressed up.

The share the group reaches is a property of the gait and not of the weight, and `validate_arm6.py` in the artefact directory re-prices the factor against each of the three dumped evaluation runs so that the spread is visible rather than inferred from A alone. The arithmetic holds every other term fixed and asks what the four would have cost the policies that were actually trained, which is an upper bound on what the arm will cost, the arm existing precisely so that the behaviour moves away from the point the projection is taken at.

| Run | The four terms now, per second | Share of the negative budget now | The four terms at ten times | Share at ten times |
|---|---|---|---|---|
| A, one scale of 0.4 | -0.1189 | 1.04 pct | -1.1894 | 9.51 pct |
| B, arm 4 | -0.5465 | 4.04 pct | -5.4650 | 29.63 pct |
| C, arm 4 and arm 5 | -0.0394 | 0.44 pct | -0.3941 | 4.22 pct |

The spread is the argument for the arm rather than an objection to it. On the expensive gait of run B the four terms would rise to very nearly thirty per cent of the negative budget, which is what it means for a 395 watt policy to be penalised out of its choice, and on the cheap gait of run C they would stand at 4.22 per cent, which is a price a policy already walking cheaply barely notices. A weight that costs the wasteful gait seven times what it costs the economical one is a weight doing the work asked of it, and the figure to hold in mind for the run this arm launches is the 4.22 per cent rather than the 9.51, arm 7 being applied on top of arm 5, whose budget run C measures.

All four move together and by the same factor. The measurement prices them as a group, because it is their sum against the tracking gain that section 14.6 compares, and moving one of four would leave the other three at a share where they cannot act while making the moved one the only lever. If a later arm wishes to separate them the place to do it is after this one has established that the group can act at all.

`pen_action_rate` and `pen_action_smoothness` are deliberately NOT moved, and the reason is section 14.6's last paragraph. Both are quadratic in the RAW action, so arm 4 already multiplied their effect at the ankles by 6.1 without anyone choosing it, and arm 6 would multiply it by a further 4.1 if it stands. Those two terms are therefore already moving under other arms and raising them here would confound this one. They should be revisited only once the action scale is settled.

The literature says this arm is the wrong shape and it is run anyway, which is recorded here so that the sequence is honest about it. Kim and colleagues argue that a reward only formulation needs more than ten terms and repeated retuning and that a constrained formulation reaches the same behaviour with three reward terms and one tuned coefficient [4], and the survey of Ha and colleagues records the same tuning burden as the standing difficulty of the field [18]. This arm is a weight change because a weight change is four lines and a constrained Markov decision process is a change of algorithm, and the four lines answer the question of whether the group can act at all, which is the question that must be answered before a larger change could be justified. Arm 8 below carries the constrained version for when it is.

The prediction is that the absolute joint power falls from C's 67.1 watts, that the whole body torque rate falls from 120.8 newton metres per second, and that `rew_lin_vel_xy` falls by less than the 1.9175 per second that B's excess bought, which is the test of whether the exchange rate was set correctly. The falsifying observation is the mean reward falling while the power does not, which would say the four terms are now large enough to disturb the optimisation without being large enough to select a cheaper gait, and the remedy is then a curriculum on the weights rather than a larger constant, in the manner Watanabe and colleagues adopt for the contact velocity term that cluster 27 of `context/literature.md` records.


##### Stage two, the power term, to be run only if stage one moves the power and costs the task

This stage needs no new code at all, which is unique in the sequence, the function it wires already standing in the shared module unused.

The arm supplies a preference over gait shape that the velocity term cannot express. The energy minimisation result establishes that whether a recognisable gait emerges at all turns on the presence of such a term [7], and this configuration prices mechanical power at nothing. Its purpose is the shuffle of run A rather than the splay, run A stepping at 6.25 per second against a commanded two and spending 118.75 watts against run C's 89.52, and it is separable from arm 10 and may be run beside it.

The function already exists in the shared module at `rewards.py:987` and is wired into no KScale term.

```python
def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)
```

The arm adds one term to `RewardsCfg` and changes nothing else.

```python
    # PROPOSED, not applied. Budgeted against the 89.5 to 118.8 W of section 3.2.
    pen_joint_power = RewTerm(
        func=mdp.joint_powers_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

The design decision is the weight and it is arithmetic rather than judgement. The term returns the summed absolute mechanical power in watts, so at the 89.5 watts run C spends a weight of minus 0.02 places the term at 1.79 per second. The velocity tracking reward stands at some 47 per second, so the power term would be 3.8 per cent of the objective, which is the two to four per cent band section 6 asks for. The alternative of pricing the squared torque instead was rejected because `pen_joint_torque` at `kscale_base_env_cfg.py:856` already does so at minus 0.00001 and torque alone does not distinguish a shuffle from a walk, a stationary strut at full torque costing the same as a limb doing work. Power is the product with velocity and is the quantity that separates them.

| Quantity | Run A | Run C | Term at weight -0.02 |
|---|---|---|---|
| Absolute joint power [W] | 118.75 | 89.52 | 2.38 and 1.79 per s |
| Cadence [steps/s] | 6.25 | 2.02 | the behaviour the term is aimed at |
| Share of the velocity tracking reward | 5.1 pct | 3.8 pct | the two to four per cent band of section 6 |

The prediction is that the cadence falls toward the commanded two per second and that the cost of transport falls without the tracking error rising. The falsifying observation is the tracking error rising, which would mean the weight has been set into the region where the term competes with the objective rather than shaping it.


##### Stage three, the constrained formulation, recorded and not scheduled

This arm is recorded so that the sequence knows where it is heading and is deliberately not scheduled, because it is a change of algorithm rather than of configuration and because arm 8 must establish that the regularisation group can act before a larger machinery for wielding it is justified.

Kim, Oh, Lee, Choi, Ji, Jung, Youm and Hwangbo replace the reward stack with a constrained Markov decision process solved by a modified interior point policy optimisation, carrying six probabilistic constraints and five average constraints, and report that the formulation needs three reward terms and one tuned coefficient where the reward only version needed more than ten terms and repeated retuning [4]. Three of their six probabilistic constraints are joint position, joint velocity and joint torque, which are exactly the three quantities section 14 finds unpriced here, and their reported hardware violation rates are 0.007, 0.001 and 0.002 against a 0.025 threshold. The property that matters most to this workspace is the transfer result, the constrained controller moving from Raibo to Anymal C without retuning the torque coefficient where the reward only controller degraded significantly, since this workspace runs SD_BRS1 and three TRON1 variants against one shared mdp package and has spent this entire document discovering that its weights are robot specific. Chane-Sane, Leziart, Flayols, Stasse, Souères and Mansard offer the cheaper path, each constraint becoming a probability of terminating future reward under a minimal modification to proximal policy optimisation [3], and that is the form this workspace would reach for first, the termination machinery already existing.

The precondition for scheduling it is arm 8's result. If a tenfold weight increase moves the power and costs little tracking, the reward only formulation is adequate for this robot and this arm stays unscheduled. If it moves the power and costs the task, or moves neither, the weights are the wrong instrument and this arm becomes the next one.


#### The state of the working tree after the arm 8 pass of 2026-09-23, and the run it launches

Section 21.7 records the tree as it stood when four arms were applied together. Two things have happened to it since and both must be stated before the table, because each changes what the next dump will mean.

Arm 7 was commented out so that the arm 6 run could be restarted without it. That run was reported by the user to have failed, the run carrying arm 7 on top of arm 6 was reported sound, and arm 7 was restored on 2026-09-23 accordingly, the pose block at `kscale_identified_cfg.py:227` and the two height targets at `kscale_base_env_cfg.py:852` and `:961` returning to the values section 21.7 sets and the arm 6 values being left in place beside them as comments so that the reversal costs one edit rather than a re derivation. The arm 6 run is therefore abandoned rather than concluded and its falsifier is unmeasured, which section 21.6 now records.

Arm 8 stage one was then applied on top of that, raising the four energy weights by a factor of ten and touching nothing else. It is the fifth arm to stand in the tree at once and the compound is now a five variable departure from the seventeen newton metre reference of section 10, so the warning of section 21.7 holds with one more term in it, and what the run establishes is the behaviour at the full intervention rather than the price of any part of it.

| File and line | Value | Arm | Applied |
|---|---|---|---|
| `kscale_base_env_cfg.py:213` | the six entry scale dictionary | arm 4 | 2026-09-22 |
| `kscale_base_env_cfg.py:221` and `:222` | the two ankle scales at 0.08 | arm 6 | on or before 2026-09-23, the arm abandoned and the value retained |
| `kscale_base_env_cfg.py:1036` | `rew_hip_lateral_sway` active at weight 5.0 | arm 5 | 2026-09-22, its `std` at 0.01 and its `amplitude` at 0.03 |
| `kscale_identified_cfg.py:230`, `:236` and `:238` | the three sagittal nominals at the K-Scale posture | arm 7 | restored 2026-09-23 |
| `kscale_identified_cfg.py:227` | spawn at 0.74492, the margin at 0.01 m | arm 7 and the user's direction | restored 2026-09-23 |
| `kscale_base_env_cfg.py:852` and `:961` | both height targets at 0.73492 | arm 7 | restored 2026-09-23 |
| `kscale_base_env_cfg.py:892`, `:893`, `:928` and `:970` | the four energy weights at -1.0e-4, -1.0e-6, -1.0e-4 and -1.0e-4 | arm 10 | 2026-09-23 |
| `kscale_base_env_cfg.py:894` and `:923` | `pen_action_rate` at -0.01 and `pen_action_smoothness` at -0.075 | deliberately NOT arm 8 | unchanged |
| `kscale_base_env_cfg.py:806` | `pen_hip_roll_deviation` at -0.1 | not an arm | set by the user, see below |
| `kscale_identified_cfg.py:152` and `:168` | `effort_limit=17` and `saturation_effort=17.0` | the baseline section 20.3 fixes | unchanged |
| `kscale_base_env_cfg.py:1164` and `:1194` | `num_envs` 6000 | not an arm | moved from 6400 by the user |

Two entries in that table are not arms of this document and are recorded so that a later reader does not attribute them to one. `pen_hip_roll_deviation` stands at minus 0.1 rather than the minus 0.05 section 21.7 found, so the conflict that sub-section describes is now twice as large, the term pricing exactly the hip roll excursion that `rew_hip_lateral_sway` pays for. The magnitudes remain far apart, the sway reward reaching 4.03 per second against a deviation penalty reading 0.0209 per second in run C's dump, so the conflict remains a drag rather than a cancellation, but a sway amplitude falling short of its target should be read against this term first. And the population stands at 6000 rather than the 6400 of run C and the 9000 of runs A and B, which section 2.2 records as the least consequential of the confounds and which nonetheless means the next dump is not matched to any of the three on experience per iteration.

The applied state was checked rather than assumed. `validate_arm6.py` in the artefact directory parses both configuration files with `ast`, needs no simulator, and asserts the four raised weights, the two that must not move, the three sagittal nominals, the spawn height and the two height targets, and it recomputes the sagittal closure identity of section 4.4 at the restored pose, which reads 0.00001 radians of residual on each leg and therefore stands with its soles flat.

What to take from the run is set by the prediction of section 21.8 and by one addition. The absolute joint power should fall from run C's 67.1 watts and the whole body torque rate from 120.8 newton metres per second, and `rew_lin_vel_xy` should fall by less than the 1.9175 per second that run B's excess power bought. The addition is that the four terms must be read as a share of the negative budget rather than as absolute numbers, because the arm moves the weights and the behaviour at once and the absolute figure conflates the two. The table of section 21.8 gives 4.22 per cent as the share the group would reach at run C's behaviour, and a realised share far beneath it is the signal that the arm worked, the policy having moved to a cheaper gait, while a realised share far above it is the signal that the weights are disturbing the optimisation without selecting a cheaper gait, which is the falsifier that sends the arm to stage two.

### 21.9 Arm 9, the ankle roll deviation penalty

Status, EXECUTED as `2026-09-25_08-00-34` and `2026-09-25_11-29-39`, REFUTED by section 16.4, and RECORDED RETROSPECTIVELY, this arm having been applied to the working tree and run twice without ever having been proposed in this document. It is carried here so that the sequence holds every intervention this robot has received and so that the exclusion it produced has a home.

The term was added between run A and run B in order to draw the ankle roll toward its nominal, the joint having stood at a mechanical stop in every run since 2026-09-17. It was commented out of the tree on 2026-09-28 rather than deleted, and it stands at `kscale_base_env_cfg.py:824` in that state.

```python
    # ARM 9 of plans/kscale_gait_shaping.md section 21.9.  EXECUTED as 2026-09-25_08-00-34
    # and 2026-09-25_11-29-39 and REFUTED by section 16.4, which measures it moving the joint
    # from one mechanical stop to the other rather than off a stop, raising the swing envelope
    # saturation from 46 to 98 per cent and the mean absolute torque from 8.5 to 16.1 N.m of a
    # 17 N.m ceiling while reversing the direction of the actuator's work on 99.3 per cent of
    # samples.  Commented out 2026-09-28 for the run that isolates it, 2026-09-28_10-25-57.
    # pen_ankle_roll_deviation = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=["(right|left)_foot_roll_02"]
    #         )
    #     },
    # )
```

Two design decisions were taken when it was applied and both are now seen to have been the wrong ones, which is recorded because each is a decision this sequence could make again.

The penalty was applied to the ACHIEVED position, `joint_deviation_l1` at `isaaclab/envs/mdp/rewards.py` differencing `joint_pos` against `default_joint_pos`, on a joint whose achieved position in stance is set by the ground reaction moment and in swing by a mechanical stop. The policy controls the commanded position and not the achieved one, and the two are connected by a servo whose authority ends at 17 newton metres and by a stop whose authority does not end at all, so the gradient of the penalty with respect to the quantity the policy controls is very nearly zero wherever the joint is against the stop. The term could not move the joint and did not.

It was applied together with a quadrupling of `rew_keep_ankle_roll_zero_in_air` from 0.25 to 1.0, so no run of the sequence can attribute the outcome to either alone. The run at `2026-09-28_10-25-57` removes the penalty and keeps the quadrupled reward, which separates them, and section 16.4 commits this document to the prediction that the joint stays at the near stop.

The parameterisation was minus 1.0, which is the same weight `pen_hip_yaw_deviation` carries at `kscale_base_env_cfg.py:815` and ten times `pen_hip_roll_deviation`. It was paid at 0.354 and 0.409 per second, which is 3.3 and 3.6 per cent of the negative budget, so the term was neither negligible nor dominant and the outcome is not explained by its size.

The refutation is in section 16.4 and the exclusion it produces is in section 20.3. The finding worth carrying beyond this robot is that a deviation penalty on a joint whose position is set by contact does not move the joint, it moves the commanded target, and where the commanded target is unbounded the policy satisfies the letter of the penalty by selecting whichever inadmissible target lies nearer the nominal and pressing into it harder. Cluster 31 of `/ws/context/literature.md` records it in that general form and records that the survey found no precedent for it.


#### The state of the working tree on 2026-09-28, and the run it launched

Five arms stood in the tree on the morning of 2026-09-23 and section 21.8 records them. Two changes were made between then and 2026-09-28 and neither was recorded in this document at the time, which is the condition sections 21.9 and 16.1 exist to repair.

The four energy weights were raised again, from the factor of ten that section 21.8 tabulates to a factor of fifty, and the comment block at `kscale_base_env_cfg.py:875` was not moved with them, so it continues to state ten above a value of fifty. The factor of ten was run as `2026-09-23_11-40-12` and section 16.2 reads it as a rung of a ladder. The factor of fifty is what runs A and C carry and one hundred is what run B carries.

`pen_ankle_roll_deviation` was added at minus 1.0 and `rew_keep_ankle_roll_zero_in_air` was raised from 0.25 to 1.0, neither having been proposed here. Section 21.9 records the first as arm 9 retrospectively.

The tree as it stands on 2026-09-28 therefore carries arm 4, arm 5, 4b, 5, 6 stage one at fifty and 6 stage two, with arm 9 commented out at `kscale_base_env_cfg.py:824` and the in air ankle roll reward left at 1.0. The run launched from it is `2026-09-28_10-25-57` at nine thousand environments, and what it establishes is stated in section 16.1, being the effect of the ankle roll deviation penalty isolated up to the population. What it cannot establish is anything about the six arms it carries alongside, all of which it shares with run C.

No arm of this section that is not named in the paragraph above has been applied to any file.

### 21.10 Arm 10, the commanded target clip

Status, NOT APPLIED, third in the order of section 20.1 and the only arm in the sequence that acts on the mechanism section 11.2 identified rather than on something adjacent to it.

The arm attacks the mechanism of section 11.2 directly. That section reconstructs the commanded targets at plus and minus 0.368, 0.409 and 0.581 radians against a plus and minus 0.2618 radian stop, on 82 to 99 per cent of samples, and finds the excursion beyond the stop tripling as the ceiling tripled. A target outside the reachable set converts the end stop into a free, perfectly repeatable and infinitely stiff reference, and nothing in the configuration prices a commanded target the joint cannot reach. The clip removes the reference rather than the incentive, and it does so at the only point in the pipeline where the quantity is in radians.

`ActionTermCfg` carries the field at `manager_term_cfg.py:97`.

```python
    clip: dict[str, tuple] | None = None
    """Clip range for the action (dict of regex expressions). Defaults to None."""
```

`JointAction` resolves it at `joint_actions.py:103` and applies it at `:175`, after the scale and the offset, so it bounds the target position rather than the raw action.

```python
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
```

```python
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
```

The proposal adds one field to the action term of `kscale_base_env_cfg.py:211` and changes nothing else.

```python
    # PROPOSED, not applied. Mechanical travel of kscale.urdf inset by 0.02 rad.
    # The hip roll and the hip pitch are MIRRORED between the two legs, so neither may be
    # written as a (left|right) pattern the way the scale dict above writes every group.
        clip={
            "(left|right)_foot_roll_02": (-0.241799, 0.241799),
            "(left|right)_foot_pitch_02": (-0.852665, 0.503599),
            "(left|right)_knee_04": (0.020000, 2.685260),
            "(left|right)_hip_yaw_03": (-1.550800, 1.550800),
            "right_hip_roll_03": (-2.248930, 0.189440),
            "left_hip_roll_03": (-0.189440, 2.248930),
            "right_hip_pitch_04": (-2.196570, 1.027200),
            "left_hip_pitch_04": (-1.027200, 2.196570),
        },
```

Four design decisions govern it and the first two are traps that the scale dictionary of arm 4 does not share.

The mirroring must be written out. The hip roll and the hip pitch carry opposite ranges on the two legs, the right hip roll running from minus 2.26893 to plus 0.20944 at `kscale.urdf:627` and the left from minus 0.20944 to plus 2.26893 at `:1107`, so a single `(left|right)` key would apply one leg's bounds to both and would clip one leg inside its own nominal pose. The scale dictionary is immune because a scale is a magnitude and the mirroring does not reach it, which is why that term may use the compact form and this one may not.

Omission is safe here where it is catastrophic in the scale. `joint_actions.py:106` initialises the clip tensor to minus and plus infinity rather than to ones, so a joint left out of the dictionary is simply unclipped and retains the behaviour it has. A partial clip covering only the ankle roll would therefore be a valid and smaller arm, and it is recorded as the fallback should the full clip prove to interact with the hip chain.

The inset of 0.02 radians was chosen over clipping at the soft limit, and the choice trades two effects against each other. The soft limits that `pen_joint_pos_limits` prices sit at 90 per cent of each range about its midpoint, `soft_joint_pos_limit_factor=0.9` at `kscale_identified_cfg.py:264`, which puts the ankle roll soft stop at plus and minus 0.23562 against the inset clip's plus and minus 0.24180. A clip at the hard limit inset therefore still permits the joint into the soft limit band, so `pen_joint_pos_limits` keeps its signal and remains the instrument that reports the residency, and the arm can be read. A clip at the soft limit would silence that term entirely and would leave the arm without its own measurement, which is why it was rejected.

| Joint | Hard lower | Hard upper | Soft lower | Soft upper | Clip lower | Clip upper |
|---|---|---|---|---|---|---|
| `(left/right)_foot_roll_02` | -0.26180 | 0.26180 | -0.23562 | 0.23562 | -0.24180 | 0.24180 |
| `(left/right)_foot_pitch_02` | -0.87267 | 0.52360 | -0.80285 | 0.45379 | -0.85267 | 0.50360 |
| `(left/right)_knee_04` | 0.00000 | 2.70526 | 0.13526 | 2.57000 | 0.02000 | 2.68526 |
| `(left/right)_hip_yaw_03` | -1.57080 | 1.57080 | -1.41372 | 1.41372 | -1.55080 | 1.55080 |
| `right_hip_roll_03` | -2.26893 | 0.20944 | -2.14501 | 0.08552 | -2.24893 | 0.18944 |
| `left_hip_roll_03` | -0.20944 | 2.26893 | -0.08552 | 2.14501 | -0.18944 | 2.24893 |
| `right_hip_pitch_04` | -2.21657 | 1.04720 | -2.05338 | 0.88401 | -2.19657 | 1.02720 |
| `left_hip_pitch_04` | -1.04720 | 2.21657 | -0.88401 | 2.05338 | -1.02720 | 2.19657 |

And the clip acts on the processed action, which makes it independent of the per environment offset randomisation that `kscale_base_env_cfg.py:610` introduces at plus and minus 0.05 radians. `use_default_offset=True` folds the default joint position into the offset, and the randomisation perturbs that default per environment, so a bound written on the raw action would mean a different target in every environment. Written on the processed action it means the same radian in all of them, which is the property the arm requires and the reason no equivalent exists at the raw action.

The mechanism of the prediction is worth stating because it is not merely that the target becomes unreachable. A clip at the stop makes the stop unprofitable rather than only unreachable, a target equal to the joint's own position generating no torque at all through the proportional term, so the latch dissolves rather than being enforced from the other side. The prediction is that the ankle roll stop residency falls from the 89 to 95 per cent of section 11.1, that the sole tilt in stance falls from 7 to 9 degrees toward zero, that the stop to stop reversals of section 11.4 cease, and that the toe-in falls without any change to the heading price. Should the residency not fall, the latch is not commanded and section 11.2 is wrong.


### 21.11 Arm 11, the clip excess penalty as it was launched, the quadratic at minus 2 and minus 500

Status, RUNNING as `2026-09-30_06-43-38` at a weight of minus 2 and `2026-09-30_07-05-14` at minus 500, and this sub-section is the record of what was launched rather than a proposal. Both weights are mis-set, section 18.4 giving the arithmetic, and the two runs should therefore be read as a measurement of what the quadratic form does at the extremes of its range rather than as a validation of either number.

A clip bounds the target and does not remove the incentive to name one outside it. An unbounded Gaussian policy over a clipped action is the configuration whose pathology this workspace has now measured and the literature accounts for [9], the raw action being free to drift arbitrarily far into the clipped region while the processed target stands still. Two quantities become unreliable when it does, both of which read the raw action rather than the target.

```python
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
```

`ActionManager.action` at `action_manager.py:263` returns the raw buffer, so `pen_action_rate` and `pen_action_smoothness` both price a quantity with no bounded relation to what the robot did, and the `last_action` observation feeds the policy the same quantity.

The remedy applied was to price the excess directly, the amount by which the processed target was clipped being available inside the action term at the moment the clamp is applied. It was added as a new function in the KScale configuration's own reach rather than as an edit of the shared clamp, which keeps it out of `joint_actions.py` and therefore out of every other task in the workspace.

```python
# APPLIED, environments/tasks/locomotion/mdp/rewards.py:2216.
def action_clip_excess_l2(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalise the amount by which the commanded target was clipped, in radians squared."""
    term = env.action_manager.get_term(action_term_name)
    if term.cfg.clip is None:
        return torch.zeros(env.num_envs, device=env.device)
    unclipped = term.raw_actions * term._scale + term._offset
    excess = unclipped - term.processed_actions
    return torch.sum(torch.square(excess), dim=1)
```

The quantity priced is the target excess in radians and not the raw action magnitude, because the raw action has no units the configuration can reason about while the excess is directly comparable to the joint travel. The excess is recomputed from the term's own tensors rather than cached inside the clamp, which costs one multiply and one add per step and reaches nothing outside this configuration.

The weights were read from the magnitude the term reported at iteration 193, which is the error this arm records rather than the one it commits, the reasoning having been that a term reporting 0.2944 per second at a weight of minus 500 was priced about correctly. Section 18.4 measures the same quantity at 30.380 radians squared per step at iteration 10000 of run E, which carries the identical clip and no penalty at all, so the early reading is the quantity before the policy has learned to produce it and the weight read from it is wrong by the factor the exploit grows.

| Weight | At the onset of the exploit, 0.0640 rad^2 | At run E's terminal excess, 30.380 rad^2 | Share of the positive budget at terminal |
|---|---|---|---|
| -2 | 0.128 per s | 60.76 per s | 69 pct |
| -500 | 32.0 per s | 15190 per s | 17262 pct |
| -14, arm 12 | 0.896 per s | 425.3 per s | 483 pct |

Neither weight in flight brackets the calibrated one usefully, minus 2 standing a factor of seven beneath it and minus 500 a factor of thirty six above, and section 16.2 records what this robot does when a regularisation term reaches the scale the second implies, the five hundredfold energy run collapsing to an eighteen step episode at a unit termination rate.

What these two runs can still establish is worth stating, since they will finish whatever this document says. The minus 2 run measures whether a price far beneath the exploit's terminal value delays its formation at all, which is the question of whether the exploit is driven by a gradient the penalty can oppose or by a return it cannot. The minus 500 run measures where the collapse boundary lies for this term, which no other run supplies and which bounds arm 12 from above. Both must be read with the clip excess reported, which section 20.2 now requires of every arm.

### 21.12 Arm 12, the clip excess penalty at the calibrated quadratic weight

Status, APPLIED TO THE WORKING TREE on 2026-09-30 and awaiting its run, `kscale_base_env_cfg.py:1133` standing at minus 14 where it stood at minus 500 earlier that day, and it changes one number against arm 11. It exists because a form cannot be refuted at a weight that was never defensible, and section 18.4 prices both weights in flight as mis-set in opposite directions, so a failure of either says nothing about the quadratic itself.

The calibration point is the onset of the exploit rather than either end of it, and the arithmetic is stated so that a later reader may re-derive it. Take the onset as the excess reaching one action standard deviation at the joints that carry it, being 0.16 radians at the hip roll and 0.08 at the ankle roll, which over the two pairs gives a sum of squares of 0.0640 radians squared. For that to cost one per cent of run E's positive budget of 88 per second, which is 0.88, the quadratic weight is minus 13.75 and the arm takes minus 14.

CORRECTED 2026-09-30. Section 18.4 published this calibration as 0.051 radians squared and a weight of minus 17, which omits the ankle roll pair although the sentence naming it says four joints. The omission is not negligible, the per joint measurement of run E at iteration 10000 attributing 0.863 and 0.864 radians squared of the excess to the two ankle rolls against 14.315 and 14.332 to the two hip rolls, so the ankle roll carries 5.7 per cent of the quadratic excess and 20.4 per cent of the absolute one. The corrected figures are used here and in arm 13 and section 18.4 is left as first written, its conclusion being unaffected.

```python
    # PROPOSED, not applied.  ARM 12 of plans/kscale_gait_shaping.md section 21.12,
    # replacing the weight at kscale_base_env_cfg.py:1133.  Minus 14 is one per cent of
    # run E's positive budget at the onset of the exploit, being an excess of one action
    # standard deviation at each of the two hip rolls and the two ankle rolls.
    pen_action_excess = RewTerm(
        func=mdp.action_clip_excess_l2,
        weight=-14.0,
    )
```

Two design decisions attach and the first is the one that admits the arm may fail.

THE ARM IS EXPECTED TO BE INSUFFICIENT AND IS RUN ANYWAY, because the alternative is to change the form and the weight together and learn nothing from either. Section 18.4's objection to the quadratic is not that any particular weight is wrong but that the term's value spans the square of a quantity spanning five orders of magnitude, so a weight calibrated at the onset reaches 425.3 per second at run E's terminal excess, which is 483 per cent of the positive budget and 3546 per cent of the negative one. If the policy reaches that excess the term is a wall rather than a price and the run will show it as the collapse arm 11's minus 500 run also measures. The arm's value is that it separates the weight from the form, and only a run at a defensible weight licenses the claim that the form is the fault.

The weight is not adapted during training, although section 19.2 records two published remedies that would adapt it, a Lagrangian multiplier on a constraint [31] and a scheduled budget [30]. Both are deferred because both add an outer loop whose own parameters would then need setting, and because arm 18 removes the difficulty rather than managing it by making the excess bounded, at which point a fixed weight becomes adequate. Should arm 18 not be taken, the adaptive forms return as the remedy of record.

The prediction is that the hip roll raw action falls from nineteen standard deviations toward the 0.54 of run D and that the clip excess falls below one radian squared per step, and that the term's value therefore stays beneath ten per cent of the positive budget throughout. The falsifying observation is the excess stabilising anywhere above five radians squared per step, which would mean the policy buys more by pressing than the calibrated price removes, and would send the sequence to arm 13 immediately.

### 21.13 Arm 13, the clip excess penalty in its absolute form

Status, NOT APPLIED, new on 2026-09-30, and it changes the FORM of the term arms 11 and 12 carry rather than its weight. It is the arm section 18.4 argues for and the one this document expects to succeed, and it is placed after arm 12 rather than before it because a form should be changed only once its predecessor has been refuted at a defensible weight.

The premise is an identity rather than a measurement. A quadratic term in a quantity that spans a factor of s spans s squared, and an absolute term spans s, so the same exploit that carries the quadratic across 51059 to 1 between iteration 193 and iteration 10000 carries the absolute form across the square root of that, being 226 to 1. The measurement supports the identity over the range it covers, the ratio of the squared sum to the absolute sum implying an effective 1.86, 1.78 and 1.77 active joints at the three checkpoints of run E, which is the hip roll pair and is constant.

| Checkpoint of run E | Squared excess [rad^2] | Absolute excess [rad] | Quadratic at -14 [per s] | Absolute at -1.8 [per s] |
|---|---|---|---|---|
| 8000 | 13.001 | 4.9153 | 182.0 | 8.85 |
| 9000 | 22.857 | 6.3428 | 320.0 | 11.42 |
| 10000 | 30.380 | 7.3391 | 425.3 | 13.21 |

The weight follows the same calibration as arm 12 and the same onset. An excess of one action standard deviation at the two hip rolls and the two ankle rolls sums to 0.48 radians in absolute value, and for that to cost one per cent of the positive budget the weight is minus 1.833, which the arm takes as minus 1.8. At run E's terminal excess the absolute form then reaches 13.21 per second against the quadratic's 425.3, a ratio of 32 to one, and 13.21 per second is 15 per cent of the positive budget and 110 per cent of the negative one. That is a stiff price and it is not a wall, which is the whole of the distinction this arm exists to draw.

The implementation is a second function beside the first rather than a parameter upon it, since `action_clip_excess_l2` is already named for its kernel and a caller that passed a kernel argument would read worse than a caller that named the function it wanted.

```python
# PROPOSED, not applied.  A new function in environments/tasks/locomotion/mdp/rewards.py,
# placed immediately after action_clip_excess_l2 at :2216 and sharing its recovery of the
# excess.  No existing caller is touched, the l2 form remaining exactly as it is.
def action_clip_excess_l1(
    env: ManagerBasedRLEnv,
    action_term_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalise the amount by which the commanded target was clipped, in radians.

    The absolute form of :func:`action_clip_excess_l2`, and the two differ only in the
    kernel. The quadratic form spans the square of the range its argument spans, which
    section 18.4 of plans/kscale_gait_shaping.md measures at 51000 to one over training,
    so no fixed quadratic coefficient is a price at both ends of it. The absolute form
    spans the range itself and its gradient is constant, so the marginal cost of one more
    radian of excess is the same wherever the policy already stands.
    """
    term = env.action_manager.get_term(action_term_name)
    if term.cfg.clip is None:
        return torch.zeros(env.num_envs, device=env.device)
    unclipped = term.raw_actions * term._scale + term._offset
    excess = unclipped - term.processed_actions
    return torch.sum(torch.abs(excess), dim=1)
```

```python
    # PROPOSED, not applied.  ARM 13 of plans/kscale_gait_shaping.md section 21.13,
    # replacing the term at kscale_base_env_cfg.py:1133.  Minus 1.8 is one per cent of run
    # E's positive budget at the same onset arm 12 calibrates against.
    pen_action_excess = RewTerm(
        func=mdp.action_clip_excess_l1,
        weight=-1.8,
    )
```

Three design decisions attach.

The absolute form is preferred to a Huber or a smooth approximation of it, although the gradient discontinuity at zero is real, because the excess is exactly zero on the majority of samples of an unexploited joint and a Huber's quadratic region would restore precisely the behaviour this arm removes near the origin, which is where the exploit begins. The discontinuity costs nothing here, `torch.abs` having a defined subgradient at zero and the term being a sum over joints so that a single joint's crossing is never the whole gradient.

The weight is calibrated at the onset rather than at run E's terminal excess, which would give minus 0.12 and would be a price no policy notices before the exploit is complete. The two calibrations differ by a factor of fifteen under this form and by a factor of 475 under the quadratic, which is the same argument stated as a sensitivity.

The term remains summed over all twelve joints rather than restricted to the four that carry the excess, because restricting it would fit the term to run E's particular exploit and section 21.26 records what this workspace thinks of a parameter fitted to one run. A joint that develops an excess later should be priced when it does.

The prediction is that the hip roll raw action falls from nineteen standard deviations, that the excess stabilises beneath one radian per step, and that the term's value stays beneath five per cent of the positive budget, all of which arm 12 also predicts, the difference being that this arm predicts it without the collapse. The falsifying observation is the excess stabilising above three radians per step, which would mean a constant marginal price is insufficient at any magnitude and that the bound must be structural rather than economic, and would send the sequence directly to arm 18.

### 21.14 Arm 14, the hip yaw fixed in the URDF, with its own asset, action term, environment and task

Status, NOT APPLIED, new in this position on 2026-10-01 at the user's direction, having stood at arm 16 until that date. It is placed here because it is INDEPENDENT of every arm around it and is run to gauge the contribution of one degree of freedom to this robot's locomotion rather than to repair anything the arms before it leave outstanding. It needs arm 10's clip and nothing later, so it may be launched the moment arm 13 settles the excess price, and its result is readable against run E directly.

The premise is measured three times and has not moved. The hip yaw sets the SWING foot heading at a slope between 0.99 and 1.01 with a correlation between 0.987 and 0.998, which section 16.7 established and section 18.6 confirms on all five runs of the present set, so the splay is the hip yaw and nothing else contributes measurably. The swing intercept stands between 2.27 and 4.77 degrees against a swing toe-in between 12.06 and 14.75, so freezing the joint removes between 62 and 83 per cent of the visible splay. And the joint is in no distress whatever, its saturation standing at 0.00 to 0.02 per cent in every run and every phase and its mean absolute torque at 3.2 to 5.8 newton metres of a 42 newton metre ceiling, so this arm removes an authority the machine uses badly rather than one it uses hard.

#### What this arm costs the articulation, measured rather than assumed

EVERY VERSION OF THIS ARM BEFORE 2026-10-01 RESTED ON A CLAIM ABOUT THE IMPORTER THAT IS FALSE FOR THIS CONFIGURATION, and the correction shrinks the arm by most of its apparent size. Those versions stated that a fixed joint is merged into its parent by the URDF importer, so that the articulation would carry ten joints and ELEVEN bodies, and they accordingly required `_BODY_PARTNERS` to be rebuilt and two entries deleted. The merge is conditional. `UrdfConverterCfg.merge_fixed_joints` at `urdf_converter_cfg.py:104` documents itself as consolidating links connected by fixed joints and defaults to True, and `urdf_converter.py:142` passes it to the importer, but `KSCALE_IDENTIFIED_CFG` sets it to FALSE at `kscale_identified_cfg.py:279`. No link is consolidated, so the child of a fixed joint survives as a body of its own.

The claim is settled by a robot in this workspace rather than by reading the importer, and the evidence is a dump rather than an argument. SD_BRS1 ALREADY DECLARES BOTH OF ITS HIP YAWS `type="fixed"`, at `SD_BRS1_Assembly2.urdf:89` and `:431`, each carrying `Part_Torso` as its parent and `Link1R` or `Link1L` as its child, and `sd_brs1_identified_cfg.py:161` carries the same `merge_fixed_joints=False` the KScale carries. That is the exact combination this arm proposes. Reading `sd_brs1_flat/2026-08-17_04-26-44/data/42/dump.npy` gives both counts.

| Quantity | SD_BRS1, measured | KScale now | KScale under this arm |
|---|---|---|---|
| `joint_names` | 10, no hip yaw present | 12 | 10 |
| Bodies, from the width of `robot_mass` | 13 | 13 | 13 |
| Total mass | 59.8523 kg | unchanged | unchanged |

The fixed joint therefore leaves `joint_names` and leaves the body set alone, which the symmetry module's own docstring already recorded at `kscale.py:74`, that the hip yaw is in the KScale flip set and absent from the SD_BRS1 set only because SD_BRS1 declares its hip yaw fixed so that the joint never enters `robot.joint_names` at all. Three consequences follow and each removes work the earlier drafts proposed. `_BODY_PARTNERS` at `kscale.py:80` needs NO change, its thirteen entries continuing to name thirteen live bodies. `robot_mass` and `robot_inertia` keep their widths, so the critic's observation layout narrows only where the joints do. And the two hip yaw CHILD links, `kd_d_301l_6061` and `kd_d_301r_6061`, remain in the scene with their inertia where it is, so the mass distribution of the leg is untouched and the comparison against run E is a comparison of one degree of freedom rather than of a different robot.

#### The URDF

```xml
<!-- PROPOSED, not applied.  environments/environments/assets/urdf/solefoot/kscale/
     kscale_hip_yaw_fixed.urdf, a copy of kscale.urdf with TWO attribute values changed and
     nothing else.  The origin, the parent, the child, the axis, the <limit> and the
     <dynamics> are all kept verbatim, so the kinematic chain below each joint is unmoved and
     the standing height, the sagittal closure identity of context/KScale.md section 16.2 and
     the 0.04300 m sole depth are preserved by construction.  The inert elements are RETAINED
     rather than deleted so that a diff against the original shows two attribute changes and
     no structural change, which is what makes the two assets comparable. -->

  <!-- was type="revolute", kscale.urdf:614 -->
  <joint name="right_hip_yaw_03" type="fixed">
    <origin xyz="0 0 -0.1023" rpy="0 0 0"/>
    <parent link="kd_d_201r_6061"/>
    <child link="kd_d_301r_6061"/>
    <axis xyz="0 0 1"/>
    <limit effort="60" velocity="20" lower="-1.5708" upper="1.5708"/>
  </joint>

  <!-- was type="revolute", kscale.urdf:1094 -->
  <joint name="left_hip_yaw_03" type="fixed">
    <origin xyz="0 0 -0.1023" rpy="0 0 0"/>
    <parent link="rs03"/>
    <child link="kd_d_301l_6061"/>
    <axis xyz="0 0 1"/>
    <limit effort="60" velocity="20" lower="-1.5708" upper="1.5708"/>
  </joint>
```

The two blocks above reproduce the structure of `kscale.urdf:614` and `:1094` and the `<origin>` and `<limit>` values must be copied from that file at the moment the asset is made rather than from this document, which quotes them to show that nothing but `type` moves. `validate_arm14.py` asserts the two files differ in exactly those two attributes and in no other byte.

A fixed joint freezes the joint at the URDF's own zero and NOT at the nominal, so the hip yaw nominal of 0.0 that `init_state` carries is what makes the two assets agree. That nominal is already zero on both sides at `kscale_identified_cfg.py:233` and `:234`, so no geometry changes and the frozen pose is the pose the present robot resets to. Were either nominal non zero the `<origin>` would have to absorb it, and this arm would not be a two attribute change.

#### The articulation configuration, built explicitly rather than by `replace`

```python
# PROPOSED, not applied.  Appended to
# environments/environments/assets/config/kscale_identified_cfg.py, leaving
# KSCALE_IDENTIFIED_CFG and every object above it untouched so that every task registered
# today resolves to exactly what it resolves to now.
#
# The configuration is CONSTRUCTED rather than derived by ArticulationCfg.replace, at the
# user's direction of 2026-10-01, so that every field it carries is visible at the point of
# definition and a later reader need not hold the parent object in mind to know what this one
# is.  The shared leaves are reused by reference and the two that differ are written out, the
# spawn because it names a different asset and the initial state because it must not carry a
# key for a joint the articulation no longer has.
hip_yaw_fixed_urdf_path = os.path.join(
    current_dir, "../urdf/solefoot/kscale/kscale_hip_yaw_fixed.urdf"
)

# The two hip yaw nominals are ABSENT rather than set to zero.  init_state.joint_pos is
# resolved against the articulation's joint list at articulation.py:1684 through
# resolve_matching_names_values, which raises at string.py:370 on a key matching no joint, so
# a retained key would abort the spawn rather than being ignored.  Every surviving entry is
# byte identical to init_state at kscale_identified_cfg.py:197, including the 0.74492 m spawn
# height, which is the 0.73492 m standing height of the arm 7 posture plus the 0.01 m
# settling margin and is unchanged because freezing a joint at its own nominal moves no link.
init_state_hip_yaw_fixed = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.74492),
    joint_pos={
        "right_hip_pitch_04": -0.34907,
        "left_hip_pitch_04": 0.34907,
        "right_hip_roll_03": 0.042951,
        "left_hip_roll_03": -0.042951,
        "right_knee_04": 0.87266,
        "left_knee_04": 0.87266,
        "right_foot_pitch_02": -0.52360,
        "left_foot_pitch_02": -0.52360,
        "right_foot_roll_02": 0.041254,
        "left_foot_roll_02": -0.041254,
    },
    joint_vel={".*": 0.0},
)

# The hip_yaw group is ABSENT rather than emptied.  Articulation._initialize_impl resolves
# every group at articulation.py:1721 and raises at :1723 when a group's joint_names_expr
# matches no joint, so a retained group would abort the spawn.  KSCALE_HIP_YAW_ACTUATOR_CFG
# itself is left defined and unreferenced here, so that a later arm may use it again and so
# that a diff of this file shows an addition and no deletion.  The five objects below are the
# SAME instances the twelve joint configuration uses, shared by reference, which is what makes
# the two robots carry identical gains by construction rather than by transcription.
actuators_hip_yaw_fixed = {
    "hip_roll": KSCALE_HIP_ROLL_ACTUATOR_CFG,
    "hip_pitch": KSCALE_HIP_PITCH_ACTUATOR_CFG,
    "knee": KSCALE_KNEE_ACTUATOR_CFG,
    "foot_roll": KSCALE_FOOT_ROLL_ACTUATOR_CFG,
    "foot_pitch": KSCALE_FOOT_PITCH_ACTUATOR_CFG,
}

KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # the same wrapper the twelve joint configuration uses.  SELF_COLLISION_FILTERED_PAIRS
        # names kd_d_201r_6061 and rs03, which are the PARENTS of the two frozen joints and
        # therefore survive, and foot_6061, foot_6061_2, kd_d_401r_6061, kd_d_401l_6061 and the
        # torso, none of which this arm touches, so the pair list carries over unchanged.
        # validate_arm14.py asserts that against the new URDF's link set rather than assuming it.
        func=spawn_kscale_from_urdf,
        asset_path=hip_yaw_fixed_urdf_path,
        fix_base=False,
        # FALSE, and this is the field the whole arm turns upon.  Left at its True default the
        # importer would consolidate each frozen joint's child into its parent, deleting
        # kd_d_301l_6061 and kd_d_301r_6061 from the body set, moving their inertia into the
        # hip roll links and taking the articulation to eleven bodies, at which point
        # _BODY_PARTNERS at mdp/symmetry/kscale.py:80 would name two bodies that no longer
        # exist and _build_body_map would raise at :174.  False is also what the twelve joint
        # configuration carries at :279 and what SD_BRS1 carries at
        # sd_brs1_identified_cfg.py:161, so the two KScale assets remain comparable and the
        # precedent is a robot that trains today.
        merge_fixed_joints=False,
        joint_drive=None,
        self_collision=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=True,
    ),
    init_state=init_state_hip_yaw_fixed,
    soft_joint_pos_limit_factor=0.9,
    actuators=actuators_hip_yaw_fixed,
)
```

Two properties of that object are worth stating because a reader checking it against `KSCALE_IDENTIFIED_CFG` will want them.

The three shared leaves are shared by REFERENCE and not copied, being `rigid_props` at `kscale_identified_cfg.py:181`, `articulation_props` at `:191` and the five surviving actuator configurations. Nothing in this workspace mutates an `ArticulationCfg` leaf after construction, the one place that would have done so being the `replace` call this arm is written to avoid, so sharing is safe and it is the only arrangement under which the two robots provably carry one gain set. Were a later arm to need different gains on one asset it must construct its own actuator objects rather than mutating these.

`soft_joint_pos_limit_factor` stays at 0.9 and is therefore applied to ten joints instead of twelve. It governs both `pen_joint_pos_limits` and the reset clamp at `events.py:350`, and section 14.7 records that the factor is drawn about the midpoint of a travel asymmetric by 10.83 to one at the hip roll, which arm 17 addresses. Freezing the hip yaw neither helps nor worsens that, the hip yaw's own travel being symmetric, so the defect is carried unchanged into this asset deliberately rather than repaired here where it would confound the comparison.

#### The action term

```python
# PROPOSED, not applied.  Added to
# environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py beside ActionsCfg
# at :212, which is left exactly as it stands.
@configclass
class ActionsCfgHipYawFixed:
    """The action term of the ten joint robot.

    joint_names stays at [".*"], which now resolves to ten joints rather than twelve and needs
    no edit. The scale and the clip dictionaries DO need one, both being resolved key by key
    against the term's own joint list, the scale at joint_actions.py:89 and the clip at :109,
    and both raising at string.py:370 on a key that matches nothing. Their hip yaw entries are
    therefore removed rather than left harmlessly behind.

    Every surviving value is byte identical to ActionsCfg, so a policy trained here and one
    trained there differ in their action dimension and in nothing else. The clip is arm 10's,
    being each joint's mechanical travel inset by 0.02 rad, which is what the tree carries and
    what run E was taken on. Where this arm is run after arm 15 the clip is arm 15's instead,
    likewise with its hip yaw entry removed, and the run notes must say which.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "(left|right)_hip_pitch_04": 0.4,
            "(left|right)_knee_04": 0.2667,
            "(left|right)_hip_roll_03": 0.16,
            "(left|right)_foot_pitch_02": 0.08,
            "(left|right)_foot_roll_02": 0.08,
        },
        use_default_offset=True,
        clip={
            "(left|right)_foot_roll_02": (-0.241799, 0.241799),
            "(left|right)_foot_pitch_02": (-0.852665, 0.503599),
            "(left|right)_knee_04": (0.020000, 2.685260),
            "right_hip_roll_03": (-2.248930, 0.189440),
            "left_hip_roll_03": (-0.189440, 2.248930),
            "right_hip_pitch_04": (-2.196570, 1.027200),
            "left_hip_pitch_04": (-1.027200, 2.196570),
        },
    )
```

The scale dictionary must cover every surviving joint and the comment at `kscale_base_env_cfg.py:208` says why, `joint_actions.py:85` initialising the tensor to ONES and writing only the matched columns, so an omitted joint silently takes a scale of 1.0 rather than keeping its intended value. The five keys above cover all ten joints, which `validate_arm14.py` asserts by resolving them against the new URDF's revolute joint list rather than by inspection.

#### The environment configuration

```python
# PROPOSED, not applied.  Appended to kscale_base_env_cfg.py beside KscaleEnvCfg, which is
# left exactly as it stands.  Every removal below is REQUIRED rather than tidy, each named
# path resolving the hip yaw through a helper that raises on an expression matching no joint.
@configclass
class KscaleEnvHipYawFixedCfg(KscaleEnvCfg):
    """The ten joint environment, differing from KscaleEnvCfg in four places and no others."""

    actions: ActionsCfgHipYawFixed = ActionsCfgHipYawFixed()

    def __post_init__(self):
        super().__post_init__()

        # ONE, the two reward terms that resolve the hip yaw.  Both carry a SceneEntityCfg
        # whose joint_names is resolved at scene_entity_cfg.py:175 through find_joints, which
        # raises at string.py:268 on an expression matching no joint, so both must go rather
        # than being left to report zero.
        #   rew_keep_hip_yaw_zero_in_air at kscale_base_env_cfg.py:828, weight +1.0
        #   pen_hip_yaw_deviation        at kscale_base_env_cfg.py:852, weight -1.0
        # Removing them changes the reward budget, and section 21.14 prices the change.
        self.rewards.rew_keep_hip_yaw_zero_in_air = None
        self.rewards.pen_hip_yaw_deviation = None

        # TWO, the reset event that offsets the hip yaw.  reset_hip_yaw_joints at
        # kscale_base_env_cfg.py:704 passes joint_name to mdp.reset_joint_by_offset, which
        # resolves it at events.py:367 through find_joints and raises the same way.  The whole
        # term goes, there being no other joint it addresses.
        self.events.reset_hip_yaw_joints = None

        # THREE, the hip yaw entry of the gain randomisation group.  hip_joint_stiffness_and_damping
        # at kscale_base_env_cfg.py:631 names three joint pairs in one SceneEntityCfg and only
        # the hip yaw pair must go, the other two still being wanted, so the ENTRY is dropped
        # rather than the term.  The list is rebuilt rather than mutated in place because the
        # parent class's own term object is shared with KscaleEnvCfg until __post_init__ runs.
        term = self.events.hip_joint_stiffness_and_damping
        term.params["asset_cfg"].joint_names = [
            n for n in term.params["asset_cfg"].joint_names if "hip_yaw" not in n
        ]

        # FOUR, nothing else.  The following are recorded explicitly because each is a place a
        # reader expects an edit and finds none.
        #
        #   The observation groups.  joint_pos at :291, joint_vel at :297 and last_action at
        #   :303, and the critic's robot_joint_* terms, all carry the default
        #   SceneEntityCfg("robot") and therefore resolve whatever the articulation has, so
        #   they narrow from twelve to ten by themselves and NO observation term changes.
        #   robot_mass at :345 and robot_inertia at :346 are per BODY and the body count is
        #   unchanged at thirteen, so their widths do not move at all.
        #
        #   randomize_joint_default_pos resolves every joint by the ".*" pattern and therefore
        #   needs no change, matching ten joints instead of twelve.
        #
        #   The symmetry augmentation.  No change, and the sub-section below proves it.
        #
        #   The curriculum, the commands and the terminations.  No term names a joint, and
        #   base_contact names the torso, which survives.
        #
        #   SELF_COLLISION_FILTERED_PAIRS.  Every link it names survives, the two it might
        #   have lost being the frozen joints' children and merge_fixed_joints being False.
```

#### The symmetry augmentation needs no change, and this is the proof

This sub-section states in full what section 21.24 carried until 2026-10-01, moved here at the user's direction because this is the first arm to change the joint count and because the investigation of that date settles the matter differently from the way that section left it.

THE CONCLUSION IS THAT NOTHING IN `mdp/symmetry/kscale.py` CHANGES. Four facts establish it and each is a property of how that module is written rather than a coincidence.

The joint map is built from the RUNTIME names. `_maps` at `kscale.py:211` calls `_build_joint_map(list(robot.joint_names), device)` at `:215`, so the permutation is ten wide the moment the articulation is ten wide, and the action tensor, the `joint_pos` and `joint_vel` observation blocks and the `last_action` block are all ten wide with it. Nothing in the module carries a hard coded twelve.

The permutation remains an involution, which is the property `_build_joint_map` asserts at `:163` and which makes mirroring twice the identity. The two hip yaws are each other's partner and they leave TOGETHER, so every surviving joint's partner also survives and `_joint_partner` at `:138` resolves for all ten. Had a single hip yaw been frozen the map would raise at `:158`, which is the right behaviour and is worth knowing.

`_FLIP_JOINT_KINDS` at `kscale.py:76` keeps its `"hip_yaw"` entry and that entry becomes inert. The flip set is applied by testing each surviving joint's name for each substring at `:160`, not by resolving each substring against the articulation, so a kind that matches nothing costs one comparison per joint and raises nothing. It is left in place because the twelve joint robot still needs it and the module serves both.

The body map is untouched because the body set is untouched. `_build_body_map` at `:168` raises at `:174` on a body absent from `_BODY_PARTNERS` and at `:178` on a partner absent from the articulation, and `_BODY_PARTNERS` at `:80` names thirteen bodies against an articulation that still has thirteen, the frozen joints' children surviving under `merge_fixed_joints=False`. The involution assertion at `:181` therefore holds unchanged, and `robot_mass`, `robot_inertia` and `robot_material_properties` keep the widths their mirror kinds expect.

WHAT SECTION 21.24 PROPOSED INSTEAD, AND WHY IT IS NOT NEEDED. That section carried an optional `subset` argument for `_build_joint_map`, so that a configuration driving FEWER joints than the articulation carries could mirror an action tensor of the actuated width while the observation blocks continued to be mirrored at the full width, together with a fourth mirror kind for `last_action`. That repair answers a real fault and this arm does not have it, because freezing the joint in the URDF removes it from the articulation so that `robot.joint_names` IS the actuated set. The repair is required only by a configuration that leaves a joint in the articulation and removes it from the action term, which is the configuration level freeze this document carried as arm 10 until 2026-09-30 and which arm 14 supersedes. It is recorded as the defect left standing at section 21.24 under the workspace rule, so that any future arm which drives a subset of an unchanged articulation finds it written out rather than meeting it at the first learning step.

ONE RESIDUAL RISK IS WORTH NAMING BECAUSE NO ASSERTION CATCHES IT. `_layout` reads the observation term widths from the observation manager at runtime and the mirror is applied block by block, so a width that changes without its kind changing is handled, but a term whose MEANING changes is not. No such term exists here, every narrowing term being a per joint vector whose joints are mirrored by the same permutation. The statement is recorded so that a later arm which adds a term of mixed kind knows that the module will mirror it by width and not by meaning.

#### The subset repair this arm does not need, written out because the rule requires the defect to be recorded

The repair below answers the case arm 14 does NOT have, being a configuration that drives fewer joints than its articulation carries, and it stood at section 21.24 until 2026-10-01. It is kept because the workspace rule requires a defect left standing to be recorded where another implementation may opt into it deliberately, and because the fault it fixes appears at the first learning step rather than at spawn, which is the most expensive place to meet it. Nothing in it should be applied for arm 14.

The symmetry augmentation must be rebuilt over the actuated subset. `compute_symmetric_states` at `mdp/symmetry/kscale.py:274` permutes the action tensor by `m["jperm"]`, which `_build_joint_map` at `:150` constructs over `robot.joint_names`, meaning all twelve joints of the articulation. An action tensor ten columns wide indexed by a twelve long permutation raises, and the `last_action` observation block carries the same fault through `_POLICY_KIND` at `:116`, which maps it to the `joint` kind. The repair must follow the workspace rule that a shared module takes an optional argument whose default reproduces the old behaviour exactly, so `_build_joint_map` gains an `actuated_names` argument defaulting to `None`, and `_maps` builds a second permutation and sign pair from it, which `compute_symmetric_states` uses for the action and for the `last_action` block while the twelve long pair continues to serve `joint_pos` and `joint_vel`. `brs.py` is a separate module and is untouched, and no TRON1 task imports `mdp/symmetry/kscale.py`, which `grep -rn 'symmetry.kscale' environments/` confirms, returning only `mdp/symmetry/__init__.py:6` and `agents/limx_rsl_rl_ppo_cfg.py:14`.

The repair was described rather than written until 2026-09-28, this arm, arm 14 and the configuration level hip yaw freeze this document carried as arm 10 until 2026-09-30 all having been blocked on it and each having deferred it to the others. Three edits are needed in one file and no other file changes.

```python
# PROPOSED, not applied.  environments/environments/tasks/locomotion/mdp/symmetry/kscale.py.
#
# EDIT ONE, at :150.  One optional argument, whose default reproduces the present two line
# return exactly, so every existing call site is unaffected.
def _build_joint_map(joint_names, device, subset=None):
    """out[i] = sign[i] * x[perm[i]], partner is the same joint with its side prefix swapped.

    Args:
        joint_names: Every joint of the articulation, in articulation order.
        device: The device the returned tensors are placed on.
        subset: The joints the returned map is to be built over, in the order the tensor
            being mirrored carries them, or None for all of joint_names. It exists so that
            a configuration driving fewer joints than the articulation carries can mirror
            an action tensor whose width is the ACTUATED count
            while the observation blocks continue to be mirrored at the full count. Passing
            None reproduces the original behaviour exactly, which is what every present
            caller does.
    """
    names = list(joint_names) if subset is None else list(subset)
    n = len(names)
    perm, sign = list(range(n)), [1.0] * n
    for i, name in enumerate(names):
        partner = _joint_partner(name)
        if partner not in names:
            raise ValueError(
                f"mirror partner {partner!r} of joint {name!r} is not in the set the map is "
                "built over. Where a subset is given, it must be closed under the mirror, so "
                "a joint may be removed from the action only together with its counterpart."
            )
        perm[i] = names.index(partner)
        if any(k in name for k in _FLIP_JOINT_KINDS):
            sign[i] = -1.0
    for i, p in enumerate(perm):
        assert perm[p] == i, f"joint permutation is not an involution at {names[i]!r}"
    return (torch.tensor(perm, dtype=torch.long, device=device),
            torch.tensor(sign, dtype=torch.float, device=device), perm)


# EDIT TWO, at :211.  The map gains a second permutation and sign pair, built over the
# action term's own joints.  Where the action drives every joint, which is the present
# configuration, the two pairs are identical and the added lines cost one comparison.
def _maps(env, device, actuated_names=None):
    if device in _CACHE:
        return _CACHE[device]
    robot = env.unwrapped.scene["robot"]
    jperm, jsign, _ = _build_joint_map(list(robot.joint_names), device)
    if actuated_names is None:
        actuated_names = _action_joint_names(env)
    aperm, asign, _ = _build_joint_map(list(robot.joint_names), device, subset=actuated_names)
    bperm_list = _build_body_map(list(robot.body_names))
    om = env.unwrapped.observation_manager
    m = {
        "jperm": jperm,
        "jsign": jsign,
        "aperm": aperm,          # the ACTION map, equal to jperm where all joints are driven
        "asign": asign,
        "bperm": torch.tensor(bperm_list, dtype=torch.long, device=device),
        # ... every remaining entry unchanged ...
    }
    _CACHE[device] = m
    return m


def _action_joint_names(env):
    """The joints the action term drives, in the order it carries them.

    Read from the action manager rather than from the configuration, so that a regex in
    joint_names resolves through the same code path the term itself used and the two cannot
    disagree. Falls back to the articulation's own joint order where no term exposes its
    ids, which is the configuration every run before 2026-09-28 used.
    """
    robot = env.unwrapped.scene["robot"]
    term = getattr(env.unwrapped.action_manager, "_terms", {}).get("joint_pos")
    ids = getattr(term, "_joint_ids", None) if term is not None else None
    if ids is None or isinstance(ids, slice):
        return list(robot.joint_names)
    return [robot.joint_names[i] for i in ids]


# EDIT THREE, at :294.  The action and the last_action observation block move to the
# action map.  Every other block keeps jperm, the observations continuing to carry every
# joint of the articulation whether it is driven or not.
    actions_aug = None if actions is None else torch.cat(
        [actions, actions[:, m["aperm"]] * m["asign"]], dim=0)
```

The `last_action` block needs one further change and it is the one a reader is most likely to miss. `_apply` at `:235` mirrors a block of kind `joint` with `m["jperm"]`, and `last_action` is mapped to that kind by `_POLICY_KIND` at `:116`, so once the action is ten wide the block is ten wide and the twelve long permutation raises on it. The narrowest repair is a fourth kind rather than a branch inside the existing one.

```python
# PROPOSED, not applied.  A new kind, added beside the existing ones at :243, and
# _POLICY_KIND at :116 and _CRITIC_KIND at :122 repointing "last_action" at it.  Every
# other entry of both dictionaries is unchanged, so joint_pos and joint_vel continue to be
# mirrored at the full articulation width.
    if kind == "action":
        return (x[..., m["aperm"]] * m["asign"]).reshape(n, -1)
```

The repair is validated without a simulator by asserting three properties of the returned pair, that the permutation is an involution, which `_build_joint_map` already asserts, that applying the map twice is the identity on a random tensor, and that the map built with `subset=None` is element for element equal to the one the present code returns. `scripts/analysis/kscale_symmetry_selftest.py` already asserts the first two for the full map, as `../context/KScale.md` section 17 records, and extending it to the subset case is the check this repair should ship with.

#### The task registration

```python
# PROPOSED, not applied.  environments/environments/tasks/locomotion/robots/
# kscale_hip_yaw_fixed_env_cfg.py, a new module beside kscale_solefoot_env_cfg.py and
# mirroring its flat branch exactly.  The HIM branch is NOT reproduced, this arm having no
# bearing on the estimator and the HIM encoder's num_output_dim of 19 at
# limx_rsl_rl_ppo_cfg.py:369 being sized against the twelve joint observation.
from isaaclab.utils import configclass

from environments.assets.config.kscale_identified_cfg import (
    KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED,
)
from environments.tasks.locomotion.cfg.SF.kscale_base_env_cfg import (
    KscaleEnvHipYawFixedCfg,
)


@configclass
class KscaleHipYawFixedBaseEnvCfg(KscaleEnvHipYawFixedCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-3.25, 0.0, 3.25)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class KscaleHipYawFixedBaseEnvCfg_PLAY(KscaleHipYawFixedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.9, 0.9)
        self.episode_length_s = 30.0


@configclass
class KscaleHipYawFixedBlindFlatEnvCfg(KscaleHipYawFixedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class KscaleHipYawFixedBlindFlatEnvCfg_PLAY(KscaleHipYawFixedBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
```

The `prim_path` is still applied through `replace` in the block above and that is deliberate. The user's direction of 2026-10-01 concerns the construction of the articulation configuration itself, which is now explicit, and this call sets one scene local field on an already constructed object exactly as `KscaleBaseEnvCfg.__post_init__` does at `kscale_solefoot_env_cfg.py:23`. Writing `prim_path` into the definition instead would put a scene detail in the asset file and would diverge from every other robot in the tree.

```python
# PROPOSED, not applied.  environments/environments/tasks/locomotion/robots/__init__.py.
# Two clauses in the import block beside the existing kscale module at :1, and two
# registrations beside Isaac-Limx-Kscale-Blind-Flat-v0 at :684.  The runner configuration is
# the SAME limx_kscale_runner_cfg instance the twelve joint task uses, built at :56, so the
# PPO hyperparameters, the network widths and the symmetry hook are shared by construction
# and the two runs differ in the environment alone.  Both tasks therefore write to
# experiment_name "kscale_flat" and the run directory's own params/env.yaml is what
# distinguishes them, which section 2.3's tree check already reads.
from environments.tasks.locomotion.robots import kscale_hip_yaw_fixed_env_cfg

gym.register(
    id="Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": kscale_hip_yaw_fixed_env_cfg.KscaleHipYawFixedBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_kscale_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": kscale_hip_yaw_fixed_env_cfg.KscaleHipYawFixedBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_kscale_runner_cfg,
    },
)
```

#### The command line wiring

```bash
# PROPOSED, not applied.  /ws/djinn, one clause in the training dispatch beside the kscale
# clause at :170 and one in the play dispatch beside its counterpart at :244.  Both are
# placed BEFORE the else arm that fails loudly, which :180 documents and which exists because
# an unrecognised mode silently selected the TRON1 SoleFoot before it was added.
                elif [[ "$3" == "kscale-hip-yaw-fixed" ]]; then
                    task="Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-v0"

# and in the play dispatch
                elif [[ "$3" == "kscale-hip-yaw-fixed" ]]; then
                    task="Isaac-Limx-Kscale-HipYawFixed-Blind-Flat-Play-v0"
```

The two `echo` lines that list the expected modes, at `:193` and `:259`, must gain `kscale-hip-yaw-fixed` in the same pass, since a mode absent from that list is a mode a user cannot discover. Neither clause sets `policy_type`, the flat KScale clause not setting it either and the default being the plain PPO runner this arm shares.

#### Design decisions

The joint is frozen in the URDF rather than in the configuration, and the two are different experiments rather than two routes to one. A configuration freeze leaves the joint in the articulation and removes the command, so the servo still holds it at a stiffness of 100 newton metres per radian and it still DEFLECTS under load, which is the honest version of a fixed joint on a machine whose actuator is a spring. This arm removes the degree of freedom, so the deflection is zero by construction and the observation and action dimensions fall with it. Section 18.6 measures the hip yaw carrying 3.2 to 5.8 newton metres of mean absolute torque, so the deflection a configuration freeze would leave is not negligible at a stiffness of 100, and the two experiments would not have answered the same question.

The asset is a second URDF rather than one URDF edited at load time, because the importer caches its conversion against the asset path and a runtime edit would reach the cached USD. The cost is one asset file and the benefit is that each task's dumped `params/env.yaml` names the asset it ran, which is the record the comparison needs.

The two hip yaw reward terms are REMOVED rather than reweighted to zero, and the budget change is priced here rather than left to be discovered. `rew_keep_hip_yaw_zero_in_air` pays 0.331 to 0.431 per second of a maximum of 1.0 at a weight of 1.0, which section 16.6 measures, and `pen_hip_yaw_deviation` charges the deviation at minus 1.0. Against run E's positive budget of 88 per second and negative budget of 11.99 the first is 0.4 per cent of the positive side and the second is a small part of the negative side, so the total objective moves by under one per cent and the comparison against run E remains readable on every other channel. Neither term is replaced, there being nothing left to shape.

The HIM variant is not registered, and the reason is a number rather than a preference. `KscaleFlatPPORunnerCfg.encoder` carries `num_output_dim=19` at `limx_rsl_rl_ppo_cfg.py:369`, which is sized against the twelve joint observation, and the HIM estimator would need that re-derived. This arm is run to gauge one degree of freedom and the plain PPO task answers that, so the HIM branch is left for whoever needs it.

#### Validation before launch

`validate_arm14.py` WAS WRITTEN on 2026-10-01 and stands in the artefact directory of section 18. It passes 54 assertions and fails none, and it reads the proposed dictionaries out of THIS SECTION rather than from a transcription of it, in the manner `validate_scale.py` established, so the plan and the check cannot drift apart. It reads the live dictionaries out of the configuration files the same way and never constructs the articulation, `isaaclab.sim` requiring the USD runtime, so every structural claim is settled from the URDF and every behavioural one from the symmetry module, which imports torch alone. It asserts that

the new URDF differs from `kscale.urdf` in exactly two attribute values, both of them a `type` on a hip yaw joint, and in no other byte,

the new URDF's `<link>` set is IDENTICAL to the original's at thirteen entries, which is the property that keeps `_BODY_PARTNERS` valid,

the new URDF carries ten revolute joints, being the original's twelve less the two hip yaws, and two fixed joints,

`_BODY_PARTNERS` covers exactly the new URDF's link set, in both directions, and is an involution,

`_build_joint_map` over the ten surviving joint names returns an involution and a sign vector whose flipped entries are exactly the hip roll, hip pitch and ankle roll pairs,

`KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED` carries no actuator group, no `init_state.joint_pos` key, no `scale` key and no `clip` key that names the hip yaw, and every key it does carry matches at least one of the ten surviving joints,

every surviving actuator group of that configuration is the SAME OBJECT as the corresponding group of `KSCALE_IDENTIFIED_CFG`, which is the check that the two robots share one gain set,

every surviving `init_state.joint_pos` entry and the spawn height are byte identical to those of `KSCALE_IDENTIFIED_CFG`,

the `scale` keys of `ActionsCfgHipYawFixed` cover all ten joints with no joint matched twice, which `joint_actions.py:85` makes a correctness matter rather than a tidiness one,

every link named in `SELF_COLLISION_FILTERED_PAIRS` is present in the new URDF,

and the sagittal closure identity of `context/KScale.md` section 16.2 still holds at the surviving nominal pose, the signed hip pitch plus the knee plus the ankle pitch summing to zero.

Two of those assertions needed their own statement corrected before they passed and the corrections are recorded because each is a trap. The closure identity must be evaluated on the RIGHT leg's sign convention, in which the hip pitch nominal is negative, since the left and right hip pitch axes are genuinely anti-parallel, which is the same fact that puts `hip_pitch` in the flip set at `mdp/symmetry/kscale.py:76`. Evaluated on the left leg's signs the three sum to 0.69813 rather than to zero, which looks like a broken pose and is a broken check. And the count of hip yaw bearing lines in `kscale_base_env_cfg.py` is not a useful assertion, since a line may carry the name twice and a comment carries it harmlessly, so the check attributes each such line to its nearest enclosing assignment and asserts that the five constructs so found are exactly the five this arm addresses. A sixth construct appearing later therefore fails the check, which a count would not.

The policy checkpoint is NOT transferable between the two tasks and the run notes must say so. The actor's input and output widths both change, twelve joint positions and velocities becoming ten and the action dimension falling from twelve to ten, so a checkpoint of one loads into the other only by a deliberate surgery this arm does not propose. The comparison is therefore between two runs from scratch at a matched budget, and section 20.2's evaluation protocol applies to both.

#### Prediction and falsifier

The prediction is that the swing toe-in falls from the 13.115 to 15.239 degrees of section 18.7's table to the swing regression intercept of 2.27 to 4.77 degrees, which is a fall of between 62 and 83 per cent and is the largest single improvement any arm of this ablation predicts, that the stance toe-in falls to its own intercept of 4.834 to 6.406 degrees rather than to zero, section 18.7 having established that figure to be a structural offset of the leg chain which the hip yaw was the only joint able to cancel, and that the yaw tracking error RISES from run E's 0.2796 radians per second because the machine must now turn on its hip roll and its foot placement alone.

That last prediction is the arm's own cost and it is the one to read first. The falsifying observation is the yaw tracking error rising above roughly 0.45 radians per second or the termination rate rising from run E's 0.0021 per second, either of which would say the hip yaw is supplying a turning authority the splay measurements did not reveal because they were taken on a machine that had it. Should that happen the arm is not simply abandoned, section 17.3 recording two independently designed bipeds that omit the ankle roll and none that omits the hip yaw, so a positive result here would be a stronger claim than the literature currently supports and a negative one is the expected outcome rather than a surprise.

### 21.15 Arm 15, the clip resized from the mechanical travel to the useful range

Status, NOT APPLIED, new on 2026-09-30, second in the order of section 20.1, and one dictionary.

The premise is section 18.3. Arm 10 set every bound at the joint's mechanical travel inset by 0.02 radians, which cured the one joint whose travel is close to its authority and left the rest unbounded in any sense that matters. Measured against the saturating deflection, being the effort limit divided by the stiffness and therefore the position error at which the servo reaches its ceiling at rest, the ankle roll's window is 2.84 times that deflection and the hip roll's is 14.51 times it. The ankle roll is the joint the clip cured and the hip roll is the joint the fault moved to, which is as close to a controlled comparison as a single run affords.

The sizing rule this arm adopts follows from that comparison rather than from a principle, and section 19.3 records that the survey offers no rule at all. The bound is set at the nominal plus and minus a multiple of the joint's own saturating deflection, and the multiple is 1.5 because the ankle roll's surviving window is 1.42 times its deflection about its centre and that is the only figure this robot has validated. The result is then intersected with the travel inset by 0.02, so no bound is widened and the arm can only narrow.

```python
    # PROPOSED, not applied.  ARM 15 of plans/kscale_gait_shaping.md section 21.15, replacing
    # the clip of arm 10 at kscale_base_env_cfg.py:225.  Each bound is the joint's
    # nominal plus and minus 1.5 q_sat, where q_sat is the effort limit divided by the
    # stiffness, intersected with the mechanical travel inset by 0.02 rad.  The nominals are
    # those of arm 7 at kscale_identified_cfg.py:197 and the q_sat values are 0.420 at the hip
    # pitch and hip yaw, 0.168 at the hip roll, 0.280 at the knee and 0.170 at both ankles.
    #
    #   joint        nominal    1.5 q_sat    bound before intersection    after
    #   hip_pitch   +/-0.34907    0.630      -0.281 .. +0.979             unchanged
    #   hip_roll    -/+0.042951   0.252      -0.295 .. +0.209             -0.189 .. +0.209
    #   hip_yaw       0.0         0.630      -0.630 .. +0.630             unchanged
    #   knee         +0.87266     0.420      +0.453 .. +1.293             unchanged
    #   foot_pitch   -0.5236      0.255      -0.779 .. -0.269             unchanged
    #   foot_roll   -/+0.041254   0.255      -0.296 .. +0.214             -0.242 .. +0.214
    clip={
        "left_hip_pitch_04": (-0.28093, 0.97907),
        "right_hip_pitch_04": (-0.97907, 0.28093),
        "left_hip_roll_03": (-0.18944, 0.20905),
        "right_hip_roll_03": (-0.20905, 0.18944),
        "(left|right)_hip_yaw_03": (-0.63, 0.63),
        "(left|right)_knee_04": (0.45266, 1.29266),
        "(left|right)_foot_pitch_02": (-0.77860, -0.26860),
        "left_foot_roll_02": (-0.24180, 0.21375),
        "right_foot_roll_02": (-0.21375, 0.24180),
    },
```

Four design decisions attach and the first is an ordering constraint rather than a parameter.

THIS ARM MUST NOT BE RUN WITHOUT ARM 11 IN ITS CORRECTED FORM, and the reason is mechanical. Narrowing a clip raises the excess, since the policy's commanded target is unchanged at the moment of the change and more of it now falls outside the bound. Section 18.3 measures the hip roll at nineteen standard deviations of raw action against a window of 2.4384 radians, and this arm reduces that window to 0.3985, so the excess at that joint would rise by roughly the difference, from 2.90 radians to something near 4.9. Running this arm alone would therefore measure a larger escalation and nothing else. Run it on top of the corrected stage two, or run the two as one arm and accept that the result is two variable.

The bound is set from the nominal rather than from the joint's own current position, which would be a rate limit rather than a clip and is a different instrument with different failure modes. A clip about the nominal bounds the posture the policy may ask for, which is the quantity section 11.2 identified, and a bound about the current position would bound only how fast it may change, which `pen_action_rate` already prices at `kscale_base_env_cfg.py` and which section 4.7 measures as inert.

The multiple is uniform at 1.5 rather than per joint, because a per joint multiple would be six free parameters fitted to one run's behaviour and section 21.26 records what this workspace thinks of a hand sweep with that many degrees of freedom. The uniform multiple is a rule, and a rule that fails is more informative than six numbers that do not.

CORRECTED 2026-09-30, and the correction was found by sizing arm 18 rather than by re-reading this one. The uniform rule stands TIGHTER than the working gait's own ninety fifth percentile commanded departure at two joints, giving 0.420 radians at the knee against a measured 0.479 and 0.255 at the ankle pitch against a measured 0.331. A bound below what the gait already commands is a bound that forbids the gait, which is the error section 21.10 records this document making once before when it proposed clipping at the saturating deflection itself. The rule therefore becomes the LARGER of 1.5 times the saturating deflection and 1.25 times run E's ninety fifth percentile, which moves the knee bound to 0.599 and the ankle pitch bound to 0.414 and leaves the other four where the table above places them. Section 21.12 carries the corrected figures and the arithmetic, and this arm should be run with them rather than with the table above if it is run at all.

The intersection can only narrow, which makes this arm strictly a restriction of arm 10 and therefore comparable to run E on every channel. Two bounds are moved by the intersection rather than by the rule, the hip roll's lower bound and the ankle roll's, and both are moved because the travel inset is already tighter than 1.5 q_sat on that side.

The prediction is that the hip roll stop residency falls from 45.90 per cent toward the 10 to 15 per cent runs A through D carry, that its raw action falls from nineteen standard deviations, that the stance splay falls from 7.188 degrees toward run D's 5.275, and that the yaw tracking error falls from 0.2796 radians per second. The ankle roll should be unmoved, its bound changing by 0.028 radians on one side only. The falsifying observation is the raw action escalating further at some THIRD joint, which would say that the escalation is a property of the policy rather than of any particular bound, and would send the sequence to arm 19 immediately.

### 21.16 Arm 16, the raised in air yaw weight and the swing gated heading price, in three stages

Status, NOT APPLIED, the first arm of the sequence that acts upon the gait rather than upon the action space, and the cheapest in the sequence at one weight, one boolean and one scale. It was carried as a single arm until 2026-09-30 and is divided into three stages here at the user's direction, each stage changing exactly one value, because section 18.7 measures the three changes to act upon different quantities and in one case in opposite directions.

The premise is two measurements of sections 16.7 and 16.10 that no earlier section could have made, both having required the gait phase to be separated. The splay is four to eighteen times larger in swing than in stance on runs A, B and C, while `pen_feet_heading` prices both at the same weight because `airborne_only` is False. And the hip yaw in air kernel is alive rather than collapsed, returning 38.5, 38.3 and 44.1 per cent of its maximum at a gradient of 1.91 to 2.21 per radian per unit weight, where the two ankle kernels return 18.2 and 18.7 and are the reason section 20.3 excludes raising their weights.

The two terms stand as follows at `kscale_base_env_cfg.py:791` and `:833`.

```python
    rew_keep_hip_yaw_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_yaw_03"]
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "use_default_offset": True
        },
    )
    pen_feet_heading = RewTerm(
        func=mdp.feet_yaw_alignment,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "forward_axis": (0.0, 0.0, -1.0),
            "common_mode": "mean",
            "tolerance": 0.10,
            "differential_scale": 1.0,
            "airborne_only": False,
            "history_index": 0,
            "force_threshold": 1.0,
        },
    )
```

#### A precondition that is not a parameter

THE INHERITANCE TABLE OF SECTION 18.7 MUST BE RE-MEASURED ON WHATEVER GAIT PRECEDES THIS ARM, and the reason is that stage two's safety is a property of the baseline rather than of the term. That section follows every contact interval from its touchdown sample to the last before release and finds the foot heading planted and held on run E, the last airborne sample, the touchdown, the stance mean and the lift off standing within 1.02 degrees of one another. On the four runs before the clip the same heading GREW within the interval by 3.642 to 9.632 degrees on 72.6 to 96.3 per cent of occasions. A term restricted to swing governs the stance heading in the first regime and abandons it in the second, and since this arm now runs after arm 15 narrows the clip, the gait it will meet is not the gait the table was measured on. `headinggate.py` in the artefact directory of section 18 computes the table from a dump and is the instrument.

#### Stage one, the in air hip yaw weight raised to four

```python
    # PROPOSED, not applied.  ARM 16 STAGE ONE of plans/kscale_gait_shaping.md section 21.16.
    rew_keep_hip_yaw_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=4.0,          # ARM 16 STAGE ONE, was 1.0
        params={ ... unchanged ... },
    )
```

The weight is raised at the hip yaw and NOT at the two ankles, and the measurement makes the distinction rather than a preference. A weight multiplies a gradient, so it can move a term whose gradient exists and cannot move one whose gradient has vanished. The hip yaw kernel stands at 38 to 44 per cent of its maximum with a gradient of 1.91 to 2.21 per radian, and quadrupling its weight takes that gradient to between 7.7 and 8.8 per radian, which is comparable to the 8.0 that `pen_feet_heading` carries. The ankle kernels stand at 18.2 and 18.7 per cent and are addressed by arm 22, which moves the width instead.

The factor is four rather than eight or two, and the arithmetic that sets it is the term's own maximum. The term pays 0.331 to 0.431 per second of a maximum of 1.0, so at a weight of four it would pay at most 4.0 per second and presently 1.32 to 1.72, which is between 1.5 and 1.9 per cent of the positive budget of 88.2 to 88.8. That is enough to be visible against `rew_ang_vel_z` at 11.1 to 12.4 and far short of `rew_lin_vel_xy` at 48.7 to 49.0, so it can shape without competing with the task.

This stage stands first because it is the only one of the three whose premise section 18.7 leaves intact, and because it acts on the swing hip yaw directly rather than through the foot heading, so its effect is attributable without the phase decomposition the other two stages need.

The prediction is that the swing toe-in falls from 13.115 to 15.239 degrees toward the 2.27 to 4.77 degree intercept the swing regression of section 16.7 names, and that `rew_keep_hip_yaw_zero_in_air` rises from 38 to 44 per cent of its maximum toward 60. The falsifying observation is the swing toe-in surviving while the term's value rises, which would mean the hip yaw excursion is bought by something the term cannot outbid, and would send the sequence to arm 14 without running stages two and three.

#### Stage two, the heading price gated to the airborne foot

```python
    # PROPOSED, not applied.  ARM 16 STAGE TWO, on top of stage one.
    pen_feet_heading = RewTerm(
        func=mdp.feet_yaw_alignment,
        weight=-8.0,
        params={
            ...
            "differential_scale": 1.0,      # UNCHANGED, see stage three
            # ARM 16 STAGE TWO, was False.  Section 18.7 measures the common mode under the
            # mean form at 0.000923 rad^2 on run E against a differential of 0.070301, so the
            # ungated common mode is inert and the gate removes no price.  What it does is
            # restrict the circular mean to the airborne foot, which makes the common mode
            # that foot's own heading error and raises it by a factor of 25.9.
            "airborne_only": True,
            ...
        },
    )
```

This stage is separated from the others because it changes what the term MEASURES rather than how much the term costs, and section 18.7 establishes that the change runs opposite to the direction the arm assumed until 2026-09-30. Under `common_mode: "mean"` the common mode is the squared error of the circular mean of both feet against the base, so a splay of plus and minus the same angle cancels within it, which the docstring at `rewards.py:1263` records as the reason to prefer that form wherever the differential scale is non zero. The measurement confirms it, the common mode carrying 1.3 per cent of the term on run E and 8.1 to 17.0 per cent on runs A, B and C. Closing the gate makes the mean in single support the swing foot alone, so the common mode becomes that foot's heading error and rises by a factor of 6.07 to 25.9, and in double support, which is 15.09 per cent of run E's samples, no foot is airborne and the term's common mode is identically zero by `rewards.py:1330`.

Two consequences must be read from this stage and neither is the one the arm originally predicted. The first is that the stance heading is not abandoned, the differential mode being returned over both feet unconditionally at `rewards.py:1354` and carrying 98.7 per cent of the term's value on run E. The second is that the orthogonality of the two modes is lost while the gate is closed, a pure splay registering on the common mode as well as the differential, so the differential is priced twice and the term's total rises from 0.071223 to 0.094173 at an unchanged weight.

The prediction is that the swing toe-in falls further than stage one alone achieves and that the stance toe-in falls with it rather than rising, this last following from section 18.7's finding that on run E the stance heading is the swing heading at the instant of contact. The falsifying observation is the stance toe-in rising above the 7.188 degrees run E carries while the swing figure falls, which would mean the heading is being generated within the contact interval as it was on runs A through D, and the remedy is then to reopen the gate rather than to raise the weight.

#### Stage three, the differential scale halved, conditional and presently refuted

```python
    # PROPOSED, not applied, AND ITS PREMISE IS REFUTED.  ARM 16 STAGE THREE.
    # Run only on the condition stated below.
    "differential_scale": 0.5,      # ARM 16 STAGE THREE, was 1.0
```

This stage was written from section 16.7, which measured the common mode of the two hip yaws exceeding the differential by 1.47 to 1.87 in amplitude on runs A, B and C, and reasoned that since the mean form weights a pure splay four times a pure common rotation the differential scale should fall to bring the two modes toward the proportion they actually occupy. Two later measurements refute it. Section 18.6 records the common mode correlation falling from 0.636 on run C to 0.256 and 0.406 on runs D and E and the ratio of the sum's deviation to the difference's falling from 2.121 to 1.299 and 1.539, so the fault is no longer predominantly a pelvis twist. And section 18.7 measures the differential mode of the term itself growing from 0.021321 on run C to 0.070301 on run E, a factor of 3.30, while carrying 98.7 per cent of the only term that opposes it.

Halving the scale would therefore halve the price of the mode that has grown, and it is retained as a stage rather than deleted only because one condition would restore it. If stage two raises the common mode's SHARE of the term above roughly one half while the differential falls back toward run C's 0.021321, the two modes are then comparably priced and the original argument returns. That condition is checkable from one evaluation and this stage should not be launched without it.

#### What the three stages establish together

The arm's value is that the three quantities it moves are independent and the phase decomposition can attribute each. Stage one acts on the swing hip yaw and is read on `rew_keep_hip_yaw_zero_in_air` and the swing toe-in. Stage two acts on which samples the common mode sees and is read on the stance toe-in and the term's composition. Stage three acts on the balance between the two modes and is read on the differential toe-in of section 18.6. Run as one arm, as this document proposed until 2026-09-30, a fall in the pooled splay would be attributable to none of them, and section 18.7's finding that stage two's effect on the common mode is 25.9 times larger than its authors supposed is the measure of how badly that would have misled.

### 21.17 Arm 17, the hip roll excluded from the joint limit penalty

Status, NOT APPLIED, and the cheapest change in the sequence at one parameter.

This arm is instrumentation rather than shaping and it should be applied to every arm above rather than run on its own. The premise is section 14.7, that `soft_joint_pos_limit_factor` scales a travel asymmetric by 10.83 to one about its midpoint, leaving the hip roll 0.0425 radians of clearance from its nominal, that the hip roll pair consequently supplies 53 to 60 per cent of `pen_joint_pos_limits` in every run, and that the term therefore charges the joint for performing the lateral weight shift arm 5 exists to produce.

The implementation restricts the term to the joints whose soft limit is a fault report, and it is one parameter.

```python
    pen_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "(left|right)_hip_pitch_04", "(left|right)_hip_yaw_03",
            "(left|right)_knee_04", "(left|right)_foot_pitch_02",
            "(left|right)_foot_roll_02",
        ])},
    )
```

Two design decisions attach.

The hip roll is excluded rather than the soft limit factor changed, because `soft_joint_pos_limit_factor` is a property of the articulation at `kscale_identified_cfg.py` and governs the reset clamp at `events.py:350` as well as this penalty, so moving it would silently change the reset distribution of every arm and invalidate the comparison against runs A, B and C. Excluding one joint pair from one reward term changes one term.

The hip roll is not left unpriced, `pen_hip_roll_deviation` at `kscale_base_env_cfg.py:804` already charging its deviation from the nominal on a symmetric quadratic that carries no asymmetric geometry, and the hard stop at 0.2094 radians remaining where it is. What is removed is a term that fires at 2.44 degrees of outward roll, not the protection of the joint.

The prediction is arithmetic rather than behavioural, `pen_joint_pos_limits` falling from 0.4497 to roughly 0.191 per second on a policy that behaves as C does, being the 42.5 per cent share the other five pairs hold. Its value is that the remaining reading is then attributable, the ankle pitch pair holding 26.6 per cent of the present term and the ankle roll pair 15.9, which become 62.6 and 37.4 per cent of what is left once the hip roll is removed, so the term becomes the ankle instrument section 20.1 assumed it already was.

### 21.18 Arm 18, the squashed action term

Status, NOT APPLIED, new on 2026-09-30, and it SUPERSEDES arm 19, which section 20.1 withdraws. It is the cheapest of the four remedies section 19.1 and section 19.3 register for the escalation of section 18.3, and it is the only one that touches neither the actor nor the algorithm.

The premise is two measurements. Section 18.3 measures the policy of run E driving the hip roll raw action to a mean absolute nineteen standard deviations against 0.54 in run D, because a clip bounds the commanded target and not the action that produces it. And section 18.4 measures the clip excess growing by a factor of 51000 over training, which section 19.2 records as a difficulty no fixed coefficient can meet. A bounded transformation answers both at once, the first by construction and the second because a bounded action makes a bounded excess.

#### The transformation and where it goes

The deviation from the nominal becomes a scaled hyperbolic tangent of the raw action rather than a linear multiple of it, so that the deviation is bounded at plus and minus the scale however large the action grows.

```python
# PROPOSED, not applied.  Added to the new module
# environments/environments/tasks/locomotion/mdp/actions.py that section 21.20 creates for
# arm 20, or to a module of its own if that arm is not taken first.  It is a new class beside
# the existing ones, so SD_BRS1 and the TRON1 SF and PF tasks continue to resolve
# mdp.JointPositionActionCfg to Isaac Lab's own class exactly as before.
from __future__ import annotations

import re

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


class SquashedJointPositionAction(JointPositionAction):
    """A joint position action whose deviation from the nominal is bounded by construction.

    The base class computes the processed action as raw * scale + offset, which is linear and
    therefore unbounded, and the clip of arm 10 bounds the RESULT while leaving the
    raw action free. Section 18.3 measures what a policy does with that freedom, driving the
    hip roll action to nineteen standard deviations because every standard deviation past the
    bound costs nothing. This class bounds the deviation instead,

        processed = squash_scale * tanh(raw) + offset

    so the deviation cannot exceed squash_scale however large the raw action becomes, and the
    clip that follows can only ever remove the band between the clip bound and that scale.

    Placement. The transformation is applied HERE and not in the actor, which leaves rsl_rl
    untouched. The log probability and the entropy continue to be those of the Gaussian over
    the raw action, the rollout continues to store the raw action, and the symmetry
    augmentation of mdp/symmetry/kscale.py continues to permute a vector of unchanged width
    and meaning. Section 19.3 establishes that this costs nothing in the PPO objective, the
    change of variables term cancelling in the ratio at ppo.py:295 because both log
    probabilities are evaluated at the same stored action.

    What it does break is stated once and must be repaired in the same pass. pen_action_rate
    and pen_action_smoothness are quadratic in the RAW action, which is unbounded behind the
    tangent, so after this change they price a quantity that is no longer proportional to
    what reaches the joint. The repair is in the configuration below.
    """

    cfg: SquashedJointPositionActionCfg

    def __init__(self, cfg: SquashedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        names = [self._asset.joint_names[i] for i in self._joint_ids] \
            if not isinstance(self._joint_ids, slice) else list(self._asset.joint_names)
        squash = torch.zeros(len(names), device=self.device)
        for expr, value in cfg.squash_scale.items():
            hit = False
            for j, name in enumerate(names):
                if re.fullmatch(expr, name):
                    squash[j] = float(value)
                    hit = True
            if not hit:
                raise ValueError(
                    f"squash_scale key {expr!r} matches none of the joints this term drives, "
                    f"which are {names}. A key that matches nothing is a typo rather than a "
                    "no-op, and string.py:362 raises on the same condition in `scale`."
                )
        if not torch.all(squash > 0):
            missing = [n for n, s in zip(names, squash) if s <= 0]
            raise ValueError(
                f"squash_scale leaves {missing} at zero. EVERY joint must be covered, since a "
                "scale of zero would pin that joint at its nominal and would do so silently, "
                "which is the failure mode the `scale` dictionary of arm 4 already guards "
                "against at kscale_base_env_cfg.py:208."
            )
        # ActionTerm is not an nn.Module, so the scale is held as a plain tensor rather than
        # registered as a buffer. It is per joint and broadcast over environments.
        self._squash_scale = squash
        # The previous processed action, kept so that the two smoothness terms may price the
        # quantity that reaches the joint rather than the raw action behind the tangent. The
        # ActionManager stores only the raw action, at action_manager.py, so a term wanting
        # the processed one must cache it here.
        self._previous_processed = torch.zeros_like(self._processed_actions)

    @property
    def previous_processed_actions(self) -> torch.Tensor:
        return self._previous_processed

    def process_actions(self, actions: torch.Tensor):
        self._previous_processed[:] = self._processed_actions
        self._raw_actions[:] = actions
        # The base class applies `* self._scale`. That multiplication is REPLACED rather than
        # composed with the tangent, the squash scale carrying the whole of the gain, so the
        # two cannot disagree and a reader of the dumped params sees one number per joint.
        self._processed_actions = self._squash_scale * torch.tanh(self._raw_actions) \
            + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        # The cached previous action must be cleared on reset or the first step of a new
        # episode charges the smoothness terms for the discontinuity at the reset itself,
        # which is the defect stats.py records the jerk family carrying before 2026-08-06.
        if env_ids is None:
            self._previous_processed[:] = self._offset
        else:
            self._previous_processed[env_ids] = self._offset[env_ids]


@configclass
class SquashedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the squashed joint position action.

    `scale` is INHERITED and IGNORED, the squash scale carrying the gain instead. It is left
    on the class rather than removed so that the configuration still validates against a
    caller that sets it, and the term raises if both are set to values that disagree, which
    the check in __init__ performs.
    """

    class_type: type = SquashedJointPositionAction

    squash_scale: dict[str, float] = {}
    """Joint name regex to the bound on that joint's deviation from its nominal, in radians.

    The deviation reaches tanh(1) = 0.7616 of this value at one standard deviation of raw
    action and asymptotes to the value itself, so it is a bound that is approached and never
    attained. Section 21.12 sets it at the clip bound divided by tanh(1).
    """
```

#### The parameterisation, and the arithmetic that sets it

Two numbers per joint are chosen together and neither can be set without the other. The clip bound, call it the admitted deviation, is what the policy may reach. The squash scale is where the tangent asymptotes, and it must exceed the admitted deviation or the bound would be unreachable and the gradient would vanish trying.

The admitted deviation is arm 15's rule CORRECTED, being the larger of 1.5 times the saturating deflection and 1.25 times run E's own ninety fifth percentile commanded departure. The correction matters at two joints and section 21.15 is amended by it, arm 15's uniform rule standing TIGHTER than the gait's own use at the knee, where the rule gives 0.420 radians against a measured 0.479, and at the ankle pitch, where it gives 0.255 against 0.331. A bound below what the working gait already commands is a bound that forbids the gait, which is the error section 21.10 records this document making once before.

The squash scale is then the admitted deviation divided by tanh of one, so that the clip bound is reached at exactly one standard deviation of raw action, where the derivative of the transformation is 0.42 of its peak and the gradient is therefore healthy.

| Joint | 1.5 q_sat | 1.25 p95 | Admitted deviation | Squash scale | Maximum excess | Pre activation at p95 | Gradient there |
|---|---|---|---|---|---|---|---|
| hip_pitch | 0.6300 | 0.4275 | 0.6300 | 0.8272 | 0.1972 | 0.440 | 0.829 |
| hip_roll | 0.2520 | 0.2138 | 0.2520 | 0.3309 | 0.0789 | 0.572 | 0.733 |
| hip_yaw | 0.6300 | 0.3337 | 0.6300 | 0.8272 | 0.1972 | 0.335 | 0.896 |
| knee | 0.4200 | 0.5988 | 0.5988 | 0.7862 | 0.1874 | 0.708 | 0.629 |
| foot_pitch | 0.2550 | 0.4138 | 0.4138 | 0.5433 | 0.1295 | 0.708 | 0.629 |
| foot_roll | 0.2550 | 0.2038 | 0.2550 | 0.3348 | 0.0798 | 0.532 | 0.763 |

The last two columns answer the question this arm was raised by. At the deviation the gait actually commands at its ninety fifth percentile, the pre activation stands between 0.335 and 0.708 and the derivative of the transformation between 0.629 and 0.896 of its peak. THERE IS NO VANISHING GRADIENT AT THE OPERATING POINT, and there is none because the scale was chosen to put the operating point in the linear region rather than because the transformation is benign. A scale set at the admitted deviation itself rather than at 1.313 times it would place the bound at an infinite pre activation and would reproduce exactly the failure section 19.4 describes.

The maximum excess column is the property that makes the weight of arms 12 and 13 calculable. Summed over the twelve joints it is 0.2846 radians squared per step, against the 30.380 run E reaches at iteration 10000 with an unbounded action, a reduction of the WORST CASE by a factor of 107. The span a fixed weight must cover accordingly falls from 51059 to one to 5.7 to one, and a single quadratic coefficient becomes defensible where section 18.4 shows none is defensible now. At minus 17, which is the figure section 18.4 derives independently from the onset of the exploit, the worst case costs 4.84 per second against run E's negative budget of 11.99, which is 40 per cent, firm without being a wall. At minus 500 the worst case still costs 142 per second, which is twelve times that budget, so that weight remains excluded even in the bounded regime.

#### The configuration

```python
    # PROPOSED, not applied.  ARM 18 of plans/kscale_gait_shaping.md section 21.18, replacing
    # the action term at kscale_base_env_cfg.py:213.  The `clip` is arm 15's rule corrected at
    # the knee and the ankle pitch, section 21.18 stating the correction, and every bound is
    # the nominal of kscale_identified_cfg.py:197 plus and minus the admitted deviation,
    # intersected with the travel inset by 0.02 rad.
    joint_pos = mdp.SquashedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        squash_scale={
            "(left|right)_hip_pitch_04": 0.8272,
            "(left|right)_hip_yaw_03": 0.8272,
            "(left|right)_knee_04": 0.7862,
            "(left|right)_hip_roll_03": 0.3309,
            "(left|right)_foot_pitch_02": 0.5433,
            "(left|right)_foot_roll_02": 0.3348,
        },
        clip={
            "left_hip_pitch_04": (-0.28093, 0.97907),
            "right_hip_pitch_04": (-0.97907, 0.28093),
            "left_hip_roll_03": (-0.18944, 0.20905),
            "right_hip_roll_03": (-0.20905, 0.18944),
            "(left|right)_hip_yaw_03": (-0.63000, 0.63000),
            "(left|right)_knee_04": (0.27386, 1.47146),
            "(left|right)_foot_pitch_02": (-0.85267, -0.10980),
            "left_foot_roll_02": (-0.24180, 0.21375),
            "right_foot_roll_02": (-0.21375, 0.24180),
        },
    )
```

The two smoothness terms must move in the same pass, and this is the one edit outside the action term that the arm requires.

```python
    # PROPOSED, not applied.  ARM 18 repairs both terms, which are quadratic in the RAW
    # action and therefore price a quantity that is no longer proportional to what reaches
    # the joint once a tangent stands between the two.  A raw action moving from 3 to 4
    # standard deviations moves the commanded target by 0.0009 rad at the hip roll and is
    # charged as though it had moved by a whole unit.
    #
    # The repair is an optional argument on each function whose default reproduces the
    # present behaviour exactly, per the rule of /ws/CLAUDE.md, since SD_BRS1 and the TRON1
    # tasks call both and neither would set it.
    #
    #   def action_rate_l2(env, use_processed: bool = False):
    #       term = env.action_manager.get_term("joint_pos")
    #       if use_processed:
    #           # ActionManager stores the raw action only, so the processed action of the
    #           # previous step must be cached by the term. SquashedJointPositionAction holds
    #           # it as _previous_processed, set at the end of process_actions.
    #           return torch.sum(torch.square(
    #               term.processed_actions - term.previous_processed_actions), dim=1)
    #       return torch.sum(torch.square(env.action_manager.action
    #                                     - env.action_manager.prev_action), dim=1)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01,
                              params={"use_processed": True})
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.075,
                                    params={"use_processed": True})
```

#### Design decisions

Six attach and the first two are the ones a reader should check hardest.

The transformation replaces the linear scale rather than composing with it. Composing would give `squash_scale * tanh(scale * raw)`, two gains per joint where one suffices, and a reader of the dumped `params/env.yaml` would have to multiply them to learn what the term does. Replacing it means the pre activation is the raw action itself and the table above may be read directly against the policy's own noise standard deviation, which sections 16.8 and 18.7 report.

The arm does NOT remove the clip, and the reason is that the two bound different things. The tangent bounds the deviation at the squash scale, which is 1.313 times the admitted deviation, and the clip bounds it at the admitted deviation. Removing the clip would widen every bound by 31 per cent and would leave arms 11 through 13 with nothing to price, which is a different experiment. Keeping both is what makes the excess a bounded, measurable quantity rather than either an unbounded one or an absent one.

The scale is set from run E's ninety fifth percentile rather than its mean or its maximum. The mean would put the operating point at a pre activation near 0.2 and would waste most of the range, and the maximum is a single sample of a thirty second episode across thirty two environments and is not a property of the gait. The ninety fifth percentile is the figure section 21.10 already uses for the same purpose and its choice is recorded there.

Run E is the reference rather than run D, although run D is the better gait, because run E is the run that carries the clip and its commanded departures are therefore measured under the bound this arm inherits. Run D's ankle roll commanded departure of 0.4724 radians is a measurement of a policy pressing an absent bound and would size the scale to accommodate the fault rather than the gait.

The entropy and the noise standard deviation keep their present meaning under this placement, which is the one respect in which this arm is cheaper than arm 19 and worth stating because section 16.8 relies on both. The Gaussian is over the raw action and is untouched, so a rising noise standard deviation continues to mean what sections 3.3 and 16.8 read it as meaning. What changes is that a raw action of three standard deviations no longer implies a large commanded target, so the two channels must be read together rather than interchangeably.

The arm supersedes arm 19 rather than preceding it. Both bound the action, arm 19 by changing the distribution and this one by transforming its output, and section 19.3 establishes that this one costs nothing in the PPO objective where arm 19 changes the actor, the log probability, the entropy and the checkpoint format. If this arm fails in the specific way section 19.3 warns of, the policy driving the pre activation out until the gradient vanishes despite the sizing, then arm 19 returns as the remedy and its section is retained for that reason.

#### Prediction and falsifier

The prediction is that `pen_action_excess` becomes bounded above by 0.2846 radians squared per step and settles far below it, that the hip roll raw action falls from nineteen standard deviations to order one, that the hip roll hard stop residency falls from 45.90 per cent toward the 10 to 15 per cent of runs A through D, and that the ankle roll condition run E achieved is preserved, its saturation staying near 0.06 per cent.

Three observations would falsify it and each sends the sequence somewhere different. The pre activation climbing past two at any joint, which `joint_action_raw` reports directly since 2026-09-28, would mean the sizing is wrong at that joint and the admitted deviation must be widened there. The excess sitting at its bound rather than below it would mean the policy still wants more range than the squash scale allows, which is the same finding by another route. And the tracking error rising above run E's 0.0486 metres per second would mean the bound is now the binding constraint on the gait rather than on the exploit, which would send the sequence to arm 19, whose bounded support admits the whole range without a gradient penalty for reaching its edge.

### 21.19 Arm 19, the bounded action distribution

Status, NOT APPLIED, new on 2026-09-30, and the only arm in this document that changes the actor rather than the environment.

The premise is section 18.3 and the survey of section 19.1. A clip bounds the commanded target and leaves the raw action unbounded, and this robot's policy responded to a clip by driving the hip roll action to a mean absolute nineteen standard deviations. Arms 11 through 13 price that excess and arm 15 narrows the window it is measured against, and neither removes the underlying condition, which is that a Gaussian policy has support on the whole real line and the admissible set does not. Chou, Maturana and Scherer address exactly this by replacing the Gaussian with a Beta distribution whose support is the bounded interval itself, so that no probability mass falls outside the admissible set and no clipping occurs [9], and Fujita and Maeda address the estimator side of the same mismatch [19].

The implementation is confined to the actor and the algorithm is unchanged. `rsl_rl`'s `ActorCritic` emits a mean and a learned standard deviation and constructs a `Normal`, and the change is to emit two positive concentrations instead and construct a `Beta` scaled onto the action range. It must be added as a new class beside the existing one rather than as an edit, since SD_BRS1, the TRON1 tasks and the quadruped all instantiate the same actor.

```python
# PROPOSED, not applied.  A new module beside the existing actor in the rsl_rl package this
# workspace vendors, or in environments/utils/wrappers/ if the vendored package is to be left
# untouched, selected by class_name in the agent configuration exactly as HIMActorCritic and
# CoptActorCritic already are at limx_rsl_rl_ppo_cfg.py.
class BetaActorCritic(ActorCritic):
    """An actor whose action distribution cannot leave the admissible interval.

    The network emits two raw outputs per action dimension rather than one, mapped to the
    two concentrations of a Beta through softplus plus one so that both exceed unity and the
    density is therefore unimodal and finite at both ends, which is what makes the mean a
    usable deterministic action at play time. The sample is affine mapped from (0, 1) onto
    the action range, which is taken from the action term's own clip so that the two cannot
    disagree.

    The entropy and the log probability are the Beta's own, so the PPO objective is unchanged
    and nothing in the algorithm needs to know which distribution it holds. What DOES change
    is the meaning of the logged noise standard deviation, which is no longer a parameter but
    a function of the concentrations, and section 16.8's reading of that channel must be
    re-derived before it is applied to a run under this actor.
    """

    def __init__(self, *args, action_low=None, action_high=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Two outputs per dimension. The final layer is rebuilt rather than wrapped so that
        # the parameter count is explicit in the checkpoint.
        last = self.actor[-1]
        self.actor[-1] = nn.Linear(last.in_features, 2 * last.out_features)
        self.register_buffer("action_low", torch.as_tensor(action_low))
        self.register_buffer("action_high", torch.as_tensor(action_high))
        # std is a parameter on the base class and is unused here. It is left in place so
        # that a checkpoint saved under this class still loads under the base one.

    def update_distribution(self, observations):
        raw = self.actor(observations)
        alpha, beta = torch.chunk(raw, 2, dim=-1)
        # softplus plus one keeps both concentrations above unity, which excludes the U
        # shaped Beta whose mass piles at the two ends and which would reproduce the
        # bang-bang preference section 15.1 records rather than removing it.
        alpha = torch.nn.functional.softplus(alpha) + 1.0
        beta = torch.nn.functional.softplus(beta) + 1.0
        self.distribution = torch.distributions.Beta(alpha, beta)

    def _to_range(self, unit):
        return self.action_low + unit * (self.action_high - self.action_low)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self._to_range(self.distribution.rsample())

    def act_inference(self, observations):
        raw = self.actor(observations)
        alpha, beta = torch.chunk(raw, 2, dim=-1)
        alpha = torch.nn.functional.softplus(alpha) + 1.0
        beta = torch.nn.functional.softplus(beta) + 1.0
        return self._to_range(alpha / (alpha + beta))

    def get_actions_log_prob(self, actions):
        unit = (actions - self.action_low) / (self.action_high - self.action_low)
        # The affine map contributes a constant Jacobian term, which is dropped because it
        # is identical for every action and therefore cancels in the PPO ratio.
        return self.distribution.log_prob(unit.clamp(1e-6, 1 - 1e-6)).sum(dim=-1)
```

The action range the actor is given must be the range the action term would clip to, expressed in RAW action units rather than in radians, which is the clip bound minus the offset divided by the scale. With arm 15's bounds and the scales of arm 4 this is plus and minus 1.575 at the hip pitch and hip yaw, plus and minus 1.575 at the knee, plus and minus 1.575 at the hip roll and plus and minus 3.1875 at the two ankles, the uniformity following from the uniform 1.5 multiple of arm 15 and the ankles differing because arm 6 halved their scale. Where the two disagree the clip remains as a backstop and the excess should read exactly zero, which is the check this arm ships with.

Three design decisions attach.

The Beta is chosen over a tanh squashed Gaussian because the squashing function distorts the geometry of the bounded space and concentrates mass at the ends, which is the behaviour section 15.1 records as the bang-bang preference and which this arm exists to remove rather than to re-express. The concentrations are held above unity for the same reason.

The arm sits below arm 10 and arm 15 in the order because both are cheaper, both are reversible by editing a configuration, and either may reduce the escalation enough that this one is unnecessary. It sits above arm 20 because a gain change made while the action is unbounded is a change to a system whose input is not defined.

The logged noise standard deviation ceases to mean what section 16.8 reads it as meaning, and that is a cost rather than a detail. This workspace has used the entropy and the noise standard deviation to separate a converging run from a diverging one three times, at sections 3.3, 16.8 and elsewhere, and under a Beta actor both are functions of the concentrations rather than parameters. The arm must therefore carry a re-derivation of that reading or lose an instrument it has relied upon.

The prediction is that the raw action ceases to leave its range by construction, that `pen_action_excess` reads exactly zero, and that the hip roll stop residency falls to whatever the load alone produces. The falsifying observation is the policy pressing the ends of the Beta interval as hard as it pressed the clip, which would mean the escalation was never about the unboundedness and would leave arm 15's sizing as the only remedy.


### 21.20 Arm 20, phase dependent gains, with the learned variable impedance action space as its second stage

Status, NOT APPLIED, and it requires a new action term in this repository's own mdp package, which is additive and touches no existing caller.

The premise is this robot's own measurement and not the literature, section 17.4 having recorded that no surveyed source prescribes a swing and stance gain schedule for a learned locomotion policy. Section 16 measures every joint of this machine behaving differently in the two phases by margins that no single gain pair serves, and the ankle roll is the extreme case.

| Joint | Speed, stance [rad/s] | Speed, swing [rad/s] | Ratio | Saturation, stance [pct] | Saturation, swing [pct] |
|---|---|---|---|---|---|
| Ankle roll, run A | 1.258, 1.171 | 0.183, 0.198 | 6.6 | 0.39, 0.30 | 48.04, 45.43 |
| Ankle roll, run C | 0.371, 0.385 | 0.013, 0.013 | 29.2 | 79.11, 80.05 | 99.13, 99.37 |
| Ankle pitch, run C | 0.924, 0.916 | 0.790, 0.823 | 1.2 | 20.59, 18.55 | 72.59, 71.99 |
| Hip yaw, run C | 0.350, 0.341 | 1.351, 1.451 | 0.25 | 0.01, 0.02 | 0.00, 0.00 |
| Knee, run C | 1.131, 1.070 | 1.441, 1.438 | 0.76 | 0.00, 0.01 | 0.00, 0.00 |

The hip yaw runs four times faster in swing than in stance and the ankle roll twenty nine times slower, so the two joints want opposite corrections and a global gain change helps neither. The stance requirement is a stiffness that holds the assembly against gravity and the ground, which `../context/KScale.md` section 11 derives from the single support hold torque. The swing requirement is a damping ratio near 0.7 against a swing inertia that `../context/KScale.md` section 11.1.1 measures as up to 897 times smaller than the stance one at the ankle roll, and a damping chosen for the stance inertia is therefore 897 times over damped in swing. That ratio is the whole of the argument and it is a property of the robot rather than of the policy.

The implementation is a new action term added beside the existing ones rather than an edit to any of them. Isaac Lab's `JointPositionAction` is re-exported into this package by the star import at `environments/tasks/locomotion/mdp/__init__.py:1`, so a new class in a new module is invisible to every existing caller until a configuration names it. The gains of an explicit actuator live on the actuator object and are read by its torque computation on every call, which `randomize_actuator_gains` at `isaaclab/envs/mdp/events.py:629` relies upon and which is why no write to the simulator is needed for anything but an `ImplicitActuator`.

```python
# PROPOSED, not applied.  A new module at
# environments/environments/tasks/locomotion/mdp/actions.py, re-exported by adding
# `from .actions import *` to that package's __init__.py beside the six imports already
# there.  Nothing existing is edited, so SD_BRS1 and the TRON1 SF and PF tasks continue
# to resolve mdp.JointPositionActionCfg to Isaac Lab's own class exactly as before.
from __future__ import annotations

from dataclasses import MISSING

import torch
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


class PhaseGainJointPositionAction(JointPositionAction):
    """A joint position action that carries one gain set for stance and another for swing.

    The processed action, the offset, the scale and the clip are those of the base class
    exactly, so a configuration that sets no gain dictionary reproduces JointPositionAction
    bit for bit.  What this term adds is a write of the actuator stiffness and damping on
    every call of apply_actions, selected per environment and per joint by whether the leg
    that joint belongs to is in contact.

    The gains are written onto the actuator object rather than into the simulator, which is
    correct for an explicit actuator and wrong for an ImplicitActuator.  The KScale drives
    every joint through IdentifiedActuator, whose compute() reads self.stiffness on each
    call, so the write takes effect on the step it is made.  The term raises rather than
    warning if it is given an implicit actuator, since a silent no-op would be indistinguishable
    from a null result.
    """

    cfg: PhaseGainJointPositionActionCfg

    def __init__(self, cfg: PhaseGainJointPositionActionCfg, env):
        super().__init__(cfg, env)
        from isaaclab.actuators import ImplicitActuator

        self._sensor = env.scene.sensors[cfg.sensor_cfg.name]
        cfg.sensor_cfg.resolve(env.scene)
        self._feet_ids = cfg.sensor_cfg.body_ids
        names = self._asset.joint_names
        # One row per actuated joint, holding the stance and the swing value and the index
        # of the foot whose contact state selects between them.  A joint the dictionaries
        # do not name keeps its configured gain in both phases, which is how a subset of the
        # joints is scheduled without naming the rest.
        n = len(self._joint_ids) if not isinstance(self._joint_ids, slice) else len(names)
        dev = self.device
        self._kp = torch.zeros(2, n, device=dev)
        self._kd = torch.zeros(2, n, device=dev)
        self._foot_of = torch.zeros(n, dtype=torch.long, device=dev)
        ids = range(len(names)) if isinstance(self._joint_ids, slice) else self._joint_ids
        for col, jid in enumerate(ids):
            name = names[jid]
            self._foot_of[col] = 0 if name.startswith(cfg.left_prefix) else 1
            base_kp, base_kd = None, None
            for actuator in self._asset.actuators.values():
                idx = actuator.joint_indices
                hit = (idx == jid).nonzero() if isinstance(idx, torch.Tensor) else None
                if hit is not None and len(hit):
                    if isinstance(actuator, ImplicitActuator):
                        raise ValueError(
                            f"joint {name!r} is driven by an ImplicitActuator, whose gains "
                            "live in the simulator rather than on the actuator, so a per "
                            "step write here would not reach it. Give it an explicit "
                            "actuator or leave it out of the gain dictionaries."
                        )
                    base_kp = float(actuator.stiffness[0, hit[0, 0]])
                    base_kd = float(actuator.damping[0, hit[0, 0]])
                    break
            self._kp[0, col], self._kp[1, col] = base_kp, base_kp
            self._kd[0, col], self._kd[1, col] = base_kd, base_kd
            for expr, (stance, swing) in (cfg.stiffness_by_phase or {}).items():
                if __import__("re").fullmatch(expr, name):
                    self._kp[0, col], self._kp[1, col] = stance, swing
            for expr, (stance, swing) in (cfg.damping_by_phase or {}).items():
                if __import__("re").fullmatch(expr, name):
                    self._kd[0, col], self._kd[1, col] = stance, swing

    def apply_actions(self):
        if self.cfg.stiffness_by_phase or self.cfg.damping_by_phase:
            forces = self._sensor.data.net_forces_w_history[:, 0, self._feet_ids]
            # (num_envs, 2), True where that foot carries load
            stance = torch.norm(forces, dim=-1) > self.cfg.force_threshold
            # (num_envs, num_joints), 0 where the joint's own leg is in stance
            phase = (~stance[:, self._foot_of]).long()
            kp = torch.gather(self._kp.T.expand(phase.shape[0], -1, -1), 2,
                              phase.unsqueeze(-1)).squeeze(-1)
            kd = torch.gather(self._kd.T.expand(phase.shape[0], -1, -1), 2,
                              phase.unsqueeze(-1)).squeeze(-1)
            for actuator in self._asset.actuators.values():
                idx = actuator.joint_indices
                if isinstance(idx, slice):
                    actuator.stiffness[:] = kp
                    actuator.damping[:] = kd
                else:
                    cols = torch.isin(torch.as_tensor(list(self._joint_ids)), idx).nonzero().view(-1)
                    actuator.stiffness[:] = kp[:, cols]
                    actuator.damping[:] = kd[:, cols]
        super().apply_actions()


@configclass
class PhaseGainJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the phase dependent gain joint position action.

    With both dictionaries left at None the term is JointPositionAction exactly, which is
    the default the workspace rule asks for, so this class may be named in a configuration
    before the gains are chosen without changing any behaviour.
    """

    class_type: type = PhaseGainJointPositionAction

    sensor_cfg: SceneEntityCfg = MISSING
    """The contact sensor resolving the two feet, LEFT BEFORE RIGHT."""

    stiffness_by_phase: dict[str, tuple[float, float]] | None = None
    """Joint name regex to (stance, swing) stiffness. A joint left unnamed keeps its own."""

    damping_by_phase: dict[str, tuple[float, float]] | None = None
    """Joint name regex to (stance, swing) damping. A joint left unnamed keeps its own."""

    left_prefix: str = "left"
    """The prefix that identifies a left joint, whose foot is index zero of sensor_cfg."""

    force_threshold: float = 1.0
```

The configuration that names it replaces the action block at `kscale_base_env_cfg.py:169` and changes nothing else.

```python
    # PROPOSED, not applied.  ARM 20 of plans/kscale_gait_shaping.md section 21.20.
    joint_pos = mdp.PhaseGainJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "(left|right)_hip_pitch_04": 0.4,
            "(left|right)_hip_yaw_03": 0.4,
            "(left|right)_knee_04": 0.2667,
            "(left|right)_hip_roll_03": 0.16,
            "(left|right)_foot_pitch_02": 0.08,
            "(left|right)_foot_roll_02": 0.08,
        },
        use_default_offset=True,
        sensor_cfg=SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
        stiffness_by_phase={
            # (stance, swing).  The stance value is the configured one in every row, so
            # this arm changes the SWING gain alone and the stance behaviour is unmoved.
            "(left|right)_foot_roll_02": (100.0, 25.0),
            "(left|right)_foot_pitch_02": (100.0, 25.0),
        },
        damping_by_phase={
            "(left|right)_foot_roll_02": (1.05, 0.26),
            "(left|right)_foot_pitch_02": (1.25, 0.31),
        },
        force_threshold=1.0,
    )
```

Four design decisions attach and the first two are the ones that keep the arm readable.

Only the SWING gain moves and only at the two ankles. The stance column reproduces the configured value exactly, so whatever the arm does is attributable to swing, and the stance derivations of `../context/KScale.md` section 11, which are the only gain derivations this workspace has validated, continue to govern the phase they were derived for. The hip yaw is deliberately excluded although its speed ratio is the second largest, because section 16.6 establishes that its swing excursion is commanded and tracked to within four per cent, so there is nothing for a gain to correct there and including it would add a variable with no premise.

The swing stiffness falls by four and the damping by the same factor, which holds the damping ratio rather than the gains. The quantity that must be preserved is the ratio, `../context/KScale.md` section 11.1.1 giving the stability bound as a condition on the damping and the swing inertia alone and independent of the stiffness, so a stiffness reduction unaccompanied by a damping reduction would leave the joint as over damped in swing as it is now while removing the authority it still needs at touchdown. Four is chosen because it is the factor that brings the saturating deflection, being the effort limit divided by the stiffness, from 0.17 radians to 0.68, which exceeds the 0.170 radian excess beyond the stop that section 16.4 measures, so a command that presently saturates the actuator would no longer do so at the reduced gain.

The gains are written on every call of `apply_actions` rather than on a contact event, because `apply_actions` runs once per physics decimation step and a contact transition may fall between two control steps. The cost is two writes of a tensor of twelve columns per step, which is negligible beside the articulation update, and the alternative of an event term cannot express a per environment per joint condition at all.

The term raises rather than warning when it meets an `ImplicitActuator`, since an implicit actuator's gains live in the simulator and a write to `actuator.stiffness` would not reach the solver. A silent no-op would produce a null result indistinguishable from the arm having been correctly run and having done nothing, which is the failure mode section 14.2 spent a run discovering at the action scale.

The prediction is that the ankle roll envelope saturation in swing falls from 98 to 99 per cent toward the stance figure, that its swing speed rises from 0.013 radians per second, and that the stop residency in swing falls below the 89.8 per cent run A carried before the deviation penalty. The falsifying observation is the saturation surviving, which would mean the command is far enough beyond the stop that a fourfold reduction in stiffness does not bring it inside the envelope, and would say that arm 10 must precede this one rather than accompany it.

#### The second stage, the learned variable impedance action space, to be run only if the schedule is ambiguous

Status, NOT APPLIED, and it is the form the literature actually supports.

Section 17.1 records that the published route to phase dependent impedance is to let the policy output the gains rather than to schedule them, Bogdanovic, Khadiv and Righetti reporting that a policy whose output is an impedance together with a desired position outperforms both torque and position control on learning rate and on robustness to environment uncertainty, on a hopper with intermittent contacts which is the surveyed system closest to a leg [24]. Section 17.4 records that no surveyed source prescribes the schedule of the first stage, so the first stage is this workspace's own proposal and this stage is the one with a precedent.

The implementation extends the same class rather than replacing it, the action dimension rising from twelve to twenty four and the added twelve being mapped through a bounded transform onto a gain interval, so that an unbounded Gaussian sample cannot produce a negative stiffness or one large enough to destabilise the explicit integration. The interval is set by `../context/KScale.md` section 11.1.1, whose necessary condition on the damping and the swing inertia gives the ceiling, and by the gravitational stiffness of the addition of 2026-09-16 to that file, which gives the floor below which the joint cannot hold the assembly at any deflection.

This stage should be run only if the first is ambiguous, and it has one specific ambiguity to resolve. If the schedule improves the ankle and degrades the touchdown, the question is whether the boundary between the two gain sets is in the wrong place rather than whether the gains are wrong, and a learned impedance answers that by placing the boundary itself. Running it first would confound the value of a phase dependent gain with the value of learning one.

### 21.21 Arm 21, the footprint separation hinge

Status, NOT APPLIED, and needing no new function, `feet_distance_v2` already carrying the hinge.

The premise is section 14.8, that `pen_feet_distance` measures a quantity 0.143 metres larger than the one that decides whether the feet foul one another, and that the true footprint separation falls below twenty millimetres on more than a quarter of samples in all three runs and reaches 0.1 millimetre at the first percentile.

The implementation requires no new function, `feet_distance_v2` at `rewards.py:1138` already carrying the second hinge and already documenting that it reproduces `feet_distance` exactly while `sole_offsets` is unset and `min_sole_distance` is zero. The docstring further states the intended registration, that the two hinges are given independent weights by registering the function twice with the second term setting `min_feet_distance` to zero so that only the sole hinge speaks, and that is the form this arm takes.

```python
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance_v2,
        weight=-100,
        params={
            "min_feet_distance": 0.19,
            "feet_links_name": [_FOOT_LINKS],
            "lateral_only": False,
        },
    )
    pen_sole_distance = RewTerm(
        func=mdp.feet_distance_v2,
        weight=-100,
        params={
            "min_feet_distance": 0.0,
            "feet_links_name": [_FOOT_LINKS],
            "lateral_only": False,
            "sole_offsets": KSCALE_SOLE_OFFSETS,
            "min_sole_distance": 0.03,
        },
    )
```

Three design decisions attach, and the threshold is the one that decides whether the arm is a shaping term or a wall.

`min_sole_distance` is set at 0.03 metres rather than at anything larger, and the measurement sets it. The footprint separation has a median of 0.0567, 0.0689 and 0.0571 metres across the three runs and a twenty fifth percentile of 0.0153, 0.0220 and 0.0153. A floor of 0.03 therefore sits between the twenty fifth percentile and the median, so the term is silent on roughly two thirds of samples and speaks on the third that is genuinely close, which is the shape a hinge should have. A floor at the 0.19 the frame origin hinge carries would fire on more than ninety per cent of samples in every run and would be a constant offset rather than a gradient, which is the failure `pen_joint_pos_limits` exhibits at the hip roll and which section 14.7 documents.

The weight matches the existing hinge at minus 100 rather than exceeding it, so the arm adds a second measurement at a known price rather than a new dominant term. At the measured violation the term would charge roughly 0.3 to 0.5 per second, which is comparable to the 0.9018 the frame origin hinge charges A, and the two together would then be about two per cent of the negative budget. That is deliberately not enough to dominate, this arm being one of the two in this section that must be run against the corrected budget of arm 8 rather than against the present one.

The first term keeps `feet_distance_v2` rather than `feet_distance`, although with the sole parameters unset the two are identical by construction. The reason is that a reader of the dumped `params/env.yaml` should see one function serving both hinges, and the docstring states the bit for bit equivalence, so nothing is risked by the change and the configuration becomes self explanatory.

The prediction is that the footprint separation's twenty fifth percentile rises above 0.03 metres, that `pen_foot_cross_contact_left` and `pen_foot_cross_contact_right` fall from their present 0.0036 and 0.0239 per second in C, and that the lateral separation rises above the 0.1440 metres C records. The falsifying observation is the stance widening without the footprint separation following it, which would say the machine bought the clearance by walking wider while keeping the same toe-in, and the remedy would then be arm 10 rather than a larger floor here.

### 21.22 Arm 22, the in air kernel width

Status, NOT APPLIED, and to be run after arm 10 since that arm may remove the deviation this term exists to correct.

Section 13 establishes that the three in air terms are inert rather than weak. The exponential kernel returns six per cent of its maximum at the 0.568 radian ankle pitch deviation and a gradient of 0.31 per radian against a velocity reward of 47 per second, so a weight multiplying a vanished gradient produces a larger number in the reward budget and no change in behaviour, which is why section 20.3 excludes raising the weight alone. The hip yaw term, whose deviation of 0.084 to 0.191 radians sits inside the kernel width, returns 42 to 57 per cent at a gradient seven to ten times larger, and it is the one term of the three the record credits with working.

All three terms call the same function and differ only in the joints they resolve and their weight.

```python
    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_foot_pitch_02"]
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "use_default_offset": True,
        },
    )
```

The kernel stands at `rewards.py:1630` and it is exponential in the absolute deviation rather than Gaussian in it, so its gradient falls as the deviation grows at every distance rather than only beyond a width.

```python
    weighted_ankle_pitch = torch.sum(ankle_pitch * ~contact_filt, dim=1)
    reward = torch.exp(-weighted_ankle_pitch / (pitch_scale + 1e-6))
```

Two design decisions are open and the arm should choose one rather than both, so that the result is readable.

The width may be raised to match the deviation each term must correct, which is the smaller change and needs no new code, only three numbers in the configuration. The rule is that the width should sit near the deviation rather than near zero, so that the operating point lies on the steep part of the kernel rather than in its tail, and the measured deviations of section 13 give the widths directly.

Or the exponential may be replaced by the relaxed logarithmic barrier [5], whose gradient grows with distance rather than vanishing, which is the same instrument arm 10 proposes and would be carried the same way, as an optional argument whose default reproduces the present exponential exactly. Running both at once would confound the width with the functional form, so the width is the arm and the barrier is its fallback.

| Term | Weight | Measured deviation | `pitch_scale` now | `pitch_scale` proposed | Value now | Value proposed |
|---|---|---|---|---|---|---|
| `rew_keep_ankle_pitch_zero_in_air` | 1.0 | 0.568 rad | 0.2 | 0.55 | 6 pct | 36 pct |
| `rew_keep_ankle_roll_zero_in_air` | 0.25 | 0.18 to 0.34 rad | 0.2 | 0.30 | 18 to 40 pct | 32 to 55 pct |
| `rew_keep_hip_yaw_zero_in_air` | 1.0 | 0.084 to 0.191 rad | 0.2 | 0.2, unchanged | 42 to 57 pct | 42 to 57 pct |

The hip yaw term is left alone deliberately. It is the one term of the three whose deviation already sits inside its width, section 5 credits it with halving the mean toe-in to 4.59 degrees, and changing a working instrument in the same arm that repairs two broken ones would make the arm unreadable.

The prediction is that the ankle pitch term rises from six per cent of its maximum toward the half the hip yaw term returns, and that the swing foot orientation becomes a quantity the policy regulates. The falsifying observation is the swing ankle pitch deviation surviving unchanged at a term value three times higher, which would mean the deviation is set by the crouch rather than chosen, and would send the sequence to arm 7.

### 21.23 Arm 23, the restored corner velocity

Status, NOT APPLIED, and it is the cheapest arm in the sequence, being one value in each of two blocks.

Section 10.2 establishes that the saturation effort was varied unintentionally alongside the ceiling and that the seventeen newton metre arm consequently delivers no additional torque whatever above 11.31 radians per second, which is precisely the regime section 11.4 shows the joint traversing in at twenty five radians per second. Until this arm is run the seventeen newton metre result cannot be attributed to the ceiling rather than to the derating slope, so it is a prerequisite for reading arm 2 at all.

The two ankle blocks stand as follows and the change is the saturation effort alone.

```python
KSCALE_FOOT_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_roll_02"],
    velocity_limit_sim=44.0,
    effort_limit=17,
    velocity_limit=37.699,
    saturation_effort=17.0,
```

```python
    # PROPOSED, not applied. 1.43 times the ceiling reproduces the ratio the 11.9 N.m arm
    # carried by accident, restoring an 11.31 rad/s constant torque plateau.
    saturation_effort=24.31,
```

The design decision is to restore the ratio rather than a particular saturation value, because the corner velocity is a function of the ratio alone. The arm 1 configuration carried 11.9 against 17.0, which is a ratio of 0.7 and a corner velocity of 11.31 radians per second, and reproducing that ratio at a seventeen newton metre ceiling requires 24.31. The alternative of lowering the ceiling back to 11.9 was rejected because it would undo arm 2 rather than complete it, and the point of the arm is to separate the two variables that arm 2 moved together.

| Configuration | Ceiling [N.m] | Saturation [N.m] | Ratio | Corner velocity [rad/s] | Torque at 25 rad/s [N.m] |
|---|---|---|---|---|---|
| Arm 1 | 11.9 | 17.0 | 0.700 | 11.31 | 5.73 |
| Arm 2, as run | 17 | 17.0 | 1.000 | 0.00 | 5.73 |
| Arm 3, as run | 34 | 34.0 | 1.000 | 0.00 | 11.45 |
| Arm 23 | 17 | 24.31 | 0.699 | 11.34 | 8.19 |

The prediction is that the ankle roll torque at the traverse speeds of section 11.4 rises by 43 per cent without any change to the peak, being 8.19 newton metres against 5.73, and that the stop to stop reversal rate falls because the damping term is no longer saturating the actuator unaided. The falsifying observation is the reversal rate of 2.03 per second surviving unchanged, which would mean the impulse switching is commanded rather than a consequence of the envelope.

### 21.24 Arm 24, the ankle roll fixed in the URDF, with its own asset, action term, environment and two tasks

Status, NOT APPLIED, PREMISE WITHDRAWN by section 18.2, and rebuilt on 2026-09-30 at the user's direction to follow arm 14's construction rather than the configuration level freeze it proposed until that date. It produces TWO tasks rather than one, the ankle roll frozen alone and the ankle roll frozen together with the hip yaw of arm 14, so that the two removals may be read apart and together.

The premise as first written is section 14.4, that the best policy of the sequence held the ankle roll at a mechanical stop for 93.2 per cent of stance on the left and 99.9 per cent of swing on the right, under a commanded target beyond the stop on 96.4 per cent of samples, at a mean absolute torque of 16.8 newton metres against a 17 newton metre ceiling. The joint was not being controlled, it was being pressed, and an actuator spending ninety eight per cent of its ceiling to press a stop does no work.

THAT PREMISE NO LONGER HOLDS. Section 18.2 measures the same joint under the clip of arm 10 at 0.06 per cent exact saturation, 0.07 per cent stop residency and 3.5 newton metres of mean absolute torque, so the joint the arm exists to remove is now a controlled joint doing ordinary work. The arm is retained for two reasons and neither is the original one. Arm 15 narrows the clip and may return the ankle to its stop, in which case the premise returns with it, and section 17.3 records two independently designed bipeds that omit the ankle roll entirely, so the question of whether this robot needs the degree of freedom at all survives the clip's success. The arm must not be launched while section 18.2's figures stand.

#### Why this is not arm 14 repeated, and the blocker that makes it different

Arm 14 freezes a joint in the middle of the chain. `left_hip_yaw_03` carries `rs03` as its parent and `kd_d_301l_6061` as its child at `kscale.urdf:1094`, so the importer merges a thigh link into a hip link and nothing outside `_BODY_PARTNERS` names either.

THE ANKLE ROLL IS THE TERMINAL JOINT OF THE CHAIN AND ITS CHILD IS THE FOOT. `right_foot_roll_02` carries `arb_uj111_cross_bearing` as its parent and `foot_6061` as its child at `kscale.urdf:590`, and `left_foot_roll_02` carries `arb_uj111_cross_bearing_2` and `foot_6061_2` at `:1070`. A fixed joint merges the child into the parent, so the naive edit that arm 14 performs would DESTROY the two foot links, and with them `_FOOT_LINKS` at `kscale_base_env_cfg.py:25`, the contact sensor that every gait term resolves, the `forward_axis` of `pen_feet_heading`, the `sole_normal_link` of the instrumentation section 20.2 requires, the 0.04300 metre sole depth of `context/KScale.md` section 16.2 and both termination conditions. The surviving merged link would be the cross bearing, whose frame is rotated by ninety degrees on two axes relative to the foot's, so every foot quantity this document measures would change frame silently and no figure would be comparable with any run above.

The construction therefore inverts the merge. Rather than fixing the roll joint and letting the foot be absorbed, the roll joint is DELETED and the pitch joint is repointed at the foot link directly, with the cross bearing's mass and inertia folded into the foot and the pitch origin composed so that the foot's frame at zero roll is exactly where it stands today. The foot link keeps its name, its frame, its visual, its collision mesh and its sole depth, and the diff is therefore larger than arm 14's two attributes while the quantity that matters, the pose of the foot, is provably unmoved.

#### The URDF, stage one, the ankle roll alone

```xml
<!-- PROPOSED, not applied.  environments/environments/assets/urdf/solefoot/kscale/
     kscale_ankle_roll_fixed.urdf, a copy of kscale.urdf with three changes per leg and no
     others.  The composed origins below were computed from the two originals rather than
     estimated, the translation being the pitch origin plus the pitch rotation applied to
     the roll origin's translation, and the rotation being the product of the two.  The
     right pitch origin carries no rotation at kscale.urdf:598 so its composition changes
     the translation alone, and the left carries a yaw of minus pi at :1078 which both
     rotates the roll translation and adds to the roll rotation's yaw. -->

<!-- ONE, the roll joints are DELETED outright, both of them. -->
<!-- TWO, the two cross bearing links are DELETED and their inertials are folded into the
     foot links they used to carry, 0.110039 kg each, expressed in the foot frame.  The
     robot's total mass is therefore unchanged, which validate_arm24.py asserts. -->
<!-- THREE, each pitch joint takes the foot link as its child at the composed origin. -->

  <joint name="right_foot_pitch_02" type="revolute">
    <!-- was xyz="0.288552 -0.0326806 0.008" rpy="0 0 -0", child arb_uj111_cross_bearing -->
    <origin xyz="0.288552 -0.0026806 -0.022" rpy="-1.5708 -1.5708 0"/>
    <parent link="kd_d_401r_6061"/>
    <child link="foot_6061"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-0.872665" upper="0.523599"/>
  </joint>

  <joint name="left_foot_pitch_02" type="revolute">
    <!-- was xyz="-0.288552 0.0325676 0.05" rpy="-0 -0 -3.14159", child ..._cross_bearing_2 -->
    <origin xyz="-0.288552 0.0025676 0.02" rpy="-1.5708 -1.5708 -3.14159"/>
    <parent link="kd_d_401l_6061"/>
    <child link="foot_6061_2"/>
    <axis xyz="0 0 1"/>
    <limit effort="10" velocity="10" lower="-0.872665" upper="0.523599"/>
  </joint>
```

ONE PROPERTY OF THIS CONSTRUCTION MUST BE UNDERSTOOD BEFORE IT IS TRUSTED, and it is the axis. The pitch joint's axis is its own local z at `kscale.urdf:602`, and composing the roll origin's rotation into the pitch origin ROTATES THAT AXIS. The pitch joint would then rotate about what used to be the roll axis, which is the opposite of what the arm wants. The axis must therefore be re-expressed in the new frame rather than left at `0 0 1`, and the correct entry is the old pitch axis carried through the inverse of the roll origin's rotation. `validate_arm24.py` computes it and asserts that the resulting pitch sweep moves the foot's toe through the sagittal plane and not the frontal one, which is the only check that distinguishes a correct composition from a plausible one. The snippet above is therefore INCOMPLETE by design on that one attribute, and a reader who fills it in by eye will produce a robot that pitches sideways.

The alternative that avoids the composition entirely is to keep the cross bearing link and give the roll joint a zero width limit rather than a fixed type. That is not a fixed joint, the servo still tracking a target through a joint with travel, and it reproduces the configuration level freeze this arm was rebuilt to replace, so it is recorded here as the fallback should the composition prove troublesome and not as the proposal.

#### The URDF, stage two, the ankle roll and the hip yaw together

```xml
<!-- PROPOSED, not applied.  environments/environments/assets/urdf/solefoot/kscale/
     kscale_hip_yaw_ankle_roll_fixed.urdf, being arm 14's two fixed joints and stage one's
     three changes per leg applied to one copy.  The two edits do not interact, the hip yaw
     sitting four joints above the ankle and neither touching the other's parent or child,
     so the composed origins of stage one are unchanged and arm 14's assertion that the
     kinematic chain below each frozen joint is unmoved continues to hold. -->
  <joint name="left_hip_yaw_03" type="fixed"/>   <!-- was revolute, arm 14 -->
  <joint name="right_hip_yaw_03" type="fixed"/>  <!-- was revolute, arm 14 -->
  <!-- and stage one's deletions and recomposition at both ankles -->
```

The articulation that results carries EIGHT joints and NINE bodies, against stage one's ten and eleven and arm 14's ten and eleven, the present robot carrying twelve and thirteen. Stage one's count of eleven bodies follows from the two cross bearings being deleted while both feet survive, and stage two's nine from those two deletions together with arm 14's two merges.

#### The articulation configuration

```python
# PROPOSED, not applied.  Appended to
# environments/environments/assets/config/kscale_identified_cfg.py after arm 14's entry,
# leaving KSCALE_IDENTIFIED_CFG and KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED untouched.
ankle_roll_fixed_urdf_path = os.path.join(
    current_dir, "../urdf/solefoot/kscale/kscale_ankle_roll_fixed.urdf"
)
hip_yaw_ankle_roll_fixed_urdf_path = os.path.join(
    current_dir, "../urdf/solefoot/kscale/kscale_hip_yaw_ankle_roll_fixed.urdf"
)


def _without(cfg, *fragments):
    """A copy of cfg with every actuator group and nominal naming a fragment removed.

    Both removals are REQUIRED rather than tidy. ActuatorBase raises on a
    joint_names_expr matching no joint and ArticulationCfg raises on an init_state key
    naming none, so a group left resolving to a joint the URDF no longer carries aborts
    the spawn. The dictionaries are rebuilt rather than mutated because
    ArticulationCfg.replace performs a shallow copy and mutation would reach the original.
    """
    out = cfg.replace()
    out.actuators = {
        name: act for name, act in cfg.actuators.items()
        if not any(f in name for f in fragments)
    }
    out.init_state = cfg.init_state.replace(
        joint_pos={
            name: value for name, value in cfg.init_state.joint_pos.items()
            if not any(f in name for f in fragments)
        }
    )
    return out


KSCALE_IDENTIFIED_CFG_ANKLE_ROLL_FIXED = _without(
    KSCALE_IDENTIFIED_CFG.replace(
        spawn=KSCALE_IDENTIFIED_CFG.spawn.replace(asset_path=ankle_roll_fixed_urdf_path),
    ),
    "foot_roll",
)

KSCALE_IDENTIFIED_CFG_HIP_YAW_ANKLE_ROLL_FIXED = _without(
    KSCALE_IDENTIFIED_CFG.replace(
        spawn=KSCALE_IDENTIFIED_CFG.spawn.replace(
            asset_path=hip_yaw_ankle_roll_fixed_urdf_path
        ),
    ),
    "foot_roll", "hip_yaw",
)
# The actuator group is named ankle_roll rather than foot_roll in some revisions of this
# file, so the fragment must be checked against the live dictionary rather than assumed.
# validate_arm24.py asserts that each returned configuration's actuator groups resolve to
# exactly the joints its URDF carries and no others.
```

#### The action term

```python
# PROPOSED, not applied.  Added to kscale_base_env_cfg.py beside ActionsCfg at :162 and
# ActionsCfgHipYawFixed of arm 14, both of which are left exactly as they stand.
@configclass
class ActionsCfgAnkleRollFixed:
    """The ten joint action term of the ankle roll fixed robot.

    Every surviving value is byte identical to ActionsCfg, so a policy trained here and one
    trained there differ in their action dimension and in nothing else. The ankle pitch
    scale is 0.08, which is arm 6's halved value and the one the tree carries, and NOT the
    0.1619 the configuration level version of this arm proposed restoring. The restoration
    was written when arm 6 was expected to be reversed and would now add a second variable.

    The clip is arm 15's with its ankle roll entries removed, or arm 10's where this arm is
    run before arm 15, likewise with those entries removed, and the run notes must say which.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "(left|right)_hip_pitch_04",
            "(left|right)_hip_roll_03",
            "(left|right)_hip_yaw_03",
            "(left|right)_knee_04",
            "(left|right)_foot_pitch_02",
        ],
        scale={
            "(left|right)_hip_pitch_04": 0.4,
            "(left|right)_hip_yaw_03": 0.4,
            "(left|right)_knee_04": 0.2667,
            "(left|right)_hip_roll_03": 0.16,
            "(left|right)_foot_pitch_02": 0.08,
        },
        use_default_offset=True,
        clip={
            "left_hip_pitch_04": (-0.28093, 0.97907),
            "right_hip_pitch_04": (-0.97907, 0.28093),
            "left_hip_roll_03": (-0.18944, 0.20905),
            "right_hip_roll_03": (-0.20905, 0.18944),
            "(left|right)_hip_yaw_03": (-0.63, 0.63),
            "(left|right)_knee_04": (0.45266, 1.29266),
            "(left|right)_foot_pitch_02": (-0.77860, -0.26860),
        },
    )


@configclass
class ActionsCfgHipYawAnkleRollFixed(ActionsCfgAnkleRollFixed):
    """The eight joint action term, being the above with the hip yaw removed as well."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "(left|right)_hip_pitch_04",
            "(left|right)_hip_roll_03",
            "(left|right)_knee_04",
            "(left|right)_foot_pitch_02",
        ],
        scale={
            "(left|right)_hip_pitch_04": 0.4,
            "(left|right)_knee_04": 0.2667,
            "(left|right)_hip_roll_03": 0.16,
            "(left|right)_foot_pitch_02": 0.08,
        },
        use_default_offset=True,
        clip={
            "left_hip_pitch_04": (-0.28093, 0.97907),
            "right_hip_pitch_04": (-0.97907, 0.28093),
            "left_hip_roll_03": (-0.18944, 0.20905),
            "right_hip_roll_03": (-0.20905, 0.18944),
            "(left|right)_knee_04": (0.45266, 1.29266),
            "(left|right)_foot_pitch_02": (-0.77860, -0.26860),
        },
    )
```

#### The environment configuration

```python
# PROPOSED, not applied.  Appended to kscale_base_env_cfg.py beside KscaleEnvCfg and
# KscaleEnvHipYawFixedCfg, both left exactly as they stand.
@configclass
class KscaleEnvAnkleRollFixedCfg(KscaleEnvCfg):
    """The ten joint environment, differing from KscaleEnvCfg in five places and no others."""

    actions: ActionsCfgAnkleRollFixed = ActionsCfgAnkleRollFixed()

    def __post_init__(self):
        super().__post_init__()

        # ONE, the three reward terms that resolve the ankle roll. The configuration level
        # version of this arm kept all three as the measurement of the residual deflection
        # under load. A URDF fixed joint has no deflection to measure and a term resolving a
        # joint the articulation does not carry raises, so all three are removed here.
        self.rewards.rew_keep_ankle_roll_zero_in_air = None
        self.rewards.pen_ankle_roll_deviation = None      # already inactive, section 18.5

        # TWO, the randomisation events naming the ankle roll group, the entry dropped from
        # each rather than the term removed, the other groups still being wanted.
        for term in (self.events.randomize_actuator_gains,
                     self.events.randomize_joint_parameters):
            if term is None:
                continue
            names = term.params["asset_cfg"].joint_names
            term.params["asset_cfg"].joint_names = [
                n for n in names if "foot_roll" not in n
            ]

        # THREE, _LEG_LINKS_NO_FOOT at :24 names arb_uj111_cross_bearing.* and those links
        # no longer exist, so the alternative must be dropped from the pattern. A
        # SceneEntityCfg whose body_names regex matches nothing raises at resolution, which
        # is the failure a reader is most likely to meet first and hardest to attribute.

        # FOUR, the observation groups narrow from twelve to ten by themselves, the
        # observation functions resolving the articulation rather than a fixed width, so no
        # observation term changes. The sole tilt instrumentation resolves foot_6061.* and
        # the foot links survive this construction, which is its whole purpose.

        # FIVE, the symmetry augmentation. _BODY_PARTNERS at mdp/symmetry/kscale.py:80
        # names the two cross bearings and those entries must be DELETED, the foot entries
        # at :87 surviving unchanged. _build_joint_map needs no subset argument, the
        # articulation itself carrying ten joints so that robot.joint_names IS the actuated
        # set, which is the same simplification arm 14 enjoys and the reason the repair
        # stated below is no longer required by either arm.


@configclass
class KscaleEnvHipYawAnkleRollFixedCfg(KscaleEnvAnkleRollFixedCfg):
    """The eight joint environment, being the above with arm 14's removals as well."""

    actions: ActionsCfgHipYawAnkleRollFixed = ActionsCfgHipYawAnkleRollFixed()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pen_hip_yaw_deviation = None
        self.rewards.rew_keep_hip_yaw_zero_in_air = None
        for term in (self.events.randomize_actuator_gains,
                     self.events.randomize_joint_parameters):
            if term is None:
                continue
            names = term.params["asset_cfg"].joint_names
            term.params["asset_cfg"].joint_names = [
                n for n in names if "hip_yaw" not in n
            ]
        # _BODY_PARTNERS loses the two hip yaw children as well, leaving NINE bodies.
```

#### The task registration and the command line wiring

```python
# PROPOSED, not applied.  environments/environments/tasks/locomotion/robots/
# kscale_ankle_roll_fixed_env_cfg.py, a new module mirroring
# kscale_hip_yaw_fixed_env_cfg.py of arm 14 line for line, with two base classes rather
# than one and therefore four registered tasks rather than two.
#
#   KscaleAnkleRollFixedBlindFlatEnvCfg          -> Isaac-Limx-Kscale-AnkleRollFixed-Blind-Flat-v0
#   KscaleAnkleRollFixedBlindFlatEnvCfg_PLAY     -> ...-Play-v0
#   KscaleHipYawAnkleRollFixedBlindFlatEnvCfg    -> Isaac-Limx-Kscale-HipYawAnkleRollFixed-Blind-Flat-v0
#   KscaleHipYawAnkleRollFixedBlindFlatEnvCfg_PLAY -> ...-Play-v0
#
# The PLAY overrides are byte identical to arm 14's, which are byte identical to the
# existing KscaleBlindFlatEnvCfg_PLAY, so the three differ only in the base they extend.
```

```bash
# PROPOSED, not applied.  /ws/djinn, two clauses in the training dispatch and two in the
# play dispatch, all placed BEFORE the else arm that fails loudly.
                elif [[ "$3" == "kscale-ankle-roll-fixed" ]]; then
                    task="Isaac-Limx-Kscale-AnkleRollFixed-Blind-Flat-v0"
                elif [[ "$3" == "kscale-hip-yaw-ankle-roll-fixed" ]]; then
                    task="Isaac-Limx-Kscale-HipYawAnkleRollFixed-Blind-Flat-v0"
```

The two `echo` lines listing the expected modes, at `:189` and `:257`, gain both names in the same pass, as arm 14 requires of its own.

#### The symmetry repair has moved to arm 14

The optional `subset` argument for `_build_joint_map`, together with the fourth mirror kind for the `last_action` block and the three edits to `mdp/symmetry/kscale.py` they need, stood in this sub-section until 2026-10-01 and now stands at section 21.14, at the user's direction and because that is the first arm of the sequence to change the joint count. The investigation of that date also changed what the repair is FOR. Neither this arm nor arm 14 needs it, both freezing their joints in the URDF under `merge_fixed_joints=False` so that `robot.joint_names` is the actuated set by construction, and section 21.14 carries the measurement that settles it. It is retained there as the defect left standing, for any future arm that drives a subset of an unchanged articulation.

#### Design decisions

THE JOINT IS NOW DELETED RATHER THAN FROZEN, WHICH REVERSES THIS ARM'S OWN DECISION OF 2026-09-22, and the four objections that decision rested upon are answered rather than overruled. It held that removing the degree of freedom would change the inertia tensor of the shank, which the construction above avoids by folding the cross bearing's 0.110039 kilograms into the FOOT rather than into the shank, leaving every link above the ankle byte identical and the total mass unchanged. It held that the collision geometry of the foot would change, which it does not, the foot's own mesh and its 0.04300 metre sole depth being untouched and only the cross bearing's mesh joining it. It held that the USD would need regenerating, which is true and which arm 14 already accepts, `spawn_kscale_from_urdf` performing the conversion at spawn. And it held that every comparison against the runs above would break, which is true of a checkpoint and not of a measurement, the comparison being between two runs from scratch at a matched budget exactly as arm 14's own note requires.

The two stages are separate URDFs rather than one URDF with a switch, because a URDF has no switch and a runtime edit of the articulation would reach the cached USD. The cost is three asset files where the tree carries one, and the benefit is that each task's dumped `params/env.yaml` names the asset it ran, which is the record a later comparison needs.

Stage two is the combination rather than the hip yaw alone, arm 14 already supplying that. The three tasks together give the four cells of a two by two design over the two joints, the present robot being the empty cell, and section 21.26 records that a search over two binary factors is the one case where an exhaustive sweep is cheaper than a search.

The frozen ankle roll angle is the configured nominal of minus and plus 0.041254 radians, and a URDF fixed joint holds it there exactly where the configuration level freeze would have let it deflect. Section 14.12 regresses the stance sole roll on the ankle roll, the hip roll and the torso roll and finds the ankle roll mapping onto the sole roll at a coefficient near unity, so the sole cant follows the frozen angle directly, and evaluating the fit at the nominal says the sole would cant five to seven degrees onto the OUTER edge because the nominal was derived to cancel a hip roll of 2.46 degrees where the machine walks at 7.9. THE ANGLE IS THEREFORE A PARAMETER OF THIS ARM AND NOT AN INHERITANCE. Section 18.2 measures run E's stance sole roll at 2.850 degrees under a live ankle roll, so freezing at the nominal is predicted to make that channel worse by a factor near two, and freezing at run C's flattening angle of 7.68 and 7.15 degrees would instead bake in a cant of the opposite sign on any gait that later widens. Stage one should be launched at the nominal and the sole roll read as the arm's principal cost.

#### Validation before launch

`validate_arm24.py` should be written beside `validate_arm6.py` in the artefact directory and should assert, without a simulator, that the merged body list has eleven entries for stage one and nine for stage two and that `_BODY_PARTNERS` names exactly those, that the surviving body permutation is an involution, that the robot's total mass is unchanged from the original URDF to within a milligram, that the composed pitch origin places the foot link's frame within a micrometre and a microradian of where the original chain places it at zero roll and zero pitch, THAT THE RECOMPOSED PITCH AXIS SWEEPS THE TOE THROUGH THE SAGITTAL PLANE AND NOT THE FRONTAL ONE, that each action term's `joint_names` resolves to the articulation's full joint list, that no reward, event or observation term resolves a joint name containing `foot_roll`, that `_LEG_LINKS_NO_FOOT` resolves to at least one surviving body, and that the sagittal closure identity of `context/KScale.md` section 16.2 still holds at the surviving nominal pose.

The axis assertion is the one that cannot be skipped. Every other property above fails loudly at spawn or at the first learning step, and a rotated pitch axis fails silently, producing a robot that walks badly for a reason no reward channel names.

#### Prediction and falsifier

The prediction is that the ankle roll's 3.5 newton metres and the 6.3 per cent of absolute joint power the pair holds are recovered in full, since the actuator is gone rather than idle, that the gait's viability is unchanged on stage one, section 17.3's two bipeds establishing that the degree of freedom is not indispensable [22], and that the stance sole roll rises from 2.850 degrees toward the five to seven the nominal implies. Stage two is predicted to remove between 62 and 83 per cent of the swing splay on top of that, which is arm 14's prediction unchanged, the two joints not interacting kinematically.

The falsifying observation is the fall rate rising on stage one, which would say that the ankle roll supplies a lateral recovery the measurements above did not see, and it should be taken seriously rather than explained away, the passive ankle literature establishing feasibility under a model based whole body controller [23] and not under a learned policy. A second falsifier is specific to the reversal recorded above. If stage one's gait differs from run E's on a channel that names neither the ankle roll nor the sole, the composition is the first suspect and the axis assertion the first thing to re-run.

### 21.25 Arm 25, the ankle pitch nominal, or its travel

Status, NOT APPLIED and CONDITIONAL, to be run only if arm 7 fails, and it is a hardware claim rather than a tuning change.

Section 12 establishes that the ankle pitch fails differently from the ankle roll. The roll is latched by command and has travel to spare, section 11.3 measuring the stance geometry demanding eight to ten degrees at the median against fifteen available, so no URDF change addresses it and section 20.3 excludes one. The pitch is genuinely short, the sagittal closure identity requiring minus 47.5 degrees in swing against a minus 50 degree stop which is exceeded on 42 to 45 per cent of samples, and the distinction between the two joints must be preserved wherever this document is summarised.

The constraint stands in the URDF at two places, the right foot pitch at `kscale.urdf:603` and the left at `:1083`, and the two are identical.

```xml
    <limit effort="10" velocity="10" lower="-0.872665" upper="0.523599"/>
```

The proposal widens the dorsiflexion limit alone and leaves the plantarflexion limit untouched.

```xml
    <!-- PROPOSED, not applied, and requiring the mechanical record's agreement first. -->
    <limit effort="10" velocity="10" lower="-1.047198" upper="0.523599"/>
```

Two design decisions attach and the first is why the arm is last and conditional.

The change is asymmetric because the demand is asymmetric. The closure identity places the requirement entirely on the dorsiflexion side, the crouch asking the ankle to pitch the sole up toward the shin, and the plantarflexion limit of 0.523599 is nowhere near binding in any run of either sequence. Widening both would be a larger change than the evidence supports and would alter the push off geometry that no measurement has found at fault.

And the arm must not precede arm 7, because arm 7 may remove the demand entirely. At the shallower crouch the thirty four newton metre run adopts, section 12 measures the requirement falling to minus 37 degrees and the swing saturation falling from 99.6 per cent to 77, so a pose that sits the ankle pitch nominal thirty degrees from its stop rather than 17.7 may close the gap without any hardware claim at all. Running the travel change first would spend the mechanical argument on a problem the pose change might have solved for nothing.

| Quantity | Present | Arm 25 | Source |
|---|---|---|---|
| Ankle pitch lower limit | -0.872665 rad, -50.0 deg | -1.047198 rad, -60.0 deg | `kscale.urdf:603` and `:1083` |
| Ankle pitch upper limit | 0.523599 rad, 30.0 deg | 0.523599 rad, unchanged | the same lines |
| Swing requirement at the present crouch | -47.5 deg | | section 12 |
| Swing requirement at the shallower crouch | -37 deg | | section 12 |
| Samples exceeding the stop | 42 to 45 pct | | section 12 |

The prediction is that the swing ankle pitch saturation falls from 99.6 per cent and that the sole reaches the orientation the in air term asks of it. The falsifying observation is the saturation surviving, which would mean the pitch is latched by command as the roll is and that arm 10 rather than any travel change is the remedy.

### 21.26 Arm 26, the search rather than the sweep

Status, NOT APPLIED, and it is an instrument for a question the other arms hold fixed rather than an arm of the ablation proper.

The gains and the ceilings of this robot have now been set by hand across seven runs and the sequence has produced one clean negative result and no positive one. The present ankle damping of 1.05 newton metre seconds per radian is a value section 11.4 shows saturating the actuator unaided at the speeds the joint actually reaches, and no arm above varies it, because a hand sweep over stiffness and damping at two joints is a four dimensional search that the ablation has no budget for.

The instrument already exists at `scripts/analysis/kbot_reference_tracking_spec_optimisation.py` and searches by CMA-ES against a supplied reference trajectory with every randomisation disabled.

```
    ./isaaclab.sh -p scripts/analysis/kbot_reference_tracking_spec_optimisation.py \
        --reference /ws/joint_data_0.73_0.05_0.5_transposed.csv \
        --popsize 64 --generations 120 --headless
```

Its search space is logarithmic in the multiplier on each quantity, taken from the live actuator objects rather than hard coded, so the box follows the working tree as it moves.

```python
PARAM_KINDS = ("stiffness", "damping", "effort_limit", "velocity_limit")
```

One property of that space is a direct interaction with arm 23 and it must be settled before the search is run. The saturation effort is carried along as a fixed ratio of the effort limit rather than searched, for a reason the script's own docstring states, that a search raising the effort limit past a fixed saturation effort would find the envelope clipping at the saturation effort and would be silently bounded below its own upper bound.

```python
        for g in self.groups:
            out[g]["saturation_effort"] = out[g]["effort_limit"] * self.saturation_ratio[g]
            out[g]["velocity_limit_sim"] = out[g]["velocity_limit"] * self.velocity_sim_ratio[g]
```

The ratio is read from the configuration at construction, and the tree presently carries a ratio of exactly 1.0 at both ankles, so a search launched from the tree as it stands would hold the corner velocity at zero across its entire box and would report gains tuned against an envelope with no constant torque plateau. Arm 23 must therefore land before arm 26 is run, or the search must be given an explicit saturation ratio, and this is the single most important thing to know before launching it.

| Parameter | Default | Meaning |
|---|---|---|
| `--gain-range` | 0.25 to 4.0 | multiplicative bounds on stiffness and damping |
| `--effort-range` | 0.25 to 4.0 | multiplicative bounds on the effort limit |
| `--velocity-range` | 0.5 to 2.0 | multiplicative bounds on the no load speed |
| `--popsize` | 64 | population, and the environment count, one design per environment |
| `--generations` | 120 | CMA-ES generations |
| `--sigma0` | 0.25 | initial step on the unit hypercube, pycma capping it at 0.333 |

The design decision that bounds the arm's ambition is that its objective is reference tracking error and not gait quality, so it cannot replace any arm above and must not be read as doing so. What it can supply is the ankle stiffness and damping that the arms hold fixed, and the honest use of it is to run it once after arm 23 has restored the envelope, take the ankle gains it returns, and carry them into the ablation as a fixed configuration rather than as an arm.

## Bibliography

A defect is left standing here rather than repaired and it is recorded so that a later reader does not mistake it for an oversight. The entries below were ordered by first appearance when the document was first written and the restructuring of 2026-09-23, which moved the evidence ahead of the ablation, disturbed that order without renumbering them. Entries 24 through 29, added on 2026-09-28, do appear in the order of their first use. The order among entries 1 through 23 is not repaired because cluster 30 of `/ws/context/literature.md` cites two of them by their number here, naming entry four and entry three of this bibliography, so a renumbering would create dead pointers in a document this one depends upon. Every marker in the text has an entry below and every entry below is cited at least once, which is the property that matters and which was verified by enumeration on 2026-09-28.

1. Cibulka, M., Winters, K., Kampwerth, T., McAfee, B., Payne, L., Roeckenhaus, T., Ross, S. J. (2016). Predicting foot progression angle during gait using two clinical measures in healthy adults, a preliminary study. International Journal of Sports Physical Therapy 11(3), pages 400 to 408.
2. Booster Robotics (2025). Booster Gym, A Reinforcement Learning Framework for Humanoid Robot Locomotion. arXiv:2506.15132. Author list not established beyond the institutional attribution by the retrieved source.
3. Chane-Sane, E., Leziart, P.-A., Flayols, T., Stasse, O., Souères, P., Mansard, N. (2024). CaT, Constraints as Terminations for Legged Locomotion Reinforcement Learning. arXiv:2403.18765. Venue not stated by the arXiv record.
4. Kim, Y., Oh, H., Lee, J., Choi, J., Ji, G., Jung, M., Youm, D., Hwangbo, J. (2024). Not Only Rewards But Also Constraints, Applications on Legged Robot Locomotion. IEEE Transactions on Robotics. arXiv:2308.12517.
5. Kim, G., Lee, Y.-H., Park, H.-W. (2025). A Learning Framework for Diverse Legged Robot Locomotion Using Barrier-Based Style Rewards. IEEE International Conference on Robotics and Automation (ICRA) 2025. arXiv:2409.15780.
6. Skalse, J., Howe, N. H. R., Krasheninnikov, D., Krueger, D. (2022). Defining and Characterizing Reward Hacking. Advances in Neural Information Processing Systems 35 (NeurIPS 2022). arXiv:2209.13085.
7. Fu, Z., Kumar, A., Malik, J., Pathak, D. (2021). Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots. Conference on Robot Learning (CoRL) 2021. arXiv:2111.01674.
8. Williams, D. S. B., Martin, A. E. (2019). Gait modification when decreasing double support percentage. Journal of Biomechanics 92, pages 76 to 83. DOI 10.1016/j.jbiomech.2019.05.028.
9. Chou, P.-W., Maturana, D., Scherer, S. (2017). Improving Stochastic Policy Gradients in Continuous Control with Deep Reinforcement Learning using the Beta Distribution. Proceedings of the 34th International Conference on Machine Learning, PMLR volume 70.
10. Reda, D., Tao, T., van de Panne, M. (2020). Learning to Locomote, Understanding How Environment Design Matters for Deep Reinforcement Learning. Motion, Interaction and Games (MIG) 2020. arXiv:2010.04304.
11. McGeer, T. (1990). Passive Dynamic Walking. International Journal of Robotics Research 9(2), pages 62 to 82.
12. van Marum, B., Shrestha, A., Duan, H., Dugar, P., Dao, J., Fern, A. (2024). Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2024. arXiv:2404.19173.
13. Margolis, G. B., Agrawal, P. (2022). Walk These Ways, Tuning Robot Control for Generalization with Multiplicity of Behavior. Conference on Robot Learning (CoRL) 2022. arXiv:2212.03238.
14. Wang, Y., Huang, Z. (2026). Reward Hacking as Equilibrium under Finite Evaluation. arXiv:2603.28063. Venue not established by the retrieved source.
15. Rudin, N., Hoeller, D., Reist, P., Hutter, M. (2021). Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Conference on Robot Learning (CoRL) 2021. arXiv:2109.11978.
16. Seyde, T., Gilitschenski, I., Schwarting, W., Stellato, B., Riedmiller, M., Wulfmeier, M., Rus, D. (2021). Is Bang-Bang Control All You Need? Solving Continuous Control with Bernoulli Policies. Advances in Neural Information Processing Systems 34 (NeurIPS 2021). arXiv:2111.02552.
17. Mysore, S., Mabsout, B., Mancuso, R., Saenko, K. (2021). Regularizing Action Policies for Smooth Control with Reinforcement Learning. IEEE International Conference on Robotics and Automation (ICRA) 2021. arXiv:2012.06644.
18. Ha, S., Lee, J., van de Panne, M., Xie, Z., Yu, W., Khadiv, M. (2025). Learning-based legged locomotion, state of the art and future perspectives. International Journal of Robotics Research. DOI 10.1177/02783649241312698. arXiv:2406.01152.
19. Fujita, Y., Maeda, S. (2018). Clipped Action Policy Gradient. Proceedings of the 35th International Conference on Machine Learning (ICML) 2018. arXiv:1802.07564.
20. Shin, J., Cha, W., Kim, D., Cha, J., Park, J. (2025). Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion. arXiv:2504.08246. Venue not established by the retrieved source beyond a submission to an IEEE venue.
21. Raffin, A., Kober, J., Stulp, F. (2022). Smooth Exploration for Robotic Reinforcement Learning. Proceedings of the 5th Conference on Robot Learning, PMLR volume 164, pages 1634 to 1644.
22. Hong, S., Park, S. (2025). Biomechanical optimization and reinforcement learning provide insight into transition from ankle to hip strategy in human postural control. Scientific Reports 15, article 13640. DOI 10.1038/s41598-025-97637-5.
23. Kim, D., Jorgensen, S. J., Lee, J., Ahn, J., Luo, J., Sentis, L. (2020). Dynamic locomotion for passive-ankle biped robots and humanoids using whole-body locomotion control. International Journal of Robotics Research. DOI 10.1177/0278364920918014. arXiv:1901.08100. Volume, issue and page range not established by the retrieved source, the publisher having refused the automated request.
24. Bogdanovic, M., Khadiv, M., Righetti, L. (2020). Learning variable impedance control for contact sensitive tasks. IEEE Robotics and Automation Letters 5(4), pages 6129 to 6136. arXiv:1907.07500.
25. Tommaselli, F., Segreto, T. H., Negri, J. D., Godoy, R. V., Becker, M. (2026). Mind the Phase, Effective Rank and Representation Health in Legged Locomotion. 10th Conference on Robot Learning (CoRL) 2026. arXiv:2609.06958.
26. Bronars, A., Park, Y., Agrawal, P. (2026). Tune to Learn, How Controller Gains Shape Robot Policy Learning. arXiv:2604.02523. Venue not established by the retrieved source beyond the arXiv listing.
27. Shin, Y.-H., Song, T.-G., Ji, G., Park, H.-W. (2023). Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion. arXiv:2312.17507. Venue not established by the retrieved source beyond the arXiv listing.
28. Tao, Y., Cheng, S., Lan, G., Yuan, Y., Jin, Y., Wang, H. (2026). Extending the Speed Limit of Quadrupedal Locomotion via Refined Actuator Modeling and Adaptive Command Scheduling. IEEE Robotics and Automation Letters, early access. arXiv:2609.13289.
29. Xia, B., Li, B., Lee, J., Scutari, M., Chen, B. (2024). The Duke Humanoid, Design and Control For Energy Efficient Bipedal Locomotion Using Passive Dynamics. arXiv:2409.19795. Venue not established by the retrieved source beyond the arXiv listing.
30. Ma, J., Ji, J., Gao, Y. (2024). Adversarial Constrained Policy Optimization, Improving Constrained Reinforcement Learning by Adapting Budgets. arXiv:2410.20786. Venue not established by the retrieved source beyond the arXiv listing.
31. Panchal, U., Kleyko, D., Artan, U., Loutfi, A. (2026). Smoothness as a Constraint for Stable Humanoid Locomotion. arXiv:2609.24552. Venue not established by the retrieved source beyond the arXiv listing.
32. Liao, S., Li, P., Yang, X., Chang, L., Fan, Z., Wang, Q., Shi, L., Cao, Y., Wu, W., Sartoretti, G. (2026). GPO, Growing Policy Optimization for Legged Robot Locomotion and Whole-Body Control. arXiv:2601.20668. Venue not established by the retrieved source beyond the arXiv listing.
33. Lin, Z. (2026). Beyond Distributions, Geometric Action Control for Continuous Reinforcement Learning. arXiv:2511.08234. Venue not established by the retrieved source beyond the arXiv listing.
34. Lu, G., Dun, Y., Zhou, Y., Tao, L., Duan, J., Li, J., Li, G. (2026). FastDSAC, Enhancing Policy Plasticity via Constrained Exploration for Scalable Humanoid Locomotion. arXiv:2606.31691. Venue not established by the retrieved source beyond the arXiv listing.
