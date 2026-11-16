# Artefacts Index

This directory holds the debugging artefacts of the simulation repository, the exported contact sheets, panels and figures that the context and plan documents of this repository cite as evidence. Its purpose is twofold, to keep the repository root clear of loose exports, and to preserve a durable experimental memory, so that a claim made in a document about a training run can still be checked against the figure that produced it long after the run itself has been deleted from the log tree.

The distinction from the training log tree matters. A run's own outputs, its checkpoints, its dumped configuration, its videos and its event files remain where the trainer writes them under `/ws/IsaacLab/logs/rsl_rl/<experiment>/<timestamp>/`, and that tree is transient, its runs being deleted as disk fills. This directory holds only the artefacts that were exported out of that tree for analysis, and those are permanent.

The workspace maintains its own such directory at `../../../context/artefacts/`, indexed separately, which holds the artefacts of investigations spanning the repository, the vendored libraries and the container tooling together. An artefact belongs here when the document that cites it belongs here.

Two conditions govern admission and both must hold. An artefact must be cited by a context or a plan document of this repository, whether individually or as a set the document reasons about, and it must be attributable to the run that produced it. An artefact that no document cites is evidence of nothing and does not belong here, and an artefact whose provenance cannot be established could not be quoted against a run even if it were cited. Both conditions are enforced by deletion rather than by annotation, so that the presence of a file is itself the assurance that some document depends upon it.

## Register

| Directory | Contents | Attributed run | Provenance of the attribution |
|---|---|---|---|
| `2026-08-25_08-14-22-train-video/` | Two contact sheets extracted from the training video, `video_frames_episode.png` sampling the whole recording at sixteen equally spaced frames and `video_frames_first_second.png` sampling its first one hundred and seventy frames, both annotated with the frame index | `quadruped_flat/2026-08-25_08-14-22`, the rough terrain quadruped task, 85 iterations | Certain, both sheets were extracted with OpenCV from `videos/train/rl-video-step-0.mp4` in that run's own directory, whose md5 is `6a0aa96c047758dc98914e2a2229228a` |
| `kscale_flat-2026-09-03_09-27-59/` | Two contact sheets from the training video at three hundred and twenty thousand steps, `train_320000_episode_span.png` sampling the whole recording at twelve equally spaced frames and `train_320000_one_cycle.png` sampling twelve frames at three frame spacing over one gait cycle, together with `scripts/`, the ten analysis programs that produced every figure quoted for this sequence, `tb2.py` extracting the scalar record of five runs, `p2.py` and `p3.py` diffing their dumped configurations through the repository loader, `an5.py` and `an6.py` reducing the training trajectory and the weighted per second reward budget, `splay3.py` computing the signed foot heading from the foot link's negative z axis, `fd2.py` the stance width hinge and its correlates, `kin2.py` the saturation, stop residency and gait quality tables, `defl.py` the commanded joint deviation reconstructed by inverting the control law, and `fr2.py` the frame extraction | `kscale_flat/2026-09-03_09-27-59`, arm 0, 23574 iterations | Certain, the sheets were extracted with OpenCV from `videos/train/rl-video-step-320000.mp4` in that run's own directory and the scripts read that directory and four others by absolute path, the run identifiers being written into each script |
| `kscale_flat-2026-09-03_09-49-22/` | One contact sheet, `train_320000_episode_span.png`, twelve equally spaced frames spanning the training video at three hundred and twenty thousand steps | `kscale_flat/2026-09-03_09-49-22`, arm R, 13444 iterations | Certain, extracted with OpenCV from `videos/train/rl-video-step-320000.mp4` in that run's own directory |
| `kscale_flat-2026-09-04_10-15-18/` | Two contact sheets, `train_320000_one_cycle.png` and `train_710000_one_cycle.png`, each twelve frames at three frame spacing over one gait cycle, at the same training step as the corresponding sheets of the other runs and at the last recorded step | `kscale_flat/2026-09-04_10-15-18`, arm 0S, 30000 iterations | Certain, extracted with OpenCV from `videos/train/rl-video-step-320000.mp4` and `rl-video-step-710000.mp4` in that run's own directory |
| `kscale_flat-2026-09-04_10-40-30/` | Two contact sheets, `train_320000_episode_span.png` spanning the whole recording at twelve equally spaced frames and `train_710000_one_cycle.png` over twelve consecutive sampled frames at the last recorded step | `kscale_flat/2026-09-04_10-40-30`, arm CS, 30000 iterations | Certain, extracted with OpenCV from `videos/train/rl-video-step-320000.mp4` and `rl-video-step-710000.mp4` in that run's own directory |
| `tron1_quadruped-mjcf-mirror-audit-2026-10-20/` | The sagittal mirror audit of the two MuJoCo documents, the script and its output | No run, the TRON1 and quadruped MJCF of the working tree of 2026-10-20 | Certain, the output regenerates from the script against documents of recorded md5 |
| `quadruped_symmetry-2026-10-20_06-54-05-sandbox-smoke/` | The probe output, iteration log, dumped symmetry configuration and environment configuration diff of a three iteration sandbox run | `quadruped_symmetry/2026-10-20_06-54-05` in `/tmp/symsandbox_20261020` of `isaac-lab-base-shreyas-gpu1`, a sandbox and not the shared log tree | Certain, extracted from that run's own log and `params/` on the day it ran |
| `sf_tron_1a_symmetry-2026-10-20_06-56-00-sandbox-smoke/` | The same for the TRON1 | `sf_tron_1a_symmetry/2026-10-20_06-56-00` in the same sandbox | Certain, as above |
| `sf_quadruped_brs_kscale-obs-term-census-2026-10-20/` | Every observation group and term of the thirty three training tasks of the SF, quadruped, SD_BRS1 and KScale families, with each term's function and parameters, and the script that produced it | No run, the environment configurations of the working tree of 2026-10-20 instantiated without a simulation | Certain, the output regenerates from the script against the tree of that date |
| `symmetry_revised-2026-10-20-sandbox-smoke/` | Excerpts of the logs of twelve sandbox training runs, two reruns, two runs of the unmodified live tree and two replays, their status table, the dumped symmetry configurations, and the scripts that launched them | The runs of `/tmp/symsandbox_v2_20261020` of `isaac-lab-base-shreyas-gpu1` on 2026-10-20, a sandbox and not the shared log tree | Certain, extracted from each run's own log and `params/` on the day it ran |
| `symmetry_applied-2026-10-20-live-smoke/` | The status table, the symmetry loss of every run read from its event file, the final checkpoint and dumped augmentation function of every run, and the script, of the eight task smoke test run on the live tree after the symmetry plan was applied | The runs under `/tmp/symmetry_smoke/logs/rsl_rl/` of `isaac-lab-base-shreyas-gpu1` on 2026-10-20, a scratch directory and not the shared log tree | Certain, read from each run's own event file and `params/` the same day |

The `2026-08-25_08-14-22-train-video/` directory is cited as a set by section 1 of [../../plans/quadruped-floot-penetration-fix.md](../../plans/quadruped-floot-penetration-fix.md), which reasons about both sheets together as the visual record of the fall through, the first establishing that the descent continues to the end of the episode and the second that it begins within the first second.

The four `kscale_flat-` directories are cited individually and as a set by section 5 of [../../plans/kscale_integration.md](../../plans/kscale_integration.md), which reasons about the four runs together as the sequence that separates the mechanical substrate from the reward set, and by section 7.15 of the same document. The span sheets of arms R and CS are the visual record of the two distinct failures, prone on the ground after an entropy collapse in the first and buckling into a crouch within a quarter second in the second, and the cycle sheets of arm 0S are the record of the walk against which the proposed baseline is to be read. The scripts are retained rather than the intermediate arrays they produced, the arrays being recoverable from the log tree while it survives and the scripts being the statement of how every quoted figure was computed.

### `kscale_flat-2026-09-08_velocity_clamp/`

Attribution is by construction. Every file here was produced on 2026-09-09 by scripts held alongside it, run against `IsaacLab/logs/rsl_rl/kscale_flat/2026-08-28_04-50-51`, `2026-09-08_06-50-00` and `2026-09-08_07-39-29`, whose `data/42/dump.npy` and event files are the sole inputs. The scripts are retained so that every figure can be regenerated, the training logs being transient.

| File | What it holds | Cited by |
|---|---|---|
| `play_contact_sheet_2026-08-28.png` | Eight frames spanning `videos/play/42/rl-video-step-0.mp4` of the reference run, the robots upright and walking | `../KScale.md`, correction of 2026-09-09 |
| `play_contact_sheet_2026-09-08_06-50-00.png` | The same eight frames of the run at action scale 0.25, the robots prone | `../KScale.md`, correction of 2026-09-09 |
| `play_contact_sheet_2026-09-08_07-39-29.png` | The same eight frames of the run at action scale 0.4, the robots prone | `../KScale.md`, correction of 2026-09-09 |
| `velocity_clamp_evidence.txt` | The four tables the correction quotes, being per joint saturation and clamp occupancy, the torque directed into the clamp, the separation decomposition, and the torque available at the clamp under each of the three configurations | `../KScale.md`, correction of 2026-09-09, and `../../plans/kscale_integration.md` chapter 6 |
| `d3.py`, `d5.py`, `d6.py`, `q1.py` | The scripts that produce the four tables, in that order | as above |

### `kscale-self-collision-2026-09-11/`

Attribution is by extraction. Every file here was cut on 2026-09-11 with OpenCV from `IsaacLab/logs/rsl_rl/kscale_flat/2026-09-11_04-56-51/videos/train/rl-video-step-0.mp4`, the frame index forming the file name. That run is the first with `enabled_self_collisions: true`, recorded at line 139 of its `params/env.yaml`, and is the run whose mean episode length fell to about two steps. The geometric audit that accompanies these frames is reproducible from `scripts/analysis/kscale_self_collision_audit.py`, which reads the URDF and the STL set directly and needs no training log.

| File | What it holds | Cited by |
|---|---|---|
| `frame_0001.png` | The first rendered frame, one robot already prone and one in the reset pose | the self collision diagnosis of 2026-09-11 |
| `frame_0010.png` | The same pair nine frames later, neither having advanced, which is what continuous resetting looks like | as above |
| `frame_1000.png` | Ten seconds in, the robot still held at the reset pose | as above |


### `kscale_flat-2026-09-14_to_09-16-limits/`

Attribution is by construction and by extraction together. The three contact sheets were cut on 2026-09-16 with the `ffmpeg` binary vendored inside `imageio_ffmpeg` from `videos/play/42/rl-video-step-0.mp4` of each named run, one frame per second over the first twelve seconds, cropped to a 460 pixel square about the robot and tiled six by two. The evidence tables were produced the same day by the three scripts held beside them, run against `data/42/statistics.npy` and `data/42/dump.npy` of the same three runs and against their event files, which are their sole inputs. The scripts are retained so that every figure may be regenerated once the log tree is gone.

| File | What it holds | Cited by |
|---|---|---|
| `play-contact-sheet-2026-09-14_10-20-50.png` | Twelve frames of the run at the original ceilings, the machine upright with a vertical torso and alternating legs | `../../plans/kscale_actuator_limits_fix.md` section 3.1 |
| `play-contact-sheet-2026-09-15_08-07-56.png` | Twelve frames of the run with the hip pitch ceiling at 84 Nm, the machine folded into a deep crouch and shuffling without advancing | as above |
| `play-contact-sheet-2026-09-16_04-52-20.png` | Twelve frames of the run at the published K-Scale ceilings, the machine collapsing repeatedly, one frame flat on the ground | as above |
| `limits_evidence.txt` | The four tables the plan quotes, being the rollout summary, the per joint velocity saturation and torque envelope residency, the mechanical stop residency, and the median joint torque conditioned on base height with the knee flexion reachability check | `../../plans/kscale_actuator_limits_fix.md` sections 3.2, 3.4, 5.1, 5.2 and 5.3 |
| `pose_evidence.txt` | The standing geometry of four candidate nominal poses and the stance load and single support hold tables at the present pose and at the published K-Scale posture, together with the entropy turning point of the three runs | `../../plans/kscale_actuator_limits_fix.md` sections 3.3, 6.2, 9.1, 9.1.1 and 9.1.2 |
| `st.py`, `dmp.py`, `tb.py` | The scripts that read the statistics record, the trajectory dump and the event files respectively, and that produce every table above | as above |
| `pose.py` | The script that produces `pose_evidence.txt`, reading the URDF alone through `scripts/analysis/kscale_sole_analysis.py` and driving `scripts/analysis/kscale_stance_analysis.py` at a substituted nominal pose | as above |

The directory is named for the span of the three runs rather than for one of them, the plan reasoning about the set as a sequence in which each run differs from its predecessor in a countable number of actuator ceilings, and no single run being the subject.

### `kscale_flat-2026-09-17_to_09-18-gait-ablation/`

Attribution is by construction. Every frame here was cut on 2026-09-21 with the `ffmpeg` binary vendored inside `imageio_ffmpeg` from `videos/play/42/rl-video-step-0.mp4` of the run its filename names, at the timestamp its filename names, the four videos being 3000 frames at one hundred frames per second so that a frame index and a dump step coincide exactly. The zoomed frames are a 420 by 380 pixel crop about the tracked robot, that robot being environment zero throughout since the viewer configuration follows the root of the robot asset at environment index zero. Every script was written the same day and reads only the `params/env.yaml`, the `data/42/` dumps and the event files of the four runs, whose absolute paths each script carries, so each figure the plan quotes may be regenerated until the log tree is cleared.

| File | What it holds | Cited by |
|---|---|---|
| `frames/A_t6.0_zoom.png`, `B_t6.0_zoom.png`, `C_t6.0_zoom.png`, `D_t6.0_zoom.png` | The four policies at six seconds, where the commanded forward velocity is 0.079 m/s and the yaw command 0.64 rad/s, being the slowest sustained interval of the recording and the one in which the splay is largest | `../../plans/kscale_gait_shaping.md` section 3.4 |
| `frames/A_t25.0_zoom.png`, `B_t25.0_zoom.png`, `C_t25.0_zoom.png`, `D_t25.0_zoom.png` | The same four at twenty five seconds, during the commanded reverse of minus 0.235 m/s, where run C reaches its deepest crouch and run D carries the machine on its locked right leg | as above |
| `frames/A_t6.0_wide.png`, `C_t6.0_wide.png` | The uncropped 1280 by 720 frames of the two contrasting families at six seconds, retained so that a reader may judge the crop | as above |
| `scripts/diff_params.py` | The four way diff of the dumped `env.yaml` through the repository loader, which establishes the five substantive parameter differences and the identity of everything else | `../../plans/kscale_gait_shaping.md` section 2.1 |
| `scripts/tb_extract.py` | The scalar record of the four runs tabulated at matched iterations, which supplies the budget confound and the divergence of run D | sections 2.2 and 3.3 |
| `scripts/stats_compare.py`, `scripts/stats2.py` | The precomputed statistics of the four evaluation dumps compared side by side, the second using the statistic names the first found absent, which together supply the gait quality and the joint saturation tables | sections 3.1, 3.2 and 4.1 |
| `scripts/raw_analysis.py` | The signed joint positions, the soft and hard limit residencies and the torque signs read from the trajectory dumps | section 4.1 |
| `scripts/toein.py` | The signed toe-in computed by rotating each foot link's negative z axis into the world, with the speed binned crouch and toe-in tables | sections 3.1 and 4.4 |
| `scripts/actionscale.py` | The action authority audit, the commanded target reconstructed by inverting the control law, and the weighted per second reward budget | sections 4.6 and 4.7 |
| `scripts/closure.py` | The sagittal closure identity, the knee against the swept height table, the off axis excursion budget and the ankle roll lateral moment balance | sections 4.2, 4.3 and 4.4 |

The directory is named for the span of the four runs rather than for one of them, the plan reasoning about the set as an ablation over a reward configuration and no single run being its subject.


### `kscale_flat-2026-09-18_to_09-21-foot-joint-ablation/`

Attribution is by the dumped configuration rather than by the directory name, each run's ankle ceiling having been read from its `params/env.yaml` through `scripts/analysis/experiment_params.py:24`. The three runs are `2026-09-18_09-37-29` at 11.9 newton metres, `2026-09-21_06-45-40` at 17 and `2026-09-21_08-40-59` at 34, and the same check establishes that `2026-09-18_09-24-17` and `2026-09-21_08-37-13` are launches that did not survive their first minute, carrying a parameter dump and an initial checkpoint alone. The directory holds programs and no arrays or frames, the arrays being recoverable from the log tree while it survives. Its own index states the validation, eleven gait quantities of the 11.9 newton metre run reproducing to every digit the figures section 3 of the plan already carries for the run it calls C, which is the same run.

| File | What it holds | Cited by |
|---|---|---|
| `scripts/run_inventory.py`, `scripts/config_diff.py` | The ankle ceiling and gains of every candidate run directory, and the exhaustive leaf path comparison finding six differing paths of 2263 | `../../plans/kscale_gait_shaping.md` section 10.1 |
| `scripts/actuator_envelope.py` | Per joint saturation against the velocity dependent envelope and against the constant ceiling, split by stance and swing, with the corner velocity of each run | sections 10.2, 11.1 and 11.4 |
| `scripts/joint_stop_residency.py` | Joint position against the hard and soft stops with stance and swing medians, which establishes the ankle roll latch | section 11.1 |
| `scripts/commanded_target.py` | The commanded target reconstructed by inverting the control law over unsaturated samples, and its excursion beyond the mechanical stop | section 11.2 |
| `scripts/sole_flatness.py`, `scripts/ankle_roll_travel.py` | The sole normal in the base yaw frame, and the ankle roll angle that would lay the sole flat against the travel the joint has | sections 11.1 and 11.3 |
| `scripts/stop_reversals.py` | Stop to stop reversals per second of the ankle roll and the traverse speed, which quantify the impulse switching | section 11.4 |
| `scripts/sagittal_closure.py` | The closure identity and the ankle pitch the crouch demands in swing | section 12 |
| `scripts/toe_in.py`, `scripts/gait_metrics.py` | The signed toe-in with the mirrored left foot frame resolved, and the gait quality comparison of the three runs | sections 10.3 and 10.4 |
| `scripts/inair_kernel.py` | The value and the gradient of the three in air deviation terms against the velocity tracking reward, which establishes the kernel collapse | section 13 |
| `scripts/common.py`, `scripts/frames.py` | The run table with the DC motor envelope of `actuator_pd.py:295`, and the quaternion rotation with the empirical foot index resolution | as above |
| `scripts/dict_scale_semantics.py`, `scripts/validate_scale.py`, `scripts/tree_vs_run.py` | The three checks written when the per joint action scale was set on 2026-09-22, being the semantics of a dictionary valued scale including the silent 1.0 it gives an uncovered joint, the resolution of the dictionary actually written to the configuration against the twelve joint names, and the comparison of the tree's sixty actuator fields against the 17 N.m run's dump | `../../plans/kscale_gait_shaping.md` sections 18.1 and 19.4, the arm section 7 numbers 3 and section 18.1 numbers 8 |
| `scripts/sway_geometry.py`, `scripts/sway_measured.py`, `scripts/validate_sway.py` | The three programs written when the lateral weight shift was merged on 2026-09-22, being the lateral travel quasi static single support demands, the lateral torso excursion the three policies actually produce, and the reward manager signature gate and body resolution of the merged term | sections 20.1 and 21.5, arm 5 |
| `scripts/cop_edge.py` | Which lateral sole edge carries the contact in stance, by transforming both sole corners into the world and comparing their heights, carrying a synthetic self test of a plate rolled by a known angle | section 11.1 as corrected 2026-09-22 |
| `scripts/arm_parameterisation.py` | The four parameter tables of section 21 that are arithmetic on the configuration rather than measurements of a run, being the per joint action scale, the lateral sway kernel at both widths, the commanded target clip against the soft limits it must not silence, and the corner velocity. It reads no dump and needs no simulator, so those tables survive the deletion of the log tree | sections 22.4, 21.5, 21.10 and 21.14 |
| `scripts/nominal_pose.py` | The pose parameters of arm 7, derived by forward kinematics against `kscale.urdf` rather than quoted, being the standing height of the present and proposed postures with the roll nominals on and off, the settling margin priced against the reset distribution, the travel each nominal leaves, the stance load the sagittal fold moves and the action scale table at both poses. It found the standing height of the posture to be 0.73492 m and not the 0.73540 the plan and context/KScale.md carried, that figure having been taken at a zero roll pose | sections 22.7 and 21.6 |

The directory is named for the span of the three runs rather than for one of them, the plan reasoning about the set as an ablation over a single actuator ceiling and no one run being its subject.

### `kscale_flat-2026-09-21_to_09-22-scale-sway-ablation/`

Attribution is by the dumped configuration, `cfgdiff.py` comparing every leaf path of the three `params/env.yaml` files through `scripts/analysis/experiment_params.py` and finding sixty three differing of two thousand three hundred and twenty one. The three runs are `2026-09-21_06-45-40`, carrying one action scale of 0.4 at every joint and labelled A by the plan, `2026-09-22_07-18-05`, adding the per joint action scale of arm 4 and labelled B, and `2026-09-22_07-40-43`, adding the hip lateral sway reward of arm 5 on top and labelled C. `pose.py` establishes that the nominal pose, the height targets and the actuator ceilings are identical across the set, so the only configured differences are the two arms and two confounds, the push curriculum ceiling falling from 3.0 to 2.0 metres per second between A and B and the environment count from nine thousand to six thousand four hundred between B and C. The evaluation checkpoint was fixed by the modification times of the saved models against those of the dumps, placing A and B at `model_10000.pt` and C at `model_11000.pt`.

| File | What it holds | Cited by |
|---|---|---|
| `scripts/cfgdiff.py`, `scripts/pose.py` | The exhaustive leaf path comparison, and the proof that the pose, the height targets and the ceilings do not vary across the set | `../../plans/kscale_gait_shaping.md` section 14.1 |
| `scripts/raw_action.py` | The raw network output implied by the recovered target, measuring the action scale compensation at 2.479 against a scale ratio of 2.471 | section 14.2 |
| `scripts/stop_reversals.py` | The stop to stop reversal rate of the ankle roll, 2.031 per second in A against 0.002 in C | section 14.3 |
| `scripts/actuator_envelope.py` | Saturation against the velocity dependent envelope, which is the authoritative measure at a corner velocity of zero and which differs from the constant ceiling by up to twenty points | sections 14.3 and 14.9 |
| `scripts/stop_residency_saturation.py`, `scripts/joint_stop_residency.py` | Hard stop residency, soft limit excursion and constant ceiling saturation across all twelve joints, with stance and swing medians | sections 14.3 and 14.9 |
| `scripts/commanded_target.py` | The commanded target recovered by inverting the control law, and its excursion beyond the mechanical stop | sections 14.2 and 14.4 |
| `scripts/sole_flatness.py`, `scripts/cop_edge.py`, `scripts/sole_cant_source.py` | The sole tilt in stance, which lateral edge carries the contact, and the ankle roll angle that would flatten the sole, which sets arm 22's parameter | sections 14.3 and 14.12 |
| `scripts/toe_in.py`, `scripts/hip_yaw_role.py` | The signed toe-in pooled and in stance, and the regression establishing that the hip yaw sets it at a slope near unity | sections 14.9 and 17 |
| `scripts/footprint_separation.py` | The true sole polygon separation against the ankle frame separation the live term measures, a gap of 0.143 metres | section 14.8 |
| `scripts/sway_measured.py` | The torso lateral excursion at the stride frequency, 5.6 millimetres in C against the 30 the reward asks for | section 14.11 |
| `scripts/reward_budget.py` | The evaluation reward budget term by term, and the energy share of the negative budget at 1.04 per cent | section 14.6 |
| `scripts/limit_penalty_share.py`, `scripts/soft_limit_geometry.py` | Which joints pay `pen_joint_pos_limits`, reproduced exactly against the dumped term, and the 2.44 degrees of hip roll clearance that explains it | section 14.7 |
| `scripts/hip_pitch_stall.py` | That run B's hip pitch saturates on position error rather than velocity, and that its knee is straightened toward its stop | section 14.5 |
| `scripts/tensorboard_compare.py` | The training progression at a matched iteration and at a matched sample budget | section 14.10 |
| `scripts/common.py`, `scripts/frames.py` | The run register with the environment counts, the push ceilings and the DC motor envelope of `actuator_pd.py:295`, and the quaternion rotation helper | as above |
| `frames/zoom_A.png` through `frames/legs_C.png` | Two cropped and tiled strips per run from the play videos, at a 460 by 400 and a 260 by 280 window about the robot | section 14.13 |
| `scripts/validate_arm6.py` | The applied state of arms 7 and 6 asserted against both configuration files with `ast` and no simulator, the sagittal closure identity recomputed at the restored pose, and the factor of ten re-priced against all three dumped runs at 9.51, 29.63 and 4.22 per cent of the negative budget | sections 22.7, 21.8 and 21.20 |

The directory held twenty two programs when it was registered and holds twenty three since `validate_arm6.py` was added on 2026-09-23, that one program reading the configuration tree rather than a dump and therefore carrying no run attribution of its own. Nine of the original twenty two are copies of programs in the directory above, carried rather than imported because each is hard wired to its own run register through its `common.py` and because a shared import would make the earlier directory's output depend on a later edit. Two of the nine carry a one line change, `commanded_target.py` and `actuator_envelope.py` holding a per run ankle saturation effort that is 17 newton metres throughout this set.

### `kscale_flat-2026-09-24_to_09-25-energy-ankle-ablation/`

Attribution is by construction rather than by inference, which distinguishes this directory from the four above it. Every figure is computed from an evaluation dump at `kscale_flat_eval/<experiment>_<checkpoint>/data/42/dump.npy`, a path whose own directory name carries the experiment and the checkpoint that produced it, and every one of those replays was configured from the copy of its run's `params/` directory that `scripts/analysis/compare.py:288` places beside the checkpoint before launching. The actuator envelope each saturation figure is measured against is read from that same dumped configuration by `scripts/ablation.py:envelope`, so no figure here depends upon the state of the working tree at the time of the analysis, and no checkpoint had to be fixed by a modification time as the directory above required.

The three runs are `2026-09-24_05-18-33`, carrying the four energy weights at fifty times their pre arm 8 baseline and labelled A by the plan, `2026-09-25_08-00-34`, carrying them at one hundred times with a joint deviation penalty added to the ankle roll and the in air ankle roll reward quadrupled, labelled B, and `2026-09-25_11-29-39`, carrying fifty times with the same two ankle roll changes and a mechanical power term added, labelled C. `paramdiff.py` finds fifty nine differing leaf paths of two thousand five hundred and eighty three, of which fifty one are the constituent fields of two reward terms present in one run and absent from another, leaving eight substantive differences and one confound, the environment count falling from nine thousand to six thousand between B and C.

| File | What it holds | Cited by |
|---|---|---|
| `scripts/paramdiff.py`, `parameter_diff.txt` | The exhaustive leaf path comparison of the three dumped configurations, which establishes what in fact differs across the set and what does not | `../../plans/kscale_gait_shaping.md` section 16.1 |
| `scripts/lineage.py` | The reward weight lineage across every run since 2026-09-21, which places these three on the energy escalation ladder and identifies the abandoned single variable arm and the run in flight | sections 16.1 and 16.10 |
| `scripts/ablation.py`, `comparative_tables.txt` | The instrument and its output, being the per joint saturation against the velocity dependent envelope, the phase conditioned deviation, speed, torque and saturation of every joint, the liftoff and touchdown transients, the splay, the gait summary and the stop residency, at the best checkpoint of each run, at the matched checkpoint ten thousand, across run A's own checkpoints and at the last checkpoint of each | sections 16.2 through 16.7 |
| `scripts/report.py` | Renders those tables from `ablation_analysis.json`, needing no numpy and therefore runnable outside the GPU containers | as above |
| `scripts/deepdive.py`, `scripts/swingankle.py`, `mechanism_probes.txt` | The three mechanism probes, being the commanded target recovered by inverting the control law and split by gait phase, the phase relation of the two hip yaws against one another and against the base yaw rate, and the sign of the ankle roll torque against its deviation with the swing joint located against its mechanical stops | sections 16.4, 16.5 and 16.6 |
| `scripts/levers.py`, `levers.txt` | The three quantities the re-ranking turns upon, being the hip yaw to toe-in regression with the stance and the swing separated and the left leg mirrored on both axes, the commanded ankle target against the mechanical stop with the saturated samples declared a bound rather than a value, and the value and gradient of the three in air kernels at their own measured deviations | sections 16.5, 16.7 and 16.10 |
| `scripts/tb.py`, `training_tables.txt` | The training progression of the three at a matched iteration and at a matched sample budget, followed by the energy escalation ladder of six runs which carries the five hundred fold collapse | sections 16.8 and 16.9 |
| `scripts/frames.py`, `frames/A_t6.0_near.png` through `frames/C_t12.0.png` | Eight frames, the full view and a crop of the nearest machine at 6.0 s and 12.0 s of each play video, with the command at each instant printed by the producing program | section 16.11 |
| `scripts/validate_instrumentation.py` | Thirty nine assertions on the four dump channels added to `scripts/rsl_rl/play.py` on 2026-09-28 for section 20.2, being the sole tilt against a plate rotated by known angles under both robots' sole normals and at three base headings, the torque envelope against `DCMotor._clip_effort` reproduced verbatim from its source across a velocity sweep at three actuator groups and agreeing to machine precision, the scatter of the commanded target under an action set with the hip yaw pair removed, and a render of the dashboard's new panel with its traces and its sixty reference lines counted. It reads the tree and one dump, imports no part of isaaclab and needs no simulator | section 20.2, and the addition of 2026-09-28 to `/ws/context/gait_metrics.md` |

Two properties of this directory should be stated because they differ from its predecessors. It carries nine programs rather than twenty two, the five measurement programs sharing one implementation through `from ablation import` rather than each carrying its own copy of the envelope and the quaternion helpers, which is possible here and was not in the directory above because every run of this set carries the identical actuator block and no per run constant has to be hard wired. And `README.md` within it records a correction rather than only a description, the first pass having resolved every joint to one foot because the foot links of this robot carry no side marker, which inverted the stance and swing columns of the left leg and produced an apparent asymmetry that does not exist. The corrected mapping is `foot_index` at `scripts/ablation.py:44` and the three independent facts that establish it are given there.


### `kscale_flat-2026-09-28_to_09-29-clip-ablation/`

Attribution is by construction, as for the directory above it, and the figures differ from every directory before it in being READ rather than inferred. `scripts/rsl_rl/play.py` began on 2026-09-28 to record the commanded target, the raw action, the pre clip torque, the velocity dependent envelope and the sole tilt, so the commanded target that the four earlier directories recovered by inverting the control law on the unsaturated minority of samples is here a recorded channel. The runs are the five of section 18.1 of the plan, being `2026-09-24_05-18-33` through `2026-09-29_06-15-58` and labelled A to E, together with `2026-09-30_06-43-38` and `2026-09-30_07-05-14`, which carry the clip excess penalty at minus 2 and minus 500 and which differ from run E in three leaf paths of 2575.

| File | What it holds | Cited by |
|---|---|---|
| `scripts/lineage5.py`, `scripts/diff_inflight.py`, `parameter_diff.txt` | The reward weights and the action transform of the five runs side by side, and the exhaustive leaf comparison of the two in flight | `../../plans/kscale_gait_shaping.md` section 18.1 |
| `scripts/clipab.py`, `scripts/clipreport.py`, `comparative_tables.txt` | The exact saturation against the actuator's own envelope with no threshold, the recorded commanded target and raw action, the clip excess and the fraction the clip bound, the saturating deflection against the clip width, the sole tilt, and the hip yaw regression split by phase | sections 18.2 through 18.7 |
| `scripts/tb5.py`, `training_tables.txt` | The training progression of all seven runs at a matched iteration and a matched sample budget | sections 18.4 and 18.7 |
| `frames/A_t6.0_near.png` through `frames/E_t12.0.png` | Sixteen frames at two matched instants from four of the five runs | section 18.10 |
| `scripts/headinggate.py`, `heading_gate_analysis.json` | The toe-in by phase with the swing to stance ratio, the decomposition of `feet_yaw_alignment` into its common and differential modes with the airborne gate open and closed, the share of the term the gate removes, and every contact interval traced from its touchdown sample to the last before release, which is what settles whether a stance heading is inherited from swing or generated in contact | section 18.7, and arm 15 at section 21.15 names it as the instrument its precondition requires |
| `scripts/excess_kernel.py` | The clip excess of run E under both kernels and per joint, recovering the action term's per environment offset by the same restriction `clipab.py` uses, which confirms section 18.4's absolute figure at 7.3391 rad and attributes 94.3 per cent of the quadratic excess and 79.1 of the absolute one to the hip roll pair | section 21.13 |
| `scripts/validate_arm14.py` | The 54 assertion pre launch check of arm 14, the hip yaw fixed in the URDF. It reads the arm's own proposed dictionaries out of section 21.14 and the live ones out of `kscale_identified_cfg.py` and `kscale_base_env_cfg.py`, so the plan and the check cannot drift, and it settles the arm's central structural claim from the SD_BRS1 dump, that robot already declaring both its hip yaws fixed under the same `merge_fixed_joints=False` and reporting ten joints against thirteen bodies | section 21.14 |
| `scripts/fixedjoint_probe.py` | The probe that established it, reading `joint_names` and the width of `robot_mass` from the SD_BRS1 evaluation dump | section 21.14 |

Its own register records a correction that a later reader would otherwise reproduce. The unclipped target is the raw action times the scale plus the action term's per environment offset, and the dump carries only environment zero's row of that offset while `randomize_joint_default_pos` moves it by up to 0.05 rad per environment. Recovering the offset from a median over all samples is correct on an unclipped run and wrong on run E, where the hip roll clip binds on 91.8 per cent of samples, and the standing repair restricts the median to the samples the clip did not bind, which recovers every run to within 0.04809 rad of the dumped row.

### `sf_tron_1a_flat-2026-10-09_06-04-18/`

Registered 2026-10-19. The run is `sf_tron_1a_flat/2026-10-09_06-04-18`, which despite its experiment name is the rough terrain task `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0`, its dumped `params/env.yaml` naming `SFIdentifiedBlindRoughEnvUrdfCfg` on its first line. Attribution is certain for both sub-directories, since every file was produced from that run's own directory by the commands recorded beside it. The source files were identified by md5 prefix on 2026-10-19, the event file `events.out.tfevents.1791526003.auk6000.175479.0` as `956bb3aa5ce7`, `videos/train/rl-video-step-0.mp4` as `e62eee6776fc`, `videos/train/rl-video-step-430000.mp4` as `c8c7099aa869` and `params/env.yaml` as `702431c96743`. All of it is cited by `../../plans/tron1_locomotion_fine_tuning.md`.

| Path | What it holds | Cited by |
|---|---|---|
| `frames/sheet_step_<N>.jpg` | One sheet per training video, forty four in all for N from 0 to 430000 in steps of 10000, eight frames each sampled evenly over the forty seconds, cropped to the tracked robot and labelled with the step, the iteration N/25 and the time | section 3.4 |
| `frames/dense_step_<N>.jpg` | Sixteen frame sheets for the ten videos from 250000 to 340000, the window in which the user reported the transition | section 3.4 |
| `frames/overview.jpg` | One representative frame per video, forty four tiles | section 3.4 |
| `tensorboard/scalars_every50.csv.gz` | Every scalar tag of the event file at iterations divisible by fifty, raw values | sections 3.5 and 7 |
| `tensorboard/plot_*.svg` | Six hand written plots of episode length, implied base height error, undesired contacts, tracking, the foot contact rewards and the noise standard deviation, each marked at iteration 12400 | section 3.5 |
| `tensorboard/extract.py` | The extraction, its tables and its fits, which reads only the event file and runs in about twenty seconds with tensorboard and scipy on the path | sections 3.5 and 7 |
| `arm0/contact_probe_model_6000.txt` | The sixty samples printed by the arm 0 contact probe of `scripts/rsl_rl/play.py` on 2026-10-19, the share of 32 environments of the rough play task whose contact force exceeded 10 N per body, replaying `model_6000.pt` for 3000 control steps, with a header recording the replay's configuration. Attribution is certain, the checkpoint path being printed in the replay's own log, from which the user transcribed the samples into the session | section 6.2.1 |

The sheets were built with the `imageio_ffmpeg` static binary and labelled with OpenCV, that binary carrying no drawtext filter. The camera follows environment zero only and does not turn with it, so only side views of that one robot can be read for support, a limitation section 3.4 of the plan states.

### `tron1_sf-model-analysis-2026-10-19/`

Registered 2026-10-19. The outputs of `scripts/analysis/biped_physical_analysis.py` and `scripts/analysis/biped_stance_analysis.py` under `scripts/analysis/biped_configs/tron1_sf.json`, run at repository commit `23513d2` with the working tree of 2026-10-19. They are attributed to the model documents rather than to a training run, `environments/environments/assets/mjcf/solefoot/tron1/base_robot.xml`, `.../urdf/solefoot/tron1/base_robot.urdf` and `.../urdf/solefoot/tron1/base_robot_tron1.urdf`, and are reproduced exactly by rerunning the two scripts, which are deterministic, the drop test seeding its generator at zero.

| File | What it holds | Cited by |
|---|---|---|
| `physical_analysis.txt` | Mass inventory, nominal pose geometry, joint limit headroom, kneeling geometry, swing inertia, symmetry and collision audit of the three documents, with their comparison table | sections 3.1 and 3.2 of `../../plans/tron1_locomotion_fine_tuning.md` |
| `stance_analysis_mjcf.txt` | Static stance torques, sag, stance and swing inertia, proposed gains, zero action drop tests on a gantry and free, and the bend sweep, for the generated MJCF | sections 3.2 and 6 |
| `stance_analysis_tron1_urdf.txt` | The same for the original LimX TRON1 URDF | section 3.2 |
| `stance_analysis_mjcf_arm7_nominal.txt` | The same for the generated MJCF under `scripts/analysis/biped_configs/tron1_sf_arm7_nominal.json`, the knee straightened by 0.2 rad, which sets the spawn height and the height target of arm 7 | section 6, arm 7 |

### `render-refresh-diagnostic-2026-10-19/`

Registered 2026-10-19. Renders of a three environment scene built by a diagnostic script run in `isaac-lab-base-shreyas-gpu1` against the working tree of 2026-10-19, not by a training run. Design A, shank scale 0.85 and so a 0.18 m shank, was converted and spawned in environments 0 and 1, design B, shank scale 1.15 and so 0.24 m, in environment 2, and B was then written into environment 0 by `apply_link_length_params`. Attribution is by construction, each image being written by the script that built the scene, with gravity off and every joint at zero.

| File | What it holds | Cited by |
|---|---|---|
| `quadruped_stale_render.png` | Environment 0 before the write, after the write, and environment 2, the written shanks drawn at A's length with their feet at B's | section 9 of `../co_optimisation_audit.md`, row dated 2026-10-19 |
| `quadruped_refreshed_render.png` | Environment 0 after the write, after `refresh_rendered_geometry`, and environment 2, the refreshed shanks reaching their feet as B's do | the same row |

### `tron1_quadruped-mjcf-mirror-audit-2026-10-20/`

Registered 2026-10-20. Not the product of a run. `mjcf_mirror_audit.py` composes every body quaternion of a MuJoCo document to the root, expresses each joint axis in the base frame and compares the mirror of each joint's axis with its partner's, and `mjcf_mirror_audit.txt` is its output against `environments/environments/assets/mjcf/solefoot/tron1/base_robot.xml`, md5 `808a09a927219e8d707cc27a4a83770b`, and `environments/environments/assets/mjcf/quadruped/quadruped.xml`, md5 `81a3b2a88f3a9d222952c3f2b0ec04d7`, both as they stood in the working tree of 2026-10-20. Attribution is by regeneration, the script reading only those two files with numpy.

| File | What it holds | Cited by |
|---|---|---|
| `mjcf_mirror_audit.py` | The audit, run with `python3 -I mjcf_mirror_audit.py` from this directory | sections 4.1 and 4.2 of `../../plans/symmetry_tron1_quadruped.md` |
| `mjcf_mirror_audit.txt` | Body counts, the worst body position mirror error, and per joint the world axis, the derived sign and the consistency of the limits | the same sections, `../tron1.md` addition of 2026-10-20, section 11 of `../quadruped.md` |

### `quadruped_symmetry-2026-10-20_06-54-05-sandbox-smoke/` and `sf_tron_1a_symmetry-2026-10-20_06-56-00-sandbox-smoke/`

Registered 2026-10-20. Each holds one file, `smoke_probe.txt`, the lines a sandbox run printed through a probe added to `compile_maps` of the proposed `common.py`, its iteration log, the `symmetry_cfg` of its dumped `params/agent.yaml`, and the diff of its dumped `params/env.yaml` against the baseline then in flight, `quadruped_flat/2026-10-20_05-09-02` and `sf_tron_1a_flat/2026-10-20_05-24-20` respectively. The runs trained three iterations on 64 environments of `Isaac-Quadruped-Blind-Rough-Symmetry-v0` and `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-Symmetry-v0` in a sandbox copy of the repository at `/tmp/symsandbox_20261020` inside `isaac-lab-base-shreyas-gpu1`, so they are absent from the shared log tree by design. Attribution is by extraction from each run's own log and directory on the day it ran. Both task identifiers belonged to the plan's first pass and were withdrawn by its revision of the same day, which never registered them, and the runtime orders these files record were reproduced on the registered tasks in `symmetry_revised-2026-10-20-sandbox-smoke/`.

| File | What it holds | Cited by |
|---|---|---|
| `smoke_probe.txt` | Runtime joint and body names and permutations, signs, shape counts, limits, group layouts, ray pattern, contact sensor order, symmetry loss per iteration, dumped symmetry configuration, environment configuration diff | section 4.4 of `../../plans/symmetry_tron1_quadruped.md` as revised on 2026-10-20, the 2026-10-20 additions of `../tron1.md` and `../quadruped.md` |

### `sf_quadruped_brs_kscale-obs-term-census-2026-10-20/`

Registered 2026-10-20. Not the product of a run. `dump_obs_terms.py` launches Isaac Sim headless, loads the environment configuration of every registered training task whose identifier begins with one of the four family prefixes, play tasks excluded, and records for each its runner class, its environment class, its spawned asset, and per observation group the history, the flattening and every term with its function and parameters, into `obs_terms.json`. It was run inside `isaac-lab-base-shreyas-gpu1` against the working tree of 2026-10-20, md5 of the output `1b8bab39e02f339185cb9537c0db1327`. Attribution is by regeneration.

| File | What it holds | Cited by |
|---|---|---|
| `dump_obs_terms.py` | The census script | section 5.1 of `../../plans/symmetry_tron1_quadruped.md` |
| `obs_terms.json` | The census | sections 3.3 and 5.1 of the same plan |

### `symmetry_revised-2026-10-20-sandbox-smoke/`

Registered 2026-10-20. The runs of the revised symmetry plan in a fresh sandbox copy of the repository at `/tmp/symsandbox_v2_20261020` inside `isaac-lab-base-shreyas-gpu1`, each three iterations on 64 environments, so they are absent from the shared log tree by design. `smoke.sh` launched the first twelve, `smoke2.sh` the two co-optimisation reruns after the `train.py` copy, `smoke3.sh` the two runs of the unmodified live tree from a directory holding no `environments` package, and `smoke4.sh` the two replays, the last as edited in place for its rerun after `scripts/analysis` was copied into the sandbox. `smoke.sh` also holds a first live tree run whose output was discarded, the sandbox directory having shadowed the installed package so that no task was registered. `add_probe.py` inserted the probe whose lines the excerpts carry. Each `<name>.txt` is the extraction of one log, the configuration lines, the group widths, the probe and symmetry lines, the iteration and loss lines, and the closing frames of any traceback, and `smoke_status.txt` gives each run's exit, traceback count and logged iterations. Attribution is by extraction from each run's own log and directory on the day it ran.

| File | What it holds | Cited by |
|---|---|---|
| `sf_ppo.txt`, `quad_ppo.txt`, `sf_him.txt`, `quad_him.txt`, `sf_copt2.txt`, `quad_copt2.txt` | The TRON1 and quadruped plain, HIM and co-optimisation runs that passed | section 11 of `../../plans/symmetry_tron1_quadruped.md`, the 2026-10-20 additions of `../tron1.md` and `../quadruped.md` |
| `sf_copt.txt`, `quad_copt.txt` | The co-optimisation runs that failed in the dump of `copt_agent.yaml` before the copy | section 11 of the plan |
| `brs_ppo.txt`, `brs_him.txt`, `live_brs_him2.txt` | The SD_BRS1 runs, and the live tree's HIM failure | section 11 of the plan, the 2026-10-20 addition of `../BRS.md` |
| `ks_ppo.txt`, `ks_yaw.txt`, `ks_him.txt` | The KScale runs | section 11 of the plan, the 2026-10-20 addition of `../KScale.md` |
| `go1_ppo.txt`, `live_go1.txt` | The Go1 asset construction failure in the sandbox and the live tree | sections 4.3 and 11 of the plan, the 2026-10-20 addition of `../quadruped.md` |
| `play_sf2.txt`, `play_quad2.txt` | The replays of the two baselines' checkpoints under the symmetry configuration | sections 10 and 11 of the plan |
| `dumped_symmetry_cfg.txt`, `smoke_status.txt`, the four scripts, `add_probe.py` | The dumped configurations, the status table and the provenance of every run | section 11 of the plan |

### `symmetry_applied-2026-10-20-live-smoke/`

Registered 2026-10-20. The smoke test of section 9.3 of `../../plans/symmetry_tron1_quadruped.md`, run by `apply_smoke.sh` on the live tree once the plan was applied, from `/tmp/symmetry_smoke` of `isaac-lab-base-shreyas-gpu1`, together with two reruns of its plain tasks, whose directories the listing also holds. `status.txt` is the script's own table, whose empty symmetry columns for the two plain tasks reflect standard output lost at the simulator's exit rather than a failure, `event_losses.txt` is the `Loss/symmetry` scalar of every run read from its event file, and `checkpoints_and_functions.txt` lists every run's checkpoints and the augmentation function its dumped configuration names. Attribution is by extraction from each run's own directory the same day.

| File | What it holds | Cited by |
|---|---|---|
| `event_losses.txt`, `checkpoints_and_functions.txt`, `status.txt`, `apply_smoke.sh` | The smoke test after application | section 13 of `../../plans/symmetry_tron1_quadruped.md`, the 2026-10-20 additions of `../tron1.md` and `../quadruped.md` recording the application |

## Removals

None to date. Record every deletion here with its reason, so that a reader who recalls a figure and cannot find it learns why rather than assuming an oversight.

Seventy two whole video frames and six tiled sheets built from them were deleted from `kscale_flat-2026-09-21_to_09-22-scale-sway-ablation/frames/` on 2026-09-23, the whole frames named `A_01.png` through `C_24.png` and the sheets `sheet_A.png` through `sheet_C.png` and `stride_A.png` through `stride_C.png`. The reason is legibility rather than redundancy. The tracking camera places the robot at roughly seventy pixels of height in a 1280 by 720 frame and the tiling preserved that scale rather than repairing it, so no gait property was visible in any of them, and section 14.13 of the plan records the same judgement. The six cropped strips that replace them are retained and cited. No document cites any deleted file.


