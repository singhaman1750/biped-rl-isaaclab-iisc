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


## Removals

None to date. Record every deletion here with its reason, so that a reader who recalls a figure and cannot find it learns why rather than assuming an oversight.
