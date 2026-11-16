# kscale_flat-2026-09-21_to_09-22-scale-sway-ablation

The analysis programs and the frames that produced every figure quoted in sections 14, 15, 16 and 17 of [../../plans/kscale_gait_shaping.md](../../plans/kscale_gait_shaping.md), written 2026-09-23 against the three runs that executed arm 3 and arm 4 of that document's ablation.

## Provenance

The three runs are `kscale_flat/2026-09-21_06-45-40`, which the plan labels A and which carries one action scale of 0.4 at every joint, `2026-09-22_07-18-05`, labelled B, which adds the per joint action scale of arm 3, and `2026-09-22_07-40-43`, labelled C, which adds the hip lateral sway reward of arm 4 on top of arm 3. The attribution was established by reading `params/env.yaml` of each through `scripts/analysis/experiment_params.py` and comparing every leaf path, which `cfgdiff.py` performs and which finds sixty three differing paths of two thousand three hundred and twenty one. The nominal pose is identical in all three, `default_joint_pos` in each dump carrying the same twelve values, so no pose confound exists across this set.

Two quantities move with the arms without belonging to them and every figure in this directory must be read against them. The push curriculum ceiling fell from 3.0 metres per second in A to 2.0 in B and C, and the parallel environment count fell from nine thousand in A and B to six thousand four hundred in C. The evaluation checkpoint was established from the modification times of the saved models against the modification times of the dumps, placing A and B at `model_10000.pt` and C at `model_11000.pt`, so C was evaluated from 78.2 per cent of the experience behind the other two.

## The programs

The scripts are retained rather than the intermediate arrays, the arrays being recoverable from the log tree while it survives and the scripts being the statement of how each figure was computed. Every one imports `common.py`, which names the three runs, their environment counts, their push ceilings and the actuator envelope. Nine are copies of programs in [../kscale_flat-2026-09-18_to_09-21-foot-joint-ablation/](../kscale_flat-2026-09-18_to_09-21-foot-joint-ablation/), carried rather than imported because each is hard wired to its own run register and because a shared import would make the earlier directory's output depend on a later edit. Two of the nine carry a one line change, `commanded_target.py` and `actuator_envelope.py` holding a per run ankle saturation effort that is 17 newton metres throughout this set.

| Program | What it establishes | New or carried |
|---|---|---|
| `common.py` | The run register, the environment counts, the push ceilings and the DCMotor envelope | New |
| `cfgdiff.py` | The exhaustive leaf path comparison, sixty three differing paths of 2321 | New |
| `pose.py` | That the nominal pose, the height targets and the actuator ceilings are identical across the three runs | New |
| `raw_action.py` | The raw network output implied by the recovered target, which measures the action scale compensation at 2.479 against a scale ratio of 2.471 | New |
| `stop_residency_saturation.py` | Hard stop residency, soft limit excursion and torque saturation against the constant ceiling, all twelve joints | New |
| `actuator_envelope.py` | Saturation against the velocity dependent envelope, which is the authoritative measure at a corner velocity of zero | Carried, one line |
| `commanded_target.py` | The commanded target recovered by inverting the control law, and its excursion beyond the mechanical stop | Carried, one line |
| `stop_reversals.py` | The stop to stop reversal rate, 2.031 per second in A against 0.002 in C | Carried |
| `joint_stop_residency.py` | Joint position against the hard and soft stops with stance and swing medians | Carried |
| `sole_flatness.py` | The sole normal in the base yaw frame, giving the lateral and fore and aft tilt in stance | Carried |
| `cop_edge.py` | Which lateral sole edge carries the contact, by transforming both corners rather than by reasoning from the normal | Carried |
| `sole_cant_source.py` | The sole cant regressed on the ankle roll, the hip roll and the torso roll, and the ankle roll angle that would flatten the sole | New |
| `toe_in.py` | The foot yaw relative to the base heading, pooled and in stance | Carried |
| `hip_yaw_role.py` | That the hip yaw sets the toe-in at a slope near unity, and what would remain at zero hip yaw | New |
| `footprint_separation.py` | The true sole polygon separation against the ankle frame separation the live term measures | New |
| `sway_measured.py` | The torso lateral excursion at the stride frequency, 5.6 millimetres in C against the 30 the reward asks for | Carried |
| `reward_budget.py` | The evaluation reward budget term by term, and the energy share of the negative budget | New |
| `limit_penalty_share.py` | Which joints pay `pen_joint_pos_limits`, reproduced exactly against the dumped term | New |
| `soft_limit_geometry.py` | Where each soft limit falls relative to the nominal, which finds the hip roll at 2.44 degrees of clearance | New |
| `hip_pitch_stall.py` | That run B's hip pitch saturates on position error and not on velocity, and that its knee is straightened | New |
| `tensorboard_compare.py` | The training progression at a matched iteration and at a matched sample budget | New |
| `frames.py` | The quaternion rotation helper and the stance axis check every frame aware program imports | Carried |
| `validate_arm6.py` | The applied state of arms 5 and 6 asserted against both configuration files with `ast` and no simulator, the sagittal closure identity recomputed at the restored pose, and the factor of ten re-priced against all three dumped runs | New, 2026-09-23 |

## The frames

Six files, two per run, are retained at `frames/`. They were extracted from `videos/play/42/rl-video-step-0.mp4` of each run with the `ffmpeg` binary carried inside `imageio_ffmpeg`, the workspace holding no other and no matplotlib.

| File | What it is |
|---|---|
| `zoom_A.png`, `zoom_B.png`, `zoom_C.png` | Six frames at eight per second from twelve seconds in, cropped to a 460 by 400 window about the robot, upscaled twofold and tiled three by two |
| `legs_A.png`, `legs_B.png`, `legs_C.png` | Four frames at the same rate and offset, cropped to a 260 by 280 window about the lower body, upscaled threefold and tiled four by one |

## Deletions

Seventy two whole frames, twenty four per run at two per second, were extracted, inspected and deleted on 2026-09-23. The tracking camera places the robot at roughly seventy pixels of height in a 1280 by 720 frame, at which scale no gait property is legible, and section 14.13 of the plan records the same judgement. Six tiled sheets built from those frames, `sheet_A` through `sheet_C` and `stride_A` through `stride_C`, were deleted on the same date and for the same reason, the tiling having preserved the scale rather than repairing it. No document cites any of the deleted files.
