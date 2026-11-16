# kscale_flat-2026-09-18_to_09-21-foot-joint-ablation

The analysis programs that produced every figure quoted in sections 10 through 13 of [../../plans/kscale_gait_shaping.md](../../plans/kscale_gait_shaping.md), written 2026-09-22 against the three surviving runs of the foot torque ablation.

## Provenance

The three runs are `kscale_flat/2026-09-18_09-37-29` at an ankle effort ceiling of 11.9 newton metres, `2026-09-21_06-45-40` at 17, and `2026-09-21_08-40-59` at 34. The attribution was established by reading `params/env.yaml` of each through the repository's own `ParamsLoader` at `scripts/analysis/experiment_params.py:24` rather than by the directory name, which matters because two directories the investigation was originally pointed at, `2026-09-18_09-24-17` and `2026-09-21_08-37-13`, carry a parameter dump and an initial checkpoint but no training record, no evaluation dump and no play video, and are therefore launches that did not survive their first minute. `run_inventory.py` performs that check and prints the ceiling of every candidate directory, and it is the program to run first when the set of runs is in doubt.

## The programs

The scripts are retained rather than the intermediate arrays, the arrays being recoverable from the log tree while it survives and the scripts being the statement of how each figure was computed. Every one imports `common.py`, which names the three runs and implements the DC motor torque speed envelope of `actuator_pd.py:295`, and several import `frames.py` for the quaternion rotation. They are run from this directory with no arguments.

| Program | What it establishes |
|---|---|
| `run_inventory.py` | The ankle ceiling, saturation effort, gains, population and action scale of every candidate run directory |
| `config_diff.py` | The exhaustive leaf path comparison of the dumped configurations, which finds six differing paths of 2263 |
| `actuator_envelope.py` | Per joint saturation measured against the velocity dependent envelope and against the constant ceiling, split by stance and swing |
| `joint_stop_residency.py` | Joint position against the hard and soft stops, with stance and swing medians |
| `commanded_target.py` | The commanded target reconstructed by inverting the control law over unsaturated samples, and its excursion beyond the mechanical stop |
| `sole_flatness.py` | The sole normal in the base yaw frame, giving the lateral and fore and aft tilt of the foot in stance |
| `ankle_roll_travel.py` | The ankle roll angle that would lay the sole flat, against the travel the joint has |
| `stop_reversals.py` | Stop to stop reversals per second of the ankle roll, and the traverse speed |
| `sagittal_closure.py` | The closure identity and the ankle pitch the crouch demands in swing |
| `toe_in.py` | The signed toe-in by the foot link axis method, with the mirrored left frame resolved |
| `gait_metrics.py` | Tracking error, cost of transport, power, jerk, cadence, double support, height, knee and separation |
| `inair_kernel.py` | The value and the gradient of the three in air deviation terms against the velocity tracking reward |
| `dict_scale_semantics.py` | The four behaviours of a dictionary valued action scale, that a complete dictionary resolves, that an INCOMPLETE one silently leaves the omitted joints at 1.0, that overlapping patterns raise, and that a pattern matching nothing raises |
| `validate_scale.py` | The scale dictionary parsed back out of `kscale_base_env_cfg.py` with `ast` and resolved against the twelve joint names the dump records, checking coverage and the spread of the action at saturation |
| `tree_vs_run.py` | The working tree's sixty actuator fields compared against those the 17 N.m run dumped, which establishes that a launch from the tree is a one variable departure from it |
| `sway_geometry.py` | The lateral travel quasi static single support demands, being the half stance, the inboard limit of the support polygon and the ankle roll moment that survives a full centre of pressure shift |
| `sway_measured.py` | The lateral torso excursion the three policies actually produce, from the base frame lateral velocity at the stride frequency with the commanded component removed |
| `cop_edge.py` | Which lateral sole edge carries the contact in stance, by transforming both sole corners into the world and comparing their heights, which needs no sign convention |
| `arm_parameterisation.py` | The four parameter tables of section 15 that are arithmetic on the configuration rather than measurements of a run, being the per joint action scale of section 21.4, the lateral sway kernel of section 21.5 at both widths, the commanded target clip of section 21.10 against the soft limits it must not silence, and the corner velocity of section 21.21. It reads no dump and needs no simulator |
| `nominal_pose.py` | The pose parameters of arm 7, derived by forward kinematics against `kscale.urdf` rather than quoted. It gives the standing height of the present and proposed postures with the roll nominals on and off, prices the settling margin against the reset distribution, reports the travel each nominal leaves and the stance load the fold moves, and re-derives the action scale table at both poses. It reads no dump and needs no simulator |
| `validate_sway.py` | The two gates the RewardManager applies to `rew_hip_lateral_sway`, being the signature match at `manager_base.py:361` against the configured params and the body resolution at `:401`, checked statically against the body list the dump records |

## Validation

`gait_metrics.py` and `toe_in.py` were checked against the figures section 3 of the plan records for the run there called C, which is `2026-09-18_09-37-29`, the same run this ablation carries as its 11.9 newton metre arm. The planar velocity error reproduces at 0.0463, the absolute cost of transport at 1.026, the absolute joint power at 89.52 watts, the joint jerk at 4563, the double support fraction at 9.28 per cent, the base height at 0.6403 metres, the mean knee angle at 1.2015 radians, the lateral foot separation at 0.1460 metres, the mean toe-in at 4.59 degrees and its ninety fifth percentile at 21.39. Eleven quantities reproducing to every digit the earlier document quotes establishes that the two pipelines agree, and any figure this directory produces for the other two runs may be read against that earlier table directly.

Two conventions are recorded because they are not inferable from the dump. Foot index 0 is `foot_6061_2` and is the LEFT foot, established in `frames.py` by correlating each foot's frame height against each knee angle, which returns plus 0.576 for the left pairing against minus 0.304 for the crossed one. And the left foot link frame is yawed by 180 degrees against the right, as `kscale.urdf:1079` declares with its `rpy` of minus pi, so `toe_in.py` resolves the forward axis per foot by choosing the sign that minimises the mean absolute heading residual rather than assuming a common convention.

## The arm 4 implementation, 2026-09-22

The last three programs were written when the per joint action scale of arm 4 was set, and they differ in kind from the rest, being checks on a configuration rather than measurements of a run. `dict_scale_semantics.py` loads `isaaclab/utils/string.py` directly by file path, bypassing the package `__init__` which imports `toml`, so it runs outside the simulation container. `validate_scale.py` reads the literal out of the configuration file rather than taking a transcription, so it cannot drift from what the tree actually holds, and it must be re-run after any edit to that dictionary. `tree_vs_run.py` is the section 2.3 check generalised, and it should be run before every launch, since the whole value of a one variable ablation rests upon the other variables having stayed still.

## The arm 5 correction, 2026-09-22

`cop_edge.py` exists because the first reading of the stance foot orientation was wrong in its prose. Section 11.1 of the plan said the machine stands on the OUTER edges of its feet, inferred from the sign of the sole normal's lateral component, and it stands on the INNER edges. The tilt measurement was never in doubt, only the mapping from a normal direction to an edge, and the error is easy to make in either direction. `cop_edge.py` avoids it by transforming the two lateral sole corners into the world and comparing their heights, which is a question with no convention attached, and it carries a synthetic self test of a plate rolled by a known angle so the reader may check the machinery before trusting the result. Any future claim about which part of a sole is loaded should be made with the corner method and not from a normal.

## The section 15 parameterisation, 2026-09-23

`arm_parameterisation.py` was added when section 15 of the plan converted every arm of the ablation into a sub-section carrying its code, its design decisions and its parameters. Four of those parameter tables are arithmetic on declared constants rather than measurements of a run, so they are derived by that program from the constants alone and each constant in it carries its source line. The program exists so that a reader may regenerate the tables against a tree that has moved rather than trust numbers transcribed into prose, which is the failure this directory has already recorded once at the arm 5 correction above.

One correction was made in the same pass. `validate_scale.py` closed with the hard coded remark that the action at saturation spanned a ratio of 3.54 to one before the per joint scale, and 3.54 is the spread at the 11.9 newton metre ankle of the plan's section 4.6 rather than at the 17 newton metre ankle the arm departs from, where it is 2.50. The line now derives both and prints both, and the plan carries the correction at its section 21.4. The lesson is the general one, that a figure derived at one configuration and quoted at another is wrong even when the arithmetic that produced it was right.

## The arm 7 implementation, 2026-09-23

`nominal_pose.py` was added when arm 7 was applied to the working tree, and it found two errors in the sub-section that specified the arm, which is the second time in two days that deriving a figure rather than quoting it has repaid the cost of writing the program.

The standing height of the published K-Scale posture is 0.73492 metres and not the 0.73540 that both the plan and the addition of 2026-09-16 to [../../context/KScale.md](../../context/KScale.md) carry. That figure is correct for the pose it was taken at, which had both roll nominals at zero, and the pose supplied on 2026-09-17 gave the hip roll and the ankle roll the mirrored values that arm 7 deliberately retains. Those tilt each leg out of the sagittal plane and cost 0.48 millimetres of height. The lesson is the same one the section 15 parameterisation records, that a figure derived at one configuration and quoted at another is wrong even when the arithmetic that produced it was right, and it is now recorded twice because the first recording did not prevent the second occurrence.

And the arm's own table inverted the quantity in its last row, naming the ankle pitch nominal as moving from 17.7 degrees of its lower stop to 30.0 when those are the nominal angles themselves and the travel remaining falls from 32.3 degrees to 20.0. A reader would have concluded that the arm gives that joint more room toward the stop the crouch drives it into, where it gives it less. What the arm buys at that joint is authority rather than headroom, the action being a delta from the default, and section F of the program computes it.

The program also prices the settling margin, which the user reduced to zero in the same pass and then set to 0.01 metres once this measurement was in hand. What it carries forward is that the reset distribution reaches very much further than the margin ever did at any value, 6.2 per cent of resets drawing a pose taller than the spawn at the 0.02 metre margin, 20.3 per cent at the 0.01 applied and 37.0 per cent at zero, the worst penetration in the three cases being 11.4, 21.4 and 31.4 millimetres. The choice is therefore one of how the reset fails and not whether it does, and full clearance against that distribution would require 0.032 metres.

## Deletions

None to date.
