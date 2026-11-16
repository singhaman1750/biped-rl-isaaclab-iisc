# Artefacts of the energy budget and ankle roll penalty ablation, 2026-09-24 to 2026-09-25

This directory holds the evidence that section 16 of [../../../plans/kscale_gait_shaping.md](../../../plans/kscale_gait_shaping.md) reasons about, being the three training runs `kscale_flat/2026-09-24_05-18-33`, `2026-09-25_08-00-34` and `2026-09-25_11-29-39`, which that section calls A, B and C, together with the twenty one checkpoint replays of them that `scripts/analysis/compare.py` staged under `IsaacLab/logs/rsl_rl/kscale_flat_eval/`.

Attribution is established by construction rather than by inference. Every figure below is computed from the evaluation dump at `kscale_flat_eval/<experiment>_<checkpoint>/data/42/dump.npy`, whose directory name carries the experiment and the checkpoint it was produced from, and every replay was configured from the copy of that run's own `params/` directory which `compare.py:288` places beside the checkpoint before launching. The actuator envelope each saturation figure is measured against is read from that same dumped configuration by `scripts/ablation.py:envelope`, so no figure here depends on the state of the working tree at the time of the analysis.

## The files

| File | What it is | Produced by |
|---|---|---|
| `scripts/ablation.py` | The instrument. Computes the per joint saturation against the velocity dependent envelope, the phase conditioned behaviour of every joint, the liftoff and touchdown transients, the splay, the gait summary and the reward budget, and writes `ablation_analysis.json` beside the staged jobs. | Written 2026-09-26 for this analysis |
| `scripts/report.py` | Renders the tables of `comparative_tables.txt` from that record. Runs outside the GPU containers, needing no numpy. | Written 2026-09-26 |
| `scripts/deepdive.py` | Inverts the control law to recover the commanded joint target, tests whether the two hip yaws move in phase, and reads the sign of the ankle roll torque against its deviation. | Written 2026-09-26 |
| `scripts/swingankle.py` | Locates the swing ankle against its mechanical stops, splits the hip yaw command by gait phase, and tabulates the reward budget. | Written 2026-09-26 |
| `scripts/tb.py` | The training record of the three runs at a matched iteration and at a matched sample budget. | Written 2026-09-26 |
| `scripts/frames.py` | Extracts the frames below at a matched instant, with the command at that instant printed beside them. | Written 2026-09-26 |
| `comparative_tables.txt` | Every table section 16 of the plan quotes, at the best checkpoint of each run, at the matched checkpoint 10000, across run A's own checkpoints and at the last checkpoint of each. | `scripts/report.py` |
| `training_tables.txt` | The training progression of the three runs and the energy escalation ladder of six. | `scripts/tb.py` |
| `frames/` | Eight frames, being the full view and a crop of the nearest machine at 6.0 s and 12.0 s of each play video. | `scripts/frames.py` |

## The frames

The three replays share a seed and a command series, every one being configured from its own run's dump and every one meeting the identical command distribution, so a frame taken at one instant is directly comparable across the three. At 6.0 s the command is a forward velocity of 0.031 m/s against a yaw rate of minus 0.300 rad/s and at 12.0 s a forward velocity of minus 0.291 m/s against a yaw rate of plus 0.300, both being slow intervals in which the splay is largest.

`A_t6.0_near.png`, `B_t6.0_near.png` and `C_t6.0_near.png` are the crops. They corroborate the measurements and add nothing to them, all three showing the same crouched posture with the two feet converged upon the midline and overlapping in projection, C's fold being visibly the deepest, which the mean knee angle of 1.088 rad against A's 0.697 states more precisely than a frame can. No figure in the plan is read off a frame.

## A correction retained rather than removed, 2026-09-26

The first pass of `scripts/ablation.py` resolved every joint to the same foot. The foot links of this robot are named `foot_6061_2` and `foot_6061` and carry no side marker, so the `startswith` test the first draft used returned false for both sides of the left leg and resolved it to foot index one. Every phase conditioned figure of that pass was therefore taken against the opposite leg's contact state, which inverted the stance and swing columns of the left leg and produced an apparent left to right asymmetry that does not exist. The repair is `foot_index` at `scripts/ablation.py:44`, which maps the left to index zero, and three independent facts establish the mapping, the articulation body order placing every left link before its right counterpart, the lateral separation of foot zero minus foot one standing at plus 0.1310 m in the base frame, and the right hip yaw correlating with foot one's toe-in at 0.95 against 0.18 for foot zero. The error is recorded because the symmetric figures of the corrected pass are the ones the plan quotes, and a reader who recomputes them with a name based side test will reproduce the fault rather than the finding.
