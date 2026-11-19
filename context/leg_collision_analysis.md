# Leg Collision Analysis of the Collapse of copt_moral/2026-11-18_05-48-29

This document establishes which collision pair drives the rise of the `leg_contact` termination in the TRON1 co-optimisation run `copt_moral/2026-11-18_05-48-29`, and judges the proposal to remove the shank from the termination. It was written on 2026-11-18 from the TensorBoard scalars of the run, the 100 designs of `designs/design_parameters_1199.csv`, a static MuJoCo scan of their documents and a replay of every design in Isaac Sim under `model_1199.pt`. Every artefact cited is in `artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/`.

## 1. The collapse

The run trains 100 designs per generation for 400 iterations, and from iteration 0 to 6000 it draws them at random inside a range that widens each generation (`../co_optimisation/co_optimisation/runners/usd_generator.py:846-860`). The share of episodes ended by `leg_contact` stood near 0.07 to 0.08 from iteration 240 to 760, rose to 0.12 at 800, to 0.24 at 840 and to 0.52 at 1200, while `time_out` fell from 0.87 at 720 to 0.31 at 1200 (`artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/tb_leg_contact.txt`). The rise therefore begins with generation 2, whose population is the one logged in `design_parameters_1199.csv`, since the runner writes that file before the swap that follows iteration 1199 (`../co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:469-473`).

The three populations differ in what they may pick. Generation 0 held only the TRON1 actuators with lengths within 1.2 percent of the base. Generation 1 added `TMotor-AK10-9` at the abduction and `Bota-Roboticer-50` at the ankle, and generation 2 draws every actuator of the catalogue at every joint, among them `TRON1-ABAD` at the ankle in 29 of the 100 designs (`artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/design_parameters_1199.csv`).

## 2. What the static geometry says

The 100 documents of generation 2 were scanned in MuJoCo over 300 samples each of every joint drawn within ±0.2 rad, the range of the reset event, counting only pairs that involve a body of the `leg_contact` list and excluding directly jointed bodies. Every collision found is between the two legs, the shank against the other shank, the other ankle housing or the other foot. The share of samples in collision is 4.1 percent in generation 2 and 3.9 percent in generation 1, and no pair involves an abduction, hip or knee housing (`artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/static_pairs.txt`). The geometry of generation 2 is therefore no more prone to contact at reset than that of generation 1, which had a `leg_contact` share of 0.08, so the rise does not come from where the bodies stand at reset.

## 3. What the replay says

`scripts/analysis/leg_contact_by_design.py` rebuilds the 100 designs from the CSV, spawns one per environment, loads `model_1199.pt` and runs 1500 policy steps. It wraps `leg_contact` so that at each evaluation, before the reset it triggers, the bodies of the list above the 1 N threshold are tallied, and one filtered `ContactSensor` per listed body splits each force into the body pair that carries it and a remainder, taken as terrain (`../scripts/analysis/leg_contact_by_design.py:102-124`, `:140-147`). The replay recorded 6590 `leg_contact` terminations in 6747 episodes.

The terminations belong to one group of designs. The 29 designs that pick `TRON1-ABAD` for the ankle account for 6556 of the 6590, a mean of 226 per design and a mean episode of 6.6 steps. The 71 others account for 34 terminations, 26 of them with at least one and none with more than 3, and run episodes of 777 steps on average. Every one of the 29 has at least 182 terminations, so the group is homogeneous, and the housing, the lengths and the other three picks do not separate it, the correlation of terminations with the thigh and shank length scales being 0.21 and 0.02 (`artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/replay_1199.txt`).

| Ankle pick | Designs | Terminations per design | Mean episode in steps | Housing diameter in m | Stiffness | Damping |
|---|---|---|---|---|---|---|
| `TRON1-ANKLE` | 31 | 0.5 | 762 | 0.060 | 10 | 0.5 |
| `Bota-Roboticer-50` | 40 | 0.5 | 790 | 0.098 | 10 | 0.5 |
| `TRON1-ABAD` | 29 | 226.1 | 6.6 | 0.0966 | 55 | 13.5 |

The housing does not explain it, since `Bota-Roboticer-50` is as wide as `TRON1-ABAD` and survives. What `TRON1-ABAD` brings to the ankle is the gains of the abduction joint, a stiffness of 55 and a damping of 13.5 against the ankle's 10 and 0.5, which `_actuator_params` copies from the catalogue entry (`../co_optimisation/co_optimisation/runners/usd_generator.py:887-898`).

## 4. The control experiment

The replay was run a second time with the ankle gains exchanged between the two picks, `TRON1-ABAD` given 10 and 0.5 and `Bota-Roboticer-50` given 55 and 13.5, every housing and length unchanged (`artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/replay_1199_ankle_gain_swap.txt`).

| Ankle pick | Gains given | Terminations per design | Mean episode in steps |
|---|---|---|---|
| `TRON1-ABAD` | 10 and 0.5 | 0.8 | 439 |
| `Bota-Roboticer-50` | 55 and 13.5 | 195.3 | 7.5 |

The failure moved with the gains to the other housing. The ankle gains alone, and not the housing, drive the collapse. The experiment does not separate the stiffness from the damping, which were exchanged together.

## 5. The colliding pairs

In the 29 failing designs the terminating contacts are all between the two legs. The shank of one leg against the foot of the other accounts for 2343 and 2259 flagged steps for the right and left shank, against the other ankle housing 303 and 285, against the other thigh 146 and 139, and against the other shank 136 (`artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/replay_1199.txt`). Of all flagged body steps in the replay the shank is present in 6822 of 7269, thigh bodies in 447 and the abduction, hip and knee housings in none. The remainder after subtracting the pair forces, taken as terrain, is 1275 of the 6556 flagged steps in the failing designs, and since the pair sensors read the last physics step alone while the net force reads four, that share is an upper bound.

The legs cross. With a foot that answers the policy's ankle command five times more stiffly and twenty seven times more damped than the ankle it was trained on, the swing foot is thrown across the midline within a few steps, and the shank of the stance leg is the body it strikes. The direct cause of the stiffness and damping acting through the foot is not measured here, only the attribution to the gains is, and the mechanism is an inference from the pair data.

## 6. Verdict on the hypothesis

The hypothesis that the shank is the body with the most contacts is confirmed, the shank being in 94 percent of the flagged steps. The conclusion that its contact should be disabled is not supported. Removing the shank from `leg_contact` would remove the termination of the 29 designs and leave their legs crossing for the whole episode, with the policy no longer told by an early reset that it failed and left only the penalty of `pen_undesired_contacts`, which does list the shank at a weight of -2.0 (`../environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:956-967`), a cost that a design with no gait to lose can accept. It would also remove the guard against kneeling that the entry of 2026-11-17 in [tron1.md](tron1.md) records the shank as serving. The shank is where the contact shows, and the cause is a design that no policy of this run can walk.

A second effect hides behind the first. The 29 designs spend nearly all their steps in 6.6 step episodes, so a third of the population yields almost no learning signal, and the fitness of those designs measured over episodes that short ranks them last for a reason unrelated to the design variables the search is meant to compare. The swapped run shows that the design with soft gains is still not stable, its episodes lasting 439 steps against 762 for the other `TRON1-ANKLE` designs, so the gains are the cause of the contact and not the whole of the difficulty.

## 7. Suggested corrections

The leg contact list stays as it is. Three changes address the cause, ordered by how little they alter.

1. Give the ankle group the gains of the joint and not of the pick. `_actuator_params` takes stiffness and damping from the catalogue entry, which is correct for a pick made for its own joint and wrong for a pick borrowed from the abduction. An optional argument naming the groups whose gains stay at the base's, defaulting to none so that every existing caller is unchanged, set only by the calling configuration, follows the rule of the project that the default reproduces the old behaviour and the defect stays recorded in the unset default.
2. Alternatively restrict the ankle's catalogue to the actuators whose gains lie within a stated factor of the ankle's, which keeps the pick's gains and removes the designs that cannot stand. This needs a bound that the data of this run cannot give, since only three ankle picks were drawn.
3. Add a filter that refuses a design whose ankle gains lie outside the range the current policy has seen, beside the `StandingFilter` already configured (`../co_optimisation/co_optimisation/utils/filters.py:283`), since that filter passed all 100 designs of generation 2, 29 of them unwalkable.

Whichever is chosen, the 29 designs of generation 2 are already trained for 400 iterations and the generation 3 population is spawned, so the run is best resumed from the checkpoint at iteration 799 with the change applied, the iterations 800 to 1199 having trained the policy on episodes of 6.6 steps for a third of its environments.

## 8. Limits

The replay is one run of 1500 steps with one seed on one checkpoint, the policy trained on the population it is replayed against, so the terrain levels and the learning state of the rest of training are not reproduced. The terrain level of each environment was not recorded. The pair attribution reads the last physics step. The replay used the working tree's `leg_contact` and the run's rewards, actuators and action scale restored from its `params/env.yaml`. The tension between the 5.5 times stiffer ankle and the policy's observations is untested, the gains reaching the policy only through the morphology observation, and whether the policy of a later generation would learn to walk on a stiff ankle is not established.

## 9. Outcome of the correction, 2026-11-19

Option 1 of section 7 was implemented, more broadly than stated there, as [../plans/actuator_gains_decoupling.md](../plans/actuator_gains_decoupling.md), the gains of every joint group becoming fields of the design specification and leaving the actuator catalogue. The same replay of the 100 designs of generation 2 under `model_1199.pt`, run on the changed tree, ended 23 episodes by `leg_contact` against 6590 before, and the 29 designs that pick `TRON1-ABAD` for the ankle fell from a mean of 226.1 terminations to 0.3, their episodes lengthening from 6.6 to 1036 steps (`artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/replay_1199_decoupled_gains.txt`). The gains read back from the simulator equal the baseline's in every environment (`artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/gain_readback_decoupled.txt`). The mechanism of section 5 remains an inference, but the attribution to the gains is now confirmed by removing them from the pick.
