# Terminating CMA-ES on Its Own, Fine-Tuning on the Final Design, and a Configuration Class for the Co-Optimisation Search

> Status, IMPLEMENTED 2026-10-28 in code, every edit E1 to E9 applied as written, with one addition, a guard in `_video_paused` that stops the walk of the wrapper chain at a layer already seen. The host checks 1 and 3 of section 8 passed in full, 22 of 22, reproduced by `../context/artefacts/copt_training_refinement-2026-10-28/host_check.py` and its output `host_check.txt`, and every edited file compiles. The container smoke test of section 8 items 4 to 7 has not been run, so the fine-tune, the aligned save, the video pause and the restore of `params/cmaes.pkl` are verified by reading and not yet by a run. The tolerances of section 3 were derived from four runs and checked against pycma's own termination test on every one of their checkpoints, two of the runs being still in training when they were read, as `../context/artefacts/copt_training_refinement-2026-10-28/snapshot.txt` records. See [README.md](README.md) for the register.

## 1. Introduction

The co-optimisation runner alternates proximal policy optimisation over a population of designs with a CMA-ES update of the distribution that samples them, and it was built to continue the search until a hard generation cap. That cap is set to `(max_iterations - ea_late_start) / ea_update_interval` (`scripts/rsl_rl/train.py:302-303`), which for the configuration every run uses, 45000 iterations, a late start at 6000 and an interval of 400 (`/ws/djinn:135`, `:154`), is 97.5 generations, so the search runs to the last of the 45000 iterations. pycma's tolerances could end it earlier, but none of them is set, and their defaults of `1e-11` for `tolx` and `tolfun` and `1e-12` for `tolfunhist` are chosen for noiseless numerical optimisation [1]. Not one of the 47 checkpoints of the four runs studied here that hold a CMA-ES strategy reports a termination under them (`../context/artefacts/copt_training_refinement-2026-10-28/es_state.txt`).

This plan makes the search end on its own and the run end soon after. It derives a fitness tolerance and a standard deviation tolerance for each robot from four runs, `copt_moral/2026-10-23_08-31-17` and `copt_moral/2026-10-26_04-55-55` for the TRON1 biped, `quadruped_copt/2026-10-23_10-36-49` and `quadruped_copt/2026-10-26_04-55-55` for the quadruped, and passes them to `CMAESDesignGenerator`, whose `_make_strategy` sets them on the strategy. Once CMA-ES terminates, the runner loads the final outcome of the search, trains the policy on it for 400 further iterations and returns.

The tolerances are properties of the present search, four parameters over link lengths and nominal pose offsets, and will change when the search space grows. They are therefore not written into the launcher or the training script. Every setting of the co-optimisation search and loop moves into a configuration class in a new package `co_optimisation/co_optimisation/config/`, with one subclass per robot, and the configuration of a run is dumped to `params/cmaes.yaml` and `params/cmaes.pkl` beside the agent and environment configurations, from which a resumed run restores it. The script arguments that carried part of this configuration are retired.

Three changes to the record of a run accompany the termination, and each is built into the runner rather than offered as an option. A checkpoint is taken before every design change and carries the same index as that generation's design file and history row. The training video films proximal policy optimisation only, never the controlled evaluation of the designs. And `log_optimised_designs` records the final design, the distribution mean that the search recommends, which is the design every later evaluation of the run uses.

The user settled four questions while the plan was written. The final outcome loaded for the fine-tune is the distribution mean, pycma's estimate of the optimum, placed in every individual slot. Checkpoints are taken with a save interval of 400, the index of the checkpoint and of the design file being made the same. `log_optimised_designs` remains one method, which always heads its record with the design `optimised_design()` returns and reports its evaluated fitness whenever that design is the one on the scene. And the configuration class holds every co-optimisation setting, the evaluation and the builder geometry included, so that the `copt` block of the agent configuration is built from it entirely.

Section 2 traces the present behaviour through the code. Section 3 defines the quantities the analysis uses, reads the four runs, and arrives at the tolerances. Section 4 sets out the edits with their code, section 5 the order of application, section 6 the callers each edit reaches, section 7 the defects left standing, and section 8 the verification.

## 2. The Present Implementation

### 2.1 Termination in pycma 4.4.4

The container installs `cma==4.4.4` (`/ws/IsaacLab/docker/Dockerfile.base:92`), and the source of that version, read on the host from the identical wheel, governs every statement in this section [1]. The generator constructs the strategy with a population, bounds, a seed, a logging prefix, the generation cap and the boundary handler, and no tolerance (`co_optimisation/co_optimisation/runners/usd_generator.py:890-903`). It asks `es.stop()` once per generation, immediately after `tell`, and sets `_terminated` when the returned dictionary is not empty (`usd_generator.py:942-953`).

pycma minimises a cost, and the generator gives it the negated fitness of each design (`usd_generator.py:948`). Three tolerances bear on the request, each of them a threshold below which a measured quantity must fall.

1. `tolx` measures the spread of the search distribution. It fires when, in every one of the four coordinates, both the standard deviation of the sampling distribution, `sigma * sigma_vec * sqrt(diag C)` in the unit box CMA-ES searches [2], and the evolution path scaled by the step size, `sigma * sigma_vec * pc`, lie below the threshold (`cma/evolution_strategy.py:4157-4159`). The second comparison is made on the signed path rather than its magnitude, so a negative component always passes it.
2. `tolfun` measures two spreads of cost at once and fires only when both lie below the threshold (`evolution_strategy.py:4163-4167`). The first is the range of the current generation, the largest cost of its 80 designs minus the smallest. The second is the range of a short history, the largest minus the smallest of the best costs of recent generations.
3. `tolfunhist` measures that history range alone, and fires once the history holds at least ten generations (`evolution_strategy.py:4171-4173`).

The history keeps the best cost of each of the last `10 + 30 N / popsize` generations, the oldest being dropped as each new one enters (`evolution_strategy.py:2657-2669`). With `N = 4` parameters and a population of 80 this is the last 11 generations. The remaining criteria cannot fire within a run. `tolstagnation` defaults to `int(100 + 100 N^1.5 / popsize)`, here 110 generations, more than the cap allows, and `tolxstagnation` requires the mean to move by less than `1e-9` over twenty or more generations.

One property of `stop()` matters to verification. It does not re-test within an iteration it has already tested unless `check_in_same_iteration=True` is passed, and its docstring states that this argument is required after the termination options are changed by hand (`evolution_strategy.py:993-1001`). The runner never meets this, because it calls `stop()` only after a `tell`, which advances the iteration.

### 2.2 The runner after termination

When the search terminates, `generate_population` sets the pending solutions to the last ones and returns `None` (`usd_generator.py:955-959`). `_update_morphology` then re-applies the previous population (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:727-733`), so the run trains on the 80 designs of the generation evaluated last. The training loop carries on to `max_iterations`, evaluating that population at every boundary and passing the fitness to an `update_with_fitness` that ignores it (`copt_on_policy_runner.py:405-419`, `usd_generator.py:946`).

`log_optimised_designs` runs once, after the final checkpoint (`copt_on_policy_runner.py:460-467`). It evaluates the population on the scene, ranks its individuals, and writes the MJCF of every individual, the MJCF of the distribution mean, a table of the individuals and a text summary to `designs/optimised/` (`:1062-1141`). The design it names as best is the fittest individual of that evaluation. The distribution mean, the design `optimised_design()` returns and the one the search recommends (`usd_generator.py:934-936`), is written as a document but never evaluated, since it is not on the scene.

### 2.3 Saving and the index offset

The runner saves when `it % save_interval == 0` (`copt_on_policy_runner.py:446-447`), so with an interval of 400 the checkpoints fall at iterations 0, 400, 800 and so on. The design swap happens when `(it + 1) % ea_update_interval == 0` (`:323-325`), at iterations 399, 799 and so on, and the design file and the history row of that generation are written with the index `it` of the swap (`:436-439`, `:994`, `:1002-1003`). A checkpoint therefore lands one iteration after each swap, and its index exceeds by one that of the design file and the history row. The two indices also name different populations. `design_parameters_399.csv` holds the population spawned at the swap, the one trained from iteration 400 onwards, whereas history row 399 holds the evaluation of the population trained up to iteration 399.

The place of the save matters for a resume as well. rsl_rl restores `current_learning_iteration` and resumes the loop at that same index, so a checkpoint saved after the swap at iteration 399 would repeat the swap on resume and run one CMA-ES generation twice, whereas a checkpoint saved before the swap repeats one iteration of training and then evaluates and swaps exactly as the uninterrupted run did.

### 2.4 Video recording through the evaluation

`train.py` wraps the environment in `gym.wrappers.RecordVideo` with a step trigger of `step % video_interval == 0` (`scripts/rsl_rl/train.py:217-230`), and `djinn` passes an interval of 10000 steps and a clip of 2000 (`/ws/djinn:216`, `:218`). In gymnasium 1.2.1, `RecordVideo.step` advances its step counter `step_id` and captures a frame on every step that passes through it (`gymnasium/wrappers/rendering.py:359-374`). The controlled evaluation steps the wrapped environment, `self.env.step` (`copt_on_policy_runner.py:926`), for 5 terrain rows, 1 column pass and 200 control steps, 1000 steps per generation, whereas one generation of proximal policy optimisation is 400 iterations of 25 steps, exactly 10000. Before the CMA-ES phase every clip therefore begins at a design swap, and from the first evaluation the counter drifts by 1000 steps a generation, so the clips walk into the evaluations. A model of the counter, described in section 3.8, places the clips at steps 160000, 270000 and 380000 of `quadruped_copt/2026-10-26_04-55-55` as starting inside an evaluation, and six of its clips as overlapping one. The evaluation's resets act on `env.unwrapped` and bypass the wrapper, so only its steps pass through.

### 2.5 Where the configuration lives today

The settings of the search are scattered. `train.py` takes the generator, the population, the generation interval, the late start and the shank gap as arguments (`scripts/rsl_rl/train.py:45-100`), holds the parameter ranges, `sigma0`, the seed and the prior size of the late start as literals (`:273-305`), and writes the evaluation settings as literals into the `copt` block of the agent configuration (`:322-333`). `djinn` supplies the arguments per robot (`/ws/djinn:135`, `:139`, `:154`, `:158`). Only the `copt` block is dumped, to `params/copt_agent.yaml` (`train.py:371-373`), and a resumed run rebuilds every setting from whatever arguments it is launched with, so nothing guarantees that it continues under the configuration it began with.

## 3. The Evidence of the Four Runs

### 3.1 The quantities and how each is computed

The analysis reads the per generation record `designs/cmaes_history.csv`, the per individual tables `designs/cmaes_generation_<g>.csv`, and the pickled strategy in every checkpoint, through the scripts of section 9. Each quantity below is stated with its procedure and with the question it answers.

1. The fitness of a design is its mean return over the controlled evaluation, in which the 80 instances of the design stand in the 80 terrain columns of each of the 5 difficulty rows for 4 s, 400 episodes in all (`copt_on_policy_runner.py:832-985`). It answers how well the current policy walks on that design.
2. A generation is one population of 80 designs, evaluated once at its end, every 400 iterations. The late window of a run is its last ten logged generations. Every late figure below is the median over that window, so that a single unusual generation does not set it.
3. The fitness spread of a generation is the standard deviation of its 80 fitnesses, the `fitness_std` column. It mixes two things, the true differences between the designs and the chance of the evaluation.
4. The split-half rank correlation `r` separates them. The evaluation is split into its even and its odd terrain columns, the 80 designs are ranked on each half, and the two rankings are correlated (`copt_on_policy_runner.py:962-969`). If the halves agree the ranking reflects the designs, and if they disagree it reflects chance. The share of the fitness variance that belongs to the designs is the reliability `2r / (1 + r)` (`scripts/analysis/cmaes_analysis.py:409-427`).
5. The evaluation noise is the standard deviation of a design's fitness that is due to chance alone, computed as `sqrt(1 - reliability) * fitness_std` and written `σ` below. It is the yardstick against which every fitness tolerance is measured, since a change of fitness smaller than the noise cannot be told from chance.
6. The current range of a generation is its largest fitness minus its smallest, `fitness_max - fitness_min`. In cost it is the same number, and it is the first quantity `tolfun` tests.
7. The history range at a generation is the largest minus the smallest of the best fitness, `fitness_max`, of that generation and the ten before it. It is the quantity `tolfunhist` tests and the second that `tolfun` tests.
8. The sampling standard deviation of a coordinate is the standard deviation the runner logs in parameter units, `std/<parameter>` (`usd_generator.py:919-932`), divided by the width of that parameter's range, so that it is in the unit box `tolx` compares against. The largest of the four is the one `tolx` must bring below its threshold.
9. The length quantum is the smallest change of length the generator can emit. It rounds every scaled length to 0.01 m (`usd_generator.py:725`), and dividing 0.01 m by the baseline length and by the range of the scale, 0.5, expresses it in the unit box.

### 3.2 The noise of the fitness

In the late window of every run the split-half rank correlation lies between -0.31 and 0.24 (`analyse.txt`), so the reliability is near zero and almost the whole fitness spread of a generation is evaluation noise. The search has narrowed until its designs differ by less than the evaluation can resolve. The noise is about three to four times larger on the biped than on the quadruped, which is why the two robots need separate fitness tolerances.

| Run | Late noise `σ` |
|---|---|
| `copt_moral/2026-10-23_08-31-17` | 0.332 |
| `copt_moral/2026-10-26_04-55-55` | 0.289 |
| `quadruped_copt/2026-10-23_10-36-49` | 0.101 |
| `quadruped_copt/2026-10-26_04-55-55` | 0.084 |

### 3.3 What `tolfun` measures here, and why it cannot detect convergence

`tolfun` requires the current range, quantity 6, to fall below its threshold. When the designs of a generation are equal in merit and differ only by noise, the current range is the range of 80 values drawn from one normal distribution of standard deviation `σ`. That range has a known expected size. Drawing 200000 sets of 80 standard normal numbers and averaging the largest minus the smallest of each set gives 4.85, so the expected current range is `4.85 σ`. It cannot be smaller however settled the search, because it is produced by the evaluation and not by the designs.

The runs bear this out. The table sets the observed late current range beside the `4.85 σ` that pure noise predicts.

| Run | Observed current range | `4.85 σ` |
|---|---|---|
| `copt_moral/2026-10-23_08-31-17` | 1.72 | 1.61 |
| `copt_moral/2026-10-26_04-55-55` | 1.40 | 1.40 |
| `quadruped_copt/2026-10-23_10-36-49` | 0.53 | 0.49 |
| `quadruped_copt/2026-10-26_04-55-55` | 0.43 | 0.41 |

The observed range is the noise prediction to within 7 per cent in every run. The current range is therefore a measurement of the evaluation noise, not of convergence. A `tolfun` threshold below about `4.85 σ` can never fire, and one above it would fire on the noise level alone, whether or not the search had settled. This corrects the expectation of `/ws/context/cmaes.md` section 5, written of an earlier two-dimensional run, that a collapsed population makes the within-generation range zero. The pose offsets of the present search are resolved to a thousandth of a radian, and the evaluation noise keeps the range wide.

### 3.4 What `tolfunhist` measures, and the fitness tolerance

The change of fitness the request names is the movement of the best design's fitness from generation to generation, which is the history range, quantity 7, that `tolfunhist` tests. Its expected size under pure noise follows the same reasoning. If neither the design nor the policy improved, the best fitness of each generation would be the largest of 80 noisy values, which varies from generation to generation with a standard deviation of `0.44 σ`, and the range of 11 such values averages `1.39 σ`, both by the same simulation. A history range near `1.39 σ` therefore means the best fitness has not moved beyond what chance produces.

| Run | Observed history range | In units of `σ` | Late change of mean fitness per generation |
|---|---|---|---|
| `copt_moral/2026-10-23_08-31-17` | 0.90 | 2.7 | +0.107 |
| `copt_moral/2026-10-26_04-55-55` | 1.07 | 3.7 | +0.009 |
| `quadruped_copt/2026-10-23_10-36-49` | 0.49 | 4.9 | -0.017 |
| `quadruped_copt/2026-10-26_04-55-55` | 0.35 | 4.1 | +0.032 |

The observed history ranges lie between 2.7 and 4.9 noise standard deviations, above the `1.39 σ` of chance. The excess is not progress of the design, since in the late window of `quadruped_copt/2026-10-26_04-55-55` every one of the 80 designs of every generation emits the same leg lengths (section 3.5). It is the generation to generation movement of the policy's performance, which the last column shows as a median change of the mean fitness of up to 0.107 a generation. The fitness tolerance is therefore set at three noise standard deviations, about twice the range of chance alone. A history range below it says that the best fitness has moved over 11 generations by no more than chance and a slow change of the policy can explain. For the biped this is `3 × 0.29` to `3 × 0.33`, rounded to 1.0, and for the quadruped `3 × 0.084` to `3 × 0.101`, rounded to 0.3.

`tolfun` is set to the same value. Because it also requires the current range to fall below the threshold, and section 3.3 shows that range to stay near `4.85 σ`, it can fire only in a generation where `tolfunhist` also fires, and adds no condition of its own. `tolfunhist` is the operative fitness criterion. On the biped it ends `copt_moral/2026-10-23_08-31-17` (section 3.6), and on the quadruped, whose history range stays above `4 σ`, the search is ended by `tolx` instead.

### 3.5 The standard deviation tolerance

The quanta of length set the physical scale of `tolx`. The baseline thigh and shank are 0.30 m and 0.35 m on the biped and 0.213 m on the quadruped (`design_generator/biped/variable_params.json:17`, `:21`, `design_generator/quadruped/variable_params.json:17`, `:21`). A 0.01 m step is a change of scale of 0.033 on the biped thigh, 0.029 on its shank and 0.047 on either quadruped link, which in the unit box, the scale range being 0.5 wide, is 0.067, 0.057 and 0.094.

| Run | Late largest sampling std | Binding coordinate | Distinct leg length pairs per late generation | In the first generations |
|---|---|---|---|---|
| `copt_moral/2026-10-23_08-31-17` | 0.072 | knee offset | 8 to 14 | 42 to 60 |
| `copt_moral/2026-10-26_04-55-55` | 0.033 | knee offset | 3 to 5 | 52 to 70 |
| `quadruped_copt/2026-10-23_10-36-49` | 0.063 | thigh offset | 1 to 4 | 46 to 59 |
| `quadruped_copt/2026-10-26_04-55-55` | 0.017 | shank length | 1 | 46 to 59 |

The distinct leg length pairs are counted by rounding the 80 sampled lengths of each generation as the generator does and counting the different thigh and shank combinations among them (`grid.py`). The search has collapsed from tens of robots to a handful, and in one run to a single robot, so the length search is resolved at the level the generator can express.

The `tolx` of each robot is half its smallest length quantum in the unit box, 0.029 for the biped, from its shank, and 0.047 for the quadruped, rounded to 0.03 and 0.05. When every coordinate's sampling standard deviation is below half a quantum, two thirds of the samples, the share of a normal distribution within one standard deviation, lie within half a step of the mean's length and round to the same length. On the pose offsets, whose ranges are 0.2 rad at the hip and 0.5 rad at the knee, the same thresholds are 0.006 rad and 0.015 rad on the biped and 0.010 rad and 0.025 rad on the quadruped.

### 3.6 The verdict of pycma on every checkpoint

The pickled strategy of every checkpoint was read without torch, its options set to the chosen values, and `stop(check_in_same_iteration=True)` asked for its verdict (`verify_stop.py`, `verify_stop.txt`). This is pycma's own test on the state the run actually held, including the evolution path condition of `tolx` that the logs do not record.

| Run | Tolerances | First checkpoint that terminates | CMA iteration, history row | Criterion | Iterations with fine-tune |
|---|---|---|---|---|---|
| `copt_moral/2026-10-23_08-31-17` | `tolx` 0.03, `tolfun` and `tolfunhist` 1.0 | 16000 | 25, row 15999 | `tolfunhist` | at most 16399 |
| `copt_moral/2026-10-26_04-55-55` | the same | 19000 | 32, row 18799 | `tolx` and `tolfunhist` | at most 19199 |
| `quadruped_copt/2026-10-23_10-36-49` | `tolx` 0.05, `tolfun` and `tolfunhist` 0.3 | 16000 | 25, row 15999 | `tolx` | at most 16399 |
| `quadruped_copt/2026-10-26_04-55-55` | the same | 12000 | 15, row 11999 | `tolx` | at most 12399 |

Checkpoints were taken every 1000 iterations, two and a half generations, so a first verdict states that termination came no later than that generation. The history alone, which records the sampling standard deviation but not the evolution path, places the first generation below the `tolx` threshold earlier in some runs, for instance at row 11199 of `quadruped_copt/2026-10-26_04-55-55` (`grid.txt`), and the true termination lies between the two. Every run terminates within its logged horizon, between iterations 11999 and 18799 where the cap would end it at 45000.

### 3.7 What stopping costs

Stopping is worth its saving only if the search would not have moved after it. For each run the distribution mean at the terminating generation of section 3.6 was compared with the mean at the last logged generation, the shift of each coordinate being expressed as a fraction of its range (`after_stop.py`, `after_stop.txt`).

| Run | Generations logged after the stop | Largest shift of a coordinate | Change of emitted leg lengths | Change of best fitness |
|---|---|---|---|---|
| `copt_moral/2026-10-23_08-31-17` | 7 | 0.064, knee offset | shank 0.32 m to 0.31 m | +0.49 |
| `copt_moral/2026-10-26_04-55-55` | 2 | 0.018, thigh offset | none | -0.14 |
| `quadruped_copt/2026-10-23_10-36-49` | 1 | 0.002, thigh offset | none | -0.01 |
| `quadruped_copt/2026-10-26_04-55-55` | 14 | 0.001, shank length | none | +0.44 |

In three runs the robot emitted at the distribution mean does not change after the stop, and in the fourth the shank shortens by a single 0.01 m step over seven generations. The changes of the best fitness lie within one to two noise standard deviations, and where they are positive the mean fitness rises with them, which is the policy improving on an unchanging design. The 400 iterations of fine-tuning on the final design deliver that improvement directly. Against these small costs, the tolerances end the search about two thirds of the way earlier than the cap.

Both quadruped runs end with the shank scale and the two pose offsets on their lower bounds, and `quadruped_copt/2026-10-23_10-36-49` with the thigh scale there too (`after_stop.txt`). A variable that consistently ends on its bound is a sign that the box, not the objective, set the outcome [3]. This plan does not act on it, and section 7 records it.

### 3.8 The video model

The model behind section 2.4 adds 25 steps per iteration and 1000 at each generation boundary from iteration 6400, when the first CMA-ES generation is evaluated, and compares the clip names recorded under `videos/train/` with the evaluation windows so obtained. It is a model built on the evaluation's step count rather than on a log of the recorder. Its finding, that clips begin inside an evaluation, is made impossible by the edit of section 4.7 whatever its exact count.

### 3.9 The chosen tolerances

| Robot | `tolx` | `tolfun` | `tolfunhist` | Grounds |
|---|---|---|---|---|
| Biped, tasks `Isaac-Limx-SF-Copt-*` | 0.03 | 1.0 | 1.0 | half the shank's 0.057 length quantum, and three noise standard deviations of 0.29 to 0.33 |
| Quadruped, tasks `Isaac-Quadruped-Copt-*` | 0.05 | 0.3 | 0.3 | half the 0.094 length quantum, and three noise standard deviations of 0.084 to 0.101 |

The generation cap is retained as a safety net. Should the noise ever grow until neither criterion can be met, the published remedies act on the evaluation rather than on the tolerances, uncertainty handling, which re-evaluates candidates and enlarges the step size or the budget when re-evaluation reorders them [4], and learning rate adaptation, which holds the signal to noise ratio of the update constant [5]. The values depend on the evaluation protocol and on the parameters searched, and are to be re-derived with the scripts of section 9 whenever the evaluation horizon, the population, the reward set or the search space changes, since each moves the noise of section 3.2 or the quanta of section 3.5. The configuration class of section 4.3 holds them per robot for that reason.

## 4. The Edits

The configuration class is the single source of every co-optimisation setting. The generator receives the tolerances as optional constructor arguments whose default leaves pycma's own, the runner receives the fine-tune length through the `copt` block of the agent configuration, which records the tolerances as well, and both are filled from the configuration class by `train.py`. A run restored from a checkpoint written before this plan resumes as it was launched, without the tolerances or the fine-tune (section 4.8).

### 4.1 E1, an optional individual builder in `DesignGeneratorBase._build_population`

`usd_generator.py:670-680`. The fine-tune population of section 4.2 is one design built through the shared pipeline, which needs `_build_population` to accept the builder of an individual. `None` reproduces the present call exactly.

```python
    def _build_population(self, generation: int, indices, individual=None) -> Population:
        """Build generation *generation* for the individual indices *indices*.

        *individual*, a callable of (generation, idx) returning (spec, actuator_params), takes
        the place of _generate_individual where given. None, the default, builds as before.
        """
        generate = individual or self._generate_individual
        usd_files: list[str] = []
        actuator_params: list[dict[str, dict]] = []
        link_length_params: list[dict[str, dict]] = []
        specs: list = []
        nominal: list = []
        parameters: list[dict] = []
        xml_files: list[str] = []
        for idx in indices:
            spec, act = generate(generation, idx)
            # ... the remainder of the loop unchanged
```

### 4.2 E2, tolerances and a final population in `CMAESDesignGenerator`

Four keyword arguments of the constructor (`usd_generator.py:841-886`), stored before the strategy is made, since `_make_strategy` reads them.

```python
        max_cma_iter: int = 10000,
        variable_params_path: str | None = None,
        tolx: float | None = None,
        tolfun: float | None = None,
        tolfunhist: float | None = None,
        final_population: str = "last",
    ) -> None:
        super().__init__(...)                                   # unchanged
        ...
        self._max_cma_iter = max_cma_iter
        # pycma's tolerances, see plans/copt_training_refinement.md section 3. None, the default,
        # leaves pycma's 1e-11, 1e-11 and 1e-12, as every run before 2026-10-28.
        self._tolerances: dict[str, float] = {
            name: float(value)
            for name, value in (("tolx", tolx), ("tolfun", tolfun), ("tolfunhist", tolfunhist))
            if value is not None
        }
        # What a terminated search returns, "last" None, so the runner re-applies the last
        # population as before, and "mean" the distribution mean in every individual slot.
        if final_population not in ("last", "mean"):
            raise ValueError(f"final_population must be 'last' or 'mean', got {final_population!r}")
        self._final_population = final_population
```

`_make_strategy` (`usd_generator.py:890-903`) passes them to the options.

```python
            "maxiter": self._max_cma_iter,
            "BoundaryHandler": "BoundTransform",
            # Empty unless the caller sets them, see __init__.
            **self._tolerances,
        })
```

A strategy restored from a checkpoint, by `es_state_path` or by `load_state`, keeps the options it was made with (`usd_generator.py:877-880`, `:1021-1028`), so a run resumed from a checkpoint written before this plan continues under pycma's defaults, as it was launched.

`generate_population` (`usd_generator.py:955-959`) returns the distribution mean when asked, and its present branch is kept under the default.

```python
        if self._terminated:
            self._pending_solutions = self._last_solutions
            if self._final_population == "mean":
                print("CMAES Terminated. Fine-tuning on the distribution mean in every individual slot")
                return self._mean_population(generation)
            print("CMAES Terminated. Fine-tuning parent policy on selected design only")
            return None
```

The present code prints its message before assigning the pending solutions, and the order of the two statements has no effect on the state. The mean is emitted and converted once and its records repeated, so the fine-tune costs one conversion rather than 80. The population keeps its size, which keeps the in-place update of `_update_morphology` valid, since that update reaches only as many environments as were given a prototype at the last respawn (`copt_on_policy_runner.py:688-699`). The repeated records are shared objects, which is safe because every consumer only reads them.

```python
    def _mean_population(self, generation: int) -> Population:
        """The distribution mean, pycma's estimate of the optimum, in every one of num_individuals slots."""
        values, picks = self.optimised_design()

        def mean(generation: int, idx: int):
            self._current_parameters = dict(values)
            spec = self._spec_from_parameters(values, picks)
            return spec, self._actuator_params(spec)

        one = self._build_population(generation, range(1), individual=mean)
        n = self.num_individuals
        return Population(
            one.get_usd_files() * n, one.get_actuator_params() * n, one.get_link_length_params() * n,
            one.get_root_body(), one.get_specs() * n, one.get_nominal() * n,
            one.get_parameters() * n, one.get_xml_files() * n,
        )
```

### 4.3 E3, the configuration class in `co_optimisation/co_optimisation/config/`

A new package of two files. `cmaes_cfg.py` holds the base class, whose values are those every `djinn` launch of a co-optimisation run has used, and one subclass per robot carrying the values of section 3.9. The class is an Isaac Lab `configclass`, as the agent and environment configurations are, so it dumps and pickles in the same way. A derived configuration is made by copying and assigning fields, not through the `configclass` replace helper.

```python
"""Configuration of the design search and the co-optimisation loop of a run.

Dumped to <experiment_dir>/params/cmaes.yaml and cmaes.pkl by scripts/rsl_rl/train.py, and
restored from cmaes.pkl when a run is resumed, so that a run continues under the configuration
it began with. See plans/copt_training_refinement.md.
"""

from __future__ import annotations

import copy

from isaaclab.utils import configclass


@configclass
class CMAESCfg:
    """Every setting of the design search and of the runner's co-optimisation loop.

    The values are those every djinn launch of a co-optimisation run used before 2026-10-28.
    The tolerances and the fine-tune are None, so CMA-ES keeps pycma's defaults and the run
    trains to max_iterations. The robot subclasses set both.
    """

    design_search: str = "length"
    """"length", CMA-ES after a random phase whose prior widens from the donor to the full box,
    "cmaes" the same after a uniform random phase, "cat" adding the actuator of every joint, and
    "random" a fresh population every generation without CMA-ES."""

    num_individuals: int = 80
    """Designs per generation, which must divide the scene's num_envs."""

    ea_update_interval: int = 400
    """Policy iterations between design generations."""

    ea_late_start: int = 6000
    """Iteration at which the random phase hands over to CMA-ES."""

    param_ranges: dict = {
        "thigh_length_scale": ("scale", (0.75, 1.25)),
        "shank_length_scale": ("scale", (0.75, 1.25)),
        "thigh_bend_offset": ("offset", (-0.10, 0.10)),
        "knee_bend_offset": ("offset", (-0.25, 0.25)),
    }
    """The searched parameters and their ranges, in the form usd_generator.DEFAULT_PARAM_RANGES takes."""

    sigma0: float = 0.25
    """Initial step size of CMA-ES in the unit box."""

    seed: int = 42
    """Seed of the CMA-ES strategy."""

    late_start_prior_divisor: int = 64
    """The random phase draws num_envs // late_start_prior_divisor designs per generation."""

    max_cma_iter: int | None = None
    """Generation cap of CMA-ES. None derives (max_iterations - ea_late_start) / ea_update_interval."""

    tolx: float | None = None
    """Largest sampling std in the unit box at which CMA-ES stops. None keeps pycma's 1e-11."""

    tolfun: float | None = None
    """Current and historic fitness range at which CMA-ES stops. None keeps pycma's 1e-11."""

    tolfunhist: float | None = None
    """Range of the best fitness over the last 10 + 30 N / popsize generations at which CMA-ES stops.
    None keeps pycma's 1e-12."""

    finetune_iterations: int | None = None
    """Iterations trained on the distribution mean once CMA-ES terminates, after which training
    ends. None trains on to max_iterations on the last population."""

    randomise_before_late_start: bool = True
    """Swap the design population at every generation of the random phase."""

    reinit_lr_each_generation: bool = True
    """Re-initialise the learning rate at every generation of the random phase."""

    eval_duration_s: float = 4.0
    """Horizon of the controlled evaluation of a generation."""

    eval_seed: int = 42
    """Seed of the Latin hypercube of commands the evaluation draws."""

    shank_foot_gap: float | None = None
    """Quadruped only, the gap in metres between each shank's collision box and its foot sphere.
    None keeps the box reaching the sphere's centre."""


@configclass
class BipedCMAESCfg(CMAESCfg):
    """The TRON1 solefoot biped, tasks Isaac-Limx-SF-Copt-*. Tolerances of plans/copt_training_refinement.md section 3.9."""

    tolx: float | None = 0.03
    tolfun: float | None = 1.0
    tolfunhist: float | None = 1.0
    finetune_iterations: int | None = 400


@configclass
class QuadrupedCMAESCfg(CMAESCfg):
    """The quadruped, tasks Isaac-Quadruped-Copt-*. Tolerances of plans/copt_training_refinement.md section 3.9."""

    tolx: float | None = 0.05
    tolfun: float | None = 0.3
    tolfunhist: float | None = 0.3
    finetune_iterations: int | None = 400
    shank_foot_gap: float | None = 0.001


# The configuration of each design generator family, as scripts/rsl_rl/train.py resolves the family from the task.
CMAES_CFG_BY_FAMILY: dict[str, type[CMAESCfg]] = {
    "biped": BipedCMAESCfg,
    "quadruped": QuadrupedCMAESCfg,
}


def without_termination(cfg: CMAESCfg) -> CMAESCfg:
    """A copy of *cfg* without the tolerances and the fine-tune, for a run begun before params/cmaes.pkl existed."""
    legacy = copy.deepcopy(cfg)
    legacy.tolx = None
    legacy.tolfun = None
    legacy.tolfunhist = None
    legacy.finetune_iterations = None
    return legacy
```

`__init__.py` exports them.

```python
from .cmaes_cfg import CMAES_CFG_BY_FAMILY, BipedCMAESCfg, CMAESCfg, QuadrupedCMAESCfg, without_termination
```

The parameter ranges are written out in full, the two pose offsets at the ranges `DEFAULT_PARAM_RANGES` gives them (`usd_generator.py:431-435`), where `train.py` today names only the two scales and leaves the offsets to the default. The generator receives the same four ranges either way. The configuration names the searched space explicitly, which is the point of moving it out of the code.

### 4.4 E4, the fine-tune, the aligned save and the end of the run in `CoptOnPolicyRunner`

The fine-tune is read from the `copt` block in `__init__` beside the other keys (`copt_on_policy_runner.py:125-142`), and two variables of state carry it, a flag that the fine-tune is under way and the iteration at which it ends.

```python
        # Iterations trained on the final design once the search terminates, after which learn
        # returns. None, the default, trains on to max_iterations, as every run before 2026-10-28.
        self._finetune_iterations: int | None = copt_cfg.get("finetune_iterations")
        # True from the generation boundary at which the search terminated, when the fine-tune begins.
        self._finetuning: bool = False
        # The last iteration of the fine-tune, set with _finetuning.
        self._finetune_stop_it: int | None = None
```

The checkpoint carries both, so a resume within the fine-tune ends where the uninterrupted run would. `save` adds them to the `copt` record (`copt_on_policy_runner.py:183-189`) and `load` reads them with defaults (`:227-233`), so a checkpoint written before this plan loads unchanged.

```python
            "cmaes_history": self._cmaes_history,
            "finetuning": self._finetuning,
            "finetune_stop_it": self._finetune_stop_it,
        }
```

```python
        self._cmaes_history = c.get("cmaes_history", [])
        self._finetuning = c.get("finetuning", False)
        self._finetune_stop_it = c.get("finetune_stop_it")
```

In `learn`, no generation is evaluated or swapped once the fine-tune has begun (`copt_on_policy_runner.py:323-325`).

```python
            is_morph_update_time = (
                (it + 1) % self._ea_update_interval == 0
            ) and (is_late_start_toggle_time or self._randomise_before_late_start) and not self._finetuning
```

The save moves to the last iteration of every save interval and is made before that iteration's evaluation and design swap, placed before the `# ---- COPT: EA generation update` block (`copt_on_policy_runner.py:404`). The design file of the population on the scene is written at the same point and under the same index. The present write after the swap (`:435-439`) and the inherited save at `it % save_interval == 0` (`:446-447`) are removed.

```python
            if self.log_dir is not None and not self.disable_logs:
                boundary = (it + 1) % self.save_interval == 0
                # Before the evaluation and the swap, so that model_<it>.pt, design_parameters_<it>.csv
                # and cmaes_history row <it> all describe the population trained up to iteration it.
                if self.current_population is not None and (is_morph_update_time or (boundary and self._finetuning)):
                    pd.DataFrame(self.current_population.get_parameters()).to_csv(
                        os.path.join(self.log_dir, "designs", f"design_parameters_{it}.csv"),
                        index_label="individual",
                    )
                if boundary:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
```

With a save interval of 400 and a generation interval of 400, the iteration `it = 400 g - 1` then yields `model_<it>.pt`, `design_parameters_<it>.csv` and history row `<it>`, all of the population trained from iteration `400 (g - 1)` to `it` and evaluated at its end. This is the pairing `play-copt` relies on, a checkpoint replayed on the best design of its own generation (`scripts/rsl_rl/play.py:334-365`). A resume from such a checkpoint repeats one iteration of training and then evaluates and tells the population the checkpoint holds, as section 2.3 requires. The untrained `model_0.pt` is no longer written.

The termination is detected right after the update that produced it (`copt_on_policy_runner.py:416-419`). By then `_update_morphology` has told CMA-ES the fitness, logged the terminating generation to `cmaes_history.csv`, whose `terminated` and `stop` columns name the criterion, and applied the population of the mean.

```python
                    self._update_morphology(it + 1, respawn=late_start_toggled)
                    if (self._finetune_iterations is not None and not self._finetuning
                            and getattr(self._design_generator, "_terminated", False)):
                        self._finetuning = True
                        self._finetune_stop_it = it + self._finetune_iterations
                        print(f"CMA-ES terminated at iteration {it}, fine-tuning on the final design "
                              f"until iteration {self._finetune_stop_it}")
```

The loop ends after the last iteration of the fine-tune, placed at the end of its body after the code state is stored (`copt_on_policy_runner.py:452-458`).

```python
            if self._finetuning and it >= self._finetune_stop_it:
                print(f"Fine-tuning complete at iteration {it}")
                break
```

A termination at the swap of iteration `k` makes iterations `k + 1` to `k + 400` the fine-tune, 400 iterations. The aligned save writes `model_<k + 400>.pt` and its design file at the last of them, and the final save after the loop writes the same checkpoint again with the same state (`copt_on_policy_runner.py:460-465`). `log_optimised_designs` follows it as today.

### 4.5 E5, the final design in `log_optimised_designs`

`copt_on_policy_runner.py:1062-1141`, updated in place. The method keeps its evaluation of the population on the scene and its table of individuals, and heads its record with the final design, the distribution mean of `optimised_design()`. After a fine-tune every slot carries that design, so the evaluation of the population is its evaluation under the fine-tuned policy, averaged over every environment. Without a fine-tune the final design is recorded unevaluated, as the distribution mean is today. The `distribution_mean/<parameter>` rows and the separate emission block of the present method become the `final design/<key>` rows.

```python
    def log_optimised_designs(self, width: int = 80, pad: int = 35) -> None:
        """Evaluate the population on the scene once more and write the final design to designs/optimised.

        The final design is the one the search recommends, optimised_design(), the distribution
        mean, and it is the design every later evaluation of the run uses. It heads
        final_results.txt, it is the single row of final_design.csv, and its MJCF is
        distribution_mean.xml. After a fine-tune every individual slot carries it, so the
        evaluation of the population on the scene is its evaluation under the fine-tuned policy,
        averaged over every environment. Without a fine-tune the population on the scene is the
        last generation sampled, which the evaluation ranks, and the final design is recorded
        unevaluated. optimised_designs.csv and individual_<k>.xml record the population on the
        scene in either case, individual k having run in environments k + N * j.
        """
        if not self._copt_started or self.current_population is None:
            return
        population = self.current_population
        with torch.inference_mode():
            evaluation = self._evaluate_population()
        fitness = evaluation["fitness"].cpu()
        order = fitness.argsort(descending=True).tolist()
        best = order[0]
        parameters = population.get_parameters()
        if len(parameters) != len(fitness):
            parameters = [{} for _ in range(len(fitness))]
        write = self.log_dir is not None and not self.disable_logs
        out = os.path.join(self.log_dir, "designs", "optimised") if write else None
        if write:
            os.makedirs(out, exist_ok=True)

        final = {}
        optimised = self._design_generator.optimised_design()
        if optimised is not None:
            values, picks = optimised
            feasible = (self._design_generator.emit_design(values, picks, os.path.join(out, "distribution_mean.xml"))
                        if write else None)
            # The terminating generation is the last history row once the fine-tune has begun.
            terminal = self._cmaes_history[-1] if self._finetuning and self._cmaes_history else {}
            on_scene = self._finetuning
            final = {
                "source": "distribution_mean",
                **{name: float(v) for name, v in values.items()},
                **(picks or {}),
                "feasible": feasible,
                "fine_tuned": on_scene,
                "fitness": fitness.mean().item() if on_scene else None,
                "fitness_std_over_slots": fitness.std().item() if on_scene else None,
                **{f"eval_{key}": (evaluation[key].mean().item() if on_scene else None)
                   for key in ("survival", "lin_vel_error", "yaw_rate_error")},
                "termination_iteration": terminal.get("iteration"),
                "termination_generation": terminal.get("generation"),
                "stop": terminal.get("stop"),
                "finetune_start_iteration": (self._finetune_stop_it - self._finetune_iterations
                                             if on_scene else None),
                "checkpoint_iteration": self.current_learning_iteration,
            }

        rows = [(f"final design/{key}", value) for key, value in final.items()]
        rows += [
            ("generation", self.generation - 1),
            ("learning iteration", self.current_learning_iteration),
            ("best individual", best),
            ("best fitness", fitness[best].item()),
        ]
        rows += [(f"best/{name}", value) for name, value in parameters[best].items()]
        rows += [
            ("fitness_mean", fitness.mean().item()),
            ("fitness_std", fitness.std().item()),
            ("split_half_rank_correlation", evaluation["split_half_rank_correlation"]),
        ]
        for key in ("survival", "lin_vel_error", "yaw_rate_error"):
            rows.append((f"eval/{key}", evaluation[key].mean().item()))
        for r, value in evaluation["row_return"].items():
            rows.append((f"eval/return_row_{r}", value))
        for key, value in (self._design_generator.search_statistics() or {}).items():
            if isinstance(value, dict):
                rows += [(f"{key}/{name}", v) for name, v in value.items()]
            else:
                rows.append((key, value))
        rows.append(("evaluation time", evaluation["time"]))

        title = " Optimised population "
        text = f"""{"#" * width}\n{title.center(width, " ")}\n\n"""
        for key, value in rows:
            shown = f"{value:.4f}" if isinstance(value, float) else f"{value}"
            text += f"""{f"{key}:":>{pad}} {shown}\n"""
        text += f"""\n{"Ranking, individual and fitness":>{pad}}\n"""
        for k in order:
            text += f"""{f"{k}:":>{pad}} {fitness[k].item():.4f}\n"""
        text += f"""{"-" * width}\n"""
        print(text, flush=True)
        if not write:
            return

        for k, path in enumerate(population.get_xml_files()):
            shutil.copy(path, os.path.join(out, f"individual_{k:04d}.xml"))
        table = pd.DataFrame(parameters)
        table["fitness"] = fitness.tolist()
        for key in ("survival", "lin_vel_error", "yaw_rate_error"):
            table[f"eval_{key}"] = evaluation[key].tolist()
        table["rank"] = fitness.argsort(descending=True).argsort().tolist()
        table.to_csv(os.path.join(out, "optimised_designs.csv"), index_label="individual")
        if final:
            pd.DataFrame([final]).to_csv(os.path.join(out, "final_design.csv"), index=False)
        with open(os.path.join(out, "final_results.txt"), "w") as fh:
            fh.write(text)
```

The terminating generation's own population, with every individual's fitness, remains in `designs/cmaes_generation_<g>.csv` (`copt_on_policy_runner.py:1043-1050`), so the record of the search is complete beside the record of the final design. A generator without a search, `RandomDesignGenerator`, returns `None` from `optimised_design()` (`usd_generator.py:655-657`), and the method then writes the population alone as before.

### 4.6 E6, `play-copt` replays the final design

`_best_design` in `scripts/rsl_rl/play.py:334-365` reads `final_design.csv` for any checkpoint taken during the fine-tune, which E5 records with `fine_tuned` and `finetune_start_iteration`. A checkpoint at or before the termination is replayed on the best design of its own generation, since its policy never trained on the mean. A run without a fine-tune, which writes no fine-tuned final design, is read exactly as before.

```python
    final = os.path.join(log_dir, "designs", "optimised", "final_design.csv")
    if os.path.exists(final):
        row = pd.read_csv(final).iloc[0]
        if bool(row.get("fine_tuned", False)) and target > int(row["finetune_start_iteration"]):
            parameters = {k: float(row[k]) for k in CMAES_PARAM_KEYS if k in row.index}
            return parameters, {
                "checkpoint_iteration": target,
                "history_iteration": int(row["termination_iteration"]),
                "generation": int(row["termination_generation"]),
                "source": "designs/optimised/final_design.csv, the fine-tuned distribution mean",
                "parameters": parameters,
            }
```

`CMAES_PARAM_KEYS` joins the import from `co_optimisation.runners.usd_generator` at `play.py:111-116`.

### 4.7 E7, a recorder that the evaluation suspends

`PausableRecordVideo` joins `HighestTerrainCameraWrapper` in `scripts/rsl_rl/highest_terrain_camera_wrapper.py`, the module of the wrappers `train.py` places around the recorder, whose docstring gains a sentence naming it.

```python
class PausableRecordVideo(gym.wrappers.RecordVideo):
    """gym.wrappers.RecordVideo with a paused flag, and RecordVideo exactly while unpaused, the default.

    Paused, a step passes straight to the wrapped environment, rendering no frame and leaving
    step_id where it was, so that the step trigger stays aligned with the steps of training alone
    and a clip in progress resumes at the next unpaused step. CoptOnPolicyRunner pauses it through
    every controlled evaluation. Written against gymnasium 1.2.1, whose RecordVideo.step advances
    step_id and captures a frame after stepping (rendering.py:359-374).
    """

    paused: bool = False

    def step(self, action):
        if self.paused:
            return self.env.step(action)
        return super().step(action)
```

`train.py` imports it beside `HighestTerrainCameraWrapper` (`scripts/rsl_rl/train.py:125`) and uses it in place of `gym.wrappers.RecordVideo` (`:230`). Every task's recording is unchanged by it, since the class behaves as its parent until something sets the flag.

```python
from highest_terrain_camera_wrapper import HighestTerrainCameraWrapper, PausableRecordVideo
...
        env = PausableRecordVideo(env, **video_kwargs)
```

The runner pauses every layer of its wrapper chain that carries the flag. Gymnasium 1.2.1 wrappers do not forward attribute lookups, exposing only `get_wrapper_attr` (`gymnasium/core.py:412`), so `hasattr` finds the flag on the recorder alone, and the chain is walked through the `env` attribute that `RslRlVecEnvWrapper` holds (`IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py:48`).

```python
import contextlib
...
    @contextlib.contextmanager
    def _video_paused(self):
        """Pause every wrapper of the chain that carries a paused flag, for the duration of the block."""
        recorders, layer = [], self.env
        while layer is not None:
            if hasattr(layer, "paused"):
                recorders.append(layer)
            layer = getattr(layer, "env", None)
        for recorder in recorders:
            recorder.paused = True
        try:
            yield
        finally:
            for recorder in recorders:
                recorder.paused = False
```

`_evaluate_population` runs its passes inside the block (`copt_on_policy_runner.py:892-893`), which covers every evaluation, those of the generations and the final one of `log_optimised_designs` alike.

```python
        try:
            with self._video_paused():
                for r, q in itertools.product(range(rows), range(passes)):
                    # ... the body of the loop unchanged, indented one level
        finally:
```

### 4.8 E8, `train.py` builds the run from the configuration class

Five arguments are retired, `--design_search`, `--copt_num_individuals`, `--ea_update_interval`, `--ea_late_start` and `--shank_foot_gap` (`scripts/rsl_rl/train.py:45-100`), their values now living in the configuration class. The import of the generators gains the configuration.

```python
from co_optimisation.config import CMAES_CFG_BY_FAMILY, CMAESCfg, without_termination
```

A module function resolves the configuration of the run, from the run being resumed where there is one, so that a restored experiment continues under exactly the configuration it began with.

```python
def _cmaes_cfg(family: str, agent_cfg, log_root_path: str) -> CMAESCfg:
    """The run's co-optimisation configuration, restored from params/cmaes.pkl of a resumed run.

    A resumed run begun before params/cmaes.pkl existed continues as it was launched, with the
    robot's configuration less the tolerances and the fine-tune, which those runs never had.
    """
    if agent_cfg.resume:
        resume_path = args_cli.checkpoint_path or get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
        saved = os.path.join(os.path.dirname(resume_path), "params", "cmaes.pkl")
        if os.path.exists(saved):
            with open(saved, "rb") as f:
                return pickle.load(f)
        print(f"[WARNING] No params/cmaes.pkl beside {resume_path}, resuming without CMA-ES tolerances or fine-tune.")
        return without_termination(CMAES_CFG_BY_FAMILY[family]())
    return CMAES_CFG_BY_FAMILY[family]()
```

The co-optimisation branch reads every setting from it (`scripts/rsl_rl/train.py:268-333`).

```python
        cmaes_cfg = _cmaes_cfg(_family, agent_cfg, log_root_path)
        _num_individuals = cmaes_cfg.num_individuals
        # ea_update_interval * num_steps_per_env (120 * 25 = 3000) should be more
        # than episode_duration / (dt * decimation) (20 / (0.005 * 4) = 1000)
        ea_update_interval = cmaes_cfg.ea_update_interval
        ea_late_start = cmaes_cfg.ea_late_start
        generator_kwargs = dict(
            family=_family,
            num_individuals=_num_individuals,
            param_ranges=dict(cmaes_cfg.param_ranges),
            output_dir=os.path.join(log_dir, "copt_usds"),
        )
        late_start_pop_size = env.num_envs // cmaes_cfg.late_start_prior_divisor
        tolerances = dict(tolx=cmaes_cfg.tolx, tolfun=cmaes_cfg.tolfun, tolfunhist=cmaes_cfg.tolfunhist)
        terminates = any(v is not None for v in tolerances.values()) or cmaes_cfg.finetune_iterations is not None
        if cmaes_cfg.design_search == "random":
            # No late start, so the runner treats every generation as a search generation.
            ea_late_start = -1
            design_generator = RandomDesignGenerator(**generator_kwargs)
            population_sizes = (_num_individuals,)
        else:
            if cmaes_cfg.design_search == "cat" and terminates:
                raise ValueError("CatCMAwM never terminates, so the tolerances and the fine-tune apply only to "
                                 "design_search 'length' and 'cmaes', see plans/copt_training_refinement.md section 7")
            generator_cls = {
                "length": GrowingDesignDistCMAESDesignGenerator,
                "cmaes": CMAESDesignGenerator,
                "cat": CatCMAESDesignGenerator,
            }[cmaes_cfg.design_search]
            termination = {} if cmaes_cfg.design_search == "cat" else dict(
                **tolerances,
                final_population="mean" if cmaes_cfg.finetune_iterations is not None else "last",
            )
            design_generator = generator_cls(
                **generator_kwargs,
                sigma0=cmaes_cfg.sigma0,
                seed=cmaes_cfg.seed,
                late_start=True,
                late_start_it=int(ea_late_start / ea_update_interval),
                late_start_prior_pop_size=late_start_pop_size,
                max_cma_iter=(cmaes_cfg.max_cma_iter if cmaes_cfg.max_cma_iter is not None
                              else (agent_cfg.max_iterations - ea_late_start) / ea_update_interval),
                **termination,
            )
            population_sizes = (late_start_pop_size, _num_individuals)
        if cmaes_cfg.shank_foot_gap is not None:
            design_generator.builder_kwargs = {"shank_foot_gap": cmaes_cfg.shank_foot_gap}
        # ... the population size check unchanged
        agent_cfg_dict["copt"] = {
            "ea_update_interval": ea_update_interval,
            "ea_late_start": ea_late_start,
            "num_individuals": _num_individuals,
            "randomise_before_late_start": cmaes_cfg.randomise_before_late_start,
            "reinit_lr_each_generation": cmaes_cfg.reinit_lr_each_generation,
            "eval_duration_s": cmaes_cfg.eval_duration_s,
            "eval_seed": cmaes_cfg.eval_seed,
            # The tolerances are read by the generator, recorded here so that copt_agent.yaml names them.
            "tolx": cmaes_cfg.tolx,
            "tolfun": cmaes_cfg.tolfun,
            "tolfunhist": cmaes_cfg.tolfunhist,
            "finetune_iterations": cmaes_cfg.finetune_iterations,
        }
        if cmaes_cfg.shank_foot_gap is not None:
            # Read by nothing in the runner, recorded so that copt_agent.yaml names the geometry.
            agent_cfg_dict["copt"]["shank_foot_gap"] = cmaes_cfg.shank_foot_gap
```

The configuration is dumped beside the agent and environment configurations, in the same two forms (`scripts/rsl_rl/train.py:367-373`).

```python
    if args_cli.policy_type in _COPT_ACTOR_CRITIC:
        # agent.yaml serialises the configclass, without the copt block the runner receives.
        dump_yaml(os.path.join(log_dir, "params", "copt_agent.yaml"), agent_cfg_dict)
        dump_yaml(os.path.join(log_dir, "params", "cmaes.yaml"), cmaes_cfg)
        dump_pickle(os.path.join(log_dir, "params", "cmaes.pkl"), cmaes_cfg)
```

The `random` search sets the late start to `-1` locally, as today, so `params/cmaes.yaml` records the configured late start and `copt_agent.yaml` the one the runner received.

### 4.9 E9, the launcher in `/ws/djinn`

The four co-optimisation arms no longer pass search arguments, the configuration class supplying them, and they set a save interval of 400. A save interval variable, 1000 for every other mode, replaces the literal of the two launch lines (`/ws/djinn:216`, `:218`), and the `copt_params` variable and its use in those lines are removed (`:118`, `:135`, `:139`, `:154`, `:158`).

```bash
                save_interval=1000
                ...
                elif [[ "$3" == "copt" ]]; then
                    task="Isaac-Limx-SF-Copt-Rough-v0"
                    policy_type="COPT"
                    save_interval=400
                elif [[ "$3" == "copt-moral" ]]; then
                    task="Isaac-Limx-SF-Copt-MoRAL-Rough-v0"
                    policy_type="COPT-MORAL"
                    save_interval=400
                ...
                elif [[ "$3" == "quadruped-copt" ]]; then
                    task="Isaac-Quadruped-Copt-Rough-v0"
                    policy_type="COPT"
                    save_interval=400
                elif [[ "$3" == "quadruped-copt-moral" ]]; then
                    task="Isaac-Quadruped-Copt-MoRAL-Rough-v0"
                    policy_type="COPT-MORAL"
                    save_interval=400
                ...
                    iexec "$container" "./isaaclab.sh -p biped/scripts/rsl_rl/train.py --task $task --video --video_length 2000 --video_interval 10000 --save_interval $save_interval --seed 24 --headless --device $device --policy-type $policy_type --checkpoint_path $checkpoint_path --resume true --load_run $run_path"
                else
                    iexec "$container" "./isaaclab.sh -p biped/scripts/rsl_rl/train.py --task $task --video --video_length 2000 --video_interval 10000 --save_interval $save_interval --seed 24 --headless --device $device --policy-type $policy_type"
```

## 5. Order of Application

1. Apply E1 and E2 to `usd_generator.py` and E3 to the new `config` package, and run the host checks 1 to 3 of section 8, which need no simulator.
2. Apply E4, E5 and the runner half of E7 to `copt_on_policy_runner.py`, then the wrapper half of E7.
3. Apply E6 to `play.py`, E8 to `train.py` and E9 to `/ws/djinn`.
4. Reinstall `co_optimisation` in both containers, since `djinn start train` installs it editable before each run (`/ws/djinn:213`).
5. Run the smoke test of section 8, then record the outcome. Mark this plan's status, add a dated entry to `/ws/context/cmaes.md` and `/ws/context/copt.md`, and update `/ws/ARCHITECTURE.md` and `/ws/CO_OPTIMISATION.md`, which describe the retired arguments (`/ws/ARCHITECTURE.md:639-641`, `:686`, `:693`, `/ws/CO_OPTIMISATION.md:2380`, `:2393-2395`) and whose descriptions of the training modes and the co-optimisation configuration this changes.

## 6. Callers and Backwards Compatibility

| Edit | Shared symbol | Callers found | Effect |
|---|---|---|---|
| E1 | `DesignGeneratorBase._build_population` | `RandomDesignGenerator.generate_population` (`usd_generator.py:817`), `CMAESDesignGenerator.generate_population` (`:963`, `:965`), `FixedDesignGenerator` (`scripts/rsl_rl/play.py:326`) | none, `individual` defaults to `_generate_individual` |
| E2 | `CMAESDesignGenerator.__init__`, `_make_strategy`, `generate_population` | `train.py:295` for `length`, `cmaes` and `cat`, `play.py:1290`, `scripts/analysis/nan_reset_diagnostic.py:123`, `co_optimisation/tests/sim_nominal_pose_check.py:170`, `co_optimisation/tests/design_generator_test.py:514`, `:523`, `:641`, `:658` | none for a caller that sets nothing, the tolerances defaulting to pycma's and `"last"` returning `None` as before |
| E4 | `CoptOnPolicyRunner` saving and fine-tune | `train.py:336`, `play.py:1308`, `nan_reset_diagnostic.py:153` | the save before each swap and the aligned design file apply to every co-optimisation training run, by the user's direction. `play.py` never calls `learn` and `nan_reset_diagnostic.py` passes no log directory, so neither saves. The fine-tune applies only where the `copt` block names it |
| E5 | `CoptOnPolicyRunner.log_optimised_designs` | `learn` (`copt_on_policy_runner.py:467`) | the record gains `final_design.csv` and its `final design` rows replace the `distribution_mean` rows, the population table and documents being unchanged |
| E7 | `PausableRecordVideo`, `_video_paused` | `train.py` for every task, `_evaluate_population` | the training video of every co-optimisation run omits its evaluations, by the user's direction. Every other task's recording is unchanged, the flag never being set |
| E8, E9 | `train.py` arguments, `/ws/djinn start train` | `djinn`, the only launcher passing the retired arguments | the four co-optimisation modes take their settings from the configuration class and save every 400 iterations. Every other mode is unchanged. A run resumed from a checkpoint written before this plan continues without the tolerances or the fine-tune |

`GrowingDesignDistCMAESDesignGenerator` inherits E2 unchanged, and its own `_generate_individual` is untouched. `CatCMAESDesignGenerator` overrides `_make_strategy` (`usd_generator.py:1093-1119`), so the tolerances never reach its strategy, and E8 refuses them for it rather than ignore them. The defaults of the retired arguments, 64 individuals, an interval of 200 and a late start at 3000 (`scripts/rsl_rl/train.py:59-100`), were never used by a co-optimisation launch, which `djinn` always made with 80, 400 and 6000, the values `CMAESCfg` carries.

## 7. Defects Left Standing

1. `CatCMAESDesignGenerator.update_with_fitness` never sets `_terminated` (`usd_generator.py:1173-1182`), so the categorical search runs to the end of training whatever its state, and `search_statistics` merely reports `should_stop` (`:1154`). A categorical search that should terminate would set `_terminated` from `self._es.should_stop()` after `tell`, and would need tolerances of its own derived from a categorical run, none of the four runs here being one. Resolved 2026-11-12 by `design_filtering.md` section 11.7, the categorical generators stopping on `maxiter`, `tolfun`, `tolfunhist` and `tolx` together with a new `tolcat`, rather than on `should_stop`, which tests the Gaussian alone against fixed tolerances. Its `tolcat` of 0.05 is provisional, a categorical run from which to derive it still being wanting.
2. Under `final_population="last"`, a terminated search keeps re-applying the last population (`usd_generator.py:955-959`, `copt_on_policy_runner.py:730-733`), and without `finetune_iterations` the run trains on to `max_iterations` and evaluates a population that no longer changes at every boundary. Both remain the behaviour of `CMAESCfg` itself and of any run resumed from a checkpoint written before this plan.
3. Both quadruped runs end with the shank scale and the two pose offsets on their lower bounds (section 3.7). The box may be cutting off the optimum, and widening it belongs to a separate plan.
4. The tolerances are tied to the present evaluation protocol and search space (section 3.9) and must be re-derived when either changes.

## 8. Verification

1. On the host, with pycma 4.4.4 installed to a scratch directory, construct `CMAESDesignGenerator` for each family with and without tolerances, and confirm that `_es.opts` carries `1e-11`, `1e-11` and `1e-12` by default and the configured values otherwise.
2. Rerun `../context/artefacts/copt_training_refinement-2026-10-28/verify_stop.py` over the four runs and confirm the table of section 3.6.
3. Call `generate_population` on a generator whose `_terminated` is set, once under `"last"`, confirming `None`, and once under `"mean"`, confirming 80 records with identical parameters equal to `optimised_design()` and a single emitted document under `gen_<g>/`. Construct `BipedCMAESCfg` and `QuadrupedCMAESCfg`, dump and pickle them as `train.py` does, confirm that the pickle loads to an equal configuration, and confirm that `without_termination` returns a copy whose four fields are `None` while the original keeps its values.
4. Smoke test in the container on both robots, with a `CMAESCfg` subclass of `ea_late_start` 400, `ea_update_interval` 40, `max_cma_iter` 20, `tolx` 0.3 and `finetune_iterations` 40, under `--save_interval 40 --max_iterations 2000`, so that the search terminates within a few generations. Confirm, in order, that `params/cmaes.yaml` and `params/cmaes.pkl` are written, that `cmaes_history.csv` closes with `terminated` 1 and a non-empty `stop`, that the terminal output announces the fine-tune and its last iteration, that training ends 40 iterations after the termination, that `model_<it>.pt` and `design_parameters_<it>.csv` exist for every boundary `it` and for no iteration outside them, and that `designs/optimised/final_design.csv` holds one row with `fine_tuned` true and parameters equal to those of `distribution_mean.xml`.
5. In the same smoke run, confirm that every clip under `videos/train/` starts at a multiple of `--video_interval` in the count of PPO steps alone, by comparing the clip names against `25 * (iteration + 1)` at each swap.
6. Resume the smoke run from the checkpoint of the boundary before the termination and confirm that it restores the configuration from `params/cmaes.pkl` and produces the same termination row, then resume it from a checkpoint inside the fine-tune and confirm that it ends at the same iteration. Resume a run begun before this plan and confirm the warning and the absence of the tolerances from its strategy.
7. Run `djinn start play-copt` on the smoke run's final checkpoint and confirm that it reports `designs/optimised/final_design.csv` as its source, and on the checkpoint of the termination and confirm that it reports the history row of that iteration.

## 9. Artefacts

The analysis of section 3 is reproduced by the scripts and outputs in `../context/artefacts/copt_training_refinement-2026-10-28/`, which read the four runs by their paths under `/ws/IsaacLab/logs/rsl_rl/`. `snapshot.txt` records when they were read and how many generations and checkpoints each run then held. `analyse.py` and `analyse.txt` reconstruct the quantities of section 3.1 from each `cmaes_history.csv` and generation table, `grid.py` and `grid.txt` give the binding coordinates, the distinct emitted leg lengths and the generation at which each candidate tolerance is first met, `es_state.py` and `es_state.txt` read the pickled strategy from every checkpoint without torch, `verify_stop.py` and `verify_stop.txt` give pycma's verdict under the tolerances of section 3.9, and `after_stop.py` and `after_stop.txt` measure what the search did after the termination. They need pycma 4.4.4, numpy and pandas on the path, and no simulator. The simulated constants 4.85, 0.44 and 1.39 of sections 3.3 and 3.4 come from 200000 sets of 80 standard normal draws with generator seed 0.

## 10. Bibliography

1. Hansen, N., Akimoto, Y., Baudis, P. CMA-ES/pycma on GitHub, package `cma` version 4.4.4, 2024. https://github.com/CMA-ES/pycma. Registered in cluster 33 of `/ws/context/literature.md`. The termination criteria, their defaults and the history window, read in `cma/evolution_strategy.py` and `cma/options_parameters.py` of the 4.4.4 wheel on 2026-10-28.
2. Hansen, N. The CMA Evolution Strategy, A Tutorial. arXiv preprint, 2016. arXiv:1604.00772. Registered in cluster 33 of `/ws/context/literature.md`. The source of the sampling distribution whose per coordinate standard deviation `tolx` tests.
3. CMA-ES source code page. https://cma-es.github.io/cmaes_sourcecode_page.html. Author and year not stated by the page retrieved on 2026-10-26. Registered in cluster 40 of `/ws/context/literature.md` and cited as entry 5 of `/ws/context/cmaes.md`. The advice that a variable consistently ending on the same bound be reconsidered.
4. Hansen, N., Niederberger, A. S. P., Guzzella, L., Koumoutsakos, P. A method for handling uncertainty in evolutionary optimization with an application to feedback control of combustion. IEEE Transactions on Evolutionary Computation 13(1), 180-197, 2009. DOI 10.1109/TEVC.2008.924423. Registered in cluster 36 of `/ws/context/literature.md`. The uncertainty handling that pycma ships as `NoiseHandler`, the published remedy should the noise of section 3.2 come to prevent termination.
5. Nomura, M., Akimoto, Y., Ono, I. CMA-ES with Learning Rate Adaptation, Can CMA-ES with Default Population Size Solve Multimodal and Noisy Problems? Genetic and Evolutionary Computation Conference (GECCO) 2023. DOI 10.1145/3583131.3590358. arXiv:2304.03473. Registered in cluster 38 of `/ws/context/literature.md`. The learning rate adaptation that holds the update's signal to noise ratio constant, the second published remedy for a noisy fitness.
