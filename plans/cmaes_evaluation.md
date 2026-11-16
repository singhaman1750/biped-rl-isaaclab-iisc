# A Controlled Evaluation of Design Fitness for CMA-ES, and the Logging of Its Generations

> Status, IMPLEMENTED on 2026-10-22, the five row passes of edit E6 following on 2026-10-23. Edits E1 to E9 and L1 to L4 were applied to the working tree on 2026-10-22 with the evaluation then placing each instance in a single row per generation, and the five pass form of E6 that section 3 now specifies was applied on 2026-10-23 through `copt_improvement.md` section 3. Two additions the user asked for during application were made beside the edits, a graceful continuation once CMA-ES terminates and a final export of the optimised population to `designs/optimised/`. Section 5 records the application, its departures from the text, and what the smoke tests did and did not exercise. Written against commit `23513d2` and the uncommitted working tree of that date. See [README.md](README.md) for the register.

## 1. Introduction

The co-optimisation runner of this repository alternates proximal policy optimisation over a population of robot designs with a CMA-ES update of the distribution that samples them. CMA-ES needs one number per design, and the quality of the search depends on whether that number orders the designs as their true merit would. Morphology and control are coupled, so a design can only be judged through a controller adapted to it [1], and the question this plan answers is under what conditions that judgement should be made.

The scene is laid out for the comparison. The `djinn` launch of both robots runs 80 designs (`/ws/djinn:135`, `:150`) over the 6400 environments of the co-optimisation scene (`environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:1388`, `environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:1579`), and environment `e` carries design `e % 80` (`co_optimisation/co_optimisation/utils/respawn.py:53-54`). Isaac Lab places environment `e` in terrain column `floor(e / (num_envs / num_cols))` (`IsaacLab/source/isaaclab/isaaclab/terrains/terrain_importer.py:348-350`), and the rough terrain of both robots has 80 columns, each a separately generated terrain sample, and 5 difficulty rows (`environments/environments/tasks/locomotion/cfg/SF/terrains_cfg.py:204-209`, `environments/environments/tasks/locomotion/cfg/quadruped/terrains_cfg.py:16-20`). Each design therefore holds 80 instances, instance `j` standing in column `j`, so every design meets every terrain column exactly once. The schedule is laid out for the comparison as well. CMA-ES begins only at the late start (`/ws/djinn:135`, `scripts/rsl_rl/train.py:84-89`), after a random design phase in which the curricula and the policy's performance on random designs have settled, so that CMA-ES sees a constant environment and curriculum state. In the run `copt_moral/2026-10-21_07-16-23` the forward, lateral and yaw command ranges and the push magnitude had all reached their configured limits by iteration 2000 and held them thereafter ([../context/artefacts/copt_moral-2026-10-21_07-16-23-fitness-evaluation/timing_summary.csv](../context/artefacts/copt_moral-2026-10-21_07-16-23-fitness-evaluation/timing_summary.csv)).

What remains is the measurement itself. The runner derives fitness from the training rollouts, adding the return of every completed episode to the account of the design its environment carries (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:369-384`) and passing the mean of those returns to `update_with_fitness` (`copt_on_policy_runner.py:801-810`, `:748-750`). The training rollouts were not built to compare designs, and six of their properties make the returns incomparable across the designs of one generation even under the layout and the schedule above.

1. Terrain difficulty differs by instance and is driven by performance. The terrain curriculum promotes an environment that walks far and demotes one that does not (`environments/environments/tasks/locomotion/mdp/curriculums.py:461-506`, `terrain_importer.py:314-329`), so the instances of a capable design climb to harder rows and are then scored on harder ground. The curriculum taxes the competence it rewards. The column of an instance is fixed by its index, its row is not, and the run above held a mean terrain level of 2.06 of 5 through its CMA-ES phase with every environment free to move ([timing_summary.csv](../context/artefacts/copt_moral-2026-10-21_07-16-23-fitness-evaluation/timing_summary.csv)).
2. Commands differ by instance. Each environment draws its own velocity command and heading, resamples them every 7.5 to 12.5 s, and derives its yaw rate command from its own heading error (`environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:113-127`, `IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py`), so the yaw command depends on the robot's own trajectory.
3. Initial states differ by instance. The reset events draw a planar offset of 0.5 m, a full turn of yaw, a base velocity of up to 0.5 in every axis and a joint offset of 0.2 rad (`limx_base_env_cfg.py:838-860`).
4. The actions are exploratory. The rollouts sample the policy distribution (`copt_on_policy_runner.py:333`), so the return of a design includes the action noise of the learner, which is not the behaviour a deployed design would show.
5. The amount of evidence differs by design. The number of completed episodes varies with the fall rate, short episodes are dropped below a threshold (`copt_on_policy_runner.py:372-379`), and the episodes still running at the swap are discarded (`:366-368`).
6. Pushes arrive at random times in each environment (`limx_base_env_cfg.py:879-890`).

The literature recognises the first of these explicitly. The authors of the terrain curriculum that Isaac Lab implements compared total episode rewards only after removing the curriculum, so that difficulty did not change underneath the comparison [2]. CMA-ES consumes only the ranking of a generation's candidates and is invariant to any order-preserving transformation of the objective [3], which fixes the requirement precisely. Within a generation every design must face the same conditions, so that a difference in fitness is a difference in the design.

This plan therefore retires the training-derived fitness entirely and makes a short controlled evaluation, run at each generation boundary of the CMA-ES phase, the only source of fitness. It meets six objectives. Every design receives the same commands, which span the commanded velocity space. Every terrain cell holds exactly one instance of every design and nothing else, and every cell of the grid, every column in every difficulty row, is visited by every design in every generation, one pass per row. The fitness is comparable across the designs of a generation. The evaluation costs as little simulated time as the coverage permits. And the training accumulators that fed the old fitness are removed rather than kept beside it. Section 2 surveys how published design and co-design work evaluates its candidates, along the four axes of command variation, environment difficulty, aggregation and data volume, and determines which CMA-ES specific methods bear on this work. Section 3 sets out the evaluation architecture and its implementation. Section 4 adds a TensorBoard head and a set of readable design logs for the CMA-ES generations, replacing the unreadable design parameter files the runner writes today.

## 2. Literature Survey

The survey proceeds from the co-design works whose evaluation protocol could be verified in their own text, through the locomotion benchmark whose terrain grid this repository inherits, to the methods that the evolutionary computation literature offers for noisy and stochastic fitness. Each subsection records what the work varies during evaluation, how it aggregates, and how much data it spends, and closes on its bearing on this plan. Section 2.17 gathers the comparison into a table and states the position the plan takes.

### 2.1 Ha, Reinforcement Learning for Improving Agent Design

Ha optimised the body parameters of the agent jointly with its policy by a population-based policy gradient over a factored Gaussian, with a population of 192 per generation [4]. Each candidate ran the task 16 times with different random seeds, and the average of those 16 returns was the fitness used for the update [4]. The tasks were single locomotion tasks without commands, among them BipedalWalkerHardcore, whose obstacles vary with the seed [4]. The final solution was judged on a separate average over 100 rollouts [4]. The work establishes repeated evaluation under randomised conditions as the means of controlling fitness noise, which the 400 cells each design meets provide here in parallel, and it separates the search signal from the final judgement, a separation the plan preserves by logging the distribution mean beside the best individual.

### 2.2 Schaff and colleagues, Jointly Learning to Construct and Control Agents

Schaff and colleagues maintained a Gaussian mixture over designs, trained a design-conditioned policy with proximal policy optimisation on eight sampled designs per iteration, and moved the mixture along the score function weighted by the average episode return of each design [5]. That return was taken from the training trajectories themselves [5], which is the scheme the runner implements today. Environments were flat ground and a five degree incline, each a separate experiment, and the final design was evaluated by the average reward over 100 episodes with the policy fine-tuned on that design alone [5]. The work is the direct precedent for the present fitness and shows its limit. In a single environment without a curriculum the training return is a defensible signal, but the confounds of section 1 do not arise there, and the work offers no protocol for a population spread across a terrain grid.

### 2.3 Luck and colleagues, Data-efficient Co-Adaptation of Morphology and Behaviour

Luck and colleagues scored a candidate design by the critic's estimate of the return, averaged over a batch of start states drawn from a buffer of all start states seen [6]. Their appendix reports that averaging over several random start states increases the performance of the method considerably [6]. The simulation baseline they compare against spent 300 episodes on each of the first five designs and 100 on each thereafter, and they found no significant difference between particle swarm optimisation and CMA-ES as the design optimiser [6]. The finding on start states bears directly on section 3, which removes the per-environment start state randomisation and gives every design the same start states, so that the variation the authors averaged out is instead held fixed across the designs being ranked.

### 2.4 Gupta and colleagues, Embodied Intelligence via Learning and Evolution

The DERL framework trained each morphology from scratch for five million environment interactions and took as fitness the average final reward over roughly the last one hundred thousand [7]. In its variable terrain task every episode generated a new course of hills, steps and rubble interleaved with flat segments, and within an evolutionary run the same random seed was used for the lifetime learning of every morphology [7]. Selection compared only the forward progress component of the reward, not the full reward [7]. The work shows two choices this plan weighs. A shared seed gives every candidate the same sequence of conditions, which is the principle of section 3. The restriction of selection to a task component shows that the fitness need not be the training reward, an option section 4 keeps open by logging tracking errors beside the return.

### 2.5 Belmonte-Baeza and colleagues, Meta Reinforcement Learning for Optimal Design of Legged Robots

This is the closest precedent. Belmonte-Baeza and colleagues optimised the thigh and shank length scales of a quadruped, each in the interval 0.6 to 1.4, for 30 generations of 35 designs [8]. Their figure describes the optimiser as a genetic algorithm and their text states that it is implemented with PyCMA, that is, CMA-ES [8]. Each design was evaluated by adapting a meta-trained policy with five proximal policy optimisation updates and then collecting 250 transitions, using 300 parallel simulated environments [8]. The fitness was the negated expected cost per step, the squared velocity tracking error plus joint torque and positive mechanical power weighted by a factor that grows with the tracking error and is clipped at 100 [8]. Evaluation commanded a randomly sampled velocity in each environment, over hilly terrain and discrete steps at several difficulty levels, with a differently generated terrain per environment and randomised friction [8]. The work confirms that velocity tracking across a range of commands and across terrain difficulty is the accepted measure of a legged design. Its random command per environment is, however, independent across designs, so two designs are compared on different command sets. Section 3 keeps the coverage and removes that independence.

### 2.6 Bjelonic and colleagues, Learning-based Design and Control for Quadrupedal Robots with Parallel-Elastic Actuators

Bjelonic and colleagues trained a design-conditioned policy and then selected the parallel spring parameters of the ANYmal knee with Bayesian optimisation [9]. The objective was the average reward per step, estimated over 1000 episodes, for forward walking at 1 m/s over a mixture of stepping stones, flat and rough terrain, with base perturbations of up to 50 N and 50 N m and friction sampled in 0.5 to 2 [9]. The authors state that command amplitude and direction were randomly sampled during design optimisation [9]. The work shows that a reward-aligned fitness, the same average reward the policy was trained on, is accepted practice, and that a sample-efficient optimiser can afford a large evaluation budget per design. A population optimiser evaluating 80 designs per generation in one batch cannot, which is why section 3 seeks variance reduction rather than volume.

### 2.7 Ghansah and colleagues, Humanoid Robot Co-Design via Hybrid Zero Dynamics

Ghansah and colleagues co-designed the thigh and shin of the ADAM humanoid together with a periodic gait in one hybrid zero dynamics optimisation, minimising a weighted mechanical cost of transport [10]. The walking speed was an outcome of the optimisation, bounded only through the step length and duration constraints, rather than a set of commanded operating points [10]. Model-based co-design of this kind evaluates a design at the single gait it optimises. The contrast motivates the command coverage of section 3, since a design scored at one operating point may be poor across the commanded range a learned controller is asked to track.

### 2.8 Zhao and colleagues, RoboGrammar

RoboGrammar searched a grammar of buildable robot topologies for the best performing designs and controllers on a given input terrain, and could optimise for a single terrain or for a combination of terrains [11]. A design optimised for one terrain is specialised to it, and a combination is the means of seeking a generalist. The terrain grid of this repository is such a combination, and section 3 weights its difficulty rows equally rather than by the proportion of environments that happen to occupy them.

### 2.9 Rudin and colleagues, Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning

Rudin and colleagues introduced the terrain curriculum that Isaac Lab implements. Terrains are tiled in a grid whose rows increase in difficulty and whose columns hold the terrain types, a robot that walks past the border of its tile is promoted and one that covers less than half the distance its command requires is demoted, and robots that solve the last level are returned to a random level [2]. Commands were held constant over an episode and resampled only at reset [2]. For evaluation the authors commanded a fixed forward velocity of 0.75 m/s with a small randomised lateral component, measured success rate by terrain type and difficulty, and used total episode reward only after removing the curriculum [2]. The work supplies three elements of section 3. The grid itself is the space of conditions. A command held for the whole evaluation episode follows their protocol. And the removal of the curriculum before reward comparison is the principle on which the whole plan rests.

### 2.10 Hansen, The CMA Evolution Strategy, A Tutorial

The tutorial establishes that CMA-ES selects by rank, so that its behaviour is invariant under any strictly increasing transformation of the objective [3]. Two consequences follow for this plan. Comparability is required only within a generation, so the evaluation conditions may change from one generation to the next without biasing the search, which section 3 exploits to rotate the difficulty rows. And a fitness offset or scale common to all designs is harmless, so the reward weights and the tracking widths, which are common to every design within a generation, need not be normalised away.

### 2.11 Hansen and colleagues, Uncertainty Handling in Evolutionary Optimisation

Hansen and colleagues proposed an uncertainty handling scheme for CMA-ES that re-evaluates a few candidates, measures how often the noise changes their ranking, and responds by enlarging the step size or the evaluation budget [12]. The scheme rests on the observation that noise harms a rank-based method only when it reorders candidates whose true values differ, and it needs only a small number of additional evaluations [12]. The installed pycma provides it as `NoiseHandler` (`../../context/cmaes.md` section 8). The plan does not adopt it now, because section 3 first removes the systematic sources of reordering. It is the escalation to apply if the split-half rank correlation logged by section 4 shows that the controlled fitness still reorders designs.

### 2.12 Heidrich-Meisner and Igel, Hoeffding and Bernstein Races for Selecting Policies

Heidrich-Meisner and Igel combined CMA-ES with racing, in which the number of episodes spent on each candidate is decided online by Hoeffding or empirical Bernstein confidence bounds, so that each performance estimate is just accurate enough to rank the candidates [13]. They motivate the scheme by the observation that the number of roll-outs needed varies during learning and between candidates, poor behaviour being detectable after a few roll-outs [13]. Racing is sequential, whereas this repository evaluates all designs simultaneously in one batch of environments, so its saving does not transfer directly. Its principle does, and section 3 sizes the evaluation to what ranking requires rather than to what an absolute estimate would.

### 2.13 Jin and Branke, Evolutionary Optimisation in Uncertain Environments

Jin and Branke classified the uncertainty an evolutionary algorithm faces into noisy fitness, perturbed design variables, approximated fitness and a moving optimum [14]. For noisy fitness the established remedies are explicit averaging, re-evaluating each candidate several times, and implicit averaging, relying on a large population to sample the neighbourhood of each region [14]. The present problem is a noisy fitness problem whose noise is partly systematic, and the systematic part cannot be averaged away by either remedy, since it is correlated with the design through the curriculum. The plan removes it by construction, after which the 400 cells of each design provide explicit averaging in parallel and the 64 designs of a generation the implicit averaging the survey describes.

### 2.14 Ng and Jordan, PEGASUS

PEGASUS converts a stochastic decision process into one with deterministic transitions by fixing the random numbers in advance, so that every candidate policy is evaluated on the same set of scenarios and the comparison between candidates is freed of sampling noise [15]. The evaluation of section 3 applies the same principle in parallel. Instance `j` of every design meets the same command, the same terrain cell and the same initial state, so that the designs are compared on common scenarios rather than on independent draws.

### 2.15 McKay, Beckman and Conover, Latin Hypercube Sampling

McKay, Beckman and Conover compared simple random sampling with stratified and Latin hypercube sampling of the inputs of a computer model, and showed that the latter improve on simple random sampling in variance for a class of estimators [16]. A Latin hypercube of `m` points places exactly one point in each of the `m` equal strata of every input axis [16]. Each design meets 400 cells and the evaluation varies four inputs per cell, the three command components and the initial yaw. A full factorial design over four axes at 400 points would allow fewer than five levels per axis, whereas a Latin hypercube of 400 points gives each axis 400 distinct values, one per stratum.

### 2.16 Kish, Survey Sampling

Kish's design effect already grounds the variance analysis of the synchronised reset in `../../context/copt_ppo_nonstationarity.md` section 7.4 [17]. The same monograph treats stratified sampling, in which a population is divided into strata, each stratum is sampled separately and the stratum means are combined with fixed weights, so that the variance between strata does not enter the estimate [17]. Section 3 stratifies by difficulty row and gives every row equal weight, which keeps the fitness from depending on whether a row received ten instances or eleven.

### 2.17 Synthesis

The table compares the protocols along the four axes of the brief.

| Work | Commands during evaluation | Environment difficulty | Aggregation | Data per design |
|---|---|---|---|---|
| Ha [4] | none, single task | seed-dependent obstacles | mean return | 16 episodes |
| Schaff and colleagues [5] | none, single task | flat or 5 degree incline, separate runs | mean training return | training episodes of the iteration |
| Luck and colleagues [6] | none, single task | single environment | mean critic value over start states | none in the loop, 100 to 300 episodes in the baseline |
| Gupta and colleagues [7] | none, forward progress | new course per episode, shared seed | mean final reward | about 100000 interactions |
| Belmonte-Baeza and colleagues [8] | random per environment | hills and steps at several levels | negated mean cost per step | 250 transitions after adaptation |
| Bjelonic and colleagues [9] | forward 1 m/s, randomised | mixed terrain, perturbations | mean reward per step | 1000 episodes |
| Ghansah and colleagues [10] | single gait | flat | cost of transport | one periodic gait |
| Zhao and colleagues [11] | none | one terrain or a combination | task objective | not established |
| Rudin and colleagues [2] | fixed 0.75 m/s forward | by type and level, curriculum removed | success rate, total reward | not established |
| This plan | Latin hypercube over the settled ranges, one command per terrain cell, shared by every design | every cell of the grid in five row passes, each cell shared by every design | mean return over the cells, equal in every row | 400 episodes of 4 s |

Three conclusions follow. The accepted measure of a legged design is velocity tracking over a range of commands and terrain difficulties, measured by the reward the policy was trained on or by its cost counterpart [8], [9]. Published protocols draw those conditions independently for each candidate [4], [8], [9], which a sequential evaluator can repair by volume but a population evaluated in one batch cannot afford. And the methods of the evolutionary literature that bear on the present work divide into those adopted now and those held in reserve. Rank invariance [3], common scenarios [15], Latin hypercube coverage [16] and stratification [17] are adopted, since they reduce variance without additional simulation. Uncertainty handling [12] and racing [13] are held in reserve, to be applied if the diagnostic of section 4 shows residual reordering. Explicit averaging by a second evaluation pass [14] is rejected, since it doubles the cost for a variance reduction that the shared conditions already provide.

## 3. Evaluation Strategy

### 3.1 Strategy Architecture

The evaluation runs once per CMA-ES generation, at the iteration of the design swap, after the policy update and before the swap. It runs whenever the CMA-ES phase has begun and a population is in the scene, and it is the only source of fitness, the training rollouts no longer contributing to it. The policy is frozen for its duration, so every design is judged by the same controller.

Instance `j` of design `k` is environment `80 j + k`, the round-robin assignment of `respawn.py:53-54`, so each of the 80 designs holds 80 instances. The rough terrain is a grid of 80 columns, each a separately generated terrain sample whose sub-terrain type is fixed by its position, and 5 rows of rising difficulty. The evaluation is made of five passes, one per row. In every pass instance `j` of every design stands in column `j`, so each terrain cell holds exactly one instance of every design and nothing else, and in pass `r` all 6400 environments stand in row `r`. Between passes every environment is reset and moved to the next row of its column. The five passes therefore take five resets, and over them every design meets every one of the 400 cells exactly once.

The rule for any population follows from one requirement, that a single generation visits every column in every row at least once. With `m` instances per design and `C` columns the evaluation makes `P = ceil(C / m)` column passes in each row, `R P` passes and resets in all, and in column pass `q` instance `j` stands in column `col(q, j) = floor((j P + q) C / (P m))`. The `P m` slots `j P + q` are spread evenly over the `C` columns, so every column is visited, the instances of one pass stand in distinct columns whenever `m <= C`, and the columns visited twice when `m` does not divide `C` are spread evenly across the grid and therefore across the sub-terrains. With more instances than columns the same rule gives one column pass, in which the extra instances repeat columns evenly spread across the grid, each with its own command, so every terrain is still covered and some are evaluated twice. The table gives the cases.

| Instances `m` per design, columns `C` | Column passes `P` | Passes in all | Columns visited twice per row |
|---|---|---|---|
| 80, 80 | 1 | 5 | none, `col = j` |
| 40, 80 | 2 | 10 | none, even then odd columns |
| 48, 80 | 2 | 10 | 16, every fifth column |
| 64, 80 | 2 | 10 | 48 |
| 100, 80 | 1 | 5 | 20, every fourth column |

```text
                 col 0     col 1     col 2            col 79
               +---------+---------+---------+-----+---------+
 pass 0  row 0 | d0..d79 | d0..d79 | d0..d79 | ... | d0..d79 |
 pass 1  row 1 | d0..d79 | d0..d79 | d0..d79 | ... | d0..d79 |
 pass 2  row 2 | d0..d79 | d0..d79 | d0..d79 | ... | d0..d79 |
 pass 3  row 3 | d0..d79 | d0..d79 | d0..d79 | ... | d0..d79 |
 pass 4  row 4 | d0..d79 | d0..d79 | d0..d79 | ... | d0..d79 |
               +---------+---------+---------+-----+---------+
 instance j of every design d0..d79 stands in column j, cell (r, j) carries its own
 command c(r, j), and a reset moves all 6400 environments one row up between passes
```

Each cell carries a condition, which fixes four things.

1. The terrain cell. In row pass `r` and column pass `q` instance `j` stands in row `r` and column `col(q, j)` of the rule above, which at 80 instances and 80 columns is its native column `j`. The coverage is complete within every generation, so no rotation across generations is needed and every generation is scored on the same cells.
2. The command. The commanded forward velocity, lateral velocity and yaw rate of cell `(r, j)` are one point of a Latin hypercube over the unit cube with one point per visited cell, `R P m` points and 400 at 80 instances and 80 columns [16], drawn once from `eval_seed` and held for the run, and mapped at each evaluation onto the command term's current ranges. Every cell therefore has its own command, which the 80 designs standing on it share. After the late start those ranges are the settled ranges the CMA-ES phase trains on, plus or minus 1.5 m/s forward, 1.0 m/s lateral and 1.35 rad/s yaw for the biped (`limx_base_env_cfg.py:1259`, `:1271`, `:1283`) and plus or minus 1.5, 0.5 and 1.35 for the quadruped (`environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:1453`, `:1464`, `:1475`). The heading controller is disabled and the yaw rate is commanded directly, so the command no longer depends on the robot's trajectory. The command is held for the whole episode, as in [2].
3. The initial state. After each reset every robot is placed at the centre of its cell at its design's spawn height (`co_optimisation/co_optimisation/utils/update.py:510`), with the yaw of the fourth axis of the same Latin hypercube point mapped onto a full turn, zero velocity, and the joints at their default positions. The random offsets of item 3 of section 1 are thereby removed, and the robots of one cell start from the same state, as [6] found to matter.
4. The disturbances. Pushes are suspended for each pass by setting the interval event timers to infinity, and command resampling by setting the command timers likewise.

Four seconds is the horizon of each pass. A cell is 15 m square (`environments/environments/tasks/locomotion/cfg/SF/terrains_cfg.py:205`, `environments/environments/tasks/locomotion/cfg/quadruped/terrains_cfg.py:17`), so a robot starting at its centre stands 7.5 m from the border. The fastest settled command of the biped, `sqrt(1.5^2 + 1.0^2) = 1.80` m/s, carries it 7.2 m in 4 s, and that of the quadruped, 1.58 m/s, 6.3 m, so each robot stays on the cell whose difficulty it is scored on. Four seconds at the control period of 0.02 s (`limx_base_env_cfg.py:1401-1405`) is 200 steps, which contains the acceleration transient of one to two seconds that `../../context/copt_ppo_nonstationarity.md` section 7.5 computes and at least two seconds of steady tracking after it. The policy acts through `act_inference`, the mean action, so the fitness measures the controller a design would deploy with rather than the learner's exploration.

The fitness of design `k` is its mean return over the 400 cells, the return of a cell being the sum of the per-step training reward from the first step of its pass until the step on which the instance terminates. Every row contributes the same 80 cells, so the plain mean is also the mean over rows of the row means, the stratified estimator of [17] with equal strata. The return is the reward the policy optimises, the measure [9] adopts. The late start holds the curriculum state constant through the CMA-ES phase, and the evaluation preserves it by suppressing every curriculum for its duration, so all designs of a generation are scored under one reward. A fall ends the accumulation of its cell, so survival enters the fitness through the reward forgone. The evaluation also measures, per design, the fraction of cells survived to the end of the horizon and the mean linear and yaw rate tracking errors over the steps alive, and per row the mean return and survival. It splits the 80 columns into two halves, the even and the odd columns, each half holding 40 columns in all five rows, 200 cells per design with every sub-terrain in equal share, and the two halves' fitnesses are rank correlated as a measure of how reliably the evaluation orders the designs. Section 4 logs all of these.

The cost is moderate. The run `copt_moral/2026-10-22_10-07-12` collected an iteration of 25 steps across 6400 environments in a median of 3.90 s through its CMA-ES phase and learned for 1.68 s, and a single pass of 200 steps with its reset took a median of 53 s ([../context/artefacts/copt_moral-2026-10-22_10-07-12-cmaes/timing_medians.txt](../context/artefacts/copt_moral-2026-10-22_10-07-12-cmaes/timing_medians.txt)). A generation at the interval of 400 iterations that the `djinn` launch passes (`/ws/djinn:135`) therefore takes about 2230 s, and the five passes about 265 s, 12 per cent of a generation, collecting 6.4 million transitions against the 64 million the generation trains on. Within that budget each design is scored on all 400 cells, every column, every sub-terrain and every row.

The evaluation must leave training exactly as it found it. It saves and restores the terrain levels, the terrain types and the environment origins it overwrites, the common step counter that gates every curriculum (`curriculums.py:175`, `:441`, `:488`), and the interval event timers. It holds both curriculum suppression flags across its five passes, so neither the terrain curriculum nor any other reads its episodes or its resets. It writes nothing to the rollout storage, since it calls the policy directly rather than `self.alg.act`. Everything else it disturbs, the episode counters, the commands, the reward sums and the robot states, is overwritten by the full reset that the design swap performs immediately afterwards (`copt_on_policy_runner.py:776-784`).

The following table gives the columns that the curriculum generator assigns to each sub-terrain in proportion (`IsaacLab/source/isaaclab/isaaclab/terrains/terrain_generator.py:245` applied to `terrains_cfg.py:216-252`), and the cells each design meets on it in one generation.

| Sub-terrain | Columns | Cells per design | Cells per split half |
|---|---|---|---|
| flat | 0 to 23 | 120 | 60 |
| pyramid slope | 24 to 31 | 40 | 20 |
| inverted pyramid slope | 32 to 39 | 40 | 20 |
| pyramid stairs | 40 to 43 | 20 | 10 |
| inverted pyramid stairs | 44 to 47 | 20 | 10 |
| waves | 48 to 63 | 80 | 40 |
| random rough | 64 to 79 | 80 | 40 |

Two sources of variation are left in place, and both are common to the comparison or independent of the design. The startup randomisation of friction, added mass, actuator gains, joint offsets and joint friction is bound to the environment index for the whole run (`limx_base_env_cfg.py:708-830`, `:888-895`), so instance `j` of design `k` carries its own draw in all five passes. Over 80 instances the friction coefficient, drawn uniformly in 0.2 to 1.25, has a per-design mean with a standard deviation of `1.05 / sqrt(12) / sqrt(80) = 0.034`. CMA-ES resamples its designs into the population slots independently every generation, so this is noise rather than bias. The observation noise likewise remains. Removing either would mean editing the shared event configuration, which rule 3 of the workspace instructions forbids for a change that only this runner asks for.

### 3.2 Implementation Plan

The plan edits three files, `co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`, `co_optimisation/co_optimisation/runners/usd_generator.py` and `scripts/rsl_rl/train.py`. The `environments` mdp package is not touched, so no task configuration changes. The edits of this section and of section 4.2 must be applied together, since edit E7 calls the logging method of section 4.2. Each edit gives the exact text to replace and its replacement.

#### Edit E1, runner imports

In `copt_on_policy_runner.py`, replace

```python
import os
import time
import torch
```

with

```python
import itertools
import math
import os
import time
import torch
```

and replace

```python
import isaaclab.sim as sim_utils
```

with

```python
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
```

#### Edit E2, the class docstring

Replace

```python
    1. Evaluates per-individual fitness from accumulated episode returns.
```

with

```python
    1. Evaluates every design under one shared set of commands, terrain cells and
       initial states, see :meth:`_evaluate_population`.
```

and replace

```python
    6. Resets fitness accumulators and fetches fresh observations.
```

with

```python
    6. Logs the generation, see :meth:`log_design_update`, and fetches fresh
       observations.
```

#### Edit E3, the runner state

In `CoptOnPolicyRunner.__init__`, replace

```python
        # Minimum step count a completed episode must reach before its return
        # contributes to individual fitness. Defaults to 0, which reproduces the
        # original behaviour of counting every completed episode, since a completed
        # episode always has length >= 1. Set explicitly in the calling
        # configuration to exclude the spurious short episodes a freshly
        # randomised phase produces (context/copt_ppo_nonstationarity.md, section 12).
        self._min_fitness_episode_length: int = copt_cfg.get(
            "min_fitness_episode_length", 0
        )
        self._design_generator = design_generator
        # Per-generation state
        self.generation: int = 0
        self.current_population: Population | None = None

        # Fitness accumulators indexed by individual
        self._individual_fitness = torch.zeros(
            self._num_individuals, dtype=torch.float, device=device
        )
        self._individual_episode_counts = torch.zeros(
            self._num_individuals, dtype=torch.long, device=device
        )
        self.encoder_cfg = train_cfg["encoder"]
```

with

```python
        # Horizon of the controlled evaluation, 200 control steps at the default.
        self._eval_duration_s: float = copt_cfg.get("eval_duration_s", 4.0)
        self._eval_seed: int = copt_cfg.get("eval_seed", 0)
        self._eval_result: dict | None = None
        self._cmaes_history: list[dict] = []
        self._design_generator = design_generator
        # Per-generation state
        self.generation: int = 0
        self.current_population: Population | None = None
        self.encoder_cfg = train_cfg["encoder"]
```

and replace

```python
        self._env_to_individual: list[int] = self._assign_individuals_to_envs()
```

with

```python
        self._env_to_individual: list[int] = self._assign_individuals_to_envs()
        self._eval_design: torch.Tensor = self._build_evaluation_design()
```

#### Edit E4, the checkpoint

In `save`, replace

```python
            "individual_fitness": self._individual_fitness.detach().cpu().clone(),
            "individual_episode_counts": self._individual_episode_counts.detach()
            .cpu()
            .clone(),
            "current_population": _population_to_dict(self.current_population),
            "design_generator": self._design_generator.get_state(),
        }
```

with

```python
            "current_population": _population_to_dict(self.current_population),
            "design_generator": self._design_generator.get_state(),
            "cmaes_history": self._cmaes_history,
        }
```

and in `load`, replace

```python
        self._individual_fitness.copy_(c["individual_fitness"].to(self.device))
        self._individual_episode_counts.copy_(
            c["individual_episode_counts"].to(self.device)
        )
        self._design_generator.load_state(c["design_generator"])
```

with

```python
        # The fitness accumulators of earlier checkpoints are ignored, the fitness of
        # the restored population being measured afresh at its generation boundary.
        self._design_generator.load_state(c["design_generator"])
        self._cmaes_history = c.get("cmaes_history", [])
```

In `_apply_restored_state`, replace

```python
        advancing the design generator, incrementing the generation counter, or
        zeroing the fitness accumulators, all of which were already restored in
        :meth:`load`.  The usual ``unwrapped_env.reset()`` is replaced by
```

with

```python
        advancing the design generator or incrementing the generation counter, both
        of which were already restored in :meth:`load`.  The usual
        ``unwrapped_env.reset()`` is replaced by
```

#### Edit E5, the training loop

In `learn`, replace

```python
                            # ---- COPT: accumulate per-individual fitness --------
                            completed_env_ids = new_ids[:, 0]
                            for env_idx in completed_env_ids.tolist():
                                if (
                                    cur_episode_length[env_idx]
                                    < self._min_fitness_episode_length
                                ):
                                    # Spurious short episode, most likely a freshly
                                    # randomised phase timing out within a few steps.
                                    # Excluded from fitness rather than counted.
                                    continue
                                ind_idx = self._env_to_individual[env_idx]
                                self._individual_fitness[ind_idx] += cur_reward_sum[
                                    env_idx
                                ]
                                self._individual_episode_counts[ind_idx] += 1
                            # ---- end COPT injection ----------------------------

                            rewbuffer.extend(
```

with

```python
                            rewbuffer.extend(
```

and replace

```python
                with torch.inference_mode():
                    self._update_morphology(it + 1, respawn=late_start_toggled)
```

with

```python
                with torch.inference_mode():
                    if self._copt_started and self.current_population is not None:
                        self._eval_result = self._evaluate_population()
                    self._update_morphology(it + 1, respawn=late_start_toggled)
```

#### Edit E6, the evaluation

Replace the method `_compute_individual_fitness`,

```python
    def _compute_individual_fitness(self) -> list[float]:
        """Return mean episode return per individual (0.0 if no episodes completed)."""
        fitness: list[float] = []
        for i in range(self._num_individuals):
            count = self._individual_episode_counts[i].item()
            if count > 0:
                fitness.append((self._individual_fitness[i] / count).item())
            else:
                fitness.append(0.0)
        return fitness
```

with

```python
    def _build_evaluation_design(self) -> torch.Tensor:
        """A Latin hypercube on the unit cube of (vx, vy, wz, yaw), one point per cell an evaluation visits.

        Returns (R, P, m, 4), point (r, q, j) the command and initial yaw of instance j in difficulty
        row r and column pass q, fixed for the run and mapped onto the command ranges at each
        evaluation. R and P are 1 on a plane.
        """
        m = self.env.num_envs // self._num_individuals
        origins = self.env.unwrapped.scene.terrain.terrain_origins
        rows, passes = (origins.shape[0], -(-origins.shape[1] // m)) if origins is not None else (1, 1)
        count = rows * passes * m
        generator = torch.Generator().manual_seed(self._eval_seed)
        axes = [
            (torch.randperm(count, generator=generator) + torch.rand(count, generator=generator)) / count
            for _ in range(4)
        ]
        return torch.stack(axes, dim=1).view(rows, passes, m, 4).to(self.env.device)

    def _evaluation_columns(self, cols: int) -> torch.Tensor:
        """Terrain column of each instance in each column pass, every column visited in every row.

        Returns (P, m) for P = ceil(cols / m) passes. Slot j * P + q is spread evenly over the
        columns, so the instances of one pass stand in distinct columns whenever m <= cols, every
        column is visited at least once, and the columns visited twice, when m does not divide
        cols, are spread evenly across the grid and so across the sub-terrains.
        """
        m = self.env.num_envs // self._num_individuals
        passes = -(-cols // m)
        q = torch.arange(passes, device=self.env.device).unsqueeze(1)
        j = torch.arange(m, device=self.env.device).unsqueeze(0)
        return (j * passes + q) * cols // (passes * m)

    def _evaluate_population(self) -> dict:
        """Score every design on one shared set of terrain cells, commands and initial states.

        Instance j of every design, environment j * N + k for design k, stands in column
        col[q, j] in column pass q, and every column pass is made in every difficulty row, so
        each cell of a pass holds one instance of every design and every design meets every cell
        of the grid at least once. Each visited cell carries its own command and initial yaw.
        Returns per design (N,) the fitness, the mean return over the R x P x m visited cells,
        the survival fraction and the mean linear and yaw rate tracking errors, and in addition
        the split-half rank correlation of the fitness over the even and odd columns, per row the
        mean return and
        survival, and the wall time. The terrain assignment, the step counter and the interval
        event timers are restored, and the reset of the design swap that follows overwrites
        everything else.
        """
        env = self.env.unwrapped
        robot = env.scene["robot"]
        terrain = env.scene.terrain
        command = env.command_manager.get_term("base_velocity")
        n = self._num_individuals
        m = env.num_envs // n
        rows, passes = self._eval_design.shape[:2]
        replica = torch.arange(env.num_envs, device=env.device) // n
        steps = int(round(self._eval_duration_s / env.step_dt))
        start = time.time()

        common_step_counter = env.common_step_counter
        interval_time_left = [t.clone() for t in env.event_manager._interval_term_time_left]
        grid = terrain.terrain_origins is not None
        if grid:
            assignment = (
                terrain.terrain_levels.clone(),
                terrain.terrain_types.clone(),
                terrain.env_origins.clone(),
            )
            col = self._evaluation_columns(terrain.terrain_origins.shape[1])
        else:
            col = torch.arange(m, device=env.device).unsqueeze(0)

        ranges = command.cfg.ranges
        lo = torch.tensor(
            [ranges.lin_vel_x[0], ranges.lin_vel_y[0], ranges.ang_vel_z[0], -math.pi], device=env.device
        )
        hi = torch.tensor(
            [ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_z[1], math.pi], device=env.device
        )
        design = lo + (hi - lo) * self._eval_design  # (R, P, m, 4)
        ret = torch.zeros(rows, passes, env.num_envs, device=self.device)
        survived = torch.zeros_like(ret)
        length = torch.zeros_like(ret)
        lin_err = torch.zeros_like(ret)
        yaw_err = torch.zeros_like(ret)
        env._suppress_terrain_curriculum = True
        env._suppress_curriculum = True
        self.eval_mode()
        try:
            for r, q in itertools.product(range(rows), range(passes)):
                if grid:
                    terrain.terrain_levels[:] = r
                    terrain.terrain_types[:] = col[q][replica]
                    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
                condition = design[r, q][replica]
                env.reset()
                # The reset events draw a planar offset, a yaw, a velocity and joint offsets
                # per environment, on which the designs would otherwise be ranked.
                root = robot.data.default_root_state.clone()
                root[:, :3] += env.scene.env_origins
                zero = torch.zeros_like(condition[:, 3])
                root[:, 3:7] = math_utils.quat_from_euler_xyz(zero, zero, condition[:, 3])
                root[:, 7:] = 0.0
                robot.write_root_state_to_sim(root)
                robot.write_joint_state_to_sim(
                    robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel)
                )
                command.vel_command_b[:] = condition[:, :3]
                command.is_heading_env[:] = False
                command.is_standing_env[:] = False
                command.time_left[:] = float("inf")
                for t in env.event_manager._interval_term_time_left:
                    t[:] = float("inf")
                env.observation_manager.reset()
                obs = TensorDict(
                    env.observation_manager.compute(update_history=True), batch_size=[env.num_envs]
                ).to(self.device)

                target = condition[:, :3].to(self.device)
                alive = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
                for _ in range(steps):
                    actions = self.alg.policy.act_inference(obs)
                    obs, rewards, dones, _ = self.env.step(actions.to(self.env.device))
                    obs, rewards = obs.to(self.device), rewards.to(self.device)
                    dones = dones.to(self.device).bool()
                    # step resets a terminated environment, so its state no longer belongs to the episode.
                    counted = alive & ~dones
                    lin_err[r, q] += counted * torch.norm(
                        target[:, :2] - robot.data.root_lin_vel_b[:, :2].to(self.device), dim=1
                    )
                    yaw_err[r, q] += counted * torch.abs(
                        target[:, 2] - robot.data.root_ang_vel_b[:, 2].to(self.device)
                    )
                    ret[r, q] += alive * rewards
                    length[r, q] += counted
                    alive &= ~dones
                survived[r, q] = alive.float()
        finally:
            env._suppress_terrain_curriculum = False
            env._suppress_curriculum = False
            env.common_step_counter = common_step_counter
            for t, saved in zip(env.event_manager._interval_term_time_left, interval_time_left):
                t.copy_(saved)
            if grid:
                for t, saved in zip(
                    (terrain.terrain_levels, terrain.terrain_types, terrain.env_origins), assignment
                ):
                    t.copy_(saved)
            self.train_mode()

        shape = (rows, passes, m, n)
        ret, survived = ret.view(shape), survived.view(shape)
        steps_alive = length.view(shape).sum((0, 1, 2)).clamp(min=1)

        # The even and the odd columns, so the halves share no terrain sample and each holds every
        # row and every sub-terrain in equal share.
        even = (col % 2 == 0).to(self.device)  # (P, m)
        split = float("nan")
        if even.any() and (~even).any():
            halves = (ret[:, even].mean((0, 1)), ret[:, ~even].mean((0, 1)))
            ranks = torch.stack([h.argsort().argsort().float() for h in halves])
            split = torch.corrcoef(ranks)[0, 1].item()
        return {
            "fitness": ret.mean((0, 1, 2)),
            "survival": survived.mean((0, 1, 2)),
            "lin_vel_error": lin_err.view(shape).sum((0, 1, 2)) / steps_alive,
            "yaw_rate_error": yaw_err.view(shape).sum((0, 1, 2)) / steps_alive,
            "split_half_rank_correlation": split,
            "row_return": {r: ret[r].mean().item() for r in range(rows)},
            "row_survival": {r: survived[r].mean().item() for r in range(rows)},
            "time": time.time() - start,
        }
```

#### Edit E7, the fitness consumed by CMA-ES

In `_update_morphology`, replace

```python
        if sim.is_playing():  # 1. stop (Fabric off)
            sim._disable_app_control_on_stop_handle = True
            sim.stop()
            sim._disable_app_control_on_stop_handle = False

        if self._ea_late_start <= it:
```

with

```python
        if sim.is_playing():  # 1. stop (Fabric off)
            sim._disable_app_control_on_stop_handle = True
            sim.stop()
            sim._disable_app_control_on_stop_handle = False

        evaluated = None
        if self._ea_late_start <= it:
```

replace

```python
            else:
                fitness = self._compute_individual_fitness()
                print("Updating Design Population")
                self._design_generator.update_with_fitness(fitness)
        elif self._reinit_lr_each_generation:
            self.alg.reinit_learning_rate()
        self.current_population = self._design_generator.generate_population(
            self.generation
        )
```

with

```python
            elif self.current_population is not None:
                fitness = self._eval_result["fitness"].tolist()
                evaluated = self.current_population
                evaluated_generation = self.generation - 1
                print("Updating Design Population")
                update_start = time.time()
                self._design_generator.update_with_fitness(fitness)
                update_time = time.time() - update_start
        elif self._reinit_lr_each_generation:
            self.alg.reinit_learning_rate()
        population_start = time.time()
        self.current_population = self._design_generator.generate_population(
            self.generation
        )
```

and at the end of the method replace

```python
            self._individual_fitness.zero_()  # 7. zero accumulators
            self._individual_episode_counts.zero_()
```

with

```python
        if evaluated is not None:
            population_time = time.time() - population_start
            self.log_design_update(locals())
        self._eval_result = None
```

#### Edit E8, the bootstrap of generation zero

In `_reload_morphology`, replace

```python
        if self.current_population is not None :
            fitness = self._compute_individual_fitness()
            print("Updating Design Population")
            self._design_generator.update_with_fitness(fitness)

        elif self.current_population is None and self._ea_late_start < 0:
            self._design_generator.sample_batch()
```

with

```python
        if self.current_population is None and self._ea_late_start < 0:
            self._design_generator.sample_batch()
```

and replace

```python
            apply_actuator_params(
                unwrapped_env, self.current_population.get_actuator_params()
            )

            # Reset fitness accumulators
            print("Resetting Accumulators")
            self._individual_fitness.zero_()
            self._individual_episode_counts.zero_()
```

with

```python
            apply_actuator_params(
                unwrapped_env, self.current_population.get_actuator_params()
            )
```

#### Edit E9, the calling configuration

In `scripts/rsl_rl/train.py`, replace

```python
            "min_fitness_episode_length": 25,
            "reinit_lr_each_generation": True,
        }
```

with

```python
            "reinit_lr_each_generation": True,
            "eval_duration_s": 4.0,
            "eval_seed": 42,
        }
```

The two values carry no switch, the evaluation running in every co-optimisation task, and are set here so that the dumped `params/copt_agent.yaml` records the horizon and the seed of each run (`train.py:354-356`).

#### Application instructions

1. Read the status banner of this plan and of [README.md](README.md) first, and apply nothing already recorded as applied.
2. Apply E1 to E9 and the edits L1 to L4 of section 4.2 in one pass, each by exact replacement of the quoted text. Every quoted old text occurs once in its file in the tree of 2026-10-22.
3. Confirm that no reference to `_compute_individual_fitness`, `_individual_fitness`, `_individual_episode_counts` or `min_fitness_episode_length` remains, `grep -rn` over `co_optimisation/co_optimisation` and `scripts`, and compile the three files with `python -m py_compile` inside `isaac-lab-base-shreyas-gpu0`.
4. Run a short training in `isaac-lab-base-shreyas-gpu1` from a working directory that holds no `environments/` folder, with `--task Isaac-Limx-SF-Copt-MoRAL-Rough-v0 --headless --copt_num_individuals 80 --ea_late_start 4 --ea_update_interval 2 --max_iterations 14`, which keeps the scene's 6400 environments, so that CMA-ES evaluates at least four generations.
5. Confirm in that run that the terminal prints one CMA-ES block per evaluated generation, that the event file carries the `CMAES/` tags of section 4.1, that `designs/` holds the `design_parameters_*.csv`, `cmaes_generation_*.csv` and `cmaes_history.csv` files with headers and scalar cells, that `CMAES/eval_time` is near the 265 s of section 3.1, that `CMAES/eval/return_row_<r>` appears for all five rows, and that `Curriculum/terrain_levels` is continuous across each evaluated generation rather than jumping to the evaluation assignment.
6. Repeat step 4 with `--task Isaac-Quadruped-Copt-Rough-v0`, and resume the first run from its last checkpoint to confirm that a checkpoint written before this plan, and one written after it, both load.
7. Record the outcome, the measured evaluation time and any divergence in a dated section of `../../context/cmaes.md`, update this banner and the register, and update `../../CO_OPTIMISATION.md` and `../../ARCHITECTURE.md` for the new fitness, as rule 19 of the workspace instructions requires.

#### Design rationale

The evaluation is a method of the runner rather than a new runner class, since the extend-the-parent rule of this workspace places a cross-cutting feature where every co-optimisation task inherits it. It is the only fitness, so the accumulators, the per-step loop over completed episodes that filled them, the episode length threshold and their checkpoint entries are removed rather than kept as a second path, and a checkpoint that still carries them loads with the entries ignored. The five rows are visited in sequential passes rather than spread across the instances, because a column can hold one instance of every design in every row only if all designs move through the rows together, and a reset between passes is the only way to relocate every robot at once. It is called from `learn` and not from inside `_update_morphology`, because `_update_morphology` begins by stopping the simulation (`copt_on_policy_runner.py:737-740`), after which no step can be taken. The update branch is guarded on a population being present, which is the state after CMA-ES has terminated and `generate_population` has returned None (`usd_generator.py:893-897`), so a terminated search is neither evaluated nor told. The random design search, whose late start is negative (`train.py:273-277`), is evaluated by the same path once its first population is in the scene, and since its `update_with_fitness` ignores the fitness (`usd_generator.py:614-619`) the evaluation serves it as a logged baseline on the same scale. The Latin hypercube is held on the unit cube and mapped onto the command ranges at each evaluation, so the evaluated commands are those the CMA-ES phase trains on, constant across generations once the curricula have settled and identical across designs within a generation in every case. The initial state is written after `env.reset()` rather than by editing the reset events, because those events are shared configuration. The overwrite reaches the buffers of the velocity command term, which `co_optimisation/co_optimisation/utils/env_state.py:74-98` already reads and writes for checkpointing, and the private `_interval_term_time_left` of the event manager, which no code of the package touched before. That list is the only interval timer store the event manager keeps (`IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py:207-229`), and it is the one new dependence on private Isaac Lab state the plan introduces.

#### Findings outside the scope of this plan

Three findings were made while planning. The categorical search does not start from the spread its configuration names. `CatCMAwM` samples from `N(m, sigma^2 A C A)` and initialises `C` to `diag(1/144)` on the unit box, so the `sigma0 = 0.25` of `train.py:286` yields an initial standard deviation of 0.021 per coordinate, which 4000 draws in `isaac-lab-base-shreyas-gpu0` measured at 0.020 to 0.021, against 0.25 for the `cma` generators given the same value. When a population contains a design that `_build_xml` refuses, the substitute design is simulated while CMA-ES is told the resulting fitness for the refused genotype (`usd_generator.py:704-722`, `:642-645`), which the `feasible` column of section 4 makes visible. And once `es.stop()` fires and `generate_population` returns None, `save` serialises the None population (`copt_on_policy_runner.py:190`) and raises. The design dump of section 4 and the guard of edit E7 handle the None population, `save` does not.

## 4. Logging Strategy

### 4.1 Architecture

A CMA-ES generation is a slower clock than a learning iteration, and its figures are lost among the per-iteration scalars when written into the existing heads. The plan adds one TensorBoard head, `CMAES/`, written once per evaluated generation at the learning iteration of the swap, so that its curves align on the same horizontal axis as `Train/` and `Loss/`. The method `log_design_update` mirrors the inherited `log` of `rsl_rl/rsl_rl/runners/on_policy_runner.py:176-289` in signature and in output, a record of scalars written to the writer and a fixed-width block printed to the terminal, and it is called from `_update_morphology` after `update_with_fitness`, so the distribution statistics describe the search that sampled the next population.

The scalars of the head fall into five groups.

| Group | Tags | Source |
|---|---|---|
| Identity | `CMAES/generation`, `CMAES/cma_iteration`, `CMAES/evaluations`, `CMAES/terminated`, `CMAES/infeasible` | runner, generator |
| Fitness | `CMAES/fitness_mean`, `_std`, `_max`, `_min`, `_median`, `CMAES/split_half_rank_correlation` | evaluation |
| Distribution | `CMAES/sigma`, `CMAES/axis_ratio`, `CMAES/mean/<parameter>`, `CMAES/std/<parameter>`, `CMAES/best/<parameter>`, `CMAES/pick_probability/<group>` | generator |
| Evaluation | `CMAES/eval/survival`, `CMAES/eval/lin_vel_error`, `CMAES/eval/yaw_rate_error`, `CMAES/eval/return_row_<r>`, `CMAES/eval/survival_row_<r>` | evaluation |
| Timing | `CMAES/eval_time`, `CMAES/update_time`, `CMAES/population_time` | runner |

The fitness statistics are those of the accumulated reward over the evaluation horizon, the quantity CMA-ES ranks. The distribution mean is reported through pycma's `xfavorite`, the mean in phenotype space, which the library documents as its current best estimate of the optimum (`cma/evolution_strategy.py:386` of the installed 4.4.4) [18], and the standard deviations through `result.stds`, which include the step size and the covariance (`cma/evolution_strategy.py:3226`) [18]. Both are converted from the unit box to the units of `param_ranges`. The best individual of a generation is logged as well, but under noise the maximum of noisy estimates is optimistically biased, and the distribution mean is the figure to report as the result of the search [18]. The split-half rank correlation is the diagnostic that decides whether the escalations of section 2.17 are needed. Each half holds the 200 cells of 40 alternate columns across all five rows, so a correlation near one shows that the evaluation orders the designs reproducibly, and a low correlation that the ranking CMA-ES receives is dominated by noise, which is the condition uncertainty handling [12] and racing [13] exist to treat.

Three files accumulate the results across generations under the run's `designs/` directory. `design_parameters_<it>.csv` is written whenever a population is applied, one row per individual with a header, carrying only the parameters the generator searches, the four of `CMAES_PARAM_KEYS` (`usd_generator.py:447-449`) and, for the categorical search, the actuator pick of each group, together with the `feasible` flag. It replaces `link_lengths_<it>.csv`, whose cells are Python dictionary representations of the full link geometry ([../context/artefacts/copt_moral-2026-10-21_07-16-23-fitness-evaluation/link_lengths_13599_head.csv](../context/artefacts/copt_moral-2026-10-21_07-16-23-fitness-evaluation/link_lengths_13599_head.csv)). `cmaes_generation_<g>.csv` is written for each evaluated generation, joining the parameters of each individual with its fitness and its evaluation metrics. `cmaes_history.csv` holds one row per generation with every scalar of the head and is rewritten whole each generation from `self._cmaes_history`, which the checkpoint carries (edit E4), so a resumed run continues its history.

The terminal block takes the following form.

```text
################################################################################
                    CMA-ES generation 3 at learning iteration 3799

                 fitness_mean: 41.2093
                  fitness_std: 6.0174
                          ...
                  update_time: 0.0312
              population_time: 61.4471
--------------------------------------------------------------------------------
```

### 4.2 Implementation Plan

#### Edit L1, the searched parameters of each individual

In `usd_generator.py`, `Population.__init__`, replace

```python
        nominal: list[dict] | None = None,
    ) -> None:
```

with

```python
        nominal: list[dict] | None = None,
        parameters: list[dict] | None = None,
    ) -> None:
```

replace

```python
        self._nominal = nominal
```

with

```python
        self._nominal = nominal
        self._parameters = parameters if parameters is not None else []
```

and replace

```python
        return self._nominal
```

with

```python
        return self._nominal

    def get_parameters(self) -> list[dict]:
        """Return the searched parameters of each individual and whether its design emitted, for logging."""
        return self._parameters
```

In `DesignGeneratorBase._build_population`, replace

```python
        nominal: list = []
        for idx in indices:
            spec, act = self._generate_individual(generation, idx)
            xml_path = self._generate_individual_mjcf(generation, idx, spec)
            if self._last_emitted is not spec:
```

with

```python
        nominal: list = []
        parameters: list[dict] = []
        for idx in indices:
            spec, act = self._generate_individual(generation, idx)
            xml_path = self._generate_individual_mjcf(generation, idx, spec)
            # The drawn design, whose genotype CMA-ES is told the fitness of even when refused.
            parameters.append({**self._design_record(spec), "feasible": self._last_emitted is spec})
            if self._last_emitted is not spec:
```

and replace

```python
        return Population(usd_files, actuator_params, link_length_params,
                          self.family.root_body, specs, nominal)
```

with

```python
        return Population(usd_files, actuator_params, link_length_params,
                          self.family.root_body, specs, nominal, parameters)
```

#### Edit L2, the generator accessors

In `DesignGeneratorBase`, replace

```python
    def load_state(self, state: dict) -> None:
        pass
```

with

```python
    def load_state(self, state: dict) -> None:
        pass

    def _design_record(self, spec) -> dict:
        """The searched parameters of the individual just drawn, for the design log."""
        return dict(self._current_parameters or {})

    def search_statistics(self) -> dict | None:
        """The search distribution in the units of ``param_ranges``, or None where there is none."""
        return None
```

In `CMAESDesignGenerator`, replace

```python
    def toggle_late_start(self) -> None:
        self.late_start = not self.late_start
```

with

```python
    def toggle_late_start(self) -> None:
        self.late_start = not self.late_start

    def search_statistics(self) -> dict | None:
        lo = np.array([self.param_ranges[k][1][0] for k in self.param_keys])
        span = np.array([self.param_ranges[k][1][1] for k in self.param_keys]) - lo
        result = self._es.result
        return {
            "cma_iteration": int(self._es.countiter),
            "evaluations": int(self._es.countevals),
            "sigma": float(self._es.sigma),
            "axis_ratio": float(self._es.D.max() / self._es.D.min()),
            "terminated": float(self._terminated),
            "stop": ", ".join(self._es.stop()),
            "mean": dict(zip(self.param_keys, (lo + span * np.asarray(result.xfavorite)).tolist())),
            "std": dict(zip(self.param_keys, (span * np.asarray(result.stds)).tolist())),
        }
```

In `CatCMAESDesignGenerator`, replace

```python
    def _dumps(self) -> bytes:
        # CatCMAwM carries no pickle_dumps of its own, unlike cma's strategy.
        return pickle.dumps(self._es)
```

with

```python
    def _dumps(self) -> bytes:
        # CatCMAwM carries no pickle_dumps of its own, unlike cma's strategy.
        return pickle.dumps(self._es)

    def _design_record(self, spec) -> dict:
        picks = {group: getattr(spec, self.family.spec_fields[group]) for group in self.cat_groups}
        return {**super()._design_record(spec), **picks}

    def search_statistics(self) -> dict | None:
        # Private state of cmaes 0.13.1, whose CatCMAwM exposes no accessor for it.
        es = self._es
        lo = np.array([self.param_ranges[k][1][0] for k in self.param_keys])
        span = np.array([self.param_ranges[k][1][1] for k in self.param_keys]) - lo
        # Sampled as N(m, sigma^2 A C A), cmaes/_catcmawm.py _sample_from_gaussian.
        a = np.asarray(es._A)
        cov = es._sigma ** 2 * a[:, None] * np.asarray(es._C) * a[None, :]
        eig = np.linalg.eigvalsh(cov)
        mean = np.clip(np.asarray(es._mean), 0.0, 1.0)
        q = np.asarray(es._q)
        return {
            "cma_iteration": int(es.generation),
            "evaluations": int(es.generation * es.population_size),
            "sigma": float(es._sigma),
            "axis_ratio": float(np.sqrt(eig.max() / eig.min())),
            "terminated": float(self._terminated),
            "stop": "should_stop" if es.should_stop() else "",
            "mean": dict(zip(self.param_keys, (lo + span * mean).tolist())),
            "std": dict(zip(self.param_keys, (span * np.sqrt(np.diag(cov))).tolist())),
            "pick_probability": {
                group: float(q[row].max()) for row, group in enumerate(self.cat_groups)
            },
        }
```

#### Edit L3, the logging method

Add the following method to `CoptOnPolicyRunner`, directly after `_evaluate_population`.

```python
    def log_design_update(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Write one CMA-ES generation to the CMAES head, the design logs and the terminal.

        Called by _update_morphology after update_with_fitness, with its locals, so the
        fitness is that of the evaluated population and the statistics those of the
        distribution that sampled the next.
        """
        it = self.current_learning_iteration
        fitness = torch.tensor(locs["fitness"], dtype=torch.float)
        evaluation = self._eval_result
        parameters = locs["evaluated"].get_parameters()
        if len(parameters) != len(fitness):
            # A population restored from a checkpoint that predates the parameter record.
            parameters = [{} for _ in range(len(fitness))]

        record = {
            "iteration": it,
            "generation": locs["evaluated_generation"],
            "fitness_mean": fitness.mean().item(),
            "fitness_std": fitness.std().item(),
            "fitness_max": fitness.max().item(),
            "fitness_min": fitness.min().item(),
            "fitness_median": fitness.median().item(),
            "split_half_rank_correlation": evaluation["split_half_rank_correlation"],
            "infeasible": sum(not p.get("feasible", True) for p in parameters),
        }
        for key in ("survival", "lin_vel_error", "yaw_rate_error"):
            record[f"eval/{key}"] = evaluation[key].mean().item()
        for r, value in evaluation["row_return"].items():
            record[f"eval/return_row_{r}"] = value
        for r, value in evaluation["row_survival"].items():
            record[f"eval/survival_row_{r}"] = value
        for key, value in (self._design_generator.search_statistics() or {}).items():
            if isinstance(value, dict):
                record.update({f"{key}/{name}": v for name, v in value.items()})
            else:
                record[key] = value
        best = parameters[int(fitness.argmax())]
        record.update({
            f"best/{name}": float(v)
            for name, v in best.items()
            if isinstance(v, float)
        })
        record["eval_time"] = evaluation["time"]
        record["update_time"] = locs["update_time"]
        record["population_time"] = locs["population_time"]
        self._cmaes_history.append(record)

        if self.log_dir is not None and not self.disable_logs:
            for key, value in record.items():
                if key != "iteration" and isinstance(value, (int, float)):
                    self.writer.add_scalar(f"CMAES/{key}", value, it)
            designs = os.path.join(self.log_dir, "designs")
            pd.DataFrame(self._cmaes_history).to_csv(
                os.path.join(designs, "cmaes_history.csv"), index=False
            )
            table = pd.DataFrame(parameters)
            table["fitness"] = fitness.tolist()
            for key in ("survival", "lin_vel_error", "yaw_rate_error"):
                table[f"eval_{key}"] = evaluation[key].tolist()
            table.to_csv(
                os.path.join(designs, f"cmaes_generation_{record['generation']:04d}.csv"),
                index_label="individual",
            )

        title = f" \033[1m CMA-ES generation {record['generation']} at learning iteration {it} \033[0m "
        log_string = f"""{"#" * width}\n{title.center(width, " ")}\n\n"""
        for key, value in record.items():
            if key in ("iteration", "generation"):
                continue
            text = f"{value:.4f}" if isinstance(value, float) else f"{value}"
            log_string += f"""{f"{key}:":>{pad}} {text}\n"""
        log_string += f"""{"-" * width}\n"""
        print(log_string)
```

#### Edit L4, the readable design dump

In `learn`, replace

```python
                    link_lengths = self.current_population.get_link_length_params()
                    keys = None
                    if len(link_lengths) > 0:
                        keys = link_lengths[0].keys()
                    df = pd.DataFrame(self.current_population.get_link_length_params())
                    df.to_csv(
                        os.path.join(self.log_dir, "designs", f"link_lengths_{it}.csv"),
                        header=False,
                        columns=keys,
                    )
```

with

```python
                    if self.current_population is not None:
                        pd.DataFrame(self.current_population.get_parameters()).to_csv(
                            os.path.join(self.log_dir, "designs", f"design_parameters_{it}.csv"),
                            index_label="individual",
                        )
```

#### Application instructions

Apply L1 to L4 in the same pass as E1 to E9 and verify them by steps 3 to 6 of section 3.2. In addition, confirm on the categorical search, `--design_search cat`, that `design_parameters_*.csv` carries one column per actuator group and that `CMAES/pick_probability/<group>` appears, and that the `cmaes_history.csv` of the resumed run of step 6 continues rather than restarts.

#### Design rationale

The parameters are recorded inside `_build_population`, the one place every generator passes through, from `_current_parameters`, which every generator already sets before emission (`usd_generator.py:776-786`, `:910-921`, `:1112-1127`), so the record needs no generator to change its sampling. They are recorded against the drawn design rather than the emitted one, because the drawn genotype is what CMA-ES is told about, and the `feasible` flag says when the two differ. The accessors live on the generator because `self._es` is private to it and differs in type between the `cma` and the `cmaes` strategies, and the base class returns None so that the random generator, which has no distribution, logs fitness, evaluation and timing only. The method is invoked with `locals()`, as the inherited `log` is (`copt_on_policy_runner.py:469`), so its inputs are named where they are computed. The readable dump replaces the link geometry dump outright, the geometry being recoverable from the emitted documents under `copt_usds/` that the generator already keeps (`usd_generator.py:728-737`), and it is written only when a population exists, which removes the dereference of a None population that the old dump performed once CMA-ES had terminated.

## 5. Implementation Outcome

### 5.1 Application

The edits of sections 3.2 and 4.2 were applied on 2026-10-22 by exact replacement, all twenty seven quoted old texts being found once and the one insertion, `log_design_update`, placed after `_evaluate_population`. The evaluation applied then placed instance `j` in column `j` and row `(j + g) mod R` in a single pass, and the five pass form of edit E6 is applied through `copt_improvement.md` section 3. The three files compile, and no reference to `_compute_individual_fitness`, `_individual_fitness`, `_individual_episode_counts` or `min_fitness_episode_length` remains in `co_optimisation/co_optimisation` or `scripts`. The evaluation is reached from `learn` (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:414`) and implemented at `:777-943`, the logging method at `:945`, and the generator accessors at `co_optimisation/co_optimisation/runners/usd_generator.py:642-662`, `:914-931` and `:1128-1160`. The installed copies of `co_optimisation` in `isaac-lab-base-shreyas-gpu0` and `isaac-lab-base-shreyas-gpu1` were reinstalled from the tree, the two training jobs then running in those containers having already imported the earlier modules and continuing unaffected.

### 5.2 Additions made during application

Two additions were made at the user's request and are not part of sections 3 and 4.

The first makes training end gracefully once CMA-ES terminates. The smoke tests of section 5.3 stop CMA-ES on `maxiter` at their last generation, which exposed the third finding of section 3.2 in a sharper form. `generate_population` returns None once the search has stopped (`usd_generator.py:893-897` of the tree before application), `_update_morphology` has by then stopped the simulation, and with no population to apply nothing restarted it, so the next `get_observations` found no physics view and raised. Before this plan the same path raised one line earlier, in the link length dump. The fix keeps the last population in the scene and lets the swap re-apply it, which restarts the simulation and continues training on the final designs (`copt_on_policy_runner.py:722-729`). The current population therefore never becomes None after the first generation, so `save` no longer meets a None population either. Once the search has terminated the evaluation and the logging continue at every generation boundary on that fixed population, `update_with_fitness` ignoring the fitness (`usd_generator.py:937-948`), which the plan's design rationale had expected to skip.

The second exports the optimised population at the end of training. After the final save, `log_optimised_designs` (`copt_on_policy_runner.py:1020`, called at `:463`) evaluates the final population once more, prints a summary, and writes to `designs/optimised/` the MJCF of every individual as `individual_<k>.xml`, copied from the documents the generator emitted and whose paths the population now records (`usd_generator.py:561`), `distribution_mean.xml`, the design the search recommends, emitted by `emit_design` from `optimised_design` (`usd_generator.py:650-662`), `optimised_designs.csv` with every individual's parameters, fitness, evaluation metrics and rank, and `final_results.txt`, the printed summary naming the best individual. The recommended design is the distribution mean `xfavorite` for the `cma` generators [18] and, for the categorical search, the clipped continuous mean with each group's most probable actuator. The save precedes the export so that the final checkpoint holds the training state rather than the evaluation's. `_population_to_dict` and `_population_from_dict` carry the MJCF paths, so a resumed run can also export, and the summary is printed with `flush=True`, which `log_design_update` now does as well, after one smoke test lost its last buffered terminal lines at shutdown.

### 5.3 Verification

Both training containers were running twenty four hour and two hour co-optimisation jobs, so the smoke tests ran at 1024 environments and 16 designs rather than the full scene of step 4 of section 3.2, on the 64 column terrain of that date, 64 instances per design with instance `j` in column `j`. Each ran 14 iterations with the late start at 4 and the interval at 2, from a scratch working directory with the edited source first on the path ([../context/artefacts/cmaes_evaluation-2026-10-22-smoke/cmaes_smoke.sh](../context/artefacts/cmaes_evaluation-2026-10-22-smoke/cmaes_smoke.sh)).

1. The biped and the quadruped each evaluated five generations, CMA-ES terminating on `maxiter` at the fifth, training continued, the final checkpoint `model_13.pt` was written, the export ran, and no traceback appeared ([../context/artefacts/cmaes_evaluation-2026-10-22-smoke/biped3/](../context/artefacts/cmaes_evaluation-2026-10-22-smoke/biped3/)). The categorical search did the same, its export carrying the actuator picks and the pick probabilities ([../context/artefacts/cmaes_evaluation-2026-10-22-smoke/biped_cat/](../context/artefacts/cmaes_evaluation-2026-10-22-smoke/biped_cat/)).
2. All forty three `CMAES/` tags of section 4.1 that apply to the `cma` generators appeared in the event files, and `design_parameters_*.csv`, `cmaes_generation_*.csv` and `cmaes_history.csv` carried headers and scalar cells. The emitted `distribution_mean.xml` and `individual_0013.xml` parse as `mujoco` documents of 15 bodies.
3. The evaluation took 35 to 44 s at 1024 environments on the shared GPUs, against the 25 s section 3.1 estimates for 4096 environments on an unshared one, which this test cannot confirm. The split-half rank correlation ranged from 0.94 to 0.99 for the biped and from -0.14 to 0.82 for the quadruped, whose instances nearly all survived the horizon after five iterations of training, so that its designs were barely separated. Every biped instance fell within the horizon after five iterations, which is expected of so young a policy.
4. A resume from a checkpoint written by the new code continued its history, rows 2 to 4 restored and row 5 appended. A resume from a copy rewritten into the earlier format, with the accumulator entries and without the history, loaded, ignored the entries and began a fresh history ([../context/artefacts/cmaes_evaluation-2026-10-22-smoke/resume_legacy/](../context/artefacts/cmaes_evaluation-2026-10-22-smoke/resume_legacy/)).
5. Not exercised. The terrain curriculum begins at step 9600 (`environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:1233-1238`), beyond a 14 iteration run, so `Curriculum/terrain_levels` stayed at zero and the save and restore of the terrain assignment, though it ran without error, was not shown to preserve a non-trivial assignment. The full scale run of step 4 of section 3.2 remains to be made.

## Bibliography

[1] N. Cheney, J. Bongard, V. SunSpiral, H. Lipson. Scalable co-optimization of morphology and control in embodied machines. Journal of the Royal Society Interface 15(143), 20170937, 2018.

[2] N. Rudin, D. Hoeller, P. Reist, M. Hutter. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Conference on Robot Learning, PMLR 164, 2022. arXiv:2109.11978.

[3] N. Hansen. The CMA Evolution Strategy, A Tutorial. 2016. arXiv:1604.00772.

[4] D. Ha. Reinforcement Learning for Improving Agent Design. Artificial Life 25(4), 352-365, 2019. arXiv:1810.03779.

[5] C. Schaff, D. Yunis, A. Chakrabarti, M. R. Walter. Jointly Learning to Construct and Control Agents using Deep Reinforcement Learning. IEEE International Conference on Robotics and Automation, 2019. arXiv:1801.01432.

[6] K. S. Luck, H. Ben Amor, R. Calandra. Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning. Conference on Robot Learning 2019, PMLR 100, 854-869, 2020. arXiv:1911.06832.

[7] A. Gupta, S. Savarese, S. Ganguli, L. Fei-Fei. Embodied intelligence via learning and evolution. Nature Communications 12, 5721, 2021. arXiv:2102.02202.

[8] A. Belmonte-Baeza, J. Lee, G. Valsecchi, M. Hutter. Meta Reinforcement Learning for Optimal Design of Legged Robots. IEEE Robotics and Automation Letters 7(4), 12134-12141, 2022. DOI 10.1109/LRA.2022.3211785. arXiv:2210.02750.

[9] F. Bjelonic, J. Lee, P. Arm, D. Sako, D. Tateo, J. Peters, M. Hutter. Learning-based Design and Control for Quadrupedal Robots with Parallel-Elastic Actuators. IEEE Robotics and Automation Letters 8(3), 1611-1618, 2023. DOI 10.1109/LRA.2023.3234809. arXiv:2301.03509.

[10] A. Ghansah, J. Kim, M. Tucker, A. D. Ames. Humanoid Robot Co-Design, Coupling Hardware Design with Gait Generation via Hybrid Zero Dynamics. 2023. arXiv:2308.10962. The retrieved records describe a conference paper of December 2023 without naming the venue.

[11] A. Zhao, J. Xu, M. Konaković Luković, J. Hughes, A. Spielberg, D. Rus, W. Matusik. RoboGrammar, Graph Grammar for Terrain-Optimized Robot Design. ACM Transactions on Graphics 39(6), SIGGRAPH Asia, 2020. DOI 10.1145/3414685.3417831, as registered in `../../context/literature.md` cluster 33. No arXiv identifier was established.

[12] N. Hansen, A. S. P. Niederberger, L. Guzzella, P. Koumoutsakos. A method for handling uncertainty in evolutionary optimization with an application to feedback control of combustion. IEEE Transactions on Evolutionary Computation 13(1), 180-197, 2009. DOI 10.1109/TEVC.2008.924423.

[13] V. Heidrich-Meisner, C. Igel. Hoeffding and Bernstein Races for Selecting Policies in Evolutionary Direct Policy Search. International Conference on Machine Learning, 401-408, 2009. DOI 10.1145/1553374.1553426.

[14] Y. Jin, J. Branke. Evolutionary Optimization in Uncertain Environments, A Survey. IEEE Transactions on Evolutionary Computation 9(3), 303-317, 2005. No DOI was established by the retrieved sources.

[15] A. Y. Ng, M. I. Jordan. PEGASUS, A Policy Search Method for Large MDPs and POMDPs. Conference on Uncertainty in Artificial Intelligence, 406-415, 2000. arXiv:1301.3878.

[16] M. D. McKay, R. J. Beckman, W. J. Conover. A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. Technometrics 21(2), 239-245, 1979. DOI 10.2307/1268522, as given by ScienceOpen and not confirmed against the publisher.

[17] L. Kish. Survey Sampling. John Wiley and Sons, New York, 1965. ISBN 9780471489009.

[18] N. Hansen, Y. Akimoto, P. Baudis. CMA-ES/pycma, package `cma` version 4.4.4. https://github.com/CMA-ES/pycma, behaviour read from the installed source at `/isaac-sim/kit/python/lib/python3.11/site-packages/cma` in `isaac-lab-base-shreyas-gpu0`.
