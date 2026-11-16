# Co-optimisation Improvement, Fixed Evaluation Widths, Fitness Reliability, an Estimator Free Policy and the Next Course for CMA-ES

> Status, IMPLEMENTED on 2026-10-23, every edit applied as written, the outcome recorded in `/ws/context/copt.md` section 12 and `/ws/context/cmaes.md` section 13. Written 2026-10-23 against the working tree of that date and run `copt_moral/2026-10-22_10-07-12`. Paths are relative to `tron1-rl-isaaclab-cozum/` unless they begin with `/ws/`. Every code change below is given as the text to find and the text to put in its place.

## 1. Introduction

The first full run of the controlled CMA-ES evaluation (`plans/cmaes_evaluation.md`) produced a fitness that rose for four generations and then fell, while the gait it measured kept improving (`/ws/context/cmaes.md` section 12). This plan answers the requests that follow from that run. Section 2 shows that the narrowing of the tracking kernel widths caused the fall, and holds the widths at their starting values during evaluation. Section 3 moves the evaluation to five passes, one per difficulty row, so that every design meets every terrain cell in every generation, gives the fitness computation under that layout, explains how the split-half rank correlation is computed and why it is there, and plans the measurement that decides whether a 4 s horizon is long enough. Section 4 adds a second policy architecture that reads the morphology directly and renames the existing one, for the biped and the quadruped alike. Section 5 reads the run's CMA-ES and PPO logs further, surveys co-design work built on CMA-ES, and sets out the next course. Section 6 replaces the tracking width curriculum's one step gate with a smoothed, two-sided gate and sets its values for the biped and the quadruped from the literature and 58 of the workspace's runs. Section 7 gives the order of application and the checks.

## 2. Evaluation at fixed tracking widths

### 2.1 Why the widths must be fixed

The two velocity tracking terms pay a Gaussian kernel of the command error, $\exp(-e^2/\sigma^2)$, weighted 25 for the linear and 7.5 for the angular term on the biped. The width $\sigma$ sets how much error is still paid. The curriculum `reduce_tracking_rewards_std` starts both widths at 0.4 and multiplies them by 0.95 whenever the population tracks well enough, down to a floor of 0.09 (`environments/environments/tasks/locomotion/mdp/curriculums.py:287-311`), so that the policy is asked for ever finer tracking as it improves. That is the right pressure for training, and the wrong one for measurement.

The evaluation steps the environment through the reward manager, so it scores each generation with whatever width the curriculum holds at that moment (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:886-899`). Within one generation every design meets the same width, and CMA-ES, which consumes only ranks, is not misled. Across generations the fitness then mixes two things, how well the designs walk and how strict the kernel has become. A falling fitness cannot be told apart from a worsening population, the fitness history cannot be compared with any other run, and any rule that reads fitness across generations, a stopping criterion or a plot, reads the curriculum rather than the designs.

### 2.2 The evidence from run `copt_moral/2026-10-22_10-07-12`

The table joins the run's CMA-ES history with its TensorBoard record ([../context/artefacts/copt_moral-2026-10-22_10-07-12-cmaes/evidence_by_generation.csv](../context/artefacts/copt_moral-2026-10-22_10-07-12-cmaes/evidence_by_generation.csv), produced by `extract_evidence.py` in the same directory). The fitness and the errors are those of the controlled evaluation. The widths are those in force at the evaluation. The kernel columns give the tracking kernel at the evaluated mean error, once under the width in force and once under the starting width of 0.4, as an illustration of the width's effect on an unchanged error. The last two columns are measured rather than modelled, the tracking terms and all other terms of the training reward over the 400 iterations before each evaluation, rescaled to the 4 s horizon.

| Gen | Iter | Fitness | Lin err (m/s) | Yaw err (rad/s) | $\sigma_{lin}$ | $\sigma_{ang}$ | $K_{lin}(\sigma)$ | $K_{lin}(0.4)$ | $K_{yaw}(\sigma)$ | $K_{yaw}(0.4)$ | Train tracking | Train other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | 6399 | 91.20 | 0.098 | 0.419 | 0.159 | 0.294 | 0.686 | 0.942 | 0.131 | 0.333 | 74.36 | 10.90 |
| 16 | 6799 | 90.35 | 0.097 | 0.392 | 0.151 | 0.279 | 0.663 | 0.943 | 0.139 | 0.383 | 75.56 | 11.28 |
| 17 | 7199 | 92.88 | 0.090 | 0.365 | 0.143 | 0.265 | 0.677 | 0.951 | 0.151 | 0.435 | 75.51 | 11.35 |
| 18 | 7599 | 93.64 | 0.090 | 0.368 | 0.143 | 0.265 | 0.676 | 0.951 | 0.147 | 0.430 | 76.46 | 11.22 |
| 19 | 7999 | 92.86 | 0.089 | 0.371 | 0.143 | 0.265 | 0.681 | 0.952 | 0.141 | 0.422 | 77.00 | 11.13 |
| 20 | 8399 | 91.94 | 0.088 | 0.369 | 0.136 | 0.265 | 0.661 | 0.953 | 0.144 | 0.427 | 76.88 | 11.06 |
| 21 | 8799 | 89.09 | 0.088 | 0.370 | 0.123 | 0.239 | 0.601 | 0.953 | 0.091 | 0.425 | 75.54 | 10.93 |
| 22 | 9199 | 89.20 | 0.085 | 0.350 | 0.111 | 0.216 | 0.557 | 0.956 | 0.073 | 0.465 | 73.24 | 11.07 |
| 23 | 9599 | 86.20 | 0.084 | 0.361 | 0.105 | 0.205 | 0.531 | 0.957 | 0.045 | 0.443 | 72.55 | 10.76 |
| 24 | 9999 | 85.25 | 0.082 | 0.362 | 0.100 | 0.195 | 0.506 | 0.958 | 0.032 | 0.440 | 70.49 | 10.90 |
| 25 | 10399 | 85.47 | 0.082 | 0.372 | 0.095 | 0.195 | 0.477 | 0.959 | 0.026 | 0.422 | 70.81 | 10.43 |
| 26 | 10799 | 86.01 | 0.083 | 0.360 | 0.095 | 0.195 | 0.468 | 0.958 | 0.033 | 0.445 | 70.81 | 10.81 |
| 27 | 11199 | 87.03 | 0.081 | 0.335 | 0.095 | 0.195 | 0.484 | 0.960 | 0.052 | 0.495 | 71.37 | 10.79 |

The run falls into three phases. Over generations 17 to 19 the widths were constant at 0.143 and 0.265 and the fitness held its peak of 92.9 to 93.6. From generation 20 to generation 25 the widths narrowed in every generation and the fitness fell to 85.5, while the linear error improved from 0.088 to 0.082 m/s and survival stayed between 0.97 and 0.99. From generation 25 the linear width rested at 0.095 and the angular at 0.195, and the fitness rose again, by 1.6 in two generations, as the yaw error improved from 0.372 to 0.335 rad/s. Across all thirteen generations the fitness correlates with the linear width at 0.89 and with the measured tracking reward of training at 0.93, while it correlates with survival at −0.19. It correlates positively with the linear error, at 0.69, which is the signature of the confound, since fitness fell exactly while the error fell. Between the peak at generation 18 and the trough at generation 24 the fitness fell by 8.4, the measured tracking terms of training by 6.0 and all other terms of training together by 0.3, so the narrowing of the widths accounts for most of the fall.

Under the starting width the kernel at the evaluated error rises over the run, from 0.942 to 0.960 for the linear term and from 0.333 to 0.495 for the yaw term, which is the improvement the evaluation was meant to show and did not.

### 2.3 The change, scoring at the starting widths

The evaluation now holds every width that `reduce_tracking_rewards_std` narrows at its configured starting value, 0.4 for both terms of both robots, and restores the training value afterwards. The starting value is read once, when the runner is constructed and before any curriculum step or checkpoint restore has touched the kernels. The floor of 0.09 is not used, because at that width the yaw kernel pays $\exp(-0.37^2/0.09^2) \approx 0$ at the run's typical error, which would remove yaw tracking from the fitness entirely.

Edit R1, `copt_on_policy_runner.py` `__init__`. Find

```python
        self._eval_design: torch.Tensor = self._build_evaluation_design()
```

and replace it with

```python
        self._eval_design: torch.Tensor = self._build_evaluation_design()
        # Every narrowed tracking width as configured, at which every generation is scored.
        self._eval_tracking_widths: dict[str, float] = self._configured_tracking_widths()
```

The two helper methods, `_configured_tracking_widths` and `_fix_tracking_widths`, and the call that holds and restores the widths around the evaluation are part of the replacement of section 3.7, so that the evaluation is replaced in one piece.

Rationale. The widths are read from the curriculum terms rather than written into the runner, so any task with such a curriculum is covered and a task without one leaves the evaluation unchanged. `get_term_cfg` returns the live configuration object, the one the curriculum itself mutates, so writing to it is the operation the curriculum performs, and restoring it in `finally` leaves training exactly as it was. The curriculum cannot move a width during the evaluation because the runner suspends it there. The snapshot is taken in `__init__` because the checkpoint carries the narrowed widths and restores them in `load` (`co_optimisation/co_optimisation/utils/env_state.py:53-62`), after construction.

One consequence follows. At a width of 0.4 the linear kernel is nearly saturated, 0.94 to 0.96 in the table, and its slope at the run's error, $2e K/\sigma^2 \approx 1.1$ per m/s, is about a fifth of its slope of 5.9 at the width of 0.143. Differences in linear tracking between designs therefore weigh less against survival and the penalty terms than they did at the narrowed widths. The yaw kernel stays in its sensitive range. If section 3.8 shows the ranking to have lost reliability, a per-term evaluation width can replace the starting value without changing this mechanism.

## 3. The split-half rank correlation, the evaluation layout, and whether 4 s is enough

### 3.1 The evaluation layout

The evaluation now visits every cell of the terrain grid in every generation, as `cmaes_evaluation.md` section 3.1 specifies. Each of the 80 designs holds 80 instances, environment `80 j + k` being instance `j` of design `k`. The grid has 80 columns, each a separately generated terrain sample, and 5 rows of rising difficulty. The evaluation makes five passes, one per row. In pass `r` every environment stands in row `r`, instance `j` of every design in column `j`, so each cell holds exactly one instance of every design and nothing else. Each cell carries its own command and initial yaw, a point of a 400 point Latin hypercube, shared by the 80 designs standing on it. A reset separates consecutive passes, five in all, and over them every design meets all 400 cells.

The rule for any population follows from one requirement, that a single generation visits every column in every row at least once. With `m` instances per design and `C` columns the evaluation makes `P = ceil(C / m)` column passes in each row, `R P` passes and resets in all, and in column pass `q` instance `j` stands in column `col(q, j) = floor((j P + q) C / (P m))`. The `P m` slots `j P + q` are spread evenly over the `C` columns, so every column is visited, the instances of one pass stand in distinct columns whenever `m <= C`, and the columns visited twice when `m` does not divide `C` are spread evenly across the grid and therefore across the sub-terrains. With more instances than columns the same rule gives one column pass, in which the extra instances repeat columns evenly spread across the grid, each with its own command, so every terrain is still covered and some are evaluated twice. The table gives the cases.

| Instances `m` per design, columns `C` | Column passes `P` | Passes in all | Columns visited twice per row |
|---|---|---|---|
| 80, 80 | 1 | 5 | none, `col = j` |
| 40, 80 | 2 | 10 | none, even then odd columns |
| 48, 80 | 2 | 10 | 16, every fifth column |
| 64, 80 | 2 | 10 | 48 |
| 100, 80 | 1 | 5 | 20, every fourth column |

The single pass applied on 2026-10-22 instead placed instance `j` in row `(j + g) mod 5`, so each column was seen at one difficulty per generation and each design met 80 cells. Section 3.7 gives the code that replaces it.

### 3.2 How the split-half rank correlation is computed

The statistic asks whether two independent halves of the evaluation would rank the designs the same way. It is computed in four steps from the per cell returns $R_{rjk}$, the return of design $k$ on the cell in row $r$ and column $j$.

1. Split the cells into two halves by column, the even columns forming half $A$ and the odd columns half $B$. Each half holds 40 columns in all 5 rows, 200 cells per design.
2. Compute each design's fitness on each half, $F^A_k = \frac{1}{200}\sum_{r}\sum_{j \in A} R_{rjk}$ and $F^B_k$ likewise.
3. Rank the 80 designs by $F^A$, and separately by $F^B$, giving two rank vectors of 80 entries.
4. Report the Pearson correlation of the two rank vectors, the Spearman correlation of $F^A$ and $F^B$.

```text
                 col 0   col 1   col 2   col 3   ...   col 78  col 79
               +-------+-------+-------+-------+-----+-------+-------+
 pass 0  row 0 |   A   |   B   |   A   |   B   | ... |   A   |   B   |
 pass 1  row 1 |   A   |   B   |   A   |   B   | ... |   A   |   B   |
 pass 2  row 2 |   A   |   B   |   A   |   B   | ... |   A   |   B   |
 pass 3  row 3 |   A   |   B   |   A   |   B   | ... |   A   |   B   |
 pass 4  row 4 |   A   |   B   |   A   |   B   | ... |   A   |   B   |
               +-------+-------+-------+-------+-----+-------+-------+
 every cell holds one instance of each of the 80 designs under that cell's command

 half A, 200 cells per design  ->  F^A (80 values)  ->  ranks of A --+
                                                                      +--> Spearman correlation
 half B, 200 cells per design  ->  F^B (80 values)  ->  ranks of B --+
```

The halves are cut by column because a column is the unit that carries a distinct terrain sample, so the two halves share no terrain and no command and are genuinely independent replicates. Alternating the columns rather than cutting the grid in two keeps the sub-terrains balanced, since the generator assigns sub-terrains to contiguous blocks of columns (`IsaacLab/source/isaaclab/isaaclab/terrains/terrain_generator.py:245`), and every half then holds 12 flat columns, 4 of each slope, 2 of each stair type and 8 each of waves and random rough. Every half covers every row because the rows are passes and every column appears in all five. The single pass layout cut its halves into alternating blocks of five instances for the same balance across rows (`copt_on_policy_runner.py:923-933`).

### 3.3 Why it was added

CMA-ES is driven only by the ranking of each generation [1], and the evaluation that produces the ranking is noisy, because the return of a design on a cell depends on the cell, the command and the chaotic course of contact as well as on the design. The planning of the evaluation held two noise remedies in reserve, uncertainty handling and racing, to be applied only if the ranking proved unreliable (`cmaes_evaluation.md` section 2.17). The split-half correlation is the measurement that decides it. It costs nothing, since it reuses the cells already simulated, and it answers the one question that matters to CMA-ES, whether the order of the designs would survive a second, independent evaluation.

### 3.4 What it measures

Write each half's fitness as a true merit plus noise, $F^h_k = \mu_k + \epsilon^h_k$, where $\mu_k$ varies across designs with variance $\sigma_d^2$ and the noise of a half has variance $\sigma_h^2$, independent between halves. Because every design meets the same cells and commands, any effect of a cell or a command that is shared by all designs cancels from the ranking [2]. What remains in $\epsilon$ is the part of a design's return that depends on the particular cell, the design by condition interaction and the chaotic divergence of contact dynamics. Then

$$\mathrm{corr}(F^A, F^B) = \frac{\sigma_d^2}{\sigma_d^2 + \sigma_h^2} = \rho_{half},$$

the reliability of a half. The full fitness averages twice as many cells, so its noise variance is $\sigma_h^2/2$ and its reliability is

$$\rho_{full} = \frac{\sigma_d^2}{\sigma_d^2 + \sigma_h^2/2} = \frac{2\rho_{half}}{1 + \rho_{half}}.$$

With $c'$ cells instead of $c$ the same algebra gives $\rho(c') = \dfrac{(c'/c)\,\rho_{full}}{1 + (c'/c - 1)\rho_{full}}$. The two variance components can be read from the halves directly, $\hat\sigma_d^2 = \mathrm{cov}_k(F^A_k, F^B_k)$ and $\hat\sigma_h^2 = \tfrac{1}{2}\mathrm{var}_k(F^A_k - F^B_k)$. The rank form is used because CMA-ES consumes ranks, and for roughly normal fitness it is close to the linear correlation.

### 3.5 Why it matters to CMA-ES

CMA-ES moves its mean toward the best $\mu$ of $\lambda$ candidates, weighted by rank, and adapts its covariance and step size from the same ranking [1]. The correlation between the observed fitness and the true merit is $\sqrt{\rho_{full}}$. Under the single pass layout, at generation 15 of the run, the split-half value of 0.55 gives $\rho_{full} = 0.71$ and a correlation of 0.84 with merit, so selection was mostly right. From generation 20 the values between −0.12 and 0.16 give $\rho_{full}$ below 0.28 and a correlation below 0.53, at worst zero, so the mean update became largely a random walk and the step size control lost the evidence it needs. The five pass layout gives each design 400 cells instead of 80, which by the formula above, if cells are independent, raises a $\rho_{full}$ of 0.71 to 0.92 and one of 0.10 to 0.36.

### 3.6 Measures that raise it

The formula has two levers, a larger spread of true merit $\sigma_d^2$ and a smaller noise $\sigma_h^2$.

1. Keep the population spread. A converging search shrinks $\sigma_d^2$ by itself. Section 5.1 shows the design spread fell fivefold over the run, which near an optimum shrinks merit differences by far more, so a falling split-half value is partly the signature of convergence. It is then a stopping signal rather than a fault.
2. More cells per design. Halving the noise variance needs twice the cells. The five pass layout already multiplies the cells by five, and a further increase needs more passes or more environments per design.
3. A longer horizon, if the per cell return becomes more design specific as the walk settles. Section 3.8 decides this.
4. A fitness with less noise per unit of merit. Survival and tracking error are the parts of the return most tied to the design, and the penalty terms add cell noise. Section 5.1 shows survival and linear error together account for most of the fitness spread in late generations.
5. Remove variation that is not shared across designs. Commands, cells, headings and initial states are already shared [2]. The policy is deterministic during evaluation (`act_inference`), so the remaining variation is the physics itself.
6. Remove the bias toward the training population, section 5.2, which adds a design dependent offset that is not merit.

### 3.7 The fitness computation under the five pass layout, and the code

Under the pass layout the fitness of design $k$ is its mean return over all $R P m$ visited cells,

$$F_k = \frac{1}{R\,P\,m}\sum_{r=0}^{R-1}\sum_{q=0}^{P-1}\sum_{j=0}^{m-1} R_{rqjk},$$

where $R_{rqjk}$ is the return of instance $j$ of design $k$ in row pass $r$ and column pass $q$, with $R = 5$ rows, $P = 1$ column pass and $m = 80$ instances at the run's configuration. Every row now contributes the same $P m$ cells, so this plain mean equals the mean over rows of the row means, and the stratification by row that the single pass needed for its unequal row counts becomes automatic. A fall ends the accumulation of its cell only, the next pass starting the instance afresh. A column visited twice, when $m$ does not divide $C$, counts twice, a small and evenly spread overweighting. Survival is the fraction of the visited cells survived to the end of the horizon, the tracking errors are averaged over the steps alive in all five passes, and the per row return and survival are the means over all cells of that row's passes, so the existing logging keys `eval/return_row_<r>` and `eval/survival_row_<r>` keep their meaning of one value per difficulty row. The logging method needs no change.

Edit R2, `copt_on_policy_runner.py`, the imports. Find

```python
import math
import os
```

and replace it with

```python
import itertools
import math
import os
```

Then replace the three methods from `    def _build_evaluation_design(self) -> torch.Tensor:` (`:776`) through the closing brace of the dictionary `_evaluate_population` returns (`:942`), which spans `_build_evaluation_design`, `_evaluation_cells` and `_evaluate_population`, with

```python
    def _configured_tracking_widths(self) -> dict[str, float]:
        """The width of every kernel reduce_tracking_rewards_std narrows, as configured before any narrowing."""
        env = self.env.unwrapped
        widths = {}
        for cfg in env.curriculum_manager._term_cfgs:
            if getattr(cfg.func, "__name__", "") == "reduce_tracking_rewards_std":
                name = cfg.params["term_name"]
                widths[name] = env.reward_manager.get_term_cfg(name).params["std"]
        return widths

    def _fix_tracking_widths(self, env) -> list[tuple[object, float]]:
        """Set each narrowed kernel to its configured width, returning the training widths to restore."""
        saved = []
        for name, width in self._eval_tracking_widths.items():
            term = env.reward_manager.get_term_cfg(name)
            saved.append((term, term.params["std"]))
            term.params["std"] = width
        return saved

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
        survival, and the wall time. Every narrowed tracking kernel is held at its configured
        width. The widths, the terrain assignment, the step counter and the interval event timers
        are restored, and the reset of the design swap that follows overwrites everything else.
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
        trace = (
            torch.zeros(rows, passes, steps, env.num_envs, dtype=torch.float16, device=self.device)
            if self._eval_save_trace
            else None
        )
        widths = self._fix_tracking_widths(env)
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
                for t in range(steps):
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
                    if trace is not None:
                        trace[r, q, t] = alive * rewards
                    length[r, q] += counted
                    alive &= ~dones
                survived[r, q] = alive.float()
        finally:
            for term, std in widths:
                term.params["std"] = std
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
        if trace is not None and self.log_dir is not None:
            torch.save(
                {"reward": trace.cpu(), "column": col.cpu(), "design": design.cpu(), "num_designs": n,
                 "step_dt": env.step_dt},
                os.path.join(self.log_dir, "designs", f"eval_trace_{self.generation:04d}.pt"),
            )
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

Rationale. The layout puts one instance of every design in each cell of each pass, so every design meets the same 400 cells and every cell compares all 80 designs directly. The passes are sequential rather than spread across the instances because a column can hold one instance of every design in every row only if all designs move through the rows together, and the reset at the start of each pass is the only way to relocate every robot at once. `_evaluation_columns` replaces `_evaluation_cells`, since the row is now the pass index, and returns the column of every instance in every column pass under the rule of section 3.1, so that a single generation covers every column in every row whether the instances are fewer than, equal to or more than the columns, and every generation is scored on the same cells. The Latin hypercube grows from 80 to `R P m` points, 400 at the run's configuration, so that every visited cell carries its own command, built once in `__init__` from the terrain the scene already holds, a plane giving one row. The widths of section 2.3 are fixed once around all five passes and restored in `finally`, and the trace of section 3.8 records every pass. The cost rises from one pass to five, about 265 s per generation against 2230 s of training at the run's measured rates ([../context/artefacts/copt_moral-2026-10-22_10-07-12-cmaes/timing_medians.txt](../context/artefacts/copt_moral-2026-10-22_10-07-12-cmaes/timing_medians.txt)), 12 per cent of a generation.

### 3.8 Is a 4 s window too short

The horizon is 200 policy steps per pass (`scripts/rsl_rl/train.py:315-316`, step 0.02 s). Every instance starts at rest with its full command, so roughly the first second is an acceleration transient, and the remaining three seconds hold only a few gait cycles. A window this short is plausibly dominated by the start, which every design shares, rather than by steady walking, which is where designs differ. Training episodes last 20 s. The question is empirical and cheap to settle, because the reliability of every shorter horizon can be computed from one long evaluation.

The measurement records the per step reward of every environment in every pass of a 16 s evaluation, and an offline script computes the fitness, $\rho_{half}$, $\rho_{full}$, $\hat\sigma_d^2$ and $\hat\sigma_h^2$ for horizons of 1, 2, 4, 8 and 16 s and for subsets of 20, 40 and all 80 columns. The horizon at which $\rho_{full}$ stops rising is the one to adopt. Repeating the analysis at an early and a late generation separates the effect of convergence from the effect of the horizon.

Edit R3, `copt_on_policy_runner.py` `__init__`. Find

```python
        self._eval_seed: int = copt_cfg.get("eval_seed", 0)
```

and replace it with

```python
        self._eval_seed: int = copt_cfg.get("eval_seed", 0)
        # Writes designs/eval_trace_<g>.pt, the per step reward of every environment in every pass.
        self._eval_save_trace: bool = copt_cfg.get("eval_save_trace", False)
```

The recording itself is part of the replacement of section 3.7. The trace is `R P` passes by 800 steps by 6400 environments in half precision, 51 MB per generation at 16 s and five passes, and is off by default, so no existing run changes. A diagnostic run sets `"eval_duration_s": 16.0, "eval_save_trace": True` in the `copt` block of `train.py` (`scripts/rsl_rl/train.py:315-316`), which makes the evaluation about 17 minutes per generation, acceptable for a diagnostic of a few generations. The analysis script is new, `scripts/analysis/eval_reliability.py`.

```python
"""Reliability of the controlled CMA-ES evaluation against horizon and column count.

Reads designs/eval_trace_<g>.pt written with eval_save_trace and prints, per horizon and
number of columns, the split-half rank correlation over even and odd columns, the full
reliability and the two variance components of section 3.4 of plans/copt_improvement.md.

Usage:
    python scripts/analysis/eval_reliability.py <eval_trace.pt>
"""

import argparse

import torch


def ranks(x):
    return x.argsort().argsort().float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    a = p.parse_args()
    d = torch.load(a.trace)
    n, dt = d["num_designs"], d["step_dt"]
    reward = d["reward"].float()                              # (R, P, T, m * n)
    rows, passes, steps_max = reward.shape[:3]
    reward = reward.view(rows, passes, steps_max, -1, n)       # (R, P, T, m, n), env = j * n + k
    col = d["column"]                                          # (P, m)
    pair = col // 2
    print("horizon_s columns split_half rho_full var_design var_noise_half")
    for h in (1, 2, 4, 8, 16):
        steps = min(int(round(h / dt)), steps_max)
        ret = reward[:, :, :steps].sum(2)                      # (R, P, m, n)
        for k in (4, 2, 1):
            keep = pair % k == 0                               # whole column pairs, both halves kept
            even, odd = keep & (col % 2 == 0), keep & (col % 2 == 1)
            fa, fb = ret[:, even].mean((0, 1)), ret[:, odd].mean((0, 1))
            sh = torch.corrcoef(torch.stack([ranks(fa), ranks(fb)]))[0, 1].item()
            full = 2 * sh / (1 + sh) if sh > -1 else float("nan")
            var_d = torch.cov(torch.stack([fa, fb]))[0, 1].item()
            var_h = 0.5 * (fa - fb).var().item()
            print(f"{h:9.0f} {col[keep].unique().numel():7d} {sh:10.3f} {full:8.3f} {var_d:10.4f} {var_h:14.4f}")
        if steps == steps_max:
            break


if __name__ == "__main__":
    main()
```

## 4. A policy that reads the morphology directly

### 4.1 What exists and what is asked

The present `CoptActorCritic` regresses the morphology from the proprioceptive state through an estimator and feeds the estimate to the actor (`co_optimisation/co_optimisation/modules/copt_actor_critic.py:39-198`), and `CoptPPO` adds the regression loss (`co_optimisation/co_optimisation/algorithms/copt_ppo.py:256-259`). The request is to rename it `CoptMorALActorCritic` with its policy type `COPT-MORAL`, and to add a new `CoptActorCritic` under policy type `COPT` whose actor takes the true `morphologyObs` group as input and has no estimator, with the NaN reporting of the existing class, for both robots, through `djinn`.

The evaluation of the code found five places the change must reach, and one it need not. `CoptPPO` unpacks `(actions, estimate)` from `act` and always computes the estimator loss, so it must accept an actor without one. The runner passes the encoder configuration as a fourth positional argument (`copt_on_policy_runner.py:493-500`), so the new class must accept and ignore it. The symmetry augmentation already mirrors every observation group, `morphologyObs` included, and raises on any unmapped term (`environments/environments/tasks/locomotion/mdp/symmetry/common.py:272-314`), so it needs no change. `train.py` and `play.py` choose the class from the policy type, and `play.py` also fixes the family to the biped (`scripts/rsl_rl/play.py:1190-1192`), which would build the wrong generator for a quadruped replay. The configuration, the registry and `djinn` carry the names.

### 4.2 Names

| Item | MoRAL, estimator | New, direct |
|---|---|---|
| Actor critic | `CoptMorALActorCritic` | `CoptActorCritic` |
| Policy type | `COPT-MORAL` | `COPT` |
| Biped agent configuration | `SFCoptMorALRunnerCfg` | `SFCoptRunnerCfg` |
| Quadruped agent configuration | `PFQuadrupedCoptMorALPPORunnerCfg` | `PFQuadrupedCoptPPORunnerCfg` |
| Biped tasks | `Isaac-Limx-SF-Copt-MoRAL-{Flat,Rough,Rough-Play}-v0`, unchanged | `Isaac-Limx-SF-Copt-{Flat,Rough,Rough-Play}-v0` |
| Quadruped tasks | `Isaac-Quadruped-Copt-MoRAL-{Flat,Rough,Rough-Play}-v0` | `Isaac-Quadruped-Copt-{Flat,Rough,Rough-Play}-v0` |
| `djinn` modes | `copt-moral`, `quadruped-copt-moral` | `copt`, `quadruped-copt` |
| Experiment directory | `copt_moral`, `quadruped_copt`, unchanged | `copt_direct`, `quadruped_copt_direct` |

The MoRAL experiment directories keep their names so that existing log trees and resumes stay where they are. The quadruped's `quadruped_copt` therefore keeps holding MoRAL runs, and the new architecture takes a distinct name rather than inheriting a directory full of the other one's history. The two architectures share every environment configuration, so a task differs from its MoRAL twin in its runner configuration alone.

Three behaviours change by request and are recorded here. The policy type `COPT` and the task ids `Isaac-Quadruped-Copt-*-v0` now name the new architecture. A run started before the change must be resumed or replayed with `COPT-MORAL` and the MoRAL task id, and passing `COPT` to an old checkpoint fails loudly when the state dict lacks the estimator's weights, rather than silently. The `djinn` mode `copt` now launches the new architecture.

### 4.3 The actor critic module

Edit A1, `co_optimisation/co_optimisation/modules/copt_actor_critic.py`. Find

```python
class CoptActorCritic(ActorCritic):
    """Actor-critic whose estimator is configured entirely by observation sets.
```

and replace it with

```python
def report_nan(tensors: TensorDict | dict[str, torch.Tensor]) -> None:
    """Print each named tensor that holds a NaN with the last axis indices it occurs at."""
    for key, item in tensors.items():
        if item.isnan().any():
            nan_indices = torch.nonzero(item.isnan(), as_tuple=False)[:, -1].unique()
            print(f"NaN in {key} at indices:\n{nan_indices}")


class CoptMorALActorCritic(ActorCritic):
    """Actor-critic whose estimator is configured entirely by observation sets.
```

Edit A2, the same file, the old class's message and its NaN helper. Find

```python
                f"CoptActorCritic requires the '{required}' observation set. "
```

and replace it with

```python
                f"CoptMorALActorCritic requires the '{required}' observation set. "
```

then find

```python
    @staticmethod
    def _report_nan(tensors: TensorDict | dict[str, torch.Tensor]) -> None:
        for key, item in tensors.items():
            if item.isnan().any():
                nan_indices = torch.nonzero(item.isnan(), as_tuple=False)[:, -1].unique()
                print(f"NaN in {key} at indices:\n{nan_indices}")
```

and replace it with

```python
    _report_nan = staticmethod(report_nan)
```

Edit A3, append to the end of the same file.

```python
class CoptActorCritic(ActorCritic):
    """Actor-critic whose actor reads the true morphology directly, with no estimator.

    The morphology enters as one more group of the policy set, obs_groups["policy"]
    naming morphologyObs, so the inherited ActorCritic sizes, normalises and
    concatenates it. act returns (actions, None) because CoptPPO unpacks an
    estimator output, and adds no estimator loss for None.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        encoder_cfg: dict | None = None,
        **kwargs: dict[str, Any],
    ):
        assert "morphologyObs" in obs_groups["policy"], (
            "CoptActorCritic reads the morphology in its actor, so obs_groups['policy'] "
            f"must name 'morphologyObs'. Found {obs_groups['policy']}"
        )
        super().__init__(obs, obs_groups, num_actions, **kwargs)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        super()._update_distribution(obs)
        report_nan({"normalised actor input": obs, "mean": self.distribution.mean,
                    "std": self.distribution.stddev})

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, None]:
        report_nan({group: obs[group] for group in self.obs_groups["policy"]})
        return super().act(obs, **kwargs), None

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        value = self.critic(critic_obs)
        report_nan({"normalised critic input": critic_obs, "value": value})
        return value
```

Edit A4, `co_optimisation/co_optimisation/modules/__init__.py`. Find

```python
from co_optimisation.modules.copt_actor_critic import CoptActorCritic
```

and replace it with

```python
from co_optimisation.modules.copt_actor_critic import CoptActorCritic, CoptMorALActorCritic
```

Rationale. The new class reports NaN at the same three points as the old one, each raw policy group by name before the actor, the normalised actor input with the distribution after it, and the critic input with the value, so a NaN is traced to its observation group in either architecture. Moving the helper to module level lets both classes share one copy, and the staticmethod alias keeps every call site of the old class unchanged. The inherited `act_inference` already serves the evaluation and the symmetry loss. The actor input is not normalised, as in the MoRAL runs, so the two architectures differ only in where the morphology comes from. The inertia channels of `morphologyObs` are of order $10^{-4}$ to $10^{-2}$ and will be faint without normalisation, and enabling `actor_obs_normalization` on the new configuration is the remedy if the policy is found to ignore them, at the cost of that clean comparison.

### 4.4 The algorithm and the runner

Edit A5, `co_optimisation/co_optimisation/algorithms/copt_ppo.py`. Find

```python
            model_estimation_loss = torch.nn.functional.mse_loss(
                predicted_model_info,
                self.policy.get_estimator_target(obs_batch).detach(),
            )
```

and replace it with

```python
            if predicted_model_info is None:
                # An actor critic without an estimator, CoptActorCritic, adds no loss.
                model_estimation_loss = torch.zeros((), device=self.device)
            else:
                model_estimation_loss = torch.nn.functional.mse_loss(
                    predicted_model_info,
                    self.policy.get_estimator_target(obs_batch).detach(),
                )
```

Edit A6, `co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`. Find

```python
from co_optimisation.modules import CoptActorCritic
```

and replace it with

```python
from co_optimisation.modules import CoptActorCritic, CoptMorALActorCritic  # noqa: F401, eval below
```

then find

```python
        actor_critic: CoptActorCritic = actor_critic_class(
```

and replace it with

```python
        actor_critic: CoptActorCritic | CoptMorALActorCritic = actor_critic_class(
```

Rationale. The MoRAL path through `CoptPPO` is unchanged, since its estimator output is never None. The runner resolves the class by `eval` of the configured name, so both classes must be importable in its module. The logged `Loss/model_estimation_loss` reads zero under the new architecture, which keeps the TensorBoard layout identical across the two.

### 4.5 The agent configurations

Edit A7, `environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py`. Find

```python
@configclass
class SFCoptMorphologyRunnerCfg(SFCoptBaseRunnerCfg):
    """Experiment 1, morphology inferred from proprioceptive history alone."""
```

and replace it with

```python
@configclass
class SFCoptRunnerCfg(SFCoptBaseRunnerCfg):
    """The true morphology read by the actor directly, no estimator, for CoptActorCritic."""

    experiment_name: str = "copt_direct"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "morphologyObs"],
        "critic": ["critic"],
    }


@configclass
class SFCoptMorALRunnerCfg(SFCoptBaseRunnerCfg):
    """Experiment 1, morphology inferred from proprioceptive history alone, for CoptMorALActorCritic."""
```

Edit A8, `environments/environments/tasks/locomotion/agents/quadruped_rsl_rl_ppo_cfg.py`. Find

```python
@configclass
class PFQuadrupedCoptPPORunnerCfg(PFQuadrupedPPORunnerCfg):
    experiment_name: str = "quadruped_copt"
```

and replace it with

```python
@configclass
class PFQuadrupedCoptPPORunnerCfg(PFQuadrupedPPORunnerCfg):
    """The true morphology read by the actor directly, no estimator, for CoptActorCritic."""

    experiment_name: str = "quadruped_copt_direct"
    max_iterations: int = 45000
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "morphologyObs"],
        "critic": ["critic"],
    }
    # Unused by CoptActorCritic, read unconditionally by CoptOnPolicyRunner.__init__.
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=19,
        hidden_dims=[1024, 512, 256],
        activation="elu",
        orthogonal_init=False,
    )


@configclass
class PFQuadrupedCoptMorALPPORunnerCfg(PFQuadrupedPPORunnerCfg):
    experiment_name: str = "quadruped_copt"
```

Rationale. The critic group of both robots already carries the link lengths, masses, inertias and nominal pose, so the critic set is unchanged. The biped's new configuration inherits the encoder block from `SFCoptBaseRunnerCfg`, and the quadruped's states one because the runner reads `train_cfg["encoder"]` whatever the class (`copt_on_policy_runner.py:144`).

### 4.6 The task registry

Edit A9, `environments/environments/tasks/locomotion/robots/__init__.py`, the imports. Find

```python
    SFCoptMorphologyRunnerCfg,
```

and replace it with

```python
    SFCoptMorALRunnerCfg,
    SFCoptRunnerCfg,
```

then find

```python
from environments.tasks.locomotion.agents.quadruped_rsl_rl_ppo_cfg import (
    PFQuadrupedCoptPPORunnerCfg,
```

and replace it with

```python
from environments.tasks.locomotion.agents.quadruped_rsl_rl_ppo_cfg import (
    PFQuadrupedCoptMorALPPORunnerCfg,
    PFQuadrupedCoptPPORunnerCfg,
```

Edit A10, the runner instances. Find

```python
limx_sf_copt_moral_runner_cfg = SFCoptMorphologyRunnerCfg()
```

and replace it with

```python
limx_sf_copt_moral_runner_cfg = SFCoptMorALRunnerCfg()

limx_sf_copt_runner_cfg = SFCoptRunnerCfg()
```

then find

```python
quadruped_copt_runner_cfg = PFQuadrupedCoptPPORunnerCfg()
```

and replace it with

```python
quadruped_copt_moral_runner_cfg = PFQuadrupedCoptMorALPPORunnerCfg()

quadruped_copt_runner_cfg = PFQuadrupedCoptPPORunnerCfg()
```

Edit A11, the biped registrations. Find

```python
#############################
# SD_BRS1 Environments
#############################
```

and replace it with

```python
gym.register(
    id="Isaac-Limx-SF-Copt-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_copt_runner_cfg,
    },
)

#############################
# SD_BRS1 Environments
#############################
```

Edit A12, the quadruped registrations. Find the whole block from `# Quadruped Copt Flat Environment` to the end of the `Isaac-Quadruped-Copt-Rough-Play-v0` registration (`robots/__init__.py:646-680`) and replace it with

```python
######################################
# Quadruped Copt Environments, MoRAL
######################################
gym.register(
    id="Isaac-Quadruped-Copt-MoRAL-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_moral_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Copt-MoRAL-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_moral_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Copt-MoRAL-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_copt_moral_runner_cfg,
    },
)

######################################
# Quadruped Copt Environments, direct
######################################
gym.register(
    id="Isaac-Quadruped-Copt-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Copt-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Copt-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)
```

`train.py` selects the design family by the substrings `Isaac-Quadruped-Copt` and `Isaac-Limx-SF-Copt` (`scripts/rsl_rl/train.py:245-248`), which every old and new id contains, so that table needs no change.

### 4.7 The launch scripts

Edit A13, `scripts/rsl_rl/cli_args.py`. Find

```python
        choices=["PPO", "HIMPPO", "COPT", "quadruped-debug", "go1-debug", "go1-default-debug"],
        help=(
            "Type of the policy to use. Can be one of PPO, HIMPPO, COPT, quadruped-debug, go1-debug or go1-default-debug, "
        ),
```

and replace it with

```python
        choices=["PPO", "HIMPPO", "COPT", "COPT-MORAL", "quadruped-debug", "go1-debug", "go1-default-debug"],
        help=(
            "Type of the policy to use. Can be one of PPO, HIMPPO, COPT, COPT-MORAL, quadruped-debug, "
            "go1-debug or go1-default-debug. COPT reads the true morphology, COPT-MORAL estimates it."
        ),
```

Edit A14, `scripts/rsl_rl/train.py`. Find

```python
    runner = None
    if args_cli.policy_type == "COPT":
```

and replace it with

```python
    runner = None
    _COPT_ACTOR_CRITIC = {"COPT": "CoptActorCritic", "COPT-MORAL": "CoptMorALActorCritic"}
    if args_cli.policy_type in _COPT_ACTOR_CRITIC:
```

then find

```python
        agent_cfg.policy.class_name = "CoptActorCritic"
        agent_cfg.algorithm.class_name = "CoptPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        # 1.0 is the unweighted sum every run to date trained with.
```

and replace it with

```python
        agent_cfg.policy.class_name = _COPT_ACTOR_CRITIC[args_cli.policy_type]
        agent_cfg.algorithm.class_name = "CoptPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        # 1.0 is the unweighted sum every run to date trained with.
```

then find

```python
    if args_cli.policy_type == "COPT":
        # agent.yaml serialises the configclass, without the copt block the runner receives.
```

and replace it with

```python
    if args_cli.policy_type in _COPT_ACTOR_CRITIC:
        # agent.yaml serialises the configclass, without the copt block the runner receives.
```

Edit A15, `scripts/rsl_rl/play.py`. Find

```python
    elif args_cli.policy_type == "COPT":

        _family = "biped"
```

and replace it with

```python
    elif args_cli.policy_type in ("COPT", "COPT-MORAL"):

        _family = "quadruped" if "Isaac-Quadruped-Copt" in args_cli.task else "biped"
```

then find

```python
        agent_cfg.policy.class_name = "CoptActorCritic"
        agent_cfg.algorithm.class_name = "CoptPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["copt"] = {
```

and replace it with

```python
        agent_cfg.policy.class_name = {"COPT": "CoptActorCritic", "COPT-MORAL": "CoptMorALActorCritic"}[
            args_cli.policy_type
        ]
        agent_cfg.algorithm.class_name = "CoptPPO"
        agent_cfg_dict = agent_cfg.to_dict()
        agent_cfg_dict["copt"] = {
```

The other assumptions of the `play.py` branch, 256 individuals and a two parameter generator, are left as they stand, since the checkpoint restores the population it was saved with. They are recorded as a defect of that branch rather than widened here.

Edit A16, `/ws/djinn`, the training chain. Find

```bash
                elif [[ "$3" == "copt" ]]; then
                    task="Isaac-Limx-SF-Copt-MoRAL-Rough-v0"
                    policy_type="COPT"
                    copt_params="--ea_update_interval 400 --ea_late_start 6000 --copt_num_individuals 80"
```

and replace it with

```bash
                elif [[ "$3" == "copt" ]]; then
                    task="Isaac-Limx-SF-Copt-Rough-v0"
                    policy_type="COPT"
                    copt_params="--ea_update_interval 400 --ea_late_start 6000 --copt_num_individuals 80"
                elif [[ "$3" == "copt-moral" ]]; then
                    task="Isaac-Limx-SF-Copt-MoRAL-Rough-v0"
                    policy_type="COPT-MORAL"
                    copt_params="--ea_update_interval 400 --ea_late_start 6000 --copt_num_individuals 80"
```

then find

```bash
                elif [[ "$3" == "quadruped-copt" ]]; then
                    task="Isaac-Quadruped-Copt-Rough-v0"
                    policy_type="COPT"
                    copt_params="--ea_update_interval 400 --ea_late_start 6000 --copt_num_individuals 80"
```

and replace it with

```bash
                elif [[ "$3" == "quadruped-copt" ]]; then
                    task="Isaac-Quadruped-Copt-Rough-v0"
                    policy_type="COPT"
                    copt_params="--ea_update_interval 400 --ea_late_start 6000 --copt_num_individuals 80"
                elif [[ "$3" == "quadruped-copt-moral" ]]; then
                    task="Isaac-Quadruped-Copt-MoRAL-Rough-v0"
                    policy_type="COPT-MORAL"
                    copt_params="--ea_update_interval 400 --ea_late_start 6000 --copt_num_individuals 80"
```

then find

```bash
                    echo "expected one of: base base-urdf berkeley him him-urdf copt"
                    echo "                 brs brs-simplified quadruped"
                    echo "                 quadruped-flat quadruped-him quadruped-copt"
```

and replace it with

```bash
                    echo "expected one of: base base-urdf berkeley him him-urdf copt copt-moral"
                    echo "                 brs brs-simplified quadruped"
                    echo "                 quadruped-flat quadruped-him quadruped-copt quadruped-copt-moral"
```

Edit A17, `/ws/djinn`, the play chain. Find

```bash
                elif [[ "$3" == "copt" ]]; then
                    task="Isaac-Limx-SF-Copt-MoRAL-Rough-Play-v0"
                    policy_type="COPT"
```

and replace it with

```bash
                elif [[ "$3" == "copt" ]]; then
                    task="Isaac-Limx-SF-Copt-Rough-Play-v0"
                    policy_type="COPT"
                elif [[ "$3" == "copt-moral" ]]; then
                    task="Isaac-Limx-SF-Copt-MoRAL-Rough-Play-v0"
                    policy_type="COPT-MORAL"
```

then find

```bash
                elif [[ "$3" == "quadruped-copt" ]]; then
                    task="Isaac-Quadruped-Copt-Rough-Play-v0"
                    policy_type="COPT"
```

and replace it with

```bash
                elif [[ "$3" == "quadruped-copt" ]]; then
                    task="Isaac-Quadruped-Copt-Rough-Play-v0"
                    policy_type="COPT"
                elif [[ "$3" == "quadruped-copt-moral" ]]; then
                    task="Isaac-Quadruped-Copt-MoRAL-Rough-Play-v0"
                    policy_type="COPT-MORAL"
```

then find

```bash
                    echo "expected one of: base base-urdf berkeley him copt brs"
                    echo "                 brs-simplified quadruped quadruped-flat quadruped-him"
                    echo "                 quadruped-copt kscale kscale-rough kscale-him"
```

and replace it with

```bash
                    echo "expected one of: base base-urdf berkeley him copt copt-moral brs"
                    echo "                 brs-simplified quadruped quadruped-flat quadruped-him"
                    echo "                 quadruped-copt quadruped-copt-moral kscale kscale-rough kscale-him"
```

### 4.8 Changes found unnecessary, and the documents to update

The symmetry module, the observation configurations of both robots and the design generators need no change, and the evaluation of section 3.7 serves both architectures alike, calling only `act_inference`. The live project documents name the old class and task ids and are updated in the same pass, `ARCHITECTURE.md` of this repository and `/ws/ARCHITECTURE.md`, `/ws/CO_OPTIMISATION.md`, and `/ws/context/copt.md`. Plans, artefacts and context entries that record past runs keep the names those runs used.

## 5. The run's logs, the literature, and the next course

### 5.1 What the CMA-ES logs show beyond section 12 of the context

Each generation's table `designs/cmaes_generation_<g>.csv` gives the fitness of all 80 designs. A regression of fitness on the four design parameters, linear and squared terms with standardised inputs, tests per generation how much of the fitness the designs explain, against a null of 300 permutations.

| Generation | Fitness std | $R^2$ quadratic | Permutation p | corr(fitness, distance from mean) | Thigh scale std | Shank scale std |
|---|---|---|---|---|---|---|
| 15 | 1.71 | 0.71 | < 0.003 | −0.75 | 0.106 | 0.107 |
| 17 | 1.11 | 0.57 | < 0.003 | −0.64 | 0.066 | 0.063 |
| 19 | 1.14 | 0.44 | < 0.003 | −0.61 | 0.055 | 0.063 |
| 21 | 1.07 | 0.34 | < 0.003 | −0.45 | 0.038 | 0.036 |
| 23 | 0.86 | 0.21 | 0.027 | −0.13 | 0.026 | 0.023 |
| 25 | 0.79 | 0.11 | 0.38 | 0.01 | 0.022 | 0.018 |

Four findings follow. First, the search did converge. The spread of the thigh and shank scales fell fivefold, to about 2 % of their nominal lengths, even though the step size CMA-ES reports stayed between 0.21 and 0.25, the covariance shrinking instead. Second, early generations carry a strong and significant signal, every squared coefficient negative, so fitness falls in every direction away from the population mean. Third, that signal fades as the spread shrinks, and from generation 22 the designs explain no more than chance, which is the loss of $\sigma_d^2$ that section 3.4 predicts and which accounts for the collapse of the split-half value. Fourth, within a generation the fitness tracks survival (correlation 0.40 to 0.74) and linear tracking error (−0.55 to −0.88) far more than yaw error (−0.28 to −0.65).

The PPO logs show a policy still improving through the search. The action noise fell from 0.60 to 0.43, the trunk contact share of terminations from 3.3 % to 1.3 %, and the explained variance stayed between 0.89 and 0.91. The command and push curricula had saturated by iteration 2800, while the tracking widths and the terrain level kept moving, the latter from 1.42 to 1.52.

### 5.2 The bias toward the training population

A shared policy is trained on the designs the search proposes, so it performs best on designs near the population mean. Any design away from the mean is then scored partly on its unfamiliarity to the policy rather than on its merit, and the negative curvature centred on the mean in every early generation is exactly what that bias predicts, as well as what a genuine optimum predicts. Co-design work identifies this as the central pathology of co-optimising a body with a controller adapted to it [3], and it is a known cause of premature convergence of the morphology [4]. The design moved little in this run, the thigh scale ending at 1.02 and the shank at 0.95, so the run cannot yet say whether it found an optimum or stayed near where the policy was best.

The two readings are separated by evaluating one frozen policy on designs spread over the whole search box. The checkpoint at iteration 6000 is the natural probe, because `randomise_before_late_start` trained it on random designs across the box (`params/copt_agent.yaml:78`), and comparing its landscape with that of the final checkpoint shows how far the search reshaped the policy around its own population. The probe can be run without new code by resuming each checkpoint with `--design_search random` and a short `--ea_update_interval`, so that each generation evaluates 80 random designs with the controlled protocol and writes them to `designs/cmaes_generation_<g>.csv`. That resume path has not been exercised and must first pass a smoke test.

### 5.3 What co-design work with CMA-ES does

The closest published construction is that of Belmonte-Baeza and colleagues, who optimise the thigh and shank scales of a quadruped with CMA-ES over 30 generations of 35 designs, scoring each design with a policy meta-trained over the whole design range and adapted briefly to it, on 250 transitions in each of 300 environments with random commands and terrain [5]. Bjelonic and colleagues likewise train one design-conditioned policy over the design space and then optimise the design against that frozen policy, averaging the reward over 1000 episodes [6]. Both decouple the distribution the policy is trained on from the distribution the optimiser searches, which is what removes the bias of section 5.2. Strgar and Kriegman reach the same conclusion from pretraining a morphology-agnostic controller and evaluating candidates zero-shot [7], and Bohlinger and Peters go further, using the value gradients of an embodiment-aware critic as the design signal [8].

The earlier co-design works spend far more evaluation per candidate than this workspace. Ha averages 16 rollouts per candidate in a population of 192 [9], Luck and colleagues average the critic over a batch of start states and found no significant difference between particle swarm optimisation and CMA-ES [10], Gupta and colleagues give every morphology a full lifetime of learning before scoring it [11], and Schaff and colleagues train one policy on a narrowing design distribution and score designs from the training trajectories [12], the scheme this workspace has now replaced.

For the noise itself, the evolution strategy literature offers three remedies. Uncertainty handling measures how often re-evaluation reorders candidates and enlarges the step size or the evaluation budget in response [13], and pycma ships it as `NoiseHandler`. Racing spends evaluations on a candidate only until its rank is settled [14]. Learning rate adaptation keeps the signal-to-noise ratio of the distribution update constant by adapting the learning rate, and solves noisy problems with the default population size [15], which the installed `cmaes` 0.13.1 exposes as `CMA(lr_adapt=True)`. A larger population averages noise implicitly [16]. Every one of these presumes that the fitness is stationary within a generation and unbiased across designs, which sections 2 and 5.2 address first.

### 5.4 The next course

The steps are ordered so that each one's result decides whether the next is needed.

1. Apply sections 2 and 3.7, so that fitness is comparable across generations and every design meets every cell. This needs no experiment.
2. Run the horizon diagnostic of section 3.8 at an early and a late generation, and adopt the shortest horizon at which $\rho_{full}$ stops rising.
3. Run the landscape probe of section 5.2 on the iteration 6000 and final checkpoints. If the frozen generalist's landscape peaks near the same designs, the run found a real optimum. If the final checkpoint's peak sits on its own population while the generalist's does not, the bias dominates.
4. If the bias dominates, keep a fixed share of environments, a quarter for instance, on designs drawn uniformly over the box throughout the search, so that the policy stays a generalist as in [5] and [6]. This is the change most likely to make the fitness a measure of the design.
5. Stop the search, or raise the evaluation budget, when the split-half value stays below a threshold such as 0.3 for three generations. A low value with a narrow population means the search has converged at the resolution of the evaluation, and further generations spend compute on noise.
6. Only then consider an evaluation-side remedy, `lr_adapt=True` for the `cmaes` searches [15] or uncertainty handling for the pycma search [13], which act on noise but not on bias.
7. Compare the new estimator-free architecture of section 4 against MoRAL on the same protocol. A policy that sees the true morphology should separate designs more sharply, which shows directly as a higher split-half value at the same population spread.

## 6. A smoothed, two-sided tracking width curriculum

### 6.1 Why the present curriculum must change

`reduce_tracking_rewards_std` narrows a tracking width by `update_rate` at every `interval` steps on which the mean kernel of the environments resetting on that one step exceeds `update_threshold`, and it never widens (`environments/environments/tasks/locomotion/mdp/curriculums.py:287-310`). Published locomotion work holds the tracking width fixed and adapts the commands [18] or the penalties [19] instead, and the only error driven width rule found, from motion tracking, settles the width at the mean tracking error of the trained policy by a moving average that is never allowed to rise [17] (`/ws/context/literature.md` cluster 39). The workspace's own runs show why the present gate cannot settle. Across 58 runs ([../context/artefacts/tracking_width_curriculum-cross-run/cross_run_widths.csv](../context/artefacts/tracking_width_curriculum-cross-run/cross_run_widths.csv), summarised in `../context/rewards.md`, addition of 2026-10-23), the quadruped's gate fired below its own threshold in 3 to 7 per cent of narrowings and left its widths near the estimated root mean square error, while every biped's gate fired below threshold in 18 to 59 per cent of linear and 54 to 100 per cent of yaw narrowings. The linear width reached its floor in 9 of the 10 SD_BRS1 runs and in both TRON1 co-optimisation runs, and the biped yaw width ended at 0.33 to 0.43 of the estimated root mean square yaw error with a mean kernel of 0.31 to 0.44, below the threshold meant to stop it. A gate that reads a handful of environments on one step ratchets the width down on favourable draws and stops only at the floor.

### 6.2 The change

Three optional arguments are added to `reduce_tracking_rewards_std`, each defaulting to the present behaviour, so that no task changes until its configuration opts in.

1. `smoothing_episodes`. When set, every episode that ends contributes its mean per step kernel, its reward sum divided by the term's maximum, to a moving average with weight `1 / smoothing_episodes`, and the gate reads that average rather than the one step's resetting environments. No decision is taken until that many episodes have contributed, so a resumed run re-warms before it acts. The average is not updated while `_suppress_curriculum` is raised, so the controlled evaluation and the design swap leave it untouched.
2. `widen_threshold`. When set, an interval on which the average falls below it widens the width by `1 / update_rate`. The width then settles where the average stays between the two thresholds, the band holding the kernel in its sensitive range, and it recovers when a harder command range, terrain or design raises the error.
3. `maximum_std`. The cap on widening, set to the term's configured starting width. On the smoothed path the narrowing is also clamped at `minimum_std` exactly, which the default path overshoots by up to one step.

Edit C1, `environments/environments/tasks/locomotion/mdp/curriculums.py`. Find

```python
def reduce_tracking_rewards_std(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    minimum_std: float,
    interval: int,
    update_threshold: float = 0.8,
    update_rate: float = 0.995,
    starting_step: float = 0.0,
):
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    if env.common_step_counter < starting_step:
        return term_cfg.params["std"]

    # Raised by CoptOnPolicyRunner across a design swap's reset.
    if env.common_step_counter % interval == 0 and not getattr(env, "_suppress_curriculum", False):
        rew = env.reward_manager._episode_sums[term_name][env_ids]
        if (
            torch.mean(rew) / env.max_episode_length
            > update_threshold * term_cfg.weight * env.step_dt
        ):
            if term_cfg.params["std"] > minimum_std:
                term_cfg.params["std"] = term_cfg.params["std"] * update_rate
    return term_cfg.params["std"]
```

and replace it with

```python
def _tracking_kernel_average(env, term_name, term_cfg, env_ids, episodes) -> tuple[float, int]:
    """Episode weighted moving average of the mean per step kernel of the episodes now ending, and its count.

    An episode's kernel is its reward sum over the term's maximum, the quantity the gate of
    reduce_tracking_rewards_std compares with its threshold, so a fall counts as the reward
    forgone. Each episode carries weight 1 / episodes, so the average spans about that many.
    """
    store = env.__dict__.setdefault("_tracking_kernel_average", {})
    average, count = store.get(term_name, (0.0, 0))
    ended = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
    if ended > 0:
        kernel = env.reward_manager._episode_sums[term_name][env_ids] / (
            env.max_episode_length * term_cfg.weight * env.step_dt
        )
        batch = kernel.mean().item()
        alpha = 1.0 - (1.0 - 1.0 / episodes) ** ended
        average = batch if count == 0 else average + alpha * (batch - average)
        count += ended
        store[term_name] = (average, count)
    return average, count


def reduce_tracking_rewards_std(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    minimum_std: float,
    interval: int,
    update_threshold: float = 0.8,
    update_rate: float = 0.995,
    starting_step: float = 0.0,
    smoothing_episodes: int | None = None,
    widen_threshold: float | None = None,
    maximum_std: float | None = None,
):
    """Narrow a tracking kernel's width as its tracking improves.

    With smoothing_episodes None, the default, the gate reads the environments resetting on
    the interval step and only narrows, the behaviour every run before 2026-10-23 trained
    under. With it set, the gate reads a moving average over that many ended episodes,
    narrows above update_threshold and, with widen_threshold set, widens below it up to
    maximum_std, so the width settles where the average stays between the two.

    Defect left standing in the default path, which every existing configuration calls, is a
    gate that fires on a single step's resetting environments, which across 58 runs narrowed
    the biped widths while their mean kernel stood below the threshold (context/rewards.md,
    addition of 2026-10-23), and a floor overshot by up to one step.
    """
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    suppressed = getattr(env, "_suppress_curriculum", False)
    if smoothing_episodes is not None and not suppressed:
        average, count = _tracking_kernel_average(env, term_name, term_cfg, env_ids, smoothing_episodes)
    if env.common_step_counter < starting_step:
        return term_cfg.params["std"]

    # Raised by CoptOnPolicyRunner across a design swap's reset.
    if env.common_step_counter % interval == 0 and not suppressed:
        if smoothing_episodes is None:
            rew = env.reward_manager._episode_sums[term_name][env_ids]
            if (
                torch.mean(rew) / env.max_episode_length
                > update_threshold * term_cfg.weight * env.step_dt
            ):
                if term_cfg.params["std"] > minimum_std:
                    term_cfg.params["std"] = term_cfg.params["std"] * update_rate
        elif count >= smoothing_episodes:
            std = term_cfg.params["std"]
            if average > update_threshold:
                term_cfg.params["std"] = max(std * update_rate, minimum_std)
            elif widen_threshold is not None and average < widen_threshold:
                cap = maximum_std if maximum_std is not None else float("inf")
                term_cfg.params["std"] = min(std / update_rate, cap)
    return term_cfg.params["std"]
```

The default path is the original code moved one level in, so every configuration that does not set `smoothing_episodes` computes exactly what it did. The five callers, `environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:1164-1185` and `:1291-1312`, `cfg/SF/brs_base_env_cfg.py:979-1001`, `cfg/SF/kscale_base_env_cfg.py:1238-1260` and `cfg/quadruped/base_env_cfg.py:1365-1386` and `:1482-1503`, are unaffected until edited.

### 6.3 The recommended values

| Parameter | Biped linear | Biped yaw | Quadruped linear | Quadruped yaw |
|---|---|---|---|---|
| Starting width, the reward term's `std` and `maximum_std` | 0.4 | 0.5 | 0.4 | 0.4 |
| Floor, `minimum_std` | 0.12 | 0.30 | 0.10 | 0.15 |
| Narrow above, `update_threshold` | 0.67 | 0.60 | 0.67 | 0.60 |
| Widen below, `widen_threshold` | 0.55 | 0.45 | 0.55 | 0.45 |
| `update_rate` | 0.95 | 0.95 | 0.925 | 0.95 |
| `smoothing_episodes` | the task's `num_envs` | the task's `num_envs` | the task's `num_envs` | the task's `num_envs` |
| `interval`, `starting_step` | unchanged | unchanged | unchanged | unchanged |

The thresholds bracket the kernel value at which a Gaussian kernel is most sensitive to the error, $e^{-1/2} \approx 0.61$ for an error of $\sigma/\sqrt{2}$. The band of 0.12 to 0.15 between them is wider than the change of about 0.05 to 0.08 in the mean kernel that one step of 5 or 7.5 per cent in the width produces near that value, so the width cannot oscillate between the two. The linear thresholds keep the 0.67 under which the quadruped's gate already behaved, and the yaw thresholds sit lower because the yaw error is the harder to drive down on every robot of the survey.

The widths come from the cross-run table. The quadruped, whose gate worked, settled its widths at 0.58 and 0.72 of the estimated root mean square error for the linear and the yaw term, so the floors are set near 0.6 of each family's estimated root mean square error at the end of training. For the bipeds that error is 0.17 to 0.20 m/s linear (TRON1 co-optimisation, SD_BRS1), giving 0.10 to 0.12, of which 0.12 is taken, and 0.48 to 0.53 rad/s yaw, giving 0.29 to 0.32, of which 0.30 is taken. For the quadruped's two most recent runs it is 0.13 to 0.14 linear and 0.13 to 0.16 yaw, giving 0.08 and 0.08 to 0.10, raised to 0.10 and 0.15 so that the floors sit no lower than the widths its gate reached by itself, a median of 0.136 and 0.223. The biped yaw start rises from 0.4 to 0.5 because its estimated root mean square yaw error already exceeds 0.4, and the TRON1 runs that never narrowed their yaw width held a mean kernel of only 0.34 to 0.39 at 0.4, below the band at the outset. The window of `num_envs` episodes is one episode per environment, about 32 iterations at 6400 environments and an average of 8 resets per step, against an interval of 200 iterations.

The root mean square estimates assume Rayleigh and half normal errors and are lower bounds for a heavy tailed error, so the floors err toward the narrow side and the widening of the band, not the floor, is meant to set the operating width. The KScale yaw errors of 1 to 2 rad/s were left out of the yaw values.

### 6.4 The configurations

Edit C2, the biped co-optimisation curriculum, `environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py`. Find

```python
    modify_linear_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_lin_vel_xy",
            "interval": 200 * 25,
            "starting_step": 400 * 25,
            "update_rate": 0.95,
            "update_threshold": 0.67,
            "minimum_std": 0.09
        }
    )
    modify_angular_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_ang_vel_z",
            "interval": 200 * 25,
            "starting_step": 0,
            "update_rate": 0.95,
            "update_threshold": 0.5,
            "minimum_std": 0.09
        }
    )
```

in `CoptCurriculumCfg` (`:1291-1312`, the second of the two occurrences, the first belonging to `CurriculumCfg` at `:1164-1185`), and replace it with

```python
    modify_linear_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_lin_vel_xy",
            "interval": 200 * 25,
            "starting_step": 400 * 25,
            "update_rate": 0.95,
            "update_threshold": 0.67,
            "minimum_std": 0.12,
            "smoothing_episodes": 6400,
            "widen_threshold": 0.55,
            "maximum_std": 0.4,
        }
    )
    modify_angular_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_ang_vel_z",
            "interval": 200 * 25,
            "starting_step": 0,
            "update_rate": 0.95,
            "update_threshold": 0.60,
            "minimum_std": 0.30,
            "smoothing_episodes": 6400,
            "widen_threshold": 0.45,
            "maximum_std": 0.5,
        }
    )
```

Edit C3, the biped co-optimisation yaw starting width, the same file, in `CoptRewardsCfg` after its docstring. Find

```python
    reproduces its original while none is published (plan section 7). 0.804 m is
    the donor's standing height, which RewardsCfg's 0.75 misses by 54 mm, and
    0.050 m the sole's depth below the foot body (biped/builder.py
    z_bottom_ankleAct2foot).
    """
```

and replace it with

```python
    reproduces its original while none is published (plan section 7). 0.804 m is
    the donor's standing height, which RewardsCfg's 0.75 misses by 54 mm, and
    0.050 m the sole's depth below the foot body (biped/builder.py
    z_bottom_ankleAct2foot).
    """

    # The biped's yaw error exceeds RewardsCfg's starting width of 0.4, copt_improvement.md section 6.3.
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=7.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
```

Edit C4, the quadruped co-optimisation curriculum, `environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py`, in `CoptCurriculumCfg` (`:1482-1503`). Find

```python
    modify_linear_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_lin_vel_xy",
            "interval": 12500,
            "starting_step": 12500,
            "update_rate": 0.925,
            "update_threshold": 0.67,
            "minimum_std": 0.09,
        },
    )
    modify_angular_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_ang_vel_z",
            "interval": 12500,
            "starting_step": 12500,
            "update_rate": 0.95,
            "update_threshold": 0.5,
            "minimum_std": 0.09,
        },
    )
```

and replace it with

```python
    modify_linear_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_lin_vel_xy",
            "interval": 12500,
            "starting_step": 12500,
            "update_rate": 0.925,
            "update_threshold": 0.67,
            "minimum_std": 0.10,
            "smoothing_episodes": 6400,
            "widen_threshold": 0.55,
            "maximum_std": 0.4,
        },
    )
    modify_angular_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_ang_vel_z",
            "interval": 12500,
            "starting_step": 12500,
            "update_rate": 0.95,
            "update_threshold": 0.60,
            "minimum_std": 0.15,
            "smoothing_episodes": 6400,
            "widen_threshold": 0.45,
            "maximum_std": 0.4,
        },
    )
```

The other tasks take the same values for their robot class at their own curriculum terms, the SD_BRS1 at `cfg/SF/brs_base_env_cfg.py:979-1001`, the KScale at `cfg/SF/kscale_base_env_cfg.py:1238-1260`, the plain and HIM TRON1 tasks at `cfg/SF/limx_base_env_cfg.py:1164-1185` and the plain and HIM quadruped tasks at `cfg/quadruped/base_env_cfg.py:1365-1386`, each with `smoothing_episodes` set to that task's `num_envs`, 4096 for the plain tasks. A biped yaw start of 0.5 is set in the reward configuration that task reads. Each such edit changes a task that trained under the old curriculum, so a run after it is not comparable with that task's earlier runs on the tracking terms, and the change is recorded in the run's dumped `params/env.yaml`.

### 6.5 Rationale and interactions

The rule follows the two practices of the literature that bear on it. Like KungfuBot it ties the width to the error the policy actually produces through a moving average [17], and like the grid curriculum of Rapid Locomotion it judges tracking over completed episodes rather than over one step [18]. It departs from both by also widening, because the co-optimisation search changes the robot under the policy and the curricula of commands and terrain raise the difficulty, and a width that cannot widen then pays almost nothing, as the biped yaw width did. Widening is this workspace's proposal and has no precedent in the surveyed work. The moving average lives on the environment and not in the checkpoint, so a resumed run takes no decision until `smoothing_episodes` episodes have ended. The curriculum is suspended during the controlled evaluation of section 3.7, so the average sees training episodes only.

The evaluation of section 2.3 scores at the configured starting widths, so after edit C3 the biped yaw term is evaluated at 0.5, which is close to $\sqrt{2}$ times its typical yaw error and so near the kernel's most sensitive width, while the linear term stays at 0.4.

The values are untested in training. A smoke run must confirm that the logged `Curriculum/modify_*_tracking_reward_std` stays constant until `smoothing_episodes` episodes have ended, then moves in steps of `update_rate`, and that across a controlled evaluation it does not move. The first full run of each family should be checked against the band, the logged mean kernel, `Episode_Reward/<term>` divided by the weight, settling between the two thresholds.

## 7. Order of application and checks

1. Apply R1 to R3 and A1 to A17 together, then the analysis script. Apply C1 to C4 of section 6 in the same pass or separately, C1 changing no task by itself. Reinstall `co_optimisation` in both containers, since it is installed as a copy, with `pip install --no-deps --no-build-isolation /workspace/isaaclab/biped/co_optimisation`.
2. Smoke test, at 1024 environments and 16 designs as in `plans/cmaes_evaluation.md` section 5, each of `COPT` and `COPT-MORAL` on `Isaac-Limx-SF-Copt-Rough-v0`, `Isaac-Limx-SF-Copt-MoRAL-Rough-v0`, `Isaac-Quadruped-Copt-Rough-v0` and `Isaac-Quadruped-Copt-MoRAL-Rough-v0`. Each must reach a CMA-ES generation and end without a traceback, the `COPT` runs printing an actor input width larger than the MoRAL runs by the width of `morphologyObs` and a `Loss/model_estimation_loss` of zero. Corrected 2026-10-23 on application, the two architectures print the same actor input width, since the MoRAL actor reads an estimate exactly as wide as `morphologyObs` (`/ws/context/cmaes.md` section 13).
3. Confirm the width fix by printing the two tracking widths before, during and after one evaluation, the middle reading 0.4 and the outer two equal. Confirm the layout by printing, in each pass, the unique terrain levels, a single value equal to the pass index, and the count of environments per terrain type, equal to the number of designs in every column, and confirm that `CMAES/eval/return_row_<r>` appears for all five rows and that `CMAES/eval_time` is near 265 s at full scale.
4. Confirm that `COPT` refuses a pre-change MoRAL checkpoint with a missing key error, and that `COPT-MORAL` loads it.
5. Run one `djinn start train copt` and one `djinn start train quadruped-copt` to their first logged iteration.
6. For C1 to C4, confirm in a smoke run that each `Curriculum/modify_*_tracking_reward_std` stays at its starting width until `smoothing_episodes` episodes have ended, then moves only in steps of `update_rate` within its floor and cap, and does not move across a controlled evaluation, and that a task without `smoothing_episodes` logs exactly the curve it logged before.
7. Record the outcome in `/ws/context/cmaes.md` and `/ws/context/copt.md`, update `ARCHITECTURE.md`, `/ws/ARCHITECTURE.md` and `/ws/CO_OPTIMISATION.md` for the new task ids and policy types, and set this plan's status in `plans/README.md`.

## 8. Bibliography

[1] N. Hansen. The CMA Evolution Strategy, A Tutorial. arXiv:1604.00772, 2016.

[2] A. Y. Ng, M. I. Jordan. PEGASUS, A Policy Search Method for Large MDPs and POMDPs. Uncertainty in Artificial Intelligence (UAI) 2000, 406-415. arXiv:1301.3878.

[3] N. Cheney, J. Bongard, V. SunSpiral, H. Lipson. Scalable co-optimization of morphology and control in embodied machines. Journal of the Royal Society Interface 15(143), 20170937, 2018.

[4] A. Mertan, N. Cheney. Investigating Premature Convergence in Co-optimization of Morphology and Control in Evolved Virtual Soft Robots. EuroGP 2024, as the arXiv record states. arXiv:2402.09231.

[5] Á. Belmonte-Baeza, J. Lee, G. Valsecchi, M. Hutter. Meta Reinforcement Learning for Optimal Design of Legged Robots. IEEE Robotics and Automation Letters 7(4), 12134-12141, 2022. DOI 10.1109/LRA.2022.3211785. arXiv:2210.02750.

[6] F. Bjelonic, J. Lee, P. Arm, D. Sako, D. Tateo, J. Peters, M. Hutter. Learning-based Design and Control for Quadrupedal Robots with Parallel-Elastic Actuators. IEEE Robotics and Automation Letters, 2023. DOI 10.1109/LRA.2023.3234809. arXiv:2301.03509.

[7] L. Strgar, S. Kriegman. Accelerated co-design of robots through morphological pretraining. arXiv:2502.10862, 2025, accepted at ICLR 2026.

[8] N. Bohlinger, J. Peters. Shape Your Body, Value Gradients for Multi-Embodiment Robot Design. arXiv:2606.00702, 2026.

[9] D. Ha. Reinforcement Learning for Improving Agent Design. Artificial Life 25(4), 352-365, 2019. arXiv:1810.03779.

[10] K. S. Luck, H. Ben Amor, R. Calandra. Data-efficient Co-Adaptation of Morphology and Behaviour with Deep Reinforcement Learning. Conference on Robot Learning 2019, PMLR 100, 854-869. arXiv:1911.06832.

[11] A. Gupta, S. Savarese, S. Ganguli, L. Fei-Fei. Embodied intelligence via learning and evolution. Nature Communications 12, 5721, 2021. arXiv:2102.02202.

[12] C. Schaff, D. Yunis, A. Chakrabarti, M. R. Walter. Jointly Learning to Construct and Control Agents using Deep Reinforcement Learning. ICRA 2019. arXiv:1801.01432.

[13] N. Hansen, A. S. P. Niederberger, L. Guzzella, P. Koumoutsakos. A method for handling uncertainty in evolutionary optimization with an application to feedback control of combustion. IEEE Transactions on Evolutionary Computation 13(1), 180-197, 2009. DOI 10.1109/TEVC.2008.924423.

[14] V. Heidrich-Meisner, C. Igel. Hoeffding and Bernstein Races for Selecting Policies in Evolutionary Direct Policy Search. ICML 2009, 401-408. DOI 10.1145/1553374.1553426.

[15] M. Nomura, Y. Akimoto, I. Ono. CMA-ES with Learning Rate Adaptation, Can CMA-ES with Default Population Size Solve Multimodal and Noisy Problems? Genetic and Evolutionary Computation Conference (GECCO) 2023. DOI 10.1145/3583131.3590358. arXiv:2304.03473.

[16] Y. Jin, J. Branke. Evolutionary Optimization in Uncertain Environments, A Survey. IEEE Transactions on Evolutionary Computation 9(3), 303-317, 2005.

[17] W. Xie, J. Han, J. Zheng, H. Li, X. Liu, J. Shi, W. Zhang, C. Bai, X. Li. KungfuBot, Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills. Advances in Neural Information Processing Systems (NeurIPS) 2025. arXiv:2506.12851.

[18] Margolis and colleagues. Rapid Locomotion via Reinforcement Learning. Robotics, Science and Systems XVIII, 2022. arXiv:2205.02824. The full author list is not recorded in `/ws/context/literature.md` and is omitted here.

[19] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, M. Hutter. Learning agile and dynamic motor skills for legged robots. Science Robotics 4(26), eaau5872, 2019. DOI 10.1126/scirobotics.aau5872. arXiv:1901.08652.
