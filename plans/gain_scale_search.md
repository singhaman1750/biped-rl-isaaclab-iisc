# Searching the servo gain scales in every continuous design generator

Status, IMPLEMENTED 2026-11-17 on the host, `design_generator_test.py` passing 122 of 122 and `design_filter_test.py` 32 with 8 skipped, the end to end check in Isaac Sim of the verification section not yet run. The outcome is recorded in section 12c of [../context/co_optimisation_audit.md](../context/co_optimisation_audit.md). The implementation followed the plan as approved, a first draft that kept a separate registry of gain keys having been withdrawn at review in favour of widening `CMAES_PARAM_KEYS`. It diverged in one respect, the test suite stub of `cmaes.CatCMAwM` gaining the `_Nco` attribute that the new dimension check reads, and the line numbers below are those of the tree before the change. Revised and implemented again later on 2026-11-17, see the revision section at the end, after which the suites passed 162 with 8 skipped.


## Context

`DEFAULT_PARAM_RANGES` carries eight per group gain scales, `abad_stiffness_scale` through `ankle_damping_scale` (`co_optimisation/co_optimisation/runners/usd_generator.py:466-479`), and `DesignGeneratorBase._actuator_params` already multiplies a catalogue entry's `stiffness` and `damping` by them when they are present in the parameters it is handed (`usd_generator.py:878-889`). The runner writes the result into the `IdentifiedActuator` rows of every environment through `apply_actuator_params` (`co_optimisation/co_optimisation/utils/respawn.py:246-305`, called at `copt_on_policy_runner.py:693`, `:737`, `:847`). `IdentifiedActuator` is an explicit `DCMotor` whose `compute` reads `self.stiffness` and `self.damping` every physics step (`environments/environments/actuators/actuator_pd.py:18-40`), so a written row takes effect at once and needs no write through to PhysX.

The scales nevertheless never reach the simulator from any search, for two reasons established in the tree.

1. `CMAES_PARAM_KEYS` lists the four geometry keys alone (`usd_generator.py:510-512`), and `CMAESDesignGenerator.__init__` truncates `param_ranges` to it (`:1068-1073`), so a gain scale named in `CMAESCfg.param_ranges` is accepted by `_typed_ranges` and then silently discarded, never becoming a search dimension and never being sampled in the random phase.
2. Every generator with a continuous block but the random one calls `_actuator_params(spec)` without its parameters, `FixedDesignGenerator` at `:997`, the fine tune's `_mean_population` at `:1205`, `CMAESDesignGenerator` at `:1226`, `GrowingDesignDistCMAESDesignGenerator` at `:1293` and `CatCMAESDesignGenerator` at `:1599`. Only `RandomDesignGenerator` (`:970`), `_individual`'s infeasible fallback (`:763`) and `describe_design` (`:693`) pass them, so the design logs would describe gains that were never applied.

The intended outcome is that naming any gain scale in a run's `param_ranges` makes it a continuous search variable of every generator that has a continuous block, sampled in the random phase, searched by CMA-ES or CatCMAwM, applied to the actuators of the emitted population, logged and replayed, while a configuration that names none behaves exactly as before.

## Approach

`CMAES_PARAM_KEYS` becomes the list of every key a CMA-ES search may carry, the four geometry keys followed by the eight gain scales, and no separate gain registry is introduced. One distinction must survive, because `_typed_ranges` merges every default into `param_ranges` (`:529-545`) and so cannot tell a named key from a defaulted one. The geometry keys are searched whether or not they are named, as they are today, `play.py`'s own search naming two of them and receiving four (`scripts/rsl_rl/play.py:1181-1190`), whereas a gain scale is searched only when named. The existing `SPEC_PARAM_FIELDS` (`:516-521`) already names the geometry keys, so the restriction reads it rather than a new structure.

| Generator | Change |
|---|---|
| `RandomDesignGenerator` | none needed, already samples and passes every key, see the defects below |
| `FixedDesignGenerator` | pass `self.parameters` to `_actuator_params` |
| `CMAESDesignGenerator` | the widened restriction, pass parameters |
| `GrowingDesignDistCMAESDesignGenerator` | inherits the restriction, pass parameters |
| `CatCMAESDesignGenerator` | inherits the restriction, pass parameters, check the continuous dimension on resume |
| `_mean_population` (fine tune of all three) | pass `values` |
| `CatOnlyCMAESDesignGenerator` | no continuous block, unchanged |

### 1. Widen `CMAES_PARAM_KEYS`, `usd_generator.py:509-512`

```python
# Every key a CMA-ES search may carry, the lengths and the pose, then the gain scales, which
# _actuator_params consumes. The order is the search vector's, so the four geometry keys keep their
# positions. CMAESDesignGenerator searches the geometry keys always and a gain scale only where named.
CMAES_PARAM_KEYS: tuple[str, ...] = (
    "thigh_length_scale", "shank_length_scale", "thigh_bend_offset", "knee_bend_offset",
    "abad_stiffness_scale", "abad_damping_scale", "hip_stiffness_scale", "hip_damping_scale",
    "knee_stiffness_scale", "knee_damping_scale", "ankle_stiffness_scale", "ankle_damping_scale",
)
```

### 2. The restriction, `CMAESDesignGenerator.__init__` (`:1068-1071`)

```python
        # Restrict CMA-ES to CMAES_PARAM_KEYS, the geometry keys always and a gain scale only where the
        # caller names it, so that a configuration naming none keeps its four dimensions and its draws.
        self.param_ranges = {k: self.param_ranges[k] for k in CMAES_PARAM_KEYS
                             if k in SPEC_PARAM_FIELDS or k in (param_ranges or {})}
        for key in self.param_ranges.keys() - SPEC_PARAM_FIELDS.keys():
            if key.split("_", 1)[0] not in self.family.groups:
                raise ValueError(f"{key} names no joint group of the {family} family, {self.family.groups}")
```

The constructor's own `param_ranges` argument is the caller's dict before the merge, so no attribute need be added to the base class. The group check refuses, for instance, `ankle_stiffness_scale` on the quadruped, whose groups are abad, hip and knee (`:448`), which would otherwise be searched as an inert dimension.

`param_keys` (`:1073`) is unchanged, since `searched_keys` is `CMAES_PARAM_KEYS` and so now admits the named gain scales, while `CatOnlyCMAESDesignGenerator.searched_keys` stays empty. `_initial_mean`, `_denormalise`, `search_statistics`, `optimised_design`, `_sample_parameter` and `_sample_parameter_v2` all iterate `param_keys` or `param_ranges`, so the gain scales enter the unit box at their initial value 1.0, are denormalised, logged as `mean` and `std`, written to `final_design.csv` by `log_optimised_designs` (`copt_on_policy_runner.py:1252-1264`) and drawn in the random phase, the geometry keys being drawn first so that their random stream is unchanged.

### 3. Pass the parameters to `_actuator_params`

```python
# FixedDesignGenerator._generate_individual, :997
        return spec, self._actuator_params(spec, self.parameters)

# CMAESDesignGenerator._mean_population.mean, :1205
            return spec, self._actuator_params(spec, values)

# CMAESDesignGenerator._generate_individual, :1226
# GrowingDesignDistCMAESDesignGenerator._generate_individual, :1293
# CatCMAESDesignGenerator._generate_individual, :1599
        return spec, self._actuator_params(spec, parameters)
```

### 4. Check the continuous dimension when CatCMAwM resumes, `CatCMAESDesignGenerator._check_dimension` (`:1407-1413`)

pycma's strategy is checked on `N` (`:1117-1121`), but the CatCMAwM check reads the population size alone, so a checkpoint resumed under a different set of gain scales would decode silently against the wrong ranges. `_Nco` is CatCMAwM's continuous dimension, zero for `CatOnlyCMAESDesignGenerator` (`cmaes/_catcmawm.py:130`, asserted at `co_optimisation/tests/design_filter_test.py:253`), so the line holds for both subclasses.

```python
        assert int(self._es._Nco) == self.num_params, (
            f"Checkpoint dimension mismatch: pickled {self._es._Nco}, expected {self.num_params}"
        )
```

### 5. Replay

`play.py:291` reads `CMAES_PARAM_KEYS` from `final_design.csv` and so picks up the gain scales without change, and the `cmaes_history.csv` route (`:304-310`) already takes every numeric `best/` column. `FixedDesignGenerator` then applies them through section 3. A `final_design.csv` written before this change carries no gain column and replays as before.

### 6. Opting a run in, an example only, the defaults are not changed

```python
@configclass
class BipedCMAESCfg(CMAESCfg):
    param_ranges: dict = {
        "thigh_length_scale": ("scale", (0.75, 1.25)),
        "shank_length_scale": ("scale", (0.75, 1.25)),
        "thigh_bend_offset": ("offset", (-0.10, 0.10)),
        "knee_bend_offset": ("offset", (-0.25, 0.25)),
        "hip_stiffness_scale": ("scale", (0.875, 1.125)),
        "knee_stiffness_scale": ("scale", (0.833, 1.167)),
        "knee_damping_scale": ("scale", (0.750, 1.250)),
    }
```

`train.py` passes `dict(cmaes_cfg.param_ranges)` to the generator (`scripts/rsl_rl/train.py:256`) and dumps the configuration to `params/cmaes.yaml`, which therefore records the choice.

## Defects left standing, to be recorded

1. `RandomDesignGenerator` samples every key of `DEFAULT_PARAM_RANGES` whether named or not (`:529-545`, `:830`), so it always randomises all eight gain scales, the ankle pair inertly on the quadruped. It is left untouched so as not to alter an existing caller.
2. `CatOnlyCMAESDesignGenerator` keeps a named gain scale in `param_ranges` but never searches or applies it, its `searched_keys` being empty, so the name is still dropped silently there.
3. The startup `randomize_actuator_gains` events (`environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:989-1023`, `PF/limx_base_env_cfg.py:277`) are overwritten by `apply_actuator_params` at the first generation, as `plans/w_eta_optimisation.md` section 1.2 already records.
4. The default gain ranges were derived from the biped's runtime randomisation (`usd_generator.py:28-35`), so a quadruped run should state its own.

`plans/w_eta_optimisation.md` (NOT IMPLEMENTED) proposes replacing these scales with a natural frequency and damping ratio per group and making `hip_stiffness_scale` raise (its line 1434), so that plan's section on the scales needs revising if both land.

## Tests

1. `co_optimisation/tests/design_generator_test.py:227` asserts `gen.param_keys == list(U.CMAES_PARAM_KEYS)` and must become `list(U.SPEC_PARAM_FIELDS)`, the four geometry keys a configuration naming no gain scale still searches. `design_filter_test.py:265` stays correct, `CatOnly`'s records carrying no key of the widened tuple.
2. New, parametrised over the families and the three continuous CMA-ES classes. Naming `hip_stiffness_scale` and `knee_damping_scale` yields six `param_keys` in `CMAES_PARAM_KEYS` order, and every individual of the random phase and of a search generation carries `actuator_params["hip"]["stiffness"] == round(catalogue_stiffness * parameters["hip_stiffness_scale"], 6)`.
3. A configuration naming no gain scale gives the catalogue's actuator parameters and the same geometry draws as before for a fixed seed.
4. `optimised_design()` returns the gain scales, and `FixedDesignGenerator` on them emits the `actuator_params` that `describe_design` records, the fine tune's mean population likewise.
5. `ankle_stiffness_scale` on the quadruped raises `ValueError`, and a CatCMAwM state pickled with four continuous dimensions fails `_check_dimension` under six.

The artefact `context/artefacts/copt_training_refinement-2026-10-28/host_check.py:110` compares the default configuration's keys with `CMAES_PARAM_KEYS` and would now report a mismatch. It records a past check and is not run by the suite, so it is noted in the artefact register rather than edited.

## Verification

1. `python -m pytest co_optimisation/tests/design_generator_test.py co_optimisation/tests/design_filter_test.py -q` in the container, the modules stubbing the converter so no simulator is needed.
2. One simulator check of the application end to end. A biped co-optimisation run with the ranges of section 6, a small `--num_envs` and an `ea_late_start` low enough to reach a search generation, asserting after `apply_actuator_params` that `articulation.actuators["knee"].stiffness[env]` equals the knee row of `current_population.get_actuator_params()[env % n]` for a sample of environments, and that `params/cmaes.yaml` and `designs/` record the gain scales.
3. Resuming that run from its checkpoint continues with six dimensions.

## Documentation, in the same pass

1. `context/co_optimisation_audit.md`, a dated entry recording the two defects, the fix and the defects left standing.
2. The module docstring of `usd_generator.py` (`:28-35`), the `CMAESDesignGenerator` docstring (`:1008-1013`) and the `FixedDesignGenerator` docstring (`:983`), stating that named gain scales are searched.
3. The `param_ranges` docstring in `config/cmaes_cfg.py:46`, and `CO_OPTIMISATION.md` at the workspace root.
4. `plans/README.md`, registering this plan once it is moved into `plans/` with its status.

## Revision of 2026-11-17, the actuator only search

At the user's direction the rule of section 2 that the geometry keys are searched whether named or not was replaced. A CMA-ES generator now searches exactly the keys of `CMAES_PARAM_KEYS` its `param_ranges` names, in that order, and the four geometry keys where it is given no ranges at all. The gain scales are listed in a `GAIN_PARAM_KEYS` tuple merged into `CMAES_PARAM_KEYS`. `_lean_max` returns zero where the lean is not searched, the user observing that an unsearched lean is the pose the robot meets throughout training. `play.py` and `scripts/analysis/nan_reset_diagnostic.py` now name all four geometry keys, which they had received unnamed.

`design_search` `actuator` maps to `CatCMAESDesignGenerator` over gain scales alone, and `train.py` raises where its `param_ranges` is empty or names a key outside `GAIN_PARAM_KEYS`. `CatOnlyCMAESDesignGenerator` is kept, unused by `train.py`. The overwriting of the startup gain randomisation, listed above among the defects left standing, is deliberate by the user's statement and is no defect. The outcome is recorded in the addition to section 12c of [../context/co_optimisation_audit.md](../context/co_optimisation_audit.md).
