# Co-Optimisation Audit

Defect audit of the design and policy co-optimisation pipeline, covering `co_optimisation/`, `design_generator/` and `environments/` for the quadruped and the TRON1 solefoot biped. First pass 2026-09-29, revised 2026-09-30 against the review recorded in section 11, and the nominal pose of section 4 redesigned on 2026-10-01 around per design bend offsets. On 2026-10-06 the implementation plan, [../plans/co_optimisation_audit_implementation.md](../plans/co_optimisation_audit_implementation.md), applied every Apply and Plan finding to a sandbox copy of the tree, tested it in the training container and corrected the claims recorded in the dated rows of sections 9 and 10. Later the same day the plan was revised against its own review, which moved the quadruped's fold into the body frames of every Isaac Lab document rather than of a separate co-optimisation asset, made all four design generators selectable, gave the categorical search its random phase and moved the population check to `train.py`, and the rows dated 2026-10-06 second revision record what that changed here. On 2026-10-07 a second review of the plan gave each robot's co-optimisation task reward and curriculum classes of its own, carried the categorical search's designs in place, and observed the nominal pose from the joint frames that hold it, and the rows dated 2026-10-07 record what that changed here. The plan was implemented on 2026-10-07, and section 12 records the validation of the tree and the four points where the application diverged from the plan's text. Every finding carries its location, its arithmetic and the edit that closes it, so that an implementation plan can be written from this document alone.

Already discharged, do not rediscover. The quadruped rough terrain built a 39,694,044 triangle collider that registered no contact, cause `horizontal_scale=0.05` at `terrains_cfg.py:28`, restored to 0.1 on 2026-09-29. See [quadruped.md](quadruped.md) section 8.

## 1. Finding register

Disposition records the decision taken on each finding in the review of section 11. Apply means the edit lands in the coming pass, Plan means it lands in the same pass but needs its own design, Record means the finding stands as context with no edit, Defer means the edit waits for a later pass.

| ID | Finding | Live for | Stops a run | Disposition | Location |
|---|---|---|---|---|---|
| B1 | Design generator family hard coded to `"biped"` | quadruped | yes | Apply | `train.py:221` |
| B2 | `capture_env_state` reads `terrain_levels`, absent on a plane | flat tasks, both robots | yes, iteration 0 | Apply | `env_state.py:144` |
| B3 | Population indexes environments with no divisibility check | both | yes, iteration 12480 | Apply | `update.py:208`, `update.py:272` |
| B4 | `cma`, `cmaes`, `pandas`, `design_generator` undeclared | all runs, all robots | if absent | Apply | `co_optimisation/setup.py` |
| B5 | `--policy-type` has no `choices`, typo falls through to plain PPO | both | silent wrong run | Apply | `cli_args.py:60` |
| B6 | `catalog_names` and `actuator_groups` expose a shelf the family already owns | both | no | Apply | `usd_generator.py:765` |
| B7 | 11 of 14 quadruped and 11 of 15 biped catalogue entries are simulation incomplete | categorical and random search | yes, once B6 widens the shelf | Apply | `actuators.json`, `actuator_catalog.py:530` |
| N1 | In place edits reach the first `num_individuals` environments only, propagation to the rest is argued and never measured | both | silently wrong fitness if the argument fails | Apply | `update.py:208`, `update.py:265` |
| D1 | Nominal pose is fixed while the search moves the kinematics, so every nominal referenced reward target prices the design. Redesigned 2026-10-01 as hip pitch and knee offsets on `DesignSpec`, solved and checked in `usd_generator.py` | both | no | Plan | section 4 |
| D1a | Quadruped `constant_params.json:53-54` states 70 and -154 deg, the Isaac asset trains at 50.67 and -101.34 deg, so the donor built from the constants stands 0.117 m high | quadruped | yes, once the pose is encoded in the XML | Apply, with D1 | `constant_params.json:53-54`, section 4.3 |
| D1b | Quadruped XML carries no pose, the biped's carries it in body quats, so the two are asymmetric and the in place editor cannot move either | both | no | Plan, with D1 | `quadruped/builder.py:501`, `update.py:152-165`, sections 4.6 and 4.9 |
| D1c | Foot constraint residual is 0 to 82 mm (quadruped) and 12 to 125 mm (biped) at the length corners with zero offsets, so it must be solved per design | both | no | Plan, with D1 | section 4.3 |
| D1d | Six actuator scales are declared and never read, ranges are untyped scales, `_current_scales`, `_sample_scales` and `dataclasses.replace` carry names and constructions the typed structure retires | both | no | Plan, with D1 | `usd_generator.py:196-236`, `:387-420`, section 4.8 |
| D2 | Quadruped estimates dynamics from true morphology, the biped estimates morphology from history, and the loss is unscaled | both | no | Plan | `quadruped_rsl_rl_ppo_cfg.py:58`, `copt_ppo.py:258` |
| D3 | Normalisation claim withdrawn, `output_detach` remains a field nothing reads | both | no | Record | `copt_actor_critic.py:145` |
| D4 | `compute_returns_design_wise` has no caller | both | no | Record | `copt_ppo.py:96` |
| D5 | Learning rate reset every generation in the random phase, deliberate but ungated | both | no | Apply | `copt_on_policy_runner.py:692` |
| D6 | 6 of 7 curriculum terms read truncated sums at every swap | both | no | Plan | `curriculums.py:476` |
| D7 | Spawn height constant while leg length is searched | both | no | Plan, with D1 | `quadruped_identified_cfg.py:93` |
| D8 | Design lattice 144 points against a population of 256 | quadruped | no | Record | `usd_generator.py:416` |
| L1 | `force_usd_conversion=True`, 55,164 imports per run | both | no | Defer | `usd_generator.py:483` |
| L2 | Observation width 12,872 floats, 7.19 GiB of rollout storage | both | no | Defer | `base_env_cfg.py:360-651` |
| L3 | `replicate_physics=False` at 6000 envs, 1500 prototypes | both | no | Defer | `respawn.py:117` |
| L4 | Actuator write is 72,000 single element device writes | both | no | Defer | `respawn.py:276` |
| X1 | Reward and termination managers not rebound on respawn | KScale only | dormant | Defer | `respawn.py:176-221` |
| X2 | `apply_actuator_params` silently drops unknown keys | both | dormant | Defer | `respawn.py:282` |
| X3 | `effort_limit_sim` and `velocity_limit_sim` never written | random or categorical search | live once B6 lands | Defer | `respawn.py:301` |
| X4 | Neither document declares a joint velocity limit, solver clamp unknown | both | verify | Defer | `quadruped.xml:19-28` |
| X5 | Zero armature on a bare catalogue entry | categorical search | absorbed into B7 | Apply, via B7 | `actuator_catalog.py:528` |
| X6 | `check_geometry` validates only `trunk_width`, never varied | both | dormant | Defer | `quadruped/builder.py:311` |
| X7 | 3 of 4 solefoot experiment configurations unregistered | biped | no | Defer | `robots/__init__.py:46` |
| X8 | Fitness buckets and design assignment use different moduli in the random phase | both | dormant | Record | `copt_on_policy_runner.py:744` |
| H1 | `.gitmodules` name disagrees with path, 3 stale docstrings | both | no | Defer | `.gitmodules` |
| H2 | Quadruped duplicates 300 lines as `CoptObservationsCfg` | quadruped | no | Defer | `base_env_cfg.py:360` |
| H3 | `robot_pos` puts absolute world coordinates in the critic | both | no | Defer | `observations.py:103` |
| H4 | `num_output_dim`, `orthogonal_init`, `self_collision` unread or inconsistent | both | no | Defer | `copt_actor_critic.py:79` |
| H5 | Morphology observations hard code `torch.device("cuda")` | both | dormant | Defer, closed by D2.3 | `observations.py:51-122` |

## 2. The pipeline as built

```
scripts/rsl_rl/train.py  --policy-type COPT
  │
  ├─ _family = "biped"                              train.py:221   ◄── B1
  ├─ _num_individuals   = 256                       train.py:222
  ├─ ea_update_interval = 480 iterations            train.py:225
  ├─ ea_late_start      = 12000 iterations          train.py:226
  ├─ late_start_pop     = num_envs / 4              train.py:259
  ├─ catalog_names=[...]                            train.py:245   ◄── B6
  │
  ├─ GrowingDesignDistCMAESDesignGenerator ─────────────────┐
  └─ CoptOnPolicyRunner(env, generator, cfg)                │
       │                                                    │
       │  learn()                                           │
       │   ├─ _reload_morphology()        gen 0, respawn    │
       │   └─ every 480 iterations:                         │
       │        _update_morphology()                        │
       │          ├─ sim.stop()                             │
       │          ├─ reinit_learning_rate()        ◄── D5   │
       │          ├─ generator.update_with_fitness(f) ◄─────┤ ask / tell
       │          ├─ generate_population(gen) ──────────────┤ emit MJCF + convert USD  ◄── L1, B7
       │          ├─ apply_link_length_params()   in place, edits USD prototypes  ◄── N1
       │          ├─ (nominal pose write)          ◄── D1, absent today
       │          ├─ env.reset()                   ◄── D6 curricula read truncated sums
       │          ├─ assert_link_lengths_applied() ◄── N1, first N envs only
       │          ├─ apply_actuator_params()       ◄── L4, X2, X3
       │          └─ episode_length_buf randomised
       │
       └─ design → environment assignment, round robin, four places that must agree
            copt_on_policy_runner.py:744   env_to_individual = i % num_individuals
            wrappers.py:114                proto_prim_paths[index % len]
            respawn.py:279                 env_idx % num_individuals
            update.py:208                  env_paths[individual]          ◄── B3, N1
```

| | quadruped | biped |
|---|---|---|
| Task ids | `Isaac-Quadruped-Copt-{Flat,Rough}-v0` | `Isaac-Limx-SF-Copt-MoRAL-{Flat,Rough}-v0` |
| Bodies / joints | 25 / 12 | 15 / 8 |
| Root prim | `/trunk/trunk` | `/base_Link/base_Link` |
| Joint expressions | `.._hip_joint`, `.._thigh_joint`, `.._calf_joint` | `abad_._Joint`, `hip_._Joint`, `knee_._Joint`, `ankle_._Joint` |
| Sagittal joint axes | `(0, 1, 0)` thigh and calf | `(0, -1, 0)` hip, knee and ankle |
| Foot contact geometry | sphere, radius 0.022 at the body origin | box, half extents 0.100 by 0.035 by 0.015 at `(0.010, 0, -0.035)` |
| Sole below the foot body origin | 0.022 | 0.050 |
| Catalogue, total and simulation complete | 14, of which 3 | 15, of which 4 |
| `num_envs` | 6000, must become 6144 by B3 | 4096, already compliant |
| `gtEncoderOut` today | `privilegedDynamicsObs` | `morphologyObs` |

The three `A1-` and four `TRON1-` catalogue entries reproduce their asset configuration field for field, so the actuator write of a baseline design is a no operation. Preserve that deliberately.

## 3. Blocking defects

### B1 Design generator family hard coded

The quadruped copt task emits biped assets. Root resolution fails first, then every joint expression matches nothing. The mapping carries the family and nothing else, the shelf being the family's own business under B6.

```python
# scripts/rsl_rl/train.py:221  BEFORE
_family = "biped"
```

```python
# scripts/rsl_rl/train.py:221  AFTER
# Derived from the task so both registered quadruped tasks reach the quadruped
# package. An unrecognised task is refused rather than defaulted, the failure it
# would otherwise raise being an unresolvable articulation root, thousands of
# lines of importer output later.
_FAMILY_BY_TASK = {
    "Isaac-Quadruped-Copt": "quadruped",
    "Isaac-Limx-SF-Copt": "biped",
}
_family = None
for _key, _fam in _FAMILY_BY_TASK.items():
    if _key in args_cli.task:
        _family = _fam
        break
if _family is None:
    raise ValueError(
        f"Task {args_cli.task!r} maps to no design generator family. "
        f"Known keys: {sorted(_FAMILY_BY_TASK)}."
    )
```

Substring keys rather than exact ids, so the Flat, Rough and Play variants of each family all resolve. `Isaac-Quadruped-Copt-Flat-v0`, `Isaac-Quadruped-Copt-Rough-v0`, `Isaac-Quadruped-Copt-Rough-Play-v0`, `Isaac-Limx-SF-Copt-MoRAL-Flat-v0`, `Isaac-Limx-SF-Copt-MoRAL-Rough-v0` and `Isaac-Limx-SF-Copt-MoRAL-Rough-Play-v0` are the six ids registered at `robots/__init__.py:388-412` and `:646-676`.

### B2 Checkpoint capture needs a terrain level a plane does not have

`terrain_levels` is created only by `_compute_env_origins_curriculum` (`terrain_importer.py:347`), and a plane takes `_compute_env_origins_grid` (`terrain_importer.py:312`). `save()` runs at `it % save_interval == 0`, true at iteration 0. Both Flat ids are registered, so the defect is reachable, and the runs of record are the Rough ids, so this is defensive rather than blocking.

```python
# co_optimisation/utils/env_state.py:144  capture_env_state  BEFORE
"terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
```

```python
# co_optimisation/utils/env_state.py:144  capture_env_state  AFTER
# A plane carries no curriculum state to restore (terrain_importer.py:312).
# None round-trips as a skip, so generated-terrain behaviour is unchanged.
_levels = getattr(terrain, "terrain_levels", None)
"terrain_levels": None if _levels is None else _levels.detach().cpu().clone(),
```

```python
# co_optimisation/utils/env_state.py:196  restore_env_state  AFTER
if state["terrain_levels"] is not None and hasattr(terrain, "terrain_levels"):
    terrain.terrain_levels.copy_(state["terrain_levels"].to(device))
terrain.env_origins.copy_(state["env_origins"].to(device))
```

Same function assumes `push_robot` and `base_velocity` exist (`env_state.py:137-138`). True in every training configuration, false in every play configuration, which never saves. Recorded, not repaired.

### B3 Population must divide the scene

| phase | population | today | after |
|---|---|---|---|
| random, iterations 0 to 12000 | `num_envs / 4` | 1500 | 1536 |
| co-optimisation, 12000 onward | `copt.num_individuals` | 256 | 256 |
| quadruped `num_envs` | | 6000, `6000 % 256 = 96` | 6144, `6144 % 256 = 0` |
| biped `num_envs` | | 4096, `4096 % 256 = 0` | unchanged |

An unequal split gives 96 of the 256 designs one more environment than the other 160, so their fitness estimates carry a different sample size and CMA-ES ranks them against each other on unequal evidence. An assertion rather than a warning forces the scene to be sized for the population. The `num_envs / 4` construction of the random phase divides for any `num_envs` that is a multiple of four, so the only binding constraint is `num_envs % num_individuals == 0`.

```python
# co_optimisation/runners/copt_on_policy_runner.py, in __init__ after
# _num_individuals is read.
# Four pathways index the first num_individuals environments, one per design:
# update.py:208, update.py:272, respawn.py:279, wrappers.py:114.
assert self._num_individuals <= env.num_envs, (
    f"num_individuals={self._num_individuals} > num_envs={env.num_envs}. "
    "Each design needs an environment to be authored through. "
    "Lower copt.num_individuals or raise --num_envs."
)
assert env.num_envs % self._num_individuals == 0, (
    f"num_envs={env.num_envs} is not a multiple of "
    f"num_individuals={self._num_individuals}, so "
    f"{env.num_envs % self._num_individuals} designs would be evaluated over one "
    "more environment than the rest and their fitness would not be comparable. "
    "Use 6144 for the quadruped, 4096 for the biped."
)
```

```python
# co_optimisation/runners/copt_on_policy_runner.py, in _update_morphology,
# immediately after generate_population returns a population.
# The random phase population is num_envs / 4 (train.py:259) and the
# co-optimisation population is copt.num_individuals, so the two phases carry
# different counts and each must divide the scene on its own.
_n = len(self.current_population.get_link_length_params())
assert unwrapped_env.num_envs % _n == 0, (
    f"generation {self.generation} carries {_n} designs over "
    f"{unwrapped_env.num_envs} environments, which does not divide."
)
```

```python
# environments/.../cfg/quadruped/base_env_cfg.py:1408  QuadrupedPFCoptEnvCfg
# 6144 = 24 * 256, the nearest multiple of the population above the 6000 this
# task was tuned at. 5888 = 23 * 256 is the nearest below.
scene: QuadrupedPFSceneCfg = QuadrupedPFSceneCfg(num_envs=6144, env_spacing=env_spacing)
```

The 2.4 per cent rise in environment count raises the rigid patch demand by the same factor, and `gpu_max_rigid_patch_count = 2**19` at `base_env_cfg.py:1433` was set with headroom above 4096 environments, so no change is needed there.

### B4 Undeclared dependencies

| import | site | scope |
|---|---|---|
| `cma` | `usd_generator.py:55`, module scope | every run, `train.py:107` imports the package unconditionally |
| `pandas` | `copt_on_policy_runner.py:23` | every run |
| `cmaes.CatCMAwM` | `usd_generator.py:830`, lazy | `--design_search cat` only |
| `design_generator` | `usd_generator.py:57` | every run |

```python
# co_optimisation/setup.py
install_requires=[
    "torch>=1.4.0", "torchvision>=0.5.0", "numpy>=1.16.4",
    "cma",              # usd_generator.py:55, module scope
    "cmaes>=0.11.0",    # CatCMAwM, usd_generator.py:830
    "pandas",           # design table, copt_on_policy_runner.py:450
],
# design_generator is a submodule of this repository, not a distribution.
# git submodule update --init --recursive, then pip install -e design_generator/
```

Verify in the training container with `python -c "import cma, cmaes, pandas, design_generator"` before anything else.

### B5 Policy type flag admits any string

`--policy-type copt` builds the co-optimisation environment and trains plain PPO on it, with no generator, no swap and no estimator, and says nothing.

```python
# scripts/rsl_rl/cli_args.py:60
choices=["PPO", "HIMPPO", "COPT",
         "quadruped-debug", "go1-debug", "go1-default-debug"],
```

### B6 The actuator shelf belongs to the family

`CatCMAESDesignGenerator` accepts `actuator_groups` and `catalog_names`, both of which restate what `FAMILIES` at `usd_generator.py:176-190` already holds. The generator serves two purposes, pre-training a generalised policy over randomly drawn actuators and searching the actuator choice by CMA-ES, and in both the shelf is a property of the robot family rather than of the launch. Every argument that restates it is a second place to update when an experiment changes.

```python
# co_optimisation/runners/usd_generator.py:765  BEFORE, in __init__
        actuator_groups: tuple[str, ...] | None = None,
        catalog_names: list[str] | None = None,
    ) -> None:
        # Resolved before super().__init__ so that _make_strategy, which it
        # calls, already sees the categorical block.
        self._pending_groups = actuator_groups
        self._pending_names = catalog_names
```

```python
# co_optimisation/runners/usd_generator.py:765  AFTER
# Both arguments and both attributes deleted. The signature is now identical to
# CMAESDesignGenerator's, so the __init__ override can go too and the class
# inherits it, keeping only _resolve_categories, _make_strategy, _check_dimension,
# _dumps, sample_batch, update_with_fitness and the decoding helpers.
```

```python
# co_optimisation/runners/usd_generator.py:789  BEFORE
    def _resolve_categories(self) -> None:
        if getattr(self, "cat_names", None) is not None:
            return
        groups = self._pending_groups or self.family.groups
        unknown = set(groups) - set(self.family.groups)
        if unknown:
            raise ValueError(...)
        names = (self._pending_names if self._pending_names is not None
                 else [a.name for a in self.family.catalog()])
        by_name = {a.name: a for a in self.family.catalog()}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(...)
        if len(names) < 2:
            raise ValueError(...)
        self.cat_groups = tuple(groups)
        self.cat_names = list(names)
        bare = [n for n in self.cat_names
                if getattr(by_name[n], "armature", 0.0) == 0.0]
        if bare:
            print("[design] warning: ...")
```

```python
# co_optimisation/runners/usd_generator.py:789  AFTER
    def _resolve_categories(self) -> None:
        if getattr(self, "cat_names", None) is not None:
            return
        # The family owns both the groups and the shelf, so an experiment changes
        # by editing the catalogue or the family, both inside design_generator.
        # Family.catalog() returns only the simulation-complete entries (B7), so
        # the unknown-group, missing-name and bare-armature checks the deleted
        # arguments made necessary are gone with them.
        groups = self.family.groups
        names = {a.name: a for a in self.family.catalog()}
        if len(names) < 2:
            raise ValueError(
                "A categorical search needs at least two simulation-complete "
                f"catalogue entries, the {self.family.name} family offers "
                f"{sorted(names)}. See design_generator/{self.family.name}/"
                "catalogs/actuators.json."
            )
        self.cat_groups: tuple[str, ...] = tuple(groups)
        self.cat_names: list[str] = list(names)
```

```python
# scripts/rsl_rl/train.py:245  the two lines deleted from the cat branch
                catalog_names=[
                    "TRON1-ABAD", "TRON1-HIP", "TRON1-KNEE", "TRON1-ANKLE",
                ],
```

The categorical block keeps one variable per group over the whole shelf, so a design may drive its abduction joint with the unit identified for the knee. That is a deliberate part of the search and is unchanged.

### B7 Most catalogue entries cannot be simulated

`IdentifiedActuator.sim_params` returns only the keys the entry states, the docstring at `actuator_catalog.py:534-540` recording the intent that "a consumer keeps whatever its own configuration holds rather than being handed a fabricated number". That intent is right for the initial asset configuration and wrong for a per-generation overwrite, where what the consumer holds after generation one is the previous design's value rather than the asset's.

| | quadruped | biped |
|---|---|---|
| Entries | 14 | 15 |
| Simulation complete | `A1-ABAD`, `A1-HIP`, `A1-KNEE` | `TRON1-ABAD`, `-HIP`, `-KNEE`, `-ANKLE` |
| Incomplete | 11, `Unitree-A1`, `Unitree-B1`, `MIT-Cheetah-3`, `TMotor-AK80-9`, `TMotor-AK80-6`, `TMotor-AK10-9`, `ANYdrive-3`, `HEBI-X8-16`, `ODrive-D5065-DD`, `MAB-MD80`, `Bota-Roboticer-50` | the same 11 |
| Keys an incomplete entry returns | `effort_limit`, `velocity_limit`, `saturation_effort`, `armature` | as quadruped |
| Keys it omits | `friction_static`, `friction_dynamic`, `activation_vel`, `stiffness`, `damping` | as quadruped |
| `armature` it returns | 0.0, neither `armature_value` nor `rotor_inertia` stated (`actuator_catalog.py:528`) | 0.0 |

Two consequences, both of which B6 turns from dormant into certain, the shelf widening from the three names the shortlist held to all fourteen, so eleven of every fourteen categorical draws land on an incomplete entry.

First, `armature` is written as zero over the identified reflected inertia, and a joint with no reflected inertia is a different plant. Second, `apply_actuator_params` writes only the keys it is handed (`respawn.py:282`), so friction, the activation velocity and the two gains stay at the previous generation's design and fitness becomes a function of evaluation order.

The filter belongs in `design_generator`, which owns the specification.

```python
# design_generator/common/actuator_catalog.py, beside SIM_PARAM_KEYS at :282
def is_sim_complete(entry) -> bool:
    """True when sim_params() states every key a per-generation overwrite sets.

    A partial entry is worse than a wrong one. apply_actuator_params writes the
    keys it is given and leaves the rest at whatever the previous design left
    (respawn.py:282), so an entry that omits damping makes fitness depend on the
    order designs were evaluated in. armature is tested by value because
    sim_params always emits the key, falling back to zero when the entry states
    neither armature_value nor rotor_inertia (actuator_catalog.py:528).
    """
    params = entry.sim_params()
    return set(params) >= set(SIM_PARAM_KEYS) and params["armature"] > 0.0
```

```python
# design_generator/quadruped/actuator.py and design_generator/biped/actuator.py,
# beside ACTUATOR_CATALOG. Kept as a separate name so the cost and mass models
# keep the whole shelf while the search sees only what can be simulated.
SIM_ACTUATOR_CATALOG: List["ActuatorLike"] = [
    a for a in ACTUATOR_CATALOG if is_sim_complete(a)
]
```

```python
# co_optimisation/runners/usd_generator.py:172  Family.catalog
    def catalog(self) -> list:
        """The entries a design may pick, being those that can be simulated.

        The eleven entries each family's catalogue states from published data
        alone carry no reflected inertia and no friction or gains, so a design
        picking one would run on a fabricated plant and would silently inherit
        the previous generation's gains. They are excluded here rather than in
        the caller so that no argument above this module names an actuator.
        """
        return list(self.module.actuator.SIM_ACTUATOR_CATALOG)

    def full_catalog(self) -> list:
        """Every entry, for cost and mass accounting that does not simulate."""
        return list(self.module.actuator.ACTUATOR_CATALOG)
```

The filter reproduces today's biped behaviour exactly, the four `TRON1-` names being precisely the shortlist that `train.py:245` passed, and gives the quadruped the three `A1-` names it could never reach. Completing the eleven entries needs identification data the workspace does not hold, so record the exclusion in `actuators.json` `_schema` notes rather than leaving a reader to infer it.

## 4. The nominal pose, D1 and D7

### 4.1 What is fixed and what the search moves

The task is accurate and efficient velocity tracking. Nothing in it fixes a standing height, and a design search whose objective is velocity tracking should be free to choose the fold it stands at, exactly as it chooses its link lengths. The nominal pose is therefore a design variable, and the rewards that reference it must take their targets from the design rather than from a constant.

| quantity | quadruped | biped |
|---|---|---|
| Nominal standing height, kinematic | 0.291952 m | 0.804343 m |
| How it is composed | `2 x 0.213 x cos(0.884337) = 0.269952` drop plus the 0.022 foot sphere | `0.1875 + 0.039565 + 0.245746 + 0.281532` chain plus the 0.050 sole |
| Independent confirmation | `quadruped_identified_cfg.py:90-92` comment states 0.292 | `design_generator/biped/variable_params.json` states `body_height = 0.80435432` |
| Where the fold lives | joint angles, `.._thigh_joint = 0.884337`, `.._calf_joint = -1.768673` | link frames, thigh 35.0 deg forward and the shank frame rotated 40.0 deg, every joint nominal 0 |
| Built in fold at the nominal pose | 101.3 deg of knee flexion | 72.2 deg between the thigh and shank directions, corrected to 75.0 in section 10 |
| Configured `pen_base_height` target | 0.292, within 0.05 mm of nominal | 0.75, which is 54 mm below nominal |
| Configured spawn height | 0.32, giving 28 mm of clearance | 0.80, putting the soles 4.3 mm below ground |
| Soft knee range | fold in [1.0053, 2.6075] rad | knee offset in [-1.0943, 1.0534] rad |
| Standing height over the length box [0.75, 1.25] | [0.2245, 0.3594] | [0.6759, 0.9328] |

The biped's two numbers are new in this pass and correct the first pass, which gave 0.641 and 0.967 against a target of 0.75. The correct statement is that the biped's standing height is affine in the two length scales, `h = 0.290497 + 0.245746 s1 + 0.268100 s2`, which is 0.804343 at the donor and moves plus or minus 0.1284 m over the box, so the configured target of 0.75 sits 54 mm below the centre of that interval rather than at it.

### 4.2 Why a fixed target prices the design

```
quadruped, weight -5.0, target pinned at 0.292              (base_env_cfg.py:1088)

  s = 0.75 ──── stands at 0.2245 m ───┐
  s = 1.00 ──── stands at 0.2920 m    ├── cost = 5 * (h - 0.292)^2 per step
  s = 1.25 ──── stands at 0.3594 m ───┘

  cost at either extreme = 5 * 0.0675^2 * 958.68 = 21.84 per episode
  cost at the donor      = 0.2723 per episode, measured
```

Measured against run `quadruped_flat/2026-09-16_07-01-06`, artefact `artefacts/quadruped_flat-2026-09-16_07-01-06-reward-budget/reward_budget.txt`.

| quantity | value |
|---|---|
| `Train/mean_reward` | 89.81 |
| `Train/mean_episode_length` | 958.68 |
| `pen_base_height` realised at the donor design | -0.2723 per episode, 0.23 per cent of the negative budget |
| implied root mean square height error held by the policy | 7.5 mm |
| penalty at either extreme of the length box | -21.84 per episode, 80 times the donor |
| share of that run's return | 24 per cent |
| share of a return rescaled to the tree's tracking weights of 25 and 12.5 | 5.5 per cent of about 397 |

That run trained at target 0.33, tracking 10 and 5, `feet_air_time` 20.0, `pen_feet_hold` -0.5, against the tree's 0.292, 25, 12.5, 10.0, -0.1. Both figures are given for that reason.

Five per cent of the fitness signal is a deterministic function of the design, symmetric about the donor, and CMA-ES maximises exactly that quantity, so it converges on the design whose natural height equals the target, which is the design it started from. Introducing the fold as a search variable widens the height interval further, so this must be closed before the fold is opened, not after.

### 4.3 The kinematic constraints and the constants that define them

Revised 2026-10-01. Sections 4.3 to 4.9 of the 2026-09-30 revision, which parameterised the quadruped by one knee flexion and left the biped empty, are superseded by sections 4.3 to 4.13 below and are recorded in section 10. Both robots now take two searched angles, a hip pitch offset and a knee offset, and the biped's ankle offset is computed from them.

```
quadruped, 3 sagittal angles about +y, q1 thigh, q2 knee (relative), foot sphere r = 0.022

  hip ──┐ q1                 foot under hip   l1 sin(q1) + l2 sin(q1 + q2) = 0
        │                    height           h = l1 cos(q1) + l2 cos(q1 + q2) + r
      thigh l1                                
        │                    height needs BOTH q1 and q2, and the foot constraint
      knee ─┐ q2             removes one degree of freedom, so one angle is free
            │
          shank l2           donor   q1 = 0.884337, q2 = -1.768673, h = 0.292000
            │
          foot (sphere, no ankle)


biped, angles about +y, t thigh, s shank, a ankle, all ABSOLUTE in the base frame

  hip ──┐ t = 35.000                     ankle under hip   dx = x_ankle - x_hip
        │                                dx = -0.0277 - l1 sin(t) - l2 sin(s)
      thigh l1                           height            h = 0.290497 + l1 cos(t) + l2 cos(s)
        │
      knee ─┐ s = -40.0002               sole flat         t + (s - t) + (a - s) = 0, so the
            │                                              foot's absolute angle is zero, or
          shank l2                                         equivalently the three offsets sum to 0
            │
      ankle ─┐ a = 0.0037 (flat)         donor   dx = +0.0252, h = 0.804357
               ▔▔▔▔▔▔▔▔▔▔▔▔▔▔ sole 0.200 long, protruding forward of the ankle, parallel to the ground
```

| quantity | quadruped | biped |
|---|---|---|
| Joints that set the standing height | hip pitch and knee, both required | hip pitch, knee and ankle, all three required |
| Constraint on the foot | foot directly below the hip actuator | ankle directly below the hip, sole flat on the ground |
| Searched offsets | hip pitch, knee | hip pitch, knee |
| Computed offset | none, there is no ankle | ankle, `ankle = -(hip + knee)` |
| Donor residual of the foot constraint | 0.0 mm | +25.2 mm, the ankle sits forward of the hip by design |
| Residual at the scale corners, `dx` over `s1, s2` in {0.75, 1.25} | 0.0, -82.4, +82.4, 0.0 mm | +12.0, +124.5, -74.1, +38.4 mm |
| Where the fold lives in the XML today | nowhere, only the MuJoCo keyframe `qpos_leg` at `quadruped/builder.py:501` | body frames, `THIGH_DIR_XZ`, `SHANK_QUAT`, `ANKLE_QUAT` at `biped/builder.py:134-151` |

The last two rows of the residual line are the first finding of this pass. The foot constraint does not hold at the donor angles once the lengths move, so a uniform draw of the scales alone already leans the quadruped by up to 82 mm and the biped by up to 125 mm, and the constraint must be evaluated per design, not once.

The second finding is that the constants the feedback names are not the donor.

| source | thigh | knee | foot x under hip | standing height |
|---|---|---|---|---|
| `quadruped/constant_params.json:53-54` | 70.0 deg | -154 deg | -11.7 mm | 0.1171 m |
| `quadruped_identified_cfg.py:100-103`, the pose every run of record trained at | 50.668778 deg (0.884337 rad) | -101.337498 deg (-1.768673 rad) | 0.0 mm | 0.2920 m |

The constants are read at `quadruped/builder.py:148-149` and consumed only by the MuJoCo home keyframe (`qpos_leg`, `:501`), which Isaac Lab never reads, so nothing has exercised the disagreement. The Isaac document must be built from the donor pose or every quadruped design stands 175 mm lower and 12 mm off its hip. The edit is to the constants, not to the builder.

```json
// design_generator/quadruped/constant_params.json:53-54
// BEFORE
  "THIGH_BEND_DEG": 70.0,
  "SHANK_BEND_DEG": -154,
// AFTER, the pose quadruped_identified_cfg.py:100-103 trains at
  "THIGH_BEND_DEG": 50.668778,
  "SHANK_BEND_DEG": -101.337498,
```

The MuJoCo keyframe then stands at the donor pose, which fixes a stale home pose rather than breaking one, since `base_z = body_height` at `builder.py:483` already assumes feet on the ground. `GEOMETRY_RULES.md:250-251` states 70.0 and -70.0 and is already out of step with the JSON, so both lines are corrected in the same pass.

### 4.4 Sign convention and what the offsets mean

All offsets are radians of rotation about the body frame's +y axis, the convention of the `*_BEND_DEG` constants, so a positive thigh offset tilts the thigh backward on the quadruped and the biped's `quat_y` convention is shared. The joint angle that reaches the same pose is `+offset` on the quadruped, whose joint axes are `(0, 1, 0)`, and `-offset` on the biped, whose axes are `(0, -1, 0)`.

| spec field | quadruped | biped | meaning |
|---|---|---|---|
| `thigh_bend_offset` | rad, `None` means 0 | rad, `None` means 0 | rigid rotation of the whole leg about the hip, added to `THIGH_BEND_DEG` |
| `knee_bend_offset` | rad, `None` means 0 | rad, `None` means 0 | rotation of the shank relative to the thigh, added to `SHANK_BEND_DEG` on the quadruped and to the relative shank angle on the biped |
| `ankle_bend_offset` | must be `None`, asserted by the builder | rad, `-(thigh + knee)` | rotation of the foot relative to the shank, keeps the sole flat |

Applied only when `for_isaaclab` is True, exactly as the feedback states, so every standalone MuJoCo use of the builder is byte identical.

The offsets that the search draws are not the offsets the spec carries. The spec carries the totals, because the foot constraint of section 4.3 moves the hip angle by up to 0.58 rad as the lengths move, and the draw is the part of the hip angle that is a free choice.

| parameter drawn | range | type | becomes |
|---|---|---|---|
| `thigh_bend_offset` | (-0.10, 0.10) rad | `offset` | `spec.thigh_bend_offset = solved + draw` under `"solved"`, `= draw` under `"default"` |
| `knee_bend_offset` | (-0.25, 0.25) rad | `offset` | `spec.knee_bend_offset = draw` |
| none | | | `spec.ankle_bend_offset = -(thigh + knee)`, biped only |

`NOMINAL_HIP_REFERENCE` selects between the two readings of the hip draw. Under `"default"` the draw is taken about the constants as the feedback words it and the foot constraint is a check followed by a repair. Under `"solved"` the draw is a lean about the hip angle that puts the foot under the hip, so the constraint holds by construction to within the lean. The measured consequence follows.

| reading | fraction of uniform draws inside the tolerance without repair | consequence for CMA-ES |
|---|---|---|
| `"default"`, quadruped, hip (-0.10, 0.10), knee (-0.25, 0.25), tolerance `1.05 (l1 + l2) sin 0.10` | 69.3 per cent | 31 per cent of hip draws are overwritten by the repair, the hip coordinate is a plateau over a third of its box |
| `"default"`, biped, same box and tolerance | 69.7 per cent | the same |
| `"default"`, quadruped, same box, a fixed tolerance of 0.02 m | 33.9 per cent | two thirds overwritten, the reason the tolerance follows the lean and is not a constant |
| `"solved"`, both | 100 per cent by construction | the hip coordinate is a lean everywhere, continuous and never overwritten |

Recommendation, `"solved"`, which is the default of section 4.5, because under the literal reading a third of the designs, those whose lengths are furthest from equal, have their hip draw discarded and the CMA-ES hip coordinate is flat there. `"default"` is retained as the alternative because it is the reading the feedback states, and the switch costs one branch.

### 4.5 The solve and the checks, inside `usd_generator.py`

```python
# co_optimisation/runners/usd_generator.py, new, beside _biped_links.
# The design generator is handed a finished DesignSpec and checks nothing about
# the pose. Everything below is the planner's, so a spec produced elsewhere is
# unconstrained and a spec produced here is constrained.
import math

# Which hip reference the draw is taken about, section 4.4.
NOMINAL_HIP_REFERENCE = "solved"        # "solved" or "default"

# The ankle's x offset from the hip at the donor, so zero draws at unit scales
# reproduce the donor exactly. The quadruped's is zero.
DX_DONOR = {"quadruped": 0.0, "biped": 0.025204}
```

```python
def solve_nominal_pose(family: str, baseline, spec_lengths: dict[str, float],
                       draws: dict[str, float],
                       reference: str = NOMINAL_HIP_REFERENCE) -> dict[str, float | None]:
    """Offsets for one design, from its lengths and the two drawn angles.

    quadruped, absolute thigh q1 = Q1 + dh, knee q2 = Q2 + dk, with Q1, Q2 the
    constants. Foot under hip, with phi = -(Q2 + dk) the interior knee flexion,
        q1* = atan2(l2 sin phi, l1 + l2 cos phi)
    so the solved hip offset is  dh* = q1* - Q1.

    biped, absolute thigh t = T + dh, shank s = S + dh + dk, with T = THIGH_BEND_DEG + 180,
    S = SHANK_BEND_DEG. The ankle is under the hip when
        dx(dh, dk) = -0.0277 - l1 sin(t) - l2 sin(s) = DX_DONOR
    which has one root in dh for a given dk, found by bisection on [-1.0, 1.0] rad.
    The ankle offset keeps the sole flat, da = -(dh + dk).
    """
    ...
```

```python
def check_nominal_pose(family: str, spec, hip_lean_max: float) -> None:
    """Assert the constraints of section 4.3 on a finished design. Raises ValueError."""
    # 1. foot under hip, quadruped dx = l1 sin(q1) + l2 sin(q1 + q2),
    #    biped dx - DX_DONOR with dx of section 4.3. Tolerance follows the lean the
    #    draw is allowed, so no constant to calibrate.
    tol = 1.05 * (spec.thigh_length + spec.shank_length) * math.sin(hip_lean_max)
    # 2. sole flat, biped, |dh + dk + da| < 1e-9.
    # 3. standing height positive and inside HEIGHT_BOX, the corridor the first run
    #    is allowed, (0.18, 0.45) quadruped and (0.55, 1.05) biped.
    # 4. the nominal pose inside the hardware joint range, the constants' angle plus
    #    the offset inside HIP_RANGE and KNEE_RANGE, with ANKLE_RANGE for the biped,
    #    each narrowed by soft_joint_pos_limit_factor = 0.9 about its midpoint.
```

| check | quadruped | biped | on failure |
|---|---|---|---|
| foot under hip, `abs(dx - dx_target) <= tol` | `dx_target = 0` | `dx_target = +0.0252`, the donor's | raise, `_build_xml`'s fallback at `usd_generator.py:449` repeats the last feasible design and logs the reason |
| sole flat, offsets sum to zero | not applicable | to `1e-9` | raise, it is a code defect, not a design |
| standing height inside the corridor | (0.18, 0.45) | (0.55, 1.05) | raise as above |
| pose inside the hardware range | hip `[-1.0472, 4.18879]`, knee `[-2.69653, -0.916298]` | hip `[-1.39626, 1.01229]`, knee `[-1.21357, 1.17267]`, ankle `[-0.791917, 1.30899]`, in joint sign | raise as above |
| `ankle_bend_offset is None` | asserted | not applicable | raise, a quadruped has no ankle |

The fallback at `:449` is today unreachable (X6), and these checks make it reachable for the first time, which is the intended use. The repeat is silent, so the check logs the substituted design's index and the reason once per generation.

Heights over the draw box, computed from the chains above.

| robot | pose at unit scales, `solved` hip, knee offset | height | hip offset solved |
|---|---|---|---|
| quadruped | knee -0.22 | 0.2542 m | +0.110 rad |
| quadruped | knee 0 | 0.2920 m | 0.000 rad |
| quadruped | knee +0.22 | 0.3265 m | -0.110 rad |
| biped | knee -0.25 | 0.7511 m | +0.130 rad |
| biped | knee 0 | 0.8044 m | 0.000 rad |
| biped | knee +0.25 | 0.8496 m | -0.131 rad |

Over both the lengths and the draws, with the `solved` hip and the lean of 0.10 of the table in section 4.4, the quadruped stands in [0.194, 0.402] m with a total hip offset in [-0.415, 0.528] rad and a worst foot residual of 36.3 mm, and the biped in [0.636, 0.983] m with a total hip offset in [-0.369, 0.424] rad, a hip plus knee sum in [-0.436, 0.346] rad, so an ankle offset in [-0.346, 0.436] rad, and a worst residual of 68.3 mm about its donor value. The biped's residual is larger because its lever, 0.65 m of thigh plus shank at the unit scales, is the longer. Both height intervals sit inside the corridors of the table above, and the largest tolerance the formula yields is 55.8 mm for the quadruped and 85.2 mm for the biped, each above the worst residual, which is the property that makes the formula a safe bound.

### 4.6 `DesignSpec`, `from_dict` and the builders

```python
# design_generator/quadruped/generate.py and design_generator/biped/generate.py,
# DesignSpec, after the clearance fields. Fixed-constant tier, so absent from
# vector(), from_vector(), labels() and DesignBounds.
from typing import Optional

    # Nominal pose offsets, radians about +y. None leaves the constants'
    # default (builder.THIGH_BEND_DEG and its siblings) untouched, and the
    # offsets are read only when build_xml is called with for_isaaclab=True.
    thigh_bend_offset: Optional[float] = None
    knee_bend_offset: Optional[float] = None
    ankle_bend_offset: Optional[float] = None      # quadruped, must stay None

    @staticmethod
    def from_dict(data: dict) -> "DesignSpec":
        """A new DesignSpec from a flat dict of its fields. Unknown keys raise."""
        known = {f.name for f in fields(DesignSpec)}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"unknown DesignSpec fields {sorted(unknown)}")
        return DesignSpec(**data)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
```

```python
# load_design_spec in both files, the one other construction site.
-    return DesignSpec(**data)
+    return DesignSpec.from_dict(data)
```

`from dataclasses import dataclass, field, fields` is the import change. `to_dict` is shallow on purpose, `dataclasses.asdict` recursing and copying, which the fields here do not need and `replace` is what is being retired.

```python
# design_generator/quadruped/builder.py, after :149. The frames of one design.
def pose_frames(spec, for_isaaclab: bool = True) -> dict:
    """Absolute thigh angle and relative knee angle, radians about +y."""
    q1, q2 = THIGH_BEND_RAD, SHANK_BEND_RAD
    if for_isaaclab:
        if spec.ankle_bend_offset is not None:
            raise ValueError("a quadruped has no ankle, ankle_bend_offset must be None")
        q1 += spec.thigh_bend_offset or 0.0
        q2 += spec.knee_bend_offset or 0.0
    return {"thigh": q1, "knee": q2}
```

```python
# design_generator/quadruped/builder.py, _leg_xml. The thigh body and the shank
# body each gain a quat, and nothing else moves, the boxes, inertials and the foot
# all being expressed in their own link frames and so rotating with them.
fr = pose_frames(spec, for_isaaclab)
thigh_q = f' quat="{_quat_y(math.degrees(fr["thigh"]))}"' if for_isaaclab else ""
shank_q = f' quat="{_quat_y(math.degrees(fr["knee"]))}"' if for_isaaclab else ""
...
<body name="thigh_{leg}_Link" pos="{_vec(*thigh_pos)}"{thigh_q}>
  ...
  <body name="shank_{leg}_Link" pos="0 0 {_fmt(-L1)}"{shank_q}>
```

The quat is emitted only under `for_isaaclab`. Emitted always it would double the pose in the MuJoCo document, whose keyframe `qpos_leg` already carries it, and the keyframe is itself dropped under `for_isaaclab` at `builder.py:431`.

The joint ranges move with the fold. At present the hip range `[-1.0472, 4.18879]` and the knee range `[-2.69653, -0.916298]` are about the folded pose through the joint angle, and once the fold is in the frames the zero of each joint is the fold, so the ranges must be shifted by the constants' fold or the nominal pose sits at a joint angle of zero in a range that never contained the fold.

```python
# quadruped/builder.py, the two joint lines of _leg_xml. Shift by the CONSTANTS'
# fold only. The per design offset is deliberately not subtracted, so the
# in place editor never has to author joint limits, section 4.9.
hip_lo, hip_hi = (v - THIGH_BEND_RAD for v in HIP_RANGE) if for_isaaclab else HIP_RANGE
knee_lo, knee_hi = (v - SHANK_BEND_RAD for v in KNEE_RANGE) if for_isaaclab else KNEE_RANGE
```

| range | before, folded joint angle | after, about the donor frame | note |
|---|---|---|---|
| quadruped hip | `[-1.0472, 4.18879]` | `[-1.9315, 3.3045]` | `v - 0.884337` |
| quadruped knee | `[-2.69653, -0.916298]` | `[-0.9279, 0.8524]` | `v + 1.768673` |
| biped | unchanged | unchanged | already about the fold |

The consequence of not subtracting the offset is that a design at hip offset 0.5 rad has its absolute hip envelope displaced by 0.5 rad against the hardware range, which the check of section 4.5 bounds by asserting the absolute nominal angle inside the hardware range.

```python
# design_generator/biped/builder.py. The frame scheme, used only when offsets are
# given under for_isaaclab, so a spec with all three None takes the existing path
# and the donor document is byte identical.
def pose_frames(spec, for_isaaclab: bool = True) -> dict:
    """Body quats of the thigh, shank and ankle, degrees about +y, each relative to its parent."""
    t0 = THIGH_BEND_DEG + 180.0                  # 35.000, the thigh direction as a +y rotation of -z
    on = for_isaaclab
    dh = math.degrees(spec.thigh_bend_offset or 0.0) if on else 0.0
    dk = math.degrees(spec.knee_bend_offset or 0.0) if on else 0.0
    da = math.degrees(spec.ankle_bend_offset or 0.0) if on else 0.0
    return {
        "thigh": t0 + dh,                        # 35.000000 + dh
        "shank": (SHANK_BEND_DEG - t0) + dk,     # -75.000189 + dk, relative to the thigh
        "ankle": ANKLE_BEND_DEG + da,            # 40.003913 + da, relative to the shank
    }
```

The flat sole is the sum of the three relative rotations. The donor's thigh, shank and ankle are `35.000000`, `-75.000189` and `40.003913` degrees, which sum to `+0.003724`, the donor's own flatness residual and the same figure the legacy scheme gives as `SHANK_BEND_DEG + ANKLE_BEND_DEG`, so the frame scheme at zero offsets is the legacy geometry. With offsets the sum becomes `0.003724 + dh + dk + da` degrees, so the sole is flat exactly when `da = -(dh + dk)`, the rule the solve of section 4.5 applies and the check asserts to `1e-9`.

```python
# biped/builder.py, _leg_xml. Frame scheme. The thigh link takes a quat, its box
# and inertial lose theirs and run along local -z as the quadruped's do, and the
# shank is hung at (0, 0, -thigh_length) in the thigh frame.
if offsets_given and for_isaaclab:
    fr = pose_frames(spec)
    thigh_attr = f' quat="{_quat_y(fr["thigh"])}"'
    shank_pos_rel = f"0.0 0.0 {_fmt(-spec.thigh_length)}"
    shank_attr = f' quat="{_quat_y(fr["shank"])}"'
    ankle_attr = f' quat="{_quat_y(fr["ankle"])}"'
    box_center_local = np.array([0.0, 0.0, -thigh_available / 2.0])
    box_quat = None                              # box_inertial_block(..., quat=None) at :288
else:
    ... existing path ...
```

The two schemes describe the same geometry. A test in `co_optimisation/tests/` builds the donor under both, parses every body's world pose with `numpy` from the emitted `pos` and `quat`, and asserts agreement to 1e-6 m, which is also the test that the proof above is right. `offsets_given` is `any(v is not None for v in (spec.thigh_bend_offset, spec.knee_bend_offset, spec.ankle_bend_offset))`.

The reason for the frame scheme is the in place editor of section 4.9. The legacy biped thigh is oriented by a box quat and an inertial quat that depend on `THIGH_DIR_XZ`, so moving the hip offset would need the box `xformOp:orient`, the link's `physics:principalAxes` and the joint rotation all rewritten in place. In the frame scheme the box is axis aligned in its link and only joint rotations move.

### 4.7 The asset configurations

| file | line | before | after |
|---|---|---|---|
| `quadruped_identified_cfg.py` | 100-103 | `".*_thigh_joint": 0.884337`, `".*_calf_joint": -1.768673` | both 0.0, the fold now in the frames, left at the old values the pose is doubled and the feet are driven through the ground |
| `solefoot_identified_cfg.py` | 86-96 | `".*_Joint": 0.0` | unchanged |
| `quadruped_identified_cfg.py` | 90-93 | spawn `z = 0.32`, comment 0.292 plus 28 mm | spawn `z` stays as the fallback until the first swap, comment updated, the runner overwrites `default_root_state[:, 2]` per environment |

`apply_standing_height` of section 4.10 asserts that the pitch joints' `default_joint_pos` are zero on every environment whose family encodes its pose in the frames, which is the check that catches a stale configuration. A consequence worth recording is that the action offset staleness hazard of the earlier revision disappears, since `default_joint_pos` is zero for every design and every generation, so `JointPositionAction._offset` (`joint_actions.py:195`) never needs refreshing and the in place path's lack of a refresh (`respawn.py:176-183`) stops mattering.

The quadruped asset config's other consumers of the old nominal, `soft_joint_pos_limit_factor = 0.9` and the PD gains, are unchanged. The soft limits are derived from the shifted USD limits of section 4.6.

### 4.8 `usd_generator.py`, one typed range structure and the renames

```python
# co_optimisation/runners/usd_generator.py, replacing DEFAULT_PARAM_RANGES at :196
# and CMAES_PARAM_RANGES at :216. One structure. Type is one of offset, scale, absolute.
ParamRange = tuple[str, tuple[float, float]]
PARAM_TYPES = ("offset", "scale", "absolute")

DEFAULT_PARAM_RANGES: dict[str, ParamRange] = {
    "thigh_length_scale":    ("scale",  (0.75, 1.25)),
    "shank_length_scale":    ("scale",  (0.75, 1.25)),
    "thigh_bend_offset":     ("offset", (-0.10, 0.10)),    # rad, section 4.4
    "knee_bend_offset":      ("offset", (-0.25, 0.25)),    # rad
    "abad_stiffness_scale":  ("scale",  (0.909, 1.091)),
    "abad_damping_scale":    ("scale",  (0.889, 1.111)),
    "hip_stiffness_scale":   ("scale",  (0.875, 1.125)),
    "hip_damping_scale":     ("scale",  (0.846, 1.154)),
    "knee_stiffness_scale":  ("scale",  (0.833, 1.167)),
    "knee_damping_scale":    ("scale",  (0.750, 1.250)),
    "ankle_stiffness_scale": ("scale",  (0.800, 1.200)),
    "ankle_damping_scale":   ("scale",  (0.800, 1.200)),
}

# CMA-ES searches these four. A tuple of names, not a second range structure.
CMAES_PARAM_KEYS = ("thigh_length_scale", "shank_length_scale",
                    "thigh_bend_offset", "knee_bend_offset")

# The DesignSpec field each spec-bound parameter writes. Parameters absent here
# (the stiffness and damping scales) are consumed by _actuator_params instead.
SPEC_PARAM_FIELDS: dict[str, str] = {
    "thigh_length_scale": "thigh_length",
    "shank_length_scale": "shank_length",
    "thigh_bend_offset":  "thigh_bend_offset",
    "knee_bend_offset":   "knee_bend_offset",
}
```

`LENGTH_SCALE_FIELDS` at `:222-225` is replaced by `SPEC_PARAM_FIELDS`, and its one other reader, `_spec_from_scales`, is rewritten below.

Six keys are deprecated and removed with their mapping, `joint_effort_scale`, `velocity_limit_scale`, `saturation_effort_scale`, `armature_scale`, `friction_static_scale` and `friction_dynamic_scale`. Their only consumer is the loop over `ACTUATOR_SCALE_KEYS` in `_actuator_params` (`:430-433`), whose keys never occur in a `param_ranges` that does not contain them, so removing them changes no behaviour. A caller that still passes one in `param_ranges` raises at construction, which is a louder failure than the silent no-op it is today.

```python
# co_optimisation/runners/usd_generator.py, deleted.
-ACTUATOR_SCALE_KEYS: dict[str, str] = {...}                        # :229-236
# _actuator_params, the loop at :430-433 is deleted and the docstring and the
# module docstring's six bullet lines (lines 14-21) with it.
```

```python
# DesignGeneratorBase.__init__, :331-343. Validate the types once.
self.param_ranges: dict[str, ParamRange] = {**DEFAULT_PARAM_RANGES}
if param_ranges is not None:
    self.param_ranges.update(param_ranges)
for name, (kind, _) in self.param_ranges.items():
    if kind not in PARAM_TYPES:
        raise ValueError(f"parameter {name!r} has unknown type {kind!r}")
self._current_parameters: dict[str, float] | None = None       # was _current_scales
```

The renames, with every site enumerated.

| old | new | sites |
|---|---|---|
| `DesignGeneratorBase._current_scales` | `_current_parameters` | `:343`, `RandomDesignGenerator._generate_individual` `:508`, `CMAESDesignGenerator` `:652`, `GrowingDesignGenerator` `:719`, `CatCMAESDesignGenerator` `:924`, and every reader of it, which `grep -n _current_scales` must return empty on completion |
| `_sample_scales` | `_sample_parameter` | `:387` and its callers in the random and growing generators |
| `_sample_scales_v2` | `_sample_parameter_v2` | `:390` and its callers. The feedback names both `_sample_parameter`, which two methods cannot share, so the second keeps its `_v2` suffix and takes the extra `scale` argument |
| local `scales` and parameter `scales` | `parameters` | `_sample_*`, `_spec_from_scales`, `_actuator_params`, `_denormalise` and the generators' locals |
| `_spec_from_scales` | `_spec_from_parameters` | `:401` and its three callers |

```python
# co_optimisation/runners/usd_generator.py:387, both samplers, bodies unchanged.
def _sample_parameter(self, rng: np.random.Generator) -> dict[str, float]:
    return {k: float(rng.uniform(lo, hi)) for k, (_, (lo, hi)) in self.param_ranges.items()}

def _sample_parameter_v2(self, rng: np.random.Generator, scale: float) -> dict[str, float]:
    parameters = {}
    for k, (_, (lo, hi)) in self.param_ranges.items():
        mean = (lo + hi) / 2
        parameters[k] = float(rng.uniform((lo - mean) * scale + mean,
                                          (hi - mean) * scale + mean))
    return parameters
```

`CMAESDesignGenerator.__init__` at `:553` and `_denormalise` at `:656` iterate `self.param_ranges`, so both read the nested tuple.

```python
# :553 BEFORE
self.param_ranges = {**CMAES_PARAM_RANGES}
# AFTER
self.param_ranges = {k: self.param_ranges[k] for k in CMAES_PARAM_KEYS}
# :656 _denormalise, the range unpack only.
-for (key, (lo, hi)), u in zip(self.param_ranges.items(), unit):
+for (key, (_, (lo, hi))), u in zip(self.param_ranges.items(), unit):
```

The optimiser is type agnostic as the feedback says, and the continuous block grows from two dimensions to four, so `_check_dimension` at `:595` correctly refuses a checkpoint written before the change. The CMA-ES step size and bounds are normalised to the unit hypercube, which makes the offsets of range 0.2 and 0.5 rad and the scales of range 0.5 commensurate without any change.

```python
# co_optimisation/runners/usd_generator.py:401, replacing _spec_from_scales.
def _spec_from_parameters(self, parameters: dict[str, float],
                          picks: dict[str, str] | None = None):
    """A new DesignSpec carrying the lengths, the nominal pose and the actuator picks.

    Built from the baseline's field dict through DesignSpec.from_dict, not
    dataclasses.replace, so every field is named once in the dict and the spec the
    caller receives is a fresh object. Lengths are rounded to two decimals, the
    quantisation the URDF era generator applied, and angles to three decimals
    (0.057 deg), which keeps the conversion cache hit rate of a near converged
    CMA-ES population.
    """
    fields_ = self.baseline.to_dict()
    draws: dict[str, float] = {}
    for name, field in SPEC_PARAM_FIELDS.items():
        if name not in parameters:
            continue
        kind = self.param_ranges[name][0]
        value = parameters[name]
        if kind == "scale":
            fields_[field] = round(fields_[field] * value, 2)
        elif kind == "offset":
            draws[field] = value                      # resolved against the lengths below
        else:                                         # "absolute"
            fields_[field] = value
    for group, name in (picks or {}).items():
        fields_[self.family.spec_fields[group]] = name
    fields_.update(self._nominal_offsets(fields_, draws))
    return type(self.baseline).from_dict(fields_)
```

```python
# DesignGeneratorBase, the one place the pose is resolved.
def _nominal_offsets(self, fields_: dict, draws: dict[str, float]) -> dict[str, float | None]:
    """thigh, knee and ankle offsets for one design, section 4.5. Empty when the
    run draws no pose, which leaves all three spec fields at the baseline's None."""
    if not draws:
        return {}
    offsets = solve_nominal_pose(self.family.name, self.baseline,
                                 {"l1": fields_["thigh_length"], "l2": fields_["shank_length"]},
                                 draws, reference=NOMINAL_HIP_REFERENCE)
    return {k: round(v, 3) if v is not None else None for k, v in offsets.items()}
```

`check_nominal_pose` is called by `_build_xml` at `:449` on the finished spec before emission, so it reaches the fallback, and `type(self.baseline).from_dict` is used rather than importing the family's class.

### 4.9 Carrying the pose to the simulation

The pose enters a running scene by two routes, and the route decides what must be written.

| route | when | what carries the pose |
|---|---|---|
| respawn, `respawn_robots` | bootstrap, the late start at `ea_late_start`, a resume | the USD the converter produced from the document, which already holds every frame, so nothing extra is written |
| in place, `apply_link_length_params` | every `ea_update_interval` iterations of the EA phase | the shared prototypes, edited on their source layers. `update.py:152-165` writes box scale and translate, mass, diagonal inertia, centre of mass and the child joint's `physics:localPos0`, and nothing else, so the new frames are NOT carried today |

This is the third finding of the pass. Moving a thigh frame needs `physics:localRot0` rewritten on the joints that connect it, and the plan carries it as one optional argument, which under CLAUDE.md rule 4 defaults to the present behaviour.

```python
# co_optimisation/runners/usd_generator.py, _quadruped_links at :126 and
# _biped_links at :96. Every entry gains joint_rot, the localRot0 each named joint
# takes, as w x y z. Built from the builder's own pose_frames, so the frames in the
# document and the frames written in place cannot disagree.
def _quat_wxyz_y(rad: float) -> list[float]:
    return [math.cos(rad / 2.0), 0.0, math.sin(rad / 2.0), 0.0]

# quadruped
fr = quadruped.builder.pose_frames(spec)
out[f"thigh_{leg}_Link"]["joint_rot"] = {
    f"thigh_{leg}_Link": _quat_wxyz_y(fr["thigh"]),      # fixed joint knee_act -> thigh
    f"{leg}_calf_joint": _quat_wxyz_y(fr["knee"]),       # shank
}
# biped, frame scheme
fr = biped.builder.pose_frames(spec)                     # degrees
out[f"thigh_{side}_Link"]["joint_rot"] = {
    f"thigh_{side}_Link": _quat_wxyz_y(math.radians(fr["thigh"])),
    f"knee_{side}_Joint": _quat_wxyz_y(math.radians(fr["shank"])),
    f"ankle_{side}_act":  _quat_wxyz_y(math.radians(fr["ankle"])),
}
```

For the biped `_biped_links` also changes its box and child entries to the frame scheme under the same condition, `pos=(0, 0, -l1/2)` and `child_pos=(0, 0, -l1)`, which `l1 * THIGH_DIR_XZ` and `(l1/2) * THIGH_BOX_DIR` currently set. The quadruped's `child_pos=(0, 0, -l1)` at `:140` is already in the link frame and needs no change.

```python
# co_optimisation/co_optimisation/utils/update.py, update_articulation_links.
# After step (3). Optional, absent for a population that searches no pose.
joint_rot = params.get("joint_rot")
if joint_rot:
    for joint_name, quat in joint_rot.items():
        jp = stage.GetPrimAtPath(f"{joints_prim_path}/{joint_name}")
        r_layer, r_path = _source_spec(jp.GetAttribute("physics:localRot0"))
        if r_layer is not None and r_path is not None:
            r_layer.GetAttributeAtPath(r_path).default = Gf.Quatf(*quat)
```

`apply_link_length_params` needs no signature change, the rotations travelling inside the entries it already receives. `physics:localRot0` is the `UsdPhysics.Joint` attribute name and `Gf.Quatf` takes `(w, x, y, z)`. Neither the importer's authored layer for these attributes nor that the fixed joint named after the thigh body carries the rotation was read in this pass, and the validation run prints `physics:localRot0` of the pose joints of one leg at the donor and compares them with the expected values, section 9. The expected values, `(w, x, y, z)`, are for the quadruped thigh `(0.9038, 0, 0.4279, 0)` and knee `(0.6338, 0, -0.7735, 0)`, and for the biped thigh `(0.9537, 0, 0.3007, 0)`, shank `(0.7934, 0, -0.6088, 0)` and ankle `(0.9397, 0, 0.3421, 0)`, each to four places, before any longer run trusts the in place path.

### 4.10 Measuring the standing height, the spawn and the call order

The standing height is derived analytically in `usd_generator.py` from the same chain as the checks and measured in the simulation, and the two are asserted equal. The pair makes the whole chain testable, the generator's solve, the builder's frames, the importer, the in place editor and the foot geometry, with one comparison.

```python
# co_optimisation/co_optimisation/utils/update.py, new, beside apply_link_length_params.
def apply_standing_height(
    env: ManagerBasedRLEnv,
    expected: list[float],
    foot_bodies: tuple[str, ...],
    foot_sole_offset: float,
    pitch_joints: list[str],
    spawn_margin: float = 0.028,
    tolerance: float = 0.002,
) -> torch.Tensor | None:
    """Measure each design's standing height and put its spawn above it.

    Returns the per environment height, published by the runner as
    ``env.copt_nominal_base_height`` for the terms of section 4.11, or None when
    the population searches no pose, leaving the asset configuration alone.

    Runs with the simulation playing, after the sim.reset() that ends
    apply_link_length_params and before the env.reset() that places the robots.
    """
    import torch
    if not expected:
        return None
    robot = env.scene.articulations["robot"]
    assignment = torch.arange(env.num_envs, device=robot.device) % len(expected)

    # 1. the frames carry the pose, so the configuration's pitch nominals must be zero,
    #    or the fold is applied twice. pitch_joints is Family.pitch_joints.
    ids, _ = robot.find_joints(pitch_joints)
    assert robot.data.default_joint_pos[:, ids].abs().max() < 1e-6, "stale nominal pose in the asset configuration"

    # 2. hold every robot at its own nominal pose above the ground, read the drop
    zeros = torch.zeros_like(robot.data.default_joint_pos)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, zeros)
    probe = robot.data.default_root_state[:, :7].clone()
    probe[:, :3] += env.scene.env_origins
    probe[:, 2] = 2.0
    robot.write_root_pose_to_sim(probe)
    foot_ids, _ = robot.find_bodies(list(foot_bodies), preserve_order=True)
    lowest = robot.data.body_link_pos_w[:, foot_ids, 2].min(dim=1).values
    standing = (probe[:, 2] - lowest) + foot_sole_offset

    # 3. the measurement must agree with the generator's own arithmetic
    want = torch.tensor(expected, device=robot.device)[assignment]
    worst = (standing - want).abs().max().item()
    if worst > tolerance:
        raise RuntimeError(f"measured standing height differs from the solved one by {worst:.4f} m")

    # 4. spawn above the standing height, so a rough tile cannot capture a foot
    robot.data.default_root_state[:, 2] = standing + spawn_margin
    return standing
```

`pitch_joints` is `[".*_thigh_joint", ".*_calf_joint"]` on the quadruped and `["hip_.*_Joint", "knee_.*_Joint", "ankle_.*_Joint"]` on the biped, carried by the `Family` so the runner passes it through. `body_link_pose_w` performs the kinematic update on read (`articulation_data.py:592`), so the probe costs no physics step.

The expected height per design is `Family.standing_height(spec)`, section 4.11, which uses the closed forms of section 4.3. The sole offsets are the same constants as before, 0.022 for the sphere and 0.050 for the biped's sole, which hold because the sole is flat to 0.004 deg, or 6.5 micrometres over its half length.

```python
# co_optimisation/runners/usd_generator.py, Family gains the three attributes the
# runner needs, all defaulted, so existing constructions are unchanged.
class Family:
    def __init__(self, name, module, groups, spec_fields, root_body, links,
                 standing_height=None, foot_bodies=(), foot_sole_offset=0.0,
                 pitch_joints=()):
        ...
        self.standing_height = standing_height          # (spec) -> float, section 4.3
        self.foot_bodies = tuple(foot_bodies)
        self.foot_sole_offset = float(foot_sole_offset)
        self.pitch_joints = list(pitch_joints)
```

| family | `foot_bodies` | `foot_sole_offset` | `standing_height(spec)` |
|---|---|---|---|
| quadruped | `foot_FR_Link`, `foot_FL_Link`, `foot_RR_Link`, `foot_RL_Link` | 0.022, the sphere radius | `l1 cos(q1) + l2 cos(q1 + q2) + 0.022` |
| biped | `foot_R_Link`, `foot_L_Link` | 0.050, the sole box 0.035 plus 0.015 | `0.290497 + l1 cos(t) + l2 cos(s)` |

The 0.290497 is read from the donor, `variable_params.json` stating `body_height = 0.80435432` and the fit being `0.804357` against it, a 2 micrometre difference. A reader replacing it with a derivation of the chain's constant terms should assert against the same value.

```
sample ──► solve_nominal_pose ──► check_nominal_pose ──► DesignSpec.from_dict ──► build_xml
 (typed      (usd_generator)        (usd_generator)         (design_generator)     (quats, for_isaaclab)
 ranges)                                                                              │
                                                                                      ▼
                                                                      MJCF ──► USD, joints carry the frames
                                                                                      │
          respawn: the USD carries the pose ────────────────────────────────────────┤
          in place: update.py writes physics:localRot0 and localPos0 ◄──── Population.link_length_params[i][body]["joint_rot"]
                                                                                      │
                                                                                      ▼
            sim.reset() ──► apply_standing_height ──► env.reset() ──► reward v2 reads env.copt_nominal_base_height
                              measures, asserts == Family.standing_height(spec), writes default_root_state[:, 2]
```

```python
# co_optimisation/co_optimisation/runners/copt_on_policy_runner.py, _update_morphology,
# after apply_link_length_params at :705 and before the reset at :714. The
# rotations already travel inside the link entries.
_bodies, _offset = self.current_population.get_foot_geometry()
_standing = apply_standing_height(
    unwrapped_env, self.current_population.get_standing_heights(), _bodies, _offset)
if _standing is not None:
    unwrapped_env.copt_nominal_base_height = _standing
    unwrapped_env.copt_nominal_pose = self.current_population.get_nominal_pose_tensor(device)
```

| call site | where | extra cost |
|---|---|---|
| `_update_morphology`, in place, `:705` | after `apply_link_length_params`, before `env.reset()` at `:714` | none |
| `_respawn_population`, `:546` | after `respawn_robots` returns, then one further `env.reset()`, step 11 of the respawn having already reset | one reset, twice per run |
| `_apply_restored_state`, `:609` | same as the respawn path, so a resumed run stands where it stood | one reset per resume |

`Population` gains `standing_heights` and `nominal_pose` (the three offsets per design) as optional constructor arguments with `None` defaults and the two getters above, and `_build_population` at `:369` fills them from `Family.standing_height(spec)` and the spec's offsets. The nominal pose tensor, three columns per environment in the round robin assignment, is what the morphology observation of section 4.11 reads.

### 4.11 Everything that depends on the nominal pose

The requirement is a v2 for each reward and termination term that takes the updated pose into account. The audit's reading, which the table applies, is that a v2 is created for each term whose value or whose configured parameter depends on the pose, and that a term which provably does not depend on it is listed with the reason, an invariance test in its place and no new function, since a copy of `track_lin_vel_xy_exp` that changes nothing would be a second thing to keep in step with the first for no gain. Where the repository's rule 4 allows an optional argument to carry the change, the entry says so instead of creating a function.

Callers, enumerated per CLAUDE.md rule 3. `base_height_rough_l2` has four, `limx_base_env_cfg.py:937`, `kscale_base_env_cfg.py:886`, `quadruped/base_env_cfg.py:1086` and `brs_base_env_cfg.py:749`. `feet_regulation` has four, `limx_base_env_cfg.py:974`, `brs_base_env_cfg.py:786`, `PF/limx_base_env_cfg.py:398` and `kscale_base_env_cfg.py:1014`. `feet_air_time` is called by `limx_base_env_cfg.py:1008`, `limx_berkeley_env_cfg.py:306`, `kscale_base_env_cfg.py:1036` and `brs_base_env_cfg.py:798`, and `feet_air_time_v2` by `quadruped/base_env_cfg.py:1132`. In place edits of any of them would change those configurations silently, so each new function below is a v2 of its own series and the originals stand.

| term | robot | depends on the pose through | v2 | change |
|---|---|---|---|---|
| `pen_base_height` | both | the target height, 0.292 and 0.75, against a standing height of 0.20 to 0.41 and 0.63 to 0.98 | `base_height_rough_l2_v2` | per environment target from `env.copt_nominal_base_height`, scalar fallback |
| `pen_feet_regulation` | biped | `base_height_target` 0.75 sets the swing height scale, and `foot_radius` 0.03 against a sole 0.050 below the foot origin | `feet_regulation_v2` | height scale per environment, and `foot_radius` set to 0.050 by the configuration, a value and not a function change |
| `rew_foot_clearance` | quadruped | `target_height` 0.10 is 34 per cent of the donor's standing height, so it is a fixed fraction of a quantity the search now moves | `foot_clearance_reward_v5` | `target_height_frac` 0.3425 times `env.copt_nominal_base_height`, `target_height` retained as the fallback, the donor reproduced exactly |
| `feet_air_time` | quadruped | thresholds 0.15 and 0.45 s assume a stride period set by the donor's leg length, which scales as the square root of the height | `feet_air_time_v3`, over the body of `feet_air_time_v2` | thresholds multiplied per environment by `sqrt(h / 0.292)` |
| `feet_air_time` | biped | thresholds 0.2 and 0.5 s, same reasoning, reference 0.804 | `feet_air_time_v2`'s name is taken by the quadruped, so `feet_air_time_nominal`, over the body of `feet_air_time` | thresholds multiplied per environment by `sqrt(h / 0.804)` |
| `pen_foot_landing_vel` | quadruped | `foot_radius` 0.022 is the sphere, fixed, and `about_landing_threshold` 0.04 is a clearance above terrain, not above the nominal stance | none | invariant, recorded, with the invariance test |
| `pen_feet_distance` | biped | `min_feet_distance` 0.115 is a lateral separation, set by the trunk width and the abduction nominal, and the offsets are sagittal | none | invariant |
| `rew_keep_ankle_pitch_zero_in_air` | biped | targets joint angle 0, which is the flat sole because the fold lives in the frames | none | invariant, and would not be had the fold been put in joint nominals, which is a point for the frame scheme |
| `pen_joint_pos_limits` | both | reads the soft limits, which shift with the fold in the quadruped's USD ranges | none | invariant, the limits being re-read after a respawn and constant in place |
| `pen_abad_deviation` | quadruped | abduction nominal, held at 0 | none | invariant |
| `pen_flat_orientation`, `pen_lin_vel_z`, `pen_ang_vel_xy`, `rew_lin_vel_xy`, `rew_ang_vel_z`, `keep_balance`, `rew_no_fly` | both | none, they read the base velocity, the gravity projection or the contact state | none | invariant |
| `pen_joint_torque`, `pen_joint_accel`, `pen_joint_vel_l2`, `pen_action_rate`, `pen_action_smoothness` | both | none through the pose, torque magnitude moves with the leg length and the mass as it did before | none | invariant |
| `pen_feet_impact`, `pen_undesired_contacts`, `feet_slide`, `pen_feet_hold` | both | none through the pose, the impact and contact thresholds scale with mass, which is the length search and not the pose | none | invariant, recorded as design dependent |
| termination `time_out` | both | none | none | invariant |
| termination `base_contact` | both | contact force on `trunk` or `base_Link` above 1.0 N, which depends on clearance and the pose sets the clearance by 0.05 to 0.2 m over the box | none | invariant as a function. The guard is `apply_standing_height`'s corridor check, which refuses a design whose trunk would start inside the terrain |

Three consumers outside the reward manager read the nominal pose and are part of the same plan.

| consumer | file | reads | change |
|---|---|---|---|
| `reset_root_state_uniform` | `mdp/events.py` via `isaaclab` | `default_root_state[:, 2]` at every reset | written by `apply_standing_height`, no code change |
| `JointPositionAction._offset`, `joint_pos_rel` | `joint_actions.py:195`, `observations.py` | `default_joint_pos`, which is zero for every design | none, the hazard of the earlier revision closes |
| `robot_link_lengths` morphology observation | `observations.py:229` | link lengths alone | `robot_morphology_v2`, a v2, concatenating the three offsets and the standing height to the lengths so the estimator of D2 predicts the pose it is now asked to infer |

The morphology observation is why section 5's D2 changes with this pass. The estimator regresses `gtEncoderOut` through `obs_groups`, a design that moves the pose and a target that omits it asks the estimator to infer something it was never given, and the observation width grows by four for the biped's three offsets and one height and by three for the quadruped's two offsets and one height, `num_output_dim` following.

Reward v2 bodies, the function bodies written out in the previous revision stand for the first two rows and are reproduced here with the names the table uses.

```python
# environments/.../mdp/rewards.py, beside base_height_rough_l2 at :1889.
def base_height_rough_l2_v2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height_attr: str = "copt_nominal_base_height",
) -> torch.Tensor:
    """As :func:`base_height_rough_l2`, with a per-environment target.

    The co-optimisation runner writes the attribute after every design swap, so the
    target tracks the design's own standing height. Falls back to the scalar
    *target_height* while the attribute is absent, on the first iteration and in
    every play configuration.

    Defect left standing in the original, which four configurations call, is that a
    design taller or shorter than the target pays a penalty that is a deterministic
    function of its geometry, 21.84 reward units per episode at either edge of the
    quadruped's length box against a measured return of 89.81.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    target = getattr(env, target_height_attr, None)
    if target is None:
        height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
        return torch.square(height.mean(dim=1) - target_height)
    fill = target.unsqueeze(1).expand_as(height)       # nan_to_num takes a float, so a where
    height = torch.where(torch.isfinite(height), height, fill)
    return torch.square(height.mean(dim=1) - target)
```

```python
# beside feet_regulation at :1850. The body of feet_regulation with its height scale per design.
def feet_regulation_v2(env, asset_cfg, foot_radius, base_height_target,
                       height_decay_scale=None, target_height_attr="copt_nominal_base_height"):
    target = getattr(env, target_height_attr, None)
    scale = base_height_target if target is None else target
    ...  # feet_regulation's body with base_height_target replaced by scale
```

```python
# beside foot_clearance_reward_v4 at :746. v5, the target a fraction of the standing height.
def foot_clearance_reward_v5(env, asset_cfg, height_sensor_cfg, sensor_cfg, target_height,
                             std, target_height_frac=None, target_height_attr="copt_nominal_base_height",
                             **kwargs):
    base = getattr(env, target_height_attr, None)
    tgt = target_height if (base is None or target_height_frac is None) else target_height_frac * base
    ...  # foot_clearance_reward_v4's body with target_height replaced by tgt (broadcast per env)
```

```python
# beside feet_air_time_v2 at :368. Thresholds follow the square root of the standing height.
def feet_air_time_v3(env, command_name, sensor_cfg, threshold_min, threshold_max,
                     reference_height=0.292, target_height_attr="copt_nominal_base_height", **kwargs):
    base = getattr(env, target_height_attr, None)
    k = 1.0 if base is None else (base / reference_height).sqrt().unsqueeze(1)
    ...  # feet_air_time_v2's body with threshold_min * k and threshold_max * k
```

```python
# environments/.../cfg/quadruped/base_env_cfg.py, the copt RewardsCfg only. The plain
# and HIM tasks keep the original terms, so no in flight comparison moves.
    pen_base_height = RewTerm(func=mdp.base_height_rough_l2_v2, weight=-5.0,
                              params={"target_height": 0.292, "sensor_cfg": SceneEntityCfg("height_scanner")})
    rew_foot_clearance = RewTerm(func=mdp.foot_clearance_reward_v5, weight=0.5, params={..., "target_height": 0.10, "target_height_frac": 0.3425})
    feet_air_time = RewTerm(func=mdp.feet_air_time_v3, weight=10.0, params={..., "reference_height": 0.292})
# cfg/SF/limx_base_env_cfg.py, the copt biped task only.
    pen_base_height = RewTerm(func=mdp.base_height_rough_l2_v2, weight=-5.0, params={"target_height": 0.75, ...})
    pen_feet_regulation = RewTerm(func=mdp.feet_regulation_v2, weight=-0.2,
                                  params={..., "base_height_target": 0.804, "foot_radius": 0.050})
    feet_air_time = RewTerm(func=mdp.feet_air_time_nominal, weight=2.0, params={..., "reference_height": 0.804})
```

The biped's fallback target moves from 0.75 to 0.804 in the copt task, the nominal, so the iterations before the first swap are no longer penalised for the 54 mm the old constant misses by. That is a change of behaviour, confined to the copt task's own configuration and recorded in its dumped `params/env.yaml`.

### 4.12 Hazards, in the order they will be met

| hazard | symptom | check |
|---|---|---|
| The constants of section 4.3 left at 70 and -154 | every quadruped stands at 0.117 m and the feet are 0.2 m from the ground at spawn | measured against expected in `apply_standing_height`, which raises |
| `quadruped_identified_cfg.py:100-103` left at the old nominals | the fold is applied twice, the legs fold through the trunk | the zero assertion of step 1 |
| Quadruped quat emitted without the range shift | the nominal sits outside `soft_joint_pos_limits`, `pen_joint_pos_limits` dominates from step one | assert in `check_nominal_pose`, and the shifted ranges of the table in section 4.6 |
| Sign of the biped offsets read as joint angles | the hip leans the wrong way and the standing height measured disagrees with the solved by more than 2 mm | the measured against expected assertion, whose tolerance is the pass criterion |
| Frame scheme and legacy scheme disagree | the zero offset biped is not the donor | the world pose test of section 4.6 |
| `physics:localRot0` not the attribute that carries the frame, or authored on a layer `_source_spec` does not resolve | the in place path leaves the previous generation's pose while the log reports new populations | the first in place update prints the joint rotation read back, and `assert_link_lengths_applied` at `update.py:241-284` gains the same comparison |
| Hip draw plateaus under `"default"` | the CMA-ES hip coordinate has no gradient over two thirds of its box | log the fraction repaired per generation, switch to `"solved"` above 0.5 |
| Pose resolved before the lengths are rounded | the foot residual shows the rounding, 0.005 m of length at 0.3 m is 1.7 per cent of the stance | `_spec_from_parameters` resolves after rounding, as written |
| Opened before the reward v2 lands | the height penalty prices the pose as well as the lengths | land 4.8, 4.10 and 4.11 in one pass |
| `copt_nominal_base_height` absent on a play configuration | falls back to the scalar, as intended | the `getattr` default |

### 4.13 Order of work inside this section

| step | edit | depends on | verification |
|---|---|---|---|
| 1 | constants of section 4.3 and `GEOMETRY_RULES.md:250-251` | none | MuJoCo keyframe stands at the donor pose |
| 2 | `DesignSpec` fields, `from_dict`, `to_dict`, `load_design_spec` in both files | none | donor loads, `from_dict` rejects an unknown key |
| 3 | `pose_frames` and the quadruped builder quats and shifted ranges | 1, 2 | the quadruped world pose test, zero offsets give `q1 = 0.884337`, `q2 = -1.768673` |
| 4 | biped frame scheme | 2 | the two scheme equivalence test to 1e-6 m |
| 5 | `quadruped_identified_cfg.py` nominals to zero | 3 | step 1 assertion of `apply_standing_height` |
| 6 | typed ranges, deprecations, renames, `_spec_from_parameters`, `solve_nominal_pose`, `check_nominal_pose` | 2 | `grep -n "_current_scales\|_sample_scales\|replace(" usd_generator.py` empty, solve and check unit tests against the tables of 4.5 |
| 7 | `joint_rot` entries, `update.py` rotation write, `Population` fields | 3, 4, 6 | validation run prints the joint rotations read back |
| 8 | `apply_standing_height` and the runner call sites | 5, 7 | measured against solved, within 2 mm, printed for the whole population |
| 9 | the reward v2 functions, `robot_morphology_v2`, copt configurations | 8 | invariance tests of section 4.11 at the donor, zero offsets reproduce the old reward to float precision |

## 5. Divergence surfaces

### D2 Policy architecture and the estimator loss

The two robots are running different experiments. The biped infers morphology from proprioceptive history, which is the deployable arrangement, and the quadruped is handed the true morphology and asked to infer dynamics from it, which is not.

| | quadruped today | biped today, `SFCoptMorphologyRunnerCfg` | quadruped after |
|---|---|---|---|
| `policy` | `["policy", "morphologyObs"]` | `["policy"]` | `["policy"]` |
| `critic` | `["critic"]` | `["critic"]` | `["critic"]` |
| `encoderIn` | `["morphologyObs", "historyObs"]` | `["historyObs"]` | `["historyObs"]` |
| `gtEncoderOut` | `["privilegedDynamicsObs"]` | `["morphologyObs"]` | `["morphologyObs"]` |
| Actor input width | 48 + 258 + 48 = 354 | 36 + 154 = 190 | 48 + 258 = 306 |
| Estimator input width | 258 + 5875 = 6133 | 5575 | 5875 |
| Estimator output width | 48 | 154 | 258 |
| Estimator parameters | 7.0 M | 6.4 M | 6.74 M |
| Deployable without privileged input | no | yes | yes |

```python
# environments/.../agents/quadruped_rsl_rl_ppo_cfg.py:58  BEFORE
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "morphologyObs"],
        "critic": ["critic"],
        "encoderIn": ["morphologyObs", "historyObs"],
        "gtEncoderOut": ["privilegedDynamicsObs"],
    }
```

```python
# environments/.../agents/quadruped_rsl_rl_ppo_cfg.py:58  AFTER
    # Identical to the biped's SFCoptMorphologyRunnerCfg
    # (limx_rsl_rl_ppo_cfg.py:161), so a difference between the two robots'
    # curves is attributable to the robot and the reward set rather than to the
    # architecture. The actor no longer reads the true morphology, which is not
    # available on hardware, and the estimator must infer it from proprioceptive
    # history, which is the MoRAL arrangement the biped tests.
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
        "encoderIn": ["historyObs"],
        "gtEncoderOut": ["morphologyObs"],
    }
```

`privilegedDynamicsObs` stays defined. Its four terms are also critic terms, and the group costs 48 columns that no observation set reads, which is 29 MB of rollout storage at 6144 environments and is what the biped does with the same group. Keeping it leaves the quadruped analogue of the biped's third experiment one line away.

Second part, the loss. The estimator's mean squared error enters the total loss at coefficient one and shares one gradient clip with the policy.

```
loss = surrogate + c_v * value_loss - c_e * entropy + 1.0 * mse(estimator)   copt_ppo.py:258
                                                      ▲
                                                      └── no coefficient
clip_grad_norm_(self.policy.parameters(), 1.0)                               copt_ppo.py:309
                 ▲
                 └── the estimator is a submodule, so its gradient norm scales
                     the actor and critic gradients down by the same factor
```

```python
# co_optimisation/algorithms/copt_ppo.py, __init__ and update.
# Defaults to 1.0, reproducing the present behaviour exactly.
self.estimator_loss_coef = estimator_loss_coef
...
loss = (surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
        + self.estimator_loss_coef * model_estimation_loss)
```

Third part, and the one the coefficient cannot reach. After the change both robots regress `morphologyObs`, whose channels are neither commensurate nor mostly variable.

| | quadruped | biped |
|---|---|---|
| `morphologyObs` width | 258 | 154 |
| `link_lengths`, `robot_link_lengths` over 8 and 4 pairs | 8, all vary | 4, all vary |
| `robot_mass`, `default_mass` over all bodies | 25, of which 8 vary | 15, of which 4 vary |
| `robot_inertia`, `default_inertia` flattened | 225, of which 24 vary and 150 are identically zero | 135, of which 12 vary and 90 are identically zero |
| Channels the design moves | 40 of 258, 15.5 per cent | 20 of 154, 13.0 per cent |
| Added by section 4.11, `robot_morphology_v2` | 2 offsets and 1 standing height, 3 channels, all vary | 3 offsets and 1 standing height, 4 channels, all vary |
| Magnitude range | lengths about 0.3, masses 0.02 to 1.0, inertias about 1e-5 to 4e-3 | lengths 0.3 to 0.35, masses 0.6 to 5.0, inertias about 1e-4 |

Five orders of magnitude between the mass channels and the inertia channels, in an unweighted sum of squares, means the inertia columns contribute nothing to the gradient and the estimator's reported loss is a statement about the masses alone. Eighty five per cent of its output capacity is spent reproducing constants.

```python
# environments/.../mdp/observations.py:51 and :60, restricting the target to the
# bodies the design moves. body_ids is slice(None) unless the caller names
# bodies, and no present caller does, so every existing caller is unchanged.
def robot_mass(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.default_mass[:, asset_cfg.body_ids].to(asset.device)


def robot_inertia(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    inertia = asset.data.default_inertia[:, asset_cfg.body_ids].to(asset.device)
    return inertia.view(inertia.shape[0], -1)
```

```python
# environments/.../cfg/quadruped/base_env_cfg.py:447, MorphologyCfg of the copt
# task. 258 channels become 8 + 8 + 72 = 88, of which 40 vary, and every
# remaining channel belongs to a link the search actually moves.
        robot_mass = ObsTerm(
            func=mdp.robot_mass,
            params={"asset_cfg": SceneEntityCfg(
                "robot", body_names=["thigh_.*_Link", "shank_.*_Link"])},
            clip=(0.0, 100.0),
        )
        robot_inertia = ObsTerm(
            func=mdp.robot_inertia,
            params={"asset_cfg": SceneEntityCfg(
                "robot", body_names=["thigh_.*_Link", "shank_.*_Link"])},
        )
        # The nominal pose is a design variable from section 4, so the estimator
        # must predict it alongside the geometry.
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
```

Both observation functions also stop hard coding `torch.device("cuda")`, which closes H5. A per-channel standardisation inside `CoptActorCritic.get_estimator_target` remains the complete answer to the magnitude spread, and a `scale` on each term of `MorphologyCfg`, visible in the dumped `params/env.yaml`, is the cheap one. The diagnostic already exists, `model_estimation_loss` being returned at `copt_ppo.py:339`, and four or more orders above `surrogate` in the first 200 iterations confirms the defect.

### D3 Estimator output normalisation, claim withdrawn

The first pass held that `encoderOut` reaches the actor unnormalised. Verified and withdrawn. `_actor_input` concatenates `encoderOut` with the rest of the actor observation and passes the whole through `self.actor_obs_normalizer` (`copt_actor_critic.py:145`), and `update_normalization` folds `encoderOut` into the running statistics before delegating upward (`copt_actor_critic.py:160-170`). The code is correct.

What remains is a configuration fact shared with every task in the repository. `empirical_normalization = False` on the runner configuration is propagated to `actor_obs_normalization` and `critic_obs_normalization` by `on_policy_runner.py:404-413`, and `ActorCritic.__init__` then installs `torch.nn.Identity` for both (`actor_critic.py:62-77`). So no observation is normalised anywhere, in the co-optimisation tasks or the plain ones, which is a deliberate inherited choice rather than a co-optimisation defect. After D2 the actor input is the policy group and the predicted morphology, exactly the biped's arrangement, so the quadruped inherits whatever that choice is worth on the biped and the two remain comparable.

One item stands. `EncoderCfg.output_detach = True` at `quadruped_rsl_rl_ppo_cfg.py:65` and `limx_rsl_rl_ppo_cfg.py:148` is read nowhere, and `copt_actor_critic.py:143` does not detach, so the actor's policy gradient flows into the estimator against the recorded intent. Either honour the field by detaching at that line or delete it. A configuration field that describes nothing is worse than no field.

### D4 Per individual advantage normalisation, retained unused

`compute_returns_design_wise` at `copt_ppo.py:96-111` has no caller, `copt_on_policy_runner.py:408` calling `compute_returns`. Left as it stands by decision, for a later experiment. Correct the register entry of `plans/COPT_INVESTIGATION_PLAN.md`, which claims three landed variance remedies where two landed.

### D5 Learning rate reset, deliberate and now gated

`reinit_learning_rate()` fires at every generation boundary before the late start (`copt_on_policy_runner.py:692`), so the adaptive schedule returns to 1e-3 every 480 iterations for the 25 generations of the random phase. That is deliberate, the random phase being a fresh design distribution each generation. It is invisible, which is the defect, so it becomes a recorded configuration choice that the dumped `params/agent.yaml` carries.

```python
# scripts/rsl_rl/train.py:266, the copt dict
        agent_cfg_dict["copt"] = {
            "ea_update_interval": ea_update_interval,
            "ea_late_start": ea_late_start,
            "num_individuals": _num_individuals,
            "randomise_before_late_start": True,
            "min_fitness_episode_length": 25,
            # Each random-phase generation draws a fresh design distribution, so
            # the adaptive schedule is returned to its initial rate rather than
            # carried across the discontinuity. True reproduces the behaviour of
            # every run to date.
            "reinit_lr_each_generation": True,
        }
```

```python
# co_optimisation/runners/copt_on_policy_runner.py, __init__
        self._reinit_lr_each_generation: bool = copt_cfg.get(
            "reinit_lr_each_generation", True
        )
```

```python
# co_optimisation/runners/copt_on_policy_runner.py:691, the random-phase branch
        else:
            if self._reinit_lr_each_generation:
                self.alg.reinit_learning_rate()
```

The single reset at the late-start transition, `copt_on_policy_runner.py:684`, is not gated. It marks the one discontinuity in the run that is not a design swap.

```python
# co_optimisation/algorithms/copt_ppo.py:349
def reinit_learning_rate(self):
    self.learning_rate = self.init_learning_rate
    # Written through for the same reason CoptOnPolicyRunner.load does at
    # copt_on_policy_runner.py:210. Invisible under the adaptive schedule, which
    # rewrites the groups each minibatch, and a no-op under a fixed one.
    for group in self.optimizer.param_groups:
        group["lr"] = self.learning_rate
```

### D6 Curriculum gates read truncated episode sums at every swap

```
_reset_idx (manager_based_rl_env.py)
  356  curriculum_manager.compute(env_ids)   ◄── reads _episode_sums
  375  reward_manager.reset(env_ids)         ◄── zeroes them

a morphology swap truncates every episode at once, so every sum read at 356 is partial.
The suppression flag exists and is read in exactly one place, curriculums.py:476.
```

`common_step_counter` is a multiple of `480 * 25 = 12000` at every swap.

| curriculum term | interval, steps | coincidence with the 12000 step swap |
|---|---|---|
| `modify_command_velocity_ang_z` | 250 x 24 = 6000 | every 2nd evaluation |
| `modify_command_velocity_lin_x` | 300 x 24 = 7200 | every 5th |
| `modify_command_velocity_lin_y` | 300 x 24 = 7200 | every 5th |
| `modify_push_force` | 400 x 24 = 9600 | every 5th |
| `modify_linear_tracking_reward_std` | 2000 x 24 = 48000 | every evaluation |
| `modify_angular_tracking_reward_std` | 2000 x 24 = 48000 | every evaluation |
| `terrain_levels` | per reset | suppressed already |

The two tracking standard deviations sharpen the velocity reward from 0.4 toward the 0.09 floor, which is how the task becomes hard. Under co-optimisation every one of their evaluations coincides with a swap, so they are frozen at their initial values for the whole run.

```python
# co_optimisation/runners/copt_on_policy_runner.py:714, and the pair at :570.
- unwrapped_env._suppress_terrain_curriculum = True
+ unwrapped_env._suppress_curriculum = True          # keep the old name as an alias
  unwrapped_env.reset()
- unwrapped_env._suppress_terrain_curriculum = False
+ unwrapped_env._suppress_curriculum = False
```

```python
# environments/.../mdp/curriculums.py, opening each of the six terms, in the
# shape the terrain term already has at line 476.
if getattr(env, "_suppress_curriculum", False):
    return command_cfg.ranges.ang_vel_z[1]   # present value, unchanged
```

Second defect in the same block. Every interval is written as a multiple of 24, the biped's `num_steps_per_env`, while the quadruped runs 25 (`quadruped_rsl_rl_ppo_cfg.py:16`). The intended 300 iteration interval is 288 and the intended 6000 iteration start is 5760, a 4 per cent error inherited from the robot the configuration was copied from. Six edits from `* 24` to `* 25`.

Addition of 2026-10-08. The quadruped's intervals are no longer multiples of 25 iterations. `CoptCurriculumCfg` now carries the step counts of run `2026-09-11_11-24-52`, 2400 on the push term and the three command velocity terms, 9600 on the two tracking standard deviations and 4800 for the terrain start (`environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:1411`), as `quadruped.md` section 8 records, so the coincidence table above describes the configuration before that date. Since 2400 divides the swap step of 12000, each of the four 2400 step terms is now evaluated at every swap, and the two 9600 step terms at every fourth swap. The gate this section prescribes is what keeps that harmless. The runner raises `_suppress_curriculum` across both swap resets (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:586-592` and `:612-618`) and each of the five functions behind the six terms reads it (`environments/environments/tasks/locomotion/mdp/curriculums.py:175`, `:302`, `:340`, `:390`, `:441`), so the re-entry a swap's reset makes is suppressed and the double step the correction of 2026-10-06 describes cannot occur.

### D7 Spawn height constant while leg length is searched

| design | standing height | spawn z today | sole relative to the ground |
|---|---|---|---|
| quadruped, s = 0.75 | 0.2245 | 0.32 | +0.0955 m, dropped |
| quadruped, s = 1.00 | 0.2920 | 0.32 | +0.0280 m, as documented |
| quadruped, s = 1.00, joint reset offset 0.2 rad | 0.3520 | 0.32 | -0.0320 m, already buried |
| quadruped, s = 1.25 | 0.3594 | 0.32 | -0.0394 m |
| quadruped, s = 1.25 and fold 1.55 | 0.4022 | 0.32 | -0.0822 m |
| biped, s = 0.75 | 0.6759 | 0.80 | +0.1241 m |
| biped, s = 1.00 | 0.8043 | 0.80 | -0.0043 m, already buried |
| biped, s = 1.25 | 0.9328 | 0.80 | -0.1328 m |

`reset_joints_by_offset` adds up to 0.2 rad to every joint (`base_env_cfg.py:1041`), so the 28 mm of clearance the quadruped asset comment claims is already consumed at the nominal design, and the biped is 4.3 mm inside the ground before any noise. Closed by step 4 of `apply_standing_height` (section 4.10, which replaced `apply_nominal_pose_params`), which writes `default_root_state[:, 2]` from the measured standing height, `reset_root_state_uniform` reading it at every reset and nothing else changing.

### D8 Design lattice smaller than the population

`_spec_from_scales` rounds each scaled length to two decimals (`usd_generator.py:416`).

| robot | baseline lengths | rounded values each | lattice | population | distinct designs possible |
|---|---|---|---|---|---|
| quadruped | 0.213, 0.213 | 12, 12 | 144 | 256 | 56 per cent of the population at best |
| biped | 0.30, 0.35 | 17, 19 | 323 | 256 | duplication likely, not certain |

Retained for reference by decision, the design space being due to widen. Note for when it does, the fold of section 4 is not rounded, so a quadruped population searching the fold as well draws from a continuum and the lattice argument no longer binds. Extends [copt_ppo_nonstationarity.md](../../context/copt_ppo_nonstationarity.md) section 3.1 to the quadruped with exact numbers.

## 6. Instancing and propagation, N1

The in-place editor writes each design once, through the first `num_individuals` environments, and relies on every other environment carrying the same design seeing the edit. The question is whether it does, and it has four parts.

```
env_i /Robot   ── Sdf.CopySpec of a template prim spec (wrappers.py:110)
                  carrying a REFERENCE to designs/individual_{i mod N}.usd
                  plus the local opinions spawn_from_usd authored

what the editor writes                where the strongest opinion lives      shared by
────────────────────────────────────  ─────────────────────────────────────  ─────────────────
xformOp:scale, xformOp:translate      the design's instancing PROTOTYPE      every env with
  on the box Xform                      /__Prototype_<N>                     that design
physics:mass, diagonalInertia,        the design's own USD layer, authored   every env with
  centerOfMass on the link prim         by the MJCF importer                 that design
physics:localPos0 on the child joint  the design's own USD layer             every env with
                                                                             that design
```

Geometry is certain. Instanced prims of one prototype cannot be overridden per instance, which is the reason the editor goes through the prototype at all, so every environment referencing a design sees its geometry edit.

The three physics attributes rest on an argument. `spawn_multi_usd_file` builds one template prim per design and copies its spec into every environment path (`wrappers.py:107-112`), so each environment's opinions are whatever `spawn_from_usd` authored, and the respawn passes only `rigid_props`, `articulation_props` and `activate_contact_sensors` (`respawn.py:107-113`), `mass_props` being left unset. None of those three authors `physics:mass`, `physics:diagonalInertia`, `physics:centerOfMass` or `physics:localPos0`, so `_source_spec`'s property stack resolves each to the design's own layer, which every environment referencing that design shares. The argument holds exactly while `mass_props` stays unset, and setting it would move the strongest opinion into each environment's own spec and confine the edit to the environment it was written through.

The verification is blind to the failure it exists to catch. `assert_link_lengths_applied` reads `positions[individual, ...]` for `individual` in `range(num_individuals)`, that is the very environments the editor wrote through, so it would pass while every environment beyond the first `num_individuals` kept the previous generation's geometry. Extending it to one further environment per design settles the question permanently at the cost of one more slice.

```python
# co_optimisation/utils/update.py:265, assert_link_lengths_applied
# BEFORE
    for individual in range(num_individuals):
        for link_name, params in link_length_params_list[individual].items():
            ...
            got = float(
                (positions[individual, names.index(child)]
                 - positions[individual, names.index(link_name)]).norm()
            )
```

```python
# co_optimisation/utils/update.py:265, assert_link_lengths_applied
# AFTER. The editor writes design d through environment d only (update.py:208)
# and relies on every environment sharing that design seeing the edit through
# the instancing prototype and the reference. Checking environment
# d + num_individuals is the only thing that distinguishes an edit that
# propagated from one that reached the environment it was written through and
# stopped there. See co_optimisation_audit.md section 6.
    num_envs = positions.shape[0]
    for individual in range(num_individuals):
        rows = [individual]
        mirror = individual + num_individuals
        if mirror < num_envs:
            rows.append(mirror)
        for env_row in rows:
            for link_name, params in link_length_params_list[individual].items():
                ...
                got = float(
                    (positions[env_row, names.index(child)]
                     - positions[env_row, names.index(link_name)]).norm()
                )
                if abs(got - want) > tol:
                    bad.append(
                        f"env {env_row} (design {individual}) {link_name} -> "
                        f"{child}: realised {got:.6f} m, requested {want:.6f} m"
                    )
```

Same treatment for `apply_standing_height` of section 4.10, whose measurement is per environment already, so it carries no equivalent blind spot.

One ordering fact is still argued rather than measured. `spawn_multi_usd_file` assigns `proto_prim_paths[index % len(proto_prim_paths)]` where `index` runs over the paths `find_matching_prim_paths` returns (`wrappers.py:114`), while the runner attributes fitness by `env_id % num_individuals` (`copt_on_policy_runner.py:744`). The two agree only if that traversal returns environments in numeric order, which follows from the cloner creating them in list order and `GetAllChildren` returning authored order, and which a lexicographic traversal would break for any `num_envs` above ten. One print of the first twenty matched paths at the first respawn settles it permanently.

## 7. Deferred, with reasons

Latency and memory, L1 to L4, stand as recorded. The design space is due to widen substantially, which weakens L1's remedy rather than strengthening it, the conversion cache paying only while the lattice is small, and which raises L2's width only through the morphology group that D2.3 shrinks from 258 channels to 88.

| ID | Mechanism | Why dormant | Trigger | Fix |
|---|---|---|---|---|
| X1 | `respawn_robots` rebinds action (`:176`), event (`:196`) and command (`:215`) managers, not reward or termination. `HipLateralSwayReward` (`rewards.py:873`), `GaitReward` (`:1968`), `JointTorqueRatePenalty` (`:2170`) cache the articulation at construction | Quadruped and TRON1 use only `ActionSmoothnessPenalty`, which caches nothing | Adopting a KScale term, or giving the KScale a copt task | Add a rebind step in the event manager's shape |
| X2 | `apply_actuator_params` has no else branch (`respawn.py:282-290`), so an unknown key vanishes | Every `SIM_PARAM_KEYS` entry happens to be a tensor attribute of `IdentifiedActuator` | A new catalogue field, or a scalar-valued attribute | Raise, or print the key |
| X3 | Only model side `effort_limit` and `velocity_limit` written, `*_sim` pushed once at `articulation.py:1772` and never again | The length search calls `_actuator_params(spec)` without scales, and `A1-*` equals the asset configuration | Live once B6 lands and a categorical or random search runs | Write `effort_limit_sim` and `velocity_limit_sim` too |
| X4 | `velocity_limit_sim` falls back to the USD value (`actuator_base.py:183`), and neither document declares a joint velocity anywhere. Both declare `armature` 0.05 in the default classes against 0.01 in the asset configuration, and `damping` 1.0 against 1.2, 1.1 and 0.9 | Unknown, not measured | Always live | Read the resolution table `_record_actuator_resolution` prints at startup, record it in [quadruped.md](quadruped.md) |
| X6 | `check_geometry` validates only `trunk_width` (`quadruped/builder.py:311`), never varied | The fallback at `usd_generator.py:449` can therefore never fire | Always live | Add the length and pose aware `check_nominal_pose` of section 4.5 |
| X7 | `SFCoptMorphologyFromDynamics`, `SFCoptDynamicsFromMorphology`, `SFCoptMorphologyAndVelocity` defined at `limx_rsl_rl_ppo_cfg.py:171-207`, only the first instantiated at `robots/__init__.py:46` | Experiments 2 to 4 cannot be launched | Wanting them | Register three more entry points |
| X8 | Design assignment is `env_idx % len(population)` and fitness accumulation is `env_idx % copt.num_individuals` (`copt_on_policy_runner.py:744`), and the two differ in the random phase where the population is `num_envs / 4` | Fitness is never read before the late start, `update_with_fitness` being reached only once `_copt_started` (`:683`) | Reading fitness during the random phase, for logging or for a warm start | Derive the modulus from the live population |

| ID | Observation | Remedy |
|---|---|---|
| H1 | `.gitmodules` names the submodule `co_optimisation/co_optimisation/runners/design_generator` with `path = design_generator`. `usd_generator.py:4`, `update.py`'s docstring and [xml_design_generator.md](../../context/xml_design_generator.md) all still place it under `co_optimisation/runners/` | Rename the submodule to its path, correct the three descriptions |
| H2 | Quadruped carries a second copy of its observation configuration as `CoptObservationsCfg` (`base_env_cfg.py:360-651`), the biped carrying copt groups as attributes of one class (`limx_base_env_cfg.py:429-432`). The two quadruped copies have already drifted, `link_lengths` existing in one critic and not the other | Bring the quadruped onto the biped's pattern, which is what CLAUDE.md rule 4 prescribes |
| H3 | `robot_pos` returns `root_pos_w` (`observations.py:103`), unclipped into the critic. On a 6 by 64 grid of 15 m tiles the y coordinate spans about plus or minus 480 m, so the value function is a function of the tile index | Subtract `env.scene.env_origins`. Every surviving run carries this, so not fatal, but free to fix |
| H4 | `EncoderCfg.num_output_dim=19` and `orthogonal_init` read nowhere, the width being derived from `gtEncoderOut` at `copt_actor_critic.py:79`. `MjcfConverterCfg.self_collision` is False in the generator and True in both asset configurations, though the articulation level `enabled_self_collisions=True` governs and survives respawn. `flatten_history_dim` carries the comment "Required by HIMActorCritic" in tasks that build no such class | Delete or honour each, make `self_collision` agree, correct the comment to name `copt_actor_critic.py:118` |
| H5 | `robot_mass`, `robot_inertia`, `robot_joint_pos`, `robot_joint_stiffness`, `robot_joint_damping`, `robot_pos`, `robot_vel` and `robot_material_properties` each build `torch.device("cuda")` rather than reading the asset's device (`observations.py:51-122`) | Closed for the two the design search moves by D2.3, the remaining six take the same treatment when next touched |

## 8. Order of work

```
 1.  B4                  python -c "import cma, cmaes, pandas, design_generator"
 2.  B3 + B5             assertions, choices and num_envs 6144, so a cheap
                         validation is possible at all
 3.  B1                  the quadruped reaches its own package
 4.  B6 + B7             shelf and family, in one pass, B6 alone would widen the
                         shelf onto eleven unsimulatable entries
 5.  B2                  defensive, the Flat ids being registered
 6.  N1                  the propagation check, before any long run trusts a
                         fitness number
 7.  validation run      both robots, reduced --num_envs that still divides the
                         population, reduced ea_update_interval. Must exercise
                         bootstrap, one in-place update, the late-start respawn,
                         one save and one resume, and must print the measured
                         standing-height interval, the first twenty matched prim
                         paths and the actuator resolution table
 8.  D1 section 4        the nine steps of section 4.13, constants and DesignSpec
                         first, builders and typed ranges next, the in place
                         rotation write, the measured standing height and the
                         reward v2 together. D7 closes with it. Nothing
                         downstream can be interpreted until this lands. The
                         validation run of step 7 must print the joint rotations
                         read back and the measured against solved heights
 9.  D2                  obs_groups first, then the loss coefficient, then the
                         target composition. Diagnose from the first 200
                         iterations of the loss log
10.  D6                  invisible in the curves, leaves the curriculum where it
                         started
11.  D5                  one flag, one write-through
```

Steps 1 to 7 make a run possible and are independent of one another apart from the ordering shown. Steps 8 to 11 change what a run means, so each wants its own before and after.

## 9. What was verified and what was not

| Claim class | Basis |
|---|---|
| Every `file:line` citation | Read in the tree on 2026-09-29 and 2026-09-30 |
| Reward budget of section 4.2 | `artefacts/quadruped_flat-2026-09-16_07-01-06-reward-budget/`, hand rolled TFRecord and protobuf reader, no TensorFlow or Isaac Sim. Weights from that run's `params/env.yaml`, md5 `36978aabc33e77cb62669c2383938ced` |
| Pose arithmetic of section 4.3 to 4.5, the offset grids, the repaired height intervals, the foot residuals at the scale corners, the repair fractions and the donor quaternions | Computed 2026-10-01 in plain Python from the chains of both builders with the donor constants, `_cluster_layout` run on the donor biped for the hip and thigh offsets, and 30,000 uniform draws of the lengths and offsets per robot. The 0.290497 biped constant and the heights are cross checked against `variable_params.json` `body_height = 0.80435432`, the fit giving 0.804357 |
| Standing heights, fold table, widths, lattice and drop arithmetic | Computed from the MJCF chains and the configuration constants. The biped's 0.804343 is confirmed independently by `design_generator/biped/variable_params.json` `body_height = 0.80435432` |
| Catalogue completeness counts of B7 | Enumerated from both `actuators.json` files against `SIM_PARAM_KEYS` and `IdentifiedActuator.sim_params` |
| D3's withdrawal | Read in `copt_actor_critic.py:142-170`, `rsl_rl/modules/actor_critic.py:62-77` and `rsl_rl/runners/on_policy_runner.py:404-413` |
| Action offset staleness | Read at `joint_actions.py:194-195` and `respawn.py:176-183`. That the in-place path has no equivalent is read, that it matters is a consequence of section 4 and has no present symptom |
| N1's physics propagation | Argued from `wrappers.py:107-118`, `respawn.py:107-113` and the property stack that `_source_spec` resolves. NOT measured, which is why the check is part of the remedy |
| Section 4.9, that the importer authors `physics:localRot0` on the joints named there, that `_source_spec` resolves it to a writable layer, and that the in place rotation write reaches every instance | NOT read in the importer and NOT measured, the Isaac Sim importer source was not re-read in this pass. The validation run prints the six pose joints of one leg and compares them against the quaternions of section 4.9 |
| Section 4.6, that the frame scheme and the legacy scheme of the biped describe one geometry | Argued from the shared sum of rotations and the shared chain, NOT tested. The world pose test of step 4 of section 4.13 settles it |
| Section 4.3, the constants disagreeing with the asset | Read at `constant_params.json:53-54` and `quadruped_identified_cfg.py:100-103`, the consequence 0.117 m computed, NOT simulated |
| Any runtime behaviour | NOT reproduced. No simulator reachable, so every consequence follows from reading |
| B4 | NOT confirmed. The Python on this filesystem is not the training interpreter |
| `co_optimisation/tests/design_generator_test.py`, 43 tests | NOT run, `pytest` absent here |
| Round robin ordering, that `Usd.Prim.GetAllChildren` returns environment prims in creation order so `wrappers.py:114` agrees with `scene.env_prim_paths` | Reasoned from the cloner's construction order, NOT measured. One print at the first respawn settles it permanently |
| `CatCMAESDesignGenerator` | Read, reachable only for the biped today by B1, never exercised in a run. The margin correction follows Hamano and colleagues [1] as implemented by `cmaes` [2] |
| 2026-10-06, section 4.9, the importer's `physics:localRot0` | Verified in the container against the converted USD of two designs per robot. A fixed joint carries the body quat, a revolute joint the body quat composed with the axis alignment that is its own `localRot1`, the rule holding to 5.4e-7 over 14 joints. See the correction of section 10 |
| 2026-10-06, section 4.6, the two biped schemes one geometry | Verified, body origins, collision geoms and centres of mass agreeing to 5.1e-7 m, and the quadruped's frame document at zero joint angles agreeing with the legacy one at its trained nominals to 2.6e-7 m |
| 2026-10-06, section 4.3, the constants' consequence | The frame donor built from the corrected constants stands at 0.292000 m with its foot 1.3e-7 m from under the hip axis, by forward kinematics of the emitted document |
| 2026-10-06, section 6, N1's physics propagation | Measured. A design written in place through environment 0 stands at its solved pose in environment 1, which shares it and was never written, to 4.1e-7 m, for geometry, `localPos0` and `localRot0` |
| 2026-10-06, section 6, round robin ordering | `find_matching_prims` walks `GetAllChildren` in authored order, and the plan asserts the matched order equals `scene.env_prim_paths` at every respawn, passing in four training runs |
| 2026-10-06, B4 | Confirmed. `cmaes` is absent from the training container, and `co_optimisation` and `design_generator` are installed there as copies, not editable |
| 2026-10-06, the 43 tests | Run in the container, 43 of 43 on the current tree, 58 of 58 after the plan, against the real `cma` 4.4.4 and the real `cmaes` 0.13.1 |
| 2026-10-06 second revision, the fold moved from the quadruped's joint nominals into its body frames | Measured in the container by the plan's appendix C. The committed quadruped document and its regeneration with the fold in the thigh and shank frames agree at 20 random configurations, the old at q and the new at q less the fold, to 7.2e-7 m in every body origin and centre of mass, 9.6e-10 kg m^2 in every world inertia, exactly in mass, and after 200 steps under the same targets to 1.4e-5 m. The biped's regeneration, whose thigh frame turns by 35.0 deg, agrees to 6.0e-7 m, 7.5e-9 kg m^2 and 7.7e-5 m after 200 steps |
| 2026-10-06 second revision, the quadruped tasks' readers of a joint angle | Read term by term in `cfg/quadruped/base_env_cfg.py`. The actions, every `joint_pos_rel` observation, `pen_joint_pos_limits`, `pen_abad_deviation`, the joint offset and joint reset events read a joint relative to its nominal or against limits that move with it, and the only absolute reader is the critic's `robot_joint_pos`, which reports `default_joint_pos` itself (`observations.py:70-76`) |
| 2026-10-06 second revision, the unit suite | 62 of 62 after the revised plan, with the suite's `cmaes` stub and with the real `cmaes` 0.13.1, against 43 of 43 on the current tree |
| 2026-10-07, the categorical search in place | Measured in the container by the plan's appendix A. A design whose lengths, pose and every actuator differ from the one its environment holds, written in place through environment 0, equals its own fresh conversion in environment 2 in every body pose at zero joint angles, every mass, inertia and centre of mass of the PhysX articulation view and every collider volume PhysX cooked, to 7.2e-7 for the quadruped and 5.0e-7 for the biped, in environment 0 and in environment 1, which was never written. Before the write the same quantities differed by at least 7.2e-5 |
| 2026-10-07, where the importer authors what an actuator pick moves | Read in a converted biped dumped prim by prim. A housing cylinder's `radius` and `height` on the `UsdGeomCylinder` inside the housing's visual and collision prototypes, under an `Xform` of unit scale carrying the axis alignment, in the design's base layer, its `extent` computed about x while its `axis` is z (`MjcfUsd.cpp:799-818`), the housing's mass, diagonal inertia and principal axes on its body prim in the physics layer, and every placement as a joint's `localPos0`, a welded body's fixed joint named after the body |
| 2026-10-07, that Isaac Lab exposes no joint frame | Read. `ArticulationData` has none, `grep -rn "localRot\|joint_frame"` over `isaaclab/` finds nothing, and the PhysX `ArticulationView` of the container's `omni.physics.tensors-107.3.26` offers link transforms, DOF state, limits, gains, masses and inertias and a metatype of names and parents, but no getter for a joint's local frames |
| 2026-10-07, the nominal pose read from the joint frames | Measured by the plan's appendix A, `localRot0 * localRot1^-1` of each folded body's joint agreeing with the generator's `pose_frames` to 1.1e-6 rad, the converted document's quaternions being written to six decimals and the in place writes agreeing to 1.2e-7 rad, and with the fold the bodies realise at zero joint angles to 1.2e-7 rad, before and after an in place write. The stage read costs 0.4 to 0.65 s per 1024 environments |
| 2026-10-07, the importer warning `Could not determine geometry type, defaulting to Sphere ()` | Read in the importer source. `LoadGeom` warns whenever a `geom` element carries neither `type` nor `mesh` (`IsaacSim/source/extensions/isaacsim.asset.importer.mjcf/plugins/isaacsim.asset.importer.mjcf/MjcfParser.cpp:189-233`), and it is also called on the `geom` of every `default` class (`MjcfParser.cpp:641`, `isDefault` true). An unnamed body geom is renamed `_geom_N` before the check (`MjcfParser.cpp:178-182`), so the empty parentheses identify a default class. Both committed documents declare three such defaults, the root, `visual` and `collision` (`environments/environments/assets/mjcf/quadruped/quadruped.xml:10-16`, the same in `solefoot/tron1/base_robot.xml`), each setting contact and display attributes only, and every one of their 53 and 33 body geoms states its own `type`, which overrides the inherited sphere. The warning therefore changes no shape, it predates the implementation since the default blocks are unchanged by it, and it repeats three times per conversion, once per default class, including each individual a respawn converts. Appendix A's collider volumes and the in place editor's `no box or cylinder` check confirm the converted shapes. |
| 2026-10-19, a design written in place is drawn with the geometry of the design it replaced | Observed in the training video of `quadruped_copt/2026-10-07_07-35-03` at step 190000, whose shanks are drawn through and past their feet ([artefacts/quadruped_copt-2026-10-07_07-35-03-train-video/](artefacts/quadruped_copt-2026-10-07_07-35-03-train-video/)), and measured by a diagnostic scene in which a 0.24 m shank was written over a spawned 0.18 m one. Physics follows the write. The foot body stands 0.2400 m from the shank, the shank collider sits at (0, 0, -0.12) m in the shank frame as in a converted copy of the same design (PhysX property query), the composed USD carries the new box, and the runner's `assert_link_lengths_applied` (`co_optimisation/co_optimisation/utils/update.py`) checks every swap of a run against the requested separations, so that run reached iteration 8000 with every foot where its design put it, and thigh and shank contacts are in any case penalised and terminal (`cfg/quadruped/base_env_cfg.py:1135`, `:1279`). The renderer does not follow. With the Fabric Scene Delegate on (`app.useFabricSceneDelegate` in `apps/isaaclab.python.headless.rendering.kit:38`), Fabric holds one copy of each instanced geometry prim per environment whose `omni:fabric:localMatrix` and `extent` are taken from the prototype at population, and an edit of the prototype's source layer reaches Fabric's prototype prim but not these copies, so the written shank was drawn at the spawned length ([artefacts/render-refresh-diagnostic-2026-10-19/quadruped_stale_render.png](artefacts/render-refresh-diagnostic-2026-10-19/quadruped_stale_render.png)). Rewriting each environment's copy of the visual boxes from the composed prototype closes it, the mean absolute difference between the written and the converted render falling from 0.685 to 0.364 against a noise floor of 0.223 between two identical designs ([artefacts/render-refresh-diagnostic-2026-10-19/quadruped_refreshed_render.png](artefacts/render-refresh-diagnostic-2026-10-19/quadruped_refreshed_render.png)). That is `refresh_rendered_geometry` in `update.py`, which the runner calls after each in place write and on resume, returning at once when the scene delegate is off, and which wrote 2048 prims in 0.12 to 0.26 s per swap of a 256 environment run with `--video`. Re-instancing the visual prims instead made them vanish from the render. The housings' cylinders are left standing, since the RTX delegate does not re-tessellate an implicit shape when its `radius` or `height` changes, writing either to the copy or to Fabric's prototype prim leaving the drawing unchanged, and a scale relative to the populated size did not reproduce the converted rendering, itself drawn through the importer's extent about the wrong axis. Only the biped's categorical search changes a housing's size. |

## 10. Corrections to the first pass

Recorded rather than deleted, per CLAUDE.md rule 17, so that the reasoning which produced the earlier claim survives.

| First pass | Correction |
|---|---|
| D3, `encoderOut` reaches the actor unnormalised and `output_detach` is ignored | The normalisation half is wrong. `copt_actor_critic.py:145` normalises the concatenated actor observation and `:160` maintains its statistics. Only the `output_detach` half stands. See section 5, D3 |
| D7, the biped stands at 0.641 m at `s = 0.75` and 0.967 m at `s = 1.25` | Wrong, and asymmetric about the target, which was the clue. The height is affine in the scales, `0.290497 + 0.245746 s1 + 0.268100 s2`, giving 0.6759 and 0.9328 about a nominal 0.804343. See section 4.1 |
| D1's remedy, a `COS_FOLD` constant of 0.633691 in the runner | Replaced. The fold is a design variable, not a constant, and the drop is measured from the simulation rather than derived. See section 4 |
| B1's remedy, a task map carrying both the family and a catalogue shortlist | Replaced. The map carries the family alone, the shelf being the family's own under B6 and B7 |
| B3's remedy, a warning when the population does not divide the scene | Replaced by an assertion, and by `num_envs = 6144` for the quadruped |
| D1 as revised 2026-09-30, `NOMINAL_POSE_RANGES` with a quadruped `knee_fold` of (1.55, 1.99), an empty biped dict, `apply_nominal_pose_params` writing per design joint nominals, `_current_scales` carrying the fold | Superseded 2026-10-01 by sections 4.3 to 4.13. Two angles are searched for both robots as offsets on `DesignSpec`, the biped's ankle is computed, the pose is carried by body frames in the XML so no joint nominal is written, and the reasoning that the drop is measured not derived survives as `apply_standing_height` with the derived value asserted against it. The knee interval of 1.55 to 1.99 rad survives as a knee offset of plus or minus 0.22 rad, widened to 0.25 |
| Section 4.1, 72.2 deg between the biped's thigh and shank directions | Wrong, the directions are 35.0 and -40.0 deg as rotations of -z, which differ by 75.0 deg. The figure was not used downstream |
| Section 4.3, the hip nominal of the quadruped fixed by the knee alone, the foot constraint holding at the donor angles at every length | Wrong when the lengths move. The residual is 0 at equal lengths and up to 82 mm at the scale corners, and 12 to 125 mm for the biped, so the constraint is solved per design, section 4.5 |
| Section 4.6, the in place path assumed to carry a pose written into the nominal joint angles | Wrong in a way that no longer matters. The in place editor writes no joint rotation at all (`update.py:152-165`), which is why the pose now travels in frames and `physics:localRot0` is added, section 4.9 |
| X5, zero armature on a bare entry, dormant because `catalog_names` restricts the shelf | Absorbed into B7 and no longer dormant, the shortlist being removed and the underlying catalogue incompleteness reaching further than armature |
| Section 4.9, `localRot0` written as the bare body quat, and the expected donor quats compared against it | Corrected 2026-10-06. Wrong for a revolute joint, whose `localRot0` is the body quat composed with its `localRot1` (`MjcfImporter.cpp:1341-1366,1530-1537`), so the bare write would turn every knee about local x. The in place write composes the two. Plan section 1, finding 1 |
| Sections 4.3 and 4.5, `DX_DONOR = 0.025204` for the biped and the fitted chains `h = 0.290497 + l1 cos(t) + l2 cos(s)` | Corrected 2026-10-06. The fitted biped chain omits the shank to ankle clearance and fixes the housings at TRON1's. The donor's ankle sits 13.987 mm ahead of the hip axis by forward kinematics of the emitted document, and the plan solves on the exact chain built from the builder's helpers. Over the box the heights are [0.1924, 0.4050] m and [0.6375, 0.9789] m, and 68.3 per cent of the biped's draws pass under `"default"` where 69.7 was given. Plan sections 1 and 5 |
| Section 4.10, `foot_sole_offset` a family constant | Corrected 2026-10-06. The biped's sole depth follows the ankle housing a categorical search picks, so the measurement compares the kinematic depth and the foot's lead over the hip, and the sole comes from a per design record. Plan section 6 |
| Section 4.7, the quadruped nominals zeroed in `quadruped_identified_cfg.py` | Corrected 2026-10-06. That configuration, the quadruped's `RewardsCfg` and `CurriculumCfg`, and the biped's `RewardsCfg` are shared with the plain and HIM tasks, so the copt quadruped gets its own configuration and frame scheme document, and every reward and curriculum change is made in the copt environment's `__post_init__`. Plan section 1, finding 4 |
| Section 4.6, the quadruped quat emitted under every `for_isaaclab` document | Corrected 2026-10-06. Only when a spec carries an offset, as for the biped, so the committed plain asset regenerates byte identical |
| Section 4.11, `foot_clearance_reward_v5`, and `feet_air_time` called by four configurations | Corrected 2026-10-06. `foot_clearance_reward_v5` exists (`rewards.py:856`) and the new function is `v6`. `feet_air_time` has two callers, the KScale and SD_BRS1 configurations calling `feet_air_time_positive_biped` |
| D5, the flag recorded in `params/agent.yaml` | Corrected 2026-10-06. `train.py` dumps the configclass, not the dict carrying `copt`, so the plan dumps that dict as `params/copt_agent.yaml` |
| D2, `robot_joint_pos` added to the estimator target, and the loss coefficient on the algorithm configuration | Corrected 2026-10-06. `default_joint_pos` is zero for every design once the pose lives in the frames, so the target takes the offsets and standing height instead, and the coefficient is a `CoptPPO` keyword because the plain runners hand the shared algorithm configuration to `PPO` |
| D6, the tracking standard deviations frozen at their initial values | Corrected 2026-10-06. The swap's reset re-enters the curriculum at the step counter of the step before it, so a term whose interval divides the swap step is evaluated twice there, the push force stepped twice, rather than frozen. The remedy is unchanged, and the rescale to x 25 covers seven terms with `terrain_levels` |
| B7, `SIM_ACTUATOR_CATALOG` bound at import | Corrected 2026-10-06. `set_active_catalog` rebinds `ACTUATOR_CATALOG`, so the filter is a function over the live global |
| Not recorded, `_build_xml`'s fallback | Added 2026-10-06. The fallback emits the last feasible spec while `_build_population` records the refused one, and a refused frame scheme design falling back to the legacy baseline would mix schemes under the in place editor. Both closed in the plan's section 5 |
| Not recorded, the categorical search | Added 2026-10-06. `--design_search cat` cannot start, its generator built with `late_start=False` while the runner is handed `ea_late_start = 12000`, so no batch is ever sampled. Reproduced on the current tree, `AssertionError: _generate_individual called before sample_batch`. Closed in the plan's section 3 |
| Not recorded, the categorical search's in place swaps | Added 2026-10-06. An actuator pick sets the biped's housing radii, from which `_cluster_layout` places the hip axis and the sole, and sets housing masses on both robots, none of which the in place editor authors. The plan's pose measurement refused the first in place swap of a biped categorical run with the foot 18.30 mm from its solved depth, the 48.3 mm radius of a TRON1 knee class housing less the 30.0 mm of the TRON1 ankle housing. Closed in the plan's section 3 by respawning the categorical search every generation, after which both robots' categorical runs stood within 0.004 mm at every swap |
| Not recorded, the categorical search's step size | Added 2026-10-06. `CatCMAwM` multiplies `sigma` by a default covariance whose standard deviation is a sixth of each continuous range's half width (`cmaes/_catcmawm.py:328-333`, version 0.13.1), so `sigma0 = 0.25` gives an initial standard deviation of about 0.021 in the unit box, measured 0.017 to 0.023, against pycma's 0.25 for the other searches. Left as a decision in the plan's section 11 |
| B3, `6000 % 256 = 96` (section 3) | Corrected 2026-10-06 second revision. `6000 % 256 = 112`, the conclusion unchanged |
| B3's remedy, an assertion in the runner's constructor and per generation | Corrected 2026-10-06 second revision. A generator's populations, `late_start_prior_pop_size` and `num_individuals`, are fixed at its construction, so `train.py` checks both there and the runner checks nothing. The constructor assertion would also have refused `play.py`, which builds the runner with 256 individuals over the 32 environments of every copt Play configuration |
| The correction above of section 4.7, a separate copt quadruped configuration and document | Reversed 2026-10-06 second revision by the review of the plan. `QUADRUPED_IDENTIFIED_CFG`'s pitch nominals are zeroed for every quadruped task as section 4.7 first proposed, `QUADRUPED_INIT_JOINT_POS` is deleted and the committed quadruped document is regenerated with its fold in the body frames. The move changes joint coordinates alone, measured in section 9's row of the same date |
| The correction above of section 4.6, the quadruped quat only when a spec carries an offset | Reversed 2026-10-06 second revision. Every `for_isaaclab` document of both robots carries its fold in its body frames, the `DesignSpec` offsets are floats defaulting to 0.0, the quadruped's spec has no ankle field, and the MuJoCo documents keep their keyframes. The committed biped document is regenerated too, its joints unchanged and its thigh frame turned by 35.0 deg |
| Section 4.4, `NOMINAL_HIP_REFERENCE` with `"solved"` and `"default"` readings, and the correction above quoting the pass rate under `"default"` | Superseded 2026-10-06 second revision. The hip is always solved and the switch is removed, so the pass rate under `"default"` describes nothing that exists |
| Section 4.8, the three range types `offset`, `scale` and `absolute` | Narrowed 2026-10-06 second revision to `scale` and `offset`, no parameter taking `absolute`, which for the pose coincides with an offset from the baseline's zero |
| Section 4.5, the foot tolerance `1.05 (l1 + l2) sin(lean)` | Refined 2026-10-06 second revision to `sin(lean + 5e-4)`, the hip offset being rounded to three decimals, without which a search of the lengths alone, both pose ranges `(0.0, 0.0)`, refuses every design |
| The closure above of the categorical search, CMA-ES from the first generation | Replaced 2026-10-06 second revision. The categorical generator is built with the late start schedule of the other CMA-ES generators, so it runs the random phase its `_generate_individual` already implements (`usd_generator.py:910-916`), every joint's actuator drawn uniformly from the shelf, and it respawns every generation of both phases |
| The correction above of section 4.7, every reward and curriculum change made in the copt environment's `__post_init__` | Superseded 2026-10-07 by the plan's second review. Each robot's file carries `CoptRewardsCfg(RewardsCfg)` and `CoptCurriculumCfg(CurriculumCfg)`, which redefine the differing terms alone and are named by the co-optimisation environment, the plain and HIM tasks keeping the shared classes. Every copt task's rewards resolve term for term, in the same order, to what the `__post_init__` wiring gave, which is the "copt RewardsCfg" section 4.11 first named |
| The closures above of the categorical search's in place swaps, by respawning every generation | Replaced 2026-10-07. The blocker was the in place editor's search for a box, every quantity a pick moves being a stage attribute it can reach, a cylinder's `radius` and `height`, a housing's mass and inertia and every placement's `localPos0`. The plan's payload carries every housing and the biped's sole, the editor writes a cylinder as it writes a box, and the categorical search swaps in place, a respawn remaining at the late start alone, both robots' runs standing within 0.005 mm at every swap |
| The correction above of D2, the target taking the offsets and the standing height | Refined 2026-10-07. The target takes the fold of every folded body, read by `robot_nominal_pose` from the body's joint frame on the stage, Isaac Lab exposing no joint frame and `default_joint_pos` being zero, and the critic takes it beside `link_lengths`. The quadruped's target names the twelve housings too, a categorical search moving their masses, 216 channels in place of 91 |
| D6, intervals written as iterations x 24 for the quadruped alone (section 5) | Extended 2026-10-07. The biped's co-optimisation runner takes 25 steps per iteration as well, `SF_TRON1AFlatPPORunnerCfg` (`limx_rsl_rl_ppo_cfg.py:98`) being the parent of `SFCoptBaseRunnerCfg` (`:143`), so the premise that 24 is the biped's count holds for the PF, WF, Berkeley, SD_BRS1 and KScale runners and not for SF, and the biped's copt curriculum is rescaled as the quadruped's |
| Not recorded, the random generator under the runner | Added 2026-10-06 second revision. `RandomDesignGenerator` has no `get_state`, `load_state` or `toggle_late_start`, which the runner calls (`copt_on_policy_runner.py:182`, `:228`, `:427`), so a random search could not be checkpointed. The plan gives the base class a stateless pair and the random search no late start |
| The 2026-10-19 row of section 9, the housings' cylinders left undrawn by `refresh_rendered_geometry` because only the biped's categorical search changes a housing's size | Corrected 2026-11-17. Since `plans/housing_clearance_and_trunk_width.md` landed on 2026-11-16 the quadruped's housings take their radius and half length from the catalogue (`design_generator/quadruped/builder.py:176-179`), so its in place swaps change housing sizes too. In `quadruped_copt/2026-11-16_08-05-55` environment 0 was spawned with the A1 knee (0.046, 0.020 m) and was drawn at that size after every swap while its body moved to the new pick's place, giving drawn hip to knee gaps of 37.5 mm in generation 1 and 19.4 mm in generation 2 against the true 10 mm of `clearance_hipAct2kneeAct_y`, and drawn knee to thigh gaps of 27.5 and 9.4 mm against a true zero ([artefacts/quadruped_copt-2026-11-16_08-05-55-train-video/](artefacts/quadruped_copt-2026-11-16_08-05-55-train-video/)). The builder and the physics were correct throughout, `assert_link_lengths_applied` checking every housing's parent separation. The refresh now redraws a cylinder whose written size differs from its spawned one through `_scaled_cylinder` in `co_optimisation/co_optimisation/utils/update.py`, drawing the spawned tessellation through a local matrix scaled in the cylinder's own frame by the written to spawned ratio, the spawned size being read from the Fabric copy's own `radius` and `height`, which nothing writes. A cylinder of unchanged size is skipped, so a run whose actuators stay fixed writes exactly what it wrote before. The scaling was checked against a USD cylinder in the training container's `pxr`. The rendered result in a live scene is not yet verified |
| The 2026-11-17 correction above, the resized housings redrawn by `_scaled_cylinder` | Corrected again 2026-11-17, later the same day. The redraw does not take effect. `quadruped_copt/2026-11-17_05-25-55` and `quadruped_copt/2026-11-17_07-11-53` both ran the fixed `update.py`, the launcher reinstalling `co_optimisation` from the working tree at every launch (`/ws/djinn:213`), the installed copy in containers gpu0 and gpu1 being byte identical to the tree and the second run starting at the moment of its reinstall, 07:11:33. In both, environment 0's knee housing changes at the first swap from radius 0.046 m and half length 0.020 m to 0.045 m and 0.0475 m, its body moving from 0.050 to 0.0775 m off the hip housing (`copt_usds/gen_0000` and `gen_0001`, `individual_0000.xml`), so a correct drawing shows a knee drum 95 mm long, 10 mm from the hip housing and flush with the thigh, and a stale one a 40 mm disc with gaps of 37.5 and 27.5 mm. Every frame of the step 10000 video of the second run, which begins at that swap, iteration 400 at 25 steps per environment, draws the 40 mm disc and both gaps ([artefacts/quadruped_copt-2026-11-17_07-11-53-train-video/](artefacts/quadruped_copt-2026-11-17_07-11-53-train-video/)), and the first run's video does the same, its oblique views hiding the gaps by perspective. The cause is not yet isolated, no console log of either run being kept and the refresh printing only a total. `_scaled_cylinder` returns None, and the cylinder is skipped silently, when the Fabric copy carries no `radius` or `height` or when they already equal the written size (`co_optimisation/co_optimisation/utils/update.py:316-323`), and either would leave the spawned tessellation drawn. If the scaled matrix was written, the renderer ignored it, which the 2026-10-19 row records already for this very approach, a scale relative to the populated size not reproducing the converted rendering. The 2026-11-17 correction was checked only as arithmetic in `pxr`, and its rendered result was never verified live, so the gaps of both runs are the original defect unrepaired. Physics is unaffected, `assert_link_lengths_applied` checking every swap |
| The cause of the second 2026-11-17 correction, isolated live | Established 2026-11-17 in container gpu1 by three runs of a diagnostic scene, env_0 spawned with the A1 quadruped and written in place with an ANYdrive-3 knee through `apply_link_length_params`, env_1 that design converted afresh as the reference, both drawn through `/OmniverseKit_Persp` with the Fabric Scene Delegate on ([artefacts/render-refresh-diagnostic-2026-11-17/](artefacts/render-refresh-diagnostic-2026-11-17/)). Two defects compound. First, after the in place write and its `sim.reset()` the Fabric copy of a housing cylinder already carries the written `radius` and `height`, 0.045 and 0.095 m, while the renderer keeps drawing the spawned tessellation, so `_scaled_cylinder`, which takes the spawned size from that copy, finds no change and returns None (`co_optimisation/co_optimisation/utils/update.py:322-323`), and the refresh skips every resized cylinder in silence (run 2). A first run that read the copies through usdrt before the write saw them keep the spawned size, which the training runner never does, so that run's skip did not occur. Second, a `omni:fabric:localMatrix` written straight after the reset is not carried into `omni:fabric:worldMatrix`, which the renderer draws from and which Fabric recomputes only through `IFabricHierarchy.updateWorldXforms()` under its change tracking, so the refresh of run 1 wrote the correct scaled matrix and was drawn unchanged, while the same value written again later was drawn correctly. With the spawned size taken from the spawned design rather than from Fabric, and `update_world_xforms()` called after the writes, through either the attribute or `set_local_xform`, every knee was drawn as the reference draws it, and still so after 40 physics steps (run 3). The `r0` and `h0` of `_scaled_cylinder`'s ratio and extent are correct as written, the ratio scaling the spawned drawing to the written size and the extent being the spawned tessellation's own |
| The repair of the cause isolated above | Implemented 2026-11-17 by [../plans/cylinder_render_refresh.md](../plans/cylinder_render_refresh.md). `_scaled_cylinder` takes the spawned (radius, height) where given (`co_optimisation/co_optimisation/utils/update.py:307`), `refresh_rendered_geometry` takes the link dicts of the population last spawned, reads each housing's spawned size as its `radius` and twice its `half_length`, calls `IFabricHierarchy.update_world_xforms()` after its writes and prints the count of cylinders redrawn (`update.py:339`, `:414`, `:440`), and the runner records that population in `_respawn_population`, through which every spawn passes, and passes it (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:210`, `:649`, `:758`). Verified the same day in container gpu1 by `co_optimisation/tests/sim_render_refresh_check.py`, which runs the runner's own sequence against the repository's `update.py`, no usdrt read preceding the refresh, and found every knee copy's world matrix scaled 2.37500 along the axis and 0.97826 across it, before any render and after 40 physics steps, the written design drawn as the reference draws it ([artefacts/render-refresh-diagnostic-2026-11-17/check_after_fix_top.png](artefacts/render-refresh-diagnostic-2026-11-17/check_after_fix_top.png)). Left standing, by the unset default of the new argument, the reading of the spawned size from the Fabric copy, which no caller now uses. A training video has not yet confirmed it, the containers' installed copy being refreshed only at the next launch |

## 11. Provenance of this revision

The dispositions of section 1 and the redesigns of sections 3 and 4 follow a review of the first pass recorded on 2026-09-30. Its decisions, in the order given, were that the generator's actuator shelf and groups belong to the family rather than to an argument, that B2 is worth applying although every co-optimisation run of record is on rough terrain, that B3 warrants an assertion rather than a warning, that B4 lands with a plan, that the nominal pose including the fold must be a searched design variable with every nominal referenced reward target following it, that the quadruped must adopt the biped's architecture in which morphology is predicted, that D3 be re-verified, that D4 be retained unused, that D5's reset be gated rather than removed, that D6 be planned, that the spawn height move with the nominal pose, that the design lattice stay small for now, that latency and memory wait for the larger design space, and that the latent defects wait.

A second review on 2026-10-01 redesigned the nominal pose. Its decisions were that both robots take a hip pitch and a knee offset and the biped's ankle is computed to keep the sole flat, that the quadruped XML carry its pose in body quats as the biped's does, that `DesignSpec` carry `thigh_bend_offset`, `knee_bend_offset` and `ankle_bend_offset` and gain `from_dict`, that every solve and check live in `usd_generator.py` with the design generator accepting a finished spec, that the single range structure be typed `offset`, `scale` or `absolute`, that `_sample_scales`, `_sample_scales_v2` and `_current_scales` take parameter names, that `dataclasses.replace` be retired, that the six unread actuator scales be deprecated, and that each reward and termination term have a v2 that accounts for the pose, which section 4.11 reads as each term that depends on it. Findings D1a to D1d and sections 4.3 to 4.13 record the investigation behind the plan, and the matters it left to the user are the hip reference of section 4.4, the corridors of section 4.5 and the tolerance of the standing height check.

A third review on 2026-10-06 revised the implementation plan rather than this audit. Its decisions were that the quadruped, like the biped, carry its fold in the body frames of its committed document and every generated one, with every pitch nominal zero and no separate co-optimisation asset, that the `DesignSpec` offsets default to 0.0, that the hip reference switch be removed, that comments carry only what the code cannot say, that the population check be made where the generator is built, that all four generators be selectable, and that the categorical search have a random phase over the actuators. The rows of sections 9 and 10 dated 2026-10-06 second revision record the consequences.

A fourth review on 2026-10-07 revised the plan again. Its decisions were that each robot's tasks be confined to that robot's configuration file, sharing what they can and taking new classes, `CoptRewardsCfg` and `CoptCurriculumCfg`, where they differ, that the categorical search's designs be written in place as the lengths are, the cylinder of a housing being handled as the box of a link, and that the nominal pose the document folds into the articulation be extracted from the simulation and observed in the morphology group, the estimator being asked to predict it as part of the morphology. The rows of sections 9 and 10 dated 2026-10-07 record the consequences.

## 12. Implementation outcome, 2026-10-07

The plan [co_optimisation_audit_implementation.md](../plans/co_optimisation_audit_implementation.md) was applied to the tree on 2026-10-07. Its diffs had been generated from a sandbox copy of the repository against a pristine snapshot, and every one of the 31 files the plan modifies was first confirmed byte identical to that snapshot, so the 31 files and the three new `sim_*_check.py` scripts were copied from the verified sandbox unchanged. The tree therefore carries the plan's edits exactly, and no edit of the user's was overwritten. The two packages were reinstalled editable in `isaac-lab-base-shreyas-gpu0`, and `design_generator` now resolves to the repository's `/workspace/isaaclab/biped/design_generator/__init__.py` rather than to `site-packages`.

The checks of the plan's section 9 were repeated on the tree, and each reproduced its sandbox figure.

| check | quadruped | biped |
|---|---|---|
| unit suite, `design_generator_test.py` | 66 of 66 passed, both families together | |
| appendix A, `localRot0` rule | 5.4e-7 | 4.8e-7 |
| appendix A, design A spawned, design B written in place | 4.7e-7 m, 3.7e-7 m | 6.2e-7 m, 3.1e-7 m |
| appendix A, in place against a fresh conversion, every quantity | 7.2e-7 | 5.0e-7 |
| appendix A, smallest change between A and B | 7.2e-5 | 2.5e-4 |
| appendix A, `robot_nominal_pose` against the realised fold | 1.2e-7 rad | 1.2e-7 rad |
| appendix A, stage read of 1024 environments | 0.51 s | 0.41 s |
| appendix B, reward invariance | PASS, 16 comparisons over both robots | |
| appendix C, 20 static configurations | 7.2e-7 m, world inertia 9.6e-10 kg m^2, limits 1.2e-7 rad | 6.0e-7 m, world inertia 7.5e-9 kg m^2, limits exact |
| appendix C, 200 steps | 1.4e-5 m, 1.5e-5 rad | 7.7e-5 m, 8.3e-7 rad |
| appendix C, inertia in the body frame changed | none | `thigh_R_Link`, `thigh_L_Link`, as section 4.1 of the plan predicts |

The validation run of the plan's section 9.8 ran from `/tmp/copt_validation` in three concurrent lanes at 256 environments and 32 individuals, with a swap every 2 iterations and the late start at 4. Sixteen runs exited cleanly with no traceback, namely `cat`, `length`, `cmaes` and `random` for 9 iterations on each robot, a resume of each robot's `cat` and `length` runs from `model_4.pt`, both flat copt tasks for one iteration and both plain tasks for two. Every swap printed `nominal pose measured`, the worst disagreement over all runs being 0.006 mm against the 2 mm tolerance. The `cat`, `length` and `cmaes` runs respawned once, at the late start, and wrote every other generation in place, the `random` runs and every resume wrote all of theirs in place. Both `length` and `cmaes` printed `aCMA-ES ... in dimension 4`. The estimator mapped 5875 inputs to 216 outputs on the quadruped and 5575 to 160 on the biped. Each `params/copt_agent.yaml` carried `reinit_lr_each_generation: true` and `estimator_loss_coef: 1.0`, and both flat copt runs wrote `model_0.pt`.

Four points diverge from the plan's text, none from its code. The plan's section 4.4 step 4 asks for `git diff --stat` over `environments/environments/assets/mjcf`, but that directory is untracked, so the regenerated documents were compared against the snapshot instead, giving the 32 and 20 changed lines the plan predicts. The run log and the design CSVs name no actuator, a categorical pick being stored as its payload, so the restriction of the shelf to the `A1-*` and `TRON1-*` entries rests on the construction of `_resolve_categories` and on the unit suite's assertion that the generator's names equal the simulable shelf (`co_optimisation/tests/design_generator_test.py:515`). A stale `co_optimisation/build/lib` from an earlier non editable install still carries the removed `catalog_names` and `actuator_groups`, but it is ignored by git (`.gitignore:38`) and lies off the import path since the editable reinstall, so it was left untouched. The registry table of the root `ARCHITECTURE.md` already omitted the quadruped copt tasks, and since the plan registers no task it was left as it stood.

The root documents were updated in the same pass, `ARCHITECTURE.md` in its sections 4.4, 4.5, 5 and 5.1 and `CO_OPTIMISATION.md` by a dated section 7 that supersedes its earlier descriptions of the generators, the swap and the nominal pose. [quadruped.md](quadruped.md) records the corrected constants, the fold in the body frames, the zero nominals and the shifted pitch ranges, and [tron1.md](tron1.md) the rotated thigh frame and the 13.99 mm ankle lead.

## 12a. The actuator data that reaches Isaac Lab, verified 2026-11-12

The builders write each motor class's `forcerange` from the picked entry's `peak_torque` (`design_generator/biped/builder.py:530-535`, the quadruped builder alike), while `IdentifiedActuator` clips against `effort_limit`, which `sim_params()` takes from the entry or else from `peak_torque` (`design_generator/common/actuator_catalog.py:552`). Emitting every simulation complete entry on every joint group gave a `forcerange` equal to both in all 16 biped and all 9 quadruped cases, and no entry of either catalogue states a distinct `effort_limit`, so the document and the simulator agree today and would part if one did. No other element of either builder's output, neither `gear` nor `actuatorfrcrange`, bounds a joint's torque.

Short training runs of both robots under the categorical searches, recorded in `../plans/design_filtering.md` section 8.2, compared every environment after the spawn and after four in place updates with the catalogue entry its design picked. `effort_limit`, `velocity_limit`, `stiffness`, `damping`, the friction triple, the promoted `saturation_effort` and the armature PhysX holds matched to float32 rounding, no PhysX joint ceiling lay below its effort limit, and every body mass PhysX reports matched the document's inertial mass to 7.2e-06 relative, 15 of 15 bodies on the biped and 25 of 25 on the quadruped. Only the seven entries that state gains, friction and armature can be picked at present, every other catalogue entry lacking all five keys and a rotor inertia, as that plan's section 7 records.

Addition of 2026-11-12, later the same day. The plan's step 9 was revised to fill each of the other entries from its nearest identified TRON1 or A1 actuator in log peak torque, peak speed and mass, leaving the seven identified entries unchanged. Applied in the sandbox only, it made all 15 biped and 14 quadruped entries pickable, the filled values reached every environment to float32 rounding, and the identified entries' parameters and both donors' documents were unchanged, so every existing task keeps its actuators. The tree's catalogues are unchanged until the plan is applied.

Addition of 2026-11-12, on application. The plan was applied to the tree the same day, the catalogues now filled as described, `design_search` reduced to `leg_geometry`, `actuator_leg_geometry` and `actuator`, and the standing filter enabled in `BipedCMAESCfg`. Both packages were reinstalled in container gpu0 and validated there, and container gpu1, then training three quadruped runs, was not reinstalled and still holds the former code.

Addition of 2026-11-12, later the same day. The plan's section 11 added the design records, every per design CSV carrying each design's absolute geometry, nominal pose, mass, actuator names, catalogue specifications and simulator parameters and filter scores beside a `designs/base_design.csv` of the base robot, the `cmaes/` TensorBoard logging of every searched quantity, its population statistics and the base robot's reference values, and the categorical random phase widening its actuator choice from the base robot's pick to the whole shelf. Both packages were then reinstalled in gpu0 and gpu1, each installed file equal to the tree's, the runs training in them continuing on their loaded modules. The installed co_optimisation copies also carry a stale `runners/design_generator` directory from the package's old `build/lib`, which nothing imports.

## 12b. The standing height check failing at the first in place swap, 2026-11-16

Two biped runs launched on 2026-11-16 with `--video --video_interval 10000 --save_interval 400 --seed 24` under `Isaac-Limx-SF-Copt-MoRAL-Rough-v0`, `copt_moral/2026-11-16_08-49-31` and `copt_moral/2026-11-16_10-20-14`, each saved `model_399.pt` and then stopped at the first in place swap, the check of `apply_standing_height` raising that the measured nominal pose differed from the solved one by 1264.83 mm in foot depth and 15019.81 mm in foot lead over the hip (`co_optimisation/co_optimisation/utils/update.py:504-507`). Both runs carried 5120 environments, 80 terrain columns and 80 designs per generation, as their dumped `params/env.yaml` records.

The depth error equals the probe height of 2.0 m less a standing depth near 0.735 m, which places the measured feet near the ground rather than under a base lifted to the probe (`co_optimisation/co_optimisation/utils/update.py:496-497`), and a lead of fifteen metres between a foot and its own knee is possible only where the link transforms were never refreshed after the probe was written. The two maxima are taken separately, so they need not come from one environment.

Eliminations were run in container gpu0 against a scratch copy of the installed package carrying diagnostics and a swap interval of three iterations, the repository and the installed package left untouched. The in place swap passed at 4096 environments, at 4096 environments with the video flags, at 5120 environments, and after resuming the crashed checkpoint at 5120 environments with 40 iterations of trained walking before the swap, every environment agreeing to within 0.003 mm and a fresh read of the link transforms matching the cached one exactly. The design draw was excluded as well, since the generator seeds each draw by generation and individual (`co_optimisation/co_optimisation/runners/usd_generator.py:966`) and generation zero was identical across the crashed runs and the tests, so the population applied in place at the first swap was the same.

The failure was reproduced only when 5120 environments, video rendering and the checkpoint save at the swap were combined. The in place `sim.reset()` (`co_optimisation/co_optimisation/utils/update.py:299`) then exhausted the GPU, PhysX reporting `Unable to allocate memory of size 671088640 for mGpuContactPairsDev` and `Failed to fetch DOF velocity attribute`, and the reinitialised articulation raising from `get_dof_velocities` (`IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py:68`, reached from `simulation_context.py:530` and `articulation.py:1578`). The 48 GB card held at that moment a foreign process of 22.5 GB outside every container, present also when the crashed run was stopped, together with about 22 GB for the run itself. The conclusion drawn is that the original runs met the same exhaustion, which there left the link transform buffer unrefreshed rather than raising, and that `apply_standing_height` caught the corrupted state as designed. The console output of the original runs was not captured, so the allocation failure is inferred for them rather than observed.

No code was changed. The configuration of the working tree had already been reduced to 4096 environments (`environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:1401`), which passed the swap with video under the same foreign allocation, and a run was relaunched on it at 12:26 UTC with its console written to `IsaacLab/logs/copt_moral_biped_2026-11-16_12-26-34.log`. A run at 5120 environments with video should be expected to fail the same way while the foreign allocation persists. The reproduction logs were deleted once this section recorded what they showed.

## 12c. The servo gain scales searched where named, 2026-11-17

`DEFAULT_PARAM_RANGES` has long carried eight multiplicative gain scales, `abad_stiffness_scale` through `ankle_damping_scale` (`co_optimisation/co_optimisation/runners/usd_generator.py:472`), and `_actuator_params` multiplies a catalogue entry's `stiffness` and `damping` by any of them it is handed (`usd_generator.py:889`). Two defects kept them from every search. `CMAES_PARAM_KEYS` named the four geometry keys alone and `CMAESDesignGenerator.__init__` truncated `param_ranges` to it, so a gain scale named in `CMAESCfg.param_ranges` was accepted by `_typed_ranges` and then discarded without a word. And every generator with a continuous block but the random one called `_actuator_params(spec)` without its parameters, so a scale could not have reached the actuators even had it been drawn, while `describe_design` passed them and the design logs would have recorded gains never applied.

Both were repaired on 2026-11-17 by the plan [gain_scale_search.md](../plans/gain_scale_search.md). `CMAES_PARAM_KEYS` now lists every key a CMA-ES search may carry, the four geometry keys followed by the eight gain scales (`usd_generator.py:519`). The restriction keeps a geometry key always and a gain scale only where the caller named it, reading the constructor's argument rather than the merged ranges, and refuses a scale naming no joint group of the family, such as `ankle_stiffness_scale` on the quadruped (`usd_generator.py:1082-1089`). `FixedDesignGenerator`, the fine tune's mean population, `CMAESDesignGenerator`, `GrowingDesignDistCMAESDesignGenerator` and `CatCMAESDesignGenerator` now hand their parameters to `_actuator_params` (`usd_generator.py:1009`, `:1223`, `:1244`, `:1311`, `:1622`), and the CatCMAwM resume check compares the pickled continuous dimension `_Nco` with the searched one (`usd_generator.py:1434`). A configuration naming no gain scale keeps its four dimensions, its initial mean, its random phase draws and its checkpoints, the gain scales being appended after the geometry in the search vector. The replay path of `play.py` reads `CMAES_PARAM_KEYS` from `final_design.csv` and so carries the scales without change.

The application to the simulator needed no change. `apply_actuator_params` writes each group's `stiffness` and `damping` rows per environment (`co_optimisation/co_optimisation/utils/respawn.py:246-305`), and `IdentifiedActuator`, an explicit `DCMotor`, reads them in `compute` at every physics step (`environments/environments/actuators/actuator_pd.py:18-40`), so a written row takes effect without a write through to PhysX.

The unit suite was extended by seven parametrised tests (`co_optimisation/tests/design_generator_test.py:260-343`) and the assertion of `:224-231` was narrowed to `SPEC_PARAM_FIELDS`, the four keys an unnamed configuration still searches. Run outside the container against `cma` 4.4.4 and `cmaes` 0.13.1, with `co_optimisation.utils.common` pre-registered because the suite's bare package stubs no longer satisfy `usd_generator.py:69`, `design_generator_test.py` passed 122 of 122 and `design_filter_test.py` 32 with 8 skipped for want of `ORIGINAL_USD_GENERATOR`. The end to end check in the simulator recommended by the plan has not been run.

Four defects are left standing. `RandomDesignGenerator` samples every key of `DEFAULT_PARAM_RANGES` whether named or not (`usd_generator.py:540`, `:841`), so it randomises all eight scales, the ankle pair inertly on the quadruped. `CatOnlyCMAESDesignGenerator` has no continuous block, so a gain scale named for it is kept in its ranges but neither searched nor applied. The startup `randomize_actuator_gains` events are overwritten by `apply_actuator_params` at the first generation, as section 1.2 of [../plans/w_eta_optimisation.md](../plans/w_eta_optimisation.md) records. And the default gain ranges were derived from the biped's runtime randomisation, so a quadruped run should state its own. That plan, not implemented, would replace the scales with a natural frequency and a damping ratio per group, and its treatment of them must be revised if both are kept.

Addition of 2026-11-17, later the same day, at the user's direction. Three statements above are revised. First, the geometry keys are no longer searched whether named or not. The keys of `param_ranges` alone decide what a CMA-ES generator tunes, in the order of `CMAES_PARAM_KEYS`, a generator constructed without ranges searching the four geometry keys as before (`co_optimisation/co_optimisation/runners/usd_generator.py:1099`). The gain scales are now listed in `GAIN_PARAM_KEYS` and merged into `CMAES_PARAM_KEYS` (`usd_generator.py:517`, `:525`). `_lean_max` returns zero where the lean is not searched, since every design then stands in its solved pose and only the rounding of the hip offset remains to be tolerated (`usd_generator.py:929`). Two callers that named the length scales alone and received the pose offsets unnamed now name all four keys, so that they search as before (`scripts/rsl_rl/play.py:1182`, `scripts/analysis/nan_reset_diagnostic.py:126-131`).

Second, `design_search` `actuator` is now served by `CatCMAESDesignGenerator` over a `param_ranges` of gain scales alone, so the actuator of every joint group is searched together with its gains on the baseline's geometry (`scripts/rsl_rl/train.py:267`). `train.py` refuses an `actuator` search whose ranges are empty or name any key outside `GAIN_PARAM_KEYS` (`train.py:274-279`). `CatOnlyCMAESDesignGenerator` is kept in the module but no longer used by `train.py` (`usd_generator.py:1644`), so the defect above that it drops a named gain scale no longer reaches a run. No robot configuration selects `actuator`.

Third, the overwriting of the startup `randomize_actuator_gains` events by `apply_actuator_params` is not a defect. The user stated that it is deliberate, the gains of a co-optimisation run being set by the design optimiser alone.

The suites were rerun as above with two new tests, that the ranges alone decide what is searched and that the actuator search keeps the baseline's geometry while tuning the picks and their gains (`co_optimisation/tests/design_generator_test.py:350`, `:359`), the combined gain tests now naming the geometry beside the gains. `design_generator_test.py` and `design_filter_test.py` together passed 162 with 8 skipped. The check in `train.py` imports the simulator and has not been exercised.

## 13. Bibliography

1. Hamano, R., Saito, S., Nomura, M., Shirakawa, S. CMA-ES with Margin, Lower-Bounding Marginal Probability for Mixed-Integer Black-Box Optimization. Genetic and Evolutionary Computation Conference (GECCO), 2022. arXiv:2205.13482. Registered in [literature.md](../../context/literature.md) cluster 16, whose entry records that the venue and author list were taken from the arXiv record and not verified against the ACM proceedings.
2. CyberAgent AI Lab. `cmaes`, a Python library for CMA Evolution Strategy. https://github.com/CyberAgentAILab/cmaes. Registered in [literature.md](../../context/literature.md) cluster 16, read at `README.md`, `examples/catcma_with_margin.py` and `cmaes/_catcmawm.py` on the `main` branch on 2026-09-25.
