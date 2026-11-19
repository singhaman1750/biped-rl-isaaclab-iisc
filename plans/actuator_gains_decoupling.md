# Decoupling the joint gains from the actuator catalogue

Status, NOT IMPLEMENTED, written 2026-11-19 for review. The change spans the submodule `design_generator` and the package `co_optimisation`, the submodule to be committed first and its pointer staged in the parent afterwards. The defect it removes is established in [../context/leg_collision_analysis.md](../context/leg_collision_analysis.md), and the line numbers below are those of the working tree on 2026-11-19.

## Context

A joint's position loop gains, its stiffness and damping, belong to the controller that drives the joint and not to the actuator that sits in it. The design generator nevertheless takes them from the catalogue entry a design picks. `CatalogActuator.sim_params` emits the entry's `stiffness` and `damping` beside its torque envelope, armature and friction (`../design_generator/common/actuator_catalog.py:545-576`), `is_sim_complete` refuses an entry that does not state them (`:282-297`), and `DesignGeneratorBase._actuator_params` copies them into the actuator parameters of the design, multiplying them by a gain scale when one is searched (`../co_optimisation/co_optimisation/runners/usd_generator.py:887-898`). The runner writes the result into every environment through `apply_actuator_params` (`../co_optimisation/co_optimisation/utils/respawn.py:246-305`), which reads the dictionary key by key and needs no change.

The consequence is that picking an actuator silently picks a controller. Every entry of the biped catalogue states a stiffness and a damping (`../design_generator/biped/catalogs/actuators.json`), the identified entries `TRON1-ABAD`, `TRON1-HIP`, `TRON1-KNEE` and `TRON1-ANKLE` carrying the gains of their own joints and the other entries the gains of the identified entry nearest in torque, speed and mass, copied by `scripts/tools/fill_actuator_sim_params.py:19`. A design that picks `TRON1-ABAD` for the ankle therefore receives the abduction joint's stiffness of 55 and damping of 13.5 at a joint whose own values are 10 and 0.5. In the run `copt_moral/2026-11-18_05-48-29` that is the whole of the collapse. The 29 of 100 generation 2 designs that did so ended 6556 of the 6590 `leg_contact` terminations of the replay in episodes of 6.6 steps, and exchanging the ankle gains between `TRON1-ABAD` and `Bota-Roboticer-50` moved the failure to the latter, whose housing is as wide, so the housing was not the cause (sections 3 and 4 of the analysis).

The gain scales that the search may already carry do not remove the defect, because they multiply the catalogue value of the pick and so inherit it. `GAIN_PARAM_KEYS` names eight of them (`usd_generator.py:510-512`) and `_actuator_params` applies a named one as `params[key] *= parameters[scale_key]` (`:893-896`), against the pick's gain and not the joint's.

The intended outcome is that the gains of each joint group are fields of `class DesignSpec`, defaulting to the values of the baseline robot of each family, set from `variable_params.json`, scaled by the gain scales exactly as the link lengths are scaled by their own, and written to the simulator through the unchanged application path. A design's pick then decides the torque envelope, armature, friction, mass and housing, and its gains are the baseline's unless a gain scale is searched.

## Evidence for the baseline values

The baseline gains of each family are the identified configurations the environments train and play with, which the baseline picks of `variable_params.json` carry in the catalogue and which agree with them to the digit.

| Family | Group | Baseline pick | Stiffness | Damping | Environment configuration |
|---|---|---|---|---|---|
| biped | abad | `TRON1-ABAD` | 55.0 | 13.5 | `../environments/environments/assets/config/solefoot_identified_cfg.py:19-20` |
| biped | hip | `TRON1-HIP` | 80.0 | 13.0 | `solefoot_identified_cfg.py:33-34` |
| biped | knee | `TRON1-KNEE` | 60.0 | 4.0 | `solefoot_identified_cfg.py:47-48` |
| biped | ankle | `TRON1-ANKLE` | 10.0 | 0.5 | `solefoot_identified_cfg.py:61-62` |
| quadruped | abad | `A1-ABAD` | 40.0 | 1.2 | `../environments/environments/assets/config/quadruped_identified_cfg.py:26-27` |
| quadruped | hip | `A1-HIP` | 40.0 | 1.1 | `quadruped_identified_cfg.py:43-44` |
| quadruped | knee | `A1-KNEE` | 50.0 | 0.9 | `quadruped_identified_cfg.py:61-62` |

The test `test_identified_entries_reproduce_their_configuration` already asserts the same figures against the catalogue (`../co_optimisation/tests/design_generator_test.py:600-636`).

## Approach

Five decisions govern the change, each chosen as the smallest measure that keeps every existing caller unaltered, in the order that the project's rules for code changes require.

1. The gains become fixed constant fields of `class DesignSpec`, not free fields. The free fields are those of `vector()`, `DesignBounds` and `from_vector()`, which index them by position, and the gains are searched through the gain scales and not through that vector, exactly as the pose offsets `thigh_bend_offset` and `knee_bend_offset` are. A fixed constant field with a Python default needs no change to `vector()`, `labels()`, `sample()` or `from_vector()`, and a `DesignSpec.from_dict` of a dictionary that names none, as every stored specification and test fixture does, receives the baseline's gains.
2. The scales reach the fields through a second dictionary, `GAIN_SPEC_FIELDS`, beside `SPEC_PARAM_FIELDS`, and `_spec_from_parameters` reads both. This departs from the instruction to expand `SPEC_PARAM_FIELDS` itself, for a reason the callers show. `SPEC_PARAM_FIELDS` is read as the set of geometry keys in four places that must keep that meaning.

| Reader | Line | What it does with the dictionary | Effect of adding the eight gain keys |
|---|---|---|---|
| `CMAESDesignGenerator.__init__` | `usd_generator.py:1092` | `dict.fromkeys(SPEC_PARAM_FIELDS)` names the four geometry keys searched when a run names no ranges | A run naming no ranges would search twelve dimensions, which changes `play.py` and every default caller |
| `CMAESDesignGenerator.__init__` | `usd_generator.py:1094` | The group check applies to the keys not in the dictionary | A gain key would skip the check, so `ankle_stiffness_scale` on the quadruped would no longer raise |
| `base_record` | `usd_generator.py:713` | Initial value of each geometry key named in the ranges | Harmless |
| `co_optimisation/tests/design_generator_test.py` | `:230`, `:242`, `:264`, `:292` | Asserts `param_keys == list(SPEC_PARAM_FIELDS)` and the geometry draws | Fails on the first, and breaks the meaning of the others |

   `GAIN_SPEC_FIELDS` leaves the dictionary a geometry dictionary and gives the gain keys one place that names their fields. If the review prefers the literal expansion, the two lines of `CMAESDesignGenerator.__init__` and the four test lines of the table are the ones to restrict to the geometry keys.
3. The gain scales are not rounded. `_spec_from_parameters` rounds a scaled length to two decimals, the quantisation of the URDF era generator (`usd_generator.py:873-876`), which would give a damping of 0.5 only twenty one levels across the scale range 0.8 to 1.2. The gain branch applies the scale or the offset without rounding, and `_actuator_params` already rounds the stored value to six decimals.
4. A gain field is applied only where the family's specification has it. The quadruped has no ankle, `RandomDesignGenerator` samples every key of `DEFAULT_PARAM_RANGES` whether named or not (`usd_generator.py:839-840`, recorded as defect 1 of [gain_scale_search.md](gain_scale_search.md)), and `DesignSpec.from_dict` raises on an unknown field, so the branch tests `field in fields` and the quadruped's inert ankle scales stay inert, as they are today.
5. `_actuator_params` reads the gains from the specification and stops multiplying them by the scale. The scale is applied once, in the specification, and applying it a second time in `_actuator_params` would square it. The `parameters` argument stays in the signature, ignored, so that no caller changes, and the call that omits the parameters (`usd_generator.py:1672`) becomes correct beside the sites that pass them, none needing an edit. This removes the second defect that [gain_scale_search.md](gain_scale_search.md) recorded, that a generator not passing its parameters logged gains it had not applied.

The catalogue is cleaned in the same pass but as a separate step, so that the behavioural change and the data removal can be reviewed and reverted apart.

## Callers

The rule of the project that every caller of a shared function be enumerated before it changes is met here for the two functions whose behaviour changes, `_spec_from_parameters` and `_actuator_params`, and for `DesignSpec`.

| Function | Callers | Behaviour with a baseline pick | Behaviour with another pick |
|---|---|---|---|
| `_spec_from_parameters` | `describe_design` `:701`, `base_record` `:716`, `emit_design` `:734`, `RandomDesignGenerator` `:985`, `FixedDesignGenerator` `:1013`, `_mean_population` `:1229`, `CMAESDesignGenerator` `:1250`, `GrowingDesignDistCMAESDesignGenerator` `:1317`, `CatCMAESDesignGenerator` `:1628`, `:1671` | Unchanged, a gain scale not named leaves the field at the baseline | Unchanged |
| `_actuator_params` | the same sites, and `_individual` `:772` | Identical to the catalogue's values, which equal the baseline | The gains are the baseline's, where they were the pick's, the intended change |
| `DesignSpec` | `load_design_spec`, `from_dict`, `to_dict`, `family.nominal`, `family.chain`, `solve_nominal_pose`, `build_xml` | Unchanged, no reader touches a gain field, and `build_xml` never reads one | Unchanged |

`apply_actuator_params`, `Population`, `_population_to_dict` and the checkpoint carry only the dictionary of actuator parameters, so a checkpoint written before the change resumes unchanged and a population restored from it keeps the gains it was trained under.

## Changes in the submodule `design_generator`

### 1. `biped/generate.py`, the fields of `class DesignSpec`

The eight fields follow the fixed constant tier, before the nominal pose offsets, with the baseline's values as defaults. The quadruped's six are the same without the ankle.

```python
    clearance_ankleAct2foot_z: float = 0.03   # must stay positive -- foot sits below the ankle joint

    # Position loop gains of each joint group, N*m/rad and N*m*s/rad. Joint properties, not the
    # picked actuator's, searched through co_optimisation's gain scales. Defaults are the baseline's.
    abad_stiffness: float = 55.0
    abad_damping: float = 13.5
    hip_stiffness: float = 80.0
    hip_damping: float = 13.0
    knee_stiffness: float = 60.0
    knee_damping: float = 4.0
    ankle_stiffness: float = 10.0
    ankle_damping: float = 0.5

    # Nominal pose offsets, radians about +y, added to the constants' fold in the
```

### 2. `quadruped/generate.py`, the fields of `class DesignSpec`

```python
    clearance_shank2foot_z: float = 0.0

    # Position loop gains of each joint group, N*m/rad and N*m*s/rad. Joint properties, not the
    # picked actuator's, searched through co_optimisation's gain scales. Defaults are the baseline's.
    abad_stiffness: float = 40.0
    abad_damping: float = 1.2
    hip_stiffness: float = 40.0
    hip_damping: float = 1.1
    knee_stiffness: float = 50.0
    knee_damping: float = 0.9

    # Nominal pose offsets, radians about +y, added to the constants' fold in the
```

`from_dict`, `to_dict` and `fields()` read every dataclass field, so neither needs an edit. `vector()`, `from_vector()`, `labels()` and `DesignBounds` list their free fields explicitly and do not reach the new ones.

### 3. `biped/variable_params.json` and `quadruped/variable_params.json`

The keys are added after the last clearance and before `body_height`, with the baseline's values, so that the file states what the dataclass defaults to and a reader of the file sees the gains beside the picks they no longer follow.

```json
  "clearance_ankleAct2foot_z": -0.01,
  "abad_stiffness": 55.0,
  "abad_damping": 13.5,
  "hip_stiffness": 80.0,
  "hip_damping": 13.0,
  "knee_stiffness": 60.0,
  "knee_damping": 4.0,
  "ankle_stiffness": 10.0,
  "ankle_damping": 0.5,
  "body_height": 0.80435432
```

```json
  "clearance_shank2foot_z": 0.0,
  "abad_stiffness": 40.0,
  "abad_damping": 1.2,
  "hip_stiffness": 40.0,
  "hip_damping": 1.1,
  "knee_stiffness": 50.0,
  "knee_damping": 0.9,
  "body_height": 0.35
```

The quadruped's block goes after `clearance_shank2foot_z` at line 24 of its file, and the biped's after `clearance_ankleAct2foot_z` at line 29. The install order in the rollout section matters, because a file carrying these keys is refused by a dataclass that lacks the fields (`from_dict` raises `KeyError` on an unknown field).

### 4. `common/actuator_catalog.py`, what a complete entry states

The torque envelope, armature and friction stay actuator properties, and the gains are named apart. `SIM_PARAM_KEYS` keeps its full content and its meaning as the keys `apply_actuator_params` receives, so every consumer of it is untouched.

```python
ACTUATOR_SIM_KEYS = (
    "effort_limit", "velocity_limit", "saturation_effort", "armature",
    "friction_static", "friction_dynamic", "activation_vel",
)
GAIN_KEYS = ("stiffness", "damping")
SIM_PARAM_KEYS = ACTUATOR_SIM_KEYS + GAIN_KEYS
```

```python
def is_sim_complete(entry) -> bool:
    """True when ``sim_params()`` states every ACTUATOR_SIM_KEYS key and a nonzero armature.

    The gains are a joint's, not the actuator's, and DesignSpec carries them, so an entry that
    states none is complete.
    """
    params = entry.sim_params()
    return set(params) >= set(ACTUATOR_SIM_KEYS) and params["armature"] > 0.0
```

Today every entry of both catalogues states its gains, so the shelf `sim_actuator_catalog` returns is the same list of 15 biped and 14 quadruped entries, and the categorical search's `cat_names` do not move (`co_optimisation/tests/design_generator_test.py:660-672`). A COMPAct composed entry, which states no friction (`actuator_catalog.py:545-556`), is still refused.

### 5. The catalogues, phase two

With the gains read from the specification, the `stiffness` and `damping` of the 29 catalogue entries are dead data that a reader would take for live. They are removed from `biped/catalogs/actuators.json` and `quadruped/catalogs/actuators.json`, together with their lines in the `units` and `fields` lists of the schema and the sentence of each `notes` that says the gains were copied. The loader reads `_SIM_JSON_KEYS` only where present (`actuator_catalog.py:608-612`), so it needs no change and a user supplied catalogue that still states gains keeps loading, the gains being ignored by `_actuator_params`.

`scripts/tools/fill_actuator_sim_params.py:19` drops the two gains from `COPIED`, and the note it appends at `:50-52` stops naming them. The script is run once and kept for provenance, so the edit is to keep a rerun from restoring them.

The step is separable. Landing sections 1 to 4 and the `usd_generator.py` changes below without it is correct and changes no value, the JSON gains being unread, and the removal can follow once the first run under the new rule confirms the gains in its logs.

## Changes in the package `co_optimisation`

### 6. `runners/usd_generator.py`, `GAIN_SPEC_FIELDS`

After `SPEC_PARAM_FIELDS` (`:523-530`).

```python
GAIN_SPEC_FIELDS: dict[str, str] = {
    f"{group}_{key}_scale": f"{group}_{key}"
    for group in ("abad", "hip", "knee", "ankle") for key in ("stiffness", "damping")
}
```

The dictionary's order is the order of `GAIN_PARAM_KEYS` (`:510-512`), which a test asserts, so the search vector's order is that of the file as before. The comment above `SPEC_PARAM_FIELDS`, which says the gain scales are "consumed by _actuator_params instead", is corrected to say they are consumed by `_spec_from_parameters` through `GAIN_SPEC_FIELDS`.

### 7. `runners/usd_generator.py`, `_spec_from_parameters`, `:864-885`

```python
        fields = self.baseline.to_dict()
        for name, field in SPEC_PARAM_FIELDS.items():
            if name in parameters:
                kind, value = self.param_ranges[name][0], parameters[name]
                fields[field] = round(fields[field] * value, 2) if kind == "scale" else fields[field] + value
        for name, field in GAIN_SPEC_FIELDS.items():
            if name in parameters and field in fields:
                kind, value = self.param_ranges[name][0], parameters[name]
                fields[field] = fields[field] * value if kind == "scale" else fields[field] + value
        for group, name in (picks or {}).items():
```

### 8. `runners/usd_generator.py`, `_actuator_params`, `:887-898`

```python
    def _actuator_params(self, spec, parameters: dict[str, float] | None = None) -> dict[str, dict]:
        """Simulator-side actuator parameters for every group of one design.

        The torque envelope, armature and friction are the picked actuator's, and the gains are the
        specification's, a joint property the gain scales have already been applied to by
        _spec_from_parameters. *parameters* is read no more and stays so that no caller changes.
        """
        out: dict[str, dict] = {}
        for group, actuator in zip(self.family.groups, spec.actuators()):
            params = actuator.sim_params()
            params.update({key: getattr(spec, f"{group}_{key}") for key in GAIN_KEYS})
            out[group] = {k: round(float(v), 6) for k, v in params.items()}
        return out
```

`GAIN_KEYS` is imported from `design_generator.common.actuator_catalog`, beside the `ACTUATOR_SPECS` the module already takes from it. The dictionary's key order differs from before only after phase two, when `stiffness` and `damping` are appended and no longer sit between the friction keys, which moves two columns of the design records and changes no name (`usd_generator.py:667-682` keys them by name).

### 9. Documentation of the module

The module docstring (`usd_generator.py:27-35`) says the gain scales are "applied post-spawn via `apply_actuator_params`". It is revised to say they scale the baseline's gain field of the specification, that the picked actuator no longer decides the gain, and that the ankle pair is the biped's alone. The `FixedDesignGenerator` docstring (`:996-1001`), which says gain scales among the parameters are applied to the actuators, is revised the same way, and `config/cmaes_cfg.py:46-49`, whose `param_ranges` text states the scales are applied to the actuators, likewise.

## What changes for a run, and what does not

A run that names no gain scale, the case of `copt_moral/2026-11-18_05-48-29` (`params/copt_agent.yaml` names the four geometry keys alone), now trains every design on the baseline's gains whatever it picks. Generation 0 is unchanged, since every design picks the baseline's actuators, and so is `base_record`, which describes the baseline. From the first generation that varies a pick, the abduction pick `TMotor-AK10-9` no longer changes the abduction gains from 55 and 13.5, and the 29 designs that pick `TRON1-ABAD` at the ankle receive 10 and 0.5. That is the single intended change of behaviour, and it is a change only for designs whose gains the pick had been changing.

A run that names gain scales, which only `design_search: actuator` runs of `scripts/rsl_rl/train.py:276` and a hand written `param_ranges` can, now scales the baseline's gain and not the pick's. For a baseline pick the two are equal. The replay path needs no edit, because `FixedDesignGenerator` reaches the fields through the same `_spec_from_parameters`, and `play.py:291` already reads `CMAES_PARAM_KEYS` from `final_design.csv`.

A checkpoint written before the change resumes with its stored actuator parameters, which carry the old gains, until the next respawn, after which the new rule applies. Resuming the run of 2026-11-18 from `model_799.pt` is the case that matters, since iterations 800 to 1199 trained a third of the environments on episodes of 6.6 steps. It trains the policy of generation 1 on gains that differ from what it saw only where its own picks had altered them.

## Defects left standing, to be recorded

1. `DEFAULT_PARAM_RANGES` keeps the gain scale ranges derived from the biped's runtime randomisation (`usd_generator.py:28-35`, `:471-478`), so a quadruped run that names gain scales should state its own.
2. `RandomDesignGenerator` still samples every key of `DEFAULT_PARAM_RANGES` whether named or not (`:839-840`), the ankle pair now being dropped by the `field in fields` test and not by being inert in `_actuator_params`.
3. The startup `randomize_actuator_gains` events remain overwritten by `apply_actuator_params` at the first generation, as `w_eta_optimisation.md` records.
4. `plans/w_eta_optimisation.md` (NOT IMPLEMENTED) computes the gains from a natural frequency and a damping ratio per group and would write them where this plan does, into the specification, so the two plans compose and the later needs its section on `_actuator_params` rebased on section 8 here.
5. The ankle gains of a stiff ankle were not separated into stiffness and damping by the replay (section 4 of the analysis), so a gain scale range that reaches 5.5 times the ankle baseline is untested, and the default ranges of 0.8 to 1.2 do not.

## Tests

The changes are in `co_optimisation/tests/design_generator_test.py`, `catalog_fill_test.py` and `design_filter_test.py`, which stub the converter and need no simulator.

1. A regression guard that records the old output as literals. For each family, `_actuator_params` of the baseline specification equals the dictionary the catalogue produced before the change, key for key, the gains being the table of the evidence section. This fails if a default drifts from the identified configuration.
2. Independence. For each family, each group and every entry of `family.catalog()`, a design that picks the entry for that group and names no gain scale has `act[group]["stiffness"]` and `["damping"]` equal to the baseline's and its other keys equal to `entry.sim_params()`. The 29 failing designs of the analysis are this case for the ankle.
3. The scale applies once. With `hip_stiffness_scale` named and drawn as 1.1, `act["hip"]["stiffness"] == round(baseline_hip * 1.1, 6)` for a baseline pick and for a non baseline pick alike, and `spec.hip_stiffness` equals the same product, so a second multiplication would fail it.
4. The gain is not rounded to two decimals. A `knee_damping_scale` of 1.037 gives `spec.knee_damping == 4.0 * 1.037` to float tolerance.
5. The quadruped tolerates the ankle keys. `RandomDesignGenerator` on the quadruped, which samples `ankle_stiffness_scale`, builds its population without raising, and `CMAESDesignGenerator` still raises `ValueError` on `ankle_stiffness_scale` named for the quadruped.
6. `tuple(GAIN_SPEC_FIELDS) == GAIN_PARAM_KEYS`, and every value of `GAIN_SPEC_FIELDS` is a field of the biped's `DesignSpec`, the quadruped's lacking the two ankle names alone.
7. The variable parameter files state the defaults. `load_design_spec()` of each family equals `DesignSpec(**json)` and every gain field equals the dataclass default, so the three places that state a gain cannot disagree.
8. Existing tests. `test_named_gain_scales_are_searched_and_applied` and `test_no_named_gain_scale_keeps_the_catalogue_gains_and_the_geometry_draws` (`design_generator_test.py:259-295`) compute their expectation from `_catalogue_params`, the catalogue's gains times the scale, and are rewritten to take the baseline's gains from the specification, which is where the new rule reads them. `test_identified_entries_reproduce_their_configuration` (`:629-636`) asserts `set(got) == set(want)` for `sim_params()` and loses the two gain keys in phase two, as does `test_unstated_simulator_fields_fall_back_rather_than_fabricate` (`:639-650`) its `stiffness` and `damping` in the omitted set. `test_the_shelf_is_the_simulation_complete_catalogue` (`:660-672`) asserts `set(params) >= set(SIM_PARAM_KEYS)` and is changed to `ACTUATOR_SIM_KEYS`. `catalog_fill_test.py:47-79` follows the same two edits.
9. `sim_filter_data_check.py:72`, which compares each entry's `sim_params()` with the logged record, is a simulator check and is run in the verification step below.

## Verification

1. `python -m pytest co_optimisation/tests/design_generator_test.py co_optimisation/tests/design_filter_test.py co_optimisation/tests/catalog_fill_test.py -q` in the container, after installing the submodule and then the package.
2. The replay that found the defect, `scripts/analysis/leg_contact_by_design.py` over `design_parameters_1199.csv` with `model_1199.pt`, run again on the changed tree. The 29 designs that pick `TRON1-ABAD` for the ankle are rebuilt with the gains of the specification, so their `ankle_actuator_stiffness` column reads 10.0 and `ankle_actuator_damping` 0.5, and their leg contact terminations per design fall from a mean of 226 towards the 0.5 of the other ankle picks. The swap experiment of the analysis predicts this, since it gave `TRON1-ABAD` the ankle's gains and its terminations fell to 0.8 per design. Its mean episode was 439 steps and not the 762 of the other `TRON1-ANKLE` designs, so a residual instability of the heavier housing is to be expected and is not a failure of this change.
3. One simulator check of the application. After `apply_actuator_params` in a small run that reaches generation 1, `articulation.actuators["ankle"].stiffness[env]` equals 10.0 for every environment whatever its ankle pick, and `articulation.actuators["hip"].stiffness[env]` equals 80.0, against the pick's gain before.
4. A run naming `hip_stiffness_scale`, with a small `--num_envs` and a low `ea_late_start`, asserting that `articulation.actuators["hip"].stiffness[env]` equals `80.0 * parameters["hip_stiffness_scale"]` for a sample of environments and that `params/cmaes.yaml` and `designs/` record the scale.

## Rollout

1. Apply sections 1 to 4 in `design_generator`, run its own checks, commit inside the submodule, then stage the pointer in the parent with sections 6 to 8 and the tests.
2. `design_generator` and `co_optimisation` are installed copies in the production containers and not editable installs, so reinstall `design_generator` first and `co_optimisation` second in each container that trains or replays. A reinstalled package with an old `variable_params.json`, or the new file with an old dataclass, raises `KeyError` on the unknown field, which is the signal that the order was wrong. Neither container is to be reinstalled while a run trains in it.
3. Resume `copt_moral/2026-11-18_05-48-29` from `model_799.pt` only after the replay of the verification step shows the 29 designs recovered.
4. Apply phase two of section 5 after the first run under the change has been inspected.

## Documentation, in the same pass

1. A dated entry in `context/co_optimisation_audit.md` recording the defect, the decoupling, the defects left standing and the figures of the verification step.
2. `design_generator/biped/DESIGN_PARAMETERS.md`, `design_generator/quadruped/DESIGN_PARAMETERS.md` and the two `catalogs/README.md` files, stating that the gains are fields of the specification and no longer catalogue fields.
3. `CO_OPTIMISATION.md` at the workspace root, where the actuator parameters are described, and `plans/gain_scale_search.md`, whose context section describes the scale as acting on the catalogue value, a dated correction to be appended and not an edit in place.
4. `plans/README.md`, whose register carries this plan with its status, and `context/README.md` and `context/leg_collision_analysis.md`, which gain a pointer to this plan as the correction proposed in section 7 of the analysis, a better one than the options listed there since it removes the cause for every group at once.

## Decisions for the review

1. Whether `GAIN_SPEC_FIELDS` as a second dictionary is accepted, or `SPEC_PARAM_FIELDS` is expanded literally and its four readers restricted, section Approach 2.
2. Whether phase two, the removal of the gains from the catalogue files, lands with this change or after the first run.
3. Whether the gains are to be exposed as free fields of `DesignSpec` later, so that an external optimiser's `vector()` carries them. The plan keeps them out, as the pose offsets are, and the gain scales stay the only search path.
