# Decoupling the joint gains from the actuator catalogue

Status, IMPLEMENTED 2026-11-19 in the working tree, uncommitted, with the quantisation of the gains and of the actuator parameters removed at the author's direction after review, see sections 7 and 8. The suites `design_generator_test.py`, `design_filter_test.py` and `catalog_fill_test.py` pass 178 with 14 skipped, and the simulator check of the verification section is recorded in the outcome section at the end. Written the same day and revised after review, and revised again after the outcome section was written, so that a gain a design does not search is left out of the actuator parameters and keeps the environment's startup randomisation, see the revision section at the end. The change spans the submodule `design_generator` and the package `co_optimisation`, the submodule to be committed first and its pointer staged in the parent afterwards. The defect it removes is established in [../context/leg_collision_analysis.md](../context/leg_collision_analysis.md), and the line numbers below are those of the working tree on 2026-11-19. The revision removed every mention of stiffness and damping from the catalogue code, folded the gain scales into `SPEC_PARAM_FIELDS` with no second dictionary, and made `_actuator_params` read the gains from the specification.

## Context

A joint's position loop gains, its stiffness and damping, belong to the controller that drives the joint and not to the actuator that sits in it. The design generator nevertheless takes them from the catalogue entry a design picks. `CatalogActuator.sim_params` emits the entry's `stiffness` and `damping` beside its torque envelope, armature and friction (`../design_generator/common/actuator_catalog.py:545-571`), `SIM_PARAM_KEYS` names them and `is_sim_complete` refuses an entry that does not state them (`:282-297`), and `DesignGeneratorBase._actuator_params` copies them into the actuator parameters of the design, multiplying them by a gain scale when one is searched (`../co_optimisation/co_optimisation/runners/usd_generator.py:887-898`). The runner writes the result into every environment through `apply_actuator_params` (`../co_optimisation/co_optimisation/utils/respawn.py:246-305`), which reads the dictionary key by key and needs no change.

The consequence is that picking an actuator silently picks a controller. Every entry of the biped catalogue states a stiffness and a damping (`../design_generator/biped/catalogs/actuators.json`), the identified entries `TRON1-ABAD`, `TRON1-HIP`, `TRON1-KNEE` and `TRON1-ANKLE` carrying the gains of their own joints and the other entries the gains of the identified entry nearest in torque, speed and mass, copied by `scripts/tools/fill_actuator_sim_params.py:19`. A design that picks `TRON1-ABAD` for the ankle therefore receives the abduction joint's stiffness of 55 and damping of 13.5 at a joint whose own values are 10 and 0.5. In the run `copt_moral/2026-11-18_05-48-29` that is the whole of the collapse. The 29 of 100 generation 2 designs that did so ended 6556 of the 6590 `leg_contact` terminations of the replay in episodes of 6.6 steps, and exchanging the ankle gains between `TRON1-ABAD` and `Bota-Roboticer-50` moved the failure to the latter, whose housing is as wide, so the housing was not the cause (sections 3 and 4 of the analysis).

The gain scales that the search may already carry do not remove the defect, because they multiply the catalogue value of the pick and so inherit it. `GAIN_PARAM_KEYS` names eight of them (`usd_generator.py:510-512`) and `_actuator_params` applies a named one as `params[key] *= parameters[scale_key]` (`:893-896`), against the pick's gain and not the joint's.

The intended outcome is that the gains of each joint group are fields of `class DesignSpec`, defaulting to the values of the baseline robot of each family, set from `variable_params.json`, searched and scaled by the same pathway as the link lengths, and written to the simulator through the unchanged application path. The catalogue then knows nothing of them. A design's pick decides the torque envelope, armature, friction, mass and housing, and its gains are the baseline's unless a gain scale is searched.

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

The test `test_identified_entries_reproduce_their_configuration` asserts the same figures against the catalogue today (`../co_optimisation/tests/design_generator_test.py:600-636`) and is the source of the literals that the new default test keeps.

## Approach

Five decisions govern the change.

1. The gains become fixed constant fields of `class DesignSpec`, not free fields. The free fields are those of `vector()`, `DesignBounds` and `from_vector()`, which index them by position, and the gains are searched through the gain scales and not through that vector, exactly as the pose offsets `thigh_bend_offset` and `knee_bend_offset` are. A fixed constant field with a Python default needs no change to `vector()`, `labels()`, `sample()` or `from_vector()`, and a `DesignSpec.from_dict` of a dictionary that names none, as every stored specification and test fixture does, receives the baseline's gains.
2. The gain scales join `SPEC_PARAM_FIELDS` and are treated as every other searched parameter. The design generator does not distinguish a geometry value from a gain, `_spec_from_parameters` applies each scale to its field in one loop, and the distinction is made only where the specification is turned into actuator parameters for the simulator, in `_actuator_params`. No second dictionary is introduced.
3. Folding the scales into the dictionary changes two of its readers, which the review judged to be wrong in their present form and which are repaired and not worked around. The table gives each reader and its disposition.

| Reader | Line | Before | After |
|---|---|---|---|
| `CMAESDesignGenerator.__init__`, the default | `usd_generator.py:1092` | `dict.fromkeys(SPEC_PARAM_FIELDS)` names the four geometry keys searched when a run names no ranges | Must keep naming the four, or a run naming no ranges, `play.py`'s replay generator and every default caller would search twelve dimensions, see section 6 |
| `CMAESDesignGenerator.__init__`, the check | `usd_generator.py:1094-1096` | Refuses a key outside `SPEC_PARAM_FIELDS` whose prefix is no joint group | Refuses a key whose field the family's specification lacks, which is the same rule stated without a value type and which `ankle_stiffness_scale` on the quadruped still fails, see section 6 |
| `base_record` | `usd_generator.py:713` | Initial value of each geometry key named in the ranges | Also the gain scales named in them, at 1.0, harmless |
| `_spec_from_parameters` | `usd_generator.py:873-876` | Applies each geometry key, rounded to two decimals | Applies every key by the same line, a gain unrounded, see section 7 |
| `co_optimisation/tests/design_generator_test.py` | `:230`, `:242`, `:264`, `:292` | Take `SPEC_PARAM_FIELDS` to be the four geometry keys | Take the geometry keys from it explicitly, see the tests |

4. A field is applied only where the family's specification has it. The quadruped has no ankle, `RandomDesignGenerator` samples every key of `DEFAULT_PARAM_RANGES` whether named or not (`usd_generator.py:839-840`, recorded as defect 1 of [gain_scale_search.md](gain_scale_search.md)), and `DesignSpec.from_dict` raises on an unknown field, so the loop in `_spec_from_parameters` skips a key whose field is absent from the baseline's dictionary, as the ankle pair is inert on the quadruped today.
5. `_actuator_params` reads a gain from the specification and stops multiplying it by the scale, and writes it only where `parameters` names the scale. The scale is applied once, in the specification, and applying it a second time in `_actuator_params` would square it. A gain that is not searched is left out of the dictionary, and so is left to the environment, see the revision section.

The catalogue is cleaned in the same change, and the gains leave its code and its data together.

## Callers

The rule of the project that every caller of a shared function be enumerated before it changes is met here for the two functions whose behaviour changes, `_spec_from_parameters` and `_actuator_params`, and for `DesignSpec`.

| Function | Callers | Behaviour with a baseline pick | Behaviour with another pick |
|---|---|---|---|
| `_spec_from_parameters` | `describe_design` `:701`, `base_record` `:716`, `emit_design` `:734`, `RandomDesignGenerator` `:985`, `FixedDesignGenerator` `:1013`, `_mean_population` `:1229`, `CMAESDesignGenerator` `:1250`, `GrowingDesignDistCMAESDesignGenerator` `:1317`, `CatCMAESDesignGenerator` `:1628`, `:1671` | Unchanged, a gain scale not named leaves its field at the baseline | Unchanged |
| `_actuator_params` | the same sites, and `_individual` `:772` | Identical to the catalogue's values, which equal the baseline | The gains are the baseline's, where they were the pick's, the intended change |
| `DesignSpec` | `load_design_spec`, `from_dict`, `to_dict`, `family.nominal`, `family.chain`, `solve_nominal_pose`, `build_xml` | Unchanged, no reader touches a gain field, and `build_xml` never reads one | Unchanged |
| `SIM_PARAM_KEYS` | `is_sim_complete`, `tests/design_generator_test.py`, `tests/catalog_fill_test.py` | The shelf is the same 15 and 14 entries | Unchanged |

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

The keys are added after the last clearance and before `body_height`, with the baseline's values, so that the file states what the dataclass defaults to and a reader of the file sees the gains beside the picks they no longer follow. The biped's block goes after `clearance_ankleAct2foot_z` at line 29 and the quadruped's after `clearance_shank2foot_z` at line 24.

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

The install order in the rollout section matters, because a file carrying these keys is refused by a dataclass that lacks the fields, `from_dict` raising `KeyError` on an unknown field.

### 4. `common/actuator_catalog.py`, the gains leave the catalogue code

No new tuple is introduced. `SIM_PARAM_KEYS` is the tuple that `is_sim_complete` reads (`:289-297`) and the tests import, so it stays, and it loses the two names, after which `is_sim_complete` needs no edit and requires exactly what an actuator can state.

```python
# Keys of the dict `sim_params` returns, the attribute names of isaaclab.actuators.DCMotor and
# environments.actuators.IdentifiedActuator that an actuator decides. The position loop gains are
# a joint's, carried by DesignSpec, and are not stated here.
SIM_PARAM_KEYS = (
    "effort_limit", "velocity_limit", "saturation_effort", "armature",
    "friction_static", "friction_dynamic", "activation_vel",
)
```

The remaining mentions are removed.

1. The comment of `:263-269` says every simulator needs "the position-loop gains the controller runs at" of an actuator, and is reworded without them.
2. `class CatalogActuator` loses its `stiffness` and `damping` fields (`:525-526`) and the pair from the tuple of `sim_params` (`:565-566`), and the docstring of `sim_params` (`:548-555`) stops saying that friction and the two gains are omitted when unstated, saying friction alone.
3. `_SIM_JSON_KEYS` (`:608-612`) loses the two names, so the loader reads no gain from a JSON entry and a user supplied catalogue that still states them loads unchanged, the gains being ignored.
4. `Actuator.sim_params` (`:388-403`) says friction and the position-loop gains have no counterpart in the published data, and is reworded to friction alone.

### 5. The catalogue data and its tool

The `stiffness` and `damping` of the 29 entries are removed from `biped/catalogs/actuators.json` and `quadruped/catalogs/actuators.json`, together with their lines in the `units` and `fields` lists of the schema (`biped/catalogs/actuators.json:25-26`, `:49-50`), the clause of each `notes` that says the gains were copied (`ODrive-D5065-DD`, `:74`, among the others), and the sentence of `_schema.simulation` that names them. `scripts/tools/fill_actuator_sim_params.py:19` drops the two names from `COPIED`, and the note it appends at `:50-52` stops naming them, so that a rerun of the tool does not restore them. `biped/catalogs/README.md` and `quadruped/catalogs/README.md` are revised to say that the gains are fields of the specification.

## Changes in the package `co_optimisation`

### 6. `runners/usd_generator.py`, the dictionary and the search restriction

`SPEC_PARAM_FIELDS` (`:523-530`) gains the eight gain scales, in the order of `GAIN_PARAM_KEYS` (`:510-512`) and so of `CMAES_PARAM_KEYS`, and its comment is corrected, the scales being consumed no longer by `_actuator_params` alone.

```python
# The DesignSpec field each searched parameter writes, a length, a pose offset or the gain of a joint
# group. _spec_from_parameters applies each, and _actuator_params hands the gains to the simulator.
SPEC_PARAM_FIELDS: dict[str, str] = {
    "thigh_length_scale": "thigh_length",
    "shank_length_scale": "shank_length",
    "thigh_bend_offset": "thigh_bend_offset",
    "knee_bend_offset": "knee_bend_offset",
    "abad_stiffness_scale": "abad_stiffness",
    "abad_damping_scale": "abad_damping",
    "hip_stiffness_scale": "hip_stiffness",
    "hip_damping_scale": "hip_damping",
    "knee_stiffness_scale": "knee_stiffness",
    "knee_damping_scale": "knee_damping",
    "ankle_stiffness_scale": "ankle_stiffness",
    "ankle_damping_scale": "ankle_damping",
}
```

`CMAESDesignGenerator.__init__` (`:1092-1096`) reads the dictionary in two places that change.

```python
        named = param_ranges or dict.fromkeys(k for k in SPEC_PARAM_FIELDS if k not in GAIN_PARAM_KEYS)
        self.param_ranges = {k: self.param_ranges[k] for k in CMAES_PARAM_KEYS if k in named}
        spec_fields = self.baseline.to_dict()
        for key in self.param_ranges:
            if SPEC_PARAM_FIELDS[key] not in spec_fields:
                raise ValueError(f"{key} names no field of the {family} specification")
```

The default of a run that names no ranges stays the four geometry keys, which keeps `play.py`'s search, the generators the tests build and every caller that names none searching what they searched, the exclusion reading the existing `GAIN_PARAM_KEYS` and introducing no constant. The check no longer reads a prefix. It asks of each searched key whether the family's specification has the field the key writes, which every geometry key and every gain key of the biped does, and which `ankle_stiffness_scale` and `ankle_damping_scale` do not on the quadruped, so that they still raise `ValueError`. `self.baseline` is set by `DesignGeneratorBase.__init__` (`:609`), which `CMAESDesignGenerator.__init__` calls first.

### 7. `runners/usd_generator.py`, `_spec_from_parameters`, `:864-885`

The loop is unchanged but for one condition, so that a key whose field the family lacks, the quadruped's ankle pair that `RandomDesignGenerator` samples, is skipped and not raised on.

```python
        fields = self.baseline.to_dict()
        for name, field in SPEC_PARAM_FIELDS.items():
            if name in parameters and field in fields:
                kind, value = self.param_ranges[name][0], parameters[name]
                scaled = fields[field] * value if kind == "scale" else fields[field] + value
                # A length is quantised to two decimals, a gain is not.
                fields[field] = round(scaled, 2) if kind == "scale" and name not in GAIN_PARAM_KEYS else scaled
        for group, name in (picks or {}).items():
```

A scaled gain is not rounded. The two decimal quantisation of a length is the URDF era generator's and would have moved a damping of 0.5 in steps of 2 percent, so the review removed it for the gains, the only place the line separates a gain from a length being the exemption of `GAIN_PARAM_KEYS`, and every length and offset keeps its rounding exactly as before.

### 8. `runners/usd_generator.py`, `_actuator_params`, `:887-898`

The gains are found among the searched parameters that the family's specification carries, by group, and written under the name the actuator gives them, where `parameters` names the scale. The iteration reads `GAIN_PARAM_KEYS`, which names the parameters that are gains, so a geometry field that happens to begin with a group name, `knee_bend_offset`, is never taken for one.

```python
    def _actuator_params(self, spec, parameters: dict[str, float] | None = None) -> dict[str, dict]:
        """Simulator-side actuator parameters for every group of one design.

        The torque envelope, armature and friction are the picked actuator's. A gain is the specification's
        field of the group, the scale already applied by _spec_from_parameters, where *parameters* names
        it, and is left out otherwise, so that the environment's own gains, randomised at startup, stay.
        """
        out: dict[str, dict] = {}
        for group, actuator in zip(self.family.groups, spec.actuators()):
            params = actuator.sim_params()
            for name in GAIN_PARAM_KEYS:
                field = SPEC_PARAM_FIELDS[name]
                if name in (parameters or {}) and field.startswith(f"{group}_") and hasattr(spec, field):
                    params[field.removeprefix(f"{group}_")] = getattr(spec, field)
            out[group] = {k: float(v) for k, v in params.items()}
        return out
```

The six decimal rounding that `_actuator_params` applied to every parameter is removed with the quantisation of the gains, so the dictionary carries the values as the specification and the catalogue state them. The dictionary's key order differs from before, `stiffness` and `damping` now following the friction keys of the catalogue and not sitting among them, which moves two columns of the design records and changes no name (`usd_generator.py:667-682` keys them by name).

### 9. Documentation of the module

The module docstring (`usd_generator.py:27-35`) says the gain scales are "applied post-spawn via `apply_actuator_params`". It is revised to say they scale the baseline's gain field of the specification, that the picked actuator no longer decides the gain, and that the ankle pair is the biped's alone. The `FixedDesignGenerator` docstring (`:996-1001`) and `config/cmaes_cfg.py:46-49`, which state that the scales are applied to the actuators, are revised the same way.

## What changes for a run, and what does not

A run that names no gain scale, the case of `copt_moral/2026-11-18_05-48-29` (`params/copt_agent.yaml` names the four geometry keys alone), now trains every environment on gains the startup randomisation drew about the baseline's, whatever its design picks, see the revision section. The base robot is described with the baseline's gains, and a design that picks `TRON1-ABAD` at the ankle holds in each environment an ankle stiffness within 8 and 12 and a damping within 0.4 and 0.6, never the abduction's 55 and 13.5. That is the single intended change of behaviour, and it is a change only for designs whose gains the pick had been changing.

A run that names gain scales, which only `design_search: actuator` runs of `scripts/rsl_rl/train.py:276` and a hand written `param_ranges` can, now scales the baseline's gain and not the pick's. For a baseline pick the two are equal. The replay path needs no edit, because `FixedDesignGenerator` reaches the fields through the same `_spec_from_parameters`, and `play.py:291` already reads `CMAES_PARAM_KEYS` from `final_design.csv`.

A checkpoint written before the change resumes with its stored actuator parameters, which carry the old gains, until the next respawn, after which the new rule applies. Resuming the run of 2026-11-18 from `model_799.pt` is the case that matters, since iterations 800 to 1199 trained a third of the environments on episodes of 6.6 steps. It trains the policy of generation 1 on gains that differ from what it saw only where its own picks had altered them.

An external catalogue that states gains is still read, and they are ignored. This is a change of behaviour for any such caller and is the point of the change.

## Defects left standing, to be recorded

1. `DEFAULT_PARAM_RANGES` keeps the gain scale ranges derived from the biped's runtime randomisation (`usd_generator.py:28-35`, `:471-478`), so a quadruped run that names gain scales should state its own.
2. `RandomDesignGenerator` still samples every key of `DEFAULT_PARAM_RANGES` whether named or not (`:839-840`), the quadruped's ankle pair now being dropped by the field test of section 7 and not by being inert in `_actuator_params`, and its gain scales, which are named in its parameters, scale the baseline's gains and not the pick's.
3. The two decimal rounding of a length remains, as the URDF era generator applied it, and a scaled gain is not rounded.
4. The startup `randomize_actuator_gains` events were overwritten by `apply_actuator_params` at the first generation, as `w_eta_optimisation.md` records, and now survive for every gain that is not searched, being applied again after each respawn, while a searched gain still replaces them.
5. `plans/w_eta_optimisation.md` (NOT IMPLEMENTED) computes the gains from a natural frequency and a damping ratio per group and would write them where this plan does, into the specification, so the two plans compose and the later needs its section on `_actuator_params` rebased on section 8 here.
6. The replay did not separate the ankle stiffness from the damping (section 4 of the analysis), so a gain scale range that reaches 5.5 times the ankle baseline is untested, and the default ranges of 0.8 to 1.2 do not.

## Tests

The changes are in `co_optimisation/tests/design_generator_test.py`, `catalog_fill_test.py` and `design_filter_test.py`, which stub the converter and need no simulator.

1. A regression guard that records the old output as literals. For each family, `_actuator_params` of the baseline specification equals the dictionary the catalogue produced before the change, key for key, the gains being the table of the evidence section. This fails if a default drifts from the identified configuration, and it replaces the gain figures that `IDENTIFIED` (`design_generator_test.py:600-625`) loses, which move into a `BASELINE_GAINS` literal.
2. Independence. For each family, each group and every entry of `family.catalog()`, a design that picks the entry for that group and names no gain scale has `act[group]["stiffness"]` and `["damping"]` equal to the baseline's and its other keys equal to `entry.sim_params()`. The 29 failing designs of the analysis are this case for the ankle.
3. The scale applies once. With `hip_stiffness_scale` named and drawn as 1.1, `act["hip"]["stiffness"] == baseline_hip * 1.1` for a baseline pick and for a non baseline pick alike, and `spec.hip_stiffness` equals the same value, so a second multiplication would fail it.
4. The quadruped tolerates the ankle keys. `RandomDesignGenerator` on the quadruped, which samples `ankle_stiffness_scale`, builds its population without raising and its specification has no ankle field, while `CMAESDesignGenerator` named with `ankle_stiffness_scale` on the quadruped raises `ValueError`, and the biped's does not.
5. `SPEC_PARAM_FIELDS` ends with the keys of `GAIN_PARAM_KEYS` in their order, `tuple(SPEC_PARAM_FIELDS) == CMAES_PARAM_KEYS`, and every value of it is a field of the biped's `DesignSpec`, the quadruped's lacking the two ankle names alone.
6. A generator given no ranges searches the four geometry keys and no gain, so `gen.param_keys` equals the geometry keys, which the tests take as `[k for k in U.SPEC_PARAM_FIELDS if k not in U.GAIN_PARAM_KEYS]`.
7. The variable parameter files state the defaults. `load_design_spec()` of each family equals `DesignSpec(**json)` and every gain field equals the dataclass default, so the three places that state a gain cannot disagree.
8. The catalogue states no gain. No entry of either family's `ACTUATOR_CATALOG` has an attribute `stiffness` or `damping`, no `sim_params()` has either key, and a JSON entry that states them still loads.
9. Existing tests that follow the code. `design_generator_test.py:230` and `:264` take `param_keys` against the geometry keys and the geometry keys plus the named gains, `:242` builds `GEOMETRY_AND_GAINS` from the geometry keys explicitly, `:292` compares the geometry draws over the geometry keys, since the gains are drawn only in the named run, and `test_named_gain_scales_are_searched_and_applied` and `test_no_named_gain_scale_keeps_the_catalogue_gains_and_the_geometry_draws` (`:259-295`) take their expectation from the specification and not from `_catalogue_params`, which is removed. `test_unstated_simulator_fields_fall_back_rather_than_fabricate` (`:639-650`) drops `stiffness` and `damping` from its omitted set, and `test_identified_entries_reproduce_their_configuration` (`:629-636`) compares the remaining keys. `catalog_fill_test.py:47-79` runs against `SIM_PARAM_KEYS` as it stands and is run unchanged.
10. `sim_filter_data_check.py:72`, which compares each entry's `sim_params()` with the logged record, is a simulator check and is run in the verification step below.

## Verification

1. `python -m pytest co_optimisation/tests/design_generator_test.py co_optimisation/tests/design_filter_test.py co_optimisation/tests/catalog_fill_test.py -q` in the container, after installing the submodule and then the package.
2. The replay that found the defect, `scripts/analysis/leg_contact_by_design.py` over `design_parameters_1199.csv` with `model_1199.pt`, run again on the changed tree. The 29 designs that pick `TRON1-ABAD` for the ankle are rebuilt with the gains of the specification, so their `ankle_actuator_stiffness` column reads 10.0 and `ankle_actuator_damping` 0.5, and their leg contact terminations per design fall from a mean of 226 towards the 0.5 of the other ankle picks. The swap experiment of the analysis predicts this, since it gave `TRON1-ABAD` the ankle's gains and its terminations fell to 0.8 per design. Its mean episode was 439 steps and not the 762 of the other `TRON1-ANKLE` designs, so a residual instability of the heavier housing is to be expected and is not a failure of this change.
3. One simulator check of the application. After `apply_actuator_params` in a small run that reaches generation 1, `articulation.actuators["ankle"].stiffness[env]` equals 10.0 for every environment whatever its ankle pick, and `articulation.actuators["hip"].stiffness[env]` equals 80.0, against the pick's gain before.
4. A run naming `hip_stiffness_scale`, with a small `--num_envs` and a low `ea_late_start`, asserting that `articulation.actuators["hip"].stiffness[env]` equals `80.0 * parameters["hip_stiffness_scale"]` for a sample of environments and that `params/cmaes.yaml` and `designs/` record the scale.

## Rollout

1. Apply sections 1 to 5 in `design_generator`, run its own checks, commit inside the submodule, then stage the pointer in the parent with sections 6 to 9 and the tests.
2. `design_generator` and `co_optimisation` are installed copies in the production containers and not editable installs, so reinstall `design_generator` first and `co_optimisation` second in each container that trains or replays. A reinstalled package with an old `variable_params.json`, or the new file with an old dataclass, raises `KeyError` on the unknown field, which is the signal that the order was wrong. Neither container is to be reinstalled while a run trains in it.
3. Resume `copt_moral/2026-11-18_05-48-29` from `model_799.pt` only after the replay of the verification step shows the 29 designs recovered.

## Documentation, in the same pass

1. A dated entry in `context/co_optimisation_audit.md` recording the defect, the decoupling, the defects left standing and the figures of the verification step.
2. `design_generator/biped/DESIGN_PARAMETERS.md`, `design_generator/quadruped/DESIGN_PARAMETERS.md` and the two `catalogs/README.md` files, stating that the gains are fields of the specification and no longer catalogue fields.
3. `CO_OPTIMISATION.md` at the workspace root, where the actuator parameters are described, and `plans/gain_scale_search.md`, whose context section describes the scale as acting on the catalogue value, a dated correction to be appended and not an edit in place.
4. `plans/README.md`, whose register carries this plan with its status, and `context/README.md` and `context/leg_collision_analysis.md`, which gain a pointer to this plan as the correction proposed in section 7 of the analysis, a better one than the options listed there since it removes the cause for every group at once.

## Decisions for the review

1. Section 6 keeps the four geometry keys as the default of a run naming no ranges, by excluding `GAIN_PARAM_KEYS` from the default. Taking the default from the whole dictionary instead would make every such run search twelve dimensions, and the exclusion is one expression that reads no new name.
2. Section 8 takes the gain from the specification field and not from `parameters`, because `parameters` holds the scale and the field holds the baseline's gain times it. The instruction to use the value from the parameters is read as the value that `_spec_from_parameters` made of them.
3. Section 7 exempts a scaled gain from the two decimal rounding and section 8 removes the six decimal rounding of the actuator parameters, both at the author's direction after review.
4. The gains stay out of `vector()`, as the pose offsets are, and the gain scales remain the only search path. Making them free fields of `DesignSpec` is a later change.

## Outcome of 2026-11-19

The change was applied to the working tree as the plan and its review specify, with the quantisation of the gains and of the actuator parameters removed. It is uncommitted, and the installed copies in the production containers are untouched, the checks below having run the tree through `PYTHONPATH` beside the container that trains.

1. The suites `design_generator_test.py`, `design_filter_test.py` and `catalog_fill_test.py` pass 178 with 14 skipped. The new tests cover the baseline defaults, the independence of the gains from the pick across every catalogue entry, the single application of a scale, the quadruped's missing ankle, the search keys, the gain free catalogue, and `test_the_gains_follow_variable_params_json`, which writes a copy of each family's `variable_params.json` with every gain edited, builds the generator on it through `variable_params_path`, and asserts that the actuator parameters of the baseline, of a design that picks another actuator for every group and of a scaled design all carry the edited values, while the unedited file still gives the baseline's.
2. The application pipeline was run in Isaac Sim, in `isaac-lab-base-shreyas-gpu0` beside the training job, by `scripts/analysis/leg_contact_by_design.py` on the 100 designs of `design_parameters_1199.csv` under `model_1199.pt`. After the runner's respawn and `apply_actuator_params`, the stiffness and damping read back from every actuator of every environment equal the baseline's, 55 and 13.5, 80 and 13, 60 and 4, 10 and 0.5, with an error of zero against both the population's parameters and the baseline, including the 29 designs that pick `TRON1-ABAD` for the ankle (`../context/artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/gain_readback_decoupled.txt`).
3. The replay of the failing generation changed as the analysis predicted.

| Ankle pick | Designs | Terminations per design before | After | Mean episode in steps before | After |
|---|---|---|---|---|---|
| `TRON1-ANKLE` | 31 | 0.5 | 0.2 | 762 | 1292 |
| `Bota-Roboticer-50` | 40 | 0.5 | 0.2 | 790 | 1250 |
| `TRON1-ABAD` | 29 | 226.1 | 0.3 | 6.6 | 1036 |

   The `leg_contact` terminations of the replay fell from 6590 to 23 (`../context/artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/replay_1199_decoupled_gains.txt`). The designs that pick `TRON1-ABAD` still run shorter episodes than the others, 1036 steps against about 1270, so a residual difficulty of that housing remains, as the swap experiment suggested.
4. Not yet done, the commit, the reinstall of the two packages in the containers, the resumption of the run from `model_799.pt`, the audit entry and the dated correction to `gain_scale_search.md`.

## Revision of 2026-11-19, the gains a design does not search

At the author's direction a gain that a design does not search is left out of the actuator parameters, so that `apply_actuator_params`, which writes only the keys it is given (`../co_optimisation/co_optimisation/utils/respawn.py:282`), leaves it as the environment holds it, randomised at startup by the `randomize_actuator_gains` events of the task (`../environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:751-800`, `../environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py:988-1020`). A first form of this revision drew the unsearched gains in `_spec_from_parameters` from a copy of those ranges, and was replaced by this one, which has fewer lines, keeps one source for the ranges and randomises per environment and not per design.

1. `_actuator_params` writes a gain only where `parameters` names its scale, section 8. `_spec_from_parameters` is unchanged from section 7, and `Family`, `base_record` and `describe_design` carry nothing for it.
2. The startup randomisation does not survive a respawn on its own. `respawn_robots` builds a new `Articulation` whose actuators carry the configuration's gains (steps 5 to 7), the event manager rebinds its terms to it (step 9) and runs no startup term again, so that after the first generation every gain would be the configuration's constant. `respawn_robots.__call__` therefore applies each startup term that is a `randomize_actuator_gains` once more after the rebinding, `respawn.py:209-216`, before the runner writes the design's actuator parameters, which replace only the gains the design searches. The term reads the articulation's own default stiffness and damping, so applying it to a fresh articulation scales the configuration's gains once.
3. The randomisation is per environment, each of the 6400 environments holding its own draw for the run, and a design is averaged over its 64 or 80 environments. The first form gave a design one draw for all its environments, which made the draw a confound of the design the search ranks, so this form is the fairer, and a gain left to the environment is a nuisance variable that averages out of a design's fitness as the environment's other randomisation does.
4. A searched gain is unchanged, the scale multiplying the baseline's gain, written for every environment of the design, so a search of the gain scales behaves as before.
5. Consequences to record. The design records carry no `<group>_actuator_stiffness` or `_damping` for a gain that is not searched, since the design has no single value of it, and `base_record` carries none. A task whose configuration has no `randomize_actuator_gains` term keeps the configuration's constant gains. The startup terms for mass, friction and armature are not applied again by this change, and whether they too are lost at a respawn is not examined here.
6. Tests. `test_a_gain_that_is_not_searched_is_left_to_the_environment` asserts that a named scale writes its gain and no other, that no gain is written without parameters, and that `base_record` carries none. The tests that asserted the baseline's gains for an unnamed gain now name every scale at 1.0 or assert the gain absent, and `test_the_gains_follow_variable_params_json` edits both files and asserts the written gains. The suites pass 182 with 14 skipped.
7. Simulator check. The replay of the 100 designs of `design_parameters_1199.csv` under `model_1199.pt` on this tree left every gain out of the population's parameters. The actuators then held 200 distinct values of every gain, 100 environments by the two legs, within the configuration's ranges, the ankle stiffness from 8.02 to 11.98 about 10 and the ankle damping from 0.4 to 0.6 about 0.5 (`../context/artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/gain_readback_environment_gains.txt`). The `leg_contact` terminations of the replay were 19 in 124 episodes against 6590 in 6747, and the 29 designs that pick `TRON1-ABAD` for the ankle averaged 0.3 terminations and episodes of 1176 steps, against 226.1 and 6.6 before (`../context/artefacts/copt_moral-2026-11-18_05-48-29-leg-collision/replay_1199_environment_gains.txt`).
