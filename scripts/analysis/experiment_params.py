"""Reconstruction of an experiment from the parameter files the run itself dumped.

Isaac Lab writes params/env.yaml and params/agent.yaml beside the checkpoints of every
run, and those files carry the configuration the policy was TRAINED under, which is not
in general the configuration the working tree holds when the policy is replayed. This
module reads them, so that an evaluation is configured by the artefact rather than by
hand.

The parsing layer imports nothing from Isaac Lab and is therefore usable from an analysis
dashboard outside the simulation container. The revival layer, which turns parsed terms
back into live configuration objects, imports lazily and is usable only inside it.
"""

from __future__ import annotations

import importlib
import os

import yaml


class ParamsLoader(yaml.SafeLoader):
    """A SafeLoader that survives the python specific tags Isaac Lab's dumper emits.

    yaml.unsafe_load would execute the imports these tags name, which fails wherever the
    task package is not installed and which is exactly the case a dashboard runs in. This
    loader records what each tag REFERRED to instead of resolving it, so the document
    parses whole and the decision to import is left to the caller.
    """


def _construct_name(loader, suffix, node):
    # A function or class reference, e.g. `!!python/name:package.module.function ''`.
    # The dotted path is retained so that the revival layer may import it on demand.
    return {"__ref__": suffix}


def _construct_object(loader, suffix, node):
    # A configuration instance. Its fields become a plain mapping, with the class it was
    # dumped from retained under a reserved key so that the revival layer can rebuild it.
    mapping = loader.construct_mapping(node, deep=True) if isinstance(
        node, yaml.MappingNode
    ) else {}
    mapping["__class__"] = suffix
    return mapping


def _construct_apply(loader, suffix, node):
    # A constructed value, e.g. `!!python/object/apply:builtins.slice [null, null, null]`.
    return {"__apply__": suffix, "args": loader.construct_sequence(node, deep=True)}


def _construct_tuple(loader, node):
    return tuple(loader.construct_sequence(node, deep=True))


ParamsLoader.add_multi_constructor("tag:yaml.org,2002:python/name:", _construct_name)
ParamsLoader.add_multi_constructor("tag:yaml.org,2002:python/object:", _construct_object)
ParamsLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/object/apply:", _construct_apply
)
ParamsLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)


def params_dir(run_dir: str) -> str:
    """The params directory of a run, given the run directory or any path inside it."""
    candidate = run_dir
    for _ in range(3):
        if os.path.isdir(os.path.join(candidate, "params")):
            return os.path.join(candidate, "params")
        candidate = os.path.dirname(candidate)
    return os.path.join(run_dir, "params")


def load_params(run_dir: str, name: str = "env") -> dict | None:
    """Parse params/<name>.yaml of a run into plain containers, or None if absent."""
    path = os.path.join(params_dir(run_dir), f"{name}.yaml")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return yaml.load(handle, Loader=ParamsLoader)


def reward_terms(env_params: dict) -> dict[str, dict]:
    """Ordered name -> {weight, func, params} for every configured reward term.

    The order is the declaration order of the configuration class, which is the order
    RewardManager registers the terms in at reward_manager.py:222 and therefore the order
    the per term series appear in rewards.npy. Verified identical, name for name and
    position for position, across the twenty six terms of run 2026-08-03_11-19-11.
    """
    terms = {}
    for name, term in (env_params or {}).get("rewards", {}).items():
        if name == "__class__" or not isinstance(term, dict):
            continue
        func = term.get("func")
        terms[name] = {
            "weight": float(term.get("weight", 0.0)),
            "func": func.get("__ref__") if isinstance(func, dict) else None,
            "params": term.get("params", {}),
        }
    return terms


def reward_weights(env_params: dict) -> dict[str, float]:
    """Name -> configured weight, the divisor that recovers an unweighted term value."""
    return {name: term["weight"] for name, term in reward_terms(env_params).items()}


def step_dt(env_params: dict) -> float | None:
    """The control period, as the product of the physics step and the decimation.

    This is the quantity every cadence, duty fraction, loading rate and cost of transport
    scales with, and reading it from the run rather than assuming it is what makes a
    figure comparable across two robots configured differently.
    """
    sim = (env_params or {}).get("sim") or {}
    if "dt" in sim and env_params.get("decimation") is not None:
        return float(sim["dt"]) * int(env_params["decimation"])
    return None


def feet_body_expression(env_params: dict, term: str = "feet_air_time") -> str | None:
    """The body name expression resolving the feet, taken from a contact keyed term.

    Every gait term that keys on the feet carries a SceneEntityCfg naming them, so the
    feet regex need not be a constant in the analysis code. Falls back through the terms
    that carry one, since a configuration need not declare any given term.
    """
    terms = reward_terms(env_params)
    for candidate in (term, "feet_slide", "rew_no_fly", "rew_gait", "pen_feet_impact"):
        params = terms.get(candidate, {}).get("params", {})
        sensor = params.get("sensor_cfg") or params.get("asset_cfg") or {}
        names = sensor.get("body_names") if isinstance(sensor, dict) else None
        if names:
            return names if isinstance(names, str) else list(names)
    return None


def _revive(value):
    """Turn a parsed mapping back into the object it was dumped from.

    Constructed through __new__ rather than __init__, because a configclass may declare
    required fields whose values are already present in the mapping and whose ordering
    the dumper does not preserve, so calling the constructor would demand arguments that
    are about to be overwritten anyway.
    """
    if isinstance(value, dict) and "__ref__" in value:
        module_path, _, attribute = value["__ref__"].rpartition(".")
        return getattr(importlib.import_module(module_path), attribute)
    if isinstance(value, dict) and "__apply__" in value:
        module_path, _, attribute = value["__apply__"].rpartition(".")
        factory = getattr(importlib.import_module(module_path), attribute)
        return factory(*[_revive(argument) for argument in value["args"]])
    if isinstance(value, dict) and "__class__" in value:
        module_path, _, attribute = value["__class__"].rpartition(".")
        cls = getattr(importlib.import_module(module_path), attribute)
        instance = cls.__new__(cls)
        for key, item in value.items():
            if key != "__class__":
                setattr(instance, key, _revive(item))
        return instance
    if isinstance(value, dict):
        return {key: _revive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_revive(item) for item in value)
    return value


def apply_reward_cfg(env_cfg, env_params: dict, strict: bool = False) -> dict:
    """Replace the reward configuration of env_cfg with the run's own, term by term.

    Returns a report of what was applied, what was added and what could not be imported,
    which the caller should print, since a silently skipped term is a policy replayed
    under a reward set it was not trained against.

    Only the reward terms are touched. Every other field of env_cfg is left exactly as
    parse_env_cfg produced it, so a caller that does not invoke this function sees no
    change whatever, which is the backwards compatibility condition of ../CLAUDE.md.
    """
    from isaaclab.managers.manager_term_cfg import RewardTermCfg

    report = {"applied": [], "added": [], "removed": [], "failed": {}}
    parsed = reward_terms(env_params)
    existing = {
        name for name in vars(env_cfg.rewards) if not name.startswith("_")
    }
    for name, term in parsed.items():
        if term["func"] is None:
            report["failed"][name] = "no func reference in params"
            continue
        try:
            func = _revive({"__ref__": term["func"]})
            params = {k: _revive(v) for k, v in (term["params"] or {}).items()}
        except Exception as error:  # noqa: BLE001
            # A term whose function the tree no longer carries. Recorded rather than
            # raised, so that one deleted reward does not make a whole run unevaluable.
            report["failed"][name] = repr(error)
            if strict:
                raise
            continue
        setattr(
            env_cfg.rewards,
            name,
            RewardTermCfg(func=func, params=params, weight=term["weight"]),
        )
        report["added" if name not in existing else "applied"].append(name)
    for name in sorted(existing - set(parsed)):
        # A term the tree carries and the run did not. Removed, because leaving it in
        # would add a reward the policy never saw to the budget the dashboard compares.
        setattr(env_cfg.rewards, name, None)
        report["removed"].append(name)
    return report


# Fields of a dumped actuator configuration that describe the class it was built from or
# the joints it binds to, rather than the drive behaviour. They are reported when they
# disagree and never written, since a joint expression or an actuator class taken from an
# older run may not match the robot the tree now spawns.
_ACTUATOR_STRUCTURAL = ("__class__", "class_type", "joint_names_expr")


def actuator_groups(env_params: dict) -> dict[str, dict]:
    """The actuator configuration of the run, group by group, as parsed mappings.

    Isaac Lab dumps these under scene.robot.actuators, one entry per actuator group, each
    carrying the stiffness, damping, armature, friction and limit fields the group was
    configured with. Returns an empty mapping where the dump predates the block or the
    run configured no articulation, which the caller should treat as nothing to apply.
    """
    robot = ((env_params.get("scene") or {}).get("robot") or {})
    groups = robot.get("actuators") or {}
    return {name: group for name, group in groups.items() if isinstance(group, dict)}


def apply_actuator_cfg(env_cfg, env_params: dict, strict: bool = False) -> dict:
    """Restore the run's actuator gains onto env_cfg, group by group and field by field.

    The drive fields are written onto the configuration object the tree already built,
    rather than the dumped object being revived whole. Reviving whole would carry the
    run's actuator CLASS into the replay, and a class the tree has since altered would
    then be constructed from fields that no longer describe it, so the safe operation is
    to leave the object and its class alone and to overwrite the numbers it holds. Fields
    the tree's actuator class does not declare are skipped rather than invented, and the
    structural fields of _ACTUATOR_STRUCTURAL are compared and reported but never written.

    Returns a report the caller should print. The `changed` entry names every field whose
    dumped value differs from the tree's, which is the evidence that the replay would
    otherwise have run the policy on gains it was never trained against.

    Only env_cfg.scene.robot.actuators is touched. A caller that does not invoke this
    function sees no change whatever, which is the backwards compatibility condition of
    ../CLAUDE.md.
    """
    report = {"applied": [], "changed": {}, "missing": [], "extra": [], "structural": {}}
    parsed = actuator_groups(env_params)
    if not parsed:
        return report
    robot = getattr(getattr(env_cfg, "scene", None), "robot", None)
    live = getattr(robot, "actuators", None) or {}
    for name, group in parsed.items():
        target = live.get(name)
        if target is None:
            report["missing"].append(name)
            continue
        for field in _ACTUATOR_STRUCTURAL:
            if field not in group:
                continue
            dumped = group[field]
            if field == "__class__":
                current = f"{type(target).__module__}.{type(target).__name__}"
            elif field == "class_type":
                current = getattr(target, field, None)
                current = (
                    f"{current.__module__}.{current.__name__}" if current else None
                )
                dumped = dumped.get("__ref__") if isinstance(dumped, dict) else dumped
            else:
                current = getattr(target, field, None)
            if current is not None and dumped is not None and current != dumped:
                report["structural"].setdefault(name, {})[field] = (current, dumped)
        changed = {}
        for field, value in group.items():
            if field in _ACTUATOR_STRUCTURAL or not hasattr(target, field):
                continue
            try:
                revived = _revive(value)
            except Exception as error:  # noqa: BLE001
                # A field naming something the tree no longer carries. Recorded rather
                # than raised, so that one stale field does not make a run unevaluable.
                report["structural"].setdefault(name, {})[field] = ("unimportable", repr(error))
                if strict:
                    raise
                continue
            current = getattr(target, field, None)
            if current != revived:
                changed[field] = (current, revived)
            setattr(target, field, revived)
        report["applied"].append(name)
        if changed:
            report["changed"][name] = changed
    report["extra"] = sorted(set(live) - set(parsed))
    return report


# Fields of a dumped action term that define the mapping from a policy output to a joint
# target. Restoring these is what makes a replayed action mean what it meant in training.
_ACTION_TRANSFORM = ("scale", "offset", "use_default_offset", "clip")

# Fields naming the class or the joints the term binds to. Reported when they disagree and
# never written, for the reason given at _ACTUATOR_STRUCTURAL.
_ACTION_STRUCTURAL = ("__class__", "class_type", "asset_name", "joint_names", "preserve_order")


def action_terms(env_params: dict) -> dict[str, dict]:
    """The action configuration of the run, term by term, as parsed mappings."""
    actions = env_params.get("actions") or {}
    return {
        name: term for name, term in actions.items()
        if isinstance(term, dict) and not name.startswith("__")
    }


def apply_action_cfg(env_cfg, env_params: dict, strict: bool = False) -> dict:
    """Restore the run's action transform onto env_cfg, term by term.

    The action scale converts a policy output into a joint position offset, so a replay at a
    scale the policy was not trained under commands a different posture for the same output.
    The defect is silent, the policy running and the rewards restoring normally while every
    commanded offset is wrong by the ratio of the two scales.

    Only the fields of _ACTION_TRANSFORM are written, onto the configuration object the tree
    already built, for the reasons given at apply_actuator_cfg. Returns a report the caller
    should print, whose `changed` entry names every field that disagreed.

    Only env_cfg.actions is touched. A caller that does not invoke this function sees no
    change whatever, which is the backwards compatibility condition of ../CLAUDE.md.
    """
    report = {"applied": [], "changed": {}, "missing": [], "extra": [], "structural": {}}
    parsed = action_terms(env_params)
    if not parsed:
        return report
    actions = getattr(env_cfg, "actions", None)
    live = {n for n in vars(actions) if not n.startswith("_")} if actions else set()
    for name, term in parsed.items():
        target = getattr(actions, name, None)
        if target is None:
            report["missing"].append(name)
            continue
        for field in _ACTION_STRUCTURAL:
            if field not in term:
                continue
            dumped = term[field]
            if field == "__class__":
                current = f"{type(target).__module__}.{type(target).__name__}"
            elif field == "class_type":
                current = getattr(target, field, None)
                current = f"{current.__module__}.{current.__name__}" if current else None
                dumped = dumped.get("__ref__") if isinstance(dumped, dict) else dumped
            else:
                current = getattr(target, field, None)
            if current is not None and dumped is not None and current != dumped:
                report["structural"].setdefault(name, {})[field] = (current, dumped)
        changed = {}
        for field in _ACTION_TRANSFORM:
            if field not in term or not hasattr(target, field):
                continue
            try:
                revived = _revive(term[field])
            except Exception as error:  # noqa: BLE001
                report["structural"].setdefault(name, {})[field] = ("unimportable", repr(error))
                if strict:
                    raise
                continue
            current = getattr(target, field, None)
            if current != revived:
                changed[field] = (current, revived)
            setattr(target, field, revived)
        report["applied"].append(name)
        if changed:
            report["changed"][name] = changed
    report["extra"] = sorted(live - set(parsed))
    return report
