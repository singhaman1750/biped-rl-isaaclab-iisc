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
