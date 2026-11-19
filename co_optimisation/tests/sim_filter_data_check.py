"""Runs scripts/rsl_rl/train.py with the run's CMAESCfg overridden and checks every spawned population.

Run with Isaac Sim, not collected by pytest. After every _reload_morphology and _update_morphology of
CoptOnPolicyRunner, every environment's actuator tensors, joint armature, PhysX joint force ceiling
and body masses are compared with the catalogue entry its design picked and with its own MJCF, and
every design is passed through the configured filters again. One line per population is printed,
prefixed [CHECK]. See plans/design_filtering.md section 6.

    COPT_CHECK='{"design_search": "actuator", "filters": ["StandingFilter"], ...}' \
    /isaac-sim/python.sh sim_filter_data_check.py path/to/train.py --task ... --policy_type COPT ...
"""
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys

OVERRIDES = json.loads(os.environ.get("COPT_CHECK", "{}"))
TRAIN = os.path.abspath(sys.argv[1])
sys.argv = [TRAIN] + sys.argv[2:]
sys.path.insert(0, os.path.dirname(TRAIN))
TOL = 1e-4


def _patch_config(module):
    for family, cls in list(module.CMAES_CFG_BY_FAMILY.items()):
        def make(cls=cls):
            cfg = cls()
            for key, value in OVERRIDES.items():
                setattr(cfg, key, value)
            return cfg
        module.CMAES_CFG_BY_FAMILY[family] = make


def _ids(actuator, articulation):
    ids = actuator.joint_indices
    return list(range(articulation.num_joints))[ids] if isinstance(ids, slice) else [int(i) for i in ids]


def _value(x, e, j):
    return float(x[e, j]) if hasattr(x, "shape") and len(getattr(x, "shape", ())) == 2 else float(x)


def check(runner, where):
    from co_optimisation.utils.filters import MjcfStatics

    env = runner.env.unwrapped
    art = env.scene.articulations["robot"]
    pop = runner.current_population
    gen = runner._design_generator
    n = len(pop.get_usd_files())
    catalog = {a.name: a for a in gen.family.catalog()}
    models = {}
    max_force = art.root_physx_view.get_dof_max_forces().cpu()
    masses = art.root_physx_view.get_masses().cpu()
    armature = art.data.joint_armature.cpu()
    keys = ("effort_limit", "velocity_limit", "stiffness", "damping", "friction_static", "friction_dynamic", "activation_vel")
    worst = {k: 0.0 for k in keys + ("saturation_effort", "armature", "xml_forcerange", "params_vs_catalog")}
    physx_below, mass_err, mass_bodies, picks = 0, 0.0, 0, set()
    for e in range(env.num_envs):
        i = e % n
        spec = pop.get_specs()[i]
        params = pop.get_actuator_params()[i]
        if i not in models:
            models[i] = MjcfStatics(pop.get_xml_files()[i])
        model = models[i]
        chosen = []
        for group, act in art.actuators.items():
            entry = catalog[getattr(spec, gen.family.spec_fields[group])]
            chosen.append(entry.name)
            # A gain the design does not search is the environment's own, randomised at startup, and is
            # held by the actuator without being in the design's parameters.
            expect = {**entry.sim_params(), **{k: params[group][k] for k in ("stiffness", "damping") if k in params[group]}}
            for key in keys:
                if key in expect:
                    worst["params_vs_catalog"] = max(worst["params_vs_catalog"], abs(params[group][key] - expect[key]))
            sat = act._saturation_effort
            for k, j in enumerate(_ids(act, art)):
                for key in keys:
                    if key in expect:
                        worst[key] = max(worst[key], abs(_value(getattr(act, key), e, k) - expect[key]))
                worst["saturation_effort"] = max(worst["saturation_effort"], abs(_value(sat, e, k) - expect["saturation_effort"]))
                worst["armature"] = max(worst["armature"], abs(float(armature[e, j]) - expect["armature"]))
                name = art.joint_names[j]
                worst["xml_forcerange"] = max(worst["xml_forcerange"],
                                              abs(model.limit[model.joint_names.index(name)] - expect["effort_limit"]))
                physx_below += float(max_force[e, j]) < expect["effort_limit"] - TOL
        picks.add(tuple(chosen))
        for b, name in enumerate(art.body_names):
            if name in model.body_names:
                ref = model.mass[model.body_names.index(name)]
                if ref > 0:
                    mass_err = max(mass_err, abs(float(masses[e, b]) - ref) / ref)
                    mass_bodies += e == 0
    refused = {f.name: len(f.filter(pop.get_xml_files())) for f in gen.filters}
    history = gen.filter_history[-1] if gen.filter_history else None
    ok = all(v < TOL for v in worst.values()) and physx_below == 0 and mass_err < 1e-3
    print("[CHECK] " + json.dumps({
        "where": where, "generation": runner.generation - 1, "designs": n, "envs": env.num_envs,
        "distinct_picks": len(picks), "max_abs_error": worst, "physx_force_below_effort_limit": physx_below,
        "mass_rel_error": mass_err, "mass_bodies_matched": mass_bodies, "of_bodies": len(art.body_names),
        "refused_on_recheck": refused, "filter_record": history, "ok": bool(ok),
    }, default=lambda o: o.item() if hasattr(o, "item") else str(o)), flush=True)


def _patch_runner(module):
    cls = module.CoptOnPolicyRunner
    for name in ("_reload_morphology", "_update_morphology"):
        original = getattr(cls, name)

        def wrapped(self, *args, _original=original, _name=name, **kwargs):
            out = _original(self, *args, **kwargs)
            if self.current_population is not None:
                check(self, _name)
            return out
        setattr(cls, name, wrapped)


PATCHES = {"co_optimisation.config": _patch_config, "co_optimisation.runners.copt_on_policy_runner": _patch_runner}


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        execute = spec.loader.exec_module

        def exec_module(module, _execute=execute, _name=name):
            _execute(module)
            PATCHES[_name](module)
        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, Hook())
runpy.run_path(TRAIN, run_name="__main__")
