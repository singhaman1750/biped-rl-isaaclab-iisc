"""Tests for co_optimisation.utils.filters, the filter loop of DesignGeneratorBase and CatOnlyCMAESDesignGenerator.

Runs without the simulator wherever numpy, cma, cmaes and design_generator are installed, the
conversion being stubbed as design_generator_test.py stubs it. See plans/design_filtering.md.

    /isaac-sim/python.sh -m pytest co_optimisation/tests/design_filter_test.py
"""
import importlib.util
import os
import pathlib
import pickle
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _stub_converter():
    class Cfg:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Converter:
        def __init__(self, cfg):
            os.makedirs(cfg.usd_dir, exist_ok=True)
            self.usd_path = os.path.join(cfg.usd_dir, cfg.usd_file_name)
            open(self.usd_path, "w").close()

    for name in ("isaaclab", "isaaclab.sim"):
        sys.modules.setdefault(name, types.ModuleType(name))
    conv = types.ModuleType("isaaclab.sim.converters")
    conv.MjcfConverter, conv.MjcfConverterCfg = Converter, Cfg
    sys.modules["isaaclab.sim.converters"] = conv


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_stub_converter()
F = _load("design_filters", ROOT / "co_optimisation/utils/filters.py")
# usd_generator imports make_filters from the package, whose __init__ needs the simulator.
for _name in ("co_optimisation", "co_optimisation.utils"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["co_optimisation.utils.filters"] = F
U = _load("usd_generator_filtered", ROOT / "co_optimisation/runners/usd_generator.py")
C = _load("design_common", ROOT / "co_optimisation/utils/common.py")
# The tree's own usd_generator, before this change, where it is present beside the sandbox.
ORIGINAL = os.environ.get("ORIGINAL_USD_GENERATOR")
O = _load("usd_generator_original", ORIGINAL) if ORIGINAL else None

QUAD_RANGES = {
    "thigh_length_scale": ("scale", (0.5, 1.50)), "shank_length_scale": ("scale", (0.5, 1.50)),
    "thigh_bend_offset": ("offset", (-0.50, 0.50)), "knee_bend_offset": ("offset", (-0.5, 0.5)),
}


class RefuseLong(F.FilterBase):
    """Refuses every design whose thigh is longer than the donor's, read from its document."""

    def __init__(self, family, limit):
        super().__init__(family)
        self.limit = limit

    def filter(self, xml_files):
        out = []
        for i, path in enumerate(xml_files):
            m = F.MjcfStatics(path)
            child = m.body_names.index("shank_R_Link" if self.family == "biped" else "shank_FR_Link")
            if np.linalg.norm(m.pos[child]) > self.limit + 1e-9:
                out.append(i)
        return out


def _generator(cls, family, tmp_path, **kw):
    kw.setdefault("output_dir", str(tmp_path))
    if cls.__name__ != "RandomDesignGenerator":
        kw.update(late_start=kw.get("late_start", True), late_start_it=4, late_start_prior_pop_size=8, sigma0=0.25, seed=3)
    g = cls(family, 8, QUAD_RANGES if family == "quadruped" else None, **kw)
    if family == "quadruped":
        g.builder_kwargs = {"shank_foot_gap": 0.001}
    return g


def _xmls(pop):
    return [open(p).read() for p in pop.get_xml_files()]


# ---------------------------------------------------------------------------
# Population and the import path
# ---------------------------------------------------------------------------


def test_population_is_reexported():
    assert U.Population.__name__ == "Population"
    assert set(dir(C.Population)) <= set(dir(U.Population))


# ---------------------------------------------------------------------------
# No filters, the populations of the tree before this change
# ---------------------------------------------------------------------------


@pytest.mark.skipif(O is None, reason="ORIGINAL_USD_GENERATOR not set")
@pytest.mark.parametrize("family", ["biped", "quadruped"])
@pytest.mark.parametrize("name", ["RandomDesignGenerator", "GrowingDesignDistCMAESDesignGenerator",
                                  "CMAESDesignGenerator", "CatCMAESDesignGenerator"])
def test_unfiltered_populations_unchanged(family, name, tmp_path):
    pops = []
    for module, sub in ((O, "old"), (U, "new")):
        g = _generator(getattr(module, name), family, tmp_path / sub)
        seen = []
        for gen in range(3):
            if gen == 2 and name != "RandomDesignGenerator":
                g.toggle_late_start()
                g.sample_batch()
            pop = g.generate_population(gen)
            if gen == 2 and name != "RandomDesignGenerator":
                g.update_with_fitness(list(np.arange(8.0)))
            seen.append((_xmls(pop), pop.get_parameters(), pop.get_actuator_params(),
                         [os.path.relpath(p, tmp_path / sub) for p in pop.get_usd_files()]))
        pops.append(seen)
    for gen, (old, new) in enumerate(zip(*pops)):
        # The categorical random phase widens its actuator choice since 2026-11-12, so only its geometry
        # draws are compared there, generation 2, the first CMA-ES one, being compared in full.
        widened = name == "CatCMAESDesignGenerator" and gen < 2
        if not widened:
            assert (old[0], old[2], old[3]) == (new[0], new[2], new[3])
        for before, after in zip(old[1], new[1]):
            kept = [k for k in before if not (widened and k in U.FAMILIES[family].groups)]
            assert {k: after[k] for k in kept} == {k: before[k] for k in kept}
            assert {"total_mass", "nominal_height", "spec_thigh_length", "knee_actuator"} <= set(after)


# ---------------------------------------------------------------------------
# The filter loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["biped", "quadruped"])
def test_random_phase_redraws_until_compliant(family, tmp_path):
    g = _generator(U.GrowingDesignDistCMAESDesignGenerator, family, tmp_path)
    limit = g.baseline.thigh_length
    g.filters = [RefuseLong(family, limit)]
    pop = g.generate_population(3)
    assert RefuseLong(family, limit).filter(pop.get_xml_files()) == []
    record = g.filter_history[-1]
    assert record["redrawn"] > 0 and record["kept_refused"] == 0 and g._designs_not_compliant == []
    assert g._filter_attempt == 0
    assert all(os.path.exists(p) for p in pop.get_usd_files())
    # The records at each position describe the design its document holds.
    for spec, links, xml in zip(pop.get_specs(), pop.get_link_length_params(), pop.get_xml_files()):
        assert spec.thigh_length <= limit + 1e-9
        assert F.MjcfStatics(xml).mass.sum() == pytest.approx(
            F.MjcfStatics(xml).mass.sum())  # parses
        assert links  # rebuilt


def test_cmaes_phase_replaces_the_told_genotypes(tmp_path):
    g = _generator(U.CMAESDesignGenerator, "biped", tmp_path, late_start=False)
    limit = g.baseline.thigh_length
    g.filters = [RefuseLong("biped", limit)]
    g.sample_batch()
    pop = g.generate_population(0)
    told = [g._denormalise(x)["thigh_length_scale"] for x in g._last_solutions]
    drawn = [p["thigh_length_scale"] for p in pop.get_parameters()]
    assert told == pytest.approx(drawn)
    assert max(round(g.baseline.thigh_length * t, 2) for t in told) <= limit + 1e-9
    g.update_with_fitness(list(np.arange(8.0)))      # pycma accepts the resampled genotypes
    assert g._es.countiter == 1


def test_categorical_phase_redraws(tmp_path):
    g = _generator(U.CatCMAESDesignGenerator, "biped", tmp_path, late_start=False)
    g.filters = [RefuseLong("biped", g.baseline.thigh_length)]
    g.sample_batch()
    pop = g.generate_population(0)
    assert RefuseLong("biped", g.baseline.thigh_length).filter(pop.get_xml_files()) == []
    g.update_with_fitness(list(np.arange(8.0)))
    assert g._es.generation == 1


def test_round_cap_keeps_the_last_draws(tmp_path):
    g = _generator(U.GrowingDesignDistCMAESDesignGenerator, "biped", tmp_path)
    g.filters = [RefuseLong("biped", 0.0)]           # refuses everything
    g.filter_max_rounds = 2
    pop = g.generate_population(1)
    assert g.filter_history[-1]["rounds"] == 2 and g.filter_history[-1]["kept_refused"] == 8
    assert len(pop.get_usd_files()) == 8 and g._filter_attempt == 0


def test_mean_population_is_not_filtered(tmp_path):
    g = _generator(U.CMAESDesignGenerator, "biped", tmp_path, late_start=False, final_population="mean")
    g.filters = [RefuseLong("biped", 0.0)]
    g._terminated = True
    g._last_solutions = []
    pop = g.generate_population(5)
    assert len(pop.get_usd_files()) == 8 and not g.filter_history


# ---------------------------------------------------------------------------
# The standing filter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["biped", "quadruped"])
def test_standing_filter_passes_the_donor(family, tmp_path):
    g = _generator(U.RandomDesignGenerator, family, tmp_path)
    path = tmp_path / "donor.xml"
    g.emit_design({}, None, str(path))
    sf = F.make_filters(["StandingFilter"], family)[0]
    assert sf.filter([str(path)]) == []
    r = sf.evaluate([str(path)])
    assert r["failure_fraction"][0] <= 0.125 + 1e-9
    assert r["failed"].shape == (1, 16, 16)


def test_standing_filter_refuses_a_weak_robot(tmp_path):
    g = _generator(U.RandomDesignGenerator, "biped", tmp_path)
    path = tmp_path / "donor.xml"
    g.emit_design({}, None, str(path))
    weak = tmp_path / "weak.xml"
    weak.write_text(path.read_text().replace('forcerange="-90.000000 90.000000"', 'forcerange="-2.000000 2.000000"'))
    assert F.StandingFilter("biped").filter([str(path), str(weak)]) == [1]


def test_make_filters_refuses_a_non_filter():
    with pytest.raises((ValueError, ImportError, AttributeError, ModuleNotFoundError)):
        F.make_filters(["numpy:ndarray"], "biped")


def test_generator_builds_its_filters(tmp_path):
    g = _generator(U.RandomDesignGenerator, "biped", tmp_path, filters=["StandingFilter"],
                   filter_params={"StandingFilter": {"num_positions": 4}})
    assert [f.name for f in g.filters] == ["StandingFilter"] and g.filters[0].num_positions == 4
    assert _generator(U.RandomDesignGenerator, "biped", tmp_path / "none").filters == []


# ---------------------------------------------------------------------------
# The categorical only search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["biped", "quadruped"])
def test_cat_only_keeps_the_geometry(family, tmp_path):
    g = _generator(U.CatOnlyCMAESDesignGenerator, family, tmp_path)
    assert g.param_keys == [] and g._es._Nco == 0 and not hasattr(g._es, "_mean")
    base = g.family.baseline()
    late = g.generate_population(1)
    g.toggle_late_start()
    g.sample_batch()
    pop = g.generate_population(2)
    for p in (late, pop):
        for spec, params in zip(p.get_specs(), p.get_parameters()):
            assert (spec.thigh_length, spec.shank_length) == (base.thigh_length, base.shank_length)
            # No searched parameter, only the picks, the feasibility and the design record columns.
            assert set(g.family.groups) | {"feasible"} <= set(params)
            assert not set(params) & set(U.CMAES_PARAM_KEYS)
    picks = {tuple(p[k] for k in g.family.groups) for p in late.get_parameters()}
    assert len(picks) > 1
    # The actuator parameters are the picked entries' own.
    catalog = {a.name: a for a in g.family.catalog()}
    for spec, act in zip(pop.get_specs(), pop.get_actuator_params()):
        for group, field in g.family.spec_fields.items():
            assert act[group]["effort_limit"] == pytest.approx(catalog[getattr(spec, field)].sim_params()["effort_limit"])
    g.update_with_fitness(list(np.arange(8.0)))
    stats = g.search_statistics()
    assert stats["cma_iteration"] == 1 and set(stats["pick_probability"]) == set(g.family.groups)
    values, chosen = g.optimised_design()
    assert values == {} and set(chosen) == set(g.family.groups)
    # The index is the catalogue's own, where it states one, and no group carries one where it states none.
    by_index = {a.index: a.name for a in g.family.full_catalog() if a.index is not None}
    assert stats["pick"] == chosen
    assert {k: by_index[i] for k, i in stats["pick_index"].items()} == (chosen if by_index else {})
    assert g.emit_design(values, chosen, str(tmp_path / "mean.xml"))
    state = pickle.loads(pickle.dumps(g.get_state()))
    g2 = _generator(U.CatOnlyCMAESDesignGenerator, family, tmp_path / "again")
    g2.load_state(state)
    assert np.allclose(g2._es._q, g._es._q)


@pytest.mark.parametrize("cls", ["CatCMAESDesignGenerator", "CatOnlyCMAESDesignGenerator"])
def test_a_categorical_state_carries_its_shelf(cls, tmp_path):
    g = _generator(getattr(U, cls), "biped", tmp_path, late_start=False)
    state = pickle.loads(pickle.dumps(g.get_state()))
    assert state["cat_names"] == g.cat_names
    g2 = _generator(getattr(U, cls), "biped", tmp_path / "again", late_start=False)
    g2.cat_names = ["renamed"] * len(g.cat_names)
    g2.load_state(state)
    assert g2.cat_names == g.cat_names


@pytest.mark.parametrize("family", ["biped", "quadruped"])
def test_design_records_carry_geometry_pose_mass_and_actuators(family, tmp_path):
    g = _generator(U.CatCMAESDesignGenerator, family, tmp_path)
    g.filters = [F.StandingFilter(family)]
    pop = g.generate_population(1)
    catalog = {a.name: a for a in g.family.full_catalog()}
    for spec, act, row, xml in zip(pop.get_specs(), pop.get_actuator_params(), pop.get_parameters(), pop.get_xml_files()):
        assert row["spec_thigh_length"] == spec.thigh_length and row["spec_knee_bend_offset"] == spec.knee_bend_offset
        assert row["nominal_height"] == pytest.approx(g.family.nominal(spec)["height"])
        assert row["total_mass"] == pytest.approx(F.MjcfStatics(xml).mass.sum())
        for group, field in g.family.spec_fields.items():
            entry = catalog[getattr(spec, field)]
            assert row[f"{group}_actuator"] == entry.name
            assert row[f"{group}_actuator_peak_torque"] == entry.peak_torque
            assert row[f"{group}_actuator_mass"] == entry.mass
            for key, value in act[group].items():
                assert row[f"{group}_actuator_{key}"] == value
        assert 0.0 <= row["standing_failure_fraction"] <= 0.25
        assert not any(k.endswith("_scale") for k in row if k not in U.DEFAULT_PARAM_RANGES)


@pytest.mark.parametrize("family", ["biped", "quadruped"])
def test_base_record_is_the_donor(family, tmp_path):
    g = _generator(U.GrowingDesignDistCMAESDesignGenerator, family, tmp_path)
    g.filters = [F.StandingFilter(family)]
    base = g.base_record(str(tmp_path / "base.xml"))
    donor = g.family.baseline()
    assert base["design"] == "base" and base["thigh_length_scale"] == 1.0 and base["knee_bend_offset"] == 0.0
    assert base["spec_thigh_length"] == donor.thigh_length and base["spec_shank_length"] == donor.shank_length
    assert base["total_mass"] == pytest.approx(F.MjcfStatics(str(tmp_path / "base.xml")).mass.sum())
    assert base["knee_actuator"] == donor.act_knee and base["standing_failure_fraction"] <= 0.125
    values, picks = g.optimised_design()
    final = g.describe_design(values, picks)
    assert set(base) - {"design", "standing_failure_fraction", "standing_unposed_fraction",
                        "standing_peak_load_median"} - set(U.SPEC_PARAM_FIELDS) <= set(final)


@pytest.mark.parametrize("family", ["biped", "quadruped"])
@pytest.mark.parametrize("cls", ["CatCMAESDesignGenerator", "CatOnlyCMAESDesignGenerator"])
def test_the_actuator_choice_widens_over_the_random_phase(family, cls, tmp_path):
    n = 15
    g = getattr(U, cls)(family, 8, QUAD_RANGES if family == "quadruped" else None, output_dir=str(tmp_path),
                        late_start=True, late_start_it=n, late_start_prior_pop_size=8)
    base = {group: getattr(g.baseline, g.family.spec_fields[group]) for group in g.cat_groups}
    catalog = {a.name: a for a in g.family.full_catalog()}
    sizes = []
    for group in g.cat_groups:
        order = g._neighbours(group)
        assert order[0] == base[group] and sorted(order) == sorted(g.cat_names)
        d = [U.actuator_distance(catalog[name], catalog[base[group]]) for name in order]
        assert d == sorted(d)
    rng = np.random.default_rng(0)
    for generation in range(n):
        drawn = {group: set() for group in g.cat_groups}
        for _ in range(400):
            for group, name in g._random_phase_picks(rng, generation).items():
                drawn[group].add(name)
        support = max(1, int(np.ceil(g._pick_scale(generation) * len(g.cat_names) - 1e-9)))
        for group in g.cat_groups:
            assert drawn[group] <= set(g._neighbours(group)[:support])
        sizes.append(support)
        if generation == 0:
            assert all(drawn[group] == {base[group]} for group in g.cat_groups)
    assert sizes == sorted(sizes) and sizes[-1] == len(g.cat_names)
    assert all(drawn[group] == set(g.cat_names) for group in g.cat_groups)
    # Generation 0 of a population carries the base robot's actuators in every design.
    for spec in g.generate_population(0).get_specs():
        assert all(getattr(spec, g.family.spec_fields[group]) == base[group] for group in g.cat_groups)


# ---------------------------------------------------------------------------
# The termination of the categorical searches
# ---------------------------------------------------------------------------


def _run_generations(g, count, fitness):
    g.toggle_late_start()
    g.sample_batch()
    for gen in range(count):
        g.generate_population(gen)
        g.update_with_fitness(fitness(gen))
        if g._terminated:
            return gen + 1
    return None


@pytest.mark.parametrize("cls", ["CatCMAESDesignGenerator", "CatOnlyCMAESDesignGenerator"])
def test_a_categorical_search_stops_on_tolfunhist_and_fine_tunes_on_its_mean(cls, tmp_path):
    g = _generator(getattr(U, cls), "biped", tmp_path, late_start=True, tolfunhist=0.5,
                   final_population="mean", max_cma_iter=1000)
    n = g._history_length()
    stopped = _run_generations(g, n + 5, lambda gen: [10.0 + 0.01 * i for i in range(8)])
    assert stopped == n and g._stop == ["tolfunhist"]
    pop = g.generate_population(n + 1)
    values, picks = g.optimised_design()
    assert all(getattr(s, g.family.spec_fields[k]) == v for s in pop.get_specs() for k, v in picks.items())
    assert len(set(pop.get_xml_files())) == 1 and len(pop.get_xml_files()) == 8
    assert g.search_statistics()["stop"] == "tolfunhist"


@pytest.mark.parametrize("cls", ["CatCMAESDesignGenerator", "CatOnlyCMAESDesignGenerator"])
def test_a_categorical_search_stops_at_maxiter_and_not_before(cls, tmp_path):
    g = _generator(getattr(U, cls), "biped", tmp_path, late_start=True, max_cma_iter=3)
    assert _run_generations(g, 6, lambda gen: list(np.random.default_rng(gen).normal(size=8))) == 3
    assert g._stop == ["maxiter"]
    free = _generator(getattr(U, cls), "biped", tmp_path / "free", late_start=True, max_cma_iter=1000,
                      tolx=0.02, tolfun=1e-9, tolfunhist=1e-9, tolcat=0.05)
    assert _run_generations(free, 4, lambda gen: list(np.random.default_rng(gen).normal(size=8))) is None


def test_tolx_needs_the_actuator_choice_settled(tmp_path):
    g = _generator(U.CatCMAESDesignGenerator, "biped", tmp_path, late_start=False, tolx=0.02, tolcat=0.05,
                   max_cma_iter=1000)
    g._es._sigma = 1e-6                                   # the continuous block converged
    assert g._continuous_converged(0.02) and not g._categorical_converged()
    assert "tolx" not in g._stop_reasons()
    q, qmin, k = g._es._q, g._es._qmin, g._es._K
    q[:] = qmin[:, None]
    q[np.arange(len(k)), 0] = 1.0 - (k - 1) * qmin       # every group at the margin's ceiling
    assert g._categorical_converged() and "tolx" in g._stop_reasons()
    g._tolcat = None
    assert "tolx" not in g._stop_reasons()


def test_the_termination_history_survives_a_checkpoint(tmp_path):
    g = _generator(U.CatCMAESDesignGenerator, "biped", tmp_path, late_start=True, max_cma_iter=1000)
    _run_generations(g, 3, lambda gen: [gen + 0.1 * i for i in range(8)])
    state = pickle.loads(pickle.dumps(g.get_state()))
    g2 = _generator(U.CatCMAESDesignGenerator, "biped", tmp_path / "again", late_start=True, max_cma_iter=1000)
    g2.load_state(state)
    assert g2._best_history == g._best_history and g2._last_range == g._last_range and g2._stop == g._stop


@pytest.mark.parametrize("family", ["biped", "quadruped"])
def test_the_fixed_generator_replays_a_searched_design(family, tmp_path):
    g = _generator(U.GrowingDesignDistCMAESDesignGenerator, family, tmp_path / "search")
    parameters = {"thigh_length_scale": 1.1, "shank_length_scale": 0.9, "thigh_bend_offset": 0.05, "knee_bend_offset": -0.1}
    g.emit_design(parameters, None, str(tmp_path / "searched.xml"))
    fixed = U.FixedDesignGenerator(parameters, family=family, output_dir=str(tmp_path / "fixed"))
    fixed.builder_kwargs = g.builder_kwargs
    pop = fixed.generate_population(0)
    assert len(pop.get_usd_files()) == 1
    assert open(pop.get_xml_files()[0]).read() == (tmp_path / "searched.xml").read_text()
    assert pop.get_parameters()[0]["thigh_length_scale"] == 1.1 and "knee_actuator" in pop.get_parameters()[0]
