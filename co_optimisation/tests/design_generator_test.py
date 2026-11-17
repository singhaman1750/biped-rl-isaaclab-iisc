"""Tests for co_optimisation.runners.usd_generator and the design generator behind it.

These are the persistent form of the gates that plans/design_generator_integration.md
section 15 marks as closable without a simulator.  ``MjcfConverter`` is stubbed to a
filesystem-only writer, and ``cma`` and ``cmaes`` are stubbed when absent, so nothing
here needs Isaac Sim.

They replace the URDF-era ``cmaes_design_generator_test.py``, every test of which
exercised the document mutation machinery that the move to the emitted MuJoCo document
retired.
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import pathlib
import pickle
import sys
import types
from dataclasses import dataclass, replace

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Stubs, installed before usd_generator is imported
# ---------------------------------------------------------------------------


class _StubMjcfConverterCfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _StubMjcfConverter:
    def __init__(self, cfg):
        self.cfg = cfg
        # reproduces mjcf_converter.py:61, which unpacks the basename split on "."
        # into exactly two names, so a second dot in the file name raises there
        stem, _ = os.path.basename(cfg.asset_path).split(".")
        os.makedirs(cfg.usd_dir, exist_ok=True)
        self.usd_path = os.path.join(cfg.usd_dir, cfg.usd_file_name)
        with open(self.usd_path, "w") as fh:
            fh.write(f"# stub conversion of {stem}\n")


class _StubCMAOptions(dict):
    def set(self, d):
        self.update(d)
        return self


class _StubCMAEvolutionStrategy:
    """Enough of pycma's surface for the generators. Picklable, which get_state needs."""

    def __init__(self, x0, sigma0, opts=None):
        self.N = len(x0)
        self._x0 = np.asarray(x0, dtype=float)
        self.sigma0 = sigma0
        self.opts = dict(opts or {})
        self.popsize = int(self.opts.get("popsize", 8))
        self._seed = int(self.opts.get("seed", 0))
        self._rng = np.random.default_rng(self._seed)

    def ask(self):
        return [np.clip(self._x0 + self.sigma0 * self._rng.standard_normal(self.N), 0, 1)
                for _ in range(self.popsize)]

    def tell(self, solutions, costs):
        assert len(solutions) == len(costs)

    def stop(self):
        return {}

    def pickle_dumps(self):
        return pickle.dumps(self)


@dataclass(frozen=True)
class _StubSolution:
    x: object = None
    z: object = None
    c: object = None


class _StubCatCMAwM:
    """Enough of cmaes.CatCMAwM's surface, matching its published signature."""

    Solution = _StubSolution

    def __init__(self, x_space=None, z_space=None, c_space=None, population_size=None,
                 mean=None, sigma=None, cov=None, cat_param=None, seed=None):
        self._x = np.asarray(x_space, dtype=float)
        # The continuous dimension, as cmaes/_catcmawm.py names it, which _check_dimension reads.
        self._Nco = len(x_space) if x_space is not None else 0
        self._c = list(c_space or [])
        self._popsize = int(population_size or 8)
        self._mean = (np.asarray(mean, dtype=float) if mean is not None
                      else self._x.mean(axis=1))
        self._sigma = float(sigma or 0.2)
        self._cat = (np.asarray(cat_param, dtype=float) if cat_param is not None
                     else np.array([[1.0 / n] * n for n in self._c]))
        for row, n in zip(self._cat, self._c):
            assert abs(row[:n].sum() - 1.0) < 1e-9
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    def population_size(self) -> int:
        return self._popsize

    def ask(self):
        lo, hi = self._x[:, 0], self._x[:, 1]
        x = np.clip(self._mean + self._sigma * self._rng.standard_normal(len(self._mean)),
                    lo, hi)
        c = np.zeros((len(self._c), max(self._c)), dtype=float)
        for row, n in enumerate(self._c):
            c[row, self._rng.choice(n, p=self._cat[row, :n])] = 1.0
        return _StubSolution(x=x, c=c)

    def tell(self, solutions):
        assert len(solutions) == self._popsize


def _stub_isaaclab_converter() -> None:
    if "isaaclab.sim.converters" in sys.modules:
        return
    MjcfConverterCfg = _StubMjcfConverterCfg
    MjcfConverter = _StubMjcfConverter
    mod = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    conv = types.ModuleType("isaaclab.sim.converters")
    conv.MjcfConverter = MjcfConverter
    conv.MjcfConverterCfg = MjcfConverterCfg
    sim.converters = conv
    mod.sim = sim
    sys.modules.update({"isaaclab": mod, "isaaclab.sim": sim,
                        "isaaclab.sim.converters": conv})


def _stub_cma() -> None:
    try:
        import cma  # noqa: F401
        return
    except ImportError:
        pass
    mod = types.ModuleType("cma")
    mod.CMAOptions = _StubCMAOptions
    mod.CMAEvolutionStrategy = _StubCMAEvolutionStrategy
    sys.modules["cma"] = mod


def _stub_cmaes() -> None:
    try:
        import cmaes  # noqa: F401
        return
    except ImportError:
        pass
    mod = types.ModuleType("cmaes")
    mod.CatCMAwM = _StubCatCMAwM
    sys.modules["cmaes"] = mod


_stub_isaaclab_converter()
_stub_cma()
_stub_cmaes()

# Loaded from its path rather than through ``co_optimisation.runners``, whose package
# __init__ imports the runner and with it tensordict and rsl_rl. usd_generator itself
# needs only numpy, cma, cmaes and design_generator, so a direct load keeps this suite
# runnable wherever those four are present.
# usd_generator imports make_filters from the package, whose __init__ needs the simulator.
_FILTERS_SPEC = importlib.util.spec_from_file_location(
    "co_optimisation.utils.filters",
    pathlib.Path(__file__).resolve().parents[1] / "co_optimisation/utils/filters.py",
)
for _name in ("co_optimisation", "co_optimisation.utils"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["co_optimisation.utils.filters"] = importlib.util.module_from_spec(_FILTERS_SPEC)
_FILTERS_SPEC.loader.exec_module(sys.modules["co_optimisation.utils.filters"])

_SPEC = importlib.util.spec_from_file_location(
    "usd_generator",
    pathlib.Path(__file__).resolve().parents[1] / "co_optimisation/runners/usd_generator.py",
)
U = importlib.util.module_from_spec(_SPEC)
sys.modules["usd_generator"] = U
_SPEC.loader.exec_module(U)

FAMILIES = ("biped", "quadruped")
N = 4


@pytest.fixture
def out_dir(tmp_path) -> str:
    return str(tmp_path / "copt")


def _make(cls, family, out_dir, **kw):
    gen = cls(family=family, num_individuals=N, output_dir=out_dir, **kw)
    gen.sample_batch()
    return gen


# ---------------------------------------------------------------------------
# The families and their guards
# ---------------------------------------------------------------------------


def test_unknown_family_is_refused(out_dir):
    with pytest.raises(ValueError, match="Unknown robot family"):
        U.RandomDesignGenerator(family="hexapod", num_individuals=1, output_dir=out_dir)


@pytest.mark.parametrize("family", FAMILIES)
def test_baseline_reproduces_the_shipped_parameter_set(family, out_dir):
    gen = U.RandomDesignGenerator(family=family, num_individuals=1, output_dir=out_dir)
    shipped = U.FAMILIES[family].module.generate.load_design_spec()
    assert gen.baseline == shipped


@pytest.mark.parametrize("family", FAMILIES)
def test_cma_search_is_the_lengths_and_the_pose(family, out_dir):
    for cls in (U.CMAESDesignGenerator, U.GrowingDesignDistCMAESDesignGenerator,
                U.CatCMAESDesignGenerator):
        gen = cls(family=family, num_individuals=N, output_dir=out_dir)
        assert gen.num_params == 4, cls.__name__
        # The geometry keys alone, a gain scale of CMAES_PARAM_KEYS being searched only where named.
        assert gen.param_keys == list(U.SPEC_PARAM_FIELDS)


# ---------------------------------------------------------------------------
# The gain scales, searched where named and applied to the actuators
# ---------------------------------------------------------------------------

GAIN_RANGES = {
    "hip_stiffness_scale": ("scale", (0.875, 1.125)),
    "knee_damping_scale": ("scale", (0.750, 1.250)),
}
# The ranges alone decide what is searched, so a search of the geometry and the gains names both.
GEOMETRY_AND_GAINS = {**{k: U.DEFAULT_PARAM_RANGES[k] for k in U.SPEC_PARAM_FIELDS}, **GAIN_RANGES}
CONTINUOUS =(U.CMAESDesignGenerator, U.GrowingDesignDistCMAESDesignGenerator, U.CatCMAESDesignGenerator)


def _catalogue_params(family, spec) -> dict[str, dict]:
    """The unscaled simulator parameters of every group of *spec*, as the catalogue states them."""
    return {g: a.sim_params() for g, a in zip(U.FAMILIES[family].groups, spec.actuators())}


def _phases(cls, family, out_dir, **kw):
    """A random phase population and then a search population of one generator."""
    gen = _make(cls, family, out_dir, late_start=True, late_start_prior_pop_size=N, late_start_it=4, seed=1, **kw)
    late = gen.generate_population(0)
    gen.toggle_late_start()
    gen.sample_batch()
    return gen, late, gen.generate_population(1)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", CONTINUOUS)
def test_named_gain_scales_are_searched_and_applied(cls, family, out_dir):
    gen, late, search = _phases(cls, family, out_dir, param_ranges=GEOMETRY_AND_GAINS)
    assert gen.param_keys == [*U.SPEC_PARAM_FIELDS, "hip_stiffness_scale", "knee_damping_scale"]
    assert gen.num_params == 6
    for pop in (late, search):
        drawn = []
        for spec, act, params in zip(pop.get_specs(), pop.get_actuator_params(), pop.get_parameters()):
            base = _catalogue_params(family, spec)
            assert act["hip"]["stiffness"] == round(base["hip"]["stiffness"] * params["hip_stiffness_scale"], 6)
            assert act["knee"]["damping"] == round(base["knee"]["damping"] * params["knee_damping_scale"], 6)
            # A scale left unnamed leaves its gain at the catalogue's.
            assert act["hip"]["damping"] == round(base["hip"]["damping"], 6)
            assert act["abad"]["stiffness"] == round(base["abad"]["stiffness"], 6)
            drawn.append(params["hip_stiffness_scale"])
        assert any(v != 1.0 for v in drawn)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", CONTINUOUS)
def test_no_named_gain_scale_keeps_the_catalogue_gains_and_the_geometry_draws(cls, family, out_dir):
    gen, late, search = _phases(cls, family, out_dir)
    for pop in (late, search):
        for spec, act in zip(pop.get_specs(), pop.get_actuator_params()):
            base = _catalogue_params(family, spec)
            for group, params in act.items():
                for key in ("stiffness", "damping"):
                    assert params[key] == round(base[group][key], 6)
    # Naming gain scales appends dimensions after the geometry, so the random phase's geometry draws are unchanged.
    _, named, _ = _phases(cls, family, str(pathlib.Path(out_dir) / "named"), param_ranges=GEOMETRY_AND_GAINS)
    for a, b in zip(late.get_parameters(), named.get_parameters()):
        assert all(a[k] == b[k] for k in U.SPEC_PARAM_FIELDS)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", [U.CMAESDesignGenerator, U.GrowingDesignDistCMAESDesignGenerator])
def test_the_optimised_gain_scales_replay_through_the_fixed_generator(cls, family, out_dir):
    gen, _, _ = _phases(cls, family, out_dir, param_ranges=GEOMETRY_AND_GAINS)
    values, picks = gen.optimised_design()
    assert {"hip_stiffness_scale", "knee_damping_scale"} <= set(values)
    record = gen.describe_design(values, picks)
    fixed = U.FixedDesignGenerator(values, family=family, output_dir=str(pathlib.Path(out_dir) / "fixed"))
    act = fixed.generate_population(0).get_actuator_params()[0]
    assert act["hip"]["stiffness"] == record["hip_actuator_stiffness"]
    assert act["knee"]["damping"] == record["knee_actuator_damping"]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", CONTINUOUS)
def test_the_fine_tune_applies_the_mean_gain_scales(cls, family, out_dir):
    gen, _, _ = _phases(cls, family, out_dir, param_ranges=GEOMETRY_AND_GAINS, final_population="mean")
    gen._terminated = True
    values, _ = gen.optimised_design()
    pop = gen.generate_population(2)
    for spec, act in zip(pop.get_specs(), pop.get_actuator_params()):
        base = _catalogue_params(family, spec)
        assert act["hip"]["stiffness"] == round(base["hip"]["stiffness"] * values["hip_stiffness_scale"], 6)


@pytest.mark.parametrize("cls", CONTINUOUS)
def test_a_gain_scale_of_a_missing_group_is_refused(cls, out_dir):
    with pytest.raises(ValueError, match="names no joint group"):
        cls(family="quadruped", num_individuals=N, output_dir=out_dir,
            param_ranges={"ankle_stiffness_scale": ("scale", (0.8, 1.2))})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", CONTINUOUS)
def test_a_checkpoint_of_another_dimension_is_refused(cls, family, out_dir):
    four = _make(cls, family, out_dir, seed=3)
    four.generate_population(0)
    state = pickle.loads(pickle.dumps(four.get_state()))
    six = _make(cls, family, str(pathlib.Path(out_dir) / "six"), seed=3, param_ranges=GEOMETRY_AND_GAINS)
    with pytest.raises(AssertionError, match="dimension mismatch"):
        six.load_state(state)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", CONTINUOUS)
def test_search_state_with_gain_scales_round_trips(cls, family, out_dir):
    gen = _make(cls, family, out_dir, seed=3, param_ranges=GEOMETRY_AND_GAINS)
    gen.generate_population(0)
    gen.update_with_fitness([float(i) for i in range(N)])
    gen.load_state(pickle.loads(pickle.dumps(gen.get_state())))
    assert gen.generate_population(1) is not None


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", CONTINUOUS)
def test_the_ranges_alone_decide_what_is_searched(cls, family, out_dir):
    lengths = {k: U.DEFAULT_PARAM_RANGES[k] for k in ("thigh_length_scale", "shank_length_scale")}
    gen = _make(cls, family, out_dir, param_ranges=lengths)
    assert gen.param_keys == ["thigh_length_scale", "shank_length_scale"]
    for params in gen.generate_population(0).get_parameters():
        assert not {"thigh_bend_offset", "knee_bend_offset"} & set(params)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_actuator_search_tunes_the_picks_and_their_gains_alone(family, out_dir):
    gen = _make(U.CatCMAESDesignGenerator, family, out_dir, param_ranges=GAIN_RANGES, late_start=True,
                late_start_prior_pop_size=N, late_start_it=4, seed=1)
    assert gen.param_keys == list(GAIN_RANGES) and gen._lean_max() == 0.0
    # The last generation of the random phase, which draws from the whole shelf.
    late = gen.generate_population(3)
    gen.toggle_late_start()
    gen.sample_batch()
    search = gen.generate_population(4)
    base = gen.baseline
    for pop in (late, search):
        for spec, act, params in zip(pop.get_specs(), pop.get_actuator_params(), pop.get_parameters()):
            # The baseline's geometry, emitted as drawn, every pick refused by no narrowed tolerance.
            assert (spec.thigh_length, spec.shank_length) == (base.thigh_length, base.shank_length)
            assert params["feasible"] and not set(U.SPEC_PARAM_FIELDS) & set(params)
            cat = _catalogue_params(family, spec)
            assert act["hip"]["stiffness"] == round(cat["hip"]["stiffness"] * params["hip_stiffness_scale"], 6)
            assert act["knee"]["damping"] == round(cat["knee"]["damping"] * params["knee_damping_scale"], 6)
    picks = {tuple(p[g] for g in gen.family.groups) for p in late.get_parameters()}
    assert len(picks) > 1
    gen.update_with_fitness([float(i) for i in range(N)])
    assert set(gen.search_statistics()["mean"]) == set(GAIN_RANGES)
    values, chosen = gen.optimised_design()
    assert set(values) == set(GAIN_RANGES) and set(chosen) == set(gen.family.groups)
    gen.load_state(pickle.loads(pickle.dumps(gen.get_state())))
    assert gen.generate_population(5) is not None


def test_ranges_are_typed_and_the_untyped_form_reads_as_scale(out_dir):
    gen = U.RandomDesignGenerator(family="biped", num_individuals=1, output_dir=out_dir,
                                  param_ranges={"thigh_length_scale": (0.8, 1.2)})
    assert gen.param_ranges["thigh_length_scale"] == ("scale", (0.8, 1.2))
    assert all(kind in U.PARAM_TYPES for kind, _ in gen.param_ranges.values())
    for bad in ({"joint_effort_scale": ("scale", (0.7, 1.3))},
                {"no_such_parameter": ("scale", (0.7, 1.3))},
                {"thigh_length_scale": ("multiplier", (0.7, 1.3))}):
        with pytest.raises(ValueError):
            U.RandomDesignGenerator(family="biped", num_individuals=1, output_dir=out_dir,
                                    param_ranges=bad)


# ---------------------------------------------------------------------------
# The population contract the runner depends upon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", [U.RandomDesignGenerator, U.CMAESDesignGenerator,
                                 U.GrowingDesignDistCMAESDesignGenerator,
                                 U.CatCMAESDesignGenerator])
def test_population_carries_every_field_the_runner_reads(cls, family, out_dir):
    gen = _make(cls, family, out_dir)
    pop = gen.generate_population(0)
    assert pop is not None
    usds = pop.get_usd_files()
    assert len(usds) == N
    assert all(os.path.exists(u) for u in usds)
    acts = pop.get_actuator_params()
    assert len(acts) == N
    groups = set(U.FAMILIES[family].groups)
    for a in acts:
        assert set(a) == groups
        for params in a.values():
            assert {"effort_limit", "velocity_limit",
                    "saturation_effort", "armature"} <= set(params)
    assert len(pop.get_link_length_params()) == N
    assert pop.get_root_body() == U.FAMILIES[family].root_body


@pytest.mark.parametrize("family", FAMILIES)
def test_emitted_document_has_exactly_one_dot_in_its_name(family, out_dir):
    gen = _make(U.CMAESDesignGenerator, family, out_dir)
    gen.generate_population(0)
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".xml"):
                assert f.count(".") == 1, f


# ---------------------------------------------------------------------------
# The in-place payload against the document the same specification emits
# ---------------------------------------------------------------------------


def _collision_box(xml: str, body: str):
    """(size, centre, mass, diagonal inertia, element) of one body's collision box or cylinder."""
    import xml.etree.ElementTree as ET

    def walk(el):
        for b in el.findall("body"):
            if b.get("name") == body:
                return b
            found = walk(b)
            if found is not None:
                return found
        return None

    b = walk(ET.fromstring(xml).find("worldbody"))
    assert b is not None, body
    geoms = [g for g in b.findall("geom") if g.get("class") == "collision"]
    assert len(geoms) == 1, (body, len(geoms))
    inertial = b.find("inertial")
    return (
        [float(v) for v in geoms[0].get("size").split()],
        [float(v) for v in (geoms[0].get("pos") or "0 0 0").split()],
        float(inertial.get("mass")),
        [float(v) for v in inertial.get("diaginertia").split()],
        b,
    )


_CAT_RANDOM_PHASE = {"late_start": True, "late_start_it": 10, "late_start_prior_pop_size": 8}


def _joint_of(body) -> str:
    """The joint a body hangs on, the importer naming a fixed one after the body."""
    joint = body.find("joint")
    return joint.get("name") if joint is not None else body.get("name")


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls,kw", [(U.RandomDesignGenerator, {}),
                                    (U.CatCMAESDesignGenerator, _CAT_RANDOM_PHASE)])
def test_link_payload_matches_the_emitted_document(cls, kw, family, out_dir):
    import xml.etree.ElementTree as ET

    gen = _make(cls, family, out_dir, **kw)
    pop = gen.generate_population(0)
    for spec, params in zip(pop.get_specs(), pop.get_link_length_params()):
        xml = gen.family.build_xml(spec)
        world = ET.fromstring(xml).find("worldbody")
        bodies = {b.get("name"): b for b in world.iter("body")}
        parents = {c.get("name"): b.get("name") for b in world.iter("body") for c in b.findall("body")}
        pos = lambda b: [float(v) for v in (b.get("pos") or "0 0 0").split()]
        for link, p in params.items():
            size, centre, mass, inertia, body = _collision_box(xml, link)
            if "half_extents" in p:
                assert np.allclose(size, p["half_extents"], atol=1e-6), link
                assert np.allclose(centre, p["box_pos"], atol=1e-6), link
            else:
                assert np.allclose(size, [p["radius"], p["half_length"]], atol=1e-6), link
            assert mass == pytest.approx(p["mass"], abs=1e-6), link
            # the document formats every float to six decimals (builder._fmt), so
            # the payload, which carries full precision, agrees only to that
            assert np.allclose(inertia, p["inertia"], atol=1e-6), link
            if "child_body" in p:
                child = bodies[p["child_body"]]
                assert _joint_of(child) == p["child_joint"], link
                assert np.allclose(pos(child), p["child_pos"], atol=1e-6), link
            if "parent_body" in p:
                assert parents[link] == p["parent_body"], link
                assert _joint_of(body) == p["parent_joint"], link
                assert np.allclose(pos(body), p["parent_pos"], atol=1e-6), link


@pytest.mark.parametrize("family", FAMILIES)
def test_the_payload_covers_everything_an_actuator_pick_moves(family, out_dir):
    import xml.etree.ElementTree as ET

    gen = _make(U.CatCMAESDesignGenerator, family, out_dir, **_CAT_RANDOM_PHASE)

    def bodies(spec):
        world = ET.fromstring(gen.family.build_xml(spec)).find("worldbody")
        return {b.get("name"): (b.get("pos"),
                                [sorted(e.attrib.items()) for e in b if e.tag in ("inertial", "geom")])
                for b in world.iter("body")}

    base = bodies(gen.baseline)
    moved = 0
    for group, field in gen.family.spec_fields.items():
        for name in gen.cat_names:
            spec = gen.family.module.generate.DesignSpec.from_dict({**gen.baseline.to_dict(), field: name})
            try:
                picked = bodies(spec)
            except ValueError:
                continue
            payload = gen.family.links(spec)
            placed = {p["child_body"] for p in payload.values() if "child_body" in p}
            placed |= {k for k, p in payload.items() if "parent_body" in p}
            for body, (where, content) in picked.items():
                if where != base[body][0]:
                    assert body in placed, (group, name, body)
                    moved += 1
                if content != base[body][1]:
                    assert body in payload, (group, name, body)
                    moved += 1
    assert moved > 0


@pytest.mark.parametrize("family", FAMILIES)
def test_link_payload_scales_with_the_length(family, out_dir):
    gen = U.RandomDesignGenerator(family=family, num_individuals=1, output_dir=out_dir)
    base = gen.family.links(gen.baseline)
    longer = gen.family.links(replace(gen.baseline,
                                      thigh_length=gen.baseline.thigh_length * 1.2))
    thigh = next(k for k in base if k.startswith("thigh_"))
    assert longer[thigh]["mass"] == pytest.approx(base[thigh]["mass"] * 1.2)
    assert longer[thigh]["half_extents"][2] == pytest.approx(
        base[thigh]["half_extents"][2] * 1.2)
    assert np.linalg.norm(longer[thigh]["child_pos"]) == pytest.approx(
        np.linalg.norm(base[thigh]["child_pos"]) * 1.2)
    shank = next(k for k in base if k.startswith("shank_"))
    assert longer[shank] == base[shank]


# ---------------------------------------------------------------------------
# for_isaaclab, which is what the converter consumes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_isaaclab_document_is_a_description_and_not_a_scene(family, out_dir):
    gen = U.RandomDesignGenerator(family=family, num_individuals=1, output_dir=out_dir)
    scene = gen.family.module.builder.build_xml(gen.baseline)
    description = gen.family.build_xml(gen.baseline)
    for tag in ("floor", "<site", "keyframe", "<sensor", "<asset"):
        assert tag in scene, tag
        assert tag not in description, tag
    root = U.FAMILIES[family].root_body
    assert f'<body name="{root}" pos="0 0 0">' in description


@pytest.mark.parametrize("family", FAMILIES)
def test_every_body_carries_a_mass(family, out_dir):
    """A massless body is legal in MuJoCo and is not legal in a PhysX articulation."""
    import xml.etree.ElementTree as ET

    gen = U.RandomDesignGenerator(family=family, num_individuals=1, output_dir=out_dir)
    root = ET.fromstring(gen.family.build_xml(gen.baseline)).find("worldbody")
    bodies = list(root.iter("body"))
    assert bodies
    for b in bodies:
        inertial = b.find("inertial")
        assert inertial is not None, b.get("name")
        assert float(inertial.get("mass")) > 0.0, b.get("name")


# ---------------------------------------------------------------------------
# The actuator catalogue
# ---------------------------------------------------------------------------

IDENTIFIED = {
    "biped": {
        "TRON1-ABAD": dict(effort_limit=60.0, velocity_limit=23.0, saturation_effort=402.0,
                           armature=6.9e-5 * 81, stiffness=55.0, damping=13.5,
                           friction_static=0.3, friction_dynamic=0.02, activation_vel=0.1),
        "TRON1-HIP": dict(effort_limit=60.0, velocity_limit=20.0, saturation_effort=443.0,
                          armature=9.4e-5 * 81, stiffness=80.0, damping=13.0,
                          friction_static=0.3, friction_dynamic=0.02, activation_vel=0.1),
        "TRON1-KNEE": dict(effort_limit=90.0, velocity_limit=14.0, saturation_effort=560.0,
                           armature=1.5e-4 * 81, stiffness=60.0, damping=4.0,
                           friction_static=0.8, friction_dynamic=0.02, activation_vel=0.1),
        "TRON1-ANKLE": dict(effort_limit=40.0, velocity_limit=20.0, saturation_effort=402.0,
                            armature=6.9e-5 * 81, stiffness=10.0, damping=0.5,
                            friction_static=0.1, friction_dynamic=0.02, activation_vel=0.1),
    },
    "quadruped": {
        "A1-ABAD": dict(effort_limit=23.622511, velocity_limit=30.0, saturation_effort=158.0,
                        armature=0.01, stiffness=40.0, damping=1.2,
                        friction_static=0.2, friction_dynamic=0.02, activation_vel=0.1),
        "A1-HIP": dict(effort_limit=23.622511, velocity_limit=30.0, saturation_effort=158.0,
                       armature=0.01, stiffness=40.0, damping=1.1,
                       friction_static=0.2, friction_dynamic=0.02, activation_vel=0.1),
        "A1-KNEE": dict(effort_limit=35.238, velocity_limit=20.0, saturation_effort=236.0,
                        armature=0.01, stiffness=50.0, damping=0.9,
                        friction_static=0.2, friction_dynamic=0.02, activation_vel=0.1),
    },
}


@pytest.mark.parametrize("family", FAMILIES)
def test_identified_entries_reproduce_their_configuration(family):
    by_name = U.FAMILIES[family].module.actuator.ACTUATOR_BY_NAME
    for name, want in IDENTIFIED[family].items():
        got = by_name[name].sim_params()
        assert set(got) == set(want), name
        for key, value in want.items():
            assert got[key] == pytest.approx(value, abs=1e-9), f"{name}.{key}"


@pytest.mark.parametrize("family", FAMILIES)
def test_unstated_simulator_fields_fall_back_rather_than_fabricate(family):
    from design_generator.common.actuator_catalog import CatalogActuator, DEFAULT_STALL_RATIO

    bare = CatalogActuator(name="bare", peak_torque=10.0, peak_speed=20.0,
                           mass=0.5, cost=1.0)
    got = bare.sim_params()
    assert got["effort_limit"] == 10.0
    assert got["velocity_limit"] == 20.0
    assert got["saturation_effort"] == pytest.approx(10.0 * DEFAULT_STALL_RATIO)
    assert got["armature"] == 0.0
    # friction and the two gains are omitted rather than invented
    assert not {"friction_static", "stiffness", "damping"} & set(got)


# ---------------------------------------------------------------------------
# The categorical search
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_categorical_search_picks_by_name_from_the_catalogue(family, out_dir):
    gen = _make(U.CatCMAESDesignGenerator, family, out_dir, seed=1)
    pop = gen.generate_population(0)
    names = {a.name for a in gen.family.catalog()}
    for spec in pop.get_specs():
        for group in gen.cat_groups:
            assert getattr(spec, gen.family.spec_fields[group]) in names


# Every entry since the fill of 2026-11-12, plans/design_filtering.md section 7, the identified
# TRON1-* and A1-* entries giving their simulator parameters to the others.
SIM_COMPLETE = {family: [a.name for a in U.FAMILIES[family].full_catalog()] for family in ("biped", "quadruped")}


@pytest.mark.parametrize("family", FAMILIES)
def test_the_shelf_is_the_simulation_complete_catalogue(family, out_dir):
    from design_generator.common.actuator_catalog import SIM_PARAM_KEYS
    shelf = U.FAMILIES[family].catalog()
    assert [a.name for a in shelf] == SIM_COMPLETE[family]
    for a in shelf:
        params = a.sim_params()
        assert set(params) >= set(SIM_PARAM_KEYS) and params["armature"] > 0.0
    # The shelf was a strict subset of the catalogue until the fill of 2026-11-12 completed every entry.
    assert len(U.FAMILIES[family].full_catalog()) == len(shelf)
    gen = U.CatCMAESDesignGenerator(family=family, num_individuals=N, output_dir=out_dir)
    assert gen.cat_names == SIM_COMPLETE[family]
    assert gen.cat_groups == U.FAMILIES[family].groups


@pytest.mark.parametrize("family", FAMILIES)
def test_the_shortlist_arguments_are_gone(family, out_dir):
    for kw in ({"catalog_names": ["A1-ABAD", "A1-HIP"]}, {"actuator_groups": ("abad",)}):
        with pytest.raises(TypeError):
            U.CatCMAESDesignGenerator(family=family, num_individuals=N, output_dir=out_dir, **kw)


# ---------------------------------------------------------------------------
# Feasibility and the checkpoint round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family,field,value", [("biped", "thigh_length", 0.01),
                                                ("quadruped", "thigh_length", 0.01)])
def test_an_infeasible_specification_falls_back_rather_than_raising(family, field, value,
                                                                   out_dir):
    # A thigh of 1 cm puts the foot far from under the hip, which check_nominal_pose refuses. The
    # trunk width was this test's example until Rule 5 made the builder set it, so that no width is infeasible.
    gen = U.RandomDesignGenerator(family=family, num_individuals=1, output_dir=out_dir)
    spec = replace(gen.baseline, **{field: value})
    xml = gen._build_xml(spec)
    assert xml.startswith("<mujoco")
    assert U.FAMILIES[family].root_body in xml
    assert gen._last_emitted is not spec     # the refused design was replaced by the last feasible one


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", [U.CMAESDesignGenerator,
                                 U.GrowingDesignDistCMAESDesignGenerator,
                                 U.CatCMAESDesignGenerator])
def test_search_state_round_trips(cls, family, out_dir):
    gen = _make(cls, family, out_dir, seed=3)
    gen.generate_population(0)
    gen.update_with_fitness([float(i) for i in range(N)])
    state = pickle.loads(pickle.dumps(gen.get_state()))
    gen.load_state(state)
    assert gen.generate_population(1) is not None


# ---------------------------------------------------------------------------
# The nominal pose, plans/co_optimisation_audit_implementation.md phases 2 to 5
# ---------------------------------------------------------------------------

STATIC_ASSET = {
    "biped": "environments/environments/assets/mjcf/solefoot/tron1/base_robot.xml",
    "quadruped": "environments/environments/assets/mjcf/quadruped/quadruped.xml",
}
REPO = pathlib.Path(__file__).resolve().parents[2]


def _world(xml: str, qpos: dict | None = None) -> dict:
    """World position of every body origin at the given hinge angles, root at the origin."""
    import xml.etree.ElementTree as ET

    def qmul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    def rot(q, v):
        return qmul(qmul(q, np.array([0.0, *v])), q * np.array([1, -1, -1, -1]))[1:]

    out = {}

    def walk(el, p, q):
        for b in el.findall("body"):
            pos = np.array([float(v) for v in (b.get("pos") or "0 0 0").split()])
            quat = np.array([float(v) for v in (b.get("quat") or "1 0 0 0").split()])
            pw, qw = p + rot(q, pos), qmul(q, quat / np.linalg.norm(quat))
            j = b.find("joint")
            if j is not None and qpos and j.get("name") in qpos:
                ax, a = np.array([float(v) for v in j.get("axis").split()]), qpos[j.get("name")]
                qw = qmul(qw, np.array([np.cos(a / 2), *(np.sin(a / 2) * ax)]))
            out[b.get("name")] = pw
            walk(b, pw, qw)

    walk(ET.fromstring(xml).find("worldbody"), np.zeros(3), np.array([1.0, 0, 0, 0]))
    return out


@pytest.mark.parametrize("family", FAMILIES)
def test_design_spec_round_trips_through_a_dict(family, out_dir):
    gen = U.RandomDesignGenerator(family=family, num_individuals=1, output_dir=out_dir)
    cls = type(gen.baseline)
    assert cls.from_dict(gen.baseline.to_dict()) == gen.baseline
    with pytest.raises(KeyError, match="unknown DesignSpec fields"):
        cls.from_dict({**gen.baseline.to_dict(), "knee_fold": 1.0})


@pytest.mark.parametrize("family", FAMILIES)
def test_the_baseline_emits_the_committed_asset(family, out_dir):
    gen = U.RandomDesignGenerator(family=family, num_individuals=1, output_dir=out_dir)
    assert gen.family.build_xml(gen.baseline) == (REPO / STATIC_ASSET[family]).read_text()


@pytest.mark.parametrize("family", FAMILIES)
def test_the_isaaclab_document_moves_the_fold_into_the_frames(family, out_dir):
    gen = U.RandomDesignGenerator(family=family, num_individuals=1, output_dir=out_dir)
    b, root = gen.family.module.builder, gen.family.root_body
    qpos = {}
    if family == "quadruped":
        for leg in b.LEG_NAMES:
            qpos[f"{leg}_thigh_joint"] = b.THIGH_BEND_RAD
            qpos[f"{leg}_calf_joint"] = b.SHANK_BEND_RAD
    mujoco = _world(b.build_xml(gen.baseline), qpos)
    isaac = _world(gen.family.build_xml(gen.baseline))
    assert max(np.abs((mujoco[k] - mujoco[root]) - (isaac[k] - isaac[root])).max()
               for k in isaac) < 1e-6


@pytest.mark.parametrize("family", FAMILIES)
def test_the_chain_is_the_emitted_document(family, out_dir):
    gen = U.RandomDesignGenerator(family=family, num_individuals=8, output_dir=out_dir)
    fam = gen.family
    for idx in range(8):
        spec, _ = gen._generate_individual(0, idx)
        world = _world(fam.build_xml(spec))
        rec = fam.nominal(spec)
        foot, hip = fam.foot_bodies[0], fam.hip_bodies[0]
        assert world[foot][0] - world[hip][0] == pytest.approx(rec["dx"], abs=2e-6)
        assert -world[foot][2] == pytest.approx(rec["depth"], abs=2e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_solved_hip_puts_the_foot_under_the_hip(family, out_dir):
    gen = U.CMAESDesignGenerator(family=family, num_individuals=N, output_dir=out_dir)
    rng = np.random.default_rng(7)
    for _ in range(50):
        parameters = {k: float(rng.uniform(lo, hi)) for k, (_, (lo, hi)) in gen.param_ranges.items()}
        parameters["thigh_bend_offset"] = 0.0
        spec = gen._spec_from_parameters(parameters)
        # the hip's rounding to 3 dp remains
        assert gen.family.nominal(spec)["dx"] == pytest.approx(gen._dx_target, abs=1e-3)
        if gen.family.has_ankle:
            total = spec.thigh_bend_offset + spec.knee_bend_offset + spec.ankle_bend_offset
            assert total == pytest.approx(0.0, abs=1e-12)
        else:
            assert not hasattr(spec, "ankle_bend_offset")


@pytest.mark.parametrize("family", FAMILIES)
def test_a_refused_pose_falls_back_to_the_last_feasible_design(family, out_dir):
    gen = U.CMAESDesignGenerator(family=family, num_individuals=N, output_dir=out_dir)
    good = gen._spec_from_parameters({"thigh_length_scale": 1.0, "shank_length_scale": 1.0,
                                      "thigh_bend_offset": 0.05, "knee_bend_offset": 0.1})
    lean = {"thigh_bend_offset": good.thigh_bend_offset + 0.6}
    if gen.family.has_ankle:
        lean["ankle_bend_offset"] = -(lean["thigh_bend_offset"] + good.knee_bend_offset)
    bad = type(good).from_dict({**good.to_dict(), **lean})
    # a 0.6 rad lean puts the foot far outside the tolerance the 0.10 rad draw sets
    with pytest.raises(ValueError, match="under the hip"):
        U.check_nominal_pose(gen.family, bad, gen._dx_target, gen._lean_max())
    assert gen._build_xml(bad) == gen.family.build_xml(gen.baseline)
    assert gen._last_emitted is gen.baseline
    gen._build_xml(good)
    assert gen._build_xml(bad) == gen.family.build_xml(good)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_population_records_the_nominal_pose(family, out_dir):
    gen = _make(U.GrowingDesignDistCMAESDesignGenerator, family, out_dir, late_start=True,
                late_start_it=25, late_start_prior_pop_size=N)
    pop = gen.generate_population(3)
    nominal = pop.get_nominal()
    assert nominal is not None and len(nominal) == N
    lo, hi = gen.family.height_corridor
    for spec, rec in zip(pop.get_specs(), nominal):
        assert rec == gen.family.nominal(spec)
        assert lo < rec["height"] < hi


@pytest.mark.parametrize("family", FAMILIES)
def test_joint_rot_is_the_documents_body_quat(family, out_dir):
    import xml.etree.ElementTree as ET
    gen = _make(U.CMAESDesignGenerator, family, out_dir, seed=5)
    pop = gen.generate_population(0)
    for spec, links in zip(pop.get_specs(), pop.get_link_length_params()):
        root = ET.fromstring(gen.family.build_xml(spec)).find("worldbody")
        by_joint = {}
        for b in root.iter("body"):
            j = b.find("joint")
            by_joint[j.get("name") if j is not None else b.get("name")] = b
        seen = 0
        for entry in links.values():
            for joint, quat in entry.get("joint_rot", {}).items():
                doc = [float(v) for v in by_joint[joint].get("quat").split()]
                assert np.allclose(doc, quat, atol=1e-6), joint
                seen += 1
        assert seen == (6 if family == "biped" else 8)


@pytest.mark.parametrize("family", FAMILIES)
def test_zero_width_pose_ranges_search_the_lengths_alone(family, out_dir):
    # The lengths are named, the ranges alone deciding what is searched.
    zero = {"thigh_length_scale": U.DEFAULT_PARAM_RANGES["thigh_length_scale"],
            "shank_length_scale": U.DEFAULT_PARAM_RANGES["shank_length_scale"],
            "thigh_bend_offset": ("offset", (0.0, 0.0)), "knee_bend_offset": ("offset", (0.0, 0.0))}
    gen = _make(U.CMAESDesignGenerator, family, out_dir, param_ranges=zero)
    for spec in gen.generate_population(0).get_specs():
        assert spec.knee_bend_offset == 0.0
        U.check_nominal_pose(gen.family, spec, gen._dx_target, gen._lean_max())


@pytest.mark.parametrize("family", FAMILIES)
def test_the_categorical_random_phase_draws_every_actuator(family, out_dir):
    # 256 draws, since the fill of 2026-11-12 widened the shelf from four entries to fifteen, the
    # chance of a given entry missing 64 uniform draws being (14/15)^64, about 1.2 per cent. The last
    # generation of the random phase, 9 of 10, since the actuator choice widens from the base robot's
    # to the whole shelf over the phase (plans/design_filtering.md section 11.2).
    gen = _make(U.CatCMAESDesignGenerator, family, out_dir, late_start=True,
                late_start_it=10, late_start_prior_pop_size=256)
    specs = gen.generate_population(9).get_specs()
    assert len(specs) == 256
    for group in gen.cat_groups:
        assert {getattr(s, gen.family.spec_fields[group]) for s in specs} == set(gen.cat_names)


# ---------------------------------------------------------------------------
# The search is centred on the initial design of the baseline MJCF
# ---------------------------------------------------------------------------

QUADRUPED_RANGES = {
    "thigh_length_scale": ("scale", (0.35, 2.00)),
    "shank_length_scale": ("scale", (0.35, 2.00)),
    "thigh_bend_offset": ("offset", (-0.50, 0.50)),
    "knee_bend_offset": ("offset", (-0.5, 0.5)),
}


@pytest.mark.parametrize("family", FAMILIES)
def test_a_symmetric_box_widens_as_it_did_about_its_midpoint(family, out_dir):
    gen = _make(U.GrowingDesignDistCMAESDesignGenerator, family, out_dir)
    for scale in (0.05, 0.4, 1.0):
        got = gen._sample_parameter_v2(np.random.default_rng(7), scale)
        rng, old = np.random.default_rng(7), {}
        for k, (_, (lo, hi)) in gen.param_ranges.items():
            mean = (lo + hi) / 2
            old[k] = float(rng.uniform((lo - mean) * scale + mean, (hi - mean) * scale + mean))
        assert got == old
    assert np.array_equal(gen._es.mean, np.full(gen.num_params, 0.5))


@pytest.mark.parametrize("family", FAMILIES)
def test_an_asymmetric_box_widens_from_the_initial_design_to_both_bounds(family, out_dir):
    gen = _make(U.GrowingDesignDistCMAESDesignGenerator, family, out_dir, param_ranges=QUADRUPED_RANGES)
    rng = np.random.default_rng(0)
    first = [gen._sample_parameter_v2(rng, 0.05) for _ in range(2000)]
    last = [gen._sample_parameter_v2(rng, 1.0) for _ in range(2000)]
    for k in ("thigh_length_scale", "shank_length_scale"):
        v = [p[k] for p in first]
        assert 1 - 0.05 * 0.65 <= min(v) < 1 < max(v) <= 1 + 0.05 * 1.0
        v = [p[k] for p in last]
        assert 0.35 <= min(v) < 0.37 and 1.98 < max(v) <= 2.0
    for k in ("thigh_bend_offset", "knee_bend_offset"):
        assert max(abs(p[k]) for p in first) <= 0.05 * 0.5


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("cls", [U.CMAESDesignGenerator, U.CatCMAESDesignGenerator])
def test_cma_es_starts_at_the_initial_design(cls, family, out_dir):
    gen = cls(family=family, num_individuals=N, output_dir=out_dir, param_ranges=QUADRUPED_RANGES)
    mean = gen._es.mean if cls is U.CMAESDesignGenerator else gen._es._mean
    assert np.allclose(mean, [0.65 / 1.65, 0.65 / 1.65, 0.5, 0.5])
    assert np.allclose(list(gen._denormalise(mean).values()), [1.0, 1.0, 0.0, 0.0])


def test_a_range_excluding_the_initial_design_is_refused(out_dir):
    with pytest.raises(ValueError, match="excludes the initial design"):
        U.RandomDesignGenerator(family="biped", num_individuals=1, output_dir=out_dir,
                                param_ranges={"thigh_length_scale": ("scale", (1.1, 1.5))})


# ---------------------------------------------------------------------------
# The housing clearance and the trunk width, Rules 5 and 6 of GEOMETRY_RULES.md
# ---------------------------------------------------------------------------


def _picked(family, abad, hip, knee):
    """The family's baseline with the three housing actuators of a leg cluster replaced."""
    fam = U.FAMILIES[family]
    fields = fam.spec_fields
    return replace(fam.baseline(), **{fields["abad"]: abad, fields["hip"]: hip, fields["knee"]: knee})


def _trunk(family, spec):
    """(a, b, c), mass and inertia of the trunk the builder emits for *spec*."""
    b = U.FAMILIES[family].module.builder
    acts = spec.actuators()
    return b.trunk_body(spec, *acts[:2])


@pytest.mark.parametrize("family", FAMILIES)
def test_every_pick_of_the_leg_cluster_builds_without_a_housing_contact(family):
    """No abduction, hip and knee pick of the catalogue is infeasible or puts two bodies in contact.

    This is the defect of run copt_moral/2026-11-12_06-55-42, where a narrow hip housing let the knee
    housing reach back into the abduction housing and the episode ended at its first step. The pose is
    the zero pose of the Isaac Lab document with the base lifted clear, as the run was audited at.
    """
    mujoco = pytest.importorskip("mujoco")
    fam = U.FAMILIES[family]
    names = [a.name for a in fam.full_catalog()]
    for abad, hip, knee in itertools.product(names, repeat=3):
        xml = fam.module.builder.build_xml(_picked(family, abad, hip, knee), for_isaaclab=True)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        data.qpos[2], data.qpos[3] = 2.0, 1.0
        mujoco.mj_forward(model, data)
        assert data.ncon == 0, (abad, hip, knee)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_hip_clears_the_knee_by_the_margin_and_keeps_the_former_offset_otherwise(family):
    fam = U.FAMILIES[family]
    b = fam.module.builder
    names = [a.name for a in fam.full_catalog()]
    for abad, hip, knee in itertools.product(names, repeat=3):
        spec = _picked(family, abad, hip, knee)
        act_abad, act_hip, act_knee = spec.actuators()[:3]
        if family == "biped":
            lay = b._cluster_layout("R", spec, act_abad, act_hip, act_knee)
            hip_x, knee_x, direction = lay["hip_pos_rel"][0], lay["hip_pos_rel"][0] + lay["knee_pos_rel"][0], -1.0
            former = -(lay["hl_abad"] + lay["r_hip"] + spec.clearance_abadAct2hipAct_x)
        else:
            lay = b._cluster_layout("RR", spec, act_abad, act_hip, act_knee)
            hip_x, knee_x, direction = lay["hip_pos"][0], lay["hip_pos"][0] + lay["knee_pos"][0], b.SIDE_SIGN_X["RR"]
            former = direction * (lay["hl_abad"] + lay["r_hip"] + spec.clearance_abadAct2hipAct_x)
        gap = direction * knee_x - lay["hl_abad"] - lay["r_knee"]
        assert gap >= b.HOUSING_MARGIN - 1e-12, (abad, hip, knee, gap)
        if abs(gap - b.HOUSING_MARGIN) > 1e-12:
            assert hip_x == former, (abad, hip, knee)    # the knee cleared, so the offset is the former one


def test_the_narrow_hip_of_the_audited_run_now_clears_by_the_margin():
    b = U.FAMILIES["biped"].module.builder
    spec = _picked("biped", "TRON1-ABAD", "ODrive-D5065-DD", "TRON1-KNEE")
    lay = b._cluster_layout("R", spec, *spec.actuators()[:3])
    knee_x = lay["hip_pos_rel"][0] + lay["knee_pos_rel"][0]
    assert -knee_x - lay["hl_abad"] - lay["r_knee"] == pytest.approx(b.HOUSING_MARGIN)


@pytest.mark.parametrize("family", FAMILIES)
def test_the_trunk_width_follows_the_abduction_actuator_and_not_the_spec(family):
    fam = U.FAMILIES[family]
    b = fam.module.builder
    base = fam.baseline()
    (a, w, c), _, _ = _trunk(family, base)
    assert w == pytest.approx(base.trunk_width)                       # the baseline is reproduced
    assert _trunk(family, replace(base, trunk_width=0.01)) == _trunk(family, base)   # the field is inert
    for act in fam.full_catalog():
        spec = _picked(family, act.name, *(getattr(base, fam.spec_fields[g]) for g in ("hip", "knee")))
        width = _trunk(family, spec)[0][1]
        assert width >= 2.0 * act.diameter + b.TRUNK_WIDTH_OFFSET - 1e-12
        if family == "quadruped":
            assert width == pytest.approx(2.0 * act.diameter + b.TRUNK_WIDTH_OFFSET)
            assert _trunk(family, spec)[0][2] == pytest.approx(act.diameter)    # Rule 4


def test_a_narrow_trunk_still_fits_the_widest_hip_of_the_biped():
    """The one pair of the shelf the abduction diameter alone left infeasible, ODrive abduction with a B1 hip."""
    b = U.FAMILIES["biped"].module.builder
    spec = _picked("biped", "ODrive-D5065-DD", "Unitree-B1", "TRON1-KNEE")
    act_abad, act_hip = spec.actuators()[:2]
    assert _trunk("biped", spec)[0][1] > 2.0 * act_abad.diameter + b.TRUNK_WIDTH_OFFSET
    b.check_geometry(spec, act_abad, act_hip)
    assert b.build_xml(spec, for_isaaclab=True).startswith("<mujoco")


@pytest.mark.parametrize("family", FAMILIES)
def test_the_link_dict_carries_the_trunk_and_every_housing_placement(family):
    fam = U.FAMILIES[family]
    b = fam.module.builder
    spec = _picked(family, "ODrive-D5065-DD", "ODrive-D5065-DD", "Unitree-B1")
    links = fam.links(spec)
    (a, w, c), mass, inertia = _trunk(family, spec)
    trunk = links[fam.root_body]
    assert trunk["half_extents"] == pytest.approx([a / 2.0, w / 2.0, c / 2.0])
    assert trunk["mass"] == pytest.approx(mass) and trunk["inertia"] == pytest.approx(list(inertia))
    leg = "R" if family == "biped" else "FR"
    acts = spec.actuators()
    if family == "biped":
        lay = b._cluster_layout(leg, spec, *acts[:3])
        keys = {"abad": "abad_pos", "hip": "hip_pos_rel", "knee": "knee_pos_rel", "thigh": "thigh_pos_rel"}
    else:
        lay = b._cluster_layout(leg, spec, *acts[:3])
        keys = {"abad": "abad_pos", "hip": "hip_pos", "knee": "knee_pos", "thigh": "thigh_pos"}
    for part, name in (("abad", f"abad_{leg}_act"), ("hip", f"hip_{leg}_act"), ("knee", f"knee_{leg}_act")):
        assert links[name]["parent_pos"] == pytest.approx([float(v) for v in lay[keys[part]]])
        assert links[name]["radius"] == pytest.approx(lay[f"r_{part}"])
        assert links[name]["half_length"] == pytest.approx(lay[f"hl_{part}"])
    assert links[f"thigh_{leg}_Link"]["parent_pos"] == pytest.approx([float(v) for v in lay[keys["thigh"]]])
