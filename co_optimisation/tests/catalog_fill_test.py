"""Tests of the catalogue fill of scripts/tools/fill_actuator_sim_params.py, plans/design_filtering.md section 7.

Compares the installed design_generator, the filled one, with the catalogues before the fill, whose
directory ORIGINAL_DESIGN_GENERATOR names, so it runs after the fill and before the reinstall is
forgotten. Runs without the simulator.

    ORIGINAL_DESIGN_GENERATOR=/path/to/unfilled/design_generator /isaac-sim/python.sh -m pytest co_optimisation/tests/catalog_fill_test.py
"""
import importlib
import json
import os
import pathlib
import sys

import pytest

from design_generator import biped, quadruped
from design_generator.common.actuator_catalog import SIM_PARAM_KEYS, is_sim_complete

ORIGINAL = os.environ.get("ORIGINAL_DESIGN_GENERATOR")
FAMILIES = {"biped": (biped, "TRON1-"), "quadruped": (quadruped, "A1-")}
COPIED = ("stiffness", "damping", "friction_static", "friction_dynamic", "activation_vel", "armature")


def _original(family):
    """The family's package loaded from the unfilled tree, under its own name so both coexist."""
    root = pathlib.Path(ORIGINAL).resolve()
    saved = {k: v for k, v in sys.modules.items() if k == "design_generator" or k.startswith("design_generator.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(root.parent))
    try:
        module = importlib.import_module(f"design_generator.{family}")
        catalog = {a.name: a for a in module.actuator.ACTUATOR_CATALOG}
        xml = module.builder.build_xml(module.generate.load_design_spec(), for_isaaclab=True)
    finally:
        sys.path.pop(0)
        for k in [k for k in sys.modules if k == "design_generator" or k.startswith("design_generator.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return catalog, xml


@pytest.mark.parametrize("family", FAMILIES)
def test_every_entry_is_simulation_complete(family):
    module, _ = FAMILIES[family]
    assert all(is_sim_complete(a) for a in module.actuator.ACTUATOR_CATALOG)
    assert len(module.actuator.sim_actuator_catalog()) == len(module.actuator.ACTUATOR_CATALOG)


@pytest.mark.parametrize("family", FAMILIES)
def test_filled_entries_copy_their_nearest_identified_entry(family):
    module, prefix = FAMILIES[family]
    catalog = {a.name: a for a in module.actuator.ACTUATOR_CATALOG}
    path = pathlib.Path(module.__file__).parent / "catalogs" / "actuators.json"
    for raw in json.load(open(path))["actuators"]:
        if raw["name"].startswith(prefix):
            continue
        ref = raw["notes"].split("copied from ")[1].split(",")[0]
        assert ref.startswith(prefix)
        mine, theirs = catalog[raw["name"]].sim_params(), catalog[ref].sim_params()
        for key in COPIED:
            assert mine[key] == pytest.approx(theirs[key], rel=1e-9), (raw["name"], key)
        assert mine["effort_limit"] == pytest.approx(raw["peak_torque"])


@pytest.mark.skipif(ORIGINAL is None, reason="ORIGINAL_DESIGN_GENERATOR not set")
@pytest.mark.parametrize("family", FAMILIES)
def test_identified_entries_and_documents_unchanged(family):
    module, prefix = FAMILIES[family]
    before, xml_before = _original(family)
    after = {a.name: a for a in module.actuator.ACTUATOR_CATALOG}
    assert set(before) == set(after)
    for name, entry in before.items():
        if name.startswith(prefix):
            assert after[name].sim_params() == entry.sim_params(), name
        for key in ("peak_torque", "peak_speed", "mass"):
            assert getattr(after[name], key) == getattr(entry, key), (name, key)
        assert set(entry.sim_params()) <= set(SIM_PARAM_KEYS)
    assert module.builder.build_xml(module.generate.load_design_spec(), for_isaaclab=True) == xml_before
