"""Emit each family's default MuJoCo document from its own parameter set.

Run after editing a family's variable_params.json or constant_params.json, or
after changing a catalog entry that its variable_params.json names. The output
is committed, so that environments/assets/config/*_identified_cfg.py can spawn
without importing the generator.

    python scripts/tools/generate_default_mjcf.py
"""

from __future__ import annotations

import pathlib

from design_generator import biped, quadruped

ASSETS = pathlib.Path(__file__).resolve().parents[2] / "environments/environments/assets/mjcf"

TARGETS = {
    "solefoot/tron1/base_robot.xml": biped,
    "quadruped/quadruped.xml": quadruped,
}


def main() -> None:
    for relative, family in TARGETS.items():
        spec = family.generate.load_design_spec()
        out = ASSETS / relative
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(family.builder.build_xml(spec, for_isaaclab=True))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
