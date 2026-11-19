"""Give every actuator catalogue entry the simulator parameters of its nearest identified actuator.

The identified actuators, TRON1-* in the biped catalogue and A1-* in the quadruped's, are those every
training run to date used, and they are left as they are. Every other entry receives, in its own
definition, the friction triple and joint side armature of the identified entry
nearest to it in log peak torque, log peak speed and log mass, the three specifications every entry
states, a tie going to the earlier identified entry in the catalogue. An entry that already states a
key keeps its own value. See plans/design_filtering.md section 7.

    python3 scripts/tools/fill_actuator_sim_params.py [--check] design_generator
"""
import argparse
import json
import math
import os

IDENTIFIED = {"biped": "TRON1-", "quadruped": "A1-"}
SPECS = ("peak_torque", "peak_speed", "mass")
COPIED = ("friction_static", "friction_dynamic", "activation_vel")
DATE = "2026-11-12"


def armature(entry: dict) -> float:
    """The joint side armature the catalogue loader derives, armature_value or rotor_inertia * ratio^2."""
    if "armature_value" in entry:
        return float(entry["armature_value"])
    return round(float(entry["rotor_inertia"]) * float(entry["ratio"]) ** 2, 9)


def distance(a: dict, b: dict) -> float:
    return math.sqrt(sum(math.log(a[k] / b[k]) ** 2 for k in SPECS))


def fill(catalog: dict, prefix: str) -> list[tuple[str, str, float]]:
    entries = catalog["actuators"]
    refs = [e for e in entries if e["name"].startswith(prefix)]
    mapping = []
    for i, entry in enumerate(entries):
        if entry in refs:
            continue
        ref = min(refs, key=lambda r: (distance(entry, r), refs.index(r)))
        values = {k: ref[k] for k in COPIED}
        values["armature_value"] = armature(ref)
        out = {}
        for key, value in entry.items():
            if key == "notes":
                out.update({k: v for k, v in values.items() if k not in entry})
            out[key] = value
        out.update({k: v for k, v in values.items() if k not in out})
        if "Simulator parameters (" not in out.get("notes", ""):
            out["notes"] = (out.get("notes", "") + f" Simulator parameters (friction, armature) copied"
                            f" from {ref['name']}, the nearest identified actuator in peak torque, peak speed and mass,"
                            f" pending identification ({DATE}).").strip()
        entries[i] = out
        mapping.append((entry["name"], ref["name"], distance(entry, ref)))
    note = (f" Corrected {DATE}, every other entry now states the simulator parameters of its nearest identified"
            f" entry, copied by scripts/tools/fill_actuator_sim_params.py, so every entry is simulation complete"
            f" and the co-optimisation search offers the whole catalogue.")
    if mapping and note not in catalog["_schema"]["simulation"]:
        catalog["_schema"]["simulation"] += note
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="the design_generator directory")
    parser.add_argument("--check", action="store_true", help="print the mapping without writing")
    args = parser.parse_args()
    for family, prefix in IDENTIFIED.items():
        path = os.path.join(args.root, family, "catalogs", "actuators.json")
        with open(path) as fh:
            catalog = json.load(fh)
        for name, ref, d in fill(catalog, prefix):
            print(f"{family:9s} {name:18s} <- {ref:12s} distance {d:.3f}")
        if not args.check:
            with open(path, "w") as fh:
                fh.write(json.dumps(catalog, indent=2) + "\n")


if __name__ == "__main__":
    main()
