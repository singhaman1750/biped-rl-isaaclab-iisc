"""USD prim-tree inspection and prototype/instance analysis utilities.

Shared helpers for printing and logging USD stage structure.  Designed for
use both from stand-alone diagnostic scripts (scripts/print_prim_tree.py) and
from inline runtime logging inside the co-optimisation loop.

Public API
----------
print_prim(prim, depth)
    Print one prim's type, path, schemas, xformOps, and key attributes.

print_prim_tree(prim, depth, filter_str, traverse_instances)
    Recursively print a subtree.

log_prim_data(stage, filter_str)
    Log all four sections: composed (hidden), composed (traversed),
    prototypes, and sub-layers.

log_prototype_and_instance_info(stage, filter_str)
    Log a focused prototype → instance mapping table suitable for
    understanding which prototype controls which individual's link geometry.
    Prints current box scale values where discoverable.
"""

from __future__ import annotations

from pxr import Usd, UsdGeom, UsdPhysics, Sdf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEOM_TYPES = {
    "Cube", "Sphere", "Cylinder", "Capsule", "Cone", "Mesh",
    "NurbsPatch", "BasisCurves", "Points", "TetMesh", "Plane",
}

KEY_ATTRS = {
    "size", "radius", "height", "width", "depth", "extent",
    "xformOp:translate", "xformOp:orient", "xformOp:scale", "xformOpOrder",
    "physics:mass", "physics:density",
    "physics:centerOfMass", "physics:diagonalInertia", "physics:principalAxes",
    "physics:collisionEnabled", "physics:rigidBodyEnabled",
    "purpose", "visibility",
}


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _xform_op_name(op_type) -> str:
    return str(op_type).split(".")[-1].replace("Type", "")


def _get_scale_op(xf: UsdGeom.Xformable) -> UsdGeom.XformOp | None:
    """Return the TypeScale xformOp on *xf*, or None if it does not exist."""
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            return op
    return None


def section(title: str) -> None:
    """Print a section header."""
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Prim printing
# ---------------------------------------------------------------------------

def print_prim(prim: Usd.Prim, depth: int, label_suffix: str = "") -> None:
    """Print one prim's diagnostic info (no recursion)."""
    indent = "  " * depth
    ptype = prim.GetTypeName() or "(no type)"
    is_geom = ptype in GEOM_TYPES
    is_inst = prim.IsInstance()
    is_in_proto = prim.IsInPrototype()

    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    ops_str = ", ".join(_xform_op_name(op.GetOpType()) for op in ops) if ops else "—"

    schemas = list(prim.GetAppliedSchemas())
    schemas_str = ", ".join(schemas) if schemas else "—"

    flags = []
    if is_inst:                  flags.append("INSTANCE")
    if is_in_proto:              flags.append("IN-PROTO")
    if prim.IsInstanceable():    flags.append("instanceable")
    flag_str = f"  [{', '.join(flags)}]" if flags else ""

    marker = " ◆" if is_geom else ""
    print(f"{indent}── {prim.GetName()} [{ptype}]{marker}{flag_str}{label_suffix}")
    print(f"{indent}   path    : {prim.GetPath()}")
    print(f"{indent}   schemas : {schemas_str}")
    print(f"{indent}   xformOps: {ops_str}")

    for attr in prim.GetAttributes():
        aname = attr.GetName()
        if aname not in KEY_ATTRS and not aname.startswith("xformOp:"):
            continue
        val = attr.Get()
        if val is None:
            continue
        print(f"{indent}   {aname} = {val}")

    # Composition arcs (internal references)
    for spec in prim.GetPrimStack():
        for ref in spec.referenceList.GetAddedOrExplicitItems():
            print(f"{indent}   ref     : {ref.assetPath or '(internal)'} @ {ref.primPath}")


def _recurse_filter(
    prim: Usd.Prim,
    depth: int,
    filter_str: str,
    traverse_instances: bool,
) -> None:
    path_str = str(prim.GetPath())
    if filter_str.lower() in path_str.lower():
        print_prim_tree(prim, depth, filter_str, traverse_instances)
        return
    for child in prim.GetChildren():
        _recurse_filter(child, depth, filter_str, traverse_instances)


def print_prim_tree(
    prim: Usd.Prim,
    depth: int = 0,
    filter_str: str | None = None,
    traverse_instances: bool = False,
) -> None:
    """Recursively print a prim and its children.

    Args:
        traverse_instances: If True, descend into instanceable prims'
            prototype children (read-only view, shown with [IN-PROTO] flag).
    """
    path_str = str(prim.GetPath())
    in_filter = filter_str is None or filter_str.lower() in path_str.lower()

    if in_filter:
        print_prim(prim, depth)

    if prim.IsInstance() and traverse_instances:
        proto = prim.GetPrototype()
        children = proto.GetChildren() if proto else []
    else:
        children = prim.GetChildren()

    for child in children:
        child_path = str(child.GetPath())
        if filter_str is None or filter_str.lower() in child_path.lower():
            print_prim_tree(child, depth + 1, filter_str, traverse_instances)
        else:
            _recurse_filter(child, depth + 1, filter_str, traverse_instances)


# ---------------------------------------------------------------------------
# High-level stage logging
# ---------------------------------------------------------------------------

def log_prim_data(stage: Usd.Stage, filter_str: str | None = None) -> None:
    """Log all four diagnostic sections for *stage*.

    Sections:
      1. Composed stage — instance contents hidden (runtime view)
      2. Composed stage — instance contents shown via prototype traversal
      3. USD Prototypes — shared geometry backing instanceable prims
      4. Sub-layers     — raw source data per layer file
    """
    usd_path = stage.GetRootLayer().identifier

    # 1. Composed stage (no instance traversal)
    section(f"COMPOSED STAGE (instance contents hidden)  {usd_path}")
    for top in stage.GetPseudoRoot().GetChildren():
        print_prim_tree(top, 0, filter_str, traverse_instances=False)

    # 2. Composed stage with instance traversal
    section("COMPOSED STAGE (instance contents shown via prototype)")
    for top in stage.GetPseudoRoot().GetChildren():
        print_prim_tree(top, 0, filter_str, traverse_instances=True)

    # 3. Prototypes
    section("USD PROTOTYPES (shared geometry backing instanceable prims)")
    prototypes = stage.GetPrototypes()
    if not prototypes:
        print("  (no prototypes found — stage may not use instancing)")
    for proto in prototypes:
        print(f"\n  Prototype: {proto.GetPath()}")
        instances = [
            str(p.GetPath()) for p in stage.Traverse()
            if p.IsInstance() and p.GetPrototype() == proto
        ]
        if filter_str:
            instances = [i for i in instances if filter_str.lower() in i.lower()]
        print(
            f"  Instances ({len(instances)}): "
            f"{instances[:6]}{'...' if len(instances) > 6 else ''}"
        )
        print()
        for child in proto.GetChildren():
            print_prim_tree(child, 1, filter_str, traverse_instances=False)

    # 4. Sub-layers (raw source data)
    seen = {stage.GetRootLayer().identifier}
    for layer in stage.GetLayerStack(includeSessionLayers=False):
        lid = layer.identifier
        if lid in seen:
            continue
        seen.add(lid)
        sub = Usd.Stage.Open(lid)
        if not sub:
            continue
        section(f"SUB-LAYER (raw source)  {lid}")
        for top in sub.GetPseudoRoot().GetChildren():
            print_prim_tree(top, 0, filter_str, traverse_instances=False)


# ---------------------------------------------------------------------------
# Prototype / instance mapping for scaling
# ---------------------------------------------------------------------------

def _collect_scale_values(proto: Usd.Prim) -> dict[str, tuple]:
    """Walk prototype children to collect all xformOp:scale values.

    Returns a dict mapping child path suffix → scale tuple, e.g.:
        {"/mesh_0": (0.05, 0.032, 0.25), "/mesh_0/box": None}
    Only entries with a non-None scale value are included.
    """
    results: dict[str, tuple] = {}

    def _walk(prim: Usd.Prim) -> None:
        xf = UsdGeom.Xformable(prim)
        op = _get_scale_op(xf)
        if op is not None:
            val = op.Get()
            if val is not None:
                results[str(prim.GetPath())] = tuple(val)
        for child in prim.GetChildren():
            _walk(child)

    for child in proto.GetChildren():
        _walk(child)
    return results


def log_prototype_and_instance_info(
    stage: Usd.Stage,
    filter_str: str | None = None,
) -> None:
    """Log a focused prototype → instance mapping table.

    For each USD prototype prints:
    - The prototype path
    - All instances that share this prototype, grouped by the env path
      and the inferred individual index (env_idx % num_individuals is
      inferred as the smallest env index that maps to this prototype)
    - All xformOp:scale values found in the prototype's subtree
      (these encode the current box dimensions for box-geometry links)
    - The prim type of each geometry child (Cube, Mesh, Cylinder, etc.)

    This output is designed to answer two questions:
      1. How many unique prototypes exist per link × individual?
      2. What are the current box dimensions stored in each prototype?
    """
    prototypes = stage.GetPrototypes()
    if not prototypes:
        print("[prototype-info] No prototypes found (stage may not use instancing).")
        return

    # Build instance → prototype map
    instance_to_proto: dict[str, Usd.Prim] = {}
    for prim in stage.Traverse():
        if prim.IsInstance():
            proto = prim.GetPrototype()
            if proto:
                instance_to_proto[str(prim.GetPath())] = proto

    section(f"PROTOTYPE ↔ INSTANCE MAPPING  ({len(prototypes)} prototypes total)")

    for proto_idx, proto in enumerate(prototypes):
        proto_path = str(proto.GetPath())

        # Gather instances that point to this prototype
        instances = [
            path for path, p in instance_to_proto.items()
            if p == proto
        ]
        if filter_str:
            instances = [i for i in instances if filter_str.lower() in i.lower()]

        print(f"\n{'─'*68}")
        print(f"  Prototype [{proto_idx}]: {proto_path}")
        print(f"  Instance count  : {len(instances)}")

        # Show instance paths (first 8, then summarise)
        for inst in instances[:8]:
            print(f"    instance → {inst}")
        if len(instances) > 8:
            print(f"    ... and {len(instances) - 8} more")

        # Show prototype children and their types
        print(f"  Children:")
        for child in proto.GetChildren():
            ctype = child.GetTypeName() or "(no type)"
            cpath = str(child.GetPath())
            is_geom = ctype in GEOM_TYPES
            marker = " ◆" if is_geom else ""
            print(f"    {child.GetName()} [{ctype}]{marker}  path={cpath}")
            # Show grandchildren (the actual geometry prims like mesh_N/box)
            for gc in child.GetChildren():
                gctype = gc.GetTypeName() or "(no type)"
                gcmarker = " ◆" if gctype in GEOM_TYPES else ""
                print(f"      {gc.GetName()} [{gctype}]{gcmarker}  path={gc.GetPath()}")

        # Show all scale values (box dimensions) in this prototype's subtree
        scale_values = _collect_scale_values(proto)
        if scale_values:
            print(f"  xformOp:scale values (box dimensions):")
            for path, val in scale_values.items():
                print(f"    {path}  →  scale={val}")
        else:
            print(f"  xformOp:scale values: (none found — may be Mesh/convex-hull geometry)")

        # Infer which env indices this prototype serves
        env_indices = []
        for inst_path in instances:
            # Instance paths look like /World/envs/env_NNN/Robot/link/visuals
            parts = inst_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    try:
                        env_indices.append(int(part[4:]))
                    except ValueError:
                        pass
                    break
        if env_indices:
            env_indices.sort()
            print(f"  Env indices served: {env_indices[:12]}{'...' if len(env_indices) > 12 else ''}")
            if len(env_indices) >= 2:
                # Infer individual index from smallest env index served
                num_individuals_guess = env_indices[1] - env_indices[0] if len(env_indices) > 1 else "?"
                print(f"  Apparent env stride (= num_individuals?): {num_individuals_guess}")
                print(f"  Individual index (env_indices[0] % stride): "
                      f"{env_indices[0] % num_individuals_guess if isinstance(num_individuals_guess, int) else '?'}")

    print(f"\n{'─'*68}")
    print(f"[prototype-info] Summary: {len(prototypes)} prototypes backing "
          f"{len(instance_to_proto)} instances.")
