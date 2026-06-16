"""In-place primitive-geometry updates for *instanced* articulation links.

The URDF importer authors each link's geometry as an instanceable internal
reference: ``<robot>/<link>/{visuals,collisions}`` are instance prims whose box
lives at ``/visuals/<link>/mesh_0`` and ``/colliders/<link>/mesh_0`` (a unit
``UsdGeomCube`` scaled by ``mesh_0``'s ``xformOp:scale``) inside the per-individual
USD layer (see CO_OPTIMISATION.md Section 4.7.6).  Instanced geometry cannot be
overridden through an environment prim, so the box size is set on the prototype's
*source* ``mesh_0`` via the ``Sdf`` layer API.

Mass/inertia (the link prim) and the child-joint anchor (the joints scope) are
NOT instanced; they reach each environment by the *reference* to the design's USD,
and -- importantly -- they are generally authored in a DIFFERENT layer than the
geometry meshes (the importer emits visuals, collisions, and physics into separate
sub-stages).  Each quantity is therefore resolved from its own authoring layer via
the corresponding prim's property stack, never assumed to share the geometry layer.
Editing one design's prototype/source updates all of its instances on the next
``sim.reset()``.

The simulation must be stopped before calling these helpers; a single timed
``sim.reset()`` reactivates physics after all per-design edits are authored.
"""

from __future__ import annotations

import time

from pxr import Gf, Sdf, Usd

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage

# Child joint relocated when a scalable link's length changes (its Body0 == link).
SCALABLE_LINK_CHILD_JOINTS = {
    "hip_R_thigh_Link": "knee_R_Joint",
    "hip_L_thigh_Link": "knee_L_Joint",
    "knee_R_Link": "ankle_R_actuator_Joint",
    "knee_L_Link": "ankle_L_actuator_Joint",
}

GEOM_MESH = "mesh_0"


def _source_spec(attr):
    """(layer, Sdf.Path) of the strongest authored opinion for ``attr`` (or None)."""
    if not attr:
        return None, None
    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    return (stack[0].layer, stack[0].path) if stack else (None, None)


def update_articulation_links(
    prototype_path: str,
    link_prim_path: str,
    extents: list[float],
    stage=None,
) -> None:
    """Author ABSOLUTE box extents for ONE geometry prototype on its source layer.

    Sets the prototype ``mesh_0``'s ``xformOp:scale``/``xformOp:translate`` and --
    idempotently -- the owning link's mass/inertia and the child joint's
    ``localPos0``.  Each quantity is authored on whichever layer actually defines it
    (resolved per-attribute), so geometry and physics may live in separate layers.
    Editing one prototype/source updates all of its instances on the next
    ``sim.reset()``.  Must run while the simulation is stopped.

    Args:
        prototype_path: Path of the implicit geometry prototype (``/__Prototype_<N>``)
            backing one ``<link>/{visuals,collisions}`` instance.
        link_prim_path: Composed (non-instanced) link prim of a representative
            environment, e.g. ``"/World/envs/env_0/Robot/hip_R_thigh_Link"``.
        extents: ``[x_abs, y_abs, z_abs]`` target box dimensions in metres.
        stage: Optional explicit USD stage; defaults to the current stage.
    """
    stage = stage or get_current_stage()
    x_new, y_new, z_new = (float(v) for v in extents)

    link_name = link_prim_path.rsplit("/", 1)[1]
    robot_path = link_prim_path.rsplit("/", 1)[0]
    child_joint = SCALABLE_LINK_CHILD_JOINTS[link_name]

    # --- geometry source (instanced): via the prototype's own mesh_0 ---
    mesh = stage.GetPrimAtPath(prototype_path).GetChild(GEOM_MESH)
    if not mesh:
        return
    g_layer, g_scale_path = _source_spec(mesh.GetAttribute("xformOp:scale"))
    if g_layer is None or g_scale_path is None:
        return
    g_trans_path = g_scale_path.GetPrimPath().AppendProperty("xformOp:translate")
    x0, y0, z0 = mesh.GetAttribute("xformOp:scale").Get()
    t0 = mesh.GetAttribute("xformOp:translate").Get()

    # --- physics sources (reference-shared, NOT instanced): each on its OWN layer ---
    link_prim = stage.GetPrimAtPath(link_prim_path)
    joint_prim = stage.GetPrimAtPath(f"{robot_path}/joints/{child_joint}")
    m_layer, m_path = _source_spec(link_prim.GetAttribute("physics:mass"))
    i_layer, i_path = _source_spec(link_prim.GetAttribute("physics:diagonalInertia"))
    j_layer, j_path = _source_spec(joint_prim.GetAttribute("physics:localPos0"))

    mass0 = link_prim.GetAttribute("physics:mass").Get()
    p0 = joint_prim.GetAttribute("physics:localPos0").Get()
    density = mass0 / (x0 * y0 * z0)
    offset = abs(p0[2]) - z0
    m_new = density * x_new * y_new * z_new

    with Sdf.ChangeBlock():
        # (1) geometry -- prototype source; the only legal resize of instanced geometry
        g_layer.GetAttributeAtPath(g_scale_path).default = Gf.Vec3d(x_new, y_new, z_new)
        if g_layer.GetAttributeAtPath(g_trans_path) is not None:
            g_layer.GetAttributeAtPath(g_trans_path).default = Gf.Vec3d(
                t0[0], t0[1], -z_new / 2.0
            )
        # (2) mass + solid-box inertia -- link prim, reference-shared (idempotent)
        if m_layer is not None and m_path is not None:
            m_layer.GetAttributeAtPath(m_path).default = m_new
        if i_layer is not None and i_path is not None:
            i_layer.GetAttributeAtPath(i_path).default = Gf.Vec3f(
                m_new * (y_new * y_new + z_new * z_new) / 12.0,
                m_new * (x_new * x_new + z_new * z_new) / 12.0,
                m_new * (x_new * x_new + y_new * y_new) / 12.0,
            )
        # (3) child-joint anchor -- joints scope, reference-shared (idempotent)
        if j_layer is not None and j_path is not None:
            j_layer.GetAttributeAtPath(j_path).default = Gf.Vec3f(
                p0[0], p0[1], -(z_new + offset)
            )


def apply_link_length_params(
    env: ManagerBasedRLEnv,
    link_length_params_list: list[dict[str, dict[str, float]]],
) -> None:
    """Author absolute link extents on every design's source layer, then a single
    timed ``sim.reset()``.

    Iterating the first ``num_individuals`` environments visits each design exactly
    once (round-robin assignment, identical to the multi-USD spawner), and editing a
    design's geometry prototype updates all of its instances.  The simulation must be
    stopped before this function is called; a safety-net stop is included.

    Args:
        env: The unwrapped ``ManagerBasedRLEnv`` instance.
        link_length_params_list: List of per-individual link-extent dicts, one per
            individual.  Each dict maps link name -> ``{"x": ..., "y": ..., "z": ...}``.
    """
    sim, scene = env.sim, env.scene
    num_individuals = len(link_length_params_list)
    env_paths = scene.env_prim_paths

    if sim.is_playing():  # safety net
        sim._disable_app_control_on_stop_handle = True
        sim.stop()
        sim._disable_app_control_on_stop_handle = False

    stage = get_current_stage()
    for individual in range(num_individuals):  # first N envs = each design once
        env_path = env_paths[individual]
        params = link_length_params_list[individual]
        for link_name, d in params.items():
            ext = [d["x"], d["y"], d["z"]]
            link_prim_path = f"{env_path}/Robot/{link_name}"
            for purpose in ("visuals", "collisions"):
                inst = stage.GetPrimAtPath(f"{link_prim_path}/{purpose}")
                proto = inst.GetPrototype() if inst else None
                if proto:
                    update_articulation_links(
                        proto.GetPath().pathString, link_prim_path, ext, stage
                    )

    print("Reactivating Physics (in-place prototype update)")
    start = time.perf_counter()
    sim.reset()
    print(
        f"Total time taken for sim.reset(): {time.perf_counter() - start:.4f} seconds"
    )
