"""In-place primitive-geometry updates for *instanced* articulation links.

The Isaac Sim MuJoCo importer authors each body's geometry as an instanceable
internal reference: ``<robot>/<root>/<body>/{visuals,collisions}`` are instance
prims whose box lives under ``/visuals/<body>/<geom>`` and
``/collisions/<body>/<geom>`` inside the per-individual Universal Scene
Description layer, each an ``Xform`` carrying the box's ``xformOp:scale`` and
``xformOp:translate`` and referencing a unit ``UsdGeomCube`` in ``/meshes``
(MjcfImporter.cpp:1236-1266, MjcfUsd.cpp:820-826).  Instanced geometry cannot
be overridden through an environment prim, so the box is set on the
prototype's own source ``Xform`` via the ``Sdf`` layer API.

Three conventions of that layout differ from the URDF importer's and are the
reason this module cannot be shared with it.  The scale is the MuJoCo
half-extent rather than the full edge length; the geometry ``Xform`` under the
prototype is named after the MuJoCo geom, which is ``_geom_<N>`` when the geom
is unnamed, rather than the URDF importer's positional ``mesh_<i>``; and every
body prim sits one level deeper, under the root body's own ``Xform``, while
the joints scope stays directly beneath the referenced asset prim.  The first
is handled by writing half-extents, the second by discovering the single box
child rather than naming it, and the third by taking the root body's name from
the population.  See plans/design_generator_integration.md section 4.6.

Mass, inertia and centre of mass (the body prim) and the child-joint anchor
(the joints scope) are NOT instanced; they reach each environment by the
*reference* to the design's Universal Scene Description, and they are
generally authored in a DIFFERENT layer than the geometry, the importer
emitting base, physics, robot and sensor into separate sub-stages that the
root layer then carries as variant-selected payloads
(PluginInterface.cpp:395-460).  Each quantity is therefore resolved from its
own authoring layer via the corresponding prim's property stack, never assumed
to share the geometry layer.  Editing one design's prototype and source
updates all of its instances on the next ``sim.reset()``.

Every number written here is computed by the design generator from the same
specification that emitted the document, so this module performs no geometry
inference of its own: it neither recovers a density nor assumes an axis along
which a link grows.  See ``co_optimisation/runners/usd_generator.py``'s
``_entry``.

The simulation must be stopped before calling these helpers; a single timed
``sim.reset()`` reactivates physics after all per-design edits are authored.
"""

from __future__ import annotations

import math
import time

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sim.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, Vt


def _source_spec(attr):
    """(layer, Sdf.Path) of the strongest authored opinion for ``attr`` (or None)."""
    if not attr:
        return None, None
    stack = attr.GetPropertyStack(Usd.TimeCode.Default())
    return (stack[0].layer, stack[0].path) if stack else (None, None)


def _box_xform(prototype) -> Usd.Prim | None:
    """The one ``Xform`` under *prototype* that positions and scales a box.

    A scalable link carries exactly one collision box and one visual box, and
    the two land in different prototypes, so the search is unambiguous.  The
    prim is identified by structure rather than by name because an unnamed
    MuJoCo geom is named ``_geom_<N>`` by the importer, with ``N`` a
    document-wide running counter (MjcfParser.cpp:180) that shifts whenever the
    emitted document gains or loses a geom.
    """
    for child in prototype.GetChildren():
        if not child.GetAttribute("xformOp:scale"):
            continue
        for descendant in Usd.PrimRange(child):
            if descendant.IsA(UsdGeom.Cube):
                return child
    return None


def _cylinder(prototype) -> Usd.Prim | None:
    """The one ``UsdGeomCylinder`` under *prototype*.

    An actuator housing carries one visual and one collision cylinder, in different
    prototypes. Unlike the box, whose size is the scale of the referencing ``Xform``,
    the importer authors a cylinder's size as the ``radius`` and ``height`` of the
    cylinder prim itself under a unit scale (MjcfUsd.cpp:799-818,
    MjcfImporter.cpp:428-444).
    """
    for prim in Usd.PrimRange(prototype):
        if prim.IsA(UsdGeom.Cylinder):
            return prim
    return None


def update_articulation_links(
    prototype_path: str,
    link_prim_path: str,
    joints_prim_path: str,
    params: dict,
    stage=None,
) -> bool:
    """Author ONE geometry prototype and its link's physics on their source layers.

    Sets the prototype box ``Xform``'s ``xformOp:scale`` and
    ``xformOp:translate``, or the prototype cylinder's ``radius``, ``height`` and
    ``extent``, and -- idempotently -- the owning link's mass, inertia and,
    for a box, centre of mass, the ``physics:localPos0`` of the child joint and
    of the link's own joint to its parent, and, when the entry carries
    ``joint_rot``, the ``physics:localRot0`` of every joint it names.
    Each quantity is authored on whichever layer actually defines it, resolved
    per attribute, since geometry and physics live in separate layers.  Editing
    one prototype and source updates all of its instances on the next
    ``sim.reset()``.  Must run while the simulation is stopped.

    Args:
        prototype_path: Path of the implicit geometry prototype
            (``/__Prototype_<N>``) backing one ``<body>/{visuals,collisions}``
            instance.
        link_prim_path: Composed (non-instanced) body prim of a representative
            environment, e.g.
            ``"/World/envs/env_0/Robot/base_Link/thigh_R_Link"``.
        joints_prim_path: The design's joints scope in the same environment,
            e.g. ``"/World/envs/env_0/Robot/joints"``.
        params: One entry of a population's link-length dict, carrying
            ``mass`` and ``inertia`` and either a box, ``half_extents`` and
            ``box_pos``, or a cylinder, ``radius`` and ``half_length``, and
            optionally ``child_joint`` and ``child_pos``, ``parent_joint`` and
            ``parent_pos``, and ``joint_rot``, a dict from joint name to the
            (w, x, y, z) quat of the body whose frame that joint's parent side
            holds.
        stage: Optional explicit USD stage; defaults to the current stage.

    Returns:
        True when the box or cylinder was found and written, False when the
        prototype carries none, which :func:`apply_link_length_params` turns
        into a raised error rather than a silent skip.
    """
    stage = stage or get_current_stage()

    mass = float(params["mass"])
    ixx, iyy, izz = (float(v) for v in params["inertia"])
    prototype = stage.GetPrimAtPath(prototype_path)

    # --- geometry source (instanced): via the prototype's own box Xform or cylinder ---
    geometry = []
    if "half_extents" in params:
        hx, hy, hz = (float(v) for v in params["half_extents"])
        px, py, pz = (float(v) for v in params["box_pos"])
        box = _box_xform(prototype)
        if not box:
            return False
        g_layer, g_scale_path = _source_spec(box.GetAttribute("xformOp:scale"))
        if g_layer is None or g_scale_path is None:
            return False
        g_trans_path = g_scale_path.GetPrimPath().AppendProperty("xformOp:translate")
    else:
        r, hl = float(params["radius"]), float(params["half_length"])
        cylinder = _cylinder(prototype)
        if not cylinder:
            return False
        # The extent the importer computes about x for a cylinder it authors about z
        # (MjcfUsd.cpp:813-815), kept so that a written design equals a converted one.
        for name, value in (("radius", r), ("height", 2.0 * hl),
                            ("extent", Vt.Vec3fArray([Gf.Vec3f(-hl, -r, -r), Gf.Vec3f(hl, r, r)]))):
            layer, path = _source_spec(cylinder.GetAttribute(name))
            if layer is None:
                return False
            geometry.append((layer, path, value))

    # --- physics sources (reference-shared, NOT instanced): each on its OWN layer ---
    link_prim = stage.GetPrimAtPath(link_prim_path)
    m_layer, m_path = _source_spec(link_prim.GetAttribute("physics:mass"))
    i_layer, i_path = _source_spec(link_prim.GetAttribute("physics:diagonalInertia"))
    c_layer, c_path = _source_spec(link_prim.GetAttribute("physics:centerOfMass"))
    placements = []
    for side in ("child", "parent"):
        if f"{side}_joint" in params:
            joint_prim = stage.GetPrimAtPath(f"{joints_prim_path}/{params[f'{side}_joint']}")
            j_layer, j_path = _source_spec(joint_prim.GetAttribute("physics:localPos0"))
            if j_layer is not None and j_path is not None:
                placements.append((j_layer, j_path, Gf.Vec3f(*(float(v) for v in params[f"{side}_pos"]))))
    # The importer authors localRot0 as the body quat composed with the joint axis
    # alignment, which it authors alone as localRot1, the identity for a fixed joint
    # (MjcfImporter.cpp:1341-1366, computeJointFrame at :1526-1537).
    rotations = []
    for joint_name, (w, x, y, z) in (params.get("joint_rot") or {}).items():
        rot_prim = stage.GetPrimAtPath(f"{joints_prim_path}/{joint_name}")
        r_layer, r_path = _source_spec(rot_prim.GetAttribute("physics:localRot0"))
        align = rot_prim.GetAttribute("physics:localRot1").Get() if rot_prim else None
        if r_layer is None or r_path is None or align is None:
            raise RuntimeError(
                f"{joints_prim_path}/{joint_name} carries no physics:localRot0 or "
                "localRot1, so the design's nominal pose cannot be written in place"
            )
        rotations.append((r_layer, r_path, Gf.Quatf(w, x, y, z) * align))

    with Sdf.ChangeBlock():
        # (1) geometry -- prototype source; the only legal resize of instanced
        #     geometry. A box's scale is the MuJoCo half-extent, the importer having
        #     left the cube at its default size of 2 and put the half-extent on
        #     the referencing Xform (MjcfUsd.cpp:820-826, MjcfImporter.cpp:399).
        if "half_extents" in params:
            g_layer.GetAttributeAtPath(g_scale_path).default = Gf.Vec3d(hx, hy, hz)
            if g_layer.GetAttributeAtPath(g_trans_path) is not None:
                g_layer.GetAttributeAtPath(g_trans_path).default = Gf.Vec3d(px, py, pz)
        for layer, path, value in geometry:
            layer.GetAttributeAtPath(path).default = value
        # (2) mass, inertia and centre of mass -- link prim, reference-shared. A
        #     housing's centre of mass is its origin whatever its size.
        if m_layer is not None and m_path is not None:
            m_layer.GetAttributeAtPath(m_path).default = mass
        if i_layer is not None and i_path is not None:
            i_layer.GetAttributeAtPath(i_path).default = Gf.Vec3f(ixx, iyy, izz)
        if c_layer is not None and c_path is not None and "box_pos" in params:
            c_layer.GetAttributeAtPath(c_path).default = Gf.Vec3f(px, py, pz)
        # (3) joint anchors -- joints scope, reference-shared. localPos0 is the
        #     child body's own origin expressed in the parent's frame, the
        #     importer having taken the joint's own position to be the child
        #     body origin (MjcfImporter.cpp:1367-1376), so relocating it is what
        #     lengthens the limb or moves a housing.
        for j_layer, j_path, pos in placements:
            j_layer.GetAttributeAtPath(j_path).default = pos
        # (4) joint frames -- joints scope, reference-shared
        for r_layer, r_path, rot in rotations:
            r_layer.GetAttributeAtPath(r_path).default = rot
    return True


def apply_link_length_params(
    env: ManagerBasedRLEnv,
    link_length_params_list: list[dict[str, dict]],
    root_body: str,
    strict: bool = True,
) -> None:
    """Author absolute link geometry on every design's source layer, then a single
    timed ``sim.reset()``.

    Iterating the first ``num_individuals`` environments visits each design
    exactly once (round-robin assignment, identical to the multi-asset
    spawner), and editing a design's geometry prototype updates all of its
    instances.  The simulation must be stopped before this function is called;
    a safety-net stop is included.

    Args:
        env: The unwrapped ``ManagerBasedRLEnv`` instance.
        link_length_params_list: Per-individual link-geometry dicts, one per
            individual, each mapping a body name to the entry documented in
            :func:`update_articulation_links`.
        root_body: The emitted document's root body name, which the Isaac Sim
            MuJoCo importer interposes between the referenced asset prim and
            every body prim.  Supplied by ``Population.get_root_body``.
        strict: Raise when a design's link could not be reached, rather than
            leaving it silently at the previous generation's geometry.
            Defaults to True.  The URDF-era editor returned early instead, so a
            layout it did not recognise cost a whole schedule of training on
            generation zero while the logs reported new populations.
    """
    sim, scene = env.sim, env.scene
    num_individuals = len(link_length_params_list)
    env_paths = scene.env_prim_paths

    if sim.is_playing():  # safety net
        sim._disable_app_control_on_stop_handle = True
        sim.stop()
        sim._disable_app_control_on_stop_handle = False

    stage = get_current_stage()
    missed: list[str] = []
    for individual in range(num_individuals):  # first N envs = each design once
        env_path = env_paths[individual]
        joints_prim_path = f"{env_path}/Robot/joints"
        for link_name, params in link_length_params_list[individual].items():
            # The importer nests the root body under an Xform of its own name, so the trunk is Robot/<root>/<root>.
            link_prim_path = f"{env_path}/Robot/{root_body}/{link_name}"
            for purpose in ("visuals", "collisions"):
                inst = stage.GetPrimAtPath(f"{link_prim_path}/{purpose}")
                proto = inst.GetPrototype() if inst else None
                if not proto:
                    missed.append(f"{link_prim_path}/{purpose} (no prototype)")
                    continue
                if not update_articulation_links(
                    proto.GetPath().pathString, link_prim_path,
                    joints_prim_path, params, stage
                ):
                    missed.append(f"{link_prim_path}/{purpose} (no box or cylinder under prototype)")

    if missed and strict:
        raise RuntimeError(
            "In-place link update reached none of the following "
            f"{len(missed)} prim(s), so their designs would have stayed at the "
            "previous generation's geometry:\n  " + "\n  ".join(missed[:20])
            + ("\n  ..." if len(missed) > 20 else "")
        )

    print("Reactivating Physics (in-place prototype update)")
    start = time.perf_counter()
    sim.reset()
    print(
        f"Total time taken for sim.reset(): {time.perf_counter() - start:.4f} seconds"
    )



def _scaled_cylinder(prim, matrix, fabric_copy, spawned=None):
    """The local matrix and extent that draw a spawned cylinder at the size its prototype now carries.

    The RTX delegate keeps the tessellation it populated at spawn, so the cylinder is drawn as that
    tessellation scaled in its own frame by the ratio of the written size to the spawned one.
    *spawned*, the (radius, height) in metres the environment's design was spawned with, is that
    baseline where given. None reads it from the Fabric copy's own radius and height, which the reset
    after an in place write overwrites with the written size while the renderer keeps the spawned
    tessellation, so that the copy then reports no change and the cylinder is left undrawn.
    None when there is no baseline to measure against, or when the written size is the spawned one,
    so that a design keeping its actuators leaves its copy exactly as spawned.
    """
    if spawned is not None:
        r0, h0 = (float(v) for v in spawned)
    elif fabric_copy.IsValid() and fabric_copy.HasAttribute("radius") and fabric_copy.HasAttribute("height"):
        r0, h0 = float(fabric_copy.GetAttribute("radius").Get()), float(fabric_copy.GetAttribute("height").Get())
    else:
        return None
    cylinder = UsdGeom.Cylinder(prim)
    axis = cylinder.GetAxisAttr().Get()
    r, h = float(cylinder.GetRadiusAttr().Get()), float(cylinder.GetHeightAttr().Get())
    if math.isclose(r, r0, rel_tol=1e-6) and math.isclose(h, h0, rel_tol=1e-6):
        return None
    i = "XYZ".index(axis)
    ratio, half = [r / r0] * 3, [r0] * 3
    ratio[i], half[i] = h / h0, h0 / 2.0
    scaled = Gf.Matrix4d().SetScale(Gf.Vec3d(*ratio)) * matrix
    # The scaled matrix maps the spawned tessellation to the written size, so the extent is the spawned one, scaled once by that matrix and not twice.
    return scaled, (Gf.Vec3d(*(-v for v in half)), Gf.Vec3d(*half))


def refresh_rendered_geometry(
    env: ManagerBasedRLEnv,
    link_length_params_list: list[dict[str, dict]],
    root_body: str,
    spawned_link_params_list: list[dict[str, dict]] | None = None,
) -> int:
    """Rewrite the renderer's copies of the visual geometry from the edited prototypes.

    With the Fabric Scene Delegate on, as it is whenever the run renders
    (``app.useFabricSceneDelegate`` in ``apps/isaaclab.python.headless.rendering.kit``),
    Fabric keeps one copy of every instanced geometry prim per environment, whose
    ``omni:fabric:localMatrix`` and ``extent`` are taken from the prototype when the
    copy is populated.  The edits of :func:`apply_link_length_params` reach USD, PhysX
    and Fabric's prototype prims but not these copies, so a design written in place is
    drawn with the boxes and cylinders of the design it replaced while its bodies sit
    where the new design puts them, a shank drawn through or short of its foot.
    Physics is unaffected, and so are the collision copies, which are not drawn.

    Boxes and cylinders are both redrawn. A box's size lives in the local matrix of its copy, the
    cube itself keeping the importer's size of 2. A cylinder's size lives in its own ``radius`` and
    ``height``, which the RTX delegate does not re-tessellate when they change, so the cylinder is
    drawn as its spawned tessellation scaled in its own frame by the ratio of the written size to the
    spawned one. A cylinder whose written size is its spawned size is not touched at all.

    *spawned_link_params_list*, the link dicts of the population last spawned, gives each housing's
    spawned size, its entry's ``radius`` and twice its ``half_length``, design ``i`` having been
    spawned as entry ``i % len(spawned_link_params_list)``. None reads the spawned size from the
    Fabric copy, which the reset has already overwritten with the written size, so that a resized
    cylinder is not redrawn, the defect this argument repairs and which an unset default leaves standing.

    The writes are carried into ``omni:fabric:worldMatrix``, which the renderer draws from, by
    ``IFabricHierarchy.update_world_xforms()``, since Fabric's change tracking does not pick up a
    local matrix written straight after the reset.

    Call after :func:`apply_link_length_params`.  Environment ``e`` shows design
    ``e % len(link_length_params_list)``, the round-robin assignment of the spawn.

    Returns:
        The number of Fabric prims written, 0 when the scene delegate is off.
    """
    import carb
    import usdrt
    import usdrt.hierarchy

    if not carb.settings.get_settings().get_as_bool("/app/useFabricSceneDelegate"):
        return 0
    stage = get_current_stage()
    rt_stage = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().Insert(stage).ToLongInt())
    xform_cache = UsdGeom.XformCache()
    env_paths = env.scene.env_prim_paths
    per_design = []
    redrawn = 0
    for individual, params in enumerate(link_length_params_list):
        spawned_params = (None if spawned_link_params_list is None
                          else spawned_link_params_list[individual % len(spawned_link_params_list)])
        entries = []
        for link_name in params:
            suffix = f"Robot/{root_body}/{link_name}/visuals"
            inst = stage.GetPrimAtPath(f"{env_paths[individual]}/{suffix}")
            proto = inst.GetPrototype() if inst else None
            if not proto:
                continue
            for prim in Usd.PrimRange(proto):
                if not (prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder)):
                    continue
                rel = f"{suffix}/{prim.GetPath().MakeRelativePath(proto.GetPath())}"
                matrix, _ = xform_cache.ComputeRelativeTransform(prim, proto)
                if prim.IsA(UsdGeom.Cube):
                    extent = UsdGeom.Boundable(prim).GetExtentAttr().Get()
                else:
                    # The spawned size from the spawned design where given, the Fabric copy otherwise.
                    entry = (spawned_params or {}).get(link_name) or {}
                    spawned = ((entry["radius"], 2.0 * entry["half_length"])
                               if "radius" in entry and "half_length" in entry else None)
                    copy = rt_stage.GetPrimAtPath(f"{env_paths[individual]}/{rel}")
                    redraw = _scaled_cylinder(prim, matrix, copy, spawned)
                    if redraw is None:
                        continue
                    matrix, extent = redraw
                    redrawn += 1
                # Fabric's extent carries the scale and rotation of the copy's local matrix, not its translation.
                linear = Gf.Matrix4d(matrix).SetTranslateOnly(Gf.Vec3d(0.0))
                box = Gf.BBox3d(Gf.Range3d(Gf.Vec3d(extent[0]), Gf.Vec3d(extent[1])), linear).ComputeAlignedRange()
                entries.append((rel, {
                    "omni:fabric:localMatrix": usdrt.Gf.Matrix4d(*(matrix[i][j] for i in range(4) for j in range(4))),
                    "extent": usdrt.Gf.Range3d(usdrt.Gf.Vec3d(*box.GetMin()), usdrt.Gf.Vec3d(*box.GetMax())),
                }))
        per_design.append(entries)
    written = 0
    for e, env_path in enumerate(env_paths):
        for rel, values in per_design[e % len(per_design)]:
            fprim = rt_stage.GetPrimAtPath(f"{env_path}/{rel}")
            if not fprim.IsValid():
                continue
            for name, value in values.items():
                if fprim.HasAttribute(name):
                    fprim.GetAttribute(name).Set(value)
            written += 1
    # The local matrices reach the world matrices the renderer draws from only through the hierarchy.
    usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId()
    ).update_world_xforms()
    if redrawn:
        print(f"[refresh] {redrawn} resized cylinders redrawn from their spawned size")
    return written


def assert_link_lengths_applied(
    env: ManagerBasedRLEnv,
    link_length_params_list: list[dict[str, dict]],
    tol: float = 1e-3,
) -> None:
    """Read the realised body separations and masses back out of the simulation.

    Called after the reset that follows :func:`apply_link_length_params`, this
    compares each scalable link's realised distance to its child body and to
    its parent body, and its mass, against what the population asked for.  It
    is the only check that can tell an edit that was authored from an edit that
    took effect, the authoring API reporting success whether or not physics
    ever reads the attribute.

    Raises:
        RuntimeError: naming every link whose realised separation differs from
            the requested one by more than *tol* metres, or whose mass differs
            by more than *tol* relative.
    """
    import torch

    robot = env.scene.articulations["robot"]
    names = robot.body_names
    positions = robot.data.body_link_pos_w
    masses = robot.data.default_mass
    num_individuals = len(link_length_params_list)
    num_envs = positions.shape[0]
    bad: list[str] = []
    for individual in range(num_individuals):
        # Design d is written through environment d alone, so environment
        # d + num_individuals tells an edit that reached the shared prototype and
        # reference from one that stopped where it was written.
        rows = [individual]
        if individual + num_individuals < num_envs:
            rows.append(individual + num_individuals)
        for env_row in rows:
            for link_name, params in link_length_params_list[individual].items():
                if link_name not in names:
                    continue
                pairs = [(link_name, params[f"{side}_body"], params[f"{side}_pos"])
                         for side in ("child", "parent") if f"{side}_body" in params]
                for a, b, pos in pairs:
                    if b not in names:
                        continue
                    want = float(torch.tensor(pos).norm())
                    got = float((positions[env_row, names.index(a)]
                                 - positions[env_row, names.index(b)]).norm())
                    if abs(got - want) > tol:
                        bad.append(
                            f"env {env_row} (design {individual}) {link_name} -> {b}: "
                            f"realised {got:.6f} m, requested {want:.6f} m"
                        )
                got = float(masses[env_row, names.index(link_name)])
                if abs(got - params["mass"]) > tol * params["mass"]:
                    bad.append(
                        f"env {env_row} (design {individual}) {link_name}: "
                        f"mass {got:.6f} kg, requested {params['mass']:.6f} kg"
                    )
    if bad:
        raise RuntimeError(
            "The in-place link update was authored but did not reach the "
            "simulation for:\n  " + "\n  ".join(bad)
        )


def apply_standing_height(
    env: ManagerBasedRLEnv,
    nominal: list[dict],
    foot_bodies: tuple[str, ...],
    hip_bodies: tuple[str, ...],
    pitch_joints: list[str],
    spawn_margin: float = 0.028,
    tolerance: float = 0.002,
):
    """Measure each environment's nominal pose, assert it against its design's, set the spawn.

    The foot depth and the foot's lead over the hip pitch axis are compared with
    ``Family.nominal``, computed from the chain the document was built from, so
    one comparison covers the solve, the builder's frames, the importer, the in
    place editor and the propagation to every environment. Each environment then
    spawns *spawn_margin* above its design's standing height, which
    ``reset_root_state_uniform`` reads at every reset. Runs after the
    ``sim.reset()`` that re-initialises the articulation and before the
    ``env.reset()`` that places the robots, and takes no physics step,
    ``body_link_pose_w`` updating the kinematics on read
    (articulation_data.py:583-598).

    Returns:
        The standing height (num_envs,).
    """
    import torch

    robot = env.scene.articulations["robot"]
    device = robot.device
    assignment = torch.arange(env.num_envs, device=device) % len(nominal)

    # The frames carry the fold, so a nonzero pitch nominal would apply it twice.
    ids, _ = robot.find_joints(pitch_joints)
    stale = robot.data.default_joint_pos[:, ids].abs().max().item()
    if stale > 1e-6:
        raise RuntimeError(
            f"pitch joint nominals of up to {stale:.4f} rad in the asset configuration, "
            "while the document folds the leg in its body frames. Set them to zero."
        )

    robot.write_joint_state_to_sim(robot.data.default_joint_pos,
                                   torch.zeros_like(robot.data.default_joint_vel))
    probe = robot.data.default_root_state[:, :7].clone()
    probe[:, :3] = env.scene.env_origins
    probe[:, 2] = 2.0
    probe[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    robot.write_root_pose_to_sim(probe)
    foot_ids, _ = robot.find_bodies(list(foot_bodies), preserve_order=True)
    hip_ids, _ = robot.find_bodies(list(hip_bodies), preserve_order=True)
    pos = robot.data.body_link_pos_w
    depth = probe[:, 2:3] - pos[:, foot_ids, 2]                         # (num_envs, legs)
    lead = pos[:, foot_ids, 0] - pos[:, hip_ids, 0]

    record = lambda key: torch.tensor([n[key] for n in nominal], device=device)[assignment]
    worst_depth = (depth - record("depth").unsqueeze(1)).abs().max().item()
    worst_dx = (lead - record("dx").unsqueeze(1)).abs().max().item()
    if worst_depth > tolerance or worst_dx > tolerance:
        raise RuntimeError(
            f"measured nominal pose differs from the solved one by {worst_depth * 1e3:.2f} mm "
            f"in foot depth and {worst_dx * 1e3:.2f} mm in foot lead over the hip"
        )

    # The margin keeps a rough terrain tile from capturing a foot at reset.
    standing = depth.mean(dim=1) + record("sole")
    robot.data.default_root_state[:, 2] = standing + spawn_margin
    print(
        f"[copt] nominal pose measured for {env.num_envs} environments, standing height "
        f"[{standing.min().item():.4f}, {standing.max().item():.4f}] m, worst disagreement "
        f"{1e3 * max(worst_depth, worst_dx):.3f} mm"
    )
    return standing
