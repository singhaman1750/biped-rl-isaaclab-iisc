"""Does refresh_rendered_geometry redraw a housing cylinder resized in place? Diagnostic of 2026-11-17.

env_0 spawns the A1 quadruped and is then written in place to the same robot with an ANYdrive-3 knee,
the change environment 0 of quadruped_copt/2026-11-17_07-11-53 met at its first swap. env_1 spawns that
ANYdrive-3 design converted afresh, the reference drawing. Both are rendered through the viewport camera
the training video records from, with the Fabric Scene Delegate on, at four stages, spawned, written
without a refresh, refreshed, and refreshed with the redraw forced from the known spawned size. The
Fabric copies of env_0's knee and hip cylinders are read back at each stage.

    ./isaaclab.sh -p logs/render_diag_2026-11-17/cylinder_redraw_diag.py logs/render_diag_2026-11-17/out
"""
import json
import os
import sys
from types import SimpleNamespace

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import usdrt  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdUtils  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402

from co_optimisation.runners.usd_generator import RandomDesignGenerator  # noqa: E402
from co_optimisation.utils import update as U  # noqa: E402

# Absolute, since the importer writes a relative output directory's sublayers as /configuration/...
OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)
report = {"fabric_scene_delegate": carb.settings.get_settings().get_as_bool("/app/useFabricSceneDelegate")}


def _fail(kind, value, tb):
    """Write what was gathered and leave, rather than hang in Kit after an exception."""
    import traceback
    traceback.print_exception(kind, value, tb)
    report["error"] = repr(value)
    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
    os._exit(1)


sys.excepthook = _fail

# ---- the two designs, emitted and converted as the runner does ----
gen = RandomDesignGenerator(family="quadruped", num_individuals=1, output_dir=os.path.join(OUT, "designs"))
gen.builder_kwargs = {"shank_foot_gap": 0.001}
spec_a = gen._spec_from_parameters({})
spec_b = gen._spec_from_parameters({}, {"knee": "ANYdrive-3"})
usd = {}
for idx, (tag, spec) in enumerate((("A1", spec_a), ("ANYdrive3_knee", spec_b))):
    xml = os.path.join(OUT, "designs", f"individual_{idx:04d}.xml")
    os.makedirs(os.path.dirname(xml), exist_ok=True)
    with open(xml, "w") as fh:
        fh.write(gen.family.build_xml(spec, **gen.builder_kwargs))
    usd[tag] = gen._generate_individual_usd(xml, idx)
links_b = gen.family.links(spec_b)
root = gen.family.root_body
report["knee_link_entry_B"] = {k: v for k, v in links_b["knee_FL_act"].items() if k in ("radius", "half_length", "parent_pos")}

# ---- the scene ----
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0", gravity=(0.0, 0.0, 0.0)))
stage = get_current_stage()
for i, (tag, y) in enumerate((("A1", -0.3), ("ANYdrive3_knee", 0.3))):
    stage.DefinePrim(f"/World/envs/env_{i}", "Xform")
    cfg = sim_utils.UsdFileCfg(usd_path=usd[tag])
    cfg.func(f"/World/envs/env_{i}/Robot", cfg, translation=(0.0, y, 0.5))
light = sim_utils.DomeLightCfg(intensity=2500.0)
light.func("/World/Light", light)
sim.reset()

rp = rep.create.render_product("/OmniverseKit_Persp", (1280, 720))
annot = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
annot.attach([rp])
VIEWS = {"front": ((1.1, 0.0, 0.75), (0.0, 0.0, 0.45)), "top": ((0.05, 0.0, 1.7), (0.0, 0.0, 0.45))}


def capture(stage_tag: str) -> None:
    for view, (eye, target) in VIEWS.items():
        sim.set_camera_view(eye, target)
        for _ in range(12):
            sim.render()
        rgb = np.asarray(annot.get_data())
        if rgb.size:
            Image.fromarray(rgb[..., :3]).save(os.path.join(OUT, f"{stage_tag}_{view}.png"))


def dump(stage_tag: str) -> None:
    """The Fabric copy of every FL cylinder of both environments, against the composed prototype."""
    rt = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().Insert(stage).ToLongInt())
    rows = {}
    for env in ("/World/envs/env_0", "/World/envs/env_1"):
        for link in ("abad_FL_act", "hip_FL_act", "knee_FL_act"):
            suffix = f"Robot/{root}/{link}/visuals"
            inst = stage.GetPrimAtPath(f"{env}/{suffix}")
            proto = inst.GetPrototype() if inst else None
            if not proto:
                rows[f"{env}/{link}"] = "no prototype"
                continue
            for prim in Usd.PrimRange(proto):
                if not prim.IsA(UsdGeom.Cylinder):
                    continue
                rel = f"{suffix}/{prim.GetPath().MakeRelativePath(proto.GetPath())}"
                fprim = rt.GetPrimAtPath(f"{env}/{rel}")
                cyl = UsdGeom.Cylinder(prim)
                row = {"usd_radius": cyl.GetRadiusAttr().Get(), "usd_height": cyl.GetHeightAttr().Get(),
                       "usd_extent": str(cyl.GetExtentAttr().Get()), "fabric_valid": fprim.IsValid()}
                if fprim.IsValid():
                    names = sorted(str(a.GetName()) for a in fprim.GetAttributes())
                    row["fabric_attributes"] = names
                    for n in ("radius", "height", "extent", "omni:fabric:localMatrix", "omni:fabric:worldMatrix"):
                        if fprim.HasAttribute(n):
                            row[f"fabric_{n}"] = str(fprim.GetAttribute(n).Get())
                    matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, proto)
                    redraw = U._scaled_cylinder(prim, matrix, fprim)
                    row["scaled_cylinder"] = None if redraw is None else str(redraw[0])
                rows[f"{env}/{link}"] = row
    report[stage_tag] = rows


capture("0_spawned")
env = SimpleNamespace(sim=sim, scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0"]))
U.apply_link_length_params(env, [links_b], root)
import usdrt.hierarchy  # noqa: E402
rt = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().Insert(stage).ToLongInt())
hier = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(rt.GetFabricId(), rt.GetStageIdAsStageId())
modes = {"FL": "attr_set_then_update", "FR": "attr_set_then_update", "RL": "set_local_xform_then_update", "RR": "set_local_xform_then_update"}
report["modes"] = modes
for leg, mode in modes.items():
    suffix = f"Robot/{root}/knee_{leg}_act/visuals"
    proto = stage.GetPrimAtPath(f"/World/envs/env_0/{suffix}").GetPrototype()
    for prim in Usd.PrimRange(proto):
        if not prim.IsA(UsdGeom.Cylinder):
            continue
        matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, proto)
        path = f"/World/envs/env_0/{suffix}/{prim.GetPath().MakeRelativePath(proto.GetPath())}"
        fprim = rt.GetPrimAtPath(path)
        redraw = U._scaled_cylinder(prim, matrix, fprim)
        report.setdefault("variant_inputs", {})[leg] = {
            "path": path, "fabric_valid": fprim.IsValid(),
            "fabric_radius": str(fprim.GetAttribute("radius").Get()) if fprim.IsValid() and fprim.HasAttribute("radius") else None,
            "fabric_height": str(fprim.GetAttribute("height").Get()) if fprim.IsValid() and fprim.HasAttribute("height") else None,
            "usd_radius": UsdGeom.Cylinder(prim).GetRadiusAttr().Get(), "scaled": redraw is not None}
        # The spawned A1 size known from the spawned design, not read from the Fabric copy.
        cyl = UsdGeom.Cylinder(prim)
        r, h, axis = cyl.GetRadiusAttr().Get(), cyl.GetHeightAttr().Get(), cyl.GetAxisAttr().Get()
        ratio = [r / 0.046] * 3
        ratio["XYZ".index(axis)] = h / 0.040
        scaled = Gf.Matrix4d().SetScale(Gf.Vec3d(*ratio)) * matrix
        m = usdrt.Gf.Matrix4d(*(scaled[a][b] for a in range(4) for b in range(4)))
        if mode == "attr_set_then_update":
            fprim.GetAttribute("omni:fabric:localMatrix").Set(m)
        else:
            hier.set_local_xform(usdrt.Sdf.Path(path), m)
hier.update_world_xforms()
rows = {}
for leg in modes:
    suffix = f"Robot/{root}/knee_{leg}_act/visuals"
    proto = stage.GetPrimAtPath(f"/World/envs/env_0/{suffix}").GetPrototype()
    for prim in Usd.PrimRange(proto):
        if prim.IsA(UsdGeom.Cylinder):
            fprim = rt.GetPrimAtPath(f"/World/envs/env_0/{suffix}/{prim.GetPath().MakeRelativePath(proto.GetPath())}")
            if not fprim.IsValid():
                rows[leg] = "invalid copy"
                continue
            rows[leg] = {"local": str(fprim.GetAttribute("omni:fabric:localMatrix").Get()),
                         "world_before_render": str(fprim.GetAttribute("omni:fabric:worldMatrix").Get())}
capture("4_fix_variants")
for leg in modes:
    suffix = f"Robot/{root}/knee_{leg}_act/visuals"
    proto = stage.GetPrimAtPath(f"/World/envs/env_0/{suffix}").GetPrototype()
    for prim in Usd.PrimRange(proto):
        if prim.IsA(UsdGeom.Cylinder):
            fprim = rt.GetPrimAtPath(f"/World/envs/env_0/{suffix}/{prim.GetPath().MakeRelativePath(proto.GetPath())}")
            if isinstance(rows.get(leg), dict):
                rows[leg]["world_after_render"] = str(fprim.GetAttribute("omni:fabric:worldMatrix").Get())
for _ in range(40):
    sim.step(render=False)
capture("5_after_40_steps")
for leg in modes:
    suffix = f"Robot/{root}/knee_{leg}_act/visuals"
    proto = stage.GetPrimAtPath(f"/World/envs/env_0/{suffix}").GetPrototype()
    for prim in Usd.PrimRange(proto):
        if prim.IsA(UsdGeom.Cylinder) and isinstance(rows.get(leg), dict):
            fprim = rt.GetPrimAtPath(f"/World/envs/env_0/{suffix}/{prim.GetPath().MakeRelativePath(proto.GetPath())}")
            rows[leg]["local_after_steps"] = str(fprim.GetAttribute("omni:fabric:localMatrix").Get())
            rows[leg]["world_after_steps"] = str(fprim.GetAttribute("omni:fabric:worldMatrix").Get())
report["fix_variants"] = rows
with open(os.path.join(OUT, "report.json"), "w") as fh:
    json.dump(report, fh, indent=1, default=str)
print("[diag] done", flush=True)
os._exit(0)
