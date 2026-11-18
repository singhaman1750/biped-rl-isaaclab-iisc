"""A housing cylinder resized in place is drawn at its written size, through the runner's own refresh.

Run with Isaac Sim and rendering, not collected by pytest. env_0 spawns the A1 quadruped and env_1
the same robot with an ANYdrive-3 knee, the change environment 0 of quadruped_copt/2026-11-17_07-11-53
met at its first swap. The ANYdrive-3 design is written into env_0 by apply_link_length_params and
refresh_rendered_geometry is called straight after, with the spawned population, as the runner calls
it, no usdrt read preceding it. Every knee cylinder copy of env_0 must then carry in
omni:fabric:worldMatrix, which the renderer draws from, the scale of the written size over the spawned
one, before any render and after 40 physics steps. A top view of both environments is written to the
output directory for comparison. See plans/cylinder_render_refresh.md.

    PYTHONPATH=biped/co_optimisation:$PYTHONPATH ./isaaclab.sh -p \
        biped/co_optimisation/tests/sim_render_refresh_check.py /workspace/isaaclab/logs/<dir>
"""
import math
import os
import sys
from types import SimpleNamespace

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import usdrt  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom, UsdUtils  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402

from co_optimisation.runners.usd_generator import RandomDesignGenerator  # noqa: E402
from co_optimisation.utils import update as U  # noqa: E402

# Absolute, since the importer writes a relative output directory's sublayers as /configuration/...
OUT = os.path.abspath(sys.argv[1])
os.makedirs(OUT, exist_ok=True)


def _fail(kind, value, tb):
    """Leave rather than hang in Kit after an exception."""
    import traceback
    traceback.print_exception(kind, value, tb)
    os._exit(1)


sys.excepthook = _fail
print(f"[check] update.py under test: {U.__file__}")

gen = RandomDesignGenerator(family="quadruped", num_individuals=1, output_dir=os.path.join(OUT, "designs"))
gen.builder_kwargs = {"shank_foot_gap": 0.001}
specs = {"A1": gen._spec_from_parameters({}), "ANYdrive3_knee": gen._spec_from_parameters({}, {"knee": "ANYdrive-3"})}
usd = {}
for idx, (tag, spec) in enumerate(specs.items()):
    xml = os.path.join(OUT, "designs", f"individual_{idx:04d}.xml")
    os.makedirs(os.path.dirname(xml), exist_ok=True)
    with open(xml, "w") as fh:
        fh.write(gen.family.build_xml(spec, **gen.builder_kwargs))
    usd[tag] = gen._generate_individual_usd(xml, idx)
links_a, links_b = gen.family.links(specs["A1"]), gen.family.links(specs["ANYdrive3_knee"])
root = gen.family.root_body

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
sim.set_camera_view((0.05, 0.0, 1.7), (0.0, 0.0, 0.45))
for _ in range(12):
    sim.render()

# The runner's sequence: the in place write and its reset, then the refresh with the spawned population.
env = SimpleNamespace(sim=sim, scene=SimpleNamespace(env_prim_paths=["/World/envs/env_0"]))
U.apply_link_length_params(env, [links_b], root)
written = U.refresh_rendered_geometry(env, [links_b], root, [links_a])
print(f"[check] refresh wrote {written} prims")


def world_scales() -> dict[str, tuple[float, float]]:
    """(scale along the cylinder's axis, scale across it) of each env_0 knee copy's world matrix."""
    rt = usdrt.Usd.Stage.Attach(UsdUtils.StageCache.Get().Insert(stage).ToLongInt())
    out = {}
    for leg in ("FL", "FR", "RL", "RR"):
        suffix = f"Robot/{root}/knee_{leg}_act/visuals"
        proto = stage.GetPrimAtPath(f"/World/envs/env_0/{suffix}").GetPrototype()
        for prim in Usd.PrimRange(proto):
            if not prim.IsA(UsdGeom.Cylinder):
                continue
            fprim = rt.GetPrimAtPath(f"/World/envs/env_0/{suffix}/{prim.GetPath().MakeRelativePath(proto.GetPath())}")
            m = fprim.GetAttribute("omni:fabric:worldMatrix").Get()
            rows = [math.sqrt(sum(m[r][c] ** 2 for c in range(3))) for r in range(3)]
            i = "XYZ".index(UsdGeom.Cylinder(prim).GetAxisAttr().Get())
            out[leg] = (rows[i], rows[(i + 1) % 3])
    return out


want = (0.095 / 0.040, 0.045 / 0.046)
failures = []
for phase in ("before_render", "after_40_steps"):
    if phase == "after_40_steps":
        for _ in range(40):
            sim.step(render=True)
    for leg, (axial, radial) in world_scales().items():
        ok = abs(axial - want[0]) < 1e-4 and abs(radial - want[1]) < 1e-4
        print(f"[check] {phase} knee_{leg}: axial {axial:.5f} radial {radial:.5f} {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"{phase} knee_{leg}")
rgb = np.asarray(annot.get_data())
if rgb.size:
    Image.fromarray(rgb[..., :3]).save(os.path.join(OUT, "top_written_env0_reference_env1.png"))
print(f"[check] {'PASSED' if not failures else 'FAILED ' + ', '.join(failures)}", flush=True)
os._exit(1 if failures else 0)
