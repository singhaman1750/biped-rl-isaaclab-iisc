# Redrawing a housing cylinder resized in place, from its spawned size and through the Fabric hierarchy

Status, IMPLEMENTED 2026-11-17 and verified in container gpu1 by `co_optimisation/tests/sim_render_refresh_check.py`, every resized knee drawn at its written size before any render and after 40 physics steps, with the host suites passing 162 with 8 skipped. The outcome is recorded in section 10 of [../context/co_optimisation_audit.md](../context/co_optimisation_audit.md). It followed the plan without divergence. The third verification, a training video, awaits the next launch. Line numbers below are those of the tree before the change.

## Context

The quadruped runs `quadruped_copt/2026-11-17_05-25-55` and `2026-11-17_07-11-53` draw every housing resized at an in place swap at its spawned size, the gaps of 37.5 and 27.5 mm at the knee that `_scaled_cylinder` was written on 2026-11-17 to close. Three diagnostic runs in container gpu1 isolated two compounding causes, recorded in the row of section 10 of `context/co_optimisation_audit.md` that isolates the cause and in `context/artefacts/render-refresh-diagnostic-2026-11-17/`.

1. After the in place write and its `sim.reset()`, the Fabric copy of a housing cylinder carries the written `radius` and `height` while the renderer keeps the spawned tessellation. `_scaled_cylinder` takes the spawned size from that copy (`co_optimisation/co_optimisation/utils/update.py:316-323`), finds no change and returns None, so the refresh skips every resized cylinder in silence.
2. A `omni:fabric:localMatrix` written straight after the reset is not carried into `omni:fabric:worldMatrix`, which the renderer draws from and which Fabric recomputes only in `IFabricHierarchy.update_world_xforms()` under its change tracking. A correct scaled matrix was written and drawn unchanged.

Run 3 drew every knee as the freshly converted reference draws it, and still so after 40 physics steps, once the spawned size was taken from the spawned design and `update_world_xforms()` was called after the writes. The change below does exactly that. Physics is untouched throughout, since only the visual Fabric copies are written.

## Changes

### 1. `_scaled_cylinder` takes the spawned size where given, `co_optimisation/co_optimisation/utils/update.py:307-329`

An optional argument whose default, None, reproduces the present reading from the Fabric copy.

```python
def _scaled_cylinder(prim, matrix, fabric_copy, spawned=None):
    """The local matrix and extent that draw a spawned cylinder at the size its prototype now carries.

    The RTX delegate keeps the tessellation it populated at spawn, so the cylinder is drawn as that
    tessellation scaled in its own frame by the ratio of the written size to the spawned one.
    *spawned*, the (radius, height) in metres the environment's design was spawned with, is that
    baseline where given. None reads it from the Fabric copy's own radius and height, which the reset
    after an in place write overwrites with the written size while the renderer keeps the spawned
    tessellation, so that the copy then reports no change and the cylinder is left undrawn, see the
    row of section 10 of context/co_optimisation_audit.md that isolates the cause.
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
    # The spawned cylinder's own extent about its true axis, which the scaled matrix then carries.
    return scaled, (Gf.Vec3d(*(-v for v in half)), Gf.Vec3d(*half))
```

The ratio and extent arithmetic is unchanged, run 3 having confirmed it.

### 2. `refresh_rendered_geometry` takes the spawned population and propagates its writes, `update.py:332-411`

A new optional `spawned_link_params_list`, whose default reproduces the present baseline, and a call of `update_world_xforms()` after the writes. The housing's spawned size is the entry's `radius` and twice its `half_length`, as `_housing` writes them (`co_optimisation/co_optimisation/runners/usd_generator.py:115-128`).

```python
def refresh_rendered_geometry(
    env: ManagerBasedRLEnv,
    link_length_params_list: list[dict[str, dict]],
    root_body: str,
    spawned_link_params_list: list[dict[str, dict]] | None = None,
) -> int:
    """... (docstring as now, with the two paragraphs below appended)

    *spawned_link_params_list*, the link dicts of the population last spawned, gives each housing's
    spawned size, design ``i`` having been spawned as entry ``i % len(spawned_link_params_list)``.
    None reads the spawned size from the Fabric copy, which the reset has already overwritten, so
    that a resized cylinder is not redrawn, the defect this argument repairs.

    The writes are carried into ``omni:fabric:worldMatrix``, which the renderer draws from, by
    ``IFabricHierarchy.update_world_xforms()``, since Fabric's change tracking does not pick up a
    local matrix written straight after the reset.
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
            ...                                    # unchanged down to the cylinder branch
                else:
                    entry = (spawned_params or {}).get(link_name) or {}
                    spawned = ((entry["radius"], 2.0 * entry["half_length"])
                               if "radius" in entry and "half_length" in entry else None)
                    copy = rt_stage.GetPrimAtPath(f"{env_paths[individual]}/{rel}")
                    redraw = _scaled_cylinder(prim, matrix, copy, spawned)
                    if redraw is None:
                        continue
                    matrix, extent = redraw
                    redrawn += 1
                ...                                # unchanged, the entry appended
        per_design.append(entries)
    written = 0
    for e, env_path in enumerate(env_paths):
        ...                                        # unchanged write loop
    # The local matrices reach the world matrices the renderer draws from only through the hierarchy.
    usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId()
    ).update_world_xforms()
    if redrawn:
        print(f"[refresh] {redrawn} resized cylinders redrawn from their spawned size")
    return written
```

`update_world_xforms()` changes nothing where no local matrix changed, and Kit already calls it before every frame, so the call only brings forward a propagation the frame would otherwise lose. It is unconditional, since it affects only the drawing and never the physics or the observations of any run.

### 3. The runner records the spawned population, `co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`

Every spawn passes through `_respawn_population`, the first population through `_reload_morphology` (`:668-688`), a resume (`:326`) and the respawn when the random phase ends (`:813`), so it is the one place to record it.

```python
# __init__, beside _restored_population (:205-207)
        # The link dicts of the population last spawned, each housing's spawned size, which the
        # in place redraw scales from. Set by _respawn_population.
        self._spawned_link_params: list[dict] | None = None

# _respawn_population, after respawn_robots (:640-645)
        try:
            self.respawn_robots(unwrapped_env, population.get_usd_files())
        finally:
            unwrapped_env._suppress_terrain_curriculum = False
            unwrapped_env._suppress_curriculum = False
        self._spawned_link_params = population.get_link_length_params()

# _refresh_rendered_geometry (:747-756)
        written = refresh_rendered_geometry(
            self.env.unwrapped,
            self.current_population.get_link_length_params(),
            self.current_population.get_root_body(),
            self._spawned_link_params,
        )
```

On resume the restored population is both spawned and written, so no housing differs from its spawned size and nothing is scaled, as now. The respawn path of `_update_morphology` spawns the new population and does not refresh, which is unchanged.

### 4. A simulator check, `co_optimisation/tests/sim_render_refresh_check.py`

New, in the manner of the other `sim_*_check.py` scripts, not collected by pytest. It is the diagnostic of `context/artefacts/render-refresh-diagnostic-2026-11-17/cylinder_redraw_diag.py` reduced to the runner's own sequence: env_0 spawns the A1 quadruped and env_1 the ANYdrive-3 knee design, `apply_link_length_params` writes the ANYdrive-3 design into env_0, and `refresh_rendered_geometry(env, [links_b], root, [links_a])` is called straight after, with no usdrt read before it. It then asserts that every knee cylinder copy of env_0 has a `omni:fabric:worldMatrix` whose scale along the cylinder's axis is 0.095 / 0.040 and across it 0.045 / 0.046 within 1e-4, before any render and after 40 physics steps, and writes a top view of both environments for inspection. The output path is made absolute and an exception exits the process, the two lessons of the diagnostic.

    ./isaaclab.sh -p biped/co_optimisation/tests/sim_render_refresh_check.py /workspace/isaaclab/logs/<dir>

## Defect left standing

With `spawned_link_params_list` unset, `refresh_rendered_geometry` still reads the spawned size from the Fabric copy and skips a resized cylinder. The runner is its only caller and sets it, and the docstring records the defect so that another caller opts in deliberately.

## Verification

1. The host suites, `design_generator_test.py` and `design_filter_test.py`, unaffected but rerun, through the scratch runner of 2026-11-17.
2. `sim_render_refresh_check.py` in container gpu1, which must pass its assertions, with its image compared against env_1, free memory confirmed above 10 GB beforehand and the training run's next swap avoided.
3. The next quadruped run with `--video`, whose step 10000 video must draw environment 0's ANYdrive-3 knees as 95 mm drums flush with the thigh, and whose console must print the `[refresh]` line at each swap.

## Documentation, in the same pass

1. `context/co_optimisation_audit.md`, a dated outcome under the row isolating the cause.
2. This plan as `plans/cylinder_render_refresh.md`, registered in `plans/README.md` with its status.
3. `plans/housing_clearance_and_trunk_width.md` section 7a, a dated sentence that the first redraw did not take effect and pointing to the new plan.
