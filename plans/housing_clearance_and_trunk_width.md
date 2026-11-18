# Housing Clearance and Trunk Width in the Design Builders

STATUS, 2026-11-16. IMPLEMENTED 2026-11-16 in the working tree of `tron1-rl-isaaclab-cozum` and its `design_generator` submodule, uncommitted, with the host tests passing, 120 with 10 skipped. NOT YET VERIFIED in Isaac Sim, the in place edit of the trunk being new editor behaviour that this host cannot exercise, so no run should train on the change until the checks of section 6 pass in the training container. One divergence from the plan as first approved, recorded in section 7.

## 1. Context

Run `copt_moral/2026-11-12_06-55-42` trained on 37 designs whose abduction housing intersects the knee housing at the nominal pose, 19 of the 2800 evaluated and 18 of the random phase, each scoring a survival of 0.0 and a single step return, the last drawn in generation 46. The hip was placed at `r_hip + clearance` from the abduction face and the knee shares the hip's x, so a knee wider than the hip plus its clearance reaches back into the abduction housing, which for a 50 mm ODrive hip beside a 96.6 mm knee is 25 + 20 − 48.3 = −3.3 mm, the overlap measured. The standing filter cannot see this and `check_geometry` refused only housings that were too large (`context/design_filtering.md` section 8).

The user chose to repair the layout in the builders, so that every design is feasible and only torque and pose tests filter designs, the CMA-ES feasibility pipeline being left as it is. Four decisions followed. The hip offset takes the knee radius into account with a margin of 0.5 mm in both builders. The quadruped takes the radius and half length of all three housings from the catalogue, with real sizes copied from the same named biped entries. A new rule sets the trunk width from the abduction actuator's diameter in both builders. A note on the topology the rules assume is kept in the local context.

## 2. Why a layout repair and not a check

A rejecting check in `_leg_xml` is the smallest edit and not the simplest correct one. A `ValueError` from a builder is caught by `_build_xml`, which substitutes the last feasible design while CMA-ES is still credited with the substitute's fitness, and `update_with_fitness` masks nothing (`co_optimisation/co_optimisation/runners/usd_generator.py:873-877`, `:725`). A repair keeps the design feasible and unique, and it is shared by the emitter, the in place editor and the nominal pose chain through `_cluster_layout`. On the 4301 designs of the run with emitted documents, the gap `r_hip + clr_abad2hip_x − clr_hip2knee_x − r_knee` separates the 37 overlapping designs from the others exactly, the first lying between −8.5 and −3.0 mm and the others no lower than +1.0 mm.

## 3. Rules and constants

Constants live in each family's `constant_params.json`, so no keyword plumbing was needed. `HOUSING_MARGIN` is 0.0005 m in both, and `TRUNK_WIDTH_OFFSET` is 0.0668 m for the biped and 0.012 m for the quadruped, the values that reproduce each baseline exactly. The quadruped's six housing constants and the derived `HIP_ACT_OFFSET_X`, `KNEE_OFFSET_Y`, `MIN_TRUNK_WIDTH` and `TRUNK_THICKNESS` are removed.

Rule 6, the hip's offset from the abduction face is `max(r_hip + clr_abadAct2hipAct_x, r_knee + HOUSING_MARGIN − u * clr_hipAct2kneeAct_x)`, with `u` the direction of the hip from the abduction, −1 for the biped and `sx` for the quadruped. It equals the former offset, to the last bit, wherever the knee clears.

Rule 5, the trunk width is `2 * d_abad + TRUNK_WIDTH_OFFSET`, raised where the first term lets two housings touch. In the biped the second term is `2 * (r_abad + r_hip − clr_trunk2abadAct_y − clr_abadAct2hipAct_y) + HOUSING_MARGIN`, the smallest width at which the hip offset's square root is real. In the quadruped it is `2 * (r_abad + hl_hip − clr_trunk2abadAct_y − clr_abadAct2hipAct_y) + HOUSING_MARGIN`, which keeps the two hip housings of an axle apart.

## 4. Changes made

1. `design_generator/biped/builder.py`, `trunk_width`, `trunk_body` and the Rule 6 offset in `_cluster_layout`, `check_geometry` and `build_xml` reading the derived width, and the constants.
2. `design_generator/quadruped/builder.py`, `_housing_dims`, `trunk_width`, `trunk_thickness`, `trunk_body`, a new `_cluster_layout` read by `_leg_xml`, `abd_x_offset` and `abd_y_offset` taking the housing size, and the constants.
3. `co_optimisation/co_optimisation/runners/usd_generator.py`, `_trunk_entry`, the trunk entry of both link dicts and the quadruped's housing sizes and parents from `_cluster_layout`.
4. `co_optimisation/co_optimisation/utils/update.py`, an entry named for the root body is the root prim itself, in `apply_link_length_params` and `refresh_rendered_geometry`.
5. `design_generator/quadruped/catalogs/actuators.json`, `diameter` and `length` copied into 11 entries from the biped catalogue, with a note in each.
6. `GEOMETRY_RULES.md` of both families, `DESIGN_PARAMETERS.md` of both, `quadruped/README.md` and the builder docstrings.
7. `co_optimisation/tests/design_generator_test.py`, ten new tests and the fallback test repointed at a 1 cm thigh.
8. `context/design_filtering.md` section 9, the artefact register and the context index.

## 5. Compatibility

The new behaviour was directed, so it replaces the old outright, with no flag. Where the knee clears and the abduction diameter is that of the baseline, documents are identical to the former builder's, which the run's 4380 design records confirm for 2677 designs (`context/artefacts/copt_moral-2026-11-12_06-55-42-design-viability/rebuild_with_housing_rules.txt`). Other abduction picks now get another trunk width, so replays of earlier designs with them differ. In the quadruped the trunk height already followed the abduction diameter by Rule 4 and now varies with the pick, and every non A1 pick changes housing size.

## 6. Verification

Done on this host. The suites `design_generator_test.py`, `design_filter_test.py` and `catalog_fill_test.py` pass, 120 with 10 skipped, including the baseline tests that pin both families to their committed assets and ten new tests that build every abduction, hip and knee combination of each shelf, 3375 and 2744, with no exception and no contact. The rebuild of the run's design records is the second check.

Still to do in the training container, since Isaac Sim is not available here. Reinstall `co_optimisation` and `design_generator`, run `sim_asset_equivalence_check.py` and `sim_nominal_pose_check.py`, and run a new check that writes a population of different abduction picks in place and reads back the trunk box size, its mass and the housing positions against the link dict. The fallback if the trunk edit does not take effect is to respawn the population when the abduction pick changes.

## 7. Divergence from the plan as approved

The quadruped's width rule was first the abduction term alone, as the user stated it. The new contact test found the left and right hip housings of an axle touching in 56 of 2744 combinations, a narrow abduction housing carrying a long hip one, which the baseline and the A1 entries never meet. The second term of Rule 5 for the quadruped was added to close it, as the biped's already was for the one pair of its shelf that the abduction term left infeasible.

## 7a. Rendering regression found 2026-11-17

Run `quadruped_copt/2026-11-16_08-05-55` showed gaps between the hip and knee housings that the documents do not contain. The cause lay outside this plan's files. `refresh_rendered_geometry` in `co_optimisation/co_optimisation/utils/update.py` redrew boxes only, on the premise that no quadruped housing changes size, which this plan voided. The refresh now redraws a resized cylinder by scale, as recorded in the 2026-11-17 correction in section 10 of `context/co_optimisation_audit.md`. Physics was unaffected. Addition of 2026-11-17, later the same day. That redraw did not take effect in runs `quadruped_copt/2026-11-17_05-25-55` and `2026-11-17_07-11-53`, the refresh skipping every resized cylinder and its writes not reaching the world matrices the renderer draws from. It was repaired by [cylinder_render_refresh.md](cylinder_render_refresh.md).

## 8. Risks

The in place trunk edit is unverified, as stated. The width rule makes leg track and trunk inertia depend on the abduction pick, which CMA-ES will see as part of that pick's effect. The eleven quadruped entries took the biped catalogue's housing sizes, and their masses differ for five of them.
