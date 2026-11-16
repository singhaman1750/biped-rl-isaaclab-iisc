"""Sandbox only. Insert runtime prints into common.compile_maps, so a smoke run records the names, orders, shapes, limits and layouts."""
import sys

path = sys.argv[1]
src = open(path).read()
probe = '''    print("[symmetry-probe] joint_names", list(robot.joint_names))
    print("[symmetry-probe] jperm", jperm.tolist(), "jsign", jsign.tolist())
    print("[symmetry-probe] body_names", list(robot.body_names))
    print("[symmetry-probe] body_perm", ctx["body_perm"])
    print("[symmetry-probe] shapes", count_shapes_per_body(robot), "max_shapes", robot.root_physx_view.max_shapes)
    print("[symmetry-probe] limits", [[round(v, 6) for v in r] for r in robot.data.joint_pos_limits[0].tolist()])
    print("[symmetry-probe] default", dict(robot.cfg.init_state.joint_pos))
    for g in om.active_terms:
        print("[symmetry-probe] group", g, list(zip(om.active_terms[g], [tuple(int(x) for x in d) for d in om.group_obs_term_dim[g]],
                                                    [c.history_length for c in om._group_obs_term_cfgs[g]])))
    for name, s in unwrapped.scene.sensors.items():
        if hasattr(s, "num_rays"):
            print("[symmetry-probe] rays", name, s.num_rays, s.cfg.ray_alignment, s.cfg.pattern_cfg.ordering)
        else:
            print("[symmetry-probe] sensor", name, list(getattr(s, "body_names", [])))
    groups = {}'''
assert src.count("    groups = {}") == 1
open(path, "w").write(src.replace("    groups = {}", probe))
print("probe inserted")
