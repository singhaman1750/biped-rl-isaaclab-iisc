"""Sandbox only. Dump every observation group and term of every task of the four robot families, from the configurations."""
import argparse, json
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(["--headless"])
app = AppLauncher(args).app

import gymnasium as gym
import environments.tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg

fams = ("Isaac-Limx-SF-", "Isaac-Quadruped-", "Isaac-Limx-SDBRS1-", "Isaac-Limx-Kscale-")
out = {}
for tid in sorted(gym.registry):
    if not tid.startswith(fams) or "Play" in tid:
        continue
    env_cfg = load_cfg_from_registry(tid, "env_cfg_entry_point")
    agent = gym.registry[tid].kwargs["rsl_rl_cfg_entry_point"]
    groups = {}
    for gname, g in vars(env_cfg.observations).items():
        if not isinstance(g, ObservationGroupCfg):
            continue
        terms = {}
        for tname, t in vars(g).items():
            if isinstance(t, ObservationTermCfg):
                params = {k: (v.name + ":" + str(getattr(v, "body_names", None)) if hasattr(v, "name") else repr(v)) for k, v in (t.params or {}).items()}
                terms[tname] = {"func": getattr(t.func, "__name__", repr(t.func)), "params": params}
        groups[gname] = {"history": g.history_length, "flatten": g.flatten_history_dim, "terms": terms}
    robot = env_cfg.scene.robot
    spawn = type(robot.spawn).__name__ + ":" + str(getattr(robot.spawn, "usd_path", None) or getattr(robot.spawn, "asset_path", None) or getattr(robot.spawn, "file_path", None))
    out[tid] = {"agent": type(agent).__name__, "env_cfg": type(env_cfg).__name__, "spawn": spawn, "groups": groups}
json.dump(out, open("/tmp/symv2/obs_terms.json", "w"), indent=1)
print("dumped", len(out))
app.close()
