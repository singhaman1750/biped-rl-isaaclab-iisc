"""Static checks on rew_hip_lateral_sway that do not need the simulator.

Reproduces the two gates the RewardManager applies at manager_base.py:361 and :401, being the
signature match against the configured params and the SceneEntityCfg body resolution.
"""
import ast, inspect, re, numpy as np
CFG = "/ws/tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py"
REW = "/ws/tron1-rl-isaaclab-cozum/environments/environments/tasks/locomotion/mdp/rewards.py"

# --- the configured params, read out of the config with ast -------------------------------
tree = ast.parse(open(CFG).read())
params = None
for n in ast.walk(tree):
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "rew_hip_lateral_sway":
        for kw in n.value.keywords:
            if kw.arg == "params":
                params = [k.value for k in kw.value.keys]
            if kw.arg == "weight":
                weight = ast.literal_eval(kw.value)
assert params, "rew_hip_lateral_sway not found"
print(f"configured weight {weight}, params {params}")

# --- the __call__ signature, read out of rewards.py with ast ------------------------------
rt = ast.parse(open(REW).read())
cls = next(n for n in ast.walk(rt) if isinstance(n, ast.ClassDef) and n.name == "HipLateralSwayReward")
call = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "__call__")
a = call.args
names = [x.arg for x in a.args]
n_def = len(a.defaults)
no_def, with_def = names[:len(names)-n_def], names[len(names)-n_def:]
print(f"__call__ mandatory {no_def}, optional {with_def}")

# manager_base.py:361-374, min_argc = 1 for rewards, +1 for self on a class term
args = no_def + with_def
min_argc = 2
lhs, rhs = set(args[min_argc:]), set(params + with_def)
print(f"\nsignature gate: {'PASS' if lhs == rhs else 'FAIL'}")
if lhs != rhs:
    print("  only in signature:", lhs - rhs, "\n  only in config:", rhs - lhs)

# --- body resolution against the real body list -------------------------------------------
BODIES = list(np.load("/ws/IsaacLab/logs/rsl_rl/kscale_flat/2026-09-21_06-45-40/data/42/dump.npy",
                      allow_pickle=True).item()["body_names"])
src = open(CFG).read()
torso = re.search(r'^_TORSO_LINK = "(.*)"', src, re.M).group(1)
feet  = re.search(r'^_FOOT_LINKS = "(.*)"', src, re.M).group(1)
mt = [b for b in BODIES if re.fullmatch(torso, b)]
mf = [b for b in BODIES if re.fullmatch(feet, b)]
print(f"\n_TORSO_LINK {torso!r} -> {len(mt)} body  {mt}   (__init__ asserts exactly 1)")
print(f"_FOOT_LINKS {feet!r} -> {len(mf)} bodies {mf}   (__init__ asserts exactly 2)")
assert len(mt) == 1 and len(mf) == 2
print("\nboth asserts in HipLateralSwayReward.__init__ will pass")
