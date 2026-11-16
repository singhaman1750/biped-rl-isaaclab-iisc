"""Compare the working tree's actuator values against those the 17 N.m run dumped."""
import sys, re
sys.path.insert(0, "/ws/tron1-rl-isaaclab-cozum/scripts/analysis")
from experiment_params import load_params
RUN = "/ws/IsaacLab/logs/rsl_rl/kscale_flat/2026-09-21_06-45-40"
CFG = "/ws/tron1-rl-isaaclab-cozum/environments/environments/assets/config/kscale_identified_cfg.py"
src = open(CFG).read()
blocks = dict(re.findall(r"KSCALE_(\w+)_ACTUATOR_CFG = IdentifiedActuatorCfg\((.*?)\n\)", src, re.S))
run = load_params(RUN)["scene"]["robot"]["actuators"]
FIELDS = ["effort_limit","saturation_effort","velocity_limit","velocity_limit_sim",
          "stiffness","damping","armature","friction_static","friction_dynamic","activation_vel"]
print(f"{'group':11}{'field':20}{'tree':>12}{'run 17Nm':>12}   match")
bad=0
for name, body in blocks.items():
    g = name.lower()
    body = "\n".join(l for l in body.splitlines() if not l.strip().startswith("#"))
    for f in FIELDS:
        m = re.search(rf"\b{f}=(\{{\".\*\": )?([-\d.]+)", body)
        if not m: continue
        tv = float(m.group(2))
        rv = run[g][f]
        rv = float(list(rv.values())[0]) if isinstance(rv, dict) else float(rv)
        ok = abs(tv-rv) < 1e-9
        if not ok: bad += 1
        print(f"{g:11}{f:20}{tv:>12.5g}{rv:>12.5g}   {'ok' if ok else 'DIFFERS'}")
print(f"\n{bad} field(s) differ between the working tree and the 17 N.m run")
