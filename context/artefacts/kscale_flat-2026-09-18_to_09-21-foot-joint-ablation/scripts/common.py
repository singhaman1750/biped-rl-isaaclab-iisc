import numpy as np, os
ROOT = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"tau11.9": "2026-09-18_09-37-29", "tau17": "2026-09-21_06-45-40", "tau34": "2026-09-21_08-40-59"}

def load(run):
    d = np.load(os.path.join(ROOT, RUNS[run], "data/42/dump.npy"), allow_pickle=True).item()
    return d

def envelope(tau_sat, tau_lim, v_lim, v):
    """DCMotor available torque magnitude at signed velocity v, actuator_pd.py:295-307."""
    v_at_lim = v_lim * (1.0 + tau_lim / tau_sat)
    vc = np.clip(v, -v_at_lim, v_at_lim)
    top = np.minimum(tau_sat * (1.0 - vc / v_lim), tau_lim)
    bot = np.maximum(tau_sat * (-1.0 - vc / v_lim), -tau_lim)
    return bot, top
