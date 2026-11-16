"""Run register for the action scale and lateral sway ablation of 2026-09-21 and 2026-09-22.

The three runs carry the labels the user's analysis brief assigns them, A being the reference
that ran one action scale of 0.4 for every joint, B adding the per joint action scale of arm 3
and C adding the hip lateral sway reward of arm 4 on top of that.  Two quantities move with the
arms without belonging to them, the push curriculum ceiling falling from 3.0 to 2.0 m/s between
A and B and the parallel environment count falling from 9000 to 6400 between B and C, and both
are recorded here because every comparison in this directory must be read against them.
"""
import numpy as np, os
ROOT = "/ws/IsaacLab/logs/rsl_rl/kscale_flat"
RUNS = {"A_scale0.4": "2026-09-21_06-45-40",
        "B_arm3":     "2026-09-22_07-18-05",
        "C_arm3+4":   "2026-09-22_07-40-43"}
NUM_ENVS = {"A_scale0.4": 9000, "B_arm3": 9000, "C_arm3+4": 6400}
PUSH_MAX = {"A_scale0.4": 3.0,  "B_arm3": 2.0,  "C_arm3+4": 2.0}

def load(run, which="dump"):
    return np.load(os.path.join(ROOT, RUNS[run], f"data/42/{which}.npy"), allow_pickle=True).item()

def envelope(tau_sat, tau_lim, v_lim, v):
    """DCMotor available torque magnitude at signed velocity v, actuator_pd.py:295-307."""
    v_at_lim = v_lim * (1.0 + tau_lim / tau_sat)
    vc = np.clip(v, -v_at_lim, v_at_lim)
    top = np.minimum(tau_sat * (1.0 - vc / v_lim), tau_lim)
    bot = np.maximum(tau_sat * (-1.0 - vc / v_lim), -tau_lim)
    return bot, top
