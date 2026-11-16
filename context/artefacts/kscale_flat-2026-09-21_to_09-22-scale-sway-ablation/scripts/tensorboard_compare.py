"""Training progression of the three runs, compared at a matched iteration count.

The environment count is not matched across the set, A and B collecting 9000 parallel
environments and C collecting 6400, so an iteration of C carries 0.711 of the experience an
iteration of A or B carries.  Both readings are printed, at a matched iteration and at a matched
sample budget, because a comparison at a matched iteration flatters C and one at a matched
sample count flatters A and B, and the honest statement needs both.
"""
import os, numpy as np, common
from tensorboard.backend.event_processing import event_accumulator

TAGS = ["Train/mean_reward","Train/mean_episode_length","Metrics/base_velocity/error_vel_xy",
        "Metrics/base_velocity/error_vel_yaw","Episode_Termination/base_contact",
        "Episode_Termination/time_out","Policy/mean_noise_std","Loss/entropy","Loss/value_function",
        "Loss/surrogate","Curriculum/modify_push_force","Curriculum/modify_command_velocity_lin_x",
        "Episode_Reward/pen_joint_torque","Episode_Reward/pen_action_rate",
        "Episode_Reward/pen_action_smoothness","Episode_Reward/pen_joint_torque_rate",
        "Episode_Reward/pen_joint_accel","Episode_Reward/pen_joint_vel_l2",
        "Episode_Reward/pen_joint_pos_limits","Episode_Reward/pen_hip_roll_deviation",
        "Episode_Reward/rew_lin_vel_xy","Episode_Reward/rew_gait","Episode_Reward/pen_base_height",
        "Episode_Reward/rew_keep_ankle_roll_zero_in_air","Episode_Reward/pen_feet_distance",
        "Episode_Reward/termination_penalty","Perf/total_fps"]

def series(run):
    ea = event_accumulator.EventAccumulator(os.path.join(common.ROOT, common.RUNS[run]),
                                            size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for t in TAGS:
        if t in ea.Tags()["scalars"]:
            ev = ea.Scalars(t)
            out[t] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out

S = {r: series(r) for r in common.RUNS}
print("recorded iteration span")
for r in common.RUNS:
    st = S[r]["Train/mean_reward"][0]
    print(f"  {r:12} {st.min():6d} to {st.max():6d}, {len(st):5d} points, {common.NUM_ENVS[r]} envs, "
          f"{common.NUM_ENVS[r]*24*(st.max()+1)/1e6:8.1f} M steps at 24 per iteration")

def at(run, tag, it):
    if tag not in S[run]: return None
    st, v = S[run][tag]
    i = int(np.argmin(np.abs(st - it)))
    return v[i]

for IT in (3000, 6000, 10000):
    print(f"\n=== matched ITERATION {IT} ===")
    print(f"{'tag':48}" + "".join(f"{r:>14}" for r in common.RUNS))
    for t in TAGS:
        vals = [at(r, t, IT) for r in common.RUNS]
        if all(v is None for v in vals): continue
        print(f"{t:48}" + "".join(f"{v:>14.4f}" if v is not None else f"{'-':>14}" for v in vals))

# matched sample budget: C needs 9000/6400 = 1.40625 times as many iterations
print("\n=== matched SAMPLE BUDGET, 9000 envs x 10000 iterations ===")
IT_FOR = {"A_scale0.4": 10000, "B_arm3": 10000, "C_arm3+4": int(round(10000*9000/6400))}
print("iterations used:", IT_FOR)
print(f"{'tag':48}" + "".join(f"{r:>14}" for r in common.RUNS))
for t in TAGS:
    vals = [at(r, t, IT_FOR[r]) for r in common.RUNS]
    if all(v is None for v in vals): continue
    print(f"{t:48}" + "".join(f"{v:>14.4f}" if v is not None else f"{'-':>14}" for v in vals))
