# The Quadruped Co-optimisation Reward, a Framework Neutral Specification

Written 2026-10-22 against `../environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py`, `../environments/environments/tasks/locomotion/mdp/rewards.py`, `../environments/environments/tasks/locomotion/mdp/curriculums.py` and the vendored Isaac Lab under `../../IsaacLab/source/isaaclab/isaaclab/`. Paths below are relative to this file.

## 1. Purpose and scope

This document specifies the reward of the quadruped co-optimisation task, `QuadrupedPFCoptEnvCfg` with `CoptRewardsCfg` and `CoptCurriculumCfg` (`base_env_cfg.py:1230-1271`, `:1426-1503`, `:1573-1588`), precisely enough to reimplement it in Isaac Gym, MuJoCo or any other simulator. Each term is given as a formula, its weight and parameters, a reference implementation in batched tensor code, and the reason it is there. Section 2 fixes the conventions every term relies on, sections 3 to 6 give the terms, section 7 the per design scaling that distinguishes co-optimisation, section 8 the curricula that change the reward during training, and section 9 a porting checklist.

The reward set descends from the legged_gym formulation [1], to which this task adds terrain referenced swing terms, an impact term in the Humanoid-Gym form [2], an abduction penalty from Extreme Parkour [3], and the scaling of three terms by each design's standing height.

## 2. Conventions

Timing. Physics runs at $\Delta t_{sim} = 0.005$ s with a decimation of 4, so one policy step lasts $\Delta t = 0.02$ s. An episode lasts 20 s, so it holds at most $T = 1000$ steps (`base_env_cfg.py:1590-1596`).

Aggregation. The manager multiplies every term by its weight and by $\Delta t$ (`../../IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:150`), so the per step reward is

$$r_t = \Delta t \sum_i w_i \, f_i(s_t, a_t).$$

A port that omits $\Delta t$ must divide every weight by 50 to reproduce the same returns. The weights below are the configured $w_i$, before that factor.

Frames. A subscript $b$ denotes the base (trunk) frame and $w$ the world frame. Base velocities $v_b, \omega_b$ are the root velocities expressed in the base frame. The projected gravity $g_b$ is the unit gravity vector in the base frame, so $g_b = (0, 0, -1)$ when the trunk is level.

Commands. The command is $c = (c_x, c_y, c_\omega)$ in the base frame, resampled every 7.5 to 12.5 s (`base_env_cfg.py:113-127`). Every environment runs in heading mode, so the yaw rate command is $c_\omega = \mathrm{clip}(0.5 \, \mathrm{wrap}(\psi^* - \psi), \pm c_{\omega,\max})$ for a sampled heading $\psi^*$ (`../../IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py:154-156`). Two percent of environments receive a zero command. The initial ranges are $\pm 1.0$, $\pm 0.1$ and $\pm 0.75$ and the curriculum of section 8 widens them.

Contacts. A contact sensor on every body stores the net contact force at each physics step with a history of 4 entries, which spans exactly one policy step. Every contact test in this document takes the maximum norm over that history and compares it with a threshold, so a contact between two policy steps is never missed. The sensor also keeps, per foot, the running air time $t^{cur}_{air}$, the air time of the last completed swing $t^{last}_{air}$, and the running contact time $t^{cur}_{con}$, with a body in contact when its force exceeds 1 N (`../../IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:416-445`). A first contact is a foot whose contact time lies in $(0, \Delta t]$, meaning it touched down during the last policy step (`contact_sensor.py:182`).

Terrain heights. A ray caster above the trunk casts a 1.6 m by 1.0 m grid at 0.1 m resolution straight down, its pattern rotated with the base yaw only (`base_env_cfg.py:75-82`). Its hit points $p^{hit}_k$ give the ground height under each grid node.

Per design height. After every design swap the co-optimisation runner measures each environment's standing height $h_e$, the mean depth of the feet below the trunk in the nominal pose plus the sole offset, and stores it as `env.copt_nominal_base_height` (`../co_optimisation/co_optimisation/utils/update.py:509`, `../co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:583`). Before any design is applied the attribute is absent and every scaled term falls back to the donor value $h_0 = 0.35$ m.

Robot naming. The asset follows Unitree naming, `<leg>_hip_joint` for abduction, `<leg>_thigh_joint` for hip pitch and `<leg>_calf_joint` for the knee, with feet `foot_<leg>_Link` modelled as spheres of radius 0.022 m. Soft joint limits are the hard range shrunk about its midpoint by a factor of 0.9 (`../environments/environments/assets/config/quadruped_identified_cfg.py:95`).

## 3. Summary table

| Term | $f_i$ | $w_i$ | Parameters |
|---|---|---|---|
| `termination_penalty` | $\mathbb{1}[\text{terminated}]$ | -2000 | time-outs excluded |
| `keep_balance` | $1$ | 0.05 | |
| `rew_lin_vel_xy` | $\exp(-\lVert c_{xy} - v_{b,xy} \rVert^2 / \sigma_v^2)$ | 10 | $\sigma_v = 0.4$, curriculum |
| `rew_ang_vel_z` | $\exp(-(c_\omega - \omega_{b,z})^2 / \sigma_\omega^2)$ | 5 | $\sigma_\omega = 0.4$, curriculum |
| `pen_base_height` | $(\bar z - h_e)^2$ | -15 | section 4.1 |
| `pen_lin_vel_z` | $v_{b,z}^2$ | -2.0 | |
| `pen_ang_vel_xy` | $\omega_{b,x}^2 + \omega_{b,y}^2$ | -0.05 | |
| `pen_flat_orientation` | $g_{b,x}^2 + g_{b,y}^2$ | -1.0 | |
| `pen_joint_torque` | $\sum_j \tau_j^2$ | -2.0e-4 | applied torque |
| `pen_joint_accel` | $\sum_j \ddot q_j^2$ | -2.5e-7 | |
| `pen_joint_vel_l2` | $\sum_j \dot q_j^2$ | -5.0e-5 | |
| `pen_joint_pos_limits` | $\sum_j \text{excess}_j$ | -2.0 | soft limits |
| `pen_abad_deviation` | $\sum_{j \in abad} \lvert q_j - \bar q_j \rvert$ | -0.5 | |
| `pen_action_rate` | $\lVert a_t - a_{t-1} \rVert^2$ | -0.01 | |
| `pen_action_smoothness` | $\lVert a_t - 2a_{t-1} + a_{t-2} \rVert^2$ | -0.01 | |
| `pen_undesired_contacts` | count of non foot bodies in contact | -2.5 | 10 N |
| `feet_air_time` | section 5.1 | 12.0 | $t_{min} = 0.15$, $t_{max} = 0.45$ s, scaled |
| `pen_feet_hold` | $\sum_f \max(0, t^{cur}_{air,f} - 0.45)$ | -1.0 | command gated |
| `feet_slide` | $\sum_f \lVert v_{f,xy} \rVert \, \mathbb{1}[\text{contact}_f]$ | -0.25 | 1 N |
| `rew_foot_clearance` | section 5.4 | 0.5 | target $0.10 \, h_e / h_0$ m |
| `pen_foot_landing_vel` | section 5.5 | -0.3 | 0.04 m window |
| `pen_feet_impact` | $\sum_f \mathrm{clip}(F_f - 160, 0, 2000)$ | -5.0e-4 | |

The weights are those of `RewardsCfg` (`base_env_cfg.py:1087-1227`) with the three overrides of `CoptRewardsCfg`.

## 4. Task, survival and base terms

Velocity tracking. The two tracking terms are Gaussian kernels on the command error, bounded in $(0, 1]$ and therefore safe to weight heavily [1]. The kernel width sets the error the policy is paid for, so it starts wide at $\sigma = 0.4$ and the curriculum of section 8 narrows it to no less than 0.09 as tracking improves. The 2 to 1 ratio of linear to angular weight follows the legged_gym convention [1]. Both use body frame velocities (`../../IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:304-330`).

```python
def track_lin_vel_xy_exp(cmd, v_b, std):        # cmd (N,3), v_b (N,3)
    return torch.exp(-((cmd[:, :2] - v_b[:, :2]) ** 2).sum(1) / std**2)

def track_ang_vel_z_exp(cmd, w_b, std):
    return torch.exp(-(cmd[:, 2] - w_b[:, 2]) ** 2 / std**2)
```

Termination and survival. The termination penalty fires on the step an episode ends by failure, never on a time-out (`rewards.py:37`). With $\Delta t$ applied it costs 40 per fall, which dominates any short term gain from a risky action. The failures are a trunk contact above 1 N and a thigh or shank contact above 1 N (`base_env_cfg.py:1274-1296`). The survival bonus of 0.05 is a constant (`../environments/environments/tasks/locomotion/mdp/rewards.py:23`).

Base stability. `pen_lin_vel_z`, `pen_ang_vel_xy` and `pen_flat_orientation` penalise bouncing, rolling and pitching rates, and trunk tilt through the horizontal components of $g_b$ [1] (`../../IsaacLab/.../envs/mdp/rewards.py:77-96`).

### 4.1 Base height above the terrain

The height of the base is measured against the mean of all ray hits beneath it, which keeps the term correct on slopes and steps (`rewards.py:2116-2152`). In the co-optimisation version each environment's target is its own design's standing height $h_e$, and a ray that hit nothing is replaced by that target so it contributes zero error (`rewards.py:2155-2183`).

$$f = \Big(z_{root} - \tfrac{1}{K}\textstyle\sum_k z^{hit}_k - h_e\Big)^2$$

```python
def base_height_rough_l2_v2(root_z, ray_hits_z, h_e):    # (N,), (N,K), (N,)
    height = root_z[:, None] - ray_hits_z
    height = torch.where(torch.isfinite(height), height, h_e[:, None].expand_as(height))
    return (height.mean(1) - h_e) ** 2
```

The reason for the per design target is recorded in the code. A fixed target makes the penalty a deterministic function of leg length, a design whose crouch stands lower being charged on every step for its geometry rather than its behaviour, which corrupts the fitness the design search ranks by (`rewards.py:2170-2175`).

## 5. Gait shaping terms

### 5.1 Feet air time with an overrun penalty

At the step a foot touches down, the term pays the duration of the swing that just ended minus a minimum, so steps shorter than $t_{min}$ are charged and longer ones paid [1]. Unlike the original form, which clamps the payment at $t_{max} - t_{min}$, this version reflects it past the cap so that an overlong swing is charged (`rewards.py:391-415`). The term is zero when the commanded planar speed is below 0.1 m/s. Both thresholds are scaled by $k_e$ of section 7 (`rewards.py:428-443`).

$$x_f = (t^{last}_{air,f} - k_e t_{min}) \, \mathbb{1}[\text{first contact}_f], \qquad g(x) = \begin{cases} x & x \le C \\ C - x & x > C \end{cases}, \qquad C = k_e (t_{max} - t_{min})$$

$$f = \mathbb{1}[\lVert c_{xy} \rVert > 0.1] \sum_f g(x_f)$$

```python
def feet_air_time_v3(first_contact, last_air_time, cmd, t_min, t_max, k):   # (N,F) bool, (N,F), (N,3), k (N,1) or 1.0
    x = (last_air_time - t_min * k) * first_contact
    cap = (t_max - t_min) * k
    x = torch.where(x > cap, cap - x, x)
    return x.sum(1) * (cmd[:, :2].norm(dim=1) > 0.1)
```

With $t_{min} = 0.15$ s and $t_{max} = 0.45$ s, the payment is largest for a swing of 0.45 s, roughly the half period of a 1 to 2 Hz trot (`base_env_cfg.py:1158-1160`). The command gate reads only $c_{xy}$, so a pure yaw command earns no air time reward.

### 5.2 Feet hold penalty

The air time term is evaluated only at touchdown, so a foot that never lands is never charged. This term reads the running air time every step and charges the excess over 0.45 s without a cap, closing that loophole (`rewards.py:2451-2472`). It carries the same command gate. Its ceiling is not scaled by $k_e$.

```python
def feet_hold_penalty(current_air_time, cmd, ceiling=0.45):
    return (current_air_time - ceiling).clamp(min=0).sum(1) * (cmd[:, :2].norm(dim=1) > 0.1)
```

### 5.3 Feet slide

Sums the planar speed of every foot in contact, contact taken as a history maximum above 1 N, using the foot's centre of mass velocity in the world frame (`rewards.py:446-453`).

### 5.4 Foot clearance above the local terrain

Each swinging foot is paid a Gaussian kernel on the error between its clearance and a target, multiplied by $\tanh(2 \lVert v_{f,xy} \rVert)$ so that only a moving foot is paid (`rewards.py:800-901`). The clearance is measured above the terrain directly beneath that foot rather than beneath the base, because on generated terrain the four feet stand on different surfaces. A foot in contact (history maximum above 1 N) earns nothing, as does a foot with no ray hit within 0.5 m. For a point foot the clearance is the height of the foot frame origin, the sphere centre, above the local ground. In the co-optimisation version the target is $0.10 \, h_e / h_0$ m, a fixed share of each design's height (`rewards.py:1122-1150`).

$$f = \sum_f \exp\!\big(-(z_f - z^{gnd}_f - z^*_e)^2 / 0.05^2\big) \tanh(2 \lVert v_{f,xy} \rVert) \, \mathbb{1}[\neg \text{contact}_f] \, \mathbb{1}[\text{valid}_f]$$

The local ground height is the maximum height among the 4 ray hits nearest the foot in the horizontal plane, ignoring hits farther than 0.5 m and rays that hit nothing. The maximum is the conservative choice at a step edge, where the foot must clear the higher surface (`rewards.py:718-798`).

```python
def terrain_height_under(points_xy, hits, k=4, max_dist=0.5):      # (N,P,2), (N,R,3)
    finite = torch.isfinite(hits).all(-1)
    d2 = ((points_xy[:, :, None, :] - hits[:, None, :, :2]) ** 2).sum(-1)
    d2 = torch.where(finite[:, None, :], d2, torch.inf)
    near_d2, idx = torch.topk(d2, min(k, hits.shape[1]), dim=-1, largest=False)
    near_z = torch.gather(hits[..., 2][:, None, :].expand(-1, points_xy.shape[1], -1), 2, idx)
    ok = torch.isfinite(near_d2) & (near_d2 <= max_dist**2)
    valid = ok.any(-1)
    z = torch.where(ok, near_z, -torch.inf).max(-1).values
    return torch.where(valid, z, 0.0), valid

def foot_clearance_v6(foot_pos, foot_vel, in_contact, hits, z_target):  # z_target scalar or (N,1)
    z_gnd, valid = terrain_height_under(foot_pos[..., :2], hits)
    err = (foot_pos[..., 2] - z_gnd - z_target) ** 2
    r = torch.exp(-err / 0.05**2) * torch.tanh(2.0 * foot_vel[..., :2].norm(dim=-1))
    return (r * valid * ~in_contact).sum(1)
```

### 5.5 Landing velocity

Charges the squared downward velocity of a foot that is about to land, defined as unloaded, descending, and within 0.04 m of the local ground measured from the bottom of the sphere, $z_f - 0.022$ (`rewards.py:171-305`). It shares the ground lookup of section 5.4 so the two terms cannot disagree about where the ground is. Position and velocity are both read at the foot link frame. The narrow window bounds what a slow descent through it can save, so the term rewards a soft final approach rather than a slow swing.

$$f = \sum_f v_{f,z}^2 \, \mathbb{1}[z_f - 0.022 - z^{gnd}_f < 0.04] \, \mathbb{1}[v_{f,z} < 0] \, \mathbb{1}[\neg \text{contact}_f] \, \mathbb{1}[\text{valid}_f]$$

### 5.6 Impact force

Charges the peak contact force of each foot over the history in excess of 160 N, clipped at 2000 N, in the Humanoid-Gym form [2] (`rewards.py:307-336`). It is exactly zero across a normal stance and so acts only on hard landings. At the configured weight it is nearly inert and serves as a guard (`base_env_cfg.py:1218-1219`).

## 6. Regularisation terms

Joint terms. `pen_joint_torque` uses the torque applied after actuator saturation, `pen_joint_accel` the finite difference of joint velocity over the physics step, and `pen_joint_vel_l2` the joint velocity, all summed over every joint (`../../IsaacLab/.../envs/mdp/rewards.py:137-178`, `../../IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation_data.py:777`). `pen_joint_pos_limits` sums the distance by which each joint lies outside its soft limits (`rewards.py:189-200`).

Abduction deviation. An L1 penalty on the abduction joints' distance from their default angles, the degree of freedom a policy widens first to enlarge its support polygon [3] (`base_env_cfg.py:1149-1156`). The default angles are themselves randomised by up to $\pm 0.05$ rad at startup (`base_env_cfg.py:1021-1029`), so the penalty is measured from each environment's perturbed default.

Action terms. Both act on the raw policy output $a_t$, before the scale of 0.25 and the default offset are applied. `pen_action_rate` is the first difference, with $a_{t-1}$ reset to zero at an episode start (`../../IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py:364-365`). `pen_action_smoothness` is the second difference, a discrete jerk, which penalises a change of rate rather than the rate itself (`rewards.py:2346-2417`). It needs two stored actions, which are zeroed at reset, and it returns zero for the first 3 steps of every episode so that the reset does not produce a spurious penalty.

```python
class ActionSmoothness:
    def __init__(self, n, m): self.a1 = self.a2 = None
    def reset(self, ids):
        if self.a1 is not None: self.a1[ids] = 0.0; self.a2[ids] = 0.0
    def __call__(self, a, episode_step):
        if self.a1 is None: self.a1 = a.clone(); return torch.zeros(len(a))
        if self.a2 is None: self.a2, self.a1 = self.a1, a.clone(); return torch.zeros(len(a))
        p = ((a - 2 * self.a1 + self.a2) ** 2).sum(1)
        self.a2, self.a1 = self.a1, a.clone()
        p[episode_step < 3] = 0.0
        return p
```

Undesired contacts. Counts the bodies other than the feet whose peak history force exceeds 10 N, over the trunk, abduction, hip and knee housings, thighs and shanks (`base_env_cfg.py:1133-1147`, `../../IsaacLab/.../envs/mdp/rewards.py:267-272`). Because a trunk, thigh or shank contact also ends the episode, the term's lasting effect falls on the actuator housings.

## 7. Per design scaling in co-optimisation

The policy is shared across designs whose legs differ in length, so three terms that encode a body size are rescaled per environment by its standing height $h_e$. Each reproduces the donor's term exactly when $h_e = h_0$ or when no height has been published.

| Term | Donor value | Scaled value |
|---|---|---|
| `pen_base_height` target | 0.35 m | $h_e$ |
| `feet_air_time` thresholds | $t_{min}, t_{max}$ | $k_e t_{min}, \; k_e t_{max}$, with $k_e = \sqrt{h_e / h_0}$ |
| `rew_foot_clearance` target | 0.10 m | $(0.10 / 0.35) \, h_e$ |

The square root scaling of the swing time follows the period of a pendulum of length $h_e$, which grows as $\sqrt{h_e / g}$ (`rewards.py:368-385`). It is the timing analogue of the dynamic similarity hypothesis, under which geometrically similar animals move alike at equal Froude number [4]. The clearance target scales linearly because it is a length. `pen_feet_hold` keeps its fixed 0.45 s ceiling. In code, $k_e$ is a column of shape (N, 1) broadcast against the feet axis (`rewards.py:421-425`).

## 8. Curricula that act on the reward

Curricula run when environments reset, before the reward sums are cleared, and read the episode sums $S_i$ of the environments resetting at that step (`../../IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:355-356`). Every term below acts only when the global step counter $n$ (policy steps since training began) is at least its start and is a multiple of its interval. The shared gate is

$$\frac{\operatorname{mean}_{e}(S_{i,e})}{T} > \theta \, w_i \, \Delta t,$$

which, since $S_i$ already carries $w_i \Delta t$, asks that the term's mean per step value over a full episode exceed the fraction $\theta$ of its maximum. A short episode is averaged over the full $T = 1000$ steps, so a falling population cannot pass it.

| Curriculum | Reads | Action when the gate passes | Interval, start | Rate, $\theta$, bound |
|---|---|---|---|---|
| Linear tracking width | `rew_lin_vel_xy` | $\sigma_v \leftarrow 0.925 \, \sigma_v$ while $\sigma_v > 0.09$ | 12500, 12500 | 0.925, 0.67, 0.09 |
| Angular tracking width | `rew_ang_vel_z` | $\sigma_\omega \leftarrow 0.95 \, \sigma_\omega$ while $\sigma_\omega > 0.09$ | 12500, 12500 | 0.95, 0.5, 0.09 |
| Forward command range | `rew_lin_vel_xy` | widen both ends by 0.03 m/s | 2500, 12500 | 0.03, 0.6, $\pm 1.5$ |
| Lateral command range | `rew_lin_vel_xy` | widen both ends by 0.01 m/s | 2500, 12500 | 0.01, 0.5, $\pm 0.5$ |
| Yaw command range | `rew_ang_vel_z` | widen both ends by 0.05 rad/s | 2500, 12500 | 0.05, 0.4, $\pm 1.35$ |

Sources are `curriculums.py:287-311` for the widths and `curriculums.py:313-459` for the ranges, with values at `base_env_cfg.py:1426-1503`.

```python
def gated(n, start, interval, S, w, dt, T, theta):
    return n >= start and n % interval == 0 and S.mean() / T > theta * w * dt

if gated(n, 12500, 12500, S["rew_lin_vel_xy"], 10, 0.02, 1000, 0.67) and sigma_v > 0.09:
    sigma_v *= 0.925
if gated(n, 12500, 2500, S["rew_lin_vel_xy"], 10, 0.02, 1000, 0.6):
    vx_range = (max(vx_range[0] - 0.03, -1.5), min(vx_range[1] + 0.03, 1.5))
```

Two further curricula change the training conditions rather than the reward. The push curriculum scales the push velocity range by 1.1 when the resetting environments contain fewer trunk contact failures than twice their time-outs, and by 0.8 when they contain more than half their time-outs, within $[0.2, 1.0]$ m/s, from step 12500 every 2500 steps (`curriculums.py:143-214`). The push itself sets a random planar base velocity every 10 to 15 s (`base_env_cfg.py:1077-1084`). On the rough task the terrain level of an environment rises when it walked more than 0.75 of a tile's size in its episode and falls when it covered less than 0.75 of the commanded distance, from step 4800 (`curriculums.py:461-506`), the game inspired promotion scheme of legged_gym [1]. The flat task removes the terrain curriculum.

The co-optimisation runner suspends every curriculum during its controlled design evaluation and during the resets that follow a design swap, through the flags `_suppress_curriculum` and `_suppress_terrain_curriculum`. A port must do the same, or a design swap will advance the schedule.

## 9. Porting checklist

1. Multiply each weighted term by the policy step $\Delta t = 0.02$ s, or divide every weight by 50, and keep the curriculum gate consistent with whichever is chosen.
2. Exclude time-outs from the termination penalty and treat them as truncations for bootstrapping.
3. Express base velocities in the base frame and the gravity vector in the base frame.
4. Store contact forces at every physics substep, keep the last 4, and take their maximum norm for every contact test. MuJoCo users can sum `mj_contactForce` per foot geom at each substep. Isaac Gym users can read `net_contact_force_tensor` at each substep.
5. Maintain per foot `current_air_time`, `last_air_time` and `current_contact_time` at physics rate with a 1 N threshold, set `last_air_time` at touchdown, and define first contact as a contact time in $(0, \Delta t]$.
6. Reproduce the 1.6 m by 1.0 m, 0.1 m, yaw aligned, vertically cast height grid, and mark missed rays as non finite so both lookups can exclude them. On flat ground all ray heights equal the floor height, which is a valid shortcut.
7. Read joint torques after actuator saturation, and compute joint acceleration by differencing velocity over the physics step.
8. Apply the action penalties to the raw policy output, reset the action history to zero at an episode start, and zero the smoothness penalty for the first 3 steps.
9. Publish a per environment standing height $h_e$ after every design change, and fall back to $h_0 = 0.35$ m where none exists.
10. Mutate the tracking widths and command ranges in place, so that the reward and the command sampler read the updated values on the next step.

## 10. Bibliography

[1] N. Rudin, D. Hoeller, P. Reist, M. Hutter. Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning. Conference on Robot Learning, PMLR 164, 91 to 100, 2022. arXiv:2109.11978.

[2] Gu et al. Humanoid-Gym. arXiv:2404.05695. The full author list, the full title, the venue and the year are not recorded in `/ws/context/literature.md` and are omitted here.

[3] X. Cheng, K. Shi, A. Agarwal, D. Pathak. Extreme Parkour with Legged Robots. IEEE International Conference on Robotics and Automation, 2024. arXiv:2309.14341.

[4] Alexander and Jayes. A Dynamic Similarity Hypothesis for the Gaits of Quadrupedal Mammals. Journal of Zoology 201(1), 135 onward, 1983. DOI 10.1111/j.1469-7998.1983.tb04266.x. Initials and the final page were not confirmed by the retrieved record and are omitted.

## Addition of 2026-10-23, where the tracking width curriculum leaves the widths, across 58 runs

`reduce_tracking_rewards_std` (`../environments/environments/tasks/locomotion/mdp/curriculums.py:287-311`) was read across every run of the workspace whose configuration carries it and whose event file is at least 20 MB, 65 runs of which 58 trained long enough and tracked well enough to be read ([artefacts/tracking_width_curriculum-cross-run/cross_run_widths.csv](artefacts/tracking_width_curriculum-cross-run/cross_run_widths.csv)). Every run starts both widths at 0.4 with a floor of 0.09 and thresholds of 0.67 and 0.5. The mean kernel below is the logged `Episode_Reward/<term>` divided by the weight, which is the quantity the gate compares with its threshold, averaged over all resets of an iteration rather than one step.

| Family | Runs | Linear narrowings below threshold | Linear runs at floor | Linear final width, median | Yaw narrowings below threshold | Yaw final width, median | Yaw final mean kernel, median |
|---|---|---|---|---|---|---|---|
| TRON1 co-optimisation | 2 | 0.59 | 2 | 0.086 | 1.00 | 0.155 | 0.38 |
| TRON1 | 4 | 0.39 | 0 | 0.337 | 1.00 | 0.390 | 0.36 |
| SD_BRS1 | 10 | 0.36 | 9 | 0.086 | 0.54 | 0.218 | 0.44 |
| KScale | 28 | 0.18 | 4 | 0.126 | 0.54 | 0.376 | 0.31 |
| Quadruped | 13 | 0.07 | 1 | 0.136 | 0.03 | 0.223 | 0.64 |

Three findings follow. The gate behaves as designed on the quadruped, whose mean kernels sit above both thresholds, only 7 and 3 per cent of its narrowings coming while the iteration mean stood below the threshold, and whose final widths land near the root mean square tracking error estimated from the logged mean error, a ratio of 0.58 for the linear and 0.72 for the yaw term. On every biped the gate fires below its own threshold, 18 to 59 per cent of linear narrowings and 54 to 100 per cent of yaw narrowings, because it reads only the few environments that reset on the step where the counter meets the interval, so the widths ratchet downward on favourable draws. The linear width reached its floor in 15 of the 58 runs, 9 of the 10 SD_BRS1 runs among them, while the yaw width reached it in none, and on the bipeds the yaw width ends at 0.33 to 0.43 of the estimated root mean square yaw error with a mean kernel of 0.31 to 0.44, below the threshold of 0.5 that was meant to stop it. The root mean square estimates assume Rayleigh and half normal errors and are lower bounds for a heavy tailed error, so the true ratios are lower still. The KScale yaw errors of 1 to 2 rad/s are large enough that its yaw figures should not be generalised. The literature on the same question is cluster 39 of `/ws/context/literature.md`.

The remedy is planned in `../plans/copt_improvement.md` section 6, an opt-in moving average gate on `reduce_tracking_rewards_std` that narrows above one threshold and widens below another, with starting widths, floors and thresholds of 0.4, 0.12, 0.67 and 0.55 for the biped linear term, 0.5, 0.30, 0.60 and 0.45 for the biped yaw term, 0.4, 0.10, 0.67 and 0.55 for the quadruped linear term and 0.4, 0.15, 0.60 and 0.45 for the quadruped yaw term, each floor near 0.6 of the family's estimated root mean square error, the ratio at which the quadruped's working gate settled.

Addition of 2026-10-23, later the same day. The gate of `../plans/copt_improvement.md` section 6 was applied. `reduce_tracking_rewards_std` carries the optional `smoothing_episodes`, `widen_threshold` and `maximum_std` (`environments/environments/tasks/locomotion/mdp/curriculums.py`), and only the two co-optimisation curricula set them, `CoptCurriculumCfg` of `cfg/SF/limx_base_env_cfg.py` and of `cfg/quadruped/base_env_cfg.py`, with the values of that section and a biped co-optimisation yaw start of 0.5 in `CoptRewardsCfg`. Every other task still runs the one step, narrow only gate, unchanged. A CPU check against the original code found the default path identical over 1500 steps, and on a synthetic error of 0.15 the smoothed path settled at a width of 0.216, a kernel of 0.618 inside the band, held its floor and cap exactly, widened to its cap when the error rose, and did not move while the curriculum was suppressed. It has not yet run in training.
