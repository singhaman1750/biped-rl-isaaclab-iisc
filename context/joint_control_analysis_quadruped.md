# Joint Control Analysis for the Quadruped, Choosing Gains When None Are Given

This document derives the proportional and derivative gains, the joint velocity ranges and the action scale for the quadruped defined by `environments/environments/assets/urdf/quadruped/quadruped.urdf`. It does not restate the theory of the proportional derivative joint controller, which [joint_control_analysis.md](joint_control_analysis.md) already establishes in full, namely the equivalence of the implicit actuator to a forced mass spring damper, the characteristic equation and its roots, the natural frequency, the damping ratio and the critical damping coefficient, the behaviour in each damping regime, and the causal chain from an excessive action scale through velocity saturation to training divergence. A reader unfamiliar with any of those should read that document first, and this one supplies only what it needs beyond it.

Two bodies of theory are contributed here that the existing document lacks, because the biped's gains arrived already identified from hardware and this robot's did not. Section 3 sets out how a proportional gain is chosen when no gain is given. Section 4 sets out how a joint velocity range is chosen. Both are stated generally before they are applied, so that the next robot to arrive without an identification need not rediscover them.

## 1. What the actuator model is, in the two sentences this document needs

Isaac Lab's implicit proportional derivative actuator computes a joint torque `tau = Kp (q_des - q) - Kd q_dot`, clipped to an effort limit, and PhysX applies it inside the physics step rather than at the policy rate. A joint of effective inertia `I` driven by that law is a mass spring damper of natural frequency `omega_n = sqrt(Kp / I)`, critical damping coefficient `Kd_crit = 2 sqrt(Kp I)` and damping ratio `zeta = Kd / (2 sqrt(Kp I))`, which is the whole of the machinery the sections below use.

The one property of the law worth naming separately, because section 3 turns on it, is that a purely proportional term cannot hold a load without an error. Under a constant load torque `tau_load` the joint settles where `Kp (q_des - q) = tau_load`, so the steady state error is `tau_load / Kp`. There is no integral term to remove it. This displacement is referred to below as the sag.

## 2. Effective inertias

The effective inertia at a joint is the inertia of everything distal to it, taken about that joint's axis with the distal joints held locked, obtained by the parallel axis theorem, plus the armature. The armature is the motor's rotor inertia reflected through the square of the gear ratio, and both MuJoCo and PhysX add it directly to the diagonal of the joint space mass matrix, so it participates in the joint's second order dynamics exactly as the link inertia does [1]. The model declares 0.01 kilogramme metres squared on every joint, a figure the conversion document records as high by roughly a factor of six for this motor, and it is nonetheless the single most consequential number in this section.

Two configurations matter and the distinction must not be elided. The nominal crouch is the operating point, the pose the robot stands in and the pose about which the policy's actions are offsets, and it is the configuration at which the gains are set. Full extension is the worst case, the pose in which each distal chain reaches its greatest lever arm, and it is the configuration against which the resulting damping ratio is checked for the excursion it suffers.

The figures below were obtained from the diagonal of the joint space mass matrix of the compiled MuJoCo model, which is by construction the effective inertia at each joint with the others locked and with the armature already included, rather than by a hand computation of parallel axis terms.

| Joint | Distal chain | At the nominal crouch | At full extension |
|---|---|---|---|
| `abad_*_Joint` | hip housing, knee housing, thigh, calf, foot, about the roll axis | 0.023674 | 0.033342 |
| `hip_*_Joint` | knee housing, thigh, calf, foot, about the pitch axis | 0.020572 | 0.027728 |
| `knee_*_Joint` | calf, foot, about the pitch axis | 0.012305 | 0.012305 |

Three observations follow. The abduction joint carries the largest effective inertia, not the smallest, because the whole leg swings about a roll axis at a lever arm of 0.080 metres laterally and up to 0.426 metres vertically, and because the two actuator housings that dominate the leg's mass both sit on that lever. This inverts the intuition carried over from the biped, where the abduction joint is the lightest, and it is the reason the abduction gains cannot simply be copied down from the hip. The knee's inertia does not vary with configuration, its distal chain containing no joint. And the armature is 42 percent of the abduction figure, 49 percent of the hip figure and 81 percent of the knee figure, so the gains derived below are, to a large extent, gains for a rotor rather than for a limb.

A correction is recorded here against the plan that specified this work. That plan tabulated the abduction inertia at the crouch as 0.024292, obtained by hand. The compiled model gives 0.023674, an overestimate of 2.5 percent in the plan's hand computed parallel axis term, the extended figure and both pitch figures agreeing exactly. The consequence is confined to two derived quantities, the abduction natural frequency rising from 6.46 to 6.54 hertz and its damping ratio from 0.609 to 0.617, and neither gain changes, both having been rounded to two significant figures.

## 3. Choosing a proportional gain when none is given

### 3.1 Why the response is the wrong thing to size against

The instinct on meeting an unidentified joint is to choose the stiffness that gives some desired response, a chosen natural frequency or a chosen settling time. That instinct is wrong, and seeing why explains an otherwise puzzling feature of every published locomotion configuration.

Sizing for a uniform natural frequency requires `Kp` proportional to `I`, so a joint of a third the inertia takes a third the stiffness. On this robot that would give the knee, whose effective inertia is half the abduction joint's, half its gain. But the knee is the joint that carries the largest static load, so it would then sag twice as far under the robot's own weight as the joint carrying almost none. The robot would stand on the softest of its joints.

This is why the reference configurations carry nearly uniform stiffness across joints of very different inertia. Isaac Lab's own A1 and Go2 configurations use a single value across all twelve joints [2], the Mini Cheetah work uses 17 throughout [3], and the rapid motor adaptation work uses 55 on the A1 [4]. Uniformity is not laziness. It is what sizing against load rather than against response produces on a machine whose joints carry comparable loads.

### 3.2 The floor, from the load the joint must hold

The criterion is therefore the sag. A tolerated sag `s` and a stance torque `tau_stance` together impose a floor `Kp >= tau_stance / s`, and the gain must clear the floor of every joint.

The tolerance adopted is 0.125 radians, half the action scale of 0.25. Beyond that the policy spends more than half its per step authority merely cancelling the sag rather than shaping the gait, which is the practical failure the floor exists to prevent.

The stance torques were measured rather than estimated, by settling the robot on rigid ground under the gains derived below and reading the steady state joint displacements. The distinction matters, and the plan that specified this work got it wrong at two of the three joints.

| Joint | Predicted by the plan | Measured in simulation | Implied floor at a sag of 0.125 rad |
|---|---|---|---|
| `abad_*_Joint` | 4.046 N m | 0.129 N m | 1.0 |
| `hip_*_Joint` | 0.226 N m | 2.256 N m | 18.0 |
| `knee_*_Joint` | 6.506 N m | 5.896 N m | 47.2 |

The plan obtained the abduction figure by multiplying a quarter of the robot's weight by the 0.080 metre lateral offset from the abduction axis to the foot. That computation counts the roll moment the ground reaction exerts about the axis and omits the opposing moment of the leg's own weight, which acts through a lever on the same side of the axis and very nearly cancels it, the legs carrying 95.7 percent of this machine. The measured abduction load is accordingly some thirty times smaller than predicted, and the hip's, which the plan judged near zero on the ground that the keyframe places each foot directly beneath its pitch axis, is ten times larger, the load not vanishing once the pose sags away from that keyframe.

The general lesson is that a stance torque should be measured by settling the robot, not computed from a lever arm, on any machine whose limbs are a material fraction of its mass.

The binding joint is therefore the knee, at a floor of 47.2, and not the abduction joint as the plan supposed.

### 3.3 The ceilings, from integration and from observability

Two ceilings bound the gain from above and only one of them binds.

The integration ceiling does not. PhysX evaluates the proportional derivative law at 200 hertz and integrates it implicitly, so a natural frequency of even 70 radians per second gives `omega_n dt` of 0.35 and presents no stiff integration problem. Explicit integrators impose a real constraint here and this one does not.

The observability ceiling binds. The policy acts at 50 hertz, and a response whose natural frequency exceeds a fifth of that rate rings and settles within a single control step, so the policy cannot observe the transient it caused and therefore cannot learn to shape it. The ceiling is `f_n <= f_ctrl / 5`, which is 10 hertz.

### 3.4 The gains adopted

The gains follow the torque ceilings of the three motors, on the reasoning that a joint sized for more torque was sized for more load and should be correspondingly stiffer. The abduction and hip joints share the 23.622511 newton metre motor and take 40. The knee's 35.238000 newton metre motor stands in a ratio of 1.4917 to the others, which would give 60, and 60 places the knee at 11.11 hertz, above the observability ceiling. The knee is therefore trimmed to 50, which lands at 10.15 hertz, exceeding the ceiling by 1.5 percent.

The derivative gain then follows from the damping ratio, `Kd = 2 zeta sqrt(Kp I)`, evaluated at the crouch and rounded to two significant figures.

| Joint | Kp | Effective inertia | Natural frequency | Fraction of the control rate | Critical damping | Kd at zeta 0.6 | Damping ratio extended | Measured sag |
|---|---|---|---|---|---|---|---|---|
| `abad_*_Joint` | 40.0 | 0.023674 | 41.10 rad/s, 6.54 Hz | 13.1 percent | 1.947 | 1.20 | 0.520 | 0.003 rad |
| `hip_*_Joint` | 40.0 | 0.020572 | 44.10 rad/s, 7.02 Hz | 14.0 percent | 1.814 | 1.10 | 0.522 | 0.056 rad |
| `knee_*_Joint` | 50.0 | 0.012305 | 63.75 rad/s, 10.15 Hz | 20.3 percent | 1.569 | 0.90 | 0.574 | 0.118 rad |

Every measured sag clears the 0.125 radian tolerance, the knee by 6 percent and the other two by an order of magnitude, so the chosen gains satisfy the criterion of section 3.2 with more margin than the plan that specified them claimed.

The damping target of 0.6 sits just below the 0.7 to 1.0 band NVIDIA recommends for a joint drive [5], deliberately, since a legged joint that must absorb an impact benefits from a little compliance and since the surveyed locomotion configurations run nearer 0.4. It gives a peak overshoot of 9.5 percent on a step and a two percent settling time of 0.105 seconds at the knee, which is 5.2 control steps, so the policy sees the whole of the transient it commands. The damping ratio falls no lower than 0.520 anywhere in the configuration space, the excursion arising because `Kd` is fixed while the effective inertia grows toward extension, and 0.520 remains comfortably underdamped without ringing.

### 3.5 What the derivation rejected, and three comparisons

The source model's own position actuators declare a proportional gain of 100 on all twelve joints. That figure is rejected. A MuJoCo `position` actuator gain is a modelling convenience with no hardware behind it, 100 is a round number rather than an identified one, and it is four times what Isaac Lab's own A1 and Go2 configurations use at `/ws/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:92` and `:175`, six times the Mini Cheetah's 17 [3] and nearly twice the rapid motor adaptation work's 55 on the A1 [4].

The source model's joint damping of 1.0, 2.0 and 2.0 would give damping ratios of 0.507, 1.102 and 1.275 against the derived proportional gains, so it is not absurd but it is inconsistent, overdamping the two pitch joints while leaving the roll joint at half their damping.

Isaac Lab's Go2 configuration uses 25 and 0.5 uniformly for the same motor family and very nearly the same link lengths, which would give this robot damping ratios of 0.32, 0.35 and 0.45 and a knee sag near 0.24 radians, so the scheme derived here is stiffer and better damped than the canonical one rather than a departure from it in kind.

The biped's leg joints in this repository run at 55, 80 and 60 against effective inertias near 0.10, placing them between 3.7 and 4.5 hertz, so the quadruped sits one octave higher, which is what a robot with a third of the limb inertia should do.

### 3.6 One consequence to carry forward

Settled on rigid ground under these gains, the robot stands at 0.2639 metres against the commanded 0.292, a droop of 28 millimetres, almost all of it the knee's 0.118 radian sag. This is inherent to a proportional law without an integral term and is not a defect of the gains.

Its practical consequence is small and worth stating so that it is not mistaken for one later. The base height reward targets 0.292 and would score a robot resting passively at its natural equilibrium at a penalty of roughly 0.004 per step against a weight of -5.0, which is negligible beside the tracking terms, and a trained policy will in any case command a slightly deeper crouch to reach the target. Should a future configuration wish the commanded pose and the standing pose to coincide exactly, the remedy is to lower the base height target to the settled figure rather than to raise the stiffness, since raising it far enough to close a 28 millimetre gap would carry the knee past the observability ceiling.

## 4. Choosing a joint velocity range

### 4.1 What a velocity limit is in this framework, and what it is not

The `velocity_limit` of a `DCMotorCfg`, from which `IdentifiedActuatorCfg` derives at `environments/environments/actuators/actuator_cfg.py:15`, is not a clamp. It is the no load speed of a linear torque speed characteristic. The available torque is `saturation_effort` times `(1 - q_dot / velocity_limit)`, then clipped to `effort_limit`, so torque falls linearly with speed and reaches zero at the declared limit.

Two consequences follow and both are design decisions rather than details. The quantity that actually bounds useful motion is the corner speed at which the falling ramp meets the effort ceiling, which is `velocity_limit (1 - effort_limit / saturation_effort)`, and below which the joint delivers constant torque. And the ratio of saturation to effort therefore selects how much of the speed range is constant torque. This repository's TRON1 configuration sets that ratio near 6.7, giving constant torque up to 85 percent of the no load speed, whereas Isaac Lab's Unitree configurations set saturation equal to effort, giving a pure ramp with no constant region at all.

The TRON1 ratio is adopted, because the source model declares a constant `forcerange` with no speed dependence whatever, so a characteristic that is flat over the operating range is the one that keeps the two models interchangeable. The Unitree alternative is recorded as the more conservative option should hardware measurements later contradict it.

### 4.2 The four bounds that determine the numbers

The motor bound. The design's torque ceilings of 23.622511 and 35.238000 newton metres match the Unitree Go1 GO-M8010-6 joint ratings of 23.7 and 35.55 newton metres to within 0.4 and 0.9 percent respectively, and its thigh and shank of 0.213 metres match the Go2's exactly, so the design is almost certainly derived from that actuator and that leg, and the corresponding rated joint speeds of 30.1 and 20.06 radians per second are the defensible no load figures [6]. Isaac Lab's own Go2 configuration independently uses 30.0 for the same motor family at `unitree.py:174`.

The slew bound. The policy may change its setpoint by the action scale on every control step, so the actuator must sustain `alpha / dt_ctrl`, which at an action scale of 0.25 and a control period of 0.02 seconds is 12.5 radians per second.

The transient bound. A step of amplitude `A` into a second order system produces a velocity `A omega_n exp(-zeta omega_n t) sin(omega_d t) / sqrt(1 - zeta^2)`, whose first peak is found by maximising over `t`. Evaluated numerically at `A` equal to 0.25 and at the gains of section 3.4 this gives 5.02 radians per second at the abduction joint, 5.47 at the hip and 8.14 at the knee. Note that the closed form upper bound `A omega_n / sqrt(1 - zeta^2)` used at section 6.5 of [joint_control_analysis.md](joint_control_analysis.md) overestimates these by a factor near 2.5, since it ignores the exponential decay before the sine reaches its first peak, and that document should record the correction.

The kinematic bound. A trot at two hertz with a duty factor of one half gives a swing duration of 0.25 seconds. A step length of 0.20 metres on a 0.270 metre standing leg requires a hip sweep of `2 arcsin(0.10 / 0.270)`, which is 0.759 radians, a mean rate of 3.04 radians per second and a half sine peak of 4.77. Knee retraction of about a radian over the same swing gives a peak near 6.3. Both are below the other bounds, which is the expected result, gait rates lying well inside actuator capability except during recovery.

### 4.3 The resulting parameterisation

The motor bound is taken as the no load speed and the corner speed is verified to exceed every demand.

| Joint | No load speed | Saturation effort | Effort limit | Corner speed | Largest demand | Margin |
|---|---|---|---|---|---|---|
| `abad_*_Joint` | 30.0 | 158.0 | 23.622511 | 25.51 | 12.5, the slew bound | 2.04 times |
| `hip_*_Joint` | 30.0 | 158.0 | 23.622511 | 25.51 | 12.5, the slew bound | 2.04 times |
| `knee_*_Joint` | 20.0 | 236.0 | 35.238000 | 17.01 | 12.5, the slew bound | 1.36 times |

## 5. The action scale

Three bounds constrain the action scale and the biped's inherited value of 0.25 clears all three.

Velocity saturation through the transient. The peak velocity scales linearly in the action scale, so the scale at which each joint's peak reaches its corner speed is 1.27 at the abduction joint, 1.16 at the hip and 0.52 at the knee.

Velocity saturation through the slew. Requiring `alpha / dt_ctrl` to stay inside the corner speed gives 0.51 at the abduction and hip joints and 0.34 at the knee.

The torque ceiling. A step of amplitude `alpha` demands `Kp alpha` at the instant it is applied, which is 10.00 newton metres at the abduction and hip joints against a ceiling of 23.622511, and 12.50 at the knee against 35.238000. The scale at which a single full step reaches the ceiling is 0.59 and 0.70 respectively.

The tightest of the nine figures is the knee's slew bound of 0.34, so the action scale of 0.25 is safe by a factor of 1.36, and the torque ceiling, which would have bound the design at 0.236 under the rejected gain of 100, no longer binds at all. This is the clearest single benefit of deriving the stiffness rather than inheriting it.

## 6. Where these figures live in the configuration

Every gain, effort limit, velocity limit and saturation effort derived above is set in `environments/environments/assets/config/quadruped_identified_cfg.py`, one `IdentifiedActuatorCfg` per joint group, selected by the expressions `abad_.._Joint`, `hip_.._Joint` and `knee_.._Joint`. The two wildcard form is deliberate and must not be relaxed to `.*`, which would additionally match the fixed joint `hip_FR_thigh_joint` and attach an actuator to a body with no degree of freedom. The action scale of 0.25 is set on the joint position action of `environments/environments/tasks/locomotion/cfg/quadruped/base_env_cfg.py`, and the 50 hertz control rate and 200 hertz physics rate that section 3.3 assumes come from the decimation of 4 and the simulation step of 0.005 seconds declared in the same file.

## 7. Bibliography

1. Guan, N., Yu, S., Zhu, S., Kim, D. Impedance Matching, Enabling an RL-Based Running Jump in a Quadruped Robot. Ubiquitous Robots, 2024. arXiv:2404.15096.
2. Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., and others. Orbit, A Unified Simulation Framework for Interactive Robot Learning Environments. IEEE Robotics and Automation Letters 8(6), 3740 to 3747, 2023. arXiv:2301.04195. The author list was not verified beyond the six named and the short form is used. The A1 and Go2 gain values were read from the local checkout at `/ws/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py`.
3. Ji, G., Mun, J., Kim, H., Hwangbo, J. Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion. IEEE Robotics and Automation Letters 7(2), 2022. arXiv:2202.05481.
4. Kumar, A., Fu, Z., Pathak, D., Malik, J. RMA, Rapid Motor Adaptation for Legged Robots. Robotics, Science and Systems, 2021. arXiv:2107.04034.
5. NVIDIA. Tutorial 6, Joint Gains Tuning. Isaac Sim OpenUSD Tuning Tutorials documentation, `docs.isaacsim.omniverse.nvidia.com`, retrieved 2026-08-19.
6. Unitree Robotics. GO Motor product specification, `unitree.com`, and Go1 Datasheet EN v3.0, together with the `unitreerobotics/unitree_ros` repository file `go2_description/urdf/go2_description.urdf`, retrieved 2026-08-19. Source of the GO-M8010-6 ratings of 23.7 newton metres at 30.1 radians per second for the hip and thigh and 35.55 newton metres at 20.06 radians per second for the calf.
