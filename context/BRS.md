# SD_BRS1 Actuator Gain Analysis: Stiffness and Damping Verification

## 1. Introduction

This document presents a rigorous, physics-grounded analysis of the stiffness and damping gains configured for the SD_BRS1 bipedal robot, as defined in `environments/environments/assets/config/sd_brs1_identified_cfg.py`. The analysis draws its physical data exclusively from the robot URDF at `assets/urdf/solefoot/SD_BRS1/SD_BRS1_Assembly2.urdf` and compares the existing gains against values derived from first principles, covering second-order system dynamics, effective inertia calculation via the parallel-axis theorem, gravitational torque estimation, and cross-referencing against the co-resident TRON1 robot configuration.

The central thesis is that the proximal joints of the BRS robot (hip pitch, hip roll, and knee pitch) are configured with stiffness values that are significantly below what the robot's mass and geometry demands, while the distal ankle joints sit in a reasonable range. The document works through every step of reasoning, beginning from the most elementary mechanical concepts, before arriving at recommended gain ranges.

---

## 2. Robot Physical Properties Extracted from URDF

### 2.1 Link Mass Inventory

The URDF lists the following rigid bodies and their masses as exported from SolidWorks (commit version 1.6.0-1-g15f4949):

| Link Name | Description | Mass (kg) |
|---|---|---|
| Part_Torso | Upper body and pelvis | 8.2464 |
| Link1R | Right hip yaw bracket | 2.7260 |
| Link1L | Left hip yaw bracket | 2.7260 |
| Link2R | Right hip roll link | 3.8609 |
| Link2L | Left hip roll link | 3.8600 |
| Link3R | Right thigh | 9.6450 |
| Link3L | Left thigh | 9.6512 |
| Link4R | Right shank | 5.8050 |
| Link4L | Left shank | 5.8054 |
| Link5R | Right ankle roll adapter | 0.0722 |
| Link5L | Left ankle roll adapter | 0.0722 |
| Link6R | Right foot sole | 3.6910 |
| Link6L | Left foot sole | 3.6910 |

Summing all links gives a total robot mass of 59.85 kg.

### 2.2 Leg Geometry

Joint origin translations in the URDF define the segment lengths:

- The KneePitch joint is offset from the HipPitch joint by (0, 0.00045, -0.44) metres, establishing a thigh length of 0.44 m along the vertical axis.
- The AnkleRoll joint is offset from the KneePitch joint by (0, 0, -0.43) metres, establishing a shank length of 0.43 m.
- The total hip-to-ankle distance with legs fully extended is therefore 0.87 m.
- The foot sole centre of mass sits approximately 0.069 m below the ankle joint (from Link6 CoM z-offset of -0.068 m).

The robot's init_state specifies a base height of 1.1 m with all joint positions at zero, which the init_state comment confirms places both foot soles at world Z = 0.0 (ground level), verified against the USD joint chain. The effective torso standing height is thus 1.1 m.

### 2.3 Joint Topology and Degrees of Freedom

The full joint chain per leg is as follows:

Part_Torso → (fixed: HipYaw) → Link1 → (revolute: HipRoll, axis X) → Link2 → (revolute: HipPitch, axis -Y/+Y) → Link3 → (revolute: KneePitch, axis Y) → Link4 → (revolute: AnkleRoll, axis -X) → Link5 → (revolute: AnklePitch, axis -Y) → Link6

A critical observation is that HipYawR and HipYawL are declared as `type="fixed"` in the URDF. They contribute no active degree of freedom. Despite this, the configuration file defines `BRS1_HIP_YAW_ACTUATOR_CFG`, however this config is absent from the `actuators` dictionary passed to `SD_BRS1_IDENTIFIED_CFG` and is therefore never applied. The robot has 10 active revolute joints in total: HipRoll, HipPitch, KneePitch, AnkleRoll, and AnklePitch, on each of the left and right legs.

Joint limits from the URDF are listed below:

| Joint | Lower (rad) | Upper (rad) | Lower (deg) | Upper (deg) |
|---|---|---|---|---|
| HipRoll | -0.349 | 0.349 | -20.0 | 20.0 |
| HipPitch | -0.576 | 0.314 | -33.0 | 18.0 |
| KneePitch | 0.000 | 1.483 | 0.0 | 85.0 |
| AnkleRoll | -0.454 | 0.454 | -26.0 | 26.0 |
| AnklePitch | -0.349 | 0.349 | -20.0 | 20.0 |

---

## 3. Theoretical Background: PD Control and Second-Order System Dynamics

### 3.1 The PD Actuator Model in IsaacLab

In this codebase, joints are governed by a proportional-derivative (PD) position controller. For each joint, the torque command is computed as:

```
τ = K × (q_des - q) - D × q_dot
```

where `q_des` is the desired joint position output by the RL policy (scaled by `scale=0.25` from the actions config), `q` is the current position, `q_dot` is the current velocity, and the result is subsequently clamped to the `effort_limit` of the `IdentifiedActuatorCfg`.

### 3.2 Second-Order System Dynamics

A single joint with effective rotational inertia `I_eff`, spring stiffness `K`, and damper `D` obeys the second-order linear ODE:

```
I_eff × q_ddot + D × q_dot + K × q = K × q_des
```

This is a classical damped harmonic oscillator. Its two characteristic quantities are:

The undamped natural frequency in radians per second:

```
ωn = sqrt(K / I_eff)
```

The dimensionless damping ratio:

```
ζ = D / (2 × sqrt(K × I_eff))
```

A system with ζ < 1 is under-damped and will oscillate when disturbed. At ζ = 1 the system is critically damped and returns to equilibrium without oscillation in the minimum possible time. At ζ > 1 the system is over-damped and returns sluggishly. For robotic locomotion, a damping ratio between 0.6 and 1.0 is generally desired, ensuring fast yet non-oscillatory tracking. The critical damping value of D for a given K and I_eff is:

```
D_crit = 2 × sqrt(K × I_eff)
```

### 3.3 Bandwidth Considerations at 50 Hz Control Rate

The environment is configured with `sim.dt = 0.005 s` and `decimation = 4`, giving a policy control period of `T_ctrl = 4 × 0.005 = 0.02 s`, equivalent to a 50 Hz control rate. The Nyquist frequency for this discrete system is `π / T_ctrl = 157 rad/s`. For stable, well-behaved tracking, the natural frequency should be well below Nyquist, typically in the range of ωn ≈ 5 to 30 rad/s for the joints of a walking biped at this control rate. Proximal joints (hip, knee) benefit from lower bandwidth (5 to 15 rad/s) to permit smooth, RL-shaped trajectories, while distal joints (ankle) benefit from higher bandwidth (15 to 30 rad/s) to enable fast terrain adaptation.

---

## 4. Effective Inertia Calculation by the Parallel-Axis Theorem

The effective rotational inertia `I_eff` at each joint about its rotation axis is computed by summing contributions from all links distal to that joint. For each distal link `k`:

```
I_k_contribution = I_k_CoM_along_axis + m_k × r_perp_k²
```

where `I_k_CoM_along_axis` is the principal inertia component of the link about its own centre of mass along the joint's rotation axis, and `r_perp_k` is the perpendicular distance from the joint axis to the link's centre of mass (computed in the 2D plane orthogonal to the joint axis). The total effective inertia is the sum of all such contributions.

All coordinates are in metres, masses in kilograms, inertias in kg·m², and torques in Newton-metres.

### 4.1 Hip Pitch (axis: Y, shared spatial origin with Hip Roll)

Distal links: Link3 (thigh), Link4 (shank), Link5 (ankle roll adapter), Link6 (foot).

Link3 CoM in HipPitch frame: (-0.01444, -0.00023, -0.24157) m. Perpendicular to Y-axis involves the X and Z components.
- r3 = sqrt(0.01444² + 0.24157²) = sqrt(0.000209 + 0.058356) = 0.2420 m
- I3_yy at CoM = 0.177697 kg·m²
- Contribution = 0.177697 + 9.645 × 0.2420² = 0.177697 + 0.564795 = 0.7425 kg·m²

Link4 CoM in HipPitch frame: knee offset (0, 0.00045, -0.44) + Link4 CoM offset (0.00346, -0.00001, -0.19188) = (0.00346, 0.00044, -0.63188) m.
- r4 = sqrt(0.00346² + 0.63188²) = sqrt(0.0000120 + 0.39927) = 0.63189 m
- I4_yy at CoM = 0.10544 kg·m²
- Contribution = 0.10544 + 5.805 × 0.63189² = 0.10544 + 2.31739 = 2.4228 kg·m²

Link5 CoM in HipPitch frame: ankle = (0, 0.00045, -0.87) + Link5 CoM offset (-0.01599, 0, 0) = (-0.01599, 0.00045, -0.87) m.
- r5 = sqrt(0.01599² + 0.87²) = sqrt(0.000256 + 0.7569) = 0.87015 m
- I5_yy at CoM = 7.715e-5 kg·m²
- Contribution = 7.715e-5 + 0.0722 × 0.87015² = 0.0000771 + 0.054668 = 0.054745 kg·m²

Link6 CoM in HipPitch frame: ankle (0, 0.00045, -0.87) + Link6 CoM (0.01069, -0.00375, -0.06865) = (0.01069, -0.00330, -0.93865) m.
- r6 = sqrt(0.01069² + 0.93865²) = sqrt(0.000114 + 0.881063) = 0.93870 m
- I6_yy at CoM = 0.021200 kg·m²
- Contribution = 0.021200 + 3.691 × 0.93870² = 0.021200 + 3.25282 = 3.27402 kg·m²

Total I_eff_hip_pitch = 0.7425 + 2.4228 + 0.054745 + 3.27402 = 6.494 kg·m²

### 4.2 Hip Roll (axis: X, same spatial origin as Hip Pitch)

Distal links: Link2, Link3, Link4, Link5, Link6.

Since the HipRollR and HipPitchR joints share the same spatial origin (both have xyz="0 0 0" in their respective parent frames and the HipYaw joint is fixed with no rotation), Links 3 through 6 have the same CoM positions relative to HipRoll as they do to HipPitch. For the roll axis (X), the perpendicular distance involves Y and Z components.

Link2 CoM in HipRoll frame: (-0.00055, 0.01004, 0.00009) m.
- r2 = sqrt(0.01004² + 0.00009²) = 0.01004 m
- I2_xx at CoM = 0.0097704 kg·m²
- Contribution = 0.0097704 + 3.8609 × 0.01004² = 0.0097704 + 0.000389 = 0.010159 kg·m²

Link3 CoM: (-0.01444, -0.00023, -0.24157) m. For X-axis perpendicular: Y and Z components.
- r3 = sqrt(0.00023² + 0.24157²) = 0.24157 m
- I3_xx at CoM = 0.187910 kg·m²
- Contribution = 0.187910 + 9.645 × 0.24157² = 0.187910 + 0.562857 = 0.750767 kg·m²

Link4 CoM: (0.00346, 0.00044, -0.63188) m.
- r4 = sqrt(0.00044² + 0.63188²) = 0.63188 m
- I4_xx at CoM = 0.098359 kg·m²
- Contribution = 0.098359 + 5.805 × 0.63188² = 0.098359 + 2.31732 = 2.41568 kg·m²

Link5 CoM: (-0.01599, 0.00045, -0.87) m.
- r5 = sqrt(0.00045² + 0.87²) = 0.870 m
- I5_xx at CoM = 3.219e-6 kg·m²
- Contribution = 3.219e-6 + 0.0722 × 0.870² = 3.219e-6 + 0.054649 = 0.054652 kg·m²

Link6 CoM: (0.01069, -0.00330, -0.93865) m.
- r6 = sqrt(0.00330² + 0.93865²) = 0.93866 m
- I6_xx at CoM = 0.015197 kg·m²
- Contribution = 0.015197 + 3.691 × 0.93866² = 0.015197 + 3.25249 = 3.26769 kg·m²

Total I_eff_hip_roll = 0.010159 + 0.750767 + 2.41568 + 0.054652 + 3.26769 = 6.499 kg·m²

### 4.3 Knee Pitch (axis: Y)

Distal links: Link4, Link5, Link6. The origin of KneePitch is at (0, 0.00045, -0.44) relative to HipPitch.

Link4 CoM relative to KneePitch: (0.00346, -0.00001, -0.19188) m.
- r4 = sqrt(0.00346² + 0.19188²) = sqrt(0.0000120 + 0.036818) = 0.19191 m
- I4_yy at CoM = 0.10544 kg·m²
- Contribution = 0.10544 + 5.805 × 0.19191² = 0.10544 + 0.21390 = 0.31934 kg·m²

Link5 CoM relative to KneePitch: ankle at (0, 0, -0.43) + Link5 CoM (-0.01599, 0, 0) = (-0.01599, 0.00045, -0.43) m.
- r5 = sqrt(0.01599² + 0.43²) = sqrt(0.000256 + 0.18490) = 0.43028 m
- I5_yy at CoM = 7.715e-5 kg·m²
- Contribution = 7.715e-5 + 0.0722 × 0.43028² = 7.715e-5 + 0.013367 = 0.013444 kg·m²

Link6 CoM relative to KneePitch: (0.01069, -0.00330, -0.43 - 0.06865) = (0.01069, -0.00330, -0.49865) m.
- r6 = sqrt(0.01069² + 0.49865²) = sqrt(0.000114 + 0.24865) = 0.49876 m
- I6_yy at CoM = 0.021200 kg·m²
- Contribution = 0.021200 + 3.691 × 0.49876² = 0.021200 + 0.91837 = 0.93957 kg·m²

Total I_eff_knee_pitch = 0.31934 + 0.013444 + 0.93957 = 1.272 kg·m²

### 4.4 Ankle Pitch (axis: -Y)

Distal link: Link6 only. AnklePitch shares the same origin as AnkleRoll.

Link6 CoM relative to AnklePitch: (0.01069, -0.00375, -0.06865) m.
- r6 = sqrt(0.01069² + 0.06865²) = sqrt(0.000114 + 0.004713) = 0.069478 m
- I6_yy at CoM = 0.021200 kg·m²
- Contribution = 0.021200 + 3.691 × 0.069478² = 0.021200 + 0.017817 = 0.039017 kg·m²

Total I_eff_ankle_pitch = 0.039017 kg·m²

### 4.5 Ankle Roll (axis: -X)

Distal links: Link5, Link6. AnkleRoll is at (0, 0, -0.43) relative to KneePitch.

Link5 CoM relative to AnkleRoll: (-0.01599, 0, 0) m. Perpendicular to X-axis requires Y and Z.
- r5 = sqrt(0² + 0²) = 0.0 m (CoM lies on the rotation axis)
- I5_xx at CoM = 3.219e-6 kg·m²
- Contribution = 3.219e-6 kg·m² (negligible)

Link6 CoM relative to AnkleRoll: (0.01069, -0.00375, -0.06865) m. Perpendicular to X-axis.
- r6 = sqrt(0.00375² + 0.06865²) = sqrt(0.0000141 + 0.004713) = 0.068753 m
- I6_xx at CoM = 0.015197 kg·m²
- Contribution = 0.015197 + 3.691 × 0.068753² = 0.015197 + 0.017449 = 0.032646 kg·m²

Total I_eff_ankle_roll = 0.032649 kg·m²

### 4.6 Summary of Effective Inertias

| Joint | I_eff (kg·m²) | Dominant contribution |
|---|---|---|
| Hip Pitch | 6.494 | Link6 foot (3.274) and Link4 shank (2.423) |
| Hip Roll | 6.499 | Link6 foot (3.268) and Link4 shank (2.416) |
| Knee Pitch | 1.272 | Link6 foot (0.940) and Link4 shank (0.319) |
| Ankle Pitch | 0.039 | Link6 foot solely |
| Ankle Roll | 0.033 | Link6 foot solely |

The hip joints carry by far the largest effective inertia because they must swing the entire leg distal to them, and the heavy foot (3.691 kg each) at a moment arm of nearly 0.94 m contributes over 3.25 kg·m² per leg to the hip's rotational load.

---

## 5. Current Actuator Configuration

The actuator gains as configured in `sd_brs1_identified_cfg.py` are:

| Joint | K (Nm/rad) | D (Nm·s/rad) | Effort Limit (Nm) | Velocity Limit (rad/s) |
|---|---|---|---|---|
| HipYaw (unused) | 60 | 8 | 351 | 20 |
| HipRoll | 80 | 14 | 351 | 20 |
| HipPitch | 60 | 14 | 298 | 20 |
| KneePitch | 80 | 4 | 420 | 20 |
| AnkleRoll | 15 | 1 | 131 | 25 |
| AnklePitch | 15 | 1 | 262 | 25 |

---

## 6. Analysis of Current Gains: Natural Frequency and Damping Ratio

Using ωn = sqrt(K / I_eff) and ζ = D / (2 × sqrt(K × I_eff)):

### 6.1 Hip Pitch (K=60, D=14, I=6.494 kg·m²)

```
ωn = sqrt(60 / 6.494) = sqrt(9.239) = 3.040 rad/s    (0.484 Hz)
ζ  = 14 / (2 × sqrt(60 × 6.494)) = 14 / 39.479 = 0.355
```

The natural frequency of 3.04 rad/s corresponds to less than 0.5 Hz. This is slower than typical human walking cadence (approximately 1 Hz) and implies that after a perturbation, the hip pitch joint takes over two full seconds to complete its first oscillation. The damping ratio of 0.355 places the system firmly in the under-damped regime, meaning that any displacement will produce significant ringing before settling.

The maximum torque achievable through stiffness at the full joint range limit (0.576 rad from zero) is K × θ_max = 60 × 0.576 = 34.6 Nm. The effort limit is 298 Nm, meaning stiffness is exploiting only 11.6% of the available actuator torque at the joint limit.

### 6.2 Hip Roll (K=80, D=14, I=6.499 kg·m²)

```
ωn = sqrt(80 / 6.499) = sqrt(12.310) = 3.509 rad/s    (0.558 Hz)
ζ  = 14 / (2 × sqrt(80 × 6.499)) = 14 / 45.604 = 0.307
```

The hip roll is slightly stiffer than hip pitch (owing to the higher K) but is even more under-damped at ζ = 0.307. The maximum stiffness torque at the joint limit (0.349 rad) is 80 × 0.349 = 27.9 Nm, exploiting only 7.9% of the 351 Nm effort limit through the proportional term.

### 6.3 Knee Pitch (K=80, D=4, I=1.272 kg·m²)

```
ωn = sqrt(80 / 1.272) = sqrt(62.893) = 7.931 rad/s    (1.262 Hz)
ζ  = 4 / (2 × sqrt(80 × 1.272)) = 4 / 20.176 = 0.198
```

The knee has a modestly higher natural frequency due to its lower effective inertia, but the damping ratio of 0.198 is the most severely under-damped of all configured joints. With ζ ≈ 0.2, the knee will exhibit approximately 4 to 5 oscillations of diminishing amplitude after any perturbation before settling. This is problematic during ground contact events where abrupt loading changes are common. The maximum stiffness torque at the upper joint limit (1.483 rad) is 80 × 1.483 = 118.6 Nm, which is 28.2% of the 420 Nm effort limit.

### 6.4 Ankle Pitch (K=15, D=1, I=0.039 kg·m²)

```
ωn = sqrt(15 / 0.039) = sqrt(384.62) = 19.61 rad/s    (3.12 Hz)
ζ  = 1 / (2 × sqrt(15 × 0.039)) = 1 / 1.530 = 0.654
```

The ankle pitch has a high natural frequency and a damping ratio of 0.654, which is within the acceptable range of 0.6 to 1.0. The low effective inertia of the foot (0.039 kg·m²) means even a moderate stiffness of 15 Nm/rad produces a fast, adequately damped response.

### 6.5 Ankle Roll (K=15, D=1, I=0.033 kg·m²)

```
ωn = sqrt(15 / 0.033) = sqrt(454.55) = 21.32 rad/s    (3.39 Hz)
ζ  = 1 / (2 × sqrt(15 × 0.033)) = 1 / 1.407 = 0.711
```

The ankle roll is the best-tuned joint in the entire configuration. A damping ratio of 0.711 is very close to the often-cited optimal value of 1/sqrt(2) ≈ 0.707 for minimum integral-squared-error tracking. The natural frequency of 21.3 rad/s provides fast lateral disturbance rejection.

### 6.6 Summary of Current Dynamic Properties

| Joint | K (Nm/rad) | D (Nm·s/rad) | I_eff (kg·m²) | ωn (rad/s) | ωn (Hz) | ζ | Assessment |
|---|---|---|---|---|---|---|---|
| Hip Pitch | 60 | 14 | 6.494 | 3.04 | 0.48 | 0.355 | Under-stiff, under-damped |
| Hip Roll | 80 | 14 | 6.499 | 3.51 | 0.56 | 0.307 | Under-stiff, under-damped |
| Knee Pitch | 80 | 4 | 1.272 | 7.93 | 1.26 | 0.198 | Marginal stiffness, severely under-damped |
| Ankle Pitch | 15 | 1 | 0.039 | 19.61 | 3.12 | 0.654 | Acceptable |
| Ankle Roll | 15 | 1 | 0.033 | 21.32 | 3.39 | 0.711 | Well-tuned |

---

## 7. Gravitational Torque Analysis

Understanding the torque demands imposed by gravity provides a lower bound on what the actuator gains must achieve to maintain meaningful tracking.

### 7.1 Hip Pitch Gravitational Torque

The hip pitch must dynamically resist the gravitational torque of all distal links. During single-support walking, the stance leg bears the full body weight. The combined mass distal to the hip pitch joint is:

```
m_leg = m_Link3 + m_Link4 + m_Link5 + m_Link6
      = 9.645 + 5.805 + 0.0722 + 3.691 = 19.213 kg
```

The composite centre of mass distance below the hip pitch joint is computed as a mass-weighted average of CoM distances:

```
r_leg_CoM = (9.645 × 0.242 + 5.805 × 0.632 + 0.0722 × 0.870 + 3.691 × 0.939) / 19.213
           = (2.334 + 3.669 + 0.063 + 3.466) / 19.213
           = 9.532 / 19.213 = 0.496 m
```

The gravitational torque at the hip pitch at a representative hip angle of 30° (0.524 rad) from vertical during swing phase is:

```
τ_grav_hip_pitch = m_leg × g × r_leg_CoM × sin(30°)
                 = 19.213 × 9.81 × 0.496 × 0.500
                 = 46.8 Nm
```

At this position, the stiffness contribution from the current K=60 setting with action scale of 0.25 rad and typical tracking error of 0.1 rad is:

```
τ_K = K × error = 60 × 0.1 = 6 Nm
```

This is roughly 13% of the gravitational torque, which means the PD controller's proportional term contributes very little to gravity compensation. The policy must output near-perfect desired position commands to compensate, placing an enormous burden on the RL training to discover the precise feedforward commands rather than relying on any closed-loop stiffness action.

### 7.2 Knee Pitch Gravitational Torque

The mass distal to the knee pitch at a typical 60° knee flexion angle (0° being fully extended) is:

```
m_lower_leg = m_Link4 + m_Link5 + m_Link6 = 5.805 + 0.0722 + 3.691 = 9.568 kg
```

The composite CoM distance below the knee:

```
r_lower_CoM = (5.805 × 0.192 + 0.0722 × 0.430 + 3.691 × 0.499) / 9.568
            = (1.114 + 0.031 + 1.842) / 9.568
            = 2.987 / 9.568 = 0.312 m
```

Gravitational torque at 60° knee bend:

```
τ_grav_knee = 9.568 × 9.81 × 0.312 × sin(60°) = 9.568 × 9.81 × 0.312 × 0.866 = 25.3 Nm
```

With K=80 and a tracking error of 0.1 rad, the stiffness torque is 8.0 Nm, which is 31.6% of the gravitational torque. This is better than the hip but still inadequate for confident gravity compensation.

### 7.3 Ankle Pitch Gravitational Torque (from foot alone)

The foot (Link6) is the only link distal to the ankle pitch. Its gravitational torque at 15° pitch angle is:

```
τ_grav_ankle = 3.691 × 9.81 × 0.069 × sin(15°) = 3.691 × 9.81 × 0.069 × 0.259 = 0.648 Nm
```

With K=15 and error 0.1 rad, the stiffness torque is 1.5 Nm, which already exceeds the gravitational torque from the foot alone. The ankle pitch actuator's primary role in locomotion is balancing the body weight transmitted through the ground contact (ground reaction force), not just supporting the foot's own weight. However, as a proportional controller the balance of GRF torques is handled through commanded position trajectories from the policy.

---

## 8. Comparison with the TRON1 Reference Configuration

The codebase also maintains the lighter TRON1 robot in `solefoot_identified_cfg.py`. Its configured gains provide a useful internal reference:

| Joint | K (Nm/rad) | D (Nm·s/rad) | Effort Limit (Nm) |
|---|---|---|---|
| TRON1 ABAD (Hip Roll) | 55 | 13.5 | 60 |
| TRON1 Hip Pitch | 80 | 13.0 | 60 |
| TRON1 Knee Pitch | 60 | 4.0 | 90 |
| TRON1 Ankle Pitch | 10 | 0.5 | 40 |

The TRON1 effort limits are 60 Nm for hip joints and 90 Nm for the knee, compared to 298 to 420 Nm for the BRS. This represents a scaling factor of roughly 5× in actuator capacity. However, the BRS stiffness gains are only 1.0 to 1.5× those of TRON1, suggesting the BRS is underexploiting its actuator capabilities. If one scales TRON1's stiffness gains proportionally to the BRS effort limits:

```
K_BRS_hip_pitch_scaled = K_TRON1_hip × (effort_BRS_hip / effort_TRON1_hip)
                       = 80 × (298 / 60) = 80 × 4.97 = 398 Nm/rad
```

```
K_BRS_knee_scaled = K_TRON1_knee × (effort_BRS_knee / effort_TRON1_knee)
                  = 60 × (420 / 90) = 60 × 4.67 = 280 Nm/rad
```

```
K_BRS_ankle_pitch_scaled = K_TRON1_ankle × (effort_BRS_ankle_pitch / effort_TRON1_ankle)
                         = 10 × (262 / 40) = 10 × 6.55 = 65.5 Nm/rad
```

These effort-scaled estimates substantially exceed the current BRS configuration and align more closely with the inertia-based targets computed in Section 9.

---

## 9. Ideal Gain Derivation

### 9.1 Methodology

For each joint, the ideal stiffness is chosen to achieve a target natural frequency `ωn_target`, and the ideal damping is chosen to achieve a target damping ratio `ζ_target`:

```
K_ideal = ωn_target² × I_eff
D_ideal = 2 × ζ_target × sqrt(K_ideal × I_eff)
        = 2 × ζ_target × ωn_target × I_eff
```

Two targets are evaluated: a conservative target suitable for RL training stability (lower bandwidth, wider action space), and an aggressive target reflecting physically appropriate servo response.

### 9.2 Saturation Check

Before finalising K_ideal, it is necessary to verify that the maximum proportional torque (K × θ_max) does not exceed the effort limit, since saturating the proportional term at the joint limit eliminates the possibility of increasing position-error-correcting force beyond the hardware ceiling. For RL, it is also beneficial to ensure that normal operating errors (of order 0.2 to 0.5 rad) produce torques meaningfully below the effort limit so the policy has dynamic range to exploit.

For HipPitch with effort_limit = 298 Nm and θ_max = 0.576 rad, the stiffness limit is K_max = 298 / 0.576 = 517 Nm/rad. For HipRoll with effort_limit = 351 Nm and θ_max = 0.349 rad, K_max = 351 / 0.349 = 1006 Nm/rad. For KneePitch with effort_limit = 420 Nm and θ_max = 1.483 rad, K_max = 420 / 1.483 = 283 Nm/rad. For AnklePitch with effort_limit = 262 Nm and θ_max = 0.349 rad, K_max = 262 / 0.349 = 751 Nm/rad. For AnkleRoll with effort_limit = 131 Nm and θ_max = 0.454 rad, K_max = 131 / 0.454 = 289 Nm/rad.

The tightest constraint is at KneePitch, where K must not exceed 283 Nm/rad to avoid saturation at the joint limit.

### 9.3 Hip Pitch

```
Target ωn = 10 rad/s, ζ_target = 0.70
K_ideal = 10² × 6.494 = 649 Nm/rad
D_ideal = 2 × 0.70 × 10 × 6.494 = 90.9 Nm·s/rad
```

However, K = 649 exceeds the saturation limit of 517 Nm/rad at the joint extreme. A practical ceiling accounting for typical operating range (errors up to 0.3 rad) and the need to avoid saturation is K ≈ 300 to 400 Nm/rad. At K = 350:

```
ωn_practical = sqrt(350 / 6.494) = sqrt(53.89) = 7.34 rad/s
D_practical = 2 × 0.70 × sqrt(350 × 6.494) = 1.40 × sqrt(2272.9) = 1.40 × 47.67 = 66.7 Nm·s/rad
```

Recommended range: K = 200 to 350 Nm/rad, D = 40 to 70 Nm·s/rad.

### 9.4 Hip Roll

```
Target ωn = 10 rad/s, ζ_target = 0.70
K_ideal = 10² × 6.499 = 650 Nm/rad
D_ideal = 2 × 0.70 × 10 × 6.499 = 91.0 Nm·s/rad
```

The saturation limit at the hip roll joint limit (0.349 rad) and effort limit (351 Nm) gives K_max = 1006 Nm/rad, which is not binding. However, for training stability, a practical ceiling of K ≈ 300 to 450 Nm/rad is appropriate.

At K = 350, D = 66.7 Nm·s/rad (same inertia as hip pitch).

Recommended range: K = 200 to 400 Nm/rad, D = 40 to 75 Nm·s/rad.

### 9.5 Knee Pitch

The knee is constrained by the saturation limit of K_max = 283 Nm/rad.

```
At K = 200, ωn = sqrt(200 / 1.272) = sqrt(157.2) = 12.54 rad/s
D_ideal = 2 × 0.70 × 12.54 × 1.272 = 22.36 Nm·s/rad
```

At K = 250 (approaching the saturation ceiling):

```
ωn = sqrt(250 / 1.272) = sqrt(196.5) = 14.02 rad/s
D_ideal = 2 × 0.70 × 14.02 × 1.272 = 24.99 Nm·s/rad
```

Recommended range: K = 150 to 250 Nm/rad, D = 15 to 25 Nm·s/rad.

### 9.6 Ankle Pitch

The current values (K=15, D=1) already achieve ωn = 19.6 rad/s and ζ = 0.654, which are within the desired range. A modest increase can be considered:

```
At K = 20, ωn = sqrt(20 / 0.039) = sqrt(512.8) = 22.65 rad/s
D_ideal = 2 × 0.70 × 22.65 × 0.039 = 1.235 Nm·s/rad
```

Recommended range: K = 15 to 25 Nm/rad, D = 1.0 to 1.5 Nm·s/rad (current values are acceptable).

### 9.7 Ankle Roll

The current values (K=15, D=1) achieve ωn = 21.3 rad/s and ζ = 0.711, which is already optimal. Slight increase is possible but not required:

Recommended range: K = 13 to 20 Nm/rad, D = 0.9 to 1.2 Nm·s/rad (current values are good).

### 9.8 Summary of Ideal vs. Current Gains

| Joint | Current K | Current D | Ideal K (range) | Ideal D (range) | Current ωn | Ideal ωn | Current ζ | Ideal ζ |
|---|---|---|---|---|---|---|---|---|
| Hip Pitch | 60 | 14 | 200 to 350 | 40 to 70 | 3.04 | 5.5 to 7.3 | 0.355 | ~0.70 |
| Hip Roll | 80 | 14 | 200 to 400 | 40 to 75 | 3.51 | 5.5 to 7.8 | 0.307 | ~0.70 |
| Knee Pitch | 80 | 4 | 150 to 250 | 15 to 25 | 7.93 | 10.9 to 14.0 | 0.198 | ~0.70 |
| Ankle Pitch | 15 | 1 | 15 to 25 | 1.0 to 1.5 | 19.61 | 19.6 to 25.3 | 0.654 | ~0.70 |
| Ankle Roll | 15 | 1 | 13 to 20 | 0.9 to 1.2 | 21.32 | 20.0 to 24.6 | 0.711 | ~0.70 |

---

## 10. Discussion

### 10.1 Why the Current Hip and Knee Gains Are Problematic

The current stiffness of K=60 to 80 Nm/rad at the hip and knee joints produces natural frequencies below 8 rad/s for a robot of 59.85 kg. These values were likely inherited from the TRON1 robot (which has similar absolute gain values) without accounting for the dramatic difference in mass. The TRON1's effort limits of 60 to 90 Nm indicate a significantly lighter platform, and the TRON1's stiffness values are appropriate for its scale.

The consequence for RL training is twofold. First, the proportional term contributes very little to gravity compensation or disturbance rejection at small tracking errors, meaning the policy must output nearly perfect desired trajectories rather than relying on closed-loop stiffness to absorb errors. This increases the effective difficulty of the learning problem. Second, the severe under-damping (ζ = 0.2 to 0.36 for the hip and knee joints) means that any external perturbation or policy-commanded step change in desired position will excite prolonged oscillations, potentially destabilising the gait and causing early episode termination.

### 10.2 Why the Damping Is Inconsistently Tuned at the Knee

The knee damping of D=4 Nm·s/rad is particularly low relative to the stiffness of K=80 Nm/rad. The ratio D/sqrt(K×I) = ζ/1 = 0.198 places the knee far into the oscillatory regime. The hip joints, despite being similarly under-stiffened, at least have D=14 which gives ζ ≈ 0.31 to 0.36. The knee's D=4 does not appear to have been derived from any dynamic analysis but rather was likely copied from the TRON1 knee setting unchanged. For the BRS knee with I_eff = 1.272 kg·m², achieving ζ = 0.70 requires D = 2 × 0.70 × sqrt(80 × 1.272) = 2 × 0.70 × 10.09 = 14.1 Nm·s/rad even at the current low stiffness of K=80. The damping should be at minimum 14 Nm·s/rad at K=80, and scaled proportionally if K is increased.

### 10.3 The Ankle Joints Are Well Configured

The ankle roll and ankle pitch joints benefit from the fact that the foot (Link6, 3.691 kg) is the only distal link, and it is relatively compact. The effective inertias of 0.033 and 0.039 kg·m² mean that even small stiffness values produce fast, well-damped responses. The current K=15, D=1 settings are physically justified and should not be changed substantially.

### 10.4 Consideration of the RL Action Scale

The actions config uses `scale=0.25` rad, meaning the policy's output of ±1 maps to ±0.25 rad of position command relative to the default pose. At K=60, the maximum proportional torque from a saturated action is 60 × 0.25 = 15 Nm for the hip pitch. This is less than one-third of the typical gravitational torque at a moderate hip angle (46.8 Nm computed in Section 7.1). At K=300, the maximum proportional torque is 300 × 0.25 = 75 Nm, which is meaningful for gravity compensation and disturbance rejection. Increasing K to the recommended range therefore also makes the action space more physically informative and may improve training convergence.

### 10.5 Effort Limits and Utilisation

The effort limits of 298 to 420 Nm for the knee and hip pitch joints are generous for a 60 kg robot. Many state-of-the-art humanoids of similar mass (Unitree H1 at 47 kg, Agility Robotics Digit at 42 kg) operate hip and knee joints with effort limits of 120 to 300 Nm. The BRS effort limits are large, suggesting powerful motors with high gear ratios. The low stiffness gains fail to exploit this actuator capacity.

---

## 11. Recommended Final Gain Values

Based on the physics analysis, the following point values are recommended as starting configurations for RL training. They represent a balanced choice within the ideal ranges computed above, respecting the effort-limit saturation constraints:

| Joint | Recommended K (Nm/rad) | Recommended D (Nm·s/rad) | Resulting ωn (rad/s) | Resulting ζ |
|---|---|---|---|---|
| HipRoll | 300 | 60 | 6.79 | 0.70 |
| HipPitch | 250 | 50 | 6.21 | 0.70 |
| KneePitch | 200 | 22 | 12.54 | 0.70 |
| AnkleRoll | 15 | 1 | 21.32 | 0.71 |
| AnklePitch | 15 | 1 | 19.61 | 0.65 |

These values give all proximal joints natural frequencies above 6 rad/s and damping ratios at or near the critically-matched value of 0.70, while keeping all joints within their effort-limit saturation boundaries for normal operating position errors.

If training stability requires starting with lower gains (e.g. to allow the policy to explore freely before committing to high joint impedance), a phased approach is possible: begin training at K = 100 to 150 for hip and knee, verify policy convergence on flat ground, then gradually increase to the recommended values above.

---

## 12. Summary

The SD_BRS1 is a 59.85 kg bipedal robot with a 0.87 m hip-to-ankle leg length, 10 active revolute joints, and actuator effort capacities of 131 to 420 Nm. The HipYaw joints are mechanically fixed and no actuator gain is applied to them in practice.

The central finding is that the hip pitch, hip roll, and knee pitch joints are configured with stiffness values (K = 60 to 80 Nm/rad) and damping values that are appropriate for a robot approximately 4 to 5 times lighter than the BRS. The effective rotational inertias of 6.49 kg·m² at the hips and 1.27 kg·m² at the knee demand proportionally higher gains to achieve usable closed-loop bandwidth and adequate damping.

The ankle joints (roll and pitch), by contrast, are well-tuned for the low effective inertia of the foot (0.033 to 0.039 kg·m²), achieving natural frequencies near 20 rad/s and damping ratios near 0.70 with their current K=15, D=1 settings.

The recommended course of action is to increase hip roll and hip pitch stiffness to the range 250 to 350 Nm/rad with damping in the range 50 to 70 Nm·s/rad, and to increase knee pitch stiffness to 150 to 250 Nm/rad with damping in the range 15 to 25 Nm·s/rad, leaving the ankle joints unchanged.
