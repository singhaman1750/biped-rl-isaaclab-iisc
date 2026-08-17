# Joint Stiffness, Damping, and Action Scaling in Bipedal Locomotion Training

## Abstract

This document provides a rigorous analysis of the relationship between proportional-derivative (PD) joint control, the mass-spring-damper dynamical system it induces, and the role of action scaling in reinforcement learning policies for bipedal locomotion. The analysis is grounded in classical control theory and contextualised within the IsaacLab simulation framework. Particular attention is given to how the choice of action scale interacts with actuator velocity limits and system damping characteristics, and how this interaction leads to training instability.

---

## 1. Introduction

Bipedal locomotion controllers trained via reinforcement learning (RL) typically interface with a simulated robot through low-level joint actuators. In IsaacLab, these actuators are modelled as implicit PD controllers: the RL policy outputs joint position targets, which are tracked by proportional-derivative feedback controllers embedded in the physics engine. The resulting closed-loop joint dynamics are precisely those of a second-order mass-spring-damper system.

Understanding this equivalence is critical for several reasons:

- The **stiffness** (proportional gain $K_p$) and **damping** (derivative gain $K_d$) parameters determine whether the joint exhibits oscillatory, critically damped, or overdamped behaviour.
- The **action scale** directly determines the amplitude of the forcing function applied to the joint, and therefore the range of velocities demanded of the actuator.
- A mismatch between demanded velocities and the actuator's velocity limit causes saturation, which breaks the linear system assumptions and can induce sustained oscillations, degraded reward signals, and ultimately training divergence.

The following sections develop these ideas formally, using the Solefoot (SF-TRON1A) bipedal robot as a concrete reference where numerical examples are required.

**Reference robot parameters (Solefoot SF-TRON1A):**

| Parameter | Leg Joints | Ankle Joints |
|---|---|---|
| Proportional gain $K_p$ (N·m/rad) | 45 | 45 |
| Derivative gain $K_d$ (N·m·s/rad) | 1.5 | 0.8 |
| Effort limit (N·m) | 80 | 80 |
| Velocity limit $\dot{q}_\text{lim}$ (rad/s) | 25 | 15 |
| Effective inertia $I$ (kg·m²) | ~0.10 | ~0.05 |

---

## 2. PD Control in IsaacLab

### 2.1 Implicit Actuator Model

IsaacLab models joints as **implicit actuators**: the physics engine (PhysX) internally computes and applies the PD torque at every simulation substep. The actuator is parameterised by a stiffness $K_p$ and a damping $K_d$, and receives a desired joint position $q_\text{des}$ as input.

At each physics substep the applied torque is:

$$\tau = K_p \left( q_\text{des} - q \right) - K_d \,\dot{q} \tag{1}$$

where $q$ is the current joint position (rad) and $\dot{q}$ is the current joint velocity (rad/s). The resulting torque is clamped to the effort limit before being applied:

$$\tau_\text{applied} = \text{clip}\left(\tau,\ {-\tau_\text{lim}},\ \tau_\text{lim}\right) \tag{2}$$

and the joint velocity is clamped to the velocity limit:

$$\dot{q}_\text{applied} = \text{clip}\left(\dot{q},\ {-\dot{q}_\text{lim}},\ \dot{q}_\text{lim}\right) \tag{3}$$

The "implicit" nature of this model means the PD computation is performed inside PhysX at each of the solver's position iterations (typically $\text{sim.dt} = 0.005$ s per substep), rather than at the coarser policy control frequency. This provides numerical stability compared to explicit actuator models.

### 2.2 Equilibrium and Setpoint Tracking

The actuator is designed to drive the joint to the setpoint $q_\text{des}$. At equilibrium ($\dot{q} = 0$, $\ddot{q} = 0$), equation (1) requires:

$$\tau_\text{eq} = K_p(q_\text{des} - q_\text{eq}) = 0 \implies q_\text{eq} = q_\text{des}$$

provided no external torques act on the joint. The joint thus asymptotically approaches $q_\text{des}$, with the transient behaviour governed by the second-order dynamics derived in Section 3.

---

## 3. The Mass-Spring-Damper Model

### 3.1 Equation of Motion

Consider a single rotational joint with effective rotational inertia $I$ (kg·m²). Applying Newton's second law for rotation:

$$I \ddot{q} = \sum \tau \tag{4}$$

Substituting the PD torque from equation (1) as the only applied torque (neglecting gravity and Coriolis terms, which is valid for small perturbations about the operating point):

$$I \ddot{q} = K_p(q_\text{des} - q) - K_d \dot{q} \tag{5}$$

Rearranging into standard second-order ODE form:

$$\boxed{I \ddot{q} + K_d \dot{q} + K_p q = K_p q_\text{des}} \tag{6}$$

This is the **equation of motion** of a forced mass-spring-damper system with:

| Mechanical analogue | Symbol | Joint equivalent |
|---|---|---|
| Mass | $m$ | Effective inertia $I$ |
| Damping coefficient | $c$ | Derivative gain $K_d$ |
| Spring constant | $k$ | Proportional gain $K_p$ |
| External forcing | $k \, x_\text{ref}$ | Setpoint force $K_p \, q_\text{des}$ |

### 3.2 Homogeneous Solution and Characteristic Equation

Setting $q_\text{des} = 0$ (free response), assume a solution of the form $q(t) = e^{\lambda t}$. Substituting into equation (6):

$$I \lambda^2 e^{\lambda t} + K_d \lambda e^{\lambda t} + K_p e^{\lambda t} = 0$$

Dividing by $e^{\lambda t} \neq 0$:

$$I \lambda^2 + K_d \lambda + K_p = 0 \tag{7}$$

This is the **characteristic equation** of the system. Its roots $\lambda$ determine the nature of the free response.

### 3.3 Natural Frequency

Dividing equation (7) by $I$:

$$\lambda^2 + \frac{K_d}{I}\lambda + \frac{K_p}{I} = 0 \tag{8}$$

The **undamped natural frequency** $\omega_n$ is defined by the constant term:

$$\boxed{\omega_n = \sqrt{\frac{K_p}{I}}} \quad \text{(rad/s)} \tag{9}$$

This is the frequency at which the system oscillates when $K_d = 0$ (no damping). Setting $K_d = 0$ in equation (8) and solving:

$$\lambda = \pm \, j \omega_n$$

The roots are purely imaginary — the free response is a sinusoid at $\omega_n$ that neither grows nor decays. $\omega_n$ is thus the oscillation frequency embedded in the characteristic equation before damping is introduced.

### 3.4 Derivation of the Damping Ratio

To extract a single dimensionless parameter characterising the system's damping, we complete the square in equation (8).

**Step 1 — Complete the square:**

$$\left(\lambda + \frac{K_d}{2I}\right)^2 - \left(\frac{K_d}{2I}\right)^2 + \omega_n^2 = 0$$

$$\left(\lambda + \frac{K_d}{2I}\right)^2 = \left(\frac{K_d}{2I}\right)^2 - \omega_n^2 \tag{10}$$

**Step 2 — Identify the critical condition:** The system transitions from oscillatory to non-oscillatory behaviour when the right-hand side of equation (10) equals zero:

$$\left(\frac{K_d}{2I}\right)^2 = \omega_n^2 \implies \frac{K_d}{2I} = \omega_n \implies K_{d,\text{crit}} = 2 I \omega_n$$

Substituting $\omega_n = \sqrt{K_p/I}$:

$$\boxed{K_{d,\text{crit}} = 2\sqrt{K_p \cdot I}} \tag{11}$$

**Step 3 — Define $\zeta$ as the ratio of actual to critical damping:**

$$\zeta \equiv \frac{K_d / 2I}{\omega_n} = \frac{K_d}{2 I \omega_n} = \frac{K_d}{2\sqrt{K_p \cdot I}} \tag{12}$$

Substituting back into equation (8) yields the **canonical second-order form**:

$$\lambda^2 + 2\zeta\omega_n \lambda + \omega_n^2 = 0 \tag{13}$$

with roots:

$$\lambda_{1,2} = \omega_n \left(-\zeta \pm \sqrt{\zeta^2 - 1}\right) \tag{14}$$

The factor of $2$ in $2\zeta\omega_n$ arises directly from the completing-the-square algebra: the term $K_d/I$ equals $2 \cdot (K_d/2I) = 2\zeta\omega_n$. It is not an empirical constant but a consequence of the quadratic structure of the ODE.

### 3.5 Behaviour by Damping Regime

The roots $\lambda_{1,2}$ in equation (14) determine the qualitative free response:

| Condition | Roots | Response |
|---|---|---|
| $\zeta < 1$ (underdamped) | Complex conjugates $-\zeta\omega_n \pm j\omega_d$ | Decaying sinusoid |
| $\zeta = 1$ (critically damped) | Repeated real $-\omega_n$ | Fastest non-oscillatory decay |
| $\zeta > 1$ (overdamped) | Two distinct negative reals | Slow exponential decay |

where the **damped natural frequency** is:

$$\omega_d = \omega_n \sqrt{1 - \zeta^2} \tag{15}$$

**Geometric interpretation:** All roots lie on a circle of radius $\omega_n$ in the complex plane, since $|\lambda|^2 = (\zeta\omega_n)^2 + \omega_d^2 = \omega_n^2(\zeta^2 + 1 - \zeta^2) = \omega_n^2$. Increasing $\zeta$ rotates the roots clockwise along this circle from the imaginary axis toward the negative real axis.

### 3.6 Underdamped Step Response

For a step change in setpoint from $q = 0$ to $q_\text{des} = A$ at $t = 0$, with zero initial conditions, the complete (particular + homogeneous) solution is:

$$q(t) = A \left[1 - e^{-\zeta\omega_n t}\left(\cos\omega_d t + \frac{\zeta}{\sqrt{1-\zeta^2}}\sin\omega_d t\right)\right] \tag{16}$$

The **peak overshoot** relative to the setpoint amplitude $A$ is:

$$M_p = \exp\!\left(\frac{-\pi\zeta}{\sqrt{1-\zeta^2}}\right) \tag{17}$$

The peak joint velocity during the step response occurs at $t = 0$ and can be bounded. The velocity of the step response is:

$$\dot{q}(t) = A \omega_n \frac{e^{-\zeta\omega_n t}}{\sqrt{1-\zeta^2}} \sin\omega_d t \tag{18}$$

The maximum velocity (occurring at $t = \arctan(\omega_d / (-\zeta\omega_n)) / \omega_d$) scales linearly with both $A$ (the amplitude of the setpoint step) and $\omega_n$, and decreases with increasing $\zeta$.

---

## 4. Joint Position Control Using Policies

### 4.1 The JointPositionAction Interface

In IsaacLab's manager-based RL framework, the `JointPositionActionCfg` class configures a discrete-time mapping from the policy output $\mathbf{a}_t \in \mathbb{R}^{n_\text{joints}}$ to the joint setpoint vector $\mathbf{q}_\text{des}$:

$$\mathbf{q}_\text{des}^{(t)} = \mathbf{q}_\text{default} + \alpha \cdot \mathbf{a}_t \tag{19}$$

where:
- $\mathbf{q}_\text{default} \in \mathbb{R}^{n_\text{joints}}$ is the robot's nominal joint configuration (the rest pose),
- $\alpha \in \mathbb{R}^+$ is the **action scale**, a scalar hyperparameter,
- $\mathbf{a}_t$ is the raw policy output, typically distributed approximately in $[-1, 1]$ early in training (before entropy collapse).

The setpoint $\mathbf{q}_\text{des}^{(t)}$ is held constant for one **control period** $\Delta t_\text{ctrl} = \text{decimation} \times \text{sim.dt}$, during which the physics engine steps `decimation` times. In the Solefoot configuration: $\Delta t_\text{ctrl} = 4 \times 0.005 = 0.02$ s.

### 4.2 Timing Hierarchy

The distinction between simulation and control timescales is important:

```
Policy inference (every control step):        Δt_ctrl = 0.02 s (50 Hz)
PhysX implicit PD evaluation (every substep): sim.dt  = 0.005 s (200 Hz)
```

The policy produces a new setpoint at 50 Hz. The PD controller tracks that setpoint at 200 Hz. The joint dynamics operate continuously at 200 Hz, meaning the transient response to each setpoint change evolves over 4 physics steps before the policy can issue a correction.

### 4.3 Effective Joint Displacement per Control Step

The maximum joint displacement commanded per control step is bounded by the action scale and the support of the policy output distribution. For a policy with output magnitude $|\mathbf{a}_t| \leq a_\text{max}$:

$$\Delta q_\text{max} = \alpha \cdot a_\text{max} \tag{20}$$

The maximum rate of change of the setpoint between consecutive control steps is:

$$\dot{q}_\text{des,max} = \frac{\Delta q_\text{max}}{\Delta t_\text{ctrl}} = \frac{\alpha \cdot a_\text{max}}{\Delta t_\text{ctrl}} \tag{21}$$

This is the maximum slew rate demanded of the joint actuator, and it must be compared against the actuator's velocity limit $\dot{q}_\text{lim}$.

---

## 5. Action Scaling and Its Impact on the Joint Model

### 5.1 Role of the Action Scale

The action scale $\alpha$ is the primary hyperparameter governing the amplitude of the forcing term in equation (6). Rewriting with the policy setpoint:

$$I \ddot{q} + K_d \dot{q} + K_p q = K_p \left(q_\text{default} + \alpha a_t\right) \tag{22}$$

Defining the displacement from default as $e = q - q_\text{default}$:

$$I \ddot{e} + K_d \dot{e} + K_p e = K_p \alpha a_t \tag{23}$$

The right-hand side $K_p \alpha a_t$ is the forcing amplitude. Larger $\alpha$ produces larger forcing, demanding greater joint velocities and larger corrective torques. The linear system of equation (23) remains valid only while the actuator limits (equations (2) and (3)) are not active.

### 5.2 Velocity Demand and the Velocity Limit

From equation (21), the maximum setpoint slew rate is $\dot{q}_\text{des,max} = \alpha a_\text{max} / \Delta t_\text{ctrl}$. The joint must track this setpoint, so the actuator must be capable of achieving at least this velocity. The condition for non-saturation is:

$$\frac{\alpha \cdot a_\text{max}}{\Delta t_\text{ctrl}} \leq \dot{q}_\text{lim} \tag{24}$$

Solving for the maximum permissible action scale:

$$\alpha_\text{safe} \leq \frac{\dot{q}_\text{lim} \cdot \Delta t_\text{ctrl}}{a_\text{max}} \tag{25}$$

For the Solefoot ankle joints ($\dot{q}_\text{lim} = 15$ rad/s, $\Delta t_\text{ctrl} = 0.02$ s, $a_\text{max} = 1$):

$$\alpha_\text{safe} \leq \frac{15 \times 0.02}{1} = 0.30 \text{ rad}$$

For the leg joints ($\dot{q}_\text{lim} = 25$ rad/s):

$$\alpha_\text{safe} \leq \frac{25 \times 0.02}{1} = 0.50 \text{ rad}$$

These represent the **maximum action scale values** at which the velocity limit is never saturated for a full-range policy output. At $\alpha = 0.25$ rad (ankle joints), the demanded slew rate is $12.5$ rad/s — comfortably within the 15 rad/s limit. At $\alpha = 0.50$ rad, the slew rate matches the velocity limit exactly; any policy output exceeding 50% of full range causes saturation.

### 5.3 Consequences of Velocity Saturation

When the velocity limit is active ($|\dot{q}| = \dot{q}_\text{lim}$), the PD control law in equation (1) is evaluated at a clamped velocity. This has two effects:

1. **The damping term is saturated.** The term $-K_d \dot{q}$ is evaluated at $\dot{q}_\text{lim}$ rather than the true velocity, meaning the damping force no longer increases proportionally with velocity. The effective damping ratio is reduced:

$$\zeta_\text{eff} < \zeta \quad \text{when } |\dot{q}| = \dot{q}_\text{lim} \tag{26}$$

2. **The system leaves the linear regime.** Equation (6) is no longer valid; the system becomes nonlinear. Superposition no longer applies: the response to multiple simultaneous setpoint changes cannot be predicted by summing individual responses.

The consequence is that the system behaves as if it is more underdamped than its $\zeta$ value suggests. Each setpoint step produces a larger overshoot than predicted by equation (17), and consecutive steps can constructively interfere — each overshoot serves as the initial condition for the next step, amplifying the oscillation amplitude over time.

### 5.4 Overshoot Growth Mechanism

Consider a sequence of alternating setpoint steps of amplitude $A$. After the $k$-th step, the joint's velocity at peak overshoot is:

$$\dot{q}_\text{peak}^{(k)} \approx A \cdot \omega_n \cdot \frac{e^{-\zeta\omega_n t_\text{peak}}}{\sqrt{1-\zeta^2}} + \dot{q}_\text{residual}^{(k-1)} \tag{27}$$

where $\dot{q}_\text{residual}^{(k-1)}$ is the velocity carried over from the previous step's ring-down. For an underdamped system with short control periods relative to the ring-down time $\tau_\text{decay} = 1/(\zeta\omega_n)$, residual velocities accumulate:

$$\tau_\text{decay} = \frac{1}{\zeta \omega_n} \gg \Delta t_\text{ctrl} \quad \implies \quad \text{velocity accumulation} \tag{28}$$

This is the core instability mechanism: the control period is short relative to the system's natural decay time, and the policy can issue large corrective actions faster than the underdamped joint can dissipate them.

---

## 6. Mathematical Analysis of the Solefoot System

### 6.1 Natural Frequency

Using equation (9) for leg joints ($K_p = 45$ N·m/rad, $I = 0.10$ kg·m²):

$$\omega_n = \sqrt{\frac{45}{0.10}} = \sqrt{450} \approx 21.2 \text{ rad/s} \approx 3.4 \text{ Hz} \tag{29}$$

For ankle joints ($I = 0.05$ kg·m²):

$$\omega_n^\text{ankle} = \sqrt{\frac{45}{0.05}} = \sqrt{900} = 30.0 \text{ rad/s} \approx 4.8 \text{ Hz} \tag{30}$$

### 6.2 Critical Damping Coefficient

Using equation (11) for leg joints:

$$K_{d,\text{crit}}^\text{leg} = 2\sqrt{45 \times 0.10} = 2\sqrt{4.5} \approx 4.24 \text{ N·m·s/rad} \tag{31}$$

For ankle joints:

$$K_{d,\text{crit}}^\text{ankle} = 2\sqrt{45 \times 0.05} = 2\sqrt{2.25} = 3.00 \text{ N·m·s/rad} \tag{32}$$

### 6.3 Damping Ratio

**Before stiffness/damping randomisation** (nominal gains):

Using equation (12) for leg joints ($K_d = 1.5$):

$$\zeta^\text{leg} = \frac{1.5}{4.24} \approx 0.354 \tag{33}$$

For ankle joints ($K_d = 0.8$):

$$\zeta^\text{ankle} = \frac{0.8}{3.00} \approx 0.267 \tag{34}$$

Both joints are **significantly underdamped** ($\zeta \ll 1$). The ankle joints are more underdamped due to their lower nominal $K_d = 0.8$ relative to their critical coefficient.

**After stiffness/damping randomisation** (uniform $K_p \in [32, 48]$, $K_d \in [2.0, 3.0]$):

For leg joints at mean randomised values ($K_p = 40$, $K_d = 2.5$, $I = 0.10$):

$$K_{d,\text{crit}} = 2\sqrt{40 \times 0.10} = 2\sqrt{4.0} = 4.00, \quad \zeta = \frac{2.5}{4.00} = 0.625 \tag{35}$$

Randomisation raises $K_d$ from 1.5 to the range $[2.0, 3.0]$, substantially improving damping ($\zeta: 0.35 \to 0.50$–$0.75$). The randomisation is thus beneficial for stability during training, acting as implicit domain randomisation over damping characteristics.

### 6.4 Decay Time Constant

The exponential envelope of the underdamped free response decays with time constant:

$$\tau_\text{decay} = \frac{1}{\zeta \omega_n} \tag{36}$$

For nominal leg joints:

$$\tau_\text{decay}^\text{leg} = \frac{1}{0.354 \times 21.2} \approx 0.133 \text{ s} = 6.7 \times \Delta t_\text{ctrl} \tag{37}$$

The control period is $\Delta t_\text{ctrl} = 0.02$ s. The free response takes approximately $6.7$ control steps to decay to $e^{-1} \approx 37\%$ of its initial amplitude. Since the policy issues a new setpoint every step, residual oscillations from step $k$ are still significant when step $k+1$ is issued. This confirms the accumulation mechanism of equation (28).

### 6.5 Peak Overshoot Analysis

Using equation (17) for nominal leg joints ($\zeta = 0.354$):

$$M_p = \exp\!\left(\frac{-\pi \times 0.354}{\sqrt{1 - 0.354^2}}\right) = \exp\!\left(\frac{-1.113}{0.935}\right) = \exp(-1.190) \approx 0.304 \tag{38}$$

A step setpoint change of amplitude $A$ produces a peak joint displacement of $A(1 + M_p) = 1.304 A$. For $\alpha = 0.25$ rad, a full-range policy output $a = 1$ commands $A = 0.25$ rad, with a peak overshoot of $0.304 \times 0.25 = 0.076$ rad above the target. The corresponding peak velocity from equation (18):

$$\dot{q}_\text{peak} \approx \frac{A \omega_n}{\sqrt{1-\zeta^2}} = \frac{0.25 \times 21.2}{0.935} \approx 5.67 \text{ rad/s} \tag{39}$$

This is well within the 25 rad/s velocity limit for leg joints: $\alpha = 0.25$ is safe.

For $\alpha = 0.50$ rad (doubled action scale), $A = 0.50$ rad:

$$\dot{q}_\text{peak} \approx \frac{0.50 \times 21.2}{0.935} \approx 11.3 \text{ rad/s} \tag{40}$$

Still within the leg velocity limit (25 rad/s), but for ankle joints ($\omega_n = 30$ rad/s, $\dot{q}_\text{lim} = 15$ rad/s):

$$\dot{q}_\text{peak} \approx \frac{0.50 \times 30.0}{\sqrt{1 - 0.267^2}} = \frac{15.0}{0.964} \approx 15.6 \text{ rad/s} \tag{41}$$

This **exceeds the ankle velocity limit of 15 rad/s**. The ankle joint saturates during the transient response, entering the nonlinear regime described in Section 5.3. The effective damping is reduced, overshoots grow, and consecutive policy steps can amplify the oscillation.

### 6.6 Causal Chain from Action Scale to Training Instability

The following chain summarises the propagation from the action scale hyperparameter to observable training metrics:

```
α: 0.25 → 0.50 (doubled action scale)
         │
         ▼
Peak velocity demand doubles (eq. 41)
         │
         ▼
Ankle velocity limit saturated → nonlinear regime (Section 5.3)
         │
         ▼
Effective ζ reduced → larger overshoots (eq. 26)
         │
         ▼
Consecutive steps accumulate oscillation energy (eq. 28)
         │
         ▼
Body oscillates vertically (pen_lin_vel_z ↑, pen_base_height ↑)
         │
         ▼
PhysX solver (4 position iterations) fails to converge
under high-velocity contact transitions
         │
         ▼
Physics divergence → NaN rewards → NaN value loss → NaN gradients
```

### 6.7 Stability Boundary

For the system to remain in the linear, well-damped regime, the following three conditions must hold simultaneously:

$$\alpha \leq \alpha_\text{safe} = \frac{\dot{q}_\text{lim} \cdot \Delta t_\text{ctrl}}{a_\text{max}} \tag{42}$$

$$\zeta \geq \zeta_\text{min} \approx 0.3 \quad \text{(sufficient ring-down within one control period)} \tag{43}$$

$$\Delta t_\text{ctrl} \geq \tau_\text{ring} \approx \frac{\pi}{\omega_d} \quad \text{(allow one oscillation half-period to complete)} \tag{44}$$

For the Solefoot system, condition (42) gives $\alpha_\text{safe} = 0.30$ rad for ankles. Conditions (43) and (44) are met by the nominal parameters. Condition (42) is violated at $\alpha = 0.50$ for ankles, which is the root cause of instability in the described training runs.

---

## 7. Summary and Recommendations

### 7.1 Summary of Key Results

| Quantity | Formula | Leg (nominal) | Ankle (nominal) |
|---|---|---|---|
| Natural frequency $\omega_n$ | $\sqrt{K_p/I}$ | 21.2 rad/s | 30.0 rad/s |
| Critical damping $K_{d,\text{crit}}$ | $2\sqrt{K_p I}$ | 4.24 N·m·s/rad | 3.00 N·m·s/rad |
| Damping ratio $\zeta$ | $K_d / 2\sqrt{K_p I}$ | 0.354 | 0.267 |
| Decay time constant $\tau_\text{decay}$ | $1/(\zeta\omega_n)$ | 0.133 s | 0.125 s |
| Peak overshoot $M_p$ | $e^{-\pi\zeta/\sqrt{1-\zeta^2}}$ | 30.4% | 39.4% |
| Safe action scale $\alpha_\text{safe}$ | $\dot{q}_\text{lim} \Delta t_\text{ctrl} / a_\text{max}$ | 0.50 rad | 0.30 rad |

### 7.2 Recommendations

**Action Scale:** Set $\alpha \leq 0.25$ rad to remain conservatively within the saturation boundary for both joint groups. This ensures the linear system assumptions remain valid throughout training, and the solver can maintain stable contact resolution.

**Damping Gains:** The nominal $K_d$ values (1.5 for legs, 0.8 for ankles) result in significantly underdamped behaviour ($\zeta \approx 0.27$–$0.35$). Increasing $K_d$ toward $K_{d,\text{crit}}/2$ (half-critical damping, $\zeta = 0.5$) would improve transient rejection without overdamping the joints. For leg joints, $K_d \approx 2.1$ N·m·s/rad corresponds to $\zeta = 0.5$; for ankles, $K_d \approx 1.5$ N·m·s/rad.

**Solver Iterations:** The solver iteration count `solver_position_iteration_count` determines how accurately contact constraints are resolved per substep. For underdamped joints experiencing contact transitions on rough terrain, a minimum of 8 position iterations is recommended.

**Reward Design:** Tracking reward kernels of the form $\exp(-\|e\|^2 / \sigma^2)$ should use $\sigma$ large enough to provide useful gradients during early training. For velocity tracking, $\sigma = 0.5$ m/s (corresponding to $\sigma^2 = 0.25$) is appropriate; $\sigma = 0.2$ m/s renders the positive reward near-zero for all policy outputs with tracking error above 0.4 m/s, suppressing learning signal during the initial phase when errors are large.

---

## Appendix A: Geometric Interpretation of the Characteristic Roots

All roots of the characteristic equation (13) lie on a circle of radius $\omega_n$ in the complex plane:

$$|\lambda|^2 = (\zeta\omega_n)^2 + (\omega_n\sqrt{1-\zeta^2})^2 = \omega_n^2(\zeta^2 + 1 - \zeta^2) = \omega_n^2$$

The roots sweep along this circle as $\zeta$ increases from 0 to 1:

- At $\zeta = 0$: roots at $\pm j\omega_n$ (purely imaginary, undamped oscillation)
- At $\zeta = 1$: roots merge at $-\omega_n$ (critically damped, real)
- At $\zeta > 1$: roots split along the negative real axis (overdamped)

The natural frequency $\omega_n$ is thus the distance from the origin to either root in the complex plane, invariant with respect to damping. The damping ratio $\zeta$ determines the angle of the root vector from the negative real axis: $\cos^{-1}(\zeta)$.

---

## Appendix B: Notation

| Symbol | Description | Units |
|---|---|---|
| $q$ | Joint position | rad |
| $\dot{q}$ | Joint velocity | rad/s |
| $\ddot{q}$ | Joint acceleration | rad/s² |
| $q_\text{des}$ | Desired joint position (setpoint) | rad |
| $q_\text{default}$ | Default (rest) joint position | rad |
| $I$ | Effective rotational inertia | kg·m² |
| $K_p$ | Proportional gain (stiffness) | N·m/rad |
| $K_d$ | Derivative gain (damping) | N·m·s/rad |
| $\tau$ | Applied joint torque | N·m |
| $\omega_n$ | Undamped natural frequency | rad/s |
| $\omega_d$ | Damped natural frequency | rad/s |
| $\zeta$ | Damping ratio | dimensionless |
| $K_{d,\text{crit}}$ | Critical damping coefficient | N·m·s/rad |
| $\tau_\text{decay}$ | Exponential decay time constant | s |
| $M_p$ | Peak overshoot ratio | dimensionless |
| $\alpha$ | Action scale | rad |
| $\mathbf{a}_t$ | Policy action output | dimensionless |
| $\Delta t_\text{ctrl}$ | Control period | s |
| $\dot{q}_\text{lim}$ | Joint velocity limit | rad/s |
