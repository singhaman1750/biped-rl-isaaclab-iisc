# Natural frequency and damping ratio optimisation of the joint servos in the CMA-ES co-optimisation

Status, NOT IMPLEMENTED, written 2026-10-27 and revised the same day against the user's review, which added the gain precedence of section 3.6, the derivation of section 3.3 and the comparison with the KScale damping rule of section 3.4, and narrowed the damping ratio range to 0.7 to 1.0. Revised again on 2026-10-29 against the second review, which asked for the modelling choice of section 3.3.2, the eigenvalue background of section 3.3.3, the derivations of section 3.3.4 and a sweep of the inertia over the pose space. The sweep found that the first version's reference inertia was wrong in two ways, an interior minimum the coarse pose grid missed and the inertial coupling of the joints, which section 3.2 and section 3.3.7 now treat, and which changed the reference, the biped's frequency override, the friction check and the randomisation risk downstream. Every number is reproduced by [scripts/analysis/inertia_analysis.py](../scripts/analysis/inertia_analysis.py) and [validate_coupled.py](../context/artefacts/w_eta_optimisation-2026-10-27/validate_coupled.py), whose outputs are kept in [../context/artefacts/w_eta_optimisation-2026-10-27/](../context/artefacts/w_eta_optimisation-2026-10-27/). Revised a third time on 2026-10-29 against the review of sections 3.2.1 to 3.3.7, which asked for the origin of Rodrigues' formula, the derivation of the free inertia and of the trigonometric law with every variable defined, the role of $P_j$, which was validated once more against MuJoCo, the meaning of $S$, $W$, $q$ and $\lambda$, a full derivation of both cases of section 3.3.7, and one consistent notation, with the state transition matrix renamed $\Phi$ throughout and a notation chart added before the bibliography. A fourth revision of 2026-10-29 rewrote case B of section 3.3.7 as ten derived steps, with the energy written in a product form whose terms are exactly the kinetic energy and the servo potential energy, and an explicit passage from its positivity to the eigenvalues of $\Phi$. A fifth revision of 2026-10-30 answered the review of the opening of case B, explaining the sign of the servo acceleration in step B1, constructing the energy of step B4 from the recurrence by the discrete multiplier method so that its kinetic weight $\mathbb{1} - \frac{1}{2}\hat D$ is derived rather than asserted, and enumerating the polarisation identity behind its midpoint form. A sixth revision the same day answered the review of the close of case B, giving the background for the positive definiteness of $\Gamma$, the purpose and every step of the state bound, the derivation of $\Phi_y$ and of $L^{-1}$, the proof that similar matrices share their eigenvalues, a rewritten passage from the hypothesis to the search box in four moves with the single joint numbers recalled, and a closing list of the conditions section 3.3.8 applies.

## 1. Introduction

### 1.1 The problem

The co-optimisation searches the thigh and shank lengths and the nominal pose of each design while a single policy learns to walk on all of them. Every joint is driven by a proportional derivative servo whose stiffness $k_p$ and damping $k_d$ are fixed per actuator in the catalogue. A servo's behaviour is set by its natural frequency and damping ratio, and both depend on the inertia the servo drives, so a fixed gain pair gives a different joint on every design the search proposes. This document proposes that the natural frequency $f_n$ and the damping ratio $\zeta$ of each actuator group become search variables of `CMAESDesignGenerator` and its two subclasses, and that the gains be computed from them against each design's own inertia, within the stability limits of the way Isaac Lab integrates this repository's actuator.

### 1.2 The current implementation

The gains are a property of the catalogue entry. `CatalogActuator` carries optional `stiffness` and `damping` fields (`design_generator/common/actuator_catalog.py:525-526`), the loader reads them from `catalogs/actuators.json` through `_SIM_JSON_KEYS` (`actuator_catalog.py:604-608`), and `sim_params` passes them on whenever the entry states them (`actuator_catalog.py:561-565`). Only the TRON1 and A1 entries state them, and those are the only entries the search may pick (`actuator_catalog.py:289-297`). The biped uses 55 and 13.5, 80 and 13, 60 and 4, and 10 and 0.5 for the abduction, hip, knee and ankle groups (`design_generator/biped/catalogs/actuators.json:235-293`), and the quadruped uses 40 and 1.2, 40 and 1.1, and 50 and 0.9 (`design_generator/quadruped/catalogs/actuators.json:208-245`).

`DesignGeneratorBase._actuator_params` copies `sim_params()` for each group (`co_optimisation/co_optimisation/runners/usd_generator.py:732-743`), and the runner hands the result to `apply_actuator_params`, which overwrites the actuator's per environment `stiffness` and `damping` rows for every environment of every design at each generation (`co_optimisation/co_optimisation/utils/respawn.py:246-310`, called at `copt_on_policy_runner.py:618-620`). The catalogue's armature is written to PhysX by the same call (`respawn.py:243`, `:301-305`).

`DEFAULT_PARAM_RANGES` holds eight multiplicative gain scales, `abad_stiffness_scale` through `ankle_damping_scale` (`usd_generator.py:436-443`), applied by `_actuator_params` when they appear in the parameters. CMA-ES never searches them. `CMAES_PARAM_KEYS` holds the two length scales and the two pose offsets alone (`usd_generator.py:447-449`), the constructor keeps only those keys (`usd_generator.py:867`), and every CMA-ES `_generate_individual` calls `_actuator_params(spec)` without the parameters (`usd_generator.py:983`, `:1050`, `:1224`). Only `RandomDesignGenerator` passes them (`usd_generator.py:824`), and it samples every key of `DEFAULT_PARAM_RANGES`. The design logs of all eight co-optimisation runs that carry them, `copt_moral` and `quadruped_copt` from 2026-10-22 to 2026-10-26, record the four length and pose keys and nothing else, so no run on disk has changed a gain.

The MJCF carries a different quantity. `builder.py` writes each joint class's passive `damping`, `armature` and `frictionloss` from `JOINT_DYNAMICS` in `constant_params.json` (`design_generator/biped/builder.py:539-545`). That is the passive joint dynamics of the document and not the servo gain. The armature it writes, 0.05 at the biped's hip, is replaced at runtime by the catalogue's armature through `apply_actuator_params`. This plan touches none of the MJCF terms.

The actuator is `IdentifiedActuator`, a `DCMotor` whose torque is computed in Python and handed to PhysX as an effort, with a friction torque subtracted (`environments/environments/actuators/actuator_pd.py:18-40`). It is evaluated once per physics step inside the decimation loop (`IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:182-189`), and both co-optimisation tasks run that loop at `sim.dt` 0.005 s with a decimation of 4 and a joint position action scale of 0.25 (`params/env.yaml` lines 22, 85 and 1316 of `copt_moral/2026-10-26_04-55-55`, and lines 22, 85 and 1245 of `quadruped_copt/2026-10-26_04-55-55`). The servo is therefore an explicit element integrated at 200 Hz, which is the source of the stability constraint of section 3.3.

One defect was found along the way, and section 3.6 repairs it at the user's direction. The biped task randomises the gains at startup through `randomize_actuator_gains`, scaling the knee stiffness by 0.833 to 1.167 and its damping by 0.75 to 1.25 for instance (`params/env.yaml:1423-1460` of `copt_moral/2026-10-26_04-55-55`). That event writes `actuator.stiffness` and `actuator.damping` (`IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:632-647`), and `apply_actuator_params` then overwrites both rows with the catalogue's values at the first generation. The startup randomisation of the gains has therefore had no effect in any co-optimisation run. The quadruped's task carries the same three terms with stiffness scales of 0.85 to 1.15 and damping scales of 0.8 to 1.2 (`params/env.yaml:1357-1470` of `quadruped_copt/2026-10-26_04-55-55`). The eight retired scales of `DEFAULT_PARAM_RANGES` carry the same ranges, which is how they were derived (`usd_generator.py:27-35`).

A second defect was found by the sweep of section 3.2. The biped's catalogue gains, its abduction damping of 13.5 above all, are outside the explicit servo's stable region at a large share of the poses its legs can reach, once the inertial coupling of the joints is counted, and a direct MuJoCo integration confirms the divergence at the worst of them (section 3.3.8). Every biped co-optimisation run to date has carried these gains. The quadruped's are stable at every pose tested.

### 1.3 What the preceding discussion established

Four results from the discussion that preceded this plan, each recorded in the workspace, bear on the design.

1. A joint servo is a second order system, and its gains follow from a natural frequency, a damping ratio and an inertia [1] [2]. Section 3.1 derives this.
2. A legged joint meets two very different inertias, its distal links in swing and the body in stance, which differ by up to 937 times on the KScale biped (`../context/KScale.md` sections 3.2 and 4.9). Bjelonic and colleagues report one to two orders of magnitude on ANYmal [3].
3. An explicitly integrated damper is only conditionally stable, and the limit is set by the smallest inertia the joint meets (`../context/KScale.md` section 11.1.1). Section 3.3.7 extends this to the coupled robot, where the smallest inertia becomes a matrix condition on the inverse mass matrix.
4. Of the actuator parameters, this workspace knows the armature of every searchable actuator from the catalogue. It does not know the friction or the command delay, the friction values in the catalogue having no recorded provenance (`/ws/context/literature.md` cluster 41).

### 1.4 Scope

The plan adds per group $f_n$ and $\zeta$ fields to both families' `DesignSpec` and `DesignBounds`, an inertia module taken from the analysis script of section 3.2.2, and changes to `usd_generator.py` confined to the CMA-ES generators and the tables they read. `RandomDesignGenerator` keeps the catalogue gains. With `CMAES_PARAM_KEYS` as it stands, every generator emits exactly the gains it emits today. It also sets which gains apply when, randomised gains during PPO training and each design's exact gains during the CMA-ES evaluation, behind a runner switch that defaults to today's behaviour. Contact phase gain schedules, gains in the policy's observation, and friction identification are left to later work and discussed where they arise.

## 2. Literature Survey

The survey covers four questions, how published work sets a servo's gains from a natural frequency, how actuators are identified so that the inertia and friction are known, how gains have been searched or learned alongside a design or a policy, and what constrains the search. Each work has its own sub-section giving the method, the architecture, the mathematics, the assumptions, how anything is optimised, and the relevance here. Where only an abstract or a listing was retrieved, the sub-section says so.

### 2.1 Setting gains from a natural frequency

#### 2.1.1 BeyondMimic, Liao and colleagues

- Method and architecture. A humanoid motion tracking controller trained in simulation with a joint position action and an onboard PD loop [1]. The gains are fixed before training by a heuristic in its supplementary section S1.
- Mathematics. For each joint $j$, $k_{p,j} = I_j\omega^2$ and $k_{d,j} = 2I_j\zeta\omega$, where $I_j = k_{g,j}^2 I_{\mathrm{rotor},j}$ is the reflected rotor inertia, $k_g$ the gear ratio, $\omega$ the natural frequency and $\zeta$ the damping ratio.
- Assumptions. Only the armature is used, because the inertia of the subtree changes with configuration. The frequency is 10 Hz and $\zeta = 2$, overdamped on purpose because the armature underestimates the true inertia.
- Optimisation. An ablation over the natural frequency, in which 10 Hz gave the best global tracking while higher frequencies cut local error but caused torque overshoot and gearbox noise.
- Relevance. The source of the requested default of 10 Hz, and of the convention mjlab ships (`/ws/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py:112-123`). Section 3.2 shows the armature-only reference does not carry over to the TRON1 biped's heavy legs.

#### 2.1.2 Isaac Sim gain tuner and Isaac Lab's natural frequency drive

- Method and architecture. Isaac Sim's gain tuner sets a joint drive from a target natural frequency and damping ratio [2]. Isaac Lab's URDF importer offers the same through `NaturalFrequencyGainsCfg` (`IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py:42-68`).
- Mathematics. $\omega_n = \sqrt{K_p/m}$ and $\zeta = K_d/(2m\omega_n)$, with $m$ the joint inertia from the masses on both sides of the joint, evaluated at the home configuration because it varies with pose.
- Assumptions. The drive is PhysX's implicit joint drive, which is unconditionally stable, so no step bound appears. Isaac Lab's importer default damping ratio is 0.005, and its natural frequency has no default.
- Optimisation. None, the values are user inputs. The Isaac Sim tuning tutorial targets a damping ratio of 0.7 to 1.0 [4].
- Relevance. The source of the damping ratio band around which the search range of section 3.5 is centred. Its implicit drive assumption does not hold for `IdentifiedActuator`.

#### 2.1.3 ZEST, Sleiman and colleagues

- Method and architecture. Motion transfer for athletic robot control, whose abstract states a procedure for selecting joint level gains from approximate analytical armature values [5]. Only the abstract was retrieved.
- Mathematics, assumptions and optimisation. Not retrieved.
- Relevance. A second instance of armature-based gain selection in recent work.

### 2.2 Identifying the actuator

#### 2.2.1 Tan and colleagues, Sim-to-Real

- Method and architecture. Quadruped locomotion learned in simulation and transferred to the Minitaur [6].
- Mathematics. An ideal DC motor, $\tau = K_t I$ and $I = (V_{pwm} - K_t\dot q)/R$, with the PD law setting the PWM duty, $V_{pwm} = V(k_p(\bar q - q) + k_d(\dot{\bar q} - \dot q))$, and a piecewise linear torque current curve for saturation.
- Assumptions. $K_t$ and $R$ from the specification, link inertia estimated from shape and mass at uniform density.
- Optimisation. No gain search. Friction came from dedicated tests, and latency from a one step PWM spike, about 3 ms in the servo and 15 to 19 ms on the companion computer.
- Relevance. Without the actuator model or the latency, the learned controllers failed on the robot. The delay this workspace does not know is a known cause of transfer failure.

#### 2.2.2 Hwangbo and colleagues, learned actuator model

- Method and architecture. One small network per series elastic drive of ANYmal, used in simulation in place of an analytical model [7].
- Mathematics. The network maps the position error and velocity at $t$, $t - 0.01$ and $t - 0.02$ s to torque.
- Assumptions. Each drive independent of the others, and logged torque available.
- Optimisation. Supervised regression on over a million samples gathered in under four minutes from sinusoidal foot motion at 1 to 25 Hz with manual disturbance. Validation error was 0.74 Nm against 3.55 Nm for the ideal model, and the authors found that the excitation must cover a wide range of frequencies.
- Relevance. An analytical model tuned for more than a week failed on unmodelled delay. A learned model belongs to one physical drive and cannot serve a catalogue entry the search has not yet built.

#### 2.2.3 Berkeley Humanoid, Liao and colleagues

- Method and architecture. A 16 kg humanoid with quasi direct drive actuators of 9 to 1 gear ratio and a simple RL controller [8]. This repository's `IdentifiedActuator` comes from its code (`actuator_pd.py:1`).
- Mathematics. Rotor inertia from CAD added as armature to the mass matrix diagonal, and a torque source without delay justified by a 1 kHz torque bandwidth.
- Assumptions. The friction form is not stated in the paper, which says only that simple per actuator experiments characterised friction and that the torque current mapping remains to be identified.
- Optimisation. None of the gains, which are deliberately not randomised. Friction is randomised by 0.9 to 1.1 and armature by 1.0 to 1.05.
- Relevance. The provenance of the actuator class, and evidence that its friction terms were not identified by a stated procedure.

#### 2.2.4 Grandia and colleagues, a bipedal robotic character

- Method and architecture. A Disney biped trained in Isaac Gym with first principles actuator models identified per actuator on a bench that measures output torque [9].
- Mathematics. $\tau = \mathrm{clamp}_{[\underline\tau(\dot q), \overline\tau(\dot q)]}(k_P(a - \tilde q) - k_D\dot q) - \tau_f$ with $\tau_f = \mu_s\tanh(\dot q/\dot q_s) + \mu_d\dot q$, a flat then linearly falling torque speed limit, a backlash term $0.5\,b\tanh(\tau_m/\tau_b)$, speed dependent encoder noise and reflected inertia as armature.
- Assumptions. Each actuator identified alone, the identified values randomised within their observed range.
- Optimisation. Gains, limits, friction, backlash, noise and inertia were identified, and their Table VI gives values for the Unitree A1 and Go1 drives. The excitation and fitting method are not described.
- Relevance. The most complete published statement of the friction model this repository runs, and the model section 3.10 analyses.

#### 2.2.5 Bjelonic, Tischhauser and Hutter, PACE

- Method and architecture. A three stage identification. A bench drive under 0.1 to 10 Hz chirps fits armature, viscous damping and Coulomb friction. The full robot in air, base fixed and joints chirped together for 20 to 40 s, fits per joint armature, damping, friction and bias with one global delay. On ground policies then use those values without dynamics randomisation [3].
- Mathematics. The joint obeys $I_a\ddot q + d\dot q = \mathrm{sat}(P_\tau(\hat q - q + \tilde q_b) - D_\tau\dot q + \tau_{\mathrm{comp}}) + \tau_f$, and linearised with the delay on the command $H_q(s) = e^{-sT_d}P_\tau/(I_a s^2 + (d + D_\tau)s + P_\tau)$. Scaling $\{I_a, d, P_\tau, D_\tau\}$ by a common factor leaves the trajectories unchanged.
- Assumptions. Tight current control, no contacts during identification, gains and saturation limits known.
- Optimisation. CMA-ES over about 49 parameters normalised to $[-1, 1]$, starting at zero mean with $\sigma = 0.5$, minimising mean squared joint position error over 4096 parallel replays. Gains were chosen low and conservative, and not by a stated procedure.
- Relevance. Its linear model is the joint model of section 3.1, and its stance observation is the reason section 3.2 references the swing inertia. Its scaling result shows that only $\omega_n$ and $\zeta$ are observable, which is the case for searching in those coordinates.

#### 2.2.6 SPI-Active, Sobanbabu and colleagues

- Method and architecture. Simulator parameters fitted by massively parallel sampling against real trajectories, then refined using commands to a pretrained policy chosen to maximise Fisher information [10]. Only the abstract and listing were retrieved.
- Optimisation. Sampling based search, reported 42 to 63 per cent better than baselines.
- Relevance. The route by which the unknown friction and delay could be identified later without a torque sensor.

#### 2.2.7 Yu, Kumar, Turk and Liu, Sim-to-Real Transfer for Biped Locomotion

- Method and architecture. Identifies only a parameter range before training, trains a policy over a latent projection of the parameters, then selects the latent by Bayesian optimisation on the Darwin OP2 [11]. Only the abstract and listing were retrieved.
- Relevance. A route that tolerates unidentified parameters by searching over them on hardware, which is what the present plan does in simulation with $f_n$ and $\zeta$.

#### 2.2.8 Masuda and Takahashi, gear driven humanoid

- Method and architecture. An actuator model for high reduction gears and an identification that uses the rewards of failed transfers on the ROBOTIS OP3 [12]. Only the abstract was retrieved.
- Relevance. States that a slight identification error can fail a transfer, which motivates the friction work of section 3.10.

#### 2.2.9 Wensing, Kim and Slotine, physically consistent inertial parameters

- Method and architecture. Link inertial parameter identification constrained by linear matrix inequalities, solved by semidefinite programming and applied to a leg of MIT Cheetah 3 [13]. Only the abstract was retrieved.
- Relevance. The link inertias here come from the generator's own mass model, so no identification is needed. This is the method to use if hardware link inertias must ever be measured.

### 2.3 Gains as design variables

#### 2.3.1 Digumarti and colleagues, concurrent design and control

- Method and architecture. StarlETH's body and leg dimensions optimised together with the parameters of a model based locomotion controller, for trotting and bounding speed [14].
- Mathematics. A weighted sum of costs, $\theta^* = \arg\min_\theta \sum_i w_i C_i(\theta, q, \dot q, \tau)$.
- Assumptions. The robot's existing actuators retained, and the swing phase tracked by a PD law.
- Optimisation. CMA-ES over 31 parameters, 6 of them design, needing about 3500 rollouts with the design free. The authors state that the actuator gains "were not tuned in the previous work, but need to be considered in the optimization when the inertial parameters of the legs change". The extracted text does not state how they were parameterised.
- Relevance. The closest precedent, and the plain statement of why gains must follow the link inertia.

#### 2.3.2 Bohlinger and Peters, Shape Your Body

- Method and architecture. A multi embodiment policy trained over randomised designs, whose frozen value function supplies gradients for searching more than 1100 embodiment parameters [15].
- Mathematics. The P and D gains and the action scale are linear multiplicative factors of plus or minus 50 per cent on their nominal values, also scaled with body size, and are observed by the policy.
- Assumptions. A design curriculum from the nominal design to the full space, the design fixed per episode.
- Optimisation. Gradient ascent of the value over design parameters. On the MIT Humanoid, copying only the optimised gains raised return from 5.8 to 12.5. On Golem, the search lowered P and raised D, which changes the damping ratio.
- Relevance. The gains carry much of the attainable improvement, and the policy observed them. The present plan does not add them to the observation, see section 4.

#### 2.3.3 Chen, He and Ciocarlie, Hardware as Policy

- Method and architecture. Hardware parameters written as a differentiable computational graph and trained with the policy [16].
- Mathematics. Parallel springs under Hooke's law and a current torque motor model, the stiffnesses $k_1 \dots k_n$ learned alongside the network weights.
- Optimisation. PPO and CMA-ES compared on a mass spring task and an underactuated hand.
- Relevance. The passive analogue of searching a stiffness, with CMA-ES as one of the optimisers.

#### 2.3.4 Fadini and colleagues, simulation aided co-design

- Method and architecture. Bi-level. A genetic algorithm searches hardware and control parameters, each candidate scored by average tracking over perturbed simulations that model friction compensation and bandwidth limits [17]. Only the abstract was retrieved.
- Mathematics. The tracking controller uses the locally optimal feedback gains of differential dynamic programming.
- Relevance. The nearest published co-design whose gains are computed from a model rather than searched, and which prefers lighter, more transparent hardware.

#### 2.3.5 Bravo-Palacios, Li and Wensing, robust co-design

- Method and architecture. Morphology, a nominal trajectory and a feedback policy co-optimised by ADMM under stochastic terrain, for parallel elastic actuation [18]. Only listing records were retrieved.
- Relevance. Feedback gains treated as co-design variables alongside the body.

### 2.4 Gains emitted by the policy

#### 2.4.1 Bogdanovic, Khadiv and Righetti, variable impedance

- Method and architecture. The policy outputs joint targets and $K_p$ per joint, with $K_d$ derived from it [19].
- Mathematics. $\tau = K_p(q_{des} - q) - K_d\dot q$ with $K_d$ varied with $\sqrt{K_p}$, which is a fixed damping ratio for a fixed inertia.
- Optimisation. Policy gradient. On a hopper, variable gains beat every fixed gain and reached $K_p = 10$ where a fixed $K_p \ge 6$ was unstable on hardware.
- Relevance. Holding the damping ratio while moving the stiffness is the published default.

#### 2.4.2 VICES, Martín-Martín and colleagues

- Method and architecture. Operational space impedance gains as policy outputs for contact rich manipulation [20]. Stiffness and damping are separate outputs per axis. The robosuite implementation also offers a `variable_kp` mode in which damping is set to keep the controller critically damped [21].
- Relevance. Both independent and ratio-tied damping appear in practice.

#### 2.4.3 Spoljaric, Yan and Lee, variable stiffness locomotion

- Method and architecture. A quadruped policy outputs joint targets and stiffness, grouped per joint, per leg or hybrid [22].
- Mathematics. $K_d = 0.2\sqrt{K_p}$ with $K_p \in [20, 60]$.
- Optimisation. PPO. Per leg stiffness won on velocity tracking and push recovery, the hybrid grouping on energy.
- Relevance. Per group gains, the grouping this plan uses.

#### 2.4.4 Radosavovic and colleagues, and learning humanoid locomotion over challenging terrain

- Method and architecture. Humanoid policies whose action carries PD targets and predicted PD gains, for 8 leg joints in the first [23] and at 50 Hz under a 2000 Hz PD loop in the second [24]. The parameterisation of the gains was not retrieved.
- Relevance. Policy-set gains at scale, an alternative to searching them with the design.

#### 2.4.5 GainAdaptor, Kim, Kwon and Kim

- Method and architecture. A second actor tunes joint PD gains by ground condition on a Unitree Go1, with energy efficiency as a stated aim [25]. Only the abstract was retrieved.
- Relevance. Gains and efficiency are linked, which supports adding energy to the fitness in later work.

#### 2.4.6 Roveda and colleagues, model based variable impedance

- Method and architecture. An ensemble of networks learns the interaction dynamics, and a cross entropy MPC updates the impedance online [26].
- Mathematics. Stiffness in $[500, 5000]$ N/m and damping ratio in $[0.1, 0.9]$ on the vertical axis, the other axes fixed at 2500 N/m and 0.7.
- Relevance. The one surveyed work that optimises a damping ratio directly.

#### 2.4.7 Hammoud, Khadiv and Righetti, risk sensitive impedance

- Method and architecture. Time varying stiffness and damping schedules produced as the feedback gains of a risk sensitive optimal controller, with damping rising ahead of an impact [27]. Only the abstract was retrieved.
- Relevance. The published answer to stance and swing needing different gains, left to later work in section 4.

### 2.5 How the gains shape learning

#### 2.5.1 Tune to Learn, Bronars, Park and Agrawal

- Method and architecture. A controlled sweep of joint gains over two orders of magnitude in $K_p$ and $K_d$, under behaviour cloning, reinforcement learning and sim-to-real transfer [28].
- Mathematics. Read through $\omega_n = \sqrt{K_p/m}$ and $\zeta = K_d/(2\sqrt{mK_p})$.
- Optimisation. Reinforcement learning found policies in every regime once its hyperparameters were retuned. In transfer, stiff overdamped gains gave the lowest identification error and the worst zero shot result through high frequency oscillation, jitter failures falling from 21.8 to 5.0 per cent as the policy rate fell from 100 to 10 Hz.
- Relevance. Within the stable region, the gains should be judged on efficiency, robustness and transfer, which a search scored on the policy's own return measures.

#### 2.5.2 Chakraborty and colleagues, actuator dynamics curricula

- Method and architecture. A curriculum anneals joint stiffness from high to the identified value as episode length grows, for a Spot handstand [29].
- Mathematics. $K(t) = K_0 + \beta(t)^p(K^* - K_0)$, with $B_i = 2\sqrt{K(t)M_i}$ recomputed every iteration so that $\zeta = 1$ while $\omega_n = \sqrt{K/M}$ moves. A higher closed loop natural frequency at critical damping enlarges the finite horizon viability kernel of a cart-pole.
- Assumptions. One effective inertia per joint, contacts as external torque.
- Relevance. The one surveyed work that moves $\omega_n$ at fixed $\zeta$ explicitly, and an argument against a low upper bound on $f_n$.

#### 2.5.3 Guan and colleagues, impedance matching

- Method and architecture. A quadruped running jump learned by RL, with simulation gains matched to hardware [30].
- Relevance. The reflected rotor inertia shifts the best fit gains, so the armature belongs in the inertia, as section 3.2 uses it.

### 2.6 Tuning on hardware without a model

#### 2.6.1 Åström and Hägglund, relay auto-tuning

- Method and architecture. A relay in the loop produces a limit cycle whose amplitude and period estimate the critical gain and frequency, and a PID controller is tuned to phase and gain margins from that point [31]. Only the abstract and records were retrieved.
- Relevance. The model free procedure for one physical joint with nothing identified.

#### 2.6.2 Berkenkamp, Schoellig and Krause, SafeOpt

- Method and architecture. Bayesian optimisation of control parameters on hardware that only tries parameters whose performance is above a safety threshold with high probability [32]. Only the abstract was retrieved.
- Relevance. A safe hardware counterpart of the search proposed here.

#### 2.6.3 Marco and colleagues, automatic LQR tuning

- Method and architecture. Bayesian optimisation over the weights of an LQR cost rather than over the gain matrix [33].
- Relevance. A precedent for searching a physically meaningful reparameterisation of the gains, which is what $f_n$ and $\zeta$ are.

### 2.7 Stability of a sampled servo

#### 2.7.1 Colgate and Schenkel, passivity of sampled data systems

- Method and architecture. A necessary and sufficient passivity condition for a virtual wall rendered by a sampled controller [34].
- Mathematics. As cited by secondary sources, in the paper's own symbols, which differ from those of section 3, $b > KT/2 + |B|$, relating the physical damping $b$ to the rendered stiffness $K$, damping $B$ and sample period $T$. The original statement was not retrieved.
- Relevance. The sampled data counterpart of the explicit integration bound of section 3.3.

### 2.8 Encoding variables for CMA-ES

Hansen's guidance is to rescale every variable to similar sensitivity, and to encode a positive variable as $10^x$ only where its typical ratio of upper to lower value exceeds 100 [35]. The ranges of section 3.4 span ratios of 2.3 and 1.8, so the existing linear unit cube encoding of `_denormalise` (`usd_generator.py:985-991`) serves unchanged.

### 2.9 Synthesis

No surveyed co-design searches a servo's $f_n$ and $\zeta$. Searched gains appear raw or as scales [14] [15], and policy-emitted gains usually tie damping to $\sqrt{k_p}$ [19] [22], which is the $\zeta$ form at a fixed inertia. The natural frequency form is the standard way of setting gains [1] [2], and the one work that moves $\omega_n$ alone holds $\zeta$ fixed [29]. This plan therefore combines the setting rule with the co-design search, as Digumarti's remark asks, using the inertia each design actually presents.

## 3. PD Gain Optimisation

### 3.1 The joint model and the gain map

Take one joint with angle $q$ [rad], commanded target $q^*$ [rad] and total inertia $I$ [kg m²] about its axis, the armature plus the links it moves. The servo applies $\tau = k_p(q^* - q) - k_d\dot q$, where $k_p$ is the stiffness [N m/rad] and $k_d$ the damping [N m s/rad], the form `IdealPDActuator` computes with zero target velocity (`IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:190-195`). Ignoring friction, saturation and delay, Newton's law gives

$$I\ddot q + k_d\dot q + k_p q = k_p q^*.$$

Dividing by $I$ and writing the standard second order form $\ddot q + 2\zeta\omega_n\dot q + \omega_n^2 q = \omega_n^2 q^*$, where $\omega_n$ is the natural frequency [rad/s] and $\zeta$ the damping ratio [-], and matching coefficients gives

$$\omega_n = \sqrt{k_p/I}, \qquad \zeta = \frac{k_d}{2\sqrt{k_p I}},$$

and, inverting, with $f_n = \omega_n/2\pi$ in Hz,

$$k_p = I(2\pi f_n)^2, \qquad k_d = 2\zeta I(2\pi f_n).$$

$\zeta$ alone sets the shape of the response, overshoot $e^{-\pi\zeta/\sqrt{1-\zeta^2}}$ below 1 and none above. $f_n$ sets its speed. Their product $\zeta\omega_n = k_d/2I$ is the decay rate and is the quantity the stability bound of section 3.3 limits. Because $k_p$ and $k_d$ scale with $I$, a search held at fixed $(f_n, \zeta)$ hands each design gains proportional to its own inertia, which is the property Digumarti and colleagues asked for [14].

The same joint meets a different inertia at every pose. At an inertia $I'$, the same gains give $\omega_n' = \omega_n\sqrt{I/I'}$ and $\zeta' = \zeta\sqrt{I/I'}$, so a joint tuned against one inertia is faster and better damped against a smaller one, and slower and less damped against a larger one.

### 3.2 Which inertia

The inertia a servo drives changes with the pose of the joints below it, and it changes again according to whether the other joints are held or free to recoil. This section derives both effects, measures them over the pose space of both families, and then chooses the reference inertia against which $f_n$ and $\zeta$ are defined. Every symbol is listed in the notation chart at the end of the document.

#### 3.2.1 How the pose moves a joint's inertia

The robot. The robot has a floating base and a set $\mathcal{H}$ of $N$ hinges, indexed by $j$ and $l$, eight on the biped and twelve on the quadruped. Its pose is the vector of hinge angles $q = (q_l)_{l \in \mathcal{H}} \in \mathbb{R}^N$. Its rigid bodies are indexed by $\nu$. Body $\nu$ has mass $m_\nu$, centre of mass at the world point $\bar p_\nu$, and inertia tensor $I_\nu$ about its centre of mass in its own inertial frame, whose orientation in the world is the rotation $R_\nu$, so that $R_\nu I_\nu R_\nu^\top$ is the same tensor in world axes. Hinge $j$ turns about the unit vector $\hat a_j$ through the world point $p_j$, and $A_j$ is the armature, the reflected rotor inertia, that PhysX adds to it. Every position, orientation and axis depends on the pose $q$.

With the base twist $v_B \in \mathbb{R}^6$ and the hinge rates $\dot q$ collected into the generalised velocity $v = (v_B, \dot q) \in \mathbb{R}^{6+N}$, the kinetic energy of the robot is $\frac{1}{2} v^\top M(q)\,v$. $M(q)$ is the joint space mass matrix [36], symmetric and positive definite, of size $(6 + N) \times (6 + N)$, and it is formed from the body Jacobians $J^v_\nu$ and $J^\omega_\nu$, which map $v$ to body $\nu$'s centre of mass velocity and angular velocity,

$$M = \sum_\nu \Big(m_\nu\,J^{v\top}_\nu J^v_\nu + J^{\omega\top}_\nu R_\nu I_\nu R_\nu^\top J^\omega_\nu\Big) + \mathrm{diag}(0_6, A),$$

with $A$ the diagonal matrix of armatures. The column of hinge $l$ in the two Jacobians is $\hat a_l \times (\bar p_\nu - p_l)$ and $\hat a_l$ for every body $\nu$ below hinge $l$, and zero for every other body. Ignoring gravity and the terms quadratic in velocity, which section 3.3 also sets aside, the equation of motion is $M\dot v = (0_6, \tau)$, where $\tau \in \mathbb{R}^N$ are the hinge torques and the base, which no actuator drives, receives none. `inertia_analysis.py:154-177` builds $M$ this way, and MuJoCo's `mj_fullM` confirms it (section 3.2.2).

The locked inertia. Suppose a torque $\tau_j$ acts at hinge $j$ while every other hinge and the base are held, $\dot v_l = 0$ for every other coordinate. Row $j$ of the equation of motion reduces to $M_{jj}\ddot q_j = \tau_j$, so the inertia hinge $j$ meets is the diagonal entry $M_{jj}$. Substituting the Jacobian columns into $M$ gives it body by body,

$$I^{\mathrm{lock}}_j(q) = M_{jj} = A_j + \sum_{\nu\,\in\,\mathcal{B}_j}\Big[\hat a_j^\top R_\nu I_\nu R_\nu^\top\hat a_j + m_\nu\,\lVert P_j(\bar p_\nu - p_j)\rVert^2\Big], \qquad P_j = \mathbb{1} - \hat a_j\hat a_j^\top,$$

with $\mathcal{B}_j$ the set of bodies below hinge $j$ and $\mathbb{1}$ the $3 \times 3$ identity. The first term is the body's own inertia about an axis through its centre of mass parallel to $\hat a_j$. The second is the parallel axis term, the mass times the squared distance of its centre of mass from the axis. This is the expression of `../context/KScale.md` section 4. The plan calls it the locked inertia, the inertia the servo meets when every other joint is held rigid.

Please note that $P_j$ is a $3 \times 3$ matrix, not a number. $\hat a_j\hat a_j^\top$ is the outer product of the axis with itself, whose action on a vector $r$ is $\hat a_j(\hat a_j^\top r)$, the component of $r$ along the axis. The inner product $\hat a_j^\top\hat a_j$ is the number one, and $1 - \hat a_j^\top\hat a_j$ would indeed be zero, but that is not the expression. $P_j r = r - \hat a_j(\hat a_j^\top r)$ removes the component of $r$ along the axis and keeps the component perpendicular to it, so $P_j$ is the orthogonal projector onto the plane normal to $\hat a_j$, with $P_j\hat a_j = 0$, $P_j^2 = P_j$ and $P_j^\top = P_j$. Moving a point along the axis does not change its distance from the axis, so only the perpendicular component $P_j(\bar p_\nu - p_j)$ enters the parallel axis term, and its squared length is the squared perpendicular distance, equal to $\lVert\hat a_j \times (\bar p_\nu - p_j)\rVert^2$, the form the Jacobian column gives. The expression was validated against MuJoCo once more for this revision. Summing it body by body (`inertia_analysis.py:187-199`) and adding the armature reproduces the diagonal of `mj_fullM` over 20 random poses to a relative error of 7.3e-15 on the biped and 1.4e-15 on the quadruped (`:321-351`).

Turning hinge $j$ itself, or any hinge proximal to it, rotates the axis, its anchor and every body of $\mathcal{B}_j$ together. A rotation applied to all of them leaves both the projection of a tensor onto the axis and a perpendicular distance from it unchanged. The locked inertia of a hinge therefore depends only on the angles of the hinges below it. The hip angle changes the abduction's locked inertia, which sits above the hip, and leaves the knee's and the ankle's unchanged, which sit below it. The interactive page of section 3.2.2 shows exactly this behaviour.

The dependence of the inertia on one distal angle needs the matrix that rotates a vector about a unit axis $\hat a$ by an angle $\theta$. Split a vector $r$ into its component along the axis, $(\hat a^\top r)\hat a$, which the rotation leaves unchanged, and its perpendicular component $r_\perp = r - (\hat a^\top r)\hat a$, which turns through $\theta$ in the plane normal to the axis. That plane is spanned by $r_\perp$ and $\hat a \times r$, which has the same length and lies a quarter turn ahead, so the rotated vector is

$$R(\hat a, \theta)\,r = (\hat a^\top r)\hat a + \cos\theta\,r_\perp + \sin\theta\,(\hat a \times r).$$

Write $[\hat a]_\times$ for the skew symmetric matrix with $[\hat a]_\times r = \hat a \times r$. The vector triple product gives $[\hat a]_\times^2 r = \hat a \times (\hat a \times r) = (\hat a^\top r)\hat a - r = -r_\perp$. Substituting $r_\perp = -[\hat a]_\times^2 r$ and $(\hat a^\top r)\hat a = r + [\hat a]_\times^2 r$, and collecting terms,

$$R(\hat a, \theta) = \mathbb{1} + \sin\theta\,[\hat a]_\times + (1 - \cos\theta)\,[\hat a]_\times^2,$$

which is Rodrigues' formula [37]. Every entry of $R(\hat a, \theta)$ is therefore an affine combination of $1$, $\cos\theta$ and $\sin\theta$, a first order trigonometric polynomial in $\theta$.

Now fixing every hinge except one hinge $l$ below $j$, and turn $l$ through $\theta = q_l$. The bodies split into two sets. Those in $\mathcal{B}_j$ but not below $l$ do not move, and their terms are a constant. Those in $\mathcal{B}_l$, the bodies below $l$, rotate rigidly about the axis $\hat a_l$ through $p_l$ by $R_l(\theta) = R(\hat a_l, \theta)$, which itself does not move since $l$ is the only hinge turning. With $\bar p^0_\nu$ and $R^0_\nu$ the centre of mass and orientation of body $\nu$ at $\theta = 0$, and $r_\nu = \bar p^0_\nu - p_l$ its offset from the axis of $l$,

$$\bar p_\nu(\theta) = p_l + R_l(\theta)\,r_\nu, \qquad R_\nu(\theta) = R_l(\theta)\,R^0_\nu.$$

Substituting these into the two terms of $I^{\mathrm{lock}}_j$ for each $\nu \in \mathcal{B}_l$ gives the following.

1. The own term becomes $\hat a_j^\top R_l\,\bar I_\nu R_l^\top\hat a_j = (R_l^\top\hat a_j)^\top\bar I_\nu(R_l^\top\hat a_j)$, with $\bar I_\nu = R^0_\nu I_\nu R^{0\top}_\nu$ a constant tensor. The vector $R_l^\top\hat a_j$ is first order in $\cos\theta$ and $\sin\theta$, so this quadratic form is second order.
2. Write $d = P_j(p_l - p_j)$, the perpendicular offset of the axis of $l$ from the axis of $j$, constant because neither axis moves. The parallel term becomes $m_\nu\lVert d + P_jR_l r_\nu\rVert^2 = m_\nu\big(\lVert d\rVert^2 + 2d^\top P_jR_l r_\nu + r_\nu^\top R_l^\top P_jR_l r_\nu\big)$, using $P_j^\top P_j = P_j$. The middle term is first order, and the last is second order.
3. A second order trigonometric polynomial is a sum of the products $1$, $\cos\theta$, $\sin\theta$, $\cos^2\theta$, $\sin^2\theta$ and $\sin\theta\cos\theta$, and the double angle identities $\cos^2\theta = \frac{1}{2}(1 + \cos 2\theta)$, $\sin^2\theta = \frac{1}{2}(1 - \cos 2\theta)$ and $\sin\theta\cos\theta = \frac{(\sin 2\theta)}{2}$ turn the last three into harmonics of $2\theta$.

Summing over the bodies, the locked inertia is a second order trigonometric polynomial in the distal angle,

$$I^{\mathrm{lock}}_j(\theta) = h_0 + h_1\cos\theta + g_1\sin\theta + h_2\cos 2\theta + g_2\sin 2\theta,$$

whose five coefficients $h_0, h_1, g_1, h_2, g_2$ [kg m²] are constants set by the masses, inertias and geometry of the bodies and by the other hinge angles.

 When $\hat a_l = \pm\hat a_j$, the hip and the knee of both families for instance, the second harmonic vanishes. $R_l$ rotates about $\hat a_j$ itself, so $R_l^\top\hat a_j = \hat a_j$ and the own term is constant. $R_l$ also maps the plane normal to $\hat a_j$ onto itself, so it commutes with $P_j$, and $R_l^\top P_jR_l = P_j$ makes the last parallel term $r_\nu^\top P_j r_\nu$ constant. Only the middle term remains, which, since $P_j d = d$ and $P_j$ commutes with $R_l$, is $2m_\nu d^\top R_l P_j r_\nu$. Summed over $\mathcal{B}_l$, with $\bar m_l = \sum_{\nu\in\mathcal{B}_l}m_\nu$ the mass below $l$ and $\bar r_l = P_j\sum_{\nu\in\mathcal{B}_l}m_\nu r_\nu/\bar m_l$ the perpendicular offset of its centre of mass from the axis of $l$ at $\theta = 0$, it is $2\bar m_l\,d^\top R_l\bar r_l$. Both $d$ and $R_l\bar r_l$ lie in the plane normal to the axis, and $R_l$ turns $\bar r_l$ through $\theta$ within it, so $d^\top R_l\bar r_l = \lVert d\rVert\lVert\bar r_l\rVert\cos(\theta + \varphi)$, with $\varphi$ the angle from $d$ to $\bar r_l$ at $\theta = 0$, measured about the axis. Hence

$$I^{\mathrm{lock}}_j(\theta) = h_0 + 2\,\bar m_l\,\lVert d\rVert\,\lVert\bar r_l\rVert\cos(\theta + \varphi),$$

with $h_0$ collecting every constant. The inertia is largest at $\theta = -\varphi$, where the centre of mass below $l$ points away from the axis of $j$ along $d$, the hip with its leg straight, and smallest at $\theta = \pi - \varphi$, where it folds back towards that axis. A single sinusoid therefore has its minimum over a joint range either at a limit or at $\theta = \pi - \varphi$.

The abduction and the hip are perpendicular on both robots, so the second harmonic survives and minima inside the range become possible. The biped abduction's locked inertia is smallest at a hip angle of $-0.995$ rad with the knee and ankle at their limits, where the folded leg's centre of mass passes closest to the abduction axis. The first version of this plan evaluated each distal joint only at its two limits and at zero, which missed that interior minimum and overstated the abduction's minimum more than threefold, 0.0415 against 0.0121 kg m². The script fits both laws to machine precision (section 3.2.2).

Holding the other joints rigidly is itself an assumption. The servos are explicit, and the other joints respond to a torque at $j$ only through their own servos, which in the fast modes of section 3.3 do not hold them rigid. Suppose instead that a torque $\tau_j$ acts at hinge $j$ and every other coordinate, the base included, is free, receiving no generalised force. Partition $v$ into the coordinate $q_j$ and the remaining $5 + N$ coordinates, collected as $r$, and $M$ conformally into the scalar $M_{jj}$, the row $M_{jr}$, the column $M_{rj} = M_{jr}^\top$ and the block $M_{rr}$. The equation of motion splits into

$$M_{jj}\ddot q_j + M_{jr}\dot v_r = \tau_j, \qquad M_{rj}\ddot q_j + M_{rr}\dot v_r = 0.$$

The second gives the recoil of the rest of the robot, $\dot v_r = -M_{rr}^{-1}M_{rj}\ddot q_j$, which $M_{rr}$, a principal block of a positive definite matrix, allows to be solved. Substituting it into the first,

$$\big(M_{jj} - M_{jr}M_{rr}^{-1}M_{rj}\big)\ddot q_j = \tau_j.$$

The bracket is the inertia hinge $j$ meets with everything else free. The same solve read the other way, $\dot v = M^{-1}(0_6, \tau)$ with only $\tau_j$ nonzero, gives $\ddot q_j = (M^{-1})_{jj}\tau_j$, so the bracket, the Schur complement of $M_{rr}$ in $M$, equals $1/(M^{-1})_{jj}$. Since $M_{rr}^{-1}$ is positive definite, the subtracted term is non negative, and

$$I^{\mathrm{free}}_j = \frac{1}{(M^{-1})_{jj}} = M_{jj} - M_{jr}M_{rr}^{-1}M_{rj} \;\le\; M_{jj} = I^{\mathrm{lock}}_j.$$

The subtracted term is the recoil. A knee torque also swings the thigh about the hip and the trunk about its centre of mass, so the shank turns further for the same torque.

With every hinge torqued at once, the same solve gives the hinge accelerations $\ddot q = W\tau$, where

$$W(q) = (M(q)^{-1})_{\mathcal{H}\mathcal{H}} \in \mathbb{R}^{N\times N}$$

is the hinge block of the inverse mass matrix, symmetric and positive definite as a principal block of a positive definite matrix. Its diagonal entries are $W_{jj} = 1/I^{\mathrm{free}}_j$, and its off diagonal entry $W_{jl}$ is the acceleration of hinge $j$ per unit torque at hinge $l$, the inertial coupling of the two. $W$ is the inverse inertia the servos act through, and sections 3.2.3 and 3.3.7 rest on it.

A contact or a fixed base adds a constraint $G\dot v = 0$, with $G$ the constraint Jacobian, enforced by a constraint force $G^\top f$. Solving $M\dot v = (0_6, \tau) + G^\top f$ together with the constraint gives $f$, and the hinge accelerations become $\ddot q = W_G\tau$ with $W_G = W - W_{\mathcal{H}G}(GM^{-1}G^\top)^{-1}W_{\mathcal{H}G}^\top$, where $W_{\mathcal{H}G} = (M^{-1}G^\top)_{\mathcal{H}}$. The subtracted matrix is positive semidefinite, so $W_G \preceq W$ in the positive semidefinite order, $X \preceq Y$ meaning that $Y - X$ is positive semidefinite. The airborne, floating robot is therefore the worst case, and the planted leg of stance only adds inertia.

#### 3.2.2 The sweep

`scripts/analysis/inertia_analysis.py` computes these quantities for any MJCF built from hinges, with or without a free root joint. It reads the kinematic tree, the joint classes of the `<default>` elements and every `<inertial>` (`inertia_analysis.py:70-118`), forms $M$ from the body Jacobians with the armature on the hinge diagonal (`:154-177`), and returns the locked and free inertias, the row sums of $|W|$ and $W$ itself at any pose (`:180-184`). `--armature` replaces the document's armature by the catalogue's, which `apply_actuator_params` writes at runtime (`co_optimisation/co_optimisation/utils/respawn.py:243`), and `--group` aggregates the hinges of an actuator group in the printed table.

The sweep (`inertia_analysis.py:231-275`) takes the locked minimum on an exact grid of 13 points over each hinge's distal hinges, the only angles it depends on. The free and reference inertias of section 3.2.3 depend on every hinge through the base, so they are swept on a grid of 7 points over each limb with the others at zero, on 2000 random whole body poses, and refined by four coordinate passes over every hinge from the worst pose found for each joint. `--check` validates the mechanics four ways. Against MuJoCo's `mj_fullM` over 20 random poses, the hinge block of $M$ agrees to a relative error of 1.1e-15 on the biped and 1.4e-15 on the quadruped, the hinge block of $M^{-1}$ to 7.1e-15 and 7.5e-16, and the projection formula of section 3.2.1 summed body by body to 7.3e-15 and 1.4e-15 (`:321-351`). The trigonometric law of section 3.2.1 fits every pair of a hinge and a distal hinge on both robots to a relative residual below 1.4e-15, at degree 1 for the parallel pairs and degree 2 for the skew ones (`:354-375`). The dashboard's JavaScript port of the mechanics agrees with the Python to 2.4e-15 and 9.5e-16, run through node (`:378-399`).

The script writes `inertia_dashboard.html`, a self contained page with one slider per hinge. Its bars show the locked and free inertia of every hinge on a log scale, with ticks at the sweep's minimum locked inertia and at the reference inertia. A table breaks the selected hinge's locked inertia into each distal body's own term and parallel axis term, side and front views draw the pose with the selected axis, a menu loads the worst pose the sweep found for any hinge, and a readout evaluates the coupled stability test of section 3.3.7 at a chosen $f_n$ and $\zeta$. The pages for the two documents of this plan are [biped](../context/artefacts/w_eta_optimisation-2026-10-27/biped/inertia_dashboard.html) and [quadruped](../context/artefacts/w_eta_optimisation-2026-10-27/quadruped/inertia_dashboard.html), produced by

```bash
python3 -I scripts/analysis/inertia_analysis.py <run>/copt_usds/gen_0008/individual_0000.xml \
    --armature 'abad_.*=0.005589' --armature 'hip_.*=0.007614' --armature 'knee_.*=0.01215' --armature 'ankle_.*=0.005589' \
    --group 'abad=abad_.*' --group 'hip=hip_.*' --group 'knee=knee_.*' --group 'ankle=ankle_.*' --check --out <dir>
python3 -I scripts/analysis/inertia_analysis.py <run>/copt_usds/gen_0008/individual_0000.xml --armature '.*=0.01' \
    --group 'abad=.*_hip_joint' --group 'hip=.*_thigh_joint' --group 'knee=.*_calf_joint' --check --out <dir>
```

on generation 8, individual 0, of `copt_moral/2026-10-26_04-55-55` and `quadruped_copt/2026-10-26_04-55-55`, with each group's catalogue armature. The printed output is kept beside each page as `inertia_analysis.txt`.

| Family, group | Armature $A_j$ | $I^{\mathrm{lock}}_j$, nominal | $I^{\mathrm{lock}}_j$, min | $I^{\mathrm{lock}}_j$, max | $I^{\mathrm{free}}_j$, min | Reference $S_j$ | Fast estimate / sweep |
|---|---|---|---|---|---|---|---|
| biped abad | 0.0056 | 0.8321 | 0.0121 | 1.2246 | 0.0102 | 0.0088 | 1.017 |
| biped hip | 0.0076 | 0.8712 | 0.2631 | 1.2273 | 0.0607 | 0.0214 | 1.150 |
| biped knee | 0.0121 | 0.2730 | 0.2650 | 0.2737 | 0.0358 | 0.0184 | 1.017 |
| biped ankle | 0.0056 | 0.0085 | 0.0085 | 0.0085 | 0.0080 | 0.0074 | 1.000 |
| quadruped abad | 0.0100 | 0.0266 | 0.0177 | 0.0303 | 0.0143 | 0.0097 | 0.998 |
| quadruped hip | 0.0100 | 0.0227 | 0.0163 | 0.0251 | 0.0156 | 0.0129 | 1.001 |
| quadruped knee | 0.0100 | 0.0122 | 0.0122 | 0.0122 | 0.0115 | 0.0095 | 1.002 |

Inertias in kg m², armature included. The last column is the per design estimate of section 3.8 E3 against the full sweep.

Three observations follow. The biped abduction's locked inertia spans a factor of 101 over its pose space, so no single locked value describes it. The biped knee's free inertia is 0.135 of its locked inertia, because the knee torque swings the heavy thigh and the trunk as well, which is the coupling section 3.3.7 shows to dominate the biped's stability. The quadruped's legs are light against their 0.01 kg m² armature, so in every group its locked, free and reference inertias stay within a factor of 3.1 of one another. The biped's swing ankle is almost entirely rotor, as the KScale ankle is (`../context/KScale.md` section 3.1).

#### 3.2.3 The choice

Five candidates exist. The armature alone is BeyondMimic's reference [1]. It ignores more than 99 per cent of the biped hip's nominal inertia and does not change when a link length changes, so it cannot make the gains follow the design, which is the purpose of this plan. The nominal locked inertia is what the Isaac Sim tuner uses [2], and the servo meets far less than it at most poses. The stance inertia, the body carried through the planted leg (`../context/KScale.md` section 4.9), is larger still and is what `KScale.md` sizes its damping target against. The minimum locked inertia was this plan's first choice.

The plan adopts the fifth. For each hinge $j \in \mathcal{H}$ its reference inertia [kg m²] is

$$S_j = \frac{1}{\max_{q}\ \sum_{l\in\mathcal{H}} \lvert W_{jl}(q)\rvert},$$

where $q$ ranges over every pose within the joint limits, $W(q)$ is the inverse inertia matrix of section 3.2.1 at that pose, and the sum runs over every hinge $l$ of the robot, so that $\sum_l|W_{jl}|$ is the absolute row sum of row $j$ of $W$. $S$ denotes the diagonal matrix $\mathrm{diag}(S_1, \dots, S_N)$. Written out, $S_j$ is the reciprocal of the largest acceleration that any combination of unit torques, at most one per hinge, could produce at hinge $j$ at any pose.

The reason for this choice is the stability of the explicit servo. Section 3.3.7 shows that the servos of the robot are stable at every pose, for every gain in the search box and every mix of gains across groups, if

$$\lambda_{\max}\big(S\,W(q)\big) \le 1 \quad \text{at every pose } q,$$

where $\lambda_{\max}(X)$ denotes the largest eigenvalue of a matrix $X$ whose eigenvalues are real. The eigenvalues of $SW$ are real and positive, because $SW = S^{1/2}(S^{1/2}WS^{1/2})S^{-1/2}$ is similar to the symmetric positive definite matrix $S^{1/2}WS^{1/2}$, and similar matrices share their eigenvalues. Gershgorin's theorem states that every eigenvalue of a square matrix $X$ lies in at least one of the discs of the complex plane centred at a diagonal entry $X_{jj}$ with radius $\sum_{l\neq j}|X_{jl}|$, the absolute sum of the rest of its row [38]. For $X = SW$, row $j$ is $S_j$ times row $j$ of $W$, and $W_{jj} > 0$, so every eigenvalue is at most $\max_j S_j\sum_l|W_{jl}|$. The definition of $S_j$ makes every such row sum at most one at every pose, which gives the condition. The script measures $\lambda_{\max}(SW)$ at 0.954 on the biped and 0.905 on the quadruped with this reference, close to the bound. With the minimum locked inertia in its place, it reaches 10.06 on the biped and 1.44 on the quadruped, so that choice would have left the biped's fast modes ten times stiffer and better damped than the box allows.

Under this reference, $f_n$ and $\zeta$ are ceilings. When every group shares them, each closed loop mode of the coupled robot has a natural frequency of at most $f_n$ and a damping ratio of at most $\zeta$, at every pose. At poses and in modes where the robot presents more inertia, the response is slower and less damped, by the factor $\sqrt{\mu_i}$ of the mode's eigenvalue $\mu_i$ of $SW$, defined in section 3.3.7, in the way section 3.1 describes for one joint. The reference follows each design's link lengths and each categorical actuator pick, since both enter $M$. The right side of an emitted biped mirrors the left and the four quadruped legs are identical about their own axes, so the joint of the first limb represents its group.

### 3.3 The stability constraints Isaac Lab imposes

#### 3.3.1 How Isaac Lab steps the servo

`IdentifiedActuator` derives from `DCMotor` and so from `IdealPDActuator`. It computes the torque in Python and hands it to PhysX as an effort, clearing the position and velocity targets (`actuator_pd.py:37-39`), and Isaac Lab writes zero PhysX drive stiffness and damping for such an actuator (`IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py:1760-1769`). Isaac Lab evaluates it once per physics step inside the decimation loop (`manager_based_rl_env.py:182-189`). The torque computed from the state at the start of a step is therefore held constant for the whole step of $\Delta t$ = 0.005 s, and PhysX advances the joint under it. The policy target $q^*$ is held constant over the four physics steps of one control step.

The servo is therefore an explicit element. Its torque responds to the state of the previous step and not the state the step ends in. Unlike PhysX's implicit drive, which is unconditionally stable, it is stable only for gains small enough against the inertia.

#### 3.3.2 The state transition equation

Sections 3.3.2 to 3.3.6 treat one joint against one inertia $I$, and section 3.3.7 extends the result to the coupled robot. For one joint the hinge subscript is dropped, and a quantity at the start of physics step $k$ is written with the step in brackets, $q[k]$, so that subscripts remain free for hinges and modes.

The state is written in the error $e[k] = q[k] - q^*$ rather than in the position $q[k]$. `../context/KScale.md` section 11.1.1 writes the same update in $q$ with the target at the origin, $q^* = 0$, which is the same choice in other words. Since $q^*$ is constant over a physics step, $e[k+1] - e[k] = q[k+1] - q[k]$, and the torque $k_p(q^* - q[k])$ is $-k_p e[k]$, so the two forms give the same matrix. The error form makes three things explicit. The equilibrium the joint should settle at is $e = 0$, which is $q = q^*$ for any target and not only for a target of zero. Stability is the statement that the error decays, whatever the target. And a new policy action changes $q^*$ at the start of a control step, which shifts $e$ by the change in target but leaves the transition matrix untouched, so the matrix derived below governs the response between any two actions. The joint velocity needs no such shift, the servo's velocity target being zero.

With friction linearised as a viscous slope (section 3.10), the torque the servo applies over step $k$ is

$$\tau[k] = -k_p e[k] - c\,\dot q[k], \qquad c = k_d + c_f,$$

where $c$ [N m s/rad] is the total damping slope, $k_d$ plus the friction slope $c_f$. PhysX's update is semi-implicit, as `../context/KScale.md` section 11.1.1 records. The velocity advances first under the held torque, and the position then advances with the new velocity,

$$\dot q[k+1] = \dot q[k] + \Delta t\frac{\tau[k]}{I}, \qquad e[k+1] = e[k] + \Delta t\,\dot q[k+1].$$

Substituting $\tau[k]$ and grouping terms introduces two dimensionless numbers,

$$a = \frac{k_p\Delta t^2}{I}, \qquad b = \frac{c\,\Delta t}{I}.$$

$a$ is the stiffness per step, the square of the natural frequency measured in radians per physics step. $b$ is the damping per step, the fraction of the joint's velocity that the damping removes in one step. The velocity update becomes

$$\dot q[k+1] = -\frac{a}{\Delta t}e[k] + (1 - b)\,\dot q[k],$$

and the position update, after substituting it,

$$e[k+1] = e[k] + \Delta t\left(-\frac{a}{\Delta t}e[k] + (1 - b)\dot q[k]\right) = (1 - a)\,e[k] + \Delta t(1 - b)\,\dot q[k].$$

In matrix form, with the state $x[k] = (e[k], \dot q[k])^\top$,

$$x[k+1] = \Phi\,x[k], \qquad \Phi = \begin{pmatrix} 1 - a & \Delta t(1 - b) \\ -a/\Delta t & 1 - b \end{pmatrix}.$$

$\Phi$ is the state transition matrix, the matrix of `../context/KScale.md` section 11.1.1. Every free response of the joint is $x[k] = \Phi^k x[0]$.

#### 3.3.3 Why the eigenvalues decide stability

The joint is stable when $x[k] \to 0$ from every initial state $x[0]$, which is a statement about $\Phi^k$ as $k \to \infty$, a limit test exactly as it appears. Its answer is carried by the eigenvalues $\lambda_1, \lambda_2$ of $\Phi$, the roots of $\det(\lambda\mathbb{1} - \Phi) = 0$, and not by its eigenvectors, which only fix the directions along which each eigenvalue acts.

Suppose first that the eigenvalues differ, so that $\Phi$ has two independent eigenvectors $\psi_1, \psi_2$ with $\Phi\psi_i = \lambda_i\psi_i$. Collecting them as the columns of $\Psi$ gives $\Phi = \Psi\,\mathrm{diag}(\lambda_1, \lambda_2)\,\Psi^{-1}$. Powers then telescope, since every inner $\Psi^{-1}\Psi$ cancels,

$$\Phi^k = \Psi\,\mathrm{diag}(\lambda_1, \lambda_2)\,\Psi^{-1}\,\Psi\,\mathrm{diag}(\lambda_1, \lambda_2)\,\Psi^{-1}\cdots = \Psi\,\mathrm{diag}(\lambda_1^k, \lambda_2^k)\,\Psi^{-1}.$$

Expanding the initial state in the eigenvectors, $x[0] = \gamma_1\psi_1 + \gamma_2\psi_2$, where $\gamma_1$ and $\gamma_2$ are arbitrary constants fixed by the initial state and carry no significance of their own, each component is simply multiplied by its eigenvalue at every step,

$$x[k] = \gamma_1\lambda_1^k\psi_1 + \gamma_2\lambda_2^k\psi_2.$$

A complex $\lambda = |\lambda|e^{i\phi}$ has $|\lambda^k| = |\lambda|^k$, so each term shrinks or grows geometrically by $|\lambda_i|$ per step, and rotates by $\phi$ per step if the pair is complex, which is the oscillation of an underdamped joint. Three cases follow. If $|\lambda_1| < 1$ and $|\lambda_2| < 1$, then $\lambda_i^k \to 0$, so $\Phi^k \to 0$ and $x[k] \to 0$ from every $x[0]$. If some $|\lambda_i| > 1$, the component along $\psi_i$ grows as $|\lambda_i|^k$ from any $x[0]$ with $\gamma_i \ne 0$, which is almost every initial state, and the joint diverges. If $|\lambda_i| = 1$, that component neither grows nor decays, which is the boundary.

If the two eigenvalues coincide and $\Phi$ lacks a second eigenvector, $\Phi = \Psi Z\Psi^{-1}$ with the Jordan block $Z = \begin{pmatrix}\lambda & 1\\ 0 & \lambda\end{pmatrix}$, whose powers are $Z^k = \begin{pmatrix}\lambda^k & k\lambda^{k-1}\\ 0 & \lambda^k\end{pmatrix}$. The extra term $k\lambda^{k-1}$ still tends to zero when $|\lambda| < 1$, since a geometric decay overcomes any polynomial growth. Both cases are summarised by the spectral radius $\rho(\Phi) = \max_i|\lambda_i|$ and Gelfand's formula $\lim_{k\to\infty}\lVert\Phi^k\rVert^{1/k} = \rho(\Phi)$, which hold for any square matrix [38]. $\Phi^k \to 0$ if and only if $\rho(\Phi) < 1$, and then $\lVert x[k]\rVert \le C(\rho(\Phi) + \epsilon)^k\lVert x[0]\rVert$ for any $\epsilon > 0$ and some constant $C$. The joint is therefore stable if and only if both eigenvalues lie strictly inside the unit circle of the complex plane, and $\rho(\Phi)$ is its decay factor per physics step. A residual of 2 per cent is reached after about $\ln 0.02/\ln\rho(\Phi)$ steps.

The circle is the discrete image of the continuous left half plane. A continuous mode $e^{st}$ sampled every $\Delta t$ becomes $\lambda^k$ with $\lambda = e^{s\Delta t}$, and $|\lambda| < 1$ exactly when $\mathrm{Re}\,s < 0$.

#### 3.3.4 The stability conditions

The trace of $\Phi$ is $\mathrm{tr}\,\Phi = 2 - a - b$ and its determinant is

$$\det\Phi = (1 - a)(1 - b) + a(1 - b) = 1 - b.$$

The eigenvalues are the roots of the characteristic polynomial

$$p(\lambda) = \det(\lambda\mathbb{1} - \Phi) = \lambda^2 - (2 - a - b)\lambda + (1 - b).$$

For a real quadratic $\lambda^2 + \alpha_1\lambda + \alpha_0$, both roots lie strictly inside the unit circle if and only if $|\alpha_0| < 1$, $p(1) > 0$ and $p(-1) > 0$. That is the Jury test for second order. The product of the roots is $\alpha_0$, so $|\alpha_0| < 1$ is necessary. The other two conditions require the upward opening parabola to be positive at $\lambda = \pm 1$, so that real roots can only lie between $-1$ and $1$, and complex roots then have modulus $\sqrt{\alpha_0} < 1$. Applied to $p$, with $\alpha_1 = -(2 - a - b)$ and $\alpha_0 = 1 - b$,

$$|\alpha_0| = |1 - b| < 1 \iff 0 < b < 2,$$

$$p(1) = 1 - (2 - a - b) + (1 - b) = a > 0,$$

$$p(-1) = 1 + (2 - a - b) + (1 - b) = 4 - a - 2b > 0.$$

The joint is therefore stable if and only if

$$a > 0, \qquad 0 < b < 2, \qquad a + 2b < 4.$$

Two checks confirm the result. The first recovers the continuous joint. A continuous pole $s$ maps to $\lambda = e^{s\Delta t} \approx 1 + s\Delta t$ when $|s\Delta t| \ll 1$, so substitute $\lambda = 1 + \varepsilon$ with $\varepsilon = s\Delta t$ into $p$ and expand,

$$p(1 + \varepsilon) = (1 + 2\varepsilon + \varepsilon^2) - (2 - a - b)(1 + \varepsilon) + (1 - b).$$

The constant terms give $1 - 2 + a + b + 1 - b = a$, the terms in $\varepsilon$ give $2\varepsilon - (2 - a - b)\varepsilon = (a + b)\varepsilon$, and the quadratic term is $\varepsilon^2$, so the identity

$$p(1 + \varepsilon) = \varepsilon^2 + (a + b)\,\varepsilon + a$$

holds exactly. Substituting $\varepsilon = s\Delta t$, $a = k_p\Delta t^2/I$ and $b = c\Delta t/I$,

$$p(1 + s\Delta t) = s^2\Delta t^2 + \left(\frac{k_p\Delta t^2}{I} + \frac{c\,\Delta t}{I}\right)s\Delta t + \frac{k_p\Delta t^2}{I} = \Delta t^2\left(s^2 + \frac{c + k_p\Delta t}{I}s + \frac{k_p}{I}\right).$$

The bracket is the continuous characteristic polynomial $Is^2 + cs + k_p$ of section 3.1, divided by $I$, with one extra damping term $k_p\Delta t$ that the scheme itself introduces. The only approximation is the map $\lambda \approx 1 + s\Delta t$, so for small steps the discrete roots are the continuous poles. The second check removes the spring. For $k_p = 0$, the conditions reduce to $0 < b < 2$, the damper analysed in section 3.3.5.

The stable region is downward closed. Write $\mathcal{T} = \{(a, b) : a > 0,\ 0 < b < 2,\ a + 2b < 4\}$, a triangle in the $(a, b)$ plane with the origin at one corner. If $(a, b) \in \mathcal{T}$ and $0 < \kappa \le 1$, then $(\kappa a, \kappa b) \in \mathcal{T}$, since $\kappa a > 0$, $0 < \kappa b \le b < 2$, and $\kappa a + 2\kappa b = \kappa(a + 2b) \le a + 2b < 4$. Geometrically, the segment from any stable point to the origin lies inside the triangle, because every condition bounds $a$ and $b$ from above by a line that the shrinking point never crosses. Shrinking both numbers by a common factor therefore never destabilises the joint. Section 3.3.6 shows that a larger inertia does exactly this.

#### 3.3.5 What $b > 1$ means, and why $b \le 1$ is kept as a margin

Take the damper alone, $k_p = 0$ and so $a = 0$. The velocity update becomes $\dot q[k+1] = (1 - b)\dot q[k]$.

- For $0 < b < 1$, the factor $1 - b$ lies between 0 and 1, and the velocity decays smoothly without changing sign, as a physical damper does.
- At $b = 1$, the velocity is removed in exactly one step.
- For $1 < b < 2$, the factor lies between $-1$ and 0. The damping removes more than the whole velocity in one step and reverses it, so the velocity alternates in sign every step while its magnitude still shrinks. That is an oscillation with a period of two physics steps, 0.01 s, a frequency of 100 Hz, which is the highest frequency a 200 Hz loop can represent, its Nyquist frequency. It is an artefact of the integration, not a physical motion.
- At $b \ge 2$, the magnitude grows and the joint diverges, bounded only by the effort limit, against which it chatters.

The alternating mode of $1 < b < 2$ is stable, but `../context/KScale.md` section 11.1.1 records that it reaches the observation and the action and torque rate penalties as noise the policy cannot act on, and so keeps $b \le 1$ as a design margin. This plan keeps the same margin for the part of $b$ the search controls, constraint C1 of section 3.3.8.

#### 3.3.6 The gains at $I_{\mathrm{ref}}$, and why $I_{\mathrm{ref}}$ cancels

For one joint, write $I_{\mathrm{ref}}$ for the inertia the gains are computed against. The plan sets the gains from the searched $f_n$ and $\zeta$, with $\omega_n = 2\pi f_n$,

$$k_p = I_{\mathrm{ref}}\,\omega_n^2, \qquad k_d = 2\zeta\,I_{\mathrm{ref}}\,\omega_n.$$

At a pose where the joint's true inertia is $I$, write $\kappa = I_{\mathrm{ref}}/I$. Substituting the gains into the definitions of $a$ and of the servo's share of $b$ gives

$$a = \frac{k_p\Delta t^2}{I} = \frac{I_{\mathrm{ref}}}{I}\,\omega_n^2\Delta t^2 = \kappa\,(\omega_n\Delta t)^2,$$

$$b_{\mathrm{servo}} = \frac{k_d\,\Delta t}{I} = \frac{I_{\mathrm{ref}}}{I}\,2\zeta\omega_n\Delta t = \kappa\,2\zeta\omega_n\Delta t.$$

The design enters only through the ratio $\kappa$, because the inertia that sized the gains reappears in the numerator of $a$ and $b$. At the pose where the joint meets its reference inertia, $I = I_{\mathrm{ref}}$, $\kappa = 1$, and the reference inertia cancels completely,

$$a_{\mathrm{ref}} = (2\pi f_n\Delta t)^2, \qquad b_{\mathrm{ref}} = 4\pi\zeta f_n\Delta t.$$

These depend on the search variables and the physics step alone.

At any other pose, $(a, b) = \kappa\,(a_{\mathrm{ref}}, b_{\mathrm{ref}})$, the point at $\kappa = 1$ shrunk towards the origin by $\kappa$. If the reference is the smallest inertia the joint ever meets, then $I \ge I_{\mathrm{ref}}$ and $0 < \kappa \le 1$ at every pose. By the downward closure of section 3.3.4, the joint is then stable at every pose as soon as $(a_{\mathrm{ref}}, b_{\mathrm{ref}})$ is stable, a condition on $(f_n, \zeta)$ alone, the same for every design. It also keeps the margin, since $b = \kappa\,b_{\mathrm{ref}} \le b_{\mathrm{ref}}$. The argument fails for a larger reference, the nominal or the stance inertia, since $\kappa$ then exceeds 1 at the poses where the joint meets less than its reference, and the worst case depends on the design. Against the nominal locked inertia, $\kappa$ reaches $0.8321/0.0121 = 69$ at the biped abduction.

For one joint, the smallest inertia is the minimum of its locked inertia. A robot is not one joint, and section 3.3.7 shows what replaces $\kappa$ when the servos of a leg act on one articulated body.

#### 3.3.7 The coupled robot

This section repeats the derivation of sections 3.3.2 to 3.3.6 for all $N$ hinges at once. It first builds the coupled transition matrix, then treats the case in which every group shares $f_n$ and $\zeta$, where the robot separates exactly into independent single joint problems, and then the general case in which the groups differ, where a sufficient condition is proved by an energy argument. The hinge errors and velocities are now vectors, $e[k], \dot q[k] \in \mathbb{R}^N$, and the gains are the diagonal matrices $K = \mathrm{diag}(k_{p,1}, \dots, k_{p,N})$ of stiffnesses and $D = \mathrm{diag}(c_1, \dots, c_N)$ of total damping slopes, each $c_j$ being the hinge's $k_d$ plus its friction slope.

The coupled transition matrix.

1. In swing, with gravity and the velocity product terms set aside as in section 3.3.2, the hinge accelerations under the torques $\tau$ are $\ddot q = W\tau$ (section 3.2.1). $W$ takes the place of $1/I$.
2. The servos apply $\tau[k] = -Ke[k] - D\dot q[k]$, held over the step.
3. The semi-implicit update is $\dot q[k+1] = \dot q[k] + \Delta t\,W\tau[k]$ and $e[k+1] = e[k] + \Delta t\,\dot q[k+1]$.
4. Substituting step 2 into the velocity update gives $\dot q[k+1] = -\Delta t\,WK\,e[k] + (\mathbb{1} - \Delta t\,WD)\,\dot q[k]$, and substituting that into the position update gives $e[k+1] = (\mathbb{1} - \Delta t^2WK)\,e[k] + \Delta t(\mathbb{1} - \Delta t\,WD)\,\dot q[k]$, with $\mathbb{1}$ now the $N \times N$ identity.
5. With the state $x[k] = (e[k], \dot q[k]) \in \mathbb{R}^{2N}$, the update is $x[k+1] = \Phi\,x[k]$ with

$$\Phi = \begin{pmatrix}\mathbb{1} - \Delta t^2WK & \Delta t(\mathbb{1} - \Delta t\,WD)\\ -\Delta t\,WK & \mathbb{1} - \Delta t\,WD\end{pmatrix}.$$

For $N = 1$ and $W = 1/I$ this is the $\Phi$ of section 3.3.2, which is why the same symbol is kept. Section 3.3.3 applies unchanged, since it holds for any square matrix, so the robot is stable if and only if $\rho(\Phi) < 1$. The inertial coupling sits in the off diagonal entries of $W$. A damping torque at the knee also accelerates the hip, through $W_{\mathrm{hip,knee}}$, and with explicit servos that acceleration is answered only one step later.

Case A, every group shares $f_n$ and $\zeta$.

1. The gains are computed against the reference inertias of section 3.2.3 with a common $\omega_n = 2\pi f_n$ and $\zeta$, so $k_{p,j} = S_j\omega_n^2$ and $k_{d,j} = 2\zeta S_j\omega_n$, that is $K = \omega_n^2S$ and $D = 2\zeta\omega_n S$, with friction left aside here and restored in section 3.3.8. The gains are then proportional to one another, hinge by hinge.
2. Eliminating $K$ and $D$, every block of $\Phi$ becomes a polynomial in the single matrix $WS$,

$$\Phi = \begin{pmatrix}\mathbb{1} - (\omega_n\Delta t)^2\,WS & \Delta t\big(\mathbb{1} - 2\zeta\omega_n\Delta t\,WS\big)\\ -\Delta t\,\omega_n^2\,WS & \mathbb{1} - 2\zeta\omega_n\Delta t\,WS\end{pmatrix}.$$

3. $WS$ can be diagonalised. It is similar to the symmetric positive definite $S^{1/2}WS^{1/2}$, since $WS = S^{-1/2}(S^{1/2}WS^{1/2})S^{1/2}$, and a symmetric matrix has an orthogonal eigendecomposition $S^{1/2}WS^{1/2} = Q\,\mathrm{diag}(\mu_1, \dots, \mu_N)\,Q^\top$ with real eigenvalues, positive here. Hence $WS = U\,\mathrm{diag}(\mu_1, \dots, \mu_N)\,U^{-1}$ with $U = S^{-1/2}Q$, whose columns $u_i$ satisfy $WSu_i = \mu_iu_i$. The index $i = 1, \dots, N$ counts these eigenpairs, one per hinge, and has nothing to do with a particular hinge. The eigenvalues of $SW$ are the same $\mu_i$, since $SW = S(WS)S^{-1}$.
4. Change to modal coordinates, $e = U\tilde e$ and $\dot q = U\tilde v$, so that $\tilde e_i$ and $\tilde v_i$ are the amplitudes of the error and the velocity along $u_i$. Multiplying the update by $U^{-1}$ and using $U^{-1}(WS)U = \mathrm{diag}(\mu_1, \dots, \mu_N)$, every block of $\Phi$ becomes diagonal, so the pair $(\tilde e_i, \tilde v_i)$ evolves without reference to any other pair,

$$\begin{pmatrix}\tilde e_i\\ \tilde v_i\end{pmatrix}[k+1] = \begin{pmatrix}1 - a_i & \Delta t(1 - b_i)\\ -a_i/\Delta t & 1 - b_i\end{pmatrix}\begin{pmatrix}\tilde e_i\\ \tilde v_i\end{pmatrix}[k], \qquad a_i = \mu_i(\omega_n\Delta t)^2, \quad b_i = \mu_i\,2\zeta\omega_n\Delta t.$$

The matrix is the scalar $\Phi$ of section 3.3.2 with $(a, b)$ replaced by $(a_i, b_i)$. In matrix terms, $\Phi = \mathrm{diag}(U, U)\,\tilde\Phi\,\mathrm{diag}(U, U)^{-1}$, and a reordering of the coordinates of $\tilde\Phi$ makes it block diagonal with these $N$ blocks of size two. This is what is meant by the coupled matrix splitting into one copy of the scalar matrix per mode. The $2N$ eigenvalues of $\Phi$ are the union of the two eigenvalues of each block.
5. A mode is the motion along one $u_i$. If the initial error and velocity are both proportional to $u_i$, they stay proportional to $u_i$ for ever, and only their common amplitude evolves, as a single joint would. The vector $u_i$ gives the relative amplitude of every hinge in that motion. The biped's leg, for instance, has a mode in which the hip and the knee turn in opposite senses, so that the thigh swings while the shank barely turns. Its inertia is small, which makes its $\mu_i$ large.
6. Each mode is therefore a single joint whose inertia ratio $\kappa$ of section 3.3.6 is $\mu_i$. For one hinge, $W = 1/I$ and $\mu = S/I = \kappa$, so this is an exact generalisation. By section 3.3.4, the robot is stable if and only if every mode satisfies $a_i > 0$, $0 < b_i < 2$ and $a_i + 2b_i < 4$.
7. Since $(a_i, b_i) = \mu_i\,(a_{\mathrm{ref}}, b_{\mathrm{ref}})$, the downward closure of section 3.3.4 makes every mode stable at a pose as soon as $(a_{\mathrm{ref}}, b_{\mathrm{ref}})$ is stable and every $\mu_i \le 1$ at that pose, and the margin then holds too, $b_i \le b_{\mathrm{ref}}$. The largest of the $\mu_i$ is $\lambda_{\max}(SW)$, so the requirement over all poses is $\lambda_{\max}(SW(q)) \le 1$, the condition of section 3.2.3. Each mode then has a natural frequency $\sqrt{\mu_i}\,\omega_n \le \omega_n$ and a damping ratio $\sqrt{\mu_i}\,\zeta \le \zeta$, which is the sense in which $f_n$ and $\zeta$ are ceilings.
8. With the minimum locked inertias in place of $S$, $\lambda_{\max}(SW)$ reaches 10.06 on the biped (section 3.2.3). At 10 Hz and 0.85 that gives a mode with $b_i$ = 5.4, far outside the stable triangle, and `validate_coupled.txt` gives a largest spectral radius of 6.44 for that choice. The single joint argument of section 3.3.6 was therefore incomplete, and the coupling, not the pose of a single joint, sets the biped's limit.

Case B, the groups have different $f_n$ and $\zeta$.

Why case A's argument does not carry over. When the groups have different gains, $K$ and $D$ are no longer multiples of one matrix $S$. The blocks $WK$ and $WD$ of $\Phi$ then have different eigenvectors in general, so no single change of coordinates diagonalises both, and the robot no longer separates into independent single joint problems. The eigenvalues of $\Phi$ can still be computed numerically for one design at one pose, which is what `validate_coupled.py` does, but that gives no condition that holds for every design and every pose at once. Case B therefore proves a sufficient condition instead, which is the matrix form of the third Jury condition $a + 2b < 4$,

$$\lambda_{\max}\big(W(\Delta t^2K + 2\Delta t\,D)\big) < 4 \quad\Longrightarrow\quad \rho(\Phi) < 1.$$

The proof does not compute the eigenvalues of $\Phi$. It follows Lyapunov's method, which finds a quantity of the state, an energy, with two properties. It never increases from one step to the next, and it is positive for every nonzero state. A quantity with both properties cannot let any solution grow, and a solution that cannot grow cannot belong to an eigenvalue outside the unit circle. Steps B1 to B3 rewrite the update in a form in which such an energy can be written down, B4 constructs it from the recurrence and explains it physically, B5 records that it never increases, B6 shows where the hypothesis makes it positive, and B7 to B9 turn those two facts into the statement about the eigenvalues of $\Phi$. B10 checks the result against the scalar case.

As in the scalar sections, the pose is held fixed over the steps considered, so that $W$ is a constant matrix, symmetric and positive definite by section 3.2.1. Every hinge has a positive stiffness and a positive total damping slope, $k_{p,j} > 0$ and $c_j > 0$, so that $K$ and $D$ are diagonal with positive entries.

Please note that $y$ is the error in scaled coordinates, $\hat K$ and $\hat D$ are the scaled stiffness and damping matrices, $\delta$ and $\delta'$ are the changes of $y$ over one step and over the next, $\xi$ and $\eta$ are placeholder vectors used once in step B4 to state an identity, $\bar y$ the average of two successive values of $y$, and $E$ is the energy. $z$, $\Phi_y$ and $L$ appear in step B7, and $\Gamma$ and $\gamma$ in step B6. All are listed in the notation chart.

Step B1, a recurrence in the error alone. The update of the coupled transition matrix carries both the error and the velocity. The energy argument is simpler with one variable, so the velocity is eliminated. The position update $e[k] = e[k-1] + \Delta t\,\dot q[k]$ gives the velocity as a difference of errors,

$$\dot q[k] = \frac{(e[k] - e[k-1])}{\Delta t}.$$

The velocity update of step 3 above is the familiar $v = u + at$, $\dot q[k+1] = \dot q[k] + \Delta t\,\ddot q[k]$, with the acceleration held over the step. The minus sign enters through the acceleration, not through the kinematics. By steps 1 and 2 the acceleration is the inverse inertia acting on the servo torque,

$$\ddot q[k] = W\tau[k] = -W\big(Ke[k] + D\dot q[k]\big).$$

The servo is a restoring law. A positive error $e = q - q^*$ means the hinge has passed its target, so the spring torque $-k_pe$ pushes it back, and a positive velocity is opposed by the damping torque $-c\,\dot q$. $W$ preserves that sense, since its positive definiteness gives $\tau^\top\ddot q = \tau^\top W\tau > 0$, so the acceleration always has a positive component along the torque that opposes the error and the velocity. Were the sign positive, the torque would drive the hinge away from its target at any timestep. Substituting this acceleration into $v = u + at$ gives $\dot q[k+1] = \dot q[k] - \Delta t\,W\big(Ke[k] + D\dot q[k]\big)$, and the position update one step later, $e[k+1] - e[k] = \Delta t\,\dot q[k+1]$, then reads

$$e[k+1] - e[k] = \Delta t\,\dot q[k] - \Delta t^2W\big(Ke[k] + D\dot q[k]\big).$$

For one hinge the last term is $-\Delta t^2(k_pe[k] + c\,\dot q[k])/I = -a\,e[k] - b\,\Delta t\,\dot q[k]$, the terms of section 3.3.2. $\Delta t\,\dot q[k]$ appears twice, once on its own and once inside the damping term, written $\Delta t^2WD\dot q[k] = \Delta t\,WD\,(\Delta t\,\dot q[k])$. Replacing $\Delta t\,\dot q[k]$ by $e[k] - e[k-1]$ in both places gives

$$e[k+1] - e[k] = \big(e[k] - e[k-1]\big) - \Delta t^2\,WK\,e[k] - \Delta t\,WD\,\big(e[k] - e[k-1]\big),$$

and moving the bracket to the left,

$$e[k+1] - 2e[k] + e[k-1] = -\Delta t^2 WK e[k] - \Delta t\,WD\,\big(e[k] - e[k-1]\big).$$

The left side is the discrete second difference, the counterpart of $\Delta t^2\ddot e$, so this is the discrete form of $\ddot e = -WKe - WD\dot e$, the servo law acting through the inverse inertia.

Step B2, why the recurrence must be symmetrised. An energy is a quadratic form, a sum of terms $u^\top Xu$, and the argument of step B5 rearranges such terms using $u^\top Xv = v^\top Xu$, which holds only for a symmetric $X$. The matrices $WK$ and $WD$ in step B1 are products of two symmetric matrices and are not symmetric themselves, so a change of variable is made that turns them into symmetric matrices with the same content.

Step B3, the symmetrised recurrence. A symmetric positive definite matrix has a unique symmetric positive definite square root. With the eigendecomposition $W = V\,\mathrm{diag}(\varpi_1, \dots, \varpi_N)\,V^\top$, with orthogonal $V$ and positive eigenvalues $\varpi_j$, it is $W^{1/2} = V\,\mathrm{diag}(\sqrt{\varpi_1}, \dots, \sqrt{\varpi_N})\,V^\top$, which satisfies $W^{1/2}W^{1/2} = W$, and its inverse is written $W^{-1/2}$. Define the scaled error and the scaled gains

$$y[k] = W^{-1/2}e[k], \qquad \hat K = \Delta t^2\,W^{1/2}KW^{1/2}, \qquad \hat D = \Delta t\,W^{1/2}DW^{1/2}.$$

$\hat K$ and $\hat D$ are symmetric, and they are positive definite, since $u^\top\hat Ku = \Delta t^2(W^{1/2} u)^\top K(W^{1/2} u) > 0$ for every $u \neq 0$, $K$ having positive diagonal entries and $W^{1/2}$ being invertible, and likewise for $\hat D$. They are also dimensionless, the matrix counterparts of $a$ and $b$. Multiply the recurrence of step B1 on the left by $W^{-1/2}$ and insert $\mathbb{1} = W^{1/2}W^{-1/2}$ before each $e$ on the right. The stiffness term becomes $W^{-1/2}\,\Delta t^2WK\,e = \Delta t^2\,W^{1/2}KW^{1/2}\,W^{-1/2}e = \hat Ky$, and the damping term likewise becomes $\hat D(y[k] - y[k-1])$, so

$$y[k+1] - 2y[k] + y[k-1] = -\hat K\,y[k] - \hat D\,\big(y[k] - y[k-1]\big).$$

For one hinge, $W = 1/I$, $y = e\sqrt{I}$, $\hat K = a$ and $\hat D = b$, and this is the scalar recurrence $e[k+1] - 2e[k] + e[k-1] = -a\,e[k] - b(e[k] - e[k-1])$.

Step B4, constructing the energy, and why its terms are kinetic and potential energy. The energy is not guessed. It is obtained from the recurrence of step B3 by the discrete form of the argument that yields the energy of a continuous oscillator, and its coefficients, the correction $-\frac{1}{2}\hat D$ included, are whatever that construction produces.

The continuous template comes first. For the oscillator $\ddot y + \mathcal{D}\dot y + \mathcal{K}y = 0$, with $\mathcal{K}$ and $\mathcal{D}$ symmetric positive definite, multiply on the left by $\dot y^\top$. Since $\dot y^\top\ddot y = \frac{d}{dt}\big(\frac{1}{2}\dot y^\top\dot y\big)$, and $\dot y^\top\mathcal{K}y = \frac{d}{dt}\big(\frac{1}{2}y^\top\mathcal{K}y\big)$ by the symmetry of $\mathcal{K}$,

$$\frac{d}{dt}\Big(\frac{1}{2}\dot y^\top \dot y + \frac{1}{2} y^\top \mathcal{K} y \Big) = -\dot y^\top \mathcal{D} \dot y \le 0.$$

The bracket is kinetic plus potential energy, and the right side is the power the damping dissipates. Three ingredients carry over to the discrete case, the equation written in a velocity variable, a multiplier that plays the part of $\dot y$, and the recognition of each resulting product as the change of some quantity from one step to the next.

1. The recurrence in step differences. Write $\delta = \delta[k] = y[k] - y[k-1]$ and $\delta' = \delta[k+1] = y[k+1] - y[k]$ for the changes over two successive steps, the counterparts of $\Delta t\,\dot y$. The left side of the recurrence of step B3 is $y[k+1] - 2y[k] + y[k-1] = \delta' - \delta$, and its damping term is $\hat D\delta$, so the discrete error equation reads

$$\delta' - \delta = -\hat K\,y[k] - \hat D\,\delta.$$

It is the discrete form of $\ddot y = -\mathcal{K}y - \mathcal{D}\dot y$, with one feature the continuous equation lacks. The damping acts on $\delta$, the change over the step that has just ended, because the explicit servo computes its damping torque from the velocity at the start of the step.

2. The multiplier. The counterpart of $\dot y$ at step $k$ is the centred difference $y[k+1] - y[k-1] = \delta + \delta'$, which spans the step before and the step after, as $\dot y(t)$ is centred at $t$. Multiply the equation of item 1 on the left by $(\delta + \delta')^\top$. Items 3 to 5 treat its three terms in turn.

3. The left side is the change of the kinetic term. $(\delta + \delta')^\top(\delta' - \delta) = \delta'^\top\delta' - \delta^\top\delta$, the cross terms $\delta^\top\delta'$ and $-\delta'^\top\delta$ cancelling. This is $\delta^\top\delta$ one step later minus its present value.

4. The stiffness term is the change of a potential, and this is where the product form comes from. Since $\delta + \delta' = y[k+1] - y[k-1]$ and $\hat K$ is symmetric,

$$-(\delta + \delta')^\top\hat K\,y[k] = -\big(y[k+1]^\top\hat K\,y[k] - y[k]^\top\hat K\,y[k-1]\big).$$

The quantity that changes is $y[k]^\top\hat K\,y[k-1]$, the stiffness coupling the two positions that bracket a step. The multiplier pairs $y[k]$ with its neighbours on either side and never with itself, so the square $y[k]^\top\hat Ky[k]$ does not appear, and the product is the only form in which this term telescopes.

5. The damping term is not yet of one sign. It is $-(\delta + \delta')^\top\hat D\,\delta = -\delta^\top\hat D\delta - \delta'^\top\hat D\delta$, and the second product has no fixed sign. Completing the square around the centred change gives

$$-\delta^\top\hat D\delta - \delta'^\top\hat D\delta = -\frac{1}{2}(\delta + \delta')^\top\hat D(\delta + \delta') + \frac{1}{2}\delta'^\top\hat D\delta' - \frac{1}{2}\delta^\top\hat D\delta,$$

which is checked by expanding the first term on the right with the symmetry of $\hat D$, $-\frac{1}{2}\delta^\top\hat D\delta - \delta'^\top\hat D\delta - \frac{1}{2}\delta'^\top\hat D\delta'$. The first term on the right is the dissipation, non positive. The other two are $\frac{1}{2}\delta^\top\hat D\delta$ one step later minus its present value, the change of a quantity exactly as in items 3 and 4.

6. Collecting the terms. Item 1, multiplied by $(\delta + \delta')^\top$ and with items 3 to 5 substituted, reads

$$\delta'^\top\delta' - \delta^\top\delta = -\big(y[k+1]^\top\hat K\,y[k] - y[k]^\top\hat K\,y[k-1]\big) - \frac{1}{2}(\delta + \delta')^\top\hat D(\delta + \delta') + \frac{1}{2}\delta'^\top\hat D\delta' - \frac{1}{2}\delta^\top\hat D\delta.$$

Every change of a quantity is moved to the left and the dissipation is left on the right. The kinetic change and the damping change combine into $\delta'^\top(\mathbb{1} - \frac{1}{2}\hat D)\delta' - \delta^\top(\mathbb{1} - \frac{1}{2}\hat D)\delta$, and the potential change is added to it, so

$$E[k+1] - E[k] = -\frac{1}{2}(\delta + \delta')^\top\hat D(\delta + \delta'), \qquad E[k] = \delta[k]^\top\Big(\mathbb{1} - \frac{1}{2}\hat D\Big)\delta[k] + y[k]^\top\hat K\,y[k-1].$$

The coefficient $\mathbb{1} - \frac{1}{2}\hat D$ of the kinetic term is therefore forced by the explicit damping. The damping acts on $\delta = \frac{1}{2}(\delta + \delta') - \frac{1}{2}(\delta' - \delta)$, the centred change less half the change of the change. Had it acted on the centred change $\frac{1}{2}(\delta + \delta')$ alone, as a centred scheme would apply it, item 5 would have produced the dissipation alone and the kinetic weight would be $\mathbb{1}$. The remainder $\frac{1}{2}(\delta + \delta')^\top\hat D(\delta' - \delta) = \frac{1}{2}\delta'^\top\hat D\delta' - \frac{1}{2}\delta^\top\hat D\delta$ is the telescoping difference of item 5, and it enters the energy as the correction $-\frac{1}{2}\delta^\top\hat D\delta$. For one hinge, $E[k] = (1 - \frac{b}{2})\,\delta[k]^2 + a\,y[k]\,y[k-1]$.

The identification is exact once the scalings are undone. Since $\delta[k] = W^{-1/2}(e[k] - e[k-1]) = \Delta t\,W^{-1/2}\dot q[k]$,

$$\delta[k]^\top\delta[k] = \Delta t^2\,\dot q[k]^\top W^{-1}\dot q[k], \qquad y[k]^\top\hat K\,y[k-1] = \Delta t^2\,e[k]^\top K\,e[k-1].$$

$W^{-1}$ is the inertia matrix the hinges present with the base free to recoil, the matrix counterpart of $I^{\mathrm{free}}_j$, so $\frac{1}{2}\dot q^\top W^{-1}\dot q$ is the kinetic energy of the hinge motion. $K$ is the stiffness of the servos, so $\frac{1}{2} e^\top Ke$ is the potential energy stored in their virtual springs, here taken as the product of the errors at the two ends of the step. $E[k]$ is therefore $2\Delta t^2$ times kinetic plus potential energy, with one correction, $-\frac{1}{2}\delta^\top\hat D\delta = -\frac{1}{2}\Delta t^3\,\dot q^\top D\,\dot q$, smaller than the kinetic term by a factor of the order of $b/2$. The correction and the product form of the potential are not chosen for their physical meaning. Items 4 and 5 of the construction above force them, and as $\Delta t \to 0$ both reduce to the continuous energy.

The midpoint form of the energy is needed by step B6. The product $y[k]^\top\hat K\,y[k-1]$ has no fixed sign, since the two positions may point in opposing directions, so the form above cannot show that $E$ is positive. Step B6 needs $E$ written instead as two quadratic forms, each in a single vector and each with one matrix whose sign can be read off. The two vectors are the change over the step, $\delta[k]$, and the midpoint of the step, $\bar y[k] = \frac{1}{2}(y[k] + y[k-1])$, which together determine both positions. The rewriting uses the polarisation identity, which expresses a product $\xi^\top X\eta$ through two squares. In it $\xi$ and $\eta$ stand for any two vectors of $\mathbb{R}^N$ and $X$ for any symmetric $N \times N$ matrix, placeholders introduced only to state the identity once, and item 4 of the list below replaces them by $y[k]$, $y[k-1]$ and $\hat K$.

1. Expand the square of the sum. $(\xi + \eta)^\top X(\xi + \eta) = \xi^\top X\xi + \xi^\top X\eta + \eta^\top X\xi + \eta^\top X\eta = \xi^\top X\xi + 2\,\xi^\top X\eta + \eta^\top X\eta$, the two cross terms being equal because $\eta^\top X\xi$ is a number and equals its own transpose, $\xi^\top X^\top\eta = \xi^\top X\eta$.
2. Expand the square of the difference in the same way. $(\xi - \eta)^\top X(\xi - \eta) = \xi^\top X\xi - 2\,\xi^\top X\eta + \eta^\top X\eta$.
3. Subtract item 2 from item 1. The squares of $\xi$ and $\eta$ cancel and the cross terms add, so $(\xi + \eta)^\top X(\xi + \eta) - (\xi - \eta)^\top X(\xi - \eta) = 4\,\xi^\top X\eta$, that is, $\xi^\top X\eta = \frac{1}{4}(\xi + \eta)^\top X(\xi + \eta) - \frac{1}{4}(\xi - \eta)^\top X(\xi - \eta)$. For numbers this is $xy = \frac{1}{4}(x + y)^2 - \frac{1}{4}(x - y)^2$.
4. Substitute $\xi = y[k]$, $\eta = y[k-1]$ and $X = \hat K$. Then $\xi + \eta = 2\bar y[k]$ and $\xi - \eta = \delta[k]$, so the first term is $\frac{1}{4}(2\bar y[k])^\top\hat K(2\bar y[k]) = \bar y[k]^\top\hat K\,\bar y[k]$, and $y[k]^\top\hat K\,y[k-1] = \bar y[k]^\top\hat K\,\bar y[k] - \frac{1}{4}\delta[k]^\top\hat K\,\delta[k]$.
5. Substitute item 4 into $E[k]$. This gives $E[k] = \delta[k]^\top(\mathbb{1} - \frac{1}{2}\hat D)\delta[k] - \frac{1}{4}\delta[k]^\top\hat K\delta[k] + \bar y[k]^\top\hat K\,\bar y[k]$, and the two quadratic forms in $\delta[k]$ share their vector, so their matrices add,

$$E[k] = \delta[k]^\top \Gamma\,\delta[k] + \bar y[k]^\top\hat K\,\bar y[k], \qquad \Gamma = \mathbb{1} - \frac{1}{4}\hat K - \frac{1}{2}\hat D.$$

In this form the potential is evaluated at the midpoint of the step, and both corrections sit in the kinetic weight $\Gamma$, the stiffness correction $-\frac{1}{4}\hat K$ having been released from the product form by item 4. For one hinge, $E[k] = (1 - \frac{a}{4} - \frac{b}{2})\,\delta[k]^2 + a\,\bar y[k]^2$, whose first coefficient is positive exactly when $a + 2b < 4$, a preview of step B6. `validate_coupled.py` confirms both forms of $E$ and the change of item 6 numerically.

Step B5, the energy never increases. Item 6 of step B4 gives the change of the energy over a step,

$$E[k+1] - E[k] = -\frac{1}{2}\,(\delta + \delta')^\top\hat D\,(\delta + \delta') \le 0.$$

The inequality holds because $\hat D$ is positive definite. This is the discrete counterpart of $-\dot y^\top\mathcal{D}\dot y$, with $\frac{1}{2}(\delta + \delta')$ the change over a step averaged across two steps. No condition on the gains was used in this step, only the recurrence itself. The same expression follows directly by computing $E[k+1] - E[k]$ from the definition and substituting $\hat K\,y[k] = \delta - \delta' - \hat D\,\delta$ from item 1.

Step B6, where the hypothesis enters, the energy is positive. A quantity that never increases controls the state only if it is positive. $E$ is not positive for every gain, and the hypothesis $\lambda_{\max}(W(\Delta t^2K + 2\Delta tD)) < 4$ is exactly what makes it so. The step has four parts, a background fact about symmetric matrices, the hypothesis rewritten as a statement about $\Gamma$, the positivity of $E$ that follows, and a quantitative form of that positivity which step B8 uses.

A real symmetric $N \times N$ matrix $X$ has real eigenvalues and an orthonormal basis of eigenvectors [38], as step B3 already used for $W$. Write $\lambda_{\min}(X)$ for its smallest eigenvalue, the counterpart of $\lambda_{\max}(X)$ of section 3.2.3. Two consequences are needed.

- Shift. If $Xu = \lambda u$, then $(\mathbb{1} - X)u = u - \lambda u = (1 - \lambda)u$. The matrix $\mathbb{1} - X$ therefore has the same eigenvectors as $X$, and each of its eigenvalues is $1$ minus the corresponding eigenvalue of $X$. Subtracting from the identity moves every eigenvalue by the same amount, because the identity acts on every vector as the number one.
- Definiteness. The smallest value of the ratio $u^\top Xu/u^\top u$ over nonzero $u$ is $\lambda_{\min}(X)$, attained at its eigenvector [38], the counterpart of the statement about $\lambda_{\max}$ used in the monotonicity property at the end of this section. Hence $u^\top Xu \ge \lambda_{\min}(X)\lVert u\rVert^2$ for every $u$, and $X$ is positive definite, $u^\top Xu > 0$ for every $u \neq 0$, exactly when $\lambda_{\min}(X) > 0$, that is, when every eigenvalue of $X$ is positive.

Three links connect the hypothesis to $\Gamma$.

1. From the definitions of step B3, $\frac{1}{4}\hat K + \frac{1}{2}\hat D = \frac{1}{4}\Delta t^2W^{1/2}KW^{1/2} + \frac{1}{2}\Delta t\,W^{1/2}DW^{1/2} = \frac{1}{4}W^{1/2}(\Delta t^2K + 2\Delta tD)W^{1/2}$.
2. For any matrix $X$, $W^{1/2}XW^{1/2} = W^{-1/2}(WX)W^{1/2}$, so $W^{1/2}XW^{1/2}$ is similar to $WX$. If $(WX)v = \lambda v$, then $W^{-1/2}(WX)W^{1/2}\,(W^{-1/2}v) = \lambda\,W^{-1/2}v$, so the two matrices share every eigenvalue, as step B7 shows in general. With $X = \Delta t^2K + 2\Delta tD$, link 1 gives $\lambda_{\max}(\frac{1}{4}\hat K + \frac{1}{2}\hat D) = \frac{1}{4}\lambda_{\max}(W(\Delta t^2K + 2\Delta tD))$, and the hypothesis is equivalent to
$$\lambda_{\max}\Big(\frac{1}{4}\hat K + \frac{1}{2}\hat D\Big) < 1.$$
3. $\frac{1}{4}\hat K + \frac{1}{2}\hat D$ is symmetric, as a sum of symmetric matrices, and positive definite, as a sum of positive definite ones, so its eigenvalues lie in the interval $(0, \infty)$, and by link 2 in $(0, 1)$. By the shift property, the eigenvalues of $\Gamma = \mathbb{1} - (\frac{1}{4}\hat K + \frac{1}{2}\hat D)$ are $1$ minus these, and so lie in $(0, 1)$. All are positive, and by the definiteness property $\Gamma$ is positive definite.

The chain runs in both directions, so the hypothesis holds exactly when $\Gamma$ is positive definite. For one hinge, $\Gamma = 1 - \frac{a}{4} - \frac{b}{2}$, positive exactly when $a + 2b < 4$.

The energy is positive. Use the midpoint form of step B4. Its first term $\delta^\top\Gamma\delta$ is positive unless $\delta = 0$, since $\Gamma$ is positive definite. Its second term $\bar y^\top\hat K\bar y$ is positive unless $\bar y = 0$, since $\hat K$ is positive definite. The step and the midpoint determine the two positions, $y[k] = \bar y[k] + \frac{1}{2}\delta[k]$ and $y[k-1] = \bar y[k] - \frac{1}{2}\delta[k]$, as adding and subtracting $\bar y[k] = \frac{1}{2}(y[k] + y[k-1])$ and $\frac{1}{2}\delta[k] = \frac{1}{2}(y[k] - y[k-1])$ confirms. Hence $E[k] = 0$ only when $y[k] = y[k-1] = 0$, and $E[k] > 0$ otherwise.

Without the hypothesis the argument, not the robot, fails. If $\Gamma$ has an eigenvector $u$ with $u^\top\Gamma u \le 0$, the nonzero state with $\delta[k] = u$ and $\bar y[k] = 0$, that is $y[k] = \frac{1}{2}u$ and $y[k-1] = -\frac{1}{2}u$, has $E[k] = u^\top\Gamma u \le 0$. An energy that is not positive for every nonzero state gives no control of the state, since a falling energy is then compatible with a growing state, and the $\lambda = -1$ case of step B9 shows that such growth does occur at the boundary of the hypothesis.

Why a quantitative form is needed. Positivity says that $E[k] > 0$ for every nonzero state, and step B5 says that $E[k]$ never exceeds $E[0]$. Together they bound the energy, not yet the state. To conclude that the state cannot grow, a large state must force a large energy, so that an energy held below $E[0]$ holds the state below a fixed size. The inequality derived below states this with an explicit constant. Since $E[k]$ is a positive definite quadratic form in the pair $(y[k], y[k-1])$, such a constant exists in any case by the definiteness property applied to that form, so this part adds no hypothesis. Its purpose is to make the constant explicit in $\Gamma$ and $\hat K$, and to be the bridge from step B5, a statement about $E$, to step B8, a statement about the solutions, which is the form in which eigenvalues can be examined.

1. The energy bounds the step and the midpoint. Let $\gamma$ be the smaller of $\lambda_{\min}(\Gamma)$ and $\lambda_{\min}(\hat K)$, positive since both matrices are positive definite. By the definiteness property, $\delta^\top\Gamma\delta \ge \lambda_{\min}(\Gamma)\lVert\delta\rVert^2 \ge \gamma\lVert\delta\rVert^2$ and $\bar y^\top\hat K\bar y \ge \gamma\lVert\bar y\rVert^2$, so adding, $E[k] \ge \gamma\big(\lVert\delta[k]\rVert^2 + \lVert\bar y[k]\rVert^2\big)$.
2. The positions bound by the step and the midpoint. The triangle inequality states that the length of a sum of two vectors is at most the sum of their lengths, $\lVert u + v\rVert \le \lVert u\rVert + \lVert v\rVert$, as one side of a triangle is no longer than the other two together. Applied to $y[k] = \bar y[k] + \frac{1}{2}\delta[k]$ and to $y[k-1] = \bar y[k] + (-\frac{1}{2}\delta[k])$, it gives $\lVert y[k]\rVert \le \lVert\bar y[k]\rVert + \frac{1}{2}\lVert\delta[k]\rVert$, and the same bound for $\lVert y[k-1]\rVert$.
3. Squaring. For two non negative numbers, the square of their sum is at most twice the sum of their squares, the difference being the square of their difference. Here $2\lVert\bar y\rVert^2 + 2(\frac{1}{2}\lVert\delta\rVert)^2 - (\lVert\bar y\rVert + \frac{1}{2}\lVert\delta\rVert)^2 = (\lVert\bar y\rVert - \frac{1}{2}\lVert\delta\rVert)^2 \ge 0$, so item 2 gives $\lVert y[k]\rVert^2 \le 2\lVert\bar y[k]\rVert^2 + \frac{1}{2}\lVert\delta[k]\rVert^2$, and the same for $\lVert y[k-1]\rVert^2$.
4. Adding the two bounds of item 3 and using item 1,
$$\lVert y[k]\rVert^2 + \lVert y[k-1]\rVert^2 \le 4\lVert\bar y[k]\rVert^2 + \lVert\delta[k]\rVert^2 \le 4\big(\lVert\bar y[k]\rVert^2 + \lVert\delta[k]\rVert^2\big) \le \frac{C_E}{\gamma}\,E[k], \qquad C_E = 4.$$
The left side is the squared length of the pair $z[k] = (y[k], y[k-1])$ of step B7.

Step B7, the state that $E$ measures, and its relation to $\Phi$. $E[k]$ depends on the pair $z[k] = (y[k], y[k-1]) \in \mathbb{R}^{2N}$. The recurrence of step B3 advances this pair by a matrix, derived in four items.

1. Start from the recurrence of step B3, $y[k+1] - 2y[k] + y[k-1] = -\hat K\,y[k] - \hat D\,\big(y[k] - y[k-1]\big)$, and solve it for the newest value by adding $2y[k] - y[k-1]$ to both sides, $y[k+1] = 2y[k] - y[k-1] - \hat K\,y[k] - \hat D\,y[k] + \hat D\,y[k-1]$.
2. Group the terms by the vector they act on, writing $2y[k] = 2\cdot\mathbb{1}\,y[k]$ and $y[k-1] = \mathbb{1}\,y[k-1]$ so that the matrices can be collected, $y[k+1] = (2\cdot\mathbb{1} - \hat K - \hat D)\,y[k] + (\hat D - \mathbb{1})\,y[k-1]$.
3. The second half of the next pair is the first half of the present one, $y[k] = \mathbb{1}\,y[k] + 0\,y[k-1]$.
4. Stacking items 2 and 3, with $z[k+1] = (y[k+1], y[k])$,
$$z[k+1] = \Phi_y\,z[k], \qquad \Phi_y = \begin{pmatrix}2\cdot\mathbb{1} - \hat K - \hat D & \hat D - \mathbb{1}\\ \mathbb{1} & 0\end{pmatrix}.$$
For one hinge, $\Phi_y = \begin{pmatrix}2 - a - b & b - 1\\ 1 & 0\end{pmatrix}$, whose trace $2 - a - b$ and determinant $1 - b$ equal those of the scalar $\Phi$ of section 3.3.2, as the similarity below requires.

The pair is an invertible linear function of the original state $x[k] = (e[k], \dot q[k])$. The position update of the coupled transition matrix, one step earlier, reads $e[k] = e[k-1] + \Delta t\,\dot q[k]$, so $e[k-1] = e[k] - \Delta t\,\dot q[k]$, and with $y = W^{-1/2}e$,
$$y[k] = W^{-1/2}e[k], \qquad y[k-1] = W^{-1/2}e[k] - \Delta t\,W^{-1/2}\dot q[k], \qquad\text{that is}\qquad z[k] = L\,x[k], \quad L = \begin{pmatrix}W^{-1/2} & 0\\ W^{-1/2} & -\Delta t\,W^{-1/2}\end{pmatrix}.$$
$L$ is invertible, and its inverse is read off by solving the two equations for $x[k]$, $e[k] = W^{1/2}y[k]$ and $\dot q[k] = (e[k] - e[k-1])/\Delta t = W^{1/2}(y[k] - y[k-1])/\Delta t$,
$$L^{-1} = \begin{pmatrix}W^{1/2} & 0\\ W^{1/2}/\Delta t & -W^{1/2}/\Delta t\end{pmatrix},$$
as multiplying out $LL^{-1} = \mathbb{1}$ confirms.

$\Phi_y$ and $\Phi$ are similar. The argument has three items.

1. The same step in two coordinates. Since $x[k+1] = \Phi x[k]$ and $x[k] = L^{-1}z[k]$, the pair one step later is $z[k+1] = Lx[k+1] = L\Phi x[k] = L\Phi L^{-1}z[k]$.
2. The two matrices are equal. Item 1 and the stacked recurrence both give $z[k+1]$ from $z[k]$, so $\Phi_yz = L\Phi L^{-1}z$ for every pair $z$ that arises from some state $x$, and since $L$ is invertible, every $z \in \mathbb{R}^{2N}$ arises, from $x = L^{-1}z$. Two matrices whose products with every vector agree are equal, as applying them to the unit vectors reads off their columns one by one. Hence $\Phi_y = L\Phi L^{-1}$, which is what it means for the two to be similar [38].
3. Similar matrices have the same eigenvalues. If $\Phi\psi = \lambda\psi$ with $\psi \neq 0$, then $\Phi_y(L\psi) = L\Phi L^{-1}L\psi = L\Phi\psi = \lambda\,L\psi$, and $L\psi \neq 0$ because $L$ is invertible, so $\lambda$ is an eigenvalue of $\Phi_y$ with eigenvector $L\psi$. The converse runs the same way with $L^{-1}$, since $\Phi = L^{-1}\Phi_yL$. Multiplicities agree as well, since the characteristic polynomials coincide, $\det(\lambda\mathbb{1} - \Phi_y) = \det\big(L(\lambda\mathbb{1} - \Phi)L^{-1}\big) = \det L\,\det(\lambda\mathbb{1} - \Phi)\,\det L^{-1} = \det(\lambda\mathbb{1} - \Phi)$, using $\lambda\mathbb{1} = L(\lambda\mathbb{1})L^{-1}$ and $\det L\,\det L^{-1} = \det(LL^{-1}) = 1$.

The two matrices describe the same motion in different coordinates, so a statement about the eigenvalues of either is a statement about both, and in particular $\rho(\Phi_y) = \rho(\Phi)$. `validate_coupled.py` confirms the similarity and the identities of steps B4 and B5 numerically.

Step B8, from a bounded energy to eigenvalues no larger than one. Take any real initial pair $z[0]$. By step B5, $E[k] \le E[k-1] \le \dots \le E[0]$, and by step B6, $\lVert z[k]\rVert^2 \le (C_E/\gamma)E[k] \le (C_E/\gamma)E[0]$. Every real solution therefore stays within a fixed bound for all $k$, the bound depending only on its own initial pair.

Suppose now that $\Phi$ had an eigenvalue $\lambda$ with $|\lambda| > 1$ and eigenvector $\psi$. The argument proceeds in four items.

1. Possibly complex. A real matrix may have complex eigenvalues, in conjugate pairs, with complex eigenvectors, so $\lambda$ and $\psi$ are allowed to be complex.
2. A growing solution. By step B7, $L\psi$ is an eigenvector of $\Phi_y$ for the same $\lambda$, and the sequence $z[k] = \lambda^kL\psi$ satisfies the recurrence, since $\Phi_y\lambda^kL\psi = \lambda^k\,\lambda L\psi = \lambda^{k+1}L\psi$.
3. Real solutions. The energy bound holds for real states, since $E$ is built from transposes of real vectors. Because $\Phi_y$ is real, $\Phi_y(w_{\mathrm{R}} + \mathrm{i}\,w_{\mathrm{I}}) = \Phi_yw_{\mathrm{R}} + \mathrm{i}\,\Phi_yw_{\mathrm{I}}$ with both products real, so equating real and imaginary parts, the real part $\mathrm{Re}(\lambda^kL\psi)$ and the imaginary part $\mathrm{Im}(\lambda^kL\psi)$ of the complex solution are each real solutions.
4. The contradiction. The squared length of a complex vector is the sum of the squared lengths of its real and imaginary parts, and $\lVert\lambda^kL\psi\rVert = |\lambda|^k\lVert L\psi\rVert$, so $\lVert\mathrm{Re}(\lambda^kL\psi)\rVert^2 + \lVert\mathrm{Im}(\lambda^kL\psi)\rVert^2 = |\lambda|^{2k}\lVert L\psi\rVert^2$. With $|\lambda| > 1$ and $L\psi \neq 0$ this grows without bound, so the two real solutions cannot both be bounded, contradicting the bound just found.

No eigenvalue of $\Phi$ therefore has $|\lambda| > 1$, and by step B7 the same holds for $\Phi_y$. Equivalently, $\rho(\Phi) \le 1$.

Step B9, no eigenvalue on the unit circle. A bounded solution is not yet a decaying one, since an eigenvalue with $|\lambda| = 1$ gives a solution that neither grows nor decays. Such eigenvalues are excluded by summing step B5. Adding the changes from step $0$ to step $k$,

$$\frac{1}{2}\sum_{m=0}^{k-1}\big(\delta[m] + \delta[m+1]\big)^\top\hat D\,\big(\delta[m] + \delta[m+1]\big) = E[0] - E[k] \le E[0].$$

The terms are non negative and their partial sums stay below $E[0]$, so the series converges, its terms tend to zero, and, $\hat D$ being positive definite, $\delta[m] + \delta[m+1] \to 0$ along every real solution.

Suppose $\Phi$ had an eigenvalue $\lambda$ with $|\lambda| = 1$ and eigenvector $\psi$. By step B7, $L\psi$ is an eigenvector of $\Phi_y$, and the second block row of $\Phi_y$ forces it into the form $L\psi = (\lambda y_0, y_0)$ with $y_0 \neq 0$, so the solution $z[k] = \lambda^{k-1}L\psi$ has $y[k] = \lambda^ky_0$. Then $\delta[k] = \lambda^k(1 - \lambda^{-1})y_0$ and $\delta[k] + \delta[k+1] = \lambda^k(1 - \lambda^{-1})(1 + \lambda)\,y_0$. The real and imaginary parts are real solutions, and by the previous paragraph both must tend to zero, so this complex sequence tends to zero. Its length is constant, $\lVert(1 - \lambda^{-1})(1 + \lambda)y_0\rVert$, since $|\lambda^k| = 1$, so it must be zero, which leaves only $\lambda = 1$ or $\lambda = -1$. Both are impossible.

- $\lambda = 1$ makes $y[k] = y_0$ constant, and the recurrence of step B3 then gives $0 = -\hat Ky_0$, impossible for $y_0 \neq 0$ since $\hat K$ is positive definite.
- $\lambda = -1$ makes $y[k] = (-1)^ky_0$. The left side of the recurrence is $(-1)^k(-1 - 2 - 1)y_0 = -4(-1)^ky_0$, and the right side is $-(-1)^k\hat Ky_0 - \hat D\big((-1)^k - (-1)^{k-1}\big)y_0 = -(-1)^k(\hat K + 2\hat D)y_0$. Equating them gives $(\frac{1}{4}\hat K + \frac{1}{2}\hat D)\,y_0 = y_0$, so $1$ would be an eigenvalue of $\frac{1}{4}\hat K + \frac{1}{2}\hat D$, which step B6 excludes.

Steps B8 and B9 together place every eigenvalue of $\Phi$ strictly inside the unit circle, $\rho(\Phi) < 1$, and by section 3.3.3 every solution decays. This proves the sufficient condition stated at the start of case B.

Step B10, the scalar check. For one hinge, $\hat K = a$ and $\hat D = b$, and the condition $\frac{1}{4}\hat K + \frac{1}{2}\hat D < 1$ reads $a/4 + b/2 < 1$, which is $a + 2b < 4$, the third Jury condition of section 3.3.4. With $a > 0$ it implies $b < 2$, the first, so the matrix condition reproduces the exact scalar region and loses nothing when $N = 1$. As a further numerical check, `validate_coupled.py` draws two hundred thousand random positive definite $W$ of order 2 to 4 with diagonal $K$ and $D$, scales each towards the boundary of the hypothesis, and finds no spectral radius above one.

Using the condition for the search box. Steps B1 to B10 certify one set of gains, for one design, at one pose. The search needs a stronger statement, a box of $(f_n, \zeta)$ values, fixed before any design is drawn, inside which every gain mix the search can produce is stable for every design at every pose. This part derives that box from the hypothesis in four moves, and section 3.3.8 turns the result into its constraint C1. Friction is set aside here, so that $c_j = k_{d,j}$ and $D = D_{\mathrm{servo}}$, the diagonal matrix of the damping gains alone, and the next part restores it.

Refresher, the single joint numbers of a hinge. Sections 3.3.2 and 3.3.6 describe one joint by two dimensionless numbers, the stiffness per step $a = k_p\Delta t^2/I$ and the damping per step $b = c\,\Delta t/I$. For hinge $j$ of the robot, measured against its reference inertia $S_j$ of section 3.2.3, they are

$$a_j = \frac{\Delta t^2k_{p,j}}{S_j}, \qquad b_j = \frac{\Delta t\,c_j}{S_j}.$$

With the gains of the plan, $k_{p,j} = S_j\omega_j^2$ and $k_{d,j} = 2\zeta_jS_j\omega_j$, where $\omega_j = 2\pi f_{n,j}$ and $\zeta_j$ are the searched values of hinge $j$'s group, $S_j$ cancels exactly as $I_{\mathrm{ref}}$ did in section 3.3.6,

$$a_j = (\omega_j\Delta t)^2, \qquad b_j = 2\zeta_j\omega_j\Delta t.$$

They are the per step numbers of hinge $j$ as if it were a single joint whose inertia equalled its reference, and they depend only on the two searched variables of its group and on $\Delta t$, not on the design.

Move 1, the hypothesis hinge by hinge. $\Delta t^2K + 2\Delta tD$ is diagonal, and by the definitions above its entry $j$ is $\Delta t^2k_{p,j} + 2\Delta t\,k_{d,j} = S_j(a_j + 2b_j)$. The left side of the hypothesis is therefore built from the single joint sums $a_j + 2b_j$, the quantity whose bound by 4 is the third Jury condition of section 3.3.4, each weighted by $S_j$.

Move 2, the corner of the box. Each group's variables are searched within fixed ranges, $f_n \in [f_{n,\min}, f_{n,\max}]$ and $\zeta \in [\zeta_{\min}, \zeta_{\max}]$, which section 3.5 sets to $[6, 14]$ Hz and $[0.7, 1.0]$. Write $\omega_{\max} = 2\pi f_{n,\max}$ for the largest natural frequency in the box. $a_j$ grows with $\omega_j$ alone and $b_j$ with the product $\zeta_j\omega_j$, and since the two ranges are independent and positive, the product is largest where both factors are, at $\zeta_{\max}\omega_{\max}$. Every hinge of every group therefore satisfies

$$a_j \le a_{\mathrm{ref}} = (\omega_{\max}\Delta t)^2, \qquad b_j \le b_{\mathrm{ref}} = 2\zeta_{\max}\omega_{\max}\Delta t.$$

These are the $a_{\mathrm{ref}}$ and $b_{\mathrm{ref}}$ of section 3.3.6 evaluated at the upper corner $(f_{n,\max}, \zeta_{\max})$ of the box, the numbers of a single joint at its reference inertia driven by the stiffest and most damped gains the search can choose. They are one pair of numbers for the whole family, fixed by the box and $\Delta t$ alone. At the ranges of section 3.5 and $\Delta t$ = 0.005 s they are $a_{\mathrm{ref}}$ = 0.193 and $b_{\mathrm{ref}}$ = 0.880.

Move 3, from the hinges to the matrix. By moves 1 and 2, each diagonal entry obeys $S_j(a_j + 2b_j) \le S_j(a_{\mathrm{ref}} + 2b_{\mathrm{ref}})$. With $S = \mathrm{diag}(S_1, \dots, S_N)$ as in section 3.2.3, this entrywise comparison of diagonal matrices is the order $\Delta t^2K + 2\Delta tD_{\mathrm{servo}} \preceq (a_{\mathrm{ref}} + 2b_{\mathrm{ref}})\,S$. Two properties of $\lambda_{\max}$ carry the comparison through the product with $W$.

- Monotonicity. Write $X \preceq Y$ when $Y - X$ is positive semidefinite (section 3.2.1). If $X \preceq Y$, then $W^{1/2}XW^{1/2} \preceq W^{1/2}YW^{1/2}$, since $u^\top W^{1/2}(Y - X)W^{1/2}u = (W^{1/2}u)^\top(Y - X)(W^{1/2}u) \ge 0$. The largest eigenvalue of a symmetric matrix is the largest value of $u^\top Xu/u^\top u$ over nonzero $u$ [38], so it cannot decrease when the matrix grows in this order. For diagonal matrices, $X \preceq Y$ simply means that every diagonal entry of $X$ is at most the corresponding entry of $Y$.
- Similarity. $\lambda_{\max}(WX) = \lambda_{\max}(W^{1/2}XW^{1/2})$ for any diagonal $X$ with positive entries, by link 2 of step B6, and in particular $\lambda_{\max}(W^{1/2}SW^{1/2}) = \lambda_{\max}(SW)$, since $SW$ is similar to $WS$ through $S$.

Applying similarity, then monotonicity, then similarity again,

$$\lambda_{\max}\big(W(\Delta t^2K + 2\Delta t\,D_{\mathrm{servo}})\big) = \lambda_{\max}\big(W^{1/2}(\Delta t^2K + 2\Delta t\,D_{\mathrm{servo}})W^{1/2}\big) \le (a_{\mathrm{ref}} + 2b_{\mathrm{ref}})\,\lambda_{\max}\big(W^{1/2}SW^{1/2}\big) = (a_{\mathrm{ref}} + 2b_{\mathrm{ref}})\,\lambda_{\max}(SW).$$

Move 4, every design and every pose. Section 3.2.3 chooses $S_j$ so that $\lambda_{\max}(SW(q)) \le 1$ at every pose $q$ of the design, by Gershgorin's theorem [38]. Move 3 then gives $\lambda_{\max}(W(\Delta t^2K + 2\Delta tD_{\mathrm{servo}})) \le a_{\mathrm{ref}} + 2b_{\mathrm{ref}}$ at every pose, for every design, and for every choice of $(f_n, \zeta)$ inside the box, group by group. The hypothesis of case B, and with it $\rho(\Phi) < 1$, therefore holds everywhere as soon as $a_{\mathrm{ref}} + 2b_{\mathrm{ref}} < 4$, a condition on the two upper limits $f_{n,\max}$ and $\zeta_{\max}$ and on $\Delta t$ alone. At the ranges of section 3.5 the sum is 0.193 + 1.760 = 1.95, about half the limit. The same moves applied to the damping alone, $\Delta t\,D_{\mathrm{servo}} \preceq b_{\mathrm{ref}}S$, give $\lambda_{\max}(\Delta t\,WD_{\mathrm{servo}}) \le b_{\mathrm{ref}}$, the matrix counterpart of the $b \le 1$ margin of section 3.3.5.

What this gives the search. The four moves replace a stability check on every design at every pose by a check on the corner of the box, made once when the generator is built. Inside the box CMA-ES may move every group's $f_n$ and $\zeta$ freely, with no rejection and no clamp, and the box itself is the only constraint the frequency and damping variables carry.

Using the condition for any diagonal gains, friction included. Move 1 holds for any diagonal gains, not only for those the box produces, with the total damping slope $c_j$ in place of $k_{d,j}$, and $a_j$, $b_j$ then defined by the first pair of equations of the refresher. Bounding each entry by $S_j\max_l(a_l + 2b_l)$ instead of by the corner gives $\Delta t^2K + 2\Delta tD \preceq \max_l(a_l + 2b_l)\,S$, and move 3 gives $\lambda_{\max}(W(\Delta t^2K + 2\Delta tD)) \le \lambda_{\max}(SW)\,\max_l(a_l + 2b_l)$. A per group check of $a_j + 2b_j$ against $S_j$ with the friction slope included is therefore a sufficient test for the coupled robot, which sections 3.3.8 and 3.6 use for the catalogue gains and the friction slope.

Two consequences close the argument. The reference $S_j$ of section 3.2.3 makes $\lambda_{\max}(SW) \le 1$ by Gershgorin's theorem, so in both cases the downward closure of section 3.3.4 carries over to the robot with $\mu_i$ in place of $\kappa$. And since a contact only lowers $W$ in the order $\preceq$ (section 3.2.1), which lowers every $\lambda_{\max}$ above, the airborne robot against which $S$ is computed bounds stance as well. The discrepancy between swing and stance inertia therefore cannot destabilise the servo. It only makes the planted joint slower and less damped than the searched ceiling, by the factor of section 3.1.

The conditions carried forward. Section 3.3.7 ends with the following mathematical statements, which section 3.3.8 applies as its constraints and its table. Throughout, $W(q)$ is the hinge block of the inverse mass matrix at pose $q$, $S = \mathrm{diag}(S_1, \dots, S_N)$ the reference inertias of section 3.2.3, $K$ and $D$ the diagonal stiffness and total damping matrices, and $\Delta t$ the physics step.

1. Coupled sufficient condition, proved in steps B1 to B9 for any diagonal gains at a fixed pose,
$$\Lambda = \lambda_{\max}\big(W(\Delta t^2K + 2\Delta t\,D)\big) < 4 \quad\Longrightarrow\quad \rho(\Phi) < 1.$$
2. Reference property, guaranteed by the construction of $S_j$ in section 3.2.3 for every design,
$$\lambda_{\max}\big(SW(q)\big) \le 1 \quad\text{for every pose } q.$$
3. Box condition, from moves 1 to 4, for every design, every pose and every $(f_n, \zeta)$ of every group within $[f_{n,\min}, f_{n,\max}] \times [\zeta_{\min}, \zeta_{\max}]$, friction aside,
$$a_{\mathrm{ref}} = (2\pi f_{n,\max}\Delta t)^2, \quad b_{\mathrm{ref}} = 4\pi\zeta_{\max}f_{n,\max}\Delta t, \qquad \Lambda_{\mathrm{servo}} \le a_{\mathrm{ref}} + 2b_{\mathrm{ref}} < 4 \quad\Longrightarrow\quad \rho(\Phi) < 1.$$
4. Damping margin, from the same moves applied to $D_{\mathrm{servo}}$ alone,
$$b_{\mathrm{ref}} \le 1 \quad\Longleftrightarrow\quad \zeta_{\max}f_{n,\max} \le \frac{1}{4\pi\Delta t}, \qquad\text{and then}\qquad \beta_{\mathrm{servo}} = \lambda_{\max}\big(\Delta t\,WD_{\mathrm{servo}}\big) \le b_{\mathrm{ref}} \le 1.$$
Since $b_{\mathrm{ref}} \le 1$ gives $a_{\mathrm{ref}} + 2b_{\mathrm{ref}} \le a_{\mathrm{ref}} + 2$, the margin implies the box condition of item 3 whenever $a_{\mathrm{ref}} < 2$, that is $f_{n,\max} < \sqrt2/(2\pi\Delta t)$ = 45.0 Hz at 0.005 s, far above any range considered.
5. Per group test with friction, a sufficient condition for any diagonal gains, with $c_j = k_{d,j} + c_{f,j}$,
$$a_j + 2b_j = \frac{\Delta t^2k_{p,j} + 2\Delta t\,c_j}{S_j} < 4 \ \text{ for every group} \quad\Longrightarrow\quad \Lambda \le \lambda_{\max}(SW)\max_j(a_j + 2b_j) < 4 \quad\Longrightarrow\quad \rho(\Phi) < 1.$$
6. Exact condition of case A, when every group shares $f_n$ and $\zeta$, with $\mu_i$ the eigenvalues of $SW$,
$$\rho(\Phi) < 1 \quad\Longleftrightarrow\quad \mu_i(a_{\mathrm{ref}}, b_{\mathrm{ref}}) \in \mathcal{T} \ \text{ for every mode } i.$$
7. Stance, from the contact property of section 3.2.1, a contact lowers $W$ in the order $\preceq$ and so lowers $\Lambda$ and $\beta$, so items 1 to 6 evaluated on the airborne robot hold in stance as well.

Section 3.3.8 enforces item 4 as its constraint C1, which by item 4 also secures item 3, reports item 5 as its constraint C2, and tabulates $\Lambda$ and $\beta$ of items 1 and 4, together with $\rho(\Phi)$ itself, for the gains of each family.

#### 3.3.8 The constraints

C1, the search box. The upper corner of every group's $f_n$ and $\zeta$ ranges must satisfy $b_{\mathrm{ref}} = 4\pi\zeta_{\max}f_{n,\max}\Delta t \le 1$, which is $\zeta_{\max}f_{n,\max} \le 1/(4\pi\Delta t) = 15.9$ Hz at 0.005 s. By section 3.3.7 it bounds every mode's servo damping per step by $b_{\mathrm{ref}}$ and the stability sum $\lambda_{\max}(W(\Delta t^2K + 2\Delta tD))$ by $\lambda_{\max}(SW)(a_{\mathrm{ref}} + 2b_{\mathrm{ref}})$, at most 1.95 at the corner of section 3.5 against the limit of 4. The twofold headroom absorbs the per design estimator of section 3.8 E3, which overstates the biped hip's reference by 15 per cent. C1 is checked once when the generator is built.

C2, the friction slope. `IdentifiedActuator` subtracts $\mu_s\tanh(\dot q/\dot q_s) + \mu_d\dot q$ explicitly as well (`actuator_pd.py:33-35`). Its slope at rest, $c_f = \mu_s/\dot q_s + \mu_d$, adds to the damping, and it depends on the picked actuator, not on the search. C2 is the per group sufficient test of section 3.3.7 with the friction slope included, $a_j + 2b_j < 4$ against $S_j$. It is reported rather than enforced, for two reasons. The identified friction is itself a stiff explicit damper against the biped's reference inertias, the abduction's slope of 3.02 alone giving $c_f\Delta t/S_j$ = 1.72. And the tanh saturates for $|\dot q| > \dot q_s$ = 0.1 rad/s, so a mode the friction destabilises grows only until the velocity leaves the linear band and then settles into a small limit cycle (section 3.10). The runner prints the share of environments failing it at every draw (section 3.6).

The following values, from [validate_coupled.py](../context/artefacts/w_eta_optimisation-2026-10-27/validate_coupled.py), evaluate the coupled robot over 4274 biped and 3536 quadruped poses, the sweep's worst poses, a 5 point grid over each limb and 3000 random whole body poses. $\Lambda$ is $\lambda_{\max}(W(\Delta t^2K + 2\Delta tD))$, stable if below 4, and $\beta$ is $\lambda_{\max}(\Delta tWD)$, inside the margin if at most 1, each at its worst pose. The spectral radius is that of $\Phi$ of section 3.3.7 with friction, and the last column is the share of the random poses at which it exceeds one.

| Family, gains | $\Lambda$ servo | $\Lambda$ with friction | $\beta$ servo | $\beta$ with friction | Largest $\rho(\Phi)$ | Random poses with $\rho(\Phi) > 1$ |
|---|---|---|---|---|---|---|
| biped, catalogue | 13.366 | 16.337 | 6.615 | 8.101 | 7.219 | 0.460 |
| biped, largest training draw | 14.849 | 17.819 | 7.351 | 8.836 | 7.967 | 0.566 |
| biped, 10 Hz, 0.85 | 1.111 | 4.021 | 0.509 | 1.967 | 1.010 | 0.000 |
| biped, 14 Hz, 1.0 | 1.860 | 4.711 | 0.838 | 2.270 | 1.369 | 0.018 |
| quadruped, catalogue | 1.061 | 2.789 | 0.490 | 1.339 | 0.939 | 0.000 |
| quadruped, largest training draw | 1.269 | 2.974 | 0.587 | 1.423 | 0.935 | 0.000 |
| quadruped, 10 Hz, 0.85 | 1.056 | 2.886 | 0.483 | 1.399 | 0.939 | 0.000 |
| quadruped, 14 Hz, 1.0 | 1.767 | 3.594 | 0.796 | 1.710 | 0.904 | 0.000 |

The searched gains keep the servo inside the margin on both robots, $\beta$ at most 0.838 and $\Lambda$ at most 1.860. Friction alone pushes the biped past the sufficient test and, at the worst pose, slightly past the true limit, in the bounded way C2 describes. The quadruped is stable under every setting, friction included.

The biped's catalogue gains are a different matter. Their servo part alone gives $\Lambda$ = 13.4, and the linear model puts 46 per cent of uniformly random poses outside the stable region. The abduction's catalogue damping of 13.5 against its free inertia of 0.0102 is $b$ = 6.6 on its own. The prediction was checked directly in MuJoCo, which integrates the same rigid body dynamics semi-implicitly. With gravity, contact, passive damping and joint limits removed, the catalogue servo applied explicitly at the worst pose grew the largest joint velocity from 3.1e-3 to 3.1e4 rad/s within ten steps, as the predicted radius of 5.73 implies. At the zero pose, radius 0.979, the same perturbation decayed. In Isaac Lab the effort limits bound such a mode, so the expected symptom is chatter at the poses concerned, with the hip flexed and the knee folded, rather than divergence. That symptom has not been sought in the runs on disk, and section 3.9 step 4 checks it before the plan is relied on.

C3, the policy loop. A servo much faster than the 50 Hz policy settles within one action, and `../context/KScale.md` section 11.1 caps $\omega_n$ at the policy's Nyquist frequency $\pi/0.02$ = 157 rad/s, 25 Hz. Under the reference of section 3.2.3 every mode is at most $f_n$, so the box of section 3.5 stays below the cap and C3 needs no code.

### 3.4 Why not the KScale damping rule

`../context/KScale.md` section 11.1 sets each KScale joint's damping as the smallest of four terms,

$$D = \min\left(2\zeta\sqrt{K I_{\mathrm{stance}}},\ \frac{\tau_{\mathrm{limit}}}{\dot q_{p95}},\ 2\sqrt{K I_{\mathrm{swing}}},\ \frac{I_{\mathrm{swing}}}{\Delta t\,m_{\mathrm{rand}}}\right),$$

with $\zeta$ = 0.70, $K$ chosen first from the stance loads, $\dot q_{p95}$ the 95th percentile joint speed measured on the trained robot, and $m_{\mathrm{rand}}$ the worst ratio of damping scale to inertia scale in the randomisation. That rule hand-derives one gain pair for one robot. The co-optimisation needs a rule that a search can move continuously across thousands of designs, and each term carries over as follows.

1. The design target, $2\zeta\sqrt{K I_{\mathrm{stance}}}$. KScale prescribes $\zeta$ = 0.70 against the stance inertia. Here $\zeta$ is a search variable, so CMA-ES finds the damping the fitness prefers instead of prescribing it. Computing the stance inertia per design would need a stance model of the body carried through the planted leg, which the generator does not build. The stance damping ratio that results is $\zeta\sqrt{I_{\mathrm{ref}}/I_{\mathrm{stance}}}$ by section 3.1, which can be logged once a stance inertia is computed. The search then measures the stance behaviour through the fitness, rather than through a target.
2. The actuator term, $\tau_{\mathrm{limit}}/\dot q_{p95}$. It needs the joint speeds of a policy already trained on the design, which do not exist when CMA-ES proposes it. The `DCMotor` torque speed envelope clips the torque in simulation regardless. The term is deferred, and could be added later from the evaluation's own logged joint speeds.
3. The swing cap, $2\sqrt{K I_{\mathrm{swing}}}$, which is $\zeta_{\mathrm{swing}} \le 1$. Under the reference of section 3.2.3, every mode's damping ratio is at most the searched $\zeta$ (section 3.3.7), so the cap is $\zeta \le 1$ on the search variable, which the box of section 3.5 enforces by its upper bound of 1.0, at every pose and in every mode.
4. The integrator term, $I_{\mathrm{swing}}/(\Delta t\,m_{\mathrm{rand}})$, which is $b \le 1/m_{\mathrm{rand}}$. This is C1. The training draws of section 3.6 for a searched group come from the box itself rather than from a scale on top of it, and the leg inertias are not randomised (the mass randomisation acts on the base alone, `params/env.yaml:1352-1385` of the biped run), so $m_{\mathrm{rand}} = 1$ and C1 is the KScale term with the reference inertia of section 3.2.3 in place of the locked swing inertia. The replacement matters. KScale's term bounds each joint against its own locked inertia, which section 3.3.7 shows does not bound the coupled modes, by a factor of 10 on the biped.

The KScale formulation is therefore kept in substance. Terms 3 and 4 become box constraints satisfied by construction, term 1 is replaced by the search, and term 2 is deferred. A literal $\min(\cdot)$ is avoided on purpose. A clamp maps a whole region of $(f_n, \zeta)$ to one gain pair, so CMA-ES would see different genotypes produce identical designs and identical fitness. That is the flat landscape that parameters pinned at a bound produced in the runs of `/ws/context/cmaes.md` section 14, where 81 to 84 per cent of the quadruped's designs sat at a bound and the design signal fell towards zero. Using the reference inertia keeps the map from search variables to gains one to one and keeps every constraint a box.

### 3.5 Search variables, defaults and ranges

Each actuator group of the family gets two variables, `<group>_natural_frequency` in Hz and `<group>_damping_ratio`. The group names are `abad`, `hip`, `knee` and, for the biped, `ankle`. The proposed ranges are $f_n \in [6, 14]$ Hz and $\zeta \in [0.7, 1.0]$.

- Defaults. CMA-ES starts every coordinate at 0.5 of its unit range (`usd_generator.py:902`), so these linear ranges start the search at exactly 10 Hz and 0.85, the requested defaults. The same values fill a variable that is switched off while its partner is on, see section 3.7.
- Literature. 10 Hz is BeyondMimic's ablated choice [1]. The damping range is the 0.7 to 1.0 band of the Isaac Sim tutorial [4], and its upper end is the swing cap of section 3.4 term 3. The frequency range spans a factor of 2.3, keeping well clear of the slow joints Chakraborty and colleagues found shrink the feasible set [29].
- Stability. $\zeta_{\max}f_{n,\max}$ = 14 Hz $\le$ 15.9 Hz satisfies C1 with $b_{\mathrm{ref}}$ = 0.880 and $a_{\mathrm{ref}}$ = 0.193, so the coupled stability sum stays at most 1.95 against the limit of 4 (section 3.3.8).
- Encoding. Linear, per section 2.8.

Against the reference inertia the defaults ask for modest gains. The table expresses each catalogue gain pair as $f_n$ and $\zeta$ against $S_j$, and gives the gains that 10 Hz and 0.85 produce, the stiffness demanded by a full scale action of 0.25 rad as a share of the effort limit, and the authority frequency $\sqrt{\tau_{\max}/(0.25\,S_j)}/2\pi$, the highest at which that action stays within the limit.

| Family, group | $S_j$ | Catalogue as $f_n$ [Hz], $\zeta$ | $k_p$, $k_d$ at 10 Hz, 0.85 | $0.25\,k_p/\tau_{\max}$ | Authority frequency [Hz] |
|---|---|---|---|---|---|
| biped abad | 0.0088 | 12.55, 9.68 | 34.9, 0.94 | 0.15 | 26.2 |
| biped hip | 0.0214 | 9.72, 4.96 | 84.6, 2.29 | 0.35 | 16.8 |
| biped knee | 0.0184 | 9.10, 1.91 | 72.4, 1.96 | 0.20 | 22.3 |
| biped ankle | 0.0074 | 5.86, 0.92 | 29.1, 0.79 | 0.18 | 23.4 |
| quadruped abad | 0.0097 | 10.21, 0.96 | 38.4, 1.04 | 0.41 | 15.7 |
| quadruped hip | 0.0129 | 8.86, 0.77 | 51.0, 1.38 | 0.54 | 13.6 |
| quadruped knee | 0.0095 | 11.53, 0.65 | 37.6, 1.02 | 0.27 | 19.4 |

The authority frequency is at least 13.6 Hz in every group, so the whole box stays within the effort limits. The first version of this section, which measured the biped's hip and knee against their locked inertias, found 10 Hz four and three times beyond those limits and recommended a 1.5 to 4.5 Hz override, which is withdrawn. Two readings of the table follow. The quadruped's catalogue gains sit at 8.9 to 11.5 Hz with $\zeta$ of 0.65 to 0.96, inside or at the edge of the box, so the defaults reproduce them closely. The biped's catalogue stiffnesses sit between 5.9 and 12.6 Hz, but their damping ratios of 1.9 to 9.7 lie far above the box, and that excess damping is what section 3.3.8 finds unstable.

### 3.6 Which gains apply when, PPO training against CMA-ES evaluation

The runner alternates two phases. During PPO training, every environment runs its design for `ea_update_interval` iterations. At each generation boundary, `_evaluate_population` scores every design under a shared set of commands and terrain cells (`co_optimisation/co_optimisation/runners/copt_on_policy_runner.py:832-986`), and `_update_morphology` then applies the next population (`copt_on_policy_runner.py:683-779`). Today both phases run every design at its nominal gains, the values `apply_actuator_params` writes (`copt_on_policy_runner.py:772-774`), and the task's startup gain randomisation is overwritten before the first iteration (section 1.2).

The two phases need different gains. PPO training should expose the policy to a spread of gains, so that it stays robust to the gains of any design the search may propose, as late start already exposes it to a spread of designs. That robustness matters for transfer [6], and Tune to Learn found that reinforcement learning reaches working policies across gain regimes [28]. The CMA-ES evaluation should score each design at exactly its own gains, so that the fitness ranks the gains the search proposed and not a random perturbation of them. The precedence is therefore as follows.

- During PPO training, the randomisation takes precedence. At every population apply, each environment draws its own gains, independently per joint.
  - A group whose $f_n$ or $\zeta$ is in `CMAES_PARAM_KEYS` draws $f_n$ and $\zeta$ uniformly over its search box and converts them against its design's $I_{\mathrm{ref}}$. The policy then trains on every gain the search can propose, not only the gains proposed so far, and every draw satisfies C1, so no further stability check is needed.
  - Every other group scales its design's nominal gains, the actuator's catalogue values, by the stiffness and damping ranges of the task's own `randomize_actuator_gains` event terms. Those ranges are read from the event manager at startup, so the task configuration remains their single source.
- During the CMA-ES evaluation, the design's gains take precedence. `_evaluate_population` writes the exact gains first, the CMA-ES gains of a searched group and the catalogue gains of any other group, and restores the training draws when it returns. The final evaluation of the optimised population in `log_optimised_designs` goes through the same method (`copt_on_policy_runner.py:1062-1080`).

The new behaviour is switched by a `gain_randomisation` key in the runner's `copt` configuration. It defaults to `False`, which reproduces every run to date, and `train.py` sets it to `True` for new runs, so the choice is recorded in each run's `params/copt_agent.yaml`. The startup event terms in the shared task configuration are left untouched. They remain inert, being overwritten as before, and the runner takes over their role in co-optimisation runs alone.

The draws are made once per population apply, at each generation, at the late start toggle and on resume. A redraw at every episode reset would add variety, but it needs a hook into the reset event path and is left for later.

One risk follows from the randomisation now taking effect for groups that are not searched, and it is not new. The per group sufficient test of section 3.3.7, $a_j + 2b_j$ against $S_j$ with the friction slope, stable wherever every group is below 4, gives the following values.

| Family, group | Catalogue gains | Largest training draw | 10 Hz, 0.85 |
|---|---|---|---|
| biped abad | 18.85 | 20.56 | 4.58 |
| biped hip | 7.57 | 8.51 | 2.58 |
| biped knee | 6.63 | 7.19 | 5.54 |
| biped ankle | 2.10 | 2.24 | 2.55 |
| quadruped abad | 3.42 | 3.68 | 3.24 |
| quadruped hip | 2.49 | 2.68 | 2.73 |
| quadruped knee | 3.19 | 3.40 | 3.29 |

The biped's catalogue gains fail the test before any draw, and section 3.3.8 shows their servo part alone to be unstable at 46 per cent of random poses. The randomisation therefore widens an instability every biped co-optimisation run already carries, from 46 to 57 per cent of random poses, rather than creating one. Narrowing the abduction's damping scale, the remedy the first version of this section proposed, cannot remove it, since the catalogue damping itself is the cause. The remedy is to search the biped's damping ratios, or at least to fix them, so that its gains come from the box. The quadruped passes under every draw. At the defaults, friction alone keeps the biped abduction and knee above 4, in the bounded way C2 describes. The runner prints, at every draw, the share of environments per group that fail the test.

### 3.7 Backward compatibility and switching variables on and off

- A `DesignSpec` gain field defaults to `None`, which means the group keeps the catalogue's `stiffness` and `damping`. `variable_params.json` states none of the new fields, so every baseline and every specification written before this change emits the gains it always did.
- A gain variable takes effect only when its key is in `CMAES_PARAM_KEYS`. With the tuple unchanged, CMA-ES has the same four dimensions, every saved checkpoint passes `_check_dimension` (`usd_generator.py:905-909`), and the per design actuator parameters are identical to today's. This is what allows comparison with earlier runs.
- If either key of a group is enabled, the group's gains come from $(f_n, \zeta)$, and the disabled partner takes its default, 10 Hz or 0.85. If neither key of a group is enabled, the group keeps its catalogue gains.
- A key for a group the family lacks, the ankle on the quadruped, is dropped with a printed notice, so one `CMAES_PARAM_KEYS` serves both robots.
- `gain_randomisation` absent or `False` leaves the runner's gain handling exactly as today, as section 3.6 describes.
- The eight gain scales are moved to `DEPRECATED_PARAM_KEYS`, so a caller naming one raises, as for the six scales retired before (`usd_generator.py:460-463`). This changes `RandomDesignGenerator`, which sampled those scales whenever `--design_search random` was used. No run on disk used random search, per section 1.2.
- `RandomDesignGenerator` drops the new keys from its ranges, so random search keeps the catalogue gains, as the user directed. With `gain_randomisation` on, its training gains are still randomised by the task's scales.
- The `*_scale` branch of `DesignGeneratorBase._actuator_params` (`usd_generator.py:738-741`) becomes unreachable and is left in place, since removing it would edit the shared base for no behavioural gain.

### 3.8 Code changes

The design generator is a git submodule, which `README.md:52-57` asks to be installed editable. `co_optimisation` is in practice installed as a copy that `djinn start train` reinstalls from the mounted tree at every launch (`/ws/context/copt.md:399`). Edits E1 to E3 are submodule commits.

#### E1, `design_generator/biped/generate.py`

Add `Optional` to the typing import and a module tuple of the gain field names after `_resolve`.

```python
from typing import List, Optional, Tuple, Union
```

```python
# Joint servo dynamics per actuator group, natural frequency [Hz] and damping
# ratio [-]. co_optimisation's usd_generator converts them to stiffness and
# damping against the joint's inertia. Appended to vector() after the act_*
# indices, so the fifteen leading positions keep their meaning.
GAIN_GROUPS = ("abad", "hip", "knee", "ankle")
GAIN_FIELDS = tuple(f"{g}_{q}" for g in GAIN_GROUPS for q in ("natural_frequency", "damping_ratio"))
```

In `DesignSpec`, after `ankle_bend_offset`,

```python
    # Servo dynamics, see GAIN_FIELDS. None keeps the catalogue entry's own
    # stiffness and damping, so every specification written before these fields
    # existed emits the gains it always did.
    abad_natural_frequency: Optional[float] = None
    abad_damping_ratio: Optional[float] = None
    hip_natural_frequency: Optional[float] = None
    hip_damping_ratio: Optional[float] = None
    knee_natural_frequency: Optional[float] = None
    knee_damping_ratio: Optional[float] = None
    ankle_natural_frequency: Optional[float] = None
    ankle_damping_ratio: Optional[float] = None
```

`to_dict` and `from_dict` iterate `fields()` and need no change. In `vector`, `from_vector` and `labels`,

```python
                self.act_abduction, self.act_hip, self.act_knee, self.act_ankle,
                # Unset gain fields travel as NaN and come back as None.
                *(np.nan if getattr(self, f) is None else getattr(self, f) for f in GAIN_FIELDS),
            ],
```

```python
            act_ankle=   int(np.clip(round(v[14]), 0, n_act - 1)),
            # A fifteen entry vector predates the gain fields and leaves them None.
            **{f: (None if len(v) <= 15 + i or np.isnan(v[15 + i]) else float(v[15 + i]))
               for i, f in enumerate(GAIN_FIELDS)},
        )
```

```python
            "act_abd", "act_hip", "act_kne", "act_ank",
            *(f"{g[:3]}_{'fn' if q == 'natural_frequency' else 'zeta'}"
              for g in GAIN_GROUPS for q in ("natural_frequency", "damping_ratio")),
        ]
```

In `DesignBounds`, after `act_ankle`, with `as_pairs`, `integrality_mask` and `sample` extended so the integer slice stays at positions 11 to 14,

```python
    # Servo dynamics, section 3.5 of plans/w_eta_optimisation.md. The search starts
    # at the midpoint, 10 Hz and 0.85, and 14 Hz x 1.0 keeps 4 pi zeta f dt <= 1
    # at the 0.005 s physics step of both co-optimisation tasks.
    abad_natural_frequency:  Tuple[float, float] = (6.0, 14.0)
    abad_damping_ratio:      Tuple[float, float] = (0.7, 1.0)
    hip_natural_frequency:   Tuple[float, float] = (6.0, 14.0)
    hip_damping_ratio:       Tuple[float, float] = (0.7, 1.0)
    knee_natural_frequency:  Tuple[float, float] = (6.0, 14.0)
    knee_damping_ratio:      Tuple[float, float] = (0.7, 1.0)
    ankle_natural_frequency: Tuple[float, float] = (6.0, 14.0)
    ankle_damping_ratio:     Tuple[float, float] = (0.7, 1.0)
```

```python
            self.act_abduction, self.act_hip, self.act_knee, self.act_ankle,
            *(getattr(self, f) for f in GAIN_FIELDS),
        ]

    def integrality_mask(self) -> np.ndarray:
        """1 = integer-valued, 0 = continuous. Matches `vector()` order."""
        return np.array([0] * 11 + [1] * 4 + [0] * len(GAIN_FIELDS))

    def sample(self, rng: np.random.Generator) -> DesignSpec:
        lo_hi = self.as_pairs()
        cont = np.array([rng.uniform(lo, hi) for (lo, hi) in lo_hi[:11]], dtype=float)
        ints = np.array([rng.integers(lo, hi + 1) for (lo, hi) in lo_hi[11:15]], dtype=float)
        gains = np.array([rng.uniform(lo, hi) for (lo, hi) in lo_hi[15:]], dtype=float)
        return DesignSpec.from_vector(np.concatenate([cont, ints, gains]))
```

#### E2, `design_generator/quadruped/generate.py`

The same edit with `GAIN_GROUPS = ("abad", "hip", "knee")`, six `DesignSpec` fields after `knee_bend_offset`, the vector and mask offsets at 6 and 9 in place of 11 and 15, and the same bounds. Its only consumer of `DesignBounds`, `graphs/mass_vs_length_sweep.py`, reads named length bounds and is unaffected.

#### E3, new module `design_generator/common/servo.py`

The module lives in the design generator rather than in `co_optimisation.utils`, whose `__init__` imports Isaac Lab through `respawn.py` (`co_optimisation/co_optimisation/utils/__init__.py:1`, `respawn.py:13-19`) and would break the simulator free loading of `usd_generator.py` the tests rely on (`co_optimisation/tests/design_generator_test.py:166-176`). It carries the mechanics of `scripts/analysis/inertia_analysis.py` verbatim, `_vec`, `_defaults`, `load_model`, `_rot`, `_axis_angle`, `kinematics`, `mass_matrix`, `inertias`, `distal`, `limbs`, `_grid` and `reference_estimate` (`inertia_analysis.py:47-303`), about 230 lines of NumPy and the standard library, validated as section 3.2.2 records. The script keeps its own copy, since the host interpreter that carries MuJoCo does not install the design generator, and E8 item 10 keeps the two equal. The quantity that sets the reference is the row sum of $|W|$,

```python
def inertias(model, q):
    """(locked, free, row sum of |W|, W) over the hinges at the pose *q*."""
    M, nb = mass_matrix(model, q)
    W = np.linalg.inv(M)[nb:, nb:]
    return np.diag(M)[nb:].copy(), 1.0 / np.diag(W), np.abs(W).sum(1), W
```

and `reference_estimate(model, joints)` returns $S_j = 1/\max_q\sum_l|W_{jl}(q)|$ over a grid of three points per hinge of the first limb and three coordinate passes over every hinge, 1.4 s per biped design and 2.4 s per quadruped design on the host, against 40 s for the full sweep it approximates to the accuracy of the table of section 3.2.2. Two helpers complete the module.

```python
def servo_gains(inertia: float, frequency_hz: float, damping_ratio: float) -> tuple[float, float]:
    """(stiffness [N m/rad], damping [N m s/rad]) placing I q'' + kd q' + kp q at the given frequency and ratio."""
    omega = 2.0 * math.pi * frequency_hz
    return inertia * omega ** 2, 2.0 * damping_ratio * inertia * omega


def step_numbers(inertia: float, stiffness: float, damping: float, dt: float) -> tuple[float, float]:
    """(a, b) of the semi implicit update, a = kp dt^2 / I and b = c dt / I for a total damping slope c."""
    return stiffness * dt ** 2 / inertia, damping * dt / inertia
```

#### E4, `co_optimisation/co_optimisation/runners/usd_generator.py`

Tables. The module docstring's paragraph on stiffness and damping scales (`:27-35`) is replaced by one describing the gain keys, and the tables become

```python
from design_generator.common.servo import limbs, load_model, reference_estimate, servo_gains

ParamRange = tuple[str, tuple[float, float]]
# "absolute" writes the drawn value into the DesignSpec field as it stands.
PARAM_TYPES = ("scale", "offset", "absolute")

# Servo dynamics per actuator group, plans/w_eta_optimisation.md section 3.5. The
# biped's groups are a superset of the quadruped's, whose DesignBounds agree.
GAIN_PARAM_KEYS: tuple[str, ...] = biped.generate.GAIN_FIELDS
DEFAULT_NATURAL_FREQUENCY = 10.0   # Hz, BeyondMimic's ablated choice
DEFAULT_DAMPING_RATIO = 0.85

DEFAULT_PARAM_RANGES: dict[str, ParamRange] = {
    "thigh_length_scale": ("scale", (0.75, 1.25)),
    "shank_length_scale": ("scale", (0.75, 1.25)),
    "thigh_bend_offset": ("offset", (-0.10, 0.10)),     # rad, a lean about the solved hip angle
    "knee_bend_offset": ("offset", (-0.25, 0.25)),      # rad
    **{k: ("absolute", getattr(biped.generate.DesignBounds(), k)) for k in GAIN_PARAM_KEYS},
}

# Names rather than a second range structure, so a caller's override of a range reaches CMA-ES.
# Add any of GAIN_PARAM_KEYS to search a group's servo, for example "hip_natural_frequency".
CMAES_PARAM_KEYS: tuple[str, ...] = (
    "thigh_length_scale", "shank_length_scale", "thigh_bend_offset", "knee_bend_offset",
)

SPEC_PARAM_FIELDS: dict[str, str] = {
    "thigh_length_scale": "thigh_length",
    "shank_length_scale": "shank_length",
    "thigh_bend_offset": "thigh_bend_offset",
    "knee_bend_offset": "knee_bend_offset",
    **{k: k for k in GAIN_PARAM_KEYS},
}

DEPRECATED_PARAM_KEYS = (
    "joint_effort_scale", "velocity_limit_scale", "saturation_effort_scale",
    "armature_scale", "friction_static_scale", "friction_dynamic_scale",
    # Retired 2026-10-27 for the natural frequency and damping ratio of each group.
    "abad_stiffness_scale", "abad_damping_scale", "hip_stiffness_scale", "hip_damping_scale",
    "knee_stiffness_scale", "knee_damping_scale", "ankle_stiffness_scale", "ankle_damping_scale",
)
```

The module gains `import re` beside `import math` (`usd_generator.py:53`).

`Family`. A keyword `group_joints=None`, stored as `self.group_joints = dict(group_joints or {})`, maps each group to the regular expression of its joints, which assigns every hinge its group's armature and picks the group's joint on the first limb.

```python
        group_joints={"abad": "abad_.*_Joint", "hip": "hip_.*_Joint",
                      "knee": "knee_.*_Joint", "ankle": "ankle_.*_Joint"},
```

```python
        group_joints={"abad": ".._hip_joint", "hip": ".._thigh_joint", "knee": ".._calf_joint"},
```

`_spec_from_parameters` (`:722-725`) learns the third type.

```python
        for name, field in SPEC_PARAM_FIELDS.items():
            if name in parameters:
                kind, value = self.param_ranges[name][0], parameters[name]
                if kind == "scale":
                    fields[field] = round(fields[field] * value, 2)
                elif kind == "offset":
                    fields[field] = fields[field] + value
                else:
                    fields[field] = value
```

`RandomDesignGenerator` gains a constructor.

```python
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Servo dynamics are searched by the CMA-ES generators alone, so random
        # search keeps every catalogue gain.
        self.param_ranges = {k: v for k, v in self.param_ranges.items() if k not in GAIN_PARAM_KEYS}
```

`CMAESDesignGenerator`. The constructor takes `physics_dt: float = 0.005`, stores it, and replaces the key restriction at `:867`.

```python
        # Restrict CMA-ES to CMAES_PARAM_KEYS, less any servo key for a group this
        # family lacks, so one tuple serves both robots.
        dropped = [k for k in CMAES_PARAM_KEYS
                   if k in GAIN_PARAM_KEYS and k.split("_", 1)[0] not in self.family.groups]
        if dropped:
            print(f"[design] {self.family.name} has no group for {dropped}, not searched")
        self.param_ranges = {k: self.param_ranges[k] for k in CMAES_PARAM_KEYS if k not in dropped}
        self.physics_dt = float(physics_dt)
        self._servo_spec = self._servo_inertia = None
        self._check_servo_box()
```

The methods below sit with the internal helpers and are inherited by both subclasses. `CatCMAESDesignGenerator._design_record` already calls `super()`, so its picks and the gains both reach the log.

```python
    # --- servo dynamics -------------------------------------------------

    def _servo_groups(self, spec) -> list[str]:
        """Groups whose gains come from (f_n, zeta), being those with either field set."""
        return [g for g in self.family.groups
                if getattr(spec, f"{g}_natural_frequency", None) is not None
                or getattr(spec, f"{g}_damping_ratio", None) is not None]

    def _servo_values(self, spec, group: str) -> tuple[float, float]:
        f = getattr(spec, f"{group}_natural_frequency")
        z = getattr(spec, f"{group}_damping_ratio")
        return (DEFAULT_NATURAL_FREQUENCY if f is None else f,
                DEFAULT_DAMPING_RATIO if z is None else z)

    def _check_servo_box(self) -> None:
        """C1 of plans/w_eta_optimisation.md section 3.3.8, b = 4 pi zeta f dt <= 1 at the box corner."""
        for g in self.family.groups:
            keys = (f"{g}_natural_frequency", f"{g}_damping_ratio")
            if not any(k in self.param_ranges for k in keys):
                continue
            f_hi = self.param_ranges[keys[0]][1][1] if keys[0] in self.param_ranges else DEFAULT_NATURAL_FREQUENCY
            z_hi = self.param_ranges[keys[1]][1][1] if keys[1] in self.param_ranges else DEFAULT_DAMPING_RATIO
            b = 4.0 * math.pi * z_hi * f_hi * self.physics_dt
            if b > 1.0:
                raise ValueError(f"{g}: f_n {f_hi} Hz and zeta {z_hi} give b = {b:.3f} > 1 "
                                 f"at dt {self.physics_dt} s, narrow the range")

    def _reference_inertias(self, spec) -> dict[str, float]:
        """Reference inertia S_j of each group, section 3.2.3, at its first limb's joint. Cached for the last spec."""
        if spec is not self._servo_spec:
            patterns = self.family.group_joints
            armature = [(patterns[g], act.armature) for g, act in zip(self.family.groups, spec.actuators())]
            model = load_model(self.family.build_xml(spec, **self.builder_kwargs), armature)
            first = {g: next(j for j in limbs(model)[0] if re.fullmatch(patterns[g], j))
                     for g in self.family.groups}
            reference = reference_estimate(model, list(first.values()))
            self._servo_inertia = {g: reference[j] for g, j in first.items()}
            self._servo_spec = spec
        return self._servo_inertia

    def _actuator_params(self, spec, parameters=None) -> dict[str, dict]:
        out = super()._actuator_params(spec, parameters)
        groups = self._servo_groups(spec)
        if not groups:
            return out
        try:
            inertia = self._reference_inertias(spec)
        except ValueError:
            # The builder refuses this spec, and _build_population substitutes the last feasible one.
            return out
        for g in groups:
            kp, kd = servo_gains(inertia[g], *self._servo_values(spec, g))
            out[g]["stiffness"], out[g]["damping"] = round(kp, 6), round(kd, 6)
        return out

    def _design_record(self, spec) -> dict:
        """The searched parameters, with each servo group's inertia and gains once (f_n, zeta) set them."""
        record = super()._design_record(spec)
        groups = self._servo_groups(spec)
        if groups:
            try:
                inertia = self._reference_inertias(spec)
            except ValueError:
                return record
            for g in groups:
                kp, kd = servo_gains(inertia[g], *self._servo_values(spec, g))
                record.update({f"{g}_inertia": inertia[g], f"{g}_stiffness": kp, f"{g}_damping": kd})
        return record
```

`Population` and the base generator carry the reference inertia to the runner, which needs it for the training draws of section 3.6. `Population.__init__` takes `servo_inertias: list[dict[str, float]] | None = None`, stores it as `self._servo_inertias`, and exposes it through `get_servo_inertias()`. `DesignGeneratorBase` gains two hooks whose defaults leave every other generator unchanged, and `_build_population` collects the second one per individual after the fallback substitution (`usd_generator.py:685-693`).

```python
    def servo_search_space(self) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
        """Per searched group, the (f_n, zeta) ranges training draws cover. None searched by default."""
        return {}

    def _servo_inertia_record(self, spec) -> dict[str, float] | None:
        """The reference inertia of every group of *spec*, or None where the generator computes none."""
        return None
```

```python
        servo_inertias: list[dict[str, float] | None] = []      # beside the other lists
        ...
            servo_inertias.append(self._servo_inertia_record(spec))   # after nominal.append
        return Population(usd_files, actuator_params, link_length_params,
                          self.family.root_body, specs, nominal, parameters, xml_files,
                          servo_inertias=None if all(i is None for i in servo_inertias) else servo_inertias)
```

`CMAESDesignGenerator` overrides both. The inertia is returned for every group, searched or not, so that the runner can check the stability of a scaled catalogue gain as well.

```python
    def servo_search_space(self) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
        out = {}
        for g in self.family.groups:
            f_key, z_key = f"{g}_natural_frequency", f"{g}_damping_ratio"
            if f_key in self.param_ranges or z_key in self.param_ranges:
                out[g] = (self.param_ranges[f_key][1] if f_key in self.param_ranges
                          else (DEFAULT_NATURAL_FREQUENCY,) * 2,
                          self.param_ranges[z_key][1] if z_key in self.param_ranges
                          else (DEFAULT_DAMPING_RATIO,) * 2)
        return out

    def _servo_inertia_record(self, spec) -> dict[str, float] | None:
        try:
            return dict(self._reference_inertias(spec))
        except ValueError:
            return None
```

`_reference_inertias` calls the builder, so a spec the builder refuses raises `ValueError` there. `_actuator_params` then returns the catalogue gains and `_servo_inertia_record` returns `None`, and `_build_population` substitutes the last feasible spec as it does today and recomputes the actuator parameters of the substitute (`usd_generator.py:685-688`). No design is refused for its gains. C1 holds by construction, and C2 is reported by the runner rather than enforced (section 3.3.8), so the first version's `_check_servo` and its `_build_xml` override are not needed. Nothing changes when no group has a servo field set, since `_servo_groups` is then empty in every method.

#### E5, `scripts/rsl_rl/train.py`

Pass the physics step.

```python
            design_generator = generator_cls(
                **generator_kwargs,
                physics_dt=env.unwrapped.physics_dt,
```

The `copt` configuration switches on the gain precedence of section 3.6 for new runs (`train.py:322-330`).

```python
        agent_cfg_dict["copt"] = {
            ...
            "eval_seed": 42,
            # Section 3.6 of plans/w_eta_optimisation.md, randomised gains in training, exact gains in evaluation.
            "gain_randomisation": True,
        }
```

`scripts/rsl_rl/play.py:1198` constructs `CMAESDesignGenerator` for replay and takes the 0.005 s default, the step of both tasks.

#### E6, `co_optimisation/co_optimisation/utils/respawn.py`

Two functions are added beside `apply_actuator_params`. The first reads the task's gain randomisation, so the task configuration stays its single source. `randomize_actuator_gains` is a class term, which the event manager replaces by its instance at construction, hence the class test. The manager's term list is private to Isaac Lab (`IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py:72-73`), which the function accepts as the price of not restating the ranges.

```python
def gain_scale_ranges(env: ManagerBasedRLEnv) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    """Stiffness and damping scale ranges of the task's randomize_actuator_gains terms, per actuator group.

    Only uniform terms with the scale operation are read, the form every
    co-optimisation task configures. A group several terms select takes the last.
    """
    from isaaclab.envs.mdp.events import randomize_actuator_gains

    articulation = env.scene.articulations["robot"]
    groups = {name: set(articulation.find_joints(act.cfg.joint_names_expr)[0])
              for name, act in articulation.actuators.items()}
    out = {}
    for cfgs in env.event_manager._mode_term_cfgs.values():
        for cfg in cfgs:
            cls = cfg.func if inspect.isclass(cfg.func) else type(cfg.func)
            if not issubclass(cls, randomize_actuator_gains) or cfg.params.get("operation") != "scale":
                continue
            assert cfg.params.get("distribution", "uniform") == "uniform", cfg.params
            joints = set(articulation.find_joints(cfg.params["asset_cfg"].joint_names)[0])
            for name, ids in groups.items():
                if joints & ids:
                    out[name] = (tuple(cfg.params["stiffness_distribution_params"]),
                                 tuple(cfg.params["damping_distribution_params"]))
    return out
```

The second draws the training gains. It runs after `apply_actuator_params` has written each design's nominal gains, which are the base the scaled groups multiply.

```python
def randomise_actuator_gains(
    env: ManagerBasedRLEnv,
    population,
    scale_ranges: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    servo_space: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    dt: float,
) -> dict[str, float]:
    """Draw every environment's training gains, plans/w_eta_optimisation.md section 3.6.

    A group the search owns draws (f_n, zeta) uniformly over its box and converts
    them against its design's reference inertia, so every draw satisfies C1. Any
    other group scales its design's nominal gains by the task's own ranges.
    Returns, per group with a known inertia, the share of environments whose draw
    fails the sufficient test a + 2b < 4 at that inertia with the friction slope,
    section 3.3.7.
    """
    articulation = env.scene.articulations["robot"]
    inertias = population.get_servo_inertias()
    n = len(population.get_actuator_params())
    individual = torch.arange(env.num_envs, device=env.device) % n
    unstable: dict[str, float] = {}
    for group, act in articulation.actuators.items():
        def uniform(lo: float, hi: float) -> torch.Tensor:
            return lo + (hi - lo) * torch.rand(act.stiffness.shape, device=env.device)

        inertia = None
        if inertias is not None and all(i is not None and group in i for i in inertias):
            inertia = torch.tensor([i[group] for i in inertias], device=env.device)[individual].unsqueeze(1)
        if group in servo_space and inertia is not None:
            (f_lo, f_hi), (z_lo, z_hi) = servo_space[group]
            omega = 2.0 * math.pi * uniform(f_lo, f_hi)
            act.stiffness[:] = inertia * omega ** 2
            act.damping[:] = 2.0 * uniform(z_lo, z_hi) * inertia * omega
        elif group in scale_ranges:
            (k_lo, k_hi), (d_lo, d_hi) = scale_ranges[group]
            act.stiffness.mul_(uniform(k_lo, k_hi))
            act.damping.mul_(uniform(d_lo, d_hi))
        if inertia is not None:
            slope = act.friction_static / act.activation_vel + act.friction_dynamic
            a = act.stiffness * dt ** 2 / inertia
            b = (act.damping + slope) * dt / inertia
            unstable[group] = float((a + 2.0 * b >= 4.0).float().mean())
    return unstable
```

`respawn.py` already imports `torch` and `inspect` (`respawn.py:11-13`), and gains `import math`.

#### E7, `co_optimisation/co_optimisation/runners/copt_on_policy_runner.py`

The configuration key, read with the others (`copt_on_policy_runner.py:126-140`), and the ranges, read once the environment exists.

```python
        # Training gains drawn about each design, and exact gains during the evaluation,
        # plans/w_eta_optimisation.md section 3.6. False reproduces every run to date.
        self._gain_randomisation: bool = copt_cfg.get("gain_randomisation", False)
```

```python
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._gain_scale_ranges = (gain_scale_ranges(self.env.unwrapped)
                                   if self._gain_randomisation else {})
```

The import at `copt_on_policy_runner.py:36` adds `gain_scale_ranges` and `randomise_actuator_gains`. One method draws the training gains. It is called directly after each of the three `apply_actuator_params` calls, in `_reload_morphology` (`:618-620`), `_apply_restored_state` (`:662-664`) and `_update_morphology` (`:772-774`).

```python
    def _apply_training_gains(self) -> None:
        """Replace the nominal gains just written by each environment's training draw, section 3.6."""
        if not self._gain_randomisation or self.current_population is None:
            return
        env = self.env.unwrapped
        unstable = randomise_actuator_gains(
            env, self.current_population, self._gain_scale_ranges,
            self._design_generator.servo_search_space(), env.physics_dt,
        )
        for group, share in unstable.items():
            if share > 0.0:
                print(f"[gains] {group}: {share:.1%} of environments drew gains that fail the "
                      "explicit servo's sufficient stability test at their reference inertia")
```

`_evaluate_population` writes the exact gains before its first pass and restores the training draws in its existing `finally` block (`:943-956`). Only the two gain keys are written, so the armature write of `apply_actuator_params` is not repeated mid-run.

```python
        # Each design is scored at its own gains, section 3.6 of plans/w_eta_optimisation.md,
        # and the training draws are put back afterwards.
        trained_gains = None
        if self._gain_randomisation:
            trained_gains = {name: (act.stiffness.clone(), act.damping.clone())
                             for name, act in robot.actuators.items()}
            apply_actuator_params(env, [
                {g: {k: p[k] for k in ("stiffness", "damping") if k in p} for g, p in params.items()}
                for params in self.current_population.get_actuator_params()
            ])
```

```python
            if trained_gains is not None:
                for name, (stiffness, damping) in trained_gains.items():
                    robot.actuators[name].stiffness.copy_(stiffness)
                    robot.actuators[name].damping.copy_(damping)
```

The checkpoint carries the inertias, so a resumed run draws as the original did. `_population_to_dict` adds `"servo_inertias": pop.get_servo_inertias()`, and `_population_from_dict` passes `servo_inertias=state.get("servo_inertias")`, which is `None` for every checkpoint written before this change.

#### E8, tests in `co_optimisation/tests/design_generator_test.py`

1. The inertia module against the numbers of section 3.2.2, read from a document emitted for a fixed spec, MuJoCo being unavailable in the container. `reference_estimate` gives the biped hip 0.0247 and the quadruped knee 0.0096 within 1e-4 for generation 8, individual 0, of the two runs named there, with the catalogue armature.
2. With `CMAES_PARAM_KEYS` unchanged, every CMA-ES generator's actuator parameters equal `sim_params()` of the picked entry for every group, the regression guard for earlier runs.
3. With `hip_natural_frequency` enabled through a monkeypatched `CMAES_PARAM_KEYS`, the hip's stiffness and damping equal `servo_gains(S_hip, f, 0.85)` with `S_hip` the generator's reference inertia and every other group keeps its catalogue gains.
4. `ankle_natural_frequency` enabled on the quadruped is dropped from `param_keys`. The assertion at line 216 becomes a comparison against the family filtered tuple.
5. A range of $[6, 20]$ Hz with $\zeta \le 1.0$ raises at construction.
6. `DesignSpec.from_vector` of a fifteen entry biped vector leaves the gain fields `None`, and `vector()` round trips a spec with gains set.
7. `test_ranges_are_typed_and_the_untyped_form_reads_as_scale` gains a case for `hip_stiffness_scale`, which must now raise.
8. `servo_search_space()` is empty with `CMAES_PARAM_KEYS` unchanged, and with `hip_natural_frequency` enabled returns the hip's frequency range and the degenerate damping ratio range $(0.85, 0.85)$.
9. Every CMA-ES generator's population carries `servo_inertias` with every group of its family, `RandomDesignGenerator`'s carries `None`, and `_population_from_dict(_population_to_dict(pop))` preserves them.
10. `design_generator.common.servo` and `scripts/analysis/inertia_analysis.py`, loaded by path, return the same `inertias` at three random poses of both documents, so the two copies cannot drift.

`gain_scale_ranges` and `randomise_actuator_gains` need a live environment and are exercised by step 6 of section 3.9.

`scripts/analysis/cmaes_analysis.py` reads parameter names from the history's `mean/` keys and recovers each range from the linear relation of `search_statistics` (`usd_generator.py:919-932`), which the linear encoding preserves, so it plots the new variables without change.

### 3.9 Implementation instructions

1. Apply E1 to E3 in the `design_generator` submodule and commit there. Apply E4 to E8 in the repository.
2. In both containers, confirm `design_generator` is installed editable, `pip install -e design_generator` (`README.md:56`), and reinstall `co_optimisation` before any launch outside `djinn`, `pip install --no-deps --no-build-isolation /workspace/isaaclab/biped/co_optimisation` (`/ws/context/copt.md:399`).
3. Run `pytest co_optimisation/tests/design_generator_test.py` in the training container. Every existing test must pass unchanged, except the line 216 assertion amended by E8 item 4.
4. Confirm the coupled prediction of section 3.3.8 in Isaac Lab before relying on it. Spawn the biped's baseline design alone in a play session with gravity disabled and the base high above the ground, set its joints to the worst pose listed in `validate_coupled.txt`, hold the action at that pose with the catalogue gains, and log the joint velocities for 0.5 s after a small perturbation. A velocity alternating in sign and growing to the effort limit confirms the chatter. Repeat at the zero pose, where the same perturbation must decay, and repeat both at 10 Hz and 0.85 against the reference inertias, where both must decay.
5. Launch a 1024 environment smoke run of each family with `CMAES_PARAM_KEYS` unchanged and `gain_randomisation` set to `False` and compare the `stiffness` and `damping` rows of the actuators after the first generation with a run from before the change. They must be equal.
6. Repeat with `gain_randomisation` on. Before the first evaluation, the `stiffness` rows of one design's environments must differ from one another and lie within the task's scales of the catalogue value. During `_evaluate_population`, they must equal the catalogue value exactly, and after it they must equal the draws again. The `[gains]` lines must name the biped abduction, hip and knee and no quadruped group, per section 3.6.
7. Add `"hip_natural_frequency", "hip_damping_ratio"` to `CMAES_PARAM_KEYS` and repeat. The TensorBoard `CMAES/mean/` and `CMAES/std/` heads must show the two new coordinates starting at 10 Hz and 0.85, and the design CSV must carry `hip_inertia`, `hip_stiffness` and `hip_damping`.
8. Record the outcome in `/ws/context/copt.md` and set this plan's status in `plans/README.md`.

### 3.10 Friction, and what it will demand when transfer matters

The friction model. `IdentifiedActuator` subtracts

$$\tau_f(\dot q) = \mu_s\tanh(\dot q/\dot q_s) + \mu_d\dot q$$

from the servo torque (`actuator_pd.py:33-35`). $\mu_s$ is the Coulomb level [N m], `friction_static`, $\dot q_s$ the velocity over which it builds [rad/s], `activation_vel`, and $\mu_d$ the viscous coefficient [N m s/rad], `friction_dynamic`. This is the form Grandia and colleagues identify on a bench [9]. The catalogue's values have no recorded identification (section 1.3).

Small velocities. Linearising at rest, $\tau_f \approx (\mu_s/\dot q_s + \mu_d)\dot q$. Near standstill, friction is a damper of slope $c_f = \mu_s/\dot q_s + \mu_d$, 8.02 at the biped knee against a catalogue $k_d$ of 4, so the effective damping ratio there is

$$\zeta_{\mathrm{eff}} = \frac{k_d + c_f}{2\sqrt{k_p I}},$$

well above the searched $\zeta$. The same slope enters the explicit update. Against the biped's reference inertias it is a stiff explicit damper on its own, $c_f\Delta t/S_j$ = 1.72 at the abduction and 2.18 at the knee, but only while $|\dot q| < \dot q_s$. A mode it destabilises grows until the velocity leaves that band, and the describing function below then gives the slope's equivalent, $4\mu_s/(\pi\hat q\,\Omega)$, which falls as the amplitude grows. The mode settles where that equivalent has fallen back to the stability limit, a velocity of the order of $\dot q_s$ = 0.1 rad/s on the biped. That bounded limit cycle is why C2 reports rather than refuses.

Large velocities. For $|\dot q| \gg \dot q_s$, the tanh saturates and friction becomes a constant torque $\mu_s\,\mathrm{sign}(\dot q)$ plus the viscous term. For a sinusoid of amplitude $\hat q$ and frequency $\Omega$, a constant torque dissipates per cycle what a viscous damper of $c_{\mathrm{eq}} = 4\mu_s/(\pi\hat q\,\Omega)$ would. That is the describing function equivalent, standard and not taken from the surveyed works. Coulomb friction therefore damps small fast motions strongly and large ones weakly, so a single $\zeta$ cannot describe the joint.

Stiction and steady error. A joint at rest stays put until the servo torque exceeds the friction, so a steady position error up to $\mu_s/k_p$ remains, 0.0133 rad at the biped knee's catalogue gains. A search that lowers $f_n$ lowers $k_p$ and widens this band in proportion to $1/f_n^2$.

Viscous friction in the gain map. The identified joint of Bjelonic and colleagues carries $d + D_\tau$ in its damping term [3], so once $\mu_d$ is known the derivative gain that realises a target $\zeta$ is $k_d = 2\zeta I\omega_n - \mu_d$.

Later work, in order of need.

1. Identify the friction triple and the command delay of the TRON1 and A1 drives, by the in air chirp procedure of Bjelonic and colleagues [3] or by sampling against logged trajectories [10]. The delay matters as much as friction, given the transfer failures of [6] and [7].
2. Subtract the identified $\mu_d$ from $k_d$ in `servo_gains`, and evaluate $\zeta$ with $c_f$ at the speeds the gait reaches rather than at rest.
3. Add a stiffness floor $k_p \ge \mu_s/e_{\mathrm{tol}}$ for a tolerated steady error $e_{\mathrm{tol}}$, beside the load floor of `../context/KScale.md` section 11.1.
4. Randomise the friction triple within its identified range, in the training draws of section 3.6.
5. If a delay is identified inside the loop rather than on the command, add the delay margin it implies to C2, since the stability analysis of section 3.3 assumes none, and a delay inside the loop adds a characteristic polynomial of higher degree.

## 4. Conclusion

The plan makes the natural frequency and damping ratio of each actuator group search variables of `CMAESDesignGenerator`, `GrowingDesignDistCMAESDesignGenerator` and `CatCMAESDesignGenerator`. Their gains are computed as $k_p = S_j(2\pi f_n)^2$ and $k_d = 2\zeta S_j(2\pi f_n)$, where the reference inertia $S_j$ is the reciprocal of the largest row sum of the inverse mass matrix over the pose space, computed per design from the emitted MJCF with the picked actuators' armature.

The choice of reference follows from the state transition of the explicit servo in section 3.3. For one joint, its stability conditions $0 < b < 2$ and $a + 2b < 4$ define a triangle that is closed under shrinking towards the origin, and a larger inertia shrinks the point, so the smallest inertia is the worst case and against it the design cancels, leaving $a = (2\pi f_n\Delta t)^2$ and $b = 4\pi\zeta f_n\Delta t$. For the robot, the inertial coupling of the joints replaces the ratio of inertias by the eigenvalues of $SW$, and a Lyapunov energy extends $a + 2b < 4$ to any mix of gains. The pose sweep of section 3.2 showed that the minimum locked inertia, the first version's reference, misses an interior minimum and leaves the biped's coupled modes ten times beyond the box, while the reference $S_j$ bounds them by Gershgorin's theorem at every pose, in stance as well as swing. The KScale damping rule is kept in substance, its swing cap and integrator bound becoming the box $\zeta \le 1$ and $4\pi\zeta f_n\Delta t \le 1$, the search replacing its fixed stance target, and the clamp it would need being avoided because a clamp would flatten the landscape CMA-ES must climb. The search starts at the requested 10 Hz and 0.85 within $[6, 14]$ Hz and $[0.7, 1.0]$, which stays within every group's effort limit, so no family needs an override.

During PPO training each environment draws its gains, over the search box for a searched group and by the task's own scales about the catalogue gains for any other, so the policy learns to handle the gains the search may propose. During the CMA-ES evaluation each design runs at exactly its own gains. A runner switch that defaults to today's behaviour carries this, and it repairs the startup gain randomisation that no co-optimisation run has yet experienced. With `CMAES_PARAM_KEYS` unchanged and the switch off, every generator and the runner behave exactly as today, so earlier runs stay comparable.

Three matters are for the user. The first is the biped's catalogue gains, whose damping is unstable at a large share of reachable poses once the coupling is counted, a defect every biped run to date has carried and which the randomisation would widen. The recommendation is to search, or at least to fix, the biped's damping ratios, adding its `*_damping_ratio` keys to `CMAES_PARAM_KEYS` for biped runs, once step 4 of section 3.9 has confirmed the prediction in Isaac Lab. The second is the identified friction, which alone keeps the biped abduction and knee above the sufficient test at any gains in the box, in a bounded way the runner reports. The third is whether the policy should observe the gains, as Bohlinger and Peters' policy does [15]. Friction identification, contact phase gain schedules [27] and an energy term in the fitness [25] are the next steps once transfer becomes the aim.

## Notation

Section 2 quotes each surveyed paper's equations in that paper's own symbols, which are local to the quotation and are not listed here. Every symbol of sections 1, 3 and 4 is listed below, with the section that introduces it. Vectors and matrices are written in world axes unless stated.

| Symbol | Meaning | Unit | Introduced |
|---|---|---|---|
| $q$, $q_j$ | pose, the vector of hinge angles, and the angle of hinge $j$, written $q$ for a single joint | rad | 3.1, 3.2.1 |
| $q^*$ | target angle commanded by the policy, held over a control step | rad | 3.1 |
| $\dot q$, $\ddot q$ | hinge velocity and acceleration | rad/s, rad/s² | 3.1 |
| $\tau$, $\tau_j$ | hinge torque, vector or of hinge $j$ | N m | 3.1, 3.2.1 |
| $k_p$, $k_d$ | servo stiffness and damping gains of one joint, $k_{p,j}$ and $k_{d,j}$ for hinge $j$ | N m/rad, N m s/rad | 3.1 |
| $I$ | inertia a single joint drives | kg m² | 3.1 |
| $I'$, $\omega_n'$, $\zeta'$ | another inertia met by the same gains, and the natural frequency and damping ratio there | kg m², rad/s, none | 3.1 |
| $I_{\mathrm{ref}}$ | inertia a single joint's gains are computed against | kg m² | 3.3.6 |
| $\omega_n$, $f_n$ | natural frequency, $\omega_n = 2\pi f_n$, and $\omega_j$, $f_{n,j}$ for hinge $j$ | rad/s, Hz | 3.1 |
| $\zeta$, $\zeta_j$ | damping ratio, shared or of hinge $j$ | none | 3.1 |
| $\mathcal{H}$, $N$ | set of hinges and their number | none | 3.2.1 |
| $j$, $l$ | hinge indices | none | 3.2.1 |
| $\nu$, $\mathcal{B}_j$ | body index, and the set of bodies below hinge $j$ | none | 3.2.1 |
| $m_\nu$, $\bar p_\nu$ | mass and centre of mass position of body $\nu$ | kg, m | 3.2.1 |
| $I_\nu$, $R_\nu$ | inertia tensor of body $\nu$ in its inertial frame, and that frame's orientation | kg m², none | 3.2.1 |
| $\hat a_j$, $p_j$ | unit axis of hinge $j$ and a point on it | none, m | 3.2.1 |
| $A_j$, $A$ | armature of hinge $j$, and their diagonal matrix | kg m² | 3.2.1 |
| $P_j$ | orthogonal projector onto the plane normal to $\hat a_j$, $\mathbb{1} - \hat a_j\hat a_j^\top$ | none | 3.2.1 |
| $\mathbb{1}$ | identity matrix of the size the context requires | none | 3.2.1 |
| $v$, $v_B$ | generalised velocity of the floating robot, and the base twist | mixed | 3.2.1 |
| $M(q)$ | joint space mass matrix of the floating robot, $(6+N)\times(6+N)$ | mixed | 3.2.1 |
| $J^v_\nu$, $J^\omega_\nu$ | Jacobians of body $\nu$'s centre of mass velocity and angular velocity | mixed | 3.2.1 |
| $I^{\mathrm{lock}}_j$, $I^{\mathrm{free}}_j$ | inertia of hinge $j$ with every other coordinate held, and free | kg m² | 3.2.1 |
| $r$ | as a subscript, every coordinate of $v$ except $q_j$, in the partition of $M$ | none | 3.2.1 |
| $W(q)$, $W_{jl}$ | hinge block of $M^{-1}$, the inverse inertia the servos act through, and its entries | 1/(kg m²) | 3.2.1 |
| $W_G$ | $W$ with a contact or a fixed base enforced | 1/(kg m²) | 3.2.1 |
| $G$, $f$ | constraint Jacobian of a contact or fixed base, and the constraint force | mixed | 3.2.1 |
| $[\hat a]_\times$ | skew symmetric matrix with $[\hat a]_\times r = \hat a \times r$ | none | 3.2.1 |
| $R(\hat a, \theta)$, $R_l(\theta)$ | rotation by $\theta$ about $\hat a$, and about the axis of hinge $l$ | none | 3.2.1 |
| $\theta$ | angle of the one distal hinge turned in section 3.2.1 | rad | 3.2.1 |
| $r_\nu$, $d$, $\bar r_l$, $\bar m_l$, $\varphi$ | offset of body $\nu$ from axis $l$, perpendicular offset between the axes of $j$ and $l$, perpendicular offset and mass of everything below $l$, and the phase of the sinusoid | m, m, m, kg, rad | 3.2.1 |
| $\bar p^0_\nu$, $R^0_\nu$, $\bar I_\nu$ | centre of mass, orientation and world inertia tensor of body $\nu$ at $\theta = 0$ | m, none, kg m² | 3.2.1 |
| $h_0, h_1, g_1, h_2, g_2$ | coefficients of the trigonometric law of the locked inertia | kg m² | 3.2.1 |
| $X$, $Y$ | generic square matrices in a stated property | as stated | 3.2.1 |
| $X \preceq Y$ | $Y - X$ is positive semidefinite | none | 3.2.1 |
| $S_j$, $S$ | reference inertia of hinge $j$, and their diagonal matrix | kg m² | 3.2.3 |
| $\lambda_{\max}(X)$ | largest eigenvalue of a matrix $X$ with real eigenvalues | as $X$ | 3.2.3 |
| $\lambda_{\min}(X)$ | smallest eigenvalue of a symmetric matrix $X$ | as $X$ | 3.3.7 |
| $\Delta t$ | physics step, 0.005 s | s | 3.3.1 |
| $k$, $x[k]$ | physics step index, and a quantity at the start of step $k$ | none | 3.3.2 |
| $e$ | joint error $q - q^*$, vector for the robot | rad | 3.3.2 |
| $c$, $c_j$, $c_f$ | total damping slope of a joint or of hinge $j$, and the friction slope at rest | N m s/rad | 3.3.2 |
| $a$, $b$ | stiffness and damping per step, $k_p\Delta t^2/I$ and $c\Delta t/I$ | none | 3.3.2 |
| $\Phi$ | state transition matrix, $2\times 2$ for one joint and $2N\times 2N$ for the robot | mixed | 3.3.2, 3.3.7 |
| $\lambda$, $\lambda_1$, $\lambda_2$ | eigenvalues of $\Phi$ | none | 3.3.3 |
| $\psi_i$, $\Psi$, $\gamma_i$, $Z$ | eigenvectors of $\Phi$ and their matrix, arbitrary expansion constants, and a Jordan block | mixed | 3.3.3 |
| $\rho(X)$ | spectral radius, the largest modulus of the eigenvalues of $X$ | none | 3.3.3 |
| $\phi$, $\epsilon$, $C$ | phase of a complex eigenvalue, an arbitrary positive number, and a constant of Gelfand's bound | rad, none, none | 3.3.3 |
| $s$ | Laplace variable, a continuous pole | 1/s | 3.3.3 |
| $p(\lambda)$, $\alpha_0$, $\alpha_1$ | characteristic polynomial of $\Phi$, and the coefficients of a generic monic quadratic | none | 3.3.4 |
| $\varepsilon$ | $\lambda - 1$, equal to $s\Delta t$ for small steps | none | 3.3.4 |
| $\mathcal{T}$ | stable triangle of $(a, b)$ | none | 3.3.4 |
| $\kappa$ | inertia ratio $I_{\mathrm{ref}}/I$ of a single joint | none | 3.3.4, 3.3.6 |
| $b_{\mathrm{servo}}$ | servo share of $b$, $k_d\Delta t/I$ | none | 3.3.6 |
| $a_{\mathrm{ref}}$, $b_{\mathrm{ref}}$ | $a$ and $b_{\mathrm{servo}}$ at the reference inertia, $(2\pi f_n\Delta t)^2$ and $4\pi\zeta f_n\Delta t$, evaluated at the box corner where stated | none | 3.3.6 |
| $K$, $D$, $D_{\mathrm{servo}}$ | diagonal matrices of the hinges' stiffnesses, total damping slopes, and damping gains alone | N m/rad, N m s/rad | 3.3.7 |
| $i$, $\mu_i$, $u_i$, $U$, $Q$ | mode index, eigenvalues and eigenvectors of $WS$, their matrix, and the orthogonal eigenvectors of $S^{1/2}WS^{1/2}$ | none | 3.3.7 |
| $\tilde e_i$, $\tilde v_i$ | modal amplitudes of the error and the velocity | rad, rad/s | 3.3.7 |
| $a_i$, $b_i$ | per step numbers of mode $i$ | none | 3.3.7 |
| $a_j$, $b_j$ | single joint numbers of hinge $j$ against $S_j$, $\Delta t^2k_{p,j}/S_j$ and $\Delta t\,c_j/S_j$, equal to $(\omega_j\Delta t)^2$ and $2\zeta_j\omega_j\Delta t$ for the plan's gains without friction | none | 3.3.7 |
| $f_{n,\min}$, $f_{n,\max}$, $\zeta_{\min}$, $\zeta_{\max}$, $\omega_{\max}$ | lower and upper ends of a group's searched ranges, and $\omega_{\max} = 2\pi f_{n,\max}$ | Hz, none, rad/s | 3.3.7 |
| $y$, $\hat K$, $\hat D$ | scaled error $W^{-1/2}e$, and the scaled gains $\Delta t^2W^{1/2}KW^{1/2}$ and $\Delta tW^{1/2}DW^{1/2}$ of case B | rad (kg m²)^{1/2}, none, none | 3.3.7 |
| $W^{1/2}$, $V$, $\varpi_j$ | symmetric square root of $W$, and the eigenvectors and eigenvalues of $W$ | mixed | 3.3.7 |
| $\delta$, $\delta'$, $\bar y$, $E$, $\Gamma$ | change of $y$ over a step and over the next step, the midpoint of a step, the energy of case B, and its kinetic weight $\mathbb{1} - \frac{1}{4}\hat K - \frac{1}{2}\hat D$ | mixed | 3.3.7 |
| $z$, $\Phi_y$, $L$ | pair $(y[k], y[k-1])$, its transition matrix, and the invertible map $z = Lx$ | mixed | 3.3.7 |
| $w_{\mathrm{R}}$, $w_{\mathrm{I}}$ | real and imaginary parts of a complex vector | as the vector | 3.3.7 |
| $\gamma$, $C_E$, $\xi$, $\eta$, $y_0$ | smallest eigenvalue of $\Gamma$ and $\hat K$, the constant 4 of the state bound, placeholder vectors of the polarisation identity, and the $y$ part of the eigenvector $L\psi$ of $\Phi_y$ | mixed | 3.3.7 |
| $\Lambda$, $\beta$, $\Lambda_{\mathrm{servo}}$, $\beta_{\mathrm{servo}}$ | coupled stability sum $\lambda_{\max}(W(\Delta t^2K + 2\Delta tD))$ and damping per step $\lambda_{\max}(\Delta tWD)$, and the same with $D_{\mathrm{servo}}$ in place of $D$ | none | 3.3.7 |
| $\tau_f$, $\zeta_{\mathrm{eff}}$ | friction torque of `IdentifiedActuator`, and the damping ratio with its slope included | N m, none | 3.10 |
| $\mu_s$, $\mu_d$, $\dot q_s$ | Coulomb friction level, viscous friction coefficient, and the velocity over which the Coulomb term builds | N m, N m s/rad, rad/s | 3.3.8, 3.10 |
| $\tau_{\max}$ | effort limit of a group | N m | 3.5 |
| $m_{\mathrm{rand}}$, $I_{\mathrm{swing}}$, $I_{\mathrm{stance}}$, $\zeta_{\mathrm{swing}}$, $\dot q_{p95}$, $\tau_{\mathrm{limit}}$ | quantities of the KScale damping rule, as `../context/KScale.md` section 11.1 defines them | mixed | 3.4 |
| $\hat q$, $\Omega$, $c_{\mathrm{eq}}$, $e_{\mathrm{tol}}$ | amplitude and frequency of a sinusoid, the describing function's equivalent damping, and a tolerated steady error | rad, rad/s, N m s/rad, rad | 3.10 |

## Bibliography

1. Liao, Q., Truong, T. E., Huang, X., Gao, Y., Tevet, G., Sreenath, K., Liu, C. K. BeyondMimic, From Motion Tracking to Versatile Humanoid Control via Guided Diffusion. arXiv preprint, 2025, venue not stated by the retrieved record. arXiv:2508.08241.
2. NVIDIA. Gain Tuner Extension. Isaac Sim 5.1 documentation, docs.isaacsim.omniverse.nvidia.com/5.1.0/robot_setup/ext_isaacsim_robot_setup_gain_tuner.html, retrieved 2026-10-26, no author list or year stated.
3. Bjelonic, F., Tischhauser, F., Hutter, M. Towards bridging the gap, Systematic sim-to-real transfer for diverse legged robots. The International Journal of Robotics Research, 2026. arXiv:2509.06342.
4. NVIDIA. Tutorial 6, Joint Gains Tuning. Isaac Sim OpenUSD Tuning Tutorials, docs.isaacsim.omniverse.nvidia.com, no author list or year stated.
5. Sleiman, J. P., Li, H., and 26 further authors. ZEST, Zero-shot Embodied Skill Transfer for Athletic Robot Control. arXiv preprint, 2026. arXiv:2602.00401.
6. Tan, J., Zhang, T., Coumans, E., Iscen, A., Bai, Y., Hafner, D., Bohez, S., Vanhoucke, V. Sim-to-Real, Learning Agile Locomotion For Quadruped Robots. Robotics, Science and Systems XIV, 2018. arXiv:1804.10332.
7. Hwangbo, J., Lee, J., Dosovitskiy, A., Bellicoso, D., Tsounis, V., Koltun, V., Hutter, M. Learning agile and dynamic motor skills for legged robots. Science Robotics 4(26), eaau5872, 2019. arXiv:1901.08652, DOI 10.1126/scirobotics.aau5872.
8. Liao, Q., Zhang, B., Huang, X., Huang, X., Li, Z., Sreenath, K. Berkeley Humanoid, A Research Platform for Learning-based Control. arXiv preprint, 2024. arXiv:2407.21781.
9. Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller, Bächer, initials not stated by the retrieved record. Design and Control of a Bipedal Robotic Character. Robotics, Science and Systems XX, 2024. DOI 10.15607/RSS.2024.XX.103, arXiv:2501.05204.
10. Sobanbabu, He, He, Yang, Shi, initials not stated by the retrieved record. Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning. Conference on Robot Learning 2025, PMLR 305. arXiv:2505.14266.
11. Yu, W., Kumar, V. C. V., Turk, G., Liu, C. K. Sim-to-Real Transfer for Biped Locomotion. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2019. arXiv:1903.01390.
12. Masuda, Takahashi, initials not stated by the retrieved record. Sim-to-Real Transfer of Compliant Bipedal Locomotion on Torque Sensor-Less Gear-Driven Humanoid. IEEE-RAS International Conference on Humanoid Robots 2023. arXiv:2204.03897.
13. Wensing, Kim, Slotine, initials not stated by the retrieved record. Linear Matrix Inequalities for Physically Consistent Inertial Parameter Identification. IEEE Robotics and Automation Letters 3(1), 60 to 67, 2018. DOI 10.1109/LRA.2017.2729659, arXiv:1701.04395.
14. Digumarti, K. M., Gehring, C., Coros, S., Hwangbo, J., Siegwart, R. Concurrent Optimization of Mechanical Design and Locomotion Control of a Legged Robot. 17th International Conference on Climbing and Walking Robots (CLAWAR) 2014. EPFL Infoscience record 200255, no DOI stated by the retrieved record.
15. Bohlinger, N., Peters, J. Shape Your Body, Value Gradients for Multi-Embodiment Robot Design. arXiv preprint, 2026. arXiv:2606.00702.
16. Chen, T., He, Z., Ciocarlie, M. Hardware as Policy, Mechanical and Computational Co-Optimization using Deep Reinforcement Learning. Conference on Robot Learning 2020, PMLR 155. arXiv:2008.04460.
17. Fadini, Flayols, Del Prete, Souères, initials not stated by the retrieved record. Simulation Aided Co-Design for Robust Robot Optimization. IEEE Robotics and Automation Letters 7(4), 11306 to 11313, 2022. DOI 10.1109/LRA.2022.3200142.
18. Bravo-Palacios, Li, Wensing, initials not stated by the retrieved record. Engineering Compliance in Legged Robots via Robust Co-Design. IEEE/ASME Transactions on Mechatronics, 2024. NSF Public Access Repository 10629373.
19. Bogdanovic, M., Khadiv, M., Righetti, L. Learning Variable Impedance Control for Contact Sensitive Tasks. arXiv preprint, venue not stated by the retrieved record. arXiv:1907.07500.
20. Martín-Martín, R., Lee, M. A., Gardner, R., Savarese, S., Bohg, J., Garg, A. Variable Impedance Control in End-Effector Space, An Action Space for Reinforcement Learning in Contact-Rich Tasks. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2019. arXiv:1906.08880.
21. robosuite. Controllers documentation, robosuite.ai/docs/modules/controllers.html, retrieved 2026-10-26, no author list or year stated.
22. Spoljaric, D., Yan, Y., Lee, D. Variable Stiffness for Robust Locomotion through Reinforcement Learning. IFAC Symposium on Robotics, 2025. arXiv:2502.09436.
23. Radosavovic and colleagues, full author list not established by the retrieved record. Real-world humanoid locomotion with reinforcement learning. Science Robotics, 2024. arXiv:2303.03381.
24. Authors not established by the retrieved record. Learning Humanoid Locomotion over Challenging Terrain. arXiv preprint, 2024. arXiv:2410.03654.
25. Kim, M., Kwon, N., Kim, J.-Y. GainAdaptor, Learning Quadrupedal Locomotion with Dual Actors for Adaptable and Energy-Efficient Walking on Various Terrains. arXiv preprint, 2024. arXiv:2412.09520.
26. Roveda, L., Maskani, J., Franceschi, P., Abdi, A., Braghin, F., Molinari Tosatti, L., Pedrocchi, N. Model-Based Reinforcement Learning Variable Impedance Control for Human-Robot Collaboration. Journal of Intelligent and Robotic Systems, 2020. DOI 10.1007/s10846-020-01183-3.
27. Hammoud, B., Khadiv, M., Righetti, L. Impedance Optimization for Uncertain Contact Interactions Through Risk Sensitive Optimal Control. IEEE Robotics and Automation Letters 6(3), 4766 to 4773, 2021. DOI 10.1109/LRA.2021.3068951, arXiv:2011.04684.
28. Bronars, A., Park, Y., Agrawal, P. Tune to Learn, How Controller Gains Shape Robot Policy Learning. arXiv preprint, 2026. arXiv:2604.02523.
29. Chakraborty, K., Rajendra, C. K., Alharbat, A., Mersha, A. Y. Actuator Dynamics Curricula for Narrow-Viability Tasks in Legged Robot Learning. Conference on Robot Learning 2026, per the authors' listing. arXiv:2609.09492.
30. Guan, N., Yu, S., Zhu, S., Kim, D. Impedance Matching, Enabling an RL-Based Running Jump in a Quadruped Robot. Ubiquitous Robots 2024. arXiv:2404.15096.
31. Åström, Hägglund, initials not stated by the retrieved record. Automatic Tuning of Simple Regulators with Specifications on Phase and Amplitude Margins. Automatica 20(5), 645 to 651, 1984. DOI 10.1016/0005-1098(84)90014-1.
32. Berkenkamp, F., Schoellig, A. P., Krause, A. Safe Controller Optimization for Quadrotors with Gaussian Processes. IEEE International Conference on Robotics and Automation (ICRA) 2016. arXiv:1509.01066.
33. Marco and colleagues, full author list and venue not established by the retrieved record. Automatic LQR Tuning based on Gaussian Process Global Optimization. Preprint hosted at ps.is.tuebingen.mpg.de, marco-MLPC15.
34. Colgate, Schenkel, initials not stated by the retrieved record. Passivity of a Class of Sampled-Data Systems, Application to Haptic Interfaces. Journal of Robotic Systems 14(1), 37 to 47, 1997. No DOI stated by the retrieved record.
35. Hansen, N. The CMA Evolution Strategy, source code page. cma-es.github.io/cmaes_sourcecode_page.html, retrieved 2026-10-26, no year stated.
36. Featherstone, R. Rigid Body Dynamics Algorithms. Springer, 2008, chapter 6, section 6.1, the joint space inertia matrix. DOI 10.1007/978-1-4899-7560-7.
37. Murray, R. M., Li, Z., Sastry, S. S. A Mathematical Introduction to Robotic Manipulation. CRC Press, Boca Raton, 1994, chapter 2, equation 2.14, Rodrigues' formula. DOI 10.1201/9781315136370.
38. Horn, Johnson, initials not stated by the retrieved record. Matrix Analysis, second edition. Cambridge University Press, 2012 by the publisher's listing and 2013 by library catalogues, chapter 6 for Gershgorin's disc theorem and chapter 5 for the spectral radius and Gelfand's formula. ISBN 978-0-521-83940-2, the DOI not established by the retrieved record.
