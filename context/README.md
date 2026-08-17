# Context Index

This directory holds the accumulated factual record of the simulation repository, the documents that establish the physical properties of the robots, the derivations behind their actuator configuration, and the mathematics of the learning architectures implemented here. A context document is descriptive rather than prescriptive, it records findings verified against the assets, the configuration, and the source, and it carries file and symbol citations so that a later reader need not repeat the derivation. Documents that propose changes rather than record facts belong in [../plans](../plans/README.md) instead.

The scope of this directory is deliberately narrow. It holds only what is specific to this repository, the robots it defines and the algorithms it implements. The wider investigation record of the workspace, covering the co-optimisation pipeline, the Isaac Lab and rsl_rl internals, the experiment history, and the literature survey, lives one level up in `../../context/` and is indexed by `../../context/README.md`. A reader seeking why a training run behaved as it did should look there, a reader seeking how this robot is parameterised should look here.

## Reading protocol

Read only the documents the register names as bearing on the matter at hand. The four are independent of one another and none obliges reading another, with the single exception that [MorAL.md](MorAL.md) refers the reader to [HIM.md](HIM.md) for the contrastive estimator it declines to re-derive.

Before altering any actuator gain, consult [BRS.md](BRS.md) rather than reasoning from the configuration alone, since the configuration records the chosen values but not the inertias and bandwidth constraints that justify them. Before altering the hybrid internal model estimator, consult [HIM.md](HIM.md), which is the only account of the contrastive machinery its implementation relies upon. Before reading a line of the vendored `MorAL/` tree, consult [MorAL.md](MorAL.md), which records its architecture in full together with the eleven defects a reader would otherwise mistake for design.

## Document register

| Document | Subject | Last revised | Currency |
|---|---|---|---|
| [BRS.md](BRS.md) | SD_BRS1 actuator gain analysis, inertias, natural frequencies, damping ratios, recommended gains | 2026-07-29 | Current, its damping recommendations were adopted only later, see the note below |
| [HIM.md](HIM.md) | The hybrid internal model architecture, its estimator, and its contrastive objective | 2026-05-15 | Current |
| [joint_control_analysis.md](joint_control_analysis.md) | PD joint control as a second order system, action scaling, velocity saturation | 2026-03-18 | Current, the analysis is analytical rather than configuration bound |
| [MorAL.md](MorAL.md) | The vendored `MorAL/` clone, its software architecture, morphology mechanism, policy and training architecture, rewards, curricula and defects | 2026-08-10 | Current, read against commit `0ac74da8` of that tree |

## Document summaries

### BRS.md

The physics grounded derivation of the stiffness and damping gains for the SD_BRS1 biped, and the authoritative source for that robot's inertial properties. It extracts the link mass inventory, the leg geometry, and the joint topology from the assembly URDF, develops the proportional derivative actuator model as a second order system with its bandwidth constraint at the control rate, and then computes the effective inertia seen by each joint through the parallel axis theorem, obtaining 6.494 for hip pitch, 6.499 for hip roll, 1.272 for the knee, 0.039 for ankle pitch, and 0.033 for ankle roll, in kilogram metres squared. From these it evaluates the natural frequency and damping ratio of the configured gains joint by joint, adds a gravitational torque analysis, compares the result against the co-resident TRON1 configuration, and closes with a recommended gain range per joint together with a saturation check against the configured effort limits.

Its practical consequence is recorded elsewhere and belongs on this index. The stiffness half of its recommendations was adopted into the configuration while the damping half was not, which left the proximal joints at damping ratios between 0.07 and 0.16 against the 0.7 the document targets, and that discrepancy was later identified as a root cause of the degraded SD_BRS1 gait. The diagnosis is in `../../context/brs_gait.md` and the remedy in `../../plans/GAIT_STRATEGY.md`, whose Phase 0 finally applied the damping figures this document derived.

### HIM.md

The mathematical account of the hybrid internal model, the learned internal state estimator that the `himloco` package implements. It explains how the architecture combines supervised velocity estimation with prototype based contrastive representation learning to compress the proprioceptive history into a compact latent, which together with a predicted base velocity augments the actor observation at every step, so that the policy may infer unobservable quantities such as terrain contact, slip, and external disturbance from sensor history alone and therefore require no privileged teacher observation at deployment. Its most useful passages are the step by step walkthrough of the Sinkhorn normalisation used to form the prototype assignment, annotated with the numerical stability measures that motivate each step, and the correlation of that derivation against the exact lines of `himloco/himloco/modules/him_estimator.py` which implements it. An appendix closes the document.

Note that this file sat at `himloco/HIM.md` until 2026-07-30 and is referenced from a comment in the estimator source, which was updated to the new path in the same pass.

### joint_control_analysis.md

An analytical treatment of the proportional derivative joint controller as a mass spring damper system, and of the way action scaling interacts with actuator velocity limits to destabilise training. It derives the equation of motion, the characteristic equation, the natural frequency, and the damping ratio, describes the response in each damping regime, and then applies the results numerically to the SoleFoot robot. Its central contribution is the causal chain from an excessive action scale through velocity saturation to sustained oscillation and training divergence, together with a stability boundary and concrete recommendations.

This is the sole copy. A byte identical duplicate stood at `../../context/joint_control_analysis.md` in the workspace record until 2026-07-30, when it was deleted and this copy made canonical, on the ground that the subject is the actuator parameterisation of a robot defined in this repository and therefore belongs beside the SD_BRS1 gain derivation in [BRS.md](BRS.md). The workspace index records the move, and the hub `../../context/knowledge_base.md` now cites this path.

### MorAL.md

The complete reading of the `MorAL/` tree vendored at this repository's root, written so that a later session may judge what it offers the co-optimisation work without re-reading seven thousand lines of Isaac Gym era code. It establishes first what the clone is, namely a StochLab reimplementation built on the HIMLoco codebase rather than the authors' release, whose commit history records the morphology network being introduced, removed and restored, and whose GenLoco flag names the procedural morphology randomisation it implements. It then traces the simulation substrate, the control loop timing and the widened seven value step signature that exists so that the estimator receives an uncontaminated next state at an episode boundary.

Its central section is the morphology mechanism. A standalone script performs parametric surgery on one donor Go1 URDF to write 3600 variants and their parameter files, sampling one overall size scale, deriving limb scales allometrically, and recovering masses from volumes through densities clipped into physical bands, a population measured here at 4.96 to 79.99 kilogrammes. Each environment then loads its own asset at construction and keeps it for the life of the run, with two derived per environment quantities that matter well beyond this clone, a nominal base height computed from the leg link lengths and an actuator gain scale computed from the total mass. The decisive negative result is that nothing reloads a morphology and no design optimiser exists anywhere in the tree, so MorAL is a generalist controller over a fixed design distribution rather than a co-design method.

The remaining sections give the exact index layout of the 264 dimensional privileged observation, the five networks of the policy with their input compositions, the three phase training architecture and the loss composition each phase selects, the effective reward weights after the GenLoco overrides, the terrain and command curricula, the domain randomisation, and the persistence and export paths. It closes with a register of eleven defects established by reading, of which three are consequential, the degree of freedom properties of the first generated asset being applied to every environment so that the mass scaled effort limits are discarded and an eighty kilogramme robot is clipped at a thirteen kilogramme robot's ceiling, the foot clearance setpoint carrying the wrong sign for the frame it is evaluated in, and the TorchScript exporter being unable to produce a runnable policy. A closing assessment separates the three architectural ideas that transfer to the co-optimisation pipeline, the supervised design regression head, the morphology dependent reward setpoint and the residual phase structure, from the code that carries them, which does not.

## Conventions

Codebase citations take the form `path:line`, resolved against this repository's root. Line numbers were correct when written and drift as the sources change, so a citation should be treated as a pointer to a symbol rather than to a position, and the symbol located by name when the line no longer matches.

Cross references within this directory use the bare filename. References to the workspace level record take the form `../../context/NAME.md` and to the workspace plans `../../plans/NAME.md`. References to this repository's own architecture document take the form `../ARCHITECTURE.md`.

The writing and context conventions governing these documents are those of the workspace, recorded in `../../CLAUDE.md`.

## Related documentation

[../ARCHITECTURE.md](../ARCHITECTURE.md) describes the Isaac Lab and rsl-rl task integration of this repository, its directory roles, class hierarchies, and the steps for adding a task. [../README.md](../README.md) covers installation and standalone usage. [../GEMINI.md](../GEMINI.md) is a high level companion overview generated for tooling. All three remain at the repository root because they describe the repository as a whole rather than a single investigation.

One further document lives with the asset it describes rather than here, `../exts/bipedal_locomotion/bipedal_locomotion/assets/urdf/solefoot/tron1/base_robot.md`, which documents the link and joint structure of the TRON1 base URDF. It is deliberately left beside that URDF, since it describes one specific file and would lose its meaning if separated from it.
