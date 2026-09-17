# KScale Physical Parameterisation, Effective Inertia, Stance Load and Actuator Gain Analysis

> Status, established 2026-08-24 against the KScale integration pass recorded in [../plans/kscale_integration.md](../plans/kscale_integration.md), restructured 2026-08-25 to follow the analytical order of [BRS.md](BRS.md) so that the same calculations are performed and validated for every robot in this workspace.
>
> Every figure is measured from `environments/environments/assets/urdf/solefoot/kscale/kscale.urdf` and its collision meshes by the scripts named in section 13, and is reproducible by re-running them. Where a figure has been superseded the superseding section is named at the point of use rather than the old figure silently corrected.

## 1. Introduction

The KScale is a sole footed biped of six actuated degrees of freedom per leg, arranged from the torso outward as a hip pitch, a hip roll, a hip yaw, a knee, an ankle pitch and an ankle roll, terminating in a flat plate that meets the ground over an area rather than at a point. It shares that topology with the SD_BRS1, which this repository refers to throughout as the BRS, and the resemblance is close enough that the KScale environment configuration entered the tree as a direct copy of the BRS configuration retargeted onto a different set of link and joint names.

The resemblance is superficial in four respects, and this document exists because none of the four is visible in the configuration file that suffers from them. The robot is roughly a third of the mass and three fifths of the leg length, so every reward parameter carrying units of length, force or height is wrong by some ratio when carried across unchanged. Its foot link frame is a full axis permutation away from the BRS convention, which caused the sole depth to be read as the foot's length. Its root link frame was rotated ninety degrees about the vertical from the convention Isaac Lab assumes, so the axis the configuration commanded as forward velocity was in fact the robot's lateral axis. And its actuator gains, when they were finally derived rather than copied, were derived against the inertia a leg presents while swinging freely and not against the load a leg carries while standing, which is a distinction the BRS analysis never had to confront because the BRS gains happened to satisfy both criteria at once.

The fourth point is the subject of the central sections below, and it is worth stating plainly at the outset because it cost a full training run. A stiffness chosen to place a joint's natural frequency inside a bandwidth band says nothing whatever about whether that joint can hold the robot up. The two questions are answered by two different inertias and two different equations, and a derivation that answers only the first will produce a robot that cannot stand. Sections 4 and 10 set out both quantities, section 11 derives gains that satisfy both, and section 19 shows the training log of the run that satisfied only one.

This document records the physical parameterisation that settles these questions, in the manner [BRS.md](BRS.md) records the BRS and [quadruped.md](quadruped.md) records the quadruped. It is the source a later reader should consult before altering any KScale reward parameter or actuator gain, since the configuration records the chosen values but not the geometry, inertias and loads that justify them.

---

## 2. Robot Physical Properties Extracted from URDF

### 2.1 Link Mass Inventory

The articulation carries thirteen links and twelve revolute joints. The masses below are as exported from Onshape, and the naming is an export artefact carrying no side convention, a point section 2.3 returns to.

| Link Name | Description | Mass (kg) |
|---|---|---|
| `assy_formfg___kd_b_102b_torso_btm` | Torso, the root link | 4.3193 |
| `kd_d_102r_6061` | Right hip pitch bracket | 0.5237 |
| `kd_d_102l_6061` | Left hip pitch bracket | 0.5237 |
| `kd_d_201r_6061` | Right hip roll link | 2.4340 |
| `rs03` | Left hip roll link | 2.4339 |
| `kd_d_301r_6061` | Right thigh, carrying the hip yaw | 2.3872 |
| `kd_d_301l_6061` | Left thigh, carrying the hip yaw | 2.3873 |
| `kd_d_401r_6061` | Right shank | 2.0984 |
| `kd_d_401l_6061` | Left shank | 2.0983 |
| `arb_uj111_cross_bearing` | Right ankle universal joint cross | 0.1100 |
| `arb_uj111_cross_bearing_2` | Left ankle universal joint cross | 0.1100 |
| `foot_6061` | Right sole plate | 0.4278 |
| `foot_6061_2` | Left sole plate | 0.4278 |

Summing all links gives a total robot mass of 20.2814 kg, against the BRS at 59.85 kg, a ratio of 0.339. The corresponding weight is 198.89 N, which is the quantity every load figure below is proportioned against.

The mass distribution differs from the BRS in a way that matters to the inertia calculation of section 4. The BRS foot masses 3.691 kg, being 6.2 per cent of that robot, where the KScale foot masses 0.4278 kg, being 2.1 per cent of this one. The BRS hip inertia is dominated by its heavy foot swung at a long moment arm, and the KScale hip inertia is dominated instead by its shank and thigh, which is why the two robots' inertia ratios do not follow their mass ratio.

### 2.2 Link Length Inventory

The leg is a chain of six joint origin offsets from the torso to the sole. The offsets are given as vectors in each joint's parent frame, as the URDF declares them, rather than as scalar norms alone, because none of the three long segments is aligned with the axis a reader would expect and the standing height cannot be recovered from the norms.

| Segment | Offset in the parent frame | Norm |
|---|---|---|
| Torso to hip pitch | +0.02461, -0.05500, -0.05951 | 0.08469 m |
| Hip pitch to hip roll | -0.02925, -0.03000, -0.07100 | 0.08244 m |
| Hip roll to hip yaw | +0.00000, -0.14200, -0.02525 | 0.14423 m |
| Hip yaw to knee, the thigh | +0.02200, +0.02100, -0.21200 | 0.21417 m |
| Knee to ankle pitch, the shank | +0.28855, -0.03268, +0.00800 | 0.29051 m |
| Ankle pitch to ankle roll | +0.00000, +0.03000, -0.03000 | 0.04243 m |
| Ankle roll to the sole plane | along the foot link frame's plus y | 0.04300 m |

The shank's offset is declared almost entirely along its parent's x and the thigh's almost entirely along minus z, so the two are expressed in frames a quarter turn apart, which is why the vertical projection of the chain must be taken by forward kinematics and not by summing norms. The thigh and the shank sum to 0.50468 m, which is the hip to ankle leg length of 0.505 m that the swing height comment at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:188` scales against. It is a segment sum rather than a measured span, so it is invariant under the pose.

Taking the chain by forward kinematics at the nominal pose of section 16 decomposes the standing height as follows, each entry being the drop from one joint origin to the next.

| From | To | Vertical drop |
|---|---|---|
| Torso origin | Hip pitch | 0.05951 m |
| Hip pitch | Hip roll | 0.02693 m |
| Hip roll | Hip yaw | 0.14381 m |
| Hip yaw | Knee | 0.21304 m |
| Knee | Ankle pitch | 0.28532 m |
| Ankle pitch | Ankle roll | 0.00000 m |
| Ankle roll | Sole plane | 0.04300 m |
| | Total | 0.77161 m |

The two hip brackets and the hip roll to hip yaw offset together consume 0.23025 m, very nearly a third of the standing height, before the leg proper begins. This robot carries an unusually tall pelvis relative to its legs, and that single fact explains a great deal of what follows, since the body centre of mass sits 0.126 m medial of each foot where the BRS sits 0.1298 m medial of a foot on a robot half again as tall.

The comparison against the BRS, at each robot's own nominal pose, is as follows.

| Quantity | KScale | BRS | Ratio |
|---|---|---|---|
| Torso origin above the lowest sole point | 0.77161 m | 1.15 m | 0.671 |
| Hip to ankle leg length | 0.505 m | 0.87 m | 0.580 |
| Foot lateral separation | 0.2520 m | 0.259 m | 0.973 |
| Total mass | 20.281 kg | 59.85 kg | 0.339 |

The foot lateral separation is very nearly the same on the two robots despite the threefold mass difference, so the stance width parameters transfer across almost unchanged where the height and force parameters do not. The KScale is a narrow robot in the fore and aft sense and a comparatively broad one across the hips.

### 2.3 Joint Topology and Degrees of Freedom

The full joint chain per leg is as follows.

Torso, then revolute hip pitch, then the hip pitch bracket, then revolute hip roll, then the hip roll link, then revolute hip yaw, then the thigh, then revolute knee, then the shank, then revolute ankle pitch, then the universal joint cross, then revolute ankle roll, then the sole plate.

Every KScale joint declares its axis as the local 0, 0, 1, with the whole orientation carried in the origin rotation, so no axis can be read off the URDF axis vector and each must be composed through the origin rotation chain to the root. Doing so at the nominal pose gives the following axes in the root frame, for the right leg.

| Joint | Axis in the root frame | Character |
|---|---|---|
| `right_hip_pitch_04` | -0.0000, +1.0000, -0.0000 | lateral, pitch |
| `right_hip_roll_03` | +0.9950, -0.0000, +0.0998 | fore and aft, roll |
| `right_hip_yaw_03` | -0.0998, +0.0000, +0.9950 | vertical, yaw |
| `right_knee_04` | +0.0000, +1.0000, -0.0000 | lateral, pitch |
| `right_foot_pitch_02` | +0.0000, +1.0000, -0.0000 | lateral, pitch |
| `right_foot_roll_02` | -1.0000, +0.0000, -0.0000 | fore and aft, roll |

The hip roll and hip yaw axes are tilted by 0.0998 in the plane, being about 5.7 degrees, because the table is composed at the nominal pose that `scripts/analysis/kscale_stance_analysis.py` sets, where `right_hip_pitch_04` is flexed to minus 0.1 rad and carries its two rigidly downstream joints with it, so the tilt is sin and cos of that flexion angle rather than a hardware mounting cant, and at the URDF's own zero pose both axes are exactly axis aligned. The three sagittal axes, the hip pitch, the knee and the ankle pitch, are exactly parallel, which is the property section 16 uses to close the sagittal chain and lay the sole flat.

The naming carries no side convention. Where a part was reused rather than mirrored in CAD the exporter named it after the part, which is why the right hip roll link is `kd_d_201r_6061` while its left partner is `rs03`, sharing no stem whatever. This is why the symmetry module of section 17 must carry an explicit body partner dictionary rather than any rule.

The hip yaw is live on the KScale and mechanically absent on the BRS, whose URDF declares both hip yaw joints with `type="fixed"` so that they never enter `robot.joint_names` at all. The KScale therefore actuates twelve joints where the BRS actuates ten, which propagates into the observation width, the action width and the symmetry permutation alike.

Joint limits from the URDF are listed below, and the last column bears on section 17.

| Joint | Right lower | Right upper | Left lower | Left upper | Left against right |
|---|---|---|---|---|---|
| `hip_pitch_04` | -2.21657 | 1.04720 | -1.04720 | 2.21657 | negated |
| `hip_roll_03` | -2.26893 | 0.20944 | -0.20944 | 2.26893 | negated |
| `hip_yaw_03` | -1.57080 | 1.57080 | -1.57080 | 1.57080 | symmetric |
| `knee_04` | 0.00000 | 2.70526 | 0.00000 | 2.70526 | identical |
| `foot_pitch_02` | -0.87267 | 0.52360 | -0.87267 | 0.52360 | identical |
| `foot_roll_02` | -0.26180 | 0.26180 | -0.26180 | 0.26180 | symmetric |

The hip pitch and hip roll are the only two joints whose left and right limits are genuinely anti symmetric, the hip yaw and ankle roll ranges being symmetric about zero and therefore invariant under negation. This is corroborating evidence for the flip set of section 17 rather than its derivation.

The ankle roll travel of plus and minus 0.26180 rad is the narrowest on the robot, and section 11.8 shows that this narrowness, rather than any inertial consideration, is what sets that joint's stiffness.

The URDF carries a uniform effort of 10 Nm and a uniform velocity of 10 rad/s on every joint without exception, from the hip pitch that carries the whole leg to the ankle roll that carries only the foot. This is the signature of an unedited export default rather than a measured hardware limit. The limits that actually bind in simulation come from the per joint actuator configuration at `environments/environments/assets/config/kscale_identified_cfg.py`, which is fortunate, but the URDF figures should not be mistaken for data.

---

## 3. Theoretical Background

The common theory is set out at [BRS.md](BRS.md) section 3 and is not repeated. In brief, each joint is governed by a proportional derivative position controller whose torque is `tau = K (q_des - q) - D q_dot` clamped to the actuator's effort limit, the closed loop is a damped harmonic oscillator of natural frequency `omega_n = sqrt(K / I_eff)` and damping ratio `zeta = D / (2 sqrt(K I_eff))`, and a damping ratio near 0.70 is sought throughout. Two extensions to that treatment are needed here, and both are consequences of this robot rather than refinements of the theory.

### 3.1 The armature belongs in the effective inertia

Isaac Lab writes the configured armature onto the PhysX joint, so it enters the mass matrix diagonal of the degree of freedom and is part of the inertia the proportional derivative loop actually sees. On the BRS this is a correction of a few per cent and can be neglected without changing any conclusion. On the KScale it cannot. At the ankle roll the armature of 0.005 kg m squared stands against a link inertia of 0.00074, exceeding it by a factor of 6.75, and at the ankle pitch 0.005 stands against 0.00260. Omitting the armature at these joints does not merely refine the answer, it changes which regime the joint is judged to be in, and an earlier draft of the integration plan that omitted it concluded that two ankle joints sat at or above the Nyquist bound of the control loop. They do not, and that conclusion is withdrawn in section 20.

### 3.2 Swing inertia and stance inertia are different quantities

The effective inertia obtained by the parallel axis theorem in section 4 is the inertia of the joint's distal subtree about its own axis. That is the inertia the joint accelerates when the leg swings freely in the air, and it is the right quantity for the swing phase. It is not the right quantity for the stance phase, and the difference is not a refinement.

With the foot planted the kinematic chain runs from the ground upward rather than from the torso downward. The links below the joint are held by the ground and contribute nothing to its inertia, and the links above it, the torso, the opposite leg and whatever of this leg is proximal to the joint, are what the joint must accelerate. Two quantities follow. The inertia is the proximal assembly taken about the joint axis, which section 4.9 computes, and gravity contributes a stiffness of its own, positive or negative according to whether rotating the joint raises or lowers that assembly, which section 10.6 computes. Neither has a counterpart in the swing regime.

Both regimes matter and on this robot they disagree by up to a factor of 937, the ankle roll swinging a 0.43 kg sole in the air and holding 19.85 kg of machine on the ground. Section 4.9 gives the reflected inertias, section 10 gives the stance loads and the gravitational stiffnesses, and section 11 derives gains that respect both. The practical rule that emerges is that stiffness is set by the stance load, by the gravitational stiffness and by the authority to command a single support posture, that damping is set by the stance inertia wherever the effort limit permits it and by the effort limit where it does not, and that the swing quantities serve as a bandwidth ceiling rather than as a design target.

---

## 4. Effective Inertia Calculation by the Parallel-Axis Theorem

The effective rotational inertia at each joint about its rotation axis is computed by summing contributions from all links distal to that joint. For each distal link `k`,

```
I_k_contribution = a . (R_k I_k R_k^T) . a  +  m_k r_perp_k^2
```

where `a` is the joint's unit axis in the root frame, `I_k` is the link's inertia tensor as the URDF declares it in its own inertial frame, `R_k` carries that frame into the root frame, and `r_perp_k` is the perpendicular distance from the joint axis to the link's centre of mass. The armature of section 3.1 is added to the total.

The own inertia term is taken in full three dimensions rather than by reading a diagonal entry. This is not pedantry on this robot. The KScale inertial frames carry non trivial rpy rotations and the joint axes are canted, as section 2.3 records, so the scalar projection that suffices for an axis aligned robot understates the result here.

All coordinates are in metres, masses in kilograms, inertias in kg m squared, and torques in Newton metres. Every figure below is at the nominal pose of section 16 and is emitted by `scripts/analysis/kscale_stance_analysis.py`.

### 4.1 Hip Pitch (axis +y, origin +0.0246, -0.0550, -0.0595)

Distal links are the hip pitch bracket, the hip roll link, the thigh, the shank, the ankle cross and the foot.

| Link | m | r_perp | own inertia | m r_perp^2 | contribution |
|---|---|---|---|---|---|
| `kd_d_102r_6061` | 0.5237 | 0.0073 | 0.001259 | 0.000028 | 0.001287 |
| `kd_d_201r_6061` | 2.4340 | 0.0886 | 0.010534 | 0.019122 | 0.029657 |
| `kd_d_301r_6061` | 2.3872 | 0.3456 | 0.015469 | 0.285055 | 0.300523 |
| `kd_d_401r_6061` | 2.0984 | 0.5118 | 0.017283 | 0.549635 | 0.566918 |
| `arb_uj111_cross_bearing` | 0.1100 | 0.6699 | 0.000026 | 0.049381 | 0.049408 |
| `foot_6061` | 0.4278 | 0.7019 | 0.001445 | 0.210761 | 0.212206 |

Link total 1.16000, plus armature 0.010, gives I_eff_hip_pitch = 1.17000 kg m squared.

The dominant contribution is the shank at 0.5669, not the foot at 0.2122, which is the reverse of the BRS ordering. The KScale foot is light and the shank is heavy, so the hip's rotational load is carried by the limb rather than by the extremity.

### 4.2 Hip Roll (axis +0.9950, 0, +0.0998, origin +0.0567, -0.1260, -0.0864)

| Link | m | r_perp | own inertia | m r_perp^2 | contribution |
|---|---|---|---|---|---|
| `kd_d_201r_6061` | 2.4340 | 0.0586 | 0.010901 | 0.008357 | 0.019259 |
| `kd_d_301r_6061` | 2.3872 | 0.3154 | 0.014027 | 0.237519 | 0.251546 |
| `kd_d_401r_6061` | 2.0984 | 0.4805 | 0.015791 | 0.484524 | 0.500315 |
| `arb_uj111_cross_bearing` | 0.1100 | 0.6325 | 0.000027 | 0.044022 | 0.044048 |
| `foot_6061` | 0.4278 | 0.6691 | 0.000306 | 0.191500 | 0.191806 |

Link total 1.00697, plus armature 0.010, gives I_eff_hip_roll = 1.01697 kg m squared.

### 4.3 Hip Yaw (axis -0.0998, 0, +0.9950, origin +0.0458, -0.1260, -0.2303)

| Link | m | r_perp | own inertia | m r_perp^2 | contribution |
|---|---|---|---|---|---|
| `kd_d_301r_6061` | 2.3872 | 0.0136 | 0.004059 | 0.000443 | 0.004503 |
| `kd_d_401r_6061` | 2.0984 | 0.0401 | 0.005009 | 0.003377 | 0.008386 |
| `arb_uj111_cross_bearing` | 0.1100 | 0.1033 | 0.000045 | 0.001173 | 0.001219 |
| `foot_6061` | 0.4278 | 0.0671 | 0.001594 | 0.001927 | 0.003520 |

Link total 0.01763, plus armature 0.010, gives I_eff_hip_yaw = 0.02763 kg m squared.

The hip yaw is the joint where the armature matters most among the proximal joints, contributing 36 per cent of the total. Every distal link sits close to the yaw axis, the perpendicular distances being at most 0.1033 m, because the leg hangs along the axis rather than across it.

### 4.4 Knee (axis +y, origin +0.0460, -0.1040, -0.4433)

| Link | m | r_perp | own inertia | m r_perp^2 | contribution |
|---|---|---|---|---|---|
| `kd_d_401r_6061` | 2.0984 | 0.1280 | 0.017283 | 0.034354 | 0.051638 |
| `arb_uj111_cross_bearing` | 0.1100 | 0.2904 | 0.000026 | 0.009280 | 0.009306 |
| `foot_6061` | 0.4278 | 0.3184 | 0.001445 | 0.043376 | 0.044821 |

Link total 0.10576, plus armature 0.015, gives I_eff_knee = 0.12076 kg m squared.

### 4.5 Ankle Pitch (axis +y, origin -0.0080, -0.0960, -0.7286)

| Link | m | r_perp | own inertia | m r_perp^2 | contribution |
|---|---|---|---|---|---|
| `arb_uj111_cross_bearing` | 0.1100 | 0.0000 | 0.000026 | 0.000000 | 0.000026 |
| `foot_6061` | 0.4278 | 0.0514 | 0.001445 | 0.001131 | 0.002576 |

Link total 0.00260, plus armature 0.005, gives I_eff_ankle_pitch = 0.00760 kg m squared.

The universal joint cross sits exactly on the ankle pitch axis, so its perpendicular distance is zero to four decimal places and it contributes only its own inertia. The armature is 1.92 times the entire link total.

### 4.6 Ankle Roll (axis -x, origin -0.0380, -0.1260, -0.7286)

| Link | m | r_perp | own inertia | m r_perp^2 | contribution |
|---|---|---|---|---|---|
| `foot_6061` | 0.4278 | 0.0328 | 0.000282 | 0.000460 | 0.000741 |

Link total 0.00074, plus armature 0.005, gives I_eff_ankle_roll = 0.00574 kg m squared.

The armature is 6.75 times the link total. This joint is, in the mass matrix, almost entirely rotor.

### 4.7 Summary of Effective Inertias

| Joint | Link total | Armature | I_eff (kg m^2) | Dominant contribution |
|---|---|---|---|---|
| Hip Pitch | 1.16000 | 0.010 | 1.17000 | shank 0.5669 and thigh 0.3005 |
| Hip Roll | 1.00697 | 0.010 | 1.01697 | shank 0.5003 and thigh 0.2515 |
| Hip Yaw | 0.01763 | 0.010 | 0.02763 | shank 0.0084, armature 0.010 |
| Knee | 0.10576 | 0.015 | 0.12076 | shank 0.0516 and foot 0.0448 |
| Ankle Pitch | 0.00260 | 0.005 | 0.00760 | foot 0.0026, armature 0.005 |
| Ankle Roll | 0.00074 | 0.005 | 0.00574 | armature 0.005 |

Against the BRS the ratios are 0.180 at the hip pitch, 0.156 at the hip roll, 0.095 at the knee, 0.195 at the ankle pitch and 0.174 at the ankle roll. None of these is the mass ratio of 0.339, and the knee is furthest from it, because inertia scales as mass times length squared rather than as mass and the KScale legs are 0.58 as long, so their inertia falls faster than their mass. This is the arithmetic behind the warning of section 1 that a gain copied by joint role will be wrong.

### 4.8 Gain-Randomisation Envelope

The event term at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:551` randomises stiffness and damping by plus and minus ten per cent at every reset. Any gain conclusion that depends on a distinction finer than ten per cent is therefore not a conclusion the simulation can distinguish, and the derivation of section 11 is quoted to no greater precision than that band supports.

### 4.9 Stance-Reflected Inertia

The inertias above are the swing quantities of section 3.2, being what a joint accelerates when its foot is free. With the foot planted the situation is exactly reversed. The links below the joint are held by the ground and the links above it, the torso, the opposite leg and whatever of this leg lies proximal to the joint, are what the joint accelerates. The stance reflected inertia is therefore the PROXIMAL assembly's inertia about the joint's own axis, computed by the same parallel axis sum as section 4 and taken over the complement of the distal subtree, with the armature added as before since the rotor is on the joint in either regime.

| Joint | I_swing | I_stance | Ratio | Proximal mass (kg) |
|---|---|---|---|---|
| Hip Pitch | 1.17000 | 1.18166 | 1.01 | 12.30 |
| Hip Roll | 1.01697 | 1.60635 | 1.58 | 12.82 |
| Hip Yaw | 0.02763 | 0.61603 | 22.30 | 15.26 |
| Knee | 0.12076 | 1.44496 | 11.97 | 17.65 |
| Ankle Pitch | 0.00760 | 4.85781 | 638.96 | 19.74 |
| Ankle Roll | 0.00574 | 5.37942 | 936.96 | 19.85 |

The ratio rises steeply as the joint grows more distal, and the reason is structural rather than incidental. A proximal joint carries nearly the same mass in both regimes, the hip pitch swinging a 9.0 kg leg below it and holding a 12.3 kg body above it, so its two inertias agree to one per cent. A distal joint carries almost nothing below it and the entire machine above it, the ankle roll swinging a 0.43 kg sole and holding 19.85 kg, so its two inertias differ by nearly three orders of magnitude. The armature dominates the swing figure at both ankles, section 4.6 recording that the ankle roll is in the mass matrix almost entirely rotor, and it is negligible against the stance figure.

Two consequences govern section 11 and both are severe.

The first is that a damping value giving a ratio of 0.70 against the swing inertia gives 0.70 divided by the square root of the ratio against the stance inertia, so a knee tuned in the air arrives in stance at 0.20 and an ankle roll at 0.023. That is not a mild degradation, it is a servo that has effectively no damping at the moment the robot is carrying its own weight. Section 11 therefore sizes damping against the stance inertia wherever the effort limit permits it.

The second is that the natural frequency ordering inverts between the regimes. In swing the ankles are the fastest joints on the robot and in stance they are the slowest, and a derivation that reads the swing figure will place a distal joint in a bandwidth band it does not occupy while it is on the ground. Section 6 tabulates both orderings and section 20 records the aliasing conclusion this correction withdraws.

One caution is recorded rather than resolved. The computation treats the planted foot as welded to the ground and the proximal assembly as rotating rigidly about the single joint axis, which is exact for single support with one joint moving and approximate in double support, where the closed loop through both legs shares the load between them. Single support occupies 0.76 of the commanded gait cycle, so the single support figure is the one section 11 uses, and the double support case would give a smaller inertia at every joint and therefore a lower damping requirement, making the choice the conservative one.

---

## 5. Calculated Actuator Configuration

The actuator gains derived in section 11 are as follows. They are named the calculated set throughout, since every stiffness in them was computed from a stance requirement rather than measured. They are the successor to the swing derived set that stood at `environments/environments/assets/config/kscale_identified_cfg.py` from the revision of 2026-08-25 until 2026-09-03, and section 7 records the identified set that stood there after it. The stiffnesses below have been carried by one training run, `2026-09-07_11-08-18`, paired with a damping set that predates section 11.1.1 and that the criterion of that section shows to be divergent at two joints, and the addition of 2026-09-08 records what it measured.

| Joint | K (Nm/rad) | D (Nm s/rad) | Armature (kg m^2) | Effort Limit (Nm) | Velocity Limit (rad/s) |
|---|---|---|---|---|---|
| Hip Yaw | 100 | 3.3 | 0.010 | 120 | 40 |
| Hip Roll | 250 | 28.0 | 0.010 | 120 | 40 |
| Hip Pitch | 200 | 21.5 | 0.010 | 120 | 40 |
| Knee | 300 | 12.0 | 0.015 | 120 | 40 |
| Ankle Roll | 120 | 1.0 | 0.005 | 34 | 20 |
| Ankle Pitch | 120 | 1.3 | 0.005 | 34 | 20 |

### 5.1 The actuator classes, identified from the joint names

The effort and velocity limits above are not derived from the URDF, which as section 2.3 records carries a uniform export default of 10 Nm and 10 rad per second on every joint of the twelve, a placeholder the converter emits where the source declares nothing. They are therefore a choice this workspace makes, and until 2026-09-08 they were an unexamined one. The evidence that settles them is in the joint names themselves.

Every actuated joint of this robot carries a two digit suffix, `hip_pitch_04`, `hip_roll_03`, `hip_yaw_03`, `knee_04`, `foot_pitch_02` and `foot_roll_02`. The K-Scale Labs K-Bot, whose actuator parameterisation is recorded at [/ws/context/kscale-opensource.md](../../context/kscale-opensource.md), names its own joints by exactly the same rule, `dof_left_hip_pitch_04`, `dof_left_hip_roll_03`, `dof_left_hip_yaw_03`, `dof_left_knee_04` and `dof_left_ankle_02`, and section 5 of that document establishes that the suffix is the RobStride actuator class the joint carries, the metadata assigning `robstride_04` to every `_04` joint, `robstride_03` to every `_03` and `robstride_02` to every `_02` without exception. The two robots share a vendor and a naming convention, and the suffix is therefore a specification rather than an ornament.

The four classes and their declared properties are reproduced from section 6 of that document, the three this robot uses being the only ones in the table that concern it.

| Class | Peak torque (Nm) | Soft limit (Nm) | Armature (kg m^2) | Declared max velocity (rad/s) | Firmware velocity ceiling (rad/s) | Firmware damping ceiling |
|---|---|---|---|---|---|---|
| robstride_02 | 17.0 | 11.9 | 0.0042 | 37.699 | 44.0 | 5.0 |
| robstride_03 | 60.0 | 42.0 | 0.0200 | 18.849 | 20.0 | 100.0 |
| robstride_04 | 120.0 | 84.0 | 0.0400 | 17.488 | 15.0 | 100.0 |

Placing this robot's six joints against that table exposes a misassignment that stood in the configuration from the original KScale integration commit until 2026-09-08. The `_04` joints, being the hip pitch and the knee, carried an effort limit of 60 Nm, which is the peak torque of the robstride_03 and not of the robstride_04 they are named for. Those two joints were under provisioned by a factor of exactly two, and the knee is the joint carrying the largest stance load on the robot at 16.47 Nm in single support. The `_03` joints, being the hip roll and the hip yaw, carried 60 Nm and were correct. The two ankles carried 17 Nm, which is the robstride_02 peak torque and is correct as a peak, though section 6 of the reference document notes that the K-Scale stack itself derates every class to seven tenths of peak as a soft limit, which would place the ankle at 11.9 Nm rather than 17.

The velocity limits were wrong in both directions and the ankle is the severe case. The two ankles carried 10 rad per second against a declared robstride_02 maximum of 37.699 and a firmware ceiling of 44.0, a limit under a third of the actuator's rating, while the four proximal joints carried 20 rad per second against declared maxima of 17.488 and 18.849, which is slightly permissive rather than restrictive. Section 11.10 gives the measurement that makes the ankle figure consequential, and it is the most consequential single number in this document.

The armatures deserve the same comparison and are not corrected here. This robot declares 0.010 at the three hip joints, 0.015 at the knee and 0.005 at the two ankles, where the RobStride classes give 0.0200 for the robstride_03, 0.0400 for the robstride_04 and 0.0042 for the robstride_02. The hip pitch and knee armatures are therefore between a quarter and three eighths of the class figure, the two robstride_03 hips half of it, and the ankles about nineteen per cent above it. The discrepancy is recorded rather than repaired, since the armature enters the mass matrix and changing it changes the plant, invalidating comparison against every run in the series, and section 11.1 records what adopting the class figures would buy at the one joint where it matters.

### 5.2 The doubled limits, and what they are for

The limits in the table of section 5 are twice the values the original integration carried, being 120 Nm and 40 rad per second at the four proximal joints and 34 Nm and 20 rad per second at the two ankles. The doubling is an experimental instrument rather than a specification claim, and it is adopted for a measured reason that section 11.10 sets out, four joints of six standing at their effort limit for about half of every episode and the two ankles reaching ninety six per cent of their velocity limit at the ninety fifth percentile of their speed distribution.

Its relation to the RobStride figures is not uniform and each case should be read on its own terms. At the hip pitch and the knee the doubled 120 Nm is exactly the robstride_04 peak torque, so the doubling at those two joints is not an experimental liberty but the correction of the misassignment section 5.1 identifies, and it is the one place where the change is defensible as hardware. At the hip roll and the hip yaw the doubled 120 Nm is twice the robstride_03 peak of 60 and is beyond specification. At the two ankles the doubled 34 Nm is twice the robstride_02 peak of 17 and is likewise beyond it, while the doubled velocity limit of 20 rad per second remains only 0.53 of the robstride_02 declared maximum and is therefore still the more restrictive of the two figures.

The purpose of the ankle doubling is specific and section 10.6 states the question it answers. The ankle roll must supply 25.07 Nm to hold the nominal pose on one foot, which no stiffness alters because the constraint is on torque rather than on deflection, so at 17 Nm that joint cannot hold single support at all and the robot must find its lateral support elsewhere. At 34 Nm it can, with 26 per cent of the limit to spare. Whether the resulting gait is achievable at 17 Nm is then answerable by running the same configuration at both limits, which is what the doubling is for, and a policy that walks only at 34 Nm has established that the ankle limit and not the reward set is what has been preventing it.

Three sets of gains have stood in this file and the distinction between them governs every comparison below. The original set was copied from the BRS by joint role. The second was derived on swing bandwidth alone in the pass of 2026-08-24 and is the set the failed training run used. The third is the calculated set, derived in section 11 against the stance regime.

| Joint | Copied K, D | Swing-derived K, D | Calculated K, D |
|---|---|---|---|
| Hip Pitch | 200, 50 | 200, 21.5 | 200, 21.5 |
| Hip Roll | 150, 45 | 150, 17.2 | 250, 28.0 |
| Hip Yaw | 40, 5 | 15, 0.85 | 100, 3.3 |
| Knee | 200, 22 | 25, 2.45 | 300, 12.0 |
| Ankle Pitch | 50, 4 | 7, 0.32 | 120, 1.3 |
| Ankle Roll | 20, 4 | 5, 0.24 | 120, 1.0 |

The calculated set agrees with the copied one at the hip pitch stiffness alone and departs from it everywhere else, upward at four joints by factors between 1.7 and 6.0. That pattern is the signature of the correction section 11.9 sets out. A gain copied by joint role, or derived against the inertia a limb presents in the air, understates what a joint must supply once its foot is on the ground, and it understates it most severely at the joints furthest from the torso, whose swing inertia is almost entirely rotor and whose stance inertia is the whole machine.

The damping moves in the opposite direction and the reason is a different constraint entirely. It falls below the copied set at every joint of the six, by a factor of 1.5 at the hip pitch and by factors between 1.8 and 4.0 at the four joints below it, and section 11.1 records the ceiling that produces it. A servo whose torque this simulation computes outside the physics step is an explicit element, and an explicit damper applied to a low inertia joint is unstable above a bound that no stance argument contains. The stance inertia says how much damping the joint wants and the integration says how much it may be given, and at four joints of six the second is the smaller quantity.

---

## 6. Analysis of Calculated Gains

Using the relations of section 3 against both inertias of section 4, for the calculated set of section 5.

| Joint | K | D | I_swing | w_swing | z_swing | I_stance | w_stance | z_stance |
|---|---|---|---|---|---|---|---|---|
| Hip Pitch | 200 | 21.5 | 1.17000 | 13.07 | 0.703 | 1.18166 | 13.01 | 0.699 |
| Hip Roll | 250 | 28.0 | 1.01697 | 15.68 | 0.878 | 1.60635 | 12.48 | 0.699 |
| Hip Yaw | 100 | 3.3 | 0.02763 | 60.16 | 0.993 | 0.61603 | 12.74 | 0.210 |
| Knee | 300 | 12.0 | 0.12076 | 49.85 | 0.997 | 1.44496 | 14.41 | 0.288 |
| Ankle Pitch | 120 | 1.3 | 0.00760 | 125.66 | 0.681 | 4.85781 | 4.97 | 0.027 |
| Ankle Roll | 120 | 1.0 | 0.00574 | 144.55 | 0.602 | 5.37942 | 4.72 | 0.020 |

Two joints reach the design target in stance and four do not, and the division is not a matter of choice. The hip pitch and the hip roll arrive at stance damping ratios of 0.699 and 0.699, which is the 0.70 sought to within the ten per cent randomisation envelope of section 4.8. The hip yaw, the knee and the two ankles arrive at 0.210, 0.288, 0.027 and 0.020, because the damping their stance inertia asks for would demand 11 to 36 Nm s per radian and the integration ceiling of section 11.1 permits 1.0 to 12.0. This is a statement about the substrate that integrates the servo and not about the derivation, and section 11.1 records the two ceilings that produce it.

The swing damping ratios are the reason the stance ratios cannot be met, and they run from 0.602 to 0.997 by construction. No joint of the six exceeds unity, which is a deliberate property of the set rather than an accident, since a servo overdamped in the air is both slow to place a foot and, at a joint whose swing inertia is small, numerically divergent under the explicit integration this simulation performs. The hip yaw is the joint at which the two regimes conflict most sharply, its stance inertia exceeding its swing inertia by a factor of 22.3, so the damping its stance mode wants is 3.3 times what its swing mode will carry. Section 11.6 records the resolution, which is that the swing constraint is a stability bound and the stance figure a preference, and a preference does not override a bound.

The natural frequencies invert the ordering a swing analysis produces, and the inversion is the central quantitative result of this section. In the swing regime the ankles are the fastest joints on the robot at 125.66 and 144.55 rad per second, the hip yaw next at 60.16, and the hips slowest at 13 to 16. In the stance regime the ordering reverses, the ankles becoming the slowest at 4.97 and 4.72 while the four proximal joints cluster between 12.48 and 14.41. The reason is that a distal joint has almost no limb below it and the entire machine above it, so its inertia in the two regimes differs by a factor of 639 at the ankle pitch and 937 at the ankle roll. Every conclusion drawn from calling the ankles fast is therefore an artefact of the wrong regime, including the aliasing warning an earlier draft raised, and section 20 records that withdrawal.

The bound that genuinely matters is the Nyquist frequency of the control loop, 157.08 rad per second for the tighter of the two control periods the KScale tasks impose, that being the rough terrain task at decimation 4 over `sim.dt` 0.005. The highest figure in the table is the ankle roll's swing frequency at 144.55 rad per second, which is 0.92 of Nyquist, and it is the binding ceiling on that joint's stiffness rather than a coincidence, section 11.8 having chosen 120 for exactly that reason. Every other joint sits below 0.80 of the bound, and in stance no joint exceeds 0.10 of it.

Against the band [BRS.md](BRS.md) section 3.3 recommends, being 5 to 15 rad per second for a proximal joint and 15 to 30 for a distal one, the four proximal joints now sit inside their band in the stance regime for the first time in any set this document records, and the two distal joints sit below theirs at 4.97 and 4.72. A distal joint slower than its band is the opposite of the failure the band exists to prevent, and it is not correctable, the stiffness that would raise those frequencies standing above the Nyquist ceiling their swing inertia imposes.

One property of the set was not aimed at and is the clearest evidence that the two regimes have been separated correctly. The four proximal joints arrive at stance natural frequencies of 13.01, 12.48, 12.74 and 14.41 rad per second, a band 15 per cent wide across joints whose stance inertias differ by a factor of 2.6 and whose stiffnesses differ by a factor of three, while their swing damping ratios arrive at 0.703, 0.878, 0.993 and 0.997, a band that spans the near critical range without crossing it. One bandwidth on the ground and one damping regime in the air is what a leg should present to a policy, and neither was imposed directly, the first following from the stiffness floors of section 11 and the second from the ceilings of section 11.1.

---

## 7. Identified Actuator Configuration

The gains of section 5 were derived rather than measured, and on 2026-09-03 they were replaced at `environments/environments/assets/config/kscale_identified_cfg.py:12` through `:88` by a set obtained from actuator system identification. These are named the identified set throughout. Every run from `2026-09-03_05-49-21` onward carries them and every run tabulated in the additions at the end of this document predates them.

| Joint | K (Nm/rad) | D (Nm s/rad) | Armature (kg m^2) | Effort Limit (Nm) | Velocity Limit (rad/s) |
|---|---|---|---|---|---|
| Hip Yaw | 500 | 18 | 0.010 | 120 | 40 |
| Hip Roll | 250 | 10 | 0.010 | 120 | 40 |
| Hip Pitch | 200 | 8 | 0.010 | 120 | 40 |
| Knee | 300 | 10 | 0.015 | 120 | 40 |
| Ankle Roll | 170 | 9 | 0.005 | 34 | 20 |
| Ankle Pitch | 170 | 9 | 0.005 | 34 | 20 |

The change is confined to the stiffness and the damping. The armature, the effort limit, the velocity limit, the saturation effort and the three friction parameters of `IdentifiedActuatorCfg` are identical in the two sets, so every inertia in section 4 and every stance torque in section 10 carries over unaltered, and the whole of the difference between the two configurations is the six pairs of numbers above. The friction model is reproduced here for completeness, being an input the analysis below does not use but a later comparison may.

| Joint | Static friction (Nm) | Dynamic friction | Activation velocity (rad/s) | Saturation effort (Nm) |
|---|---|---|---|---|
| Hip Yaw | 0.2 | 0.02 | 0.1 | 120 |
| Hip Roll | 0.3 | 0.02 | 0.1 | 120 |
| Hip Pitch | 0.3 | 0.02 | 0.1 | 120 |
| Knee | 0.8 | 0.02 | 0.1 | 120 |
| Ankle Roll | 0.1 | 0.02 | 0.1 | 34 |
| Ankle Pitch | 0.1 | 0.02 | 0.1 | 34 |

One caveat governs everything that follows. No measurement record, test procedure or fitting residual for the identification is held in this workspace, so the six pairs are taken as given and are not audited here. The analysis below therefore establishes what the identified set implies for this robot under the same relations the calculated set was judged by, and where the two disagree the document records the disagreement rather than adjudicating it, since a measured gain and a derived gain are not evidence of the same kind.

---

## 8. Analysis of Identified Gains

The relations are those of section 3, unchanged,

```
w_n  = sqrt(K / I)
zeta = D / (2 sqrt(K I))
```

evaluated against both inertias of section 4, the swing inertia of section 4.7 and the stance reflected inertia of section 4.9. The final column gives the swing natural frequency as a fraction of the Nyquist bound of 157.08 rad/s, that bound following from the tighter of the two control periods the KScale tasks impose, `w_nyq = pi / T` at `T` of 0.02 s, being decimation 4 over a `sim.dt` of 0.005 at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:1037`.

| Joint | K | D | I_swing | w_swing | z_swing | I_stance | w_stance | z_stance | w_swing / w_nyq |
|---|---|---|---|---|---|---|---|---|---|
| Hip Pitch | 200 | 8 | 1.17000 | 13.07 | 0.261 | 1.18166 | 13.01 | 0.260 | 0.083 |
| Hip Roll | 250 | 10 | 1.01697 | 15.68 | 0.314 | 1.60635 | 12.48 | 0.250 | 0.100 |
| Hip Yaw | 500 | 18 | 0.02763 | 134.52 | 2.422 | 0.61603 | 28.49 | 0.513 | 0.856 |
| Knee | 300 | 10 | 0.12076 | 49.84 | 0.831 | 1.44496 | 14.41 | 0.240 | 0.317 |
| Ankle Pitch | 170 | 9 | 0.00760 | 149.56 | 3.958 | 4.85781 | 5.92 | 0.157 | 0.952 |
| Ankle Roll | 170 | 9 | 0.00574 | 172.10 | 4.555 | 5.37942 | 5.62 | 0.149 | 1.096 |

Three readings follow, and they are of unequal weight.

Every joint of the identified set is underdamped in stance, and this is the reading that governs the rest. The stance damping ratios run from 0.149 at the ankle roll to 0.513 at the hip yaw against the 0.70 target section 11.1 sets, with the hip pitch at 0.260, the hip roll at 0.250, the knee at 0.240 and the ankle pitch at 0.157. The damping restoring the target is `D = 2 zeta sqrt(K I_stance)`, which asks 21.5 at the hip pitch against the configured 8, 28.1 at the hip roll against 10, 24.6 at the hip yaw against 18, 29.2 at the knee against 10 and 40.2 and 42.3 at the two ankles against 9. The set therefore carries between a fifth and three quarters of the damping its own stiffnesses ask for, and the deficit is worst where the stiffness is highest. The consequence is oscillatory rather than fatal, a ratio of 0.25 giving an overshoot of some 44 per cent to a step and a settling that rings for several cycles, and it is the direction of error section 4.9 names as the unsafe one, since the joint rings while carrying the body weight. It is also the behaviour the training record shows, the two runs on these gains carrying a base angular velocity root mean square of 6.52 and 5.63 rad per second against 1.60 under the calculated set.

The distal joints are overdamped in SWING and underdamped in stance, and the pair of statements is not a contradiction but the arithmetic of section 4.9. The ankle roll sits at a swing damping ratio of 4.555 and a stance ratio of 0.149, the ankle pitch at 3.958 and 0.157, and the hip yaw at 2.422 and 0.513, because their two inertias differ by factors of 937, 639 and 22.3 and one damping value must serve both. The practical reading is that these joints behave as first order lags in the air, with time constants `D / K` of 0.053 s at the ankles and 0.036 s at the hip yaw, being between two and five control periods and therefore visible to the policy as a delay, while on the ground they ring. No choice of damping removes both symptoms, which is why section 11.1 sizes for stance and accepts the swing consequence, and why the identified set, which sizes for neither, arrives at the worse of the two everywhere.

The three joints whose swing damping ratio exceeds unity are numerically divergent, and this is the reading that supersedes the first. Section 11.1.1 gives the criterion, an explicitly integrated damper being stable only while `D dt / I_swing` stays below two, and the identified set stands at 3.258 at the hip yaw, 5.919 at the ankle pitch and 7.838 at the ankle roll. The spectral radii of the corresponding swing modes are 2.584, 5.391 and 7.491, rising to 3.170, 6.413 and 8.845 at the worst corner of the randomisation envelope, so a disturbance at the ankle roll is multiplied by seven and a half at every physics step and reaches the effort limit from any perturbation within four of them, in every environment of the randomisation envelope rather than in some fraction of it. The first reading above, that these joints are underdamped in stance and ring, describes a servo that does not exist. What the logs record as ringing is this divergence saturating in the air and re-entering stance as an impulse, and it is why the base angular velocity root mean square of 6.52 and 5.63 rad per second is an order of magnitude above what a stance damping ratio of 0.15 to 0.51 would produce. No stiffness addresses it and the only remedies are a smaller damping or a smaller physics timestep.

The ankle roll swing natural frequency exceeds the Nyquist bound. At 172.10 rad/s against 157.08 it stands at 1.096 of that bound, and the ankle pitch in swing at 149.56 stands at 0.952 of it. This is the one entry in the table that is not a matter of preference, and it is the ceiling section 11.8 sets the calculated ankle stiffness of 120 to respect, that value standing at 144.55 rad/s and 0.920 of the bound. A mode above the Nyquist frequency of the loop that drives it cannot be represented in the sampled system and appears instead as an alias at a lower frequency, so the ankle roll under these gains is not a joint whose response the controller shapes but a joint whose response the controller samples too slowly to see. Two mitigations apply. The joint is heavily overdamped in swing, at 4.555 having little oscillatory mode to alias in the first place, and the frequency is a swing quantity only, the same joint standing at 5.62 rad/s in stance where section 4.9 gives its true reflected inertia. The remedy is available without touching the gain, the flat terrain task at decimation 2 giving a control period of 0.01 s and a Nyquist bound of 314.16 rad/s at `kscale_base_env_cfg.py:1012` under which every entry in the table is comfortably sampled.

---

## 9. Calculated and Identified Gain Comparison

The two sets are compared here over every quantity this document computes for a gain set. The stance torques are omitted from the comparison because they do not admit one, the torque of section 10.1 depending on the mass distribution, the pose and the ground reaction force alone and not on any gain, so both sets face the identical load and differ only in what they do under it.

| Quantity | Joint | Calculated | Identified | Ratio or change |
|---|---|---|---|---|
| Stiffness K (Nm/rad) | Hip Pitch | 200 | 200 | 1.00 |
| | Hip Roll | 250 | 250 | 1.00 |
| | Hip Yaw | 100 | 500 | 5.00 |
| | Knee | 300 | 300 | 1.00 |
| | Ankle Pitch | 120 | 170 | 1.42 |
| | Ankle Roll | 120 | 170 | 1.42 |
| Damping D (Nm s/rad) | Hip Pitch | 21.5 | 8 | 0.37 |
| | Hip Roll | 28.0 | 10 | 0.36 |
| | Hip Yaw | 3.3 | 18 | 5.45 |
| | Knee | 12.0 | 10 | 0.83 |
| | Ankle Pitch | 1.3 | 9 | 6.92 |
| | Ankle Roll | 1.0 | 9 | 9.00 |
| Stance damping ratio z | Hip Pitch | 0.699 | 0.260 | -0.439 |
| | Hip Roll | 0.699 | 0.250 | -0.449 |
| | Hip Yaw | 0.210 | 0.513 | +0.303 |
| | Knee | 0.288 | 0.240 | -0.048 |
| | Ankle Pitch | 0.027 | 0.157 | +0.130 |
| | Ankle Roll | 0.020 | 0.149 | +0.129 |
| Swing spectral radius | Hip Pitch | 0.953 | 0.983 | stable in both |
| | Hip Roll | 0.929 | 0.975 | stable in both |
| | Hip Yaw | 0.823 | 2.584 | DIVERGENT |
| | Knee | 0.847 | 0.766 | stable in both |
| | Ankle Pitch | 0.381 | 5.391 | DIVERGENT |
| | Ankle Roll | 0.359 | 7.491 | DIVERGENT |
| Stance frequency w (rad/s) | Hip Pitch | 13.01 | 13.01 | 1.00 |
| | Hip Roll | 12.48 | 12.48 | 1.00 |
| | Hip Yaw | 12.74 | 28.49 | 2.24 |
| | Knee | 14.41 | 14.41 | 1.00 |
| | Ankle Pitch | 4.97 | 5.92 | 1.19 |
| | Ankle Roll | 4.72 | 5.62 | 1.19 |
| Sag, double support (rad) | Hip Pitch | 0.037 | 0.037 | 1.00 |
| | Knee | 0.027 | 0.027 | 1.00 |
| | Ankle Pitch | 0.026 | 0.019 | 0.73 |
| Sag, single support (rad) | Hip Pitch | 0.068 | 0.068 | 1.00 |
| | Knee | 0.055 | 0.055 | 1.00 |
| | Ankle Pitch | 0.051 | 0.036 | 0.71 |
| Settled base height (m) | whole body | 0.76213 | 0.76249 | +0.4 mm |
| Settled sag from nominal (mm) | whole body | 9.5 | 9.1 | 0.96 |
| Ankle roll lateral deflection (rad) | Ankle Roll | 0.035 | 0.025 | 0.71 |
| Saturating position error (rad) | Hip Pitch | 0.600 | 0.600 | 1.00 |
| | Hip Roll | 0.480 | 0.480 | 1.00 |
| | Hip Yaw | 1.200 | 0.240 | 0.20 |
| | Knee | 0.400 | 0.400 | 1.00 |
| | Ankle Pitch | 0.283 | 0.200 | 0.71 |
| | Ankle Roll | 0.283 | 0.200 | 0.71 |
| Highest swing w against Nyquist | Ankle Roll in both | 0.920 | 1.096 | 1.19 |

The comparison resolves into four statements.

The two sets have converged on the stance requirement and differ almost not at all upon it. Every sagittal sag is now equal between them but for the ankle pitch, where the identified 170 sags 0.019 rad against the calculated 120's 0.026, both inside the 0.05 rad budget, and the settled base heights differ by 0.4 mm. Three of the six stiffnesses coincide outright, the hip pitch at 200, the hip roll at 250 and the knee at 300, which is a corroboration of the derivation rather than an argument for it, the identification having measured a machine that the stance analysis independently predicts.

The identified set carries more stance damping at four joints of six and none of it survives the integrator, which is the single most important line of this comparison and the one that reverses its apparent verdict. On the stance damping ratio alone the identified set looks the better servo at the hip yaw and the two ankles, reaching 0.513, 0.157 and 0.149 against 0.210, 0.027 and 0.020. Evaluating the swing spectral radius of section 11.1.1 at those same three joints gives 2.584, 5.391 and 7.491, every one of them past unity in every environment of the randomisation envelope, so the modes those damping ratios describe do not decay, they grow. A spectral radius of 7.491 multiplies a disturbance by that factor at every physics step and reaches the effort limit from any perturbation within four steps.

The consequence is that the identified set's stance damping figures are not measurements of a servo's behaviour but of a servo's configuration, and the two part company at three joints. This also supplies the mechanism for an observation section 9 previously recorded without one. The identified set was described as ringing where the calculated set settles, its two runs carrying a base angular velocity root mean square of 6.52 and 5.63 rad per second against 1.60, and that ringing was attributed to underdamping. It is not underdamping. A joint whose stance damping ratio is 0.513 does not ring at 6.5 rad per second, and a joint whose swing mode grows by a factor of 2.6 per physics step at the hip yaw and 7.5 at the ankle roll does. The identified set's ringing is numerical divergence in the swing phase, saturating against the effort limit and re-entering the stance phase as an impulse, and no adjustment of the stiffness addresses it.

At the two joints where both sets are stable, the hip pitch and the hip roll, the calculated set is the better damped by factors of 2.7 and 2.8 in stance, and there the ordinary reading holds.

The identified set is more aggressive at the two joints the derivation bounds by the Nyquist ceiling. Its ankle roll at 170 stands at 172.10 rad per second in swing, being 1.096 of the 157.08 rad per second bound, where the calculated 120 stands at 144.55 and 0.920 of it. The calculated ceiling was chosen for exactly this reason, section 11.8 taking the largest round stiffness that keeps the joint sampled, and the identified value crosses it. The margin is small and the joint is heavily overdamped in swing at 4.555, so it has little oscillatory mode to alias, and the remedy is available without touching the gain, the flat terrain task at decimation 2 giving a Nyquist bound of 314.16 rad per second at `kscale_base_env_cfg.py:1012` under which every entry is comfortably sampled.

The identified set buys its stiffness with authority. The saturating position error `q_sat = tau_limit / K` is the offset at which the proportional term reaches the effort limit and ceases to be proportional, and it is smaller under the identified set at every joint but the hip pitch and the hip roll, most sharply at the hip yaw where 0.600 rad becomes 0.120 and at the two ankles where 0.142 becomes 0.100. Against the action scale of 0.25 rad at `kscale_base_env_cfg.py:148` a saturated action clips at all six joints under the identified set and at four of six under the calculated set, so the difference is one of degree rather than of kind, and section 11.10 shows that four joints reach that condition in ordinary walking under either.

---

## 10. Gravitational and Stance Load Analysis

This section supplies the quantity the swing derivation omitted. It corresponds to [BRS.md](BRS.md) section 7, and it is developed further here because on this robot it is decisive rather than merely informative.

### 10.1 Method

With the foot planted the joint reacts the ground reaction force. The static torque at a joint is obtained by balancing the free body distal to it, which comprises the ground reaction force acting at the centre of pressure together with the weight of every distal link,

```
tau_j = a_j . [ (r_cop - o_j) x F_grf ]  +  sum_k m_k a_j . [ (c_k - o_j) x g ]
```

where `F_grf` is the vertical ground reaction force borne by that foot, being half the body weight in symmetric double support and the whole of it in single support, and the centre of pressure is placed directly beneath the ankle roll axis, which is the balanced case. The static sag at that joint is then `tau_j / K`, being the position error the proportional term must sustain to hold the load.

### 10.2 Static Stance Loads

| Joint | tau, double support | tau, single support | K | sag, double | sag, single | Effort limit |
|---|---|---|---|---|---|---|
| Hip Pitch | 7.423 | 13.651 | 200 | 0.037 | 0.068 | 120 |
| Hip Roll | 0.008 | 0.007 | 250 | 0.000 | 0.000 | 120 |
| Hip Yaw | 0.000 | 0.000 | 100 | 0.000 | 0.000 | 120 |
| Knee | 8.109 | 16.468 | 300 | 0.027 | 0.055 | 120 |
| Ankle Pitch | 3.150 | 6.133 | 120 | 0.026 | 0.051 | 34 |
| Ankle Roll | 0.000 | 0.000 | 120 | 0.000 | 0.000 | 34 |

The hip roll, hip yaw and ankle roll carry no static load in a balanced stance, their axes passing through the centre of pressure or lying parallel to the force. They are not thereby unimportant, and section 11.8 sizes the ankle roll on a different criterion entirely, but they impose no stiffness floor by this argument.

The sagittal joints do carry load, and the ordering is worth noting. The knee's 8.109 Nm exceeds the hip pitch's 7.423 Nm despite the hip carrying more of the leg, because the knee sits closer to the line of the ground reaction force in the fore and aft sense and its moment arm to the centre of pressure is the larger of the two at this pose.

### 10.3 The Ankle in Single Support

The ankle effort limit is the tightest actuator constraint on the robot and it deserves its own examination, since the table above places the single support ankle pitch load at 6.133 Nm, which is comfortable, only because the centre of pressure was placed beneath the ankle. Two figures are in play and the distinction governs the rest of this section. The robstride_02 peak torque of 17 Nm, which section 5.1 establishes as the class this joint is named for, and the doubled experimental limit of 34 Nm that section 5.2 adopts and that the configuration now carries.

A robot in single support must place its centre of mass over its stance foot, and at the nominal pose the body centre of mass sits 0.126 m medial of the foot. Holding that offset without leaning would demand 198.89 times 0.126, being 25.07 Nm about the ankle roll axis. At the robstride_02 peak of 17 Nm that is not attainable at all, the demand exceeding the limit by a factor of 1.47, and shifting the centre of pressure to the outer edge of the sole relieves at most 198.89 times 0.0423, being 8.41 Nm, leaving 16.65 Nm, which is 98 per cent of the limit and only reached with the foot on the verge of rolling over its own edge. At the doubled 34 Nm the same demand is 74 per cent of the limit and is met with the sole flat, which is the whole of what the doubling buys and the reason section 5.2 adopts it.

The comparison with the BRS is instructive rather than alarming. That robot's body centre of mass sits 0.1298 m medial of its foot, almost exactly the KScale figure, but its sole is 0.194 m wide against 0.0846 m, so the centre of pressure shift relieves 56.94 Nm and leaves 19.22 Nm against a 131 Nm limit, a margin of 6.8 times where the KScale has none.

The resolution is that neither robot in fact holds this load at the ankle. A biped balances in single support by leaning the pelvis over the stance foot, which the hip roll accomplishes, so the equilibrium ankle roll torque is near zero and the figure above describes a transient rather than a stance. The finding is therefore not that the KScale ankle is undersized in the ordinary sense, but that the KScale has far less lateral authority in reserve than the BRS and cannot rely on ankle strategy for lateral balance at all. It must lean. Section 10.6 makes the statement quantitative and sharper than this section can, the single support hold torque at the ankle roll being 25.07 Nm against a limit of 17, so the joint cannot hold the nominal pose on one foot even momentarily. This is recorded because it bears directly on the hip roll gain, which section 11.5 sets at 250 Nm/rad for exactly this reason, the hip roll being the joint that must supply the lean the ankle cannot, and because it predicts that lateral disturbance rejection will be this robot's weakest axis.

### 10.4 Where the Robot Actually Settles

The static sag figures above are first order, and they understate the truth because sag is self reinforcing. As a knee flexes under load its moment arm to the ground reaction force grows, which demands more torque, which produces more flexion. The honest question is therefore not how far a joint sags but whether an equilibrium exists at all, and that is answered by minimising the total potential energy with the sole planted,

```
E(q) = M g z_com(q)  +  sum_j K_j (q_j - q_j_default)^2
```

over the three sagittal joints of both legs, subject to the joint limits. The minimiser is the configuration the robot comes to rest in under its own weight, and the base height there is what `pen_base_height` and the `low_height` termination will actually observe.

| Gain set | Settled hip | Settled knee | Settled ankle | Settled base height | Sag from nominal |
|---|---|---|---|---|---|
| Swing-derived, knee 25 and ankle 7 | +0.051 | 1.523 | +0.193 | 0.4362 m | 335.4 mm |
| Knee 50, ankle 15 | | 0.848 | +0.047 | 0.6571 m | 114.5 mm |
| Knee 100, ankle 25 | | 0.548 | -0.119 | 0.7375 m | 34.1 mm |
| Knee 150, ankle 40 | -0.035 | 0.484 | -0.198 | 0.7522 m | 19.4 mm |
| Calculated, knee 300 and ankle 120 | -0.0445 | 0.4359 | -0.2698 | 0.76213 m | 9.5 mm |

The first row is the result that explains the failed training run. Under the swing derived gains the robot's equilibrium is a deep squat at a knee of 1.523 rad and a base height of 0.4362 m, which is 335 mm below the nominal 0.77161 m. The robot was never standing. It was collapsing to a crouch the instant it was set down, and the only question was how far past that crouch its momentum carried it.

The last row is the calculated set. A settled height of 0.76213 m against a nominal 0.77161 m is a sag of 9.5 mm, which is a stance rather than a collapse, and it leaves 0.492 m of margin above the `low_height` termination floor of 0.27 m. The intermediate rows show that the transition from collapse to stance is not gradual, the base height rising 220 mm between the first two rows and 5 mm between the last two, so a stiffness deficit at the knee and the ankle expresses itself as a cliff rather than as a proportional sag, and a set that clears the cliff by any margin behaves much as any other.

A note on the height reward follows from this. `pen_base_height.target_height` is set to 0.772, being the unloaded kinematic height, and the robot will settle 9.5 mm below it and pay a small standing penalty it can never discharge. The figure is within the noise of the term's own weighting and no change is proposed, but it is recorded so that a reader does not mistake the residual for a tracking failure.

### 10.5 Analysis for Identified Gains

The sections above were computed for the calculated set. This one repeats them for the identified set of section 7, and it begins with the observation that determines how much repetition is actually required.

The static stance torque of section 10.1 does not depend on any gain. Restating it,

```
tau_j = a_j . [ (r_cop - o_j) x F_grf ]  +  sum_k m_k a_j . [ (c_k - o_j) x g ]
```

every quantity on the right is a mass, a position or a force. The joint axis `a_j` and the origin `o_j` come from the URDF and the pose, the centre of pressure `r_cop` from the contact geometry, the ground reaction `F_grf` from the body weight and the support fraction, and the link centres of mass `c_k` from the inertial blocks. The stiffness enters nowhere. The identified set therefore faces exactly the loads section 10.2 tabulated, and the two sets are distinguished not by the torque demanded of them but by the position error at which each delivers it.

| Joint | tau, double support | tau, single support | K identified | sag, double | sag, single | Effort limit |
|---|---|---|---|---|---|---|
| Hip Pitch | 7.423 | 13.651 | 200 | 0.037 | 0.068 | 120 |
| Hip Roll | 0.008 | 0.007 | 250 | 0.000 | 0.000 | 120 |
| Hip Yaw | 0.000 | 0.000 | 500 | 0.000 | 0.000 | 120 |
| Knee | 8.109 | 16.468 | 300 | 0.027 | 0.055 | 120 |
| Ankle Pitch | 3.150 | 6.133 | 170 | 0.019 | 0.036 | 34 |
| Ankle Roll | 0.000 | 0.000 | 170 | 0.000 | 0.000 | 34 |

The sag is `delta q_j = tau_j / K_j`, being the position error a proportional term must hold to produce the required torque, since the actuator commands `tau = K (q_des - q) - D q_dot` and at rest the derivative term vanishes. Against the 0.05 rad budget of section 11.1 every joint now passes in double support, where the calculated set failed at the ankle pitch with 0.063 rad. The improvement is exactly the stiffness ratio, the ankle pitch sag falling by the factor 50 over 170, being 0.294, and the knee by 200 over 300, being 0.667. The hip pitch is unchanged because its stiffness is unchanged.

In single support the whole body weight passes through one foot, so `F_grf` doubles and the torque slightly more than doubles, the increment above a factor of two coming from the distal link weights which do not scale with the support fraction. The largest single support demand is the knee's 16.468 Nm against a 120 Nm limit, and the ankle pitch's 6.133 Nm against 34 Nm, so no joint approaches saturation from the stance load alone under either set. The single support sags of 0.068, 0.055 and 0.036 rad are all within the budget, which the calculated set achieved at none of the three.

The equilibrium is obtained as in section 10.4 by minimising the total potential energy with the sole planted,

```
E(q) = M g z_com(q)  +  sum_j K_j (q_j - q_j_default)^2
```

over the three sagittal joints of both legs, subject to the joint limits. The first term falls as the robot crouches and the second rises, so the minimiser is the configuration at which the gravitational moment and the spring moment balance, and raising a stiffness steepens the second term and moves the minimum toward the default pose. This is the honest form of the sag question, because the first order figure `tau / K` understates the truth, a flexing knee lengthening its own moment arm to the ground reaction force and demanding more torque still.

| Gain set | Settled hip | Settled knee | Settled ankle | Settled base height | Sag from nominal |
|---|---|---|---|---|---|
| Calculated, knee 300 and ankle 120 | -0.0445 | 0.4359 | -0.2698 | 0.76213 m | 9.5 mm |
| Identified, knee 300 and ankle 170 | -0.0448 | 0.4358 | -0.2789 | 0.76249 m | 9.1 mm |

The identified set settles 0.4 mm higher, at a sag of 9.1 mm against 9.5 mm, and it does so with a marginally deeper ankle, minus 0.2789 rad against minus 0.2698, the two sets having converged on the same equilibrium once the calculated set is derived against the stance regime. The settled configuration is therefore no longer a quantity that distinguishes the two, and the distinctions that remain are the damping ratios of section 8 and the saturating position errors of section 9.

The lateral criterion of section 11.8 is the one place where the improvement over the swing derived set is large. Taking the worst lateral case as the centre of pressure held at the edge of the sole under half the body weight, the demand is unchanged at 99.45 times 0.0423, being 4.21 Nm, and the deflection `delta q = tau / K` falls from 4.21 over 20, being 0.210 rad, to 4.21 over 170, being 0.025 rad, the calculated 120 giving 0.035 rad. Against a travel of plus and minus 0.26180 rad that is a fall from 80 per cent of the travel to 9.5 per cent of it. The addition of 2026-09-03 records that the ankle roll stood within 0.02 rad of a stop for 80.5 per cent of steps in the reference run and 86.4 per cent in the last, and this is the calculation that predicts it, the calculated stiffness having left the joint 0.052 rad of margin where the identified one leaves 0.235 rad.

The cost is stated last because it is the only one. The proportional term saturates at the position error

```
q_sat = tau_limit / K
```

beyond which the commanded torque is clipped at the effort limit and further position error buys nothing. The action term at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:148` scales a saturated action to 0.25 rad of offset from the default pose, and the static sag adds to it, so the relevant comparison is `q_sat` against that 0.25 rad.

| Joint | Effort limit | K calculated | q_sat calculated | K identified | q_sat identified |
|---|---|---|---|---|---|
| Hip Pitch | 120 | 200 | 0.600 | 200 | 0.600 |
| Hip Roll | 120 | 150 | 0.800 | 250 | 0.480 |
| Hip Yaw | 120 | 120 | 1.000 | 500 | 0.240 |
| Knee | 120 | 200 | 0.600 | 300 | 0.400 |
| Ankle Pitch | 34 | 50 | 0.680 | 170 | 0.200 |
| Ankle Roll | 34 | 20 | 1.700 | 170 | 0.200 |

Under the identified set the hip yaw at 0.240 rad and the two ankles at 0.200 rad reach their effort limit at or below a full scale action of 0.25 rad. Under the calculated set no joint does, the thinnest margin being the ankles at 0.283 rad. Section 11.2 performs this check at the same configured 0.25 rad and gives the feasible stiffness band each set is judged against, which the calculated set satisfies at every joint and the identified set at three. The finding is therefore that the identified set introduces clipping in the nominal case where the calculated set does not, and the consequence is that the policy's effective action range at the hip yaw and the ankles is set by the effort limit rather than by the action scale. Whether that is a defect depends on the hardware the identification measured, since an actuator that genuinely cannot exceed its limit is described correctly by a configuration that clips, and it is recorded here rather than resolved for the reason section 7 gives.

### 10.6 Single Support, the Proximal Hold Torque and the Gravitational Stiffness

Sections 10.2 through 10.5 read the stance load from the ground reaction force beneath a planted sole, which is the right quantity for the sag of a joint under weight and the wrong one for two questions section 11 must answer. Both concern single support, which the commanded gait occupies for 0.76 of every cycle, and both are properties of the assembly ABOVE the joint rather than below it.

With one foot planted the links distal to a joint are held by the ground and the links proximal to it, being the torso, the opposite leg and whatever of this leg lies above the joint, are what the joint carries and accelerates. Two quantities follow from the potential energy of that assembly as the joint rotates about its own axis, taken by central difference at the nominal pose with the sole planted. The first derivative is the torque the joint must supply to hold the pose on one foot. The second derivative is the stiffness gravity itself contributes, and its sign decides whether the joint sits in a potential well or stands on a hill,

```
tau_hold = | d U_proximal / d theta |          k_grav = - d^2 U_proximal / d theta^2
```

with the convention that a POSITIVE `k_grav` is destabilising. A joint whose own stiffness does not exceed a positive `k_grav` has a net restoring torque of the wrong sign and cannot hold the assembly upright at any deflection, which is a floor no swing analysis and no double support load analysis produces.

| Joint | Proximal mass (kg) | tau_hold (Nm) | k_grav (Nm/rad) | Effort limit | Reading |
|---|---|---|---|---|---|
| Hip Pitch | 12.30 | 1.199 | -23.94 | 120 | restoring, hold trivial |
| Hip Roll | 12.82 | 24.949 | -20.09 | 120 | restoring, hold is 21 per cent of the limit |
| Hip Yaw | 15.26 | 2.503 | -0.17 | 120 | neutral, as a vertical axis must be |
| Knee | 17.65 | 1.618 | +32.28 | 120 | destabilising, buckles under load |
| Ankle Pitch | 19.74 | 8.717 | +84.90 | 34 | destabilising, hold is 26 per cent of the limit |
| Ankle Roll | 19.85 | 25.067 | +84.90 | 34 | destabilising, hold is 74 per cent of the limit |

Four readings follow and each governs a decision in section 11.

The hip roll carries 24.95 Nm in single support, being 21 per cent of its effort limit, where its double support load is 0.008 Nm. The two figures differ by a factor of three thousand and the derivation must use the larger, since the robot spends three quarters of its cycle in the regime that produces it. At a stiffness of 150 that torque sags the joint 0.166 rad against an outward abduction travel of only 0.209 rad, consuming 80 per cent of the range before the policy commands anything, which is the arithmetic behind the measured hip roll stop residency of 78.9 per cent in the reference run.

The two ankles share a gravitational stiffness of 84.90 Nm per radian because both axes pivot the same assembly about the same planted sole, and the figure is the linearised inverted pendulum stiffness of a 19.8 kg mass at the height its centre lies above the ground. A stiffness below it cannot hold the robot up about that axis, which places the ankle roll at 20 Nm per radian and the ankle pitch at 50 firmly on the wrong side of the boundary.

The ankle roll hold torque of 25.07 Nm is the figure on which the effort limit of that joint turns, and it is the reason section 5.2 doubles it. At the robstride_02 peak of 17 Nm the demand exceeds the limit by a factor of 1.47, and no stiffness corrects that, the constraint being on torque rather than on deflection, so the joint cannot hold the nominal pose on one foot at all and the robot must carry its centre of mass over the stance foot by other means. This is the quantitative form of the lateral authority problem section 10.3 states, and it is the mechanical reason the splay of the addition of 2026-08-28 is not merely tolerated but necessary. At the doubled 34 Nm the hold is 74 per cent of the limit and the joint holds single support directly, so the splay ceases to be the only lateral authority available and becomes one option among two. Every statement in this document that treats the splay as necessary is conditional on the 17 Nm figure and is withdrawn at 34, which is the single most consequential difference the doubling makes and the reason it is worth an experiment rather than an assumption.

The hip yaw's gravitational stiffness of minus 0.17 confirms that a vertical axis contributes no gravitational term to first order, which section 4.9 assumes and which is the one place the earlier reasoning about this joint was correct. Its INERTIAL term is a different matter and section 4.9 gives it.

---

---

## 11. Ideal Gain Derivation

### 11.1 Methodology

The derivation proceeds in four steps, and the order matters because the steps are not of equal authority. The stiffness is chosen first against the loads the robot must hold, the damping second against the inertia it must arrest, and the damping is then bounded by the substrate that integrates it, which is a constraint of a different kind from the other two and overrides them because a numerically divergent servo has no dynamics to shape.

The stiffness is bounded below by three requirements, of which the binding one differs from joint to joint, and above by one.

The first floor is the static stance load of section 10.2, the smallest stiffness holding the double support torque within a sag budget, `K_load = tau_2sup / 0.05`. A robot which cannot hold its own weight standing cannot learn anything, so this is a hard constraint. It is silent at the three joints whose axes carry no gravitational moment in a symmetric stance, being the hip roll, the hip yaw and the ankle roll.

The second floor is the gravitational stiffness of section 10.6 and it applies where the first is silent as readily as where it binds. With one foot planted a joint carries the assembly above it, and rotating the joint moves that assembly's centre of mass. Where the movement lowers it the assembly stands on a potential hill and gravity contributes a NEGATIVE stiffness, so a servo whose own stiffness does not exceed it cannot hold the joint at any deflection whatever, the net restoring torque having the wrong sign,

```
K_grav = -d^2 U_proximal / d theta^2   where positive, otherwise silent
```

This criterion decides the two ankles, which no other criterion reaches, and it is the reason a stiffness sized on the swing inertia leaves this robot unable to stand on one foot.

The third floor is authority. The joint must be able to COMMAND the posture that carries the body over the stance foot, and the largest offset a saturated action can command is the action scale, so

```
K_auth = tau_hold_1sup / action_scale
```

with `tau_hold_1sup` the single support hold torque of section 10.6. This binds at the hip roll and the ankle roll, the two joints that carry the lateral moment.

The ceiling is the Nyquist bound of the control loop taken against the SWING inertia, `K_nyq = I_swing * omega_nyquist^2`, since the swing regime is where a joint reaches its highest natural frequency and where an aliased servo would first misbehave. At a control period of 0.02 s the bound is 157.08 rad per second and the ceiling it imposes is loose at every proximal joint and tight at the two ankles, whose swing inertias are almost entirely rotor.

One quantity that an earlier reading of this problem treated as a ceiling is demoted here to a diagnostic. The stiffness at which a full scale action, added to the static sag, first reaches the effort limit is reported in section 11.2 and is not a constraint. Where it falls below a floor the joint clips under a full scale action, and that is accepted, because a joint which clips occasionally under an extreme command fails less often than a joint which cannot hold the load it carries continuously. The measurement of section 11.10 shows this is not a hypothetical trade, four joints of six clipping for about half of every episode under gains chosen to avoid clipping entirely.

A criterion that suggests itself and is deliberately NOT used should be recorded, since its omission is a judgement rather than an oversight. It is tempting to size a stiffness by dividing a logged joint torque by an allowed deflection, on the reading that the logged torque is a disturbance the joint must resist. The reading does not hold at a position controlled joint. The torque a proportional derivative servo delivers is the sum of a disturbance the joint meets from outside and the spring's own response to the offset the policy has commanded, and only the first is a quantity a stiffness may be sized against. The two are separated by inverting the control law, the commanded deviation being `(tau + D q_dot) / K`, and where that reconstruction returns an excursion comparable to the action scale the logged torque is the policy driving the joint rather than the world disturbing it. Applied to the hip yaw over the replay of `2026-08-28_04-50-51`, the reconstruction returns a ninety fifth percentile commanded deviation of 1.28 rad against an actual deviation of 0.77 rad, so the joint is being driven and not disturbed, and a stiffness derived from its logged torque would be derived from the wrong quantity. Raising a stiffness against a command the policy is free to raise in turn does not reduce the deflection, it multiplies the torque, and the joint arrives at its effort limit rather than at the intended posture.

The damping follows last, and it takes the smallest of four quantities,

```
D = min( 2 zeta sqrt(K I_stance) ,  tau_limit / q_dot_p95 ,  2 sqrt(K I_swing) ,  I_swing / (dt_phys x m_rand) )
```

at a target damping ratio of 0.70 against the stance reflected inertia of section 4.9.

The first term is the design target and it matters that it is taken against the stance inertia rather than the swing inertia, since the two differ by up to a factor of 937 on this robot and a damping sized in the air arrives in stance at 0.70 divided by the square root of that ratio, which is 0.02 at the ankle roll.

The second is what the actuator can deliver, being the largest damping whose torque does not consume the whole effort limit at the ninety fifth percentile joint speed the robot is measured to reach. Under the doubled limits of section 5.2 it binds at no joint of the six, where under the original limits it bound at four, and section 11.10 records the measurement that motivated the doubling.

The third and the fourth are the constraints the substrate imposes and they are the subject of section 11.1.1, which no earlier treatment of this robot contained and which decides the damping at four joints of six.

### 11.1.1 An explicit servo has a stability bound, and it is tighter than any dynamic criterion

The actuator class this robot uses is not a PhysX joint drive. `IdentifiedActuator` at `environments/environments/actuators/actuator_pd.py:18` derives from `DCMotor` and therefore from `IdealPDActuator`, and it sets `control_action.joint_positions = None` at `environments/environments/actuators/actuator_pd.py:37`, so the proportional derivative law is evaluated in Python and its result is handed to the physics engine as a joint effort and nothing else. The servo is an explicit element. Isaac Lab evaluates it inside the decimation loop at `IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:182` through `:190`, once per physics step rather than once per control step, so the loop runs at the reciprocal of `sim.dt`, being 200 Hz at the 0.005 s this robot's tasks configure.

An explicit damper integrated at a finite step is conditionally stable and the condition is not on the damping ratio. Writing the semi implicit update the engine performs, the velocity advancing on the torque computed from the state at the start of the step and the position advancing on the new velocity, the joint's free response is governed by

```
a = dt^2 K / I_swing        b = dt D / I_swing

M = [[1 - a, dt (1 - b)], [-a / dt, 1 - b]]      det M = 1 - b      trace M = 2 - a - b
```

and the mode grows without bound wherever the spectral radius of `M` exceeds unity. The determinant alone gives the necessary condition `b < 2`, which is the form section 11 of [/ws/context/kscale-opensource.md](../../context/kscale-opensource.md) states for the K-Bot and uses to identify the wrist as the binding joint of that robot at 1.68 of the bound. The full spectral radius is tighter than the determinant wherever `a` is not small, and on this robot it is not small at the two ankles, whose stiffness is large against a swing inertia that is almost entirely rotor.

The inertia in the criterion is the SWING inertia and this is the whole of why the constraint had been missed. A leg carrying its own weight presents the stance reflected inertia of section 4.9, which is large, and a servo damped against it is far inside the bound. The same servo, with the same damping, meets the swing inertia the moment the foot leaves the ground, and the commanded gait spends 0.76 of its cycle with one foot in the air. The ratio between the two inertias is 22.30 at the hip yaw and 638.96 and 936.96 at the two ankles, so a damping chosen for the ground is met by an inertia between one and three orders of magnitude smaller within the same stride.

Two consequences are taken as constraints rather than as observations. The first is that the swing damping ratio is capped at unity, `D <= 2 sqrt(K I_swing)`, which is a dynamic argument as well as a numerical one, an overdamped joint placing a foot slowly being a defect in its own right. The second is that the worst draw of the randomisation envelope must satisfy `b <= 1`, giving `D <= I_swing / (dt_phys x m_rand)` with `m_rand` the product of the largest damping scale and the reciprocal of the smallest inertia scale. Section 4.8 gives the damping envelope as plus and minus ten per cent and the armature and mass envelope as plus and minus five, so `m_rand = 1.1 / 0.95 = 1.158` and the ceiling is `0.864 I_swing / dt_phys`, which is `172.7 I_swing` at the configured timestep.

The choice of `b <= 1` rather than `b < 2` is deliberate and is the margin the criterion carries. At `b` above unity the velocity reverses sign at every physics step, which is a bounded mode and not a divergence, but it is a mode at the Nyquist frequency of the integrator and it enters the observation, the action rate penalty and the torque rate penalty as noise the policy cannot act on. At `b` above two the mode grows and the joint chatters against its effort limit. Requiring the worst randomised draw to sit at or below unity places the nominal configuration at 0.86 of it and leaves the whole envelope monotone.

Section 11.9 gives the resulting ceilings and the joints they bind. It should be recorded that adopting the RobStride armatures of section 5.1 would relax this constraint at the one joint where it costs the most, the hip yaw swing inertia rising from 0.02763 to 0.03763 with an armature of 0.0200 in place of 0.0100 and its damping ceiling rising from 4.77 to 6.50 accordingly, which is not enough to reach the stance target of 11.0 and is therefore a mitigation rather than a resolution.

### 11.2 Saturation Check

The maximum proportional torque a joint can command is `K` times the largest position error it will see, and where that exceeds the effort limit the proportional term ceases to be a proportional term. Two errors are relevant. The action term at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:148` carries a scale of 0.25 rad, so a saturated action commands that much offset from the default pose, and the static sag of section 10.2 adds to it.

| Joint | K | Effort limit | K x 0.25 | Static sag torque | Sum | Headroom | q_sat | Travel |
|---|---|---|---|---|---|---|---|---|
| Hip Pitch | 200 | 120 | 50.0 | 7.4 | 57.4 | 52.2 per cent | 0.600 | 3.264 |
| Hip Roll | 250 | 120 | 62.5 | 0.0 | 62.5 | 47.9 per cent | 0.480 | 2.478 |
| Hip Yaw | 100 | 120 | 25.0 | 0.0 | 25.0 | 79.2 per cent | 1.200 | 3.142 |
| Knee | 300 | 120 | 75.0 | 8.1 | 83.1 | 30.8 per cent | 0.400 | 2.705 |
| Ankle Pitch | 120 | 34 | 30.0 | 3.2 | 33.2 | 2.4 per cent | 0.283 | 1.396 |
| Ankle Roll | 120 | 34 | 30.0 | 0.0 | 30.0 | 11.8 per cent | 0.283 | 0.524 |

Every joint of the six carries a positive headroom under the limits of section 5.2, which is to say that a full scale action added to the static sag commands a torque the actuator delivers without clipping anywhere. This is the second thing the doubling buys and it is worth separating from the first. Under the original limits four joints of six carried a negative headroom, the ankle pitch at minus 95.0 per cent and the knee at minus 38.5, so a full scale action at those joints commanded a torque that was clipped and the proportional term ceased to be a proportional term over most of its commanded range.

The quantity that governs a joint's behaviour beyond its linear range is `q_sat`, the position error at which the joint reaches its limit, and the comparison that decides the matter is against the joint's mechanical travel rather than against the action scale. At the ankle roll `q_sat` is 0.283 rad against a travel of 0.2618, so the joint cannot reach its effort limit anywhere within its own range of motion and is a proportional element over the whole of it. At the original 17 Nm the same figure was 0.142 rad, which is 54 per cent of the travel, and a stiffness of 20 reached its limit only at 0.850 rad, which is 3.2 times the entire travel, so that joint never behaved as a spring anywhere in its range and the section 11.10 measurement of an 80 per cent stop residency is the consequence. The three settings fall on either side of two different boundaries and the resulting behaviour is not monotone in the stiffness alone.

The table is computed at an action scale of 0.25 and the result is conditional on it, which must be stated because the configuration has carried both 0.25 and 0.4 at different dates and the reference run `2026-08-28_04-50-51` trained at 0.4. Repeating the arithmetic at that scale gives headrooms of 27.1, 16.7, 66.7, minus 6.8, minus 50.4 and minus 41.2 per cent, so the knee and the two ankles return to clipping and the clip free property above is a property of the pair rather than of the limits alone. A run pairing the gains of section 12 with an action scale of 0.4 is therefore a different experiment from the one section 11.2 describes, and whichever is chosen the choice should be deliberate.

The margin at the ankle pitch is the thinnest in the table at 2.4 per cent and is reported rather than removed. Lowering that stiffness to widen it would carry the joint below the gravitational floor of 84.90 which section 11.7 shows it must clear, and the trade is settled in favour of the floor, a joint that clips on the largest commands it will rarely issue failing less often than a joint that cannot hold the machine above it upright at all.

### 11.3 Hip Pitch

The load requirement gives `K_min = 7.423 / 0.05 = 148.5 Nm/rad` and the gravitational stiffness is minus 23.94, a restoring value, so the second floor is silent. The Nyquist ceiling against a swing inertia of 1.17 is 28869 and is not remotely binding. 200 Nm/rad is taken, holding the double support sag at 0.037 rad within the budget and the single support hold at 0.006 rad.

```
K = 200,  I_stance = 1.18166
D_target   = 2 x 0.70 x sqrt(200 x 1.18166) = 21.5
D_effort   = 120 / 3.019 = 39.8
D_swingcrit= 2 x sqrt(200 x 1.17000) = 30.6
D_integ    = 172.7 x 1.17000 = 202.1
D = 21.5 Nm s/rad
```

The design target stands, every ceiling being loose by a factor between 1.4 and 9.4, and the joint arrives at a stance damping ratio of 0.699 and a swing ratio of 0.703. This is the only joint of the six at which all four quantities agree that the target is attainable, and the reason is that its swing and stance inertias differ by one per cent, so there is no regime for the derivation to fall between. The stance natural frequency of 13.01 rad per second sits inside the 5 to 15 band [BRS.md](BRS.md) section 3.3 prescribes for a proximal joint.

### 11.4 Knee

The load requirement gives `K_min = 8.109 / 0.05 = 162.2 Nm/rad` in double support and 329.4 in single, and the gravitational stiffness is plus 32.28, the knee being the one sagittal joint whose stance mode gravity destabilises, since flexing it lowers the body. Both floors are cleared with margin at 300 Nm/rad, which holds the double support sag at 0.027 rad and the single support hold at 0.005, and stands 9.3 times above the gravitational floor. The Nyquist ceiling is 2980 and is not binding.

```
K = 300,  I_stance = 1.44496
D_target   = 2 x 0.70 x sqrt(300 x 1.44496) = 29.2
D_effort   = 120 / 5.791 = 20.7
D_swingcrit= 2 x sqrt(300 x 0.12076) = 12.0
D_integ    = 172.7 x 0.12076 = 20.9
D = 12.0 Nm s/rad
```

The swing critical ceiling binds and the joint arrives at a stance damping ratio of 0.288 against a swing ratio of 0.997. This is the first of the four joints at which the substrate rather than any choice made here decides the damping, and the consequence is recorded rather than resolved. The ordering of the four candidates is worth noting because it recurs below. The stance target is the largest, the effort limit and the integration ceiling sit close together in the middle, and the swing critical value is the smallest, so a derivation carrying only the first two would have selected 20.7 and left the joint overdamped in swing at a ratio of 1.72 with a stability number of 0.86 at the nominal draw and 0.99 at the worst. That is inside the bound but at its edge, and the margin the swing critical ceiling supplies is what keeps it there.

### 11.5 Hip Roll

The load requirement is silent, the double support torque being 0.008 Nm, and the gravitational stiffness of minus 20.09 is restoring, so neither of the first two floors reaches this joint. The third does, and it is the criterion this joint turns on. Section 10.6 gives a single support hold torque of 24.95 Nm, the stance hip roll carrying the lateral moment of the torso and the swinging leg together, and the authority floor is therefore `24.95 / 0.25 = 99.8 Nm/rad`.

The travel budget raises it further and is the binding form of the requirement on this robot. Outward abduction is capped at 0.209 rad by the asymmetric joint limits of section 2.3, and the hold torque sags the joint by `24.95 / K`. Requiring that sag to consume no more than half the available abduction gives `K >= 24.95 / 0.105 = 238 Nm/rad`. 250 is taken, sagging 0.100 rad and leaving 0.109 rad of abduction for the policy to command with.

```
K = 250,  I_stance = 1.60635
D_target   = 2 x 0.70 x sqrt(250 x 1.60635) = 28.1
D_effort   = 120 / 0.927 = 129.4
D_swingcrit= 2 x sqrt(250 x 1.01697) = 31.9
D_integ    = 172.7 x 1.01697 = 175.7
D = 28.0 Nm s/rad
```

The hip roll is the joint at which every ceiling is loose by the widest margin, its measured ninety fifth percentile speed being 0.927 rad per second against 3.0 to 9.6 elsewhere and its swing inertia being the second largest on the robot, so the design target stands and the joint arrives at a stance damping ratio of 0.699 and a swing ratio of 0.878. That it is the slowest joint on the robot is itself a consequence of the stiffness having been too low, a joint pinned against its stop for four fifths of an episode having nowhere to move.

### 11.6 Hip Yaw

The joint carries no static load in double support, its axis being vertical and passing near the centre of pressure, and its gravitational stiffness of minus 0.17 is negligible in both directions. The single support hold torque is 2.50 Nm, giving an authority floor of 10.0 Nm/rad, and the sag budget gives 50.1. Both are weak, and the joint would be left undetermined across an order of magnitude if nothing else spoke, which is the position every criterion above leaves it in.

Bandwidth matching decides it, and the ground is the stance inertia of 0.61603 kg m squared that section 4.9 computes. A leg whose proximal servos share one natural frequency presents the policy with a single timescale to shape a trajectory against, and a leg mixing a slow hip with a fast yaw presents two, the faster of which the policy is free to exploit for motion the slower joints cannot follow. The other three proximal joints arrive at stance natural frequencies of 13.01, 12.48 and 14.41 rad per second, so matching the hip yaw to that band gives

```
K = omega^2 I_stance = 12.74^2 x 0.61603 = 100 Nm/rad
```

At that stiffness the single support hold sags the joint 0.025 rad, the saturating position error is 0.600 rad which is 2.4 times the action scale, and the swing natural frequency is 60.2 rad per second, being 0.38 of the Nyquist bound. Every criterion is satisfied with margin, which is the first time this joint has had one that binds it at all.

```
K = 100,  I_stance = 0.61603
D_target   = 2 x 0.70 x sqrt(100 x 0.61603) = 11.0
D_effort   = 120 / 5.159 = 23.3
D_swingcrit= 2 x sqrt(100 x 0.02763) = 3.3
D_integ    = 172.7 x 0.02763 = 4.8
D = 3.3 Nm s/rad
```

This is the joint at which the two regimes conflict most sharply and it is worth setting out why the smaller quantity wins, since the stance target is 3.3 times it and the argument for the stance target is the same argument that fixed the stiffness. The stance reflected inertia of 0.61603 is 22.30 times the swing inertia of 0.02763, and a damping of 11.0 chosen against the first meets the second the moment the foot leaves the ground. Section 11.1.1 gives the consequence, a stability number of 1.99 at the nominal draw and 2.31 at the worst, so the mode grows rather than decaying and the joint chatters against its effort limit within a third of a second of any disturbance. Sixty per cent of the randomisation envelope lies past the bound.

The stance target is therefore not attainable at this joint and no arrangement of the stiffness recovers it, since raising `K` raises the target and the ceiling in the same proportion, the first as the square root and the second not at all. What is attainable is 3.3, giving a swing damping ratio of 0.993 and a stance ratio of 0.210, and the honest description of that servo is that it is correctly damped in the air and lightly damped on the ground. The lightly damped stance mode is arrested by the policy and by the other joints rather than by the servo, which is the same accommodation sections 11.7 and 11.8 make at the two ankles for the same reason.

One consequence of the reduction must be stated because it reverses a property an earlier reading of this joint valued. A hip yaw damped at 11.0 is sluggish in swing and cannot splay the foot quickly, which is convenient given that splay is a fault this robot's reward set spends four terms attempting to price, and at 3.3 that incidental brake is gone. The convenience was never a design argument and should not have been treated as one. The commanded joint position is set by the action and the action is unbounded, so no damping restricts the posture the policy may ask for and a bound on the splay must be imposed as a bound, whether by a clip, a joint limit or a termination. What removes the need for the splay is not a gain at this joint at all but the ankle roll authority that section 5.2 supplies.

Recommended K = 90 to 115 Nm/rad, D = 3.0 to 3.6 Nm s/rad.

### 11.7 Ankle Pitch

Both of the first two floors reach this joint and the second is the higher. The load requirement gives `K_min = 3.150 / 0.05 = 63.0 Nm/rad`, and the gravitational stiffness of plus 84.90 is the true floor, being the stiffness below which the joint cannot hold the machine above it upright at any deflection, the whole robot pivoting about a planted sole being an inverted pendulum in the sagittal plane. The Nyquist ceiling against a swing inertia of 0.00760 is 187.6 Nm/rad and is genuinely binding here, this joint being almost entirely rotor in the air. The band is therefore 85 to 188 and 120 is taken near its geometric centre, clearing the gravitational floor by 41 per cent and reaching 0.80 of the Nyquist bound in swing.

```
K = 120,  I_stance = 4.85781
D_target   = 2 x 0.70 x sqrt(120 x 4.85781) = 33.8
D_effort   = 34 / 9.561 = 3.56
D_swingcrit= 2 x sqrt(120 x 0.00760) = 1.91
D_integ    = 172.7 x 0.00760 = 1.31
D = 1.3 Nm s/rad
```

The integration ceiling binds and it is the tightest of the four by a factor of 26 against the stance target. The stance damping ratio is 0.027 and nothing raises it, so the stance mode of this joint is essentially undamped and is arrested by the policy and by the other joints rather than by the servo. This is a property of the robot rather than of the derivation, an ankle carrying 4.86 kg m squared of reflected inertia through a servo whose own inertia is 0.0076 having no setting at which it damps both.

The doubling of section 5.2 is worth examining here because it does not help. Raising the effort limit from 17 to 34 raises the effort ceiling from 1.78 to 3.56, which is the largest relative relaxation any ceiling receives anywhere in this derivation, and the damping does not rise with it because the integration ceiling of 1.31 was always the smaller quantity and is untouched by the effort limit. What the doubling buys at this joint is the saturation headroom of section 11.2 and the velocity headroom of section 11.10, not damping. A ceiling that was believed to bind and did not is a useful thing to have measured, since it removes the effort limit from the list of quantities a future retuning of this joint may usefully move.

### 11.8 Ankle Roll

The load requirement is silent in double support and the gravitational stiffness of plus 84.90 is the floor, identical to the ankle pitch's because both axes pivot the same assembly about the same planted sole. The authority floor is higher still and is the binding one, section 10.6 giving a single support hold torque of 25.07 Nm and hence `K_auth = 25.07 / 0.25 = 100.3 Nm/rad`. The Nyquist ceiling against a swing inertia of 0.00574 is 141.7 Nm/rad, the tightest ceiling on the robot. The band is 100 to 142 and 120 is taken within it, reaching 0.92 of the Nyquist bound in swing.

The lateral criterion of section 10.3 is satisfied incidentally and is no longer the binding statement it was. Holding the centre of pressure at the edge of the sole demands 4.21 Nm, which deflects the joint 0.035 rad at this stiffness against a travel of 0.2618, where a stiffness of 20 deflects it 0.210 rad and consumes 80 per cent of the travel on a single static requirement.

```
K = 120,  I_stance = 5.37942
D_target   = 2 x 0.70 x sqrt(120 x 5.37942) = 35.6
D_effort   = 34 / 9.592 = 3.54
D_swingcrit= 2 x sqrt(120 x 0.00574) = 1.66
D_integ    = 172.7 x 0.00574 = 0.99
D = 1.0 Nm s/rad
```

The integration ceiling binds and it is the tightest on the robot, 36 times below the stance target, this joint having the smallest swing inertia and the largest stance inertia of the twelve. The stance damping ratio is 0.020 and the reading is the same as at the ankle pitch.

Two consequences must be stated plainly and the first is the hardest constraint on this machine. The single support hold torque of 25.07 Nm exceeds the robstride_02 peak of 17 Nm by a factor of 1.47, so at that limit this joint cannot hold the nominal pose on one foot at any stiffness whatever, the constraint being on torque rather than on deflection. The robot must then bring its centre of mass over the stance foot by other means, and the two available to it are hip roll adduction, which section 11.5 shows is itself constrained to 0.209 rad of travel, and widening the effective base of support by yawing the slender sole, which the addition of 2026-08-28 measures at 31.5 per cent of lateral extent for 7.7 degrees of splay. That the robot adopts the second is not a reward defect at 17 Nm, it is the only lateral authority its ankle does not have.

At the doubled 34 Nm the hold is 74 per cent of the limit and the joint holds single support directly, which is the change section 5.2 exists to test. The deflection it holds at is the second consequence and it is not comfortable. A hold torque of 25.07 Nm against a stiffness of 120 deflects the joint 0.209 rad, which is 80 per cent of its 0.2618 rad travel, so the joint holds the pose but arrives at it with almost no margin for the policy to command with. Restoring half the travel would require a stiffness of 191 Nm/rad and the Nyquist ceiling of 141.7 forbids it, so the two requirements are in genuine conflict at this joint and the conflict is not resolvable by any gain. What resolves it is a wider sole or a lower centre of mass, both of which are mechanical changes, and this is recorded as the standing hardware question that section 12 states.

### 11.9 Summary of Ideal against Previous Gains

| Joint | Swing-derived K, D | Calculated K, D | Binding stiffness floor | Binding damping term | w_stance | z_stance |
|---|---|---|---|---|---|---|
| Hip Pitch | 200, 21.5 | 200, 21.5 | double support load, 148.5 | stance target | 13.01 | 0.699 |
| Hip Roll | 150, 17.2 | 250, 28.0 | abduction travel budget, 238 | stance target | 12.48 | 0.699 |
| Hip Yaw | 15, 0.85 | 100, 3.3 | bandwidth matching, 100 | swing critical damping | 12.74 | 0.210 |
| Knee | 25, 2.45 | 300, 12.0 | single support sag, 329 | swing critical damping | 14.41 | 0.288 |
| Ankle Pitch | 7, 0.32 | 120, 1.3 | gravitational stiffness, 84.9 | swing integration | 4.97 | 0.027 |
| Ankle Roll | 5, 0.24 | 120, 1.0 | single support authority, 100.3 | swing integration | 4.72 | 0.020 |

Two features of the table deserve comment because they overturn expectations a swing analysis creates.

The first is that no two joints are decided by the same criterion. The hip pitch stands on the double support load, the knee on the single support sag, the hip roll on a travel budget peculiar to its asymmetric limits, the ankle pitch on the gravitational stiffness of an inverted pendulum, the ankle roll on the authority to command a lateral hold, and the hip yaw on nothing but the wish that the proximal servos share a bandwidth. A derivation that applies one rule to six joints will therefore be wrong at five of them, and the earlier swing bandwidth derivation, which applied exactly one rule, was.

The second is the direction of the correction, and it runs opposite ways for the two gains. Sizing the STIFFNESS on the swing inertia understates every distal value severely, by a factor of 24 at the ankle roll and 17 at the ankle pitch, because the swing inertia of a distal joint is almost entirely rotor while its stance inertia is the whole machine. It also inverts the natural frequency ordering, a swing analysis placing the ankles at 59 to 81 rad per second and the hips at 12, where the correct stance analysis places the ankles at 4.7 to 5.0 and the hips at 12.5 to 14.4. The ankles are the SLOWEST joints on this robot in the regime that matters and the fastest in the regime that does not, and every conclusion that follows from calling them fast, the aliasing warnings among them, is an artefact of the wrong inertia.

The DAMPING inverts that lesson exactly and this is the part that is easiest to get wrong, having got the stiffness right. The stance inertia says how much damping a joint wants and the swing inertia says how much it may be given, and the second is a bound where the first is a preference. Four joints of six therefore take a damping the stance analysis calls too small by factors between 2.4 and 36, and the reason is that the same servo meets both inertias within one stride. A derivation that applies the stance inertia to both gains, which is the natural reading of the correction the previous paragraph describes, produces a hip yaw damping of 11.0 that diverges in sixty per cent of randomised environments. Section 11.1.1 gives the criterion and section 11.9's damping column gives the joints it decides.

The comparison against the BRS is retained because it corroborates the sagittal figures without being an argument for them. The BRS knee at 200 Nm/rad sags 0.123 rad in double support and the KScale knee at 300 sags 0.027, so the KScale is the better provided of the two at the joint where the two robots are most nearly comparable. Where the two diverge is at the distal and lateral joints, and the reason is the ankle effort limit rather than the derivation, the robstride_02 peak of 17 Nm on a 198.89 N robot against a sole 0.0846 m wide leaving no configuration in which the ankle roll holds single support.

A third comparison is available and it is the closest this workspace has to an independent check. The K-Bot recorded at [/ws/context/kscale-opensource.md](../../context/kscale-opensource.md) is a humanoid of 36.719 kg from the same vendor, carrying the same RobStride classes at the same named joints, and section 9 of that document reconstructs its published damping as a critical damping construction performed against the FREE LIMB inertia at a ratio near 0.96. That is the swing critical ceiling of section 11.1 applied as the design rule rather than as a bound, and section 10 of the same document measures what it costs, the K-Bot's stance damping ratios falling to 0.573 at the hip pitch, 0.112 at the knee and 0.017 at the ankle. The two derivations are therefore each other's complement. The K-Scale rule is right about the bound and silent about the load, this document's earlier readings were right about the load and silent about the bound, and the set above is the first to carry both. The corroboration is worth its weight because the K-Bot's hip yaw at a proportional gain of 100, which is this robot's figure exactly, carries a published damping of 3.419, which is within four per cent of the 3.3 that section 11.6 derives here by a different route on a robot of a different mass.

### 11.10 The measured saturation, and the limits it establishes

Replaying `2026-08-28_04-50-51` at seed 42 shows what the robot actually did under the swing derived gains that preceded this section, at the original effort limits of 60 and 17 Nm and the original velocity limits of 20 and 10 rad per second. The measurement serves two purposes. It is the empirical warrant for demoting the no clip ceiling from a constraint to a diagnostic, and it is the evidence on which section 5.2 doubles both limits.

| Joint | Effort limit | Fraction of samples at 99 per cent of the limit or above |
|---|---|---|
| Hip Pitch | 60 | 0.463 and 0.464 |
| Hip Roll | 60 | 0.536 and 0.493 |
| Hip Yaw | 60 | 0.000 and 0.000 |
| Knee | 60 | 0.002 and 0.003 |
| Ankle Pitch | 17 | 0.546 and 0.501 |
| Ankle Roll | 17 | 0.555 and 0.509 |

The two figures are the left and the right joint. That run trained and was replayed at an action scale of 0.4, and its gains were chosen so that no joint would clip at the configured scale. Four of six clipped regardless, for about half of every episode. A ceiling that a set constructed to respect it violates at two thirds of its joints is not describing the constraint that binds, and the quantity it should be replaced by is `q_sat`, the position error at which the joint reaches its limit, which section 11.2 reports.

Three consequences follow and each is measured rather than argued.

The stiffness of a joint that saturates half the time is not the quantity that governs its behaviour, so a gain change at such a joint buys less than the derivation implies. The ankle roll stands within 0.02 rad of its travel limit for 73.9 per cent of samples at a stiffness of 20 and 66.9 per cent at a stiffness of 170, a change of 8.5 times in the gain moving the residency by seven points, because both settings saturate. What the derivation buys at that joint is not the removal of saturation but the crossing of the gravitational floor of 84.90, below which the servo has the wrong sign and above which it has a basin of attraction, and the two settings compared above fall on opposite sides of it.

The action scale is not the remedy it appears to be, and this is worth stating because it is the obvious candidate. The commanded offset is the product of the scale and the policy output, so halving the scale would halve the position error at every joint at once if the policy output were fixed. It is not. The action term carries `clip` unset, so the scale multiplies an unbounded Gaussian and the policy compensates, the reconstructed ninety fifth percentile commanded hip yaw deviation standing at 0.91 rad at a scale of 0.4 and 0.95 rad at 0.25 on configurations differing in that line alone. The measured effect at the ankle roll is in the wrong direction outright, the stop residency rising from 0.669 to 0.765 under the reduction. The action scale governs the resolution of the policy's control and it does not govern the reachable joint set.

The stance load floor and the gravitational floor remain the correct constraints and the stiffnesses that satisfy them are the ones above. What the measurement adds is the knowledge that at four joints those stiffnesses are, for much of the episode, not the operative quantity, and that the operative quantity is the effort limit.

The velocity limit is the fourth consequence and it is the largest single number in this document, having gone unexamined until the actuator classes of section 5.1 prompted the comparison. `DCMotor`, from which `IdentifiedActuator` derives, does not clip the torque at a constant. It clips at a four quadrant torque speed curve, the available torque falling linearly from the stall value at rest to zero at the declared velocity limit, and with `saturation_effort` set equal to `effort_limit` throughout this configuration the full effort is available only at zero speed. The ninety fifth percentile joint speeds measured in the same replay give the following.

| Joint | q_dot p95 (rad/s) | Original limit | Fraction of it | Torque available there (Nm) | Doubled limit | Fraction | Torque available (Nm) |
|---|---|---|---|---|---|---|---|
| Hip Pitch | 3.019 | 20 | 0.15 | 50.94 | 40 | 0.08 | 110.94 |
| Hip Roll | 0.927 | 20 | 0.05 | 57.22 | 40 | 0.02 | 117.22 |
| Hip Yaw | 5.159 | 20 | 0.26 | 44.52 | 40 | 0.13 | 104.52 |
| Knee | 5.791 | 20 | 0.29 | 42.63 | 40 | 0.14 | 102.63 |
| Ankle Pitch | 9.561 | 10 | 0.96 | 0.75 | 20 | 0.48 | 17.75 |
| Ankle Roll | 9.592 | 10 | 0.96 | 0.69 | 20 | 0.48 | 17.69 |

The two ankles reach 96 per cent of their declared velocity limit at the ninety fifth percentile of their speed distribution, at which point the torque speed curve has left them 0.69 and 0.75 Nm. For the fastest twenty steps of every hundred those joints are not 17 Nm actuators, they are free hinges. The effect compounds in exactly the wrong direction, since the damping torque `D q_dot` is largest at the speed where the available torque is smallest, so the servo loses authority precisely at the moment its damping term is asking for the most. The four proximal joints are nowhere near this condition, reaching 0.05 to 0.29 of their limits with 43 to 57 Nm still available.

Doubling the ankle velocity limit to 20 rad per second raises the available torque at the same measured speed from 0.69 to 17.69 Nm, a factor of 25.5, and it does so without changing the stall torque at all. That single figure is the strongest argument in this document for the doubling of section 5.2 and it is independent of the torque limit argument, since the original 10 rad per second stands against a robstride_02 declared maximum of 37.699 and a firmware ceiling of 44.0, so it was not a specification but an error. The doubled 20 rad per second remains only 0.53 of the declared figure and is therefore still conservative.

The ankle effort limit of 17 Nm is therefore not the only tight constraint at that joint as section 10.3 records. Under the original configuration the ankle was constrained four ways at once, by a torque limit it could not hold single support within, by a velocity limit that removed almost all of that torque at the speeds it reached, by a gravitational stiffness its stiffness stood below, and by an integration ceiling its damping stood above. Sections 5.2, 11.7 and 11.8 address the first three and section 11.1.1 the fourth.

---

## 12. Recommended Final Gain Values

The values below are the calculated set. They stood at `environments/environments/assets/config/kscale_identified_cfg.py` from 2026-08-25 until 2026-09-03, when the identified set of section 7 replaced them, and this section is retained as the record of the derivation rather than as a description of the current configuration. They are reproduced by `python3 scripts/analysis/kscale_stance_analysis.py`, whose default gain set is the calculated one, and the identified set is obtained from the same script with `--gains identified`.

| Joint | K (Nm/rad) | D (Nm s/rad) | w_stance (rad/s) | z_stance | z_swing | b_worst | Sag, double support | Sag, single support hold |
|---|---|---|---|---|---|---|---|---|
| Hip Pitch | 200 | 21.5 | 13.01 | 0.699 | 0.703 | 0.11 | 0.037 rad | 0.006 rad |
| Hip Roll | 250 | 28.0 | 12.48 | 0.699 | 0.878 | 0.16 | 0.000 rad | 0.100 rad |
| Hip Yaw | 100 | 3.3 | 12.74 | 0.210 | 0.993 | 0.69 | 0.000 rad | 0.025 rad |
| Knee | 300 | 12.0 | 14.41 | 0.288 | 0.997 | 0.58 | 0.027 rad | 0.005 rad |
| Ankle Pitch | 120 | 1.3 | 4.97 | 0.027 | 0.681 | 0.99 | 0.026 rad | 0.073 rad |
| Ankle Roll | 120 | 1.0 | 4.72 | 0.020 | 0.602 | 1.01 | 0.000 rad | 0.209 rad |

The `b_worst` column is the stability number `D dt / I_swing` of section 11.1.1 evaluated at the worst corner of the randomisation envelope, and no entry exceeds unity. The spectral radius of the swing mode, computed over four hundred thousand draws of that envelope, exceeds unity in no environment at any joint of the six.

Under this set the robot settles at a base height of 0.76213 m against a nominal 0.77161 m, a sag of 9.5 mm, with 0.492 m of margin above the `low_height` termination floor, and the double support sag lies within the 0.05 rad budget at every joint of the six. The stiffnesses are unchanged from those the load analysis alone prescribes, the whole of the difference the stability criterion makes falling on the damping, so the settled pose and every static figure in section 10 carry over untouched.

The set has one property worth stating separately because it was not aimed at and is the clearest evidence that the stance inertia is the right quantity to size against. The four proximal joints arrive at stance natural frequencies of 13.01, 12.48, 12.74 and 14.41 rad per second, a band 15 per cent wide across joints whose inertias differ by a factor of 2.6 and whose stiffnesses differ by a factor of three. A leg whose proximal servos share one bandwidth presents the policy with a single timescale to shape trajectories against, where a leg mixing a 12 rad per second hip with a 66 rad per second yaw presents two and invites the policy to exploit the fast one. The two distal joints sit at 4.97 and 4.72 rad per second, slower than the proximal band rather than faster, which is the reverse of the ordering a swing analysis predicts and is the direct consequence of a distal joint carrying the whole machine above it once its foot is on the ground.

Three cautions stand and one earlier caution is withdrawn.

The first is that the hip yaw, the knee and the two ankles are damped by the substrate rather than by the design target, at stance damping ratios of 0.210, 0.288, 0.027 and 0.020 against the 0.70 sought, and no gain closes that gap at the configured physics timestep. The remedy, if the stance damping is ever wanted, is not a gain at all. Halving `sim.dt` from 0.005 to 0.0025 doubles every integration ceiling in section 11.9 and would carry the hip yaw to 9.5 and the knee to 24.1, at twice the simulation cost, and that is the trade a future revision should weigh rather than attempting to recover the damping from the gain set.

The second concerns the ankle roll travel. Section 11.8 shows the single support hold deflects that joint 0.209 rad of the 0.2618 rad its limits permit, so the joint holds the pose with 20 per cent of its travel remaining, and the stiffness that would restore half the travel stands above the Nyquist ceiling its swing inertia imposes. The conflict is not resolvable by any gain and its resolution is mechanical, a wider sole or a lower centre of mass.

The third is the standing hardware question and it is now sharper than an open question. Section 5.1 identifies the actuator classes from the joint name suffixes against the published K-Scale Labs metadata, which assigns the robstride_04 a peak torque of 120 Nm, the robstride_03 60 Nm and the robstride_02 17 Nm. The identification is corroborated independently by the three actuator housing meshes, `rs02.stl`, `rs03.stl` and `rs04.stl`, placed largest at the hip pitch and smallest at the shank with mesh volumes of 514.71, 181.16 and 60.63 cubic centimetres, a naming and size ordering consistent with that line. If it holds, the four proximal limits of 120 Nm are correct at the hip pitch and the knee and twice the specification at the hip roll and the hip yaw, and the two ankle limits of 34 Nm are twice the specification, so the set above is not realisable on RobStride hardware at the ankle and the lateral authority of section 5.2 is a simulation result awaiting a hardware answer. What would settle it is a specification, and no manufacturer name, part number or specification reference appears anywhere in the URDF or the surrounding configuration.

The caution that is withdrawn is the one an earlier reading of the ankle limit produced, that four joints of six clip under a full scale action. Under the limits of section 5.2 none does, section 11.2 giving positive headroom at all six, and the ankle roll cannot reach its effort limit anywhere within its own mechanical travel.

---

## 13. The Measurement Scripts

Four scripts produce every figure above. All are pure numpy, scipy and the standard library, with no dependency on Isaac Lab, on torch or on the task package, so that each runs inside the simulation container and equally in a plain interpreter against a checked out URDF.

`scripts/analysis/kscale_sole_analysis.py` parses the URDF, runs forward kinematics to determine which axis of each foot link frame points downward, transforms the collision mesh vertices into the link frame, isolates those lying on the lowest plane, takes their convex hull and reduces it to a requested number of points by repeatedly deleting the vertex whose removal costs the least polygon area. It emits a frame convention check and a fidelity sweep beside the table. It is robot agnostic, taking the URDF path and a regular expression naming the foot links, so it serves the BRS and any future robot as readily as the KScale.

`scripts/analysis/kscale_physical_analysis.py` reports the standing geometry at a given pose, the swing effective inertia at each joint with the natural frequency and damping ratio the configured gains imply, the symmetry flip set composed from the joint origin rotation chains, and an axis aligned bounding box audit of non adjacent link pairs.

`scripts/analysis/kscale_stance_analysis.py` supplies sections 4.9, 8, 9, 10 and 11, being the static stance torque at each joint under the ground reaction force, the settled configuration obtained by minimising the gravitational plus spring potential energy with the sole planted, the stance reflected inertia taken as the proximal assembly's inertia about the joint axis, the single support hold torque and gravitational stiffness of section 10.6 obtained by central difference of that assembly's potential energy, the damping derivation of section 11.1 with its four ceilings, the swing mode spectral radius and randomisation sweep of section 11.1.1, the ankle roll lateral criterion, and the saturation, feasible band and Nyquist checks. Its `stance_inertia_proximal` and `single_support_gravity` are the functions section 4.9 and section 10.6 quote, and the older `stance_inertia`, which reflects the body mass through the vertical lift alone, is retained unchanged so that tables computed before 2026-09-04 reproduce and is used by nothing. It carries two gain sets, selected by `--gains`, the calculated set of section 5 being the default so that every table in this document is reproduced by the bare invocation, and the identified set of section 7 being obtained by `--gains identified`. The action scale and the control period the saturation and Nyquist checks use are exposed as `--action-scale` and `--control-period`. It was written in the pass of 2026-08-25 and it is the script whose absence permitted the failure of section 19.

`scripts/analysis/kscale_reframe_urdf.py` performs the root frame repair of section 14, and `scripts/analysis/kscale_symmetry_selftest.py` asserts the involution and double mirror properties of section 17 without a running simulator.

---

## 14. The Root Frame Convention, and its Repair

The two feet of the KScale as exported were separated in the root frame by the vector 0.2520, 0.0001, 0.0000 in metres, so the lateral axis was x. The two feet of the BRS are separated by 0.000, minus 0.259, 0.000, so its lateral axis is y, which is the Isaac Lab convention.

Three independent lines of evidence agreed and the finding did not rest on the separation vector alone. The two hip pitch joint origins differed only in x, at plus and minus 0.055, and were identical in y and z. The foot's long axis of 0.21 m lay along the root frame y. The toe extended toward minus y from the ankle while the heel sat 0.02 m to plus y, so forward was minus y. The joint naming was nonetheless correct throughout, the joints named as pitch rotating about the lateral axis and those named as roll about the fore and aft axis, so nothing in the URDF was mislabelled. It was the frame convention alone that differed.

The consequences reached four places. Isaac Lab evaluates the velocity command and the base linear velocity observation in the root body frame, so the `lin_vel_x` range of minus 0.3 to 0.8 commanded a leftward sidestep of up to 0.8 m/s while `lin_vel_y` at plus and minus 0.01 pinned the true fore and aft velocity near zero. The height scanner's `GridPatternCfg` of size 1.6 by 1.0 laid its long axis across the robot rather than along its path. The symmetry mirror plane was the y and z plane rather than the x and z plane, which inverts the flip set. And any per axis foot separation statistic read the stride where it intended to read the stance width.

The repair re-expresses the root frame in place by a rotation of plus ninety degrees about z, which is the rotation carrying the robot's forward direction of minus y onto plus x. In place re-expression was preferred over the more conventional insertion of a massless `base_link` above the torso joined by a fixed joint, because it preserves the body count exactly and therefore changes no observation width, several critic observations iterating over bodies and an extra link silently widening them. Every quantity expressed in the root frame was premultiplied by the rotation, being the root link's inertial, visual and collision origins, its inertia tensor which transforms as `R I R^T`, and the origins of the two joints whose parent is the root. Nothing below the root changed, each subtree being rigidly attached to its parent joint and moving with it. Twenty two lines of the URDF were rewritten.

The repair is self checking and was verified three ways. The frame convention check now reports the foot separation as 0.0001, minus 0.2520, 0.0000 and the lateral axis as y. The sole table of section 15 is bit identical before and after, as it must be, the foot link frame being untouched. And the flip set of section 17, recomputed about the x and z plane in the corrected frame, reproduces the set obtained about the y and z plane in the exported frame, since the flip set is a physical property of the robot and not of the frame it is expressed in.

One caution attaches. The URDF is a generated artefact and a re-export from CAD reintroduces the original frame, which is why `kscale_reframe_urdf.py` is kept rather than discarded.

---

## 15. The Sole Geometry

The foot link frame is a full axis permutation away from the BRS convention. Its x is the sole's width, its y is the vertical with the positive sense pointing downward, and its z is the fore and aft length. Both feet use the same mesh under the same collision origin, the visual origin being identical to the collision origin, so the two feet share one table and the mirroring is carried entirely by the link rotations.

The sole is a genuinely flat manufactured surface. Of the 51228 vertices, 462 lie within one millimetre of the extreme, and those 462 span a range of one micron, which is a real machined face and not a numerical accident. A second, shallower flat band exists four millimetres above it, an interior recessed face bordered by the deeper rim, and the outer rim is the physically correct contact surface since it is what first touches the ground under any orientation.

| Quantity | KScale | BRS | Ratio |
|---|---|---|---|
| Sole plane in the link frame | y equal to plus 0.0430 | z equal to minus 0.1240 | |
| Sole depth below the link origin | 0.0430 m | 0.124 m | 0.347 |
| Sole length | 0.2100 m | 0.2612 m | 0.804 |
| Sole width | 0.0846 m | 0.194 m | 0.436 |
| Convex hull vertices on the sole plane | 78 | | |

The KScale therefore stands on a foot nearly as long as the BRS foot but less than half as wide, and at a third of the depth beneath the ankle. A parameter derived from the depth may be scaled by the mass ratio with some confidence, the depth ratio of 0.347 tracking the mass ratio of 0.339 closely, and a parameter derived from the footprint may not. The width ratio of 0.436 against a lateral centre of mass offset that is 0.971 of the BRS figure is the geometric statement of the lateral authority problem section 10.3 sets out.

The figure of 0.19 that an earlier comment in the configuration reported is arithmetically correct as a minimum of the link frame z, and it is the foot's length rather than its depth. That comment's arithmetic was sound and its choice of which axis to call the lowest point was not, because it carried the BRS convention that a foot link's z is its vertical onto a robot for which that is false. The author recorded a suspicion that the number was a misread and withheld the four reward terms depending on it rather than shipping against a figure nobody trusted, which was the right judgement.

The twelve point table the reduction produces is reproduced at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py` as the module constant `KSCALE_SOLE_OFFSETS`, beside the class that consumes it so that the clearance reward and the landing gate cannot drift apart.

The point count is settled by a fidelity sweep rather than chosen. The sweep rotates both the full vertex set and the reduced table through the ankle's own roll and pitch limits, read from the two joints immediately proximal to the foot, and reports the largest height by which the reduced table overestimates the true clearance. The comparison is performed after aligning the sole to face world down, so that the sweep exercises the foot's real roll and pitch axes rather than the arbitrary axes of a permuted link frame. This distinction is not pedantic, sweeping in the unaligned frame reporting an error of 8.399 mm for the twelve point table where the same table measured correctly commits 0.836 mm.

| Points retained | Worst height error over the ankle's travel |
|---|---|
| 4 | 25.116 mm |
| 8 | 2.530 mm |
| 12 | 0.836 mm |
| 16 | 0.657 mm |
| 24 | 0.389 mm |

Twelve points is the knee of that curve, improving on eight by a factor of three and improved upon by sixteen by less than a quarter. It also matches the cardinality of the BRS table and lands within a twentieth of a millimetre of the 0.85 mm fidelity the BRS table achieves over its own travel, which is the closest thing to an independent calibration available. Four points, which a straight sided trapezoid's corners would give, commits 25 mm of error, because the KScale toe is a rounded arc rather than a straight taper and a chord drawn across that arc falls some two centimetres inside the true boundary.

---

## 16. The Nominal Standing Pose

### 16.1 The pose and its history

The nominal pose at `environments/environments/assets/config/kscale_identified_cfg.py` sets the hip pitches to minus 0.1 rad on the right and plus 0.1 on the left, the knees to 0.4 rad, the ankle pitches to minus 0.3 rad, and every other joint to zero. Three earlier poses stood before it and the standing heights are as follows, each being the torso origin above the lowest point of any collision mesh, which is the quantity `base_height_rough_l2` compares against its target on flat ground.

| Pose | Knee | Ankle pitch | Hip pitch | Standing height |
|---|---|---|---|---|
| Zero | 0.00 | 0.00 | 0.00 | 0.77506 m |
| First nominal | 0.30 | -0.15 | 0.00 | 0.79526 m |
| Second nominal | 0.30 | -0.15 | -+0.10 | 0.78352 m |
| Current | 0.40 | -0.30 | -+0.10 | 0.77161 m |

An earlier revision of this document reported 0.7953 m as the standing height and parameterised three reward terms against it. That figure was taken at the first nominal pose while the asset configuration had already acquired the hip pitches, so it described a pose the robot never held, and it is superseded by the 0.77161 m above.

### 16.2 The pose closes the sagittal chain

The three sagittal joints of a leg act about a common axis, as section 2.3 establishes, so the foot's pitch relative to the ground is the sum of the hip pitch, the knee and the ankle pitch taken about the world plus y, and the sole is horizontal exactly when that sum vanishes. Under the anti parallel hip pitch axes the right leg contributes its joint value directly and the left contributes its negation, so both legs give minus 0.1 plus 0.4 minus 0.3, which is zero. The pose is closed.

It is the first nominal pose of this robot that is, and the consequence is larger than the height change.

| Pose | Sagittal sum | Sole tilt | Vertices within 1 mm of the ground | Ankle roll origin height |
|---|---|---|---|---|
| First nominal | +0.150 rad | 8.594 deg | 90 | 0.07091 m |
| Second nominal | +0.050 rad | 2.864 deg | 161 | 0.05244 m |
| Current | 0.000 rad | 0.000 deg | 462 | 0.04300 m |

The 462 figure is the whole machined face of section 15, the one that spans a range of one micron, so the robot now stands on its sole rather than on the rim of its toe. The ankle roll origin height falling to exactly the 0.04300 m sole depth is the same statement read off the forward kinematics, and it is the cheapest check available that a pose is closed, since a tilted foot lifts its link origin above the sole depth and a flat one does not. Preserve that identity if these angles are changed again.

This matters to `pen_feet_regulation` at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:870` in a way that no height parameter records. That term gates a horizontal foot speed penalty on `exp(-(z - r) / s)` with `r` the sole depth of 0.043 m and `s` the decay scale of 0.03 m, and it is designed to reach unity for a foot resting on the ground. At the first nominal pose a foot flat on the ground reported a frame height of 0.07091 m, so the gate evaluated to 0.394 and the term stood at two fifths of its intended authority throughout stance. At the second it evaluated to 0.730. It now evaluates to 1.000. The term is not being retuned, it is being un-detuned by the geometry, and the weight of minus 0.2 should be read against a stance penalty some two and a half times larger than the one that weight was chosen beside.

### 16.3 The crouch is shallow, and the reason is kinematic

The pose is described as a crouch and it lowers the robot by 11.9 mm against the pose that preceded it, of which only 4.0 mm comes from the knee and 9.4 mm from the ankle correction that lets the foot lie flat. Against the zero pose it is lower by 3.5 mm. This is worth recording plainly, because a reader who reads a knee of 0.4 rad as a meaningful squat and parameterises a height reward against that expectation will be surprised by the logged base height.

Sweeping the knee with the ankle held at the closure condition, so that the sole stays flat throughout, gives the following.

| Knee | Ankle pitch | Standing height | Against the zero pose |
|---|---|---|---|
| 0.00 | +0.10 | 0.77015 m | -0.00491 m |
| 0.10 | 0.00 | 0.77484 m | -0.00022 m |
| 0.20 | -0.10 | 0.77666 m | +0.00160 m |
| 0.30 | -0.20 | 0.77558 m | +0.00052 m |
| 0.40 | -0.30 | 0.77161 m | -0.00345 m |
| 0.50 | -0.40 | 0.76479 m | -0.01027 m |
| 0.60 | -0.50 | 0.75519 m | -0.01987 m |
| 0.70 | -0.60 | 0.74290 m | -0.03216 m |
| 0.80 | -0.70 | 0.72804 m | -0.04702 m |
| 0.90 | -0.80 | 0.71077 m | -0.06429 m |
| 0.97 | -0.87 | 0.69734 m | -0.07772 m |

The height is stationary across the first four rows and reaches a maximum near a knee of 0.2 rad, which says that the zero pose is not the fully extended leg, the built in offsets of the thigh and the shank leaving a residual bend that the first fifth of a radian of knee flexion removes rather than adds to. Real crouch begins near 0.5 rad, where the derivative reaches minus 0.068 m per radian, and grows to minus 0.173 m per radian by 0.9 rad. A pose intended to lower the robot appreciably therefore wants a knee between 0.6 and 0.9 rad, not 0.4.

The depth is bounded, and by the ankle rather than the knee. Closure demands an ankle pitch of 0.1 minus the knee, and the ankle pitch lower limit of minus 0.87267 rad from the table of section 2.3 caps the knee at 0.97267 rad, at which the robot stands 0.69734 m, being 0.078 m below the zero pose. The knee's own upper limit of 2.70526 rad is nowhere near binding. Any deeper stance requires either a hip pitch that leans the torso or an ankle that abandons the flat sole, and the second forfeits the gate authority of section 16.2.

### 16.4 The spawn height

The spawn height was lowered from 0.85 m to 0.79 m in the pass of 2026-08-25. The former value stood 0.0784 m above the 0.77161 m stance, so the robot was dropped rather than set down, and it arrived at the ground with 1.24 m/s of vertical speed after a free fall of 0.126 s. That is a real impact rather than a settle, and it compounded the static collapse of section 10.4 by adding momentum to a leg that could not hold the static load in the first place.

The new figure follows the convention the BRS already records at `sd_brs1_identified_cfg.py`, which spawns at its 1.15 m standing height plus a 0.02 m settling margin. The same margin on the measured 0.77161 m gives 0.79161 m, rounded to 0.79 m, leaving 0.0184 m of clearance beneath the sole and an arrival speed of 0.60 m/s. The spawn height and the standing height are one quantity and must be re-measured together whenever the nominal pose moves.

### 16.5 The reward parameters the pose sets

| Parameter | Location | Value | Basis |
|---|---|---|---|
| `pen_base_height.target_height` | `kscale_base_env_cfg.py:829` | 0.772 | The measured 0.77161 m unloaded height |
| `pen_feet_regulation.base_height_target` | `kscale_base_env_cfg.py:876` | 0.772 | Held in step with the above |
| `low_height.minimum_height` | `kscale_base_env_cfg.py:1018` | 0.27 | 0.348 of the standing height |
| `init_state.pos` z | `kscale_identified_cfg.py` | 0.79 | Standing height plus the 0.02 m BRS settling margin |

The termination floor follows the BRS ratio rather than any new argument, that robot terminating at 0.4 m against a 1.15 m stance, which is 0.348, and the same fraction of 0.77161 m giving 0.2685 m.

Four parameters were checked and are deliberately unchanged by the pose. The stance width floor of `pen_feet_distance` stands at 0.24 m because the foot lateral separation remains 0.2520 m at every pose considered here, the hips being unmoved in roll. The sole depth of `pen_feet_regulation.foot_radius` stands at 0.043 m because it is a property of the foot link and not of the pose. The swing height of 0.05 m and the clearance kernel width of 0.02 m stand because both scale against the 0.505 m segment sum leg length, which is pose invariant. The impact force threshold of 290 N stands because it scales against the mass.

### 16.6 An incidental effect on the reset distribution

`reset_joint_by_offset` at `environments/environments/tasks/locomotion/mdp/events.py:350` clamps its sampled position to the soft joint limits before writing it, so a perturbation that reaches past a limit is not an error but a mass of probability piled onto that limit. The knee's soft lower limit is 0.13526 rad, being the midpoint of its zero to 2.70526 rad range less nine tenths of its half range, and the knee default is perturbed twice, once at startup by the plus and minus 0.05 rad of `joint_offsets` and again at every reset by the plus and minus 0.24 rad of `reset_knee_pitch_joints`.

At the former default of 0.3 rad, 15.67 per cent of resets clamped, so roughly one environment in six began its episode with its knee pinned exactly at the soft limit rather than sampled from the intended interval, and `pen_joint_pos_limits` at weight minus 2.0 met a policy already standing on the boundary. At the current default of 0.4 rad the figure is 0.65 per cent. The ankle pitch, whose default moved further, spans minus 0.50 to minus 0.10 rad under the same two perturbations against soft limits of minus 0.80286 and plus 0.45379 rad, so it never clamps at either pose.

---

## 17. The Symmetry Mirror

Under a reflection a rotation about an axis maps to a rotation about the image of that axis carrying the determinant's sign, so the axis transforms as a pseudovector, `a` mapping to `det(M) M a`. Composing each joint's origin rotation chain to the root and comparing each left joint's mirrored world axis against its right partner's actual axis settles the flip set without any appeal to a naming convention or to a roll against pitch rule, which is the method the MorphoSymm framework establishes [1] and the reason the hip pitch flips despite being a pitch joint.

Composing about the x and z plane in the corrected frame gives the following.

| Joint | Left world axis | Mirrored | Right world axis | Flips |
|---|---|---|---|---|
| `hip_pitch_04` | minus y | minus y | plus y | Yes |
| `hip_roll_03` | plus x | minus x | plus x | Yes |
| `hip_yaw_03` | plus z | minus z | plus z | Yes |
| `knee_04` | plus y | plus y | plus y | No |
| `foot_pitch_02` | plus y | plus y | plus y | No |
| `foot_roll_02` | minus x | plus x | minus x | Yes |

This is the physically expected pattern, the roll and yaw degrees of freedom flipping and the pitch degrees of freedom not, with the single exception of the hip pitch, which flips because its left and right URDF axes are genuinely anti parallel. That exception is corroborated independently by the joint limits of section 2.3, and it is exactly the anomaly the BRS symmetry module documents for itself. The two robots are therefore structurally identical under the mirror, the KScale simply adding the live hip yaw to the flip set where the BRS hip yaw is a fixed joint.

Reflecting about the wrong plane, which is what a reader assuming the Isaac Lab convention would have done against the exported frame, yields the flip set of hip yaw, knee and ankle pitch, which is the physically implausible pattern of the pitch joints flipping and the roll joints not. That an incorrect mirror plane produces a plausible looking table is why the frame of section 14 had to be corrected before the mirror was written, rather than the mirror written against the frame as it stood.

Three properties of the mirror are worth stating because each is a silent failure mode rather than a loud one. The joint partner rule must swap the `right_` and `left_` prefix, the BRS rule of swapping a trailing L for a trailing R matching nothing on a robot whose every joint name ends in a digit and degenerating to an identity permutation that raises no error and teaches the policy the robot is symmetric under doing nothing. The body partners need an explicit dictionary rather than any rule at all, for the naming reason recorded in section 2.3. And the gait phase, a sine and cosine pair of one shared clock, negates both channels together, because the two feet are placed in antiphase by the command's offset of 0.5 rather than by the observation, so exchanging the feet is exactly a half cycle shift of the shared clock [2]. A naive treatment of the pair as an even and an odd channel would negate only one and is wrong.

The critic group is mirrored as well as the policy group, since a value function that observes an unmirrored privileged state cannot be invariant under the reflection.

One quantity cannot be determined statically. The number of collision shapes per body, which the mirror of `robot_material_properties` requires, is known only at runtime, and the module carries it as `None` until the count has been read from a running environment, under which that term falls back to an identity mirror rather than to a wrong one. The BRS module carries the same placeholder for the same reason.

The mechanism chosen is data augmentation rather than the mirror loss. The catalogue of four mechanisms distinguishes duplication of transitions through the mirror, an auxiliary equivariance penalty, phase replay and a hard equivariant network, and reports that no single method dominates across robots [3]. The primary symmetry paper excludes the loss from its comparison on the strength of the Isaac Lab finding that augmentation converges faster and behaves better [2][4], and recommends augmentation specifically for intrinsic motion symmetry on a real biped, where actuator and mass asymmetries break the perfect symmetry assumption and a hard constraint becomes brittle under distribution shift. The KScale, like the BRS, carries an asymmetric inertial model. A caution from the same literature bears on the reset events, since a strictly symmetric policy cannot leave a symmetric neutral pose, so training must begin from a noised non neutral posture [3]. The KScale reset events perturb every joint, the hip yaw now included, so this is satisfied, and it is recorded here so that the perturbations are not removed as an economy.

---

## 18. Self Collision and Clearance Audit

No left leg link overlaps any right leg link at either the zero pose or the nominal pose, the two chains remaining separated throughout by the lateral offset of the hip. No link penetrates the ground plane at the spawn height of section 16.4.

Two non adjacent axis aligned bounding box overlaps exist at both poses. The torso overlaps each hip roll link across a region of 0.116 by 0.003 by 0.089 m, two joints removed. Each shank overlaps its own foot, one hop around the ankle bearing, over a region growing from 0.096 by 0.085 by 0.035 m at the zero pose to 0.161 by 0.085 by 0.039 m at the nominal pose. Bounding box overlap is a conservative test and neither pair is necessarily a true mesh intersection.

The configuration reads as internally contradictory on the matter, `kscale_identified_cfg.py` setting `enabled_self_collisions` to False on the articulation properties while setting `self_collision` to True on the spawn configuration. The BRS carries the identical pairing at `environments/environments/assets/config/sd_brs1_identified_cfg.py`, so the KScale is not diverging from its exemplar here and the pairing was deliberately left as it stands, resolving it on the KScale alone being the opposite of parity. In effect the articulation root property governs at runtime, so self collision is off for both robots, which is also why the standing bounding box overlaps above do not produce persistent interpenetration forces.

The exposure this leaves is nonetheless real and is recorded so that it is not rediscovered. `../../context/brs_gait.md` records a confirmed BRS exploit in which a trained policy pressed its legs together to forge a contact signal and thereby defeated every contact keyed reward term at once. The KScale carries the same exposure with the same geometry and, unlike the BRS, carries no instrumentation that would reveal it, no equivalent of the sole clearance logging the BRS work stream built. A forged contact check, a non zero contact force on a foot whose true sole clearance is well above zero, should precede any trust in a KScale contact keyed reward.

---

## 19. The Training Failure of 2026-08-24

The first KScale flat training run was launched on 2026-08-24 at 11:22 and ran to 18000 iterations before it was stopped. Its logs are at `IsaacLab/logs/rsl_rl/kscale_flat/2026-08-24_11-22-46`. It failed completely, and this section records the diagnosis because the evidence is unusually clean and because the failure mode is one a reader could otherwise spend a long time attributing to the reward set.

### 19.1 What the logs show

The controlling observation is the episode length, taken from `Train/mean_episode_length`.

| Iteration | Mean episode length | Mean reward | base_contact | low_height | time_out |
|---|---|---|---|---|---|
| 0 | 12.82 | -59.47 | 0.000 | 0.000 | 0.006 |
| 1000 | 39.97 | -16.23 | 0.000 | 1.000 | 0.000 |
| 5000 | 40.05 | -9.25 | 0.000 | 1.000 | 0.000 |
| 10000 | 40.64 | -7.36 | 0.000 | 1.000 | 0.000 |
| 18000 | 40.96 | -5.59 | 0.000 | 1.000 | 0.000 |

The episode length reaches 40 steps by iteration 1000 and does not move thereafter. Seventeen thousand further iterations buy 0.99 steps, which is one hundredth of a second at the flat task's 100 Hz control rate. The mean reward improves throughout, from minus 16.23 to minus 5.59, so the policy is learning something, but what it is learning is how to be less heavily penalised during a fall it cannot prevent.

The termination breakdown is decisive and it is what separates this failure from an ordinary one. Every episode ends by `low_height` and none by `base_contact`, at every iteration from 1000 onward. The robot is not falling over. A robot that topples strikes the ground with its torso and trips `illegal_contact` on the torso link. This robot sinks vertically, its torso passing below the 0.27 m floor while never touching anything, which is the signature of legs folding underneath a body rather than a body losing its balance over its feet.

Forty steps is 0.4 s. The videos at `videos/train/rl-video-step-440000.mp4` confirm the reading directly, the robot being already collapsed into a deep crouch with its pelvis near the ground by frame 10, which is 0.1 s, and fully down by frame 20.

### 19.2 The root cause

Section 10.4 supplies it. Under the gains that run used, the knee at 25 Nm/rad and the ankle pitch at 7 Nm/rad, the robot's static equilibrium under its own weight is a knee of 1.523 rad at a base height of 0.4362 m. The robot could not stand at any policy, because standing was not a configuration the joint springs admitted. The spawn height of 0.85 m then added a 0.126 s free fall arriving at 1.24 m/s, and the momentum carried the collapse past the 0.27 m floor within the 0.4 s the logs record.

The reward set is not implicated and the reward breakdown corroborates that. `rew_gait` is the largest term in magnitude at minus 0.64 and it never improves, which is exactly what a gait clock reward does when the robot never takes a step. `rew_lin_vel_xy` rises from 0.20 to 0.53 and `pen_ang_vel_xy` recovers from minus 2.86 to minus 0.13, so the shaping terms behave as designed within the fraction of a second available to them. No reward parameter would have changed the outcome.

### 19.3 Why the derivation permitted it

The gains that run used were derived in the pass of 2026-08-24 by placing each joint's natural frequency inside a bandwidth band, computed against the swing inertia of section 4. That derivation was internally correct and it was answering the wrong question, for the reason section 3.2 sets out. No step in it examined whether a joint could hold the robot's weight, because the quantity that would have revealed the problem, the static stance torque of section 10.2, was never computed. The knee's requirement of 8.109 Nm against a stiffness of 25 Nm/rad was visible in one line of arithmetic and nobody performed it.

Two further defects in the same derivation are recorded so that they are not attributed to anything else. The ankle roll at 5 Nm/rad deflected 0.841 rad under the lateral criterion of section 11.8, being 3.2 times its own travel, so that joint sat pinned at its stop whenever the robot leaned. And the spawn height of 0.85 m was left unexamined against a standing height that had moved twice, so the robot was dropped from 78 mm rather than set down.

### 19.4 What was changed

The knee stiffness was raised from 25 to 200 Nm/rad and its damping from 2.45 to 10.0, the ankle pitch from 7 to 50 and 0.32 to 1.7, and the ankle roll from 5 to 20 and 0.24 to 0.5. The hip pitch, hip roll and hip yaw stiffnesses were unchanged, their damping adjusted by less than five per cent to the stance inertia basis. The spawn height was lowered from 0.85 m to 0.79 m. `scripts/analysis/kscale_stance_analysis.py` was written so that the omitted calculation is available and reproducible rather than remembered. Those figures are the immediate repair made on the day and not the calculated set of section 5, which the derivation of section 11 arrived at afterwards and which raises three of them further, the knee to 300 and both ankles to 120. The repair sufficed to stop the collapse, the settled base height rising from 0.4362 m to 0.7522 m at the knee stiffness of 150 that section 10.4 tabulates and higher still at 200, and it did not address the single support requirements that sections 10.6 and 11 later identified.

The settled base height under the new set is 0.76213 m with 0.492 m of margin above the termination floor, against 0.4362 m and 0.166 m before. This is a prediction from statics and not yet an observation, and it should be confirmed by inspecting the first few hundred iterations of the next run, where the diagnostic to watch is the episode length rising off 40 steps and the termination mix moving away from an unbroken `low_height`.

---

## 20. Corrections to the Earlier Record

Five claims standing in the tree before or during this work are refuted and are recorded here so that they are not reintroduced.

The BRS hip yaw joints are declared `type="fixed"` in that robot's URDF and are absent from its joint list entirely, rather than being present with a zero width limit, which several comments asserted. The distinction matters because a zero width limit would still occupy an index in `robot.joint_names` and therefore in the observation and action widths, and a fixed joint does not.

The figure of roughly 0.13 m by which the KScale foot was said to stand off to the side of the hip, which an earlier docstring cited as evidence that the zero pose was anomalous, is measured from the torso centreline and is correct as such, being the sum of two fixed bracket offsets. The foot sits directly beneath the hip roll axis to within four microns. The zero pose is a clean reference frame after all, and the closed chain inverse kinematics derivation the BRS uses would carry over to the KScale if it were wanted.

`scripts/rsl_rl/play.py` selected its foot link pattern and its sole offset table from a pair of hand edited module constants, which was correct as long as exactly one robot was ever dumped and silently ceased to be so on the arrival of a second sole footed biped. A KScale dump read under the BRS pattern resolves no feet at all, and a KScale dump read under the BRS table would have reported clearances measured against a sole three times too deep in the wrong axis. The constants are now resolved at runtime against the articulation's own body names, the BRS path resolving to exactly the values it carried before, and a robot for which no table has been measured resolving its feet while disabling the clearance logging rather than reporting it against another robot's geometry.

The integration plan reported that two KScale ankle joints sat at or above the 157.08 rad/s Nyquist bound of the rough terrain control loop. They do not, under the calculated set. That calculation omitted the armature, which at those two joints exceeds the link inertia by factors of 1.92 and 6.75, and including it puts the highest ankle swing natural frequency at 144.55 rad/s, being 0.920 of Nyquist, which section 11.8 treats as the binding ceiling on that joint's stiffness rather than as a coincidence. The claim as originally stated is withdrawn and section 6 gives the corrected figures, with the qualification recorded in the addition below that the identified set does cross the bound and that every Nyquist figure in this document is a swing quantity, the same joints standing between 4.7 and 5.9 rad/s once their feet are on the ground.

An earlier revision of this document derived the actuator gains from the swing inertia alone and presented the result as settled. It was not, and section 19 records what it cost. The specific error is that the effective inertia of section 4 was treated as the only effective inertia, where it is one of two, and the stance load of section 10 was never computed at all. The swing inertias themselves are unchanged and remain correct for what they describe. What is withdrawn is the conclusion drawn from them, being the knee at 25 Nm/rad, the ankle pitch at 7 and the ankle roll at 5.

Correction, 2026-08-26. The paragraph closing section 2.3 states that the 0.0998 tilt of the hip roll and hip yaw axes against the sagittal plane is a real design cant and not a rounding artefact. That is wrong, and the true source is a pose artefact rather than a hardware one. The table of section 2.3 was composed at the nominal standing pose of section 16, under which `right_hip_pitch_04` sits at minus 0.1 rad, and `right_hip_roll_03` and `right_hip_yaw_03` are rigidly downstream of `right_hip_pitch_04` in the kinematic chain, so the whole distal subtree rotates with it. Recomposing the same chain at the URDF's own zero pose, every joint angle at zero rather than at the nominal pose, gives a `right_hip_roll_03` axis of exactly plus 1, 0, 0 and a `right_hip_yaw_03` axis of exactly 0, 0, plus 1 to floating point precision, the residual of order 1e-6 tracing to the URDF's own truncation of pi over 2 to 1.5708 rather than to any cant. The figure 0.0998 is sin of 0.1 exactly, and 0.9950 is cos of 0.1 exactly, so the reported tilt is the hip pitch flexion angle of the standing pose propagating down the chain to its two downstream joints, not an independent mounting offset carried in their own origin rotations. `right_hip_roll_03`'s origin rpy of `0 -1.5708 0` and `right_hip_yaw_03`'s origin rpy of `-1.5708 -0 -0`, at `environments/environments/assets/urdf/solefoot/kscale/kscale.urdf:623` and `:615` respectively, are themselves exact quarter turns carrying no cant term. The effective inertia figures of sections 4.2 and 4.3, and any reward or gain derivation that reads the tilt as fixed hardware geometry rather than as an artefact of the pose at which the axis was evaluated, should be understood accordingly. The three sagittal axes remaining exactly parallel is unaffected by this correction, since hip pitch, knee and ankle pitch all rotate about the same lateral axis regardless of pose.

Addition, 2026-08-26, made in the same pass as the feet heading reward of `../plans/kscale_integration.md` section 5, which is the first term to depend on this geometry. The direction a KScale foot points, taken as the toe and therefore as the foot link's negative z after section 15, is a function of the hip yaw joint alone. The ankle roll axis lies along the toe direction itself and moves the heading by 0.0001 degrees at its limit, and the three sagittal joints cannot yaw the foot at all. The coefficient relating the two follows the pose exactly as the axis tilt above does, being 1.000000 at the URDF zero pose, 0.995 at the nominal stance, 1.073 at a hip pitch of minus 0.5 rad and 1.524 at minus 1.0 rad, so it varies by more than half across the travel of a joint the swing phase sweeps most of. A quantity derived from the 0.995 must therefore carry the pose it was measured at, and a reward regulating foot direction through the hip yaw coordinate would under price the heading error precisely during swing, which is why the term that was added regulates the toe direction in task space instead.

The two hip yaw axes are furthermore CO DIRECTED, reading minus 0.0998, plus 0.0000, plus 0.9950 on the right and minus 0.0998, minus 0.0000, plus 0.9950 on the left at the nominal pose, a mutual dot product of 1.000000, so a positive command at either hip turns that foot the same way and the heading gain carries the same sign on both legs. This is not evident from the URDF, whose two hip yaw joints declare opposite origin rotations of minus and plus 1.5708 about x at `environments/environments/assets/urdf/solefoot/kscale/kscale.urdf:615` and `:1095`, the opposition being cancelled by parent links that are themselves oppositely oriented. Booster Gym's T1 shares the convention, declaring both hip yaws with `axis xyz="0 0 1"` and `origin rpy="0 0 0"`, which is why the feet yaw reward of that work ports here without a sign correction. Every figure in these two paragraphs is asserted by `scripts/analysis/kscale_feet_heading_selftest.py`, which fails rather than warns.

Addition, 2026-09-04, made in the pass that added sections 7, 8, 9 and 10.5 for the identified gains. Two claims above are qualified rather than refuted by that set.

The correction that no KScale joint reaches the Nyquist bound was established against the calculated gains and holds for them, the highest figure there being the ankle roll at 144.55 rad/s in swing, which is 0.920 of the 157.08 rad/s bound and is the ceiling section 11.8 chose that stiffness to respect. It does not hold for the identified set. The ankle roll at a stiffness of 170 stands at 172.10 rad/s, which is 1.096 of the bound, and the ankle pitch in swing at 149.56 rad/s stands at 0.952 of it. The original correction remains sound on its own terms, since the error it repaired was the omission of the armature from the effective inertia and that omission is still an error, but the conclusion it drew is specific to the gains it was drawn for. A second qualification belongs with it. Every Nyquist figure in this document is a SWING quantity, and section 4.9 establishes that a distal joint's stance natural frequency is between one and two orders of magnitude lower, the ankle roll standing at 4.72 rad/s on the ground against 144.55 in the air. Aliasing is therefore a swing phase concern at this robot's distal joints and not a stance phase one, and any earlier reading that treated the two as one quantity is withdrawn. Section 8 records the current position and the remedy, which is the decimation 2 control period of the flat terrain task rather than a change of gain.

Addition, 2026-09-04, second entry, on the action scale and on the hip yaw. Two records above are superseded rather than corrected. The saturation check of section 11.2 was originally computed at an action scale of 0.25 rad while the configuration at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:148` carried 0.4, an inconsistency that stood from 2026-08-25 until the configuration was set to 0.25 on 2026-09-04. The check and the configuration now agree and section 11.2 is computed at the configured value, so the inconsistency is closed rather than papered over. It is recorded because every run in the series to date, `2026-08-28_04-50-51` and `2026-09-03_09-27-59` among them, trained at 0.4, and their measured behaviour must be read against that scale and not against the one the configuration now carries. Section 11.10 gives the measurement that makes the distinction matter, four joints of six standing at their effort limit for about half of every episode at a scale of 0.4. The hip yaw stiffness of section 11.6 is likewise a derivation performed for the first time against the bandwidth matching criterion, that joint having previously been sized on swing bandwidth alone because it is the one joint bearing no stance load and the two criteria then in use were both silent for it. What made the matching possible is the stance reflected inertia of section 4.9, which is 22.3 times the swing figure at this joint and which no earlier revision of this document computed, the reflection having previously been taken through the vertical lift alone and therefore vanishing at a vertical axis.

---

## Bibliography

1. Ordonez Apraez et al., On discrete symmetries of robotics systems, a group theoretic and data driven analysis, RSS 2023 and IJRR 2024, arXiv:2302.10433.
2. Su et al., Leveraging Symmetry in RL-based Legged Locomotion Control, IROS, 2024, arXiv:2403.17320.
3. Abdolhosseini et al., On Learning Symmetric Locomotion, Motion in Games, 2019, DOI 10.1145/3359566.3360070.
4. Mittal et al., Symmetry Considerations for Learning Task Symmetric Robot Policies, 2024, arXiv:2403.04359.

### Addition, 2026-08-28, the sole geometry gives the splay a measurable stability value

The first training run to carry the feet heading reward, `2026-08-26_08-08-13`, established that the reward is paid rather than obeyed, and the reason lies in the sole geometry this document already records rather than in the reward's formulation. The contact offsets at `environments/environments/tasks/locomotion/cfg/SF/kscale_base_env_cfg.py:28` to `:41` give a sole measuring 0.2099 m fore and aft against 0.0846 m across, an aspect ratio of 2.481. A foot yawed by an angle presents a lateral half extent about the roll axis of half the length times the absolute sine plus half the width times the absolute cosine, which rises from 0.0423 m at zero to 0.0793 m at the 22.5 degrees the run actually reached, a factor of 1.874. The commanded gait fixes stance duration at 0.62 against an offset of 0.5, so single support occupies 0.76 of the cycle and the support polygon is the single stance foot throughout that interval. Splaying the foot therefore very nearly doubles the lateral stability margin for three quarters of every cycle, which is a structural benefit no penalty of the magnitude budgeted can outbid, and it is specific to a long narrow sole, so the SD_BRS1's immunity to this fault is not solely a consequence of its hip yaw being fixed.

The differential form of the fault follows from the same argument. A common rotation of both feet carries the commanded heading with it and forfeits the tracking reward, whereas an equal and opposite splay buys the margin on both feet while leaving the mean heading intact, so the policy occupies the one mode of the decomposition that the tracking terms do not charge for.

Two measurements that a reader may expect to find here are absent and their absence is deliberate. The SIGN of the splay is not established, because the heading penalty is quadratic and `joint_deviation_l1` takes an absolute value, so neither instrument distinguishes toes turned inward from toes turned outward, and the training video is recorded from a fixed oblique overview in which a foot occupies some ten to twenty five pixels and cannot settle it either. The resolved ORDER of the hip yaw joints against the foot bodies is also not established. The mass inventory above lists every paired leg link with the right member before the left, including `foot_6061` before `foot_6061_2`, and `SceneEntityCfg` resolves through `resolve_matching_names` at `IsaacLab/source/isaaclab/isaaclab/utils/string.py:178` in target list order rather than regex order, so the pairing is plausible, but no dumped `params/env.yaml` records the resolved order and no runtime check has yet been run. Any reward that pairs a joint against a foot positionally, as `keep_ankle_pitch_zero_in_air` does, is therefore unverified for pairing correctness until that check is on record.

## Addition, 2026-09-02, the off axis excursion is conserved under reweighting and the stance width parameter is unsatisfiable

Three findings from the runs `2026-08-31_04-57-06` and `2026-09-01_06-53-24` are recorded here rather than only in the plan, because each is a property of this robot rather than of the investigation that found it.

The first is that the off axis hip excursion behaves as a budget the policy allocates rather than as a quantity it minimises. At matched iterations 19195 to 19394 the hip yaw excursion fell from a raw rate of 0.57152 to 0.38435 between the two runs, a reduction of 33 per cent, while the hip roll excursion rose from 0.23580 to 0.40581, an increase of 72 per cent, and the sum moved from 0.80731 to 0.79017, a change of 2.1 per cent. Per joint the hip yaw falls from 16.4 to 11.0 degrees and the hip roll rises from 6.8 to 11.6 degrees. The physical form of the trade is that the feet are drawn toward the midline, which requires adduction at the hip roll, in place of being splayed about the vertical. A single pair of runs cannot establish the conservation as a law and it should not be quoted as one, but the near cancellation of two components that moved by a third and by three quarters is far outside what noise on those components would produce.

The second is that `pen_feet_distance` as configured for this robot demands a stance the robot's own proportions cannot deliver while walking, and that the working tree's revision of it is unsatisfiable outright. The quantity read is the separation between the two foot link frames, each frame being laterally centred upon its sole, so the derived quantity that governs collision is the clearance between the soles' inner edges, being the separation less the sole width of 0.0846 m. The nominal foot separation at the standing pose is 0.252 m and the runs of this sequence set `min_feet_distance` to 0.24 m, which permits 0.012 m of adduction and demands a clearance of 0.155 m between soles only 0.085 m wide. The working tree sets 0.26 m, which exceeds the nominal separation and leaves the penalty with no achievable zero anywhere in the configuration space. Human anthropometry cannot correct the value and the reason should be recorded, since it will recur for any parameter ported by proportion. The step width to step length ratio of 0.101 to 0.183 reported over 294 adults aged seventy and above, applied to this robot's step length of 0.25 to 0.45 m, which follows from a commanded gait frequency of 1.0 cycle per second at an offset of 0.5 and hence two steps per second, yields a step width between 0.025 and 0.082 m. Every value in that interval lies below the sole width, so a human proportioned stance is geometrically impossible on this machine, which has a short stride and disproportionately wide feet. The correct port is from the SD_BRS1, which carries `min_feet_distance` of 0.25 against a sole width of 0.194 m and a nominal separation of 0.259 m, demanding an inner edge clearance of 0.056 m. Porting on the fraction of the nominal separation preserves the wrong invariant, the two robots' nominal separations differing by three per cent while their sole widths differ by a factor of 2.29, and it is the origin of the 0.24 m figure. Porting on the absolute clearance gives 0.141 m and porting on the clearance as a fraction of nominal gives 0.139 m, and the value adopted is 0.14 m. A separate defect is that the term as run measures the planar Euclidean separation and therefore conflates stance width with step length, so a policy taking short steps registers as narrow stanced and the reverse, and a fore and aft stride inflates the measured quantity. The `lateral_only` argument at `environments/environments/tasks/locomotion/mdp/rewards.py:569` repairs it and is correct on this robot only because the root frame correction was carried out, the lateral component being taken at base frame index one.

The third is that the swing gated hip yaw reward is not the cause of the stiff legged gait, notwithstanding that the two appeared together. The run of 2026-09-01 carries `rew_keep_ankle_pitch_zero_in_air` applied to the hip yaw at the same weight of 1.0 while the terms it competes with were scaled down by a factor near a third, so the gate is relatively some 3.3 times stronger in the run without the stiffness than in the run with it, and it holds its own raw rate at 0.44312, the highest of any run carrying it. The stiffness is attributable to the heading penalty at minus 4.0 imposed against gait shaping terms at full magnitude, under which the cheapest way to hold both the hip yaw and the foot heading fixed is to swing the leg as a rigid pendulum, which the frames of that run's video show directly and which the logs corroborate, the foot clearance reward falling 59 per cent while the air time reward rose 93 per cent.

One caution carried forward from the addition of 2026-08-28 is unchanged. The SIGN of the splay remains unestablished, since the heading penalty is quadratic and `joint_deviation_l1` absolute, and although the videos of the two later runs are recorded by a close tracking camera in which a foot occupies roughly 40 to 55 pixels rather than the 10 to 25 of the earlier fixed overview, the base heading against which a foot would be judged is not drawn in the frame. Signed instrumentation remains the first requirement of the next run.

## Addition, 2026-09-03, the splay is pigeon toed, the gait block is the only block that pays for leaving the ground, and two joints have been running against their mechanical stops

Three policies were evaluated against one another on 2026-09-03, `kscale_flat/2026-08-28_04-50-51`, `2026-09-02_06-59-31` which is Baseline 3 of `plans/kscale_integration.md`, and `2026-09-02_10-51-29` which repeats Baseline 3 with the hip yaw travel reduced to plus and minus 0.22 rad. Five findings are recorded and two of them correct earlier entries in this file.

The sign of the splay is established and the caution carried since 2026-08-28 is discharged. Rotating each foot link's negative z axis into the world, projecting it onto the horizontal and differencing against the base heading gives the left foot at minus 7.70 degrees and the right at plus 7.46 in the reference run, both toes converging on the midline. The gait is PIGEON TOED. `foot_6061_2` is the left foot and `foot_6061` the right, established from `kscale.urdf:1070` and `:590` rather than from the name, the `_2` suffix carrying no side convention. Baseline 3 removes 70 per cent of that bias, 7.58 degrees of symmetric inward splay becoming 2.29, while removing only 19 per cent of the per foot magnitude, so what survives is symmetric jitter about zero rather than a posture. Note that `stats.py` reports a foot yaw misalignment near 1.5 rad for every run, which is the ninety degree axis permutation of this robot's foot link frame and not a measurement of splay, so that record must not be quoted for this quantity.

The splay has a geometric cause specific to this robot's proportions. The sole is 0.2100 m long and 0.0846 m wide, a slenderness of 2.48, and the lateral base of support of a rectangular sole at a yaw of t is L sin t plus W cos t, so 7.7 degrees raises it by 31.5 per cent, 84.6 mm becoming 111.3 mm, at no cost in any term reading a joint coordinate. Splay is the cheapest lateral support this robot can buy, which is why three successive instruments relocated the excursion rather than removing it against an unchanged lateral demand from `pen_flat_orientation` at minus 50 and `pen_ang_vel_xy` at minus 5. The same arithmetic bounds it, the two toes converging by 27.4 mm of inner gap at that angle against a fifth percentile gap of 23.7 mm in that run, so at the narrow tail of that stance distribution the toes crossed. The splay is a standing postural bias and not a reflex, its within run correlation against instantaneous lateral velocity, roll rate and lateral projected gravity lying between minus 0.26 and plus 0.17.

Baseline 3 achieved every objective set for it and stopped the robot walking. Air time fell 78 per cent, clearance 62 per cent, step length by half, the fraction of steps with a sole more than 20 mm off the ground fell from 10.6 to 4.7 per cent, and the double support fraction rose from 41.1 to 51.1 against a commanded 24, while falls fell 69 per cent, posture improved by half and velocity tracking improved. Aggregating the evaluation budget by group gives the cause. The gait block was not demoted against the task block, their ratio moving only from 1.043 to 0.969. It was demoted against everything else, the style block rising 2.71 fold against it, the regularisation block 1.94 fold and the flat survival bonus 28.7 fold, and each of those three is cheaper to satisfy when the robot does not leave the ground.

Two joints have been running against their mechanical stops in every run to date, and this CORRECTS the belief recorded in section 7.7 of the plan that raising the ankle roll stiffness from 5 to 20 Nm per radian had cured that joint. It had not. The ankle roll stood within 0.02 rad of a stop for 80.5 per cent of steps in the reference run and 86.4 per cent in the last, on a joint whose entire travel is plus and minus 0.2618 rad. And the hip yaw, at a stiffness of 15 against 150 to 200 at every other joint, carried the robot's highest mean joint speed at 7.3 rad per second and an acceleration near 1300 rad per second squared at an amplitude of only 0.15 rad, with the base yaw rate reaching 4.5 to 6.5 rad per second against commands near 0.6 at a command correlation of zero. The one run that pinned the joint mechanically restored the yaw tracking reward from a raw 0.037 to 0.293 and cut the yaw rate error from 7.52 to 1.78 rad per second. New gains obtained by actuator identification are now in the tree and address both, hip yaw moving to 500 and 18, ankle roll and ankle pitch to 170 and 9, knee to 300, hip roll to 250 and hip pitch damping to 8, and every run tabulated above predates them.

The evaluation of `2026-09-02_10-51-29` is set aside and only its training record may be quoted. The hip yaw restriction was made in the URDF and reverted before the evaluation ran, so the policy trained against stops at 0.22 rad and was replayed against stops at 1.5708, all three dumps carrying identical `joint_position_limits` and the observed hip yaw in its own replay reaching 1.586 rad. Its training mean episode length reaches the full 2000 steps of 2000 while 89.3 per cent of its evaluation episodes terminate early, and that gap measures the mismatch rather than the policy. The two `params/env.yaml` files of the 09-02 series are byte identical, so an asset change of this kind leaves no provenance in the dumped record at all.

## Addition, 2026-09-04, an experimental confirmation of the stance derivation, and a measured warning against sizing a stiffness from a driven joint's own torque

The run `kscale_flat/2026-09-04_10-40-30` is the first experimental test of stiffnesses sized against the swing regime at joints that carry the machine in stance, and it is recorded here because it confirms section 11 by failing in exactly the manner section 11 predicts. The run carried the calculated hip pitch and the calculated action scale of 0.25, but with the hip yaw at 120 Nm per radian and 2.5 Nm s per radian in place of the 100 and 11.0 that section 11.6 derives, the hip roll at 150 and 17.3 in place of 250 and 28.0, the knee at 200 and 10.0 in place of 300 and 10.0, the ankle roll at 20 and 0.5 and the ankle pitch at 50 and 1.7 in place of 120 and 1.8. Those figures are what a swing bandwidth argument produces at the distal joints and what a disturbance quotient produces at the hip yaw, and the run is what happens when a robot is asked to walk on them.

It never stood. Its mean episode length rose to 55.8 at initialisation, fell to 16.3 by iteration 500 and stood at 23.4 at iteration 29000, its `base_contact` termination rate was 1.0000 at every checkpoint across the full thirty thousand iterations, and its mean reward never exceeded minus 2.4. Its policy entropy never collapsed, holding between 5.0 and 6.9 from iteration 10000 onward against a noise standard deviation near 0.53, so the optimiser explored for thirty thousand iterations and found nothing. An optimiser that keeps exploring and never improves is reporting that the environment is not survivable, not that the objective is badly shaped, and the distinction is worth insisting upon because the companion run `2026-09-03_09-49-22` failed at the same task by the opposite mechanism, its entropy falling to minus 1.38 under a halved tracking weight.

The reward record identifies the failure precisely and it is lateral. At iteration 2000 the run paid minus 26.21 per second on `pen_ang_vel_xy` and minus 21.79 on `pen_flat_orientation`, against minus 3.45 and minus 0.45 in the reference run `2026-08-28_04-50-51`, the two lateral stability terms an order of magnitude worse than in any other run of the sequence while its tracking, its regularisation and its contact terms sat within the ordinary range. Three readings follow and each is the empirical form of a statement section 11 makes on analytical grounds alone.

The ankle roll at 20 Nm per radian stands below the gravitational stiffness of 84.90 that section 10.6 computes, so its net restoring torque about the planted sole has the wrong sign and the joint supplies no lateral authority whatever. It stood within 0.02 rad of a mechanical stop on 73.9 and 73.4 per cent of samples in the reference run with 57.8 and 53.5 per cent effort saturation, which is what a servo below its gravitational floor looks like in a log. Section 11.8 sets that joint at 120 for this reason and the run is the demonstration that the floor is real rather than notional. The hip roll at 150 stands below its own travel budget floor of 238 by the same logic, sagging 0.166 rad under a single support hold of 24.95 Nm against an outward abduction travel of 0.209, and its measured stop residency of 78.9 per cent is the consequence.

A gain set whose joints share one function is not separable, and this is why the run tumbled rather than merely tracked poorly. The addition of 2026-08-28 records that splay is the cheapest lateral support this robot can buy, worth 31.5 per cent of lateral base at 7.7 degrees on a sole of slenderness 2.48, and section 10.6 records why it is necessary, the ankle roll being unable to hold single support at any stiffness. A robot whose ankle roll cannot supply lateral authority depends on the splay, and raising the hip yaw eightfold above the swing derived 15 makes the splay eight times more expensive in torque. Removing the substitute while leaving the original unavailable removes the lateral support altogether, which is what the two penalties above record. The identified set of section 7 raises the hip yaw to 500 and simultaneously the ankle roll to 170 and the hip roll to 250, so the lateral duty transfers rather than vanishing, and the mean stance toe-in accordingly halves from 13.07 to 7.81 degrees without a fall. A stiffness may therefore not be varied one joint at a time where several joints serve one function, and the derivation of section 11 is to be applied as a set or not at all.

A logged torque at a position controlled joint is not a disturbance and must not be divided by an allowed deflection to obtain a stiffness. Section 11.1 states the principle and this run is the measurement behind it. Inverting the ideal control law over the reference run's dump, taking the commanded deviation as the applied torque plus the damping term divided by the stiffness, gives a ninety fifth percentile commanded hip yaw deviation of 1.28 rad against an actual deviation of 0.77 rad at a stiffness of 15, so the recorded torque is the spring's own response to a very large policy command that the joint tracked to sixty per cent. The two runs on the identified gains confirm the consequence directly, their ninety fifth percentile commanded hip yaw deviation standing at 0.91 rad at a stiffness of 500 and an action scale of 0.4 and at 0.95 rad at the same stiffness and a scale of 0.25, the same absolute excursion at both scales, so the policy raises its command to hold the posture it wants whatever the stiffness and whatever the scale. At 120 Nm per radian that excursion demands 110 Nm against a limit of 60.

Two measurements from the same sequence bear on section 11.10 and are recorded with the rest. The first is that the reduction of the action scale from 0.4 to 0.25 does not reduce saturation at the joint it would most be wanted for. The ankle roll stop residency is 0.739 and 0.734 at a stiffness of 20, 0.669 and 0.667 at 170 and an action scale of 0.4, and 0.765 and 0.774 at 170 and a scale of 0.25, so the residency RISES under the reduction. The action term carries `clip` unset, so the scale multiplies an unbounded Gaussian and the policy compensates, and the scale governs the resolution of the control rather than the reachable joint set. Any statement that lowering it restricts what the policy may command is false on this configuration.

The second is favourable and is recorded with the rest. The reduction of the action scale to 0.25 removes the delay in gait emergence that the identified gains impose. The reference run on the swing derived hip yaw and the calculated sagittal joints reached a mean episode length of 235 steps by iteration 1000, the identified gains at a scale of 0.4 reached 31 and did not exceed 100 until after iteration 5000, and the identified gains at a scale of 0.25 reached 317, which reproduces the reference schedule. The delay is a property of the identified gains driven at a scale sized for something else and not of the gains themselves.

One question the sequence raises and does not answer should be recorded as the next experiment rather than left implicit. The calculated set of section 12 has never been run in full. Every run to date carries either the swing derived distal gains, which sections 11.7 and 11.8 show fall below the gravitational floor, or the identified set, which section 8 shows is underdamped at every joint in stance. The derivation predicts that the calculated set walks with the identified set's stance authority and without its ringing, the two differing chiefly in damping, and that prediction is falsifiable in one run.

---

## Addition, 2026-09-08, the run that carried the stance stiffnesses, and the criterion it produced

The addition of 2026-09-04 closed by naming one experiment the sequence had not performed, the calculated set of section 12 having never been run in full, and predicted that it would walk with the identified set's stance authority and without its ringing. The run `kscale_flat/2026-09-07_11-08-18` performed it. The prediction was wrong, the reason is recorded in section 11.1.1, and the damping half of the set given in sections 5, 6, 11 and 12 is what the reason produced.

The run carried the six stiffnesses of section 12 exactly, at the action scale of 0.25 that section 11.2 assumes, against the original effort limits of 60 and 17 Nm and the original velocity limits of 20 and 10 rad per second. Flattening its dumped configuration and that of the reference run `2026-08-28_04-50-51` to leaves and comparing them gives a difference of exactly three items in 1670, the gain set, the action scale of 0.25 against 0.4, and `pen_feet_distance.min_feet_distance` of 0.14 against 0.24. The damping it carried was the pre criterion set, 20.0, 28.0, 11.0, 10.0, 1.8 and 1.8.

It never stood. Its mean episode length reached 29.3 steps by iteration 1000 and 29.0 at iteration 9000, against the reference run's 213.9 and 1651.2, and its `base_contact` termination rate was 1.0000 at every checkpoint of 9915. Its entropy fell from 17.06 only as far as 9.91 and its policy noise standard deviation held at 0.65, so the optimiser was still exploring vigorously at the end, while its value function loss fell to 0.84, the critic having learned the environment perfectly and learned that every episode ends in a torso contact after 0.29 s. This is the signature of an environment that is not survivable rather than of an objective that is badly shaped, and it is the same signature `2026-09-04_10-40-30` produced.

The gain set is not what failed and the reward record establishes it. At iteration 2000 the run paid 2.5691 per second on `pen_ang_vel_xy` and 0.0826 on `pen_flat_orientation` in raw units, against 5.4524 and 0.4793 for `2026-09-04_10-40-30`, which carried the swing derived distal gains at the same action scale. Crossing the gravitational floor of 84.90 at both ankles therefore cut the orientation penalty by a factor of 5.8 and the lateral rate penalty by 2.1, which is the improvement sections 11.7 and 11.8 predict and the first direct measurement of it. Against the reference run's 0.6482 and 0.0088 the same figures remain four and nine times worse, so the improvement is real and incomplete.

What failed is the damping, and the mechanism is numerical rather than dynamic. Section 11.1.1 sets out the criterion and it need not be repeated, only its verdict on the three sets this document records, evaluated as the spectral radius of the swing mode at the worst corner of the randomisation envelope.

| Joint | Swing-derived, 2026-08-28 | Calculated damping as run, 2026-09-07 | Identified set, section 7 | Section 12 |
|---|---|---|---|---|
| Hip Pitch | 0.943 | 0.949 | 0.980 | 0.945 |
| Hip Roll | 0.949 | 0.917 | 0.971 | 0.917 |
| Hip Yaw | 0.901 | 1.366 | 3.170 | 0.834 |
| Knee | 0.883 | 0.788 | 0.788 | 0.858 |
| Ankle Pitch | 0.858 | 0.701 | 6.413 | 0.534 |
| Ankle Roll | 0.704 | 1.137 | 8.845 | 0.407 |

Sweeping the envelope gives the fraction of environments carrying at least one divergent joint. The reference run carries none, `2026-09-04_10-40-30` carries none, the run of 2026-09-07 carries 61.2 per cent, and the set of section 12 carries none. The hip yaw diverges in 59.0 per cent of environments on its own and the ankle roll in 5.1 per cent. The identified set of section 7 diverges at three joints in every environment without exception, which section 8 now records. The two divergent joints of the 2026-09-07 run stand either side of the boundary before randomisation, the hip yaw at a nominal spectral radius of 1.037 and the ankle roll at 0.800, so the hip yaw was already past the bound at its configured values and the ankle roll is a joint the randomisation alone carries over it in a twentieth of environments.

Two independent measurements corroborate the diagnosis and neither was used to construct it.

The first is in the run's own telemetry, taken over iterations 0 to 60 where the policy is still near random at every gain set and the comparison therefore isolates the plant. The joint acceleration penalty runs at 1.66 to 3.05 times the reference run's raw value while the action smoothness penalty runs at 0.83 to 1.16 times it and the joint velocity penalty at 1.15 to 1.48. Acceleration rising by a factor of three at unchanged action smoothness and barely changed velocity is high frequency content the plant is injecting rather than content the policy is commanding, which is what a mode oscillating at the integrator's Nyquist frequency looks like in an aggregate. The ratio grows monotonically from 1.66 at iteration 0 to 3.05 at iteration 60, which is the interval over which the policy first begins lifting feet and therefore first places the hip yaw in the swing regime where the instability lives.

The second is the K-Bot. Section 11 of [/ws/context/kscale-opensource.md](../../context/kscale-opensource.md) derives the same dimensionless number for that robot, states the same bound of two, and identifies its wrist as the binding joint at 1.68 of the bound under the proportional and derivative scale of 2.0 that the K-Scale training script's own inline comment describes as unstable in edge cases. Two robots, two simulators and two independent derivations arrive at the same criterion, and on the K-Bot the authors appear to have found the boundary empirically and retreated from it without naming what they had found.

Three further matters belong with this record.

The run's `pen_feet_distance` carried `lateral_only` unset, which is to say False, against a `min_feet_distance` of 0.14. Section 5 of [../plans/kscale_integration.md](../plans/kscale_integration.md) specifies that pairing as 0.14 with `lateral_only` True, the two being one change rather than two, and at False the separation is the planar Euclidean norm at `environments/environments/tasks/locomotion/mdp/rewards.py:604`, which a fore and aft stride satisfies at zero stance width. The term therefore placed no floor under lateral separation at all, its measured mean violation standing at 0.0025 m against the reference run's 0.0071, and the extracted video frames show the legs crossing at the shank with the feet in a single line. This is a configuration defect rather than a gain defect and it is recorded here because it was a co-factor in the failure, the ankle roll at 17 Nm being unable to hold single support while the substitute the robot had been using was simultaneously withdrawn.

The velocity limit finding of section 11.10 was made in this pass and is the largest quantity the pass produced. The two ankles reached 96 per cent of their declared 10 rad per second limit at the ninety fifth percentile of their measured speed, at which point the `DCMotor` torque speed curve had left them 0.69 and 0.75 Nm of a nominal 17. That the ankles were free hinges for the fastest twentieth of every episode had stood unnoticed through six runs, and it was the comparison against the robstride_02 declared maximum of 37.699 rad per second, which the joint name suffix of section 5.1 made available, that exposed it.

The question this addition leaves open is narrower than the one it closes. The set of section 12 is stable in every environment of the randomisation envelope, clears every stiffness floor, clips at no joint and holds single support at the ankle roll, and none of that is a demonstration that it walks. What it establishes is that the four failures the sequence has recorded, the entropy collapse of `2026-09-03_09-49-22`, the lateral collapse of `2026-09-04_10-40-30`, the divergence of `2026-09-07_11-08-18` and the ringing of the identified set, each had a distinct and now identified cause, and that no run to date has been an unconfounded test of the stance derivation. The next run should carry the section 12 gains at the section 5.2 limits against the reward set of `2026-08-28_04-50-51` unchanged, `min_feet_distance` at 0.24 and `lateral_only` at False included, so that the actuator parameterisation is the only thing that differs from the run this document's reference figures are all drawn from. The stance width question of section 5 of [../plans/kscale_integration.md](../plans/kscale_integration.md) is a separate experiment and pairing it with this one would confound both, the failure of 2026-09-07 having been caused by exactly that pairing. This is the first run in the series for which the derivation makes a prediction it cannot escape.

## Correction, 2026-09-09. The velocity limit the derivation raised was never the one that binds

Two runs were executed on 2026-09-08 against the gains this document derived. `2026-09-08_06-50-00` carried an action scale of 0.25 and `2026-09-08_07-39-29` carried 0.4, the two being otherwise identical, and both carried `min_feet_distance` at 0.20 where the reference carries 0.24. A structural diff of the dumped configurations over 1670 leaves establishes that the pair differ from each other in the action scale alone, and from `2026-08-28_04-50-51` in the six stiffness and damping pairs, the six effort and velocity limits, the separation floor and, for the first of them, the action scale. The `lateral_only` key that appears in the new dumps and not in the reference is the default of False at `environments/environments/tasks/locomotion/mdp/rewards.py:791`, so the term is unchanged in behaviour and the pair is a cleaner ablation than the run of 2026-09-07 was.

Both failed, and they failed in the same way. Mean episode length tracked the reference to iteration 200, reaching 45.9 steps against the reference's 33.9, and then fell to 17.4 by iteration 400 and stood between 15 and 22 for the eleven thousand iterations that followed. Per step return floored at approximately minus 1.3 in both, a value the reference passes through at iteration 700 on its way to zero at iteration 1000 and to plus 0.17 at 1200. The evaluation dumps record 4799 and 5746 terminations against 32 timeouts for the reference, a mean survival of 0.20 and 0.17 seconds against the reference's full twenty, and the contact sheets at `artefacts/kscale_flat-2026-09-08_velocity_clamp/` show the reference walking while both new policies lie prone.

The cause is not the gains, and it is not the separation floor. It is a mismatch between two velocity limits that this document treated as one.

### The two limits, and which of them the solver obeys

`IdentifiedActuatorCfg` exposes `velocity_limit`, which sets the knee of the four quadrant torque speed curve at `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:297`, and `velocity_limit_sim`, which is written into the PhysX articulation at `IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py:1773` and is the speed the solver will actually permit. Where `velocity_limit_sim` is left unset it resolves to the value carried by the asset, at `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_base.py:183`. The KScale configuration at `environments/environments/assets/config/kscale_identified_cfg.py` sets `velocity_limit` on all six actuators and `velocity_limit_sim` on none of them, and every one of the twelve joints in `environments/environments/assets/urdf/solefoot/kscale/kscale.urdf` carries the placeholder `velocity="10"`, the same figure on a hip pitch as on an ankle roll. The solver clamp is therefore 10 rad per second on every joint of this robot and has been in every run of the series, which the dumped `joint_velocity_limits_sim` confirms at 9.999999 for all twelve joints of all three runs.

The doubling of section 5.2 raised `velocity_limit` and could not raise the clamp, so the joints gained no speed. What the doubling changed was the torque the actuator delivers while pressed against a clamp that did not move.

| Joint group | tau at 10 rad/s, reference | tau at 10 rad/s, 2026-09-08 | Ratio |
|---|---|---|---|
| Hip yaw, hip roll, hip pitch, knee | 30.00 Nm, half of 60 | 90.00 Nm, three quarters of 120 | 3.0 |
| Ankle pitch, ankle roll | 0.00 Nm, the curve reaching zero exactly at the clamp | 17.00 Nm, half of 34 | unbounded |

The reference ankle was matched to the clamp by coincidence, its `velocity_limit` of 10 being the URDF placeholder repeated, so its torque decayed to exactly zero as the joint approached the speed the solver would refuse to exceed, and the joint decelerated of its own accord rather than pressing against the constraint. The reference hips carried a two to one margin that still halved their torque there. At 40 and 20 rad per second the margins become four to one and two to one, and the actuator arrives at the clamp with three quarters and one half of a stall torque that has itself been doubled.

### What the robot does with that torque

The measurement is unambiguous. Of the samples at which a joint of the 2026-09-08 runs stands at the clamp, between 96.4 and 100.0 per cent carry an actuator torque directed into the clamp, which is to say the actuator is commanding an acceleration the solver silently refuses. The reference shows 19.8 per cent at the ankle roll and 51.5 per cent at the knee, which is to say no systematic direction at all, and it arrives there carrying 0.64 Nm.

| Run | Joint | Per cent of steps at the clamp | Mean torque there | Per cent driving into it | Mean continuous dwell |
|---|---|---|---|---|---|
| 2026-08-28 | left knee | 0.07 | 16.31 Nm | 51.5 | 13.2 ms |
| 2026-08-28 | left ankle roll | 1.92 | 0.64 Nm | 19.8 | 17.2 ms |
| 2026-09-08_06-50-00 | left knee | 51.38 | 89.19 Nm | 100.0 | 56.1 ms |
| 2026-09-08_06-50-00 | left hip yaw | 25.09 | 41.61 Nm | 96.4 | 44.2 ms |
| 2026-09-08_07-39-29 | left ankle roll | 27.76 | 16.48 Nm | 99.8 | 32.3 ms |

A joint held against a velocity clamp is a joint whose acceleration is set by the solver and not by the controller, so the action that commands it has no effect for as long as the dwell lasts. The knee of `2026-09-08_06-50-00` occupies that dead zone for 51.4 per cent of every timestep, in continuous stretches reaching 240 ms, and the mean joint power summed over the twelve joints stands at 2957.5 W against the reference's 302.2 W, rising to 4039.3 W in the run at the larger action scale. Ten to thirteen times the reference power passes through the actuators of a 20.28 kg robot without producing controlled motion.

### The correction this makes to section 11.10 and to section 5.2

Section 11.10 recorded that the ankles reached 96 per cent of their declared limit and were left 0.69 and 0.75 Nm by the torque speed curve, and treated that as a defect to be repaired by raising the limit. The observation was correct and the diagnosis was not. The ankle was not being starved of torque by a mis set parameter, it was being brought to rest by a curve correctly matched to the speed ceiling the solver imposes, and the ceiling rather than the curve is what wanted raising. Section 5.2 proposed the doubling as the repair and the doubling could not reach the binding quantity, which is why the run it recommended failed.

The corroboration is in the neighbouring robot. The TRON1 configuration at `environments/environments/assets/config/solefoot_cfg.py:50`, `61`, `111` and `122` sets `velocity_limit_sim` to 25.0 explicitly on every actuator group, and that configuration produces the working gaits this workspace has recorded. The KScale configuration sets it in no place, and the difference between the two files is the difference between a speed ceiling chosen by a robot's author and one inherited from a URDF field that nobody filled in.

### Why the policy chose to fall, which is not what it was avoiding

The failure presents as a policy that stops actuating and lets the robot drop, and the reward accounting does not support that reading. Summed over the twelve joints the failing policies apply thirteen times the reference's mechanical power, and the terms that price effort are negligible in the return, `pen_joint_torque` contributing minus 0.0053 per step and `pen_joint_accel` minus 0.0186 against a floor of minus 1.3, which is four tenths of one per cent and one and a half per cent respectively. The policy is not declining to apply torque. It is applying more torque than the reference ever did and receiving no control authority in exchange.

What follows is a consequence of the termination structure rather than of the effort terms. `base_contact` at `terminations` carries no penalty and `time_out` is the only other terminal, so an episode whose per step return is negative is worth more the sooner it ends. The reference escapes because it reaches a positive per step return at iteration 1000. A policy that cannot stabilise never reaches one, and the shortening of the episode from 45.9 steps at iteration 200 to 17.4 at iteration 400 is the only remaining direction of improvement available to it. The collapse is the optimiser working correctly against a plant in which the action has lost its effect.

### The solver iteration counts, which are an amplifier and not a cause

`solver_position_iteration_count` and `solver_velocity_iteration_count` both stand at 2 at `environments/environments/assets/config/kscale_identified_cfg.py:113` and `114`, unchanged across all three runs, so they cannot be the differential cause. They are nonetheless implicated, and the measurement shows how. The clamp at 10 rad per second is not enforced exactly, the maximum joint speed observed being 14.37 rad per second in the reference, 14.26 in the run at action scale 0.25 and 23.68 in the run at 0.4. A constraint overshot by 44 per cent in the reference is overshot by 137 per cent when the torque pressed against it triples, because two velocity iterations do not converge the projection and the residual scales with the force driving it. The same setting that was adequate at 30 Nm is not adequate at 90 Nm, not because the setting changed but because the operating point moved into the regime where it binds. Position iterations are less implicated, the ninety ninth percentile penetration beyond a hard joint limit standing at 0.0046 rad at worst, so the limits themselves are being held.

### What the separation floor did, which is nothing

The reduction of `min_feet_distance` from 0.24 to 0.20 is not implicated and the data excludes it. At the plateau `pen_feet_distance` contributes minus 0.0014 per step in the run at action scale 0.25, which is one tenth of one per cent of the floor that must be recovered, and the measured planar separation falls below the threshold on 4.23 per cent of samples against the reference's 47.72 per cent at its own higher threshold. The reference violates its floor on nearly half of all samples and walks. The failing runs barely touch theirs, their mean planar separation standing at 0.307 and 0.832 m because a fallen robot's feet are far apart rather than close together.

The defect that section 5 of [../plans/kscale_integration.md](../plans/kscale_integration.md) records at line 320 nonetheless stands. With `lateral_only` at False the term measures the planar Euclidean norm, and the decomposition of the reference's separation into 0.168 m of lateral and 0.142 m of fore and aft component shows that a stride of ordinary length supplies a third of the threshold on its own. Any future test of the stance width hypothesis must set `lateral_only` to True, or it will not be testing stance width.

### The action scale, which is exonerated as a cause and retained as a concern

The two runs bracket the action scale at 0.25 and 0.4 with every other quantity held, and they fail identically, at 21.1 and 18.5 steps of mean episode length and at minus 1.28 and minus 1.31 of per step return. The action scale is therefore not the discriminator between success and failure, and section 11.2's clip free property at 0.25 buys nothing while the velocity clamp binds.

It remains a concern for a reason the clamp analysis exposes. The ankle roll's entire travel is 0.5236 rad, and an action scale of 0.4 asks a unit action to command 0.4 rad of offset, which is 76 per cent of that travel, while the same scale at the hip pitch commands 12 per cent of a travel of 3.2637 rad. A single scalar scale is not a single quantity when the joints differ in range by a factor of six. `JointPositionActionCfg.scale` accepts a dictionary keyed by joint name expression at `IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py:35`, resolved at `joint_actions.py:86`, so a per joint scale is available without any change to a shared module. The measured range used by the ankle roll exceeds unity in all three runs, at 1.05, 1.02 and 1.15 of its full travel, which is the signature of a command that spends much of its time outside the reachable set.

## Addition, 2026-09-09. The actuator limits are taken from the published K-Bot parameterisation rather than from a derivation

The torque and speed ceilings carried by `environments/assets/config/kscale_identified_cfg.py` had until now been set by the derivations of sections 5 and 11 and by the arbitrary ceilings that preceded them, with no external reference against which either could be checked. The K-Scale Labs stack has since been surveyed at `/ws/context/kscale-opensource.md`, and it publishes the four RobStride actuator classes with their peak torques, their derated soft limits and their velocity ceilings, which supplies exactly that reference. The joint names of this robot carry the class designation in their own suffixes, `_03` at the two upper hip joints, `_04` at the hip pitch and the knee, and `_02` at the two ankle joints, so the mapping requires no judgement.

The nomenclature caution of that document's section 3 must be restated before the numbers are used. The K-Bot is a twenty degree of freedom humanoid of 36.719 kg with a single ankle pitch per leg, and this robot is a twelve degree of freedom sole footed biped of 20.281 kg with an ankle roll besides. Nothing about the gains, the inertias or the damping ratios transfers between them. What does transfer is the actuator hardware, since the two machines are built from the same RobStride line, and it is only the hardware ceilings that have been carried across.

Three quantities were read for each class, the peak torque and the declared maximum velocity from `actuator_type_to_metadata` in `kbot-models/kbot/metadata.json`, the derated soft limit from the per joint records of the same file, and the firmware velocity ceiling from line 19 of the corresponding driver source in `actuator/actuator/robstride/src/actuators/`. The peak torque was confirmed independently against the `actuatorfrcrange` of the default classes at `kbot-models/kbot/robot.mjcf:4-19`, which agrees exactly. The soft limit is seven tenths of the peak at every class without exception, and it is the value the control law actually clips at, at `ksim/ksim/actuators.py:215-217`, so it is the ceiling the hardware enforces rather than the torque the motor could momentarily produce.

| Class | Joints here | Peak, Nm | Soft, Nm | Declared velocity, rad/s | Firmware, rad/s | Set here, rad/s |
|---|---|---|---|---|---|---|
| robstride_03 | hip yaw, hip roll | 60.0 | 42.0 | 18.849 | 20.0 | 18.849 |
| robstride_04 | hip pitch, knee | 120.0 | 84.0 | 17.488 | 15.0 | 15.0 |
| robstride_02 | ankle roll, ankle pitch | 17.0 | 11.9 | 37.699 | 44.0 | 37.699 |

The decomposition onto the two fields of the DCMotor model follows from what each field means. `saturation_effort` is the stall torque from which the four quadrant curve descends at `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:294-306`, and it therefore takes the peak. `effort_limit` is the flat ceiling that curve is clipped against on the same lines, and it therefore takes the soft limit. The flat region consequently extends to three tenths of the velocity ceiling at the two larger classes and to three tenths at the smallest, which is 5.65 rad/s at the hips, 4.50 at the hip pitch and knee, and 11.31 at the ankles, beyond which the available torque falls linearly to nothing.

The velocity fields were set to the lesser of the declared and the firmware figure, and both fields were set to the same number. The first choice is the conservative one, a simulation that permits a rate the hardware refuses being a simulation that trains a policy the hardware cannot execute, and it binds only at the robstride_04, whose declared 17.488 exceeds its firmware ceiling of 15.0. The second choice is the direct consequence of the correction recorded above under the 2026-09-09 heading. Where `velocity_limit_sim` is left unset it resolves to the asset value at `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_base.py:183`, which for this robot is the URDF placeholder of ten radians per second carried identically by all twelve joints at `environments/assets/urdf/solefoot/kscale/kscale.urdf`, and where it is set below the curve knee the joint sits pinned against a solver clamp while the actuator continues to press torque into it. Setting the clamp and the knee to one number removes that failure mode by construction, since the curve delivers exactly zero torque at the speed the solver will not exceed.

Two matters are left standing. The armature values in the configuration, 0.01 at the hips, 0.015 at the knee and 0.005 at the ankles, disagree with the class figures of 0.02, 0.04 and 0.0042 that both the metadata and `robot.mjcf:4-19` declare, and the disagreement is a factor of two at the hips and of nearly three at the knee. It has not been corrected here because the request was confined to the limits and because the armature enters the damping ratio derivations of sections 5 and 11, which would have to be recomputed alongside it. The second is that the ankle ceiling has fallen from the 34.0 Nm this configuration previously carried to 11.9 Nm, which is a reduction of 65 per cent, and section 10 of the survey document records that the K-Bot's own ankle already spends 41 per cent of that soft limit merely standing. The ankle authority available to this robot is therefore now materially smaller than any run before this change enjoyed, and the first run under it should be read with that in mind rather than compared naively against the historical record.

## Correction, 2026-09-10. The published K-Bot ceilings were applied and then withdrawn, and the tree is returned to the 2026-08-28 reference

The addition of 2026-09-09 above records the replacement of this robot's effort and velocity ceilings by the published RobStride class figures. That change was withdrawn the following day and never trained, so the section preceding this one describes a configuration that the tree does not carry and should be read as a derivation retained for a later arm rather than as a record of what ran.

The reason for the withdrawal is a matter of experimental design rather than of the derivation being wrong. The series of 2026-09-07 and 2026-09-08 established that the only run in this robot's history to produce a serviceable gait is `kscale_flat/2026-08-28_04-50-51`, and every departure from it since has failed. A configuration that differs from that run in six gain pairs, six ceilings, a separation floor, an action scale and a pair of solver iteration counts cannot attribute a failure to any one of them, and the K-Bot ceilings would have added a seventh and eighth axis of variation to a comparison that already had too many. The tree is accordingly returned to the reference in full, with exactly one departure carried forward, the one the correction of 2026-09-09 identified as a defect rather than a choice.

The departure is the setting of `velocity_limit_sim` equal to `velocity_limit` at every actuator group, which is 20.0 rad/s at the four proximal groups and 10.0 rad/s at the two ankle groups. Left unset the field resolves to the asset value at `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_base.py:183`, which is the placeholder `velocity="10"` that every one of the twelve joints of `environments/assets/urdf/solefoot/kscale/kscale.urdf` carries identically, a hip pitch and an ankle roll given one number being plainly a placeholder rather than a measurement. The consequence at the ankles is nil, since their curve knee already stood at 10.0 rad/s and the coincidence is what allowed their torque to decay to zero at the clamp in the reference run. The consequence at the four proximal groups is that the solver clamp rises from 10.0 to 20.0 rad/s and meets the knee, so a joint driven to its ceiling now arrives there with no torque behind it rather than pressing 30 Nm into a barrier it cannot pass. This is the one behavioural difference between the working tree and the reference run.

The revert was verified against `kscale_flat/2026-08-28_04-50-51/params/env.yaml` rather than against the git history, the run directory's `git/` folder being empty and the closest committed state, `cf82fc4` of 2026-08-26, disagreeing with the dump at three terms, so the dump is the only authority for what actually ran. Sixty actuator fields were compared field by field and agree exactly, comprising the six effort limits, the six velocity limits, the six saturation efforts, the eighteen gain and armature entries and the friction and activation parameters. One hundred and three reward fields across twenty nine terms were compared the same way and agree exactly, including the action scale at 0.4, the separation floor at 0.24, `keep_balance` at 0.05 and `pen_feet_heading` at minus 4.0, the last two of which the committed state of 2026-08-26 does not carry and which therefore confirm that the reference run was launched from an uncommitted working tree. The solver iteration counts were found at four and returned to the two the dump records.

Two differences remain that are visible in a dump and are not behavioural. The `feet_distance` term now passes `lateral_only` explicitly at `False`, which the reference did not pass at all and which is the parameter's default at `mdp/rewards.py:961`, so the computed separation is identical. And the shared reward module carries `foot_landing_vel_v3` and the terrain lookup it depends upon, added 2026-09-09 for the quadruped, which no KScale term references. The only lines removed from that module in the whole of that work are three of docstring prose inside `foot_landing_vel`, so `foot_clearance_reward_v3`, `foot_landing_vel_v2` and `feet_impact_force`, which the KScale configuration does call, are untouched.

---

## Addition, 2026-09-11. The shank and foot bounding box overlap of section 18 is a true convex hull intersection, and it is why self collision collapsed the episode length

Section 18 recorded two non adjacent bounding box overlaps and declined to say whether either was a true intersection, bounding box overlap being a conservative test. Enabling `enabled_self_collisions` on 2026-09-11 forced the question, the mean episode length of `kscale_flat/2026-09-11_04-56-51` falling to about two steps, and the answer supersedes the closing judgement of that section for this robot, self collision no longer being off here.

The distinction that section 18 could not draw is between the geometry the mesh carries and the geometry PhysX holds. The IsaacLab URDF converter defaults to `collider_type="convex_hull"` at `IsaacLab/source/isaaclab/isaaclab/sim/converters/urdf_converter_cfg.py:119`, and `environments/environments/assets/config/kscale_identified_cfg.py` does not override it, so every `<collision>` element becomes one convex hull and a concave part has its concavity filled. The shank is a fork with a slot for the ankle and the foot is a concave plate, so hulling the fork fills the slot and the filled slot then contains the foot. Measured by `scripts/analysis/kscale_self_collision_audit.py`, the two hulls admit a ball of 8.10 mm radius, the deepest vertex lies 11.05 mm inside the opposing hull, and the overlap encloses 16.82 cm³, the left leg agreeing to within 0.01 mm. The hull inflates the foot to 3.50 times its true volume, the shank to 5.52 times and the ankle bracket `kd_d_402r_6061` to 7.87 times.

The raw meshes do not touch. Their clearance is 4.90 mm at the nominal pose and closes to 0.32 mm only at the ankle pitch lower limit of `-0.872665` rad given at `environments/environments/assets/urdf/solefoot/kscale/kscale.urdf:603`, where the end stop parts `r_end_stop_ankle_6061` of `kscale.urdf:428` hold the mechanism apart. The interference is therefore an artefact of the import and not a property of the robot, which is the finding section 18 left open.

The overlap is structural rather than a consequence of the reset pose. Sweeping the ankle pitch across its full range leaves the hull overlap between 6.26 and 11.85 mm and never clears it, and a sweep of 1500 poses drawn from the joint ranges scaled by the `soft_joint_pos_limit_factor` of 0.9 found both same leg foot and shank pairs overlapping in every pose. No other pair appeared in more than 0.5 per cent of poses, the next being the torso against `kd_d_201r_6061` at 1.59 mm, so the defect is confined to two pairs and the remaining overlaps are genuine self collisions at extreme configurations.

Why this ends an episode requires one further step, the termination being `base_contact` on the torso at a threshold of 1.0 N and not a foot term. At the reset pose the torso is clear, the audit finding no torso pair overlapping there, so the terminating contact is not present at reset and must be produced by motion. Gravity cannot produce it, free fall over the spawn height of 0.79 m taking `sqrt(2 × 0.79 / 9.81)`, which is 0.40 s, or forty control steps at the decimation of 2 and the step of 0.005 s, against the two steps observed. The foot and shank pair is two joints apart across the ankle cross bearing and so is not among the directly jointed pairs PhysX filters, which leaves an eleven millimetre penetration on both legs for the solver to resolve at every reset against the `solver_position_iteration_count` of 2. That this injects the energy is inference and has not been observed directly, the depenetration impulse never having been logged, but it is the only candidate the measurements leave standing.

The remedy carried in the tree excludes the two pairs through `UsdPhysics.FilteredPairsAPI`, applied by `spawn_kscale_from_urdf` in `kscale_identified_cfg.py` and wired as the `func` of the spawn configuration. Two routes were tried first and rejected. Setting `collider_type` to `convex_decomposition` addresses the cause directly but exhausts GPU memory at this environment count. Applying the exclusion through an event term in the `prestartup` mode is refused outright by `IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py:363` whenever `replicate_physics` is True, a guard aimed at randomization that does not distinguish an edit identical across instances. Spawning happens at `IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py:181` and the cloner replicates at line 184, so the relationship written on the source prim propagates to every environment at no per environment cost, which is why the spawn function is the hook rather than the event manager.

Crossed leg pairs are deliberately left live, being both genuine and the subject of the `pen_foot_cross_contact_right` and `pen_foot_cross_contact_left` terms added the same day, which read `force_matrix_w_history` through filtered contact sensors and are identically zero unless self collision is enabled. Two exposures remain recorded rather than closed. The torso against `kd_d_201r_6061` at 1.59 mm in 0.5 per cent of poses has not been tested against its raw meshes, so whether it is a second hull artefact or a genuine interference is unknown, and both bodies feed `pen_undesired_contacts` while the torso also drives a termination at 1.0 N, a threshold low enough that any brush ends the episode. And the forged contact exposure of section 18 is unchanged by any of this, self collision now being enabled having if anything widened it.

Continued the same day, the exclusion list extended from two pairs to four and the torso question closed. The paragraph above leaves the torso against `kd_d_201r_6061` standing as an untested exposure. It is now excluded, together with the torso against `rs03`, on the ground that the torso is wanted clean of self contact whatever the provenance of that overlap, since `base_contact` terminates on it at a threshold of 1.0 N and `pen_undesired_contacts` names it among the bodies it charges, and neither term was calibrated against anything but the ground. A sweep of 20000 poses, reproducible through the `--torso-sweep` mode of `scripts/analysis/kscale_self_collision_audit.py`, establishes that these two hip roll links are the only links able to reach the torso at all, at 0.24 and 0.30 per cent of poses and at most 3.10 and 3.28 mm. Every thigh, shank, ankle cross bearing and foot never reaches it across that sample, and the two hip pitch brackets `kd_d_102r_6061` and `kd_d_102l_6061` are directly jointed to the torso and therefore already filtered by PhysX, so no further pair is available to be excluded. This confirms and quantifies the bounding box finding of section 18, which recorded the torso overlapping each hip roll link two joints removed and could not say whether the overlap was real.

The naming of that pair is worth recording, since a reader searching for a symmetric pair will not find one. The right hip roll link follows the convention and is `kd_d_201r_6061`, while the left is named `rs03` after the RobStride motor part it carries, a name that also appears as a collision part within three other links, so the link and the part must not be confused when reading either the URDF or a body name list.

Whether these exclusions restore the episode length is unverified at the time of writing. The earlier claim that the foot and shank overlap is what collapsed it remains inference, and the run that carried the first two exclusions collapsed again, which is equally consistent with the exclusions having failed to apply, no confirmation having been logged at that point. `spawn_kscale_from_urdf` now prints the count of instances it excludes, so the next run distinguishes the two cases rather than leaving them entangled.

## Addition, 2026-09-16. The nominal pose is replaced by the published K-Scale standing posture, and the standing height must not be read from the sweep of section 16.3

The nominal pose of `environments/environments/assets/config/kscale_identified_cfg.py` was changed on this date from the hip pitch of plus and minus 0.1, knee of 0.4 and ankle pitch of minus 0.3 that section 16.1 records, to the published K-Scale K-Bot standing posture of plus and minus 0.34907, 0.87266 and minus 0.52360, being 20, 50 and minus 30 degrees read from section 7 of [/ws/context/kscale-opensource.md](../../context/kscale-opensource.md). The change is specified by section 9.1 of [../plans/kscale_actuator_limits_fix.md](../plans/kscale_actuator_limits_fix.md) and its justification belongs there. Four quantities are recorded here because they are properties of this robot rather than of that experiment.

The pose closes the sagittal chain exactly as its predecessor did, the signed sum of the three angles vanishing about the world plus y under the anti parallel hip pitch axes, and the check of section 16.2 passes, the ankle roll origin standing at 0.04300 m and both soles lying flat to within 0.1 mm. The lateral foot separation is unchanged at 0.2520 m, so no parameter of `pen_feet_distance` moves with the pose.

The standing height is 0.73540 m against the 0.77161 m of the pose it replaces, computed by forward kinematics against `kscale.urdf` and confirmed by `scripts/analysis/kscale_stance_analysis.py`. This figure MUST NOT be taken from the knee sweep of section 16.3, which holds the hip pitch at plus and minus 0.1 throughout and therefore describes a different family of poses, and which would give 0.712 m at this knee angle. That sweep remains correct for what it states and is not withdrawn, but it answers the question of what a knee angle alone does and not the question of what a three joint posture does, and the two differ here by 23 mm. Three parameters carry the standing height and all three were moved together, the spawn height of the asset configuration from 0.79 to 0.755, and the targets of `pen_base_height` and `pen_feet_regulation` from 0.772 to 0.735.

The deeper fold moves the stance loads in opposite directions at the two sagittal joints and the direction at the hip pitch was not anticipated. The single support load FALLS there by 17 per cent, from 13.651 Nm to 11.288, and rises at the knee by 69 per cent, from 16.468 to 27.877. The knee figure is held with margin by every effort limit this robot has carried, standing at 33 per cent of the 84 Nm the published K-Scale parameterisation gives that joint, but it deflects the knee 0.139 rad at the configured stiffness of 200 against 0.082 before. The ankle pitch and ankle roll loads do not move at all, both being fixed by the sole geometry and the lateral lean rather than by the sagittal fold. The hip yaw is the joint the pose costs, its single support hold torque rising from 2.503 Nm to 8.575, which at the configured stiffness of 15 deflects it 0.572 rad, and section 11.6 having already derived an authority floor of 100 Nm per radian for that joint against that 15. The pose does not create that defect and it multiplies its consequence by 3.4.

A finding of the same pass belongs here rather than in the plan, being a property of this robot at every pose and every limit set it has carried. The gravitational stiffness of both ankles in single support is plus 84.90 Nm per radian at the old pose and plus 80.04 at the new, a destabilising sign in both cases, and section 10.6 establishes that a joint whose stiffness does not exceed its gravitational stiffness cannot hold the assembly above it upright at any deflection whatever. The configured ankle pitch stiffness is 50 and the configured ankle roll 20. Both ankles of this robot are therefore statically unstable in single support and have been in every run of its history, which is not a shortfall of margin but an error of sign, and it is the mechanical reason the machine must reach a mechanical stop rather than settle at a deflection. It is the same conclusion section 11.8 reaches for the ankle roll by the authority argument, arrived at independently and extended to the ankle pitch, which that section does not reach. The measured stop residencies of 53 to 56 per cent at the ankle roll and 35 to 37 at the ankle pitch in `kscale_flat/2026-09-14_10-20-50`, the only run of this robot to walk, are the direct expression of it.

The two analysis scripts carry the pose in their own `NOMINAL_POSE` constants and both were moved in the same pass, at `scripts/analysis/kscale_physical_analysis.py` and `scripts/analysis/kscale_stance_analysis.py`, so every table in this document that either reproduces is now computed at the new pose. A reader reproducing a table dated before 2026-09-16 must restore the old angles first.
