# Converting a MuJoCo Model to a URDF, with the Quadruped as the Worked Case

This document records the conversion of `my_design.xml`, the MuJoCo description of the project's quadruped, into the Isaac Lab asset at `environments/environments/assets/urdf/quadruped/quadruped.urdf`. It serves two purposes at once. It is the factual record of that particular conversion, including the four corrections applied to the source model and the reasoning that produced each, and it is the general procedure for any future MuJoCo to URDF conversion undertaken in this project. A reader who has never opened either format should be able to follow it without recourse to another document, so the conventions of both formats are stated before they are used.

## 1. Why a conversion is needed at all, and why it is done by hand

Isaac Lab spawns an articulation from a Universal Scene Description asset, which it will produce on demand from a URDF through its own converter, and it has no path from a MuJoCo XML description [2]. A model authored in MuJoCo must therefore be restated in URDF before this project can train against it. The restatement is not mechanical, because the two formats differ in what they make explicit, and the differences are precisely where errors enter.

A converter exists, `mjcf_urdf_simple_converter`, and its output is a useful cross check. It is not used to produce the asset, for three reasons that bear repeating whenever the question is raised again. It exports every geometry as a triangulated mesh, which forfeits the analytic inertia validation that section 4 depends upon, since a mesh has no declared primitive against which a declared inertia may be checked. It produces an asset whose link lengths cannot be rewritten by editing a joint origin, which the co-optimisation design generator requires, that generator scaling a limb by rewriting the z offset of the joint beneath it. And a mesh asset is large enough that converting it at every design swap becomes a material cost in a co-optimisation run, where the biped's own asset is converted thousands of times over a training job.

The asset is therefore emitted from a generator script, `environments/environments/assets/urdf/quadruped/gen_quadruped_urdf.py`, which holds every dimension once and writes the twenty five links and twenty four joints from four leg definitions. A later change to a segment length or a mass is made in one place rather than in twelve, and the correspondence with the MuJoCo source is auditable by reading forty lines rather than by diffing two thousand.

## 2. The conventions of the two formats, stated before they are used

### 2.1 How each format describes a shape

MuJoCo's `size` attribute is not a dimension. It is a half dimension, and the rule differs by primitive, which is the single most common source of error in a conversion [1].

| Primitive | MuJoCo `size` | The full dimensions it denotes | The URDF form |
|---|---|---|---|
| Box | Three half extents | Twice each entry | `<box size="Lx Ly Lz"/>`, full extents |
| Cylinder | Radius, then half length | Radius, and twice the second entry | `<cylinder radius="r" length="L"/>`, full length |
| Sphere | Radius | Radius | `<sphere radius="r"/>` |
| Capsule | Radius, then half length of the cylindrical section | See section 2.2 | No equivalent exists |

A conversion that carries a MuJoCo box `size` into a URDF `box size` unchanged produces a body of half the intended extent in every direction, and therefore an eighth of the intended volume. Where the inertia is transcribed separately, as it is here, the error is silent, the body rendering and colliding at the wrong size while its dynamics remain correct. The trunk of this robot is declared `size="0.085000 0.098000 0.046000"` in MuJoCo at `environments/environments/assets/urdf/quadruped/my_design.xml:26` and is emitted as `0.170000 0.196000 0.092000` in the URDF, and the doubling is deliberate.

A cylinder carries the same trap in its second entry alone. The actuator housings of this robot are declared `size="0.046000 0.020000"`, a radius of 0.046 metres and a half length of 0.020, and become a URDF cylinder of radius 0.046 and length 0.040.

### 2.2 The capsule, which URDF does not have

MuJoCo offers a capsule, a cylinder closed by a hemispherical cap at each end, and URDF offers no such primitive [1]. A conversion must therefore decide what to render in its place, and the decision has an inertial consequence that is easy to overlook.

A capsule of radius `r` and cylindrical length `L` occupies the volume of the cylinder plus the volume of a sphere of that radius, and it extends a distance `r` beyond the cylindrical section at each end, so its overall length is `L + 2r`. At a fixed density it is therefore heavier than the cylinder it contains, by the volume of one sphere, and its mass is distributed further from the centre, so both its transverse and its axial inertias exceed the cylinder's. Rendering a capsule as a cylinder of the same radius and length keeps the inertia consistent only if the inertia was computed for a cylinder in the first place.

That is exactly the situation this model presented, and section 4.2 records how it was resolved.

### 2.3 How each format places a body

Here the two formats agree, and the agreement is what makes the transcription safe.

A MuJoCo `<body pos="...">` gives the position of that body's frame origin expressed in its parent body's frame. A URDF `<joint><origin xyz="..."/></joint>` gives the position of the child link's frame origin expressed in the parent link's frame. These are the same quantity under two names [1][2], so a MuJoCo body position transcribes directly into the origin of the URDF joint that carries it, with no transformation and no sign change. The generator relies on this throughout, and the cross check of section 6 confirms it held for all twenty four joints.

Two differences remain and both must be handled. MuJoCo nests bodies, so the tree structure is carried by the XML nesting, whereas URDF is flat and carries the structure in the `<parent>` and `<child>` elements of each joint, which means the conversion must name every link explicitly. And MuJoCo attaches a joint to the body it moves, whereas URDF attaches it between two links, so a MuJoCo body carrying a joint becomes a URDF link plus a URDF joint whose child is that link.

### 2.4 How each format expresses an inertia

Both formats express the inertia about the body's own centre of mass. MuJoCo's `<inertial pos="..." mass="..." diaginertia="ixx iyy izz"/>` gives the centre of mass position in the body frame together with the three principal moments, and assumes the principal axes coincide with the body frame axes unless a quaternion says otherwise. URDF's `<inertial>` carries an `<origin>`, a `<mass>` and a full symmetric `<inertia>` tensor with six independent entries. Where the MuJoCo model gives a diagonal, the URDF off diagonal entries are zero, and the transcription is direct.

The formulae for the primitives this model uses are as follows, and they are the basis of the validation in section 4.

For a solid box of mass `m` and full extents `a`, `b`, `c` along the three axes, the principal moments are `Ixx = m(b^2+c^2)/12`, and its two cyclic permutations.

For a solid cylinder of mass `m`, radius `r` and full length `L`, the moment about the cylinder axis is `m r^2 / 2` and the moment about either transverse axis is `m(3r^2 + L^2)/12`.

For a solid sphere of mass `m` and radius `r`, all three moments equal `2 m r^2 / 5`.

For a solid capsule, of mass `m` distributed over a cylinder of radius `r` and length `L` closed by two hemispheres of the same radius, the axial moment is the cylinder's plus the two caps' contribution and the transverse moment adds a parallel axis term for each cap displaced by `L/2` plus a shape term, so both exceed the cylinder's figures at equal density.

Where a rigid body is displaced from an axis by a distance `d`, its moment about that axis is its own moment about a parallel axis through its centre of mass plus `m d^2`. This is the parallel axis theorem, and it is the whole of what section 5 of the companion document needs to compute an effective inertia at a joint.

### 2.5 The rotation of a primitive, and why the URDF carries roll pitch yaw

MuJoCo orients a geometry with a quaternion in its `quat` attribute, or with a `fromto` pair naming the two endpoints of the axis. URDF orients a shape with a roll pitch yaw triple in the `<origin rpy="...">` of its `<visual>` or `<collision>` element. A cylinder is authored along z in both formats, so a cylinder whose axis lies along x takes `rpy="0 1.5708 0"`, a rotation of ninety degrees about y, and one whose axis lies along y takes `rpy="1.5708 0 0"`. The generator emits exactly these two values, at `gen_quadruped_urdf.py:88` for the abduction housing and `:98` for the two pitch housings.

### 2.6 The frame of a foot, and the error it invites

This deserves its own statement, because it recurs on every robot with a rounded foot and it is invisible in the model file.

A foot rendered as a box or a mesh usually has its frame origin placed at the sole, so the height at which the robot stands equals the vertical drop computed from the joint angles, and the two numbers coincide. A foot rendered as a sphere or a capsule has its frame origin at the centre of that primitive, which lies one radius above the lowest point. The standing height is then the kinematic drop plus the foot radius, and a keyframe or an initial state that omits the radius buries the foot in the ground by exactly that amount.

The rule to carry forward is that the standing height of a robot with a spherical or capsular foot exceeds its kinematic drop by the foot radius, on every such robot, without exception. Section 4.4 records how this model violated the rule and how it was corrected.

## 3. The joint semantic audit, the step most costly to skip

Before any geometry is converted, the physical meaning of every joint must be established from its axis rather than from its name. A model's joint names are the author's convention and need not match the target repository's, and where they differ but overlap, a conversion that carries the names across will silently attach the wrong meaning to the wrong degree of freedom.

This model presented exactly that hazard, and it is the single most consequential fact of the conversion. Its naming and the repository's naming use the same three words for different joints.

| MuJoCo joint | Its axis | The physical degree of freedom | The URDF joint it becomes |
|---|---|---|---|
| `FR_hip_joint` | `1 0 0`, roll | Abduction and adduction | `abad_FR_Joint` |
| `FR_thigh_joint` | `0 1 0`, pitch | Hip flexion and extension | `hip_FR_Joint` |
| `FR_calf_joint` | `0 1 0`, pitch | Knee flexion and extension | `knee_FR_Joint` |

The MuJoCo name `hip` denotes the roll axis, whereas this repository's TRON1 convention reserves `hip` for the pitch axis and calls the roll axis `abad`, following the abduction and adduction terminology. A conversion carrying the names across unchanged would place the abduction joint under the name `hip`, and every reward, event and observation that selects joints by regular expression would then act on the wrong axis. The abduction deviation penalty of the reward configuration would penalise hip pitch, and the actuator gains derived for a roll joint would be applied to a pitch joint of different inertia.

The general procedure is therefore to tabulate every joint against its declared axis, to name each by the physical motion the axis produces rather than by the source model's label, and to record the mapping in the conversion document before writing a line of the target file.

## 4. The audit of this model, and the four corrections it forced

Every inertial entry was recomputed from its declared primitive and its declared mass, using the formulae of section 2.4 and the half extent rules of section 2.1. The result is the table below.

| Link | Primitive | Declared mass | Declared diagonal | Recomputed | Verdict |
|---|---|---|---|---|---|
| `base_Link` | Box 0.170 by 0.196 by 0.092 | 0.674397 | 0.002635, 0.002100, 0.003783 | 0.002635, 0.002100, 0.003783 | Exact |
| `abad_*_actuator_Link` | Cylinder r 0.046, L 0.040, axis X | 1.028375 | 0.001088, 0.000681, 0.000681 | 0.001088, 0.000681, 0.000681 | Exact |
| `hip_*_actuator_Link` | Cylinder r 0.046, L 0.040, axis Y | 1.028375 | 0.000681, 0.001088, 0.000681 | 0.000681, 0.001088, 0.000681 | Exact |
| `knee_*_actuator_Link` | Cylinder r 0.046, L 0.040, axis Y | 1.434746 | 0.000950, 0.001518, 0.000950 | 0.000950, 0.001518, 0.000950 | Exact |
| `hip_*_thigh_Link` | Cylinder r 0.020, L 0.213, axis Z | 0.187365 | 0.000727, 0.000727, 0.000037 | 0.000727, 0.000727, 0.000037 | Exact |
| `knee_*_Link` | Cylinder r 0.014, L 0.213, axis Z | 0.187365 | 0.000727, 0.000727, 0.000037 | 0.000352, 0.000352, 0.000009 | Wrong, at a mass of 0.091809 |
| `foot_*_Link` | Sphere r 0.022 | 0.020000 | 0.000004 on all three | 0.000004 | Exact to the stated precision |

### 4.1 Establishing the model's own convention before judging its errors

The audit's method is worth stating, because it is what allows a discrepancy to be corrected rather than merely noted. An inertia that disagrees with its primitive may mean the inertia is wrong or that the primitive is wrong, and nothing in the file itself says which. The way to decide is to recover the author's implied convention from the entries that agree, and then to hold the disagreeing entry to it.

Dividing the thigh's declared mass of 0.187365 kilogrammes by the volume of a solid cylinder of radius 0.020 metres and length 0.213 metres gives 700.0015 kilogrammes per cubic metre, which is 700 to five significant figures. Its declared axial moment of 0.000037 is exactly `m r^2 / 2` for that radius. Five matching significant figures is not coincidence, so the leg segments were dimensioned as solid cylinders at a uniform density of 700, and that density is the convention against which the calf must be judged.

### 4.2 The four corrections

The first correction is the calf's mass and inertia. It carries the thigh's figures verbatim although its declared radius is 0.014 metres and not 0.020. At the model's own density of 700 the calf's mass is 0.091809 kilogrammes and its diagonal is 0.000352, 0.000352, 0.000009. The declared mass is therefore too large by a factor of 2.04 and the declared axial moment by a factor of 4.16. The error is not dynamically inert, an axial error alone would be nearly so, but the mass and the transverse moments both enter the knee's effective inertia and the whole robot's weight. Correcting it reduces the total mass from 16.219301 to 15.837077 kilogrammes.

The second correction is the geometry type. The thigh and the calf are declared `capsule` but are dimensioned and inertiated as cylinders, per section 4.1. Since URDF has no capsule in any case, the URDF must declare a cylinder whatever the source says, and the correction is to change the two geometry types in the MuJoCo file so that the declared shape, the declared mass and the declared inertia agree in both files. The alternative resolution, recomputing the inertias for capsules and keeping the smoother collision primitive, is recorded here so that it remains available without repeating the derivation, a capsule of the thigh's radius and length at the same density having a mass of 0.210822 and a transverse moment of 0.001034, and one of the calf's a mass of 0.099855 and a transverse moment of 0.000452.

The third correction is the abduction axis, and section 5 sets out its reasoning in full.

The fourth correction is the standing height, and section 4.3 derives it.

All four were applied to `my_design.xml` before the URDF was generated, so that the two files describe one robot, and the corrected MuJoCo file is kept beside the URDF at `environments/environments/assets/urdf/quadruped/my_design.xml` rather than at the workspace root, so that a later divergence between the two models is visible within one directory.

One property was recorded as a defect rather than corrected, because it is a modelling choice and not an internal inconsistency. The armature of 0.01 kilogramme metres squared is declared identically on all twelve joints. Armature is the rotor inertia reflected through the square of the gear ratio, and a GO-M8010-6 with a reduction of 6.33 and a rotor inertia of order 4 times 10 to the minus 5 reflects roughly 1.6 times 10 to the minus 3 [3], so the declared figure is high by something near a factor of six. It is carried across unaltered so that the two models remain interchangeable, and its consequence, that it supplies between 41 and 81 percent of every effective inertia and therefore inflates the derived gains, is recorded in the companion document.

### 4.3 The standing pose, and the buried foot

The MuJoCo keyframe placed the trunk at 0.270 metres with the hip pitch at 0.884337 radians and the knee at -1.768673 radians. Since the knee angle is the negation of twice the hip angle to within a microradian, the calf's absolute pitch is the negation of the thigh's, the fore and aft components of the two segments cancel, and the foot lands exactly beneath the hip pitch axis. The vertical drop is `0.213 cos(0.884337) + 0.213 cos(-0.884336)`, which evaluates to 0.270000 metres.

Every intermediate offset in the chain from the trunk to the hip pitch axis is purely lateral or longitudinal, so the pitch axis lies at the trunk's own height and a trunk at 0.270 metres places the foot frame's origin at exactly zero. By the rule of section 2.6 that origin is the centre of the contact sphere, so the keyframe buried 22 millimetres of the foot in the ground. The correction raises the keyframe's third position entry to 0.292000, and 0.292 is thereafter the standing height that the articulation's initial state, the base height reward target and the terrain scaling all use.

The same figure of 0.270 appeared a second time, as the default position of the trunk body itself, which is the height at which the model spawns when no keyframe is loaded. The plan's prescription named only the keyframe. Both were corrected, on the ground that leaving them split would make the model contradict itself depending on how it is loaded, and the divergence is recorded here so that a reader comparing the plan against the file is not left to wonder.

## 5. Mirroring the abduction axes, a convention rather than a kinematic change

The source model gives all four legs the abduction axis `1 0 0`. The TRON1 URDF gives `abad_R_Joint` the axis `-1 0 0` and `abad_L_Joint` the axis `1 0 0`, at `environments/environments/assets/urdf/solefoot/tron1/base_robot.urdf:121` and `:393`, over an identical limit range. The quadruped adopts TRON1's convention, and the reason is a sign semantic rather than a kinematic one.

Consider a leg pointing downward, so that its foot sits at a displacement `(0, 0, -L)` from the abduction axis. Rotating by a positive angle `theta` about the axis `1 0 0` carries the foot to `(0, +L sin(theta), -L cos(theta))`, so the foot swings toward positive `y`. On a left leg, which sits at positive `y`, that is outward, and on a right leg, which sits at negative `y`, that is inward. An unmirrored model therefore makes a positive command mean abduction on one side and adduction on the other. Mirroring the axis on the right legs makes positive mean outward on both.

Three things follow from adopting it. A laterally symmetric stance takes equal abduction values on all four legs rather than opposed ones, which is what the initial joint position of the articulation configuration assumes. A left to right symmetry augmentation becomes a pure permutation of the twelve joint states rather than a permutation composed with a sign flip on four of them, which is what any future augmentation for this robot will want. And the two robots' abduction states carry the same meaning, so a reward or an event written against one transfers to the other without a hidden sign.

The mirroring costs nothing in reachable configuration, the abduction range of -1.0 to 1.0 being symmetric about zero, so the limits transcribe unchanged and only the axis vector differs. The URDF gives `abad_FR_Joint` and `abad_RR_Joint` the axis `-1 0 0` and `abad_FL_Joint` and `abad_RL_Joint` the axis `1 0 0`, and the MuJoCo file receives the same change on its `FR_hip_joint` and `RR_hip_joint`. The pitch axes are already consistent, `0 1 0` on all four legs meaning forward flexion everywhere, and are carried across unaltered.

The effect was verified in simulation rather than asserted. Commanding the front right abduction joint to its upper limit of 1.0 radian carries its foot from `y = -0.1320` to `y = -0.3224`, which is outward, whereas under the unmirrored axis the same command carried it to `y = +0.1320`, the front left foot's nominal position.

## 6. The resulting topology, and its verification against the source

The naming convention is `[abad/hip/knee]_[FR/FL/RR/RL]_[thigh/actuator]_[Link/joint]`, following TRON1. Two departures from the literal form are deliberate. The twelve actuated joints take the suffix `_Joint` with a capital letter, because that is what TRON1 uses, while the fixed joints take a lowercase `_joint`, exactly as TRON1's own `abad_R_fixed_joint` does. The foot takes the name `foot_FR_Link` rather than a form built from a joint name, because the convention has no slot for a terminal segment, and because a feet selecting expression of `foot_.*_Link` is then disjoint from the `abad_.*`, `hip_.*` and `knee_.*` expressions that the undesired contact penalty needs.

The chain, given for the front right leg and repeated for the other three, is as follows.

| URDF link | MuJoCo body | Parent joint | Type | Origin in parent |
|---|---|---|---|---|
| `base_Link` | `trunk` | root | free | root |
| `abad_FR_actuator_Link` | `FR_abd_act` | `abad_FR_fixed_joint` | fixed | `0.105 -0.052 0` |
| `hip_FR_actuator_Link` | `FR_hip_act` | `abad_FR_Joint` | revolute, axis `-1 0 0` | `0.071 0 0` |
| `knee_FR_actuator_Link` | `FR_knee_act` | `hip_FR_Joint` | revolute, axis `0 1 0` | `0 -0.050 0` |
| `hip_FR_thigh_Link` | `FR_thigh` | `hip_FR_thigh_joint` | fixed | `0 -0.030 0` |
| `knee_FR_Link` | `FR_calf` | `knee_FR_Joint` | revolute, axis `0 1 0` | `0 0 -0.213` |
| `foot_FR_Link` | `FR_foot` | `foot_FR_joint` | fixed | `0 0 -0.213` |

The corrected total mass is 15.837077 kilogrammes, being 0.674397 for the trunk and 3.790670 for each leg, and the weight is 155.36 newtons. The four legs carry 95.7 percent of the machine and the two actuator housings on each leg carry 91 percent of the leg, a distribution that matters to the gain derivation, where it makes the abduction joint the most inertially loaded of the three rather than the least.

The verification was performed by compiling the corrected MuJoCo model with the MuJoCo library itself [1] and comparing it body by body against the generated URDF, which is a stronger check than comparing against a converter's draft, since it compares against the source model's own compiled representation rather than against a second transcription of it. For all twenty five bodies the mass, the inertia diagonal and the frame origin agreed to within one part in a billion, and for all twelve revolute joints the axis and the limit range agreed exactly. The URDF was independently confirmed to declare twenty five links and twenty four joints of which twelve are revolute, to have a single root at `base_Link`, and to be acyclic with every link reaching that root.

The standing pose was verified in the same pass. Loading the corrected keyframe and evaluating the forward kinematics places all four foot centres at `z = +0.022000`, so each sole rests at exactly `z = 0`, which is the check that the correction of section 4.3 is both necessary and sufficient.

## 7. The clearance audit and the self collision decision

At the nominal pose every non adjacent pair of bodies clears. The abduction housing spans `x` from 0.085 to 0.125 and the hip housing from 0.130 to 0.222, a gap of 5 millimetres. The hip housing spans `y` from -0.072 to -0.032 and the knee housing from -0.122 to -0.082, a gap of 10 millimetres. The hip housing and the thigh clear by 40 millimetres in `y`. The trunk, whose half extent in `x` is 0.085, abuts the abduction housing exactly without penetrating it. The single overlapping pair is the thigh against the knee housing, which interpenetrate in `y` over the interval -0.122 to -0.112, and that pair is joined by the fixed `hip_FR_thigh_joint`, so PhysX excludes it from self collision automatically, as it excludes every directly jointed pair [2].

Self collision must nevertheless remain enabled, because the abduction range permits the legs to reach one another. Under the mirrored axis it is the lower limit rather than the upper that carries a foot inward, driving the front right foot from `y = -0.132` toward and past the midline, so an unconstrained policy can drive one leg into another during exploration. The prescription is `self_collision=True` on the spawn configuration and `enabled_self_collisions=True` on the articulation root properties, matching both the biped configuration and MuJoCo's own default of colliding all pairs except parents and children.

The measured stance is 0.264 metres between the two front feet, against a leg of 0.448 metres from the abduction axis to the sole, which is a narrow machine and is the reason the reward configuration carries an abduction deviation penalty.

## 8. A checklist for the next conversion

1. Tabulate every joint against its declared axis and name it by the motion the axis produces, never by the source model's label. Record the mapping before writing anything.
2. Recompute every declared inertia from its declared primitive and mass. Where an entry disagrees, recover the model's implied density from the entries that agree, and hold the disagreeing entry to it rather than guessing.
3. Double every box half extent and every cylinder half length when emitting the URDF geometry.
4. Decide what replaces each capsule, and confirm the inertia that accompanies it was computed for the shape actually emitted.
5. If any foot is a sphere or a capsule, add its radius to the kinematic drop to obtain the standing height, and correct every place that height appears, the keyframe and the model's own default spawn among them.
6. Mirror any laterally paired axis so that a positive command means the same physical direction on both sides, and confirm the limit range is symmetric before doing so, since it must otherwise be mirrored too.
7. Transcribe each MuJoCo body position directly into the origin of the URDF joint that carries it, without transformation.
8. Compile the corrected source model with the MuJoCo library and compare it body by body against the emitted URDF, on mass, inertia, origin, axis and limit. This is the check that catches everything the eye does not.
9. Confirm the emitted tree is acyclic with a single root, and that its summed mass equals the source model's.
10. Load the standing pose and confirm every sole rests at zero.

## 9. Bibliography

1. Todorov, E., Erez, T., Tassa, Y. MuJoCo, A physics engine for model-based control. IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012. DOI 10.1109/IROS.2012.6386109. The `size`, `pos`, `fromto` and `inertial` semantics used throughout section 2 were confirmed against the MuJoCo XML reference documentation and against the compiled model produced by the MuJoCo Python package version 3.11.0.
2. Mittal, M., Yu, C., Yu, Q., Liu, J., Rudin, N., Hoeller, D., and others. Orbit, A Unified Simulation Framework for Interactive Robot Learning Environments. IEEE Robotics and Automation Letters 8(6), 3740 to 3747, 2023. arXiv:2301.04195. The author list was not verified beyond the six named and the short form is used. The URDF import path and the articulation root properties referred to here were read from the local checkout at `/ws/IsaacLab`.
3. Unitree Robotics. GO Motor product specification, `unitree.com`, and Go1 Datasheet EN v3.0, retrieved 2026-08-19. Source of the GO-M8010-6 reduction and rotor inertia figures used in the armature discussion of section 4.2.
