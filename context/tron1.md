# base_robot.urdf — Moment of Inertia Calculations

All inertia values are expressed at the link's centre of mass (CoM) in the link's local frame.
The CoM of every link is placed at the geometric centre of its visual/collision primitive.
Off-diagonal products of inertia are zero for all primitive shapes aligned with the link axes.

---

## Formulae Reference

**Solid box** (dimensions a×b×c along x,y,z):
```
Ixx = m(b² + c²) / 12
Iyy = m(a² + c²) / 12
Izz = m(a² + b²) / 12
```

**Solid cylinder** (radius r, length L):
- Axis along **X**: `Ixx = m·r²/2`,  `Iyy = Izz = m(3r² + L²)/12`
- Axis along **Y**: `Iyy = m·r²/2`,  `Ixx = Izz = m(3r² + L²)/12`
- Axis along **Z**: `Izz = m·r²/2`,  `Ixx = Iyy = m(3r² + L²)/12`

---

## Links — No change required

### base_Link

| Property | Value |
|----------|-------|
| Shape | Box |
| Dimensions (x × y × z) | 0.27 m × 0.26 m × 0.19 m |
| Mass | 5.0 kg |
| CoM in link frame | (0, 0, 0) |

```
Ixx = 5.0 × (0.26² + 0.19²) / 12 = 5.0 × (0.0676 + 0.0361) / 12 = 5.0 × 0.1037 / 12 = 0.043208 kg·m²
Iyy = 5.0 × (0.27² + 0.19²) / 12 = 5.0 × (0.0729 + 0.0361) / 12 = 5.0 × 0.1090 / 12 = 0.045417 kg·m²
Izz = 5.0 × (0.27² + 0.26²) / 12 = 5.0 × (0.0729 + 0.0676) / 12 = 5.0 × 0.1405 / 12 = 0.058542 kg·m²
```

---

### limx_imu

| Property | Value |
|----------|-------|
| Shape | Box |
| Dimensions (x × y × z) | 0.015 m × 0.015 m × 0.004 m |
| Mass | 0.01 kg |
| CoM in link frame | (0, 0, 0) |

```
Ixx = 0.01 × (0.015² + 0.004²) / 12 = 0.01 × (0.000225 + 0.000016) / 12 = 2.0e-7 ≈ 1e-6 kg·m²
Iyy = 0.01 × (0.015² + 0.004²) / 12 = 2.0e-7 ≈ 1e-6 kg·m²
Izz = 0.01 × (0.015² + 0.015²) / 12 = 0.01 × 0.000450 / 12 = 3.75e-7 ≈ 1e-6 kg·m²
```

*(Rounded to 1e-6 — IMU link inertia is negligible.)*

---

### abad_R_actuator_Link / abad_L_actuator_Link

| Property | Value |
|----------|-------|
| Shape | Solid cylinder |
| Cylinder axis | X (visual/collision rpy="0 1.57 0" → Y-rotation 90° maps Z→X) |
| Radius | 0.0483 m |
| Length | 0.0545 m |
| Mass | 0.95 kg |
| CoM in link frame | (+0.02725, 0, 0) — geometric centre of cylinder |

```
r² = 0.0483² = 0.002333 m²
L² = 0.0545² = 0.002970 m²

Ixx = m·r²/2          = 0.95 × 0.002333 / 2                     = 0.001108 kg·m²
Iyy = m(3r² + L²)/12  = 0.95 × (3×0.002333 + 0.002970) / 12
                       = 0.95 × 0.009969 / 12                     = 0.000789 kg·m²
Izz = Iyy                                                         = 0.000789 kg·m²
```

---

### hip_R_actuator_Link / hip_L_actuator_Link

*(Renamed from abad_R_Link / abad_L_Link — geometry, mass, and inertia unchanged.)*

| Property | Value |
|----------|-------|
| Shape | Solid cylinder |
| Cylinder axis | Y (visual rpy="1.57 0 0" → X-rotation 90° maps Z→Y) |
| Radius | 0.0483 m |
| Length | 0.0545 m |
| Mass | 1.469 kg |
| CoM in link frame | (0, +0.035, 0) for R / (0, −0.035, 0) for L |

```
r² = 0.0483² = 0.002333 m²
L² = 0.0545² = 0.002970 m²

Iyy = m·r²/2          = 1.469 × 0.002333 / 2                    = 0.001713 kg·m²
Ixx = m(3r² + L²)/12  = 1.469 × (3×0.002333 + 0.002970) / 12
                       = 1.469 × 0.009969 / 12                    = 0.001220 kg·m²
Izz = Ixx                                                         = 0.001220 kg·m²
```

---

### knee_R_Link / knee_L_Link

| Property | Value |
|----------|-------|
| Shape | Box |
| Dimensions (x × y × z) | 0.025 m × 0.032 m × 0.3 m |
| Mass | 1.22 kg |
| CoM in link frame | (0, 0, −0.15) — geometric centre of box |

```
a = 0.025 m,  b = 0.032 m,  c = 0.3 m
a² = 0.000625,  b² = 0.001024,  c² = 0.090000

Ixx = m(b² + c²) / 12 = 1.22 × (0.001024 + 0.090000) / 12 = 1.22 × 0.091024 / 12 = 0.009254 kg·m²
Iyy = m(a² + c²) / 12 = 1.22 × (0.000625 + 0.090000) / 12 = 1.22 × 0.090625 / 12 = 0.009214 kg·m²
Izz = m(a² + b²) / 12 = 1.22 × (0.000625 + 0.001024) / 12 = 1.22 × 0.001649 / 12 = 0.000168 kg·m²
```

---

### ankle_R_Link / ankle_L_Link

| Property | Value |
|----------|-------|
| Shape | Box |
| Dimensions (x × y × z) | 0.2 m × 0.07 m × 0.03 m |
| Mass | 0.62 kg |
| CoM in link frame | (0.01, 0, −0.035) — geometric centre of box |

```
a = 0.2 m,  b = 0.07 m,  c = 0.03 m
a² = 0.04000,  b² = 0.00490,  c² = 0.00090

Ixx = m(b² + c²) / 12 = 0.62 × (0.00490 + 0.00090) / 12 = 0.62 × 0.00580 / 12 = 0.000300 kg·m²
Iyy = m(a² + c²) / 12 = 0.62 × (0.04000 + 0.00090) / 12 = 0.62 × 0.04090 / 12 = 0.002113 kg·m²
Izz = m(a² + b²) / 12 = 0.62 × (0.04000 + 0.00490) / 12 = 0.62 × 0.04490 / 12 = 0.002320 kg·m²
```

---

## Links — Updated

### hip_R_Link / hip_L_Link

The hip link visual/collision changed from a cylinder (r=0.0483, L=0.0545) to a shorter
cylinder (r=0.0483, L=0.032), and mass reduced from 0.95 to 0.5 kg.
CoM corrected to geometric centre of the visual cylinder.

| Property | Old Value | New Value |
|----------|-----------|-----------|
| Shape | Cylinder r=0.0483 L=0.0545 | Cylinder r=0.0483 **L=0.032** |
| Mass | 0.95 kg | **0.5 kg** |
| CoM (R) | (0, −0.08275, 0) | **(0, −0.016, 0)** |
| CoM (L) | (0, +0.08275, 0) | **(0, +0.016, 0)** |
| Cylinder axis | Y (rpy="1.57 0 0") | Y (unchanged) |

```
r² = 0.0483² = 0.002333 m²
L² = 0.032² = 0.001024 m²

Iyy = m·r²/2          = 0.5 × 0.002333 / 2                      = 0.000583 kg·m²
Ixx = m(3r² + L²)/12  = 0.5 × (3×0.002333 + 0.001024) / 12
                       = 0.5 × 0.008023 / 12                      = 0.000334 kg·m²
Izz = Ixx                                                         = 0.000334 kg·m²
```

*Old values replaced: ixx=0.011378, iyy=0.011562, izz=0.000441 (erroneously copied from thigh box).*

---

### knee_R_actuator_Link / knee_L_actuator_Link

New links added during kinematic chain refactoring. The visual/collision rpy="1.57 0 0"
means the cylinder axis is along Y. CoM corrected to geometric centre at visual origin.

| Property | Copied Value | Corrected Value |
|----------|-------------|-----------------|
| Shape | Cylinder r=0.0483 L=0.0545 | (unchanged) |
| Cylinder axis | Z (from old rpy="0 0 0") | **Y** (rpy="1.57 0 0") |
| Mass | 0.95 kg | (unchanged) |
| CoM (R) | (0, −0.08275, 0) | **(0, −0.02725, 0)** |
| CoM (L) | (0, +0.08275, 0) | **(0, +0.02725, 0)** |

```
r² = 0.0483² = 0.002333 m²
L² = 0.0545² = 0.002970 m²

Iyy = m·r²/2          = 0.95 × 0.002333 / 2                     = 0.001108 kg·m²
Ixx = m(3r² + L²)/12  = 0.95 × (3×0.002333 + 0.002970) / 12
                       = 0.95 × 0.009969 / 12                     = 0.000789 kg·m²
Izz = Ixx                                                         = 0.000789 kg·m²
```

*Old values replaced: ixx=0.000789, iyy=0.000789, izz=0.001108 (axis-Z values, Iyy/Izz swapped).*

---

### hip_R_thigh_Link / hip_L_thigh_Link

The thigh visual/collision box is 0.05×0.032×0.25 (c=0.25). The previous .md incorrectly
used c=0.3 to compute inertia. CoM corrected to geometric centre of the box at visual origin.

| Property | Old Value | Corrected Value |
|----------|-----------|-----------------|
| Shape | Box 0.05×0.032×0.25 | (unchanged) |
| Mass | 1.5 kg | (unchanged) |
| c dimension used for inertia | 0.3 (wrong) | **0.25** |
| CoM (R) | (0, +0.04325, −0.175) | **(0, 0, −0.125)** |
| CoM (L) | (0, +0.04325, −0.175) | **(0, 0, −0.125)** |

```
a = 0.05 m,  b = 0.032 m,  c = 0.25 m
a² = 0.002500,  b² = 0.001024,  c² = 0.062500

Ixx = m(b² + c²) / 12 = 1.5 × (0.001024 + 0.062500) / 12 = 1.5 × 0.063524 / 12 = 0.007941 kg·m²
Iyy = m(a² + c²) / 12 = 1.5 × (0.002500 + 0.062500) / 12 = 1.5 × 0.065000 / 12 = 0.008125 kg·m²
Izz = m(a² + b²) / 12 = 1.5 × (0.002500 + 0.001024) / 12 = 1.5 × 0.003524 / 12 = 0.000441 kg·m²
```

*Old values replaced: ixx=0.011378, iyy=0.011563 (computed with c=0.3 instead of 0.25). Izz unchanged.*

---

### ankle_R_actuator_Link / ankle_L_actuator_Link

The ankle actuator visual/collision rpy="1.57 0 0" means the cylinder axis is along Y, but
the previous inertia was computed for axis X. The ankle_L collision rpy was also corrected
from "0 1.57 0" to "1.57 0 0" in the URDF update. Ixx and Iyy need to be swapped.

| Property | Old Value | Corrected Value |
|----------|-----------|-----------------|
| Shape | Cylinder r=0.03 L=0.0545 | (unchanged) |
| Cylinder axis | X (wrong — based on old collision rpy) | **Y** (rpy="1.57 0 0") |
| Mass | 0.95 kg | (unchanged) |
| CoM | (−0.02725, 0, 0) | (unchanged) |

```
r² = 0.03² = 0.000900 m²
L² = 0.0545² = 0.002970 m²

Iyy = m·r²/2          = 0.95 × 0.000900 / 2                     = 0.000428 kg·m²
Ixx = m(3r² + L²)/12  = 0.95 × (3×0.000900 + 0.002970) / 12
                       = 0.95 × 0.005670 / 12                     = 0.000449 kg·m²
Izz = Ixx                                                         = 0.000449 kg·m²
```

*Old values replaced: ixx=0.000428, iyy=0.000449 (axis-X values, Ixx/Iyy swapped).*

---

## Summary Table

| Link | Shape | m (kg) | CoM (link frame, m) | Ixx (kg·m²) | Iyy (kg·m²) | Izz (kg·m²) | Changed? |
|------|-------|--------|---------------------|-------------|-------------|-------------|----------|
| base_Link | Box 0.27×0.26×0.19 | 5.0 | (0, 0, 0) | 0.043208 | 0.045417 | 0.058542 | No |
| limx_imu | Box 0.015×0.015×0.004 | 0.01 | (0, 0, 0) | 1e-6 | 1e-6 | 1e-6 | No |
| abad_R_actuator_Link | Cyl r=0.0483 L=0.0545 axis X | 0.95 | **(+0.02725, 0, 0)** | 0.001108 | 0.000789 | 0.000789 | **Yes** |
| abad_L_actuator_Link | Cyl r=0.0483 L=0.0545 axis X | 0.95 | **(+0.02725, 0, 0)** | 0.001108 | 0.000789 | 0.000789 | **Yes** |
| hip_R_actuator_Link | Cyl r=0.0483 L=0.0545 axis Y | 1.469 | **(0, +0.035, 0)** | 0.001220 | 0.001713 | 0.001220 | **Yes** |
| hip_L_actuator_Link | Cyl r=0.0483 L=0.0545 axis Y | 1.469 | **(0, −0.035, 0)** | 0.001220 | 0.001713 | 0.001220 | **Yes** |
| **hip_R_Link** | Cyl r=0.0483 L=0.032 axis Y | 0.5 | **(0, −0.016, 0)** | **0.000334** | **0.000583** | **0.000334** | **Yes** |
| **hip_L_Link** | Cyl r=0.0483 L=0.032 axis Y | 0.5 | **(0, +0.016, 0)** | **0.000334** | **0.000583** | **0.000334** | **Yes** |
| **knee_R_actuator_Link** | Cyl r=0.0483 L=0.0545 axis Y | 0.95 | **(0, −0.02725, 0)** | **0.000789** | **0.001108** | **0.000789** | **Yes** |
| **knee_L_actuator_Link** | Cyl r=0.0483 L=0.0545 axis Y | 0.95 | **(0, +0.02725, 0)** | **0.000789** | **0.001108** | **0.000789** | **Yes** |
| **hip_R_thigh_Link** | Box 0.05×0.032×0.25 | 1.5 | **(0, 0, −0.125)** | **0.007941** | **0.008125** | **0.000441** | **Yes** |
| **hip_L_thigh_Link** | Box 0.05×0.032×0.25 | 1.5 | **(0, 0, −0.125)** | **0.007941** | **0.008125** | **0.000441** | **Yes** |
| knee_R_Link | Box 0.025×0.032×0.3 | 1.22 | (0, 0, −0.15) | 0.009254 | 0.009214 | 0.000168 | No |
| knee_L_Link | Box 0.025×0.032×0.3 | 1.22 | (0, 0, −0.15) | 0.009254 | 0.009214 | 0.000168 | No |
| **ankle_R_actuator_Link** | Cyl r=0.03 L=0.0545 axis Y | 0.95 | (−0.02725, 0, 0) | **0.000449** | **0.000428** | **0.000449** | **Yes** |
| **ankle_L_actuator_Link** | Cyl r=0.03 L=0.0545 axis Y | 0.95 | (−0.02725, 0, 0) | **0.000449** | **0.000428** | **0.000449** | **Yes** |
| ankle_R_Link | Box 0.2×0.07×0.03 | 0.62 | (+0.01, 0, −0.035) | 0.000300 | 0.002113 | 0.002320 | No |
| ankle_L_Link | Box 0.2×0.07×0.03 | 0.62 | (+0.01, 0, −0.035) | 0.000300 | 0.002113 | 0.002320 | No |

All off-diagonal inertia terms (ixy, ixz, iyz) are zero — the CoM frames are aligned with the
principal axes of each primitive shape.

---

## Correction of 2026-10-07, the thigh frame of the regenerated document

The thigh entries of the section on updated links describe the thigh box in its own frame, centred at (0, 0, −0.125) and diagonal, which is the inertia the generator computes. This note records how the regenerated Isaac Lab document `environments/environments/assets/mjcf/solefoot/tron1/base_robot.xml`, emitted by `design_generator/biped/builder.py`, places that frame, and it leaves the entries above unedited.

The thigh frame is now rotated by `THIGH_BEND_DEG + 180` degrees about +y, that is 35.0 degrees, through `pose_frames` (`design_generator/biped/builder.py:154-168`). Each thigh body carries `quat="0.953717 0 0.300706 0"`, its inertial and both geoms are laid along local −z at (0, 0, −0.150000) without a quaternion of their own, and the shank body is placed at (0, 0, −0.300000) with `quat="0.793352 0 -0.608763 0"` (`base_robot.xml:63-68`, and the left leg at `:112-117`). Before the change the thigh frame was unrotated, the box was laid along the bend direction by a quaternion on the geom and the inertial, and the shank stood at (−0.172073, 0, −0.245746) with `quat="0.939692 0 -0.342022 0"`. Every joint keeps its coordinate and the shank and ankle bodies keep their world orientation, so the world inertia is unchanged while the thigh's inertia in its own frame is now the diagonal (0.007940, 0.008125, 0.000441) at its stated centre, which is what the observation `robot_inertia` reads for these two bodies.

The ankle sits 13.99 mm ahead of the hip pitch axis in the donor document. Composing the thigh offset (−0.027704, 0, −0.039565) with the shank placement and the ankle housing at (−0.017221, 0, −0.353079) under the 35 degree and −40 degree rotations gives +0.013990 m in x (`base_robot.xml:63`, `:68`, `:74`), which agrees with the 13.987 mm the plan records from forward kinematics of the emitted document, and the chain's earlier assumption of 25.2 mm in the audit is superseded (`plans/co_optimisation_audit_implementation.md:82`).

---

## Addition of 2026-10-19, the nominal pose, the stance load and the kneeling geometry, measured

This section records the physical findings of `../plans/tron1_locomotion_fine_tuning.md` sections 3.1 and 3.2, so that a later session need not rerun them. The figures come from `scripts/analysis/biped_physical_analysis.py` and `scripts/analysis/biped_stance_analysis.py` under `scripts/analysis/biped_configs/tron1_sf.json`, and are preserved in `artefacts/tron1_sf-model-analysis-2026-10-19/`. Both scripts load the MJCF or the URDF through MuJoCo with a floating root, take every robot specific quantity from the configuration, and serve any biped.

The generated `environments/environments/assets/mjcf/solefoot/tron1/base_robot.xml` and the simplified `environments/environments/assets/urdf/solefoot/tron1/base_robot.urdf` carry one nominal pose, the thigh at -35.0 degrees from the downward vertical, the shank at +37.2 degrees and the knee included angle at 107.8 degrees, the URDF writing it into joint origin rotations at `base_robot.urdf:154`, `:257` and `:292`. The original LimX document `base_robot_tron1.urdf` stands at -30.0, +30.0 and 119.9 degrees. The generated robot stands 0.8044 m from root to sole against 0.8397, weighs 21.328 kg against 20.813, and carries a trunk of 6.910 kg against 9.595 and 2.790 kg below each knee against 1.840, so 0.676 of its mass is leg. Its sole lies 0.0500 m below the foot body origin and its feet stand 0.1799 m apart laterally.

Under the identified gains of `environments/environments/assets/config/solefoot_identified_cfg.py:14-67` the single support stance torque is 34.40 Nm at the knee and 5.68 Nm at the ankle, sagging each by 0.57 rad at stiffnesses of 60 and 10, so that 2.3 action units at the scale of 0.25 are needed to hold the nominal pose on one leg. The original geometry needs 32.79 Nm at the knee. With the trunk held level and zero action, the generated robot settles at 0.7546 m, which is the 0.75 m target of `pen_base_height`. The swing inertia including armature is 0.804, 0.842, 0.273 and 0.0085 kg m^2 at the abad, hip, knee and ankle, and the stance inertia 1.285, 1.086, 2.776 and 6.066.

The kneeling geometry is a property of the backward knee and not of the generator. Within the joint limits the highest posture resting on a leg link rather than a sole holds the root 0.5418 m above the ground on the shanks, against 0.5749 m for the original document. The simplified URDF's left knee range, -0.873 to 1.361 rad at `base_robot.urdf:528`, disagrees with its right knee and with the MJCF, both -1.214 to 1.173, and the spawn height of 0.8 m embeds the soles 4.4 mm.

## Addition of 2026-10-19, the contact sensor reports the lower leg loads

Arm 0 of `../plans/tron1_locomotion_fine_tuning.md`, recorded in its section 6.2.1, replayed `model_6000.pt` of run `sf_tron_1a_flat/2026-10-09_06-04-18` with a probe of the `contact_forces` sensor and established that the sensor reports every lower leg body of the generated MJCF, `shank_L_Link` exceeding 10 N in a mean of 20.5 per cent of environments, `shank_R_Link` in 2.7 and the thighs occasionally. The undesired contact count it measured, 0.270 bodies per environment step, matches the 0.275 the training record implies at that iteration. The knee mode of that checkpoint is an asymmetric gait on the right foot, loaded in 80.8 per cent of environments, and the intermittently loaded left shin. The ankle housings `ankle_*_act`, outside the undesired contact set, register in up to 3.6 per cent of environments as a foot pitches. The samples are preserved in `artefacts/sf_tron_1a_flat-2026-10-09_06-04-18/arm0/contact_probe_model_6000.txt`.

## Addition of 2026-10-20, the sagittal mirror of the TRON1

Established for `../plans/symmetry_tron1_quadruped.md` sections 4.1 and 4.3. The generated MJCF is mirrored exactly about the base x-z plane, every body position mapping to its partner's with zero error and the masses and inertias of each pair identical (`artefacts/tron1_quadruped-mjcf-mirror-audit-2026-10-20/mjcf_mirror_audit.txt`). Composing the body chain to the root gives the abduction axes (-1, 0, 0) on the right and (1, 0, 0) on the left (`../environments/environments/assets/mjcf/solefoot/tron1/base_robot.xml:52`, `base_robot.xml:101`) and the hip, knee and ankle axes (0, -1, 0) on both sides, so under the reflection a rotation axis a maps to -M a, every left axis equals its right partner's image, and no joint coordinate changes sign. The left and right limits are identical in every pair, and since the abduction limits (-0.384, 1.396) are asymmetric about zero, that identity corroborates the empty flip set decisively. A mirror written with the rule that roll joints flip is therefore wrong for this asset.

At runtime, recorded on 2026-10-20 in a sandbox run of `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-Symmetry-v0` (`artefacts/sf_tron_1a_symmetry-2026-10-20_06-56-00-sandbox-smoke/smoke_probe.txt`), the articulation resolves its joints depth major and right first, `abad_R_Joint, abad_L_Joint, hip_R_Joint, hip_L_Joint, knee_R_Joint, knee_L_Joint, ankle_R_Joint, ankle_L_Joint`, and its fifteen bodies depth major with `base_Link` first, each body carrying exactly one collision shape. The contact sensor lists the same bodies leg major, the base then the right leg's seven bodies then the left's. The policy group of the rough task is 360 wide and the critic group 4790, the height scan 187 rays in the "xy" ordering, aligned with the base yaw.

## Addition of 2026-10-20, the mirror under the revised plan, and a correction

Established for the revision of `../plans/symmetry_tron1_quadruped.md` of the same date. Correction to the preceding addition. The task it names, `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-Symmetry-v0`, was proposed by the plan's first pass and withdrawn by its revision, which sets the symmetry configuration on `SF_TRON1AFlatPPORunnerCfg` itself, so the identifier was never registered. The runtime orders the addition records were reproduced unchanged on the registered task `Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0` in the revised sandbox (`artefacts/symmetry_revised-2026-10-20-sandbox-smoke/sf_ppo.txt`), and they stand.

The revision adds four facts. The configured default pose sets every named joint to zero (`../environments/environments/tasks/locomotion/robots/limx_solefoot_env_cfg.py:1031-1038`) and is its own mirror, whereas the `default_joint_pos` buffer of any one environment is not, the per joint offset event of `../environments/environments/tasks/locomotion/cfg/SF/limx_base_env_cfg.py:807-808` perturbing it, by 0.048 rad on the right hip of environment 0 in the sandbox, so a check of the flip set against the default pose must read the configuration and not the buffer. The co-optimisation policy group inherits the plain group's ten frame history since the override that set it to zero was removed from `SFCoptEnvCfg` on 2026-10-20, and it is 360 wide, the width of the plain group. The HIM layout is a policy group of 33, an `obsHistory` of (10, 33), a `targetEnc` of 36 and an `estimatorGT` of 3, and the co-optimisation layout adds a `morphologyObs` of 160, being the mass and inertia of all fifteen bodies with four link lengths and six folds, a `privilegedDynamicsObs` of 28 and a `historyObs` of 5575, the critic growing to 4890 (`artefacts/symmetry_revised-2026-10-20-sandbox-smoke/sf_him.txt`, `sf_copt2.txt`). And all three layouts trained three iterations under the mirror with every check passing, the symmetry loss reading 0.0063, 0.0045 and 0.0036 on the plain task, 0.0035 at the HIM task's first logged iteration and 0.0053 at the co-optimisation task's.


## Addition of 2026-10-20, the mirror applied

`../plans/symmetry_tron1_quadruped.md` was applied on 2026-10-20 without divergence in code, its section 13 recording the application. From that date every TRON1 SF task served by `SF_TRON1AFlatPPORunnerCfg` and its subclasses trains with sagittal augmentation and the mirror loss at 0.5 (`../environments/environments/tasks/locomotion/agents/limx_rsl_rl_ppo_cfg.py:129-134`), a run's `params/agent.yaml` carrying `symmetry_cfg` where an earlier run's does not. On the live tree the plain rough task logged a symmetry loss of 0.0063, 0.0045 and 0.0036 over three iterations, equal to the sandbox, the HIM task 0.0035, 0.0015 and 0.0011, and the co-optimisation task 0.0053, 0.0031 and 0.0031 (`artefacts/symmetry_applied-2026-10-20-live-smoke/event_losses.txt`). The baseline then in flight, `sf_tron_1a_flat/2026-10-20_05-24-20`, was launched before the change and is the control.
