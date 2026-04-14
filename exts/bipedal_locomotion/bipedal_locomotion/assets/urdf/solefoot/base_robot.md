# base_robot.urdf — Moment of Inertia Calculations

All inertia values are expressed at the link's centre of mass (CoM) in the link's local frame.
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

## Links — Already Correct (no change required)

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
| CoM in link frame | (−0.02725, 0, 0) — geometric centre of cylinder |

```
r² = 0.0483² = 0.002333 m²
L² = 0.0545² = 0.002970 m²

Ixx = m·r²/2          = 0.95 × 0.002333 / 2                     = 0.001108 kg·m²
Iyy = m(3r² + L²)/12  = 0.95 × (3×0.002333 + 0.002970) / 12
                       = 0.95 × 0.009969 / 12                     = 0.000789 kg·m²
Izz = Iyy                                                         = 0.000789 kg·m²
```

---

### hip_R_Link / hip_L_Link

| Property | Value |
|----------|-------|
| Shape | Solid cylinder |
| Cylinder axis | Y (visual rpy="1.57 0 0" → X-rotation 90° maps Z→Y) |
| Radius | 0.0483 m |
| Length | 0.0545 m |
| Mass | 0.95 kg |
| CoM in link frame | (0, −0.08275, 0) for R / (0, +0.08275, 0) for L |

```
r² = 0.002333 m²,  L² = 0.002970 m²

Iyy = m·r²/2          = 0.95 × 0.002333 / 2                     = 0.001108 kg·m²
Ixx = m(3r² + L²)/12  = 0.95 × 0.009969 / 12                    = 0.000789 kg·m²
Izz = Ixx                                                         = 0.000789 kg·m²
```

---

### hip_R_thigh_Link / hip_L_thigh_Link

The thigh is modelled as a single box matching the collision geometry, which spans the full
thigh segment from the hip actuator flange to the knee joint.

| Property | Value |
|----------|-------|
| Shape | Box (collision geometry used for inertia) |
| Dimensions (x × y × z) | 0.05 m × 0.032 m × 0.3 m |
| Mass | 1.5 kg |
| CoM in link frame | (0, +0.04325, −0.15) for R / (0, −0.04325, −0.15) for L |

```
a = 0.05 m,  b = 0.032 m,  c = 0.3 m
a² = 0.002500,  b² = 0.001024,  c² = 0.09000

Ixx = m(b² + c²) / 12 = 1.5 × (0.001024 + 0.09000) / 12 = 1.5 × 0.091024 / 12 = 0.011378 kg·m²
Iyy = m(a² + c²) / 12 = 1.5 × (0.002500 + 0.09000) / 12 = 1.5 × 0.092500 / 12 = 0.011563 kg·m²
Izz = m(a² + b²) / 12 = 1.5 × (0.002500 + 0.001024) / 12 = 1.5 × 0.003524 / 12 = 0.000441 kg·m²
```

---

## Links — Updated (TRON1 values replaced)

### abad_R_Link / abad_L_Link

The abad link in base_robot.urdf is a single cylinder connecting the abad actuator output to
the hip joint, unlike the complex bracket geometry in TRON1.

| Property | Value |
|----------|-------|
| Shape | Solid cylinder |
| Cylinder axis | Y (visual rpy="1.57 0 0" → X-rotation 90° maps Z→Y) |
| Radius | 0.0483 m |
| Length | 0.0545 m |
| Mass | 1.469 kg |
| CoM in link frame | (−0.0683, +0.035, 0) for R / (−0.0683, −0.035, 0) for L |

```
r² = 0.0483² = 0.002333 m²
L² = 0.0545² = 0.002970 m²

Iyy = m·r²/2          = 1.469 × 0.002333 / 2                    = 0.001713 kg·m²
Ixx = m(3r² + L²)/12  = 1.469 × (3×0.002333 + 0.002970) / 12
                       = 1.469 × 0.009969 / 12                    = 0.001220 kg·m²
Izz = Ixx                                                         = 0.001220 kg·m²
```

*TRON1 values replaced: ixx was 1555e-6, iyy was 2359e-6, izz was 2081e-6 (complex bracket geometry).*

---

### knee_R_Link / knee_L_Link

The knee/shank link in base_robot.urdf is a thin rectangular bar, unlike the combined
thigh-and-shank assembly in TRON1.

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

*TRON1 values replaced: ixx was 10938e-6, iyy was 14358e-6, izz was 4088e-6 (longer, heavier thigh assembly).*

---

### ankle_R_actuator_Link / ankle_L_actuator_Link

The ankle actuator uses a smaller-diameter cylinder (r=0.03) than the abad actuator (r=0.0483).
The TRON1-copied values incorrectly used the abad actuator radius.

| Property | Value |
|----------|-------|
| Shape | Solid cylinder |
| Cylinder axis | X (collision rpy="0 1.57 0" → Y-rotation 90° maps Z→X) |
| Radius | 0.03 m |
| Length | 0.0545 m |
| Mass | 0.95 kg |
| CoM in link frame | (−0.02725, 0, 0) — geometric centre of cylinder |

```
r² = 0.03² = 0.0009 m²
L² = 0.0545² = 0.002970 m²

Ixx = m·r²/2          = 0.95 × 0.0009 / 2                       = 0.000428 kg·m²
Iyy = m(3r² + L²)/12  = 0.95 × (3×0.0009 + 0.002970) / 12
                       = 0.95 × 0.005670 / 12                     = 0.000449 kg·m²
Izz = Iyy                                                         = 0.000449 kg·m²
```

*TRON1 values replaced: ixx was 1108e-6, iyy/izz were 789e-6 (used abad radius 0.0483 instead of 0.03).*

---

### ankle_R_Link / ankle_L_Link

The ankle link is modelled as a flat rectangular foot plate.

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

*TRON1 values replaced: ixx was 525e-6, iyy was 1812e-6, izz was 1974e-6 (different foot geometry and non-zero cross terms).*

---

## Summary Table

| Link | Shape | m (kg) | CoM (link frame, m) | Ixx (kg·m²) | Iyy (kg·m²) | Izz (kg·m²) | Changed? |
|------|-------|--------|---------------------|-------------|-------------|-------------|----------|
| base_Link | Box 0.27×0.26×0.19 | 5.0 | (0, 0, 0) | 0.043208 | 0.045417 | 0.058542 | No |
| limx_imu | Box 0.015×0.015×0.004 | 0.01 | (0, 0, 0) | 1e-6 | 1e-6 | 1e-6 | No |
| abad_R_actuator_Link | Cylinder r=0.0483 L=0.0545 axis X | 0.95 | (−0.02725, 0, 0) | 0.001108 | 0.000789 | 0.000789 | No |
| abad_L_actuator_Link | Cylinder r=0.0483 L=0.0545 axis X | 0.95 | (−0.02725, 0, 0) | 0.001108 | 0.000789 | 0.000789 | No |
| **abad_R_Link** | Cylinder r=0.0483 L=0.0545 axis Y | 1.469 | (−0.0683, +0.035, 0) | **0.001220** | **0.001713** | **0.001220** | **Yes** |
| **abad_L_Link** | Cylinder r=0.0483 L=0.0545 axis Y | 1.469 | (−0.0683, −0.035, 0) | **0.001220** | **0.001713** | **0.001220** | **Yes** |
| hip_R_Link | Cylinder r=0.0483 L=0.0545 axis Y | 0.95 | (0, −0.08275, 0) | 0.000789 | 0.001108 | 0.000789 | No |
| hip_L_Link | Cylinder r=0.0483 L=0.0545 axis Y | 0.95 | (0, +0.08275, 0) | 0.000789 | 0.001108 | 0.000789 | No |
| hip_R_thigh_Link | Box 0.05×0.032×0.3 | 1.5 | (0, +0.04325, −0.15) | 0.011378 | 0.011563 | 0.000441 | No |
| hip_L_thigh_Link | Box 0.05×0.032×0.3 | 1.5 | (0, −0.04325, −0.15) | 0.011378 | 0.011563 | 0.000441 | No |
| **knee_R_Link** | Box 0.025×0.032×0.3 | 1.22 | (0, 0, −0.15) | **0.009254** | **0.009214** | **0.000168** | **Yes** |
| **knee_L_Link** | Box 0.025×0.032×0.3 | 1.22 | (0, 0, −0.15) | **0.009254** | **0.009214** | **0.000168** | **Yes** |
| **ankle_R_actuator_Link** | Cylinder r=0.03 L=0.0545 axis X | 0.95 | (−0.02725, 0, 0) | **0.000428** | **0.000449** | **0.000449** | **Yes** |
| **ankle_L_actuator_Link** | Cylinder r=0.03 L=0.0545 axis X | 0.95 | (−0.02725, 0, 0) | **0.000428** | **0.000449** | **0.000449** | **Yes** |
| **ankle_R_Link** | Box 0.2×0.07×0.03 | 0.62 | (+0.01, 0, −0.035) | **0.000300** | **0.002113** | **0.002320** | **Yes** |
| **ankle_L_Link** | Box 0.2×0.07×0.03 | 0.62 | (+0.01, 0, −0.035) | **0.000300** | **0.002113** | **0.002320** | **Yes** |

All off-diagonal inertia terms (ixy, ixz, iyz) are zero — the CoM frames are aligned with the
principal axes of each primitive shape.
