# Abduction Sweep Clearance, Flush Hip Placement and Trunk Clearance in the Design Builders

STATUS, 2026-11-18. IMPLEMENTED in the working tree of `tron1-rl-isaaclab-cozum` and its `design_generator` submodule, uncommitted, with the committed baseline assets regenerated. The three suites `design_generator_test.py`, `design_filter_test.py` and `catalog_fill_test.py` pass in the container `isaac-lab-base-shreyas-gpu0`, 168 with 14 skipped, and the two MuJoCo contact tests, skipped there for want of `mujoco`, pass on the host. NOT YET VERIFIED in Isaac Sim, section 7. The rules are always applied, as directed, with no switch to restore the former layout. One rule was added to the plan as first written, Rule 7d of section 3.7, which the existing nominal pose contact test forced.

## 1. Context and request

The design builders place each leg as a chain of actuator housings, the abduction housing welded to the trunk, the hip housing carrying the abduction joint, and the knee housing carrying the hip pitch joint (`design_generator/biped/builder.py:6-34`). The abduction joint rotates the hip housing, the knee housing and the whole leg about the x axis through the hip housing's centre, so the housings sweep through the space beneath and beside the trunk, which the present layout does not guard. Three requirements were stated.

1. The distance between the two abduction actuators must be at least $$2 (r_h \sin\theta_m + r_a)$$, with $$r_h$$ and $$r_a$$ the hip and abduction housing radii and $$\theta_m$$ the outer limit of the abduction range, and the trunk width must follow, so that the hip housings of the two legs can never collide.
2. The hip housing must start at the inner limit of the abduction housing, and the abduction to hip y offset must be set accordingly, replacing the square root formula at `design_generator/biped/builder.py:369-374`, for both legs.
3. At the largest abduction and adduction the hip and knee housings must not hit the trunk, and this fixes the z offset between the abduction housing and the trunk.

Both the biped and the quadruped are in scope, and the changes stay within `_cluster_layout` and `trunk_width` of each builder, with one small shared helper module and the `check_geometry` of the biped trimmed. Every rule is applied to every design. A fourth rule, 7d, which keeps the quadruped's thigh clear of its abduction housing, followed from the first two and is derived in section 3.7.

## 2. Variables and conventions

Every symbol used in section 3 is defined here, with its unit and the code name it has in the builders. A symbol keeps one meaning throughout.

Frame. The trunk frame has its origin at the centre of the trunk box, x forward, y to the left, z up. All housing positions below are in this frame unless stated. Lengths are in metres and angles in radians.

Housings. Each housing is a solid cylinder. The abduction housing has its axis along x, and the hip and knee housings have their axes along y when the abduction angle is zero (`QUAT_ALIGN_X`, `QUAT_ALIGN_Y`, `biped/builder.py:147-148`). The radius is half the catalogue diameter and the half length is half the catalogue length (`_housing_dims`, `biped/builder.py:212-215`).

| Symbol | Meaning | Unit | Code name |
| --- | --- | --- | --- |
| $$s$$ | Side sign of a leg, +1 for a left leg and −1 for a right leg | none | `sign`, `sy` |
| $$u$$ | Outward coordinate of a point, $$u = s\,y$$, so that it grows away from the centreline for either leg | m | none |
| $$a$$ | Trunk length along x | m | `spec.trunk_length` |
| $$c$$ | Trunk thickness along z, the spec's in the biped and $$2 r_a$$ in the quadruped | m | `spec.trunk_thickness`, `trunk_thickness(act_abad)` |
| $$W$$ | Trunk width along y, derived by the rules below and not read from the spec | m | `trunk_width(...)` |
| $$r_a, h_a$$ | Radius and half length of the abduction housing | m | `r_abad`, `hl_abad` |
| $$r_h, h_h$$ | Radius and half length of the hip housing | m | `r_hip`, `hl_hip` |
| $$r_k, h_k$$ | Radius and half length of the knee housing | m | `r_knee`, `hl_knee` |
| $$t$$ | Half extent of the thigh box along y, the gap the biped leaves between the hip and knee housings | m | `THIGH_BOX_HALF[1]` |
| $$c_{ty}, c_{tz}$$ | Trunk to abduction clearances along y and z | m | `clearance_trunk2abadAct_y`, `_z` |
| $$c_{hy}, c_{hz}$$ | Abduction to hip clearances along y and z | m | `clearance_abadAct2hipAct_y`, `_z` |
| $$c_{ky}, c_{kz}$$ | Hip to knee clearances along y and z | m | `clearance_hipAct2kneeAct_y`, `_z` |
| $$g$$ | The outward part of the abduction to hip clearance, $$g = \max(c_{hy}, 0)$$ | m | `max(spec.clearance_abadAct2hipAct_y, 0.0)` |
| $$m$$ | Minimum clear distance between housings | m | `HOUSING_MARGIN`, 0.0005 |
| $$\theta$$ | Abduction joint angle, positive when it lifts the outward side of a housing, see the sign note below | rad | joint position |
| $$\theta_{lo}, \theta_{hi}$$ | Lower and upper limits of the abduction range, with $$\theta_{lo} \le 0 \le \theta_{hi}$$ | rad | `ABAD_RANGE` |
| $$\theta_m$$ | Outer limit of the abduction range, $$\max(\lvert\theta_{lo}\rvert, \lvert\theta_{hi}\rvert)$$ | rad | none |
| $$S$$ | $$\sin(\min(\theta_m, \pi/2))$$ | none | `ABAD_SWEEP_SIN` |
| $$Y$$ | Outward coordinate of the abduction housing's centre | m | none |
| $$Y_p$$ | Outward coordinate of the hip housing's centre, which is the pivot of the abduction joint | m | none |
| $$u_k, z_k$$ | Outward and vertical offset of the knee housing's centre from the pivot at zero abduction | m | none |
| $$\ell(\theta)$$ | Height of the highest point of a housing above the pivot at abduction $$\theta$$ | m | `lift_max` |
| $$\rho(\theta)$$ | Outward coordinate of the innermost point of a housing, measured from the pivot's outward coordinate | m | `reach_min` |
| $$e$$ | Effective clearance between the trunk's lower face and the abduction housing's upper surface | m | `clearance_trunk2abadAct_z` or its floor |
| $$z_p$$ | Height of the pivot in the trunk frame | m | none |
| $$x_h, x_k$$ | x coordinates of the hip and knee housing centres | m | none |
| $$x_a, c_{tx}$$ | x coordinate of the abduction housing's centre and the trunk to abduction clearance along x | m | `abad_pos[0]`, `clearance_trunk2abadAct_x` |
| $$d_a$$ | Diameter of the abduction housing, $$2 r_a$$ | m | `act_abad.diameter` |
| $$c_{kt}$$ | Knee to thigh clearance along y | m | `clearance_kneeAct2thigh_y` |
| $$c_{ky}^{*}$$ | Effective hip to knee clearance along y, at least $$c_{ky}$$, of Rule 7d | m | `knee_gap(...)` |
| $$w$$ | Half extent of the thigh box along y in the quadruped, which is $$t$$ | m | `THIGH_BOX_HALF[1]` |
| $$z_a$$ | Height of the abduction housing's centre in the trunk frame | m | `abad_pos[2]` |
| $$y_{hip}$$ | Offset of the hip housing from the abduction body along y | m | `hip_pos_rel[1]`, `hip_pos[1]` |
| $$W_h, W_k$$ | Trunk width demanded by the hip sweep and by the knee sweep | m | none |

Sign note. The abduction joint axis is $$(1, 0, 0)$$ for a left leg and $$(-1, 0, 0)$$ for a right leg in the biped (`biped/builder.py:409`), so a positive angle abducts both legs outward and lifts the outward side of every housing of either leg. The quadruped uses $$(1, 0, 0)$$ for all four legs (`quadruped/builder.py:502`), so a positive angle lifts the outward side of a left leg's housings and lowers that of a right leg's. The quadruped's range is symmetric, $$\theta_{lo} = -\theta_{hi}$$, so the set of angles a right leg meets is the set a left leg meets and every expression below applies to both legs unchanged. This is an assumption, and a constants file giving the quadruped an asymmetric abduction range would need the mirrored interval for its right legs.

Pivot. The abduction joint is declared on the hip body (`biped/builder.py:504`, `quadruped/builder.py:502`), so its axis passes through the hip housing's centre. The hip housing therefore turns about its own centre, and the knee housing and everything below it turn about that point with it. The hip pitch joint turns the knee housing about its own axis, which is the cylinder's axis, and leaves its geometry unchanged.

Rotation. A point at outward coordinate $$u$$ and height $$z$$ relative to the pivot moves, for a positive $$\theta$$, to

$$
u' = u\cos\theta - z\sin\theta, \qquad z' = u\sin\theta + z\cos\theta
$$

A cylinder with its axis along y, half length $$h$$ and radius $$r$$, tilted by $$\theta$$ about x, reaches $$h\cos\theta + r\lvert\sin\theta\rvert$$ either side of its centre along the outward direction and $$h\lvert\sin\theta\rvert + r\cos\theta$$ either side along z. This holds for either sign of $$\theta$$, since the disc of the cylinder tilts equally far towards both sides.

## 3. Derivation

### 3.1 Flush placement of the hip, Rule 7a

1. The abduction housing's axis lies along x, so its extent along y is its radius either side of its centre. Its centre is at outward coordinate

$$
Y = \frac{W}{2} - r_a + c_{ty}
$$

   which is the offset `_cluster_layout` already uses (`biped/builder.py:362`, `quadruped/builder.py:389`). Its inner edge, the face nearest the centreline, is at $$Y - r_a$$.

2. The hip housing starts at that edge, so its inner face is on it and the housing extends outward by its length $$2 h_h$$, with the outward part of the clearance added on top. The pivot is at

$$
Y_p = Y - r_a + h_h + g
$$

3. Relative to the abduction body the hip offset along y is therefore

$$
y_{hip} = s\,(h_h - r_a + g)
$$

   which replaces the square root expression for the biped and the unsigned clearance for the quadruped. The clearance $$c_{hy}$$ enters through $$g$$ alone, so a negative value is read as zero. A negative clearance would push the hip inboard of the abduction housing's inner edge and void the bound of section 3.2.

### 3.2 Hip sweep and the trunk width, Rule 7b

4. At abduction $$\theta$$ the innermost point of the hip housing, measured from the pivot's outward coordinate, is

$$
\rho_h(\theta) = -\left(h_h\cos\theta + r_h\lvert\sin\theta\rvert\right)
$$

   so its outward coordinate from the centreline is

$$
Y_p + \rho_h(\theta) = (Y - r_a + g) + h_h(1 - \cos\theta) - r_h\lvert\sin\theta\rvert
$$

5. The middle term is never negative and $$\sin\lvert\theta\rvert$$ grows with $$\lvert\theta\rvert$$ up to $$\pi/2$$, so for every angle of the range

$$
Y_p + \rho_h(\theta) \ge Y - r_a + g - r_h S
$$

   The bound is the requested term $$r_h\sin\theta_m$$, made conservative by the dropped $$h_h(1 - \cos\theta)$$. Both signs of the range limits are covered by the absolute value in $$\theta_m$$.

6. The two legs of an axle can sit at their worst angles at once, since their joints are independent, so the two hip housings stay $$m$$ apart when twice the innermost coordinate is at least $$m$$. Dropping $$g$$, which only helps,

$$
2\left(Y - r_a - r_h S\right) \ge m \quad\Longleftrightarrow\quad 2Y \ge 2\left(r_a + r_h S\right) + m
$$

   The left side of the second form is the distance between the centres of the two abduction actuators, so this is the first requirement of section 1 with the margin of Rule 6 added to avoid contact at equality.

### 3.3 Knee sweep, Rule 7c

7. The knee housing sits outboard of the hip housing, at offset $$(u_k, z_k)$$ from the pivot, with

$$
u_k = h_h + 2t + c_{ky} + h_k \;\text{(biped)}, \qquad u_k = h_h + c_{ky} + h_k \;\text{(quadruped)}, \qquad z_k = c_{kz}
$$

   The biped's thigh box lies between the two housings, which the extra $$2t$$ accounts for (`biped/builder.py:377`, `quadruped/builder.py:407`). The rotation of section 2 moves the knee housing's centre and tilts its cylinder, so the outward coordinate of its innermost point, measured from the pivot, is

$$
\rho_k(\theta) = u_k\cos\theta - z_k\sin\theta - h_k\cos\theta - r_k\lvert\sin\theta\rvert
$$

8. The knee housing must also keep $$m/2$$ clear of the centreline at every angle, which gives

$$
Y_p + \min_\theta \rho_k(\theta) \ge \frac{m}{2}
$$

   and, with $$Y_p = Y - r_a + h_h + g$$ and $$g$$ dropped,

$$
Y \ge \frac{m}{2} + r_a - h_h - \min_{\theta \in [\theta_{lo}, \theta_{hi}]} \rho_k(\theta)
$$

   On each side of zero, $$\rho_k$$ has the form $$-(A\sin\theta - B\cos\theta)$$, a single sinusoid, so its minimum over an interval is at an end or at the angle $$\operatorname{atan2}(A, -B)$$ where its derivative vanishes. Section 6.1 evaluates it in closed form.

9. The knee term was added because the sweep of section 7 found that Rule 7a with the hip term alone raises the biped's knee to knee contacts from 75 to 255 designs. The hip moves inward by the flush rule and carries the knee with it, and the hip term does not see the knee.

### 3.4 Trunk width

10. Substituting $$Y = W/2 - r_a + c_{ty}$$ into steps 6 and 8 gives the width demanded by the sweep of each housing

$$
W_h = 2\left(2 r_a + r_h S - c_{ty}\right) + m
$$

$$
W_k = 2\left(\frac{m}{2} + 2 r_a - h_h - c_{ty} - \min_\theta \rho_k(\theta)\right)
$$

   and the trunk is as wide as the largest of these and the former Rule 5 width, twice the abduction diameter plus `TRUNK_WIDTH_OFFSET`,

$$
W = \max\left(2 d_a + \text{TRUNK\_WIDTH\_OFFSET},\; W_h,\; W_k\right), \qquad d_a = 2 r_a
$$

   The former second argument, which kept the biped's square root real and the quadruped's two hips of an axle apart, is superseded, the square root being no longer evaluated and the hips no longer centred on their abduction housings.

### 3.5 Trunk clearance, Rule 8

11. The pivot height in the trunk frame is $$z_p = z_a + c_{hz}$$ with the abduction housing's centre at $$z_a = -(c/2 + r_a + e)$$ in the biped (`biped/builder.py:363`). A housing reaches height $$\ell(\theta)$$ above the pivot, and the trunk's lower face is at $$-c/2$$, so it clears the trunk by $$m$$ at every angle when

$$
z_p + \max_\theta \ell(\theta) \le -\frac{c}{2} - m
$$

12. For the hip housing, whose centre is the pivot,

$$
\ell_h(\theta) = h_h\lvert\sin\theta\rvert + r_h\cos\theta
$$

and for the knee housing, from the rotation of section 2,

$$
\ell_k(\theta) = u_k\sin\theta + z_k\cos\theta + h_k\lvert\sin\theta\rvert + r_k\cos\theta
$$

The knee housing is carried outward by the lever arm $$u_k$$, so abducting rotates it upward by about $$u_k\sin\theta$$, which is why the knee housing and not the hip housing is the one that reaches the trunk.

13. Substituting $$z_p$$ and solving for the clearance gives the floor on $$e$$

$$
e \ge \max_\theta \ell(\theta) - r_a + c_{hz} + m
$$

and the effective clearance is the larger of the spec's $$c_{tz}$$ and the floor of every housing that is above the trunk in plan view,

$$
e = \max\left(c_{tz},\; \max_{i \in \{h, k\}}\left[\max_\theta \ell_i(\theta) - r_a + c_{hz} + m\right]\right)
$$

14. A housing is above the trunk in plan view when its x extent meets the trunk's. Rotation about x leaves the x extent of a cylinder with its axis along y at $$r$$ either side of its centre, so housing $$i$$ takes part when

$$
\lvert x_i\rvert - r_i < \frac{a}{2}
$$

The test ignores y, which treats a housing as meeting the trunk's width whenever it meets its length. That is conservative, and the floor applies to every design for which it holds.

15. In the quadruped the test never holds. Rule 1 puts the abduction housing flush against the trunk's end face, $$\lvert x_a\rvert = a/2 + h_a + c_{tx}$$, and Rule 6 puts the hip at least $$h_a + r_h$$ further out, so

$$
\lvert x_h\rvert - r_h \ge \frac{a}{2} + 2 h_a + c_{tx} > \frac{a}{2}
$$

for any clearance $$c_{tx} > -2 h_a$$ and a nonnegative abduction to hip x clearance, and the knee shares the hip's x. The rule therefore adds no code to the quadruped, and a test asserts the separation, so that a future placement that breaks it is caught. This is the case in the sweep, which found no trunk contact in the quadruped under any rule set.

16. The maxima of $$\ell$$ over the range are evaluated in closed form. On each side of zero, $$\ell$$ has the form $$A\sin\theta + B\cos\theta$$, whose maximum over an interval is at an end or at $$\operatorname{atan2}(A, B)$$. Section 6.1 gives the function.

### 3.6 Baseline values

Computed from the unmodified builders and from the prototype with every rule applied, for the baseline `variable_params.json` of each family.

| Quantity | Biped now | Biped after | Quadruped now | Quadruped after |
| --- | --- | --- | --- | --- |
| $$r_a, r_h$$ [m] | 0.0483, 0.0483 | same | 0.046, 0.046 | same |
| $$\theta_m$$ [rad], $$S$$ | 1.39626, 0.98481 | same | 0.80285, 0.71934 | same |
| Trunk width $$W$$ [m] | 0.2600 | 0.2888 | 0.1960 | 0.2507 |
| Abduction y [m] | 0.0817 | 0.0961 | 0.0520 | 0.0793 |
| Hip y relative to abduction [m] | 0.0000 | −0.0211 | 0.0000 | −0.0260 |
| Knee y relative to hip [m] | −0.0078 | −0.0078 | 0.0500 | 0.0525 |
| Abduction z [m] | −0.1875 | −0.1875 | 0.0000 | 0.0000 |

Across the shelves, the rules widen the trunk of the biped by 0 to 39.1 mm, 24.5 mm on average over 3375 abduction, hip and knee picks, and that of the quadruped by 19.7 to 65.5 mm, 53.8 mm on average over 2744 picks. The z floor binds for 36 of the 3375 biped picks, by up to 23.3 mm, and for none of the quadruped's. The knee gap floor of Rule 7d binds for 684 of the 2744 quadruped picks at the updated baseline clearance, by up to 18 mm, and the baseline's own gap rises from 10.0 to 12.5 mm.

### 3.7 Thigh clearance of the abduction housing, Rule 7d, quadruped

17. In the quadruped the knee housing is at $$h_h + h_k + c_{ky}$$ outboard of the hip housing, and the thigh box lies a further $$h_k + w + c_{kt}$$ outboard of the knee centre (`quadruped/builder.py:407`, `:416`). With the hip placed by step 3, the thigh's centre is at outward offset from the abduction axis

$$
d = 2 h_h + 2 h_k + w + g + c_{ky} + c_{kt} - r_a
$$

    and its inner face at $$d - w$$. Before Rule 7a the hip was centred on the abduction housing and this face lay beyond the abduction housing's outer face. Rule 7a pulls the leg in by $$r_a - h_h$$, which brings the thigh over the housing's extent along y, so that a thigh that leans or pitches towards it meets it.

18. The thigh clears the housing at every hip pitch when its inner face is beyond the housing's outer face by $$m$$, which is $$d - w \ge r_a + m$$, and so

$$
c_{ky}^{*} = \max\left(c_{ky},\; 2 r_a + m - 2 h_h - 2 h_k - g - c_{kt}\right)
$$

    This is the knee's y offset of `_cluster_layout` and the $$u_k$$ of step 7 for the quadruped, so the width of section 3.4 uses it too. The biped needs no such rule, its thigh lying between the hip and knee housings in a layout that already overlapped the abduction housing's extent before this change, and its nominal pose passing the contact test.

19. The rule was found by the existing test `test_every_pick_of_the_leg_cluster_builds_without_a_housing_contact`, which builds every pick at the zero pose of the Isaac Lab document and demands no contact. With Rules 7a to 7c alone, 78 of the 2744 quadruped picks put a front thigh into its abduction housing, for example Unitree B1 abduction, Unitree A1 hip and MIT Cheetah 3 knee, by 0.7 mm to 1.0 mm. Rule 7d removes them and keeps the thigh clear at every pitch.

## 4. Options evaluated

The workspace rules ask for the smallest change that keeps callers intact, so each choice was weighed against them.

1. Carrying the range on `DesignBounds`. The request offered `DesignBounds` attached to every `DesignSpec`. `DesignBounds` holds only the search interval of each free variable (`design_generator/biped/generate.py:250-277`, `quadruped/generate.py:209-237`), and no joint range. The range that matters is `ABAD_RANGE`, which the builders already load from `constant_params.json` (`biped/builder.py:112`, `quadruped/builder.py:129`) and write into the abduction joint's `range` (`biped/builder.py:504`), so the joint stop and the sweep bound are one number by construction. Attaching a bounds object would put a nested dataclass in a frozen `DesignSpec` whose `to_dict` returns every field (`biped/generate.py:122-125`) and whose dictionary `CMAESDesignGenerator._spec_from_parameters` re-reads (`co_optimisation/co_optimisation/runners/usd_generator.py:872-885`), the design records and replay tests would have to serialise it, and nothing would vary it. The plan reads `ABAD_RANGE` and leaves `DesignBounds` and `DesignSpec` untouched. Should a later study sweep the range per design, it becomes a fixed constant tier field of two floats read by the same lines.

2. A switch. None is added, as directed. The consequence is that the baseline of each family no longer reproduces its donor asset, section 5.

3. Where the helper lives. The two builders are separate packages with their own copies of every helper, but the two sinusoid extremum functions are identical for both. They go in one new module in `design_generator/common/`, beside `actuator_catalog.py`, which both builders already import from through their actuator modules, rather than being written twice.

4. The knee actuator in `trunk_width`. `trunk_width(spec, act_abad, act_hip)` has callers in `build_xml`, `trunk_body` and the in place editor's `_trunk_entry` (`usd_generator.py:138-147`), and adding a parameter would change all of them. `spec.actuators()` already resolves the knee at index 2 in both families, so `trunk_width` reads it from the spec and its signature is unchanged.

5. The clearance clamp. `max(c_hy, 0)` makes the biped baseline's `clearance_abadAct2hipAct_y` of −0.008877 m inert, since the square root form that absorbed it is gone. The alternative, raising on a negative value, would be caught by `_build_xml` and replaced by the last feasible design with CMA-ES still credited, the failure mode of `plans/housing_clearance_and_trunk_width.md` section 2, so the clamp is kept and recorded in the geometry document.

## 5. Callers and compatibility

Every site that reads the two methods was enumerated before choosing the change.

| Site | Reads | Effect |
| --- | --- | --- |
| `biped/builder.py` `build_xml`, `trunk_body`, `check_geometry` | `trunk_width`, `_cluster_layout` | New width and offsets |
| `quadruped/builder.py` `build_xml`, `trunk_body` | same | same |
| `usd_generator.py:192-198`, `:246-252` | `_cluster_layout` keys `abad_pos`, `hip_pos_rel`, `hip_pos` | Keys unchanged, the in place editor writes the new offsets with no edit |
| `usd_generator.py:138-147` | `trunk_body` | The trunk entry follows the width with no edit |
| `usd_generator.py:293-305`, `:263-271` | z and x terms of the layout and the clearances | The y offset is not read, so the nominal pose solve is unchanged except through the biped's abduction z, which `_biped_chain` reads at `:305` and which the rule changes for the 36 picks above |
| `scripts/tools/generate_default_mjcf.py` | each family's baseline | Regenerates both committed assets |

The rules are always applied, so they replace the former layout outright, in the manner of the 2026-11-16 housing plan (`plans/housing_clearance_and_trunk_width.md` section 5). Four things change as a result.

1. The baselines move. The biped's hip moves 6.6 mm inboard and the trunk widens by 28.8 mm, and the quadruped's hip moves 26.0 mm inboard and the trunk widens by 54.7 mm, section 3.6. The baseline designs were updated with it. `design_generator/biped/variable_params.json` carries `trunk_width` 0.288832, the derived width, and `clearance_abadAct2hipAct_y` 0.0, its former −0.008877 being read as zero, and `design_generator/quadruped/variable_params.json` carries `trunk_width` 0.250679 and `clearance_hipAct2kneeAct_y` 0.0125, the gap Rule 7d gives it. The committed assets `environments/environments/assets/mjcf/solefoot/tron1/base_robot.xml` and `environments/environments/assets/mjcf/quadruped/quadruped.xml` were regenerated with `scripts/tools/generate_default_mjcf.py`, and the donor equivalence of both robots is then an equivalence to the former layout and not to the donor, which the geometry documents record.

2. Earlier design records and runs replay differently for every pick, since the width and the hip offset now follow the rules. The runs in flight keep their installed copies until `design_generator` is reinstalled.

3. Tests that pin the former layout change, section 6.5.

4. The defect left standing in the former layout, recorded in both geometry documents, is that the hip and knee housings of an axle were not guaranteed clear of each other, nor the knee housing clear of the trunk, through the abduction range. Section 7 counts it.

## 6. Code changes

Six places take code. Line numbers are those of the working tree on 2026-11-18.

### 6.1 New module `design_generator/common/housing_sweep.py`

```python
"""Extremes over an abduction range of the housings that the abduction joint sweeps.

See plans/design_generator_update.md sections 2 and 3 for the geometry. Both functions assume the
range contains zero, which holds for every abduction range the builders load.
"""
from __future__ import annotations

import math


def trig_max(a: float, b: float, t0: float, t1: float) -> float:
    """Maximum of a * sin(t) + b * cos(t) over [t0, t1]."""
    candidates = [t0, t1]
    t_star = math.atan2(a, b)
    if t0 < t_star < t1:
        candidates.append(t_star)
    return max(a * math.sin(t) + b * math.cos(t) for t in candidates)


def lift_max(r: float, hl: float, u: float, z: float, lo: float, hi: float) -> float:
    """Largest height of a y axis cylinder above the pivot over the range, equation of step 12.

    r and hl are the cylinder's radius and half length, (u, z) its centre's outward and vertical
    offset from the pivot at zero abduction, and lo and hi the range limits.
    """
    return max(trig_max(u + hl, z + r, max(lo, 0.0), max(hi, 0.0)),
               trig_max(u - hl, z + r, min(lo, 0.0), min(hi, 0.0)))


def reach_min(r: float, hl: float, u: float, z: float, lo: float, hi: float) -> float:
    """Smallest outward coordinate of a y axis cylinder from the pivot's over the range, step 7."""
    p = u - hl
    return min(-trig_max(z + r, -p, max(lo, 0.0), max(hi, 0.0)),
               -trig_max(z - r, -p, min(lo, 0.0), min(hi, 0.0)))
```

Both functions were checked against a 20001 point brute force over 3000 random cylinders and ranges, with no mismatch above 1e-6.

### 6.2 `design_generator/biped/builder.py`

The import and the derived constant, after `HOUSING_MARGIN` and `TRUNK_WIDTH_OFFSET` at line 131.

```python
from design_generator.common.housing_sweep import lift_max, reach_min

ABAD_SWEEP_SIN = math.sin(min(max(abs(v) for v in ABAD_RANGE), math.pi / 2.0))
```

`trunk_width` at lines 289 to 303 is replaced, its docstring stating equations of sections 3.2 to 3.4.

```python
def trunk_width(spec, act_abad, act_hip) -> float:
    r_abad, _ = _housing_dims(act_abad)
    r_hip, hl_hip = _housing_dims(act_hip)
    r_knee, hl_knee = _housing_dims(spec.actuators()[2])
    lo, hi = ABAD_RANGE
    u_knee = hl_hip + 2.0 * THIGH_BOX_HALF[1] + spec.clearance_hipAct2kneeAct_y + hl_knee
    reach = reach_min(r_knee, hl_knee, u_knee, spec.clearance_hipAct2kneeAct_z, lo, hi)
    half_hip = r_abad + r_hip * ABAD_SWEEP_SIN
    half_knee = r_abad - hl_hip - reach
    half = max(half_hip, half_knee) + HOUSING_MARGIN / 2.0
    own = 2.0 * act_abad.diameter + TRUNK_WIDTH_OFFSET
    return max(own, 2.0 * (half + r_abad - spec.clearance_trunk2abadAct_y))
```

`_cluster_layout` at lines 355 to 392. The abduction housing's z is computed last, once the hip and knee offsets are known, so its line at 363 moves, and the hip's y offset at lines 371 and 372 loses the square root.

```python
    x_offset_trunk2abadAct = hl_abad + spec.clearance_trunk2abadAct_x
    y_offset_trunk2abadAct = trunk_width(spec, act_abad, act_hip) / 2.0 - r_abad + spec.clearance_trunk2abadAct_y

    x_offset_abadAct2hipAct = -max(hl_abad + r_hip + spec.clearance_abadAct2hipAct_x,
                                   hl_abad + r_knee + HOUSING_MARGIN + spec.clearance_hipAct2kneeAct_x)
    y_offset_abadAct2hipAct = sign * (hl_hip - r_abad + max(spec.clearance_abadAct2hipAct_y, 0.0))
    z_offset_abadAct2hipAct = spec.clearance_abadAct2hipAct_z
    hip_pos_rel = np.array([x_offset_abadAct2hipAct, y_offset_abadAct2hipAct, z_offset_abadAct2hipAct])

    # knee_pos_rel and thigh_pos_rel as before

    lo, hi = ABAD_RANGE
    x_hip = x_offset_trunk2abadAct + x_offset_abadAct2hipAct
    x_knee = x_hip + knee_pos_rel[0]
    lift = -math.inf
    if abs(x_hip) - r_hip < spec.trunk_length / 2.0:
        lift = lift_max(r_hip, hl_hip, 0.0, 0.0, lo, hi)
    if abs(x_knee) - r_knee < spec.trunk_length / 2.0:
        lift = max(lift, lift_max(r_knee, hl_knee, sign * knee_pos_rel[1], knee_pos_rel[2], lo, hi))
    clearance_z = max(spec.clearance_trunk2abadAct_z, lift - r_abad + z_offset_abadAct2hipAct + HOUSING_MARGIN)
    z_offset_trunk2abadAct = -(spec.trunk_thickness / 2.0 + r_abad + clearance_z)
    abad_pos = np.array([x_offset_trunk2abadAct, sign * y_offset_trunk2abadAct, z_offset_trunk2abadAct])
```

`check_geometry` at lines 330 to 348 keeps its first check and loses the square root block, which no longer applies. The first check cannot fail either, since the width exceeds the abduction diameter, and it is left in place as a guard.

### 6.3 `design_generator/quadruped/builder.py`

The import and constant after `ABAD_RANGE` at line 129.

```python
from design_generator.common.housing_sweep import reach_min

ABAD_SWEEP_SIN = math.sin(min(max(abs(v) for v in ABAD_RANGE), math.pi / 2.0))
```

`knee_gap`, Rule 7d, is added before `trunk_width`.

```python
def knee_gap(spec, r_abad: float, hl_hip: float, hl_knee: float) -> float:
    g = max(spec.clearance_abadAct2hipAct_y, 0.0)
    floor = (2.0 * r_abad + HOUSING_MARGIN - 2.0 * hl_hip - 2.0 * hl_knee - g
             - spec.clearance_kneeAct2thigh_y)
    return max(spec.clearance_hipAct2kneeAct_y, floor)
```

`trunk_width` at lines 182 to 197 is replaced, and reads the knee's offset through `knee_gap`.

```python
def trunk_width(spec, act_abad, act_hip) -> float:
    r_abad, _ = _housing_dims(act_abad)
    r_hip, hl_hip = _housing_dims(act_hip)
    r_knee, hl_knee = _housing_dims(spec.actuators()[2])
    lo, hi = ABAD_RANGE
    u_knee = hl_hip + knee_gap(spec, r_abad, hl_hip, hl_knee) + hl_knee
    reach = reach_min(r_knee, hl_knee, u_knee, spec.clearance_hipAct2kneeAct_z, lo, hi)
    half_hip = r_abad + r_hip * ABAD_SWEEP_SIN
    half_knee = r_abad - hl_hip - reach
    half = max(half_hip, half_knee) + HOUSING_MARGIN / 2.0
    own = 2.0 * act_abad.diameter + TRUNK_WIDTH_OFFSET
    return max(own, 2.0 * (half + r_abad - spec.clearance_trunk2abadAct_y))
```

`_cluster_layout` at lines 398 to 409, the hip offset and the knee offset, which takes its y from `knee_gap`.

```python
    hip_pos = (
        sx * hip_dx,
        sy * (hl_hip - r_abad + max(spec.clearance_abadAct2hipAct_y, 0.0)),
        spec.clearance_abadAct2hipAct_z,
    )
    knee_pos = (
        spec.clearance_hipAct2kneeAct_x,
        sy * (hl_hip + hl_knee + knee_gap(spec, r_abad, hl_hip, hl_knee)),
        spec.clearance_hipAct2kneeAct_z,
    )
```

The hip's x offset, `thigh_pos` and `check_geometry` are unchanged, and the abduction's z offset needs no floor, section 3.5 step 15.

### 6.4 Documents

Both `GEOMETRY_RULES.md` files gain Rules 7 and 8, stating the equations of section 3, the clamp on the clearance, the table of section 3.6 for that family, and the standing defect of section 5. The biped's `y_hip` and `z_abad` blocks at lines 61 to 69 take the new forms. Both `DESIGN_PARAMETERS.md` files record that `trunk_width` and the hip y clearance's negative values are inert. The module docstrings of the two builders gain one paragraph. `context/design_filtering.md` section 9 gains a dated paragraph with the rule, the baseline table and the sweep counts, and `plans/README.md` gains the row of this plan.

### 6.5 Tests in `co_optimisation/tests/design_generator_test.py`

Changed. `test_the_trunk_width_follows_the_abduction_actuator_and_not_the_spec` at line 1033 asserts that the baseline width equals the spec's and that the quadruped's width is exactly twice the abduction diameter plus the offset, both of which the rules void. It keeps the assertions that the width ignores `spec.trunk_width` and is at least the Rule 5 width, and drops the two equalities. `test_the_baseline_emits_the_committed_asset` at line 780 passes again once the assets are regenerated.

Added.

```python
def _hips(family, spec):
    b = U.FAMILIES[family].module.builder
    a, h, k = spec.actuators()[:3]
    leg = "L" if family == "biped" else "FL"
    lay = b._cluster_layout(leg, spec, a, h, k)
    hip_key = "hip_pos_rel" if family == "biped" else "hip_pos"
    knee_key = "knee_pos_rel" if family == "biped" else "knee_pos"
    return b, lay, lay[hip_key], lay[knee_key]


@pytest.mark.parametrize("family", FAMILIES)
def test_the_hip_starts_at_the_abduction_inner_edge_and_the_housings_never_near_the_centreline(family):
    fam = U.FAMILIES[family]
    names = [a.name for a in fam.full_catalog()]
    for abad, hip, knee in itertools.product(names, repeat=3):
        spec = _picked(family, abad, hip, knee)
        b, lay, hip_rel, knee_rel = _hips(family, spec)
        lo, hi = b.ABAD_RANGE
        theta = np.linspace(lo, hi, 161)
        y_hip = lay["abad_pos"][1] + hip_rel[1]
        g = max(spec.clearance_abadAct2hipAct_y, 0.0)
        assert y_hip - lay["hl_hip"] == pytest.approx(lay["abad_pos"][1] - lay["r_abad"] + g, abs=1e-9)
        hip_in = y_hip - (lay["hl_hip"] * np.cos(theta) + lay["r_hip"] * np.abs(np.sin(theta)))
        u, z = knee_rel[1], knee_rel[2]
        knee_in = y_hip + u * np.cos(theta) - z * np.sin(theta) - (
            lay["hl_knee"] * np.cos(theta) + lay["r_knee"] * np.abs(np.sin(theta)))
        assert hip_in.min() >= b.HOUSING_MARGIN / 2.0 - 1e-9, (abad, hip, knee)
        assert knee_in.min() >= b.HOUSING_MARGIN / 2.0 - 1e-9, (abad, hip, knee)


def test_the_biped_housings_clear_the_trunk_through_the_abduction_range():
    fam = U.FAMILIES["biped"]
    names = [a.name for a in fam.full_catalog()]
    for abad, hip, knee in itertools.product(names, repeat=3):
        spec = _picked("biped", abad, hip, knee)
        b, lay, hip_rel, knee_rel = _hips("biped", spec)
        theta = np.linspace(*b.ABAD_RANGE, 161)
        z_pivot = lay["abad_pos"][2] + hip_rel[2]
        x_hip = lay["abad_pos"][0] + hip_rel[0]
        x_knee = x_hip + knee_rel[0]
        trunk_floor = -spec.trunk_thickness / 2.0
        if abs(x_hip) - lay["r_hip"] < spec.trunk_length / 2.0:
            top = z_pivot + lay["hl_hip"] * np.abs(np.sin(theta)) + lay["r_hip"] * np.cos(theta)
            assert top.max() <= trunk_floor - b.HOUSING_MARGIN + 1e-9, (abad, hip, knee)
        if abs(x_knee) - lay["r_knee"] < spec.trunk_length / 2.0:
            top = (z_pivot + knee_rel[1] * np.sin(theta) + knee_rel[2] * np.cos(theta)
                   + lay["hl_knee"] * np.abs(np.sin(theta)) + lay["r_knee"] * np.cos(theta))
            assert top.max() <= trunk_floor - b.HOUSING_MARGIN + 1e-9, (abad, hip, knee)


def test_the_quadruped_housings_stay_beyond_the_trunk_end_face():
    fam = U.FAMILIES["quadruped"]
    names = [a.name for a in fam.full_catalog()]
    for abad, hip, knee in itertools.product(names, repeat=3):
        spec = _picked("quadruped", abad, hip, knee)
        b, lay, hip_rel, knee_rel = _hips("quadruped", spec)
        x_hip = abs(lay["abad_pos"][0] + hip_rel[0])
        assert x_hip - lay["r_hip"] > spec.trunk_length / 2.0
        assert abs(x_hip + knee_rel[0]) - lay["r_knee"] > spec.trunk_length / 2.0
```

The tests build the left leg, so `knee_rel[1]` is already the outward offset $$u_k$$. All three tests are analytic, and the first two take a few seconds over the 3375 and 2744 picks.

A MuJoCo contact test completes the set, skipped where the module is absent. For a stride of 13 through the abduction, hip and knee picks of each family it builds `build_xml(spec)`, sets both abduction joints to every pair from the limits, zero and a grid between, calls `mujoco.mj_kinematics` and `mj_collision`, ignores contacts with the world and those shallower than 0.1 mm, and asserts that no contact exists between the two hip bodies, between the two knee bodies, or between the trunk and any hip or knee body. The sweep script of section 7 is the full version of it.

## 7. Verification

Done on this host in a scratch copy of `design_generator` with the rules applied, and no repository file edited. The sweep builds every abduction, hip and knee pick, 3375 for the biped and 2744 for the quadruped, sets the two abduction joints to every pair of a grid of the range, 100 pose pairs for the biped and 81 for the quadruped, with the pitch joints at zero, calls `mj_collision`, and counts a contact when two bodies other than the world penetrate by more than 0.1 mm. The arms are the former layout, Rule 7 with the hip term alone, Rule 7 with the hip and knee terms, Rule 8 alone, and every rule together. Contacts are counted per design, a design counting once if any pose contacts, and per design and pose.

Biped, 3375 designs.

| Contact | Former layout | Rule 7 hip term | Rule 7 hip and knee | Rule 8 alone | Every rule |
| --- | --- | --- | --- | --- | --- |
| Hip with trunk | 0 | 0 | 0 | 0 | 0 |
| Knee with trunk, designs | 18 | 27 | 27 | 0 | 0 |
| Knee with trunk, design and pose records | 425 | 1207 | 1207 | 0 | 0 |
| Hip with hip, designs | 0 | 0 | 0 | 0 | 0 |
| Knee with knee, designs | 75 | 255 | 0 | 75 | 0 |
| Knee with knee, design and pose records | 149 | 990 | 0 | 149 | 0 |

Quadruped, 2744 designs.

| Contact | Former layout | Rule 7 hip term | Rule 7 hip and knee | Every rule |
| --- | --- | --- | --- | --- |
| Hip with hip, designs | 1008 | 0 | 0 | 0 |
| Hip with hip, design and pose records | 29204 | 0 | 0 | 0 |
| Any hip or knee with trunk | 0 | 0 | 0 | 0 |
| Knee with knee | 0 | 0 | 0 | 0 |

The rules therefore prevent every one of the contacts they address. In the biped, 18 designs and 425 design and pose records of the knee striking the trunk, and 75 designs and 149 records of the two knee housings meeting, fall to none, and the quadruped's 1008 designs and 29204 records of the two hip housings meeting fall to none. Three points qualify the result.

1. Rule 7 with the hip term alone is worse than the former layout for the biped, raising the knee to trunk contacts from 18 to 27 designs and the knee to knee contacts from 75 to 255. The flush hip carries the knee inward, so the knee term of Rule 7c and the clearance of Rule 8 are not optional parts of the change. The knee term removes the knee to knee contacts and Rule 8 removes the trunk contacts, and each is needed for its own.

2. The biped's hip never struck the trunk under any arm, since at baseline the hip's abduction depth of 0.0925 m exceeds its reach $$\sqrt{h_h^2 + r_h^2}$$ of 0.0555 m. Rule 8 is exercised by the knee, whose lever arm $$u_k$$ lifts it.

3. The quadruped never struck the trunk under any arm, as step 15 predicts.

Contacts remain between the two legs' thighs, shanks and feet in every design of either robot, 3375 of 3375 biped and 2744 of 2744 quadruped designs for the shank and foot bodies against each other. They occur at no pose where both legs abduct outward and at no nominal pose, so they arise where a leg adducts across the centreline and the lower leg swings into the other. The rules do not address them, the hip and knee housings being the request, and the biped's thigh to thigh contacts rise from 2895 to 3375 designs because the leg moves 6.6 mm inboard. The remainder of the design space, a smaller adduction limit or a leg of reduced reach, is outside this plan.

The sweep has limits. The pitch joints are at zero, so the thighs and shanks are straight and no pitch posture is covered, the counts are those of MuJoCo, whose parent and child filter differs from PhysX's articulation self collision, and the Isaac Sim checks below were not run. The sweep script, its outputs and the diff of the prototype against the tree are kept in `context/artefacts/design_generator_update-2026-11-18-rule-sweep/`.

After implementation. The sweep script was run again on the production builders with no prototype switch, and gave the same counts, no hip, knee or trunk contact in either robot, the quadruped's thigh and shank counts changing slightly with Rule 7d, to 2618 designs of lower leg against thigh and 2604 of thigh against thigh. The tests of section 6.5 pass, and the existing nominal pose test `test_every_pick_of_the_leg_cluster_builds_without_a_housing_contact` passes for every pick of both shelves, which it did not before Rule 7d. The three suites pass in the container `isaac-lab-base-shreyas-gpu0`, 168 with 14 skipped, run with the repository's own modules on the path and not the installed copies, and the two MuJoCo contact tests, which the container skips for want of `mujoco`, pass on the host in a scratch virtual environment. That environment lacks `cma` and `cmaes`, so 20 tests that exercise the search stubs fail there for the stubs' missing attributes alone, as they do for any change.

A defect that predates this change. The three suites do not collect as they stand, since `usd_generator.py:63` imports `co_optimisation.utils.common` and the suites' module stubs do not provide it, the guarded import it once had being commented out. They were run through a pytest plugin that loads `utils/common.py` first, a scratch file that is not part of the repository, and the suites' own headers were left alone.

Still to do. In the training containers reinstall `design_generator` and `co_optimisation`, which are installed copies and not editable, and run `sim_asset_equivalence_check.py` and `sim_nominal_pose_check.py`, together with a population written in place, to confirm the editor places the hip and widens the trunk to the link dict. The containers were idle on 2026-11-18, and the Isaac python could not import `mujoco` there, so the sweep ran on this host.

## 8. Risks

1. The x overlap test of step 14 ignores y, so it floors the abduction z for housings that the trunk's width may not in fact cover. The sweep shows the floor binding for 36 of 3375 picks, by at most 23.3 mm, and the cost is a lower leg for those picks.

2. The trunk widens for every pick, so its mass and inertia and the leg track change and CMA-ES sees them as part of the abduction, hip and knee picks. The quadruped's trunk widens by 54 mm on average.

3. The distance in the request was read as centre to centre, which is what the derivation yields. Read as edge to edge it would add $$2 r_a$$ with no collision argument behind it.

4. A $$\theta_m$$ above $$\pi/2$$ is capped by $$S$$, so a constants file widening `ABAD_RANGE` past 90 degrees does not widen the hip term further, while the bound of step 5 holds there as well. The knee term and the lift use the range directly.

5. The biped baseline's negative `clearance_abadAct2hipAct_y` is read as zero, section 4 item 5.

6. The quadruped's mirrored abduction axes are assumed to leave the range symmetric, section 2, and the helper assumes the range contains zero.

## 9. Conclusion

The three requirements reduce to a flush hip offset, equation of step 3, a trunk width taken from the hip and knee sweeps, steps 10, and a floor on the abduction to trunk clearance, step 13, all read from the abduction range the builders already hold, with the helper module of section 6.1 doing the trigonometric extremes. They change `trunk_width` and `_cluster_layout` of both builders and leave the editor, the nominal pose solve and the catalogues as they are. A sweep of the prototype over both shelves found that they remove all 1008 quadruped designs with hip contact and all 18 biped designs with knee to trunk contact and 75 with knee to knee contact, and that the hip term alone would have made the biped worse, so the knee term and the clearance floor belong with it.

## Notation chart

| Symbol | Meaning | Unit | Section |
| --- | --- | --- | --- |
| $$s$$ | Side sign of a leg, +1 left and −1 right | none | 2 |
| $$u$$ | Outward coordinate, $$s\,y$$ | m | 2 |
| $$a$$ | Trunk length | m | 2 |
| $$c$$ | Trunk thickness | m | 2 |
| $$W$$ | Derived trunk width | m | 2 |
| $$r_a, h_a$$ | Abduction housing radius and half length | m | 2 |
| $$r_h, h_h$$ | Hip housing radius and half length | m | 2 |
| $$r_k, h_k$$ | Knee housing radius and half length | m | 2 |
| $$d_a$$ | Abduction housing diameter, $$2 r_a$$ | m | 3.4 |
| $$t$$ | Half extent of the thigh box along y | m | 2 |
| $$c_{ty}, c_{tz}$$ | Trunk to abduction clearances | m | 2 |
| $$c_{hy}, c_{hz}$$ | Abduction to hip clearances | m | 2 |
| $$c_{ky}, c_{kz}$$ | Hip to knee clearances | m | 2 |
| $$c_{tx}$$ | Trunk to abduction clearance along x | m | 3.5 |
| $$g$$ | Outward part of $$c_{hy}$$, $$\max(c_{hy}, 0)$$ | m | 2 |
| $$m$$ | Housing margin | m | 2 |
| $$\theta$$ | Abduction joint angle | rad | 2 |
| $$\theta_{lo}, \theta_{hi}$$ | Abduction range limits | rad | 2 |
| $$\theta_m$$ | Outer limit of the range | rad | 2 |
| $$S$$ | $$\sin(\min(\theta_m, \pi/2))$$ | none | 2 |
| $$Y$$ | Outward coordinate of the abduction centre | m | 2 |
| $$Y_p$$ | Outward coordinate of the pivot | m | 2 |
| $$u_k, z_k$$ | Knee centre's offset from the pivot | m | 2 |
| $$\ell(\theta)$$ | Height of a housing's highest point above the pivot | m | 2 |
| $$\rho(\theta)$$ | Outward coordinate of a housing's innermost point from the pivot | m | 2 |
| $$e$$ | Effective abduction to trunk clearance | m | 2 |
| $$z_p, z_a$$ | Heights of the pivot and of the abduction centre | m | 3.5 |
| $$x_h, x_k$$ | x coordinates of the hip and knee centres | m | 2 |
| $$y_{hip}$$ | Hip offset from the abduction body along y | m | 3.1 |
| $$W_h, W_k$$ | Trunk width demanded by the hip and by the knee sweep | m | 3.4 |
