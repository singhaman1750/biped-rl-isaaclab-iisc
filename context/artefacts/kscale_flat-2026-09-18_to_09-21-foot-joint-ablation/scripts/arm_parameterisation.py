"""Every parameter table of section 21 of plans/kscale_gait_shaping.md, derived rather than quoted.

Section 15 converts each arm of the ablation into a sub-section carrying its code, its design
decisions and its parameters.  Four of those parameter tables are arithmetic on the configuration
rather than measurements of a run, so they are reproduced here from the declared constants alone
and the program needs no dump and no simulator.  Run it from this directory with no arguments and
compare its output against the tables of sections 15.4, 15.5, 15.9 and 15.11.

The constants below are transcribed from the tree of 2026-09-23 and each carries its source.
"""
import numpy as np

# kscale_identified_cfg.py, the six actuator groups, with the URDF travel of kscale.urdf and the
# nominal pose of kscale_identified_cfg.py:197.  The range is that of the RIGHT leg, the left
# being its mirror at the hip roll and the hip pitch and identical elsewhere.
#
# Refreshed 2026-09-23 when arm 7 moved the three sagittal nominals and arm 6 halved the two
# ankle scales.  The nominals are those of kscale_identified_cfg.py:221, :227 and :229 and the
# scales those of kscale_base_env_cfg.py:214, so the tables below describe the tree as it stands
# and NOT the arm 4 reference, whose ankle scale of 0.1619 is carried separately for comparison.
GROUPS = {
    # name          K      tau   scale   lower      upper     nominal
    "hip_pitch": (200.0,  84.0, 0.4000, -2.21657,  1.04720, -0.34907),
    "hip_yaw":   (100.0,  42.0, 0.4000, -1.57080,  1.57080,  0.0),
    "knee":      (300.0,  84.0, 0.2667,  0.0,      2.70526,  0.87266),
    "hip_roll":  (250.0,  42.0, 0.1600, -2.26893,  0.20944,  0.042951),
    "foot_pitch":(100.0,  17.0, 0.0800, -0.87267,  0.52360, -0.52360),
    "foot_roll": (100.0,  17.0, 0.0800, -0.26180,  0.26180,  0.041254),
}
ARM3_ANKLE_SCALE = 0.1619             # what the sizing rule gives, before arm 6 halved it
OLD_SCALE = 0.4                       # kscale_base_env_cfg.py:193 before 2026-09-22
SOFT_FACTOR = 0.9                     # kscale_identified_cfg.py:241
CLIP_INSET = 0.02                     # the inset section 21.10 proposes
ANKLE_VEL_LIMIT = 37.699              # kscale_identified_cfg.py:153 and :169


def section_15_4():
    print("section 21.4, the per joint action scale\n")
    print("  every column is taken at the nominal pose the tree now holds, so the single scale")
    print("  column is the counterfactual of that pose under one scale of 0.4 and not the state")
    print("  of any run this document reports, the runs having carried the shallower pose too\n")
    print(f"{'group':11}{'K':>7}{'tau':>7}{'tau/K':>8}{'arm 4':>8}{'tree':>8}{'a_sat':>8}"
          f"{'a_stop, 0.4':>13}{'a_stop, arm 4':>15}{'a_stop, tree':>14}")
    sat = []
    for g, (K, tau, sc, lo, hi, nom) in GROUPS.items():
        d = tau / K
        arm3 = ARM3_ANKLE_SCALE if g in ("foot_pitch", "foot_roll") else sc
        near = min(abs(nom - lo), abs(hi - nom))
        sat.append(d / sc)
        print(f"{g:11}{K:>7.0f}{tau:>7.1f}{d:>8.4f}{arm3:>8.4f}{sc:>8.4f}{d/sc:>8.3f}"
              f"{near/OLD_SCALE:>13.3f}{near/arm3:>15.3f}{near/sc:>14.3f}")
    old = [GROUPS[g][1] / GROUPS[g][0] / OLD_SCALE for g in GROUPS]
    print(f"\naction at saturation, spread at one scale of 0.4 {max(old)/min(old):.2f} to one,"
          f" under arm 4 alone 1.0002, under the tree {max(sat)/min(sat):.4f} to one")
    print("arm 4 equalises that quantity and arm 6 deliberately breaks it again at the ankles\n")


def section_15_5():
    print("section 21.5, the lateral sway kernel at the merged width and at the tree's\n")
    a, w, n = 0.03, 5.0, 400000
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cases = {
        "perfect tracking":      np.zeros_like(th),
        "half amplitude, phase": -0.5 * a * np.sin(th),
        "no sway at all":        -a * np.sin(th),
        "a quarter cycle late":  a * np.sin(th - np.pi / 2) - a * np.sin(th),
        "anti phase":            -2 * a * np.sin(th),
    }
    print(f"{'torso behaviour':24}{'std 0.03':>12}{'std 0.01':>12}")
    vals = {}
    for name, e in cases.items():
        row = [w * np.mean(np.exp(-e ** 2 / s ** 2)) for s in (0.03, 0.01)]
        vals[name] = row
        print(f"{name:24}{row[0]:>12.3f}{row[1]:>12.3f}")
    d = [vals["perfect tracking"][i] - vals["no sway at all"][i] for i in (0, 1)]
    print(f"{'discrimination':24}{d[0]:>12.3f}{d[1]:>12.3f}")
    for i, s in enumerate((0.03, 0.01)):
        e = -a * np.sin(th)
        gr = np.abs(-2 * e / s ** 2 * np.exp(-e ** 2 / s ** 2)) * w
        tag = f"std {s}"
        print(f"  {tag:10} mean grad {gr.mean():8.1f}/m  peak {gr.max():8.1f}/m"
              f"  stride fraction above a tenth of peak {100*np.mean(gr > 0.1*gr.max()):5.1f} pct")
    print("\nthe narrowing does NOT collapse the kernel, because the target moves past a still")
    print("torso twice per stride and so samples the steep region every cycle regardless\n")


def section_15_9():
    print("section 21.10, the commanded target clip, against the soft limits it must not silence\n")
    print(f"{'joint':22}{'hard lo':>10}{'hard hi':>10}{'soft lo':>10}{'soft hi':>10}"
          f"{'clip lo':>10}{'clip hi':>10}")
    mirrored = {"hip_roll", "hip_pitch"}
    for g, (K, tau, sc, lo, hi, nom) in GROUPS.items():
        for side, (l, h) in (("right", (lo, hi)),
                             ("left", (-hi, -lo) if g in mirrored else (lo, hi))):
            mid, half = (l + h) / 2, (h - l) / 2 * SOFT_FACTOR
            print(f"{side + '_' + g:22}{l:>10.5f}{h:>10.5f}{mid-half:>10.5f}{mid+half:>10.5f}"
                  f"{l+CLIP_INSET:>10.5f}{h-CLIP_INSET:>10.5f}")
    print("\nthe clip must sit OUTSIDE the soft limit or pen_joint_pos_limits is silenced and the")
    print("arm loses its own measurement, which the inset of 0.02 rad satisfies at every joint\n")


def section_15_11():
    print("section 21.21, the corner velocity the saturation effort sets\n")
    print(f"{'configuration':16}{'ceiling':>9}{'saturation':>12}{'ratio':>8}"
          f"{'corner':>9}{'tau at 25 rad/s':>17}")
    for name, eff, sat in (("arm 1", 11.9, 17.0), ("arm 2, as run", 17.0, 17.0),
                           ("arm 3, as run", 34.0, 34.0), ("arm 19", 17.0, 24.31)):
        corner = ANKLE_VEL_LIMIT * max(0.0, 1.0 - eff / sat)
        tau25 = min(sat * (1.0 - 25.0 / ANKLE_VEL_LIMIT), eff)
        print(f"{name:16}{eff:>9.2f}{sat:>12.2f}{eff/sat:>8.3f}{corner:>9.2f}{tau25:>17.2f}")
    print("\nthe corner velocity is a function of the RATIO alone, so arm 19 restores the 0.700")
    print("that arm 1 carried by accident rather than any particular saturation value\n")


if __name__ == "__main__":
    section_15_4()
    section_15_5()
    section_15_9()
    section_15_11()
