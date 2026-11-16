"""MJCF-based robot design generator for co-optimisation.

Every design in a population is a document emitted by the vendored parametric
generator at ``co_optimisation/runners/design_generator``, converted to
Universal Scene Description by :class:`isaaclab.sim.converters.MjcfConverter`
and spawned through the multi-asset pathway.  The generator is the single
source of truth: it owns the geometry, the mass model and the actuator
catalog, and it hands this module the three things the runner consumes, a
Universal Scene Description path per individual, the actuator parameters that
cannot be expressed in the asset, and the absolute link geometry the in-place
editor in ``co_optimisation/utils/update.py`` writes.

``param_ranges`` maps a parameter name to ``(type, (lo, hi))``, the type being
``"scale"`` (a multiplier on a baseline field) or ``"offset"`` (an addition, in
the field's own unit). See :data:`DEFAULT_PARAM_RANGES` for every key, its type
and its range.  A bare ``(lo, hi)`` is read as ``("scale", (lo, hi))``, the only
type the untyped form ever carried, so a caller written before the types
existed is unchanged.

- ``"thigh_length_scale"``, ``"shank_length_scale"`` -- scales on
  ``DesignSpec.thigh_length`` and ``shank_length``, (0.75, 1.25).
- ``"thigh_bend_offset"``, ``"knee_bend_offset"`` -- the nominal pose, radians
  about +y, (-0.10, 0.10) and (-0.25, 0.25), the first a lean about the hip
  angle that :func:`solve_nominal_pose` finds for the design's lengths and
  knee.  Checked by :func:`check_nominal_pose` before the specification is
  emitted, the biped's ankle offset being computed so that the sole stays flat.
- Per-group stiffness and damping scales, applied post-spawn via
  ``apply_actuator_params``.  Ranges are the symmetric equivalents of the
  runtime domain-randomisation absolute ranges, an asymmetric absolute range
  being resolved by taking the smaller half: ``"abad_stiffness_scale"``
  (0.909, 1.091), ``"abad_damping_scale"`` (0.889, 1.111),
  ``"hip_stiffness_scale"`` (0.875, 1.125), ``"hip_damping_scale"``
  (0.846, 1.154), ``"knee_stiffness_scale"`` (0.833, 1.167),
  ``"knee_damping_scale"`` (0.750, 1.250), ``"ankle_stiffness_scale"`` and
  ``"ankle_damping_scale"`` (0.800, 1.200), biped only.

Six actuator scales are retired, ``joint_effort_scale``,
``velocity_limit_scale``, ``saturation_effort_scale``, ``armature_scale``,
``friction_static_scale`` and ``friction_dynamic_scale``.  No launcher ever
passed one and a search never read one, so a caller that names one now raises
rather than being silently ignored.

Three keys of the URDF-era generator are gone.  ``link_mass_scale`` has no
counterpart because a link's mass is ``lambda * length`` under the generator's
own model, so it moves with the length scales rather than independently;
``actuator_radius_scale`` and ``actuator_length_scale`` have none because the
housing envelope is a property of the catalog entry the design picks, so it
moves with :class:`CatCMAESDesignGenerator`'s categorical variables instead.
"""

from __future__ import annotations

import math
import numpy as np
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import cma

# try:
from co_optimisation.utils.common import Population  # noqa: F401, re-exported for every importer of this module
from co_optimisation.utils.filters import make_filters
from design_generator import biped, quadruped

# except ImportError:
#     # Loaded by its path without the simulator, as co_optimisation/tests does, where the package
#     # __init__ would import the runner, so the sibling file is loaded by its path as well.
#     import importlib.util as _util
#
#     _spec = _util.spec_from_file_location("co_optimisation_utils_common", Path(__file__).resolve().parents[1] / "utils" / "common.py")
#     _common = _util.module_from_spec(_spec)
#     _spec.loader.exec_module(_common)
#     Population = _common.Population

# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


def _box_inertia(mass: float, half) -> tuple[float, float, float]:
    """Diagonal moment of inertia of a solid box about its own centre."""
    hx, hy, hz = (float(v) for v in half)
    return (
        mass * ((2 * hy) ** 2 + (2 * hz) ** 2) / 12.0,
        mass * ((2 * hx) ** 2 + (2 * hz) ** 2) / 12.0,
        mass * ((2 * hx) ** 2 + (2 * hy) ** 2) / 12.0,
    )


def _entry(half, pos, mass, child_joint=None, child_body=None, child_pos=None) -> dict:
    """One scalable link's absolute geometry, as ``update.py`` consumes it.

    ``half`` are MuJoCo half-extents, which is also the scale the Isaac Sim
    MuJoCo importer authors on the box prototype, so they are written through
    unchanged.  ``pos`` is the box centre in the link frame, which for these
    links is also the centre of mass.  ``child_pos`` is the child body's origin
    in the link frame, which is what the child joint's ``physics:localPos0``
    holds.
    """
    mass = float(mass)
    out = {
        "half_extents": [float(v) for v in half],
        "box_pos": [float(v) for v in pos],
        "mass": mass,
        "inertia": list(_box_inertia(mass, half)),
    }
    if child_joint is not None:
        out.update(child_joint=child_joint, child_body=child_body,
                   child_pos=[float(v) for v in child_pos])
    return out


def _housing(builder, actuator_mass, radius, half_length, parent=None) -> dict:
    """One actuator housing, a cylinder whose size and mass follow the actuator the design picks.

    ``radius`` and ``half_length`` are the MuJoCo cylinder size, which the importer
    authors as the radius and twice the height of a ``UsdGeomCylinder``
    (MjcfUsd.cpp:799-818). The inertia is the builder's own, about the cylinder's
    principal axes, which the document fixes. ``parent`` is ``(body, joint, pos)``,
    the housing's own origin in its parent body, which that joint's
    ``physics:localPos0`` holds.
    """
    i_perp, i_axial = builder.cyl_inertia(float(actuator_mass), radius, 2.0 * half_length)
    out = {
        "radius": float(radius),
        "half_length": float(half_length),
        "mass": float(actuator_mass),
        "inertia": [i_perp, i_perp, i_axial],
    }
    if parent is not None:
        body, joint, pos = parent
        out.update(parent_body=body, parent_joint=joint, parent_pos=[float(v) for v in pos])
    return out


def _quat_wxyz_y(rad: float) -> list[float]:
    return [math.cos(rad / 2.0), 0.0, math.sin(rad / 2.0), 0.0]


def _biped_links(spec) -> dict[str, dict]:
    b = biped.builder
    l1, l2 = float(spec.thigh_length), float(spec.shank_length)
    fr = b.pose_frames(spec)
    act_abad, act_hip, act_knee, act_ankle = spec.actuators()
    r_ankle, hl_ankle = b._housing_dims(act_ankle)
    out: dict[str, dict] = {}
    for side in b.LEG_SIDES:
        out[f"thigh_{side}_Link"] = _entry(
            half=(b.THIGH_BOX_HALF[0], b.THIGH_BOX_HALF[1],
                  b.LINK_BOX_LENGTH_FRAC * l1 / 2.0),
            pos=(0.0, 0.0, -l1 / 2.0),
            mass=b.LAMBDA_THIGH * l1,
            child_joint=f"knee_{side}_Joint",
            child_body=f"shank_{side}_Link",
            child_pos=(0.0, 0.0, -l1),
        )
        out[f"shank_{side}_Link"] = _entry(
            half=(b.SHANK_BOX_HALF[0], b.SHANK_BOX_HALF[1],
                  b.LINK_BOX_LENGTH_FRAC * l2 / 2.0),
            pos=(0.0, 0.0, -l2 / 2.0),
            mass=b.LAMBDA_SHANK * l2,
            # ankle_{S}_act carries no joint, so the importer names its fixed
            # joint after the body (MjcfImporter.cpp:1353).
            child_joint=f"ankle_{side}_act",
            child_body=f"ankle_{side}_act",
            child_pos=(spec.clearance_shank2ankleAct_x,
                       spec.clearance_shank2ankleAct_y,
                       -l2 + spec.clearance_shank2ankleAct_z),
        )
        # Keyed by the joint each body quat sits on, a body with no joint of its
        # own hanging on a fixed joint named after it.
        out[f"thigh_{side}_Link"]["joint_rot"] = {
            f"thigh_{side}_Link": _quat_wxyz_y(fr["thigh"]),
            f"knee_{side}_Joint": _quat_wxyz_y(fr["shank"]),
        }
        out[f"shank_{side}_Link"]["joint_rot"] = {f"ankle_{side}_act": _quat_wxyz_y(fr["ankle"])}

        # The housings and every placement that follows their size, _cluster_layout's chain.
        lay = b._cluster_layout(side, spec, act_abad, act_hip, act_knee)
        out[f"abad_{side}_act"] = _housing(b, act_abad.mass, lay["r_abad"], lay["hl_abad"],
                                           ("base_Link", f"abad_{side}_act", lay["abad_pos"]))
        out[f"hip_{side}_act"] = _housing(b, act_hip.mass, lay["r_hip"], lay["hl_hip"],
                                          (f"abad_{side}_act", f"abad_{side}_Joint", lay["hip_pos_rel"]))
        out[f"knee_{side}_act"] = _housing(b, act_knee.mass, lay["r_knee"], lay["hl_knee"],
                                           (f"hip_{side}_act", f"hip_{side}_Joint", lay["knee_pos_rel"]))
        out[f"thigh_{side}_Link"].update(parent_body=f"knee_{side}_act", parent_joint=f"thigh_{side}_Link",
                                          parent_pos=[float(v) for v in lay["thigh_pos_rel"]])
        out[f"ankle_{side}_act"] = _housing(b, act_ankle.mass, r_ankle, hl_ankle)
        # The sole hangs below the ankle housing, so its box and centre of mass follow its radius.
        out[f"foot_{side}_Link"] = _entry(
            half=(spec.foot_length / 2.0, b.FOOT_BOX_HALF[0], b.FOOT_BOX_HALF[1]),
            pos=(spec.clearance_ankleAct2foot_x, 0.0,
                 -(r_ankle + b.FOOT_BOX_HALF[1] + spec.clearance_ankleAct2foot_z)),
            mass=b.LAMBDA_FOOT * spec.foot_length,
        )
    return out


def _quadruped_links(spec) -> dict[str, dict]:
    q = quadruped.builder
    l1, l2 = float(spec.thigh_length), float(spec.shank_length)
    fr = q.pose_frames(spec)
    act_abad, act_hip, act_knee = spec.actuators()
    out: dict[str, dict] = {}
    for leg in q.LEG_NAMES:
        out[f"thigh_{leg}_Link"] = _entry(
            half=(q.THIGH_BOX_HALF[0], q.THIGH_BOX_HALF[1],
                  l1 * q.VISUAL_LINK_SCALE / 2.0),
            pos=(0.0, 0.0, -l1 / 2.0),
            mass=q.LAMBDA_THIGH * l1,
            child_joint=f"{leg}_calf_joint",
            child_body=f"shank_{leg}_Link",
            child_pos=(0.0, 0.0, -l1),
        )
        out[f"shank_{leg}_Link"] = _entry(
            half=(q.SHANK_BOX_HALF[0], q.SHANK_BOX_HALF[1],
                  l2 * q.VISUAL_LINK_SCALE / 2.0),
            pos=(0.0, 0.0, -l2 / 2.0),
            mass=q.LAMBDA_SHANK * l2,
            child_joint=f"foot_{leg}_Link",
            child_body=f"foot_{leg}_Link",
            child_pos=(spec.clearance_shank2foot_x,
                       spec.clearance_shank2foot_y,
                       -l2 + spec.clearance_shank2foot_z),
        )
        out[f"thigh_{leg}_Link"]["joint_rot"] = {
            f"thigh_{leg}_Link": _quat_wxyz_y(fr["thigh"]),
            f"{leg}_calf_joint": _quat_wxyz_y(fr["shank"]),
        }
        # Fixed housing sizes (quadruped/builder.py), so a pick moves the masses alone.
        out[f"abad_{leg}_act"] = _housing(q, act_abad.mass, q.ABD_RADIUS, q.ABD_HALFLEN)
        out[f"hip_{leg}_act"] = _housing(q, act_hip.mass, q.HIP_RADIUS, q.HIP_HALFLEN)
        out[f"knee_{leg}_act"] = _housing(q, act_knee.mass, q.KNEE_RADIUS, q.KNEE_HALFLEN)
    return out


# ---------------------------------------------------------------------------
# Nominal pose: the sagittal chain, the solve and the checks
# ---------------------------------------------------------------------------

def _rot_y(angle: float, x: float, z: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return x * c + z * s, -x * s + z * c


def _quadruped_chain(spec, dh: float, dk: float, da: float = 0.0) -> tuple[float, float, float]:
    """(foot x ahead of the hip pitch axis, foot body depth below the root, sole below the foot body).

    The hip pitch axis is the origin of knee_{leg}_act, which Rules 1 to 4 put at
    the trunk's mid-height, so only the z clearances move it.
    """
    q = quadruped.builder
    l1, l2 = float(spec.thigh_length), float(spec.shank_length)
    a_thigh = q.THIGH_BEND_RAD + dh
    a_shank = a_thigh + q.SHANK_BEND_RAD + dk
    x = spec.clearance_kneeAct2thigh_x
    z = (spec.clearance_trunk2abadAct_z + spec.clearance_abadAct2hipAct_z
         + spec.clearance_hipAct2kneeAct_z + spec.clearance_kneeAct2thigh_z)
    for angle, vx, vz in ((a_thigh, 0.0, -l1),
                          (a_shank, spec.clearance_shank2foot_x, -l2 + spec.clearance_shank2foot_z)):
        dx, dz = _rot_y(angle, vx, vz)
        x, z = x + dx, z + dz
    return x, -z, q.FOOT_RADIUS


def _biped_chain(spec, dh: float, dk: float, da: float) -> tuple[float, float, float]:
    """(ankle joint x ahead of the hip pitch axis, its depth below the root, sole below it).

    The hip pitch axis is the origin of knee_{S}_act, whose height _cluster_layout
    takes from the abduction housing's radius, and the sole's depth follows the
    ankle housing's, so both move with the actuators a categorical search picks.
    """
    b = biped.builder
    act_abad, act_hip, act_knee, act_ankle = spec.actuators()
    lay = b._cluster_layout("R", spec, act_abad, act_hip, act_knee)
    l1, l2 = float(spec.thigh_length), float(spec.shank_length)
    a_thigh = math.radians(b.THIGH_BEND_DEG + 180.0) + dh
    a_shank = math.radians(b.SHANK_BEND_DEG) + dh + dk
    a_foot = a_shank + math.radians(b.ANKLE_BEND_DEG) + da
    x = float(lay["thigh_pos_rel"][0])
    z = float(lay["abad_pos"][2] + lay["hip_pos_rel"][2] + lay["knee_pos_rel"][2] + lay["thigh_pos_rel"][2])
    for angle, vx, vz in ((a_thigh, 0.0, -l1),
                          (a_shank, spec.clearance_shank2ankleAct_x, -l2 + spec.clearance_shank2ankleAct_z)):
        dx, dz = _rot_y(angle, vx, vz)
        x, z = x + dx, z + dz
    r_ankle, _ = b._housing_dims(act_ankle)
    bottom = -(r_ankle + 2.0 * b.FOOT_BOX_HALF[1] + spec.clearance_ankleAct2foot_z)
    _, sole_z = _rot_y(a_foot, spec.clearance_ankleAct2foot_x, bottom)
    sole = -sole_z + 0.5 * spec.foot_length * abs(math.sin(a_foot))   # lowest corner of the sole
    return x, -z, sole


def _bisect(f, lo: float, hi: float, iterations: int = 60) -> float:
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0.0:
        return lo
    if (f_lo > 0.0) == (f_hi > 0.0):
        raise ValueError(f"no hip angle in [{lo}, {hi}] rad puts the foot under the hip")
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if (f_mid > 0.0) == (f_lo > 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_nominal_pose(family: "Family", spec, dx_target: float) -> dict[str, float]:
    """The pose offsets of *spec*, whose thigh offset is read as a lean about the solved hip angle.

    The hip angle that puts the foot *dx_target* ahead of the hip pitch axis is
    found on the family's chain for the design's lengths and knee, and the
    biped's ankle keeps the sole at the donor's tilt. The knee is rounded to
    three decimals before the solve and the hip after it, so only the hip's
    rounding remains in the foot's position.
    """
    dk = round(spec.knee_bend_offset, 3)
    hip = _bisect(lambda h: family.chain(spec, h, dk, -(h + dk))[0] - dx_target, -1.0, 1.0)
    dh = round(hip + spec.thigh_bend_offset, 3)
    offsets = {"thigh_bend_offset": dh, "knee_bend_offset": dk}
    if family.has_ankle:
        offsets["ankle_bend_offset"] = -(dh + dk)
    return offsets


def check_nominal_pose(family: "Family", spec, dx_target: float, lean_max: float) -> None:
    """Refuse a design whose nominal pose breaks the constraints. Raises ValueError.

    The foot under the hip to a tolerance set by the largest lean a draw may
    take, the sole flat, the standing height inside the family's corridor, and
    every pitch joint's nominal inside its hardware range narrowed by the
    asset's soft_joint_pos_limit_factor of 0.9.
    """
    dh, dk = spec.thigh_bend_offset, spec.knee_bend_offset
    da = getattr(spec, "ankle_bend_offset", 0.0)
    if family.has_ankle and abs(dh + dk + da) > 1e-9:
        raise ValueError(f"sole not flat, offsets sum to {dh + dk + da:.3e} rad")
    dx, depth, sole = family.chain(spec, dh, dk, da)
    # 5e-4 rad is the rounding of the hip offset to three decimals.
    tol = 1.05 * (spec.thigh_length + spec.shank_length) * math.sin(lean_max + 5e-4)
    if abs(dx - dx_target) > tol:
        raise ValueError(f"foot {1e3 * (dx - dx_target):+.1f} mm from under the hip, tolerance {1e3 * tol:.1f} mm")
    lo, hi = family.height_corridor
    if not lo < depth + sole < hi:
        raise ValueError(f"standing height {depth + sole:.4f} m outside ({lo}, {hi})")
    for name, value, (r_lo, r_hi) in family.pose_ranges(dh, dk, da):
        mid, half = 0.5 * (r_lo + r_hi), 0.45 * (r_hi - r_lo)
        if not mid - half <= value <= mid + half:
            raise ValueError(f"{name} nominal {value:+.4f} rad outside the soft hardware range")


class Family:
    """Everything this module needs to know about one robot family."""

    def __init__(self, name, module, groups, spec_fields, root_body, links,
                 chain=None, has_ankle=False, height_corridor=(0.0, float("inf")),
                 pose_ranges=None, foot_bodies=(), hip_bodies=(), pitch_joints=()):
        self.name = name
        self.module = module
        self.groups = groups
        self.spec_fields = spec_fields      # actuator group -> DesignSpec field
        self.root_body = root_body
        self.links = links
        # chain(spec, dh, dk, da) -> (dx, depth, sole) is the sagittal chain of the
        # emitted document, the bodies and joints are what the runner measures.
        self.chain = chain
        self.has_ankle = has_ankle
        self.height_corridor = height_corridor
        self.pose_ranges = pose_ranges
        self.foot_bodies = tuple(foot_bodies)
        self.hip_bodies = tuple(hip_bodies)
        self.pitch_joints = list(pitch_joints)

    def baseline(self, path=None):
        return (self.module.generate.load_design_spec(path) if path is not None
                else self.module.generate.load_design_spec())

    def build_xml(self, spec, **builder_kwargs) -> str:
        return self.module.builder.build_xml(spec, for_isaaclab=True, **builder_kwargs)

    def catalog(self) -> list:
        """The simulation-complete entries, the only ones a design may pick."""
        return self.module.actuator.sim_actuator_catalog()

    def full_catalog(self) -> list:
        return list(self.module.actuator.ACTUATOR_CATALOG)

    def nominal(self, spec) -> dict:
        dh, dk = spec.thigh_bend_offset, spec.knee_bend_offset
        da = getattr(spec, "ankle_bend_offset", 0.0)
        dx, depth, sole = self.chain(spec, dh, dk, da)
        return {"offsets": [dh, dk, da], "dx": dx, "depth": depth, "sole": sole, "height": depth + sole}


def _quadruped_pose_ranges(dh, dk, da):
    # Absolute joint angles, the hardware range being about the folded pose.
    q = quadruped.builder
    return (("hip", q.THIGH_BEND_RAD + dh, q.HIP_RANGE),
            ("knee", q.SHANK_BEND_RAD + dk, q.KNEE_RANGE))


def _biped_pose_ranges(dh, dk, da):
    # Joint sign, the axes being (0, -1, 0), the hardware range about the donor fold.
    b = biped.builder
    return (("hip", -dh, b.HIP_RANGE), ("knee", -dk, b.KNEE_RANGE), ("ankle", -da, b.ANKLE_RANGE))


FAMILIES: dict[str, Family] = {
    "biped": Family(
        "biped", biped,
        groups=("abad", "hip", "knee", "ankle"),
        spec_fields={"abad": "act_abduction", "hip": "act_hip",
                     "knee": "act_knee", "ankle": "act_ankle"},
        root_body="base_Link", links=_biped_links,
        chain=_biped_chain, has_ankle=True, height_corridor=(0.55, 1.05),
        pose_ranges=_biped_pose_ranges,
        foot_bodies=("foot_R_Link", "foot_L_Link"),
        hip_bodies=("knee_R_act", "knee_L_act"),
        pitch_joints=["hip_.*_Joint", "knee_.*_Joint", "ankle_.*_Joint"],
    ),
    "quadruped": Family(
        "quadruped", quadruped,
        groups=("abad", "hip", "knee"),
        spec_fields={"abad": "act_abduction", "hip": "act_hip", "knee": "act_knee"},
        root_body="trunk", links=_quadruped_links,
        chain=_quadruped_chain, has_ankle=False, height_corridor=(0.18, 0.45),
        pose_ranges=_quadruped_pose_ranges,
        foot_bodies=tuple(f"foot_{leg}_Link" for leg in quadruped.builder.LEG_NAMES),
        hip_bodies=tuple(f"knee_{leg}_act" for leg in quadruped.builder.LEG_NAMES),
        pitch_joints=[".._thigh_joint", ".._calf_joint"],
    ),
}

# ---------------------------------------------------------------------------
# Search ranges
# ---------------------------------------------------------------------------

ParamRange = tuple[str, tuple[float, float]]
PARAM_TYPES = ("scale", "offset")

DEFAULT_PARAM_RANGES: dict[str, ParamRange] = {
    "thigh_length_scale": ("scale", (0.75, 1.25)),
    "shank_length_scale": ("scale", (0.75, 1.25)),
    "thigh_bend_offset": ("offset", (-0.10, 0.10)),     # rad, a lean about the solved hip angle
    "knee_bend_offset": ("offset", (-0.25, 0.25)),      # rad
    "abad_stiffness_scale": ("scale", (0.909, 1.091)),
    "abad_damping_scale": ("scale", (0.889, 1.111)),
    "hip_stiffness_scale": ("scale", (0.875, 1.125)),
    "hip_damping_scale": ("scale", (0.846, 1.154)),
    "knee_stiffness_scale": ("scale", (0.833, 1.167)),
    "knee_damping_scale": ("scale", (0.750, 1.250)),
    "ankle_stiffness_scale": ("scale", (0.800, 1.200)),
    "ankle_damping_scale": ("scale", (0.800, 1.200)),
}

def actuator_distance(a, b) -> float:
    """The distance of two catalogue actuators in log peak torque, log peak speed and log mass.

    The metric scripts/tools/fill_actuator_sim_params.py fills the catalogues by, see
    plans/design_filtering.md section 7.1.
    """
    return math.sqrt(sum(math.log(getattr(a, k) / getattr(b, k)) ** 2 for k in ("peak_torque", "peak_speed", "mass")))

# Names rather than a second range structure, so a caller's override of a range reaches CMA-ES.
CMAES_PARAM_KEYS: tuple[str, ...] = (
    "thigh_length_scale", "shank_length_scale", "thigh_bend_offset", "knee_bend_offset",
)

# The DesignSpec field each spec-bound parameter writes. A parameter absent here
# (the stiffness and damping scales) is consumed by _actuator_params instead.
SPEC_PARAM_FIELDS: dict[str, str] = {
    "thigh_length_scale": "thigh_length",
    "shank_length_scale": "shank_length",
    "thigh_bend_offset": "thigh_bend_offset",
    "knee_bend_offset": "knee_bend_offset",
}

DEPRECATED_PARAM_KEYS = (
    "joint_effort_scale", "velocity_limit_scale", "saturation_effort_scale",
    "armature_scale", "friction_static_scale", "friction_dynamic_scale",
)


def _typed_ranges(param_ranges: dict | None) -> dict[str, ParamRange]:
    out: dict[str, ParamRange] = {**DEFAULT_PARAM_RANGES}
    for name, value in (param_ranges or {}).items():
        if name in DEPRECATED_PARAM_KEYS:
            raise ValueError(f"parameter {name!r} is retired, see usd_generator's module docstring")
        if name not in DEFAULT_PARAM_RANGES:
            raise ValueError(f"unknown parameter {name!r}, known {sorted(DEFAULT_PARAM_RANGES)}")
        if isinstance(value[0], str):
            kind, (lo, hi) = value
        else:
            kind, (lo, hi) = "scale", value       # the untyped form, which only ever carried scales
        if kind not in PARAM_TYPES:
            raise ValueError(f"parameter {name!r} has unknown type {kind!r}, expected one of {PARAM_TYPES}")
        out[name] = (kind, (float(lo), float(hi)))
    return out


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class DesignGeneratorBase(ABC):
    """Abstract base class for design generators.

    Provides all shared specification, emission and conversion machinery.
    Subclasses implement :meth:`_generate_individual` to choose one
    individual's :class:`DesignSpec`.

    Args:
        family: ``"biped"`` or ``"quadruped"``, selecting which of the vendored
            generator's two packages supplies the geometry and the catalog.
        num_individuals: Number of designs per generation.
        param_ranges: Optional dict overriding entries in
            :data:`DEFAULT_PARAM_RANGES`.  Only the keys given are overridden.
        output_dir: Directory for the emitted documents and their conversions.
        variable_params_path: Optional path to a ``variable_params.json`` that
            supplies the baseline specification.  Defaults to the one the
            family's own package ships, which reproduces the donor robot.
        filters: Optional names of the filters every population must pass, as
            ``co_optimisation.utils.filters.make_filters`` takes them.  None or
            empty, the default, leaves the class attribute ``filters`` empty and
            filters nothing.
        filter_params: Optional keyword arguments of each filter, keyed as
            *filters* names it.
    """

    def __init__(
        self,
        family: str,
        num_individuals: int,
        param_ranges: dict[str, ParamRange] | None = None,
        output_dir: str = "/tmp/copt_usds",
        variable_params_path: str | None = None,
        filters: list[str] | None = None,
        filter_params: dict | None = None,
    ) -> None:
        if family not in FAMILIES:
            raise ValueError(
                f"Unknown robot family {family!r}. Known families: {sorted(FAMILIES)}."
            )
        self.family = FAMILIES[family]
        self.num_individuals = num_individuals
        self.output_dir = output_dir
        self.param_ranges: dict[str, ParamRange] = _typed_ranges(param_ranges)
        # The search is centred on the initial design, so every range must contain it.
        for key, (_, (lo, hi)) in self.param_ranges.items():
            if not lo <= self._initial_value(key) <= hi:
                raise ValueError(
                    f"{key} range ({lo}, {hi}) excludes the initial design's {self._initial_value(key)}."
                )
        self.baseline = self.family.baseline(variable_params_path)
        self._dx_target = self.family.chain(self.baseline, 0.0, 0.0, 0.0)[0]
        # The last specification that emitted without raising, substituted for
        # any later specification the builder rejects. See _build_xml.
        self._last_feasible = self.baseline
        self._last_emitted = self.baseline
        self._current_parameters: dict[str, float] | None = None
        # The positions of the population being filtered that some filter still refuses, and the redraw
        # in progress, 0 outside filter_population so that every draw keeps its former seed.
        self._designs_not_compliant: list[int] = []
        self._filter_attempt = 0
        self.filter_history: list[dict] = []
        if filters:
            self.filters = make_filters(list(filters), family, filter_params)

    # Extra keyword arguments of the family's build_xml, set by the caller after
    # construction, for example {"shank_foot_gap": 0.001} for the quadruped. Empty,
    # the default, emits every document as before.
    builder_kwargs: dict = {}

    filters: list = []
    filter_max_rounds: int = 100

    # ---- abstract design hooks ----------------------------------------------

    @abstractmethod
    def _generate_individual(self, generation: int, idx: int):
        """Return ``(spec, actuator_params)`` for one individual."""
        ...

    @abstractmethod
    def generate_population(self, generation: int) -> Population | None:
        """Generate a new population for *generation*."""
        ...

    def update_with_fitness(self, fitness: list[float]) -> None:
        """Optionally update internal state using per-individual fitness scores.

        The default implementation is a no-op (random search).
        """
        pass

    def sample_batch(self) -> None:
        pass

    def get_state(self) -> dict:
        return {}

    def load_state(self, state: dict) -> None:
        pass

    def search_statistics(self) -> dict | None:
        """The search distribution in the units of ``param_ranges``, or None where there is none."""
        return None

    def optimised_design(self) -> tuple[dict[str, float], dict[str, str] | None] | None:
        """The parameters and actuator picks the search recommends, or None where there is no search."""
        return None

    def emit_design(self, parameters: dict[str, float], picks: dict[str, str] | None, path: str) -> bool:
        """Write the MJCF of one design to *path*, False where the builder refused it for the last feasible one."""
        self._current_parameters = parameters
        spec = self._spec_from_parameters(parameters, picks)
        xml = self._build_xml(spec)
        with open(path, "w") as fh:
            fh.write(xml)
        return self._last_emitted is spec

    # ---- shared population pipeline -----------------------------------------

    def _build_population(self, generation: int, indices, individual=None, filtered: bool = True) -> Population:
        """Build generation *generation* for the individual indices *indices*.

        *individual*, a callable of (generation, idx) returning (spec, actuator_params), takes
        the place of _generate_individual where given. None, the default, builds as before.
        Every document is emitted, then filtered where *filtered* and filters are set, and only
        then converted, so that a refused design never costs a conversion.
        """
        generate = individual or self._generate_individual
        indices = list(indices)
        rows = [self._individual(generation, idx, generate) for idx in indices]
        population = Population(*([row[key] for row in rows] for key in (
            "usd_file", "actuator_params", "link_length_params")), self.family.root_body,
            *([row[key] for row in rows] for key in ("spec", "nominal", "parameters", "xml_file")))
        if filtered and self.filters:
            self.filter_population(generation, population, indices, generate)
        usd_files = population.get_usd_files()
        for k, idx in enumerate(indices):
            usd_files[k] = self._generate_individual_usd(population.get_xml_files()[k], idx)
        return population

    def _individual(self, generation: int, idx: int, generate) -> dict:
        """Draw and emit one individual, every record of it but its conversion."""
        spec, act = generate(generation, idx)
        xml_path = self._generate_individual_mjcf(generation, idx, spec)
        # The drawn design, whose genotype CMA-ES is told the fitness of even when refused.
        parameters = {**self._design_record(spec), "feasible": self._last_emitted is spec}
        if self._last_emitted is not spec:
            # _build_xml refused the design and wrote the last feasible one.
            spec = self._last_emitted
            act = self._actuator_params(spec, self._current_parameters)
        # The design actually emitted and trained, its geometry, pose, mass and actuators.
        with open(xml_path) as fh:
            parameters.update(self.design_record(spec, act, fh.read()))
        return {
            "usd_file": None, "actuator_params": act, "link_length_params": self.family.links(spec),
            "spec": spec, "nominal": self.family.nominal(spec), "parameters": parameters, "xml_file": xml_path,
        }

    # ---- filtering -------------------------------------------------------------

    def filter_population(self, generation: int, population: Population, indices: list[int], generate) -> None:
        """Redraw every design of *population* some filter refuses, until every filter passes all of them.

        _designs_not_compliant begins as every position of *population*, each round leaving the positions
        some filter refused, which are redrawn under a fresh seed or a fresh sample of the search
        distribution, _redraw, and filtered again. *indices* are the individual indices of the positions.
        """
        self._designs_not_compliant = list(range(len(indices)))
        record = {"generation": generation, "designs": len(indices), "rounds": 0, "redrawn": 0, "refused": {}}
        try:
            while self._designs_not_compliant:
                pending = self._designs_not_compliant
                xml_files = [population.get_xml_files()[k] for k in pending]
                refused: set[int] = set()
                for f in self.filters:
                    hits = f.filter(xml_files)
                    record["refused"][f.name] = record["refused"].get(f.name, 0) + len(hits)
                    refused.update(hits)
                self._designs_not_compliant = [pending[i] for i in sorted(refused)]
                if not self._designs_not_compliant:
                    break
                if record["rounds"] == self.filter_max_rounds:
                    print(f"[filter] generation {generation}: {len(self._designs_not_compliant)} designs still "
                          f"refused after {self.filter_max_rounds} rounds, kept as drawn")
                    break
                record["rounds"] += 1
                record["redrawn"] += len(self._designs_not_compliant)
                self._filter_attempt = record["rounds"]
                for k in self._designs_not_compliant:
                    self._redraw(indices[k])
                    population.set_individual(k, self._individual(generation, indices[k], generate))
        finally:
            self._filter_attempt = 0
        record["kept_refused"] = len(self._designs_not_compliant)
        self.filter_history.append(record)
        # Every filter's per design scores, for the design logs, on the population as it now stands.
        rows = population.get_parameters()
        for f in self.filters:
            for name, values in f.scores(population.get_xml_files()).items():
                for k, value in enumerate(values):
                    rows[k][name] = float(value)
        print(f"[filter] generation {generation}: {record['redrawn']} redraws over {record['rounds']} rounds, "
              f"refusals {record['refused']}")

    def _redraw(self, idx: int) -> None:
        """Replace the draw behind individual *idx* before it is generated again. The seeded draws need
        nothing, _rng_seed folding in the round, so only a search distribution's sample is replaced."""
        pass

    def _rng_seed(self, generation: int, idx: int):
        """The seed of individual *idx*'s draw, unchanged outside a redraw, the round folded in during one."""
        seed = generation * 10000 + idx
        return seed if self._filter_attempt == 0 else [seed, self._filter_attempt]

    # ---- parameter sampling --------------------------------------------------

    def _sample_parameter(self, rng: np.random.Generator) -> dict[str, float]:
        return {k: float(rng.uniform(lo, hi)) for k, (_, (lo, hi)) in self.param_ranges.items()}

    def _initial_value(self, key: str) -> float:
        """The value of *key* that reproduces the initial design of the baseline MJCF."""
        return 1.0 if self.param_ranges[key][0] == "scale" else 0.0

    def _sample_parameter_v2(self, rng: np.random.Generator, scale: float) -> dict[str, float]:
        """Uniform draws from the box shrunk by *scale* towards the initial design.

        Each side covers *scale* of its own distance from the initial design, so an asymmetric
        box widens at an uneven pace and both sides reach their bounds together at scale 1.
        Before 2026-10-30 the box shrank towards its midpoint, which moved every early design
        off the initial one whenever the box was asymmetric.
        """
        parameters = {}
        for k, (_, (lo, hi)) in self.param_ranges.items():
            c = self._initial_value(k)
            lo_n = (lo - c) * scale + c
            hi_n = (hi - c) * scale + c
            parameters[k] = float(rng.uniform(lo_n, hi_n))
        return parameters

    # ---- specification assembly ---------------------------------------------

    def _spec_from_parameters(self, parameters: dict[str, float], picks: dict[str, str] | None = None):
        """A new DesignSpec from the baseline's fields, the parameters and the actuator picks.

        Not through ``DesignSpec.from_vector``, which drops every fixed-constant
        clearance back to its Python default. Scaled lengths are rounded to two
        decimals, the quantisation the URDF-era generator applied to the box
        extents, and the pose is solved for the rounded lengths.
        """
        fields = self.baseline.to_dict()
        for name, field in SPEC_PARAM_FIELDS.items():
            if name in parameters:
                kind, value = self.param_ranges[name][0], parameters[name]
                fields[field] = round(fields[field] * value, 2) if kind == "scale" else fields[field] + value
        for group, name in (picks or {}).items():
            fields[self.family.spec_fields[group]] = name
        design_spec = self.family.module.generate.DesignSpec
        offsets = solve_nominal_pose(self.family, design_spec.from_dict(fields), self._dx_target)
        return design_spec.from_dict({**fields, **offsets})

    def _actuator_params(self, spec, parameters: dict[str, float] | None = None) -> dict[str, dict]:
        """Simulator-side actuator parameters for every group of one design."""
        parameters = parameters or {}
        out: dict[str, dict] = {}
        for group, actuator in zip(self.family.groups, spec.actuators()):
            params = actuator.sim_params()
            for key in ("stiffness", "damping"):
                scale_key = f"{group}_{key}_scale"
                if key in params and scale_key in parameters:
                    params[key] *= parameters[scale_key]
            out[group] = {k: round(float(v), 6) for k, v in params.items()}
        return out

    # ---- emission and conversion --------------------------------------------

    def _build_xml(self, spec) -> str:
        """Emit *spec*, falling back to the last feasible specification.

        ``check_geometry`` raises when a sampled trunk width or actuator radius
        makes the placement chain degenerate, and ``check_nominal_pose`` when the
        pose breaks a constraint.  A raised generation would abort training, so
        an infeasible individual silently repeats the last design that did emit,
        which costs one individual's diversity rather than the run.  The design
        actually emitted is left in ``_last_emitted`` for _build_population.
        """
        try:
            check_nominal_pose(self.family, spec, self._dx_target, self._lean_max())
            xml = self.family.build_xml(spec, **self.builder_kwargs)
        except ValueError as exc:
            print(f"[design] infeasible specification, reusing the last feasible one: {exc}")
            self._last_emitted = self._last_feasible
            return self.family.build_xml(self._last_emitted, **self.builder_kwargs)
        self._last_feasible = self._last_emitted = spec
        return xml

    def _lean_max(self) -> float:
        _, (lo, hi) = self.param_ranges["thigh_bend_offset"]
        return max(abs(lo), abs(hi))

    def _generate_individual_mjcf(self, generation: int, idx: int, spec) -> str:
        gen_dir = os.path.join(self.output_dir, f"gen_{generation:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        # Exactly one dot in the basename: MjcfConverter._convert_asset splits
        # the basename on "." and unpacks the result into two names, so a second
        # dot raises ValueError there (mjcf_converter.py:61).
        xml_path = os.path.join(gen_dir, f"individual_{idx:04d}.xml")
        with open(xml_path, "w") as fh:
            fh.write(self._build_xml(spec))
        return xml_path

    def _generate_individual_usd(self, xml_path: str, idx: int) -> str:
        # Imported here rather than at module scope so that the generators can
        # be exercised without a running simulator.
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

        usd_out_dir = os.path.join(os.path.dirname(xml_path), f"individual_{idx:04d}_usd")
        cfg = MjcfConverterCfg(
            asset_path=xml_path,
            usd_dir=usd_out_dir,
            usd_file_name=f"{self.family.name}_{idx}.usd",
            link_density=0.0,
            import_inertia_tensor=True,
            fix_base=False,
            self_collision=False,
            make_instanceable=True,
            force_usd_conversion=True,
        )
        return MjcfConverter(cfg).usd_path


# ---------------------------------------------------------------------------
# Random design generator
# ---------------------------------------------------------------------------


class RandomDesignGenerator(DesignGeneratorBase):
    """Generates robot design populations by randomly perturbing the baseline.

    Each individual is produced by sampling every parameter of
    ``param_ranges``, applying the lengths and the nominal pose to the baseline
    specification, emitting the document, converting it, and recording the
    actuator parameters that the asset cannot carry.
    """

    def generate_population(self, generation: int) -> Population | None:
        return self._build_population(generation, range(self.num_individuals))

    def _generate_individual(self, generation: int, idx: int):
        rng = np.random.default_rng(seed=self._rng_seed(generation, idx))
        parameters = self._sample_parameter(rng)
        self._current_parameters = parameters
        spec = self._spec_from_parameters(parameters)
        return spec, self._actuator_params(spec, parameters)


# ---------------------------------------------------------------------------
# Fixed design generator
# ---------------------------------------------------------------------------


class FixedDesignGenerator(DesignGeneratorBase):
    """Emits one given design as a population of one, for replaying a searched design.

    Mirrors CMAESDesignGenerator._generate_individual, actuator parameters included, so the
    replayed robot is the one the search emitted. The base class's default ranges carry the
    scale or offset kind of every CMAES_PARAM_KEYS entry and the thigh_bend_offset range that
    _lean_max reads, both as train.py gave them to the run.
    """

    def __init__(self, parameters: dict[str, float], **kwargs):
        super().__init__(num_individuals=1, **kwargs)
        self.parameters = dict(parameters)

    def generate_population(self, generation: int):
        return self._build_population(generation, range(self.num_individuals))

    def _generate_individual(self, generation: int, idx: int):
        self._current_parameters = dict(self.parameters)
        spec = self._spec_from_parameters(self.parameters)
        return spec, self._actuator_params(spec)


# ---------------------------------------------------------------------------
# CMA-ES design generator
# ---------------------------------------------------------------------------


class CMAESDesignGenerator(DesignGeneratorBase):
    """Generates robot designs by sampling from a CMA-ES search distribution.

    The design vector is the unit-hypercube encoding of the parameters named by
    :data:`CMAES_PARAM_KEYS`, the two length scales and the two pose offsets.
    CMA-ES adapts the multivariate Gaussian search distribution from the
    per-individual mean episode return reported by :class:`CoptOnPolicyRunner`.
    """

    # The keys of param_ranges the search distribution carries, CMAES_PARAM_KEYS but for a
    # search over the actuator picks alone.
    searched_keys: tuple[str, ...] = CMAES_PARAM_KEYS

    def __init__(
        self,
        family: str,
        num_individuals: int,
        param_ranges: dict[str, ParamRange] | None = None,
        output_dir: str = "/tmp/copt_usds",
        sigma0: float = 0.2,
        seed: int = 0,
        es_state_path: str | None = None,
        late_start: bool = False,
        late_start_it: int = 8000,
        late_start_prior_pop_size: int = 4096,
        max_cma_iter: int = 10000,
        variable_params_path: str | None = None,
        tolx: float | None = None,
        tolfun: float | None = None,
        tolfunhist: float | None = None,
        final_population: str = "last",
        tolcat: float | None = None,
        filters: list[str] | None = None,
        filter_params: dict | None = None,
    ) -> None:
        super().__init__(
            family=family,
            num_individuals=num_individuals,
            param_ranges=param_ranges,
            output_dir=output_dir,
            variable_params_path=variable_params_path,
            filters=filters,
            filter_params=filter_params,
        )
        # pycma's tolerances, see plans/copt_training_refinement.md section 3. None, the default,
        # leaves pycma's 1e-11, 1e-11 and 1e-12, as every run before 2026-10-28.
        self._tolerances: dict[str, float] = {
            name: float(value)
            for name, value in (("tolx", tolx), ("tolfun", tolfun), ("tolfunhist", tolfunhist))
            if value is not None
        }
        # What a terminated search returns, "last" None, so the runner re-applies the last
        # population as before, and "mean" the distribution mean in every individual slot.
        if final_population not in ("last", "mean"):
            raise ValueError(f"final_population must be 'last' or 'mean', got {final_population!r}")
        self._final_population = final_population
        # The categorical searches' own termination, see CatCMAESDesignGenerator._stop_reasons, which
        # pycma's strategy neither needs nor reads.
        self._tolcat = None if tolcat is None else float(tolcat)
        self._best_history: list[float] = []
        self._last_range = float("inf")
        self._stop: list[str] = []
        self.late_start_prior_num_individual = late_start_prior_pop_size
        # Restrict CMA-ES to CMAES_PARAM_KEYS.
        # Caller-supplied param_ranges may tune the bounds of these keys
        # but cannot add new dimensions (any extras would be inert).
        self.param_ranges = {k: self.param_ranges[k] for k in CMAES_PARAM_KEYS}

        self.param_keys: list[str] = [k for k in self.param_ranges if k in self.searched_keys]
        self.num_params: int = len(self.param_keys)
        self._sigma0: float = float(sigma0)
        self._seed: int = int(seed)
        self.late_start = late_start
        self.late_start_it = late_start_it
        self._max_cma_iter = max_cma_iter

        if es_state_path is not None:
            with open(es_state_path, "rb") as fh:
                self._es = pickle.loads(fh.read())
            self._check_dimension()
        else:
            self._es = self._make_strategy()

        self._pending_solutions: list | None = None
        self._last_solutions: list | None = None
        self._terminated = False

    # --- strategy ------------------------------------------------------

    def _make_strategy(self):
        opts = cma.CMAOptions()
        opts.set({
            "popsize": int(self.num_individuals),
            "bounds": [np.zeros(self.num_params), np.ones(self.num_params)],
            "seed": self._seed,
            "verb_disp": 1,
            "verb_filenameprefix": str(Path(self.output_dir) / "cma_log") + "/",
            "maxiter": self._max_cma_iter,
            "BoundaryHandler": "BoundTransform",
            **self._tolerances,
        })
        return cma.CMAEvolutionStrategy(self._initial_mean(), self._sigma0, opts)

    def _initial_mean(self) -> np.ndarray:
        """The initial design in the unit box, where the search starts. The box midpoint before 2026-10-30."""
        lo = np.array([self.param_ranges[k][1][0] for k in self.param_keys])
        span = np.array([self.param_ranges[k][1][1] for k in self.param_keys]) - lo
        x0 = np.array([self._initial_value(k) for k in self.param_keys]) - lo
        # A zero-width range pins its parameter wherever the mean sits, so it keeps the former 0.5.
        return np.divide(x0, span, out=np.full(self.num_params, 0.5), where=span > 0)

    def _check_dimension(self) -> None:
        assert self._es.N == self.num_params, (
            f"Checkpoint dimension mismatch: "
            f"pickled {self._es.N}, expected {self.num_params}"
        )

    def _dumps(self) -> bytes:
        return self._es.pickle_dumps()

    # --- Public API ---------------------------------------------------

    def _redraw(self, idx: int) -> None:
        # The random phase draws from a seed, a search generation from the distribution, whose
        # replacement sample is the genotype tell then pairs with the fitness.
        if not self.late_start:
            self._pending_solutions[idx] = self._es.ask(1)[0]

    def toggle_late_start(self) -> None:
        self.late_start = not self.late_start

    def search_statistics(self) -> dict | None:
        lo = np.array([self.param_ranges[k][1][0] for k in self.param_keys])
        span = np.array([self.param_ranges[k][1][1] for k in self.param_keys]) - lo
        result = self._es.result
        return {
            "cma_iteration": int(self._es.countiter),
            "evaluations": int(self._es.countevals),
            "sigma": float(self._es.sigma),
            "axis_ratio": float(self._es.D.max() / self._es.D.min()),
            "terminated": float(self._terminated),
            "stop": ", ".join(self._es.stop()),
            "mean": dict(zip(self.param_keys, (lo + span * np.asarray(result.xfavorite)).tolist())),
            "std": dict(zip(self.param_keys, (span * np.asarray(result.stds)).tolist())),
        }

    def optimised_design(self) -> tuple[dict[str, float], dict[str, str] | None] | None:
        # xfavorite, the distribution mean, is pycma's estimate of the optimum.
        return self._denormalise(self._es.result.xfavorite), None

    def sample_batch(self) -> None:
        assert self._pending_solutions is None, "sample_batch called twice consecutively"
        self._pending_solutions = self._es.ask()

    def update_with_fitness(self, fitness: list[float]) -> None:
        assert self._last_solutions is not None, (
            "update_with_fitness called before generate_population"
        )
        if not self._terminated:
            print("Updating Designs with PPO Training Roll")
            costs = [self._sanitise_cost(-float(f)) for f in fitness]
            self._es.tell(self._last_solutions, costs)
            if self._es.stop():
                print(f"[CMA-ES] terminated: {self._es.stop()}")
                self._terminated = True
            self.sample_batch()

    def generate_population(self, generation: int) -> Population | None:  # type: ignore[override]
        if self._terminated:
            self._pending_solutions = self._last_solutions
            if self._final_population == "mean":
                print("CMAES Terminated. Fine-tuning on the distribution mean in every individual slot")
                return self._mean_population(generation)
            print("CMAES Terminated. Fine-tuning parent policy on selected design only")
            return None
        print("Generating New Designs using CMAES output")
        if self.late_start:
            print("late start enabled sampling random designs")
            pop = self._build_population(generation, range(self.late_start_prior_num_individual))
        else:
            pop = self._build_population(generation, range(self.num_individuals))
        self._last_solutions = self._pending_solutions
        self._pending_solutions = None
        return pop

    # --- Internal helpers --------------------------------------------

    def _mean_population(self, generation: int) -> Population:
        """The distribution mean, pycma's estimate of the optimum, in every one of num_individuals slots.

        The mean is emitted and converted once and its records repeated, so the fine-tune costs one
        conversion rather than num_individuals. The size is kept so that the runner's in-place update
        still reaches every environment. The repeated records are shared, every consumer only reading them.
        """
        values, picks = self.optimised_design()

        def mean(generation: int, idx: int):
            self._current_parameters = dict(values)
            spec = self._spec_from_parameters(values, picks)
            return spec, self._actuator_params(spec)

        one = self._build_population(generation, range(1), individual=mean, filtered=False)
        n = self.num_individuals
        return Population(
            one.get_usd_files() * n, one.get_actuator_params() * n, one.get_link_length_params() * n,
            one.get_root_body(), one.get_specs() * n, one.get_nominal() * n,
            one.get_parameters() * n, one.get_xml_files() * n,
        )

    def _generate_individual(self, generation: int, idx: int):
        if self.late_start:
            rng = np.random.default_rng(seed=self._rng_seed(generation, idx))
            parameters = self._sample_parameter(rng)
        else:
            assert self._pending_solutions is not None, (
                "_generate_individual called before sample_batch"
            )
            parameters = self._denormalise(self._pending_solutions[idx])
        self._current_parameters = parameters
        spec = self._spec_from_parameters(parameters)
        return spec, self._actuator_params(spec)

    def _denormalise(self, x) -> dict[str, float]:
        parameters: dict[str, float] = {}
        for i, key in enumerate(self.param_keys):
            _, (lo, hi) = self.param_ranges[key]
            xi = float(np.clip(np.asarray(x)[i], 0.0, 1.0))
            parameters[key] = float(lo + xi * (hi - lo))
        return parameters

    @staticmethod
    def _sanitise_cost(c: float, fallback: float = 1e6) -> float:
        if not np.isfinite(c):
            return fallback
        return c

    def save_state(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self._dumps())

    def get_state(self) -> dict:
        """Return a picklable snapshot of the full search state.

        Widens the strategy-object-only :meth:`save_state` into the complete
        bookkeeping required for an identical resume, namely the pending and
        last ``ask`` solutions that pair genotypes with their fitness, the
        termination flag, and the late-start flag.  The two subclasses inherit
        this unchanged.
        """
        return {
            "es": self._dumps(),
            "pending_solutions": self._pending_solutions,
            "last_solutions": self._last_solutions,
            "terminated": self._terminated,
            "late_start": self.late_start,
        }

    def load_state(self, state: dict) -> None:
        """Restore the search state produced by :meth:`get_state`."""
        self._es = pickle.loads(state["es"])
        self._check_dimension()
        self._pending_solutions = state["pending_solutions"]
        self._last_solutions = state["last_solutions"]
        self._terminated = state["terminated"]
        self.late_start = state["late_start"]


class GrowingDesignDistCMAESDesignGenerator(CMAESDesignGenerator):
    """CMA-ES whose late-start prior widens from the initial design to the full box."""

    def _generate_individual(self, generation: int, idx: int):
        if self.late_start:
            rng = np.random.default_rng(seed=self._rng_seed(generation, idx))
            # Saturation point of the design distribution's growth
            s = 0.1
            # Total number of generations of random population sampling
            n = self.late_start_it
            scale = 0.95 * (generation / (n - s)) + 0.05
            parameters = self._sample_parameter_v2(rng, scale)
        else:
            assert self._pending_solutions is not None, (
                "_generate_individual called before sample_batch"
            )
            parameters = self._denormalise(self._pending_solutions[idx])
        self._current_parameters = parameters
        spec = self._spec_from_parameters(parameters)
        return spec, self._actuator_params(spec)


# ---------------------------------------------------------------------------
# Mixed continuous and categorical CMA-ES design generator
# ---------------------------------------------------------------------------


class CatCMAESDesignGenerator(CMAESDesignGenerator):
    """CMA-ES with Margin over the link lengths *and* the actuator picks.

    The continuous part is the same unit-hypercube encoding of the length
    scales and pose offsets that :class:`CMAESDesignGenerator` searches.
    Alongside it sits one categorical variable per actuator group, whose
    categories are the entries of that family's catalog, so a genotype names
    both how the leg is shaped and which unit drives each joint.  The two are
    optimised jointly by ``cmaes.CatCMAwM``, which carries a multivariate Gaussian over the
    continuous block and a categorical distribution per discrete block and
    applies the margin correction of Hamano et al. [1] so the discrete
    marginals cannot collapse before the continuous block has converged.

    The groups and the shelf are the family's own, ``Family.groups`` and
    ``Family.catalog()``, the latter being the simulation-complete entries only,
    so no constructor argument restates either. Before the late start every
    group draws its entry uniformly from the shelf, so the policy has met every
    actuator on every joint before the search begins.
    """

    # --- strategy ------------------------------------------------------

    def _resolve_categories(self) -> None:
        if getattr(self, "cat_names", None) is not None:
            return
        names = [a.name for a in self.family.catalog()]
        if len(names) < 2:
            raise ValueError(
                "A categorical search needs at least two simulation-complete "
                f"catalogue entries, the {self.family.name} family offers {names}. "
                f"See design_generator/{self.family.name}/catalogs/actuators.json."
            )
        self.cat_groups: tuple[str, ...] = tuple(self.family.groups)
        self.cat_names: list[str] = names

    def _growth_scale(self, generation: int) -> float:
        """The random phase's widening, 0.05 at generation 0 and about 1 at its last, the geometry's own schedule."""
        return 0.95 * (generation / (self.late_start_it - 0.1)) + 0.05

    def _neighbours(self, group: str) -> list[str]:
        """The shelf ordered by actuator_distance from the base robot's pick for *group*, nearest first.

        The base robot's own pick leads, even against an entry at distance zero such as the quadruped's
        A1-ABAD beside its A1-HIP, and a tie among the rest keeps the shelf's order. A base pick missing
        from the shelf leaves the shelf's order.
        """
        catalog = {a.name: a for a in self.family.full_catalog()}
        base = catalog.get(getattr(self.baseline, self.family.spec_fields[group]))
        if base is None or base.name not in self.cat_names:
            return list(self.cat_names)
        return sorted(self.cat_names, key=lambda name: (name != base.name, actuator_distance(catalog[name], base),
                                                        self.cat_names.index(name)))

    def _pick_scale(self, generation: int) -> float:
        """The share of each group's shelf the random phase draws from, 0.05 at generation 0 and 1 at its last.

        The geometry's schedule, _growth_scale, with its end pinned to the random phase's last generation,
        late_start_it - 1, which _growth_scale reaches only at about 0.94 for 15 generations, so that every
        actuator is met before CMA-ES begins.
        """
        last = self.late_start_it - 1
        return 1.0 if last <= 0 else min(1.0, 0.05 + 0.95 * generation / last)

    def _random_phase_picks(self, rng: np.random.Generator, generation: int) -> dict[str, str]:
        """Each group's pick, uniform over the nearest ceil(scale K) of its K shelf entries to the base robot's.

        Generation 0 gives every design the base robot's actuators, at most 20 entries to a shelf, and the
        last generation of the random phase the whole shelf, so the actuator choice widens with the
        geometry box. Before 2026-11-12 every group drew uniformly from the whole shelf from generation 0.
        """
        scale = self._pick_scale(generation)
        picks = {}
        for group in self.cat_groups:
            order = self._neighbours(group)
            support = max(1, math.ceil(scale * len(order) - 1e-9))
            picks[group] = order[int(rng.integers(support))]
        return picks

    def _make_strategy(self):
        from cmaes import CatCMAwM

        self._resolve_categories()
        n_cat = len(self.cat_names)
        # The continuous block keeps the unit-hypercube convention of the
        # parent class, so _denormalise is reused unchanged.
        x_space = [[0.0, 1.0] for _ in range(self.num_params)]
        c_space = [n_cat for _ in self.cat_groups]
        # Start every categorical marginal uniform except for the baseline's
        # own pick, which is given the plurality so that generation zero is
        # centred on the donor robot rather than on an arbitrary shelf entry.
        cat_param = np.full((len(self.cat_groups), n_cat), 1.0 / n_cat)
        for row, group in enumerate(self.cat_groups):
            pick = getattr(self.baseline, self.family.spec_fields[group])
            if pick in self.cat_names:
                cat_param[row, :] = 0.5 / (n_cat - 1)
                cat_param[row, self.cat_names.index(pick)] = 0.5
        return CatCMAwM(
            x_space=x_space,
            c_space=c_space,
            cat_param=cat_param,
            population_size=int(self.num_individuals),
            mean=self._initial_mean(),
            sigma=self._sigma0,
            seed=self._seed,
        )

    def _check_dimension(self) -> None:
        self._resolve_categories()
        got = int(self._es.population_size)
        assert got == int(self.num_individuals), (
            f"Checkpoint population size mismatch: pickled {got}, "
            f"expected {self.num_individuals}"
        )

    def _dumps(self) -> bytes:
        # CatCMAwM carries no pickle_dumps of its own, unlike cma's strategy.
        return pickle.dumps(self._es)

    def get_state(self) -> dict:
        # The shelf the strategy's category indices refer to, which the catalogue may since have widened,
        # and the fitness history the termination criteria read.
        return {**super().get_state(), "cat_names": list(self.cat_names),
                "best_history": list(self._best_history), "last_range": self._last_range, "stop": list(self._stop)}

    def load_state(self, state: dict) -> None:
        """Restore the state, the shelf its category indices refer to and the termination history."""
        super().load_state(state)
        self.cat_names = list(state["cat_names"])
        self._best_history = list(state["best_history"])
        self._last_range = float(state["last_range"])
        self._stop = list(state["stop"])

    # --- termination ----------------------------------------------------

    def _history_length(self) -> int:
        """pycma's 10 + 30 N / popsize, N counting the continuous variables and the actuator groups."""
        return 10 + math.ceil(30 * (self.num_params + len(self.cat_groups)) / int(self._es.population_size))

    def _categorical_converged(self) -> bool:
        """Every group's likeliest pick within tolcat of the probability the margin lets it reach.

        CatCMAwM keeps every category at or above q_min (cmaes/_catcmawm.py, _init_categorical), so a
        group's likeliest pick can reach at most 1 - (K - 1) q_min, about 0.924 for 15 entries over four
        groups. None, the default, never counts the categorical block converged.
        """
        if self._tolcat is None:
            return False
        q, qmin, k = np.asarray(self._es._q), np.asarray(self._es._qmin), np.asarray(self._es._K)
        ceiling = 1.0 - (k - 1) * qmin
        return bool(np.all(q.max(axis=1) >= ceiling - self._tolcat))

    def _continuous_converged(self, tolx: float) -> bool:
        """pycma's tolx, every sampling standard deviation and evolution path entry in the unit box below tolx."""
        if self.num_params == 0:
            return True
        es = self._es
        a = np.asarray(es._A)
        std = es._sigma * a * np.sqrt(np.diag(np.asarray(es._C)))
        return bool(np.all(std < tolx) and np.all(np.abs(es._sigma * a * np.asarray(es._pc)) < tolx))

    def _stop_reasons(self) -> list[str]:
        """The criteria the search now meets, the CatCMAwM counterparts of pycma's that a leg_geometry run stops on.

        maxiter, the generation cap max_cma_iter. tolfunhist, the range of the best fitness over the last
        _history_length generations below tolfunhist. tolfun, that range and the range of the last
        generation's fitness both below tolfun. tolx, the continuous block converged by tolx and the
        categorical block by tolcat, both being required since either alone leaves the design unsettled.
        CatCMAwM's own should_stop checks its Gaussian against fixed tolerances of 1e-12 and returns
        False without one, so it is not consulted. Before 2026-11-12 a categorical search never stopped.
        """
        reasons = []
        if self._max_cma_iter is not None and int(self._es.generation) >= self._max_cma_iter:
            reasons.append("maxiter")
        n = self._history_length()
        if len(self._best_history) >= n:
            spread = max(self._best_history[-n:]) - min(self._best_history[-n:])
            if "tolfunhist" in self._tolerances and spread < self._tolerances["tolfunhist"]:
                reasons.append("tolfunhist")
            if "tolfun" in self._tolerances and max(spread, self._last_range) < self._tolerances["tolfun"]:
                reasons.append("tolfun")
        if ("tolx" in self._tolerances and self._categorical_converged()
                and self._continuous_converged(self._tolerances["tolx"])):
            reasons.append("tolx")
        return reasons


    def search_statistics(self) -> dict | None:
        # Private state of cmaes 0.13.1, whose CatCMAwM exposes no accessor for it.
        es = self._es
        lo = np.array([self.param_ranges[k][1][0] for k in self.param_keys])
        span = np.array([self.param_ranges[k][1][1] for k in self.param_keys]) - lo
        # Sampled as N(m, sigma^2 A C A), cmaes/_catcmawm.py _sample_from_gaussian.
        a = np.asarray(es._A)
        cov = es._sigma ** 2 * a[:, None] * np.asarray(es._C) * a[None, :]
        eig = np.linalg.eigvalsh(cov)
        mean = np.clip(np.asarray(es._mean), 0.0, 1.0)
        q = np.asarray(es._q)
        return {
            "cma_iteration": int(es.generation),
            "evaluations": int(es.generation * es.population_size),
            "sigma": float(es._sigma),
            "axis_ratio": float(np.sqrt(eig.max() / eig.min())),
            "terminated": float(self._terminated),
            "stop": ", ".join(self._stop or self._stop_reasons()),
            "mean": dict(zip(self.param_keys, (lo + span * mean).tolist())),
            "std": dict(zip(self.param_keys, (span * np.sqrt(np.diag(cov))).tolist())),
            **self._pick_statistics(q),
        }


    def optimised_design(self) -> tuple[dict[str, float], dict[str, str] | None] | None:
        q = np.asarray(self._es._q)
        picks = {group: self.cat_names[int(q[row].argmax())] for row, group in enumerate(self.cat_groups)}
        return self._denormalise(np.clip(np.asarray(self._es._mean), 0.0, 1.0)), picks

    # --- ask and tell ---------------------------------------------------

    def sample_batch(self) -> None:
        assert self._pending_solutions is None, "sample_batch called twice consecutively"
        self._pending_solutions = [self._es.ask() for _ in range(self._es.population_size)]

    def _redraw(self, idx: int) -> None:
        if not self.late_start:
            self._pending_solutions[idx] = self._es.ask()

    def update_with_fitness(self, fitness: list[float]) -> None:
        assert self._last_solutions is not None, (
            "update_with_fitness called before generate_population"
        )
        if self._terminated:
            return
        print("Updating Designs with PPO Training Roll")
        costs = [self._sanitise_cost(-float(f)) for f in fitness]
        self._es.tell(list(zip(self._last_solutions, costs)))
        self._best_history.append(min(costs))
        self._last_range = max(costs) - min(costs)
        self._stop = self._stop_reasons()
        if self._stop:
            print(f"[CatCMAwM] terminated: {self._stop}")
            self._terminated = True
        self.sample_batch()

    # --- decoding -------------------------------------------------------

    @staticmethod
    def _category_indices(c) -> list[int]:
        """Category index per variable, from either encoding CatCMAwM emits.

        ``Solution.c`` is documented as a one-hot matrix of shape
        ``(n_variables, n_categories)``; a one-dimensional array of indices is
        accepted too so that a future release which drops the encoding does not
        break the decode.
        """
        arr = np.asarray(c)
        if arr.ndim == 2:
            return [int(i) for i in arr.argmax(axis=1)]
        return [int(round(float(v))) for v in arr.reshape(-1)]

    def _picks_from_solution(self, solution) -> dict[str, str]:
        indices = self._category_indices(solution.c)
        n = len(self.cat_names)
        return {
            group: self.cat_names[int(np.clip(indices[row], 0, n - 1))]
            for row, group in enumerate(self.cat_groups)
        }

    def _generate_individual(self, generation: int, idx: int):
        if self.late_start:
            rng = np.random.default_rng(seed=self._rng_seed(generation, idx))
            parameters = self._sample_parameter_v2(rng, self._growth_scale(generation))
            picks = self._random_phase_picks(rng, generation)
        else:
            assert self._pending_solutions is not None, (
                "_generate_individual called before sample_batch"
            )
            solution = self._pending_solutions[idx]
            parameters = self._denormalise(solution.x)
            picks = self._picks_from_solution(solution)
        self._current_parameters = parameters
        spec = self._spec_from_parameters(parameters, picks)
        return spec, self._actuator_params(spec)


# ---------------------------------------------------------------------------
# Categorical only CMA-ES design generator
# ---------------------------------------------------------------------------


class CatOnlyCMAESDesignGenerator(CatCMAESDesignGenerator):
    """The actuator picks alone on the baseline's link lengths and nominal pose.

    ``cmaes.CatCMAwM`` with a categorical block and no continuous one, its x_space being empty,
    which the library supports by skipping its Gaussian, so that only a categorical distribution
    per actuator group is sampled and updated. pycma, the ``cma`` package, offers no categorical
    search, and ``cmaes.CatCMA`` requires a continuous mean. Before the late start every group
    draws its entry uniformly from the shelf, as in :class:`CatCMAESDesignGenerator`.
    """

    searched_keys: tuple[str, ...] = ()

    def search_statistics(self) -> dict | None:
        q = np.asarray(self._es._q)
        return {
            "cma_iteration": int(self._es.generation),
            "evaluations": int(self._es.generation * self._es.population_size),
            "terminated": float(self._terminated),
            "stop": ", ".join(self._stop or self._stop_reasons()),
            **self._pick_statistics(q),
        }

    def optimised_design(self) -> tuple[dict[str, float], dict[str, str] | None] | None:
        q = np.asarray(self._es._q)
        return {}, {group: self.cat_names[int(q[row].argmax())] for row, group in enumerate(self.cat_groups)}

    def _generate_individual(self, generation: int, idx: int):
        if self.late_start:
            rng = np.random.default_rng(seed=self._rng_seed(generation, idx))
            picks = self._random_phase_picks(rng, generation)
        else:
            assert self._pending_solutions is not None, "_generate_individual called before sample_batch"
            picks = self._picks_from_solution(self._pending_solutions[idx])
        self._current_parameters = {}
        spec = self._spec_from_parameters({}, picks)
        return spec, self._actuator_params(spec)
