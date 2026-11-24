"""Filters that refuse a design from its emitted MJCF before the design is converted and trained.

A filter reads the documents of a population, which every family's builder emits with the same
conventions, and returns the positions of the designs it refuses. DesignGeneratorBase.filter_population
redraws those designs until every filter passes them. See plans/design_filtering.md.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod

import numpy as np

GRAVITY = 9.81


class FilterBase(ABC):
    """A test every design of a population must pass, holding whatever state and statistics it keeps.

    Args:
        family: The design generator family, "biped" or "quadruped", for a filter that needs it.
    """

    # True for a filter whose filter() also takes the design records, one dict per document, as
    # DesignGeneratorBase.design_record writes them. False, the default, keeps filter(xml_files).
    uses_records: bool = False

    def __init__(self, family: str) -> None:
        self.family = family
        self.statistics: dict = {}

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def filter(self, xml_files: list[str]) -> list[int]:
        """The positions in *xml_files* of the designs this filter refuses, in ascending order."""
        ...

    def scores(self, xml_files: list[str]) -> dict[str, list[float]]:
        """Per design measures the design logs record, keyed by column name. None by default."""
        return {}


# ---------------------------------------------------------------------------
# MJCF model
# ---------------------------------------------------------------------------


def _vec(text: str | None, default) -> np.ndarray:
    return np.array([float(v) for v in text.split()]) if text else np.array(default, dtype=float)


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class MjcfStatics:
    """The rigid body tree of one emitted MJCF, as the arrays the vectorised statics consume.

    Bodies are kept in document order, so a parent always precedes its children. Every body carries at
    most one hinge, the root's free joint excepted. The ground contacts are the collision geoms of the
    leaves of the tree, a sphere contributing its centre and radius and a box the four corners of its
    lowest face, which in the builders' documents are the quadruped's feet and the biped's soles.
    A joint's torque ceiling is its motor's forcerange, taken from the motor's own attribute or its class.
    """

    def __init__(self, path: str) -> None:
        root = ET.parse(path).getroot()
        force_by_class = {
            d.get("class"): _vec(d.find("motor").get("forcerange"), [0, 0])
            for d in root.iter("default") if d.get("class") and d.find("motor") is not None
        }
        force_by_joint = {}
        for motor in root.iter("motor"):
            if motor.get("joint"):
                force = motor.get("forcerange")
                force_by_joint[motor.get("joint")] = (
                    _vec(force, [0, 0]) if force else force_by_class.get(motor.get("class"), np.zeros(2)))

        self.body_names: list[str] = []
        parents, pos, rot, mass, ipos = [], [], [], [], []
        self.joint_names: list[str] = []
        self.joint_classes: list[str] = []
        joint_body, axis, jpos, ranges, limits = [], [], [], [], []
        contact_body, contact_point, contact_radius = [], [], []

        def visit(element, parent: int) -> None:
            index = len(self.body_names)
            self.body_names.append(element.get("name"))
            parents.append(parent)
            pos.append(_vec(element.get("pos"), [0, 0, 0]))
            rot.append(_quat_matrix(_vec(element.get("quat"), [1, 0, 0, 0])))
            inertial = element.find("inertial")
            mass.append(float(inertial.get("mass")) if inertial is not None else 0.0)
            ipos.append(_vec(inertial.get("pos") if inertial is not None else None, [0, 0, 0]))
            hinges = [j for j in element.findall("joint") if j.get("type", "hinge") == "hinge"]
            if len(hinges) > 1:
                raise ValueError(f"body {element.get('name')!r} carries {len(hinges)} hinges, the statics take one")
            for joint in hinges:
                self.joint_names.append(joint.get("name"))
                self.joint_classes.append(joint.get("class", ""))
                joint_body.append(index)
                axis.append(_vec(joint.get("axis"), [0, 0, 1]))
                jpos.append(_vec(joint.get("pos"), [0, 0, 0]))
                ranges.append(_vec(joint.get("range"), [-np.pi, np.pi]))
                limits.append(float(np.max(np.abs(force_by_joint.get(joint.get("name"), np.zeros(2))))))
            children = element.findall("body")
            if not children:
                for geom in element.findall("geom"):
                    if geom.get("class") != "collision":
                        continue
                    centre = _vec(geom.get("pos"), [0, 0, 0])
                    g_rot = _quat_matrix(_vec(geom.get("quat"), [1, 0, 0, 0]))
                    size = _vec(geom.get("size"), [0, 0, 0])
                    if geom.get("type") == "sphere":
                        contact_body.append(index)
                        contact_point.append(centre)
                        contact_radius.append(size[0])
                    elif geom.get("type") == "box":
                        for sx in (-1.0, 1.0):
                            for sy in (-1.0, 1.0):
                                contact_body.append(index)
                                contact_point.append(centre + g_rot @ (size * np.array([sx, sy, -1.0])))
                                contact_radius.append(0.0)
            for child in children:
                visit(child, index)

        visit(root.find("worldbody").find("body"), -1)
        self.parents = np.array(parents)
        self.pos, self.rot = np.array(pos), np.array(rot)
        self.mass, self.ipos = np.array(mass), np.array(ipos)
        self.joint_body = np.array(joint_body)
        self.axis = np.array(axis) / np.linalg.norm(axis, axis=1, keepdims=True)
        self.jpos, self.range, self.limit = np.array(jpos), np.array(ranges), np.array(limits)
        self.contact_body = np.array(contact_body)
        self.contact_point, self.contact_radius = np.array(contact_point), np.array(contact_radius)
        if len(self.contact_body) < 3:
            raise ValueError(f"{path} offers {len(self.contact_body)} ground contacts, a stance needs three")

    def topology(self) -> tuple:
        return (tuple(self.parents), tuple(self.joint_body), tuple(self.contact_body), tuple(self.joint_classes))

    def subtree(self) -> tuple[np.ndarray, np.ndarray]:
        """(J, B) and (J, C) masks, the bodies and the contacts each joint carries."""
        inside = np.eye(len(self.parents), dtype=bool)
        for b, p in enumerate(self.parents):
            if p >= 0:
                inside[:, b] |= inside[:, p]       # a body is inside every subtree its parent is inside
        bodies = inside[self.joint_body]
        return bodies, bodies[:, self.contact_body]


def _stack(models: list[MjcfStatics], name: str) -> np.ndarray:
    return np.stack([getattr(m, name) for m in models])


def _axis_angle(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotation matrices about unit *axis* (..., 3) by *angle* (...), Rodrigues' formula."""
    k = np.zeros(axis.shape[:-1] + (3, 3))
    k[..., 0, 1], k[..., 0, 2] = -axis[..., 2], axis[..., 1]
    k[..., 1, 0], k[..., 1, 2] = axis[..., 2], -axis[..., 0]
    k[..., 2, 0], k[..., 2, 1] = -axis[..., 1], axis[..., 0]
    s, c = np.sin(angle)[..., None, None], np.cos(angle)[..., None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def standing_statics(models: list[MjcfStatics], q: np.ndarray, coplanar_tol: float = 1e-3) -> dict:
    """The static stance of every design at every joint configuration, vectorised over both.

    *q* is (D, S, J), D designs sharing one topology and S configurations. The base is free, so each
    configuration is turned until its contacts rest on a level ground, gravity is balanced by vertical
    contact forces of least norm, and every joint's torque is the moment its subtree must resist.
    Returns arrays over (D, S) and (D, S, J) in the ground frame, its origin the contacts' centroid.
    """
    D, S, J = q.shape
    B = len(models[0].parents)
    pos, rot = _stack(models, "pos")[:, None], _stack(models, "rot")[:, None]          # (D, 1, B, ...)
    mass, ipos = _stack(models, "mass"), _stack(models, "ipos")[:, None]
    axis, jpos = _stack(models, "axis")[:, None], _stack(models, "jpos")[:, None]      # (D, 1, J, 3)
    joint_of_body = np.full(B, -1)
    joint_of_body[models[0].joint_body] = np.arange(J)

    # Forward kinematics, X_b = X_parent (pos, quat) (jpos, R(axis, q)) (-jpos), MuJoCo's convention.
    R = np.zeros((D, S, B, 3, 3))
    p = np.zeros((D, S, B, 3))
    o = np.zeros((D, S, J, 3))                    # joint anchors
    a = np.zeros((D, S, J, 3))                    # joint axes
    for b, parent in enumerate(models[0].parents):
        R_p = np.broadcast_to(np.eye(3), (D, S, 3, 3)) if parent < 0 else R[:, :, parent]
        p_p = np.zeros((D, S, 3)) if parent < 0 else p[:, :, parent]
        R_l = R_p @ rot[:, :, b]
        p_l = p_p + np.einsum("dsij,dsj->dsi", R_p, np.broadcast_to(pos[:, :, b], (D, S, 3)))
        j = joint_of_body[b]
        if j < 0:
            R[:, :, b], p[:, :, b] = R_l, p_l
            continue
        R_q = _axis_angle(np.broadcast_to(axis[:, :, j], (D, S, 3)), q[..., j])
        anchor = np.broadcast_to(jpos[:, :, j], (D, S, 3))
        o[:, :, j] = p_l + np.einsum("dsij,dsj->dsi", R_l, anchor)
        a[:, :, j] = np.einsum("dsij,dsj->dsi", R_l, np.broadcast_to(axis[:, :, j], (D, S, 3)))
        R[:, :, b] = R_l @ R_q
        p[:, :, b] = o[:, :, j] - np.einsum("dsij,dsj->dsi", R[:, :, b], anchor)

    com_b = p + np.einsum("dsbij,dsbj->dsbi", R, np.broadcast_to(ipos, (D, S, B, 3)))
    total = mass.sum(axis=1)                                                            # (D,)
    com = np.einsum("db,dsbi->dsi", mass, com_b) / total[:, None, None]
    cb = models[0].contact_body
    c_local = _stack(models, "contact_point")[:, None]
    radius = _stack(models, "contact_radius")[:, None]                                  # (D, 1, C)
    centre = p[:, :, cb] + np.einsum("dscij,dscj->dsci", R[:, :, cb], np.broadcast_to(c_local, (D, S) + c_local.shape[2:]))

    # The ground plane, through the contacts, its normal the least principal axis, pointing at the body.
    mid = centre.mean(axis=2, keepdims=True)
    vt = np.linalg.svd(centre - mid)[2]
    n = vt[..., -1, :]
    n = n * np.where(np.einsum("dsi,dsi->ds", com - mid[:, :, 0], n) < 0.0, -1.0, 1.0)[..., None]
    coplanar = np.abs(np.einsum("dsci,dsi->dsc", centre - mid, n)).max(axis=2) <= coplanar_tol
    contact = centre - radius[..., None] * n[:, :, None]
    ground_mid = contact.mean(axis=2, keepdims=True)

    # Turn the ground frame so that n is +z, the least rotation, about n x z.
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(n, z)
    angle = np.arctan2(np.linalg.norm(cross, axis=-1), n @ z)
    unit = cross / np.maximum(np.linalg.norm(cross, axis=-1, keepdims=True), 1e-12)
    G = _axis_angle(unit, angle)                                                        # (D, S, 3, 3)

    def ground(x):
        return np.einsum("dsij,ds...j->ds...i", G, x - ground_mid.reshape(D, S, *([1] * (x.ndim - 3)), 3))

    com_g, com_bg, contact_g, o_g = ground(com), ground(com_b), ground(contact), ground(o)
    a_g = np.einsum("dsij,dskj->dski", G, a)
    origin_g = ground(p)
    upright = np.einsum("dsij,dsj->dsi", G, R[:, :, 0, :, 2])[..., 2] > 0.0       # the root's z axis

    # Validity, the root upright and nothing but the contacts on the ground. Support, the centre of mass over the
    # rectangle the contacts span, which for the symmetric stances sampled here is their hull.
    leaves = np.zeros(B, dtype=bool)
    leaves[cb] = True
    clearance = np.minimum(origin_g[..., 2], com_bg[..., 2])[..., ~leaves].min(axis=2)
    lo, hi = contact_g[..., :2].min(axis=2), contact_g[..., :2].max(axis=2)
    supported = np.all((com_g[..., :2] >= lo) & (com_g[..., :2] <= hi), axis=-1)

    # Vertical contact forces of least norm balancing the weight and its moments about the centre of mass.
    C = len(cb)
    A = np.stack([np.ones(contact_g.shape[:3]),
                  contact_g[..., 0] - com_g[..., None, 0],
                  contact_g[..., 1] - com_g[..., None, 1]], axis=2)                    # (D, S, 3, C)
    weight = total[:, None] * GRAVITY
    rhs = np.zeros((D, S, 3))
    rhs[..., 0] = weight
    force = np.einsum("dsci,dsi->dsc", np.linalg.pinv(A), rhs)                          # (D, S, C)

    # Joint torques, tau_j = -a_j . [sum over the subtree of (r - o_j) x F], gravity on the bodies and
    # the contact forces on the contacts.
    body_mask, contact_mask = models[0].subtree()
    gravity = np.zeros((D, S, B, 3))
    gravity[..., 2] = -mass[:, None, :] * GRAVITY
    push = np.zeros((D, S, C, 3))
    push[..., 2] = force
    moment = (np.einsum("jb,dsjbi->dsji", body_mask,
                        np.cross(com_bg[:, :, None] - o_g[:, :, :, None], gravity[:, :, None]))
              + np.einsum("jc,dsjci->dsji", contact_mask,
                          np.cross(contact_g[:, :, None] - o_g[:, :, :, None], push[:, :, None])))
    torque = -np.einsum("dsji,dsji->dsj", a_g, moment)

    return {
        "torque": torque, "force": force, "com": com_g, "contact": contact_g, "supported": supported,
        "coplanar": coplanar, "clearance": clearance, "upright": upright, "ground_rotation": G, "ground_origin": ground_mid[:, :, 0],
        "normal": n, "body_rotation": R, "body_position": p, "mass": total,
    }


# ---------------------------------------------------------------------------
# Standing filter
# ---------------------------------------------------------------------------


class StandingFilter(FilterBase):
    """Refuse a design that cannot hold its own weight in too many of a fixed set of stances.

    Every design meets the same num_positions joint configurations, a Latin hypercube over the soft
    range of each pitch joint class scaled towards the nominal pose by position_fraction, every leg
    taking the same values and every other joint its nominal, and the same num_torques torque
    ceilings, fractions of each motor's forcerange spread evenly over torque_fraction. A pair fails
    where the stance leaves something other than the contacts on the ground, the centre of mass outside
    the support, or a joint's static torque above the ceiling, and a design whose failed share exceeds
    max_failure_fraction is refused.

    Args:
        family: The design generator family.
        num_positions: Joint configurations per design.
        num_torques: Torque ceilings per configuration.
        max_failure_fraction: Largest share of failed pairs a design may show.
        position_fraction: Share of each pitch joint's soft range sampled, about the nominal pose. 1.0
            samples the whole soft range, which refuses both donors, see plans/design_filtering.md section 2.5.
        torque_fraction: (lo, hi), the range of the ceilings as fractions of forcerange. (0.0, 1.0)
            spreads them over the whole limit, which refuses both donors with position_fraction 1.0.
        soft_limit_factor: The asset's soft_joint_pos_limit_factor, which narrows the range about its middle.
        seed: Seed of the Latin hypercube, so that every design and every generation meets the same stances.
    """

    def __init__(
        self,
        family: str,
        num_positions: int = 16,
        num_torques: int = 16,
        max_failure_fraction: float = 0.25,
        position_fraction: float = 0.25,
        torque_fraction: tuple[float, float] = (0.5, 1.0),
        soft_limit_factor: float = 0.9,
        seed: int = 0,
    ) -> None:
        super().__init__(family)
        self.num_positions = int(num_positions)
        self.num_torques = int(num_torques)
        self.max_failure_fraction = float(max_failure_fraction)
        self.position_fraction = float(position_fraction)
        self.torque_fraction = (float(torque_fraction[0]), float(torque_fraction[1]))
        self.soft_limit_factor = float(soft_limit_factor)
        self.seed = int(seed)
        lo, hi = self.torque_fraction
        self.ceilings = lo + (hi - lo) * (np.arange(self.num_torques) + 0.5) / self.num_torques

    def _pitch_classes(self, model: MjcfStatics) -> list[str]:
        """The joint classes whose axes lie along the base's y axis at the nominal pose."""
        R = np.eye(3)[None].repeat(len(model.parents), axis=0)
        for b, parent in enumerate(model.parents):
            R[b] = model.rot[b] if parent < 0 else R[parent] @ model.rot[b]
        world = np.einsum("jik,jk->ji", R[model.joint_body], model.axis)
        pitch = np.abs(world[:, 1]) > 0.99
        return sorted({c for c, flag in zip(model.joint_classes, pitch) if flag})

    def configurations(self, models: list[MjcfStatics]) -> np.ndarray:
        """(D, S, J) joint positions, every design's own range read through one unit Latin hypercube."""
        classes = self._pitch_classes(models[0])
        rng = np.random.default_rng(self.seed)
        S = self.num_positions
        unit = (rng.permuted(np.tile(np.arange(S), (len(classes), 1)), axis=1).T
                + rng.uniform(size=(S, len(classes)))) / S                              # (S, K)
        ranges = _stack(models, "range")                                                # (D, J, 2)
        mid, half = ranges.mean(axis=-1), 0.5 * self.soft_limit_factor * (ranges[..., 1] - ranges[..., 0])
        lo = self.position_fraction * np.minimum(mid - half, 0.0)
        hi = self.position_fraction * np.maximum(mid + half, 0.0)
        q = np.zeros((len(models), S, ranges.shape[1]))
        for k, name in enumerate(classes):
            cols = [j for j, c in enumerate(models[0].joint_classes) if c == name]
            q[:, :, cols] = (lo[:, None, cols] + unit[None, :, k, None] * (hi - lo)[:, None, cols])
        return q

    def evaluate(self, xml_files: list[str]) -> dict:
        """Every quantity the filter decides on, for every design of *xml_files*."""
        models = [MjcfStatics(path) for path in xml_files]
        if any(m.topology() != models[0].topology() for m in models[1:]):
            raise ValueError("the documents of one population must share one topology")
        q = self.configurations(models)
        stance = standing_statics(models, q)
        limit = _stack(models, "limit")                                                 # (D, J)
        load = np.abs(stance["torque"]) / np.maximum(limit[:, None, :], 1e-9)           # (D, S, J)
        posed = stance["coplanar"] & stance["upright"] & (stance["clearance"] > 0.0) & stance["supported"]
        held = load.max(axis=-1)[..., None] <= self.ceilings                           # (D, S, T)
        failed = ~(posed[..., None] & held)
        return {
            "q": q, "posed": posed, "load": load, "failed": failed,
            "failure_fraction": failed.mean(axis=(1, 2)), **stance,
        }

    def filter(self, xml_files: list[str]) -> list[int]:
        if not xml_files:
            return []
        result = self.evaluate(xml_files)
        fraction = result["failure_fraction"]
        refused = np.flatnonzero(fraction > self.max_failure_fraction)
        self.statistics = {
            "designs": len(xml_files),
            "refused": int(refused.size),
            "failure_fraction_mean": float(fraction.mean()),
            "failure_fraction_max": float(fraction.max()),
            "unposed_fraction_mean": float(1.0 - result["posed"].mean()),
            "peak_load_median": float(np.median(result["load"].max(axis=-1))),
        }
        return refused.tolist()


    def scores(self, xml_files: list[str]) -> dict[str, list[float]]:
        if not xml_files:
            return {}
        result = self.evaluate(xml_files)
        return {
            "standing_failure_fraction": result["failure_fraction"].tolist(),
            "standing_unposed_fraction": (1.0 - result["posed"].mean(axis=1)).tolist(),
            "standing_peak_load_median": np.median(result["load"].max(axis=-1), axis=1).tolist(),
        }



# ---------------------------------------------------------------------------
# Actuator adequacy filter
# ---------------------------------------------------------------------------


class ActuatorAdequacyFilter(FilterBase):
    """Refuse a design whose picked actuators cannot carry the body they drive under the task's fixed gains.

    Robot agnostic. Every quantity is read per joint class from the emitted MJCF, the classes being whatever the
    family's builder sets on its joints (abad, hip, knee for the quadruped, abad, hip, knee, ankle for the biped),
    and every threshold is a dictionary keyed by joint class, a class it does not name being left untested. No
    family's numbers are built in, each family's CMAESCfg.filter_params supplying its own.

    Three tests per joint class, each read at the nominal pose q = 0, where standing_statics balances the weight on
    every ground contact of the document.
      torque: the static torque times peak_factor, the gait's peak foot force over the all contact static share,
              against the motor's forcerange.
      sag:    the static torque over the class's stiffness, the steady state PD deflection the policy must cancel
              with an action offset, the gains being the environment's and not the pick's.
      speed:  the picked velocity_limit, read from the design record key "<class>_actuator_velocity_limit",
              against min_velocity. A record without the key passes, so the test needs the design record.
    See /ws/plans/quadruped_copt_improvement2.md section 6.1.

    Args:
        family: The design generator family, unused by the tests.
        stiffness: Kp of each joint class as the environment configures it, required by max_sag_rad.
        peak_factor: Peak foot force of the gait the policy is expected to learn over the static share.
        max_peak_load: Largest peak torque as a fraction of forcerange, per joint class.
        max_sag_rad: Largest static PD deflection in radians, per joint class.
        min_velocity: Smallest velocity_limit in rad/s, per joint class.
    """

    uses_records = True

    def __init__(
        self,
        family: str,
        stiffness: dict | None = None,
        peak_factor: float = 1.0,
        max_peak_load: dict | None = None,
        max_sag_rad: dict | None = None,
        min_velocity: dict | None = None,
    ) -> None:
        super().__init__(family)
        self.stiffness = {k: float(v) for k, v in (stiffness or {}).items()}
        self.peak_factor = float(peak_factor)
        self.max_peak_load = {k: float(v) for k, v in (max_peak_load or {}).items()}
        self.max_sag_rad = {k: float(v) for k, v in (max_sag_rad or {}).items()}
        self.min_velocity = {k: float(v) for k, v in (min_velocity or {}).items()}
        missing = sorted(set(self.max_sag_rad) - set(self.stiffness))
        if missing:
            raise ValueError(f"max_sag_rad names joint classes {missing} that stiffness does not give a Kp for")

    def evaluate(self, xml_files: list[str], records: list[dict] | None = None) -> dict:
        """Every quantity the filter decides on, per design and joint class."""
        models = [MjcfStatics(path) for path in xml_files]
        if any(m.topology() != models[0].topology() for m in models[1:]):
            raise ValueError("the documents of one population must share one topology")
        stance = standing_statics(models, np.zeros((len(models), 1, len(models[0].joint_names))))
        tau = np.abs(stance["torque"][:, 0])                                            # (D, J)
        limit = _stack(models, "limit")                                                 # (D, J)
        classes = np.array(models[0].joint_classes)
        out, failed = {}, np.zeros(len(models), dtype=bool)
        for name in sorted(set(classes)):
            cols = classes == name
            load = (self.peak_factor * tau[:, cols] / np.maximum(limit[:, cols], 1e-9)).max(axis=1)
            out[f"actuator_peak_load_{name}"] = load
            if name in self.max_peak_load:
                failed |= load > self.max_peak_load[name]
            if name in self.stiffness:
                sag = tau[:, cols].max(axis=1) / self.stiffness[name]
                out[f"actuator_sag_rad_{name}"] = sag
                if name in self.max_sag_rad:
                    failed |= sag > self.max_sag_rad[name]
            if name in self.min_velocity and records is not None:
                speed = np.array([float(r.get(f"{name}_actuator_velocity_limit") or np.inf) for r in records])
                out[f"actuator_velocity_limit_{name}"] = speed
                failed |= speed < self.min_velocity[name]
        out["failed"] = failed
        return out

    def filter(self, xml_files: list[str], records: list[dict] | None = None) -> list[int]:
        if not xml_files:
            return []
        result = self.evaluate(xml_files, records)
        refused = np.flatnonzero(result["failed"])
        self.statistics = {"designs": len(xml_files), "refused": int(refused.size)}
        return refused.tolist()

    def scores(self, xml_files: list[str]) -> dict[str, list[float]]:
        if not xml_files:
            return {}
        result = self.evaluate(xml_files)
        return {k: np.asarray(v, dtype=float).tolist() for k, v in result.items() if k != "failed"}


FILTERS: dict[str, type[FilterBase]] = {
    "StandingFilter": StandingFilter,
    "ActuatorAdequacyFilter": ActuatorAdequacyFilter,
}


def make_filters(names: list[str], family: str, params: dict | None = None) -> list[FilterBase]:
    """The filters *names* name, each a key of FILTERS or a "module:Class" path, built with *params*[name]."""
    filters = []
    for name in names:
        if name in FILTERS:
            cls = FILTERS[name]
        else:
            from isaaclab.utils.string import string_to_callable

            cls = string_to_callable(name)
        if not (isinstance(cls, type) and issubclass(cls, FilterBase)):
            raise ValueError(f"filter {name!r} resolves to {cls!r}, not a FilterBase subclass")
        filters.append(cls(family, **dict((params or {}).get(name, {}))))
    return filters
