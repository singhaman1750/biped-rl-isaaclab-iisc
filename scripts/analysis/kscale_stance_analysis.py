"""Stance load analysis for a sole footed biped.

The companion script ``kscale_physical_analysis.py`` computes the effective inertia of each
joint's distal subtree about its own axis, which is the inertia a leg presents while swinging
freely in the air. That is not the quantity which decides whether a robot can stand up. With the
foot planted the joint reacts the ground reaction force, and the static torque it must hold is
set by the body weight and a moment arm rather than by the limb's inertia.

This script supplies the three quantities that follow from that observation. The static stance
torque at each joint, obtained by balancing the distal free body against the ground reaction
force at the centre of pressure. The settled configuration, obtained by minimising the sum of
the gravitational and the proportional spring potential energy with the sole planted, which
answers the question of where the robot actually comes to rest under its own weight. And the
stance reflected inertia, being the body mass seen through the joint's own Jacobian, which is
the inertia the derivative term should be sized against.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kscale_sole_analysis import forward_kinematics, parse_urdf, rpy_to_matrix  # noqa: E402

GRAVITY = 9.80665

DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "environments", "environments", "assets", "urdf", "solefoot", "kscale", "kscale.urdf",
)

# Mirrors init_state.joint_pos at assets/config/kscale_identified_cfg.py.
# Updated 2026-09-16 to the published K-Scale K-Bot standing posture adopted by
# environments/environments/assets/config/kscale_identified_cfg.py on that date, per section 9.1
# of plans/kscale_actuator_limits_fix.md. It replaces the shallower knee 0.4 and ankle -0.3, and
# it closes the sagittal chain exactly as that pose did, the standing height falling from
# 0.77161 m to 0.73540 m. Mirror any further change here and in the asset configuration together.
NOMINAL_POSE = {
    "right_hip_pitch_04": -0.20369, "left_hip_pitch_04": 0.20369,
    "right_hip_roll_03": 0.042951, "left_hip_roll_03": -0.042951,
    "right_hip_yaw_03": 0.0, "left_hip_yaw_03": 0.0,
    "right_knee_04": 0.51239, "left_knee_04": 0.51239,
    "right_foot_pitch_02": -0.3087, "left_foot_pitch_02": -0.3087,
    "right_foot_roll_02": 0.041254, "left_foot_roll_02": -0.041254,
}

# joint -> (stiffness, damping, armature, effort limit, lower limit, upper limit)
#
# The calculated set is the one derived in KScale.md sections 6 and 11 from the single support
# hold torque, the gravitational stiffness of the stance mode and the swing Nyquist ceiling, at
# the action scale of 0.25 the configuration carries. The identified set comes from actuator system
# identification rather than from derivation. Armature, effort limits and joint travel are
# common to both, only the stiffness and the damping differing.
CALCULATED_GAINS = {
    "right_hip_pitch_04": (200.0, 21.5, 0.010, 120.0, -2.21657, 1.04720),
    "right_hip_roll_03": (250.0, 28.0, 0.010, 120.0, -2.26893, 0.20944),
    "right_hip_yaw_03": (100.0, 3.3, 0.010, 120.0, -1.57080, 1.57080),
    "right_knee_04": (300.0, 12.0, 0.015, 120.0, 0.0, 2.70526),
    "right_foot_pitch_02": (120.0, 1.3, 0.005, 34.0, -0.87267, 0.52360),
    "right_foot_roll_02": (120.0, 1.0, 0.005, 34.0, -0.26180, 0.26180),
}

# The set obtained from actuator system identification, recorded at KScale.md section 7. The
# effort and velocity limits are common to both sets and are the doubled values of KScale.md
# section 5.2, being the robstride_04 peak torque at the two `_04` joints and twice the
# robstride_03 and robstride_02 peaks elsewhere.
IDENTIFIED_GAINS = {
    "right_hip_pitch_04": (200.0, 8.0, 0.010, 120.0, -2.21657, 1.04720),
    "right_hip_roll_03": (250.0, 10.0, 0.010, 120.0, -2.26893, 0.20944),
    "right_hip_yaw_03": (500.0, 18.0, 0.010, 120.0, -1.57080, 1.57080),
    "right_knee_04": (300.0, 10.0, 0.015, 120.0, 0.0, 2.70526),
    "right_foot_pitch_02": (170.0, 9.0, 0.005, 34.0, -0.87267, 0.52360),
    "right_foot_roll_02": (170.0, 9.0, 0.005, 34.0, -0.26180, 0.26180),
}

# The set standing in environments/environments/assets/config/kscale_identified_cfg.py as of
# 2026-09-17, being the calculated damping of section 12 paired with the PUBLISHED K-Scale effort
# limits rather than the doubled figures the two sets above carry. The effort column is the
# robstride soft limit, 84.0 at the two `_04` joints, 42.0 at the two `_03` and 11.9 at the two
# `_02`, read from /ws/context/kscale-opensource.md section 6. Added as a third set rather than by
# editing either of the above, so that the no argument invocation continues to reproduce the tables
# of KScale.md exactly.
CONFIGURED_GAINS = {
    "right_hip_pitch_04": (200.0, 21.5, 0.010, 84.0, -2.21657, 1.04720),
    "right_hip_roll_03": (250.0, 28.0, 0.010, 42.0, -2.26893, 0.20944),
    "right_hip_yaw_03": (100.0, 3.5, 0.010, 42.0, -1.57080, 1.57080),
    "right_knee_04": (300.0, 12.0, 0.015, 84.0, 0.0, 2.70526),
    "right_foot_pitch_02": (120.0, 5.0, 0.005, 11.9, -0.87267, 0.52360),
    "right_foot_roll_02": (120.0, 4.0, 0.005, 11.9, -0.26180, 0.26180),
}

GAIN_SETS = {"calculated": CALCULATED_GAINS, "identified": IDENTIFIED_GAINS,
             "configured": CONFIGURED_GAINS}

# The module level functions below read GAINS as a global. It is rebound by main() when a
# gain set other than the default is asked for, so that the no argument invocation reproduces
# the tables of KScale.md exactly as it did before the identified set was added.
GAINS = CALCULATED_GAINS

# Measured 90th percentile of the absolute applied joint torque, and the fraction of samples
# at or above 99 per cent of the effort limit, taken from the replay of 2026-08-28_04-50-51 at
# seed 42 on 2026-09-04, right leg. A joint that saturates does not report its disturbance, the
# clipped torque being a lower bound on the torque actually demanded, so the disturbance
# criterion below is reported for the unsaturated joints alone.
# joint -> (p90 |tau| in Nm, saturated fraction)
MEASURED_DISTURBANCE = {
    "right_hip_pitch_04": (60.338, 0.464),
    "right_hip_roll_03": (60.305, 0.493),
    "right_hip_yaw_03": (12.134, 0.000),
    "right_knee_04": (28.106, 0.003),
    "right_foot_pitch_02": (17.118, 0.501),
    "right_foot_roll_02": (17.027, 0.509),
}

# Ninety fifth percentile of the absolute joint speed, right leg, from the replay of
# 2026-08-28_04-50-51 at seed 42 on 2026-09-04. The damping ceiling of KScale.md section 11.1
# is the effort limit divided by this figure, being the largest damping the actuator sustains
# at the speed the joint is actually asked to reach.
MEASURED_SPEED_P95 = {
    "right_hip_pitch_04": 3.019,
    "right_hip_roll_03": 0.927,
    "right_hip_yaw_03": 5.159,
    "right_knee_04": 5.791,
    "right_foot_pitch_02": 9.561,
    "right_foot_roll_02": 9.592,
}

# joint -> (original velocity limit, doubled velocity limit) in rad/s, KScale.md section 5.2.
VELOCITY_LIMITS = {
    "right_hip_pitch_04": (20.0, 40.0),
    "right_hip_roll_03": (20.0, 40.0),
    "right_hip_yaw_03": (20.0, 40.0),
    "right_knee_04": (20.0, 40.0),
    "right_foot_pitch_02": (10.0, 20.0),
    "right_foot_roll_02": (10.0, 20.0),
}

# The velocity_limit standing in the asset configuration as of 2026-09-17, being the
# published declared maximum velocity of each robstride class, against the
# velocity_limit_sim the solver enforces, being that class's firmware ceiling.
CONFIGURED_VELOCITY = {
    "right_hip_pitch_04": (17.488, 15.0),
    "right_hip_roll_03": (18.849, 20.0),
    "right_hip_yaw_03": (18.849, 20.0),
    "right_knee_04": (17.488, 15.0),
    "right_foot_pitch_02": (37.699, 44.0),
    "right_foot_roll_02": (37.699, 44.0),
}

# The physics timestep the KScale tasks configure, at which the explicit actuator model is
# evaluated, and the extremes of the startup randomisation of KScale.md section 4.8.
PHYSICS_DT = 0.005
RAND_DAMPING_MAX = 1.1
RAND_INERTIA_MIN = 0.95

ANKLE_ROLL_JOINT = "right_foot_roll_02"
SOLE_DEPTH = 0.0430          # metres below the ankle roll origin, along the foot frame's plus y
SOLE_NORMAL = (0.0, 1.0, 0.0)
SOLE_HALF_WIDTH = 0.0846 / 2

# The three sagittal joints, in the order the settled pose solver varies them.
SAGITTAL = ("right_hip_pitch_04", "right_knee_04", "right_foot_pitch_02")


def parse_inertials(path: str) -> dict[str, tuple[float, np.ndarray, np.ndarray]]:
    """Mass, centre of mass offset and inertia tensor of every link with an inertial block.

    The tensor is rotated out of the inertial frame into the link frame by ``R I R^T``, since
    these inertial frames carry a non trivial rpy and are not aligned with the link.
    """
    out: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for link in ET.parse(path).getroot().iter("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        origin = inertial.find("origin")
        text = origin.get("xyz") if origin is not None else None
        com = np.array([float(v) for v in (text or "0 0 0").split()])
        rpy_text = origin.get("rpy") if origin is not None else None
        rpy = np.array([float(v) for v in (rpy_text or "0 0 0").split()])
        node = inertial.find("inertia")
        ixx, ixy, ixz = float(node.get("ixx")), float(node.get("ixy")), float(node.get("ixz"))
        iyy, iyz, izz = float(node.get("iyy")), float(node.get("iyz")), float(node.get("izz"))
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        rotation = rpy_to_matrix(*rpy)
        out[link.get("name", "")] = (
            float(inertial.find("mass").get("value")), com, rotation @ tensor @ rotation.T,
        )
    return out


def descendants(joints, link: str) -> set[str]:
    """Every link at or below the given link in the kinematic tree."""
    children: dict[str, list[str]] = {}
    for joint in joints.values():
        children.setdefault(joint.parent, []).append(joint.child)
    seen: set[str] = set()
    stack = [link]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(children.get(node, []))
    return seen


def joint_axis(joints, rot, name: str) -> np.ndarray:
    """The joint's rotation axis expressed in the root frame, normalised."""
    joint = joints[name]
    axis = rot[joint.parent] @ joint.origin_rot @ joint.axis
    return axis / np.linalg.norm(axis)


class Model:
    """Forward kinematics, mass properties and ground contact for one URDF."""

    def __init__(self, urdf: str):
        self.links, self.joints, self.root = parse_urdf(urdf)
        self.inertials = parse_inertials(urdf)
        self.mass = {n: m for n, (m, _, _) in self.inertials.items()}
        self.total_mass = sum(self.mass.values())
        self.weight = self.total_mass * GRAVITY
        self.foot = self.joints[ANKLE_ROLL_JOINT].child

    def pose_from(self, hip: float, knee: float, ankle: float) -> dict[str, float]:
        """Both legs at the mirrored equivalent of the three sagittal angles given.

        The hip pitch axes are anti parallel between the legs, so the left hip takes the
        negation of the right for one physical posture, while the knee and the ankle pitch
        axes are parallel and take the same sign on both legs.
        """
        pose = dict(NOMINAL_POSE)
        pose.update({
            "right_hip_pitch_04": hip, "left_hip_pitch_04": -hip,
            "right_knee_04": knee, "left_knee_04": knee,
            "right_foot_pitch_02": ankle, "left_foot_pitch_02": ankle,
        })
        return pose

    def kinematics(self, pose: dict[str, float]):
        """Link rotations and positions, plus the height shift that plants the sole."""
        rot, pos = forward_kinematics(self.joints, self.root, pose)
        sole = pos[self.foot] + rot[self.foot] @ (SOLE_DEPTH * np.array(SOLE_NORMAL))
        return rot, pos, -sole[2]

    def com_world(self, rot, pos, shift: float) -> np.ndarray:
        """Whole body centre of mass with the sole resting on the ground plane."""
        total = sum(
            self.mass[n] * (pos[n] + rot[n] @ com)
            for n, (_, com, _) in self.inertials.items() if n in pos
        )
        return total / self.total_mass + np.array([0.0, 0.0, shift])

    def base_height(self, pose: dict[str, float]) -> float:
        rot, pos, shift = self.kinematics(pose)
        return pos[self.root][2] + shift


def stance_torques(model: Model, pose: dict[str, float], support: float) -> dict[str, float]:
    """Static torque at each right leg joint with the sole planted.

    ``support`` is the fraction of the body weight borne by this foot, being one half in
    symmetric double support and one in single support. The centre of pressure is placed
    directly beneath the ankle roll axis, which is the balanced case.
    """
    rot, pos, shift = model.kinematics(pose)
    ground = np.array([pos[model.foot][0], pos[model.foot][1], -shift])
    grf = np.array([0.0, 0.0, model.weight * support])
    gravity = np.array([0.0, 0.0, -GRAVITY])
    out: dict[str, float] = {}
    for name in GAINS:
        origin = pos[model.joints[name].child]
        axis = joint_axis(model.joints, rot, name)
        torque = float(axis @ np.cross(ground - origin, grf))
        for link in descendants(model.joints, model.joints[name].child):
            if link not in model.inertials:
                continue
            mass, com, _ = model.inertials[link]
            world = pos[link] + rot[link] @ com
            torque += mass * float(axis @ np.cross(world - origin, gravity))
        out[name] = torque
    return out


def swing_inertia(model: Model, name: str) -> float:
    """Distal subtree inertia about the joint axis at the nominal pose, armature included.

    Each link contributes its own inertia resolved along the axis, ``a . R I R^T . a``, plus
    the parallel axis term ``m r_perp^2``. The own inertia term is taken in full three
    dimensions rather than by reading a diagonal entry, because these inertial frames are not
    in general aligned with the axis acting upon them.
    """
    rot, pos, _ = model.kinematics(NOMINAL_POSE)
    origin = pos[model.joints[name].child]
    axis = joint_axis(model.joints, rot, name)
    total = 0.0
    for link in descendants(model.joints, model.joints[name].child):
        if link not in model.inertials:
            continue
        mass, com, tensor = model.inertials[link]
        world_tensor = rot[link] @ tensor @ rot[link].T
        offset = pos[link] + rot[link] @ com - origin
        perpendicular = offset - float(offset @ axis) * axis
        total += float(axis @ world_tensor @ axis) + mass * float(perpendicular @ perpendicular)
    return total + GAINS[name][2]


def settled_pose(model: Model, stiffness: dict[str, float]):
    """Minimise gravitational plus spring potential energy with the sole planted.

    Both legs are constrained to the same posture, so each spring term is counted twice, and
    the constant factor of one half in the spring energy is dropped since it scales the whole
    objective. The result is where the robot comes to rest under its own weight.
    """
    q0 = np.array([NOMINAL_POSE[j] for j in SAGITTAL])
    bounds = [(GAINS[j][4], GAINS[j][5]) for j in SAGITTAL]
    gains = np.array([stiffness[j] for j in SAGITTAL])

    def energy(q: np.ndarray) -> float:
        rot, pos, shift = model.kinematics(model.pose_from(*q))
        potential = model.total_mass * GRAVITY * model.com_world(rot, pos, shift)[2]
        return potential + float(np.sum(gains * (q - q0) ** 2))

    best = None
    for start in (q0, np.array([-0.1, 0.8, -0.5]), np.array([-0.3, 1.5, -0.8])):
        result = minimize(energy, start, bounds=bounds, method="L-BFGS-B")
        if best is None or result.fun < best.fun:
            best = result
    return best.x, model.base_height(model.pose_from(*best.x))


def stance_inertia(model: Model) -> dict[str, float]:
    """Swing inertia plus the body mass reflected through the joint's own Jacobian.

    A joint that lifts the whole body accelerates far more than its own distal subtree. The
    reflected term is ``M (d z_com / d q)^2``, taken by central difference at the nominal pose,
    and it applies only to the sagittal joints, the remaining joints not raising the body to
    first order in a symmetric stance.
    """
    q0 = np.array([NOMINAL_POSE[j] for j in SAGITTAL])
    step = 1e-4
    out: dict[str, float] = {}
    for name in GAINS:
        swing = swing_inertia(model, name)
        if name not in SAGITTAL:
            out[name] = swing
            continue
        index = SAGITTAL.index(name)
        heights = []
        for sign in (+1.0, -1.0):
            q = q0.copy()
            q[index] += sign * step
            rot, pos, shift = model.kinematics(model.pose_from(*q))
            heights.append(model.com_world(rot, pos, shift)[2])
        jacobian = (heights[0] - heights[1]) / (2 * step)
        out[name] = swing + model.total_mass * jacobian ** 2
    return out


def proximal_links(model: Model, name: str) -> set[str]:
    """Every link the joint moves when its own foot is planted, being the complement of the
    distal subtree. In single support the links below the joint are held by the ground and the
    links above it, the torso and the opposite leg, are what the joint accelerates."""
    return set(model.inertials) - descendants(model.joints, model.joints[name].child)


def stance_inertia_proximal(model: Model) -> dict[str, float]:
    """Inertia of the proximal assembly about the joint axis, armature included.

    This is the inertia the joint actually accelerates in single support, and it supersedes
    :func:`stance_inertia` for every purpose in KScale.md section 11. The older function
    reflects the body mass through ``M (d z_com / d q)^2`` alone, which captures the vertical
    lifting term and misses the horizontal rotational one entirely, so it reports the swing
    value at the lateral and vertical axes where the true figure is up to nine hundred times
    larger. It is retained unchanged so that tables computed before 2026-09-04 reproduce.
    """
    rot, pos, _ = model.kinematics(NOMINAL_POSE)
    out: dict[str, float] = {}
    for name in GAINS:
        axis = joint_axis(model.joints, rot, name)
        origin = pos[model.joints[name].child]
        total = 0.0
        for link in proximal_links(model, name):
            mass, com, tensor = model.inertials[link]
            world_tensor = rot[link] @ tensor @ rot[link].T
            offset = pos[link] + rot[link] @ com - origin
            perpendicular = offset - float(offset @ axis) * axis
            total += float(axis @ world_tensor @ axis) + mass * float(perpendicular @ perpendicular)
        out[name] = total + GAINS[name][2]
    return out


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def single_support_gravity(model: Model, step: float = 1e-4):
    """Hold torque and gravitational stiffness of the proximal assembly about each joint axis.

    With one foot planted the joint carries the weight of everything above it, and rotating it
    changes that assembly's potential energy. The first derivative is the torque the joint must
    supply to hold the nominal pose on one foot and the second is the stiffness gravity itself
    contributes, reported with the sign convention that a POSITIVE value is destabilising, the
    assembly standing on a potential hill rather than sitting in a well.
    """
    rot, pos, _ = model.kinematics(NOMINAL_POSE)
    hold: dict[str, float] = {}
    gravity: dict[str, float] = {}
    for name in GAINS:
        axis = joint_axis(model.joints, rot, name)
        origin = pos[model.joints[name].child]
        links = [(model.inertials[l][0], pos[l] + rot[l] @ model.inertials[l][1])
                 for l in proximal_links(model, name)]
        energies = []
        for angle in (step, 0.0, -step):
            rotation = _axis_rotation(axis, angle)
            energies.append(GRAVITY * sum(m * (origin + rotation @ (p - origin))[2] for m, p in links))
        hold[name] = abs((energies[0] - energies[2]) / (2 * step))
        gravity[name] = -(energies[0] - 2 * energies[1] + energies[2]) / step ** 2
    return hold, gravity


def swing_spectral_radius(K: float, D: float, inertia: float, dt: float = PHYSICS_DT) -> float:
    """Spectral radius of the semi implicit update of an explicitly integrated PD joint.

    KScale.md section 11.1.1. The actuator model of ``environments/actuators/actuator_pd.py``
    hands the physics engine a torque rather than a position target, so the proportional and
    derivative terms are integrated explicitly at the physics timestep. Writing the velocity
    update on the torque computed from the state at the start of the step and the position
    update on the resulting velocity gives the companion matrix below, whose spectral radius
    exceeding unity means the free response grows rather than decays.

    The inertia is the SWING inertia, since that is what the joint meets whenever its foot is
    off the ground, and the commanded gait spends 0.76 of its cycle in single support.
    """
    a = dt * dt * K / inertia
    b = dt * D / inertia
    companion = np.array([[1.0 - a, dt * (1.0 - b)], [-a / dt, 1.0 - b]])
    return float(max(abs(np.linalg.eigvals(companion))))


def damping_ceilings(K: float, inertia_swing: float, inertia_stance: float,
                     effort: float, speed_p95: float, zeta: float = 0.70,
                     dt: float = PHYSICS_DT) -> dict[str, float]:
    """The four candidate damping values of KScale.md section 11.1, smallest of which is taken.

    The stance target is the design preference, the effort ceiling is what the actuator can
    deliver at the speed the joint is measured to reach, the swing critical ceiling holds the
    swing damping ratio at or below unity, and the integration ceiling holds the stability
    number of ``swing_spectral_radius`` at or below unity at the worst corner of the startup
    randomisation envelope of section 4.8.
    """
    margin = RAND_DAMPING_MAX / RAND_INERTIA_MIN
    return {
        "stance target": 2.0 * zeta * np.sqrt(K * inertia_stance),
        "effort limit": effort / speed_p95,
        "swing critical": 2.0 * np.sqrt(K * inertia_swing),
        "swing integration": inertia_swing / (dt * margin),
    }


def randomised_divergence_fraction(K: float, D: float, link_inertia: float, armature: float,
                                   samples: int = 200000, seed: int = 0,
                                   dt: float = PHYSICS_DT) -> float:
    """Fraction of randomised environments whose swing mode spectral radius exceeds unity.

    The startup events of KScale.md section 4.8 scale stiffness and damping over 0.9 to 1.1 and
    the link mass and the armature over 0.95 to 1.05, independently per environment, so a gain
    set sitting near the bound at its nominal values crosses it in some fraction of them.
    """
    rng = np.random.default_rng(seed)
    inertia = (link_inertia * rng.uniform(0.95, 1.05, samples)
               + armature * rng.uniform(0.95, 1.05, samples))
    a = dt * dt * (K * rng.uniform(0.9, 1.1, samples)) / inertia
    b = dt * (D * rng.uniform(0.9, 1.1, samples)) / inertia
    trace, det = 2.0 - a - b, 1.0 - b
    disc = trace * trace - 4.0 * det
    root = np.sqrt(np.abs(disc))
    real = np.maximum(np.abs((trace + root) / 2.0), np.abs((trace - root) / 2.0))
    radius = np.where(disc >= 0.0, real, np.sqrt(np.abs(det)))
    return float((radius > 1.0).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Stance load analysis for a sole footed biped.")
    parser.add_argument("--urdf", default=os.path.normpath(DEFAULT_URDF))
    parser.add_argument("--budget", type=float, default=0.05,
                        help="acceptable static sag in radians before a stiffness is flagged")
    parser.add_argument("--gains", choices=sorted(GAIN_SETS), default="calculated",
                        help="which actuator gain set to analyse")
    parser.add_argument("--action-scale", type=float, default=0.25,
                        help="joint position action scale, for the saturation check")
    parser.add_argument("--control-period", type=float, default=0.02,
                        help="control period in seconds, for the Nyquist check")
    args = parser.parse_args()

    global GAINS
    GAINS = GAIN_SETS[args.gains]

    model = Model(args.urdf)
    print(f"gain set      {args.gains}")
    print(f"URDF          {args.urdf}")
    print(f"total mass    {model.total_mass:.4f} kg, weight {model.weight:.2f} N")
    print(f"nominal pose  base height {model.base_height(NOMINAL_POSE):.5f} m")

    double = stance_torques(model, NOMINAL_POSE, 0.5)
    single = stance_torques(model, NOMINAL_POSE, 1.0)
    print("\nStatic stance load, centre of pressure beneath the ankle roll axis")
    print(f"{'joint':22}{'tau_2sup':>10}{'tau_1sup':>10}{'K':>8}{'sag_2sup':>10}"
          f"{'sag_1sup':>10}{'effort':>9}{'verdict':>22}")
    for name, (K, _D, _a, effort, lo, hi) in GAINS.items():
        t2, t1 = double[name], single[name]
        sag2, sag1 = abs(t2) / K, abs(t1) / K
        if max(abs(t2), abs(t1)) > effort:
            verdict = "EXCEEDS EFFORT LIMIT"
        elif sag1 > (hi - lo):
            verdict = "DEFLECTS PAST RANGE"
        elif sag2 > args.budget:
            verdict = "sag over budget"
        else:
            verdict = "ok"
        print(f"{name:22}{t2:10.3f}{t1:10.3f}{K:8.1f}{sag2:10.3f}{sag1:10.3f}"
              f"{effort:9.1f}{verdict:>22}")

    q, height = settled_pose(model, {n: g[0] for n, g in GAINS.items()})
    nominal = model.base_height(NOMINAL_POSE)
    print("\nSettled configuration under body weight, symmetric double support")
    print(f"  hip pitch {q[0]:+.4f}  knee {q[1]:.4f}  ankle pitch {q[2]:+.4f}")
    print(f"  base height {height:.5f} m against a nominal {nominal:.5f} m, "
          f"sag {(nominal - height) * 1000:.1f} mm")

    hold, gravity = single_support_gravity(model)
    print("\nSingle support hold torque and gravitational stiffness about each joint axis")
    print(f"{'joint':22}{'M_prox':>9}{'tau_hold':>10}{'k_grav':>9}{'effort':>8}"
          f"{'sag at K':>10}{'verdict':>24}")
    for name, (K, _D, _a, effort, _lo, _hi) in GAINS.items():
        if hold[name] > effort:
            verdict = "HOLD EXCEEDS EFFORT LIMIT"
        elif gravity[name] > K:
            verdict = "STANCE MODE UNSTABLE"
        else:
            verdict = "ok"
        mass = sum(model.mass[l] for l in proximal_links(model, name))
        print(f"{name:22}{mass:9.2f}{hold[name]:10.3f}{gravity[name]:9.2f}{effort:8.1f}"
              f"{hold[name] / K:10.3f}{verdict:>26}")
    print("  A positive gravitational stiffness is destabilising, so a joint whose stiffness")
    print("  does not exceed it cannot hold the assembly above it upright at any deflection.")

    print("\nStance reflected inertia against swing inertia, and the damping each implies")
    reflected = stance_inertia_proximal(model)
    print(f"{'joint':22}{'I_swing':>10}{'I_stance':>10}{'ratio':>8}{'K':>7}{'D':>7}"
          f"{'z_swing':>9}{'z_stance':>10}{'w_stance':>10}")
    for name, (K, D, _a, _e, _lo, _hi) in GAINS.items():
        swing = swing_inertia(model, name)
        stance = reflected[name]
        print(f"{name:22}{swing:10.5f}{stance:10.5f}{stance / swing:8.2f}{K:7.1f}{D:7.2f}"
              f"{D / (2 * (K * swing) ** 0.5):9.3f}{D / (2 * (K * stance) ** 0.5):10.3f}"
              f"{(K / stance) ** 0.5:10.2f}")

    torque = (model.weight / 2) * SOLE_HALF_WIDTH
    K_roll, _, _, _, lo, hi = GAINS[ANKLE_ROLL_JOINT]
    print("\nDamping derivation, the stance target against the effort limited ceiling")
    print(f"{'joint':22}{'K':>7}{'I_stance':>10}{'D_z070':>9}{'qd_p95':>9}"
          f"{'D_max_eff':>11}{'D':>7}{'qd_sat':>9}{'binding':>18}")
    for name, (K, D, _a, effort, _lo, _hi) in GAINS.items():
        target = 2 * 0.70 * (K * reflected[name]) ** 0.5
        ceiling = effort / MEASURED_SPEED_P95[name]
        print(f"{name:22}{K:7.1f}{reflected[name]:10.4f}{target:9.2f}"
              f"{MEASURED_SPEED_P95[name]:9.3f}{ceiling:11.2f}{D:7.2f}{effort / D:9.2f}"
              f"{('effort limit' if ceiling < target else 'stance target'):>18}")

    print("\nAnkle roll lateral criterion, centre of pressure held at the edge of the sole")
    print(f"  demand {torque:.2f} Nm, deflection at K = {K_roll:.1f} is "
          f"{torque / K_roll:.3f} rad against a travel of {lo:+.4f} to {hi:+.4f} rad")
    print(f"  minimum stiffness to stay inside the travel "
          f"{torque / max(abs(lo), abs(hi)):.2f} Nm/rad")

    nyquist = np.pi / args.control_period
    print(f"\nSaturation headroom at an action scale of {args.action_scale:.2f} rad, "
          f"and bandwidth against a Nyquist bound of {nyquist:.2f} rad/s")
    print(f"{'joint':22}{'K':>7}{'K*scale':>9}{'tau_2sup':>10}{'sum':>9}{'effort':>8}"
          f"{'headroom':>10}{'q_sat':>8}{'w_stance':>10}{'w/nyq':>8}")
    for name, (K, _D, _a, effort, _lo, _hi) in GAINS.items():
        commanded = K * args.action_scale
        total = commanded + abs(double[name])
        w = (K / reflected[name]) ** 0.5
        print(f"{name:22}{K:7.1f}{commanded:9.2f}{abs(double[name]):10.3f}{total:9.2f}"
              f"{effort:8.1f}{100 * (effort - total) / effort:9.1f}%{effort / K:8.3f}"
              f"{w:10.2f}{w / nyquist:8.3f}")

    print(f"\nFeasible stiffness band at an action scale of {args.action_scale:.2f} rad, "
          f"the load floor taken at a sag budget of {args.budget:.2f} rad")
    print(f"{'joint':22}{'K_min_load':>12}{'K_min_grav':>12}{'K_auth':>9}"
          f"{'K_max_nyq':>11}{'K_max_noclip':>14}{'K':>7}{'verdict':>18}")
    for name, (K, _D, _a, effort, _lo, _hi) in GAINS.items():
        floor = abs(double[name]) / args.budget
        gravity_floor = max(gravity[name], 0.0)
        authority = hold[name] / args.action_scale
        nyquist_ceiling = swing_inertia(model, name) * nyquist ** 2
        ceiling = (effort - abs(double[name])) / args.action_scale
        binding = max(floor, gravity_floor, authority)
        if binding > nyquist_ceiling:
            verdict = "BAND EMPTY"
        elif not binding <= K <= nyquist_ceiling:
            verdict = "K outside band"
        else:
            verdict = "ok"
        print(f"{name:22}{floor:12.1f}{gravity_floor:12.1f}{authority:9.1f}"
              f"{nyquist_ceiling:11.1f}{ceiling:14.1f}{K:7.1f}{verdict:>18}")
    print("  The no clip ceiling is reported as a diagnostic and is not a constraint. Where it")
    print("  falls below a floor the joint clips under a full scale action, which is accepted,")
    print("  a joint unable to hold its stance load failing always rather than occasionally.")

    print("\nWhy a stiffness is NOT sized from a logged torque, section 11.1")
    print(f"{'joint':22}{'p90 tau':>9}{'sat':>7}{'K':>7}{'dev at K':>10}"
          f"{'K@0.05':>9}{'K@0.10':>9}{'K@0.20':>9}{'note':>14}")
    for name, (K, _D, _a, _e, _lo, _hi) in GAINS.items():
        tau, sat = MEASURED_DISTURBANCE[name]
        note = "SATURATED" if sat > 0.05 else "valid"
        print(f"{name:22}{tau:9.2f}{sat:7.3f}{K:7.1f}{tau / K:10.3f}"
              f"{tau / 0.05:9.1f}{tau / 0.10:9.1f}{tau / 0.20:9.1f}{note:>14}")
    print("  The K columns are what a disturbance quotient would ask for and are reported as a")
    print("  diagnostic, not a requirement. A saturated joint reports a lower bound on its torque")
    print("  rather than a measurement of it, and at an unsaturated joint the logged torque is")
    print("  dominated by the policy's own commanded offset rather than by anything external, so")
    print("  neither case yields a disturbance. Section 11.1 gives the reconstruction that")
    print("  separates the two and the reason this table decides no gain in this document.")

    swing = {n: swing_inertia(model, n) for n in GAINS}
    stance = stance_inertia_proximal(model)
    print("\nDamping ceilings of section 11.1, the smallest of the four being taken")
    print(f"{'joint':22}{'K':>7}{'stance':>9}{'effort':>9}{'swing_cr':>10}{'integ':>8}"
          f"{'D':>7}{'D_cfg':>8}{'binding':>20}")
    for name, (K, D, _a, effort, _lo, _hi) in GAINS.items():
        c = damping_ceilings(K, swing[name], stance[name], effort, MEASURED_SPEED_P95[name])
        binding = min(c, key=c.get)
        print(f"{name:22}{K:7.1f}{c['stance target']:9.2f}{c['effort limit']:9.2f}"
              f"{c['swing critical']:10.2f}{c['swing integration']:8.2f}"
              f"{c[binding]:7.2f}{D:8.1f}{binding:>20}")

    print("\nSwing mode stability of section 11.1.1, explicit PD at a physics dt of "
          f"{PHYSICS_DT} s")
    print(f"{'joint':22}{'I_swing':>10}{'b_nom':>8}{'b_worst':>9}{'rho_nom':>9}"
          f"{'rho_worst':>11}{'P(diverge)':>12}{'verdict':>12}")
    margin = RAND_DAMPING_MAX / RAND_INERTIA_MIN
    worst_any = 0.0
    for name, (K, D, armature, _e, _lo, _hi) in GAINS.items():
        inertia = swing[name]
        rho_nom = swing_spectral_radius(K, D, inertia)
        rho_bad = swing_spectral_radius(K * RAND_DAMPING_MAX, D * RAND_DAMPING_MAX,
                                        inertia * RAND_INERTIA_MIN)
        frac = randomised_divergence_fraction(K, D, inertia - armature, armature)
        worst_any = max(worst_any, frac)
        verdict = "DIVERGENT" if rho_bad > 1.0 else "stable"
        print(f"{name:22}{inertia:10.5f}{D * PHYSICS_DT / inertia:8.3f}"
              f"{D * PHYSICS_DT * margin / inertia:9.3f}{rho_nom:9.4f}{rho_bad:11.4f}"
              f"{frac * 100:11.1f}%{verdict:>12}")
    print("  The stability number b is D dt / I_swing and the mode diverges above 2, oscillating")
    print("  at the integrator's Nyquist frequency above 1. The derivation of section 11.1 holds")
    print("  the worst randomised draw at or below 1, which places the nominal set at 0.86 of it.")
    print(f"  Worst per joint divergence fraction over the randomisation envelope, {worst_any * 100:.1f} per cent.")

    print("\nTorque speed derating of section 11.10, DCMotor at saturation_effort = effort_limit")
    print(f"{'joint':22}{'qd_p95':>9}{'v_old':>8}{'tau_old':>10}{'v_new':>8}{'tau_new':>10}{'gain':>8}")
    for name, (_K, _D, _a, effort, _lo, _hi) in GAINS.items():
        qd = MEASURED_SPEED_P95[name]
        v_old, v_new = VELOCITY_LIMITS[name]
        t_old = (effort / 2.0) * max(0.0, 1.0 - qd / v_old)
        t_new = effort * max(0.0, 1.0 - qd / v_new)
        print(f"{name:22}{qd:9.3f}{v_old:8.1f}{t_old:10.2f}{v_new:8.1f}{t_new:10.2f}"
              f"{t_new / max(t_old, 1e-9):7.1f}x")
    print("  The old columns are the original limits, half the effort and half the velocity of")
    print("  the doubled set of section 5.2, and the available torque is the four quadrant curve")
    print("  the DCMotor base class applies, falling linearly to zero at the velocity limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
