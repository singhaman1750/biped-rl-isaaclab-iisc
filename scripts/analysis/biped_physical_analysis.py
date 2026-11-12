"""Physical parameterisation of any biped, from its MJCF or URDF document and a shared configuration.

This script generalises ``kscale_physical_analysis.py``, which is bound to the KScale URDF, its
mesh feet and a pose written into its own source. Here the model is loaded through MuJoCo, which
reads both document formats, and everything robot specific comes from a JSON configuration that
the companion ``biped_stance_analysis.py`` shares, being the joints of each leg with their degrees
of freedom and roles, the foot bodies, the nominal pose set outside the link geometry, the
actuator parameters, the control timing and the reward targets. An example is
``biped_configs/tron1_sf.json``.

It reports, in order,

* the mass inventory, separating the trunk from the moving leg mass and the mass below the knee,
* the nominal pose geometry, being the sagittal angle of each segment, the knee included angle,
  the leg extension, the ankle lead over the hip, the standing height above the sole, the centre
  of mass against the sole and the sole depth, from which the height and foot reward parameters
  follow,
* the joint limits at the nominal pose, with the headroom to the soft limits in radians and in
  action units,
* the kneeling geometry, the base heights at which a non foot link and not the sole is the lowest
  point of the robot, with the base height penalty such a posture incurs,
* the swing effective inertia of each joint, read from the mass matrix diagonal, with the
  natural frequency, damping ratio and discrete stability the configured gains imply,
* the symmetry flip set and a self collision audit at the nominal pose.

Any further models given with ``--compare`` are analysed under the same configuration and the
headline quantities are tabulated side by side, which is how a generated document is checked
against the URDF it replaces.

Usage:
    python scripts/analysis/biped_physical_analysis.py --config scripts/analysis/biped_configs/tron1_sf.json \
        --compare environments/environments/assets/urdf/solefoot/tron1/base_robot.urdf \
                  environments/environments/assets/urdf/solefoot/tron1/base_robot_tron1.urdf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
GRAVITY = 9.81


def resolve(path: str) -> str:
    """A configuration path, taken relative to the repository root unless already absolute."""
    return path if os.path.isabs(path) or os.path.exists(path) else os.path.join(REPO_ROOT, path)


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_spec(path: str) -> mujoco.MjSpec:
    """A MuJoCo specification of the document, with a floating root whatever the format.

    MuJoCo welds a URDF's root to the world and fuses every fixed joint, which would discard the
    trunk and rename the links, so the URDF is rewritten first. Visual elements are dropped,
    collision meshes are resolved against the document's ``meshes`` directory and dropped when
    absent or in a format MuJoCo cannot read, static fusion is disabled and a free joint is given
    to the root body.
    """
    path = resolve(path)
    if not path.lower().endswith(".urdf"):
        return mujoco.MjSpec.from_file(path)
    root = ET.parse(path).getroot()
    here = os.path.dirname(path)
    for link in root.iter("link"):
        for elem in list(link):
            if elem.tag == "visual":
                link.remove(elem)
            elif elem.tag == "collision" and elem.find("geometry/mesh") is not None:
                mesh = elem.find("geometry/mesh")
                name = os.path.basename(mesh.get("filename", ""))
                local = os.path.join(here, "meshes", name)
                if os.path.splitext(name)[1].lower() in (".stl", ".obj") and os.path.exists(local):
                    mesh.set("filename", local)
                else:
                    link.remove(elem)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="true", balanceinertia="true")
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    spec.worldbody.first_body().add_freejoint()
    return spec


class Biped:
    """A compiled model together with the configuration's view of it.

    The model's own joint damping, armature and friction are zeroed, since Isaac Lab replaces them
    with the actuator configuration, and the configured armature is added back wherever an
    inertia is reported.
    """

    def __init__(self, path: str, cfg: dict, spec: mujoco.MjSpec | None = None):
        self.path, self.cfg = path, cfg
        self.model = (spec or load_spec(path)).compile()
        m = self.model
        m.dof_damping[:] = 0.0
        m.dof_armature[:] = 0.0
        m.dof_frictionloss[:] = 0.0
        self.data = mujoco.MjData(m)
        self.free_root = int(m.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)
        names = [m.body(i).name for i in range(m.nbody)]
        self.root = names.index(cfg["root_body"])
        self.legs: dict[str, dict] = {}
        for side, leg in cfg["legs"].items():
            joints = {}
            for name, info in leg["joints"].items():
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise KeyError(f"{path}: joint {name} not found")
                dof = {int(mujoco.mjtJoint.mjJNT_HINGE): 1, int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
                       int(mujoco.mjtJoint.mjJNT_BALL): 3}[int(m.jnt_type[jid])]
                if dof != info["dof"]:
                    raise ValueError(f"{name}: configuration states {info['dof']} dof, the model has {dof}")
                joints[name] = dict(info, id=jid, qpos=m.jnt_qposadr[jid], dofadr=m.jnt_dofadr[jid])
            if sum(j["dof"] for j in joints.values()) != cfg["dof_per_leg"]:
                raise ValueError(f"leg {side} does not carry the {cfg['dof_per_leg']} dof configured")
            foot = [i for i, n in enumerate(names) if re.fullmatch(leg["foot_body"], n)]
            if len(foot) != 1:
                raise KeyError(f"{path}: foot pattern {leg['foot_body']} matches {len(foot)} bodies")
            role = {j["role"]: n for n, j in joints.items()}
            self.legs[side] = dict(joints=joints, foot=foot[0], role=role)
        self.collision = np.flatnonzero(((m.geom_contype | m.geom_conaffinity) != 0) & (m.geom_bodyid != 0))
        self.foot_geoms = {s: [g for g in self.collision if m.geom_bodyid[g] in self.subtree(l["foot"])]
                           for s, l in self.legs.items()}

    # -- structure ---------------------------------------------------------------------------
    def subtree(self, body: int) -> set[int]:
        out = {body}
        for b in range(body + 1, self.model.nbody):
            if self.model.body_parentid[b] in out:
                out.add(b)
        return out

    def joint_names(self) -> list[str]:
        return [n for leg in self.legs.values() for n in leg["joints"]]

    def joint(self, name: str) -> dict:
        for leg in self.legs.values():
            if name in leg["joints"]:
                return leg["joints"][name]
        raise KeyError(name)

    def actuator(self, name: str) -> dict:
        return self.cfg["actuators"][self.joint(name)["group"]]

    def trunk_bodies(self) -> set[int]:
        """The root and every body welded to it, which move with the trunk."""
        m, out = self.model, {self.root}
        for b in range(self.root + 1, m.nbody):
            if m.body_parentid[b] in out and m.body_jntnum[b] == 0:
                out.add(b)
        return out

    # -- kinematics --------------------------------------------------------------------------
    def set_pose(self, pose: dict[str, float] | None = None, offsets: dict[str, float] | None = None,
                 base_height: float = 0.0) -> None:
        """Place the root at the origin, level, with the configured nominal pose plus offsets."""
        m, d = self.model, self.data
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0.0
        if self.free_root:
            d.qpos[0:3] = (0.0, 0.0, base_height)
            d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        else:
            d.qpos[0] = base_height
        pose = dict(self.cfg["nominal_pose"]) if pose is None else pose
        for name, value in pose.items():
            d.qpos[self.joint(name)["qpos"]] = value
        for name, value in (offsets or {}).items():
            d.qpos[self.joint(name)["qpos"]] += value
        mujoco.mj_forward(m, d)

    def geom_lowest(self, g: int) -> float:
        """World z of the lowest point of a collision geom at the current pose."""
        m, d = self.model, self.data
        c, rot, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        t = int(m.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_BOX):
            return float(c[2] - np.abs(rot[2]) @ size)
        if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return float(c[2] - size[0])
        if t in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            az = abs(rot[2, 2])
            radial = size[0] * (1.0 if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else math.sqrt(max(0.0, 1.0 - az * az)))
            return float(c[2] - size[1] * az - radial)
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = m.geom_dataid[g]
            verts = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            return float((verts @ rot.T + c)[:, 2].min())
        return float(c[2] - m.geom_rbound[g])

    def lowest(self, geoms) -> tuple[float, int]:
        values = [(self.geom_lowest(g), g) for g in geoms]
        return min(values) if values else (math.inf, -1)

    def sagittal_offsets(self, by_role: dict[str, float], ref_side: str) -> dict[str, float]:
        """Joint offsets moving every leg alike, given per role offsets for the reference leg.

        A document may mirror a joint axis between the legs, the original TRON1 URDF turning its
        right knee about +y and its left about -y, so equal coordinates would bend the legs in
        opposite senses. Each leg's offset is signed by its axis against the reference leg's.
        """
        self.set_pose()
        out = {}
        for side, leg in self.legs.items():
            for r, value in by_role.items():
                ref = self.axis(self.legs[ref_side]["role"][r])
                out[leg["role"][r]] = value * float(np.sign(self.axis(leg["role"][r]) @ ref))
        return out

    def anchor(self, name: str) -> np.ndarray:
        return self.data.xanchor[self.joint(name)["id"]].copy()

    def axis(self, name: str) -> np.ndarray:
        return self.data.xaxis[self.joint(name)["id"]].copy()

    def com(self) -> np.ndarray:
        return self.data.subtree_com[self.root].copy()

    # -- dynamics ----------------------------------------------------------------------------
    def mass_matrix_diagonal(self) -> np.ndarray:
        full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, full)
        return np.diag(full).copy()

    def axis_inertia(self, bodies, name: str) -> float:
        """Inertia of the given bodies about the joint's axis line, by the parallel axis theorem."""
        m, d = self.model, self.data
        a, p = self.axis(name), self.anchor(name)
        total = 0.0
        for b in bodies:
            rot = d.ximat[b].reshape(3, 3)
            r = d.xipos[b] - p
            perp = r - (r @ a) * a
            total += float(a @ rot @ np.diag(m.body_inertia[b]) @ rot.T @ a) + m.body_mass[b] * float(perp @ perp)
        return total


def segment_angle(vec: np.ndarray) -> float:
    """Sagittal angle of a segment from the downward vertical, positive when its distal end leads."""
    return math.degrees(math.atan2(vec[0], -vec[2]))


def nominal_geometry(bp: Biped, side: str) -> dict:
    """Standing geometry of one leg and of the robot at the nominal pose, base level at the origin."""
    m, d = bp.model, bp.data
    bp.set_pose()
    role = bp.legs[side]["role"]
    hip, knee, ankle = (bp.anchor(role[r]) for r in ("hip_pitch", "knee", "ankle_pitch"))
    thigh, shank = knee - hip, ankle - knee
    included = math.degrees(math.acos(np.clip(-thigh @ shank / np.linalg.norm(thigh) / np.linalg.norm(shank), -1, 1)))
    span = np.linalg.norm(ankle - hip)
    foot = bp.legs[side]["foot"]
    sole_z, _ = bp.lowest(bp.foot_geoms[side])
    floor_z, floor_geom = bp.lowest(bp.collision)
    sole_pts = []
    for g in bp.foot_geoms[side]:
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
            rot, c, s = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
            corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) * s
            sole_pts.append(corners @ rot.T + c)
    sole_x = np.concatenate(sole_pts)[:, 0] if sole_pts else np.array([d.xpos[foot][0]])
    feet = [bp.legs[s]["foot"] for s in bp.legs]
    foot_x_axis = d.xmat[foot].reshape(3, 3)[:, 0]
    com = bp.com()
    return {
        "thigh_length": float(np.linalg.norm(thigh)),
        "shank_length": float(np.linalg.norm(shank)),
        "thigh_angle_deg": segment_angle(thigh),
        "shank_angle_deg": segment_angle(shank),
        "knee_included_deg": included,
        "foot_pitch_deg": math.degrees(math.asin(np.clip(-foot_x_axis[2], -1, 1))),
        "hip_ankle_span": float(span),
        "hip_ankle_vertical": float(hip[2] - ankle[2]),
        "extension_ratio": float(span / (np.linalg.norm(thigh) + np.linalg.norm(shank))),
        "ankle_lead_over_hip": float(ankle[0] - hip[0]),
        "knee_behind_hip": float(knee[0] - hip[0]),
        "standing_height": float(-floor_z),
        "lowest_geom_body": m.body(m.geom_bodyid[floor_geom]).name,
        "sole_depth_below_foot_body": float(d.xpos[foot][2] - sole_z),
        "sole_clear_of_floor": float(sole_z - floor_z),
        "com_height": float(com[2] - floor_z),
        "com_x": float(com[0]),
        "sole_x_range": (float(sole_x.min()), float(sole_x.max())),
        "foot_lateral_separation": float(abs(d.xpos[feet[0]][1] - d.xpos[feet[1]][1])),
        "hip_pitch_origin": hip.tolist(),
    }


def mass_inventory(bp: Biped, side: str) -> dict:
    m = bp.model
    total = float(m.body_mass.sum())
    trunk = float(sum(m.body_mass[b] for b in bp.trunk_bodies()))
    role = bp.legs[side]["role"]
    leg_root = m.jnt_bodyid[bp.joint(next(iter(bp.legs[side]["joints"])))["id"]]
    knee_body = m.jnt_bodyid[bp.joint(role["knee"])["id"]]
    return {
        "total": total,
        "trunk": trunk,
        "leg_moving": float(sum(m.body_mass[b] for b in bp.subtree(leg_root))),
        "below_knee": float(sum(m.body_mass[b] for b in bp.subtree(knee_body))),
        "leg_fraction": float(1.0 - trunk / total),
    }


def kneeling_geometry(bp: Biped, ref_side: str, samples: int = 25) -> dict:
    """Sweep the sagittal joints symmetrically within their limits, trunk level, abduction nominal.

    A posture is a kneeling one when the lowest point of the robot belongs to a non foot link by
    more than a millimetre, so that the robot would rest on that link rather than on its soles.
    The base height of such a posture is the height of the root origin above that lowest point.
    """
    m = bp.model
    foot_set = set(g for gs in bp.foot_geoms.values() for g in gs)
    other = [g for g in bp.collision if g not in foot_set]
    trunk = bp.trunk_bodies()
    grids = []
    for r in ("hip_pitch", "knee", "ankle_pitch"):
        jid = bp.joint(bp.legs[ref_side]["role"][r])["id"]
        lo, hi = m.jnt_range[jid]
        nominal = bp.cfg["nominal_pose"].get(bp.legs[ref_side]["role"][r], 0.0)
        grids.append(np.linspace(lo, hi, samples) - nominal)
    signs = bp.sagittal_offsets({"hip_pitch": 1.0, "knee": 1.0, "ankle_pitch": 1.0}, ref_side)
    hips, knees, ankles = ([leg["role"][r] for leg in bp.legs.values()] for r in ("hip_pitch", "knee", "ankle_pitch"))
    best, worst, count, total = None, None, 0, 0
    for dh in grids[0]:
        for dk in grids[1]:
            for da in grids[2]:
                bp.set_pose(offsets={n: v * signs[n] for n, v in
                                     {**{n: dh for n in hips}, **{n: dk for n in knees}, **{n: da for n in ankles}}.items()})
                foot_z, _ = bp.lowest(foot_set)
                other_z, g = bp.lowest(other)
                total += 1
                if other_z < foot_z - 1e-3:
                    body = m.geom_bodyid[g]
                    if body in trunk:
                        continue
                    count += 1
                    rec = (-other_z, m.body(body).name, dh, dk, da)
                    best = rec if best is None or rec[0] > best[0] else best
                    worst = rec if worst is None or rec[0] < worst[0] else worst
    return {"fraction": count / total, "highest": best, "lowest": worst}


def report(bp: Biped, side: str) -> dict:
    cfg, m = bp.cfg, bp.model
    print("=" * 96)
    print(f"MODEL {bp.path}")
    print("=" * 96)

    masses = mass_inventory(bp, side)
    print("\nMASS INVENTORY")
    print(f"  total {masses['total']:.3f} kg, trunk with its welded bodies {masses['trunk']:.3f} kg, "
          f"one moving leg {masses['leg_moving']:.3f} kg, below the knee {masses['below_knee']:.3f} kg")
    print(f"  legs carry {100 * masses['leg_fraction']:.1f} percent of the total mass")

    geo = nominal_geometry(bp, side)
    print(f"\nNOMINAL POSE GEOMETRY, leg {side}, nominal pose of the configuration, trunk level")
    print(f"  thigh, hip pitch axis to knee axis      {geo['thigh_length']:.4f} m at {geo['thigh_angle_deg']:+.2f} deg from the downward vertical")
    print(f"  shank, knee axis to ankle axis          {geo['shank_length']:.4f} m at {geo['shank_angle_deg']:+.2f} deg")
    print(f"  knee included angle                     {geo['knee_included_deg']:.2f} deg (180 is a straight leg)")
    print(f"  sole pitch                              {geo['foot_pitch_deg']:+.2f} deg")
    print(f"  hip to ankle span                       {geo['hip_ankle_span']:.4f} m, vertical {geo['hip_ankle_vertical']:.4f} m, "
          f"extension ratio {geo['extension_ratio']:.3f}")
    print(f"  knee behind hip / ankle ahead of hip    {geo['knee_behind_hip']:+.4f} m / {geo['ankle_lead_over_hip']:+.4f} m")
    print(f"  standing height, root above lowest point {geo['standing_height']:.4f} m, lowest point on {geo['lowest_geom_body']}")
    print(f"  sole depth below the foot body origin   {geo['sole_depth_below_foot_body']:.4f} m (the foot_radius of the foot rewards)")
    print(f"  centre of mass height / fore-aft        {geo['com_height']:.4f} m / {geo['com_x']:+.4f} m, "
          f"sole spans x {geo['sole_x_range'][0]:+.4f} to {geo['sole_x_range'][1]:+.4f} m")
    print(f"  foot lateral separation                 {geo['foot_lateral_separation']:.4f} m")
    target = cfg["rewards"].get("base_height_target")
    if target is not None:
        print(f"  reward base height target {target:.3f} m lies {geo['standing_height'] - target:+.4f} m below the standing height")

    if cfg.get("design_constants") and os.path.exists(resolve(cfg["design_constants"])):
        with open(resolve(cfg["design_constants"])) as f:
            const = json.load(f)
        bends = {k: v for k, v in const.items() if k.endswith("_BEND_DEG")}
        if bends:
            print(f"  design constants {', '.join(f'{k} {v:+.3f}' for k, v in bends.items())}")

    scale = cfg["control"]["action_scale"]
    soft = cfg["control"]["soft_joint_pos_limit_factor"]
    print(f"\nJOINT LIMITS AT THE NOMINAL POSE, soft limit factor {soft}, action scale {scale}")
    print(f"  {'joint':<16}{'lower':>9}{'upper':>9}{'nominal':>9}{'soft lo':>9}{'soft hi':>9}{'room lo':>9}{'room hi':>9}{'act lo':>8}{'act hi':>8}")
    for name in bp.legs[side]["joints"]:
        lo, hi = m.jnt_range[bp.joint(name)["id"]]
        q0 = cfg["nominal_pose"].get(name, 0.0)
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * soft
        slo, shi = mid - half, mid + half
        print(f"  {name:<16}{lo:>9.3f}{hi:>9.3f}{q0:>9.3f}{slo:>9.3f}{shi:>9.3f}{q0 - slo:>9.3f}{shi - q0:>9.3f}"
              f"{(q0 - slo) / scale:>8.2f}{(shi - q0) / scale:>8.2f}")

    sides = list(bp.legs)
    kneel = kneeling_geometry(bp, side)
    print("\nKNEELING GEOMETRY, symmetric sagittal sweep within the joint limits, trunk level")
    print(f"  {100 * kneel['fraction']:.1f} percent of the sampled postures rest on a non foot leg link")
    weight = cfg["rewards"].get("base_height_weight")
    for label in ("highest", "lowest"):
        rec = kneel[label]
        if rec is None:
            continue
        line = (f"  {label} kneeling base height {rec[0]:.4f} m on {rec[1]}, offsets hip {rec[2]:+.3f} knee {rec[3]:+.3f} "
                f"ankle {rec[4]:+.3f} rad")
        if target is not None and weight is not None:
            line += f", base height penalty rate {weight * (rec[0] - target) ** 2:+.4f} per second"
        print(line)

    dt = cfg["control"]["physics_dt"]
    control_dt = dt * cfg["control"]["decimation"]
    print(f"\nSWING EFFECTIVE INERTIA AND CONFIGURED GAINS at the nominal pose, physics dt {dt}, "
          f"control Nyquist {math.pi / control_dt:.1f} rad/s")
    bp.set_pose()
    diag = bp.mass_matrix_diagonal()
    print(f"  {'joint':<16}{'I_link':>9}{'arm':>9}{'I_eff':>9}{'tau_g':>8}{'K':>7}{'D':>7}{'w_n':>8}{'zeta':>7}{'rho':>7}")
    inertias = {}
    for name in bp.legs[side]["joints"]:
        j, act = bp.joint(name), bp.actuator(name)
        link = float(diag[j["dofadr"]])
        inertia = link + act["armature"]
        tau_g = abs(float(bp.data.qfrc_bias[j["dofadr"]]))
        K, D = act["stiffness"], act["damping"]
        a, b = dt * dt * K / inertia, dt * D / inertia
        rho = float(max(abs(np.linalg.eigvals(np.array([[1 - a, dt * (1 - b)], [-a / dt, 1 - b]])))))
        inertias[name] = (link, inertia, tau_g)
        print(f"  {name:<16}{link:>9.5f}{act['armature']:>9.5f}{inertia:>9.5f}{tau_g:>8.2f}{K:>7.1f}{D:>7.2f}"
              f"{math.sqrt(K / inertia):>8.2f}{D / (2 * math.sqrt(K * inertia)):>7.2f}{rho:>7.3f}")
    print("  I in kg m^2 from the mass matrix diagonal, tau_g the gravity torque of the distal subtree in Nm,")
    print("  rho the spectral radius of the explicitly integrated PD update, above 1 diverging.")

    print("\nSYMMETRY FLIP SET, reflection in the x and z plane")
    bp.set_pose()
    mirror = np.diag([1.0, -1.0, 1.0])
    if len(sides) == 2:
        a_side, b_side = sides
        for ra in bp.legs[a_side]["role"]:
            na, nb = bp.legs[a_side]["role"][ra], bp.legs[b_side]["role"].get(ra)
            if nb is None:
                continue
            mirrored = -(mirror @ bp.axis(na))
            flip = float(mirrored @ bp.axis(nb)) < 0.0
            lo_a, hi_a = m.jnt_range[bp.joint(na)["id"]]
            lo_b, hi_b = m.jnt_range[bp.joint(nb)["id"]]
            mirrored_range = (-hi_a, -lo_a) if flip else (lo_a, hi_a)
            agree = np.allclose(mirrored_range, (lo_b, hi_b), atol=1e-3)
            print(f"  {ra:<12} {na} axis {np.round(bp.axis(na), 3)} vs {nb} axis {np.round(bp.axis(nb), 3)}, "
                  f"flip {'YES' if flip else 'no'}, mirrored range {'agrees' if agree else 'DISAGREES'} "
                  f"({lo_a:+.3f}, {hi_a:+.3f}) / ({lo_b:+.3f}, {hi_b:+.3f})")

    print("\nSELF COLLISION AUDIT at the nominal pose, non adjacent collision geoms in penetration")
    mujoco.mj_collision(m, bp.data)
    hits = [(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name, c.dist)
            for c in bp.data.contact[: bp.data.ncon] if c.dist < 0]
    for a, b, dist in hits:
        print(f"  {a} / {b} penetration {-dist * 1000:.1f} mm")
    if not hits:
        print("  none")
    print()
    return {"mass": masses, "geometry": geo, "kneel": kneel, "inertia": inertias}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="shared biped analysis configuration")
    parser.add_argument("--model", default=None, help="MJCF or URDF document, defaulting to the configuration's")
    parser.add_argument("--compare", nargs="*", default=[], help="further documents analysed under the same configuration")
    parser.add_argument("--side", default=None, help="leg to report, defaulting to the first configured")
    parser.add_argument("--json-out", default=None, help="write the headline quantities of every model to this file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    side = args.side or next(iter(cfg["legs"]))
    paths = [args.model or cfg["model"]] + args.compare
    results = {p: report(Biped(p, cfg), side) for p in paths}

    if len(paths) > 1:
        print("=" * 96)
        print("COMPARISON")
        print("=" * 96)
        rows = [
            ("total mass, kg", lambda r: r["mass"]["total"]),
            ("trunk mass, kg", lambda r: r["mass"]["trunk"]),
            ("mass below the knee, kg", lambda r: r["mass"]["below_knee"]),
            ("thigh angle, deg", lambda r: r["geometry"]["thigh_angle_deg"]),
            ("shank angle, deg", lambda r: r["geometry"]["shank_angle_deg"]),
            ("knee included angle, deg", lambda r: r["geometry"]["knee_included_deg"]),
            ("thigh length, m", lambda r: r["geometry"]["thigh_length"]),
            ("shank length, m", lambda r: r["geometry"]["shank_length"]),
            ("extension ratio", lambda r: r["geometry"]["extension_ratio"]),
            ("ankle lead over hip, m", lambda r: r["geometry"]["ankle_lead_over_hip"]),
            ("standing height, m", lambda r: r["geometry"]["standing_height"]),
            ("centre of mass height, m", lambda r: r["geometry"]["com_height"]),
            ("highest kneeling height, m", lambda r: r["kneel"]["highest"][0] if r["kneel"]["highest"] else float("nan")),
        ]
        names = [os.path.basename(p) for p in paths]
        print(f"  {'quantity':<30}" + "".join(f"{n[:22]:>24}" for n in names))
        for label, get in rows:
            print(f"  {label:<30}" + "".join(f"{get(results[p]):>24.4f}" for p in paths))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
