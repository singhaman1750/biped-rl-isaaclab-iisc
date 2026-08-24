"""Emit quadruped.urdf from the primitive parameters of my_design.xml.

Every number below is transcribed from the MuJoCo model and none is recomputed, so the
URDF is equivalent to the MJCF by construction rather than by a converter's fidelity.
The MJCF cylinder ``size`` is a radius and a HALF length, so the URDF length is twice
the second entry, and the MJCF box ``size`` is a half extent triple, so the URDF box is
twice each entry.
"""

import xml.etree.ElementTree as ET
from xml.dom import minidom

PI_2 = 1.5707963267948966

# leg -> (abad mount x, abad mount y, hip actuator x offset, knee actuator y offset,
#         thigh y offset, abduction axis sign). The first five are transcribed from the
# four <body> chains of the MJCF. The sixth mirrors the abduction axis so that a positive
# command means outward on both sides, as TRON1 does, see section 4.2.
LEGS = {
    "FR": (0.105, -0.052, 0.071, -0.050, -0.030, -1),
    "FL": (0.105, 0.052, 0.071, 0.050, 0.030, 1),
    "RR": (-0.105, -0.052, -0.071, -0.050, -0.030, -1),
    "RL": (-0.105, 0.052, -0.071, 0.050, 0.030, 1),
}

THIGH_L = 0.213
CALF_L = 0.213
FOOT_R = 0.022

LIMITS = {           # joint -> (lower, upper, effort, velocity)
    "abad": (-1.0, 1.0, 23.622511, 30.0),
    "hip": (-1.2, 3.5, 23.622511, 30.0),
    "knee": (-2.9, -0.5, 35.238000, 20.0),
}


def _sub(parent, tag, **attrib):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrib.items()})


def _inertial(link, mass, ixx, iyy, izz, xyz="0 0 0"):
    node = _sub(link, "inertial")
    _sub(node, "origin", xyz=xyz, rpy="0 0 0")
    _sub(node, "mass", value=f"{mass:.6f}")
    _sub(node, "inertia", ixx=f"{ixx:.9f}", ixy="0", ixz="0",
         iyy=f"{iyy:.9f}", iyz="0", izz=f"{izz:.9f}")


def _shape(link, kind, geom_attr, xyz, rpy, rgba):
    for tag in ("visual", "collision"):
        node = _sub(link, tag)
        _sub(node, "origin", xyz=xyz, rpy=rpy)
        geom = _sub(node, "geometry")
        _sub(geom, kind, **geom_attr)
        if tag == "visual":
            mat = _sub(node, "material", name=rgba[0])
            _sub(mat, "color", rgba=rgba[1])


def _joint(root, name, jtype, parent, child, xyz, axis=None, limit=None):
    j = _sub(root, "joint", name=name, type=jtype)
    _sub(j, "origin", xyz=xyz, rpy="0 0 0")
    _sub(j, "parent", link=parent)
    _sub(j, "child", link=child)
    if axis is not None:
        _sub(j, "axis", xyz=axis)
    if limit is not None:
        lo, hi, eff, vel = limit
        _sub(j, "limit", lower=f"{lo}", upper=f"{hi}", effort=f"{eff}", velocity=f"{vel}")
        _sub(j, "dynamics", damping="0.0", friction="0.0")


def build():
    root = ET.Element("robot", {"name": "quadruped"})

    base = _sub(root, "link", name="base_Link")
    _inertial(base, 0.674397, 0.002635, 0.002100, 0.003783)
    _shape(base, "box", {"size": "0.170000 0.196000 0.092000"},
           "0 0 0", "0 0 0", ("trunk", "0.20 0.20 0.25 1"))

    for leg, (ax, ay, hx, ky, ty, asign) in LEGS.items():
        # abduction actuator housing, rigid to the trunk, cylinder axis along X
        link = _sub(root, "link", name=f"abad_{leg}_actuator_Link")
        _inertial(link, 1.028375, 0.001088, 0.000681, 0.000681)
        _shape(link, "cylinder", {"radius": "0.046000", "length": "0.040000"},
               "0 0 0", f"0 {PI_2} 0", ("abad", "0.85 0.35 0.10 1"))
        _joint(root, f"abad_{leg}_fixed_joint", "fixed", "base_Link",
               f"abad_{leg}_actuator_Link", f"{ax:.6f} {ay:.6f} 0.000000")

        # hip pitch actuator housing, carried by the abduction joint, axis along Y
        link = _sub(root, "link", name=f"hip_{leg}_actuator_Link")
        _inertial(link, 1.028375, 0.000681, 0.001088, 0.000681)
        _shape(link, "cylinder", {"radius": "0.046000", "length": "0.040000"},
               "0 0 0", f"{PI_2} 0 0", ("hip", "0.15 0.45 0.85 1"))
        _joint(root, f"abad_{leg}_Joint", "revolute", f"abad_{leg}_actuator_Link",
               f"hip_{leg}_actuator_Link", f"{hx:.6f} 0.000000 0.000000",
               axis=f"{asign} 0 0", limit=LIMITS["abad"])

        # knee actuator housing, carried by the hip pitch joint, axis along Y
        link = _sub(root, "link", name=f"knee_{leg}_actuator_Link")
        _inertial(link, 1.434746, 0.000950, 0.001518, 0.000950)
        _shape(link, "cylinder", {"radius": "0.046000", "length": "0.040000"},
               "0 0 0", f"{PI_2} 0 0", ("knee", "0.15 0.75 0.30 1"))
        _joint(root, f"hip_{leg}_Joint", "revolute", f"hip_{leg}_actuator_Link",
               f"knee_{leg}_actuator_Link", f"0.000000 {ky:.6f} 0.000000",
               axis="0 1 0", limit=LIMITS["hip"])

        # thigh segment, rigid to the knee actuator. The MJCF declares a capsule but
        # dimensions and inertias it as a cylinder, and after the correction of section
        # 4.3 both files say cylinder.
        link = _sub(root, "link", name=f"hip_{leg}_thigh_Link")
        _inertial(link, 0.187365, 0.000727, 0.000727, 0.000037,
                  xyz=f"0 0 {-THIGH_L / 2:.6f}")
        _shape(link, "cylinder", {"radius": "0.020000", "length": f"{THIGH_L:.6f}"},
               f"0 0 {-THIGH_L / 2:.6f}", "0 0 0", ("thigh", "0.6 0.6 0.6 1"))
        _joint(root, f"hip_{leg}_thigh_joint", "fixed", f"knee_{leg}_actuator_Link",
               f"hip_{leg}_thigh_Link", f"0.000000 {ty:.6f} 0.000000")

        # calf segment, carried by the knee joint
        # The mass and the inertia below are the CORRECTED figures of section 4.3,
        # computed for a cylinder of radius 0.014 at the model's own leg density of
        # 700 kg/m^3, not the thigh's figures that the MJCF erroneously repeats here.
        link = _sub(root, "link", name=f"knee_{leg}_Link")
        _inertial(link, 0.091809, 0.000352, 0.000352, 0.000009,
                  xyz=f"0 0 {-CALF_L / 2:.6f}")
        _shape(link, "cylinder", {"radius": "0.014000", "length": f"{CALF_L:.6f}"},
               f"0 0 {-CALF_L / 2:.6f}", "0 0 0", ("calf", "0.45 0.45 0.45 1"))
        _joint(root, f"knee_{leg}_Joint", "revolute", f"hip_{leg}_thigh_Link",
               f"knee_{leg}_Link", f"0.000000 0.000000 {-THIGH_L:.6f}",
               axis="0 1 0", limit=LIMITS["knee"])

        # spherical foot, rigid to the calf, the only intended ground contact
        link = _sub(root, "link", name=f"foot_{leg}_Link")
        _inertial(link, 0.020000, 0.000004, 0.000004, 0.000004)
        _shape(link, "sphere", {"radius": f"{FOOT_R:.6f}"},
               "0 0 0", "0 0 0", ("foot", "0.05 0.05 0.05 1"))
        _joint(root, f"foot_{leg}_joint", "fixed", f"knee_{leg}_Link",
               f"foot_{leg}_Link", f"0.000000 0.000000 {-CALF_L:.6f}")

    return root


if __name__ == "__main__":
    import sys
    xml = minidom.parseString(ET.tostring(build())).toprettyxml(indent="  ")
    open(sys.argv[1], "w").write(xml)
