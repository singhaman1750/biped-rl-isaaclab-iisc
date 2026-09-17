import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from environments.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
urdf_path = os.path.join(current_dir, "../urdf/solefoot/kscale/kscale.urdf")

# Pairs excluded from self collision, in two groups, both measured by
# scripts/analysis/kscale_self_collision_audit.py. PhysX filters only directly jointed parent and
# child pairs, so everything listed here is two or more joints apart and would otherwise be live.
#
# The first group is each leg's foot against its own shank. The two are separated by the ankle
# cross bearing rather than directly jointed, and the convex hull of the shank fork fills the slot
# the ankle occupies, leaving the foot about 11 mm inside it at every pose of a 1500 pose sweep.
# The raw meshes clear one another by 4.90 mm at the reset pose and by 0.32 mm at the ankle pitch
# lower limit, where the end stops of kscale.urdf:428 hold them apart, so the overlap is an
# artefact of the import rather than a property of the mechanism and excluding it discards nothing
# real. Left live it is worse than a one off disturbance, since a permanent contact on both feet
# enters the net force that `contact_forces` reports and would corrupt `feet_air_time`, the feet
# contact observations and every feet keyed reward term.
#
# The second group is the torso against each hip roll link, which keeps the torso reporting ground
# contact alone so that the `base_contact` termination and the torso entry of
# `pen_undesired_contacts` mean what they do if self collision is enabled.
SELF_COLLISION_FILTERED_PAIRS = [
    ("foot_6061", "kd_d_401r_6061"),
    ("foot_6061_2", "kd_d_401l_6061"),
    ("assy_formfg___kd_b_102b_torso_btm", "kd_d_201r_6061"),
    ("assy_formfg___kd_b_102b_torso_btm", "rs03"),
]


def spawn_kscale_from_urdf(
    prim_path, cfg, translation=None, orientation=None, **kwargs
):
    """Spawn the KScale URDF, then exclude :data:`SELF_COLLISION_FILTERED_PAIRS` from self collision.

    The exclusion is applied here rather than through an event term because USD filtered pairs are
    read when PhysX parses the stage, inside ``sim.reset()`` at manager_based_env.py:173. The only
    event mode running before that point is ``prestartup``, which event_manager.py:363 rejects
    outright while ``replicate_physics`` is True. Spawning happens earlier still, at
    interactive_scene.py:181, before the cloner replicates at line 184, so writing the relationship
    on the source prim carries it to every environment and costs nothing per environment.

    The path expression resolves to the source environment alone under replication and to all
    environments without it, and both cases are handled by filtering whatever the expression
    matches at this moment.
    """
    from pxr import Sdf, UsdPhysics

    prim = sim_utils.spawn_from_urdf(prim_path, cfg, translation, orientation, **kwargs)

    stage = sim_utils.get_current_stage()
    applied = 0
    for link_a, link_b in SELF_COLLISION_FILTERED_PAIRS:
        matches = sim_utils.find_matching_prim_paths(f"{prim_path}/{link_a}")
        if not matches:
            raise ValueError(
                f"Cannot exclude self collision pair ('{link_a}', '{link_b}'). The expression"
                f" '{prim_path}/{link_a}' matched no prim, so the link name is wrong or the asset"
                " did not spawn as expected."
            )
        for path_a in matches:
            path_b = path_a.rsplit("/", 1)[0] + f"/{link_b}"
            prim_a, prim_b = stage.GetPrimAtPath(path_a), stage.GetPrimAtPath(path_b)
            if not prim_a.IsValid() or not prim_b.IsValid():
                raise ValueError(
                    f"Cannot exclude self collision pair ('{link_a}', '{link_b}'). Resolved"
                    f" '{path_a}' and '{path_b}', of which at least one is not a valid prim."
                )
            UsdPhysics.FilteredPairsAPI.Apply(
                prim_a
            ).CreateFilteredPairsRel().AddTarget(Sdf.Path(path_b))
            applied += 1
    print(
        f"[INFO] spawn_kscale_from_urdf: excluded {applied} self collision pair instances from"
        f" {len(SELF_COLLISION_FILTERED_PAIRS)} pairs under '{prim_path}'"
    )
    return prim


KSCALE_HIP_YAW_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_hip_yaw_03"],
    velocity_limit_sim=20.0,
    effort_limit=42.0,
    velocity_limit=18.849,
    saturation_effort=60.0,
    # stiffness={".*": 15.0},
    # damping={".*": 0.9},
    stiffness={".*": 100.0},
    damping={".*": 3.5},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_HIP_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_hip_roll_03"],
    effort_limit=42.0,
    velocity_limit=18.849,
    velocity_limit_sim=20.0,
    saturation_effort=60.0,
    # stiffness={".*": 250.0},
    # damping={".*": 28.0},
    stiffness={".*": 250.0},
    damping={".*": 28.0},
    armature={".*": 0.01},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_HIP_PITCH_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_hip_pitch_04"],
    velocity_limit_sim=15.0,
    effort_limit=84.0,
    velocity_limit=17.488,
    saturation_effort=120.0,
    stiffness={".*": 200.0},
    damping={".*": 21.5},
    # stiffness={".*": 200.0},
    # damping={".*": 21.5},
    armature={".*": 0.01},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_KNEE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_knee_04"],
    velocity_limit_sim=15.0,
    effort_limit=84.0,
    velocity_limit=17.488,
    saturation_effort=120.0,
    stiffness={".*": 300.0},
    damping={".*": 12.0},
    # stiffness={".*": 300.0},
    # damping={".*": 12.0},
    armature={".*": 0.015},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_FOOT_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_roll_02"],
    velocity_limit_sim=44.0,
    effort_limit=11.9,
    velocity_limit=37.699,
    saturation_effort=17.0,
    stiffness={".*": 100.0},
    damping={".*": 1.05},
    # stiffness={".*": 120.0},
    # damping={".*": 1.0},
    armature={".*": 0.005},
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_FOOT_PITCH_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_pitch_02"],
    velocity_limit_sim=44.0,
    effort_limit=11.9,
    velocity_limit=37.699,
    saturation_effort=17.0,
    stiffness={".*": 100.0},
    damping={".*": 1.25},
    # stiffness={".*": 120.0},
    # damping={".*": 1.3},
    armature={".*": 0.005},
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

rigid_props = sim_utils.RigidBodyPropertiesCfg(
    rigid_body_enabled=True,
    disable_gravity=False,
    retain_accelerations=False,
    linear_damping=0.0,
    angular_damping=0.0,
    max_linear_velocity=1000.0,
    max_angular_velocity=1000.0,
    max_depenetration_velocity=1.0,
)
articulation_props = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=2,
    solver_velocity_iteration_count=2,
)

init_state = ArticulationCfg.InitialStateCfg(
    # 0.76633 m standing height plus the 0.02 m settling margin this workspace uses.
    pos=(0.0, 0.0, 0.786),
    joint_pos={
        "right_hip_pitch_04": -0.20369,
        "left_hip_pitch_04": 0.20369,
        "right_hip_roll_03": 0.042951,
        "left_hip_roll_03": -0.042951,
        "right_hip_yaw_03": 0.0,
        "left_hip_yaw_03": 0.0,
        "right_knee_04": 0.51239,
        "left_knee_04": 0.51239,
        "right_foot_pitch_02": -0.3087,
        "left_foot_pitch_02": -0.3087,
        "right_foot_roll_02": 0.041254,
        "left_foot_roll_02": -0.041254,
    },
    joint_vel={".*": 0.0},
)

actuators = {
    "hip_yaw": KSCALE_HIP_YAW_ACTUATOR_CFG,
    "hip_roll": KSCALE_HIP_ROLL_ACTUATOR_CFG,
    "hip_pitch": KSCALE_HIP_PITCH_ACTUATOR_CFG,
    "knee": KSCALE_KNEE_ACTUATOR_CFG,
    "foot_roll": KSCALE_FOOT_ROLL_ACTUATOR_CFG,
    "foot_pitch": KSCALE_FOOT_PITCH_ACTUATOR_CFG,
}

KSCALE_IDENTIFIED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # wraps the stock spawn_from_urdf to exclude SELF_COLLISION_FILTERED_PAIRS on the source
        # prim, which enabled_self_collisions below depends on
        func=spawn_kscale_from_urdf,
        asset_path=urdf_path,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=None,
        self_collision=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=True,
    ),
    init_state=init_state,
    soft_joint_pos_limit_factor=0.9,
    actuators=actuators,
)
