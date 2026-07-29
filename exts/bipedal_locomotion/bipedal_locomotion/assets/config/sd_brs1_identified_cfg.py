import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from bipedal_locomotion.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
urdf_path = os.path.join(current_dir, "../urdf/solefoot/SD_BRS1/SD_BRS1_Assembly2.urdf")
urdf_path2 = os.path.join(current_dir, "../urdf/solefoot/SD_BRS1/SD_BRS.urdf")

# Hip yaw
BRS1_HIP_YAW_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["HipYaw[LR]"],
    effort_limit=351.0,
    velocity_limit=20.0,
    saturation_effort=351.0,
    stiffness={".*": 60.0},
    damping={".*": 8.0},
    armature={".*": 0.0},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Hip roll
BRS1_HIP_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["HipRoll[LR]"],
    effort_limit=500.0,
    velocity_limit=40.0,
    saturation_effort=500.0,
    stiffness={".*": 150.0},
    damping={".*": 45.0},
    armature={".*": 0.01},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Hip pitch
BRS1_HIP_PITCH_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["HipPitch[LR]"],
    effort_limit=500.0,
    velocity_limit=40.0,
    saturation_effort=500.0,
    stiffness={".*": 200.0},
    damping={".*": 50.0},
    armature={".*": 0.01},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Knee
BRS1_KNEE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["KneePitch[LR]"],
    effort_limit=600.0,
    velocity_limit=40.0,
    saturation_effort=600.0,
    stiffness={".*": 200.0},
    damping={".*": 22.0},
    armature={".*": 0.015},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Ankle roll
BRS1_ANKLE_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["AnkleRoll[LR]"],
    effort_limit=131.0,
    velocity_limit=25.0,
    saturation_effort=131.0,
    stiffness={".*": 20.0},
    damping={".*": 4.0},
    armature={".*": 0.005},
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Ankle pitch
BRS1_ANKLE_PITCH_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["AnklePitch[LR]"],
    effort_limit=262.0,
    velocity_limit=25.0,
    saturation_effort=262.0,
    stiffness={".*": 50.0},
    damping={".*": 4.0},
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
    enabled_self_collisions=False,
    solver_position_iteration_count=2,
    solver_velocity_iteration_count=2,
)
init_state = ArticulationCfg.InitialStateCfg(
    # Standing crouch solved from the sagittal 2-link closed chain, placing the ankle
    # directly beneath the hip so the stance is balanced, with the ankle pitch absorbing
    # the shank tilt so the sole stays flat. Derived in /ws/NATURAL_GAIT_PLAN.md section
    # 2.9 for a standing torso height of 1.15 m:
    #   d = H - 0.305077,  k = arccos((d^2 - l1^2 - l2^2) / (2*l1*l2)),
    #   beta = atan2(l2*sin(k), l1 + l2*cos(k)),  ankle = k - beta
    # with l1 = 0.44 (thigh), l2 = 0.43 (shank), and the 0.305077 constant being the
    # 0.181077 torso-to-hip drop plus the 0.124 ankle-frame-to-sole drop.
    #
    # Signs follow the URDF joint axes, which are NOT uniformly mirrored. HipPitchR is
    # (0,-1,0) against HipPitchL (0,1,0), so the two hips take OPPOSITE signs for the same
    # physical flexion. KneePitchR/L are both (0,1,0) and AnklePitchR/L are both (0,-1,0),
    # so neither is mirrored and both take the SAME sign on the two legs. This is
    # corroborated by the limits, KneePitch being [0, 1.483] on both legs.
    #
    # The previous pose set hip, knee and ankle all to 0.2618, which does not satisfy the
    # closed chain and placed the ankle 0.114 m forward of the hip. It was also dead code,
    # overwritten by _JOINT_POS_DEFAULTS in tasks/locomotion/robots/brs_solefoot_env_cfg.py.
    # Both files now carry the same pose.
    #
    # Spawn height is the 1.15 m standing height plus the 0.02 m settling margin that the
    # previous straight-leg value of 1.2 m carried over its own 1.175 m standing height.
    pos=(0.0, 0.0, 1.17),
    joint_pos={
        "HipRollR": 0.0,
        "HipRollL": 0.0,
        "HipPitchR": 0.2379,
        "HipPitchL": -0.2379,
        "KneePitchR": 0.4814,
        "KneePitchL": 0.4814,
        "AnkleRollR": 0.0,
        "AnkleRollL": 0.0,
        "AnklePitchR": 0.2435,
        "AnklePitchL": 0.2435,
    },
    joint_vel={".*": 0.0},
)
actuators = {
    "hip_roll": BRS1_HIP_ROLL_ACTUATOR_CFG,
    "hip_pitch": BRS1_HIP_PITCH_ACTUATOR_CFG,
    "knee": BRS1_KNEE_ACTUATOR_CFG,
    "ankle_roll": BRS1_ANKLE_ROLL_ACTUATOR_CFG,
    "ankle_pitch": BRS1_ANKLE_PITCH_ACTUATOR_CFG,
}

SD_BRS1_IDENTIFIED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
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
articulation_props2 = sim_utils.ArticulationRootPropertiesCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=2,
    solver_velocity_iteration_count=2,
)
SD_BRS1_IDENTIFIED_CFG2 = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=urdf_path2,
        fix_base=False,
        merge_fixed_joints=False,
        joint_drive=None,
        self_collision=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props2,
        activate_contact_sensors=True,
    ),
    init_state=init_state,
    soft_joint_pos_limit_factor=0.9,
    actuators=actuators,
)
