import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from bipedal_locomotion.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
usd_path = os.path.join(current_dir, "../usd/SF_TRON1A/SF_TRON1A.usd")
urdf_path = os.path.join(current_dir, "../urdf/solefoot/base_robot.urdf")
usd_path_sf = os.path.join(current_dir, "../usd/SF_TRON1A/SF_TRON1A.usd")
usd_path_pf = os.path.join(current_dir, "../usd/PF_TRON1A/PF_TRON1A.usd")
usd_path_wf = os.path.join(current_dir, "../usd/WF_TRON1A/WF_TRON1A.usd")

# Hip adduction/abduction (maps to Berkeley HXX: HR + HAA)
TRON1_ABAD_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["abad_L_Joint", "abad_R_Joint"],
    effort_limit=60.0,
    velocity_limit=23,
    saturation_effort=402,
    stiffness={".*": 55.0},
    damping={".*": 13.5},
    armature={".*": 6.9e-5 * 81},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Hip flexion/extension (maps to Berkeley HFE)
TRON1_HIP_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["hip_L_Joint", "hip_R_Joint"],
    effort_limit=60.0,
    velocity_limit=20,
    saturation_effort=443,
    stiffness={".*": 80.0},
    damping={".*": 13.0},
    armature={".*": 9.4e-5 * 81},
    friction_static=0.3,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Knee flexion/extension (maps to Berkeley KFE)
TRON1_KNEE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["knee_L_Joint", "knee_R_Joint"],
    effort_limit=90.0,
    velocity_limit=14,
    saturation_effort=560,
    stiffness={".*": 60.0},
    damping={".*": 4.0},
    armature={".*": 1.5e-4 * 81},
    friction_static=0.8,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Ankle flexion/extension (maps to Berkeley FFE: feet flexion/extension)
TRON1_ANKLE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["ankle_L_Joint", "ankle_R_Joint"],
    effort_limit=40.0,
    velocity_limit=20,
    saturation_effort=402,
    stiffness={".*": 10.0},
    damping={".*": 0.5},
    armature={".*": 6.9e-5 * 81},
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
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=4,
)
activate_contact_sensors = True
init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.8),
    joint_pos={
        ".*_Joint": 0.0,
    },
    joint_vel={".*": 0.0},
)
soft_joint_pos_limit_factor = 0.9
actuators = {
    "abad": TRON1_ABAD_ACTUATOR_CFG,
    "hip": TRON1_HIP_ACTUATOR_CFG,
    "knee": TRON1_KNEE_ACTUATOR_CFG,
    "ankle": TRON1_ANKLE_ACTUATOR_CFG,
}

SOLEFOOT_IDENTIFIED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=activate_contact_sensors,
    ),
    init_state=init_state,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
)

spawn = sim_utils.UrdfFileCfg(
    asset_path=urdf_path,
    fix_base=False,
    merge_fixed_joints=True,
    self_collision=True,
    joint_drive=None,
    rigid_props=rigid_props,
    articulation_props=articulation_props,
    activate_contact_sensors=activate_contact_sensors,
)
SOLEFOOT_IDENTIFIED_CFG_URDF = ArticulationCfg(
    spawn=spawn,
    init_state=init_state,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
)

SOLEFOOT_IDENTIFIED_MULTIUSD_CFG = ArticulationCfg(
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[usd_path_sf, usd_path_pf, usd_path_wf],
        random_choice=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=activate_contact_sensors,
    ),
    init_state=init_state,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
)
