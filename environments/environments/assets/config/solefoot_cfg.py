import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(__file__)
mjcf_path = os.path.join(current_dir, "../mjcf/solefoot/tron1/base_robot.xml")

SOLEFOOT_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=mjcf_path,
        fix_base=False,
        self_collision=True,
        import_inertia_tensor=True,
        link_density=0.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    # See solefoot_identified_cfg.py for why the root must be named outright.
    articulation_root_prim_path="/base_Link/base_Link",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_Joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ],
            effort_limit_sim=80.0,
            velocity_limit_sim=25.0,
            stiffness=45.0,
            damping=1.5,
            friction=0.0,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                "ankle_L_Joint",
                "ankle_R_Joint",
            ],
            effort_limit_sim=80.0,
            velocity_limit_sim=25.0,
            stiffness=45.0,
            damping=0.8,
            friction=0.0,
        ),
    },
)
