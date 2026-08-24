import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from environments.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
urdf_path = os.path.join(current_dir, "../urdf/quadruped/quadruped.urdf")

# Abduction and adduction, the roll axis. MuJoCo names this joint FR_hip_joint.
# The effort limit is the MJCF forcerange and the velocity limit the Go1 GO-M8010-6
# rated joint speed, this design's torque ceiling matching that motor to within 0.4
# percent. Kp is DERIVED, not taken from the MJCF position actuator gain of 100, which
# is a modelling convenience with no hardware behind it. It is set by the stance torque
# of 4.05 N m at a tolerated sag of 0.125 rad, half the action scale, and lands at
# 6.54 Hz, well inside the observability ceiling of a fifth of the 50 Hz control rate.
# Kd gives a damping ratio of 0.62 against an effective inertia of 0.023674 kg m^2 at
# the nominal crouch. See joint_control_analysis_quadruped.md section 5.3 throughout.
QUADRUPED_ABAD_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["abad_.._Joint"],
    effort_limit=23.622511,
    velocity_limit=30.0,
    saturation_effort=158.0,
    stiffness={".*": 40.0},
    damping={".*": 1.2},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Hip flexion and extension, the pitch axis. MuJoCo names this joint FR_thigh_joint.
# It shares the abduction joint's motor and takes the same Kp, its own stance torque
# being near zero at the crouch, where the keyframe places each foot directly beneath
# its pitch axis. Kd is set against an effective inertia of 0.020572 kg m^2.
QUADRUPED_HIP_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["hip_.._Joint"],
    effort_limit=23.622511,
    velocity_limit=30.0,
    saturation_effort=158.0,
    stiffness={".*": 40.0},
    damping={".*": 1.1},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Knee flexion and extension. MuJoCo names this joint FR_calf_joint. Its motor stands
# in a ratio of 1.4917 to the other two, which would give a Kp of 60, but 60 places the
# joint at 11.11 Hz, above the observability ceiling. It is trimmed to 50, landing at
# 10.15 Hz and accepting a stance sag of 0.130 rad against the 0.125 target. Kd is set
# against an effective inertia of 0.012305 kg m^2, which does not vary with pose.
QUADRUPED_KNEE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["knee_.._Joint"],
    effort_limit=35.238000,
    velocity_limit=20.0,
    saturation_effort=236.0,
    stiffness={".*": 50.0},
    damping={".*": 0.9},
    armature={".*": 0.01},
    friction_static=0.2,
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
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
)
activate_contact_sensors = True

# 0.292 m is the height at which the foot sphere rests on the ground in the nominal
# crouch, being the 0.270 m kinematic drop plus the 0.022 m foot radius. The spawn
# adds 28 mm of clearance so that a rough terrain tile cannot capture the foot at reset.
init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.32),
    joint_pos={
        "abad_.._Joint": 0.0,
        "hip_.._Joint": 0.884337,
        "knee_.._Joint": -1.768673,
    },
    joint_vel={".*": 0.0},
)
soft_joint_pos_limit_factor = 0.9
actuators = {
    "abad": QUADRUPED_ABAD_ACTUATOR_CFG,
    "hip": QUADRUPED_HIP_ACTUATOR_CFG,
    "knee": QUADRUPED_KNEE_ACTUATOR_CFG,
}

spawn = sim_utils.UrdfFileCfg(
    asset_path=urdf_path,
    fix_base=False,
    merge_fixed_joints=False,
    self_collision=True,
    joint_drive=None,
    rigid_props=rigid_props,
    articulation_props=articulation_props,
    activate_contact_sensors=activate_contact_sensors,
)

QUADRUPED_IDENTIFIED_CFG = ArticulationCfg(
    spawn=spawn,
    init_state=init_state,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
)
