import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from environments.actuators import IdentifiedActuatorCfg

current_dir = os.path.dirname(__file__)
mjcf_path = os.path.join(current_dir, "../mjcf/quadruped/quadruped.xml")

# Abduction and adduction, the roll axis, named .._hip_joint by the design generator
# that emits the asset.
# The effort limit is the MJCF forcerange and the velocity limit the Go1 GO-M8010-6
# rated joint speed, this design's torque ceiling matching that motor to within 0.4
# percent. Kp is DERIVED, not taken from the MJCF position actuator gain of 100, which
# is a modelling convenience with no hardware behind it. It is set by the stance torque
# of 4.05 N m at a tolerated sag of 0.125 rad, half the action scale, and lands at
# 6.54 Hz, well inside the observability ceiling of a fifth of the 50 Hz control rate.
# Kd gives a damping ratio of 0.62 against an effective inertia of 0.023674 kg m^2 at
# the nominal crouch. See joint_control_analysis_quadruped.md section 5.3 throughout.
#
# velocity_limit_sim is the PhysX joint velocity cap, set on every group at 1.2 times
# velocity_limit, the factor co_optimisation/utils/respawn.py VELOCITY_LIMIT_SIM_FACTOR applies
# to every picked actuator. Unset, it resolves to the importer's FLT_MAX in deg/s, 5.94e36 rad/s,
# leaving only the DC motor curve's braking torque against an overspeed
# (isaaclab_diagnostics.md sections 6c, 6d and 7a). The curve clamps at
# velocity_limit * (1 + effort_limit / saturation_effort), 34.5 rad/s here and 23.0 at the
# knee, so the cap of 36 and 24 sits just past it, the curve governing normal running and the
# solver only an overspeed.
QUADRUPED_ABAD_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".._hip_joint"],
    effort_limit=23.622511,
    velocity_limit=30.0,
    velocity_limit_sim=36.0,
    saturation_effort=158.0,
    stiffness={".*": 40.0},
    damping={".*": 1.2},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Hip flexion and extension, the pitch axis, named .._thigh_joint by the emitter.
# It shares the abduction joint's motor and takes the same Kp, its own stance torque
# being near zero at the crouch, where the keyframe places each foot directly beneath
# its pitch axis. Kd is set against an effective inertia of 0.020572 kg m^2.
QUADRUPED_HIP_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".._thigh_joint"],
    effort_limit=23.622511,
    velocity_limit=30.0,
    velocity_limit_sim=36.0,
    saturation_effort=158.0,
    stiffness={".*": 40.0},
    damping={".*": 1.1},
    armature={".*": 0.01},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

# Knee flexion and extension, named .._calf_joint by the emitter. Its motor stands
# in a ratio of 1.4917 to the other two, which would give a Kp of 60, but 60 places the
# joint at 11.11 Hz, above the observability ceiling. It is trimmed to 50, landing at
# 10.15 Hz and accepting a stance sag of 0.130 rad against the 0.125 target. Kd is set
# against an effective inertia of 0.012305 kg m^2, which does not vary with pose.
QUADRUPED_KNEE_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=[".._calf_joint"],
    effort_limit=35.238000,
    velocity_limit=20.0,
    velocity_limit_sim=24.0,
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

init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.38),
    joint_pos={
        ".._hip_joint": 0.0,
        ".._thigh_joint": 0.0,
        ".._calf_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)
soft_joint_pos_limit_factor = 0.9
actuators = {
    "abad": QUADRUPED_ABAD_ACTUATOR_CFG,
    "hip": QUADRUPED_HIP_ACTUATOR_CFG,
    "knee": QUADRUPED_KNEE_ACTUATOR_CFG,
}

QUADRUPED_IDENTIFIED_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=mjcf_path,
        fix_base=False,
        self_collision=True,
        import_inertia_tensor=True,
        link_density=0.0,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=activate_contact_sensors,
    ),
    articulation_root_prim_path="/trunk/trunk",
    init_state=init_state,
    soft_joint_pos_limit_factor=soft_joint_pos_limit_factor,
    actuators=actuators,
)
