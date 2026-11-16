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
    armature={".*": 0.02},
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
    armature={".*": 0.02},
    friction_static=0.2,
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
    armature={".*": 0.04},
    friction_static=0.2,
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
    armature={".*": 0.04},
    friction_static=0.2,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_FOOT_ROLL_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_roll_02"],
    velocity_limit_sim=44.0,
    effort_limit=17,
    velocity_limit=37.699,
    saturation_effort=17.0,
    stiffness={".*": 100.0},
    damping={".*": 1.05},
    # stiffness={".*": 120.0},
    # damping={".*": 1.0},
    armature={".*": 0.0042},
    friction_static=0.1,
    activation_vel=0.1,
    friction_dynamic=0.02,
)

KSCALE_FOOT_PITCH_ACTUATOR_CFG = IdentifiedActuatorCfg(
    joint_names_expr=["(left|right)_foot_pitch_02"],
    velocity_limit_sim=44.0,
    effort_limit=17,
    velocity_limit=37.699,
    saturation_effort=17.0,
    stiffness={".*": 100.0},
    damping={".*": 1.25},
    # stiffness={".*": 120.0},
    # damping={".*": 1.3},
    armature={".*": 0.0042},
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
    # Arm 7 of plans/kscale_gait_shaping.md, set 2026-09-23, which section 21.7 details, the
    # duplicate numbering by which it was also arm 20 having been collapsed when that plan was
    # restructured on the same date.  The three sagittal nominals move to the published K-Scale K-Bot
    # standing posture of 20, 50 and minus 30 degrees, recorded at /ws/context/kscale-opensource.md
    # section 7 and adopted once before on 2026-09-16 before the pose supplied on 2026-09-17
    # replaced it.  The posture satisfies this robot's sagittal closure identity, the signed hip
    # pitch plus the knee plus the ankle pitch summing to zero, so the soles lie flat and the ankle
    # roll origin stands at exactly the 0.04300 m sole depth, which context/KScale.md section 16.2
    # gives as the cheapest check that a pose is closed.  The hip roll and ankle roll nominals are
    # deliberately unmoved, the sagittal chain and the lateral chain being separable here and the
    # lateral one carrying the stance width that arm 5's premise is measured against.
    #
    # 0.73492 m standing height, computed by forward kinematics against kscale.urdf by
    # context/artefacts/kscale_flat-2026-09-18_to_09-21-foot-joint-ablation/scripts/nominal_pose.py.
    # It is NOT the 0.73540 m that context/KScale.md records for this posture, that figure having
    # been taken at a pose whose two roll nominals were zero, the retained roll costing 0.48 mm.
    # The 0.02 m settling margin this workspace has used since 2026-08-25 was reduced to zero at
    # the user's direction on 2026-09-23 and then set to 0.01 by the user on the same date, so the
    # spawn stands 10 mm above the standing height that pen_base_height now targets.  Section C of
    # nominal_pose.py prices the three choices against the reset distribution, which reaches very
    # much further than any of them, the reset perturbations of kscale_base_env_cfg.py:656 onward
    # drawing a pose taller than the spawn on 6.2 per cent of resets at 0.02 m, 20.3 per cent at
    # 0.01 and 37.0 per cent at zero, the worst penetration being 11.4, 21.4 and 31.4 mm.  Full
    # clearance against that distribution would need 0.032 m.  The quantity to watch in the first
    # thousand iterations is the termination rate at episode start rather than the height itself.
    # ARM 7 pose.  ACTIVE, restored 2026-09-23.  The arm 6 run taken without it was abandoned
    # and the arm 7 run taken on top of arm 6 proved sound, so the tree returns to arm 7 and
    # ARM 8 stage one is launched on top of it.  0.74492 m is the 0.73492 m standing height of
    # this posture plus the 0.01 m settling margin the user set on 2026-09-23.
    pos=(0.0, 0.0, 0.74492),
    joint_pos={
        "right_hip_pitch_04": -0.34907,
        "left_hip_pitch_04": 0.34907,
        "right_hip_roll_03": 0.042951,
        "left_hip_roll_03": -0.042951,
        "right_hip_yaw_03": 0.0,
        "left_hip_yaw_03": 0.0,
        "right_knee_04": 0.87266,
        "left_knee_04": 0.87266,
        "right_foot_pitch_02": -0.52360,
        "left_foot_pitch_02": -0.52360,
        "right_foot_roll_02": 0.041254,
        "left_foot_roll_02": -0.041254,
    },
    # ARM 6 pose, the pose supplied 2026-09-17, 0.76633 m standing height plus the 0.02 m
    # settling margin.  Commented out 2026-09-23 on the return to arm 7.  Uncomment this block
    # and comment out the arm 7 block above to stand at arm 6 alone.
    # pos=(0.0, 0.0, 0.776),
    # joint_pos={
    #     "right_hip_pitch_04": -0.20369,
    #     "left_hip_pitch_04": 0.20369,
    #     "right_hip_roll_03": 0.042951,
    #     "left_hip_roll_03": -0.042951,
    #     "right_hip_yaw_03": 0.0,
    #     "left_hip_yaw_03": 0.0,
    #     "right_knee_04": 0.51239,
    #     "left_knee_04": 0.51239,
    #     "right_foot_pitch_02": -0.3087,
    #     "left_foot_pitch_02": -0.3087,
    #     "right_foot_roll_02": 0.041254,
    #     "left_foot_roll_02": -0.041254,
    # },
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


# ARM 14 of plans/kscale_gait_shaping.md section 21.14, the hip yaw frozen in the URDF.
# APPLIED 2026-10-01.  Everything above this line is untouched, so every task registered
# before this date resolves to exactly what it resolved to before.
#
# The articulation is CONSTRUCTED rather than derived by ArticulationCfg.replace, at the
# user's direction, so that every field it carries is visible at the point of definition and
# a later reader need not hold KSCALE_IDENTIFIED_CFG in mind to know what this object is.
# The shared leaves are reused by reference, nothing in this workspace mutating an
# ArticulationCfg leaf after construction, and the two that differ are written out, the spawn
# because it names a different asset and the initial state because it must not carry a key for
# a joint the articulation no longer has.
#
# WHAT THE FREEZE COSTS THE ARTICULATION, measured rather than assumed.  Twelve joints become
# TEN and thirteen bodies stay THIRTEEN.  merge_fixed_joints below is False, so the importer
# does not consolidate a fixed joint's child into its parent and kd_d_301l_6061 and
# kd_d_301r_6061 survive with their inertia where the URDF puts it.  SD_BRS1 already trains in
# this exact configuration, declaring both its hip yaws type="fixed" at
# SD_BRS1_Assembly2.urdf:89 and :431 under the same merge_fixed_joints=False at
# sd_brs1_identified_cfg.py:161, and its evaluation dump reports ten joint_names against a
# robot_mass thirteen columns wide.  The consequence is that mdp/symmetry/kscale.py needs NO
# change, _BODY_PARTNERS at :80 continuing to name thirteen live bodies and _build_joint_map
# narrowing by itself because _maps builds it from robot.joint_names at :215.
hip_yaw_fixed_urdf_path = os.path.join(
    current_dir, "../urdf/solefoot/kscale/kscale_hip_yaw_fixed.urdf"
)

# The two hip yaw nominals are ABSENT rather than set to zero.  init_state.joint_pos is
# resolved against the articulation's joint list at articulation.py:1684 through
# resolve_matching_names_values, which raises at string.py:370 on a key matching no joint, so a
# retained key would abort the spawn rather than being ignored.  Every surviving entry is byte
# identical to init_state above, including the 0.74492 m spawn height, and it is unchanged
# because a fixed joint holds the joint at the URDF's own zero and both hip yaw nominals were
# already zero, so no link moves and the frozen pose is the pose this robot resets to today.
init_state_hip_yaw_fixed = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.74492),
    joint_pos={
        "right_hip_pitch_04": -0.34907,
        "left_hip_pitch_04": 0.34907,
        "right_hip_roll_03": 0.042951,
        "left_hip_roll_03": -0.042951,
        "right_knee_04": 0.87266,
        "left_knee_04": 0.87266,
        "right_foot_pitch_02": -0.52360,
        "left_foot_pitch_02": -0.52360,
        "right_foot_roll_02": 0.041254,
        "left_foot_roll_02": -0.041254,
    },
    joint_vel={".*": 0.0},
)

# The hip_yaw group is ABSENT rather than emptied.  Articulation._initialize_impl resolves
# every group at articulation.py:1721 and raises at :1723 when a group's joint_names_expr
# matches no joint, so a retained group would abort the spawn.  KSCALE_HIP_YAW_ACTUATOR_CFG is
# left defined and unreferenced here, so a later arm may use it again and a diff of this file
# shows an addition and no deletion.  The five objects below are the SAME instances the twelve
# joint configuration uses, shared by reference, which is what makes the two robots carry one
# gain set by construction rather than by transcription.
actuators_hip_yaw_fixed = {
    "hip_roll": KSCALE_HIP_ROLL_ACTUATOR_CFG,
    "hip_pitch": KSCALE_HIP_PITCH_ACTUATOR_CFG,
    "knee": KSCALE_KNEE_ACTUATOR_CFG,
    "foot_roll": KSCALE_FOOT_ROLL_ACTUATOR_CFG,
    "foot_pitch": KSCALE_FOOT_PITCH_ACTUATOR_CFG,
}

KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        # the same wrapper the twelve joint configuration uses.  SELF_COLLISION_FILTERED_PAIRS
        # names kd_d_201r_6061 and rs03, which are the PARENTS of the two frozen joints and
        # therefore survive, together with the two feet, the two shanks and the torso, none of
        # which this arm touches, so the pair list carries over unchanged.
        func=spawn_kscale_from_urdf,
        asset_path=hip_yaw_fixed_urdf_path,
        fix_base=False,
        # FALSE, and this is the field the whole arm turns upon.  Left at its True default the
        # importer would consolidate each frozen joint's child into its parent, deleting
        # kd_d_301l_6061 and kd_d_301r_6061 from the body set, moving their inertia into the
        # hip roll links and taking the articulation to eleven bodies, at which point
        # _BODY_PARTNERS at mdp/symmetry/kscale.py:80 would name two bodies that no longer
        # exist and _build_body_map would raise at :174.
        merge_fixed_joints=False,
        joint_drive=None,
        self_collision=True,
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        activate_contact_sensors=True,
    ),
    init_state=init_state_hip_yaw_fixed,
    # Applied to ten joints instead of twelve.  It governs both pen_joint_pos_limits and the
    # reset clamp at events.py:350, and section 14.7 of the plan records that the factor is
    # drawn about the midpoint of a travel asymmetric by 10.83 to one at the hip roll, which
    # arm 17 addresses.  Freezing the hip yaw neither helps nor worsens that, the hip yaw's own
    # travel being symmetric, so the defect is carried unchanged into this asset deliberately
    # rather than repaired here where it would confound the comparison.
    soft_joint_pos_limit_factor=0.9,
    actuators=actuators_hip_yaw_fixed,
)
