import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from environments.tasks.locomotion import mdp
from environments.tasks.locomotion.mdp.curriculums import reduce_tracking_rewards_std

_LEG_LINKS_NO_FOOT = "kd_d_102[rl]_6061|kd_d_201r_6061|rs03|kd_d_301[rl]_6061|kd_d_401[rl]_6061|arb_uj111_cross_bearing.*"
_FOOT_LINKS = "foot_6061.*"
_FOOT_LINK_RIGHT = "foot_6061"
_FOOT_LINK_LEFT = "foot_6061_2"
_SHANK_LINK_RIGHT = "kd_d_401r_6061"
_SHANK_LINK_LEFT = "kd_d_401l_6061"
_TORSO_LINK = "assy_formfg___kd_b_102b_torso_btm"

KSCALE_SOLE_OFFSETS = [
    [-0.0423, +0.0430, +0.0138],
    [-0.0362, +0.0430, -0.1572],
    [-0.0326, +0.0430, -0.1712],
    [-0.0258, +0.0430, -0.1805],
    [-0.0108, +0.0430, -0.1889],
    [+0.0051, +0.0430, -0.1899],
    [+0.0213, +0.0430, -0.1841],
    [+0.0326, +0.0430, -0.1712],
    [+0.0362, +0.0430, -0.1572],
    [+0.0423, +0.0430, +0.0138],
    [+0.0363, +0.0430, +0.0200],
    [-0.0363, +0.0430, +0.0200],
]

##################
# Scene Definition
##################

env_spacing = 2.5


@configclass
class KscaleSceneCfg(InteractiveSceneCfg):
    """Configuration for the kscale scene"""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
        env_spacing=env_spacing,
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = MISSING

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + _TORSO_LINK,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )

    foot_pair_contact_right = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + _FOOT_LINK_RIGHT,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/" + _FOOT_LINK_LEFT,
            "{ENV_REGEX_NS}/Robot/" + _SHANK_LINK_LEFT,
        ],
        history_length=4,
        update_period=0.0,
    )

    foot_pair_contact_left = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + _FOOT_LINK_LEFT,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/" + _SHANK_LINK_RIGHT],
        history_length=4,
        update_period=0.0,
    )

##############
# MDP settings
##############


@configclass
class CommandsCfg:
    """Command terms for the MDP"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        debug_vis=True,
        resampling_time_range=(7.5, 12.5),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.8),
            lin_vel_y=(-0.01, 0.01),
            ang_vel_z=(-0.3, 0.3),
            heading=(-math.pi, math.pi),
        ),
    )

    # gait_command = mdp.UniformGaitCommandCfg(
    #     resampling_time_range=(10.0, 10.0),
    #     debug_vis=False,
    #     ranges=mdp.UniformGaitCommandCfg.Ranges(
    #         frequencies=(1.0, 1.0),
    #         offsets=(0.5, 0.5),
    #         durations=(0.62, 0.62),
    #         swing_height=(0.05, 0.05),
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    # Per joint action scale, set 2026-09-22 as arm 4 of plans/kscale_gait_shaping.md, which
    # section 21.4 details, reordered ahead of the nominal pose change at the user's direction.
    # It was written as arm 5 before the lateral weight shift took that number on the same date,
    # and the duplicate numbering by which it was also arm 10 was collapsed when that plan was
    # restructured on 2026-09-23.  This comment was repointed at the restructured sections on
    # the same date.  A single
    # scale of 0.4 met six joint groups whose ceiling to stiffness ratios span 0.168 to 0.420 rad,
    # so one action magnitude saturated the ankles where two and a half times as much was needed
    # at the hip pitch, and the two joints reached at the smallest action are exactly the two the
    # evaluation dumps find pinned against their stops.  Each scale below is the hip pitch's 0.4
    # times the ratio of that group's saturation deflection to the hip pitch's, which equalises
    # the action at saturation across the set at 1.050 and is the sizing rule section 4.6 states.
    #
    # The table below is re-derived 2026-09-23 by section G of nominal_pose.py in the ablation
    # artefact scripts and it records TWO further interventions on top of arm 4 rather than one.
    #
    # The two ankle scales stand at 0.08 and not at the 0.1619 the rule above gives, which is
    # ARM 6 of plans/kscale_gait_shaping.md at section 21.6, a halving that deliberately breaks
    # the equalisation arm 4 established so that those two groups saturate at an action of 2.125
    # against the 1.050 of the other four.  Its target is the commanded latch of section 11 of
    # that plan, which finds the ankle roll driven onto its stop BY COMMAND rather than held there
    # by torque, and it moves that stop to an action of 2.757 where the rule placed it at 1.362.
    #
    # And the stop column moves with the nominal pose, ARM 7 having deepened the sagittal fold on
    # the same date, which is why the column is given at both poses.
    #
    # A run launched from this tree therefore carries arms 4, 4, 4b and 5 at once and cannot
    # attribute an effect to any one of them.  That is the user's direction of 2026-09-23 and is
    # recorded at section 21.7 of the plan rather than left to be inferred from the dump.
    #
    #   group        K     tau    tau/K    scale   action at saturation   action to nearer stop
    #                                                                     2026-09-17     arm 7
    #   hip_pitch   200    84.0   0.4200   0.4000          1.050             3.127        3.491
    #   hip_yaw     100    42.0   0.4200   0.4000          1.050             3.927        3.927
    #   knee        300    84.0   0.2800   0.2667          1.050             1.921        3.272
    #   hip_roll    250    42.0   0.1680   0.1600          1.050             1.041        1.041
    #   foot_pitch  100    17.0   0.1700   0.0800          2.125             7.050        4.363
    #   foot_roll   100    17.0   0.1700   0.0800          2.125             2.757        2.757
    #
    # The ankle figures are tied to the 17.0 N.m ceiling at kscale_identified_cfg.py:152 and :168
    # and must be re-derived if that ceiling moves, a 34 N.m ankle giving 0.3238 under the rule.
    #
    # Two properties of the dict form govern the entries.  Every joint MUST be covered, because
    # joint_actions.py:85 initialises the scale tensor to ONES and writes only the matched
    # columns, so an omitted joint silently takes a scale of 1.0 rather than keeping 0.4.  And the
    # patterns must be mutually exclusive, a joint matching two keys raising at string.py:328.
    # A key matching no joint raises at string.py:362, so a typo fails loudly rather than
    # silently, which is the one failure mode that needs no guarding here.
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "(left|right)_hip_pitch_04": 0.4,
            "(left|right)_hip_yaw_03": 0.4,
            "(left|right)_knee_04": 0.2667,
            "(left|right)_hip_roll_03": 0.16,
            "(left|right)_foot_pitch_02": 0.08,
            "(left|right)_foot_roll_02": 0.08,
        },
        use_default_offset=True,
        # ARM 10 of plans/kscale_gait_shaping.md section 21.10, run as
        # 2026-09-29_06-15-58.  Each bound is the joint's mechanical travel inset by 0.02 rad.
        # The clamp is applied AFTER the default_joint_pos offset at joint_actions.py:175, so
        # every pair below is an ABSOLUTE joint target and not a deviation.
        #
        # IT SUCCEEDED, and section 18.2 of that plan reports it.  The ankle roll saturation
        # fell from 60.17 per cent to 0.06, its hard stop residency from 42.30 to 0.07, its
        # sustained actuator demand from 13.0 N.m of a 17 N.m ceiling to 3.5, and the ankle
        # pitch left its own stop with it.  This block is therefore a PRECONDITION of every
        # later arm rather than an arm, and it should not be removed while any of them is
        # being read.
        #
        # ITS SIZING RULE IS NONETHELESS WRONG AND SECTION 21.15 REPLACES IT.  Measured
        # against the saturating deflection, being the effort limit divided by the stiffness,
        # the ankle roll window below is 2.84 times that deflection and the hip roll window
        # 14.51 times it.  The ankle roll is the joint this cured and the hip roll is the
        # joint the fault moved to, section 18.3 measuring the hip roll at its stop for 45.9
        # per cent of every episode under a raw action of nineteen standard deviations, the
        # policy having answered a bound it could not reach by commanding far past it.  Arm 15
        # sets every bound at the nominal plus and minus 1.5 times the deflection instead.
        #
        # ARM 11, pen_action_excess below, prices what this block removes.  Section
        # 18.4 measures that excess growing from 0.000595 rad^2 per step at iteration 193 to
        # 30.380 at iteration 10000 of the run that carries this clip WITHOUT the penalty, a
        # factor of 51000, so a weight read from the early magnitude is read before the policy
        # has learned to produce the quantity it prices.
        clip={
            "(left|right)_foot_roll_02": (-0.241799, 0.241799),
            "(left|right)_foot_pitch_02": (-0.852665, 0.503599),
            "(left|right)_knee_04": (0.020000, 2.685260),
            "(left|right)_hip_yaw_03": (-1.550800, 1.550800),
            "right_hip_roll_03": (-2.248930, 0.189440),
            "left_hip_roll_03": (-0.189440, 2.248930),
            "right_hip_pitch_04": (-2.196570, 1.027200),
            "left_hip_pitch_04": (-1.027200, 2.196570),
        },

    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP (standard PPO)"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group"""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # gait_phase = ObsTerm(func=mdp.get_gait_phase)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100.0, 100.0), scale=1.0)
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque, clip=(-500.0, 500.0))
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc, clip=(-2000.0, 2000.0))
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS)},
            clip=(-20.0, 20.0),
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos, clip=(-100.0, 100.0))
        robot_vel = ObsTerm(func=mdp.robot_vel, clip=(-20.0, 20.0))
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties, clip=(-10.0, 10.0))
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS)},
            clip=(-5000.0, 5000.0),
        )

        # gait_phase = ObsTerm(func=mdp.get_gait_phase)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()


@configclass
class HIMObservationsCfg:
    """Observation specifications for the MDP (HIM architecture)"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group"""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HistoryObsCfg(ObsGroup):
        """Observations for history group (fed to HIM encoder)"""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoise(mean=0.0, std=0.05),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoise(mean=0.0, std=0.025),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class TargetEncCfg(ObsGroup):
        """Privileged observations for the target encoder"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100.0, 100.0), scale=1.0)
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged observations for critic"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, clip=(-100.0, 100.0), scale=1.0)
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
        )
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque, clip=(-500.0, 500.0))
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc, clip=(-2000.0, 2000.0))
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS)},
            clip=(-20.0, 20.0),
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos, clip=(-100.0, 100.0))
        robot_vel = ObsTerm(func=mdp.robot_vel, clip=(-20.0, 20.0))
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties, clip=(-10.0, 10.0))
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS)},
            clip=(-5000.0, 5000.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    @configclass
    class EstimatorGTCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            noise=GaussianNoise(mean=0.0, std=0.00),
            scale=1.0,
        )

    policy: PolicyCfg = PolicyCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    estimatorGT: EstimatorGTCfg = EstimatorGTCfg()
    targetEnc: TargetEncCfg = TargetEncCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    prepare_quantity_for_kscale = EventTerm(
        func=mdp.prepare_quantity_for_tron,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_TORSO_LINK),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[_LEG_LINKS_NO_FOOT, _FOOT_LINKS]),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    hip_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "(right|left)_hip_roll_03",
                    "(right|left)_hip_pitch_04",
                    "(right|left)_hip_yaw_03",
                ]
            ),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    knee_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["(right|left)_knee_04"]),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    ankle_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_foot_roll_02", "(right|left)_foot_pitch_02"]
            ),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    joint_offsets = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.05, 0.05),
            "operation": "add",
        },
    )
    joint_friction = EventTerm(
        func=mdp.randomize_joint_friction_model,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "friction_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )
    scale_all_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    # Joint-specific position randomisation. UNVERIFIED ranges -- copied structurally
    # from SD_BRS1's reset events (fractions of that robot's soft joint limits around its
    # nominal pose) but not re-derived from kscale's own URDF limits or its (placeholder)
    # nominal pose in kscale_identified_cfg.py. Revisit once the standing pose is fixed.
    reset_hip_roll_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "(right|left)_hip_roll_03",
            "position_range": (-0.1295, 0.1295),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_hip_yaw_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            # Range comparable to the hip roll's, the two being off-axis degrees of freedom of
            # similar authority. See the note in hip_joint_stiffness_and_damping above.
            "joint_name": "(right|left)_hip_yaw_03",
            "position_range": (-0.1295, 0.1295),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_hip_pitch_r_joint = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "right_hip_pitch_04",
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_hip_pitch_l_joint = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "left_hip_pitch_04",
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_knee_pitch_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "(right|left)_knee_04",
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_ankle_roll_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "(right|left)_foot_roll_02",
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_ankle_pitch_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "(right|left)_foot_pitch_02",
            "position_range": (-0.15, 0.15),
            "velocity_range": (-0.5, 0.5),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-4000.0)

    keep_balance = RewTerm(func=mdp.stay_alive, weight=0.05)

    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=50,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=15,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    rew_no_fly = RewTerm(
        func=mdp.NoFlyWithGrace,
        weight=15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "threshold": 1.0,
            "history_index": 0,
            "grace_steps": 20,
        },
    )
    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_foot_pitch_02"]
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "use_default_offset": True,
        },
    )
    rew_keep_ankle_roll_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_foot_roll_02"]
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "use_default_offset": True
        },
    )
    rew_keep_hip_yaw_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        # 1.0 reproduces run 2026-09-29_06-15-58 (arm 10).  Arm 16 stage 1 raised it to 5.0,
        # as run 2026-10-01_11-16-37 did.
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_yaw_03"]
            ),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "require_airborne": True,
            "history_index": 0,
            "force_threshold": 1.0,
            "pitch_scale": 0.2,
            "use_default_offset": True
        },
    )
    pen_hip_roll_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_roll_03"]
            )
        },
    )
    pen_hip_yaw_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["(right|left)_hip_yaw_03"]
            )
        },
    )
    # pen_ankle_roll_deviation = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=["(right|left)_foot_roll_02"]
    #         )
    #     },
    # )
    pen_feet_heading = RewTerm(
        func=mdp.feet_yaw_alignment,
        weight=-8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "forward_axis": (0.0, 0.0, -1.0),
            "common_mode": "mean",
            "tolerance": 0.10,
            "differential_scale": 1.0,
            "airborne_only": False,
            "history_index": 0,
            "force_threshold": 1.0,
        },
    )
    pen_base_height = RewTerm(
        func=mdp.base_height_rough_l2,
        params={
            # Arm 7 of plans/kscale_gait_shaping.md, set 2026-09-23.  0.73492 m is the standing
            # height of the nominal pose at kscale_identified_cfg.py:197, computed by forward
            # kinematics against kscale.urdf by nominal_pose.py in the ablation artefact scripts,
            # and it is now also the spawn height, the settling margin having been zeroed in the
            # same pass.  The three values are one value rather than three, the standing height
            # following from the pose by forward kinematics, so a target set independently of the
            # pose asks the policy to hold a height its own posture does not produce, which is the
            # condition section 4.5 of that plan diagnoses.  It replaces the 0.73 that stood here
            # from 2026-09-22 unrecorded and the 0.76633 of the pose supplied 2026-09-17.
            # ARM 7 target.  ACTIVE, restored 2026-09-23 on the return to arm 7, the arm 6
            # run taken without it having been abandoned and the arm 7 run taken on top of
            # arm 6 having proved sound.  ARM 8 stage one is launched on top of it.
            "target_height": 0.73492,
            # ARM 6 target, 0.766297 m, the standing height of the pose supplied 2026-09-17
            # and computed by forward kinematics against kscale.urdf by nominal_pose.py in
            # the ablation artefact scripts, with the mirrored hip roll and ankle roll of
            # that pose counted.  Commented out 2026-09-23 on the return to arm 7.
            # "target_height": 0.76630,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
        weight=-45.0,
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-5)
    # ARM 8 of plans/kscale_gait_shaping.md section 21.8, set 2026-09-23 on top of
    # arms 4, 4, 4b and 5.  The four energy weights are raised by a factor of ten and nothing
    # else is touched.  Section 14.6 of that plan measures the four together at 1.04 per cent
    # of run A's negative reward budget and 0.44 per cent of run C's, and prices the exchange
    # the optimiser actually faces, run B burning 3.61 times run A's mechanical power, paying
    # 0.4276 per second more for it and collecting 1.9175 per second more on tracking alone,
    # an exchange rate of 4.48 to one in favour of the expensive gait.  Ten leaves a margin of
    # 2.2 over the indifference point that 4.48 would set, indifference being the worst
    # available choice since an objective indifferent between two gaits selects between them by
    # the exploration noise, and it takes the group from 1.04 per cent of the negative budget
    # to 9.4 per cent, a share that can shape without dominating.  All four move together and
    # by the same factor, the measurement pricing them as a group rather than singly.
    #
    # pen_action_rate at :894 and pen_action_smoothness at :923 are deliberately NOT moved.
    # Both are quadratic in the RAW action, so arm 4 multiplied their effect at the ankles by
    # 6.1 and arm 6 by a further 4.1 without anyone choosing it, and raising them here would
    # confound this arm with those two.  They are revisited once the action scale is settled.
    #
    #   term                    before     applied    factor
    #   pen_joint_torque        -1.0e-5    -5.0e-4      50
    #   pen_joint_accel         -1.0e-7    -5.0e-6      50
    #   pen_joint_vel_l2        -1.0e-5    -5.0e-4      50
    #   pen_joint_torque_rate   -1.0e-5    -5.0e-4      50
    #
    # CORRECTED 2026-09-30.  The table above read a factor of TEN until this date while the
    # values beneath it were FIFTY, the tenfold rung having been run separately as
    # 2026-09-23_11-40-12 and the fiftyfold one applied here without the comment moving with
    # it.  Section 16.1 of the plan records the discrepancy and section 16.2 reads both as
    # rungs of one ladder, which runs at one, ten, fifty, one hundred and five hundred times
    # the pre arm 8 value.  ARM 8 SUCCEEDED.  The first order of magnitude carries most of the
    # benefit, the mean reward rising from 887.6 to 955.4 with the velocity tracking reward
    # unmoved, the second is very nearly free, and the FIVE HUNDREDFOLD RUN COLLAPSED to an
    # eighteen step episode at a unit termination rate with nothing in the hundredfold run
    # foreshadowing it.  Section 20.4 accordingly excludes going beyond one hundred.
    #
    # The prediction is that the absolute joint power falls from run C's 67.1 W, that the whole
    # body torque rate falls from 120.8 N.m/s, and that rew_lin_vel_xy falls by less than the
    # 1.9175 per second run B's excess power bought.  The falsifier is the mean reward falling
    # while the power does not, which sends the arm to stage two.
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-4)  # ARM 8, was -1.0e-5
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-5.0e-6)  # ARM 8, was -1.0e-7
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    pen_joint_power = RewTerm(
        func=mdp.joint_powers_l1,
        weight=-0.02,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[_LEG_LINKS_NO_FOOT, _TORSO_LINK],
            ),
            "threshold": 10.0,
        },
    )
    pen_foot_cross_contact_right = RewTerm(
        func=mdp.filtered_contacts,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg("foot_pair_contact_right"),
            "threshold": 1.0,
        },
    )
    pen_foot_cross_contact_left = RewTerm(
        func=mdp.filtered_contacts,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg("foot_pair_contact_left"),
            "threshold": 1.0,
        },
    )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.075)
    pen_joint_torque_rate = RewTerm(
        func=mdp.JointTorqueRatePenalty,
        # ARM 8 stage one, raised by ten 2026-09-23 with the three terms at :892, :893 and
        # :970, was -1.0e-5.  The reasoning is the comment block above pen_joint_torque.
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-50.0)
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100,
        params={
            # The nominal pose supplied 2026-09-17 placed the ankle frames 0.1992 m apart,
            # the mirrored hip roll having drawn them 53 mm closer than the 0.2520 m every
            # earlier pose held. 0.24 stood 0.012 m below the old nominal and 0.19 stands
            # 0.0092 m below that one, so the floor keeps its clearance below the pose
            # rather than firing at every reset, which at 0.24 it now would.  Arm 7 deepened
            # the sagittal fold on 2026-09-23 and the separation widened to 0.2044 m with it,
            # the rolled hips standing further apart as the leg folds, so the floor now sits
            # 0.0144 m below the nominal and is left where it is.
            "min_feet_distance": 0.19,
            "feet_links_name": [_FOOT_LINKS],
            "lateral_only": False,
        },
    )
    pen_feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[_FOOT_LINKS]),
            # The same standing height as pen_base_height above, moved with it 2026-09-23 for
            # arm 7.  It is INERT in this configuration and the value changes no behaviour,
            # feet_regulation at rewards.py:1850 reading base_height_target only as the height
            # gate length scale and only when height_decay_scale is absent, which it is not.  It
            # is nonetheless held in step, being dumped to params/env.yaml where a later reader
            # will take it for the standing height, as this document's own sections have.
            # ARM 7 target.  ACTIVE, restored 2026-09-23 with pen_base_height above.
            "base_height_target": 0.73492,
            # ARM 6 target, commented out 2026-09-23 on the return to arm 7.
            # "base_height_target": 0.76630,
            "foot_radius": 0.043,
            "height_decay_scale": 0.03,
        },
    )
    # ARM 8 stage one, raised by ten 2026-09-23 with the three terms at :892, :893 and :928,
    # was -1.0e-05.  The reasoning is the comment block above pen_joint_torque.
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-4)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=12.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
        },
    )
    rew_foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward_v2,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "target_height": 0.05,
            "std": 0.02,
            "tanh_mult": 1.0,
            "sole_offsets": KSCALE_SOLE_OFFSETS,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "force_threshold": 1.0,
        },
    )
    pen_foot_landing_vel = RewTerm(
        func=mdp.foot_landing_vel_v2,
        weight=-30.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "sole_offsets": KSCALE_SOLE_OFFSETS,
            "about_landing_threshold": 0.04,
            "force_threshold": 1.0,
        },
    )
    pen_feet_impact = RewTerm(
        func=mdp.feet_impact_force,
        weight=-3.0e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
            "force_threshold": 290.0,
        },
    )
    # rew_gait = RewTerm(
    #     func=mdp.GaitReward,
    #     weight=40.0,
    #     params={
    #         "tracking_contacts_shaped_force": -1.0,
    #         "tracking_contacts_shaped_vel": -1.0,
    #         "gait_force_sigma": 25.0,
    #         "gait_vel_sigma": 0.25,
    #         "kappa_gait_probs": 0.05,
    #         "command_name": "gait_command",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_FOOT_LINKS),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
    #     },
    # )
    # rew_hip_lateral_sway = RewTerm(
    #     func=mdp.HipLateralSwayReward,
    #     weight=5.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=_TORSO_LINK),
    #         "feet_cfg": SceneEntityCfg("robot", body_names=_FOOT_LINKS),
    #         "command_name": "gait_command",
    #         "std": 0.01,
    #         "amplitude": 0.03,
    #         "phase_shift": 0.0,
    #         "resync_threshold": 0.04,
    #         "resync_patience": 50,
    #     },
    # )

    # ARM 12 of plans/kscale_gait_shaping.md section 21.12, prices the amount by
    # which the clip at :251 removed the commanded target.  APPLIED TO THE WORKING TREE on
    # 2026-09-30 at the CALIBRATED weight and awaiting its run.  ARM 11 is the same term at
    # the two weights it was launched with, RUNNING as 2026-09-30_07-05-14 at -500 and as
    # 2026-09-30_06-43-38 at -2, and section 21.11 is its record.
    #
    # SECTION 18.4 PRICES BOTH WEIGHTS AS MIS-SET AND THE READING THAT CHOSE THEM AS THE
    # WRONG ONE.  The excess is a LEARNED EXPLOIT rather than an initial condition, standing
    # at 0.000595 rad^2 per step at iteration 193 and at 30.380 at iteration 10000 of
    # 2026-09-29_06-15-58, which carries the identical clip with no penalty, a factor of
    # 51000.  The magnitude this term reports early in training is therefore the quantity
    # BEFORE the policy has learned to produce it, and a weight read from it is wrong by that
    # factor.  Against run E's negative budget of 11.99 per second and positive budget of 88,
    # -2 reaches 60.8 per second at iteration 10000 and -500 reaches 15190, which is 173 times
    # the whole task reward and which the arm 8 ladder above has already watched this robot
    # collapse under at a smaller multiple.
    #
    # -14 IS THE CALIBRATED WEIGHT AND SECTION 21.12 DERIVES IT.  The onset of the exploit is
    # taken as the excess reaching one action standard deviation at the joints that carry it,
    # being 0.16 rad at each hip roll and 0.08 at each ankle roll, which sums to 0.0640 rad^2,
    # and -14 places that at 0.896 per second, being one per cent of run E's positive budget.
    # Section 18.4 published this calibration as 0.051 rad^2 and -17, which omits the ankle
    # roll pair although the sentence naming it says four joints, and section 21.12 carries
    # the correction.  The per joint measurement attributes 0.863 and 0.864 rad^2 of run E's
    # excess to the two ankle rolls against 14.315 and 14.332 to the two hip rolls, so the
    # pair carries 5.7 per cent of the quadratic excess and 20.4 per cent of the absolute one.
    #
    # THE FORM IS STILL THE PART TO CHANGE AND THIS WEIGHT DOES NOT SETTLE IT, which is why
    # arm 13 follows.  A quadratic term on a quantity spanning five orders of magnitude spans
    # ten, so no fixed coefficient is right at both ends, which section 19.2 records as a
    # difficulty with a published name.  At -14 this form reaches 425.3 per second at run E's
    # terminal excess, which is 483 per cent of the positive budget, where the l1 form of
    # arm 13 at -1.8 reaches 13.21, a ratio of 32 to 1.  Arm 12 is expected to be insufficient
    # and is run so that the form is refuted at a weight that was defensible.
    # pen_action_excess = RewTerm(
    #     func = mdp.action_clip_excess_l2,
    #     weight=-4,
    # )
    # pen_action_excess = RewTerm(
    #     func=mdp.action_clip_excess_l1,
    #     weight=-0.02,
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=_TORSO_LINK),
            "threshold": 1.0,
        },
    )
    # low_height = DoneTerm(
    #     func=mdp.root_height_below_minimum,
    #     params={"minimum_height": 0.27},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP"""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_delayed,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "starting_step": 200 * 24,
        },
    )

    modify_push_force = CurrTerm(
        func=mdp.modify_push_force,
        params={
            "term_name": "push_robot",
            "max_velocity": (2.0, 2.0),
            "interval": 300 * 24,
            "starting_step": 800 * 24,
            "increment_rate": 1.1,
            "decrement_rate": 0.9,
        },
    )

    modify_command_velocity_lin_x = CurrTerm(
        func=mdp.modify_command_velocity_x,
        params={
            "term_name": "rew_lin_vel_xy",
            "max_velocity": (-0.5, 1.0),
            "interval": 300 * 24,
            "starting_step": 1000 * 24,
            "update_rate": 0.005,
            "update_threshold": 0.75,
        },
    )

    modify_command_velocity_lin_y = CurrTerm(
        func=mdp.modify_command_velocity_y,
        params={
            "term_name": "rew_lin_vel_xy",
            "max_velocity": (-0.2, 0.2),
            "interval": 300 * 24,
            "starting_step": 1000 * 24,
            "update_rate": 0.005,
            "update_threshold": 0.75,
        },
    )

    modify_command_velocity_ang_z = CurrTerm(
        func=mdp.modify_command_velocity_angular,
        params={
            "term_name": "rew_ang_vel_z",
            "max_velocity": (-0.9, 0.9),
            "interval": 300 * 24,
            "starting_step": 1000 * 24,
            "update_rate": 0.005,
            "update_threshold": 0.7,
        },
    )

    modify_linear_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_lin_vel_xy",
            "interval": 300 * 24,
            "starting_step": 900 * 24,
            "update_rate": 0.95,
            "update_threshold": 0.67,
            "minimum_std": 0.09,
        },
    )

    modify_angular_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_ang_vel_z",
            "interval": 300 * 24,
            "starting_step": 0,
            "update_rate": 0.975,
            "update_threshold": 0.5,
            "minimum_std": 0.09,
        },
    )


########################
# Environment definition
########################


@configclass
class KscaleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the kscale standard PPO environment"""

    scene: KscaleSceneCfg = KscaleSceneCfg(num_envs=4096, env_spacing=env_spacing)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.physx.gpu_max_rigid_patch_count = 2**19 
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        self.sim.dt = 0.005
        self.seed = 42
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.foot_pair_contact_right is not None:
            self.scene.foot_pair_contact_right.update_period = self.sim.dt
        if self.scene.foot_pair_contact_left is not None:
            self.scene.foot_pair_contact_left.update_period = self.sim.dt


@configclass
class KscaleHIMEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the kscale HIM environment"""

    scene: KscaleSceneCfg = KscaleSceneCfg(num_envs=4096, env_spacing=env_spacing)
    observations: HIMObservationsCfg = HIMObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.sim.physx.gpu_max_rigid_patch_count = 2**19 
        self.sim.physx.gpu_collision_stack_size=2**28,  # 256 MB; bump to 2**29 / 2**30 if needed
        self.sim.physx.gpu_max_rigid_contact_count=2**23,
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        self.sim.dt = 0.005
        self.seed = 42
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.foot_pair_contact_right is not None:
            self.scene.foot_pair_contact_right.update_period = self.sim.dt
        if self.scene.foot_pair_contact_left is not None:
            self.scene.foot_pair_contact_left.update_period = self.sim.dt


# ============================================================================================
# ARM 14 of plans/kscale_gait_shaping.md section 21.14, the hip yaw frozen in the URDF.
# APPLIED 2026-10-01.  Everything above this line is untouched, so KscaleEnvCfg and
# KscaleHIMEnvCfg resolve to exactly what they resolved to before this date.
# ============================================================================================


@configclass
class ActionsCfgHipYawFixed:
    """The action term of the ten joint robot, for KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED.

    joint_names stays at [".*"], which now resolves to TEN joints rather than twelve and needs
    no edit.  The scale and the clip dictionaries do need one, both being resolved key by key
    against the term's own joint list, the scale at joint_actions.py:89 and the clip at :109,
    and both raising at string.py:370 on a key that matches nothing.  Their hip yaw entries are
    therefore removed rather than left harmlessly behind.

    Every surviving value is byte identical to ActionsCfg at :212, so a policy trained here and
    one trained there differ in their action dimension and in nothing else.  The clip is arm
    10's, being each joint's mechanical travel inset by 0.02 rad, which is what the tree
    carries and what 2026-09-29_06-15-58 was taken on.  Where this is run after arm 15 the clip
    becomes arm 15's instead, likewise with its hip yaw entry removed, and the run notes must
    say which.

    The scale dictionary must cover every surviving joint and the reason is at :208,
    joint_actions.py:85 initialising the scale tensor to ONES and writing only the matched
    columns, so an omitted joint silently takes a scale of 1.0 rather than keeping its intended
    value.  The five keys below cover all ten joints, which validate_arm14.py asserts against
    the new URDF's revolute joint list rather than by inspection.
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            "(left|right)_hip_pitch_04": 0.4,
            "(left|right)_knee_04": 0.2667,
            "(left|right)_hip_roll_03": 0.16,
            "(left|right)_foot_pitch_02": 0.08,
            "(left|right)_foot_roll_02": 0.08,
        },
        use_default_offset=True,
        clip={
            "(left|right)_foot_roll_02": (-0.241799, 0.241799),
            "(left|right)_foot_pitch_02": (-0.852665, 0.503599),
            "(left|right)_knee_04": (0.020000, 2.685260),
            "right_hip_roll_03": (-2.248930, 0.189440),
            "left_hip_roll_03": (-0.189440, 2.248930),
            "right_hip_pitch_04": (-2.196570, 1.027200),
            "left_hip_pitch_04": (-1.027200, 2.196570),
        },
    )


@configclass
class KscaleEnvHipYawFixedCfg(KscaleEnvCfg):
    """The ten joint environment, differing from KscaleEnvCfg in four places and no others.

    Every removal below is REQUIRED rather than tidy, each named path resolving the hip yaw
    through a helper that raises on an expression matching no joint, so a retained reference
    aborts at spawn or at the first reset rather than reporting zero.

    Mutating the inherited terms here is safe.  configclass converts every mutable class
    default into a field whose default_factory deep copies it per instance, at
    configclass.py:381 and :491, so self.events is this instance's own object before
    __post_init__ runs and nothing written here reaches KscaleEnvCfg or any other subclass.
    """

    actions: ActionsCfgHipYawFixed = ActionsCfgHipYawFixed()

    def __post_init__(self):
        super().__post_init__()

        # ONE, the two reward terms that resolve the hip yaw.  Both carry a SceneEntityCfg
        # whose joint_names is resolved at scene_entity_cfg.py:175 through find_joints, which
        # raises at string.py:268 on an expression matching no joint, so both must go rather
        # than being left to report zero.  Removing them moves the objective by under one per
        # cent, rew_keep_hip_yaw_zero_in_air paying 0.331 to 0.431 per second of a maximum of
        # 1.0 against a positive budget of 88, which section 21.14 prices.
        self.rewards.rew_keep_hip_yaw_zero_in_air = None
        self.rewards.pen_hip_yaw_deviation = None

        # TWO, the reset event that offsets the hip yaw.  reset_hip_yaw_joints passes
        # joint_name to mdp.reset_joint_by_offset, which resolves it at events.py:367 through
        # find_joints and raises the same way.  The whole term goes, there being no other
        # joint it addresses.
        self.events.reset_hip_yaw_joints = None

        # THREE, the hip yaw entry of the gain randomisation group.  That term names three
        # joint pairs in one SceneEntityCfg and only the hip yaw pair must go, the other two
        # still being wanted, so the ENTRY is dropped rather than the term.
        term = self.events.hip_joint_stiffness_and_damping
        term.params["asset_cfg"].joint_names = [
            n for n in term.params["asset_cfg"].joint_names if "hip_yaw" not in n
        ]

        # FOUR, nothing else.  The following are recorded because each is a place a reader
        # expects an edit and finds none.
        #
        #   The observation groups.  joint_pos at :291, joint_vel at :297 and last_action at
        #   :303, and the critic's robot_joint_* terms, all carry the default
        #   SceneEntityCfg("robot") and therefore resolve whatever the articulation has, so
        #   they narrow from twelve to ten by themselves and NO observation term changes.
        #   robot_mass at :345 and robot_inertia at :346 are per BODY and the body count is
        #   unchanged at thirteen, so their widths do not move at all.
        #
        #   randomize_joint_default_pos resolves every joint by ".*" and so needs no change,
        #   matching ten joints instead of twelve.
        #
        #   The symmetry augmentation.  NO CHANGE.  _maps builds the joint permutation from
        #   robot.joint_names at mdp/symmetry/kscale.py:215, so it is ten wide the moment the
        #   articulation is; the two hip yaws are each other's mirror partner and leave
        #   together, so the involution asserted at :163 still holds; "hip_yaw" stays in
        #   _FLIP_JOINT_KINDS at :76 and becomes inert, the set being tested as a substring of
        #   each surviving name rather than resolved against the articulation; and
        #   _BODY_PARTNERS at :80 still names thirteen live bodies because the asset carries
        #   merge_fixed_joints=False.  Section 21.14 carries the proof and validate_arm14.py
        #   asserts every clause of it.
        #
        #   The curriculum, the commands and the terminations.  No term names a joint, and
        #   base_contact names the torso, which survives.
