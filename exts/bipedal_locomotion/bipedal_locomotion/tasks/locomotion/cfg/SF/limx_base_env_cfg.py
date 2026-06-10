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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg as BaseCommandsCfg

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.mdp.curriculums import reduce_tracking_rewards_std

from .terrains_cfg import BERKELEY_MIMIC_TERRAINS_CFG

##################
# Scene Definition
##################

env_spacing=2.5

@configclass
class SFSceneCfg(InteractiveSceneCfg):
    """Configuration for the test scene"""

    # terrain
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
        env_spacing=env_spacing
    )

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # bipedal robot
    robot: ArticulationCfg = MISSING

    # height sensors (Berkeley Mimic - Actor Visible)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_Link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )


##############
# MDP settings
##############


@configclass
class CommandsCfg:
    """Command terms for the MDP"""

    # gait_command = mdp.UniformGaitCommandCfg(
    #     resampling_time_range=(5.0, 5.0),  # Fixed resampling time of 5 seconds
    #     debug_vis=False,  # No debug visualization needed
    #     ranges=mdp.UniformGaitCommandCfg.Ranges(
    #         frequencies=(0.8, 1.6), # (1.5, 2.5),  # Gait frequency range [Hz]
    #         offsets=(0.5, 0.5),  # Phase offset range [0-1]
    #         durations=(0.5, 0.5),  # Contact duration range [0-1]
    #         swing_height=(0.1, 0.2)
    #     ),
    # )

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        debug_vis=True,
        resampling_time_range=(7.5, 12.5),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.75, 0.75),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            noise=GaussianNoise(mean=0.0, std=0.05),
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

        # robot joint measurements
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

        # last action
        last_action = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        # gaits
        # gait_phase = ObsTerm(func=mdp.get_gait_phase)
        # gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            # Required by HIMActorCritic
            self.flatten_history_dim = True

    # @configclass
    # class HistoryObsCfg(ObsGroup):
    #     """Observation for policy group"""
    #
    #     # robot base measurements
    #     base_lin_vel = ObsTerm(
    #         func=mdp.base_lin_vel,
    #         clip=(-100.0, 100.0),
    #         noise=GaussianNoise(mean=0.0, std=0.05),
    #         scale=1.0,
    #     )
    #     base_ang_vel = ObsTerm(
    #         func=mdp.base_ang_vel,
    #         noise=GaussianNoise(mean=0.0, std=0.05),
    #         clip=(-100.0, 100.0),
    #         scale=0.25,
    #     )
    #     proj_gravity = ObsTerm(
    #         func=mdp.projected_gravity,
    #         noise=GaussianNoise(mean=0.0, std=0.025),
    #         clip=(-100.0, 100.0),
    #         scale=1.0,
    #     )
    #
    #     # robot joint measurements
    #     joint_pos = ObsTerm(
    #         func=mdp.joint_pos_rel,
    #         noise=GaussianNoise(mean=0.0, std=0.01),
    #         clip=(-100.0, 100.0),
    #         scale=1.0,
    #     )
    #     joint_vel = ObsTerm(
    #         func=mdp.joint_vel_rel,
    #         noise=GaussianNoise(mean=0.0, std=0.01),
    #         clip=(-100.0, 100.0),
    #         scale=0.05,
    #     )
    #
    #     # last action
    #     last_action = ObsTerm(
    #         func=mdp.last_action,
    #         noise=GaussianNoise(mean=0.0, std=0.01),
    #         clip=(-100.0, 100.0),
    #         scale=1.0,
    #     )
    #     velocity_commands = ObsTerm(
    #         func=mdp.generated_commands, params={"command_name": "base_velocity"}
    #     )
    #
    #     # gaits
    #     # gait_phase = ObsTerm(func=mdp.get_gait_phase)
    #     # gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})
    #
    #     def __post_init__(self):
    #         self.enable_corruption = True
    #         self.concatenate_terms = True
    #         self.history_length = 25
    #         self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # robot base measurements
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # last action
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        # velocity command
        # vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # gait_phase = ObsTerm(func=mdp.get_gait_phase)
        # gait_command = ObsTerm(func=mdp.get_gait_command, params={"command_name": "gait_command"})

        # Privileged observation
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_.*")},
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*")
            },
        )
        # heights = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 1.0),
        # )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            # Required by HIMActorCritic
            self.flatten_history_dim = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    # obsHistory: HistoryObsCfg = HistoryObsCfg()


@configclass
class HIMObservationsCfg:
    """Observation specifications for the MDP for HIM architecture"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
        # base_lin_vel = ObsTerm(
        #     func=mdp.base_lin_vel,
        #     clip=(-100.0, 100.0),
        #     noise=GaussianNoise(mean=0.0, std=0.05),
        #     scale=1.0,
        # )
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

        # robot joint measurements
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

        # last action
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
        """Observation for history group"""

        # robot base measurements
        # base_lin_vel = ObsTerm(
        #     func=mdp.base_lin_vel,
        #     clip=(-100.0, 100.0),
        #     noise=GaussianNoise(mean=0.0, std=0.05),
        #     scale=1.0,
        # )
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

        # robot joint measurements
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

        # last action
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
            self.history_length = 25
            # Required by HIMActorCritic
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        """Observation for critic group"""

        # robot base measurements
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # robot joint measurements
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        # last action
        last_action = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

        # heights scan
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        )

        # Privileged observation
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="ankle_.*")},
        )
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*")
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 25
            self.flatten_history_dim = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
    
    @configclass
    class EstimatorGTCfg(ObsGroup):
        """Observation for policy group"""

        # robot base measurements
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


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    prepare_quantity_for_tron1_piper = EventTerm(
        func=mdp.prepare_quantity_for_tron,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            # Converted from add(-5.0, 5.0) on baseline 9.585 kg → ±52.2% symmetric scale
            "mass_distribution_params": (0.478, 1.522),
            "operation": "scale",
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.8, 1.2),
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
    robot_joint_stiffness_and_damping_knee = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["knee_[LR]_Joint"]),
            # Converted from abs(50,70)/(3,5) on baselines 60/4 → ±16.7%/±25% symmetric
            "stiffness_distribution_params": (0.833, 1.167),
            "damping_distribution_params": (0.750, 1.250),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping_hip = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_[LR]_Joint"]),
            # Converted from abs(70,90)/(10,15) on baselines 80/13
            # Stiffness: ±12.5% symmetric; Damping: 23.1%/15.4% asymmetric → take 15.4%
            "stiffness_distribution_params": (0.875, 1.125),
            "damping_distribution_params": (0.846, 1.154),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping_abad = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["abad_[LR]_Joint"]),
            # Converted from abs(50,70)/(12,15) on baselines 55/13.5
            # Stiffness: 9.1%/27.3% asymmetric → take 9.1%; Damping: ±11.1% symmetric
            "stiffness_distribution_params": (0.909, 1.091),
            "damping_distribution_params": (0.889, 1.111),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping_ankle = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["ankle_[LR]_Joint"]),
            # Converted from abs(8,12)/(0.4,0.6) on baselines 10/0.5 → ±20% symmetric
            "stiffness_distribution_params": (0.800, 1.200),
            "damping_distribution_params": (0.800, 1.200),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # Custom Berkeley Calibration Error
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

    # robot_center_of_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_coms,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "com_distribution_params": (
    #             (-0.075, 0.075),
    #             (-0.075, 0.075),
    #             (-0.075, 0.075),
    #         ),
    #         "operation": "add",
    #         "distribution": "uniform",
    #     },
    # )

    # reset
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

    reset_robot_joints = EventTerm(
        # func=mdp.reset_joints_by_scale,
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque_stochastic,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
    #         "force_range": {
    #             "x": (-500.0, 500.0),
    #             "y": (-500.0, 500.0),
    #             "z": (-0.0, 0.0),
    #         },  # force = mass * dv / dt
    #         "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
    #         "probability": 0.002,  # Expect step = 1 / probability
    #     },
    # )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )
    
    # # Note: Using standard isaaclab randomize_joint_parameters if available
    scale_all_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            # Corrected from (1.0, 1.05): original had no downward perturbation.
            # Symmetric ±5% applied in both directions.
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    # termination related rewards
    keep_balance = RewTerm(func=mdp.stay_alive, weight=0.05)

    # tracking rewards
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=7.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.09)},
    )
    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["ankle_L_Joint", "ankle_R_Joint"]),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["ankle_[RL]_Link"]
            ),
        },
    )
    rew_no_fly = RewTerm(
        func=mdp.no_fly,
        weight=1.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names="ankle_[RL]_Link"
            ),
            "threshold": 1.0,
        },
    )

    # penalizations
    pen_base_height = RewTerm(
        func=mdp.base_height_rough_l2,
        params={
            "target_height": 0.75,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
        weight=-5.0,
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)  # -0.5
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00008)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-5e-7)  # -2.5e-7
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)  # -0.01)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)

    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"],
            ),
            "threshold": 10.0,
        },
    )
    pen_action_smoothness = RewTerm(
        func=mdp.ActionSmoothnessPenalty, weight=-0.075
    )  # -0.01)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100,
        params={"min_feet_distance": 0.115, "feet_links_name": ["ankle_[RL]_Link"]},
    )
    pen_feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.2,  # -0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ankle_[RL]_Link"]),
            "base_height_target": 0.75,
            "foot_radius": 0.03,
        },
    )
    # pen_foot_landing_vel = RewTerm(
    #     func=mdp.foot_landing_vel,
    #     weight=-0.15,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=["ankle_[RL]_Link"]),
    #             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["ankle_[RL]_Link"]),
    #              "foot_radius": 0.03, "about_landing_threshold": 0.08},
    # )
    # pen_joint_power_l1 = RewTerm(func=mdp.joint_powers_l1, weight=-2e-5)
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-05)

    # Gait reward
    # test_gait_reward = RewTerm(
    #     func=mdp.GaitReward,
    #     weight=0.1,
    #     params={
    #         "tracking_contacts_shaped_force": -2.0,
    #         "tracking_contacts_shaped_vel": -2.0,
    #         "gait_force_sigma": 25.0,
    #         "gait_vel_sigma": 0.25,
    #         "kappa_gait_probs": 0.05,
    #         "command_name": "gait_command",
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names="ankle_.*"),
    #     },
    # )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*"),
            "command_name": "base_velocity",
            "threshold_min": 0.2,
            "threshold_max": 0.5,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="ankle_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="ankle_.*"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP (Berkeley Style)"""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel_delayed,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "starting_step": 200*24
        }
    )

    modify_push_force = CurrTerm(
        func=mdp.modify_push_force,
        params={
            "term_name": "push_robot",
            "max_velocity": (3.0, 3.0),
            "interval": 200 * 24,
            "starting_step": 1500 * 24,
        },
    )

    modify_command_velocity_lin_x = CurrTerm(
        func=mdp.modify_command_velocity_x,
        params={
            "term_name": "rew_lin_vel_xy",
            "max_velocity": (-1.5, 1.5),
            "interval": 200 * 24,
            "starting_step": 2500 * 24,
            "update_rate": 0.04,
            "update_threshold": 0.6,
        }
    )

    modify_command_velocity_lin_y = CurrTerm(
        func=mdp.modify_command_velocity_y,
        params={
            "term_name": "rew_lin_vel_xy",
            "max_velocity": (-1, 1),
            "interval": 200 * 24,
            "starting_step": 2500 * 24,
            "update_rate": 0.04,
            "update_threshold": 0.25,
        }
    )
    
    modify_command_velocity_ang_z = CurrTerm(
        func=mdp.modify_command_velocity_angular,
        params={
            "term_name": "rew_ang_vel_z",
            "max_velocity": (-1.35, 1.35),
            "interval": 200 * 24,
            "starting_step": 2500 * 24,
            "update_rate": 0.04,
            "update_threshold": 0.25,
        }
    )

    modify_linear_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_lin_vel_xy",
            "interval": 300 * 24,
            "starting_step": 900 * 24,
            "update_rate": 0.95,
            "update_threshold": 0.67,
            "minimum_std": 0.09
        }
    )
    modify_angular_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_ang_vel_z",
            "interval": 300 * 24,
            "starting_step": 0,
            "update_rate": 0.975,
            "update_threshold": 0.5,
            "minimum_std": 0.09
        }
    )

    # velocity_curriculum = CurrTerm(
    #     func=mdp.velocity_curriculum,
    #     params={
    #         "command_name": "base_velocity",
    #         "max_steps": 15000 * 24,  # Placeholder, updated in train.py
    #         "x_config": {
    #             "start_frac": 0.0,
    #             "end_frac": 0.2,
    #             "min_range": (-0.4, 0.4),
    #             "max_range": (-1, 1.5),
    #         },
    #         "y_config": {
    #             "start_frac": 0.0,
    #             "end_frac": 0.2,
    #             "min_range": (-0.1, 0.1), # Not used as it starts at (0,0) till start_frac
    #             "max_range": (-0.7, 0.7),
    #         },
    #         "z_config": {
    #             "start_frac": 0.0,
    #             "end_frac": 0.2,
    #             "min_range": (-0.3, 0.3), # Not used as it starts at (0,0) till start_frac
    #             "max_range": (-1.0, 1.0),
    #         },
    #     },
    # )
    #
    # lin_vel_curriculum = CurrTerm(
    #         func=mdp.lin_vel_curriculum,
    #         params={
    #             "command_name": "base_velocity",
    #             "rwd_threshold": 0.7,
    #             "time_step": 2e-4 / 24,
    #             "max_lin_vel_x": (-1.0, 1.0),
    #             "max_lin_vel_y": (-0.75, 0.75),
    #         },
    # )


########################
# Environment definition
########################


@configclass
class SFEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: SFSceneCfg = SFSceneCfg(num_envs=4096, env_spacing=env_spacing)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization"""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        # simulation settings
        self.sim.dt = 0.005
        self.seed = 42
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

@configclass
class SFHIMEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: SFSceneCfg = SFSceneCfg(num_envs=4096, env_spacing=env_spacing)
    # Basic settings
    observations: HIMObservationsCfg = HIMObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization"""
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        # simulation settings
        self.sim.dt = 0.005
        self.seed = 42
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

# @configclass
# class SFHIMEnvCfg(SFEnvCfg):
#     observations: HIMObservationsCfg = HIMObservationsCfg()
