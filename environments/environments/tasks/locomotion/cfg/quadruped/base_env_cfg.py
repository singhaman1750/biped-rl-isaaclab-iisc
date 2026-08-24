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

from environments.tasks.locomotion import mdp
from environments.tasks.locomotion.mdp.curriculums import reduce_tracking_rewards_std

from .terrains_cfg import QUADRUPED_ROUGH_TERRAINS_CFG

##################
# Scene Definition
##################

env_spacing=2.5

@configclass
class QuadrupedPFSceneCfg(InteractiveSceneCfg):
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names="foot_.*_Link")},
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
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link")
            },
        )
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
        )

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
class CoptObservationsCfg:
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

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class MorphologyCfg(ObsGroup):
        """P_1, morphology and terrain privileged information"""

        link_lengths = ObsTerm(
            func=mdp.robot_link_lengths,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "parent_body_names": [
                    "hip_FR_thigh_Link",
                    "hip_FL_thigh_Link",
                    "hip_RR_thigh_Link",
                    "hip_RL_thigh_Link",
                    "knee_FR_Link",
                    "knee_FL_Link",
                    "knee_RR_Link",
                    "knee_RL_Link",
                ],
                "child_body_names": [
                    "knee_FR_Link",
                    "knee_FL_Link",
                    "knee_RR_Link",
                    "knee_RL_Link",
                    "foot_FR_Link",
                    "foot_FL_Link",
                    "foot_RR_Link",
                    "foot_RL_Link",
                ],
            },
            clip=(0.0, 100.0),
        )
        robot_mass = ObsTerm(func=mdp.robot_mass, clip=(0.0, 100.0))
        robot_inertia = ObsTerm(func=mdp.robot_inertia)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PredictedMorphologyCfg(ObsGroup):
        """P_1, morphology and terrain privileged information"""

        link_lengths = ObsTerm(
            func=mdp.robot_link_lengths,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "parent_body_names": [
                    "hip_FR_thigh_Link",
                    "hip_FL_thigh_Link",
                    "hip_RR_thigh_Link",
                    "hip_RL_thigh_Link",
                    "knee_FR_Link",
                    "knee_FL_Link",
                    "knee_RR_Link",
                    "knee_RL_Link",
                ],
                "child_body_names": [
                    "knee_FR_Link",
                    "knee_FL_Link",
                    "knee_RR_Link",
                    "knee_RL_Link",
                    "foot_FR_Link",
                    "foot_FL_Link",
                    "foot_RR_Link",
                    "foot_RL_Link",
                ],
            },
            clip=(0.0, 100.0),
        )
        robot_mass = ObsTerm(func=mdp.robot_mass, clip=(0.0, 100.0))
        robot_inertia = ObsTerm(func=mdp.robot_inertia)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PredictedPrivilegedCfg(ObsGroup):
        """P_2, ground-truth dynamic state, the decoder regression target"""

        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link")
            },
        )
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="foot_.*_Link")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

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
            params={"asset_cfg": SceneEntityCfg("robot", body_names="foot_.*_Link")},
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
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link")
            },
        )
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
        )
        link_lengths = ObsTerm(
            func=mdp.robot_link_lengths,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "parent_body_names": [
                    "hip_FR_thigh_Link",
                    "hip_FL_thigh_Link",
                    "hip_RR_thigh_Link",
                    "hip_RL_thigh_Link",
                    "knee_FR_Link",
                    "knee_FL_Link",
                    "knee_RR_Link",
                    "knee_RL_Link",
                ],
                "child_body_names": [
                    "knee_FR_Link",
                    "knee_FL_Link",
                    "knee_RR_Link",
                    "knee_RL_Link",
                    "foot_FR_Link",
                    "foot_FL_Link",
                    "foot_RR_Link",
                    "foot_RL_Link",
                ],
            },
            clip=(0.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 10
            # Required by HIMActorCritic
            self.flatten_history_dim = True

    @configclass
    class HistoryObsCfg(ObsGroup):
        """H, the n-step rolling history of the actor state"""

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
        heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-5.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 25
            self.flatten_history_dim = False

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    morphologyObs: MorphologyCfg = MorphologyCfg()
    predictedMorphologyObs: PredictedMorphologyCfg = PredictedMorphologyCfg()
    predictedPrivilegedObs: PredictedPrivilegedCfg = PredictedPrivilegedCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()


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
            self.history_length = 10
            # Required by HIMActorCritic
            self.flatten_history_dim = False


    @configclass
    class TargetEncCfg(ObsGroup):
        """Observation for critic group"""

        # robot base measurements
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

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

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
            clip=(-1.0, 1.0),
        )

        # Privileged observation
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(
            func=mdp.feet_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="foot_.*_Link")},
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
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link")
            },
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
    targetEnc: TargetEncCfg = TargetEncCfg()


@configclass
class EventsCfg:
    """Configuration for events"""

    # startup
    # A payload on the trunk. The range is an ABSOLUTE add and is deliberately small and
    # asymmetric, the trunk weighing only 0.674397 kg, so that the sampled mass cannot go
    # negative. randomize_rigid_body_mass does not guard against a negative mass unless
    # min_mass is given, and min_mass is given here as a second line of defence.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-0.3, 0.6),
            "operation": "add",
            "min_mass": 0.2,
        },
    )
    # The trunk is only 4.3 percent of this robot's 15.837077 kg, so perturbing it alone
    # cannot deliver the plus or minus ten percent of TOTAL mass the reference
    # configurations randomise over. This term supplies that, scaling every body.
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.90, 1.10),
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=["knee_.._Joint"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping_hip = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["hip_.._Joint"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    robot_joint_stiffness_and_damping_abad = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["abad_.._Joint"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.80, 1.20),
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
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # interval. Halved against the biped, Isaac Lab's own Go2 rough configuration
    # disabling the push entirely on the ground that a light robot on randomised
    # terrain is already perturbed enough.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP"""

    # termination related rewards
    keep_balance = RewTerm(func=mdp.stay_alive, weight=0.05)

    # tracking rewards. The 2 to 1 ratio of linear to angular is the one every
    # surveyed quadruped configuration uses.
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=12.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )

    # penalisations
    # 0.292 m is the standing height at which the 0.022 m foot sphere rests on the
    # ground given the 0.270 m kinematic drop of the nominal crouch.
    pen_base_height = RewTerm(
        func=mdp.base_height_rough_l2,
        params={
            "target_height": 0.292,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
        weight=-5.0,
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-5.0e-05)
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.075)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # Every body except the four feet. The feet are named foot_<LEG>_Link and are
    # therefore disjoint from all three expressions below.
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_Link", "abad_.*", "hip_.*", "knee_.*"],
            ),
            "threshold": 10.0,
        },
    )

    # The abduction joints are the degrees of freedom a policy widens first, to buy a
    # larger support polygon, and this robot's 0.264 m stance is narrow against its
    # 0.426 m leg. Follows the hip_pos term of Extreme Parkour.
    pen_abad_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["abad_.._Joint"])},
    )

    # gait shaping
    # threshold_min is half the stride period of a 2 Hz trot at a duty factor of one
    # half, which is what the canonical 0.5 s is for ANYmal's 1 Hz trot.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
            "command_name": "base_velocity",
            "threshold_min": 0.25,
            "threshold_max": 0.40,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*_Link"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*_Link"),
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
            "starting_step": 2000*24
        }
    )

    modify_push_force = CurrTerm(
        func=mdp.modify_push_force,
        params={
            "term_name": "push_robot",
            "max_velocity": (3.0, 3.0),
            "interval": 400 * 24,
            "starting_step": 6000 * 24,
            "increment_rate": 1.1,
            "decrement_rate" : 0.8,
            "minimum_velocity": 0.2
        },
    )

    modify_command_velocity_lin_x = CurrTerm(
        func=mdp.modify_command_velocity_x,
        params={
            "term_name": "rew_lin_vel_xy",
            "max_velocity": (-1.5, 1.5),
            "interval": 300 * 24,
            "starting_step": 6000 * 24,
            "update_rate": 0.015,
            "update_threshold": 0.6,
        }
    )

    modify_command_velocity_lin_y = CurrTerm(
        func=mdp.modify_command_velocity_y,
        params={
            "term_name": "rew_lin_vel_xy",
            "max_velocity": (-1, 1),
            "interval": 300 * 24,
            "starting_step": 6000 * 24,
            "update_rate": 0.015,
            "update_threshold": 0.4,
        }
    )

    modify_command_velocity_ang_z = CurrTerm(
        func=mdp.modify_command_velocity_angular,
        params={
            "term_name": "rew_ang_vel_z",
            "max_velocity": (-1.35, 1.35),
            "interval": 250 * 24,
            "starting_step": 6000 * 24,
            "update_rate": 0.05,
            "update_threshold": 0.4,
        }
    )

    modify_linear_tracking_reward_std = CurrTerm(
        func=mdp.reduce_tracking_rewards_std,
        params={
            "term_name": "rew_lin_vel_xy",
            "interval": 2000 * 24,
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
            "interval": 2000 * 24,
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
class QuadrupedPFEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: QuadrupedPFSceneCfg = QuadrupedPFSceneCfg(num_envs=4096, env_spacing=env_spacing)
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
class QuadrupedPFHIMEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: QuadrupedPFSceneCfg = QuadrupedPFSceneCfg(num_envs=4096, env_spacing=env_spacing)
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

@configclass
class QuadrupedPFCoptEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the test environment"""

    # Scene settings
    scene: QuadrupedPFSceneCfg = QuadrupedPFSceneCfg(num_envs=4096, env_spacing=env_spacing)
    # Basic settings
    observations: CoptObservationsCfg = CoptObservationsCfg()
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
        self.episode_length_s = 20
        self.sim.render_interval = 2 * self.decimation
        # simulation settings
        self.sim.dt = 0.005
        self.seed = 42
        # PhysX GPU rigid contact patch buffer. Default (5 * 2**15 = 163840) is too
        # small at num_envs=4096 with the population-wide resets this task performs
        # at every design swap, observed overflowing with "please increase its size
        # to at least 167741". Set with headroom above that observed minimum rather
        # than matched to it, since a transient spike from a synchronised reset can
        # exceed any single observed value.
        self.sim.physx.gpu_max_rigid_patch_count = 2**19  # 524288
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt



# @configclass
# class QuadrupedPFHIMEnvCfg(QuadrupedPFEnvCfg):
#     observations: HIMObservationsCfg = HIMObservationsCfg()
