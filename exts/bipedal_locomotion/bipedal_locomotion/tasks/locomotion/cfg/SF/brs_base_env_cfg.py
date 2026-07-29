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

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.mdp.curriculums import reduce_tracking_rewards_std

##################
# Scene Definition
##################

env_spacing = 2.5


@configclass
class SDBRS1SceneCfg(InteractiveSceneCfg):
    """Configuration for the SD_BRS1 scene"""

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
        prim_path="{ENV_REGEX_NS}/Robot/Part_Torso",
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

    # Periodic walking clock, consumed by rew_gait in RewardsCfg and observed through
    # ObservationsCfg.PolicyCfg.gait_phase. Added 2026-07-20, see /ws/GAIT_STRATEGY.md section 2.7.
    # Run 2026-07-20_07-52-14 converged onto a one legged hold, single support occupancy at least
    # 0.909 with single support phases near 2.5 s against a human 0.4 s, because every gait term in
    # the reward set is maximised by holding and not one has a value that depends on the number of
    # steps taken. The clock family (Siekmann arXiv:2011.01387, Walk These Ways arXiv:2212.03238,
    # Humanoid-Gym arXiv:2404.05695) is the only surveyed construction that states WHEN each foot
    # should be down, so it cannot be farmed by holding, holding being off schedule by definition.
    #
    # frequency 1.0 Hz with stance duration 0.6 and anti phase offset 0.5 gives, per 1.0 s cycle
    # and per foot, a 0.4 s swing, a 0.4 s single support phase and a 0.1 s double support phase on
    # each side, which is exactly the reference walk the hold was priced against. The ranges are
    # degenerate on purpose, a fixed clock removes a confound from this experiment, and widening
    # them is the follow up once alternation is established. If they are ever widened, add
    # mdp.get_gait_command to the observation groups, the parameters carry no information while
    # they are constant.
    gait_command = mdp.UniformGaitCommandCfg(
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=mdp.UniformGaitCommandCfg.Ranges(
            frequencies=(1.0, 1.0),
            offsets=(0.5, 0.5),
            durations=(0.6, 0.6),
            # unused by GaitReward, carried because GaitCommand samples four parameters, set to
            # the clearance target of rew_foot_clearance so the two agree if it is ever read
            swing_height=(0.08, 0.08),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.4,
        use_default_offset=True,
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
        # [sin, cos] of the walking clock phase. Without this the gait reward is not learnable,
        # the policy cannot know which foot the schedule wants down at a given instant so the term
        # would average to noise. get_gait_phase uses the same
        # remainder(episode_length_buf * step_dt * frequency, 1) that
        # GaitReward.compute_contact_targets uses, so observation and reward agree exactly.
        gait_phase = ObsTerm(func=mdp.get_gait_phase)

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
            params={"asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]")},
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
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]")},
            clip=(-5000.0, 5000.0),
        )

        gait_phase = ObsTerm(func=mdp.get_gait_phase)

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
        last_action = ObsTerm(
            func=mdp.last_action,
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]")},
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
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]")},
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

    prepare_quantity_for_sd_brs1 = EventTerm(
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
            "asset_cfg": SceneEntityCfg("robot", body_names="Part_Torso"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Link[1-6][LR]"),
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=["HipRoll[LR]", "HipPitch[LR]"]),
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=["KneePitch[LR]"]),
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
            "asset_cfg": SceneEntityCfg("robot", joint_names=["AnkleRoll[LR]", "AnklePitch[LR]"]),
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
    # joint-specific position randomisation, ranges taken from the URDF joint limits
    # (SD_BRS1_Assembly2.urdf), clamped to the soft joint limits inside the event term
    reset_hip_roll_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "HipRoll[RL]",
            "position_range": (-0.1295, 0.1295),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_hip_pitch_r_joint = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "HipPitchR",
            # recentred on the crouch nominal, see NATURAL_GAIT_PLAN.md 5.2.3. The offset is
            # ADDED to the default, so a symmetric range straddles the new pose.
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_hip_pitch_l_joint = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "HipPitchL",
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_knee_pitch_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "KneePitch[RL]",
            # was (0.0, 0.475), one-sided because the old default sat on the extension stop.
            # About the 0.4814 nominal this samples [0.241, 0.721], inside the soft band.
            "position_range": (-0.24, 0.24),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_ankle_roll_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "AnkleRoll[RL]",
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_ankle_pitch_joints = EventTerm(
        func=mdp.reset_joint_by_offset,
        mode="reset",
        params={
            "joint_name": "AnklePitch[RL]",
            # narrower than the others because the ankle keeps only 0.165 rad of soft-limit
            # margin at the nominal, so a wider sample would clamp. Gives [0.094, 0.394].
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
        func=mdp.no_fly,
        weight=15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
            "threshold": 1.0,
            # Read the CURRENT contact frame. The sensor writes the newest sample to index
            # 0, so the function's default of -1 reads the OLDEST of the four buffered
            # samples, roughly 15 ms stale, which a provenance survey established to be a
            # porting error rather than a design. See NATURAL_GAIT_PLAN.md section 5.2.6.
            # Set here only, so the TRON1 SF caller keeps its behaviour bit for bit.
            "history_index": 0,
        },
    )
    rew_keep_ankle_pitch_zero_in_air = RewTerm(
        func=mdp.keep_ankle_pitch_zero_in_air,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["AnklePitchL", "AnklePitchR"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
            # without this the term returns its maximum of 1 whenever both feet are planted,
            # paying a standing bonus that opposes the stepping objective
            "require_airborne": True,
            # index 0 pairs with 1, giving the current-or-previous debounce the Isaac Gym
            # ancestor expresses as contact_filt = contact OR last_contacts. The default of
            # -1 pairs with -2, the two oldest samples. See section 5.2.6.
            "history_index": 0,
        },
    )
    # Phase A2 of NATURAL_GAIT_PLAN.md, swing-phase knee flexion, gated to swing so the
    # efficient extended stance knee is left untouched.
    #
    # This is the v2 MONOTONE form. The v1 Gaussian was trained as run 2026-07-23_11-31-57
    # and produced literally nothing, never exceeding 1.05e-6 against a saturation of 10,
    # because its target sat 4.4 tolerances from the policy's actual swing knee of 0.22 rad
    # and so carried no usable gradient anywhere the policy stood. See section 2.6. The ramp
    # instead pays for flexion BEYOND the stance nominal and has the constant gradient
    # 1 / (cap - nominal) across its whole active range.
    #
    # Comment this term out to run the ablation against the pose change alone, which is how
    # every surveyed humanoid (G1, H1, Go1) obtains knee flexion, with no knee reward at all.
    # rew_knee_flexion = RewTerm(
    #     func=mdp.knee_flexion_in_swing_v2,
    #     weight=10.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["KneePitchL", "KneePitchR"]),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
    #         # the stance nominal carried by _JOINT_POS_DEFAULTS, so the term asks only for
    #         # flexion BEYOND the posture the robot already holds
    #         "nominal": 0.4814,
    #         "cap": 0.9,
    #         "force_threshold": 1.0,
    #     },
    # )
    # Section 5.2.5. The established remedy for a limb escaping into an off-axis degree of
    # freedom, mirroring the IsaacLab G1 recipe which applies joint_deviation_l1 to hip roll
    # and hip yaw at -0.1 and deliberately NOT to hip pitch or the knee, the two joints that
    # must swing freely. HipYaw is a fixed joint on this robot, so hip roll is the whole
    # off-axis set. Also discharges the dead joint_deviation_l1 import noted in section 2.4.
    pen_hip_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["HipRoll[LR]"])},
    )
    # Penalise ankle joints deviating from the nominal crouch pose. This serves the same
    # purpose as pen_hip_deviation but for the ankle complex, discouraging the policy from
    # driving the ankle into extreme positions where the shank motor housing (Link4) and
    # the foot yoke (Link6) interpenetrate. Link4 and Link6 are grandparent-grandchild
    # (separated by the collision-less Link5) so PhysX does not filter their contacts, but
    # with enabled_self_collisions=False the simulator generates no contact forces at all,
    # leaving the policy free to drive the ankle through the shank. This soft penalty
    # provides the missing gradient. Mirrors the G1 recipe which applies joint_pos_limits
    # at -1.0 specifically to ankle joints (rough_env_cfg.py lines 51-55) and
    # joint_deviation_l1 at -0.1 to hip roll/yaw.
    pen_ankle_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["AnklePitch[LR]", "AnkleRoll[LR]"]
            )
        },
    )

    pen_base_height = RewTerm(
        func=mdp.base_height_rough_l2,
        params={
            # The previous 1.0 m target is KINEMATICALLY UNREACHABLE. For a balanced,
            # flat-footed, vertical-torso stance the ankle pitch limit of 0.454 rad binds
            # first and puts the floor at 1.0891 m; 1.0 m would need 0.654 rad of ankle,
            # 44 percent past the stop. See NATURAL_GAIT_PLAN.md section 2.8. The penalty
            # could therefore never be discharged by bending the knee and was instead
            # discharged by trunk pitch and leg splay, which is to say the -300 weight was
            # buying the forward lean that pen_flat_orientation pays to prevent.
            # 1.15 m is the standing height of the new crouched nominal pose.
            "target_height": 1.15,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
        # The crouch is now carried by the default pose, so this term reverts to
        # discouraging collapse and hyperextension rather than forcing a posture.
        weight=-30.0,
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-5)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00001)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-2.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["Link[1-5][LR]", "Part_Torso"],
            ),
            "threshold": 10.0,
        },
    )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.075)
    pen_joint_torque_rate = RewTerm(
        func=mdp.JointTorqueRatePenalty,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-50.0)
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100,
        params={"min_feet_distance": 0.21, "feet_links_name": ["Link6[RL]"]},
    )
    pen_feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link6[RL]"]),
            # kept in step with pen_base_height's target_height above
            "base_height_target": 1.15,
            # the Link6 frame sits at the ankle and the sole plate extends 0.124 m below it,
            # verified against the collision mesh; the previous 0.03 is TRON1 point-foot geometry
            "foot_radius": 0.124,
            # ~0.025 * base_height_target, so only ground-level foot speed is penalised
            "height_decay_scale": 0.03,
        },
    )
    pen_joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-05)

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=12.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
            "asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]"),
        },
    )
    rew_foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward_v2,
        weight=20.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]"),
            # true sole clearance above the ground, not a body frame height (see v2 docstring).
            # 0.08 m matches the intent of the previous frame target of 0.20 m, which for a
            # level foot was 0.20 - 0.124 = 0.076 m of sole clearance.
            "target_height": 0.08,
            "std": 0.035,
            "tanh_mult": 1.0,
            # Support set of the Link6 sole, taken from the collision mesh. The sole face is
            # at z=-0.124 over x -0.1091..0.1521, y +/-0.0970, with chamfered fore and aft
            # edges rising to z=-0.1144. Reproduces the mesh's lowest point to within 0.85 mm
            # over pitch +/-50 deg and roll +/-30 deg. Same table serves both feet.
            "sole_offsets": [
                [-0.1091, -0.0970, -0.1240],
                [-0.1091, 0.0970, -0.1240],
                [0.1521, -0.0970, -0.1240],
                [0.1521, 0.0970, -0.1240],
                [-0.1206, -0.0970, -0.1203],
                [-0.1206, 0.0970, -0.1203],
                [0.1636, -0.0970, -0.1203],
                [0.1636, 0.0970, -0.1203],
                [-0.1262, -0.0970, -0.1144],
                [-0.1262, 0.0970, -0.1144],
                [0.1692, -0.0970, -0.1144],
                [0.1692, 0.0970, -0.1144],
            ],
            # redundant with the sole measurement, kept as defence in depth
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
            "force_threshold": 1.0,
        },
    )

    # Periodic phase reward, Siekmann's construction (arXiv:2011.01387) in the Walk These Ways
    # form (arXiv:2212.03238). Penalises foot force during scheduled swing and foot velocity
    # during scheduled stance against the clock in CommandsCfg.gait_command, so it is zero for a
    # gait that matches the schedule and negative otherwise. Added 2026-07-20 to answer the one
    # legged hold of run 2026-07-20_07-52-14, see /ws/GAIT_STRATEGY.md section 2.7 and phase 1b.
    #
    # Weight arithmetic, computed by reimplementing compute_contact_targets and both reward halves
    # over one cycle rather than estimated. With both scales at -1.0 the function lies in [-2, 0].
    # The observed hold scores -0.208 on the force half, since the planted foot carries load
    # through the 40 percent of the cycle the clock assigns to its swing, and -0.280 on the
    # velocity half, since the raised foot is waved at 0.86 m/s through the 60 percent assigned to
    # its stance, totalling -0.488. A walk matching the clock scores -0.079, not zero, because the
    # kappa smoothing makes the scheduled transitions soft while real contact switches are hard.
    # The discriminating gap is therefore 0.408, not the 0.5 first estimated by hand. The barrier
    # the hold enjoys over a real walk is 13.3 per second, so a weight of 40 turns the gap into
    # 16.3 per second and overturns it with a margin near a quarter, whereas 30 would have given
    # 12.3 and fallen short. Raise to 50 if alternation still does not appear, lower to 30 if early
    # training destabilises, accepting that 30 alone will not overturn the hold.
    #
    # Known property. GaitReward carries no zero command gate, so the roughly twenty percent of
    # commands below the 0.1 m/s threshold ask the robot to march in place. That is acceptable
    # here, it is still alternation, and it is van Marum's first objection to clocks
    # (arXiv:2404.19173). Set the weight to 0.0 to disable the term without removing the wiring.
    rew_gait = RewTerm(
        func=mdp.GaitReward,
        weight=40.0,
        params={
            "tracking_contacts_shaped_force": -1.0,
            "tracking_contacts_shaped_vel": -1.0,
            # 1 - exp(-F^2/sigma), saturated above roughly 10 N, so any real load during swing counts
            "gait_force_sigma": 25.0,
            # 1 - exp(-v^2/sigma), 0.22 at 0.25 m/s and 0.63 at 0.5 m/s, the stance slip band
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            # both must resolve the feet in the same order, which the identical pattern guarantees
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Link6[LR]"),
            "asset_cfg": SceneEntityCfg("robot", body_names="Link6[LR]"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Part_Torso"),
            "threshold": 1.0,
        },
    )
    low_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        # Lowered from 0.65. This term stands in for a fall detector because the base of
        # this robot may never contact the ground even when it falls, so base_contact alone
        # is insufficient. At 0.65 against a 1.15 m stance it also truncated exactly the
        # low excursions in which knee flexion would be discovered, and no IsaacLab
        # humanoid reference config uses a height termination at all. See section 3.7.
        params={"minimum_height": 0.4},
    )


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
        func=mdp.modify_push_force_v2,
        params={
            "term_name": "push_robot",
            "max_velocity": (3.0, 3.0),
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
class SDBRS1EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the SD_BRS1 standard PPO environment"""

    scene: SDBRS1SceneCfg = SDBRS1SceneCfg(num_envs=4096, env_spacing=env_spacing)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        self.sim.dt = 0.005
        self.seed = 42
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class SDBRS1HIMEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the SD_BRS1 HIM environment"""

    scene: SDBRS1SceneCfg = SDBRS1SceneCfg(num_envs=4096, env_spacing=env_spacing)
    observations: HIMObservationsCfg = HIMObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        self.sim.dt = 0.005
        self.seed = 42
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
