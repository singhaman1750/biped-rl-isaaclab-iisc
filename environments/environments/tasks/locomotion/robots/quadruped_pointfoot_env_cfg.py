from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import (
    UnitreeGo1RoughEnvCfg,
)

from environments.assets.config.quadruped_identified_cfg import QUADRUPED_IDENTIFIED_CFG
from environments.tasks.locomotion.cfg.quadruped.base_env_cfg import (
    QuadrupedPFCoptEnvCfg,
    QuadrupedPFEnvCfg,
    QuadrupedPFHIMEnvCfg,
)
from environments.tasks.locomotion.cfg.quadruped.terrains_cfg import (
    QUADRUPED_ROUGH_TERRAINS_CFG,
    QUADRUPED_ROUGH_TERRAINS_PLAY_CFG,
)

# The nominal crouch of the MuJoCo keyframe. The knee is the negation of twice the hip,
# which places each foot exactly beneath its hip pitch axis at a drop of 0.270 m.
QUADRUPED_INIT_JOINT_POS = {
    "abad_.._Joint": 0.0,
    "hip_.._Joint": 0.884337,
    "knee_.._Joint": -1.768673,
}


######################
# Quadruped Base Environments
######################


@configclass
class QuadrupedPFBaseEnvCfg(QuadrupedPFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = QUADRUPED_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = dict(QUADRUPED_INIT_JOINT_POS)

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # The viewer offsets are scaled to a 0.292 m robot, the biped's (-2.5, 0, 2.5)
        # framing a body three times this one's standing height.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.2, 0.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.25)


@configclass
class QuadrupedPFBaseEnvCfg_PLAY(QuadrupedPFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomisation for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.terrain_levels = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.curriculum.modify_linear_tracking_reward_std = None
        self.curriculum.modify_angular_tracking_reward_std = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


######################
# Quadruped Base Environments
######################


@configclass
class QuadrupedPFHIMBaseEnvCfg(QuadrupedPFHIMEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = QUADRUPED_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = dict(QUADRUPED_INIT_JOINT_POS)

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # The viewer offsets are scaled to a 0.292 m robot, the biped's (-2.5, 0, 2.5)
        # framing a body three times this one's standing height.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.2, 0.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.25)


@configclass
class QuadrupedPFHIMBaseEnvCfg_PLAY(QuadrupedPFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomisation for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.terrain_levels = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.curriculum.modify_linear_tracking_reward_std = None
        self.curriculum.modify_angular_tracking_reward_std = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


######################
# Quadruped Base Environments
######################


@configclass
class QuadrupedPFCoptBaseEnvCfg(QuadrupedPFCoptEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.replicate_physics = False

        self.scene.robot = QUADRUPED_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = dict(QUADRUPED_INIT_JOINT_POS)

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # The viewer offsets are scaled to a 0.292 m robot, the biped's (-2.5, 0, 2.5)
        # framing a body three times this one's standing height.
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.2, 0.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.25)


@configclass
class QuadrupedPFCoptBaseEnvCfg_PLAY(QuadrupedPFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomisation for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.terrain_levels = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.curriculum.modify_linear_tracking_reward_std = None
        self.curriculum.modify_angular_tracking_reward_std = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


@configclass
class QuadrupedPFBlindFlatEnvCfg(QuadrupedPFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # the scene terrain is already a plane, only the level curriculum must go
        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFBlindFlatEnvCfg_PLAY(QuadrupedPFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFBlindRoughEnvCfg(QuadrupedPFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG


@configclass
class QuadrupedPFBlindRoughEnvCfg_PLAY(QuadrupedPFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_PLAY_CFG


@configclass
class QuadrupedPFHIMBlindFlatEnvCfg(QuadrupedPFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # the scene terrain is already a plane, only the level curriculum must go
        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFHIMBlindFlatEnvCfg_PLAY(QuadrupedPFHIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFHIMBlindRoughEnvCfg(QuadrupedPFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG


@configclass
class QuadrupedPFHIMBlindRoughEnvCfg_PLAY(QuadrupedPFHIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_PLAY_CFG


@configclass
class QuadrupedPFCoptBlindFlatEnvCfg(QuadrupedPFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # the scene terrain is already a plane, only the level curriculum must go
        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFCoptBlindFlatEnvCfg_PLAY(QuadrupedPFCoptBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class QuadrupedPFCoptBlindRoughEnvCfg(QuadrupedPFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG


@configclass
class QuadrupedPFCoptBlindRoughEnvCfg_PLAY(QuadrupedPFCoptBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_PLAY_CFG


######################################
# Go1 Asset Ablation, Debug
######################################


@configclass
class QuadrupedPFGo1AssetBaseEnvCfg(QuadrupedPFEnvCfg):
    """The quadruped task's own environment and MDP stack, unchanged, with the Unitree
    Go1's shipped USD in place of QUADRUPED_IDENTIFIED_CFG.

    Reward magnitudes tuned to the quadruped's own geometry (pen_base_height's
    target_height=0.33, the 0.022 m foot radius in the landing and clearance terms) are
    left untouched and are therefore wrong for Go1's 0.4 m standing height, this task
    exists to measure throughput, not to converge a gait.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"

        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "trunk"

        for term in (
            self.rewards.feet_air_time,
            self.rewards.pen_feet_hold,
            self.rewards.feet_slide,
            self.rewards.rew_foot_clearance,
            self.rewards.pen_foot_landing_vel,
            self.rewards.pen_feet_impact,
        ):
            if "sensor_cfg" in term.params:
                term.params["sensor_cfg"].body_names = ".*_foot"
            if "asset_cfg" in term.params:
                term.params["asset_cfg"].body_names = ".*_foot"
        self.observations.critic.feet_lin_vel.params["asset_cfg"].body_names = ".*_foot"
        self.observations.critic.feet_contact_force.params["sensor_cfg"].body_names = (
            ".*_foot"
        )

        # Go1's own hip_joint/thigh_joint/calf_joint are respectively the abduction, hip
        # flexion, and knee axes, see the derivation comments in quadruped_identified_cfg.py.
        self.rewards.pen_abad_deviation.params["asset_cfg"].joint_names = [
            ".*_hip_joint"
        ]
        self.events.robot_joint_stiffness_and_damping_abad.params[
            "asset_cfg"
        ].joint_names = [".*_hip_joint"]
        self.events.robot_joint_stiffness_and_damping_hip.params[
            "asset_cfg"
        ].joint_names = [".*_thigh_joint"]
        self.events.robot_joint_stiffness_and_damping_knee.params[
            "asset_cfg"
        ].joint_names = [".*_calf_joint"]

        # No verified Go1 equivalent for the quadruped's four-name non-foot contact list.
        # Go1's own rough_env_cfg.py:54 disables this term rather than guess one, followed
        # here.
        self.rewards.pen_undesired_contacts = None

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.4, 0.0, 1.6)
        self.viewer.lookat = (0.0, 0.0, 0.4)


@configclass
class QuadrupedPFGo1AssetBlindRoughEnvCfg(QuadrupedPFGo1AssetBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG


###############################################
# Go1 Native, Population and Terrain Matched, Debug
###############################################


@configclass
class Go1NativeMatchedEnvCfg(UnitreeGo1RoughEnvCfg):
    """IsaacLab's own shipped Go1 rough task, `UnitreeGo1RoughEnvCfg`, completely
    unchanged except for the population and the terrain generator, both overridden here
    to match this repository's quadruped task, `QuadrupedPFBlindRoughEnvCfg`, and its Go1
    asset ablation, `QuadrupedPFGo1AssetBlindRoughEnvCfg` above, exactly.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 7000
        self.scene.terrain.terrain_generator = QUADRUPED_ROUGH_TERRAINS_CFG
