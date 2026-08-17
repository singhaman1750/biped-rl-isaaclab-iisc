import math

from isaaclab.envs.mdp import joint_deviation_l1
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.sd_brs1_identified_cfg import (
    SD_BRS1_IDENTIFIED_CFG,
    SD_BRS1_IDENTIFIED_CFG2,
)
from bipedal_locomotion.tasks.locomotion.cfg.SF.brs_base_env_cfg import (
    SDBRS1EnvCfg,
    SDBRS1HIMEnvCfg,
)
from bipedal_locomotion.tasks.locomotion.cfg.SF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
)

# Signs follow the URDF axes. Only HipPitch is mirrored between the legs, so the two hips
# take opposite signs, while KneePitch and AnklePitch share an axis across legs and take
# the same sign on both.
_JOINT_POS_DEFAULTS = {
    "HipRollR": 0.0,
    "HipRollL": 0.0,
    "HipPitchR": 0.2379,
    "HipPitchL": -0.2379,
    "KneePitchR": 0.4814,
    "KneePitchL": 0.4814,
    "AnkleRollR": 0.0,
    "AnkleRollL": 0.0,
    "AnklePitchR": 0.2435,
    "AnklePitchL": 0.2435,
}


######################
# SD_BRS1 Base Environments
######################


@configclass
class SDBRS1BaseEnvCfg(SDBRS1EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SD_BRS1_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = _JOINT_POS_DEFAULTS

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-4.5, 0.0, 4.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SDBRS1BaseEnvCfg_PLAY(SDBRS1BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.9, 0.9)
        self.episode_length_s = 30.0


@configclass
class SDBRS1BaseEnv2Cfg(SDBRS1EnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SD_BRS1_IDENTIFIED_CFG2.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = _JOINT_POS_DEFAULTS

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-4.5, 0.0, 4.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SDBRS1BaseEnv2Cfg_PLAY(SDBRS1BaseEnv2Cfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.9, 0.9)
        self.episode_length_s = 30.0


######################
# SD_BRS1 HIM Base Environments
######################


@configclass
class SDBRS1HIMBaseEnvCfg(SDBRS1HIMEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SD_BRS1_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = _JOINT_POS_DEFAULTS

        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.1)


@configclass
class SDBRS1HIMBaseEnvCfg_PLAY(SDBRS1HIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.add_link_mass = None
        self.curriculum.modify_push_force = None
        self.curriculum.modify_command_velocity_lin_x = None
        self.curriculum.modify_command_velocity_lin_y = None
        self.curriculum.modify_command_velocity_ang_z = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.9, 0.9)
        self.episode_length_s = 30.0


######################
# SD_BRS1 Blind Flat
######################


@configclass
class SDBRS1BlindFlatEnvCfg(SDBRS1BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class SDBRS1BlindFlatEnvCfg_PLAY(SDBRS1BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class SDBRS1BlindFlatEnv2Cfg(SDBRS1BaseEnv2Cfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class SDBRS1BlindFlatEnv2Cfg_PLAY(SDBRS1BaseEnv2Cfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


######################
# SD_BRS1 Blind Rough
######################


@configclass
class SDBRS1BlindRoughEnvCfg(SDBRS1BaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class SDBRS1BlindRoughEnvCfg_PLAY(SDBRS1BaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG


######################
# SD_BRS1 HIM Blind Flat
######################


@configclass
class SDBRS1HIMBlindFlatEnvCfg(SDBRS1HIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.heights = None
        self.curriculum.terrain_levels = None


@configclass
class SDBRS1HIMBlindFlatEnvCfg_PLAY(SDBRS1HIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.heights = None
        self.curriculum.terrain_levels = None


######################
# SD_BRS1 HIM Blind Rough
######################


@configclass
class SDBRS1HIMBlindRoughEnvCfg(SDBRS1HIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.heights = None
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class SDBRS1HIMBlindRoughEnvCfg_PLAY(SDBRS1HIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.heights = None
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG
