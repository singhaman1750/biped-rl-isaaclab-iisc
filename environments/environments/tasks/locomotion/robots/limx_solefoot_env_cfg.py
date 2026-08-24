import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from environments.assets.config.solefoot_cfg import SOLEFOOT_CFG, SOLEFOOT_CFG_URDF
from environments.assets.config.solefoot_identified_cfg import (  # SOLEFOOT_IDENTIFIED_MULTIUSD_CFG,
    SOLEFOOT_IDENTIFIED_CFG,
    SOLEFOOT_IDENTIFIED_CFG_URDF,
)
from environments.tasks.locomotion import mdp
from environments.tasks.locomotion.cfg.SF.limx_base_env_cfg import (
    SFCoptEnvCfg,
    SFEnvCfg,
    SFHIMEnvCfg,
)
from environments.tasks.locomotion.cfg.SF.limx_berkeley_env_cfg import (
    SFBerkeleyEnvCfg,
)
from environments.tasks.locomotion.cfg.SF.terrains_cfg import (
    BERKELEY_MIMIC_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)

######################
# Solefoot Base Environment
######################


@configclass
class SFBaseEnvCfg(SFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SFBaseEnvCfg_PLAY(SFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.modify_command_velocity = None
        self.curriculum.modify_push_force = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


@configclass
class SFHIMBaseEnvCfg(SFHIMEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SFHIMBaseEnvCfg_PLAY(SFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.modify_command_velocity = None
        self.curriculum.modify_push_force = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)

@configclass
class SFCoptBaseEnvCfg(SFCoptEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.replicate_physics = False

        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SFCoptBaseEnvCfg_PLAY(SFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.modify_command_velocity = None
        self.curriculum.modify_push_force = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


@configclass
class SFBaseEnvUrdfCfg(SFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_CFG_URDF.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFBaseEnvUrdfCfg_PLAY(SFBaseEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.modify_command_velocity = None
        self.curriculum.modify_push_force = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


@configclass
class SFHIMBaseEnvUrdfCfg(SFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_CFG_URDF.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMBaseEnvUrdfCfg_PLAY(SFHIMBaseEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.modify_command_velocity = None
        self.curriculum.modify_push_force = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.35, 1.35)


############################
# Solefoot Blind Flat Environment
############################


@configclass
class SFBlindFlatEnvCfg(SFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class SFBlindFlatEnvCfg_PLAY(SFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None

@configclass
class SFBlindFlatEnvUrdfCfg(SFBaseEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class SFBlindFlatEnvUrdfCfg_PLAY(SFBaseEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class SFHIMBlindFlatEnvUrdfCfg(SFHIMBaseEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class SFHIMBlindFlatEnvUrdfCfg_PLAY(SFHIMBaseEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None

@configclass
class SFCoptBlindFlatEnvCfg(SFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class SFCoptBlindFlatEnvCfg_PLAY(SFCoptBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


#############################
# Solefoot Blind Rough Environment
#############################


@configclass
class SFBlindRoughEnvCfg(SFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG


@configclass
class SFBlindRoughEnvCfg_PLAY(SFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class SFBlindRoughEnvUrdfCfg(SFBaseEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG


@configclass
class SFBlindRoughEnvUrdfCfg_PLAY(SFBaseEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class SFHIMBlindRoughEnvUrdfCfg(SFHIMBaseEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG


@configclass
class SFHIMBlindRoughEnvUrdfCfg_PLAY(SFHIMBaseEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False

@configclass
class SFCoptBlindRoughEnvCfg(SFCoptBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG


@configclass
class SFCoptBlindRoughEnvCfg_PLAY(SFCoptBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False




##############################
# Solefoot Blind Stairs Environment
##############################


@configclass
class SFBlindStairEnvCfg(SFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.rewards.rew_lin_vel_xy.weight = 2.0
        self.rewards.rew_ang_vel_z.weight = 1.5
        self.rewards.pen_lin_vel_z.weight = -1.0
        self.rewards.pen_ang_vel_xy.weight = -0.05
        self.rewards.pen_action_rate.weight = -0.01
        self.rewards.pen_flat_orientation.weight = -2.5
        self.rewards.pen_undesired_contacts.weight = -1.0

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class SFBlindStairEnvCfg_PLAY(SFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(
            difficulty_range=(0.5, 0.5)
        )


#############################
# Solefoot Flat Environment
#############################


@configclass
class SFFlatEnvCfg(SFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.curriculum.terrain_levels = None


@configclass
class SFFlatEnvCfg_PLAY(SFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.curriculum.terrain_levels = None


#############################
# Solefoot Rough Environment
#############################


@configclass
class SFRoughEnvCfg(SFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SFRoughEnvCfg_PLAY(SFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG


@configclass
class SFRoughEnvUrdfCfg(SFBaseEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SFRoughEnvUrdfCfg_PLAY(SFBaseEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG


@configclass
class SFHIMRoughEnvUrdfCfg(SFHIMBaseEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SFHIMRoughEnvUrdfCfg_PLAY(SFHIMBaseEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG


##############################
# Solefoot Blind Stairs Environment
##############################


@configclass
class SFStairEnvCfg(SFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class SFStairEnvCfg_PLAY(SFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(
            difficulty_range=(0.5, 0.5)
        )


#############################
# Solefoot HIM Environments
#############################

@configclass
class SFHIMBlindFlatEnvCfg(SFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.heights = None
        # self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class SFHIMBlindFlatEnvCfg_PLAY(SFHIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.heights = None
        # self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class SFHIMBlindRoughEnvCfg(SFHIMBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG
        self.observations.policy.heights = None
        # self.observations.critic.heights = None


@configclass
class SFHIMBlindRoughEnvCfg_PLAY(SFHIMBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.heights = None
        # self.observations.critic.heights = None

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BERKELEY_MIMIC_TERRAINS_CFG
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False


@configclass
class SFBerkeleyBaseEnvCfg(SFBerkeleyEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class SFBerkeleyBaseEnvCfg_PLAY(SFBerkeleyEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SOLEFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        # self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)

        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"

        # update viewport camera to follow the robot root
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-2.5, 0.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 0.5)

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None
        self.events.add_link_mass = None

        # disable curriculum for play
        self.curriculum.modify_command_velocity = None
        self.curriculum.modify_push_force = None

        # set maximum commanded velocity
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 3.0)


@configclass
class SFBerkeleyRoughEnvCfg(SFBerkeleyBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # general settings
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


class SFBerkeleyRoughEnvCfg_PLAY(SFBerkeleyBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        # general settings
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False


#########################################
# Solefoot Identified Actuator Environments
# Inherits all MDP settings from the base classes; only swaps the robot to
# use the Berkeley IdentifiedActuator model instead of ImplicitActuator.
#########################################


@configclass
class SFIdentifiedBlindFlatEnvCfg(SFBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBlindFlatEnvCfg_PLAY(SFBlindFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMIdentifiedBlindFlatEnvCfg(SFHIMBlindFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMIdentifiedBlindFlatEnvCfg_PLAY(SFHIMBlindFlatEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBlindFlatEnvUrdfCfg(SFBlindFlatEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBlindFlatEnvUrdfCfg_PLAY(SFBlindFlatEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMIdentifiedBlindFlatEnvUrdfCfg(SFHIMBlindFlatEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMIdentifiedBlindFlatEnvUrdfCfg_PLAY(SFHIMBlindFlatEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBlindRoughEnvCfg(SFBlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBlindRoughEnvCfg_PLAY(SFBlindRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMIdentifiedBlindRoughEnvCfg(SFHIMBlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMIdentifiedBlindRoughEnvCfg_PLAY(SFHIMBlindRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBlindRoughEnvUrdfCfg(SFBlindRoughEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

@configclass
class SFIdentifiedBlindRoughCoptEnvUrdfCfg(SFIdentifiedBlindRoughEnvUrdfCfg):
    def __post_init__(self):
        # This one is without privileged observations
        super().__post_init__()
        self.scene.replicate_physics = False
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBlindRoughEnvUrdfCfg_PLAY(SFBlindRoughEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMIdentifiedBlindRoughEnvUrdfCfg(SFHIMBlindRoughEnvUrdfCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFHIMIdentifiedBlindRoughEnvUrdfCfg_PLAY(SFHIMBlindRoughEnvUrdfCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG_URDF.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBerkeleyRoughEnvCfg(SFBerkeleyRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }


@configclass
class SFIdentifiedBerkeleyRoughEnvCfg_PLAY(SFBerkeleyRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = SOLEFOOT_IDENTIFIED_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }
