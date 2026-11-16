"""The ten joint KScale, its hip yaw frozen in the URDF.

ARM 14 of plans/kscale_gait_shaping.md section 21.14, APPLIED 2026-10-01. The arm is
INDEPENDENT of every arm around it and is run to gauge what one degree of freedom contributes
to this robot's locomotion rather than to repair anything, so it needs only arm 10's commanded
target clip and its result is readable against 2026-09-29_06-15-58 directly.

This module mirrors the flat branch of kscale_solefoot_env_cfg.py line for line. The HIM
branch is deliberately NOT reproduced, this arm having no bearing on the estimator and the HIM
encoder's num_output_dim of 19 at agents/limx_rsl_rl_ppo_cfg.py:369 being sized against the
twelve joint observation, which would have to be re-derived first.
"""

from isaaclab.utils import configclass

from environments.assets.config.kscale_identified_cfg import (
    KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED,
)
from environments.tasks.locomotion.cfg.SF.kscale_base_env_cfg import (
    KscaleEnvHipYawFixedCfg,
)

###############################################
# kscale hip yaw fixed Base Environments
###############################################


@configclass
class KscaleHipYawFixedBaseEnvCfg(KscaleEnvHipYawFixedCfg):
    def __post_init__(self):
        super().__post_init__()

        # prim_path is still applied through replace and that is deliberate.  The user's
        # direction of 2026-10-01 concerns the construction of the ARTICULATION configuration,
        # which kscale_identified_cfg.py now writes out explicitly, and this call sets one
        # scene local field on an already constructed object exactly as
        # KscaleBaseEnvCfg.__post_init__ does at kscale_solefoot_env_cfg.py:23.  Writing
        # prim_path into the definition would put a scene detail in the asset file and would
        # diverge from every other robot in the tree.
        self.scene.robot = KSCALE_IDENTIFIED_CFG_HIP_YAW_FIXED.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (-3.25, 0.0, 3.25)
        self.viewer.lookat = (0.0, 0.0, 0.5)


@configclass
class KscaleHipYawFixedBaseEnvCfg_PLAY(KscaleHipYawFixedBaseEnvCfg):
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


###############################################
# kscale hip yaw fixed Blind Flat
###############################################


@configclass
class KscaleHipYawFixedBlindFlatEnvCfg(KscaleHipYawFixedBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None


@configclass
class KscaleHipYawFixedBlindFlatEnvCfg_PLAY(KscaleHipYawFixedBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
