import gymnasium as gym

from environments.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    PF_TRON1AFlatPPORunnerCfg,
    SD_BRS1FlatPPORunnerCfg,
    SF_Berkeley_PPORunnerCfg,
    SF_TRON1AFlatPPORunnerCfg,
    SFCoptLearnedModelPPORunnerCfg,
    SFCoptPPORunnerCfg,
    WF_TRON1AFlatPPORunnerCfg,
)
from environments.tasks.locomotion.agents.quadruped_rsl_rl_ppo_cfg import (
    PFQuadrupedCoptLearnedModelPPORunnerCfg,
    PFQuadrupedCoptPPORunnerCfg,
    PFQuadrupedPPORunnerCfg,
)

from ..cfg.SF import brs_base_env_cfg, limx_berkeley_env_cfg
from ..envs.him_env import HIMManagerBasedRLEnv
from . import (
    brs_solefoot_env_cfg,
    limx_pointfoot_env_cfg,
    limx_solefoot_env_cfg,
    limx_wheelfoot_env_cfg,
    quadruped_pointfoot_env_cfg,
)

##
# Create PPO runners for RSL-RL
##

limx_pf_blind_flat_runner_cfg = PF_TRON1AFlatPPORunnerCfg()

limx_wf_blind_flat_runner_cfg = WF_TRON1AFlatPPORunnerCfg()

limx_sf_blind_flat_runner_cfg = SF_TRON1AFlatPPORunnerCfg()

limx_sf_him_blind_flat_runner_cfg = SF_TRON1AFlatPPORunnerCfg()

limx_sf_berkeley_mimic_runner_cfg = SF_Berkeley_PPORunnerCfg()

limx_sf_copt_runner_cfg = SFCoptPPORunnerCfg()

limx_sf_copt_learned_runner_cfg = SFCoptLearnedModelPPORunnerCfg()

quadruped_runner_cfg = PFQuadrupedPPORunnerCfg()

quadruped_him_runner_cfg = PFQuadrupedPPORunnerCfg()

quadruped_copt_runner_cfg = PFQuadrupedCoptPPORunnerCfg()

quadruped_copt_learned_runner_cfg = PFQuadrupedCoptLearnedModelPPORunnerCfg()

##
# Register Gym environments
##

############################
# PF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-PF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

#############################
# WF Blind Flat Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)


############################
# SF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-SF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

############################
# SF Blind Rough Environment
############################
gym.register(
    id="Isaac-Limx-SF-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

#############################
# SF HIM Environment
#############################
gym.register(
    id="Isaac-Limx-SF-HIM-Blind-Flat-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_him_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Blind-Flat-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_him_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Blind-Rough-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_him_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Blind-Rough-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_him_blind_flat_runner_cfg,
    },
)

#############################
# SF Berkeley Mimic Environment
#############################
gym.register(
    id="Isaac-Limx-SF-Berkeley-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBerkeleyRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_berkeley_mimic_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Berkeley-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBerkeleyRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_berkeley_mimic_runner_cfg,
    },
)

#############################
# SF Identified Actuator Environments (Blind Rough)
#############################
gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Identified-Blind-Flat-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMIdentifiedBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Identified-Blind-Flat-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMIdentifiedBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Flat-Urdf-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBlindFlatEnvUrdfCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Flat-Play-Urdf-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBlindFlatEnvUrdfCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Identified-Blind-Rough-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMIdentifiedBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Identified-Blind-Rough-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMIdentifiedBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Rough-Urdf-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBlindRoughEnvUrdfCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Blind-Rough-Play-Urdf-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBlindRoughEnvUrdfCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Identified-Blind-Rough-Urdf-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMIdentifiedBlindRoughEnvUrdfCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-HIM-Identified-Blind-Rough-Play-Urdf-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFHIMIdentifiedBlindRoughEnvUrdfCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Berkeley-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBerkeleyRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_berkeley_mimic_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Identified-Berkeley-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFIdentifiedBerkeleyRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_berkeley_mimic_runner_cfg,
    },
)

#############################
# SF Co-Optimisation Environment
#############################
gym.register(
    id="Isaac-Limx-SF-Copt-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_copt_runner_cfg,
    },
)

#############################
# SF Co-Optimisation Learned-Model Environment
#############################
gym.register(
    id="Isaac-Limx-SF-Copt-Learned-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_learned_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Learned-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_copt_learned_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Learned-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_copt_learned_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Copt-Learned-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFCoptBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_copt_learned_runner_cfg,
    },
)

#############################
# SD_BRS1 Environments
#############################

limx_sd_brs1_runner_cfg = SD_BRS1FlatPPORunnerCfg()

gym.register(
    id="Isaac-Limx-SDBRS1-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1BlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SDBRS1-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1BlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SDBRS1-Blind-Flat2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1BlindFlatEnv2Cfg,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SDBRS1-Blind-Flat2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1BlindFlatEnv2Cfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)


gym.register(
    id="Isaac-Limx-SDBRS1-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1BlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SDBRS1-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1BlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SDBRS1-HIM-Blind-Flat-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1HIMBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SDBRS1-HIM-Blind-Flat-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1HIMBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SDBRS1-HIM-Blind-Rough-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1HIMBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SDBRS1-HIM-Blind-Rough-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": brs_solefoot_env_cfg.SDBRS1HIMBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sd_brs1_runner_cfg,
    },
)

##################################
# Quadruped Blind Flat Environment
##################################
gym.register(
    id="Isaac-Quadruped-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

###################################
# Quadruped Blind Rough Environment
###################################
gym.register(
    id="Isaac-Quadruped-Blind-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Blind-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_runner_cfg,
    },
)

######################################
# Quadruped HIM Blind Flat Environment
######################################
gym.register(
    id="Isaac-Quadruped-HIM-Blind-Flat-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFHIMBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_him_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-HIM-Blind-Flat-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFHIMBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_him_runner_cfg,
    },
)

#######################################
# Quadruped HIM Blind Rough Environment
#######################################
gym.register(
    id="Isaac-Quadruped-HIM-Blind-Rough-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFHIMBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_him_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-HIM-Blind-Rough-Play-v0",
    entry_point=HIMManagerBasedRLEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFHIMBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_him_runner_cfg,
    },
)

#################################
# Quadruped Copt Flat Environment
#################################
gym.register(
    id="Isaac-Quadruped-Copt-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)

##################################
# Quadruped Copt Rough Environment
##################################
gym.register(
    id="Isaac-Quadruped-Copt-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)

gym.register(
    id="Isaac-Quadruped-Copt-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": quadruped_copt_runner_cfg,
    },
)

################################################
# Quadruped Copt Learned Model Rough Environment
################################################
gym.register(
    id="Isaac-Quadruped-Copt-Learned-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": quadruped_pointfoot_env_cfg.QuadrupedPFCoptBlindRoughEnvCfg,
        "rsl_rl_cfg_entry_point": quadruped_copt_learned_runner_cfg,
    },
)
