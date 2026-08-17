import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import (
    PF_TRON1AFlatPPORunnerCfg,
    SF_Berkeley_PPORunnerCfg,
    SF_TRON1AFlatPPORunnerCfg,
    SFCoptLearnedModelPPORunnerCfg,
    SFCoptPPORunnerCfg,
    WF_TRON1AFlatPPORunnerCfg,
)

from ..cfg.SF import limx_berkeley_env_cfg
from ..envs.him_env import HIMManagerBasedRLEnv
from . import limx_pointfoot_env_cfg, limx_solefoot_env_cfg, limx_wheelfoot_env_cfg

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
