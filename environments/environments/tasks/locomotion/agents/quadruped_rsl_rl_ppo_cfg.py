from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg

from environments.utils.wrappers.rsl_rl.rl_mlp_cfg import (
    DecoderCfg,
    EncoderCfg,
    RslRlPpoAlgorithmMlpCfg,
)


# Mirrors SF_TRON1AFlatPPORunnerCfg exactly, differing only in experiment_name. The
# policy and algorithm hyperparameters are deliberately identical to the biped's so
# that a difference between the two robots' learning curves is attributable to the
# robot and the reward set rather than to the optimiser.
@configclass
class PFQuadrupedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 25
    max_iterations = 30000
    save_interval = 500
    experiment_name = "quadruped_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=19,
        hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        orthogonal_init=False,
    )


# The co-optimisation runner reads obs_groups to decide which observation groups feed
# the actor and which the critic, at copt_on_policy_runner.py:483.
@configclass
class PFQuadrupedCoptPPORunnerCfg(PFQuadrupedPPORunnerCfg):
    experiment_name: str = "quadruped_copt"
    max_iterations: int = 45000
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy", "morphologyObs"],
        "critic": ["critic"],
    }


@configclass
class PFQuadrupedCoptLearnedModelPPORunnerCfg(PFQuadrupedCoptPPORunnerCfg):
    experiment_name: str = "quadruped_copt_learned"
    obs_groups: dict[str, list[str]] = {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    decoder = DecoderCfg(
        output_detach=False,
        num_output_dim=3,
        hidden_dims=[128, 256, 512],
        activation="elu",
        orthogonal_init=False,
    )
