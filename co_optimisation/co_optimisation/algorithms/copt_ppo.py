"""CoptPPO: PPO variant for the co-optimisation runner.

Extends :class:`rsl_rl.algorithms.PPO` with two additions.

1. Per-individual advantage normalisation.  The population assigns environments to
   individuals round-robin, so each individual owns a contiguous block of
   environments.  Normalising advantages within each individual's block removes the
   mixture bias introduced by pooling designs with different return scales into one
   global mean and standard deviation.

2. Explained variance logging.  The scalar is computed from the pre-update storage
   tensors (old values against bootstrapped returns) and added to the ``loss_dict``
   returned by :meth:`update`, exposing a calibrated critic-fit diagnostic under the
   key ``"explained_variance"``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO


class CoptPPO(PPO):
    """PPO with per-individual advantage normalisation and explained variance metric.

    Args:
        policy: The actor-critic module, forwarded unchanged to :class:`PPO`.
        num_individuals: Number of distinct designs in the population.
        env_to_individual: Round-robin mapping from environment index to individual
            index, of length ``num_envs``.  Produced by
            :meth:`CoptOnPolicyRunner._assign_individuals_to_envs`.
        **kwargs: Remaining keyword arguments forwarded to :class:`PPO`.
    """

    def __init__(
        self,
        policy,
        num_individuals: int,
        env_to_individual: list[int],
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            policy,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device,
            normalize_advantage_per_mini_batch,
            rnd_cfg,
            symmetry_cfg,
            multi_gpu_cfg,
        )
        self.num_individuals = num_individuals
        num_envs = len(env_to_individual)
        self._ind_env_mask: list[torch.Tensor] = [
            torch.tensor(
                [env_to_individual[j] == ind for j in range(num_envs)],
                dtype=torch.bool,
                device=self.device,
            )
            for ind in range(num_individuals)
        ]

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE returns and apply per-individual advantage normalisation.

        Global normalisation (inside :meth:`RolloutStorage.compute_returns`) is
        disabled; the per-individual normalisation replaces it so that each design's
        advantages are scaled relative only to that design's own return distribution.
        """
        last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=False
        )
        adv = self.storage.advantages  # [T, N, 1]
        for ind in range(self.num_individuals):
            mask = self._ind_env_mask[ind]  # bool [N]
            block = adv[:, mask, :]  # [T, n_envs_for_ind, 1]
            adv[:, mask, :] = (block - block.mean()) / (block.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run the PPO update and append an explained variance diagnostic.

        The explained variance is computed from the pre-update storage tensors so
        that it reflects the quality of the old value function against the
        bootstrapped returns, before any gradient step modifies the critic.
        """
        with torch.no_grad():
            flat_returns = self.storage.returns.flatten(0, 1)
            flat_values = self.storage.values.flatten(0, 1)
            var_y = flat_returns.var()
            explained_var = (
                1.0 - (flat_returns - flat_values).var() / (var_y + 1e-8)
            ).item()
        loss_dict = super().update()
        loss_dict["explained_variance"] = explained_var
        return loss_dict
