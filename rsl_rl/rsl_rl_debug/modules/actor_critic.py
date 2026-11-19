import torch
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import ActorCritic


class DebugActorCritic(ActorCritic):
    is_recurrent = False

    @staticmethod
    def _report_nan(tensors: TensorDict | dict[str, torch.Tensor]) -> None:
        for key, item in tensors.items():
            if item.isnan().any():
                nan_indices = torch.nonzero(item.isnan(), as_tuple=False)[
                    :, -1
                ].unique()
                print(f"NaN in {key} at indices:\n{nan_indices}")

    def _network_weights(
        self, network_names: tuple[str, ...]
    ) -> dict[str, torch.Tensor]:
        weights = {
            f"{network_name}_{weight_name}": weight
            for network_name in network_names
            for weight_name, weight in getattr(self, network_name).named_parameters()
        }
        weights.update(self.named_parameters(recurse=False))
        return weights

    def _update_distribution(self, obs: torch.Tensor) -> None:
        # The inherited act passes the concatenated and normalised actor input,
        # not the TensorDict its annotation names, so the per group check is
        # made in act. Normal validation is disabled by ActorCritic, so NaN
        # passes through the distribution and is reported here.
        super()._update_distribution(obs)
        self._report_nan(
            {
                "normalised actor input": obs,
                "mean": self.distribution.mean,
                "std": self.distribution.stddev,
            }
        )

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._report_nan(self._network_weights(("actor",)))
        self._report_nan(obs)
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        self._update_distribution(obs)
        return self.distribution.sample()

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._report_nan(self._network_weights(("critic",)))
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        value = self.critic(obs)
        self._report_nan({"normalised critic input": obs, "value": value})
        return value
