# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP


def report_nan(tensors: TensorDict | dict[str, torch.Tensor]) -> None:
    """Print each named tensor that holds a NaN with the last axis indices it occurs at."""
    for key, item in tensors.items():
        if item.isnan().any():
            nan_indices = torch.nonzero(item.isnan(), as_tuple=False)[:, -1].unique()
            print(f"NaN in {key} at indices:\n{nan_indices}")


def network_weights(
    module: torch.nn.Module, network_names: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    """Map <network_name>_<weight_name> to each weight of the named networks of module.

    The parameters held directly by module, the action std of ActorCritic, keep their own name.
    """
    weights = {
        f"{network_name}_{weight_name}": weight
        for network_name in network_names
        for weight_name, weight in getattr(module, network_name).named_parameters()
    }
    weights.update(module.named_parameters(recurse=False))
    return weights


class CoptMorALActorCritic(ActorCritic):
    """Actor-critic whose estimator is configured entirely by observation sets.

    The estimator input is the concatenation of the groups named by
    ``obs_groups["encoderIn"]`` and its output width is that of the groups named
    by ``obs_groups["gtEncoderOut"]``, which are also the regression target
    CoptPPO reads through :meth:`get_estimator_target`. The estimator output is
    presented to the inherited machinery as one further actor observation under
    the reserved key ``encoderOut``, so that the actor width, the empirical
    normaliser and the concatenation are all handled by :class:`ActorCritic`
    without arithmetic of our own, and so that the estimator output is
    normalised together with the rest of the actor input.
    """

    is_recurrent = False
    ENCODER_OUTPUT_KEY = "encoderOut"

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        encoder_cfg: dict,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "log",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ):
        for required in ("encoderIn", "gtEncoderOut"):
            assert required in obs_groups, (
                f"CoptMorALActorCritic requires the '{required}' observation set. "
                f"Found sets: {sorted(obs_groups)}"
            )

        num_encoder_in = self._flat_width(obs, obs_groups["encoderIn"], "encoderIn")
        num_encoder_out = self._flat_width(
            obs, obs_groups["gtEncoderOut"], "gtEncoderOut"
        )

        obs_groups = {name: list(groups) for name, groups in obs_groups.items()}
        obs_groups["policy"] = obs_groups["policy"] + [self.ENCODER_OUTPUT_KEY]
        reference = obs[obs_groups["critic"][0]]
        obs = obs.copy()
        obs[self.ENCODER_OUTPUT_KEY] = reference.new_zeros(
            reference.shape[0], num_encoder_out
        )

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization,
            critic_obs_normalization,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            noise_std_type,
            state_dependent_std,
            **kwargs,
        )

        self.estimator = MLP(
            num_encoder_in,
            num_encoder_out,
            encoder_cfg.get("hidden_dims", [128, 64, 16]),
            encoder_cfg.get("activation", activation),
        )
        print(f"Estimator MLP: {self.estimator}")

    @staticmethod
    def _flat_width(obs: TensorDict, groups: list[str], set_name: str) -> int:
        width = 0
        for group in groups:
            assert len(obs[group].shape) == 2, (
                f"Observation group '{group}' in set '{set_name}' has shape "
                f"{tuple(obs[group].shape)}. Every estimator observation must be "
                "flat, so a history group must set flatten_history_dim."
            )
            width += obs[group].shape[-1]
        return width

    def _get_estimator_input(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["encoderIn"]], dim=-1)

    def get_estimator_target(self, obs: TensorDict) -> torch.Tensor:
        """The regression target CoptPPO minimises the estimator against."""
        return torch.cat(
            [obs[group] for group in self.obs_groups["gtEncoderOut"]], dim=-1
        )

    def _with_encoder_output(
        self, obs: TensorDict, encoder_output: torch.Tensor
    ) -> TensorDict:
        obs = obs.copy()
        obs[self.ENCODER_OUTPUT_KEY] = encoder_output
        return obs

    _report_nan = staticmethod(report_nan)

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

    def _actor_input(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.estimator(self._get_estimator_input(obs))
        actor_obs = self.get_actor_obs(self._with_encoder_output(obs, encoder_output))
        return self.actor_obs_normalizer(actor_obs), encoder_output

    def act(
        self, obs: TensorDict, **kwargs: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._report_nan(network_weights(self, ("actor", "estimator")))
        actor_obs, encoder_output = self._actor_input(obs)
        self._report_nan(self._with_encoder_output(obs, encoder_output))
        self._update_distribution(actor_obs)
        return self.distribution.sample(), encoder_output

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        self._report_nan(network_weights(self, ("critic",)))
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        value = self.critic(critic_obs)
        self._report_nan({"normalised critic input": critic_obs, "value": value})
        return value

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs, _ = self._actor_input(obs)
        if self.state_dependent_std:
            return self.actor(actor_obs)[..., 0, :]
        return self.actor(actor_obs)

    def update_normalization(self, obs: TensorDict) -> None:
        # The estimator output is part of the actor input and must therefore
        # contribute to the actor statistics. Without this the inherited
        # implementation raises KeyError on 'encoderOut', since PPO passes the
        # raw environment observation at ppo.py:160.
        if self.actor_obs_normalization:
            with torch.no_grad():
                obs = self._with_encoder_output(
                    obs, self.estimator(self._get_estimator_input(obs))
                )
        super().update_normalization(obs)


class CoptActorCritic(ActorCritic):
    """Actor-critic whose actor reads the true morphology directly, with no estimator.

    The morphology enters as one more group of the policy set, obs_groups["policy"]
    naming morphologyObs, so the inherited ActorCritic sizes, normalises and
    concatenates it. act returns (actions, None) because CoptPPO unpacks an
    estimator output, and adds no estimator loss for None.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        encoder_cfg: dict | None = None,
        **kwargs: dict[str, Any],
    ):
        assert "morphologyObs" in obs_groups["policy"], (
            "CoptActorCritic reads the morphology in its actor, so obs_groups['policy'] "
            f"must name 'morphologyObs'. Found {obs_groups['policy']}"
        )
        super().__init__(obs, obs_groups, num_actions, **kwargs)

    def _update_distribution(self, obs: torch.Tensor) -> None:
        super()._update_distribution(obs)
        report_nan({"normalised actor input": obs, "mean": self.distribution.mean,
                    "std": self.distribution.stddev})

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, None]:
        report_nan(network_weights(self, ("actor",)))
        report_nan({group: obs[group] for group in self.obs_groups["policy"]})
        return super().act(obs, **kwargs), None

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        report_nan(network_weights(self, ("critic",)))
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        value = self.critic(critic_obs)
        report_nan({"normalised critic input": critic_obs, "value": value})
        return value
