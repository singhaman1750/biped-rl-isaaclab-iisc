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

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn, Tuple

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP

from co_optimisation.modules.copt_estimator import CoptEstimator


class CoptActorCritic(ActorCritic):
    is_recurrent = False

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
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ):
        num_actor_obs = 0
        enc_size = obs["predictedPrivilegedObs"].shape[-1]
        enc_hidden_dims = encoder_cfg.get("hidden_dims", [128, 64, 16])
        for obs_group in obs_groups["policy"]:
            assert (
                len(obs[obs_group].shape) == 2
            ), "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_actor_obs += enc_size
        print(f"number of actor obs: {num_actor_obs}")

        super(CoptActorCritic, self).__init__(
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

        # Actor
        if self.state_dependent_std:
            self.actor = MLP(
                num_actor_obs, [2, num_actions], actor_hidden_dims, activation
            )
        else:
            self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)

        # Estimator
        num_one_step_obs = obs["obsHistory"].shape[-1]
        self.history_size = obs["obsHistory"].shape[1]
        print("privileged observation shape: ", obs["morphologyObs"].shape)
        print("one step observation size: ", num_one_step_obs)
        num_one_step_privileged_obs = obs["morphologyObs"].shape[-1]
        enc_size = obs["predictedPrivilegedObs"].shape[-1]
        enc_hidden_dims = enc_hidden_dims[:-1]
        self.estimator = MLP(
            num_one_step_privileged_obs + self.history_size * num_one_step_obs,
            enc_size,
            enc_hidden_dims,
            activation,
        )

        # Action noise
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_we/ights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(
                mod for mod in sequential if isinstance(mod, nn.Linear)
            )
        ]

    def _get_estimator_input(self, obs: TensorDict) -> torch.Tensor:
        history_obs = torch.flatten(
            self.actor_obs_normalizer(obs["obsHistory"]), start_dim=1
        )
        return torch.cat((obs["morphologyObs"], history_obs), dim=-1)

    def _update_distribution(self, obs: TensorDict):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        pred = self.estimator(self._get_estimator_input(obs))
        actor_input = torch.cat((actor_obs, pred), dim=-1)
        mean = self.actor(actor_input)
        if torch.any(torch.isnan(self.log_std)):
            print("received nan values in the log std")
        std = torch.exp(self.log_std)
        self.distribution = Normal(mean, std)
        return pred

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        pred = self._update_distribution(obs)
        return self.distribution.sample(), pred

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        privileged_obs = self._get_estimator_input(obs)
        latent = self.estimator(privileged_obs)
        actions_mean = self.actor(torch.cat((actor_obs, latent), dim=-1))
        return actions_mean


class CoptLearnedModelActorCritic(CoptActorCritic):
    """Co-optimisation actor-critic with a learned-model estimator.

    Replaces the parent's plain MLP estimator with a :class:`CoptEstimator`
    whose encoder consumes the privileged morphology observations together
    with the flattened actor observation history, and whose decoder predicts
    the robot dynamic state.  ``act`` additionally returns the prediction so
    that :class:`CoptLearnedModelPPO` can compute the model estimation loss.

    Note, ``actor_obs_normalization`` must remain False for this class, the
    empirical normalizer is sized for the flat actor input and cannot consume
    the 3D observation history.
    """

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        encoder_cfg: dict,
        decoder_cfg: dict,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        **kwargs: dict[str, Any],
    ):
        super(CoptLearnedModelActorCritic, self).__init__(
            obs,
            obs_groups,
            num_actions,
            encoder_cfg,
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

        # Estimator, replaces the parent's plain MLP estimator.
        enc_hidden_dims = encoder_cfg.get("hidden_dims", [128, 64, 16])
        dec_hidden_dims = decoder_cfg.get("hidden_dims", [64, 128])
        assert len(obs["obsHistory"].shape) == 3
        self.history_size = obs["obsHistory"].shape[1]
        num_one_step_obs = obs["obsHistory"].shape[-1]
        num_privileged_obs = obs["morphologyObs"].shape[-1]
        num_predicted_privileged_obs = obs["predictedPrivilegedObs"].shape[-1]
        print("observation history shape: ", obs["obsHistory"].shape)
        print("one step observation size: ", num_one_step_obs)
        self.estimator = CoptEstimator(
            temporal_steps=self.history_size,
            num_one_step_obs=num_one_step_obs,
            num_privileged_obs=num_privileged_obs,
            num_predicted_privileged_obs=num_predicted_privileged_obs,
            enc_hidden_dims=enc_hidden_dims,
            dec_hidden_dims=dec_hidden_dims,
            activation=encoder_cfg.get("activation", "elu"),
        )

    def _get_estimator_input(self, obs: TensorDict) -> torch.Tensor:
        history_obs = torch.flatten(
            self.actor_obs_normalizer(obs["obsHistory"]), start_dim=1
        )
        return torch.cat((obs["morphologyObs"], history_obs), dim=-1)

    def _update_distribution(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        latent, pred_info = self.estimator(self._get_estimator_input(obs))
        actor_input = torch.cat((actor_obs, latent), dim=-1)
        mean = self.actor(actor_input)
        if torch.any(torch.isnan(self.log_std)):
            print("received nan values in the log std")
        std = torch.exp(self.log_std)
        self.distribution = Normal(mean, std)
        return pred_info

    def act(
        self, obs: TensorDict, **kwargs: dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_info = self._update_distribution(obs)
        return self.distribution.sample(), pred_info

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        latent = self.estimator.encode(self._get_estimator_input(obs))
        return self.actor(torch.cat((actor_obs, latent), dim=-1))


class CoptLearnedModelV2ActorCritic(CoptLearnedModelActorCritic):
    def _get_estimator_input(self, obs: TensorDict) -> torch.Tensor:
        history_obs = torch.flatten(
            self.actor_obs_normalizer(obs["obsHistory"]), start_dim=1
        )
        return history_obs
