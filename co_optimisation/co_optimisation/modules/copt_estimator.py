import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "silu":
        return nn.SiLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None


class CoptEstimator(nn.Module):
    """Encoder-decoder estimator for the learned-model co-optimisation policy.

    The encoder consumes ``temporal_steps`` steps of actor observations
    flattened together with the morphology and terrain privileged
    observations, and emits a latent of width ``enc_hidden_dims[-1]``.  The
    decoder regresses the robot dynamic state (``predictedPrivilegedObs``)
    from that latent.  Unlike ``HIMEstimator`` this module owns no optimiser,
    the model estimation loss returned by :meth:`update` is folded into the
    PPO loss and minimised by the single shared optimiser, so the encoder
    receives gradients from both the PPO objective and the decoder objective.
    """

    def __init__(
        self,
        temporal_steps,
        num_one_step_obs,
        num_privileged_obs,
        num_predicted_privileged_obs,
        enc_hidden_dims=[128, 64, 16],
        dec_hidden_dims=[64, 128],
        activation="elu",
        learning_rate=1e-3,
        max_grad_norm=10.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "CoptEstimator.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super(CoptEstimator, self).__init__()
        activation = get_activation(activation)

        self.temporal_steps = temporal_steps
        self.num_one_step_obs = num_one_step_obs
        self.num_latent = enc_hidden_dims[-1]
        self.max_grad_norm = max_grad_norm
        self.num_predicted_privileged_obs = num_predicted_privileged_obs
        self.num_privileged_obs = num_privileged_obs

        # Encoder
        enc_input_dim = self.temporal_steps * self.num_one_step_obs + self.num_privileged_obs
        print("encoder input dim: ", enc_input_dim)
        print("temporal_steps: ", self.temporal_steps)
        print("num one step observations: ", self.num_one_step_obs)
        print("num privileged observations: ", self.num_privileged_obs)
        enc_layers = []
        for l in range(len(enc_hidden_dims) - 1):
            enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[l]), activation]
            enc_input_dim = enc_hidden_dims[l]
        enc_layers += [nn.Linear(enc_input_dim, enc_hidden_dims[-1])]
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_input_dim = enc_hidden_dims[-1]
        dec_layers = []
        for l in range(len(dec_hidden_dims)):
            dec_layers += [nn.Linear(dec_input_dim, dec_hidden_dims[l]), activation]
            dec_input_dim = dec_hidden_dims[l]
        dec_layers += [nn.Linear(dec_input_dim, self.num_predicted_privileged_obs)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, estimator_input):
        # The outputs are part of the policy computation graph and must NOT be
        # detached, the encoder is trained by both the PPO and decoder losses.
        z = self.encoder(estimator_input)
        pred = self.decoder(z)
        z = F.normalize(z, dim=-1, p=2)
        return z, pred

    def encode(self, estimator_input):
        z = self.encoder(estimator_input.detach())
        z = F.normalize(z, dim=-1, p=2)
        return z

    def get_latent(self, estimator_input):
        return self.encode(estimator_input).detach()

    def update(self, predicted, target):
        """Return the model estimation loss ``|| P_2 - P_2_pred ||^2``.

        No optimiser step happens here, the caller adds the returned tensor to
        the PPO loss so a single optimiser updates encoder, decoder, actor,
        and critic together.
        """
        return F.mse_loss(predicted, target.detach())
