# =============================================================================
# src/networks.py
# PyTorch neural network components for the LSTM-SAC agent.
#
# Architecture:
#   LSTMFeatureExtractor  →  shared backbone
#   SACActorNetwork       →  Gaussian policy for continuous actions
#   SACCriticNetwork      →  Twin Q-networks to reduce value overestimation
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from copy import deepcopy
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Helper: build a fully-connected block
# ---------------------------------------------------------------------------

def _build_fc(input_size: int, hidden_sizes: List[int]) -> nn.Sequential:
    """
    Stack Linear → ReLU → LayerNorm blocks.
    The final hidden size becomes the output size.
    """
    layers: list = []
    prev = input_size
    for h in hidden_sizes:
        layers += [
            nn.Linear(prev, h),
            nn.ReLU(inplace=True),
            nn.LayerNorm(h),
        ]
        prev = h
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# LSTM Feature Extractor
# ---------------------------------------------------------------------------

class LSTMFeatureExtractor(nn.Module):
    """
    Encodes a (batch, seq_len, input_size) observation into a fixed-size
    feature vector by running an LSTM and taking only the last timestep output,
    followed by layer normalisation and a small MLP.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        num_layers:  int,
        dropout:     float,
        fc_sizes:    List[int],
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_layers  = _build_fc(hidden_size, fc_sizes)
        self.output_size = fc_sizes[-1]

        # Initialise LSTM weights with orthogonal initialisation
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_size)

        Returns
        -------
        features : (batch, output_size)
        """
        # Hidden state is reset to zero for every batch (no cross-episode memory)
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_out    = lstm_out[:, -1, :]    # take only the final timestep
        normed      = self.layer_norm(last_out)
        features    = self.fc_layers(normed)
        return features                     # (batch, output_size)


# ---------------------------------------------------------------------------
# SAC Actor Network
# ---------------------------------------------------------------------------

class SACActorNetwork(nn.Module):
    """
    Stochastic actor that outputs a Gaussian distribution over the action
    space, squashed through tanh and shifted to [0, 1] for position fractions.
    """

    LOG_STD_MIN = -20
    LOG_STD_MAX =   2

    def __init__(
        self,
        feature_extractor: LSTMFeatureExtractor,
        action_dim: int = 1,
    ):
        super().__init__()
        self.extractor    = feature_extractor
        self.mean_layer   = nn.Linear(feature_extractor.output_size, action_dim)
        self.log_std_layer = nn.Linear(feature_extractor.output_size, action_dim)

        # Small init for the output heads → stable early training
        nn.init.uniform_(self.mean_layer.weight,    -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias,      -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.bias,   -3e-3, 3e-3)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : (batch, seq_len, n_features)

        Returns
        -------
        mean    : (batch, action_dim)
        log_std : (batch, action_dim)  clamped to [LOG_STD_MIN, LOG_STD_MAX]
        """
        features = self.extractor(obs)
        mean     = self.mean_layer(features)
        log_std  = self.log_std_layer(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action using the reparameterisation trick, squash with tanh,
        and shift from (-1, 1) to (0, 1).

        Returns
        -------
        action   : (batch, action_dim)  in [0, 1]
        log_prob : (batch, 1)           log-probability with squashing correction
        mean_act : (batch, action_dim)  deterministic action (tanh(mean) shifted)
        """
        mean, log_std = self.forward(obs)
        std    = log_std.exp()
        dist   = Normal(mean, std)

        x_t    = dist.rsample()                     # reparameterised sample
        y_t    = torch.tanh(x_t)                    # squash to (-1, 1)
        action = (y_t + 1.0) / 2.0                 # shift to (0, 1)

        # Log-probability with tanh squashing correction
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1.0 - y_t.pow(2) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)   # (batch, 1)

        mean_act = (torch.tanh(mean) + 1.0) / 2.0   # deterministic action

        return action, log_prob, mean_act


# ---------------------------------------------------------------------------
# SAC Critic Network (Twin Q)
# ---------------------------------------------------------------------------

class SACCriticNetwork(nn.Module):
    """
    Twin Q-networks (Q1, Q2) that estimate the soft action-value function.
    Using two independent critics and taking the minimum reduces overestimation.
    """

    def __init__(
        self,
        feature_extractor: LSTMFeatureExtractor,
        action_dim: int = 1,
    ):
        super().__init__()
        # Each critic gets its OWN separate feature extractor (independent weights)
        self.extractor1 = feature_extractor
        self.extractor2 = deepcopy(feature_extractor)

        feat_dim = feature_extractor.output_size

        self.q1 = nn.Sequential(
            nn.Linear(feat_dim + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(feat_dim + action_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        # Small output init
        for net in (self.q1, self.q2):
            last_layer = [m for m in net.modules() if isinstance(m, nn.Linear)][-1]
            nn.init.uniform_(last_layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(last_layer.bias,   -3e-3, 3e-3)

    def forward(
        self,
        obs:    torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs    : (batch, seq_len, n_features)
        action : (batch, action_dim)

        Returns
        -------
        q1_value, q2_value  — each of shape (batch, 1)
        """
        feat1 = self.extractor1(obs)   # (batch, output_size)
        feat2 = self.extractor2(obs)

        x1 = torch.cat([feat1, action], dim=-1)   # (batch, output_size + action_dim)
        x2 = torch.cat([feat2, action], dim=-1)

        return self.q1(x1), self.q2(x2)           # each (batch, 1)
