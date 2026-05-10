# =============================================================================
# src/agent.py
# Soft Actor-Critic (SAC) agent with experience replay.
# Implements the full SAC update loop from scratch:
#   - Twin Q-network critic update
#   - Actor update via reparameterised policy gradient
#   - Automatic entropy temperature tuning
#   - Polyak soft-update of target critic
# =============================================================================

from __future__ import annotations

import os
import logging
from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.networks import LSTMFeatureExtractor, SACActorNetwork, SACCriticNetwork

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer backed by pre-allocated numpy arrays.
    Observations are stored as float32 for memory efficiency.
    """

    def __init__(
        self,
        capacity:   int,
        obs_shape:  Tuple,
        action_dim: int,
        device:     torch.device,
    ):
        self.capacity   = capacity
        self.device     = device
        self.ptr        = 0     # write pointer
        self.size       = 0     # current number of valid entries

        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self.dones    = np.zeros((capacity, 1),          dtype=np.float32)

    def add(
        self,
        obs:      np.ndarray,
        next_obs: np.ndarray,
        action:   np.ndarray,
        reward:   float,
        done:     bool,
    ) -> None:
        """Store a single transition in the buffer (overwrites oldest if full)."""
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = float(reward)
        self.dones[self.ptr]    = float(done)

        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Uniformly sample a mini-batch of transitions.

        Returns
        -------
        Dictionary of tensors on self.device with keys:
          obs, next_obs, actions, rewards, dones.
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            "obs":      torch.FloatTensor(self.obs[idxs]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[idxs]).to(self.device),
            "actions":  torch.FloatTensor(self.actions[idxs]).to(self.device),
            "rewards":  torch.FloatTensor(self.rewards[idxs]).to(self.device),
            "dones":    torch.FloatTensor(self.dones[idxs]).to(self.device),
        }

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """
    Soft Actor-Critic agent for continuous position-sizing.

    Components
    ----------
    - Actor  : LSTM → Gaussian policy → action ∈ [0, 1]
    - Critic : Twin Q-networks with shared LSTM backbone
    - Target : Polyak-averaged copy of the critic (frozen parameters)
    - Buffer : Circular experience replay buffer
    - Alpha  : Learnable entropy temperature (automatic tuning)
    """

    def __init__(
        self,
        obs_shape:  Tuple,
        action_dim: int,
        config,
        device:     str | torch.device = "cuda",
    ):
        if isinstance(device, str):
            device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.device     = device
        self.config     = config
        self.action_dim = action_dim

        logger.info("Initialising SACAgent on device: %s", self.device)

        # ---- Infer dimensions from obs_shape (window, features) ---------
        seq_len, n_features = obs_shape[0], obs_shape[1]

        # ---- Build actor ------------------------------------------------
        actor_extractor = LSTMFeatureExtractor(
            input_size  = n_features,
            hidden_size = config.LSTM_HIDDEN_SIZE,
            num_layers  = config.LSTM_NUM_LAYERS,
            dropout     = config.LSTM_DROPOUT,
            fc_sizes    = config.FC_HIDDEN_SIZES,
        )
        self.actor = SACActorNetwork(actor_extractor, action_dim).to(device)

        # ---- Build critic (twin Q) with its own extractor ---------------
        critic_extractor = LSTMFeatureExtractor(
            input_size  = n_features,
            hidden_size = config.LSTM_HIDDEN_SIZE,
            num_layers  = config.LSTM_NUM_LAYERS,
            dropout     = config.LSTM_DROPOUT,
            fc_sizes    = config.FC_HIDDEN_SIZES,
        )
        self.critic = SACCriticNetwork(critic_extractor, action_dim).to(device)

        # ---- Target critic (frozen copy of critic) ----------------------
        self.critic_target = deepcopy(self.critic).to(device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # ---- Optimisers -------------------------------------------------
        self.actor_optimizer  = Adam(self.actor.parameters(),
                                     lr=config.LEARNING_RATE_ACTOR)
        self.critic_optimizer = Adam(self.critic.parameters(),
                                     lr=config.LEARNING_RATE_CRITIC)

        # ---- Automatic entropy tuning -----------------------------------
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=config.LEARNING_RATE_ALPHA)
        self.target_entropy  = float(config.TARGET_ENTROPY)

        # ---- Replay buffer ----------------------------------------------
        self.buffer = ReplayBuffer(
            capacity   = config.BUFFER_SIZE,
            obs_shape  = obs_shape,
            action_dim = action_dim,
            device     = device,
        )

        # Training counters
        self.total_updates = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Choose an action given a single observation.

        Parameters
        ----------
        obs           : numpy array of shape obs_shape.
        deterministic : If True, return tanh(mean) (no exploration noise).

        Returns
        -------
        action : numpy array of shape (action_dim,) in [0, 1].
        """
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, win, feat)
        self.actor.eval()
        with torch.no_grad():
            action, _, mean_act = self.actor.sample(obs_t)
            result = mean_act if deterministic else action
        self.actor.train()
        return result.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------
    # SAC update step
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, batch_size: int) -> Dict[str, float]:
        """
        Sample a mini-batch and perform one full SAC update.

        Returns
        -------
        Dictionary with scalar losses for TensorBoard logging:
          critic_loss, actor_loss, alpha_loss, alpha.
        """
        batch = self.buffer.sample(batch_size)
        obs      = batch["obs"]        # (B, win, feat)
        next_obs = batch["next_obs"]   # (B, win, feat)
        actions  = batch["actions"]    # (B, action_dim)
        rewards  = batch["rewards"]    # (B, 1)
        dones    = batch["dones"]      # (B, 1)

        # ================================================================
        # Step 1 — Update Critic
        # ================================================================
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            min_q_next       = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            target_q         = rewards + (1.0 - dones) * self.config.GAMMA * min_q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, target_q) \
                    + nn.functional.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # ================================================================
        # Step 2 — Update Actor
        # ================================================================
        new_action, log_pi, _ = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_action)
        min_q_new      = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha.detach() * log_pi - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ================================================================
        # Step 3 — Update Alpha (automatic entropy tuning)
        # ================================================================
        # Re-sample log_pi with no_grad for alpha update
        with torch.no_grad():
            _, log_pi_alpha, _ = self.actor.sample(obs)

        alpha_loss = -(self.log_alpha * (log_pi_alpha + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ================================================================
        # Step 4 — Soft update target critic (Polyak)
        # ================================================================
        tau = self.config.TAU
        with torch.no_grad():
            for param, target_param in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        self.total_updates += 1

        losses = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss":  float(actor_loss.item()),
            "alpha_loss":  float(alpha_loss.item()),
            "alpha":       float(self.alpha.item()),
        }

        # Sanity checks
        assert not np.isnan(losses["critic_loss"]), "critic_loss is NaN"
        assert not np.isnan(losses["actor_loss"]),  "actor_loss is NaN"

        return losses

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all network weights, optimiser states, and alpha."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor_state":          self.actor.state_dict(),
            "critic_state":         self.critic.state_dict(),
            "critic_target_state":  self.critic_target.state_dict(),
            "actor_optimizer":      self.actor_optimizer.state_dict(),
            "critic_optimizer":     self.critic_optimizer.state_dict(),
            "alpha_optimizer":      self.alpha_optimizer.state_dict(),
            "log_alpha":            self.log_alpha.data,
            "total_updates":        self.total_updates,
        }, path)
        logger.info("Checkpoint saved → %s", path)

    def load(self, path: str) -> None:
        """Load all weights and states from a checkpoint file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor_state"])
        self.critic.load_state_dict(checkpoint["critic_state"])
        self.critic_target.load_state_dict(checkpoint["critic_target_state"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.log_alpha.data = checkpoint["log_alpha"]
        self.total_updates  = checkpoint.get("total_updates", 0)
        logger.info("Checkpoint loaded ← %s  (updates=%d)", path, self.total_updates)
