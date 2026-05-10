# =============================================================================
# training/train.py
# Main training loop for the LSTM-SAC trading agent.
# =============================================================================

from __future__ import annotations

import os
import sys
import logging
import time
from typing import Optional

import numpy as np
from tqdm import tqdm

# Allow imports from project root when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.agent       import SACAgent
from src.environment import TradingEnvironment
from src.reward      import RewardCalculator

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates the SAC training loop including:
      - Warm-up with random actions
      - Agent updates at each step
      - Periodic evaluation on validation set
      - TensorBoard logging
      - Model checkpointing
    """

    def __init__(
        self,
        agent:     SACAgent,
        train_env: TradingEnvironment,
        val_env:   TradingEnvironment,
        cfg = config,
    ):
        self.agent     = agent
        self.train_env = train_env
        self.val_env   = val_env
        self.config    = cfg

        # Create output directories
        os.makedirs(cfg.LOG_DIR,        exist_ok=True)
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cfg.PLOT_DIR,       exist_ok=True)

        # TensorBoard
        if _TB_AVAILABLE:
            self.writer = SummaryWriter(log_dir=cfg.LOG_DIR)
        else:
            self.writer = None
            logger.warning("tensorboard not available — skipping TB logging.")

        self.reward_calc  = RewardCalculator()
        self.best_sharpe  = -np.inf
        self.global_step  = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop for TOTAL_TIMESTEPS environment steps."""
        cfg = self.config

        obs = self.train_env.reset()
        episode_reward  = 0.0
        episode_values  = [cfg.INITIAL_CAPITAL]
        episode_number  = 0
        episode_start_t = time.time()

        logger.info("Starting training for %d timesteps …", cfg.TOTAL_TIMESTEPS)

        pbar = tqdm(total=cfg.TOTAL_TIMESTEPS, desc="Training", unit="step", dynamic_ncols=True)

        for step in range(cfg.TOTAL_TIMESTEPS):
            self.global_step = step

            # ---- Select action ------------------------------------------
            if step < cfg.WARMUP_STEPS:
                action = self.train_env.action_space.sample()
            else:
                action = self.agent.select_action(obs, deterministic=False)

            # ---- Environment step ---------------------------------------
            next_obs, reward, done, truncated, info = self.train_env.step(action)

            # Store transition (mark done=True only if natural episode end,
            # not if we forcibly terminated due to drawdown)
            real_done = done and not truncated
            self.agent.buffer.add(obs, next_obs, action, reward, real_done)

            obs             = next_obs
            episode_reward += float(reward)
            episode_values.append(info["portfolio_value"])

            # ---- Agent update -------------------------------------------
            if (step >= cfg.WARMUP_STEPS
                    and len(self.agent.buffer) >= cfg.BATCH_SIZE):
                for _ in range(cfg.GRADIENT_STEPS):
                    losses = self.agent.update(cfg.BATCH_SIZE)

                if step % 1000 == 0:
                    self._log_losses(losses, step)

            # ---- Episode end -------------------------------------------
            if done or truncated:
                episode_number += 1
                metrics = RewardCalculator.compute_episode_metrics(episode_values)
                elapsed = time.time() - episode_start_t

                pbar.set_postfix({
                    "ep":      episode_number,
                    "reward":  f"{episode_reward:+.2f}",
                    "sharpe":  f"{metrics['sharpe_ratio']:.3f}",
                    "pv":      f"{episode_values[-1]:,.0f}",
                })

                self._log_episode(metrics, episode_reward, episode_number, step)

                # Reset for next episode
                obs             = self.train_env.reset()
                episode_reward  = 0.0
                episode_values  = [cfg.INITIAL_CAPITAL]
                episode_start_t = time.time()

            # ---- Validation evaluation ----------------------------------
            if step % cfg.EVAL_FREQUENCY == 0 and step > 0:
                val_metrics = self._evaluate_on_val()
                self._log_val_metrics(val_metrics, step)

                # Save best model based on Sharpe ratio
                if val_metrics["sharpe_ratio"] > self.best_sharpe:
                    self.best_sharpe = val_metrics["sharpe_ratio"]
                    best_path = os.path.join(cfg.CHECKPOINT_DIR, "best_model.pt")
                    self.agent.save(best_path)
                    logger.info(
                        "Step %d | New best Sharpe: %.4f → saved to %s",
                        step, self.best_sharpe, best_path,
                    )

            # ---- Periodic checkpoint ------------------------------------
            if step % cfg.CHECKPOINT_FREQ == 0 and step > 0:
                ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"step_{step}.pt")
                self.agent.save(ckpt_path)

            pbar.update(1)

        pbar.close()
        if self.writer:
            self.writer.close()

        logger.info("Training complete. Best val Sharpe: %.4f", self.best_sharpe)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_on_val(self, n_episodes: int = 1) -> dict:
        """Run deterministic evaluation on the validation environment."""
        from training.evaluate import evaluate
        metrics, _ = evaluate(self.agent, self.val_env,
                               n_episodes=n_episodes, deterministic=True)
        return metrics

    def _log_losses(self, losses: dict, step: int) -> None:
        if self.writer:
            for k, v in losses.items():
                self.writer.add_scalar(f"train/{k}", v, step)

    def _log_episode(
        self, metrics: dict, episode_reward: float,
        episode_number: int, step: int,
    ) -> None:
        if self.writer:
            self.writer.add_scalar("episode/reward",       episode_reward,              episode_number)
            self.writer.add_scalar("episode/sharpe_ratio", metrics["sharpe_ratio"],     episode_number)
            self.writer.add_scalar("episode/total_return", metrics["total_return_pct"], episode_number)
            self.writer.add_scalar("episode/max_drawdown", metrics["max_drawdown_pct"], episode_number)
            self.writer.add_scalar("episode/step",         step,                        episode_number)

    def _log_val_metrics(self, metrics: dict, step: int) -> None:
        if self.writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"val/{k}", v, step)

        sharpe = metrics.get("sharpe_ratio", 0.0)
        ret    = metrics.get("total_return_pct", 0.0)
        dd     = metrics.get("max_drawdown_pct", 0.0)
        logger.info(
            "Step %7d | Val → Sharpe: %+.3f | Return: %+.1f %% | MaxDD: %.1f %%",
            step, sharpe, ret, dd,
        )
