# =============================================================================
# src/reward.py
# Reward shaping and episode-level performance metrics.
# =============================================================================

import numpy as np
from typing import Dict, List


class RewardCalculator:
    """
    Computes per-step reward signals and end-of-episode performance metrics
    for the LSTM-SAC trading agent.
    """

    # ------------------------------------------------------------------
    # Per-step reward
    # ------------------------------------------------------------------

    @staticmethod
    def compute(
        portfolio_values: List[float],
        current_step: int,
        action: float,
        prev_action: float,
        config,
    ) -> float:
        """
        Combine three reward components into a single scalar, then apply a
        transaction-cost penalty for excessive action changes.

        Components (configurable weights in config):
          1. Rolling Sharpe ratio  — weight REWARD_SHARPE_WEIGHT
          2. Log return            — weight REWARD_RETURN_WEIGHT
          3. Drawdown penalty      — weight REWARD_DRAWDOWN_WEIGHT

        Parameters
        ----------
        portfolio_values : History of portfolio values up to and including
                           the current step.
        current_step     : Current environment step index (unused directly
                           but available for debugging).
        action           : Action taken at the current step [0, 1].
        prev_action      : Action taken at the previous step [0, 1].
        config           : Project config module.

        Returns
        -------
        Scalar reward clipped to [-10, +10].
        """
        pv = np.array(portfolio_values, dtype=np.float64)

        if len(pv) < 2:
            return 0.0

        # ---- Component 1: Rolling Sharpe --------------------------------
        lookback = min(config.REWARD_LOOKBACK, len(pv))
        window   = pv[-lookback:]
        returns  = np.diff(np.log(window + 1e-8))

        if len(returns) > 1 and np.std(returns) > 1e-8:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        sharpe_reward = sharpe * config.REWARD_SHARPE_WEIGHT

        # ---- Component 2: Log Return ------------------------------------
        log_return    = np.log(pv[-1] / (pv[-2] + 1e-8))
        return_reward = log_return * config.REWARD_RETURN_WEIGHT

        # ---- Component 3: Drawdown penalty ------------------------------
        peak          = np.max(pv)
        drawdown      = (peak - pv[-1]) / (peak + 1e-8)
        drawdown_reward = -drawdown * config.REWARD_DRAWDOWN_WEIGHT

        # ---- Transaction cost penalty -----------------------------------
        action_delta = abs(float(action) - float(prev_action))
        if action_delta > config.TRANSACTION_COST_THRESHOLD:
            cost_penalty = -action_delta * 0.01
        else:
            cost_penalty = 0.0

        reward = sharpe_reward + return_reward + drawdown_reward + cost_penalty

        # Clip to prevent exploding gradients
        return float(np.clip(reward, -10.0, 10.0))

    # ------------------------------------------------------------------
    # End-of-episode metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_episode_metrics(
        portfolio_values: List[float],
        risk_free_rate: float = 0.05,
    ) -> Dict[str, float]:
        """
        Compute a comprehensive set of risk-adjusted performance metrics
        for a completed trading episode.

        Parameters
        ----------
        portfolio_values : Ordered list of portfolio values (one per step).
        risk_free_rate   : Annualised risk-free rate (default 5 %).

        Returns
        -------
        Dictionary with keys:
          total_return_pct, annualised_return_pct, annualised_volatility_pct,
          sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown_pct,
          win_rate_pct, profit_factor, num_trades.
        """
        pv      = np.array(portfolio_values, dtype=np.float64)
        n_steps = len(pv)

        if n_steps < 2:
            return {k: 0.0 for k in [
                "total_return_pct", "annualised_return_pct",
                "annualised_volatility_pct", "sharpe_ratio", "sortino_ratio",
                "calmar_ratio", "max_drawdown_pct", "win_rate_pct",
                "profit_factor", "num_trades",
            ]}

        # Daily log returns
        daily_returns = np.diff(np.log(pv + 1e-8))

        # ---- Total and Annualised Return --------------------------------
        total_return     = (pv[-1] / pv[0]) - 1.0
        n_days           = n_steps  # treat each step as one trading day
        ann_return       = (1.0 + total_return) ** (252.0 / max(n_days, 1)) - 1.0

        # ---- Annualised Volatility --------------------------------------
        ann_vol = float(np.std(daily_returns) * np.sqrt(252))

        # ---- Sharpe Ratio -----------------------------------------------
        daily_rf   = risk_free_rate / 252.0
        excess_ret = daily_returns - daily_rf
        if np.std(excess_ret) > 1e-8:
            sharpe = float(np.mean(excess_ret) / np.std(excess_ret) * np.sqrt(252))
        else:
            sharpe = 0.0

        # ---- Sortino Ratio (downside volatility only) -------------------
        neg_rets       = excess_ret[excess_ret < 0]
        downside_std   = float(np.std(neg_rets) * np.sqrt(252)) if len(neg_rets) > 1 else 1e-8
        sortino        = float(ann_return / downside_std) if downside_std > 1e-8 else 0.0

        # ---- Max Drawdown -----------------------------------------------
        peak          = np.maximum.accumulate(pv)
        drawdowns     = (peak - pv) / (peak + 1e-8)
        max_drawdown  = float(np.max(drawdowns))

        # ---- Calmar Ratio -----------------------------------------------
        calmar = float(ann_return / max_drawdown) if max_drawdown > 1e-8 else 0.0

        # ---- Trade-level stats ------------------------------------------
        # A "trade" is detected whenever the portfolio value changes direction
        # relative to a zero-action baseline; here we use step-over-step sign changes.
        step_returns = np.diff(pv)
        wins         = step_returns[step_returns > 0]
        losses       = step_returns[step_returns < 0]
        num_trades   = len(wins) + len(losses)

        win_rate = float(len(wins) / num_trades * 100.0) if num_trades > 0 else 0.0

        sum_wins    = float(np.sum(wins))    if len(wins)   > 0 else 0.0
        sum_losses  = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-8
        profit_factor = sum_wins / sum_losses if sum_losses > 1e-8 else 0.0

        return {
            "total_return_pct":         float(total_return   * 100.0),
            "annualised_return_pct":    float(ann_return     * 100.0),
            "annualised_volatility_pct":float(ann_vol        * 100.0),
            "sharpe_ratio":             sharpe,
            "sortino_ratio":            sortino,
            "calmar_ratio":             calmar,
            "max_drawdown_pct":         float(max_drawdown   * 100.0),
            "win_rate_pct":             win_rate,
            "profit_factor":            profit_factor,
            "num_trades":               int(num_trades),
        }
