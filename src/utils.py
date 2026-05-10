# =============================================================================
# src/utils.py
# Utility functions: seeding, rolling statistics, annualisation,
# drawdown computation, and a trade tracker.
# =============================================================================

import os
import random
import logging
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Fix random seeds for Python built-ins, NumPy, and PyTorch (CPU + CUDA)
    to ensure reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may slow training; disable if speed matters)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set to %d", seed)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def compute_rolling_stats(
    values: List[float],
    window: int,
) -> Tuple[List[float], List[float]]:
    """
    Compute rolling mean and standard deviation over a sliding window.

    Parameters
    ----------
    values : Ordered list of scalar observations.
    window : Number of recent observations to include.

    Returns
    -------
    means : List of rolling means (same length as `values`).
    stds  : List of rolling standard deviations.
    """
    means: List[float] = []
    stds:  List[float] = []
    buf: deque = deque(maxlen=window)

    for v in values:
        buf.append(v)
        arr = np.array(buf)
        means.append(float(np.mean(arr)))
        stds.append(float(np.std(arr)) if len(arr) > 1 else 0.0)

    return means, stds


# ---------------------------------------------------------------------------
# Return annualisation
# ---------------------------------------------------------------------------

def annualise_return(total_return: float, n_days: int) -> float:
    """
    Convert a total (cumulative) return over `n_days` trading days to an
    annualised return, assuming 252 trading days per year.

    Formula: (1 + total_return)^(252 / n_days) - 1

    Parameters
    ----------
    total_return : Decimal total return (e.g. 0.25 for +25 %).
    n_days       : Number of trading days in the period.

    Returns
    -------
    Annualised return as a decimal.
    """
    if n_days <= 0:
        return 0.0
    return float((1.0 + total_return) ** (252.0 / n_days) - 1.0)


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

def compute_max_drawdown(portfolio_values: List[float]) -> float:
    """
    Compute the maximum peak-to-trough drawdown over the portfolio history.

    Parameters
    ----------
    portfolio_values : Time-ordered list of portfolio values.

    Returns
    -------
    Maximum drawdown as a positive fraction (e.g. 0.30 = 30 % drawdown).
    """
    pv   = np.array(portfolio_values, dtype=np.float64)
    peak = np.maximum.accumulate(pv)
    dd   = (peak - pv) / (peak + 1e-8)
    return float(np.max(dd))


# ---------------------------------------------------------------------------
# Trade tracker
# ---------------------------------------------------------------------------

class TradeTracker:
    """
    Records every completed trade and computes running statistics used by
    the Kelly criterion position sizer and evaluation reports.

    A "trade" is defined as: open at `entry_price`, close at `exit_price`,
    with `size` shares traded.
    """

    def __init__(self, rolling_window: int = 50):
        """
        Parameters
        ----------
        rolling_window : Number of recent trades to use for rolling metrics.
        """
        self.rolling_window = rolling_window
        self._trades: List[dict] = []          # full history
        self._rolling: deque = deque(maxlen=rolling_window)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        entry_price: float,
        exit_price:  float,
        size:        float,
        entry_step:  int,
        exit_step:   int,
    ) -> None:
        """
        Log a completed round-trip trade.

        Parameters
        ----------
        entry_price : Price at position open.
        exit_price  : Price at position close.
        size        : Number of shares (or position fraction) traded.
        entry_step  : Environment step at which the position was opened.
        exit_step   : Environment step at which the position was closed.
        """
        pnl    = (exit_price - entry_price) * size
        is_win = pnl > 0.0

        trade = {
            "entry_price": entry_price,
            "exit_price":  exit_price,
            "size":        size,
            "pnl":         pnl,
            "is_win":      is_win,
            "entry_step":  entry_step,
            "exit_step":   exit_step,
        }
        self._trades.append(trade)
        self._rolling.append(trade)

    # ------------------------------------------------------------------
    # Aggregate statistics
    # ------------------------------------------------------------------

    def win_rate(self, use_rolling: bool = True) -> Optional[float]:
        """
        Fraction of trades that were profitable.
        Returns None if no trades have been recorded.
        """
        source = list(self._rolling) if use_rolling else self._trades
        if not source:
            return None
        wins = sum(1 for t in source if t["is_win"])
        return wins / len(source)

    def win_loss_ratio(self, use_rolling: bool = True) -> Optional[float]:
        """
        Ratio of average winning P&L to average (absolute) losing P&L.
        Returns None if there are no wins or no losses.
        """
        source = list(self._rolling) if use_rolling else self._trades
        wins   = [t["pnl"] for t in source if t["is_win"]]
        losses = [abs(t["pnl"]) for t in source if not t["is_win"]]

        if not wins or not losses:
            return None

        avg_win  = float(np.mean(wins))
        avg_loss = float(np.mean(losses))
        return avg_win / (avg_loss + 1e-8)

    def profit_factor(self) -> float:
        """
        Gross profit divided by gross loss (over the full trade history).
        Returns 0 if there are no losing trades.
        """
        gross_profit = sum(t["pnl"] for t in self._trades if t["is_win"])
        gross_loss   = abs(sum(t["pnl"] for t in self._trades if not t["is_win"]))
        return float(gross_profit / (gross_loss + 1e-8))

    def average_win(self) -> float:
        """Mean P&L of winning trades."""
        wins = [t["pnl"] for t in self._trades if t["is_win"]]
        return float(np.mean(wins)) if wins else 0.0

    def average_loss(self) -> float:
        """Mean absolute P&L of losing trades (returned as a positive number)."""
        losses = [abs(t["pnl"]) for t in self._trades if not t["is_win"]]
        return float(np.mean(losses)) if losses else 0.0

    def num_trades(self) -> int:
        return len(self._trades)

    def all_trades(self) -> List[dict]:
        return list(self._trades)

    def reset(self) -> None:
        """Clear all recorded trades (e.g. between train episodes)."""
        self._trades.clear()
        self._rolling.clear()

    def summary(self) -> dict:
        return {
            "num_trades":      self.num_trades(),
            "win_rate":        self.win_rate(use_rolling=False),
            "win_loss_ratio":  self.win_loss_ratio(use_rolling=False),
            "profit_factor":   self.profit_factor(),
            "average_win":     self.average_win(),
            "average_loss":    self.average_loss(),
        }
