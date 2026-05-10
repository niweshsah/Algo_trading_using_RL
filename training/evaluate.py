# =============================================================================
# training/evaluate.py
# Deterministic evaluation, backtesting, and result visualisation.
# =============================================================================

from __future__ import annotations

import os
import sys
import logging
from typing import Tuple, Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward      import RewardCalculator
from src.environment import TradingEnvironment
from src.agent       import SACAgent

logger = logging.getLogger(__name__)

# Common plot style
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    agent:         SACAgent,
    env:           TradingEnvironment,
    n_episodes:    int  = 1,
    deterministic: bool = True,
) -> Tuple[Dict[str, float], Dict[str, list]]:
    """
    Run the agent deterministically on an environment and collect trajectories.

    Parameters
    ----------
    agent         : Trained SACAgent.
    env           : TradingEnvironment (val or test split).
    n_episodes    : Number of evaluation episodes to average over.
    deterministic : Use tanh(mean) action (no exploration noise).

    Returns
    -------
    metrics    : Dictionary of averaged performance metrics.
    trajectory : Dictionary of lists (portfolio_values, actions, positions,
                 prices, dates) from the last episode.
    """
    all_metrics: List[Dict] = []
    trajectory: Dict[str, list] = {}

    for ep in range(n_episodes):
        obs = env.reset()
        done, truncated = False, False

        ep_portfolio_values: List[float] = [env.config.INITIAL_CAPITAL]
        ep_actions:    List[float] = []
        ep_positions:  List[float] = []
        ep_prices:     List[float] = []
        ep_dates:      List       = []

        while not (done or truncated):
            action = agent.select_action(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)

            ep_portfolio_values.append(info["portfolio_value"])
            ep_actions.append(float(info["action_taken"]))
            ep_positions.append(float(info["position_fraction"]))
            ep_prices.append(float(info["price"]))
            ep_dates.append(info["step"])

        metrics = RewardCalculator.compute_episode_metrics(ep_portfolio_values)
        all_metrics.append(metrics)

        # Keep trajectory from the last episode
        trajectory = {
            "portfolio_values": ep_portfolio_values,
            "actions":          ep_actions,
            "positions":        ep_positions,
            "prices":           ep_prices,
            "dates":            ep_dates,
        }

    # Average metrics across episodes
    avg_metrics: Dict[str, float] = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        avg_metrics[key] = float(np.mean(vals))

    return avg_metrics, trajectory


# ---------------------------------------------------------------------------
# Backtest + plotting
# ---------------------------------------------------------------------------

def backtest_and_plot(
    agent:    SACAgent,
    test_env: TradingEnvironment,
    save_dir: str,
) -> Dict[str, float]:
    """
    Run a full deterministic backtest on the test environment, generate 5
    analysis plots, and print a formatted summary table.

    Parameters
    ----------
    agent    : Trained (best checkpoint) SACAgent.
    test_env : Test-split TradingEnvironment.
    save_dir : Directory in which to save PNG plots.

    Returns
    -------
    Dictionary of backtest performance metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    metrics, trajectory = evaluate(agent, test_env, n_episodes=1, deterministic=True)

    pv      = np.array(trajectory["portfolio_values"], dtype=np.float64)
    actions = np.array(trajectory["actions"])
    pos     = np.array(trajectory["positions"])
    prices  = np.array(trajectory["prices"])
    steps   = np.arange(len(pv))

    # ---- Buy-and-Hold baseline -----------------------------------------
    n_price_steps = len(prices)
    bh_scale      = pv[0] / prices[0] if prices[0] > 0 else 1.0
    bh_values     = prices * bh_scale              # same starting capital
    bh_pv_full    = np.concatenate([[pv[0]], bh_values])[:len(pv)]

    bh_metrics = RewardCalculator.compute_episode_metrics(list(bh_pv_full))

    # ====================================================================
    # Plot 1 — Portfolio value vs Buy-and-Hold with drawdown shading
    # ====================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax1, ax2 = axes

    ax1.plot(steps, pv,         label="SAC Agent",      color="#2196F3", lw=2)
    ax1.plot(steps[:len(bh_pv_full)], bh_pv_full,
             label="Buy & Hold", color="#FF9800", lw=2, ls="--")

    # Shade drawdown periods (where agent PV < running peak)
    peak   = np.maximum.accumulate(pv)
    in_dd  = pv < peak
    _shade_regions(ax1, steps, in_dd, color="red", alpha=0.10, label="Drawdown period")

    ax1.set_title("Portfolio Value — Agent vs Buy & Hold", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Drawdown curve in lower panel
    drawdown_curve = (peak - pv) / (peak + 1e-8) * 100.0
    ax2.fill_between(steps, -drawdown_curve, 0, color="red", alpha=0.4)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Step")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "plot1_portfolio_value.png"))

    # ====================================================================
    # Plot 2 — Position sizing over time with price overlay
    # ====================================================================
    fig, ax_pos = plt.subplots(figsize=(14, 5))
    ax_px = ax_pos.twinx()

    pos_steps = np.arange(len(pos))
    ax_pos.fill_between(pos_steps, pos, alpha=0.4, color="#4CAF50", label="Position fraction")
    ax_pos.plot(pos_steps, pos, color="#4CAF50", lw=1.5)
    ax_pos.set_ylabel("Position Fraction [0 – 1]", color="#4CAF50")
    ax_pos.set_ylim(0, 1.1)
    ax_pos.tick_params(axis="y", labelcolor="#4CAF50")

    ax_px.plot(pos_steps, prices[:len(pos)], color="#9C27B0", lw=1.5,
               alpha=0.7, label="Close Price")
    ax_px.set_ylabel("Close Price ($)", color="#9C27B0")
    ax_px.tick_params(axis="y", labelcolor="#9C27B0")

    ax_pos.set_title("Position Sizing and Price", fontsize=14, fontweight="bold")
    ax_pos.set_xlabel("Step")

    lines1, labels1 = ax_pos.get_legend_handles_labels()
    lines2, labels2 = ax_px.get_legend_handles_labels()
    ax_pos.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "plot2_position_sizing.png"))

    # ====================================================================
    # Plot 3 — Return distribution: Agent vs Buy-and-Hold
    # ====================================================================
    agent_rets = np.diff(pv) / (pv[:-1] + 1e-8) * 100.0
    bh_rets    = np.diff(bh_pv_full) / (bh_pv_full[:-1] + 1e-8) * 100.0

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(
        min(agent_rets.min(), bh_rets.min()),
        max(agent_rets.max(), bh_rets.max()),
        50,
    )
    ax.hist(agent_rets, bins=bins, alpha=0.6, color="#2196F3", label="SAC Agent", density=True)
    ax.hist(bh_rets,    bins=bins, alpha=0.6, color="#FF9800", label="Buy & Hold", density=True)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Density")
    ax.set_title("Daily Return Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "plot3_return_distribution.png"))

    # ====================================================================
    # Plot 4 — Rolling 30-day Sharpe ratio
    # ====================================================================
    log_rets   = np.diff(np.log(pv + 1e-8))
    window     = 30
    roll_sharpe = _rolling_sharpe(log_rets, window)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.arange(len(roll_sharpe)), roll_sharpe, color="#E91E63", lw=2)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.fill_between(np.arange(len(roll_sharpe)), roll_sharpe, 0,
                    where=roll_sharpe >= 0, color="#4CAF50", alpha=0.25, label="Positive Sharpe")
    ax.fill_between(np.arange(len(roll_sharpe)), roll_sharpe, 0,
                    where=roll_sharpe < 0,  color="#F44336", alpha=0.25, label="Negative Sharpe")
    ax.set_xlabel("Step")
    ax.set_ylabel("Rolling Sharpe (30-day)")
    ax.set_title("Rolling 30-Day Sharpe Ratio", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "plot4_rolling_sharpe.png"))

    # ====================================================================
    # Plot 5 — Trade scatter: entry vs exit price
    # ====================================================================
    trades = _extract_trades(pos, prices)

    fig, ax = plt.subplots(figsize=(10, 6))
    if trades:
        entry_prices = [t["entry_price"] for t in trades]
        exit_prices  = [t["exit_price"]  for t in trades]
        sizes        = [t["position"]    for t in trades]
        colours      = ["#4CAF50" if t["pnl"] >= 0 else "#F44336" for t in trades]
        marker_sizes  = [max(20, s * 300) for s in sizes]

        sc = ax.scatter(entry_prices, exit_prices,
                        c=colours, s=marker_sizes, alpha=0.7, edgecolors="white", lw=0.5)

        # Diagonal (break-even line)
        lo = min(min(entry_prices), min(exit_prices))
        hi = max(max(entry_prices), max(exit_prices))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Break-even")

        green_patch = mpatches.Patch(color="#4CAF50", label="Profit")
        red_patch   = mpatches.Patch(color="#F44336", label="Loss")
        ax.legend(handles=[green_patch, red_patch, plt.Line2D([0],[0],color="k",ls="--",label="Break-even")])
    else:
        ax.text(0.5, 0.5, "No completed trades", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")

    ax.set_xlabel("Entry Price ($)")
    ax.set_ylabel("Exit Price ($)")
    ax.set_title("Trade Analysis: Entry vs Exit (size = position fraction)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(save_dir, "plot5_trade_analysis.png"))

    # ====================================================================
    # Print summary table
    # ====================================================================
    _print_summary(metrics, bh_metrics)

    return metrics


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved → %s", path)


def _shade_regions(ax, x, mask, **kwargs) -> None:
    """Shade contiguous True regions of `mask` on `ax`."""
    label = kwargs.pop("label", None)
    first = True
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            ax.axvspan(x[start], x[i - 1], label=label if first else None, **kwargs)
            first = False
            start = None
    if start is not None:
        ax.axvspan(x[start], x[-1], label=label if first else None, **kwargs)


def _rolling_sharpe(returns: np.ndarray, window: int) -> np.ndarray:
    """Compute annualised rolling Sharpe ratio."""
    n = len(returns)
    sharpe = np.zeros(n)
    for i in range(window, n):
        chunk = returns[i - window: i]
        if np.std(chunk) > 1e-8:
            sharpe[i] = np.mean(chunk) / np.std(chunk) * np.sqrt(252)
    return sharpe


def _extract_trades(positions: np.ndarray, prices: np.ndarray) -> list:
    """
    Detect round-trip trades from position and price arrays.
    A trade starts when position moves from 0 to >0 and ends when it drops back to 0.
    """
    trades = []
    in_pos    = False
    entry_px  = 0.0
    entry_pos = 0.0

    for i, (p, px) in enumerate(zip(positions, prices)):
        if not in_pos and p > 0.01:
            in_pos    = True
            entry_px  = px
            entry_pos = p
        elif in_pos and p < 0.01:
            in_pos = False
            pnl = (px - entry_px) * entry_pos
            trades.append({
                "entry_price": entry_px,
                "exit_price":  px,
                "position":    entry_pos,
                "pnl":         pnl,
            })

    return trades


def _print_summary(agent_m: dict, bh_m: dict) -> None:
    metrics_order = [
        ("total_return_pct",         "Total Return (%)"),
        ("annualised_return_pct",    "Annual Return (%)"),
        ("annualised_volatility_pct","Annual Volatility (%)"),
        ("sharpe_ratio",             "Sharpe Ratio"),
        ("sortino_ratio",            "Sortino Ratio"),
        ("calmar_ratio",             "Calmar Ratio"),
        ("max_drawdown_pct",         "Max Drawdown (%)"),
        ("win_rate_pct",             "Win Rate (%)"),
        ("profit_factor",            "Profit Factor"),
        ("num_trades",               "Num Trades"),
    ]
    w = 55
    print("\n" + "=" * w)
    print(f"{'BACKTEST RESULTS SUMMARY':^{w}}")
    print("=" * w)
    print(f"  {'Metric':<28} {'Agent':>10}  {'Buy&Hold':>10}")
    print("-" * w)
    for key, label in metrics_order:
        a = agent_m.get(key, 0.0)
        b = bh_m.get(key, 0.0)
        if key == "num_trades":
            bh_str = "N/A"
            print(f"  {label:<28} {int(a):>10d}  {bh_str:>10}")
        elif key in ("win_rate_pct", "profit_factor"):
            bh_str = "N/A"
            print(f"  {label:<28} {a:>10.3f}  {bh_str:>10}")
        else:
            print(f"  {label:<28} {a:>10.2f}  {b:>10.2f}")
    print("=" * w + "\n")
