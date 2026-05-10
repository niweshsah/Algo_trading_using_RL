# =============================================================================
# src/position_sizing.py
# Risk-management position sizing methods.
# Each method takes a raw RL action in [0, 1] and returns an adjusted action
# in [0, 1] after applying the relevant constraint.
# =============================================================================

import numpy as np


class PositionSizer:
    """
    Collection of static position-sizing methods.

    The environment calls `atr_vol_combined()` at every step to convert the
    actor's raw continuous action into a risk-adjusted position fraction.
    """

    # ------------------------------------------------------------------
    # Individual sizing methods
    # ------------------------------------------------------------------

    @staticmethod
    def kelly_sizing(
        raw_action: float,
        win_rate: float,
        win_loss_ratio: float,
        kelly_fraction: float = 0.5,
    ) -> float:
        """
        Half-Kelly criterion.

        Parameters
        ----------
        raw_action      : Actor output in [0, 1].
        win_rate        : Fraction of trades that were profitable.
        win_loss_ratio  : Average win P&L / average loss P&L (absolute).
        kelly_fraction  : Safety multiplier (0.5 = half-Kelly).

        Returns
        -------
        Adjusted position fraction in [0, 1].
        """
        loss_rate  = 1.0 - win_rate
        full_kelly = (win_rate * win_loss_ratio - loss_rate) / (win_loss_ratio + 1e-8)
        half_kelly = full_kelly * kelly_fraction
        half_kelly = float(np.clip(half_kelly, 0.0, 1.0))
        adjusted   = min(raw_action, half_kelly)
        return max(0.0, adjusted)

    @staticmethod
    def volatility_target_sizing(
        raw_action: float,
        asset_vol: float,
        target_vol: float = 0.15,
    ) -> float:
        """
        Scale position so that the portfolio's expected annualised volatility
        equals `target_vol`.

        Parameters
        ----------
        raw_action : Actor output in [0, 1].
        asset_vol  : Annualised historical volatility of the asset (e.g. 0.30).
        target_vol : Target portfolio volatility (default 15 % p.a.).

        Returns
        -------
        Adjusted position fraction in [0, 1].
        """
        vol_scalar = target_vol / max(asset_vol, 1e-8)
        adjusted   = raw_action * min(vol_scalar, 1.0)
        return float(np.clip(adjusted, 0.0, 1.0))

    @staticmethod
    def atr_sizing(
        raw_action: float,
        atr: float,
        current_price: float,
        portfolio_value: float,
        max_risk: float = 0.02,
        atr_mult: float = 2.0,
    ) -> float:
        """
        Limit position so that a stop-loss at `atr_mult × ATR` below entry
        would lose at most `max_risk × portfolio_value`.

        Parameters
        ----------
        raw_action      : Actor output in [0, 1].
        atr             : Average True Range of the asset at current bar.
        current_price   : Latest close price.
        portfolio_value : Total portfolio value in currency units.
        max_risk        : Maximum portfolio fraction at risk per trade.
        atr_mult        : Stop-loss distance expressed as multiples of ATR.

        Returns
        -------
        Adjusted position fraction in [0, 1].
        """
        stop_distance    = atr * atr_mult
        max_loss_dollars = portfolio_value * max_risk
        # How many shares can we buy before a stop-out costs max_loss_dollars?
        shares_limit     = max_loss_dollars / (stop_distance + 1e-8)
        dollar_limit     = shares_limit * current_price
        atr_fraction     = dollar_limit / (portfolio_value + 1e-8)
        adjusted         = min(raw_action, atr_fraction)
        return max(0.0, float(adjusted))

    @staticmethod
    def drawdown_guard(
        raw_action: float,
        current_drawdown: float,
        max_drawdown: float = 0.20,
    ) -> float:
        """
        Linearly reduce position size as drawdown approaches `max_drawdown`.
        At zero drawdown the full raw action passes through; at max_drawdown
        the position is forced to zero.

        Parameters
        ----------
        raw_action       : Actor output in [0, 1].
        current_drawdown : Current drawdown from peak as a positive fraction
                           (e.g. 0.10 = 10 % below peak).
        max_drawdown     : Threshold at which position is zeroed out.

        Returns
        -------
        Adjusted position fraction in [0, 1].
        """
        scale    = max(0.0, 1.0 - current_drawdown / (max_drawdown + 1e-8))
        adjusted = raw_action * scale
        return float(np.clip(adjusted, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Master method used by the environment
    # ------------------------------------------------------------------

    @staticmethod
    def atr_vol_combined(
        raw_action: float,
        atr: float,
        current_price: float,
        portfolio_value: float,
        asset_vol: float,
        current_drawdown: float,
        config,
        win_rate: float | None = None,
        win_loss_ratio: float | None = None,
    ) -> float:
        """
        Apply all four sizing constraints in sequence:
          1. ATR-based stop-loss sizing
          2. Volatility targeting
          3. Drawdown guard
          4. Half-Kelly (only if sufficient trade history is available)

        Parameters
        ----------
        raw_action      : Raw actor output in [0, 1].
        atr             : Current ATR value.
        current_price   : Latest asset close price.
        portfolio_value : Current total portfolio value.
        asset_vol       : Annualised rolling 20-day asset return volatility.
        current_drawdown: Current drawdown from peak (positive fraction).
        config          : Project config module (contains constants).
        win_rate        : Rolling trade win rate; None → skip Kelly step.
        win_loss_ratio  : Rolling win/loss ratio; None → skip Kelly step.

        Returns
        -------
        Final position fraction in [0, 1].
        """
        action = float(raw_action)

        # Step 1: ATR stop-loss constraint
        action = PositionSizer.atr_sizing(
            raw_action      = action,
            atr             = atr,
            current_price   = current_price,
            portfolio_value = portfolio_value,
            max_risk        = config.MAX_RISK_PER_TRADE,
            atr_mult        = config.ATR_MULTIPLIER,
        )

        # Step 2: Volatility targeting
        action = PositionSizer.volatility_target_sizing(
            raw_action = action,
            asset_vol  = asset_vol,
            target_vol = config.TARGET_VOLATILITY,
        )

        # Step 3: Drawdown guard
        action = PositionSizer.drawdown_guard(
            raw_action       = action,
            current_drawdown = current_drawdown,
            max_drawdown     = 0.20,
        )

        # Step 4: Half-Kelly (only when trade history is available)
        if (win_rate is not None
                and win_loss_ratio is not None
                and win_rate > 0.0
                and win_loss_ratio > 0.0):
            action = PositionSizer.kelly_sizing(
                raw_action      = action,
                win_rate        = win_rate,
                win_loss_ratio  = win_loss_ratio,
                kelly_fraction  = config.KELLY_FRACTION,
            )

        return float(np.clip(action, 0.0, 1.0))
