# =============================================================================
# src/environment.py
# Custom OpenAI Gym environment for continuous-action RL trading.
# =============================================================================

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, Any, Optional

from src.position_sizing import PositionSizer
from src.reward import RewardCalculator
from src.utils import TradeTracker


class TradingEnvironment(gym.Env):
    """
    Single-asset trading environment with continuous position sizing.

    Observation
    -----------
    Shape (window_size, n_features + 3).
    The +3 portfolio state features are appended to each timestep slice:
      - current_position : fraction of capital currently in the stock [0, 1]
      - unrealised_pnl   : (current_price - entry_price) / entry_price
      - drawdown         : peak-to-current drawdown as a positive fraction

    Action
    ------
    Box(0, 1, shape=(1,), dtype=float32).
    Scalar representing the DESIRED position fraction [0 = all cash, 1 = fully invested].
    The raw action is passed through PositionSizer.atr_vol_combined() before execution.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        sequences:   np.ndarray,   # (n_samples, window_size, n_features)
        prices:      np.ndarray,   # (n_samples,) raw close prices
        dates:       np.ndarray,   # (n_samples,) datetime-like labels
        config,                    # project config module
        mode: str = "train",
    ):
        super().__init__()

        self.sequences    = sequences.astype(np.float32)
        self.prices       = prices.astype(np.float32)
        self.dates        = dates
        self.config       = config
        self.mode         = mode

        self.n_steps      = len(sequences)
        self.window_size  = sequences.shape[1]
        self.n_features   = sequences.shape[2]
        self.n_obs_feat   = self.n_features + 3   # +3 portfolio state features

        # ---- Gym spaces -------------------------------------------------
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, self.n_obs_feat),
            dtype=np.float32,
        )

        # ---- Portfolio state (initialised in reset) ---------------------
        self.capital:         float = config.INITIAL_CAPITAL
        self.cash:            float = config.INITIAL_CAPITAL
        self.shares:          float = 0.0
        self.position:        float = 0.0   # current position fraction
        self.entry_price:     float = 0.0
        self.peak_value:      float = config.INITIAL_CAPITAL
        self.current_step:    int   = 0
        self.portfolio_history: list = []
        self.prev_action:     float = 0.0

        # ---- Trade tracking for Kelly / evaluation ----------------------
        self.trade_tracker = TradeTracker(rolling_window=50)
        self._in_position  = False
        self._entry_step:  int   = 0

        # Pre-compute ATR and rolling volatility arrays from price sequence
        self._atr_array: np.ndarray = self._precompute_atr()
        self._vol_array: np.ndarray = self._precompute_vol()

    # ------------------------------------------------------------------
    # Core Gym interface
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset environment to initial state and return first observation."""
        self.capital      = self.config.INITIAL_CAPITAL
        self.cash         = self.config.INITIAL_CAPITAL
        self.shares       = 0.0
        self.position     = 0.0
        self.entry_price  = 0.0
        self.peak_value   = self.config.INITIAL_CAPITAL
        self.current_step = 0
        self.prev_action  = 0.0
        self.portfolio_history = [self.config.INITIAL_CAPITAL]
        self.trade_tracker.reset()
        self._in_position  = False
        self._entry_step   = 0

        obs = self._build_obs()
        return obs

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one trading step.

        Parameters
        ----------
        action : np.ndarray of shape (1,) with value in [0, 1].

        Returns
        -------
        obs, reward, done, truncated, info
        """
        raw_action = float(np.clip(action[0], self.config.MIN_POSITION,
                                               self.config.MAX_POSITION))

        current_price = float(self.prices[self.current_step])
        atr           = float(self._atr_array[self.current_step])
        asset_vol     = float(self._vol_array[self.current_step])
        portfolio_val = self._portfolio_value(current_price)

        # Drawdown for position sizing
        drawdown = max(0.0, (self.peak_value - portfolio_val) / (self.peak_value + 1e-8))

        # Pull Kelly stats from tracker (None if insufficient history)
        win_rate       = self.trade_tracker.win_rate(use_rolling=True)
        win_loss_ratio = self.trade_tracker.win_loss_ratio(use_rolling=True)

        # Apply composite position sizing
        adjusted_action = PositionSizer.atr_vol_combined(
            raw_action      = raw_action,
            atr             = atr,
            current_price   = current_price,
            portfolio_value = portfolio_val,
            asset_vol       = asset_vol,
            current_drawdown= drawdown,
            config          = self.config,
            win_rate        = win_rate,
            win_loss_ratio  = win_loss_ratio,
        )

        desired_position = float(np.clip(adjusted_action, 0.0, 1.0))

        # ---- Execute trade ----------------------------------------------
        trade_cost = self._execute_trade(desired_position, current_price, portfolio_val)

        # ---- Update portfolio ------------------------------------------
        new_portfolio_val = self._portfolio_value(current_price)
        self.peak_value   = max(self.peak_value, new_portfolio_val)
        current_drawdown  = max(0.0, (self.peak_value - new_portfolio_val)
                                     / (self.peak_value + 1e-8))
        self.portfolio_history.append(new_portfolio_val)

        # ---- Compute reward --------------------------------------------
        reward = RewardCalculator.compute(
            portfolio_values = self.portfolio_history,
            current_step     = self.current_step,
            action           = desired_position,
            prev_action      = self.prev_action,
            config           = self.config,
        )

        self.prev_action = desired_position

        # ---- Advance step ----------------------------------------------
        self.current_step += 1
        done      = self.current_step >= self.n_steps - 1
        truncated = current_drawdown > 0.50   # stop if 50 % of capital lost

        # ---- Build observation -----------------------------------------
        obs = self._build_obs()

        # ---- Info dict -------------------------------------------------
        unrealised_pnl = (
            (current_price - self.entry_price) / (self.entry_price + 1e-8)
            if self._in_position else 0.0
        )
        info: Dict[str, Any] = {
            "portfolio_value":     new_portfolio_val,
            "cash":                self.cash,
            "shares":              self.shares,
            "position_fraction":   self.position,
            "trade_cost":          trade_cost,
            "drawdown":            current_drawdown,
            "step":                self.current_step,
            "price":               current_price,
            "action_taken":        raw_action,
            "atr_adjusted_action": adjusted_action,
        }

        # Sanity check
        assert 0.0 <= info["position_fraction"] <= 1.0, \
            f"position_fraction out of bounds: {info['position_fraction']}"

        return obs, reward, done, truncated, info

    def render(self, mode: str = "human") -> None:
        if mode == "human":
            price = float(self.prices[self.current_step])
            pv    = self._portfolio_value(price)
            print(
                f"  Step {self.current_step:>5} | "
                f"Price: {price:>10.2f} | "
                f"Portfolio: {pv:>12.2f} | "
                f"Cash: {self.cash:>12.2f} | "
                f"Shares: {self.shares:>8.2f} | "
                f"Pos: {self.position:.3f}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """
        Concatenate the current LSTM feature window with three portfolio
        state scalars broadcast across every timestep.
        Returns ndarray of shape (window_size, n_features + 3).
        """
        step = min(self.current_step, self.n_steps - 1)
        feat_window = self.sequences[step].copy()   # (window_size, n_features)

        price = float(self.prices[step])
        pv    = self._portfolio_value(price)

        current_pos    = float(self.position)
        unrealised_pnl = (
            (price - self.entry_price) / (self.entry_price + 1e-8)
            if self._in_position else 0.0
        )
        drawdown = max(0.0, (self.peak_value - pv) / (self.peak_value + 1e-8))

        # Append the three scalars at every timestep row
        extras = np.array(
            [current_pos, unrealised_pnl, drawdown], dtype=np.float32
        )                                       # (3,)
        extras_tiled = np.tile(extras, (self.window_size, 1))   # (window_size, 3)

        obs = np.concatenate([feat_window, extras_tiled], axis=1)  # (window, n+3)
        return obs.astype(np.float32)

    def _execute_trade(
        self,
        desired_position: float,
        current_price:    float,
        portfolio_value:  float,
    ) -> float:
        """
        Compute and execute the trade implied by moving from the current
        position fraction to `desired_position`.

        Returns the total transaction cost (commission + slippage) in $$.
        """
        delta = desired_position - self.position   # positive = buy, negative = sell

        # Skip tiny trades to avoid excessive churning
        trade_value = abs(delta * portfolio_value)
        if trade_value < portfolio_value * self.config.TRANSACTION_COST_THRESHOLD:
            return 0.0

        commission = trade_value * self.config.COMMISSION_RATE
        slippage   = trade_value * self.config.SLIPPAGE_RATE
        trade_cost = commission + slippage

        if delta > 0.0:
            # ---- Buying -------------------------------------------------
            # Effective spend = cost of shares + transaction fees
            dollars_to_invest = delta * portfolio_value - trade_cost
            if dollars_to_invest <= 0.0 or self.cash < dollars_to_invest + trade_cost:
                return 0.0
            shares_bought  = dollars_to_invest / (current_price + 1e-8)
            self.shares   += shares_bought
            self.cash     -= dollars_to_invest + trade_cost

            # Record entry price (weighted average)
            if not self._in_position:
                self.entry_price  = current_price
                self._in_position = True
                self._entry_step  = self.current_step
            else:
                # Weighted average entry
                old_val  = self.position * portfolio_value
                new_val  = desired_position * portfolio_value
                self.entry_price = (
                    self.entry_price * old_val + current_price * (new_val - old_val)
                ) / (new_val + 1e-8)

        elif delta < 0.0:
            # ---- Selling ------------------------------------------------
            shares_to_sell = abs(delta) * portfolio_value / (current_price + 1e-8)
            shares_to_sell = min(shares_to_sell, self.shares)
            proceeds       = shares_to_sell * current_price - trade_cost
            self.shares   -= shares_to_sell
            self.cash     += proceeds

            if desired_position <= 0.01:
                # Fully closed — record trade
                if self._in_position:
                    self.trade_tracker.record(
                        entry_price = self.entry_price,
                        exit_price  = current_price,
                        size        = shares_to_sell,
                        entry_step  = self._entry_step,
                        exit_step   = self.current_step,
                    )
                self._in_position = False
                self.entry_price  = 0.0

        self.position = desired_position
        return trade_cost

    def _portfolio_value(self, current_price: float) -> float:
        return self.cash + self.shares * current_price

    def _precompute_atr(self, period: int = 14) -> np.ndarray:
        """
        Compute a simple ATR array from the raw price sequence.
        Since we only have Close prices in `self.prices`, approximate:
          TR ≈ |Close_t - Close_{t-1}|
        """
        n   = len(self.prices)
        atr = np.zeros(n, dtype=np.float32)
        tr  = np.abs(np.diff(self.prices, prepend=self.prices[0]))

        # Exponential moving average of TR
        alpha = 2.0 / (period + 1)
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

        # Replace zeros with a small fallback
        atr = np.where(atr < 1e-4, 1e-4, atr)
        return atr

    def _precompute_vol(self, window: int = 20) -> np.ndarray:
        """
        Annualised rolling 20-day return volatility for each step.
        """
        n   = len(self.prices)
        vol = np.zeros(n, dtype=np.float32)

        log_returns = np.diff(np.log(self.prices + 1e-8), prepend=np.log(self.prices[0] + 1e-8))

        for i in range(n):
            start  = max(0, i - window + 1)
            sample = log_returns[start : i + 1]
            if len(sample) > 1:
                vol[i] = float(np.std(sample) * np.sqrt(252))
            else:
                vol[i] = 0.15   # default to target vol if insufficient history

        return vol
