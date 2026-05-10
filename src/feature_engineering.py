# =============================================================================
# src/feature_engineering.py
# Pure-numpy/pandas technical indicator computation + normalisation + sequencing.
# No ta-lib dependency — all indicators implemented from scratch to avoid
# C-extension installation issues on arbitrary environments.
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List


class FeatureEngineer:
    """
    Computes 22 technical indicators, normalises with RobustScaler (fit on
    train only), and packages data into (samples, window, features) sequences.
    """

    def __init__(self):
        self.scaler: RobustScaler | None = None
        self._feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Append all technical indicators to a copy of the OHLCV DataFrame.
        Returns the augmented DataFrame with NaN rows dropped.

        Parameters
        ----------
        df : DataFrame with columns [Open, High, Low, Close, Volume].
        """
        out = df.copy()

        close  = out["Close"]
        high   = out["High"]
        low    = out["Low"]
        volume = out["Volume"]

        # ---- Trend: moving averages ------------------------------------
        out["SMA_5"]  = self._sma(close, 5)
        out["SMA_20"] = self._sma(close, 20)
        out["SMA_50"] = self._sma(close, 50)
        out["EMA_12"] = self._ema(close, 12)
        out["EMA_26"] = self._ema(close, 26)

        # ---- MACD -------------------------------------------------------
        macd_line          = out["EMA_12"] - out["EMA_26"]
        macd_signal        = self._ema(macd_line, 9)
        out["MACD"]        = macd_line
        out["MACD_signal"] = macd_signal
        out["MACD_hist"]   = macd_line - macd_signal

        # ---- Momentum: RSI (14) ----------------------------------------
        out["RSI_14"] = self._rsi(close, 14)

        # ---- Bollinger Bands (20, 2σ) -----------------------------------
        bb_mid            = self._sma(close, 20)
        bb_std            = close.rolling(20, min_periods=1).std()
        out["BB_upper"]   = bb_mid + 2 * bb_std
        out["BB_middle"]  = bb_mid
        out["BB_lower"]   = bb_mid - 2 * bb_std
        out["BB_width"]   = (out["BB_upper"] - out["BB_lower"]) / (bb_mid + 1e-8)

        # ---- Volatility: ATR (14) --------------------------------------
        out["ATR_14"] = self._atr(high, low, close, 14)

        # ---- Volume: OBV -----------------------------------------------
        out["OBV"] = self._obv(close, volume)

        # ---- Stochastic Oscillator %K, %D ------------------------------
        stoch_k, stoch_d   = self._stochastic(high, low, close, k=14, d=3)
        out["STOCH_k"]     = stoch_k
        out["STOCH_d"]     = stoch_d

        # ---- Trend strength: ADX (14) ----------------------------------
        out["ADX_14"] = self._adx(high, low, close, 14)

        # ---- CCI (20) --------------------------------------------------
        out["CCI_20"] = self._cci(high, low, close, 20)

        # ---- ROC (10) --------------------------------------------------
        out["ROC_10"] = self._roc(close, 10)

        # ---- MFI (14) --------------------------------------------------
        out["MFI_14"] = self._mfi(high, low, close, volume, 14)

        # ---- VWAP (cumulative, reset per calendar day) -----------------
        out["VWAP"] = self._vwap(high, low, close, volume)

        # Record feature column names (excluding raw OHLCV)
        ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
        self._feature_names = [c for c in out.columns if c not in ohlcv_cols]

        # Drop rows containing NaN (warm-up period from long windows)
        out.dropna(inplace=True)

        return out

    def normalize_features(
        self,
        train_df: pd.DataFrame,
        val_df:   pd.DataFrame,
        test_df:  pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
        """
        Fit RobustScaler on training features ONLY; transform all three splits.

        Returns
        -------
        train_scaled, val_scaled, test_scaled : DataFrames with same index/columns.
        scaler : the fitted RobustScaler instance.
        """
        feature_cols = self._feature_names
        if not feature_cols:
            raise RuntimeError("Call compute_indicators() before normalize_features().")

        self.scaler = RobustScaler()

        # Fit on train features only (prevent data leakage)
        train_features = train_df[feature_cols].values.astype(np.float32)
        self.scaler.fit(train_features)

        def _transform(df: pd.DataFrame) -> pd.DataFrame:
            scaled_values = self.scaler.transform(
                df[feature_cols].values.astype(np.float32)
            )
            scaled_df = df.copy()
            scaled_df[feature_cols] = scaled_values
            return scaled_df

        train_scaled = _transform(train_df)
        val_scaled   = _transform(val_df)
        test_scaled  = _transform(test_df)

        return train_scaled, val_scaled, test_scaled, self.scaler

    def create_sequences(
        self,
        df: pd.DataFrame,
        window_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create overlapping sliding-window sequences for the LSTM.

        Parameters
        ----------
        df          : Scaled DataFrame containing at least feature columns
                      and the raw Close price.
        window_size : Number of timesteps per input sequence.

        Returns
        -------
        X      : (n_samples, window_size, n_features)  float32
        prices : (n_samples,)  raw Close prices aligned with the LAST step
                 of each window (i.e. the current price the env will use).
        dates  : (n_samples,)  DatetimeIndex values aligned with the last step.
        """
        feature_cols = self._feature_names
        data = df[feature_cols].values.astype(np.float32)
        close_prices = df["Close"].values.astype(np.float32)
        dates = df.index.values

        n = len(data)
        n_samples = n - window_size + 1  # last window ends at index n-1

        X      = np.zeros((n_samples, window_size, len(feature_cols)), dtype=np.float32)
        prices = np.zeros(n_samples, dtype=np.float32)
        out_dates = np.empty(n_samples, dtype=dates.dtype)

        for i in range(n_samples):
            X[i]         = data[i : i + window_size]
            prices[i]    = close_prices[i + window_size - 1]
            out_dates[i] = dates[i + window_size - 1]

        # Sanity checks
        assert X.shape == (n_samples, window_size, len(feature_cols)), \
            f"Unexpected X shape: {X.shape}"
        assert not np.isnan(X).any(), "NaN values detected in sequence array X"

        return X, prices, out_dates

    def get_feature_names(self) -> List[str]:
        """Return list of all indicator column names in the order they appear."""
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Private indicator implementations (pure numpy/pandas)
    # ------------------------------------------------------------------

    @staticmethod
    def _sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period, min_periods=1).mean()

    @staticmethod
    def _ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False, min_periods=1).mean()

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta  = close.diff()
        gain   = delta.clip(lower=0)
        loss   = (-delta).clip(lower=0)
        avg_g  = gain.ewm(com=period - 1, adjust=False, min_periods=1).mean()
        avg_l  = loss.ewm(com=period - 1, adjust=False, min_periods=1).mean()
        rs     = avg_g / (avg_l + 1e-8)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series,
              close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False, min_periods=1).mean()

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        sign = np.sign(close.diff().fillna(0))
        return (volume * sign).cumsum()

    @staticmethod
    def _stochastic(
        high: pd.Series, low: pd.Series, close: pd.Series,
        k: int = 14, d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        low_k  = low.rolling(k, min_periods=1).min()
        high_k = high.rolling(k, min_periods=1).max()
        pct_k  = 100.0 * (close - low_k) / (high_k - low_k + 1e-8)
        pct_d  = pct_k.rolling(d, min_periods=1).mean()
        return pct_k, pct_d

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series,
              close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index via Wilder smoothing."""
        prev_high  = high.shift(1)
        prev_low   = low.shift(1)
        prev_close = close.shift(1)

        up_move   = high  - prev_high
        down_move = prev_low - low

        plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)

        atr     = tr.ewm(span=period, adjust=False, min_periods=1).mean()
        plus_di = 100.0 * pd.Series(plus_dm,  index=close.index).ewm(
            span=period, adjust=False, min_periods=1).mean() / (atr + 1e-8)
        minus_di = 100.0 * pd.Series(minus_dm, index=close.index).ewm(
            span=period, adjust=False, min_periods=1).mean() / (atr + 1e-8)

        dx  = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        adx = dx.ewm(span=period, adjust=False, min_periods=1).mean()
        return adx

    @staticmethod
    def _cci(high: pd.Series, low: pd.Series,
              close: pd.Series, period: int = 20) -> pd.Series:
        typical  = (high + low + close) / 3.0
        sma_typ  = typical.rolling(period, min_periods=1).mean()
        mean_dev = typical.rolling(period, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        return (typical - sma_typ) / (0.015 * mean_dev + 1e-8)

    @staticmethod
    def _roc(close: pd.Series, period: int = 10) -> pd.Series:
        return (close - close.shift(period)) / (close.shift(period) + 1e-8) * 100.0

    @staticmethod
    def _mfi(high: pd.Series, low: pd.Series, close: pd.Series,
              volume: pd.Series, period: int = 14) -> pd.Series:
        """Money Flow Index."""
        typical  = (high + low + close) / 3.0
        raw_flow = typical * volume
        diff     = typical.diff()

        pos_flow = raw_flow.where(diff > 0, 0.0)
        neg_flow = raw_flow.where(diff < 0, 0.0).abs()

        sum_pos  = pos_flow.rolling(period, min_periods=1).sum()
        sum_neg  = neg_flow.rolling(period, min_periods=1).sum()

        mfi_ratio = sum_pos / (sum_neg + 1e-8)
        return 100.0 - (100.0 / (1.0 + mfi_ratio))

    @staticmethod
    def _vwap(high: pd.Series, low: pd.Series,
               close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Cumulative VWAP reset per calendar day.
        Works correctly on both single-day and multi-day DataFrames.
        """
        typical = (high + low + close) / 3.0
        df_tmp = pd.DataFrame({
            "typical": typical,
            "volume":  volume,
            "tp_vol":  typical * volume,
        })

        # Group by date to reset the cumsum each day
        df_tmp["date"] = df_tmp.index.normalize()
        cum_tp_vol = df_tmp.groupby("date")["tp_vol"].cumsum()
        cum_vol    = df_tmp.groupby("date")["volume"].cumsum()
        return cum_tp_vol / (cum_vol + 1e-8)
