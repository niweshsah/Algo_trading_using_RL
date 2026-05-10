# =============================================================================
# src/data_pipeline.py
# Handles downloading, caching, and splitting market data.
# =============================================================================

import os
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Manages the full data lifecycle:
      1. Download OHLCV data from Yahoo Finance (with local cache).
      2. Clean and validate the raw data.
      3. Split chronologically into train / val / test sets.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_or_download(self, ticker: str, start: str, end: str,
                         interval: str = "1d") -> pd.DataFrame:
        """
        Return a cleaned OHLCV DataFrame.
        If a cached CSV exists it is loaded; otherwise data is downloaded
        from Yahoo Finance, cleaned, and saved to disk.
        """
        cache_path = os.path.join(self.data_dir, f"raw_{ticker}.csv")

        if os.path.exists(cache_path):
            logger.info("Loading cached data from %s", cache_path)
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            logger.info("Downloading %s from %s to %s …", ticker, start, end)
            df = self.download_data(ticker, start, end, interval)

        self._validate(df, ticker)
        return df

    def download_data(self, ticker: str, start: str, end: str,
                      interval: str = "1d") -> pd.DataFrame:
        """
        Download OHLCV data from Yahoo Finance, clean, and cache to CSV.

        Parameters
        ----------
        ticker   : Stock symbol, e.g. "AAPL".
        start    : Start date string "YYYY-MM-DD".
        end      : End date string  "YYYY-MM-DD".
        interval : Bar size; default "1d" (daily).

        Returns
        -------
        pd.DataFrame with columns [Open, High, Low, Close, Volume].
        """
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,   # adjust for splits and dividends
            progress=False,
        )

        # yfinance may return a MultiIndex when downloading a single ticker
        # with auto_adjust=True; flatten if needed.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Keep only the canonical columns
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Handle missing values: forward-fill first, then backward-fill
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Drop any rows that are still NaN after filling
        df.dropna(inplace=True)

        # Persist to disk
        cache_path = os.path.join(self.data_dir, f"raw_{ticker}.csv")
        df.to_csv(cache_path)
        logger.info("Saved raw data to %s  (rows=%d)", cache_path, len(df))

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float   = 0.15,
        test_ratio: float  = 0.15,
    ):
        """
        Chronological (non-shuffled) train / val / test split.

        Parameters
        ----------
        df          : Full OHLCV DataFrame sorted by date (ascending).
        train_ratio : Fraction of rows used for training.
        val_ratio   : Fraction used for validation.
        test_ratio  : Fraction used for testing.

        Returns
        -------
        train_df, val_df, test_df  —  three non-overlapping DataFrames.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        n = len(df)
        train_end = int(n * train_ratio)
        val_end   = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy()
        val_df   = df.iloc[train_end:val_end].copy()
        test_df  = df.iloc[val_end:].copy()

        # Print summary
        print("\n" + "=" * 55)
        print("  DATA SPLIT SUMMARY")
        print("=" * 55)
        for name, part in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            print(
                f"  {name:<6}  rows={len(part):>5}  "
                f"{part.index[0].date()} → {part.index[-1].date()}"
            )
        print("=" * 55 + "\n")

        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame, ticker: str) -> None:
        """Raise ValueError if the DataFrame does not meet quality thresholds."""
        min_rows = 500
        if len(df) < min_rows:
            raise ValueError(
                f"Insufficient data for {ticker}: "
                f"got {len(df)} rows, need at least {min_rows}."
            )

        required_cols = {"Open", "High", "Low", "Close", "Volume"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in data: {missing}")

        nan_count = df[list(required_cols)].isna().sum().sum()
        if nan_count > 0:
            raise ValueError(f"Data still contains {nan_count} NaN values after cleaning.")

        logger.info(
            "Data validation passed: %d rows, %s → %s",
            len(df),
            df.index[0].date(),
            df.index[-1].date(),
        )
