# =============================================================================
# config.py — Central Configuration for LSTM + SAC Trading System
# =============================================================================

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
TICKER         = "AAPL"
START_DATE     = "2018-01-01"
END_DATE       = "2023-12-31"
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
TEST_RATIO     = 0.15
DATA_INTERVAL  = "1d"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
WINDOW_SIZE    = 30        # LSTM lookback window (timesteps)
INDICATORS = [
    "SMA_5", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_middle", "BB_lower", "BB_width",
    "ATR_14",
    "OBV",
    "STOCH_k", "STOCH_d",
    "ADX_14",
    "CCI_20",
    "ROC_10",
    "MFI_14",
    "VWAP",
]

# ---------------------------------------------------------------------------
# Environment settings
# ---------------------------------------------------------------------------
INITIAL_CAPITAL              = 100_000.0   # USD
COMMISSION_RATE              = 0.001       # 0.1% per trade
SLIPPAGE_RATE                = 0.0005      # 0.05% slippage
MAX_POSITION                 = 1.0         # max fraction of capital in one stock
MIN_POSITION                 = 0.0         # 0 = fully out of market
TRANSACTION_COST_THRESHOLD   = 0.001       # minimum action delta to count as trade

# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------
ATR_MULTIPLIER      = 2.0         # stop-loss = ATR × this
MAX_RISK_PER_TRADE  = 0.02        # max 2% portfolio loss per trade
TARGET_VOLATILITY   = 0.15        # annual target portfolio vol (15%)
KELLY_FRACTION      = 0.5         # half-Kelly safety factor

# ---------------------------------------------------------------------------
# Reward function weights
# ---------------------------------------------------------------------------
REWARD_SHARPE_WEIGHT    = 0.4
REWARD_RETURN_WEIGHT    = 0.4
REWARD_DRAWDOWN_WEIGHT  = 0.2
REWARD_LOOKBACK         = 20      # timesteps for rolling Sharpe

# ---------------------------------------------------------------------------
# LSTM architecture
# ---------------------------------------------------------------------------
LSTM_HIDDEN_SIZE    = 128
LSTM_NUM_LAYERS     = 2
LSTM_DROPOUT        = 0.2
FC_HIDDEN_SIZES     = [256, 128]

# ---------------------------------------------------------------------------
# SAC hyperparameters
# ---------------------------------------------------------------------------
LEARNING_RATE_ACTOR     = 3e-4
LEARNING_RATE_CRITIC    = 3e-4
LEARNING_RATE_ALPHA     = 3e-4
GAMMA                   = 0.99
TAU                     = 0.005       # soft update rate for target network
BUFFER_SIZE             = 100_000
BATCH_SIZE              = 256
WARMUP_STEPS            = 1_000
GRADIENT_STEPS          = 1
TARGET_ENTROPY          = -1.0        # for automatic entropy tuning
UPDATE_INTERVAL         = 1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS     = 500_000
EVAL_FREQUENCY      = 10_000
CHECKPOINT_FREQ     = 50_000
SEED                = 42
DEVICE              = "cuda"          # or "cpu"

# ---------------------------------------------------------------------------
# Logging & I/O
# ---------------------------------------------------------------------------
LOG_DIR         = "results/logs"
CHECKPOINT_DIR  = "results/checkpoints"
PLOT_DIR        = "results/plots"
DATA_DIR        = "data"
