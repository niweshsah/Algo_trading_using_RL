# Reinforcement Learning Trading Agent

A production-grade reinforcement learning system for continuous-action algorithmic trading, combining LSTM-based feature extraction with Soft Actor-Critic (SAC) optimization. Demonstrates state-of-the-art deep RL techniques applied to quantitative finance.

## Table of Contents

- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Installation Instructions](#installation-instructions)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Evaluation & Backtesting](#evaluation--backtesting)
- [Configuration](#configuration)
- [Checkpoint Handling](#checkpoint-handling)
- [Logging & Monitoring](#logging--monitoring)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

---

## Project Overview

### What the Project Does

This project implements an end-to-end deep reinforcement learning trading system that learns to manage a continuous position in a single asset (Apple stock) by:

- **Processing sequential market data** via an LSTM feature extractor that ingests 22+ technical indicators computed from raw OHLCV (Open-High-Low-Close-Volume) data
- **Making continuous position decisions** through a Soft Actor-Critic (SAC) agent that outputs a scalar position fraction [0, 1]
- **Managing risk** via multi-layer position sizing incorporating ATR-based stops, volatility targeting, Kelly criterion, and drawdown limits
- **Optimizing risk-adjusted returns** by maximizing a reward signal weighted across Sharpe ratio, log returns, and drawdown constraints

### Problem Solved

Traditional algorithmic trading systems rely on hand-crafted rules and fixed thresholds. This project demonstrates how **deep RL learns adaptive trading policies directly from raw market data**, automatically discovering the optimal balance between:

- Profitable trade selection
- Risk management and position sizing
- Market regime adaptation
- Transaction cost minimization

### Main Features & Capabilities

| Feature | Details |
|---------|---------|
| **RL Algorithm** | Soft Actor-Critic (SAC) with automatic entropy regularization |
| **Feature Engineering** | 22 custom technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, CCI, ROC, MFI, VWAP, OBV) |
| **Architecture** | Shared LSTM backbone + Twin Q-networks + Gaussian policy actor |
| **Risk Management** | ATR-based stops, Kelly criterion, volatility targeting, drawdown limits |
| **Data Pipeline** | Automatic yfinance downloading, train/val/test splitting, rolling normalization |
| **Evaluation** | Deterministic backtesting, performance metrics (Sharpe, max drawdown, win rate), visualization |
| **Monitoring** | TensorBoard logging, mid-training checkpoint evaluation, detailed trade tracking |
| **Reproducibility** | Seed control, configuration-driven system, checkpoint save/load |

### High-Level Workflow

```
Raw Price Data (yfinance)
         ↓
   Feature Engineering (22 indicators)
         ↓
   Sliding-window sequences (LSTM input)
         ↓
   Train/Val/Test split + normalization
         ↓
   RL Training Loop:
   - Agent exploration via SAC policy
   - Experience replay + twin Q-networks
   - Periodic validation & checkpointing
         ↓
   Deterministic Backtesting
         ↓
   Performance visualization & metrics
```

### Key Technologies

- **PyTorch 2.0+**: Neural network implementation
- **Stable-Baselines3 utilities**: RL reference implementations (custom SAC from scratch)
- **OpenAI Gym**: Standard RL environment interface
- **Pandas/NumPy**: Data manipulation and technical indicator computation
- **scikit-learn**: RobustScaler for feature normalization
- **TensorBoard**: Training visualization and monitoring
- **yfinance**: Automatic stock price data downloading
- **Matplotlib/Seaborn**: Backtesting visualization

---

## Folder Structure

```
Quantifiers/
├── README.md                          # This file
├── main.py                            # Entry point: end-to-end pipeline
├── config.py                          # Central configuration (all hyperparameters)
├── mid_eval.py                        # Mid-training checkpoint evaluation (safe to run anytime)
├── requirements.txt                   # Python dependencies
│
├── src/                               # Core source code
│   ├── agent.py                       # SAC agent + experience replay buffer
│   ├── environment.py                 # Custom Gym environment for trading
│   ├── feature_engineering.py         # Technical indicator computation
│   ├── networks.py                    # LSTM + Actor/Critic PyTorch modules
│   ├── position_sizing.py             # Risk management position sizing methods
│   ├── reward.py                      # Multi-component reward shaping
│   ├── data_pipeline.py               # Data loading, downloading, splitting
│   └── utils.py                       # Utility functions (seed setting, trade tracking)
│
├── training/                          # Training and evaluation modules
│   ├── train.py                       # Main SAC training loop
│   └── evaluate.py                    # Deterministic evaluation & backtesting
│
├── data/                              # Dataset directory
│   └── raw_AAPL.csv                   # Apple stock OHLCV data (2018–2023)
│
└── results/                           # Training artifacts (auto-created)
    ├── checkpoints/                   # Saved model weights (.pt files)
    │   ├── best_model.pt              # Highest validation Sharpe ratio
    │   ├── step_50000.pt              # Checkpoint at 50k environment steps
    │   ├── step_100000.pt             # Checkpoint at 100k steps
    │   └── step_150000.pt             # Checkpoint at 150k steps
    ├── logs/                          # TensorBoard event files
    │   └── events.out.tfevents.*      # Training curves (loss, reward, metrics)
    └── plots/                         # Backtesting visualizations
        ├── mid_eval_best_model/       # Plots from best_model.pt evaluation
        ├── mid_eval_step_50000/       # Plots from step_50000.pt evaluation
        └── ...                        # Additional checkpoint evaluations
```

### Key Files Explained

| File | Purpose |
|------|---------|
| **main.py** | Complete pipeline: loads data → computes features → trains agent → backtests. Run this for end-to-end training. |
| **config.py** | All hyperparameters, paths, and settings. Modify this to customize training (learning rate, batch size, architecture, etc.). |
| **mid_eval.py** | Standalone evaluation of checkpoints. Safe to run while training is ongoing in another terminal. |
| **src/agent.py** | SAC implementation: actor/critic networks, experience replay, entropy tuning. |
| **src/environment.py** | Gym-compatible trading environment: step-by-step trading simulation with portfolio tracking. |
| **src/feature_engineering.py** | Computes all 22 technical indicators from scratch (no TA-Lib dependency). |
| **src/networks.py** | PyTorch LSTM encoder, Gaussian policy actor, twin Q-network critic. |
| **training/train.py** | Training loop: experience collection, agent updates, validation, checkpointing. |
| **training/evaluate.py** | Backtesting: run trained agent on historical data, compute metrics, generate plots. |

---

## Installation Instructions

### System Requirements

| Requirement | Details |
|-------------|---------|
| **Operating System** | Linux, macOS, Windows (tested on Linux) |
| **Python** | 3.8–3.11 (recommended: 3.10+) |
| **GPU Support** | CUDA 11.8+ & cuDNN 8.6+ for NVIDIA GPUs (optional but recommended) |
| **RAM** | Minimum 8 GB (16+ GB recommended for large batch sizes) |
| **Storage** | ~5 GB for checkpoint + log files during training |

### Step-by-Step Installation

#### 1. Clone/Setup Project

```bash
cd /path/to/your/projects
# Clone the repository if applicable, or navigate to existing directory
cd RL_trading/Quantifiers
```

#### 2. Create Virtual Environment

```bash
# Using venv (recommended for beginners)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using Conda (if you prefer)
conda create -n lstm-sac python=3.10
conda activate lstm-sac
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Dependency Summary:**
- **torch>=2.0.0**: Core neural network framework
- **numpy, pandas**: Numerical computing and data manipulation
- **yfinance>=0.2.28**: Automatic stock price downloading
- **gym>=0.26.0**: Standard RL environment interface
- **stable-baselines3>=2.0.0**: Reference RL implementations (utilities only)
- **scikit-learn>=1.3.0**: Feature normalization
- **matplotlib, seaborn**: Visualization
- **tensorboard>=2.13.0**: Training monitoring
- **tqdm>=4.65.0**: Progress bars
- **scipy>=1.11.0**: Scientific computing

#### 4. GPU Setup (Optional but Recommended)

**Verify GPU availability:**

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output if GPU is available:
```
True
Tesla T4  # (or your GPU name)
```

**If CUDA not detected (GPU training unavailable):**

The code defaults to CPU, but GPU training is significantly faster (~10-50x speedup). To use GPU:

```bash
# Install PyTorch with CUDA support (example for CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Check [PyTorch official page](https://pytorch.org/get-started/locally/) for your CUDA version.

#### 5. Verify Installation

```bash
python -c "import torch, numpy, pandas, gym, stable_baselines3; print('✓ All imports successful')"
```

### Common Installation Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **ImportError: No module named 'torch'** | PyTorch not installed | Run `pip install torch -r requirements.txt` |
| **RuntimeError: CUDA out of memory** | GPU memory exhausted | Reduce `BATCH_SIZE` in `config.py` or switch to CPU |
| **ModuleNotFoundError: sklearn** | scikit-learn not installed | Run `pip install scikit-learn` |
| **Could not find yfinance data** | Network issue or invalid ticker | Check internet connection; verify `TICKER` in `config.py` |
| **Windows: Permission denied (bash scripts)** | Git line endings issue | Run `git config --global core.autocrlf true` before cloning |

### Development Setup (Optional)

For debugging and development:

```bash
# Install optional dev packages
pip install pytest ipython jupyter

# Run tests (if available)
pytest tests/

# Start interactive notebook
jupyter notebook
```

---

## Dataset Preparation

### Data Overview

The project uses **daily OHLCV data** for Apple Inc. (ticker: AAPL) spanning **2018-01-02 to 2023-12-29** (~1500 trading days).

| Field | Type | Example |
|-------|------|---------|
| **Date** | timestamp | 2023-01-03 |
| **Open** | float | 150.23 |
| **High** | float | 151.50 |
| **Low** | float | 149.80 |
| **Close** | float | 150.95 |
| **Volume** | int | 52,237,600 |

### Automatic Data Download

The code automatically downloads data via `yfinance` on first run:

```bash
python main.py
```

On first execution:
1. Checks if `data/raw_AAPL.csv` exists
2. If missing, downloads 2018-01-01 to 2023-12-31 data
3. Stores locally for future runs (avoids repeated API calls)

### Data Format & Preprocessing

**Raw data pipeline:**

```
Step 1: Download/Load Raw CSV
         └─ Columns: Date, Open, High, Low, Close, Volume

Step 2: Compute Technical Indicators (22 total)
         └─ SMA, EMA, RSI, MACD, Bollinger Bands, ATR, ADX, 
            Stochastic, CCI, ROC, MFI, VWAP, OBV

Step 3: Train/Val/Test Split (70% / 15% / 15%)
         └─ Split chronologically to avoid data leakage

Step 4: Normalization (fit on TRAIN only)
         └─ RobustScaler (median ± IQR, handles outliers well)
         └─ Val/Test scaled using train statistics (no future data!)

Step 5: Sliding Window Sequences (window_size=30 timesteps)
         └─ Shape per sample: (30, n_features)
         └─ Creates overlapping windows: ~1500 → ~1470 sequences

Step 6: Create Environments (Gym interface)
         └─ Simulation of trading: observe → act → step
```

**Feature List (22 indicators):**

| Category | Features |
|----------|----------|
| **Trend** | SMA_5, SMA_20, SMA_50, EMA_12, EMA_26 |
| **Momentum** | RSI_14, MACD, MACD_signal, MACD_hist |
| **Volatility** | BB_upper, BB_middle, BB_lower, BB_width, ATR_14 |
| **Strength** | ADX_14, CCI_20 |
| **Volume** | OBV, MFI_14 |
| **Oscillators** | STOCH_k, STOCH_d |
| **Rate of Change** | ROC_10, VWAP |

### Expected Directory Structure

After running the code once:

```
data/
├── raw_AAPL.csv                    # ~60 KB raw data
├── processed_train.npy             # Preprocessed sequences (auto-created)
└── processed_val.npy
└── processed_test.npy
```

### Using Custom Data

To train on a different stock or time period, modify `config.py`:

```python
TICKER       = "MSFT"               # Change stock ticker
START_DATE   = "2020-01-01"         # Change start date
END_DATE     = "2024-01-01"         # Change end date
TRAIN_RATIO  = 0.70                 # Adjust split ratios
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
```

Then run:
```bash
python main.py  # Downloads new data and retrains
```

### Data Validation Checks

The code performs automatic checks:

- ✓ No NaN values in raw OHLCV
- ✓ Monotonic timestamps
- ✓ No duplicate dates
- ✓ Features computed without look-ahead bias
- ✓ Train/Val/Test splits are non-overlapping
- ✓ Normalization fitted on train only (no data leakage)

---

## Training

### Quick Start

To train the full model from scratch:

```bash
# Basic training (uses GPU if available, CPU otherwise)
python main.py

# Expected runtime: ~15-30 minutes on NVIDIA GPU, ~2-4 hours on CPU
```

This executes the complete pipeline:
1. Data loading and feature engineering
2. Train/val/test split and normalization
3. SAC training loop (500k environment steps)
4. Periodic validation and checkpointing
5. Final backtesting on test set

### Training Configuration

All training hyperparameters are centralized in `config.py`. Key parameters:

```python
# Training duration
TOTAL_TIMESTEPS     = 500_000        # Total environment steps
EVAL_FREQUENCY      = 10_000         # Validation every N steps
CHECKPOINT_FREQ     = 50_000         # Save checkpoint every N steps
WARMUP_STEPS        = 1_000          # Random exploration first

# SAC algorithm
LEARNING_RATE_ACTOR = 3e-4           # Actor network learning rate
LEARNING_RATE_CRITIC = 3e-4          # Critic network learning rate
GAMMA               = 0.99           # Discount factor
TAU                 = 0.005          # Soft update rate (target network)
BATCH_SIZE          = 256            # Mini-batch size
BUFFER_SIZE         = 100_000        # Replay buffer capacity

# LSTM architecture
LSTM_HIDDEN_SIZE    = 128            # LSTM hidden dimension
LSTM_NUM_LAYERS     = 2              # Number of stacked LSTMs
LSTM_DROPOUT        = 0.2            # Dropout rate
FC_HIDDEN_SIZES     = [256, 128]     # FC layer sizes after LSTM

# Reward shaping
REWARD_SHARPE_WEIGHT = 0.4           # Weight for Sharpe ratio component
REWARD_RETURN_WEIGHT = 0.4           # Weight for return component
REWARD_DRAWDOWN_WEIGHT = 0.2         # Weight for drawdown penalty
REWARD_LOOKBACK      = 20            # Timesteps for rolling metrics
```

### Multi-GPU Training (If Applicable)

The current implementation is single-GPU optimized. For multi-GPU training, modify `src/agent.py`:

```python
# Example modification (not in base code):
if torch.cuda.device_count() > 1:
    agent.actor = nn.DataParallel(agent.actor)
    agent.critic = nn.DataParallel(agent.critic)
```

However, this requires additional synchronization code not currently included.

### Monitoring Training Progress

Open TensorBoard in a separate terminal while training:

```bash
tensorboard --logdir results/logs/ --port 6006
```

Then navigate to `http://localhost:6006` to see:
- Actor loss over time
- Critic loss
- Reward curves (train & validation)
- Sharpe ratio progression
- Exploration entropy

### Resuming Training from Checkpoint

To resume training from a saved checkpoint instead of restarting:

```python
# In training/train.py, modify the Trainer initialization:
agent = SACAgent(obs_shape=(config.WINDOW_SIZE, n_features), 
                 action_dim=1, config=config, device=config.DEVICE)
agent.load(checkpoint_path="results/checkpoints/step_100000.pt")

trainer = Trainer(agent, train_env, val_env, config)
trainer.train()  # Continues from step 100,000
```

Current code starts fresh each run. For persistent resumption, add this to `main.py`:

```python
import glob
latest_checkpoint = max(glob.glob("results/checkpoints/*.pt"), 
                       key=os.path.getctime, default=None)
if latest_checkpoint:
    agent.load(latest_checkpoint)
    print(f"Resumed from {latest_checkpoint}")
```

### Common Training Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **Loss diverges to NaN** | Exploding gradients | Reduce `LEARNING_RATE_ACTOR` or `LEARNING_RATE_CRITIC` |
| **Training too slow** | GPU not being used | Check: `python -c "import torch; print(torch.cuda.is_available())"` |
| **Out of memory (OOM)** | Batch too large | Reduce `BATCH_SIZE` in `config.py` (e.g., 256→128) |
| **Poor final performance** | Underfitting | Increase `TOTAL_TIMESTEPS` or `LSTM_HIDDEN_SIZE` |
| **Reward stuck at zero** | Reward clipping too aggressive | Adjust weights in `config.py` |

---

## Evaluation & Backtesting

### Running Evaluation

#### Option 1: Full Pipeline with Backtesting (Recommended)

```bash
python main.py
```

Automatically backtests the trained agent on the test set at the end.

#### Option 2: Evaluate Specific Checkpoint

Run the standalone evaluation script (safe to use during training):

```bash
# Evaluate the best model found during training
python mid_eval.py --checkpoint results/checkpoints/best_model.pt

# Evaluate a specific training step
python mid_eval.py --checkpoint results/checkpoints/step_100000.pt

# Auto-select the latest checkpoint
python mid_eval.py

# Evaluate on validation set instead of test
python mid_eval.py --split val

# Save plots to custom directory
python mid_eval.py --checkpoint results/checkpoints/best_model.pt \
                  --plot-dir custom_plots/
```

### Evaluation Metrics

The evaluation produces detailed performance metrics:

| Metric | Definition | Interpretation |
|--------|------------|-----------------|
| **Total Return** | Final portfolio value / initial capital | Absolute profit/loss |
| **Sharpe Ratio** | (μ_return - r_f) / σ_return × √252 | Risk-adjusted returns (higher better) |
| **Max Drawdown** | Peak-to-trough decline / peak | Worst-case loss (lower better, <-0.20 is bad) |
| **Win Rate** | # profitable trades / # total trades | Consistency (>50% is good) |
| **Profit Factor** | Sum of wins / abs(sum of losses) | Return per unit risk (>1.5 is good) |
| **Calmar Ratio** | annual return / max drawdown | Return per unit risk |

### Backtesting Output

After evaluation, the code generates:

```
results/plots/mid_eval_best_model/
├── portfolio_value.png        # Equity curve over time
├── actions.png                # Position sizing decisions
├── prices_with_trades.png     # Buy/sell signals overlaid on price
├── returns_distribution.png   # Histogram of daily returns
├── cumulative_returns.png     # Cumulative P&L
├── drawdown.png               # Underwater plot
└── metrics.txt                # Numerical results summary
```

### Comparing Checkpoints

To compare multiple checkpoints side-by-side:

```bash
# Evaluate best model
python mid_eval.py --checkpoint results/checkpoints/best_model.pt \
                  --plot-dir results/plots/compare_best/

# Evaluate step 50k
python mid_eval.py --checkpoint results/checkpoints/step_50000.pt \
                  --plot-dir results/plots/compare_50k/

# Evaluate step 100k
python mid_eval.py --checkpoint results/checkpoints/step_100000.pt \
                  --plot-dir results/plots/compare_100k/

# Then compare the equity curves visually
```

### Performance Interpretation Guide

**Excellent performance:**
```
Sharpe Ratio > 1.5
Max Drawdown < -10%
Win Rate > 60%
Total Return > 50% over 5 years
```

**Good performance:**
```
Sharpe Ratio 0.8–1.5
Max Drawdown -10% to -25%
Win Rate 50–60%
Total Return 10–50% over 5 years
```

**Poor performance:**
```
Sharpe Ratio < 0.5
Max Drawdown < -50%
Win Rate < 50%
Total Return negative
```

### Deterministic vs. Stochastic Evaluation

The evaluation script uses **deterministic actions** (mean of policy distribution):

```python
action = agent.select_action(obs, deterministic=True)  # Use mean
# vs
action = agent.select_action(obs, deterministic=False)  # Sample from distribution
```

Deterministic evaluation is standard for final performance reporting. Stochastic is useful for understanding policy variance.

---

## Configuration

### Central Configuration File (`config.py`)

All hyperparameters are grouped into logical sections. Modify values to customize training:

```python
# =============================================================================
# config.py
# =============================================================================

# --- DATA SETTINGS ---
TICKER = "AAPL"                      # Stock symbol
START_DATE = "2018-01-01"            # Training data start
END_DATE = "2023-12-31"              # Training data end
TRAIN_RATIO = 0.70                   # 70% train
VAL_RATIO = 0.15                     # 15% validation
TEST_RATIO = 0.15                    # 15% test
DATA_INTERVAL = "1d"                 # Daily data

# --- FEATURE ENGINEERING ---
WINDOW_SIZE = 30                     # LSTM lookback (30 days)
INDICATORS = [...]                   # 22 technical indicators (see list)

# --- ENVIRONMENT ---
INITIAL_CAPITAL = 100_000.0          # Starting capital in USD
COMMISSION_RATE = 0.001              # 0.1% per trade
SLIPPAGE_RATE = 0.0005               # 0.05% slippage
MAX_POSITION = 1.0                   # Max 100% invested
MIN_POSITION = 0.0                   # Min 0% (can be fully in cash)
TRANSACTION_COST_THRESHOLD = 0.001   # Minimum position change to count as trade

# --- POSITION SIZING ---
ATR_MULTIPLIER = 2.0                 # Stop-loss = ATR × 2
MAX_RISK_PER_TRADE = 0.02            # Max 2% of portfolio per trade
TARGET_VOLATILITY = 0.15             # Target 15% annual portfolio volatility
KELLY_FRACTION = 0.5                 # Half-Kelly for safety

# --- REWARD FUNCTION ---
REWARD_SHARPE_WEIGHT = 0.4           # Weight Sharpe ratio
REWARD_RETURN_WEIGHT = 0.4           # Weight log returns
REWARD_DRAWDOWN_WEIGHT = 0.2         # Weight drawdown penalty
REWARD_LOOKBACK = 20                 # 20-step rolling window for metrics

# --- LSTM ARCHITECTURE ---
LSTM_HIDDEN_SIZE = 128               # Hidden state dimension
LSTM_NUM_LAYERS = 2                  # Number of stacked LSTM layers
LSTM_DROPOUT = 0.2                   # Dropout rate
FC_HIDDEN_SIZES = [256, 128]         # Fully-connected layers after LSTM

# --- SAC HYPERPARAMETERS ---
LEARNING_RATE_ACTOR = 3e-4           # Actor learning rate
LEARNING_RATE_CRITIC = 3e-4          # Critic learning rate
LEARNING_RATE_ALPHA = 3e-4           # Entropy temperature learning rate
GAMMA = 0.99                         # Discount factor
TAU = 0.005                          # Soft update (target network = 0.995 * current + 0.005 * target)
BUFFER_SIZE = 100_000                # Replay buffer capacity
BATCH_SIZE = 256                     # Mini-batch size
WARMUP_STEPS = 1_000                 # Random exploration first
GRADIENT_STEPS = 1                   # Updates per environment step
TARGET_ENTROPY = -1.0                # Automatic entropy tuning target
UPDATE_INTERVAL = 1                  # Update frequency

# --- TRAINING ---
TOTAL_TIMESTEPS = 500_000            # Total environment steps
EVAL_FREQUENCY = 10_000              # Validation every N steps
CHECKPOINT_FREQ = 50_000             # Checkpoint every N steps
SEED = 42                            # Random seed for reproducibility
DEVICE = "cuda"                      # "cuda" (GPU) or "cpu"

# --- I/O PATHS ---
LOG_DIR = "results/logs"             # TensorBoard logs
CHECKPOINT_DIR = "results/checkpoints"
PLOT_DIR = "results/plots"
DATA_DIR = "data"
```

### Configuration Customization Examples

**Example 1: Speed up training (smaller model, fewer steps)**

```python
TOTAL_TIMESTEPS = 100_000            # 5x fewer steps
LSTM_HIDDEN_SIZE = 64                # Smaller LSTM
FC_HIDDEN_SIZES = [128, 64]          # Smaller FC layers
BATCH_SIZE = 128                     # Smaller batches
EVAL_FREQUENCY = 5_000               # Less frequent eval
```

Expected runtime: ~3-5 minutes on GPU, ~30-60 minutes on CPU.

**Example 2: More aggressive risk management**

```python
MAX_RISK_PER_TRADE = 0.01            # 1% max risk (was 2%)
TARGET_VOLATILITY = 0.10             # 10% target vol (was 15%)
KELLY_FRACTION = 0.25                # Quarter-Kelly (was 0.5)
REWARD_DRAWDOWN_WEIGHT = 0.5         # Penalize drawdowns more (was 0.2)
```

**Example 3: Train on different asset**

```python
TICKER = "MSFT"                      # Microsoft stock
START_DATE = "2020-01-01"            # Last 3-4 years
END_DATE = "2024-01-01"
```

Then run: `python main.py`

### Environment Variables

For deployment or CI/CD, override config with environment variables:

```bash
export DEVICE=cpu                    # Use CPU instead of GPU
export INITIAL_CAPITAL=50000          # $50k instead of $100k
export TOTAL_TIMESTEPS=1000000        # 1 million steps
python main.py
```

To use in code (not currently implemented):

```python
import os
device = os.getenv("DEVICE", config.DEVICE)
```

---

## Checkpoint Handling

### Checkpoint Format

Checkpoints are PyTorch `.pt` files containing:

```python
{
    "actor_state_dict": {...},       # Actor network weights
    "critic_state_dict": {...},      # Critic network weights
    "target_critic_state_dict": {...},  # Target critic (for soft updates)
    "actor_optimizer_state": {...},  # Adam optimizer state
    "critic_optimizer_state": {...},
    "alpha": float,                  # Entropy temperature
    "alpha_optimizer_state": {...},
    "global_step": int,              # Training step number
    "timestamp": str,                # When checkpoint was saved
}
```

### Automatic Checkpointing

During training, checkpoints are saved automatically:

```
At step 50,000:  results/checkpoints/step_50000.pt
At step 100,000: results/checkpoints/step_100000.pt
At step 150,000: results/checkpoints/step_150000.pt
...
Best (highest val Sharpe): results/checkpoints/best_model.pt
```

### Manual Checkpoint Operations

**Load a checkpoint for evaluation:**

```python
from src.agent import SACAgent
import torch

agent = SACAgent(obs_shape=(30, 25), action_dim=1, config=config, device="cuda")

# Load weights from disk
checkpoint = torch.load("results/checkpoints/best_model.pt")
agent.actor.load_state_dict(checkpoint["actor_state_dict"])
agent.critic.load_state_dict(checkpoint["critic_state_dict"])
agent.target_critic.load_state_dict(checkpoint["target_critic_state_dict"])

# Now use agent.select_action() for inference
action = agent.select_action(observation, deterministic=True)
```

**Save current agent state:**

```python
checkpoint = {
    "actor_state_dict": agent.actor.state_dict(),
    "critic_state_dict": agent.critic.state_dict(),
    "target_critic_state_dict": agent.target_critic.state_dict(),
    "actor_optimizer_state": agent.actor_opt.state_dict(),
    "critic_optimizer_state": agent.critic_opt.state_dict(),
    "alpha": float(agent.alpha),
    "alpha_optimizer_state": agent.alpha_opt.state_dict(),
    "global_step": trainer.global_step,
    "timestamp": datetime.now().isoformat(),
}
torch.save(checkpoint, "results/checkpoints/custom_name.pt")
```

### Checkpoint Management Best Practices

| Practice | Benefit |
|----------|---------|
| Keep only `best_model.pt` for production | Saves disk space (~50 MB each checkpoint) |
| Archive old checkpoints to external storage | Prevents disk overflow during long training |
| Compare multiple checkpoints via `mid_eval.py` | Identify overfitting: best_model vs later checkpoints |
| Document which checkpoint each plot comes from | Reproducibility & experiment tracking |

### Troubleshooting Checkpoints

| Issue | Solution |
|-------|----------|
| **File corrupted (unpickle error)** | Checkpoint partially written; training crashed. Delete & restart. |
| **Weights don't match model** | Architecture changed. Verify `LSTM_HIDDEN_SIZE` hasn't changed. |
| **Memory error loading checkpoint** | Checkpoint on GPU but device changed to CPU. Use `map_location="cpu"`. |

---

## Logging & Monitoring

### TensorBoard Monitoring

Real-time visualization of training curves:

```bash
# Start TensorBoard (during or after training)
tensorboard --logdir results/logs/ --port 6006

# View in browser
# http://localhost:6006
```

**Available plots in TensorBoard:**

| Metric | Description | Good Value |
|--------|-------------|-----------|
| `agent/actor_loss` | Policy gradient loss | Decreasing trend |
| `agent/critic_loss` | Q-function MSE loss | Decreasing trend |
| `agent/alpha` | Entropy temperature | ~0.1–0.2 |
| `train/reward` | Mean training reward | Increasing |
| `val/sharpe_ratio` | Validation Sharpe ratio | >0.8 |
| `val/total_return` | Validation cumulative return | Positive |
| `val/max_drawdown` | Validation worst-case loss | >-30% |

### Console Logging

The code prints progress to stdout:

```
2024-01-15 10:30:45  INFO     main: === Stage 1: Data ===
2024-01-15 10:30:46  INFO     main: === Stage 2: Feature Engineering ===
2024-01-15 10:30:48  INFO     main: === Stage 3: Normalisation ===
2024-01-15 10:30:48  INFO     main: Features (22): ['SMA_5', 'SMA_20', ...]
2024-01-15 10:30:50  INFO     main: === Stage 4: Sequence Creation ===
2024-01-15 10:30:51  INFO     main: === Stage 5: Environment Building ===
2024-01-15 10:30:52  INFO     training.train: Training started | Total steps: 500000
...
[Epoch  50000 / 500000] Reward: 0.045  Sharpe: 0.823  Actor Loss: -0.156
...
```

### Debugging with Logging Levels

Increase verbosity for debugging:

```python
# In config.py or main.py, set logging level:
import logging
logging.basicConfig(level=logging.DEBUG)  # More verbose
# logging.basicConfig(level=logging.WARNING)  # Less verbose
```

### Trade Tracking & Statistics

The environment tracks trade statistics (accessed during evaluation):

```python
# Access trade history
trade_stats = env.trade_tracker.get_statistics()
# Returns: {
#     "total_trades": 42,
#     "winning_trades": 25,
#     "losing_trades": 17,
#     "win_rate": 0.595,
#     "avg_win": 0.0234,  # per-trade return
#     "avg_loss": -0.0156,
#     "profit_factor": 1.85,
# }
```

---

## Troubleshooting

### Common Issues & Solutions

#### 1. **CUDA Out of Memory (OOM)**

**Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Causes:**
- Batch size too large for GPU VRAM
- Model too large for GPU memory
- Replay buffer too large

**Solutions:**

```python
# Option A: Reduce batch size (config.py)
BATCH_SIZE = 128  # Was 256

# Option B: Reduce replay buffer size
BUFFER_SIZE = 50_000  # Was 100_000

# Option C: Reduce model size
LSTM_HIDDEN_SIZE = 64  # Was 128
FC_HIDDEN_SIZES = [128, 64]  # Was [256, 128]

# Option D: Switch to CPU
DEVICE = "cpu"  # Train on CPU (slower but uses RAM instead)
```

Then retry: `python main.py`

#### 2. **Data Download Fails**

**Error:**
```
ConnectionError: yfinance could not download data for AAPL
```

**Causes:**
- No internet connection
- yfinance API temporary outage
- Stock ticker invalid

**Solutions:**

```bash
# Check internet
ping google.com

# Verify ticker
python -c "import yfinance; yfinance.Ticker('AAPL').history(period='1y")"

# Use pre-downloaded data
# Place raw_AAPL.csv in data/ manually, then:
python main.py
```

#### 3. **Shape Mismatch Error**

**Error:**
```
RuntimeError: Expected input of size (batch, 30, 25) but got (batch, 30, 24)
```

**Causes:**
- `WINDOW_SIZE` changed without retraining
- Feature computation changed
- Data split corrupted

**Solutions:**

```bash
# Delete cached data
rm -rf data/processed_*.npy

# Retrain from scratch
python main.py

# Or ensure config.py WINDOW_SIZE matches checkpoint
WINDOW_SIZE = 30  # Must match what checkpoint was trained with
```

#### 4. **Poor Training Performance**

**Symptoms:**
- Reward stuck near zero
- Sharpe ratio negative
- Model barely better than random

**Causes:**
- Reward weights misconfigured
- Learning rates too high/low
- Model capacity too small
- Insufficient training steps

**Solutions:**

```python
# Option A: Increase training duration
TOTAL_TIMESTEPS = 1_000_000  # Was 500k

# Option B: Tune learning rates
LEARNING_RATE_ACTOR = 1e-4  # Lower if diverging
# or
LEARNING_RATE_ACTOR = 1e-3  # Higher if converging slowly

# Option C: Balance reward components
REWARD_SHARPE_WEIGHT = 0.5  # Emphasize Sharpe more
REWARD_RETURN_WEIGHT = 0.3
REWARD_DRAWDOWN_WEIGHT = 0.2

# Option D: Increase model capacity
LSTM_HIDDEN_SIZE = 256  # Was 128
FC_HIDDEN_SIZES = [512, 256]  # Was [256, 128]
```

Then restart: `python main.py`

#### 5. **Diverging Loss (NaN)**

**Error:**
```
RuntimeError: invalid loss (NaN)
```

**Causes:**
- Learning rate too high
- Reward clipping insufficient
- Numerical instability

**Solutions:**

```python
# Reduce learning rates significantly
LEARNING_RATE_ACTOR = 1e-5    # 30x reduction
LEARNING_RATE_CRITIC = 1e-5
LEARNING_RATE_ALPHA = 1e-5

# Increase batch size (more stable gradient estimates)
BATCH_SIZE = 512  # Was 256

# Check reward bounds (should be clipped to [-10, 10])
# In src/reward.py, verify: return np.clip(reward, -10, 10)
```

#### 6. **Windows Line Ending Issues**

**Error (on Windows):**
```
bash: ./venv/Scripts/activate: No such file or directory
```

**Solution:**

```bash
# Use Windows activation script
venv\Scripts\activate

# Or convert line endings (Git setting)
git config --global core.autocrlf true
```

#### 7. **Import Errors After Installation**

**Error:**
```
ModuleNotFoundError: No module named 'stable_baselines3'
```

**Solution:**

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Or install specific package
pip install stable-baselines3==2.0.0
```

#### 8. **Checkpoint Loading Fails**

**Error:**
```
RuntimeError: expected scalar type Double but found Float
```

**Solution:**

```python
# Ensure checkpoint and model use same dtype
checkpoint = torch.load(..., map_location=torch.device('cpu'))
# Checkpoint loaded on CPU first, then move to GPU
agent.actor.to("cuda")
agent.critic.to("cuda")
```

---

## Examples

### Example 1: Full Training Pipeline (Default)

```bash
# Train from scratch with all defaults
python main.py

# Expected output:
# Stage 1: Data        ✓ Downloaded AAPL 2018-2023
# Stage 2: Features    ✓ Computed 22 indicators
# Stage 3: Normalise   ✓ Fitted scaler on train
# Stage 4: Sequences   ✓ Created 1470 sequences
# Stage 5: Environments ✓ Built 3 Gym environments
# Stage 6: Agent       ✓ Initialized SAC agent
# Training            ✓ 500k steps (15-30 min GPU, 2-4 hrs CPU)
# Evaluation           ✓ Test set backtesting
# Plots                ✓ Saved to results/plots/
```

### Example 2: Training Only (Skip Evaluation)

For faster iteration during development:

```python
# Modify main.py to comment out evaluation:
# backtest_and_plot(...)  # Comment this line
```

Then:
```bash
python main.py  # Trains quickly without evaluation step
```

### Example 3: Evaluate Existing Checkpoint

```bash
# During training (safe to run in parallel terminal)
python mid_eval.py --checkpoint results/checkpoints/step_50000.pt

# Expected output:
# Loading checkpoint...
# Building environment...
# Running evaluation...
# Metrics:
#   Total Return: +23.5%
#   Sharpe Ratio: 0.876
#   Max Drawdown: -18.3%
#   Win Rate: 58.2%
# Plots saved to: results/plots/mid_eval_step_50000/
```

### Example 4: Train on Different Stock

```python
# Edit config.py
TICKER = "MSFT"
START_DATE = "2019-01-01"
END_DATE = "2024-01-01"
```

```bash
python main.py
# Downloads MSFT data and trains
```

### Example 5: Quick Test Run (Small Model, Few Steps)

```python
# Edit config.py
TOTAL_TIMESTEPS = 5_000              # 100x fewer steps
EVAL_FREQUENCY = 1_000               # Less frequent validation
LSTM_HIDDEN_SIZE = 32                # Tiny model
FC_HIDDEN_SIZES = [64, 32]
BATCH_SIZE = 32
BUFFER_SIZE = 10_000
CHECKPOINT_FREQ = 2_500              # Save less often
```

```bash
python main.py
# Completes in ~30 seconds for testing/debugging
```

### Example 6: Train on CPU Only

```bash
# Either set environment variable
export DEVICE=cpu
python main.py

# Or edit config.py
DEVICE = "cpu"
```

### Example 7: Continuous Monitoring During Training

Terminal 1 (Training):
```bash
python main.py
```

Terminal 2 (Monitor TensorBoard):
```bash
tensorboard --logdir results/logs/
```

Terminal 3 (Periodic evaluation):
```bash
# Wait 2 minutes for training to start saving checkpoints
sleep 120
# Then evaluate periodically
while true; do
  python mid_eval.py --split val
  sleep 300  # Every 5 minutes
done
```

### Example 8: Compare Two Checkpoints

```bash
# Evaluate checkpoint A
python mid_eval.py --checkpoint results/checkpoints/step_100000.pt \
                  --plot-dir results/compare/step_100k/

# Evaluate checkpoint B
python mid_eval.py --checkpoint results/checkpoints/best_model.pt \
                  --plot-dir results/compare/best/

# Manually compare the equity curves in the generated PNG files
ls results/compare/*/portfolio_value.png
```

### Example 9: Production Inference (Use Trained Model)

```python
import torch
import numpy as np
from src.agent import SACAgent
from src.feature_engineering import FeatureEngineer
from src.environment import TradingEnvironment
import config

# Load trained agent
agent = SACAgent(
    obs_shape=(config.WINDOW_SIZE, 25),  # 25 = 22 indicators + 3 portfolio state
    action_dim=1,
    config=config,
    device="cuda"
)
checkpoint = torch.load("results/checkpoints/best_model.pt")
agent.actor.load_state_dict(checkpoint["actor_state_dict"])
agent.actor.eval()  # Evaluation mode

# Simulate live trading (example: one step)
# In practice, you'd update obs with new data
obs = np.random.randn(30, 25).astype(np.float32)  # (window, features)
with torch.no_grad():
    action = agent.select_action(torch.from_numpy(obs).unsqueeze(0), 
                                 deterministic=True)
position_fraction = float(action[0, 0])
print(f"Recommended position: {position_fraction:.2%}")
```

### Example 10: Hyperparameter Search (Manual)

```python
# Test 3 configurations
configs = [
    {"LEARNING_RATE_ACTOR": 1e-4, "BATCH_SIZE": 128},
    {"LEARNING_RATE_ACTOR": 3e-4, "BATCH_SIZE": 256},
    {"LEARNING_RATE_ACTOR": 1e-3, "BATCH_SIZE": 512},
]

for cfg in configs:
    # Update config.py with values
    for key, val in cfg.items():
        setattr(config, key, val)
    
    # Train
    python main.py
    
    # Results automatically saved to results/checkpoints/ & results/plots/
    # Compare Sharpe ratios manually
```

---

## Contributing & Citation

If you use this code for research or commercial applications, please cite:

```bibtex
@misc{lstm_sac_trader,
  author = {Your Name},
  title = {LSTM-SAC Trading Agent: Deep Reinforcement Learning for Quantitative Trading},
  year = {2024},
  url = {https://github.com/yourusername/lstm-sac-trader}
}
```

---

## License

This project is provided as-is for educational and research purposes. No warranty of any kind.

---

## References

**Key Papers:**
- Haarnoja, T., et al. (2018). "Soft Actor-Critic Algorithms and Applications." *ICML*.
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*.

**Useful Resources:**
- [OpenAI Gym Documentation](https://gym.openai.com/)
- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [TensorBoard User Guide](https://www.tensorflow.org/tensorboard/get_started)

---

## Support

For issues, questions, or suggestions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review relevant source code comments
3. Open an issue on GitHub (if applicable)
4. Submit a pull request with improvements

---

**Last Updated:** January 2024  
**Python Version:** 3.8+  
**PyTorch Version:** 2.0+  
**Status:** Active Development
