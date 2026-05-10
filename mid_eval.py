# =============================================================================
# mid_eval.py
# Evaluate any saved checkpoint independently — safe to run while training
# is ongoing in another terminal. Reads only from disk, never touches the
# training process or its memory.
# Usage:
#   python mid_eval.py --checkpoint results/checkpoints/step_50000.pt
#   python mid_eval.py --checkpoint results/checkpoints/best_model.pt
#   python mid_eval.py   # auto-picks the latest checkpoint
# =============================================================================

import os
import sys
import glob
import argparse
import logging

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_pipeline       import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.environment         import TradingEnvironment
from src.agent               import SACAgent
from src.utils               import set_seed
from training.evaluate       import backtest_and_plot, evaluate

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("mid_eval")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Mid-training checkpoint evaluator")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pt checkpoint. If omitted, auto-selects the latest one."
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"],
        help="Which data split to evaluate on (default: test)."
    )
    parser.add_argument(
        "--plot-dir", type=str, default=None,
        help="Where to save plots. Defaults to results/plots/mid_eval_<step>/"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint auto-detection
# ---------------------------------------------------------------------------

def find_latest_checkpoint() -> str:
    """Return the path of the most recently modified .pt file in CHECKPOINT_DIR."""
    pattern = os.path.join(config.CHECKPOINT_DIR, "*.pt")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No checkpoints found in {config.CHECKPOINT_DIR}. "
            "Has training produced any checkpoints yet?"
        )
    # Sort by modification time — most recent last
    files.sort(key=os.path.getmtime)
    return files[-1]


def checkpoint_label(path: str) -> str:
    """Extract a short label from the checkpoint filename for plot sub-directory."""
    name = os.path.splitext(os.path.basename(path))[0]  # e.g. "step_50000"
    return name


# ---------------------------------------------------------------------------
# Data + environment (same pipeline as main.py, results are identical)
# ---------------------------------------------------------------------------

def build_environment(split: str) -> TradingEnvironment:
    logger.info("Building data pipeline …")
    pipeline = DataPipeline(data_dir=config.DATA_DIR)
    raw_df   = pipeline.load_or_download(
        ticker   = config.TICKER,
        start    = config.START_DATE,
        end      = config.END_DATE,
        interval = config.DATA_INTERVAL,
    )
    train_df, val_df, test_df = pipeline.split_data(
        raw_df,
        train_ratio = config.TRAIN_RATIO,
        val_ratio   = config.VAL_RATIO,
        test_ratio  = config.TEST_RATIO,
    )

    fe = FeatureEngineer()
    train_feat = fe.compute_indicators(train_df)
    val_feat   = fe.compute_indicators(val_df)
    test_feat  = fe.compute_indicators(test_df)

    train_scaled, val_scaled, test_scaled, _ = fe.normalize_features(
        train_feat, val_feat, test_feat
    )

    split_map = {"train": train_scaled, "val": val_scaled, "test": test_scaled}
    chosen_df = split_map[split]

    X, prices, dates = fe.create_sequences(chosen_df, config.WINDOW_SIZE)

    env = TradingEnvironment(X, prices, dates, config, mode=split)
    logger.info(
        "Environment ready — split=%s  steps=%d  obs_shape=%s",
        split, len(X), env.observation_space.shape,
    )
    return env


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(config.SEED)

    # ---- Resolve checkpoint path ----------------------------------------
    ckpt_path = args.checkpoint or find_latest_checkpoint()
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    label    = checkpoint_label(ckpt_path)
    plot_dir = args.plot_dir or os.path.join(config.PLOT_DIR, f"mid_eval_{label}")
    os.makedirs(plot_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Checkpoint : %s", ckpt_path)
    logger.info("  Split      : %s", args.split)
    logger.info("  Plot dir   : %s", plot_dir)
    logger.info("=" * 60)

    # ---- Build environment ----------------------------------------------
    env = build_environment(args.split)
    obs_shape = env.observation_space.shape

    # ---- Load agent from checkpoint -------------------------------------
    # Build a fresh agent then load weights — training process is untouched
    agent = SACAgent(
        obs_shape  = obs_shape,
        action_dim = 1,
        config     = config,
        device     = config.DEVICE,
    )
    agent.load(ckpt_path)
    logger.info("Agent loaded from checkpoint.")

    # ---- Quick metrics print (no plots) ---------------------------------
    logger.info("Running deterministic evaluation …")
    metrics, trajectory = evaluate(agent, env, n_episodes=1, deterministic=True)

    print("\n" + "=" * 55)
    print(f"  MID-TRAINING EVAL  —  {label}")
    print("=" * 55)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<35} {v:>10.4f}")
        else:
            print(f"  {k:<35} {v:>10}")
    print("=" * 55 + "\n")

    # ---- Full backtest + 5 plots ----------------------------------------
    logger.info("Generating plots in %s …", plot_dir)

    # Re-build env (evaluate() advances the internal step counter)
    env2 = build_environment(args.split)
    backtest_and_plot(agent, env2, plot_dir)

    logger.info("Done. All plots saved to %s/", plot_dir)


if __name__ == "__main__":
    main()