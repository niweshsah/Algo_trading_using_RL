# =============================================================================
# main.py
# Entry point: data → features → environments → agent → train → backtest.
# =============================================================================

import os
import sys
import logging

# Ensure the project root is on PYTHONPATH when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_pipeline       import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.environment         import TradingEnvironment
from src.agent               import SACAgent
from src.utils               import set_seed
from training.train          import Trainer
from training.evaluate       import backtest_and_plot

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ----------------------------------------------------------------
    # 0. Reproducibility
    # ----------------------------------------------------------------
    set_seed(config.SEED)

    # ----------------------------------------------------------------
    # 1. Download / load raw OHLCV data
    # ----------------------------------------------------------------
    logger.info("=== Stage 1: Data ===")
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

    # ----------------------------------------------------------------
    # 2. Feature engineering (compute indicators)
    # ----------------------------------------------------------------
    logger.info("=== Stage 2: Feature Engineering ===")
    fe = FeatureEngineer()

    train_feat = fe.compute_indicators(train_df)
    val_feat   = fe.compute_indicators(val_df)
    test_feat  = fe.compute_indicators(test_df)

    logger.info(
        "Indicator shapes — train: %s  val: %s  test: %s",
        train_feat.shape, val_feat.shape, test_feat.shape,
    )

    # ----------------------------------------------------------------
    # 3. Normalise — fit ONLY on train (no data leakage)
    # ----------------------------------------------------------------
    logger.info("=== Stage 3: Normalisation ===")
    train_scaled, val_scaled, test_scaled, scaler = fe.normalize_features(
        train_feat, val_feat, test_feat
    )
    feature_names = fe.get_feature_names()
    logger.info("Features (%d): %s", len(feature_names), feature_names)

    # ----------------------------------------------------------------
    # 4. Create sliding-window sequences
    # ----------------------------------------------------------------
    logger.info("=== Stage 4: Sequence Creation (window=%d) ===", config.WINDOW_SIZE)
    X_train, p_train, d_train = fe.create_sequences(train_scaled, config.WINDOW_SIZE)
    X_val,   p_val,   d_val   = fe.create_sequences(val_scaled,   config.WINDOW_SIZE)
    X_test,  p_test,  d_test  = fe.create_sequences(test_scaled,  config.WINDOW_SIZE)

    logger.info(
        "Sequence shapes — train: %s  val: %s  test: %s",
        X_train.shape, X_val.shape, X_test.shape,
    )

    # ----------------------------------------------------------------
    # 5. Build environments
    # ----------------------------------------------------------------
    logger.info("=== Stage 5: Environments ===")
    train_env = TradingEnvironment(X_train, p_train, d_train, config, mode="train")
    val_env   = TradingEnvironment(X_val,   p_val,   d_val,   config, mode="val")
    test_env  = TradingEnvironment(X_test,  p_test,  d_test,  config, mode="test")

    obs_shape = train_env.observation_space.shape
    logger.info("Observation space shape: %s", obs_shape)

    # Sanity check on environment
    obs = train_env.reset()
    assert obs.shape == obs_shape, f"obs shape mismatch: {obs.shape} vs {obs_shape}"
    sample_action = train_env.action_space.sample()
    obs2, reward, done, truncated, info = train_env.step(sample_action)
    assert 0.0 <= info["position_fraction"] <= 1.0, "position_fraction out of bounds"
    logger.info("Environment sanity check passed.")
    train_env.reset()   # reset before training

    # ----------------------------------------------------------------
    # 6. Build SAC agent
    # ----------------------------------------------------------------
    logger.info("=== Stage 6: Agent ===")
    agent = SACAgent(
        obs_shape  = obs_shape,
        action_dim = 1,
        config     = config,
        device     = config.DEVICE,
    )

    # Verify update mechanics with one synthetic batch
    import numpy as np
    for _ in range(config.BATCH_SIZE + 10):
        dummy_obs  = np.random.randn(*obs_shape).astype("float32")
        dummy_nobs = np.random.randn(*obs_shape).astype("float32")
        agent.buffer.add(dummy_obs, dummy_nobs, np.array([0.5]), 0.0, False)
    losses = agent.update(config.BATCH_SIZE)
    assert "critic_loss" in losses, "Missing critic_loss in update output"
    assert "actor_loss"  in losses, "Missing actor_loss in update output"
    assert not np.isnan(losses["critic_loss"]), "critic_loss is NaN"
    logger.info("Agent update sanity check passed: %s", losses)

    # Re-initialise buffer (clear dummy data)
    from src.agent import ReplayBuffer
    agent.buffer = ReplayBuffer(
        capacity   = config.BUFFER_SIZE,
        obs_shape  = obs_shape,
        action_dim = 1,
        device     = agent.device,
    )

    # ----------------------------------------------------------------
    # 7. Train
    # ----------------------------------------------------------------
    logger.info("=== Stage 7: Training ===")
    trainer = Trainer(agent, train_env, val_env, config)
    trainer.train()

    # ----------------------------------------------------------------
    # 8. Load best model and run backtest on test set
    # ----------------------------------------------------------------
    logger.info("=== Stage 8: Backtest ===")
    best_ckpt = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    if os.path.exists(best_ckpt):
        agent.load(best_ckpt)
        logger.info("Loaded best model from %s", best_ckpt)
    else:
        logger.warning(
            "No best_model.pt found in %s — using final model weights.",
            config.CHECKPOINT_DIR,
        )

    backtest_and_plot(agent, test_env, config.PLOT_DIR)

    logger.info("All done. Plots saved to %s/", config.PLOT_DIR)


if __name__ == "__main__":
    main()
