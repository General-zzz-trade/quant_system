"""BTC P0 Alpha — Live Paper Trading.

Assembles: EnrichedFeatureComputer + LGBM model + FundingPoller + LivePaperRunner.

Usage:
    python3 -m scripts.run_btc_paper
    python3 -m scripts.run_btc_paper --balance 10000 --threshold 0.001 --testnet
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description="BTC P0 Alpha — Live Paper Trading")
    p.add_argument("--balance", type=float, default=10000.0, help="Starting balance (USDT)")
    p.add_argument("--threshold", type=float, default=0.001, help="ML score threshold")
    p.add_argument("--risk-pct", type=float, default=0.5, help="Equity %% per position")
    p.add_argument("--testnet", action="store_true", help="Use Binance testnet")
    p.add_argument("--interval", default="1h", help="Kline interval (default: 1h)")
    p.add_argument("--log-level", default="INFO", help="Log level")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    model_path = Path("models/BTCUSDT/lgbm_alpha_final.pkl")
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    # 1. Load model
    from alpha.models.lgbm_alpha import LGBMAlphaModel

    model = LGBMAlphaModel(name="lgbm_alpha")
    model.load(model_path)
    logger.info("Loaded model from %s (%d features)", model_path, len(model.feature_names))

    # 2. Feature computer
    from features.enriched_computer import EnrichedFeatureComputer

    computer = EnrichedFeatureComputer()

    # 3. Funding poller
    from execution.adapters.binance.funding_poller import BinanceFundingPoller

    poller = BinanceFundingPoller(symbol="BTCUSDT", testnet=args.testnet)
    poller.start()

    # 4. Decision module
    from decision.ml_decision import MLDecisionModule

    decision = MLDecisionModule(
        symbol="BTCUSDT", risk_pct=args.risk_pct, threshold=args.threshold,
    )

    # 5. Build runner
    from runner.live_paper_runner import LivePaperConfig, LivePaperRunner

    config = LivePaperConfig(
        symbols=("BTCUSDT",),
        starting_balance=args.balance,
        kline_interval=args.interval,
        testnet=args.testnet,
    )
    runner = LivePaperRunner.build(
        config,
        decision_modules=[decision],
        feature_computer=computer,
        alpha_models=[model],
        funding_rate_source=poller.get_rate,
    )

    logger.info(
        "Starting BTC paper trading (balance=%.0f, threshold=%.4f, testnet=%s)",
        args.balance, args.threshold, args.testnet,
    )
    try:
        runner.start()
    finally:
        poller.stop()


if __name__ == "__main__":
    main()
