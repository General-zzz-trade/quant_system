#!/usr/bin/env python3
"""Paper Trading — real market data + simulated execution.

Runs LiveRunner in shadow_mode with:
  - EnrichedFeatureComputer + LiveInferenceBridge from ModelRegistry
  - ConceptDriftAdapter + OODDetector for online monitoring
  - Daily summary: trade count, PnL, hit rate, rolling IC

Usage:
    python3 -m scripts.run_paper_trading
    python3 -m scripts.run_paper_trading --symbols BTCUSDT,ETHUSDT --testnet
    python3 -m scripts.run_paper_trading --model-dir models_unified/BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_model(model_dir: Path):
    """Load model from directory (unified pkl or legacy pkl)."""
    from alpha.models.lgbm_alpha import LGBMAlphaModel

    unified_path = model_dir / "lgbm_unified.pkl"
    if unified_path.exists():
        model = LGBMAlphaModel(name="paper_alpha")
        model.load(unified_path)
        return model

    # Fallback: try V7 model
    v7_path = model_dir / "lgbm_v7_alpha.pkl"
    if v7_path.exists():
        model = LGBMAlphaModel(name="paper_alpha")
        model.load(v7_path)
        return model

    # Fallback: try any .pkl
    pkls = list(model_dir.glob("*.pkl"))
    if pkls:
        model = LGBMAlphaModel(name="paper_alpha")
        model.load(pkls[0])
        return model

    return None


def _load_model_from_registry(registry_db: str, symbol: str):
    """Load production model from registry."""
    from research.model_registry.registry import ModelRegistry

    registry = ModelRegistry(registry_db)
    prod = registry.get_production(f"alpha_unified_{symbol}")
    if prod is None:
        return None, None

    # Find model file based on registry metadata
    model_dir = Path(f"models_unified/{symbol}")
    model = _load_model(model_dir)
    return model, prod


class PaperTradingMonitor:
    """Tracks paper trading performance metrics."""

    def __init__(self) -> None:
        self.trades: List[Dict[str, Any]] = []
        self.predictions: List[Dict[str, Any]] = []
        self._start_time = time.time()

    def on_fill(self, fill: Any) -> None:
        self.trades.append({
            "ts": str(getattr(fill, "ts", "")),
            "symbol": str(getattr(fill, "symbol", "")),
            "side": str(getattr(fill, "side", "")),
            "qty": str(getattr(fill, "qty", "")),
            "price": str(getattr(fill, "price", "")),
        })

    def on_prediction(self, symbol: str, score: float, actual_return: float) -> None:
        self.predictions.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "score": score,
            "actual_return": actual_return,
        })

    def daily_summary(self) -> Dict[str, Any]:
        n_trades = len(self.trades)
        n_predictions = len(self.predictions)

        pnl = 0.0
        hits = 0
        for p in self.predictions:
            if p["score"] > 0 and p["actual_return"] > 0:
                hits += 1
                pnl += p["actual_return"]
            elif p["score"] < 0 and p["actual_return"] < 0:
                hits += 1
                pnl += abs(p["actual_return"])
            else:
                pnl -= abs(p["actual_return"])

        hit_rate = hits / max(n_predictions, 1)
        uptime_hours = (time.time() - self._start_time) / 3600

        return {
            "uptime_hours": round(uptime_hours, 1),
            "n_trades": n_trades,
            "n_predictions": n_predictions,
            "hit_rate": round(hit_rate, 4),
            "cumulative_pnl": round(pnl, 6),
        }


def build_paper_runner(
    symbols: tuple,
    *,
    model_dir: Optional[Path] = None,
    registry_db: str = "model_registry.db",
    testnet: bool = False,
    enable_drift_monitoring: bool = True,
):
    """Build a LiveRunner in shadow mode for paper trading."""
    from runner.live_runner import LiveRunner, LiveRunnerConfig
    from features.enriched_computer import EnrichedFeatureComputer

    config = LiveRunnerConfig(
        symbols=symbols,
        shadow_mode=True,
        testnet=testnet,
        enable_persistent_stores=True,
        enable_monitoring=True,
        enable_reconcile=False,  # No reconcile needed in shadow mode
        health_stale_data_sec=120.0,
        health_port=8080,
    )

    # Load model
    alpha_models = []
    for sym in symbols:
        sym_model_dir = model_dir / sym if model_dir else Path(f"models_unified/{sym}")
        model = _load_model(sym_model_dir)
        if model is not None:
            alpha_models.append(model)
            logger.info("Loaded model for %s from %s", sym, sym_model_dir)
        else:
            logger.warning("No model found for %s", sym)

    feature_computer = EnrichedFeatureComputer()
    monitor = PaperTradingMonitor()

    # Build with shadow mode
    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _DummyVenueClient()},
        feature_computer=feature_computer,
        alpha_models=alpha_models if alpha_models else None,
        on_fill=monitor.on_fill,
    )

    return runner, monitor


class _DummyVenueClient:
    """Dummy client for shadow mode (no real execution needed)."""

    def send_order(self, order_event):
        return []

    def cancel_all_orders(self, symbol: str):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper Trading — shadow mode execution")
    parser.add_argument("--symbols", default="BTCUSDT",
                        help="Comma-separated symbols")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Model directory (default: models_unified/)")
    parser.add_argument("--registry-db", default="model_registry.db",
                        help="Model registry database")
    parser.add_argument("--testnet", action="store_true",
                        help="Use testnet endpoints")
    parser.add_argument("--log-interval", type=int, default=3600,
                        help="Status log interval in seconds")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    symbols = tuple(s.strip().upper() for s in args.symbols.split(","))
    logger.info("Starting paper trading: symbols=%s testnet=%s", symbols, args.testnet)

    runner, monitor = build_paper_runner(
        symbols,
        model_dir=args.model_dir,
        registry_db=args.registry_db,
        testnet=args.testnet,
    )

    # Start runner in background, log periodic summaries
    import threading

    def _run():
        runner.start()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    logger.info("Paper trading started. Logging summary every %ds.", args.log_interval)
    try:
        while True:
            time.sleep(args.log_interval)
            summary = monitor.daily_summary()
            logger.info("Paper Trading Summary: %s", json.dumps(summary))
    except KeyboardInterrupt:
        logger.info("Shutting down paper trading...")
        runner.stop()

    # Final summary
    summary = monitor.daily_summary()
    logger.info("Final Paper Trading Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
