#!/usr/bin/env python3
"""Retrain daemon — automated model retraining with drift detection.

Combines:
  - RetrainTrigger (time-based + degradation-based)
  - train_unified pipeline
  - ModelRegistry registration + promotion
  - ConceptDriftAdapter integration

Dual trigger:
  - Weekly scheduled retrain
  - Drift-triggered retrain (ConceptDriftAdapter.recommendation == "retrain")

Promotion requires:
  - OOS IC > 0
  - H2 IC > 0
  - Deflated Sharpe > 0
  - Better than current production

Usage:
    python3 -m scripts.retrain_daemon
    python3 -m scripts.retrain_daemon --check-interval 3600 --symbols BTCUSDT
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def retrain_symbol(
    symbol: str,
    out_base: Path,
    registry_db: str,
    n_trials: int = 20,
) -> Optional[Dict[str, Any]]:
    """Run unified training for one symbol and register results."""
    from scripts.train_unified import run_one
    from research.model_registry.registry import ModelRegistry

    registry = ModelRegistry(registry_db)

    result = run_one(
        symbol, out_base,
        top_k=25,
        horizon=0,  # auto-select
        target_mode="",  # auto-select
        n_folds=5,
        n_trials=n_trials,
        use_icir_select=True,
        regime_split=False,
        registry=registry,
    )

    return result


def check_drift_trigger(symbol: str) -> bool:
    """Check if drift adapter recommends retraining.

    Returns True if drift state recommends "retrain".
    In production, this would read from a running drift adapter.
    Here we check persisted drift state if available.
    """
    drift_path = Path(f"data/live/drift_state_{symbol}.json")
    if not drift_path.exists():
        return False

    try:
        with open(drift_path) as f:
            state = json.load(f)
        return state.get("recommendation") == "retrain"
    except Exception:
        return False


def check_ood_trigger(symbol: str, threshold: float = 0.1) -> bool:
    """Check if OOD rate exceeds threshold."""
    ood_path = Path(f"data/live/ood_state_{symbol}.json")
    if not ood_path.exists():
        return False

    try:
        with open(ood_path) as f:
            state = json.load(f)
        return state.get("ood_rate", 0) > threshold
    except Exception:
        return False


class RetrainDaemon:
    """Daemon that monitors triggers and runs retraining."""

    def __init__(
        self,
        symbols: tuple,
        *,
        check_interval_sec: float = 3600.0,
        weekly_retrain: bool = True,
        drift_retrain: bool = True,
        out_base: Path = Path("models_unified"),
        registry_db: str = "model_registry.db",
        n_trials: int = 20,
    ) -> None:
        self._symbols = symbols
        self._check_interval = check_interval_sec
        self._weekly_retrain = weekly_retrain
        self._drift_retrain = drift_retrain
        self._out_base = out_base
        self._registry_db = registry_db
        self._n_trials = n_trials
        self._last_retrain: Dict[str, datetime] = {}
        self._running = False

    def _should_weekly_retrain(self, symbol: str) -> bool:
        """Check if 7 days have passed since last retrain."""
        last = self._last_retrain.get(symbol)
        if last is None:
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed > 7 * 86400

    def _run_retrain(self, symbol: str, reason: str) -> Optional[Dict[str, Any]]:
        """Execute retraining for one symbol."""
        logger.info("Starting retrain for %s (reason: %s)", symbol, reason)
        start = time.time()

        try:
            result = retrain_symbol(
                symbol, self._out_base,
                self._registry_db, self._n_trials,
            )

            duration = time.time() - start
            if result:
                promoted = result.get("registry", {}).get("promoted", False)
                logger.info(
                    "Retrain complete for %s in %.0fs: promoted=%s",
                    symbol, duration, promoted,
                )
                self._last_retrain[symbol] = datetime.now(timezone.utc)

                # Save retrain record
                record = {
                    "symbol": symbol,
                    "reason": reason,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_sec": duration,
                    "promoted": promoted,
                    "cv_ic": result.get("cv_ic"),
                    "oos_ic": result.get("oos_extended", {}).get("overall", {}).get("ic"),
                }
                record_path = self._out_base / symbol / "retrain_history.jsonl"
                record_path.parent.mkdir(parents=True, exist_ok=True)
                with open(record_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

            return result

        except Exception:
            logger.exception("Retrain failed for %s", symbol)
            # Record last retrain time even on failure to prevent infinite retry loop
            self._last_retrain[symbol] = datetime.now(timezone.utc)
            return None

    def check_and_retrain(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """Check all triggers and retrain as needed."""
        results = {}

        for symbol in self._symbols:
            reason = None

            # Check drift trigger
            if self._drift_retrain and check_drift_trigger(symbol):
                reason = "drift_detected"

            # Check OOD trigger
            elif self._drift_retrain and check_ood_trigger(symbol):
                reason = "ood_rate_exceeded"

            # Check weekly schedule
            elif self._weekly_retrain and self._should_weekly_retrain(symbol):
                reason = "weekly_schedule"

            if reason:
                results[symbol] = self._run_retrain(symbol, reason)
            else:
                results[symbol] = None

        return results

    def run(self) -> None:
        """Run the daemon loop."""
        self._running = True
        logger.info("RetrainDaemon started: symbols=%s interval=%ss",
                     self._symbols, self._check_interval)

        while self._running:
            try:
                self.check_and_retrain()
            except Exception:
                logger.exception("RetrainDaemon check failed")

            # Wait for next check
            for _ in range(int(self._check_interval)):
                if not self._running:
                    break
                time.sleep(1.0)

    def stop(self) -> None:
        self._running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Model retrain daemon")
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT",
                        help="Comma-separated symbols")
    parser.add_argument("--check-interval", type=float, default=3600.0,
                        help="Check interval in seconds")
    parser.add_argument("--no-weekly", action="store_true",
                        help="Disable weekly retrain")
    parser.add_argument("--no-drift", action="store_true",
                        help="Disable drift-triggered retrain")
    parser.add_argument("--out", default="models_unified",
                        help="Output directory")
    parser.add_argument("--registry-db", default="model_registry.db",
                        help="Registry database path")
    parser.add_argument("--n-trials", type=int, default=20,
                        help="Optuna trials per fold")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    symbols = tuple(s.strip().upper() for s in args.symbols.split(","))

    daemon = RetrainDaemon(
        symbols,
        check_interval_sec=args.check_interval,
        weekly_retrain=not args.no_weekly,
        drift_retrain=not args.no_drift,
        out_base=Path(args.out),
        registry_db=args.registry_db,
        n_trials=args.n_trials,
    )

    if args.once:
        daemon.check_and_retrain()
    else:
        try:
            daemon.run()
        except KeyboardInterrupt:
            daemon.stop()


if __name__ == "__main__":
    main()
