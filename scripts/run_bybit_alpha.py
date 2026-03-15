#!/usr/bin/env python3
"""Run alpha strategy on Bybit demo trading.

Connects to Bybit demo API, fetches 1h klines, computes features via
RustFeatureEngine, runs LightGBM inference, and trades BTCUSDT perpetual.

Usage:
    python3 -m scripts.run_bybit_alpha                    # live demo trading
    python3 -m scripts.run_bybit_alpha --dry-run          # signal only, no orders
    python3 -m scripts.run_bybit_alpha --once              # single bar then exit
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────

MODEL_BASE = Path("models_v8")
INTERVAL = "60"  # Bybit: "60" = 1h
WARMUP_BARS = 200
POLL_INTERVAL = 60  # seconds between checks

# Default symbols + position sizes
SYMBOL_CONFIG = {
    "BTCUSDT": {"size": 0.001, "model_dir": "BTCUSDT_gate_v2"},
    "ETHUSDT": {"size": 0.01, "model_dir": "ETHUSDT_gate_v2"},
}


def load_model(model_dir: Path) -> dict:
    """Load model config + LightGBM model from disk.

    Uses pickle for LightGBM model files — these are trusted local
    artifacts produced by our own training pipeline.
    """
    import pickle  # noqa: S403 — trusted local model files only

    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    primary = config["horizon_models"][0]
    lgbm_path = model_dir / primary["lgbm"]
    with open(lgbm_path, "rb") as f:
        raw = pickle.load(f)  # noqa: S301 — trusted local artifact

    # Model is stored as {"model": Booster, "features": [...]}
    model = raw["model"] if isinstance(raw, dict) else raw

    return {
        "config": config,
        "model": model,
        "features": primary["features"],
        "deadzone": config.get("deadzone", 2.0),
        "min_hold": config.get("min_hold", 12),
        "max_hold": config.get("max_hold", 96),
        "zscore_window": config.get("zscore_window", 720),
        "zscore_warmup": config.get("zscore_warmup", 180),
    }


def create_adapter():
    """Create Bybit demo adapter from env vars or defaults."""
    from execution.adapters.bybit import BybitAdapter, BybitConfig

    api_key = os.environ.get("BYBIT_API_KEY", "ODwzdOy3bgfi6Hjqp9")
    api_secret = os.environ.get("BYBIT_API_SECRET",
                                "SYNHdt42n7jcOzT3vFnTPlgGokvRdajs9pAU")
    base_url = os.environ.get("BYBIT_BASE_URL", "https://api-demo.bybit.com")

    config = BybitConfig(api_key=api_key, api_secret=api_secret,
                         base_url=base_url)
    adapter = BybitAdapter(config)
    if not adapter.connect():
        raise RuntimeError("Failed to connect to Bybit")
    return adapter


class AlphaRunner:
    """Runs alpha strategy on Bybit with RustFeatureEngine + LightGBM."""

    def __init__(self, adapter: Any, model_info: dict, symbol: str,
                 dry_run: bool = False, position_size: float = 0.001):
        self._adapter = adapter
        self._symbol = symbol
        self._model = model_info["model"]
        self._features = model_info["features"]
        self._config = model_info["config"]
        self._deadzone = model_info["deadzone"]
        self._min_hold = model_info["min_hold"]
        self._max_hold = model_info["max_hold"]
        self._zscore_window = model_info["zscore_window"]
        self._zscore_warmup = model_info["zscore_warmup"]
        self._dry_run = dry_run
        self._position_size = position_size

        self._predictions: list[float] = []
        self._current_signal = 0
        self._hold_count = 0
        self._bars_processed = 0

        from _quant_hotpath import RustFeatureEngine
        self._engine = RustFeatureEngine()

    def warmup(self) -> int:
        """Fetch historical bars and warm up feature engine."""
        bars = self._adapter.get_klines(self._symbol, interval=INTERVAL,
                                        limit=WARMUP_BARS)
        bars.reverse()  # Bybit returns newest first

        for bar in bars:
            self._engine.push_bar(
                bar["close"], bar["volume"], bar["high"], bar["low"],
                bar["open"], funding_rate=float("nan"),
            )
            features = self._engine.get_features()
            if features:
                feat_dict = dict(features)
                x = [feat_dict.get(f, 0.0) or 0.0 for f in self._features]
                if not any(np.isnan(x)):
                    pred = self._model.predict([x])[0]
                    self._predictions.append(float(pred))

        self._bars_processed = len(bars)
        logger.info("Warmup: %d bars, %d predictions", len(bars), len(self._predictions))
        return len(bars)

    def process_bar(self, bar: dict) -> dict:
        """Process one bar: features → predict → signal → trade."""
        self._bars_processed += 1

        self._engine.push_bar(
            bar["close"], bar["volume"], bar["high"], bar["low"],
            bar["open"], funding_rate=float("nan"),
        )

        features = self._engine.get_features()
        if not features:
            return {"action": "no_features", "bar": self._bars_processed}

        feat_dict = dict(features)
        x = [feat_dict.get(f, 0.0) or 0.0 for f in self._features]

        if any(np.isnan(x)):
            return {"action": "nan_features", "bar": self._bars_processed}

        pred = float(self._model.predict([x])[0])
        self._predictions.append(pred)

        if len(self._predictions) < self._zscore_warmup:
            return {"action": "warmup", "bar": self._bars_processed,
                    "pred": pred, "buffer": len(self._predictions)}

        window = self._predictions[-self._zscore_window:]
        mean = np.mean(window)
        std = np.std(window)
        z = (pred - mean) / std if std > 1e-12 else 0.0

        prev_signal = self._current_signal
        desired = 0
        if z > self._deadzone:
            desired = 1
        elif z < -self._deadzone:
            desired = -1

        if self._hold_count < self._min_hold:
            new_signal = prev_signal
            self._hold_count += 1
        elif desired != prev_signal:
            new_signal = desired
            self._hold_count = 1
        else:
            new_signal = desired
            self._hold_count += 1

        if self._hold_count >= self._max_hold and new_signal != 0:
            new_signal = 0
            self._hold_count = 1

        self._current_signal = new_signal

        result = {
            "action": "signal", "bar": self._bars_processed,
            "pred": round(pred, 6), "z": round(z, 4),
            "signal": new_signal, "prev_signal": prev_signal,
            "hold_count": self._hold_count, "close": bar["close"],
        }

        if new_signal != prev_signal:
            result["trade"] = self._execute_signal_change(prev_signal, new_signal, bar["close"])

        return result

    def _execute_signal_change(self, prev: int, new: int, price: float) -> dict:
        if self._dry_run:
            return {"action": "dry_run", "from": prev, "to": new}

        if prev != 0:
            self._adapter.close_position(self._symbol)
            logger.info("Closed position")

        if new != 0:
            side = "buy" if new == 1 else "sell"
            result = self._adapter.send_market_order(self._symbol, side, self._position_size)
            logger.info("Opened %s %.4f BTC @ ~$%.1f: %s", side, self._position_size, price, result)
            return {"side": side, "qty": self._position_size, "result": result}

        return {"action": "flat"}


def main():
    parser = argparse.ArgumentParser(description="Bybit alpha strategy runner")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Symbols to trade (default: BTCUSDT ETHUSDT)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    adapter = create_adapter()
    bal = adapter.get_balances()
    usdt = bal.get("USDT")
    logger.info("USDT balance: %s", usdt.total if usdt else "?")

    # Build per-symbol runners
    runners: dict[str, AlphaRunner] = {}
    last_bar_times: dict[str, int] = {}

    for symbol in args.symbols:
        sym_cfg = SYMBOL_CONFIG.get(symbol, {"size": 0.001, "model_dir": f"{symbol}_gate_v2"})
        model_dir = MODEL_BASE / sym_cfg["model_dir"]
        if not (model_dir / "config.json").exists():
            logger.warning("No model for %s at %s, skipping", symbol, model_dir)
            continue

        model_info = load_model(model_dir)
        logger.info(
            "%s: model v%s, %d features, dz=%.1f, hold=%d-%d, size=%.4f",
            symbol, model_info["config"]["version"],
            len(model_info["features"]), model_info["deadzone"],
            model_info["min_hold"], model_info["max_hold"], sym_cfg["size"],
        )

        runner = AlphaRunner(
            adapter=adapter, model_info=model_info, symbol=symbol,
            dry_run=args.dry_run, position_size=sym_cfg["size"],
        )
        logger.info("%s: warming up %d bars...", symbol, WARMUP_BARS)
        runner.warmup()
        runners[symbol] = runner
        last_bar_times[symbol] = 0

    if not runners:
        logger.error("No symbols with models. Exiting.")
        return

    if args.once:
        for symbol, runner in runners.items():
            bars = adapter.get_klines(symbol, interval=INTERVAL, limit=2)
            if bars:
                result = runner.process_bar(bars[0])
                logger.info("%s: %s", symbol, json.dumps(result, default=str))
        return

    logger.info(
        "Starting multi-symbol alpha: %s, poll=%ds, dry=%s",
        list(runners.keys()), POLL_INTERVAL, args.dry_run,
    )

    heartbeat_interval = 300  # log heartbeat every 5 minutes
    last_heartbeat = time.time()
    cycle_count = 0

    try:
        while True:
            cycle_count += 1
            for symbol, runner in runners.items():
                try:
                    bars = adapter.get_klines(symbol, interval=INTERVAL, limit=2)
                    if not bars:
                        continue

                    latest = bars[0]
                    bar_time = latest["time"]

                    if bar_time > last_bar_times[symbol]:
                        last_bar_times[symbol] = bar_time
                        result = runner.process_bar(latest)
                        if result.get("action") == "signal":
                            logger.info(
                                "%s bar %d: $%.1f z=%+.3f sig=%d hold=%d%s",
                                symbol, result["bar"], result["close"],
                                result["z"], result["signal"],
                                result["hold_count"],
                                f" TRADE={result['trade']}" if "trade" in result else "",
                            )
                except Exception:
                    logger.exception("Error processing %s", symbol)

            # Heartbeat: log status every 5 minutes
            now = time.time()
            if now - last_heartbeat >= heartbeat_interval:
                last_heartbeat = now
                sigs = {s: r._current_signal for s, r in runners.items()}
                holds = {s: r._hold_count for s, r in runners.items()}
                logger.info(
                    "HEARTBEAT cycle=%d signals=%s holds=%s",
                    cycle_count, sigs, holds,
                )

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Stopped")
        if not args.dry_run:
            for symbol, runner in runners.items():
                if runner._current_signal != 0:
                    logger.info("Closing %s position on exit...", symbol)
                    adapter.close_position(symbol)


if __name__ == "__main__":
    main()
