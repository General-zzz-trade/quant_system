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
    """Load model config + all horizon models for IC-weighted ensemble.

    Uses pickle for LightGBM/XGBoost model files — trusted local
    artifacts produced by our own training pipeline.
    """
    import pickle  # noqa: S403 — trusted local model files only

    config_path = model_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    # Load ALL horizon models for ensemble
    horizon_models = []
    for hm in config.get("horizon_models", []):
        lgbm_path = model_dir / hm["lgbm"]
        if not lgbm_path.exists():
            continue
        with open(lgbm_path, "rb") as f:
            raw = pickle.load(f)  # noqa: S301 — trusted local artifact
        model = raw["model"] if isinstance(raw, dict) else raw

        # Also load XGBoost if available
        xgb_model = None
        xgb_path = model_dir / hm.get("xgb", "")
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                xgb_raw = pickle.load(f)  # noqa: S301
            xgb_model = xgb_raw["model"] if isinstance(xgb_raw, dict) else xgb_raw

        horizon_models.append({
            "horizon": hm["horizon"],
            "lgbm": model,
            "xgb": xgb_model,
            "features": hm["features"],
            "ic": hm.get("ic", 0.01),
        })

    if not horizon_models:
        raise RuntimeError(f"No models found in {model_dir}")

    # Primary model = first horizon (for feature list compatibility)
    primary = horizon_models[0]

    return {
        "config": config,
        "model": primary["lgbm"],  # backward compat
        "features": primary["features"],
        "horizon_models": horizon_models,
        "lgbm_xgb_weight": config.get("lgbm_xgb_weight", 0.5),
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
                 dry_run: bool = False, position_size: float = 0.001,
                 adaptive_sizing: bool = True, risk_per_trade: float = 0.05,
                 min_size: float = 0.01, max_size_pct: float = 0.30):
        self._adapter = adapter
        self._symbol = symbol
        self._model = model_info["model"]  # primary (backward compat)
        self._features = model_info["features"]
        self._horizon_models = model_info.get("horizon_models", [])
        self._lgbm_xgb_weight = model_info.get("lgbm_xgb_weight", 0.5)
        self._config = model_info["config"]
        self._deadzone_base = model_info["deadzone"]
        self._deadzone = model_info["deadzone"]  # current (adapted)
        self._vol_median = 0.0063  # ETH 1h median vol (calibrated from history)
        self._min_hold = model_info["min_hold"]
        self._max_hold = model_info["max_hold"]
        self._zscore_window = model_info["zscore_window"]
        self._zscore_warmup = model_info["zscore_warmup"]
        self._dry_run = dry_run
        self._base_position_size = position_size
        self._position_size = position_size

        # Adaptive position sizing
        self._adaptive_sizing = adaptive_sizing
        self._risk_per_trade = risk_per_trade  # fraction of equity to risk per trade
        self._min_size = min_size              # minimum position (exchange lot size)
        self._max_size_pct = max_size_pct      # max position as % of equity

        self._predictions: list[float] = []
        self._current_signal = 0
        self._hold_count = 0
        self._bars_processed = 0

        # Regime filter state
        self._closes: list[float] = []
        self._rets: list[float] = []
        self._regime_active = True
        self._vol_threshold = 0.004   # 20-bar ret stdev
        self._trend_threshold = 0.04  # |close/MA480 - 1|
        self._ma_window = 480

        # P&L tracking + drawdown circuit breaker
        self._entry_price: float = 0.0
        self._total_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0
        self._max_drawdown_pct: float = 15.0  # kill at 15% drawdown
        self._killed: bool = False

        from _quant_hotpath import RustFeatureEngine
        self._engine = RustFeatureEngine()

    def warmup(self) -> int:
        """Fetch historical bars and warm up feature engine."""
        bars = self._adapter.get_klines(self._symbol, interval=INTERVAL,
                                        limit=WARMUP_BARS)
        bars.reverse()  # Bybit returns newest first

        for bar in bars:
            self._check_regime(bar["close"])  # build regime state
            self._engine.push_bar(
                bar["close"], bar["volume"], bar["high"], bar["low"],
                bar["open"], funding_rate=float("nan"),
            )
            features = self._engine.get_features()
            if features:
                feat_dict = dict(features)
                pred = self._ensemble_predict(feat_dict)
                if pred is not None:
                    self._predictions.append(pred)

        self._bars_processed = len(bars)
        regime_str = "active" if self._regime_active else "filtered"
        logger.info("Warmup: %d bars, %d predictions, regime=%s",
                     len(bars), len(self._predictions), regime_str)
        return len(bars)

    def _ensemble_predict(self, feat_dict: dict) -> float | None:
        """IC-weighted ensemble across horizons, LightGBM + XGBoost blend."""
        if not self._horizon_models:
            # Fallback: single model
            x = [feat_dict.get(f, 0.0) or 0.0 for f in self._features]
            if any(np.isnan(x)):
                return None
            return float(self._model.predict([x])[0])

        weighted_sum = 0.0
        weight_total = 0.0

        for hm in self._horizon_models:
            feats = hm["features"]
            x = [feat_dict.get(f, 0.0) or 0.0 for f in feats]
            if any(np.isnan(x)):
                continue

            ic = max(hm["ic"], 0.001)  # floor to avoid zero weight

            # LightGBM prediction
            lgbm_pred = float(hm["lgbm"].predict([x])[0])

            # XGBoost prediction (if available)
            if hm["xgb"] is not None:
                import xgboost as xgb
                dm = xgb.DMatrix(np.array([x], dtype=np.float32),
                                 feature_names=feats)
                xgb_pred = float(hm["xgb"].predict(dm)[0])
                # Blend LightGBM + XGBoost
                w = self._lgbm_xgb_weight
                pred = lgbm_pred * w + xgb_pred * (1 - w)
            else:
                pred = lgbm_pred

            weighted_sum += pred * ic
            weight_total += ic

        if weight_total <= 0:
            return None
        return weighted_sum / weight_total

    def _compute_position_size(self, price: float) -> float:
        """Compute adaptive position size based on current equity and volatility.

        Formula: size = (equity * risk_pct) / (price * vol_stop)
        - equity: current USDT balance from exchange
        - risk_pct: fraction of equity willing to lose per trade
        - vol_stop: estimated stop distance (2x 20-bar vol)
        - Clamped to [min_size, equity * max_size_pct / price]
        """
        if not self._adaptive_sizing:
            return self._base_position_size

        try:
            bal = self._adapter.get_balances()
            usdt = bal.get("USDT")
            equity = float(usdt.total) if usdt else 0
        except Exception:
            return self._base_position_size

        if equity <= 0 or price <= 0:
            return self._base_position_size

        # Volatility-based stop: 2x recent 20-bar vol
        if len(self._rets) >= 20:
            vol = np.std(self._rets[-20:])
            stop_distance = max(vol * 2, 0.005)  # at least 0.5%
        else:
            stop_distance = 0.02  # default 2%

        # Position size = risk_amount / (price * stop_distance)
        risk_amount = equity * self._risk_per_trade
        size = risk_amount / (price * stop_distance)

        # Clamp: min lot size and max % of equity
        max_size = (equity * self._max_size_pct) / price
        size = max(self._min_size, min(size, max_size))

        # Round to exchange lot size (0.01 for ETH on Bybit)
        size = round(size, 2)

        if size != self._position_size:
            logger.info(
                "%s adaptive size: %.4f → %.4f (equity=$%.0f vol=%.4f stop=%.3f)",
                self._symbol, self._position_size, size,
                equity, stop_distance, stop_distance,
            )

        self._position_size = size
        return size

    def _check_regime(self, close: float) -> bool:
        """Check if current market regime is favorable for trading.

        Returns True if regime is active (OK to trade).
        Three-layer filter:
        1. Vol + trend (original): skip dead markets
        2. Ranging detector: skip choppy range-bound markets
        3. Dynamic deadzone: adapt to current volatility

        Walk-forward analysis: RANGE/LOW-VOL folds lose money (avg -2.4 Sharpe).
        BULL/BEAR folds make money (avg +1.7 Sharpe). This filter targets the gap.
        """
        self._closes.append(close)
        if len(self._closes) >= 2:
            ret = np.log(close / self._closes[-2])
            self._rets.append(ret)

        # Need enough history
        if len(self._rets) < 20:
            return True

        # 20-bar realized volatility
        vol_20 = np.std(self._rets[-20:])

        # Trend strength: |close / MA(480) - 1|
        if len(self._closes) >= self._ma_window:
            ma = np.mean(self._closes[-self._ma_window:])
            trend = abs(close / ma - 1)
        else:
            trend = 0.1  # assume active during warmup

        # Layer 1: original vol + trend filter
        base_active = (vol_20 >= self._vol_threshold) or (trend >= self._trend_threshold)

        # Layer 2: ranging detector — detect choppy, directionless markets
        # If price has bounced back and forth without making progress over 100 bars,
        # it's ranging. Net displacement / total path should be low.
        is_ranging = False
        if len(self._closes) >= 100:
            window = self._closes[-100:]
            net_move = abs(window[-1] - window[0])
            total_path = sum(abs(window[j] - window[j-1]) for j in range(1, len(window)))
            efficiency = net_move / total_path if total_path > 0 else 0
            # Low efficiency (<0.08) = choppy range-bound
            # High efficiency (>0.15) = trending
            is_ranging = efficiency < 0.08 and trend < 0.04

        self._regime_active = base_active and not is_ranging

        # Dynamic deadzone: scale with vol, clamp to [0.15, 0.6]
        if self._vol_median > 0:
            ratio = vol_20 / self._vol_median
            self._deadzone = max(0.15, min(0.6, self._deadzone_base * (ratio ** 0.5)))

        return self._regime_active

    def process_bar(self, bar: dict) -> dict:
        """Process one bar: regime → features → predict → signal → trade."""
        self._bars_processed += 1

        # Regime filter
        regime_ok = self._check_regime(bar["close"])

        # Get funding rate from ticker (if available)
        funding_rate = bar.get("funding_rate", float("nan"))
        if np.isnan(funding_rate):
            try:
                tick = self._adapter.get_ticker(self._symbol)
                funding_rate = tick.get("fundingRate", float("nan"))
            except Exception:
                funding_rate = float("nan")

        self._engine.push_bar(
            bar["close"], bar["volume"], bar["high"], bar["low"],
            bar["open"], funding_rate=funding_rate,
        )

        features = self._engine.get_features()
        if not features:
            return {"action": "no_features", "bar": self._bars_processed}

        feat_dict = dict(features)

        # Multi-horizon IC-weighted ensemble prediction
        pred = self._ensemble_predict(feat_dict)
        if pred is None:
            return {"action": "nan_features", "bar": self._bars_processed}
        self._predictions.append(pred)

        if len(self._predictions) < self._zscore_warmup:
            return {"action": "warmup", "bar": self._bars_processed,
                    "pred": pred, "buffer": len(self._predictions)}

        window = self._predictions[-self._zscore_window:]
        mean = np.mean(window)
        std = np.std(window)
        z = (pred - mean) / std if std > 1e-12 else 0.0

        prev_signal = self._current_signal

        # Regime filter: force flat when regime is unfavorable
        if not regime_ok:
            desired = 0
        elif z > self._deadzone:
            desired = 1
        elif z < -self._deadzone:
            desired = -1
        else:
            desired = 0

        # Smart exit: z-score reversal after min_hold
        z_reversal_threshold = -0.3
        force_exit = False
        if (prev_signal != 0 and self._hold_count >= self._min_hold):
            if prev_signal > 0 and z < z_reversal_threshold:
                force_exit = True  # long + z reversed negative
            elif prev_signal < 0 and z > -z_reversal_threshold:
                force_exit = True  # short + z reversed positive

        if force_exit:
            new_signal = 0
            self._hold_count = 1
        elif self._hold_count < self._min_hold and prev_signal != 0:
            # Min-hold only locks when IN a position (not when flat)
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
            "regime": "active" if regime_ok else "filtered",
            "dz": round(self._deadzone, 3),
        }

        if new_signal != prev_signal:
            # Recompute position size before entering new position
            if new_signal != 0:
                self._compute_position_size(bar["close"])
            result["trade"] = self._execute_signal_change(prev_signal, new_signal, bar["close"])
            result["size"] = self._position_size

        return result

    def _execute_signal_change(self, prev: int, new: int, price: float) -> dict:
        if self._killed:
            return {"action": "killed", "reason": "drawdown_breaker"}

        trade_info: dict = {}

        # Close existing position → track P&L
        if prev != 0 and self._entry_price > 0:
            if prev == 1:  # was long
                pnl_pct = (price - self._entry_price) / self._entry_price * 100
            else:  # was short
                pnl_pct = (self._entry_price - price) / self._entry_price * 100
            pnl_usd = pnl_pct / 100 * self._entry_price * self._position_size
            self._total_pnl += pnl_usd
            self._trade_count += 1
            if pnl_usd > 0:
                self._win_count += 1
            trade_info["closed_pnl"] = round(pnl_usd, 4)
            trade_info["closed_pct"] = round(pnl_pct, 2)
            logger.info(
                "%s CLOSE %s: pnl=$%.4f (%.2f%%) total=$%.4f wins=%d/%d",
                self._symbol, "long" if prev == 1 else "short",
                pnl_usd, pnl_pct, self._total_pnl,
                self._win_count, self._trade_count,
            )

        # Drawdown check
        self._peak_equity = max(self._peak_equity, self._total_pnl)
        if self._peak_equity > 0:
            dd = (self._peak_equity - self._total_pnl) / self._peak_equity * 100
            if dd >= self._max_drawdown_pct:
                self._killed = True
                logger.critical(
                    "%s DRAWDOWN KILL: dd=%.1f%% peak=$%.2f current=$%.2f",
                    self._symbol, dd, self._peak_equity, self._total_pnl,
                )
                if not self._dry_run:
                    self._adapter.close_position(self._symbol)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}

        if self._dry_run:
            trade_info["action"] = "dry_run"
            trade_info["from"] = prev
            trade_info["to"] = new
            return trade_info

        # Close venue position
        if prev != 0:
            self._adapter.close_position(self._symbol)

        # Open new position
        if new != 0:
            side = "buy" if new == 1 else "sell"
            result = self._adapter.send_market_order(self._symbol, side, self._position_size)
            self._entry_price = price
            logger.info("Opened %s %.4f @ ~$%.1f: %s", side, self._position_size, price, result)
            trade_info.update({"side": side, "qty": self._position_size, "result": result})
        else:
            self._entry_price = 0.0
            trade_info["action"] = "flat"

        return trade_info


def _run_ws_mode(runners: dict, adapter: Any, dry_run: bool) -> None:
    """WebSocket-based event loop — processes bars on confirmed kline push."""
    from execution.adapters.bybit.ws_client import BybitWsClient

    symbols = list(runners.keys())

    def on_ws_bar(symbol: str, bar: dict) -> None:
        runner = runners.get(symbol)
        if not runner:
            return
        result = runner.process_bar(bar)
        if result.get("action") == "signal":
            regime = result.get("regime", "?")
            logger.info(
                "WS %s bar %d: $%.1f z=%+.3f sig=%d hold=%d regime=%s dz=%.3f%s",
                symbol, result["bar"], result["close"],
                result["z"], result["signal"],
                result["hold_count"], regime, result.get("dz", 0),
                f" TRADE={result['trade']}" if "trade" in result else "",
            )

    ws = BybitWsClient(
        symbols=symbols, interval=INTERVAL,
        on_bar=on_ws_bar, demo=True,
    )

    logger.info(
        "Starting multi-symbol alpha (WebSocket): %s, dry=%s",
        symbols, dry_run,
    )
    ws.start()

    try:
        while True:
            time.sleep(300)
            sigs = {s: r._current_signal for s, r in runners.items()}
            pnls = {s: f"${r._total_pnl:.2f}" for s, r in runners.items()}
            trades = {s: f"{r._win_count}/{r._trade_count}" for s, r in runners.items()}
            sizes = {s: f"{r._position_size:.2f}" for s, r in runners.items()}
            logger.info("WS HEARTBEAT sigs=%s pnl=%s trades=%s size=%s", sigs, pnls, trades, sizes)
    except KeyboardInterrupt:
        logger.info("Stopped")
        ws.stop()
        if not dry_run:
            for symbol, runner in runners.items():
                if runner._current_signal != 0:
                    logger.info("Closing %s on exit...", symbol)
                    adapter.close_position(symbol)


def main():
    parser = argparse.ArgumentParser(description="Bybit alpha strategy runner")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Symbols to trade (default: BTCUSDT ETHUSDT)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--ws", action="store_true", help="Use WebSocket instead of REST polling")
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

    # WebSocket mode: push-based, low latency
    if args.ws:
        _run_ws_mode(runners, adapter, args.dry_run)
        return

    logger.info(
        "Starting multi-symbol alpha (REST poll): %s, poll=%ds, dry=%s",
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
                            regime = result.get("regime", "?")
                            logger.info(
                                "%s bar %d: $%.1f z=%+.3f sig=%d hold=%d regime=%s dz=%.3f%s",
                                symbol, result["bar"], result["close"],
                                result["z"], result["signal"],
                                result["hold_count"], regime,
                                result.get("dz", 0),
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
                regimes = {s: "active" if r._regime_active else "FILTERED" for s, r in runners.items()}
                pnls = {s: f"${r._total_pnl:.2f}" for s, r in runners.items()}
                trades = {s: f"{r._win_count}/{r._trade_count}" for s, r in runners.items()}
                sizes = {s: f"{r._position_size:.2f}" for s, r in runners.items()}
                logger.info(
                    "HEARTBEAT cycle=%d sigs=%s holds=%s regimes=%s pnl=%s trades=%s size=%s",
                    cycle_count, sigs, holds, regimes, pnls, trades, sizes,
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
