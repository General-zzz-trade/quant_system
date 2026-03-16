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
    # 15m alpha: separate model, different interval
    "ETHUSDT_15m": {"size": 0.01, "model_dir": "ETHUSDT_15m", "symbol": "ETHUSDT",
                    "interval": "15", "warmup": 800},
    # SUI 1h alpha (walk-forward 6/7 PASS, Sharpe 1.63, +150%)
    "SUIUSDT": {"size": 0.1, "model_dir": "SUIUSDT"},
    # AXS 1h alpha (walk-forward 13/17 PASS, Sharpe 1.25, +241%)
    "AXSUSDT": {"size": 1.0, "model_dir": "AXSUSDT"},
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
                xgb_raw = pickle.load(f)  # noqa: S301 — trusted local artifact
            xgb_model = xgb_raw["model"] if isinstance(xgb_raw, dict) else xgb_raw

        # Also load Ridge if available (walk-forward winner: 15/20 PASS)
        ridge_model = None
        ridge_features = None
        ridge_name = hm.get("ridge", "")
        ridge_path = model_dir / ridge_name if ridge_name else None
        if ridge_path and ridge_path.is_file():
            with open(ridge_path, "rb") as f:
                ridge_raw = pickle.load(f)  # noqa: S301 — trusted local artifact
            ridge_model = ridge_raw["model"] if isinstance(ridge_raw, dict) else ridge_raw
            ridge_features = ridge_raw.get("features") if isinstance(ridge_raw, dict) else None

        horizon_models.append({
            "horizon": hm["horizon"],
            "lgbm": model,
            "xgb": xgb_model,
            "ridge": ridge_model,
            "ridge_features": ridge_features,  # may differ from lgbm features
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
        "min_hold": config.get("min_hold", 18),
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

    # Kelly-optimal leverage ladder (validated by Monte Carlo simulation 2026-03-15)
    # Full Kelly = 1.3x, half-Kelly = 0.65x. At 3x+, bust rate > 50%.
    # Geometric mean: 1.5x=14.3%/q (best), 2x=11.0%/q, 3x=-4.4%/q (negative!)
    # Ladder is flat at 1.5x — Kelly optimal doesn't depend on account size.
    LEVERAGE_LADDER = [
        (0,      1.5),    # $0-$5K:      1.5x (Kelly optimal, 2% bust rate)
        (5000,   1.5),    # $5K-$20K:    1.5x (same — Kelly is scale-invariant)
        (20000,  1.0),    # $20K-$50K:   1.0x (half-Kelly, capital preservation)
        (50000,  1.0),    # $50K+:       1.0x (pure alpha, no leverage risk)
    ]

    def __init__(self, adapter: Any, model_info: dict, symbol: str,
                 dry_run: bool = False, position_size: float = 0.001,
                 adaptive_sizing: bool = True, risk_per_trade: float = 0.10,
                 min_size: float = 0.01, max_size_pct: float = 10.00):
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

        # Regime filter state — thresholds scale with timeframe
        self._closes: list[float] = []
        self._rets: list[float] = []
        self._regime_active = True
        # Detect timeframe from model config: 15m bars have smaller per-bar returns
        is_15m = "15m" in model_info.get("config", {}).get("version", "")
        if is_15m:
            # 15m bars: vol is ~2x lower, scale down threshold
            self._vol_threshold = 0.002   # 20-bar vol on 15m ≈ 5h
            self._trend_threshold = 0.02  # trend more sensitive on short TF
            self._ma_window = 480 * 4     # 480×4 = 1920 bars = 20 days (same real time as 1h)
            self._ranging_window = 400    # 400 bars = ~4 days (same as 100 1h bars)
            self._ranging_threshold = 0.06  # slightly more lenient
        else:
            self._vol_threshold = 0.004   # 20-bar vol on 1h ≈ 20h
            self._trend_threshold = 0.04  # |close/MA480 - 1|
            self._ma_window = 480         # 480 bars = 20 days
            self._ranging_window = 100    # 100 bars = ~4 days
            self._ranging_threshold = 0.08

        # P&L tracking + drawdown circuit breaker
        self._entry_price: float = 0.0
        self._total_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0
        self._max_drawdown_pct: float = 15.0  # kill at 15% drawdown
        self._killed: bool = False

        # Adaptive stop-loss state (grid-search optimized 2026-03-15)
        # ATR×2.0 initial, trail at 0.8×ATR profit, step 0.3×ATR → 18/20 folds, 75% trail wins
        self._atr_buffer: list[float] = []  # recent ATR values for adaptive stop
        self._trade_peak_price: float = 0.0  # highest favorable price since entry (for trailing)
        self._atr_stop_mult: float = 2.0     # ATR multiplier for initial stop
        self._trail_atr_mult: float = 0.8    # trail activates after 0.8×ATR profit (was 1.5)
        self._trail_step: float = 0.3        # tight trail: 0.3×ATR (was 0.5)
        self._breakeven_atr: float = 1.0     # move stop to breakeven after 1x ATR profit

        from _quant_hotpath import RustFeatureEngine
        self._engine = RustFeatureEngine()

    def warmup(self, limit: int = WARMUP_BARS, interval: str = INTERVAL) -> int:
        """Fetch historical bars and warm up feature engine."""
        bars = self._adapter.get_klines(self._symbol, interval=interval,
                                        limit=limit)
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

    def _current_atr(self) -> float:
        """Get current ATR (average true range) from recent bar data.

        Uses 14-bar ATR as percentage of price. Falls back to 1.5% if
        insufficient data.
        """
        if len(self._atr_buffer) < 5:
            return 0.015  # 1.5% default
        return float(np.mean(self._atr_buffer[-14:]))

    def _compute_stop_price(self, current_price: float) -> float:
        """Compute adaptive stop price based on ATR + trailing logic.

        Three-phase stop:
        1. Initial: entry ± ATR × 2.0 (wide, let trade breathe)
        2. Breakeven: after 1×ATR profit, move stop to entry price
        3. Trailing: after 1.5×ATR profit, trail at peak - ATR×0.5

        Hard floor: never allow stop wider than 5% (capital protection).
        Hard ceiling: never allow stop tighter than 0.3% (avoid noise stops).
        """
        if self._entry_price <= 0 or self._current_signal == 0:
            return 0.0

        atr = self._current_atr()
        side = self._current_signal  # +1=long, -1=short
        entry = self._entry_price

        # Update trade peak (best price since entry)
        if side > 0:
            self._trade_peak_price = max(self._trade_peak_price, current_price)
            profit_pct = (self._trade_peak_price - entry) / entry
        else:
            self._trade_peak_price = min(self._trade_peak_price, current_price)
            profit_pct = (entry - self._trade_peak_price) / entry

        # Phase 1: Initial stop — wide, based on ATR
        initial_stop_dist = atr * self._atr_stop_mult  # typically 2×ATR ≈ 3%

        # Phase 2: Breakeven — after 1×ATR profit, move stop to entry
        if profit_pct >= atr * self._breakeven_atr:
            # Phase 3: Trailing — after 1.5×ATR profit, trail tightly
            if profit_pct >= atr * self._trail_atr_mult:
                trail_dist = atr * self._trail_step  # tight trail: 0.5×ATR
                if side > 0:
                    stop = self._trade_peak_price * (1 - trail_dist)
                else:
                    stop = self._trade_peak_price * (1 + trail_dist)
            else:
                # Breakeven stop (at entry + tiny buffer)
                buffer = atr * 0.1  # 0.1×ATR above entry
                if side > 0:
                    stop = entry * (1 + buffer)
                else:
                    stop = entry * (1 - buffer)
        else:
            # Initial wide stop
            if side > 0:
                stop = entry * (1 - initial_stop_dist)
            else:
                stop = entry * (1 + initial_stop_dist)

        # Hard floor: max 5% loss (capital protection)
        if side > 0:
            floor = entry * 0.95
            stop = max(stop, floor)
        else:
            ceil = entry * 1.05
            stop = min(stop, ceil)

        # Hard ceiling: min 0.3% distance (avoid noise stops)
        min_dist = entry * 0.003
        if side > 0 and current_price - stop < min_dist:
            stop = min(stop, current_price - min_dist)
        elif side < 0 and stop - current_price < min_dist:
            stop = max(stop, current_price + min_dist)

        return stop

    def check_realtime_stoploss(self, price: float) -> bool:
        """Check adaptive stop-loss against real-time price.

        Called on every tick (~100ms). Uses ATR-based trailing stop
        that adapts to market volatility.
        """
        if self._current_signal == 0 or self._entry_price <= 0 or self._killed:
            return False

        stop = self._compute_stop_price(price)

        triggered = False
        if self._current_signal > 0 and price <= stop:
            triggered = True
        elif self._current_signal < 0 and price >= stop:
            triggered = True

        if triggered:
            if self._current_signal > 0:
                unrealized = (price - self._entry_price) / self._entry_price
            else:
                unrealized = (self._entry_price - price) / self._entry_price

            atr = self._current_atr()
            phase = "TRAIL" if unrealized > 0 else ("BREAKEVEN" if abs(unrealized) < atr else "INITIAL")

            logger.warning(
                "%s ADAPTIVE STOP [%s]: price=$%.2f stop=$%.2f entry=$%.2f "
                "pnl=%.2f%% atr=%.2f%% peak=$%.2f",
                self._symbol, phase, price, stop, self._entry_price,
                unrealized * 100, atr * 100, self._trade_peak_price,
            )
            if not self._dry_run:
                self._adapter.close_position(self._symbol)

            pnl_usd = unrealized * self._entry_price * self._position_size
            self._total_pnl += pnl_usd
            self._trade_count += 1
            if pnl_usd > 0:
                self._win_count += 1
            logger.info(
                "%s STOP CLOSED: pnl=$%.4f total=$%.4f trades=%d/%d",
                self._symbol, pnl_usd, self._total_pnl,
                self._win_count, self._trade_count,
            )

            self._current_signal = 0
            self._hold_count = 0
            self._entry_price = 0.0
            self._trade_peak_price = 0.0
            return True

        return False

    def _ensemble_predict(self, feat_dict: dict) -> float | None:
        """Ensemble: Ridge (primary) + LightGBM (secondary).

        Ridge won 20-fold walk-forward (15/20 PASS, Sharpe 0.54, +433%).
        Config weights: ridge_weight (default 0.6) + lgbm_weight (default 0.4).
        """
        if not self._horizon_models:
            x = [feat_dict.get(f, 0.0) or 0.0 for f in self._features]
            if any(np.isnan(x)):
                return None
            return float(self._model.predict([x])[0])

        ridge_w = self._config.get("ridge_weight", 0.6)
        lgbm_w = self._config.get("lgbm_weight", 0.4)

        weighted_sum = 0.0
        weight_total = 0.0

        for hm in self._horizon_models:
            feats = hm["features"]
            x = [feat_dict.get(f, 0.0) or 0.0 for f in feats]
            if any(np.isnan(x)):
                continue

            ic = max(hm["ic"], 0.001)

            # Ridge prediction (primary — walk-forward winner)
            if hm.get("ridge") is not None:
                # Ridge may use different features than LGBM
                rf = hm.get("ridge_features") or feats
                rx = [feat_dict.get(f, 0.0) or 0.0 for f in rf]
                if not any(np.isnan(rx)):
                    ridge_pred = float(hm["ridge"].predict([rx])[0])
                    lgbm_pred = float(hm["lgbm"].predict([x])[0])
                    pred = ridge_pred * ridge_w + lgbm_pred * lgbm_w
                else:
                    pred = float(hm["lgbm"].predict([x])[0])
            else:
                pred = float(hm["lgbm"].predict([x])[0])

            weighted_sum += pred * ic
            weight_total += ic

        if weight_total <= 0:
            return None
        return weighted_sum / weight_total

    def _get_leverage_for_equity(self, equity: float) -> float:
        """Look up leverage from ladder based on current equity."""
        lev = 1.0
        for threshold, lev_val in self.LEVERAGE_LADDER:
            if equity >= threshold:
                lev = lev_val
        return lev

    def _compute_position_size(self, price: float) -> float:
        """Compute position size using Kelly-optimal leverage ladder.

        1. Look up leverage from equity-based ladder
        2. Position = equity * leverage / price
        3. Clamp to [min_size, exchange limit]
        4. Set exchange leverage to match

        Kelly math: 20x optimal, but real-world safety margin for
        stop-loss slippage means 5-10x at small equity.
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

        # Get leverage from ladder
        target_lev = self._get_leverage_for_equity(equity)

        # Position = equity * leverage / price
        size = (equity * target_lev) / price

        # Clamp to min lot size
        size = max(self._min_size, size)

        # Round to exchange lot size (0.01 for ETH on Bybit)
        size = round(size, 2)

        # Round to exchange lot size (0.01 for ETH on Bybit)
        size = round(size, 2)

        if size != self._position_size:
            logger.info(
                "%s SIZING: equity=$%.0f lev=%.0fx → %.2f ETH ($%.0f notional)",
                self._symbol, equity, target_lev, size, size * price,
            )

        # Set exchange leverage to match (only if changed)
        if not hasattr(self, "_current_exchange_lev") or self._current_exchange_lev != int(target_lev):
            try:
                self._adapter._client.post("/v5/position/set-leverage", {
                    "category": "linear", "symbol": self._symbol,
                    "buyLeverage": str(int(target_lev)),
                    "sellLeverage": str(int(target_lev)),
                })
                self._current_exchange_lev = int(target_lev)
                logger.info("%s exchange leverage set to %dx", self._symbol, int(target_lev))
            except Exception:
                pass  # non-fatal

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
        rw = self._ranging_window
        if len(self._closes) >= rw:
            window = self._closes[-rw:]
            net_move = abs(window[-1] - window[0])
            total_path = sum(abs(window[j] - window[j-1]) for j in range(1, len(window)))
            efficiency = net_move / total_path if total_path > 0 else 0
            is_ranging = efficiency < self._ranging_threshold and trend < self._trend_threshold

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

        # Update ATR buffer (True Range as % of close)
        if len(self._closes) >= 2:
            prev_close = self._closes[-2]
            tr = max(
                bar["high"] - bar["low"],
                abs(bar["high"] - prev_close),
                abs(bar["low"] - prev_close),
            )
            atr_pct = tr / bar["close"] if bar["close"] > 0 else 0
            self._atr_buffer.append(atr_pct)
            if len(self._atr_buffer) > 50:
                self._atr_buffer = self._atr_buffer[-50:]

        # Smart exits (in priority order)
        force_exit = False
        exit_reason = ""

        # 1. Adaptive stop loss: ATR-based with trailing
        if prev_signal != 0 and self._entry_price > 0:
            stop = self._compute_stop_price(bar["close"])
            if prev_signal > 0 and bar["low"] <= stop:
                force_exit = True
                exit_reason = "atr_stop"
            elif prev_signal < 0 and bar["high"] >= stop:
                force_exit = True
                exit_reason = "atr_stop"

        # 2. Z-score reversal after min_hold
        z_reversal_threshold = -0.3
        if not force_exit and (prev_signal != 0 and self._hold_count >= self._min_hold):
            if prev_signal > 0 and z < z_reversal_threshold:
                force_exit = True
                exit_reason = "z_reversal"
            elif prev_signal < 0 and z > -z_reversal_threshold:
                force_exit = True
                exit_reason = "z_reversal"

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
        if force_exit:
            result["exit_reason"] = exit_reason

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
            self._trade_peak_price = price  # initialize trailing peak
            atr = self._current_atr()
            stop = self._compute_stop_price(price)
            logger.info(
                "Opened %s %.4f @ ~$%.1f stop=$%.2f (ATR=%.2f%%): %s",
                side, self._position_size, price, stop, atr * 100, result,
            )
            trade_info.update({"side": side, "qty": self._position_size, "result": result,
                               "stop": round(stop, 2), "atr_pct": round(atr * 100, 2)})
        else:
            self._entry_price = 0.0
            self._trade_peak_price = 0.0
            trade_info["action"] = "flat"

        return trade_info


class PortfolioCombiner:
    """Combines signals from multiple alphas into a single net position.

    Each alpha produces signal ∈ {-1, 0, +1} with a weight.
    Net signal = weighted average → discretize to {-1, 0, +1}.

    Prevents: double-sizing when both agree, fee waste when they disagree,
    and oversized positions from independent execution.

    Position management:
    - Net signal > threshold → long
    - Net signal < -threshold → short
    - Otherwise → flat
    - Single position on exchange, sized by combined conviction
    """

    def __init__(self, adapter: Any, symbol: str, weights: dict[str, float],
                 threshold: float = 0.3, dry_run: bool = False,
                 min_size: float = 0.01):
        self._adapter = adapter
        self._symbol = symbol
        self._weights = weights  # runner_key → weight (e.g. {"ETHUSDT": 0.5, "ETHUSDT_15m": 0.5})
        self._threshold = threshold
        self._dry_run = dry_run
        self._min_size = min_size

        self._signals: dict[str, int] = {k: 0 for k in weights}
        self._current_position: int = 0  # -1, 0, +1
        self._position_size: float = 0.0
        self._entry_price: float = 0.0
        self._total_pnl: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0

    def update_signal(self, runner_key: str, signal: int, price: float) -> dict | None:
        """Update one alpha's signal and recompute net position.

        Returns trade info dict if position changed, None otherwise.
        """
        if runner_key not in self._signals:
            return None

        old_signal = self._signals[runner_key]
        if signal == old_signal:
            return None  # no change

        self._signals[runner_key] = signal

        # Compute weighted net signal
        net = 0.0
        total_weight = 0.0
        for k, w in self._weights.items():
            net += self._signals[k] * w
            total_weight += w
        if total_weight > 0:
            net /= total_weight

        # AGREE ONLY: trade only when ALL alphas agree direction
        # Backtest: AGREE Sharpe=5.48 vs weighted COMBO Sharpe=3.18 (+72%)
        n_long = sum(1 for s in self._signals.values() if s > 0)
        n_short = sum(1 for s in self._signals.values() if s < 0)
        n_total = len(self._signals)

        if n_long == n_total:
            desired = 1    # unanimous long
        elif n_short == n_total:
            desired = -1   # unanimous short
        else:
            desired = 0    # any disagreement → flat

        if desired == self._current_position:
            return None  # net position unchanged

        # Position change needed
        trade = self._execute_change(desired, price)
        return trade

    def _execute_change(self, desired: int, price: float) -> dict:
        """Execute net position change on exchange."""
        prev = self._current_position
        trade_info = {
            "from": prev, "to": desired, "price": price,
            "signals": dict(self._signals),
        }

        # Close existing
        if prev != 0 and self._entry_price > 0:
            if prev > 0:
                pnl_pct = (price - self._entry_price) / self._entry_price
            else:
                pnl_pct = (self._entry_price - price) / self._entry_price
            pnl_usd = pnl_pct * self._entry_price * self._position_size
            self._total_pnl += pnl_usd
            self._trade_count += 1
            if pnl_usd > 0:
                self._win_count += 1
            trade_info["closed_pnl"] = round(pnl_usd, 4)
            logger.info(
                "COMBO CLOSE %s: pnl=$%.4f total=$%.4f wins=%d/%d",
                "long" if prev > 0 else "short",
                pnl_usd, self._total_pnl, self._win_count, self._trade_count,
            )

            if not self._dry_run:
                self._adapter.close_position(self._symbol)

        # Compute new position size
        if desired != 0:
            try:
                bal = self._adapter.get_balances()
                usdt = bal.get("USDT")
                equity = float(usdt.total) if usdt else 0
            except Exception:
                equity = 100

            # Conviction scaling: both agree = full size, one only = half
            agree_count = sum(1 for s in self._signals.values() if s == desired)
            conviction = agree_count / len(self._signals)  # 0.5 = one alpha, 1.0 = both
            leverage = 1.5
            size = (equity * leverage * conviction) / price
            size = max(self._min_size, round(size, 2))

            self._position_size = size
            self._entry_price = price

            # Cap at 30% of equity per symbol (leave room for other symbols)
            max_notional = equity * 0.30 * leverage
            size = min(size, max_notional / price)
            size = max(self._min_size, round(size, 2))

            self._position_size = size
            side = "buy" if desired > 0 else "sell"
            if not self._dry_run:
                result = self._adapter.send_market_order(self._symbol, side, size)
                trade_info["result"] = result
                logger.info(
                    "COMBO ORDER result: %s %s %.2f → %s",
                    side, self._symbol, size, result,
                )
            logger.info(
                "COMBO OPEN %s %.2f @ $%.1f conviction=%.0f%% signals=%s",
                side, size, price, conviction * 100, self._signals,
            )
        else:
            self._entry_price = 0.0
            self._position_size = 0.0

        self._current_position = desired
        return trade_info

    def get_status(self) -> dict:
        return {
            "position": self._current_position,
            "signals": dict(self._signals),
            "pnl": f"${self._total_pnl:.2f}",
            "trades": f"{self._win_count}/{self._trade_count}",
            "size": self._position_size,
        }


def _run_ws_mode(runners: dict, adapter: Any, dry_run: bool,
                 runner_intervals: dict | None = None) -> None:
    """WebSocket-based event loop — processes bars on confirmed kline push.

    Supports multiple intervals (e.g. 1h + 15m) via separate WS clients.
    runner_intervals maps runner_key → (real_symbol, interval).
    """
    from execution.adapters.bybit.ws_client import BybitWsClient

    if runner_intervals is None:
        runner_intervals = {s: (s, INTERVAL) for s in runners}

    # Group runners by interval → separate WS clients
    interval_groups: dict[str, dict[str, str]] = {}  # interval → {real_symbol: runner_key}
    for runner_key, (real_symbol, interval) in runner_intervals.items():
        interval_groups.setdefault(interval, {})[real_symbol] = runner_key

    # Build portfolio combiner for symbols with multiple alphas
    # Group runner_keys by real_symbol
    symbol_runners: dict[str, list[str]] = {}
    for rkey, (rsym, _) in runner_intervals.items():
        symbol_runners.setdefault(rsym, []).append(rkey)

    combiners: dict[str, PortfolioCombiner] = {}
    for rsym, rkeys in symbol_runners.items():
        if len(rkeys) > 1:
            # Multiple alphas on same symbol → use combiner
            weights = {k: 0.5 for k in rkeys}
            combiners[rsym] = PortfolioCombiner(
                adapter=adapter, symbol=rsym, weights=weights,
                threshold=0.3, dry_run=dry_run,
            )
            # Disable direct trading in individual runners
            for rk in rkeys:
                runners[rk]._dry_run = True  # signals only, combiner executes
            logger.info("COMBO mode: %s runners=%s weights=%s", rsym, rkeys, weights)

    def make_bar_handler(group: dict[str, str]):
        def on_ws_bar(symbol: str, bar: dict) -> None:
            runner_key = group.get(symbol)
            if not runner_key:
                return
            runner = runners.get(runner_key)
            if not runner:
                return
            result = runner.process_bar(bar)
            if result.get("action") == "signal":
                regime = result.get("regime", "?")
                label = runner_key if runner_key != symbol else symbol
                trade_str = ""

                # Route signal through combiner if available
                real_sym = runner_intervals.get(runner_key, (symbol, ""))[0]
                if real_sym in combiners:
                    combo_trade = combiners[real_sym].update_signal(
                        runner_key, result["signal"], result["close"],
                    )
                    if combo_trade:
                        trade_str = f" COMBO={combo_trade}"
                elif "trade" in result:
                    trade_str = f" TRADE={result['trade']}"

                logger.info(
                    "WS %s bar %d: $%.1f z=%+.3f sig=%d hold=%d regime=%s dz=%.3f%s",
                    label, result["bar"], result["close"],
                    result["z"], result["signal"],
                    result["hold_count"], regime, result.get("dz", 0),
                    trade_str,
                )
        return on_ws_bar

    def on_ws_tick(symbol: str, price: float) -> None:
        """Real-time stop-loss check — routes to all runners for this symbol."""
        for rkey, (rsym, _) in runner_intervals.items():
            if rsym == symbol:
                runner = runners.get(rkey)
                if runner:
                    runner.check_realtime_stoploss(price)

    # Start one WS client per interval
    ws_clients = []
    for interval, group in interval_groups.items():
        real_symbols = list(group.keys())
        ws = BybitWsClient(
            symbols=real_symbols, interval=interval,
            on_bar=make_bar_handler(group), on_tick=on_ws_tick, demo=True,
        )
        ws_clients.append(ws)

    logger.info(
        "Starting multi-symbol alpha (WebSocket + realtime stop): %s, dry=%s",
        list(runners.keys()), dry_run,
    )
    for ws in ws_clients:
        ws.start()

    try:
        while True:
            time.sleep(300)
            sigs = {s: r._current_signal for s, r in runners.items()}
            pnls = {s: f"${r._total_pnl:.2f}" for s, r in runners.items()}
            trades = {s: f"{r._win_count}/{r._trade_count}" for s, r in runners.items()}
            combo_status = {sym: c.get_status() for sym, c in combiners.items()} if combiners else {}
            if combo_status:
                logger.info("WS HEARTBEAT sigs=%s combo=%s", sigs, combo_status)
            else:
                logger.info("WS HEARTBEAT sigs=%s pnl=%s trades=%s", sigs, pnls, trades)
    except KeyboardInterrupt:
        logger.info("Stopped")
        for ws in ws_clients:
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

    runner_intervals: dict[str, tuple[str, str]] = {}  # runner_key → (real_symbol, interval)

    for symbol in args.symbols:
        sym_cfg = SYMBOL_CONFIG.get(symbol, {"size": 0.001, "model_dir": f"{symbol}_gate_v2"})
        model_dir = MODEL_BASE / sym_cfg["model_dir"]
        if not (model_dir / "config.json").exists():
            logger.warning("No model for %s at %s, skipping", symbol, model_dir)
            continue

        model_info = load_model(model_dir)
        real_symbol = sym_cfg.get("symbol", symbol)  # e.g. ETHUSDT_15m → ETHUSDT
        interval = sym_cfg.get("interval", INTERVAL)
        warmup = sym_cfg.get("warmup", WARMUP_BARS)

        logger.info(
            "%s: model v%s, %d features, dz=%.1f, hold=%d-%d, size=%.4f",
            symbol, model_info["config"]["version"],
            len(model_info["features"]), model_info["deadzone"],
            model_info["min_hold"], model_info["max_hold"], sym_cfg["size"],
        )

        runner = AlphaRunner(
            adapter=adapter, model_info=model_info, symbol=real_symbol,
            dry_run=args.dry_run, position_size=sym_cfg["size"],
        )
        logger.info("%s: warming up %d bars...", symbol, warmup)
        runner.warmup(limit=warmup, interval=interval)
        runners[symbol] = runner
        runner_intervals[symbol] = (real_symbol, interval)
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
        _run_ws_mode(runners, adapter, args.dry_run, runner_intervals=runner_intervals)
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
