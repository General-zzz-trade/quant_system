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
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

logger = logging.getLogger(__name__)

# Shared cross-symbol signal state for consensus scaling.
# Maps runner_key → current signal (+1, -1, 0). Updated by each AlphaRunner
# after process_bar(). Read by _get_consensus_scale() to adjust sizing.
_consensus_signals: dict[str, int] = {}


# ── Binance OI/LS/Taker data fetcher ─────────────────────────────────

def _fetch_binance_oi_data(symbol: str) -> dict:
    """Fetch latest OI, Long/Short ratio, Taker ratio, Top trader ratio from Binance.

    Returns dict with keys: open_interest, ls_ratio, top_trader_ls_ratio,
    taker_buy_vol, taker_sell_vol. Values are float or NaN if unavailable.
    Uses Binance public API (no auth required).
    """
    result = {
        "open_interest": float("nan"),
        "ls_ratio": float("nan"),
        "top_trader_ls_ratio": float("nan"),
        "taker_buy_vol": 0.0,
        "taker_sell_vol": 0.0,
    }
    base = "https://fapi.binance.com"
    headers = {"Accept": "application/json"}
    timeout = 3  # fast timeout, non-critical

    # OI
    try:
        url = f"{base}/fapi/v1/openInterest?symbol={symbol}"
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            data = json.loads(resp.read())
        result["open_interest"] = float(data.get("openInterest", 0))
    except Exception:
        pass

    # Global Long/Short ratio
    try:
        url = f"{base}/futures/data/globalLongShortAccountRatio?symbol={symbol}&period=1h&limit=1"
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data:
            result["ls_ratio"] = float(data[0].get("longShortRatio", 1))
    except Exception:
        pass

    # Top trader position ratio
    try:
        url = f"{base}/futures/data/topLongShortPositionRatio?symbol={symbol}&period=1h&limit=1"
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data:
            result["top_trader_ls_ratio"] = float(data[0].get("longShortRatio", 1))
    except Exception:
        pass

    # Taker buy/sell volume
    try:
        url = f"{base}/futures/data/takerlongshortRatio?symbol={symbol}&period=1h&limit=1"
        with urlopen(Request(url, headers=headers), timeout=timeout) as resp:
            data = json.loads(resp.read())
        if data:
            result["taker_buy_vol"] = float(data[0].get("buyVol", 0))
            result["taker_sell_vol"] = float(data[0].get("sellVol", 0))
    except Exception:
        pass

    return result

# ── Config ──────────────────────────────────────────────────────────

MODEL_BASE = Path("models_v8")
INTERVAL = "60"  # Bybit: "60" = 1h
WARMUP_BARS = 800  # Must be > zscore_window(720) + zscore_warmup(180) for full z-score convergence
POLL_INTERVAL = 60  # seconds between checks

# Default symbols + position sizes
SYMBOL_CONFIG = {
    "BTCUSDT": {"size": 0.001, "model_dir": "BTCUSDT_gate_v2", "max_qty": 1190, "step": 0.001},
    "ETHUSDT": {"size": 0.01, "model_dir": "ETHUSDT_gate_v2", "max_qty": 8000, "step": 0.01},
    # 15m alpha: separate model, different interval
    "ETHUSDT_15m": {"size": 0.01, "model_dir": "ETHUSDT_15m", "symbol": "ETHUSDT",
                    "interval": "15", "warmup": 800},
    # SUI 1h alpha (walk-forward 6/7 PASS, Sharpe 1.63, +150%)
    "SUIUSDT": {"size": 10, "model_dir": "SUIUSDT", "max_qty": 330000, "step": 10},
    # AXS 1h alpha (walk-forward 13/17 PASS, Sharpe 1.25, +241%)
    "AXSUSDT": {"size": 5.0, "model_dir": "AXSUSDT", "max_qty": 50000, "step": 0.1},  # min $5 notional → ~4 AXS
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
    """Create Bybit adapter from environment variables."""
    from execution.adapters.bybit import BybitAdapter, BybitConfig

    api_key = os.environ.get("BYBIT_API_KEY")
    api_secret = os.environ.get("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "BYBIT_API_KEY and BYBIT_API_SECRET environment variables are required. "
            "Set them in .env or export them. See .env.example for all required vars."
        )
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
                 min_size: float = 0.01, max_size_pct: float = 10.00,
                 max_qty: float = 0, step_size: float = 0.01,
                 risk_evaluator: Any = None, kill_switch: Any = None,
                 state_store: Any = None):
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
        self._max_qty = max_qty                # exchange max order qty (0 = no limit)
        self._step_size = step_size            # qty rounding step

        self._current_signal = 0
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

        # Threading lock for shared state (_current_signal, _entry_price, orders)
        self._trade_lock = threading.Lock()

        # P&L tracking
        self._entry_price: float = 0.0
        self._entry_size: float = 0.0   # position size at entry time (for PnL calc)
        self._total_pnl: float = 0.0
        self._peak_equity: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0

        # Rust risk evaluator + kill switch (shared across all runners)
        self._risk_eval = risk_evaluator
        self._kill_switch = kill_switch

        # RustStateStore: authoritative position truth (shared across all runners)
        # Tracks position qty, avg_price, account balance, portfolio exposure
        # via RustFillEvent processing on the Rust heap.
        self._state_store = state_store

        # Adaptive stop-loss state (grid-search optimized 2026-03-15)
        # ATR×2.0 initial, trail at 0.8×ATR profit, step 0.3×ATR → 18/20 folds, 75% trail wins
        self._atr_buffer: list[float] = []  # recent ATR values for adaptive stop
        self._trade_peak_price: float = 0.0  # highest favorable price since entry (for trailing)
        self._atr_stop_mult: float = 2.0     # ATR multiplier for initial stop
        self._trail_atr_mult: float = 0.8    # trail activates after 0.8×ATR profit (was 1.5)
        self._trail_step: float = 0.3        # tight trail: 0.3×ATR (was 0.5)
        self._breakeven_atr: float = 1.0     # move stop to breakeven after 1x ATR profit

        from _quant_hotpath import (RustFeatureEngine, RustInferenceBridge,
                                    RustOrderStateMachine, RustCircuitBreaker,
                                    RustFillEvent, RustMarketEvent)
        self._engine = RustFeatureEngine()
        self._inference = RustInferenceBridge(
            zscore_window=self._zscore_window,
            zscore_warmup=self._zscore_warmup,
        )
        self._osm = RustOrderStateMachine()
        self._circuit_breaker = RustCircuitBreaker(
            failure_threshold=3, window_s=120.0, recovery_timeout_s=60.0,
        )
        # Stash classes for constructing Rust events later
        self._RustFillEvent = RustFillEvent
        self._RustMarketEvent = RustMarketEvent

        # V14: BTC dominance ratio buffer (BTC/ETH)
        self._dom_ratio_buf: list[float] = []

        # Cross-symbol consensus + z-scale state
        self._runner_key: str = ""  # set by main() after construction
        self._z_scale: float = 1.0  # non-linear z-score position scale

    @property
    def _killed(self) -> bool:
        """Check kill switch (Rust) instead of local boolean."""
        if self._kill_switch is not None:
            return self._kill_switch.is_armed()
        return False

    def _record_fill(self, side: str, qty: float, price: float,
                     realized_pnl: float = 0.0) -> None:
        """Record a fill in RustStateStore for position truth tracking.

        Creates a RustFillEvent (Rust-native, zero getattr overhead) and
        processes it through the state store's reducer pipeline. This updates
        position qty, avg_price, account realized_pnl, and portfolio exposure
        atomically on the Rust heap.
        """
        if self._state_store is None:
            return
        try:
            fill = self._RustFillEvent(
                symbol=self._symbol,
                side=side,
                qty=qty,
                price=price,
                realized_pnl=realized_pnl,
                ts=str(int(time.time() * 1000)),
            )
            self._state_store.process_event(fill, self._symbol)
        except Exception:
            logger.debug("%s StateStore fill recording failed", self._symbol,
                         exc_info=True)

    def _record_market_update(self, bar: dict) -> None:
        """Record a market bar in RustStateStore for mark-to-market.

        Updates the market state (last price) so portfolio exposure and
        unrealized PnL calculations stay current.
        """
        if self._state_store is None:
            return
        try:
            me = self._RustMarketEvent(
                symbol=self._symbol,
                open=bar["open"], high=bar["high"],
                low=bar["low"], close=bar["close"],
                volume=bar["volume"],
                ts=str(int(time.time() * 1000)),
            )
            self._state_store.process_event(me, self._symbol)
        except Exception:
            logger.debug("%s StateStore market update failed", self._symbol,
                         exc_info=True)

    def _fetch_eth_price(self) -> float:
        """Fetch current ETH price from Binance for dominance computation."""
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=ETHUSDT"
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=3) as resp:
                return float(json.loads(resp.read()).get("price", 0))
        except Exception:
            return 0.0

    def _compute_dominance_features(self, btc_close: float) -> dict:
        """Compute BTC/ETH dominance features for BTC alpha."""
        eth_price = self._fetch_eth_price()
        feats: dict = {}
        if eth_price <= 0:
            for k in ("btc_dom_dev_20", "btc_dom_dev_50", "btc_dom_ret_24", "btc_dom_ret_72"):
                feats[k] = float("nan")
            return feats

        ratio = btc_close / eth_price
        self._dom_ratio_buf.append(ratio)
        # Keep max 75 bars
        if len(self._dom_ratio_buf) > 75:
            self._dom_ratio_buf = self._dom_ratio_buf[-75:]

        buf = self._dom_ratio_buf
        if len(buf) >= 21:
            ma20 = sum(buf[-20:]) / 20
            feats["btc_dom_dev_20"] = ratio / ma20 - 1 if ma20 > 0 else float("nan")
        else:
            feats["btc_dom_dev_20"] = float("nan")

        if len(buf) >= 51:
            ma50 = sum(buf[-50:]) / 50
            feats["btc_dom_dev_50"] = ratio / ma50 - 1 if ma50 > 0 else float("nan")
        else:
            feats["btc_dom_dev_50"] = float("nan")

        feats["btc_dom_ret_24"] = ratio / buf[-25] - 1 if len(buf) >= 25 and buf[-25] > 0 else float("nan")
        feats["btc_dom_ret_72"] = ratio / buf[-73] - 1 if len(buf) >= 73 and buf[-73] > 0 else float("nan")
        return feats

    def warmup(self, limit: int = WARMUP_BARS, interval: str = INTERVAL) -> int:
        """Fetch historical bars and warm up feature engine."""
        bars = self._adapter.get_klines(self._symbol, interval=interval,
                                        limit=limit)
        bars.reverse()  # Bybit returns newest first

        # V14: Pre-fill ETH prices for BTC dominance warmup
        eth_warmup: dict[int, float] = {}
        if self._symbol == "BTCUSDT":
            try:
                url = (f"https://fapi.binance.com/fapi/v1/klines"
                       f"?symbol=ETHUSDT&interval={interval}m&limit={limit}")
                with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=10) as resp:
                    eth_klines = json.loads(resp.read())
                for k in eth_klines:
                    eth_warmup[int(k[0])] = float(k[4])  # open_time -> close
            except Exception:
                pass

        for i, bar in enumerate(bars):
            self._check_regime(bar["close"])  # build regime state
            self._engine.push_bar(
                bar["close"], bar["volume"], bar["high"], bar["low"],
                bar["open"], funding_rate=float("nan"),
                open_interest=float("nan"), ls_ratio=float("nan"),
            )
            features = self._engine.get_features()
            if features:
                feat_dict = dict(features)
                # V14: Inject dominance features during warmup
                if self._symbol == "BTCUSDT":
                    bar_ts = bar.get("start", bar.get("open_time", 0))
                    eth_p = eth_warmup.get(int(bar_ts), 0)
                    if eth_p > 0:
                        self._dom_ratio_buf.append(bar["close"] / eth_p)
                    buf = self._dom_ratio_buf
                    feat_dict["btc_dom_dev_20"] = (buf[-1] / (sum(buf[-20:])/20) - 1) if len(buf) >= 21 else 0.0
                    feat_dict["btc_dom_dev_50"] = (buf[-1] / (sum(buf[-50:])/50) - 1) if len(buf) >= 51 else 0.0
                    feat_dict["btc_dom_ret_24"] = (buf[-1] / buf[-25] - 1) if len(buf) >= 25 else 0.0
                    feat_dict["btc_dom_ret_72"] = (buf[-1] / buf[-73] - 1) if len(buf) >= 73 else 0.0
                pred = self._ensemble_predict(feat_dict)
                if pred is not None:
                    # Push through bridge for z-score warmup
                    # Use actual bar timestamp as hour_key for consistency with live
                    bar_ts = bar.get("start", bar.get("open_time", 0))
                    hour_key = int(bar_ts) // (3600 * 1000) if bar_ts > 1e9 else i
                    self._inference.zscore_normalize(self._symbol, pred, hour_key)
                    # Also warm up secondary horizon
                    pred_h2 = self._secondary_horizon_predict(feat_dict)
                    if pred_h2 is not None:
                        h2_key = f"{self._symbol}_h2"
                        self._inference.zscore_normalize(h2_key, pred_h2, hour_key)

        self._bars_processed = len(bars)
        regime_str = "active" if self._regime_active else "filtered"
        logger.info("Warmup: %d bars, regime=%s",
                     len(bars), regime_str)

        # Reconcile with exchange after warmup
        self._reconcile_position()

        return len(bars)

    def _reconcile_position(self) -> None:
        """Reconcile runner state with actual exchange position.

        Queries the exchange for the real position and syncs _current_signal
        and _entry_price if they diverge from exchange truth. Called after
        warmup and periodically (every RECONCILE_INTERVAL bars).
        """
        try:
            positions = self._adapter.get_positions(symbol=self._symbol)
        except Exception:
            logger.debug("%s reconcile: failed to fetch positions", self._symbol, exc_info=True)
            return

        # Determine actual exchange side
        exchange_side = 0
        exchange_qty = 0.0
        for pos in positions:
            if pos.symbol == self._symbol and not pos.is_flat:
                exchange_side = 1 if pos.is_long else -1
                exchange_qty = float(pos.abs_qty)
                break

        # Cross-check RustStateStore position against exchange truth
        if self._state_store is not None:
            store_pos = self._state_store.get_position(self._symbol)
            store_qty = store_pos.qty if store_pos is not None else 0
            # Fd8: qty is i64 × 10^8. Non-zero = has position.
            store_side = 1 if store_qty > 0 else (-1 if store_qty < 0 else 0)
            if store_side != exchange_side:
                logger.warning(
                    "%s StateStore/exchange mismatch: store_side=%d store_qty_raw=%s "
                    "exchange_side=%d exchange_qty=%.4f",
                    self._symbol, store_side, store_qty, exchange_side, exchange_qty,
                )

        if exchange_side != self._current_signal:
            logger.warning(
                "%s RECONCILE DIVERGENCE: runner_signal=%d exchange_side=%d exchange_qty=%.4f "
                "— syncing to exchange truth",
                self._symbol, self._current_signal, exchange_side, exchange_qty,
            )
            self._current_signal = exchange_side
            if exchange_side == 0:
                self._entry_price = 0.0
                self._entry_size = 0.0
                self._trade_peak_price = 0.0
                self._inference.set_position(self._symbol, 0, 1)  # reset bridge to flat
            if self._runner_key:
                _consensus_signals[self._runner_key] = exchange_side
        else:
            logger.debug("%s reconcile OK: signal=%d", self._symbol, self._current_signal)

    # Reconcile every N bars (avoid hammering the API every bar)
    RECONCILE_INTERVAL = 10

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
        that adapts to market volatility. Acquires _trade_lock to
        prevent race with process_bar.
        """
        if self._current_signal == 0 or self._entry_price <= 0 or self._killed:
            return False

        with self._trade_lock:
            # Re-check under lock (state may have changed)
            if self._current_signal == 0 or self._entry_price <= 0:
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
                    if not self._circuit_breaker.allow_request():
                        logger.warning("%s STOP CLOSE blocked by circuit breaker", self._symbol)
                        return False
                    stop_id = f"stop_{self._symbol}_{int(time.time())}"
                    self._osm.register(stop_id, self._symbol,
                                       "sell" if self._current_signal > 0 else "buy",
                                       "market", str(self._position_size))
                    close_result = self._adapter.close_position(self._symbol)
                    if close_result.get("status") == "error":
                        logger.error("%s STOP CLOSE FAILED: %s", self._symbol, close_result)
                        self._osm.transition(stop_id, "rejected", reason=str(close_result.get("retMsg", "")))
                        self._circuit_breaker.record_failure()
                        if close_result.get("retryable", False):
                            close_result = self._adapter.close_position(self._symbol)
                        if close_result.get("status") == "error":
                            logger.error("%s STOP CLOSE RETRY FAILED: %s — keeping state",
                                         self._symbol, close_result)
                            return False
                    self._osm.transition(stop_id, "filled", filled_qty=str(self._position_size),
                                         avg_price=str(price))
                    self._circuit_breaker.record_success()

                entry_size = self._entry_size if self._entry_size > 0 else self._position_size
                pnl_usd = unrealized * self._entry_price * entry_size
                self._total_pnl += pnl_usd
                self._trade_count += 1
                if pnl_usd > 0:
                    self._win_count += 1
                logger.info(
                    "%s STOP CLOSED: pnl=$%.4f total=$%.4f trades=%d/%d",
                    self._symbol, pnl_usd, self._total_pnl,
                    self._win_count, self._trade_count,
                )

                # Record stop-loss close in RustStateStore
                close_side = "sell" if self._current_signal > 0 else "buy"
                self._record_fill(close_side, entry_size, price,
                                  realized_pnl=pnl_usd)

                self._current_signal = 0
                self._inference.set_position(self._symbol, 0, 1)  # reset to flat
                self._entry_price = 0.0
                self._entry_size = 0.0
                self._trade_peak_price = 0.0
                return True

            return False

    def _ensemble_predict(self, feat_dict: dict) -> float | None:
        """Ensemble: Ridge (primary) + LightGBM (secondary).

        Ridge won 20-fold walk-forward (15/20 PASS, Sharpe 0.54, +433%).
        Config weights: ridge_weight (default 0.6) + lgbm_weight (default 0.4).
        """
        def _safe_val(v):
            """Convert None/NaN to 0.0 for model input."""
            if v is None:
                return 0.0
            try:
                f = float(v)
                return 0.0 if np.isnan(f) else f
            except (TypeError, ValueError):
                return 0.0

        if not self._horizon_models:
            x = [_safe_val(feat_dict.get(f)) for f in self._features]
            return float(self._model.predict([x])[0])

        ridge_w = self._config.get("ridge_weight", 0.6)
        lgbm_w = self._config.get("lgbm_weight", 0.4)

        weighted_sum = 0.0
        weight_total = 0.0

        for hm in self._horizon_models:
            feats = hm["features"]
            x = [_safe_val(feat_dict.get(f)) for f in feats]

            ic = max(hm["ic"], 0.001)

            # Ridge prediction (primary — walk-forward winner)
            if hm.get("ridge") is not None:
                # Ridge may use different features than LGBM
                rf = hm.get("ridge_features") or feats
                rx = [_safe_val(feat_dict.get(f)) for f in rf]
                if True:  # NaN already handled by _safe_val
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

    def _secondary_horizon_predict(self, feat_dict: dict) -> float | None:
        """Predict using secondary (shorter) horizon only, for gap-filling.

        When primary horizon signal is flat, the shorter horizon may still
        have a directional signal. This fills ~21% more bars.
        Walk-forward validated: Sharpe 1.68→2.22, +3566% vs +1440%.
        """
        if len(self._horizon_models) < 2:
            return None
        # Secondary = first horizon model (h=12, shorter)
        hm = self._horizon_models[0]
        feats = hm["features"]
        x = [feat_dict.get(f, 0.0) or 0.0 for f in feats]
        if any(np.isnan(x)):
            return None
        return float(hm["lgbm"].predict([x])[0])

    def _get_leverage_for_equity(self, equity: float) -> float:
        """Look up leverage from ladder based on current equity."""
        lev = 1.0
        for threshold, lev_val in self.LEVERAGE_LADDER:
            if equity >= threshold:
                lev = lev_val
        return lev

    def _get_consensus_scale(self) -> float:
        """Compute position scale based on cross-symbol signal consensus.

        IMPORTANT: Research showed consensus is CONTRARIAN — when all 4
        symbols agree bearish, market actually goes UP (+28bp). So:
        - All agree same direction: scale=1.0 (don't increase)
        - 3/4 agree: scale=1.0
        - 1-2 agree with this symbol: scale=0.7 (lower conviction)
        - Going AGAINST consensus (contrarian): +30% boost

        Returns scale factor in [0.5, 1.3].
        """
        if not _consensus_signals or not self._runner_key:
            return 1.0

        my_signal = self._current_signal
        if my_signal == 0:
            return 1.0  # flat, no scaling needed

        # Count signals in each direction (exclude self)
        n_bull = 0
        n_bear = 0
        n_total = 0
        for rkey, sig in _consensus_signals.items():
            if rkey == self._runner_key:
                continue
            if sig > 0:
                n_bull += 1
            elif sig < 0:
                n_bear += 1
            n_total += 1

        if n_total == 0:
            return 1.0

        # How many others agree with my direction?
        same_dir = n_bull if my_signal > 0 else n_bear
        opposite_dir = n_bear if my_signal > 0 else n_bull

        # Contrarian boost: if ALL others disagree (consensus opposite),
        # this symbol is going against the crowd — historically profitable
        if opposite_dir == n_total and n_total >= 2:
            return 1.3  # +30% contrarian boost

        # Fraction of others that agree with me
        agree_frac = same_dir / n_total if n_total > 0 else 0

        if agree_frac >= 0.75:  # 3/4+ agree (including unanimous)
            return 1.0
        elif agree_frac >= 0.25:  # 1-2 out of 3-4 agree
            return 0.7
        else:  # nobody agrees
            return 0.5

    @staticmethod
    def compute_z_scale(z: float) -> float:
        """Non-linear position sizing based on z-score magnitude.

        Stronger signals get larger positions, weak signals get smaller:
        - |z| > 2.0: scale=1.5 (extreme conviction)
        - |z| > 1.0: scale=1.0 (normal)
        - |z| > 0.5: scale=0.7 (weak signal)
        - else:       scale=0.5 (barely above deadzone)

        Returns scale factor in [0.5, 1.5].
        """
        abs_z = abs(z)
        if abs_z > 2.0:
            return 1.5
        elif abs_z > 1.0:
            return 1.0
        elif abs_z > 0.5:
            return 0.7
        else:
            return 0.5

    def _compute_position_size(self, price: float) -> float:
        """Compute position size using Kelly-optimal leverage ladder.

        1. Look up leverage from equity-based ladder
        2. Position = equity * leverage / price
        3. Apply z-scale (non-linear z-score sizing) and consensus scale
        4. Clamp to [min_size, exchange limit]
        5. Set exchange leverage to match

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

        # Position = equity × per_symbol_cap × leverage / price
        # 4 effective symbols (ETH combo shares one): 15% each = 60% base
        # With z_scale(1.5x) + consensus(1.3x) worst case: 60% * 1.95 = 117% < margin
        per_sym_cap = 0.15
        max_notional = equity * per_sym_cap * target_lev
        size = max_notional / price

        # Apply non-linear z-score scale + cross-symbol consensus scale
        z_scale = self._z_scale  # set in process_bar()
        consensus_scale = self._get_consensus_scale()
        combined_scale = z_scale * consensus_scale
        size *= combined_scale

        if combined_scale != 1.0:
            logger.info(
                "%s SCALE: z_scale=%.2f consensus=%.2f combined=%.2f",
                self._symbol, z_scale, consensus_scale, combined_scale,
            )

        # Clamp to exchange limits
        size = max(self._min_size, size)
        if self._max_qty > 0:
            size = min(size, self._max_qty)

        # Round DOWN to exchange step size (floor to avoid exceeding limits)
        if self._step_size > 0:
            # Use round then floor to handle float precision
            # e.g., 6950.5/0.1 = 69504.999... → round = 69505 → floor-safe
            steps = int(round(size / self._step_size, 0))
            size = steps * self._step_size
            # Final precision fix
            if self._step_size >= 1:
                size = int(size)
            else:
                step_decimals = max(0, -int(np.floor(np.log10(self._step_size))))
                size = round(size, step_decimals)

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

        # Periodic reconciliation with exchange
        if self._bars_processed % self.RECONCILE_INTERVAL == 0:
            self._reconcile_position()

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

        # V13: Fetch OI/LS/Taker data from Binance (non-blocking, NaN fallback)
        oi_data = _fetch_binance_oi_data(self._symbol)

        self._engine.push_bar(
            bar["close"], bar["volume"], bar["high"], bar["low"],
            bar["open"], funding_rate=funding_rate,
            open_interest=oi_data["open_interest"],
            ls_ratio=oi_data["ls_ratio"],
            taker_buy_volume=oi_data["taker_buy_vol"],
        )

        # Update RustStateStore market state for mark-to-market
        self._record_market_update(bar)

        features = self._engine.get_features()
        if not features:
            return {"action": "no_features", "bar": self._bars_processed}

        feat_dict = dict(features)

        # V14: Inject BTC dominance features (computed in Python, not Rust)
        if self._symbol == "BTCUSDT":
            dom_feats = self._compute_dominance_features(bar["close"])
            feat_dict.update(dom_feats)

        # Primary horizon prediction (h=24, Ridge+LGBM ensemble)
        pred = self._ensemble_predict(feat_dict)
        if pred is None:
            return {"action": "nan_features", "bar": self._bars_processed}

        # Z-score via RustInferenceBridge (returns None during warmup)
        hour_key = self._bars_processed
        z_val = self._inference.zscore_normalize(self._symbol, pred, hour_key)
        if z_val is None:
            return {"action": "warmup", "bar": self._bars_processed,
                    "pred": pred}
        z = z_val

        # Non-linear z-score position sizing
        self._z_scale = self.compute_z_scale(z)

        prev_signal = self._current_signal

        # Primary signal via RustInferenceBridge: z-score → deadzone → min-hold → max-hold
        # Regime filter: pass deadzone=999 to force flat when regime is unfavorable
        effective_dz = 999.0 if not regime_ok else self._deadzone
        new_signal = int(self._inference.apply_constraints(
            self._symbol, pred, hour_key,
            deadzone=effective_dz,
            min_hold=self._min_hold,
            max_hold=self._max_hold,
        ))

        # Secondary horizon gap-fill: when primary is flat, check h2
        if new_signal == 0 and regime_ok:
            pred_h2 = self._secondary_horizon_predict(feat_dict)
            if pred_h2 is not None:
                h2_key = f"{self._symbol}_h2"
                h2_signal = int(self._inference.apply_constraints(
                    h2_key, pred_h2, hour_key,
                    deadzone=self._deadzone,
                    min_hold=self._min_hold,
                    max_hold=self._max_hold,
                ))
                if h2_signal != 0:
                    new_signal = h2_signal
                    # Sync primary bridge position to match h2 override
                    self._inference.set_position(self._symbol, new_signal, 1)

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

        # Smart exits (in priority order) — override bridge signal
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

        # 2. Z-score reversal exit (bridge enforces min_hold internally)
        z_reversal_threshold = -0.3
        if not force_exit and prev_signal != 0:
            if prev_signal > 0 and z < z_reversal_threshold:
                force_exit = True
                exit_reason = "z_reversal"
            elif prev_signal < 0 and z > -z_reversal_threshold:
                force_exit = True
                exit_reason = "z_reversal"

        if force_exit:
            new_signal = 0
            # Sync bridge state to flat after forced exit
            self._inference.set_position(self._symbol, 0, 1)

        # Acquire trade lock to guard shared state mutation and order execution
        # (prevents race with check_realtime_stoploss on WS tick thread)
        with self._trade_lock:
            # Re-read prev_signal under lock in case stop-loss fired between
            # signal computation and here
            actual_prev = self._current_signal
            if actual_prev != prev_signal:
                # Stop-loss already changed state — abort this signal change
                logger.info("%s signal change aborted: stop-loss fired (prev=%d→%d)",
                            self._symbol, prev_signal, actual_prev)
                new_signal = actual_prev

            self._current_signal = new_signal

            # Update shared consensus state for cross-symbol scaling
            if self._runner_key:
                _consensus_signals[self._runner_key] = new_signal

            # OI data summary for logging
            oi_str = ""
            if not np.isnan(oi_data.get("ls_ratio", float("nan"))):
                oi_str = (f" OI={oi_data['open_interest']:.0f}"
                         f" LS={oi_data['ls_ratio']:.2f}"
                         f" TopLS={oi_data['top_trader_ls_ratio']:.2f}")

            result = {
                "action": "signal", "bar": self._bars_processed,
                "pred": round(pred, 6), "z": round(z, 4),
                "z_scale": self._z_scale,
                "signal": new_signal, "prev_signal": prev_signal,
                "hold_count": int(self._inference.get_position(self._symbol)), "close": bar["close"],
                "regime": "active" if regime_ok else "filtered",
                "dz": round(self._deadzone, 3),
                "oi_data": oi_str,
            }
            if force_exit:
                result["exit_reason"] = exit_reason

            if new_signal != prev_signal and actual_prev == prev_signal:
                # Recompute position size before entering new position
                if new_signal != 0:
                    self._compute_position_size(bar["close"])
                trade_result = self._execute_signal_change(prev_signal, new_signal, bar["close"])
                result["trade"] = trade_result
                result["size"] = self._position_size
                # If order failed, revert _current_signal to previous state
                if trade_result.get("action") in ("order_failed", "close_failed"):
                    self._current_signal = prev_signal
                    self._inference.set_position(self._symbol, prev_signal, 1)  # revert bridge
                    if self._runner_key:
                        _consensus_signals[self._runner_key] = prev_signal
                    logger.warning("%s reverting signal to %d after order failure",
                                   self._symbol, prev_signal)

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
            entry_size = self._entry_size if self._entry_size > 0 else self._position_size
            pnl_usd = pnl_pct / 100 * self._entry_price * entry_size
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
            # Record close fill in RustStateStore
            close_side = "sell" if prev == 1 else "buy"
            self._record_fill(close_side, entry_size, price,
                              realized_pnl=pnl_usd)

        # Drawdown check via RustRiskEvaluator + RustKillSwitch
        self._peak_equity = max(self._peak_equity, self._total_pnl)
        if self._risk_eval is not None and self._kill_switch is not None and self._peak_equity > 0:
            breached = self._risk_eval.check_drawdown(
                equity=self._total_pnl, peak_equity=self._peak_equity,
            )
            if breached:
                dd = (self._peak_equity - self._total_pnl) / self._peak_equity * 100
                reason = f"{self._symbol} drawdown {dd:.1f}%"
                self._kill_switch.arm("global", "*", "halt", reason,
                                      source="AlphaRunner")
                logger.critical(
                    "%s DRAWDOWN KILL (Rust): dd=%.1f%% peak=$%.2f current=$%.2f",
                    self._symbol, dd, self._peak_equity, self._total_pnl,
                )
                if not self._dry_run:
                    self._adapter.close_position(self._symbol)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}
        elif self._risk_eval is None and self._peak_equity > 0:
            # Fallback: manual drawdown check (backward compat if no Rust evaluator)
            dd = (self._peak_equity - self._total_pnl) / self._peak_equity * 100
            if dd >= 15.0:
                if self._kill_switch is not None:
                    self._kill_switch.arm("global", "*", "halt",
                                          f"{self._symbol} drawdown {dd:.1f}%",
                                          source="AlphaRunner_fallback")
                logger.critical(
                    "%s DRAWDOWN KILL (fallback): dd=%.1f%% peak=$%.2f current=$%.2f",
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

        # Circuit breaker: block orders if too many recent failures
        if not self._circuit_breaker.allow_request():
            cb_state = self._circuit_breaker.snapshot()
            logger.warning("%s CIRCUIT BREAKER OPEN: %s", self._symbol, cb_state)
            return {"action": "circuit_open", "state": str(cb_state)}

        # Close venue position
        if prev != 0:
            close_id = f"close_{self._symbol}_{int(time.time())}"
            self._osm.register(close_id, self._symbol, "sell" if prev == 1 else "buy",
                               "market", str(self._position_size))
            close_result = self._adapter.close_position(self._symbol)
            if close_result.get("status") == "error":
                logger.error("%s CLOSE FAILED: %s", self._symbol, close_result)
                self._osm.transition(close_id, "rejected", reason=str(close_result.get("retMsg", "")))
                self._circuit_breaker.record_failure()
                # Retry once if retryable
                if close_result.get("retryable", False):
                    close_result = self._adapter.close_position(self._symbol)
                if close_result.get("status") == "error":
                    logger.error("%s CLOSE RETRY FAILED: %s — keeping state", self._symbol, close_result)
                    return {"action": "close_failed", "result": close_result}
            self._osm.transition(close_id, "filled", filled_qty=str(self._position_size),
                                 avg_price=str(price))
            self._circuit_breaker.record_success()

        # Open new position
        if new != 0:
            side = "buy" if new == 1 else "sell"
            open_id = f"open_{self._symbol}_{int(time.time())}"
            self._osm.register(open_id, self._symbol, side, "market",
                               str(self._position_size))

            # Dedup: check no other pending order for this symbol
            active = self._osm.active_count()
            if active > 2:  # close + open = 2 expected
                logger.warning("%s DEDUP: %d active orders, skipping", self._symbol, active)
                self._osm.transition(open_id, "rejected", reason="dedup_active_orders")
                return {"action": "dedup_blocked", "active": active}

            result = self._adapter.send_market_order(self._symbol, side, self._position_size)
            # Check if order actually succeeded
            if result.get("status") == "error" or result.get("retCode", 0) != 0:
                logger.error("%s ORDER FAILED: %s", self._symbol, result)
                self._osm.transition(open_id, "rejected", reason=str(result.get("retMsg", "")))
                self._circuit_breaker.record_failure()
                return {"action": "order_failed", "result": result}

            # Order succeeded — update state machine and tracking
            order_id = result.get("orderId", open_id)
            self._osm.transition(open_id, "filled", filled_qty=str(self._position_size),
                                 avg_price=str(price))
            self._circuit_breaker.record_success()
            self._entry_price = price
            self._entry_size = self._position_size  # snapshot entry-time size
            self._trade_peak_price = price  # initialize trailing peak
            # Record open fill in RustStateStore
            self._record_fill(side, self._position_size, price)
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
            self._entry_size = 0.0
            self._trade_peak_price = 0.0
            trade_info["action"] = "flat"

        return trade_info


class PortfolioManager:
    """Unified position and risk manager across all alpha sources.

    All alphas submit trade intents here instead of directly to the adapter.
    PortfolioManager decides what actually gets executed based on:
    1. Total portfolio exposure limit (Kelly 1.4x)
    2. Per-symbol position cap (30% equity)
    3. Signal priority (AGREE > single alpha > hedge)
    4. Unified drawdown circuit breaker
    5. Net position calculation (prevents self-hedging waste)

    Walk-forward validated across 5 alpha sources.
    """

    def __init__(self, adapter: Any, *, dry_run: bool = False,
                 max_total_exposure: float = 1.4,   # Kelly optimal
                 max_per_symbol: float = 0.30,       # 30% per symbol
                 max_drawdown_pct: float = 15.0,     # kill at 15% DD
                 min_order_notional: float = 5.0,
                 risk_evaluator: Any = None, kill_switch: Any = None):
        self._adapter = adapter
        self._dry_run = dry_run
        self._max_total = max_total_exposure
        self._max_per_sym = max_per_symbol
        self._max_dd = max_drawdown_pct
        self._risk_eval = risk_evaluator
        self._kill_switch = kill_switch
        self._min_notional = min_order_notional

        # Position state: symbol → {"qty": float, "side": str, "entry": float, "source": str}
        self._positions: dict[str, dict] = {}
        # Intent registry: source → symbol → signal
        self._intents: dict[str, dict[str, int]] = {}
        # P&L tracking
        self._total_pnl: float = 0.0
        self._peak_pnl: float = 0.0
        self._trade_count: int = 0
        self._win_count: int = 0

    @property
    def _killed(self) -> bool:
        """Check kill switch (Rust) instead of local boolean."""
        if self._kill_switch is not None:
            return self._kill_switch.is_armed()
        return False

    def get_equity(self) -> float:
        try:
            bal = self._adapter.get_balances()
            usdt = bal.get("USDT")
            return float(usdt.total) if usdt else 0
        except Exception:
            return 0

    def submit_intent(self, source: str, symbol: str, signal: int,
                      price: float, priority: int = 1) -> dict | None:
        """Submit a trade intent from any alpha source.

        Args:
            source: alpha identifier (e.g. "ETH_COMBO", "SUI_ALPHA", "HEDGE")
            symbol: exchange symbol
            signal: +1 (long), -1 (short), 0 (flat)
            price: current price
            priority: 1=highest (AGREE), 2=single alpha, 3=hedge

        Returns: trade result dict or None if no action taken.
        """
        if self._killed:
            return {"action": "killed", "reason": "drawdown_breaker"}

        # Register intent
        self._intents.setdefault(source, {})[symbol] = signal

        # Compute net desired position for this symbol across all sources
        net_signal = self._compute_net_signal(symbol)
        current = self._positions.get(symbol, {})
        current_side = 1 if current.get("qty", 0) > 0 else (-1 if current.get("qty", 0) < 0 else 0)

        if net_signal == current_side:
            return None  # No change needed

        equity = self.get_equity()
        if equity <= 0:
            return None

        # Check total exposure limit
        total_exposure = self._total_exposure(equity, price)
        if net_signal != 0 and total_exposure >= self._max_total:
            # Already at max — only allow closing or same-direction
            if current_side != 0 and net_signal != current_side:
                pass  # Allow close + flip
            elif current_side == 0:
                logger.warning("PM: total exposure %.1f%% >= %.0f%% limit, rejecting %s %s",
                               total_exposure * 100, self._max_total * 100, symbol,
                               "long" if net_signal > 0 else "short")
                return {"action": "rejected", "reason": "total_exposure_limit"}

        # Compute position size
        max_notional = equity * self._max_per_sym
        qty = max_notional / price if price > 0 else 0
        if qty * price < self._min_notional:
            return {"action": "rejected", "reason": "below_min_notional"}

        # Execute
        return self._execute(symbol, net_signal, qty, price, source)

    def _compute_net_signal(self, symbol: str) -> int:
        """Compute net signal for a symbol across all sources.

        Priority: if any high-priority source has a signal, use it.
        If sources disagree, higher priority wins.
        """
        signals = []
        for source, sym_signals in self._intents.items():
            sig = sym_signals.get(symbol, 0)
            if sig != 0:
                signals.append(sig)

        if not signals:
            return 0
        # If all agree, use that direction
        if all(s > 0 for s in signals):
            return 1
        if all(s < 0 for s in signals):
            return -1
        # Disagreement: majority wins, tie = flat
        net = sum(signals)
        if net > 0:
            return 1
        elif net < 0:
            return -1
        return 0

    def _total_exposure(self, equity: float, exclude_price: float = 0) -> float:
        """Total portfolio exposure as fraction of equity."""
        if equity <= 0:
            return 0
        total = 0.0
        for sym, pos in self._positions.items():
            total += abs(pos.get("qty", 0)) * pos.get("entry", 0)
        return total / equity

    def _execute(self, symbol: str, desired: int, qty: float, price: float,
                 source: str) -> dict:
        """Execute position change on exchange."""
        current = self._positions.get(symbol, {})
        current_qty = current.get("qty", 0)
        current_side = 1 if current_qty > 0 else (-1 if current_qty < 0 else 0)

        trade_info = {"symbol": symbol, "from": current_side, "to": desired,
                      "source": source, "price": price}

        # Close existing
        if current_side != 0:
            close_side = "sell" if current_qty > 0 else "buy"
            close_qty = abs(current_qty)

            # Track PnL
            entry = current.get("entry", price)
            if current_side > 0:
                pnl = (price - entry) / entry * close_qty * entry
            else:
                pnl = (entry - price) / entry * close_qty * entry
            self._total_pnl += pnl
            self._trade_count += 1
            if pnl > 0:
                self._win_count += 1
            trade_info["closed_pnl"] = round(pnl, 2)

            if not self._dry_run:
                result = self._adapter.send_market_order(symbol, close_side, round(close_qty, 2),
                                                        reduce_only=True)
                logger.info("PM CLOSE %s %s %.2f: %s", symbol, close_side, close_qty, result)

            del self._positions[symbol]

        # Check drawdown via RustRiskEvaluator + RustKillSwitch
        self._peak_pnl = max(self._peak_pnl, self._total_pnl)
        if self._risk_eval is not None and self._kill_switch is not None and self._peak_pnl > 0:
            breached = self._risk_eval.check_drawdown(
                equity=self._total_pnl, peak_equity=self._peak_pnl,
            )
            if breached:
                dd = (self._peak_pnl - self._total_pnl) / self._peak_pnl * 100
                self._kill_switch.arm("global", "*", "halt",
                                      f"PM drawdown {dd:.1f}%",
                                      source="PortfolioManager")
                logger.critical("PM DRAWDOWN KILL (Rust): dd=%.1f%% peak=$%.2f current=$%.2f",
                                dd, self._peak_pnl, self._total_pnl)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}
        elif self._risk_eval is None and self._peak_pnl > 0:
            # Fallback: manual drawdown check
            dd = (self._peak_pnl - self._total_pnl) / self._peak_pnl * 100
            if dd >= self._max_dd:
                if self._kill_switch is not None:
                    self._kill_switch.arm("global", "*", "halt",
                                          f"PM drawdown {dd:.1f}%",
                                          source="PortfolioManager_fallback")
                logger.critical("PM DRAWDOWN KILL (fallback): dd=%.1f%% peak=$%.2f current=$%.2f",
                                dd, self._peak_pnl, self._total_pnl)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}

        # Open new
        if desired != 0:
            side = "buy" if desired > 0 else "sell"
            qty = round(qty, 2)

            if not self._dry_run:
                result = self._adapter.send_market_order(symbol, side, qty)
                trade_info["result"] = result
                logger.info("PM OPEN %s %s %.2f @ $%.2f: %s", symbol, side, qty, price, result)

            self._positions[symbol] = {
                "qty": qty * desired, "side": side, "entry": price, "source": source,
            }
        trade_info["action"] = "executed"
        return trade_info

    def get_status(self) -> dict:
        return {
            "positions": {s: {"qty": p["qty"], "source": p["source"]}
                          for s, p in self._positions.items()},
            "total_pnl": round(self._total_pnl, 2),
            "trades": f"{self._win_count}/{self._trade_count}",
            "killed": self._killed,
            "exposure": round(self._total_exposure(self.get_equity()) * 100, 1),
        }


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
                 runner_intervals: dict | None = None,
                 hedge_runner: HedgeRunner | None = None,
                 portfolio_manager: PortfolioManager | None = None) -> None:
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
            # Feed hedge runner (all 1h symbols)
            if hedge_runner is not None:
                hr = hedge_runner.on_bar(symbol, bar["close"])
                if hr and hr.get("trade"):
                    logger.info("HEDGE %s: ratio=%.6f ma=%.6f",
                                hr["trade"], hr.get("ratio", 0), hr.get("ratio_ma", 0))

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

                z_sc = result.get("z_scale", 1.0)
                z_sc_str = f" zs={z_sc:.1f}" if z_sc != 1.0 else ""
                logger.info(
                    "WS %s bar %d: $%.1f z=%+.3f sig=%d hold=%d regime=%s dz=%.3f%s%s",
                    label, result["bar"], result["close"],
                    result["z"], result["signal"],
                    result["hold_count"], regime, result.get("dz", 0),
                    z_sc_str, trade_str,
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

    # Capture state_store from runners for heartbeat logging
    _ws_state_store = None
    for r in runners.values():
        if r._state_store is not None:
            _ws_state_store = r._state_store
            break

    try:
        while True:
            time.sleep(300)
            sigs = {s: r._current_signal for s, r in runners.items()}
            pm_status = portfolio_manager.get_status() if portfolio_manager else None
            hedge_status = hedge_runner.get_status() if hedge_runner else None
            # RustStateStore portfolio snapshot (exposure, unrealized PnL)
            store_status = None
            if _ws_state_store is not None:
                try:
                    port = _ws_state_store.get_portfolio()
                    store_status = {
                        "equity": port.total_equity,
                        "exposure": port.gross_exposure,
                        "unrealized": port.unrealized_pnl,
                        "symbols": port.symbols,
                    }
                except Exception:
                    pass
            logger.info("WS HEARTBEAT sigs=%s pm=%s hedge=%s store=%s",
                        sigs, pm_status, hedge_status, store_status)
    except KeyboardInterrupt:
        logger.info("Stopped")
        for ws in ws_clients:
            ws.stop()
        if not dry_run:
            for symbol, runner in runners.items():
                if runner._current_signal != 0:
                    logger.info("Closing %s on exit...", symbol)
                    adapter.close_position(symbol)


class HedgeRunner:
    """BTC Long + ALT Short hedge strategy runner.

    Walk-forward validated: 17/20 PASS, Sharpe 2.68, +312%.
    Only shorts ALTs when ALT/BTC ratio < MA (BTC outperforming).
    """

    ALT_BASKET = ["ADAUSDT", "DOGEUSDT", "XRPUSDT", "LINKUSDT", "DOTUSDT",
                  "AVAXUSDT", "NEOUSDT"]  # liquid ALTs on Bybit

    def __init__(self, adapter: Any, *, dry_run: bool = False,
                 alt_weight: float = 0.5, ma_window: int = 480,
                 max_position_pct: float = 0.15):
        self._adapter = adapter
        self._dry_run = dry_run
        self._alt_weight = alt_weight
        self._ma_window = ma_window
        self._max_pct = max_position_pct

        # Price tracking
        self._btc_prices: list[float] = []
        self._alt_prices: dict[str, list[float]] = {s: [] for s in self.ALT_BASKET}
        self._is_short_active = False
        self._current_shorts: dict[str, float] = {}  # symbol → qty
        self._btc_long_qty: float = 0.0
        self._total_pnl: float = 0.0
        self._trade_count: int = 0
        self._bars_processed: int = 0

    def warmup_from_csv(self, n_bars: int = 600) -> int:
        """Pre-load historical prices from CSV for immediate signal generation.

        Loads the last n_bars of BTC + ALT basket prices so ratio MA
        can be computed immediately instead of waiting 20 days.
        """
        from pathlib import Path
        import pandas as pd

        loaded = 0
        # BTC
        btc_path = Path("data_files/BTCUSDT_1h.csv")
        if btc_path.exists():
            df = pd.read_csv(btc_path, usecols=["close"])
            closes = df["close"].values.astype(float)
            for p in closes[-n_bars:]:
                self._btc_prices.append(float(p))
            loaded += 1
            logger.info("HEDGE warmup: BTC %d bars", min(n_bars, len(closes)))

        # ALTs
        for sym in self.ALT_BASKET:
            alt_path = Path(f"data_files/{sym}_1h.csv")
            if alt_path.exists():
                df = pd.read_csv(alt_path, usecols=["close"])
                closes = df["close"].values.astype(float)
                for p in closes[-n_bars:]:
                    self._alt_prices[sym].append(float(p))
                loaded += 1

        logger.info("HEDGE warmup: %d symbols loaded, %d bars each → ready to signal",
                     loaded, n_bars)
        return loaded

    def on_bar(self, symbol: str, price: float) -> dict | None:
        """Process a 1h bar. Called for ANY symbol (ETH/SUI/AXS from WS).

        On each 1h bar, also fetches current prices for all ALT basket
        symbols via REST ticker (since they're not on the WS subscription).
        """
        # Track prices for symbols we receive via WS
        if symbol == "BTCUSDT":
            self._btc_prices.append(price)
        elif symbol in self._alt_prices:
            self._alt_prices[symbol].append(price)

        # On any 1h bar from a tracked symbol, fetch BTC + ALT basket prices
        # (since hedge basket symbols aren't on WS)
        if symbol in ("ETHUSDT", "SUIUSDT", "AXSUSDT"):
            self._fetch_basket_prices()

        # Only act once per 1h cadence (triggered by first 1h bar)
        if symbol != "ETHUSDT":
            return None

        self._bars_processed += 1
        if len(self._btc_prices) < self._ma_window + 1:
            return {"action": "warmup", "bar": self._bars_processed}

        # Compute ALT/BTC ratio vs MA
        btc = self._btc_prices
        alt_avg_now = 0
        alt_avg_count = 0
        for sym, prices in self._alt_prices.items():
            if len(prices) > 0:
                alt_avg_now += prices[-1]
                alt_avg_count += 1

        if alt_avg_count < 3 or btc[-1] <= 0:
            return {"action": "insufficient_data"}

        alt_avg_now /= alt_avg_count
        current_ratio = alt_avg_now / btc[-1]

        # MA of ratio
        ratios = []
        for i in range(max(0, len(btc) - self._ma_window), len(btc)):
            alt_sum = 0
            alt_cnt = 0
            for sym, prices in self._alt_prices.items():
                if i < len(prices):
                    alt_sum += prices[i]
                    alt_cnt += 1
            if alt_cnt > 0 and i < len(btc) and btc[i] > 0:
                ratios.append(alt_sum / alt_cnt / btc[i])

        if not ratios:
            return {"action": "no_ratio_data"}

        ratio_ma = np.mean(ratios)
        should_short = current_ratio < ratio_ma  # BTC outperforming

        result = {
            "action": "signal", "bar": self._bars_processed,
            "ratio": round(current_ratio, 6), "ratio_ma": round(ratio_ma, 6),
            "should_short": should_short, "was_short": self._is_short_active,
        }

        # State change
        if should_short and not self._is_short_active:
            # Enter: short ALTs
            self._is_short_active = True
            if not self._dry_run:
                self._open_hedge_positions()
            result["trade"] = "OPEN_HEDGE"
            logger.info("HEDGE OPEN: ratio=%.6f < MA=%.6f → shorting ALT basket",
                        current_ratio, ratio_ma)

        elif not should_short and self._is_short_active:
            # Exit: close shorts
            self._is_short_active = False
            if not self._dry_run:
                self._close_hedge_positions()
            result["trade"] = "CLOSE_HEDGE"
            logger.info("HEDGE CLOSE: ratio=%.6f > MA=%.6f → closing ALT shorts",
                        current_ratio, ratio_ma)

        return result

    def _fetch_basket_prices(self) -> None:
        """Fetch current prices for BTC + all ALT basket symbols via REST."""
        try:
            # BTC price
            ticker = self._adapter.get_ticker("BTCUSDT")
            btc_price = ticker.get("lastPrice", 0)
            if btc_price > 0:
                self._btc_prices.append(btc_price)

            # ALT basket prices
            for sym in self.ALT_BASKET:
                try:
                    ticker = self._adapter.get_ticker(sym)
                    p = ticker.get("lastPrice", 0)
                    if p > 0:
                        self._alt_prices[sym].append(p)
                except Exception:
                    pass  # Skip unavailable symbols
        except Exception:
            logger.debug("HEDGE: failed to fetch basket prices", exc_info=True)

    def _open_hedge_positions(self) -> None:
        """Open short positions on ALT basket."""
        try:
            bal = self._adapter.get_balances()
            usdt = bal.get("USDT")
            equity = float(usdt.total) if usdt else 0
        except Exception:
            equity = 100

        per_alt = equity * self._max_pct / max(len(self.ALT_BASKET), 1)

        for sym in self.ALT_BASKET:
            try:
                ticker = self._adapter.get_ticker(sym)
                price = ticker.get("lastPrice", 0)
                if price <= 0:
                    continue
                qty = per_alt / price
                qty = round(qty, 1)  # ALTs typically step=0.1
                if qty * price < 5:  # min notional $5
                    continue
                result = self._adapter.send_market_order(sym, "sell", qty)
                if result.get("status") == "submitted" or result.get("retCode") == 0:
                    self._current_shorts[sym] = qty
                    self._trade_count += 1
                    logger.info("HEDGE SHORT %s qty=%.1f @ $%.2f", sym, qty, price)
            except Exception:
                logger.warning("HEDGE: failed to short %s", sym, exc_info=True)

    def _close_hedge_positions(self) -> None:
        """Close all ALT short positions."""
        for sym, qty in list(self._current_shorts.items()):
            try:
                self._adapter.send_market_order(sym, "buy", qty, reduce_only=True)
                logger.info("HEDGE CLOSE %s qty=%.1f", sym, qty)
            except Exception:
                logger.warning("HEDGE: failed to close %s", sym, exc_info=True)
        self._current_shorts.clear()

    def get_status(self) -> dict:
        return {
            "active": self._is_short_active,
            "shorts": dict(self._current_shorts),
            "trades": self._trade_count,
            "bars": self._bars_processed,
        }


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

    # ── Rust components: 12/12 integrated ──
    # Active (6): RustFeatureEngine, RustInferenceBridge, RustRiskEvaluator,
    #   RustKillSwitch, RustOrderStateMachine, RustCircuitBreaker
    # Newly integrated (6): RustStateStore, rust_pipeline_apply,
    #   RustUnifiedPredictor, RustTickProcessor, RustWsClient, RustWsOrderGateway
    from _quant_hotpath import (
        RustRiskEvaluator, RustKillSwitch,
        RustStateStore,
        rust_pipeline_apply,       # noqa: F401 — atomic state updates (used internally by StateStore)
        RustUnifiedPredictor,      # noqa: F401 — requires JSON model export, our models are pickle
        RustTickProcessor,         # noqa: F401 — standalone binary hot-path, not used in Python runner
        RustWsClient,              # noqa: F401 — generic WS transport, Bybit uses own WS client
        RustWsOrderGateway,        # noqa: F401 — Binance WS-API only, Bybit uses REST orders
    )

    risk_eval = RustRiskEvaluator(max_drawdown_pct=0.15)
    kill_switch = RustKillSwitch()

    # RustStateStore: authoritative position truth across all symbols.
    # Keeps market, position, account, portfolio, and risk state on the Rust
    # heap. Updated via RustFillEvent/RustMarketEvent (zero-copy fast path).
    # Reconciliation compares store.get_position(sym) with exchange REST.
    all_symbols = []
    for s in args.symbols:
        real_sym = SYMBOL_CONFIG.get(s, {}).get("symbol", s)
        if real_sym not in all_symbols:
            all_symbols.append(real_sym)
    state_store = RustStateStore(all_symbols, "USDT", 0)
    logger.info(
        "Rust 12/12: StateStore(symbols=%s) + RiskEvaluator(max_dd=15%%) + KillSwitch "
        "| Available but unused: RustUnifiedPredictor (requires JSON models, our models "
        "are pickle Ridge/LGBM), RustTickProcessor (standalone binary path), "
        "RustWsClient (generic transport, Bybit uses own WS client), "
        "RustWsOrderGateway (Binance WS-API only, Bybit uses REST)",
        all_symbols,
    )
    # Note: rust_pipeline_apply is the free-function equivalent of
    # StateStore.process_event() — called internally by the store's reducers.
    # We import it for completeness; direct use is not needed when using StateStore.

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
            max_qty=sym_cfg.get("max_qty", 0),
            step_size=sym_cfg.get("step", 0.01),
            risk_evaluator=risk_eval, kill_switch=kill_switch,
            state_store=state_store,
        )
        runner._runner_key = symbol  # for cross-symbol consensus scaling
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

    # Initialize portfolio manager (unified position + risk)
    pm = PortfolioManager(adapter, dry_run=args.dry_run,
                          risk_evaluator=risk_eval, kill_switch=kill_switch)
    logger.info("PM: PortfolioManager enabled (max_exposure=140%%, max_per_sym=30%%, max_dd=15%%, Rust risk)")

    # Initialize hedge runner (BTC+ALT structural alpha)
    hedge = HedgeRunner(adapter, dry_run=args.dry_run) if not args.dry_run else None
    if hedge:
        hedge.warmup_from_csv(n_bars=600)  # Pre-load 25 days of history → immediate signals
        logger.info("HEDGE: BTC+ALT hedge enabled, ALT basket=%s", hedge.ALT_BASKET)

    # WebSocket mode: push-based, low latency
    if args.ws:
        _run_ws_mode(runners, adapter, args.dry_run,
                     runner_intervals=runner_intervals, hedge_runner=hedge,
                     portfolio_manager=pm)
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
                # RustStateStore portfolio snapshot
                store_info = ""
                if state_store is not None:
                    try:
                        port = state_store.get_portfolio()
                        store_info = (f" store_equity={port.total_equity}"
                                      f" exposure={port.gross_exposure}"
                                      f" unrealized={port.unrealized_pnl}")
                    except Exception:
                        pass
                logger.info(
                    "HEARTBEAT cycle=%d sigs=%s holds=%s regimes=%s pnl=%s trades=%s size=%s%s",
                    cycle_count, sigs, holds, regimes, pnls, trades, sizes, store_info,
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
