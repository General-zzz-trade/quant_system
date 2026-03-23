"""AlphaRunner — DEPRECATED. Use decision/modules/alpha.py (AlphaDecisionModule) instead.

New entry point: runner/alpha_main.py
This file is kept for backward compatibility with runner/main.py.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from runner.strategy_config import (
    INTERVAL, MAX_ORDER_NOTIONAL, WARMUP_BARS,
    _consensus_signals, get_max_order_notional,
)
from execution.balance_utils import get_total_and_free_balance
from data.oi_cache import BinanceOICache, _fetch_binance_oi_data  # noqa: F401
from execution.order_utils import reliable_close_position
from attribution.pnl_tracker import PnLTracker
from state.checkpoint import CheckpointManager
from runner.gates.evaluator import GateEvaluator
from portfolio.entry_scaler import EntryScaler
from infra.errors import (
    VenueError,
)

logger = logging.getLogger(__name__)

# Neutral fallback values for NaN features — 0.0 would be a directional signal
# for ratio/RSI features where the neutral value is not zero.
_NEUTRAL_DEFAULTS: dict[str, float] = {
    "ls_ratio": 1.0,
    "top_trader_ls_ratio": 1.0,
    "taker_buy_ratio": 0.5,
    "vol_regime": 1.0,
    "bb_pctb_20": 0.5,
    "rsi_14": 50.0,
    "rsi_6": 50.0,
}


class AlphaRunner:
    """Runs alpha strategy on Bybit with RustFeatureEngine + LightGBM."""

    # Leverage ladder: 10x across all tiers for demo signal validation.
    # For real money, scale down (edit this ladder).
    LEVERAGE_LADDER = [
        (0,      10.0),   # all tiers: 10x
    ]

    def __init__(self, adapter: Any, model_info: dict, symbol: str,
                 dry_run: bool = False, position_size: float = 0.001,
                 adaptive_sizing: bool = True, risk_per_trade: float = 0.10,
                 min_size: float = 0.01, max_size_pct: float = 10.00,
                 max_qty: float = 0, step_size: float = 0.01,
                 risk_evaluator: Any = None, kill_switch: Any = None,
                 state_store: Any = None, pnl_tracker: PnLTracker | None = None,
                 oi_cache: Any = None, start_oi_cache: bool = True):
        self._adapter = adapter
        self._symbol = symbol
        self._model = model_info["model"]  # primary (backward compat)
        self._features = model_info["features"]
        self._horizon_models = model_info.get("horizon_models", [])
        self._lgbm_xgb_weight = model_info.get("lgbm_xgb_weight", 0.5)
        self._config = model_info["config"]
        self._deadzone_base = model_info["deadzone"]
        self._deadzone = model_info["deadzone"]  # current (adapted)
        self._long_only = model_info.get("long_only", False)
        # Median realized vol per bar, calibrated from history per timeframe
        # Used for regime-adaptive deadzone scaling (vol_ratio = rv_20 / vol_median)
        version_str_init = model_info.get("config", {}).get("version", "")
        if "15m" in version_str_init:
            self._vol_median = 0.003   # 15m bars: ~half of 1h vol
        elif "4h" in version_str_init:
            self._vol_median = 0.013   # 4h bars: ~2x of 1h vol
        else:
            self._vol_median = 0.0063  # 1h bars (default, calibrated from ETH history)
        self._min_hold_base = model_info["min_hold"]
        self._max_hold_base = model_info["max_hold"]
        self._min_hold = self._min_hold_base
        self._max_hold = self._max_hold_base
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

        # Regime filter state — adaptive thresholds from rolling percentiles.
        # Replaces hardcoded thresholds that couldn't adapt to changing markets.
        # Each metric uses its own rolling buffer to compute p20 (bottom quintile = "dead").
        self._closes: list[float] = []
        self._rets: list[float] = []
        self._regime_active = True

        # Detect timeframe from model config
        version_str = model_info.get("config", {}).get("version", "")
        is_15m = "15m" in version_str
        is_4h = "4h" in version_str

        # MA window scales with timeframe to cover same real time (~20 days)
        if is_15m:
            self._ma_window = 1920       # 20 days at 15m
            self._ranging_window = 400   # ~4 days at 15m
        elif is_4h:
            self._ma_window = 120        # 20 days at 4h
            self._ranging_window = 25    # ~4 days at 4h
        else:
            self._ma_window = 480        # 20 days at 1h
            self._ranging_window = 100   # ~4 days at 1h

        # Rolling buffers for adaptive thresholds (populated during warmup + live)
        self._vol_history: list[float] = []      # rolling vol_20 values
        self._trend_history: list[float] = []    # rolling |trend| values
        self._eff_history: list[float] = []      # rolling efficiency values
        _adaptive_window = 200  # bars to compute percentile over (~8 days 1h, ~33 days 4h)
        self._adaptive_window = _adaptive_window

        # Seed thresholds (used before enough history; conservative = allow trading)
        self._vol_threshold = 0.0       # 0 = always pass vol check until adaptive kicks in
        self._trend_threshold = 0.0     # 0 = always pass trend check
        self._ranging_threshold = 0.0   # 0 = never ranging

        # Threading lock for shared state (_current_signal, _entry_price, orders)
        self._trade_lock = threading.Lock()

        # P&L tracking (unified via PnLTracker)
        self._entry_price: float = 0.0
        self._entry_size: float = 0.0   # position size at entry time (for PnL calc)
        self._pnl = pnl_tracker if pnl_tracker is not None else PnLTracker()

        # Dynamic weight: rolling trade PnL for Sharpe-based degradation
        self._recent_trade_pnls: list[float] = []  # last 30 trade PnLs
        self._dynamic_scale: float = 1.0  # multiplied into per_sym_cap

        # Rust risk evaluator + kill switch (shared across all runners)
        self._risk_eval = risk_evaluator
        self._kill_switch = kill_switch

        # RustStateStore: authoritative position truth (shared across all runners)
        # Tracks position qty, avg_price, account balance, portfolio exposure
        # via RustFillEvent processing on the Rust heap.
        self._state_store = state_store

        # Adaptive stop-loss state (regime-aware).
        # Base: 1.2x ATR initial, breakeven at 0.5×ATR, trail step 0.2×ATR
        # Regime scaling: high vol → wider stops, low vol → tighter stops
        self._atr_buffer: list[float] = []  # recent ATR values for adaptive stop
        self._trade_peak_price: float = 0.0  # highest favorable price since entry (for trailing)
        self._atr_stop_mult_base: float = 1.2
        self._atr_stop_mult: float = 1.2
        self._trail_atr_mult: float = 0.5    # trail activates after 0.5×ATR profit
        self._trail_step: float = 0.2        # trail distance: 0.2×ATR
        self._breakeven_atr: float = 0.5     # breakeven at 0.5×ATR profit
        self._vol_regime_history: list[float] = []

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

        # Composite regime detection (opt-in, default off)
        self._use_composite_regime = model_info.get("use_composite_regime", False)
        self._regime_params: Any = None  # Optional[RegimeParams]
        if self._use_composite_regime:
            from regime.composite import CompositeRegimeDetector
            from regime.param_router import RegimeParamRouter
            self._composite_detector = CompositeRegimeDetector()
            self._param_router = RegimeParamRouter()
        else:
            self._composite_detector = None
            self._param_router = None

        # Cross-symbol consensus + z-scale state
        self._runner_key: str = ""  # set by main() after construction
        self._z_scale: float = 1.0  # non-linear z-score position scale
        self._last_bar_time: float = 0.0  # time.time() of last process_bar call

        # Alpha expansion gates (scale position based on context features)
        from runner.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
        from runner.gates.liquidation_cascade_gate import LiquidationCascadeGate
        from runner.gates.carry_cost_gate import CarryCostGate
        from runner.gates.vpin_entry_gate import VPINEntryGate
        self._mtf_gate = MultiTFConfluenceGate()
        self._liq_gate = LiquidationCascadeGate()
        self._carry_gate = CarryCostGate()
        self._vpin_gate = VPINEntryGate()
        self._gate_scale: float = 1.0  # cumulative gate scale for logging
        self._gate_evaluator = GateEvaluator(
            liq_gate=self._liq_gate, mtf_gate=self._mtf_gate,
            carry_gate=self._carry_gate, vpin_gate=self._vpin_gate,
        )

        # Checkpoint manager (extracted)
        self._ckpt = CheckpointManager()

        # Entry scaler (extracted)
        self._entry_scaler_module = EntryScaler()

        # Online Ridge: incremental weight updates between weekly retrains
        self._online_ridge: Any = None
        self._online_ridge_features: list[str] = []
        self._last_close: float = 0.0  # for computing realized returns
        self._prev_feat_dict: dict | None = None  # store previous bar features to avoid look-ahead

        # Cross-market data cache (SPY/QQQ/VIX/TLT from Yahoo Finance daily)
        self._cross_market: dict[str, float] = {}
        self._cross_market_last_update: float = 0.0
        self._load_cross_market()

        # External data caches for push_bar injection
        # NOTE: Disabled until next planned retrain (injection changed prediction
        # distribution → z-score buffer mismatch → unreliable signals for 30 days).
        # Re-enable after auto-retrain rebuilds models with these features.
        # self._onchain_data: dict[str, float] = {}
        # self._fgi_value: float = float("nan")
        # self._spot_close: float = float("nan")
        # self._load_external_data()

        # Limit order entry: try maker (0 bps) before taker (4 bps) for new positions
        self._use_limit_entry: bool = True
        self._limit_entry_timeout: float = 30.0  # default; overridden by adaptive logic
        self._limit_entry_poll_interval: float = 2.0  # poll interval in seconds
        # Tick sizes for price improvement (Bybit USDT-M perps)
        self._tick_sizes: dict[str, float] = {"BTCUSDT": 0.10, "ETHUSDT": 0.01}
        # Try to fetch actual tick size from exchange
        try:
            info = self._adapter._client.get("/v5/market/instruments-info", {
                "category": "linear", "symbol": symbol,
            })
            items = info.get("result", {}).get("list", [])
            if items:
                tick = float(items[0].get("priceFilter", {}).get("tickSize", 0))
                if tick > 0:
                    self._tick_sizes[symbol] = tick
        except Exception as exc:
            logger.debug("%s tick_size fetch failed, using default: %s", symbol, exc)
        # Fill rate tracking: limit fills vs market fallbacks
        self._limit_fills: int = 0
        self._market_fallbacks: int = 0

        # 5m Position Scaler: scale entry size by Bollinger-band position.
        # Uses existing _closes buffer (bar-level) to compute short-term
        # overbought/oversold. Reduces size when chasing (buying high),
        # increases when entering on pullback.
        # Validated: MaxDD -50%, Sharpe -10% (net risk-adjusted improvement).
        self._entry_scaler_enabled: bool = True
        self._entry_scaler_window: int = 12  # BB lookback (bars at current TF)
        self._entry_scaler_last: float = 1.0  # last computed scale for logging

        # Background OI cache: fetches Binance OI/LS/Taker every 55s in a daemon thread.
        # Allow injection/disable in tests and non-networked environments.
        self._oi_cache = oi_cache if oi_cache is not None else BinanceOICache(self._symbol)
        if not hasattr(self._oi_cache, "get"):
            raise TypeError("oi_cache must provide get()")
        if start_oi_cache:
            if not hasattr(self._oi_cache, "start"):
                raise TypeError("oi_cache must provide start() when start_oi_cache=True")
            self._oi_cache.start()

    # Backward-compat properties: delegate to PnLTracker
    @property
    def _total_pnl(self) -> float:
        return self._pnl.total_pnl

    @property
    def _peak_equity(self) -> float:
        return self._pnl.peak_equity

    @property
    def _trade_count(self) -> int:
        return self._pnl.trade_count

    @property
    def _win_count(self) -> int:
        return self._pnl.win_count

    @property
    def _killed(self) -> bool:
        """Check kill switch (Rust) instead of local boolean."""
        if self._kill_switch is not None:
            return self._kill_switch.is_armed()
        return False

    def hot_reload_model(self, model_info: dict) -> bool:
        """Hot-reload model weights without restarting.

        Called on SIGHUP. Preserves feature engine state, checkpoint,
        position, and signal pipeline — only swaps model weights and config.
        Returns True if reload succeeded.
        """
        try:
            self._model = model_info["model"]
            self._features = model_info["features"]
            self._horizon_models = model_info.get("horizon_models", [])
            self._lgbm_xgb_weight = model_info.get("lgbm_xgb_weight", 0.5)
            new_cfg = model_info["config"]
            self._deadzone_base = model_info["deadzone"]
            self._deadzone = model_info["deadzone"]
            self._long_only = model_info.get("long_only", False)
            self._min_hold_base = model_info["min_hold"]
            self._max_hold_base = model_info["max_hold"]
            self._min_hold = self._min_hold_base
            self._max_hold = self._max_hold_base
            # Update config reference for version detection
            self._config = new_cfg
            logger.info(
                "%s HOT RELOAD: model=%s features=%d dz=%.2f mh=%d maxh=%d",
                self._runner_key or self._symbol,
                new_cfg.get("version", "?"), len(self._features),
                self._deadzone, self._min_hold, self._max_hold,
            )
            return True
        except Exception as exc:
            logger.error("%s hot reload FAILED: %s", self._symbol, exc)
            return False

    def _force_flat_local_state_locked(self) -> None:
        self._current_signal = 0
        self._entry_price = 0.0
        self._entry_size = 0.0
        self._trade_peak_price = 0.0
        self._inference.set_position(self._symbol, 0, 1)
        if self._runner_key:
            _consensus_signals[self._runner_key] = 0

    def force_flat_local_state(self) -> None:
        """Reset local runner state after a global kill or forced unwind."""
        with self._trade_lock:
            self._force_flat_local_state_locked()

    def stop(self) -> None:
        """Stop background resources owned by the runner."""
        stop = getattr(self._oi_cache, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception:
                logger.debug("%s OI cache stop failed", self._symbol, exc_info=True)

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

    def _load_cross_market(self) -> None:
        """Load T-1 cross-market features from daily CSV.

        Called on init and refreshed every 6h. Uses the previous day's row
        (date < today) to avoid look-ahead bias — US markets close at ~21:00
        UTC, so same-day data is not yet available for most crypto bars.
        """
        cm_path = Path("data_files/cross_market_daily.csv")
        if not cm_path.exists():
            return
        try:
            import pandas as pd
            df = pd.read_csv(cm_path, index_col="date", parse_dates=True)
            if df.empty:
                return
            # T-1 shift: use the most recent row where date < today (UTC)
            # to avoid look-ahead bias — US markets close at ~21:00 UTC,
            # so same-day data is not yet available for most crypto bars.
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).date()
            df_past = df[df.index.date < today]
            if df_past.empty:
                return
            latest = df_past.iloc[-1]
            self._cross_market = {}
            for col in ["spy_ret_1d", "qqq_ret_1d", "spy_ret_5d", "vix_level",
                         "tlt_ret_5d", "uso_ret_5d", "coin_ret_1d", "spy_extreme",
                         "treasury_10y_chg_5d", "eem_ret_5d", "gld_ret_5d",
                         "ethe_ret_1d", "gbtc_ret_1d", "ibit_ret_1d",
                         "bito_ret_1d", "gbtc_premium_dev",
                         "etha_ret_1d", "bitx_ret_1d", "biti_ret_1d",
                         "mara_ret_1d", "riot_ret_1d"]:
                val = latest.get(col)
                if val is not None and val == val:  # not NaN
                    self._cross_market[col] = float(val)
            self._cross_market_last_update = time.time()
            logger.info("%s cross-market loaded (T-1): %d features, date=%s",
                        self._symbol, len(self._cross_market),
                        df_past.index[-1].strftime("%Y-%m-%d"))
        except Exception as e:
            logger.debug("%s cross-market load failed: %s", self._symbol, e)

    def _load_external_data(self) -> None:
        """Load on-chain, FGI, and spot data for push_bar injection."""
        import pandas as pd
        sym = self._symbol

        # On-chain (daily, T-1 shifted)
        asset = "btc" if "BTC" in sym else "eth"
        try:
            oc_path = Path(f"data_files/{asset}_onchain_daily.csv")
            if oc_path.exists():
                oc = pd.read_csv(oc_path)
                last = oc.iloc[-1]
                self._onchain_data = {
                    "oc_flow_in": float(last.get("exchange_inflow", 0)),
                    "oc_flow_out": float(last.get("exchange_outflow", 0)),
                    "oc_addr": 0.0,  # not in this CSV format
                    "oc_tx": 0.0,
                }
                logger.debug("%s on-chain loaded: %s", sym, self._onchain_data)
        except Exception as e:
            logger.debug("%s on-chain load failed: %s", sym, e)

        # Fear & Greed index
        try:
            fgi_path = Path("data_files/fear_greed_index.csv")
            if fgi_path.exists():
                fgi = pd.read_csv(fgi_path)
                self._fgi_value = float(fgi.iloc[-1].get("value", float("nan")))
        except Exception as e:
            logger.debug("%s FGI load failed: %s", sym, e)

        # Spot close (for basis computation)
        try:
            spot_path = Path(f"data_files/{sym}_spot_1h.csv")
            if spot_path.exists():
                spot = pd.read_csv(spot_path)
                self._spot_close = float(spot["close"].iloc[-1])
        except Exception as e:
            logger.debug("%s spot close load failed: %s", sym, e)

    def _fetch_eth_price(self) -> float:
        """Fetch current ETH price from Binance for dominance computation."""
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/price?symbol=ETHUSDT"
            with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=3) as resp:
                return float(json.loads(resp.read()).get("price", 0))
        except Exception as e:
            logger.warning("%s _fetch_eth_price failed: %s", self._symbol, e)
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

    _CHECKPOINT_DIR = Path("data/runtime/checkpoints")

    def _save_checkpoint(self) -> None:
        """Save engine + inference state to disk for fast restart."""
        ckpt_name = self._runner_key or self._symbol
        extra = {
            "bars_processed": self._bars_processed,
            "regime_active": self._regime_active,
            "deadzone": float(self._deadzone),
            "atr_buffer": [float(x) for x in self._atr_buffer[-50:]],
            "dom_ratio_buf": [float(x) for x in self._dom_ratio_buf[-75:]],
            "closes": [float(x) for x in self._closes[-500:]],
            "rets": [float(x) for x in self._rets[-500:]],
        }
        self._ckpt.save(
            ckpt_name,
            self._engine.checkpoint(),
            self._inference.checkpoint(),
            extra=extra,
        )

    def _restore_checkpoint(self) -> bool:
        """Restore engine + inference state from disk. Returns True if restored."""
        ckpt_name = self._runner_key or self._symbol
        data = self._ckpt.restore(ckpt_name)
        if data is None:
            return False
        try:
            self._engine.restore_checkpoint(data["engine"])
            inference_data = data["inference"]
            if isinstance(inference_data, str):
                inference_data = json.loads(inference_data)
            self._inference.restore(inference_data)
            self._bars_processed = data.get("bars_processed", 0)
            self._regime_active = data.get("regime_active", True)
            self._deadzone = data.get("deadzone", self._deadzone)
            self._atr_buffer = data.get("atr_buffer", [])
            self._dom_ratio_buf = data.get("dom_ratio_buf", [])
            self._closes = data.get("closes", [])
            self._rets = data.get("rets", [])
            logger.info("%s checkpoint restored (bars=%d, regime=%s)",
                        self._symbol, self._bars_processed,
                        "active" if self._regime_active else "filtered")
            return True
        except Exception as e:
            logger.warning("%s checkpoint restore failed: %s — full warmup", self._symbol, e)
            return False

    def warmup(self, limit: int = WARMUP_BARS, interval: str = INTERVAL) -> int:
        """Fetch historical bars and warm up feature engine.

        Tries to restore from checkpoint first (instant). Falls back to
        full warmup from historical klines if no checkpoint exists.
        """
        # Try fast restore from checkpoint
        if self._restore_checkpoint():
            # Still reconcile with exchange
            if not self._dry_run:
                self._reconcile_position()
            return self._bars_processed

        bars = self._adapter.get_klines(self._symbol, interval=interval,
                                        limit=limit)
        bars.reverse()  # Bybit returns newest first

        # V14: Pre-fill ETH prices for BTC dominance warmup
        eth_warmup: dict[int, float] = {}
        if self._symbol == "BTCUSDT":
            try:
                # Convert Bybit interval to Binance format: "60" → "1h", "15" → "15m"
                _interval_map = {"60": "1h", "15": "15m", "240": "4h", "D": "1d"}
                binance_interval = _interval_map.get(interval, f"{interval}m")
                url = (f"https://fapi.binance.com/fapi/v1/klines"
                       f"?symbol=ETHUSDT&interval={binance_interval}&limit={limit}")
                with urlopen(Request(url, headers={"Accept": "application/json"}), timeout=10) as resp:
                    eth_klines = json.loads(resp.read())
                for k in eth_klines:
                    eth_warmup[int(k[0])] = float(k[4])  # open_time -> close
            except Exception as e:
                logger.warning("Failed to fetch ETH warmup klines for BTC dominance: %s", e)

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

        # Save checkpoint after successful warmup
        self._save_checkpoint()

        # Reconcile with exchange after warmup
        if not self._dry_run:
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
        except VenueError as exc:
            logger.error("VENUE_ERROR symbol=%s context=reconcile type=%s",
                         self._symbol, type(exc).__name__)
            return
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
            else:
                # Sync RustStateStore with exchange position on restart
                exchange_price = 0.0
                for pos in positions:
                    if pos.symbol == self._symbol and not pos.is_flat:
                        exchange_price = float(pos.entry_price) if pos.entry_price else 0.0
                        break

                # Fallback: if exchange doesn't return entry_price, use current
                # market price. This prevents entry_price=0 which would cause
                # trailing stop to immediately trigger (peak_price=0 bug).
                if exchange_price <= 0:
                    try:
                        tick = self._adapter.get_ticker(self._symbol)
                        exchange_price = float(tick.get("lastPrice", 0))
                    except Exception as e:
                        logger.warning("%s reconcile: failed to fetch ticker price: %s", self._symbol, e)
                if exchange_price <= 0 and self._closes:
                    exchange_price = self._closes[-1]

                if exchange_qty > 0 and exchange_price > 0:
                    self._record_fill(
                        "buy" if exchange_side == 1 else "sell",
                        exchange_qty, exchange_price,
                    )
                    # Sync entry state for stop-loss and PnL tracking
                    self._entry_price = exchange_price
                    self._entry_size = exchange_qty
                    self._trade_peak_price = exchange_price
                    self._position_size = exchange_qty
                    logger.info(
                        "%s RECONCILE: synced StateStore with exchange position side=%d qty=%.4f price=%.2f",
                        self._symbol, exchange_side, exchange_qty, exchange_price,
                    )
                elif exchange_qty > 0 and exchange_price <= 0:
                    # Critical: exchange has position but we cannot determine price.
                    # Set entry_price to a safe value to prevent stop-loss chaos.
                    logger.error(
                        "%s RECONCILE INVARIANT: exchange has qty=%.4f but no price available. "
                        "Setting signal to flat to avoid undefined stop-loss behavior.",
                        self._symbol, exchange_qty,
                    )
                    self._current_signal = 0
                    self._entry_price = 0.0
                    self._entry_size = 0.0
                    self._trade_peak_price = 0.0
                    self._inference.set_position(self._symbol, 0, 1)
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

        # Regime-aware stop scaling:
        # High vol → wider multiplier (let trade breathe in volatile market)
        # Low vol → tighter multiplier (capture small moves precisely)
        if len(self._atr_buffer) >= 20:
            recent_vol = np.mean(self._atr_buffer[-5:])
            median_vol = np.median(self._atr_buffer[-20:])
            if median_vol > 0:
                vol_ratio = recent_vol / median_vol
                # Scale ATR multiplier: ratio>1.5 → widen 30%, ratio<0.7 → tighten 20%
                regime_scale = np.clip(0.7 + 0.3 * vol_ratio, 0.8, 1.3)
                self._atr_stop_mult = self._atr_stop_mult_base * regime_scale

        # Update trade peak (best price since entry)
        # Safety: if peak was never initialized (e.g., after reconcile restart),
        # set it to entry price to avoid min(0, price)=0 for shorts.
        if self._trade_peak_price <= 0.0:
            self._trade_peak_price = entry
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
                    close_result = reliable_close_position(self._adapter, self._symbol)
                    if close_result["status"] == "failed":
                        logger.error("%s STOP CLOSE FAILED after retries — keeping state",
                                     self._symbol)
                        self._osm.transition(stop_id, "rejected", reason="reliable_close_failed")
                        self._circuit_breaker.record_failure()
                        return False
                    if not close_result.get("verified", True):
                        logger.warning("%s STOP CLOSE: position verification failed, proceeding",
                                       self._symbol)
                    self._osm.transition(stop_id, "filled", filled_qty=str(self._position_size),
                                         avg_price=str(price))
                    self._circuit_breaker.record_success()

                entry_size = self._entry_size if self._entry_size > 0 else self._position_size
                trade = self._pnl.record_close(
                    symbol=self._symbol, side=self._current_signal,
                    entry_price=self._entry_price, exit_price=price,
                    size=entry_size, reason="stop_loss",
                )
                logger.info(
                    "%s STOP CLOSED: pnl=$%.4f total=$%.4f trades=%d/%d",
                    self._symbol, trade["pnl_usd"], self._pnl.total_pnl,
                    self._pnl.win_count, self._pnl.trade_count,
                )

                # Record stop-loss close in RustStateStore
                close_side = "sell" if self._current_signal > 0 else "buy"
                self._record_fill(close_side, entry_size, price,
                                  realized_pnl=trade["pnl_usd"])

                self._current_signal = 0
                self._inference.set_position(self._symbol, 0, 1)  # reset to flat
                self._entry_price = 0.0
                self._entry_size = 0.0
                self._trade_peak_price = 0.0
                return True

            return False

    def enable_online_ridge(self, horizon_idx: int = 0) -> None:
        """Enable Online Ridge for incremental weight updates.

        Call after construction to activate. Loads static Ridge weights
        and begins RLS updates as new bars arrive.
        """
        from alpha.online_ridge import OnlineRidge
        if not self._horizon_models:
            logger.warning("%s no horizon models — cannot enable online ridge", self._symbol)
            return
        hm = self._horizon_models[horizon_idx]
        ridge_model = hm.get("ridge")
        if ridge_model is None:
            logger.warning("%s no ridge model in horizon %d", self._symbol, horizon_idx)
            return
        rf = hm.get("ridge_features") or hm["features"]
        n_feat = len(rf)
        self._online_ridge = OnlineRidge(
            n_features=n_feat,
            forgetting_factor=0.997,
            min_samples_for_update=50,
            max_update_magnitude=0.05,
        )
        coef = np.asarray(ridge_model.coef_, dtype=np.float64).ravel()
        intercept = float(ridge_model.intercept_)
        self._online_ridge.load_from_weights(coef, intercept)
        self._online_ridge_features = list(rf)
        logger.info("%s Online Ridge enabled: %d features, drift monitoring active",
                     self._symbol, n_feat)

    def _update_online_ridge(self, feat_dict: dict, close: float) -> None:
        """Feed realized return to Online Ridge for incremental update.

        Uses PREVIOUS bar's features paired with the return that was realized
        AFTER those features were observed (avoids look-ahead bias).
        """
        if self._online_ridge is None:
            self._prev_feat_dict = dict(feat_dict)
            self._last_close = close
            return
        if self._last_close <= 0 or self._prev_feat_dict is None:
            self._prev_feat_dict = dict(feat_dict)
            self._last_close = close
            return
        # Realized return (1-bar forward return of previous bar)
        realized_ret = (close - self._last_close) / self._last_close
        # Use PREVIOUS bar's features (not current bar's)
        x = np.array([
            float(self._prev_feat_dict.get(f, _NEUTRAL_DEFAULTS.get(f, 0.0)) or 0.0)
            for f in self._online_ridge_features
        ])
        self._online_ridge.update(x, realized_ret)
        # Store current for next iteration
        self._prev_feat_dict = dict(feat_dict)
        self._last_close = close
        # Log drift periodically
        if self._online_ridge.n_updates % 100 == 0 and self._online_ridge.n_updates > 0:
            drift = self._online_ridge.weight_drift
            logger.info("%s OnlineRidge: %d updates, drift=%.4f",
                         self._symbol, self._online_ridge.n_updates, drift)
            if drift > 0.5:
                logger.warning("%s OnlineRidge drift=%.4f exceeds threshold!",
                               self._symbol, drift)

    def _ensemble_predict(self, feat_dict: dict) -> float | None:
        """Ensemble: Ridge (primary) + LightGBM (secondary).

        Ridge won 20-fold walk-forward (15/20 PASS, Sharpe 0.54, +433%).
        Config weights: ridge_weight (default 0.6) + lgbm_weight (default 0.4).

        4h models use Ridge-only (OOS IC=0.035 vs LGBM IC=-0.022; LGBM overfits).
        """
        def _safe_val(v, feat_name: str = "") -> float:
            """Convert None/NaN to neutral value for model input.

            Uses _NEUTRAL_DEFAULTS for features where 0.0 is a directional
            signal (e.g. ls_ratio neutral=1.0, rsi_14 neutral=50.0).
            """
            neutral = _NEUTRAL_DEFAULTS.get(feat_name, 0.0)
            if v is None:
                return neutral
            try:
                f = float(v)
                return neutral if np.isnan(f) else f
            except (TypeError, ValueError):
                return neutral

        if not self._horizon_models:
            x = [_safe_val(feat_dict.get(f), f) for f in self._features]
            return float(self._model.predict([x])[0])

        # 4h models: pure Ridge (LGBM overfits on 4h frequency)
        version_str = self._config.get("version", "")
        ridge_only_4h = "4h" in version_str

        ridge_w = self._config.get("ridge_weight", 0.6)
        lgbm_w = self._config.get("lgbm_weight", 0.4)

        weighted_sum = 0.0
        weight_total = 0.0

        for hm in self._horizon_models:
            feats = hm["features"]
            x = [_safe_val(feat_dict.get(f), f) for f in feats]

            ic = max(hm["ic"], 0.001)

            # Ridge prediction (primary — walk-forward winner)
            if hm.get("ridge") is not None:
                # Ridge may use different features than LGBM
                rf = hm.get("ridge_features") or feats
                rx = [_safe_val(feat_dict.get(f), f) for f in rf]
                if True:  # NaN already handled by _safe_val
                    # Use Online Ridge if available, else static Ridge
                    if self._online_ridge is not None and self._online_ridge.n_updates >= 50:
                        ridge_pred = self._online_ridge.predict(np.array(rx))
                    else:
                        ridge_pred = float(hm["ridge"].predict([rx])[0])

                    if ridge_only_4h:
                        # 4h: Ridge-only, skip LGBM (OOS IC 0.035 vs -0.022)
                        pred = ridge_pred
                    else:
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
        x = [feat_dict.get(f, _NEUTRAL_DEFAULTS.get(f, 0.0)) or _NEUTRAL_DEFAULTS.get(f, 0.0) for f in feats]
        if any(np.isnan(x)):
            return None
        return float(hm["lgbm"].predict([x])[0])

    def _get_leverage_for_equity(self, equity: float) -> float:
        """Look up base leverage from ladder, then scale by volatility.

        Low vol → higher leverage (same risk budget), high vol → lower.
        Formula: lev × (1 / vol_ratio^0.5), clamped to [0.5×, 1.5×] of base.
        At vol_ratio=2.0 (2× median vol): lev × 0.71 (reduce by 29%).
        At vol_ratio=0.5 (half median vol): lev × 1.41 (increase by 41%).
        """
        lev = 1.0
        for threshold, lev_val in self.LEVERAGE_LADDER:
            if equity >= threshold:
                lev = lev_val

        # Vol-adaptive leverage: reduce from base when vol spikes, keep base otherwise.
        # Crypto vol is too high (30-50% annual) for target-risk style leverage.
        # Instead: base leverage × (1 / vol_ratio^0.3), clamped to [0.5×, 1.0×] of base.
        # This only REDUCES leverage during high vol, never increases above base.
        if len(self._rets) >= 20:
            rv_20 = float(np.std(self._rets[-20:]))
            vol_median = float(np.median(self._vol_history)) if len(self._vol_history) >= 50 else self._vol_median
            if vol_median > 0:
                vol_ratio = rv_20 / vol_median
                vol_ratio = max(0.5, min(3.0, vol_ratio))
                # Only reduce: high vol → scale down, normal/low vol → keep base
                vol_scale = min(1.0, 1.0 / (vol_ratio ** 0.3))
                lev *= vol_scale

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
        # n_total - same_dir - opposite_dir = neutrals (not penalized)

        if agree_frac >= 0.5:   # majority agree
            return 1.0
        elif opposite_dir > same_dir:  # majority actively disagree
            return 0.85  # was 0.7 — less penalty (only 2 runners, disagreement is normal)
        else:  # mixed or mostly neutral — don't penalize for others being flat
            return 0.90  # was 0.85

    def _auto_tune_params(self) -> None:
        """Periodically auto-tune base params from recent trade performance.

        Runs every 24h (checked by _bars_processed). Adjusts:
        1. Primary horizon: switch to whichever horizon has best recent IC
        2. long_only: disable if recent shorts are profitable

        Uses realized trade data, not hypothetical — only changes what's proven.
        """
        # Only run every ~24h (24 bars at 1h, 6 bars at 4h)
        is_4h = "4h" in (self._runner_key or "")
        tune_interval = 6 if is_4h else 24
        if self._bars_processed % tune_interval != 0 or self._bars_processed < tune_interval:
            return

        # Auto-tune horizon: pick the one with best recent IC from trade results
        if len(self._horizon_models) > 1 and len(self._recent_trade_pnls) >= 10:
            # Compare performance of trades that were influenced by each horizon
            # Simple heuristic: if overall PnL is negative, the current primary may not be best
            pnls = self._recent_trade_pnls
            if np.mean(pnls[-10:]) < 0 and len(self._horizon_models) > 1:
                # Current horizon is losing → log suggestion (don't auto-switch yet)
                logger.info(
                    "%s AUTO_TUNE: recent 10-trade avg PnL negative (%.4f), "
                    "consider switching primary horizon on next retrain",
                    self._symbol, np.mean(pnls[-10:]),
                )

        # Auto-tune long_only: if recent shorts are profitable, consider enabling shorts
        if self._long_only and len(self._recent_trade_pnls) >= 20:
            # This is informational only — actual change requires retrain
            logger.debug("%s AUTO_TUNE: long_only=True, tracking short opportunity", self._symbol)

    def _update_dynamic_scale(self) -> None:
        """Update dynamic position scale based on rolling trade Sharpe.

        If recent trades have negative Sharpe → reduce position size.
        This prevents a losing streak from compounding losses.
        - Rolling Sharpe > 0: scale = 1.0 (full size)
        - Rolling Sharpe in [-0.5, 0]: scale = 0.5 (half size)
        - Rolling Sharpe < -0.5: scale = 0.0 (stop trading this symbol)
        """
        pnls = self._recent_trade_pnls
        if len(pnls) < 5:
            self._dynamic_scale = 1.0
            return
        import numpy as _np
        arr = _np.array(pnls)
        mu = _np.mean(arr)
        std = _np.std(arr)
        sharpe = mu / std if std > 1e-8 else 0.0
        if sharpe > 0:
            self._dynamic_scale = 1.0
        elif sharpe > -0.5:
            self._dynamic_scale = 0.5
        else:
            self._dynamic_scale = 0.0
            logger.warning(
                "%s DYNAMIC SCALE → 0: rolling Sharpe=%.2f (%d trades), pausing",
                self._symbol, sharpe, len(pnls),
            )

    @staticmethod
    def compute_z_scale(z: float) -> float:
        """Non-linear position sizing based on z-score magnitude.

        Shared live/backtest contract:
        - |z| > 2.0: scale=1.5
        - |z| > 1.0: scale=1.0
        - |z| > 0.5: scale=0.7
        - else:       scale=0.5
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

    def _round_to_step(self, size: float) -> float:
        """Round qty to exchange step size (floor to never exceed)."""
        if self._step_size <= 0:
            return size
        steps = int(size / self._step_size)  # floor, not round — never exceed
        size = steps * self._step_size
        if self._step_size >= 1:
            size = int(size)
        else:
            step_decimals = max(0, -int(np.floor(np.log10(self._step_size))))
            size = round(size, step_decimals)
        return size

    def _compute_position_size(self, price: float) -> float:
        """Compute position size using Kelly-optimal leverage ladder.

        1. Look up leverage from equity-based ladder
        2. Position = equity * leverage / price
        3. Apply z-scale (non-linear z-score sizing) and consensus scale
        4. Clamp to [min_size, exchange limit]
        5. Set exchange leverage to match

        Uses the same 1.5x/1.0x leverage brackets as the converged gate chain.
        """
        if not self._adaptive_sizing:
            size = self._round_to_step(self._base_position_size)
            self._position_size = size
            return size

        try:
            equity, _free = get_total_and_free_balance(self._adapter.get_balances())
            if equity is None:
                logger.warning("%s sizing fallback: USDT total unavailable", self._symbol)
                equity = 0.0
        except VenueError as exc:
            logger.error("VENUE_ERROR symbol=%s context=sizing type=%s retry=true",
                         self._symbol, type(exc).__name__)
            size = self._round_to_step(self._base_position_size)
            self._position_size = size
            return size
        except Exception as exc:
            logger.warning("%s sizing fallback: failed to fetch balances: %s", self._symbol, exc)
            size = self._round_to_step(self._base_position_size)
            self._position_size = size
            return size

        if equity <= 0 or price <= 0:
            size = self._round_to_step(self._base_position_size)
            self._position_size = size
            return size

        # Get leverage from ladder
        target_lev = self._get_leverage_for_equity(equity)

        # Dynamic leverage: reduce when in drawdown (vol-aware thresholds)
        dd_pct = self._pnl.drawdown_pct if self._pnl.peak_equity > 0 else 0
        vol_ratio = (self._deadzone / self._deadzone_base) if self._deadzone_base > 0 else 1.0
        dd_scale = self._entry_scaler_module.leverage_scale(dd_pct, vol_ratio=vol_ratio)
        target_lev *= dd_scale
        if dd_scale < 1.0:
            logger.info("%s DD_SCALE: dd=%.1f%% → lev_scale=%.2f → effective_lev=%.1fx",
                        self._symbol, dd_pct, dd_scale, target_lev)

        # Position = equity × per_symbol_cap × leverage / price
        # Sharpe-weighted allocation: BTC+ETH only (altcoins removed 2026-03-21)
        # Walk-forward Sharpe: ETH 4.67, BTC 4.37 (monthly-gate optimized)
        # Equal weight since both are strong; cap at 0.45 each
        # Per-symbol position cap (fraction of equity allocated to this symbol).
        # Strategy H: 4h primary direction + 1h scale.
        # 4h runners are primary allocators; 1h/15m serve as scalers via gate.
        # At 10x leverage, effective exposure = cap × 10:
        #   BTC_4h 0.15 × 10 = 1.5x effective, ETH_4h 0.10 × 10 = 1.0x effective
        #   BTC_1h 0.08 × 10 = 0.8x (scaler), ETH_1h 0.06 × 10 = 0.6x (scaler)
        # Base allocation weights per runner (fraction of equity before leverage).
        # ── Dynamic base weights: adapt to equity size + market regime ──
        # Small account (<$500): concentrate on fewer positions, higher cap
        # Medium ($500-$10K): standard allocation
        # Large (>$10K): diversify, lower per-symbol cap
        if equity < 500:
            _BASE_WEIGHTS = {
                "BTCUSDT": 0.25, "ETHUSDT": 0.25, "ETHUSDT_15m": 0.0,
                "BTCUSDT_15m": 0.0, "BTCUSDT_4h": 0.35, "ETHUSDT_4h": 0.30,
            }
        elif equity < 10000:
            _BASE_WEIGHTS = {
                "BTCUSDT": 0.18, "ETHUSDT": 0.18, "ETHUSDT_15m": 0.05,
                "BTCUSDT_15m": 0.05, "BTCUSDT_4h": 0.25, "ETHUSDT_4h": 0.20,
            }
        else:
            _BASE_WEIGHTS = {
                "BTCUSDT": 0.12, "ETHUSDT": 0.12, "ETHUSDT_15m": 0.05,
                "BTCUSDT_15m": 0.05, "BTCUSDT_4h": 0.18, "ETHUSDT_4h": 0.15,
            }

        base_cap = _BASE_WEIGHTS.get(self._runner_key or self._symbol, 0.10)

        # Market regime scaling: trending → boost, ranging → shrink
        # Uses regime_active from _check_regime (already computed this bar)
        if self._regime_active:
            # Active regime: use full cap
            regime_cap_scale = 1.0
        else:
            # Filtered/ranging: reduce to 60% (still trade, just smaller)
            regime_cap_scale = 0.6
        base_cap *= regime_cap_scale

        # Adaptive allocation: scale base weight by runner's rolling IC health
        # IC healthy (GREEN) → 1.2x base, IC weak (YELLOW) → 0.8x, IC dead (RED) → 0.4x
        try:
            ic_path = Path("data/runtime/ic_health.json")
            if ic_path.exists() and not hasattr(self, '_ic_scale_cache_ts'):
                self._ic_scale_cache_ts = 0.0
                self._ic_scale_cache: dict[str, float] = {}
            # Refresh IC data every 10 min (small JSON, ensures retrain updates are picked up fast)
            if hasattr(self, '_ic_scale_cache_ts') and time.time() - self._ic_scale_cache_ts > 600:
                import json as _json
                ic_data = _json.loads(ic_path.read_text())
                for m in ic_data.get("models", []):
                    model_name = m.get("model", "")
                    status = m.get("overall_status", "GREEN")
                    scale = {"GREEN": 1.2, "YELLOW": 0.8, "RED": 0.4}.get(status, 1.0)
                    self._ic_scale_cache[model_name] = scale
                self._ic_scale_cache_ts = time.time()
        except Exception as exc:
            logger.debug("%s IC scale cache update failed: %s", self._symbol, exc)

        # Map runner_key to model name for IC lookup
        _runner_to_model = {
            "BTCUSDT": "BTCUSDT_gate_v2", "ETHUSDT": "ETHUSDT_gate_v2",
            "BTCUSDT_4h": "BTCUSDT_4h", "ETHUSDT_4h": "ETHUSDT_4h",
        }
        model_name = _runner_to_model.get(self._runner_key or self._symbol, "")
        ic_scale = getattr(self, '_ic_scale_cache', {}).get(model_name, 1.0)
        per_sym_cap = base_cap * ic_scale

        # Dynamic degradation: reduce cap when rolling Sharpe is negative
        per_sym_cap *= self._dynamic_scale

        # Correlation-aware sizing: reduce when highly correlated with active positions
        try:
            from runner.main import _correlation_computer
            if _correlation_computer is not None:
                active = [s for s, sig in _consensus_signals.items()
                          if sig != 0 and s != self._symbol and s != self._runner_key]
                if active:
                    avg_corr = _correlation_computer.position_correlation(self._symbol, active)
                    if avg_corr is not None and avg_corr > 0.6:
                        corr_scale = max(0.3, 1.0 - (avg_corr - 0.6) / 0.4)
                        per_sym_cap *= corr_scale
        except Exception:
            logger.debug("%s correlation sizing unavailable", self._symbol, exc_info=True)

        # Directional hedge: if BTC and ETH have opposite signals,
        # they partially hedge each other → allow slightly larger combined position
        my_key = self._runner_key or self._symbol
        peer_key = "ETHUSDT" if "BTC" in my_key else "BTCUSDT"
        # Check all timeframes of peer
        peer_sigs = [v for k, v in _consensus_signals.items() if peer_key in k and v != 0]
        if peer_sigs and self._current_signal != 0:
            peer_dir = 1 if sum(peer_sigs) > 0 else -1
            if peer_dir != self._current_signal:
                # Opposite directions = natural hedge → boost cap slightly
                per_sym_cap *= 1.15
                logger.debug("%s HEDGE_BOOST: opposite to %s → cap*1.15", self._symbol, peer_key)

        max_notional = equity * per_sym_cap * target_lev
        size = max_notional / price

        # Vol-targeting: normalize position size by realized vol so each trade
        # risks roughly the same amount regardless of market regime.
        # target_vol = 1.5% (1h bar), scale = target / realized.
        # High vol → smaller position, low vol → larger position.
        # Apply non-linear z-score scale + consensus + confidence-based cap
        z_scale = self._z_scale  # set in process_bar()
        consensus_scale = self._get_consensus_scale()
        # Confidence cap: strong z-score → larger position, weak → smaller
        confidence_scale = self._entry_scaler_module.confidence_cap_scale(
            z_score=self._last_z_score if hasattr(self, '_last_z_score') else 0,
            base_dz=self._deadzone_base,
        )
        combined_scale = z_scale * consensus_scale * confidence_scale
        pre_scale_size = size
        size *= combined_scale

        logger.info(
            "%s SCALE: cap=%.3f lev=%.1f base_size=%.4f × z=%.2f×cons=%.2f×conf=%.2f=%.2f → size=%.4f",
            self._symbol, per_sym_cap, target_lev, pre_scale_size,
            z_scale, consensus_scale, confidence_scale, combined_scale, size,
        )

        # Clamp to exchange limits
        size = max(self._min_size, size)
        if self._max_qty > 0:
            size = min(size, self._max_qty)

        # Round DOWN to exchange step size (floor to avoid exceeding limits)
        size = self._round_to_step(size)

        if size != self._position_size:
            logger.info(
                "%s SIZING: equity=$%.0f lev=%.0fx → %.2f %s ($%.0f notional)",
                self._symbol, equity, target_lev, size, self._symbol.replace("USDT", ""), size * price,
            )

        # Set exchange leverage to match (only if changed)
        # Bug fix: Bybit requires integer leverage >= 2. int(1.5)=1 was silently setting 1x.
        lev_int = max(2, int(round(target_lev)))
        if not hasattr(self, "_current_exchange_lev") or self._current_exchange_lev != lev_int:
            try:
                result = self._adapter._client.post("/v5/position/set-leverage", {
                    "category": "linear", "symbol": self._symbol,
                    "buyLeverage": str(lev_int),
                    "sellLeverage": str(lev_int),
                })
                if isinstance(result, dict):
                    ret_code = result.get("retCode", -1)
                    if ret_code != 0:
                        logger.warning(
                            "%s set_leverage API failed: retCode=%s retMsg=%s",
                            self._symbol, ret_code, result.get("retMsg"),
                        )
                self._current_exchange_lev = lev_int
                logger.info("%s exchange leverage set to %dx", self._symbol, lev_int)
            except Exception as e:
                logger.warning("%s set_leverage failed (non-fatal): %s", self._symbol, e)

        # Portfolio max exposure guard: total notional across all symbols
        # should not exceed 2x equity (prevents over-leverage even with hedging)
        try:
            # RustStateStore uses Fd8 (qty × 10^8). Must divide by _SCALE.
            _SCALE = 100_000_000
            total_exposure = sum(
                abs(self._state_store.get_position(s).qty) / _SCALE * price
                for s in ["BTCUSDT", "ETHUSDT"]
                if self._state_store is not None
            ) if self._state_store else 0
            # Portfolio cap scales with equity: small accounts can be more aggressive
            if equity < 500:
                max_total = equity * 8.0   # small: allow up to 8x total
            elif equity < 10000:
                max_total = equity * 6.0   # medium: 6x
            else:
                max_total = equity * 5.0   # large: 5x (more conservative)
            if total_exposure + (size * price) > max_total and size > self._min_size:
                allowed = max(0, max_total - total_exposure) / price
                if allowed < self._min_size:
                    size = 0
                else:
                    size = self._round_to_step(allowed)
                logger.info("%s PORTFOLIO_CAP: total_exposure=$%.0f + new=$%.0f > max=$%.0f → size=%.4f",
                            self._symbol, total_exposure, size * price, max_total, size)
        except Exception as exc:
            logger.debug("%s portfolio exposure check unavailable: %s", self._symbol, exc)

        # Final safety: ensure size is valid (not NaN/Inf/negative)
        if np.isnan(size) or np.isinf(size) or size < 0:
            logger.error("%s SIZING BLOCKED: invalid size=%.6f — falling back to min_size",
                         self._symbol, size)
            size = self._min_size

        self._position_size = size
        return size

    def _compute_entry_scale(self, signal: int) -> float:
        """Scale entry size by Bollinger-band position of current price.

        When entering long into an oversold dip → larger size (up to 1.2x).
        When entering long at overbought highs → smaller size (down to 0.3x).
        Mirror logic for shorts.

        Uses the _closes buffer (already maintained by _check_regime).
        Validated via backtest: MaxDD reduced ~50%, Sharpe cost ~10%.
        """
        if not self._entry_scaler_enabled:
            return 1.0
        scale = self._entry_scaler_module.bb_scale(
            signal, self._closes, self._entry_scaler_window,
        )
        self._entry_scaler_last = scale
        return scale

    def _evaluate_gates(self, signal: int, feat_dict: dict) -> float:
        """Evaluate alpha expansion gates — returns cumulative scale factor.

        Delegates to GateEvaluator (extracted module).
        Returns scale ∈ [0.0, ~1.5].
        """
        scale = self._gate_evaluator.evaluate(
            signal, feat_dict, _consensus_signals,
            self._runner_key or self._symbol, self._symbol,
        )
        self._gate_scale = scale
        return scale

    def _check_regime(self, close: float, feat_dict: dict | None = None) -> bool:
        """Check if current market regime is favorable for trading.

        Uses **adaptive thresholds** from rolling percentiles instead of hardcoded
        values. Each metric (vol, trend, efficiency) maintains a history buffer;
        the 20th percentile defines "dead market" — below that = filter.

        This auto-calibrates to the current market:
        - Bull run: vol/trend naturally higher → threshold rises → only truly dead periods filtered
        - Ranging: vol/trend naturally lower → threshold drops → system stays active in quiet periods
        - Pre-breakout: vol starts rising from low base → immediately detectable

        Args:
            close: Current bar close price.
            feat_dict: Feature dict from RustFeatureEngine (optional, for composite).
        """
        self._closes.append(close)
        if len(self._closes) >= 2:
            ret = np.log(close / self._closes[-2])
            self._rets.append(ret)

        # Truncate to bounded window
        _max_history = self._ma_window + 100
        if len(self._closes) > _max_history:
            self._closes = self._closes[-_max_history:]
        if len(self._rets) > _max_history:
            self._rets = self._rets[-_max_history:]

        # Need enough history
        if len(self._rets) < 20:
            return True

        # ── Compute current metrics ──
        vol_20 = float(np.std(self._rets[-20:]))

        if len(self._closes) >= self._ma_window:
            ma = np.mean(self._closes[-self._ma_window:])
            trend = abs(close / ma - 1)
        else:
            trend = 0.1  # assume active during warmup

        efficiency = 1.0
        rw = self._ranging_window
        if len(self._closes) >= rw:
            window = self._closes[-rw:]
            net_move = abs(window[-1] - window[0])
            total_path = sum(abs(window[j] - window[j-1]) for j in range(1, len(window)))
            efficiency = net_move / total_path if total_path > 0 else 0

        # ── Update rolling history for adaptive thresholds ──
        self._vol_history.append(vol_20)
        self._trend_history.append(trend)
        self._eff_history.append(efficiency)
        aw = self._adaptive_window
        if len(self._vol_history) > aw:
            self._vol_history = self._vol_history[-aw:]
        if len(self._trend_history) > aw:
            self._trend_history = self._trend_history[-aw:]
        if len(self._eff_history) > aw:
            self._eff_history = self._eff_history[-aw:]

        # ── Adaptive thresholds: p20 of rolling history ──
        # Before enough history (< 50 bars), use permissive defaults (= active)
        if len(self._vol_history) >= 50:
            self._vol_threshold = float(np.percentile(self._vol_history, 20))
            self._trend_threshold = float(np.percentile(self._trend_history, 20))
            self._ranging_threshold = float(np.percentile(self._eff_history, 25))

        # ── Layer 1: Vol + Trend (OR) ──
        # "Dead market" = both vol AND trend are in their bottom quintile
        # Use strict > (not >=) so zero vol/trend at exactly p20 = still dead
        base_active = (vol_20 > self._vol_threshold) or (trend > self._trend_threshold)

        # ── Layer 2: Ranging detector ──
        # Choppy = low efficiency AND low trend (both below adaptive threshold)
        is_ranging = (
            efficiency < self._ranging_threshold
            and trend < self._trend_threshold
        ) if len(self._eff_history) >= 50 else False

        self._regime_active = base_active and not is_ranging

        # ── Vol-adaptive deadzone ──
        # (replaces hardcoded vol_median with rolling median)
        if len(self._vol_history) >= 50:
            vol_median = float(np.median(self._vol_history))
        else:
            vol_median = self._vol_median  # fallback to hardcoded seed

        if vol_median > 0:
            ratio = vol_20 / vol_median
            # Clamp ratio [0.5, 2.0] — prevents extreme deadzone swings
            ratio = max(0.5, min(2.0, ratio))
            self._deadzone = self._deadzone_base * ratio

            # Adaptive hold times: scale with vol
            self._min_hold, self._max_hold = self._entry_scaler_module.adaptive_hold(
                self._min_hold_base, self._max_hold_base, ratio,
            )

        # Composite regime (opt-in, overrides deadzone/min_hold via ParamRouter)
        if (
            self._use_composite_regime
            and self._composite_detector is not None
            and feat_dict is not None
        ):
            self._apply_composite_regime(feat_dict)

        return self._regime_active

    def _apply_composite_regime(self, feat_dict: dict) -> None:
        """Run CompositeRegimeDetector and update params via ParamRouter."""
        from datetime import datetime, timezone

        label = self._composite_detector.detect(
            symbol=self._symbol,
            ts=datetime.now(timezone.utc),
            features=feat_dict,
        )
        if label is None or label.meta is None:
            return
        composite = label.meta.get("composite")
        if composite is None:
            return

        params = self._param_router.route(composite)
        self._regime_params = params
        # Feed back regime params to trading parameters
        self._deadzone = params.deadzone
        self._min_hold = params.min_hold
        # Crisis → block trading
        if composite.is_crisis:
            self._regime_active = False
        # Ranging → use ParamRouter scale/deadzone (reduced sizing, wider dz).
        # No longer hard-blocks; h=6 had negative IC but h=24 (primary) is positive.
        # ParamRouter already sets dz=1.2 + scale=0.4 for ranging, which is sufficient.
        logger.info(
            "%s CompositeRegime: %s|%s → dz=%.2f mh=%d scale=%.2f",
            self._symbol, composite.trend, composite.vol,
            params.deadzone, params.min_hold, params.position_scale,
        )

    @property
    def seconds_since_last_bar(self) -> float:
        """Seconds since last process_bar call. Returns inf if never called."""
        if self._last_bar_time == 0.0:
            return float("inf")
        return time.time() - self._last_bar_time

    def process_bar(self, bar: dict) -> dict:
        """Process one bar: regime → features → predict → signal → trade."""
        self._last_bar_time = time.time()
        self._bars_processed += 1

        # Periodic checkpoint save (every 10 bars) — but only if we have
        # enough history. Prevents overwriting a 800-bar checkpoint with a
        # 10-bar one after restore + a few new bars.
        # Use per-runner minimum: 4h needs 200 bars, 1h/15m needs 800.
        min_ckpt_bars = 200 if "4h" in (self._runner_key or "") else WARMUP_BARS
        if self._bars_processed % 10 == 0 and self._bars_processed >= min_ckpt_bars:
            try:
                self._save_checkpoint()
            except Exception:
                logger.debug("%s checkpoint save failed", self._symbol, exc_info=True)

        # Periodic reconciliation with exchange
        if not self._dry_run and self._bars_processed % self.RECONCILE_INTERVAL == 0:
            self._reconcile_position()

        # Post-reconcile invariant check: if in position, entry fields must be set.
        # This catches the historical bug where reconcile set _current_signal!=0
        # but left _entry_price=0 and _trade_peak_price=0.
        if self._current_signal != 0 and self._entry_price <= 0:
            logger.warning(
                "%s INVARIANT VIOLATION: signal=%d but entry_price=%.4f — "
                "using current close as fallback",
                self._symbol, self._current_signal, self._entry_price,
            )
            self._entry_price = bar["close"]
            self._entry_size = self._position_size
        if self._current_signal != 0 and self._trade_peak_price <= 0:
            logger.warning(
                "%s INVARIANT VIOLATION: signal=%d but trade_peak_price=%.4f — "
                "using entry_price as fallback",
                self._symbol, self._current_signal, self._trade_peak_price,
            )
            self._trade_peak_price = self._entry_price

        # Regime filter
        regime_ok = self._check_regime(bar["close"])

        # Get funding rate from ticker (if available)
        funding_rate = bar.get("funding_rate", float("nan"))
        try:
            funding_rate = float(funding_rate)
        except (ValueError, TypeError):
            funding_rate = float("nan")
        if np.isnan(funding_rate):
            try:
                tick = self._adapter.get_ticker(self._symbol)
                raw = tick.get("fundingRate", float("nan"))
                funding_rate = float(raw) if raw is not None else float("nan")
            except (ValueError, TypeError, Exception):
                funding_rate = float("nan")

        # V13: OI/LS/Taker data from background cache (refreshed every 55s, never blocks)
        oi_data = self._oi_cache.get()

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

        # V21: Inject cross-market features (SPY/QQQ/VIX/TLT/GLD from Yahoo daily)
        # Refresh cache every 6h (data updates daily at 22:00 UTC)
        if time.time() - self._cross_market_last_update > 21600:  # 6h
            self._load_cross_market()
        feat_dict.update(self._cross_market)

        # Online Ridge: feed realized return for incremental weight update
        self._update_online_ridge(feat_dict, bar["close"])

        # V14: Inject BTC dominance features (computed in Python, not Rust)
        if self._symbol == "BTCUSDT":
            dom_feats = self._compute_dominance_features(bar["close"])
            feat_dict.update(dom_feats)

        # Composite regime update (2nd pass with features)
        if self._use_composite_regime:
            regime_ok = self._check_regime(bar["close"], feat_dict=feat_dict)

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
        # Clip extreme z-scores: after service restart, rolling z-score window
        # has limited history and can produce z=13+ during large price moves.
        # Cap at ±5.0 to prevent distorted signal/sizing behavior.
        z = max(-5.0, min(5.0, z_val))

        # Non-linear z-score position sizing
        self._z_scale = self.compute_z_scale(z)
        self._last_z_score = z  # for confidence_cap_scale in _compute_position_size

        # Z-score reliability guard: extreme z (|z| > 3.5) with no position
        # is suspicious — likely caused by low prediction buffer variance.
        # Force flat if unreliable, regardless of warmup stage.
        if abs(z) > 3.5 and self._current_signal == 0:
            # A genuine z > 3.5 is < 0.02% probability under normal distribution.
            # Most likely cause: prediction buffer has near-zero std (all predictions ~same).
            # Cap z at ±3.0 to allow strong-but-not-extreme signals, block noise.
            old_z = z
            z = 3.0 if z > 0 else -3.0
            self._z_scale = self.compute_z_scale(z)
            self._last_z_score = z
            logger.info(
                "%s Z_CLAMP: |z|=%.1f capped to %.1f (extreme z with no position → likely unreliable)",
                self._symbol, abs(old_z), z,
            )
            self._last_z_score = z

        prev_signal = self._current_signal

        # Regime-adaptive deadzone: scale with realized vol
        # Low vol → lower deadzone (capture weaker signals)
        # Vol-adaptive deadzone is now handled in _check_regime() using rolling median

        # Primary signal via RustInferenceBridge: z-score → deadzone → min-hold → max-hold
        # Regime filter: pass deadzone=999 to force flat when regime is unfavorable.
        effective_dz = 999.0 if not regime_ok else self._deadzone
        new_signal = int(self._inference.apply_constraints(
            self._symbol, pred, hour_key,
            deadzone=effective_dz,
            min_hold=self._min_hold,
            max_hold=self._max_hold,
            long_only=self._long_only,
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

        # 2. Quick loss exit (5x leverage protection)
        # At 5x, a -1% move = -5% equity. Cut losses fast.
        if not force_exit and prev_signal != 0 and self._entry_price > 0:
            if prev_signal > 0:
                unrealized_pct = (bar["close"] - self._entry_price) / self._entry_price
            else:
                unrealized_pct = (self._entry_price - bar["close"]) / self._entry_price
            if unrealized_pct < -0.01:  # -1% adverse move → exit at 5x = -5% equity
                force_exit = True
                exit_reason = f"quick_loss_{unrealized_pct:.2%}"

        # 3. 4h z-score reversal exit: when 4h model signal flips against position
        # This is stronger than 1h z-reversal because 4h model has IC 0.29-0.43
        if not force_exit and prev_signal != 0:
            base_sym = self._symbol.replace("USDT", "") + "USDT"
            key_4h = f"{base_sym}_4h"
            sig_4h = _consensus_signals.get(key_4h, 0)
            # Only apply to non-4h runners (4h runners have their own exit logic)
            is_4h_runner = "4h" in getattr(self, '_config', {}).get('version', '')
            if not is_4h_runner and sig_4h != 0 and sig_4h == -prev_signal:
                force_exit = True
                exit_reason = f"4h_reversal_sig={sig_4h}"

        # 4. Z-score reversal exit (bridge enforces min_hold internally)
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

        # ── Direction alignment: ETH follows BTC's lead ──────────────
        # BTC is crypto market leader (70% dominance). When ETH wants to
        # open a NEW position opposing BTC's direction, block it.
        # Only applies to: new entries (prev=0 → new≠0) on ETH runners.
        # Does NOT apply to: exits, BTC runners, or same-direction trades.
        if new_signal != 0 and prev_signal == 0 and "ETH" in self._symbol:
            # Get BTC consensus direction (any timeframe)
            btc_sigs = [v for k, v in _consensus_signals.items()
                        if "BTC" in k and v != 0]
            if btc_sigs:
                btc_dir = 1 if sum(btc_sigs) > 0 else -1
                if new_signal != btc_dir:
                    logger.info(
                        "%s DIRECTION_ALIGN: ETH wants %s but BTC consensus is %s → blocked",
                        self._symbol,
                        "LONG" if new_signal > 0 else "SHORT",
                        "LONG" if btc_dir > 0 else "SHORT",
                    )
                    new_signal = 0

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

            # Orphan position safety: if signal is now 0 but we have entry_price set
            # from reconcile, AND exchange still has a position, keep the signal alive
            # so stop-loss continues to protect the position.
            if new_signal == 0 and self._entry_price > 0 and not self._dry_run:
                try:
                    positions = self._adapter.get_positions(symbol=self._symbol)
                    has_real = any(not p.is_flat for p in positions)
                    if has_real:
                        # Exchange has position but signal says flat → keep old signal for protection
                        exchange_side = 1 if any(p.is_long for p in positions if not p.is_flat) else -1
                        self._current_signal = exchange_side
                        new_signal = exchange_side
                        logger.warning(
                            "%s ORPHAN GUARD: signal=0 but exchange has %s position — "
                            "keeping signal=%d for stop-loss protection",
                            self._symbol, "LONG" if exchange_side == 1 else "SHORT", exchange_side,
                        )
                except Exception as exc:
                    logger.debug("%s orphan check API failed: %s", self._symbol, exc)

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
                "gate_scale": round(self._gate_scale, 3),
                "entry_scale": round(self._entry_scaler_last, 2),
                "signal": new_signal, "prev_signal": prev_signal,
                "hold_count": int(self._inference.get_position(self._symbol)), "close": bar["close"],
                "regime": "active" if regime_ok else "filtered",
                "dz": round(self._deadzone, 3),
                "oi_data": oi_str,
            }
            if force_exit:
                result["exit_reason"] = exit_reason

            if new_signal != prev_signal and actual_prev == prev_signal:
                # Evaluate alpha expansion gates for position scaling
                gate_scale = self._evaluate_gates(new_signal, feat_dict)
                if gate_scale <= 0.0 and new_signal != 0 and prev_signal == 0:
                    # Gate blocked entry — stay flat
                    new_signal = 0
                    self._current_signal = 0
                    self._inference.set_position(self._symbol, 0, 1)
                    result["gate_blocked"] = True
                    result["signal"] = 0
                else:
                    result["gate_scale"] = round(gate_scale, 3)

                # Recompute position size before entering new position
                if new_signal != 0:
                    self._compute_position_size(bar["close"])
                    # Apply gate scaling to position size
                    if gate_scale < 1.0:
                        self._position_size = max(
                            self._min_size,
                            self._position_size * gate_scale,
                        )
                        self._position_size = self._round_to_step(self._position_size)
                    # Apply entry scaler: reduce size when chasing, boost on pullback
                    entry_scale = self._compute_entry_scale(new_signal)
                    if entry_scale != 1.0:
                        self._position_size = max(
                            self._min_size,
                            self._position_size * entry_scale,
                        )
                        self._position_size = self._round_to_step(self._position_size)
                        logger.info(
                            "%s ENTRY_SCALER: bb_scale=%.2f → size=%.4f",
                            self._symbol, entry_scale, self._position_size,
                        )
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
                elif trade_result.get("action") == "killed":
                    self._force_flat_local_state_locked()
                    result["signal"] = 0
                    result["hold_count"] = 0
                    logger.warning("%s kill switch armed — flattening local strategy state",
                                   self._symbol)

        return result

    def _execute_limit_entry(self, symbol: str, side: str, qty: float,
                              price: float) -> dict:
        """Try to open a position via limit order for maker fee (0 bps).

        Places a PostOnly limit order at best bid + 1 tick (buy) or best ask
        - 1 tick (sell) for improved queue position without crossing.
        Uses adaptive timeout based on spread width:
          - spread < 0.02%: 15s (tight, high fill probability)
          - spread 0.02-0.05%: 30s (normal)
          - spread > 0.05%: 5s (wide, avoid adverse selection)

        Returns the same dict shape as ``send_market_order`` with an extra
        ``entry_method`` key ("limit" or "market_fallback").
        """
        # 1. Fetch current best bid/ask
        ticker = self._adapter.get_ticker(symbol)
        if not ticker or not ticker.get("bid1Price") or not ticker.get("ask1Price"):
            logger.warning("%s LIMIT ENTRY: ticker unavailable, falling back to market",
                           symbol)
            result = self._adapter.send_market_order(symbol, side, qty)
            result["entry_method"] = "market_fallback"
            self._market_fallbacks += 1
            self._log_fill_rate(symbol)
            return result

        bid = float(ticker["bid1Price"])
        ask = float(ticker["ask1Price"])
        spread_pct = (ask - bid) / bid * 100 if bid > 0 else 0

        # Adaptive timeout based on spread width
        # BTC/ETH spreads are almost always <0.02% (1-2 ticks).
        # 15s was too short — most limit orders need 20-40s to fill on tight spreads.
        # Adaptive timeout based on historical fill rate
        total_attempts = self._limit_fills + self._market_fallbacks
        if total_attempts >= 5:
            fill_rate = self._limit_fills / total_attempts
            # High fill rate → longer timeout (patient), low → shorter (don't waste time)
            base_timeout = 15.0 + 30.0 * fill_rate  # 15s (0% fill) to 45s (100% fill)
        else:
            base_timeout = 25.0  # default before enough data

        if spread_pct < 0.02:
            timeout = base_timeout
        elif spread_pct <= 0.05:
            timeout = base_timeout * 0.8  # slightly less patient for normal spread
        else:
            timeout = 5.0   # wide spread — quick fallback to avoid adverse selection

        # Price improvement: place 1 tick inside the spread for better queue position
        # Bug fix: when spread is exactly 1 tick, bid+tick=ask which crosses the book
        # and PostOnly would reject. In that case, place at bid/ask (no improvement).
        tick = self._tick_sizes.get(symbol, 0.01)
        spread_ticks = round((ask - bid) / tick) if tick > 0 else 999
        if side.lower() == "buy":
            limit_price = bid if spread_ticks <= 1 else bid + tick
        else:
            limit_price = ask if spread_ticks <= 1 else ask - tick

        logger.info(
            "%s LIMIT ENTRY: %s %.4f @ $%.2f (bid=$%.2f ask=$%.2f spread=%.4f%% timeout=%.0fs tick=%.2f)",
            symbol, side, qty, limit_price, bid, ask, spread_pct, timeout, tick,
        )

        # 2. Submit PostOnly limit order (rejected if would cross = guaranteed maker)
        limit_result = self._adapter.send_limit_order(
            symbol, side, qty, limit_price, tif="PostOnly",
        )
        if limit_result.get("status") == "error":
            logger.warning(
                "%s LIMIT ENTRY: PostOnly rejected (%s), falling back to market",
                symbol, limit_result.get("retMsg", ""),
            )
            result = self._adapter.send_market_order(symbol, side, qty)
            result["entry_method"] = "market_fallback"
            self._market_fallbacks += 1
            self._log_fill_rate(symbol)
            return result

        order_id = limit_result.get("orderId", "")
        logger.info("%s LIMIT ENTRY: order submitted orderId=%s", symbol, order_id)

        # 3. Poll for fill (adaptive timeout)
        deadline = time.time() + timeout
        filled = False
        while time.time() < deadline:
            time.sleep(self._limit_entry_poll_interval)
            try:
                open_orders = self._adapter.get_open_orders(symbol=symbol)
                # Check if our order is still in the open orders list
                still_open = any(
                    o.order_id == order_id for o in open_orders
                )
                if not still_open:
                    # Order no longer open — check fills to confirm it was filled
                    fills = self._adapter.get_recent_fills(symbol=symbol)
                    for f in fills:
                        if getattr(f, "order_id", "") == order_id:
                            filled = True
                            break
                    if not filled:
                        # Order disappeared but no fill found — likely self-cancelled
                        # or exchange rejected. Still check position as safety net.
                        positions = self._adapter.get_positions(symbol=symbol)
                        if any(not p.is_flat for p in positions):
                            filled = True
                    break
            except Exception as exc:
                logger.warning("%s LIMIT ENTRY: poll error: %s", symbol, exc)

        if filled:
            logger.info("%s LIMIT ENTRY: FILLED as maker (0 bps) orderId=%s",
                        symbol, order_id)
            self._limit_fills += 1
            self._log_fill_rate(symbol)
            return {
                "orderId": order_id,
                "status": "submitted",
                "entry_method": "limit",
            }

        # 4. Not filled within timeout — cancel and fall back to market
        logger.info(
            "%s LIMIT ENTRY: not filled after %.0fs, cancelling and using market order",
            symbol, timeout,
        )
        try:
            cancel_result = self._adapter.cancel_order(symbol, order_id)
            if cancel_result.get("status") == "canceled":
                logger.info("%s LIMIT ENTRY: cancelled orderId=%s", symbol, order_id)
            else:
                # Cancel failed — order may have filled in the meantime
                logger.warning(
                    "%s LIMIT ENTRY: cancel failed (%s) — checking if filled",
                    symbol, cancel_result.get("retMsg", ""),
                )
                fills = self._adapter.get_recent_fills(symbol=symbol)
                for f in fills:
                    if getattr(f, "order_id", "") == order_id:
                        logger.info("%s LIMIT ENTRY: filled during cancel, using as maker",
                                    symbol)
                        self._limit_fills += 1
                        self._log_fill_rate(symbol)
                        return {
                            "orderId": order_id,
                            "status": "submitted",
                            "entry_method": "limit",
                        }
        except Exception as exc:
            logger.warning("%s LIMIT ENTRY: cancel error: %s", symbol, exc)

        # Before market fallback: try repricing once at current best bid/ask
        # This catches cases where price moved away from our limit during the wait
        try:
            self._adapter.cancel_order(symbol, order_id)
            # Re-fetch current price
            ticker2 = self._adapter.get_ticker(symbol)
            if ticker2 and ticker2.get("bid1Price"):
                bid2 = float(ticker2["bid1Price"])
                ask2 = float(ticker2["ask1Price"])
                if side.lower() == "buy":
                    reprice = bid2 if spread_ticks <= 1 else bid2 + tick
                else:
                    reprice = ask2 if spread_ticks <= 1 else ask2 - tick

                logger.info("%s LIMIT REPRICE: %s @ $%.2f (was $%.2f)", symbol, side, reprice, limit_price)
                reprice_result = self._adapter.send_limit_order(symbol, side, qty, reprice, post_only=True)
                if reprice_result.get("orderId"):
                    # Wait another 15s for the repriced order
                    reprice_id = reprice_result["orderId"]
                    time.sleep(15)
                    fills = self._adapter.get_recent_fills(symbol=symbol)
                    for f in fills:
                        if getattr(f, "order_id", "") == reprice_id:
                            logger.info("%s LIMIT REPRICE: FILLED as maker", symbol)
                            self._limit_fills += 1
                            self._log_fill_rate(symbol)
                            return {"orderId": reprice_id, "status": "submitted", "entry_method": "limit_reprice"}
                    # Still not filled, cancel and fall through to market
                    self._adapter.cancel_order(symbol, reprice_id)
        except Exception as e:
            logger.warning("%s LIMIT REPRICE failed: %s", symbol, e)

        # Fall back to market order
        result = self._adapter.send_market_order(symbol, side, qty)
        result["entry_method"] = "market_fallback"
        self._market_fallbacks += 1
        self._log_fill_rate(symbol)
        logger.info("%s LIMIT ENTRY: market fallback result=%s", symbol, result)
        return result

    def _log_fill_rate(self, symbol: str) -> None:
        """Log limit fill rate every 10 trades."""
        total = self._limit_fills + self._market_fallbacks
        if total > 0 and total % 10 == 0:
            rate = self._limit_fills / total * 100
            logger.info(
                "%s LIMIT FILL RATE: %.1f%% (%d limit / %d market / %d total)",
                symbol, rate, self._limit_fills, self._market_fallbacks, total,
            )

    def _execute_signal_change(self, prev: int, new: int, price: float) -> dict:
        kill_armed = self._killed
        if kill_armed and prev == 0 and new != 0:
            return {"action": "killed", "reason": "drawdown_breaker"}

        trade_info: dict = {}

        # Prepare close PnL data (but defer recording until venue confirms)
        _pending_close_trade: dict | None = None
        _pending_close_size: float = 0.0
        if prev != 0 and self._entry_price > 0:
            _pending_close_size = self._entry_size if self._entry_size > 0 else self._position_size

        # Drawdown check via RustRiskEvaluator + RustKillSwitch
        # Use actual account equity (not cumulative PnL) for drawdown calculation
        try:
            _dd_equity, _ = get_total_and_free_balance(self._adapter.get_balances())
            _dd_equity = _dd_equity or 0.0
        except Exception:
            _dd_equity = 0.0
        _dd_peak = max(self._pnl.peak_equity, _dd_equity) if self._pnl.peak_equity > 0 else _dd_equity
        if self._risk_eval is not None and self._kill_switch is not None and _dd_peak > 0:
            breached = self._risk_eval.check_drawdown(
                equity=_dd_equity, peak_equity=_dd_peak,
            )
            if breached:
                dd = self._pnl.drawdown_pct
                reason = f"{self._symbol} drawdown {dd:.1f}%"
                self._kill_switch.arm("global", "*", "halt", reason,
                                      source="AlphaRunner")
                logger.critical(
                    "%s DRAWDOWN KILL (Rust): dd=%.1f%% peak=$%.2f current=$%.2f",
                    self._symbol, dd, self._pnl.peak_equity, self._pnl.total_pnl,
                )
                if not self._dry_run:
                    reliable_close_position(self._adapter, self._symbol)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}
        elif self._risk_eval is None and self._pnl.peak_equity > 0:
            # Fallback: manual drawdown check (backward compat if no Rust evaluator)
            dd = self._pnl.drawdown_pct
            if dd >= 15.0:
                if self._kill_switch is not None:
                    self._kill_switch.arm("global", "*", "halt",
                                          f"{self._symbol} drawdown {dd:.1f}%",
                                          source="AlphaRunner_fallback")
                logger.critical(
                    "%s DRAWDOWN KILL (fallback): dd=%.1f%% peak=$%.2f current=$%.2f",
                    self._symbol, dd, self._pnl.peak_equity, self._pnl.total_pnl,
                )
                if not self._dry_run:
                    reliable_close_position(self._adapter, self._symbol)
                return {"action": "killed", "reason": f"drawdown_{dd:.0f}%"}

        if self._dry_run:
            # In dry_run, record PnL immediately (no venue to confirm)
            if prev != 0 and self._entry_price > 0:
                trade = self._pnl.record_close(
                    symbol=self._symbol, side=prev,
                    entry_price=self._entry_price, exit_price=price,
                    size=_pending_close_size, reason="signal_change",
                )
                trade_info["closed_pnl"] = round(trade["pnl_usd"], 4)
                trade_info["closed_pct"] = round(trade["pnl_pct"], 2)
                self._recent_trade_pnls.append(trade["pnl_pct"])
                if len(self._recent_trade_pnls) > 30:
                    self._recent_trade_pnls = self._recent_trade_pnls[-30:]
                self._update_dynamic_scale()
                self._auto_tune_params()
            # Set entry state for dry_run new positions
            if new != 0:
                self._entry_price = price
                self._entry_size = self._position_size
                self._trade_peak_price = price
            elif prev != 0:
                self._entry_price = 0.0
                self._entry_size = 0.0
                self._trade_peak_price = 0.0
            trade_info["action"] = "dry_run"
            trade_info["from"] = prev
            trade_info["to"] = new
            return trade_info

        # Circuit breaker: block orders if too many recent failures
        if not self._circuit_breaker.allow_request():
            cb_state = self._circuit_breaker.snapshot()
            logger.warning("%s CIRCUIT BREAKER OPEN: %s", self._symbol, cb_state)
            return {"action": "circuit_open", "state": str(cb_state)}

        # Close venue position — first verify exchange actually has one
        if prev != 0:
            # Guard: don't try to close a phantom position (from checkpoint restore)
            try:
                positions = self._adapter.get_positions(symbol=self._symbol)
                has_real_pos = any(not p.is_flat for p in positions)
            except Exception:
                has_real_pos = True  # assume yes on API error (safer to attempt close)

            if not has_real_pos:
                logger.warning(
                    "%s PHANTOM CLOSE: signal was %d but exchange has no position — "
                    "resetting internal state without PnL",
                    self._symbol, prev,
                )
                self._entry_price = 0.0
                self._entry_size = 0.0
                self._trade_peak_price = 0.0
                return {"action": "phantom_close", "from": prev, "to": new}

            close_id = f"close_{self._symbol}_{int(time.time())}"
            self._osm.register(close_id, self._symbol, "sell" if prev == 1 else "buy",
                               "market", str(self._position_size))
            close_result = reliable_close_position(self._adapter, self._symbol)
            if close_result["status"] == "failed":
                logger.error("%s CLOSE FAILED after retries — keeping state", self._symbol)
                self._osm.transition(close_id, "rejected", reason="reliable_close_failed")
                self._circuit_breaker.record_failure()
                return {"action": "close_failed", "result": close_result}
            if not close_result.get("verified", True):
                logger.warning("%s CLOSE: position verification failed, proceeding", self._symbol)
            self._osm.transition(close_id, "filled", filled_qty=str(self._position_size),
                                 avg_price=str(price))
            self._circuit_breaker.record_success()

            # Record PnL AFTER venue close confirmed (prevents phantom PnL on close failure)
            if self._entry_price > 0:
                trade = self._pnl.record_close(
                    symbol=self._symbol, side=prev,
                    entry_price=self._entry_price, exit_price=price,
                    size=_pending_close_size, reason="signal_change",
                )
                trade_info["closed_pnl"] = round(trade["pnl_usd"], 4)
                trade_info["closed_pct"] = round(trade["pnl_pct"], 2)
                logger.info(
                    "%s CLOSE %s: pnl=$%.4f (%.2f%%) total=$%.4f wins=%d/%d",
                    self._symbol, "long" if prev == 1 else "short",
                    trade["pnl_usd"], trade["pnl_pct"], self._pnl.total_pnl,
                    self._pnl.win_count, self._pnl.trade_count,
                )
                close_side = "sell" if prev == 1 else "buy"
                self._record_fill(close_side, _pending_close_size, price,
                                  realized_pnl=trade["pnl_usd"])
                self._recent_trade_pnls.append(trade["pnl_pct"])
                if len(self._recent_trade_pnls) > 30:
                    self._recent_trade_pnls = self._recent_trade_pnls[-30:]
                self._update_dynamic_scale()

        if kill_armed:
            return {
                "action": "killed",
                "reason": "drawdown_breaker",
                "from": prev,
                "to": 0,
            }

        # Open new position
        if new != 0:
            side = "buy" if new == 1 else "sell"
            open_id = f"open_{self._symbol}_{int(time.time())}"
            order_type = "limit" if self._use_limit_entry else "market"
            self._osm.register(open_id, self._symbol, side, order_type,
                               str(self._position_size))

            # Dedup: check no other pending order for this symbol
            active = self._osm.active_count()
            if active > 2:  # close + open = 2 expected
                logger.warning("%s DEDUP: %d active orders, skipping", self._symbol, active)
                self._osm.transition(open_id, "rejected", reason="dedup_active_orders")
                return {"action": "dedup_blocked", "active": active}

            # Ensure step rounding is applied (safety net for all code paths)
            self._position_size = self._round_to_step(self._position_size)

            # Dynamic notional cap: scales with equity so position sizes
            # stay proportional as account grows/shrinks.
            try:
                _eq_for_cap, _ = get_total_and_free_balance(self._adapter.get_balances())
                _eq_for_cap = _eq_for_cap or 0.0
            except Exception:
                _eq_for_cap = 0.0
            dynamic_cap = get_max_order_notional(_eq_for_cap) if _eq_for_cap > 0 else MAX_ORDER_NOTIONAL
            effective_cap = dynamic_cap  # safety cap, no z_scale reduction (sizing handles it)
            notional = self._position_size * price
            logger.debug("%s NOTIONAL CHECK: size=%.4f price=%.2f notional=$%.2f limit=$%.2f (z_cap=$%.0f)",
                         self._symbol, self._position_size, price, notional, dynamic_cap, effective_cap)
            if notional > effective_cap:
                logger.warning(
                    "%s NOTIONAL CLAMP: $%.0f exceeds z-scaled limit $%.0f (z_scale=%.2f) — reducing size",
                    self._symbol, notional, effective_cap, self._z_scale,
                )
                self._position_size = self._round_to_step(effective_cap / price)
                notional = self._position_size * price
                if self._position_size < self._min_size:
                    self._osm.transition(open_id, "rejected", reason="below_min_after_clamp")
                    return {"action": "blocked", "reason": "below_min_after_clamp"}

            # Pre-flight margin check: avoid "ab not enough" errors
            try:
                _equity, avail = get_total_and_free_balance(self._adapter.get_balances())
                lev = getattr(self, '_current_exchange_lev', 1) or 1
                margin_needed = notional / lev
                if avail is None:
                    logger.warning("%s MARGIN PRECHECK skipped: USDT free balance unavailable", self._symbol)
                elif margin_needed > avail * 0.95:  # 5% buffer
                    logger.warning(
                        "%s MARGIN SKIP: need $%.0f margin but only $%.0f available (lev=%dx)",
                        self._symbol, margin_needed, avail, lev,
                    )
                    self._osm.transition(open_id, "rejected", reason="insufficient_margin")
                    return {"action": "margin_skip", "need": margin_needed, "avail": avail}
            except VenueError as exc:
                logger.error("VENUE_ERROR symbol=%s type=%s context=margin_precheck",
                             self._symbol, type(exc).__name__)
            except Exception as exc:
                logger.warning("%s MARGIN PRECHECK failed: %s", self._symbol, exc)

            # Safety: reject NaN/zero/negative position size before sending any order
            if (np.isnan(self._position_size) or self._position_size <= 0
                    or np.isnan(price) or price <= 0):
                logger.error(
                    "%s ORDER BLOCKED: invalid size=%.6f or price=%.2f (NaN/zero)",
                    self._symbol, self._position_size, price,
                )
                self._osm.transition(open_id, "rejected", reason="invalid_size_or_price")
                return {"action": "blocked", "reason": "nan_or_zero_size"}

            # Try limit entry for maker fee (0 bps) on new positions;
            # fall back to market order if limit not filled within timeout.
            if self._use_limit_entry:
                result = self._execute_limit_entry(self._symbol, side, self._position_size, price)
                entry_method = result.get("entry_method", "limit")
                if entry_method != "limit":
                    logger.info("%s ENTRY: used market fallback (4 bps taker)", self._symbol)
                else:
                    logger.info("%s ENTRY: filled as maker (0 bps)", self._symbol)
            else:
                result = self._adapter.send_market_order(self._symbol, side, self._position_size)
            # Check if order actually succeeded
            if result.get("status") == "error" or result.get("retCode", 0) != 0:
                ret_msg = str(result.get("retMsg", ""))
                if "ab not enough" in ret_msg.lower() or "insufficient" in ret_msg.lower():
                    logger.error("MARGIN_INSUFFICIENT symbol=%s msg=%s", self._symbol, ret_msg)
                else:
                    logger.error("ORDER_REJECTED symbol=%s reason=%s", self._symbol, ret_msg)
                self._osm.transition(open_id, "rejected", reason=ret_msg)
                self._circuit_breaker.record_failure()
                return {"action": "order_failed", "result": result}

            # Order succeeded — update state machine and tracking
            result.get("orderId", open_id)
            self._osm.transition(open_id, "filled", filled_qty=str(self._position_size),
                                 avg_price=str(price))
            self._circuit_breaker.record_success()

            # Use actual fill price (not bar close) to correctly track PnL + stops
            actual_price = price  # fallback to bar close
            if result.get("orderId"):
                try:
                    time.sleep(0.3)  # brief wait for fill to propagate
                    fills = self._adapter.get_recent_fills(symbol=self._symbol)
                    if fills:
                        actual_price = float(fills[0].price)
                        # Log slippage vs bar close AND vs order price (true execution cost)
                        bar_slip = (actual_price - price) / price * 100
                        entry_method = result.get("entry_method", "unknown")
                        logger.info(
                            "%s FILL: method=%s fill=$%.2f bar_close=$%.2f bar_slip=%.3f%%",
                            self._symbol, entry_method, actual_price, price, bar_slip,
                        )
                except Exception as e:
                    logger.warning("%s failed to get fill price, using bar close: %s",
                                   self._symbol, e)

            self._entry_price = actual_price
            self._entry_size = self._position_size  # snapshot entry-time size
            self._trade_peak_price = actual_price  # initialize trailing peak
            # Record open fill in RustStateStore
            self._record_fill(side, self._position_size, actual_price)
            atr = self._current_atr()
            stop = self._compute_stop_price(actual_price)
            logger.info(
                "Opened %s %.4f @ ~$%.1f stop=$%.2f (ATR=%.2f%%): %s",
                side, self._position_size, actual_price, stop, atr * 100, result,
            )
            trade_info.update({"side": side, "qty": self._position_size, "result": result,
                               "stop": round(stop, 2), "atr_pct": round(atr * 100, 2)})
        else:
            self._entry_price = 0.0
            self._entry_size = 0.0
            self._trade_peak_price = 0.0
            trade_info["action"] = "flat"

        return trade_info
