"""AlphaRunner — runs alpha strategy on Bybit with RustFeatureEngine + LightGBM."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from scripts.ops.config import (
    INTERVAL, MAX_ORDER_NOTIONAL, WARMUP_BARS,
    _consensus_signals,
)
from scripts.ops.data_fetcher import _fetch_binance_oi_data

logger = logging.getLogger(__name__)


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

            # Hard safety limit: block orders exceeding MAX_ORDER_NOTIONAL
            notional = self._position_size * price
            if notional > MAX_ORDER_NOTIONAL:
                logger.critical(
                    "%s ORDER BLOCKED: notional=$%.2f exceeds hard limit $%.2f. size=%.4f price=%.2f",
                    self._symbol, notional, MAX_ORDER_NOTIONAL, self._position_size, price,
                )
                self._osm.transition(open_id, "rejected", reason=f"notional ${notional:.0f} > limit")
                return {"action": "blocked", "reason": f"notional ${notional:.0f} > ${MAX_ORDER_NOTIONAL:.0f}"}

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
