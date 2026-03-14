"""ML Signal Decision Module for event-driven backtest engine.

Implements DecisionModule protocol to bridge ML model predictions
into the existing event-driven backtest infrastructure.

This module:
  - Receives snapshots from the pipeline (bar-by-bar)
  - Runs ML model predictions (LGBM+XGB ensemble)
  - Applies rolling z-score normalization
  - Generates IntentEvent + OrderEvent via the existing event system
  - Tracks position state for hold constraints

All look-ahead bias prevention is handled by the engine's
EmbargoExecutionAdapter (fills at next bar's open price).
"""
from __future__ import annotations

import json
import pickle
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from event.header import EventHeader
from event.types import EventType, IntentEvent, OrderEvent
from runner.backtest.adapter import _make_id
from state.position import PositionState


@dataclass
class _ZScoreBuf:
    """Rolling z-score normalization (causal, backward-looking only)."""
    window: int = 720
    warmup: int = 180
    _buf: Deque[float] = field(default_factory=deque)

    def __post_init__(self):
        self._buf = deque(maxlen=self.window)

    def push(self, value: float) -> float:
        self._buf.append(value)
        if len(self._buf) < self.warmup:
            return 0.0
        arr = np.array(self._buf)
        std = float(np.std(arr))
        if std < 1e-12:
            return 0.0
        return (value - float(np.mean(arr))) / std

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self.warmup


class MLSignalDecisionModule:
    """Decision module that generates orders from ML model predictions.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. "BTCUSDT").
    model_dir : str or Path
        Directory with lgbm_v8.pkl, xgb_v8.pkl, config.json.
    equity : float
        Starting equity for position sizing.
    risk_fraction : float
        Fraction of equity per trade (before leverage).
    deadzone : float
        Z-score threshold for entry. If None, uses config.json value.
    min_hold : int
        Minimum bars before exit allowed.
    max_hold : int
        Maximum bars before forced exit.
    long_only : bool
        Only take long positions.
    trend_follow : bool
        Extend existing positions when trend remains favorable.
    trend_indicator : str
        Feature name used for trend gating.
    trend_threshold : float
        Threshold applied to trend feature.
    monthly_gate : bool
        If enabled, only allow entries when close is above trailing MA window.
    monthly_gate_window : int
        Window size for the monthly gate moving average.
    vol_target : float, optional
        If set, scale position size by target_vol / realized_vol_feature.
    vol_feature : str
        Feature name used for vol targeting.
    zscore_window : int
        Rolling z-score window (bars).
    leverage : float
        Fixed leverage multiplier for position sizing.
    v11_config : V11Config, optional
        V11 config object. If provided, overrides hardcoded params.
    """

    def __init__(
        self,
        symbol: str,
        model_dir: str | Path,
        equity: float = 10_000.0,
        risk_fraction: float = 0.05,
        deadzone: float | None = None,
        min_hold: int | None = None,
        max_hold: int | None = None,
        long_only: bool = False,
        trend_follow: bool = False,
        trend_indicator: str = "tf4h_close_vs_ma20",
        trend_threshold: float = 0.0,
        monthly_gate: bool = False,
        monthly_gate_window: int = 480,
        vol_target: float | None = None,
        vol_feature: str = "atr_norm_14",
        bear_model: Any | None = None,
        bear_thresholds: Sequence[tuple] | None = None,
        short_model: Any | None = None,
        short_score_key: str = "ml_short_score",
        zscore_window: int = 720,
        leverage: float = 2.0,
        v11_config=None,
    ):
        self.symbol = symbol.upper()
        self._equity = equity
        self._risk_fraction = risk_fraction
        self._long_only = long_only
        self._zscore_window = zscore_window
        self._leverage = leverage
        self._origin = f"ml_signal_{self.symbol.lower()}"

        # Load config
        model_dir = Path(model_dir)
        with open(model_dir / "config.json") as f:
            self._config = json.load(f)

        # V11 config: use if provided, else build from config.json
        self._v11 = v11_config
        if self._v11 is None:
            from alpha.v11_config import V11Config
            self._v11 = V11Config.from_config_json(self._config)

        # Multi-horizon or single model
        self._multi_horizon = self._config.get("multi_horizon", False)
        self._horizon_models: List[Dict[str, Any]] = []

        # Resolve params: v11_config > constructor args > config.json defaults
        cfg = self._v11
        warmup = cfg.zscore_warmup
        lgbm_xgb_w = cfg.lgbm_xgb_weight

        if self._multi_horizon:
            # Load per-horizon models
            for hcfg in self._config.get("horizon_models", []):
                lgbm_path = model_dir / hcfg["lgbm"]
                xgb_path = model_dir / hcfg["xgb"]
                with open(lgbm_path, "rb") as f:
                    lgbm_data = pickle.load(f)
                xgb_model = None
                if xgb_path.exists():
                    with open(xgb_path, "rb") as f:
                        xgb_model = pickle.load(f)["model"]
                self._horizon_models.append({
                    "horizon": hcfg["horizon"],
                    "lgbm": lgbm_data["model"],
                    "xgb": xgb_model,
                    "features": lgbm_data["features"],
                    "zscore_buf": _ZScoreBuf(window=zscore_window,
                                             warmup=warmup),
                })
            # Primary features (union for feature extraction)
            self._features = self._horizon_models[0]["features"]
            # Single z-score buf not used in multi-horizon mode
            self._lgbm = self._horizon_models[0]["lgbm"]
            self._xgb = self._horizon_models[0]["xgb"]
        else:
            # Single model (backward compatible)
            with open(model_dir / "lgbm_v8.pkl", "rb") as f:
                lgbm_data = pickle.load(f)
            self._lgbm = lgbm_data["model"]
            self._features = lgbm_data["features"]
            self._xgb = None
            xgb_path = model_dir / "xgb_v8.pkl"
            if xgb_path.exists():
                with open(xgb_path, "rb") as f:
                    self._xgb = pickle.load(f)["model"]

        # Use config values as defaults, allow override
        self._deadzone = deadzone if deadzone is not None else cfg.deadzone
        self._min_hold = min_hold if min_hold is not None else cfg.min_hold
        self._max_hold = max_hold if max_hold is not None else cfg.max_hold
        self._long_only = long_only if long_only else cfg.long_only
        self._lgbm_xgb_w = lgbm_xgb_w
        self._trend_follow = bool(trend_follow)
        self._trend_indicator = str(trend_indicator)
        self._trend_threshold = float(trend_threshold)
        self._monthly_gate = bool(monthly_gate)
        self._monthly_gate_window = max(int(monthly_gate_window), 1)
        self._vol_target = None if vol_target is None else float(vol_target)
        self._vol_feature = str(vol_feature)
        self._bear_model = bear_model
        self._bear_thresholds = bear_thresholds
        self._short_model = short_model
        self._short_score_key = str(short_score_key)

        # State (z-score buf used only to feed ExitManager z_score input)
        self._zscore_buf = _ZScoreBuf(window=zscore_window, warmup=warmup)
        self._position: float = 0.0  # +1 long, -1 short, 0 flat
        self._entry_bar: int = 0
        self._bar_count: int = 0
        self._current_qty = Decimal("0")
        self._close_buf: Deque[float] = deque(maxlen=self._monthly_gate_window)

        # Rust constraint bridge: z-score + discretize + min-hold in one call
        self._rust_bridge = None
        try:
            from _quant_hotpath import RustInferenceBridge
            self._rust_bridge = RustInferenceBridge(
                zscore_window=zscore_window,
                zscore_warmup=warmup,
                default_gate_window=self._monthly_gate_window,
            )
        except ImportError:
            pass  # Fall back to Python _ZScoreBuf + _discretize_signal

        # V11 modules
        from decision.exit_manager import ExitManager
        from decision.regime_gate import RegimeGate
        self._exit_mgr = ExitManager(
            config=cfg.exit,
            min_hold=self._min_hold,
            max_hold=self._max_hold,
        )
        self._regime_gate = RegimeGate(config=cfg.regime_gate)

        # Adaptive ensemble (multi-horizon only)
        self._ensemble = None
        if self._multi_horizon:
            from alpha.horizon_ensemble import AdaptiveHorizonEnsemble
            self._ensemble = AdaptiveHorizonEnsemble(
                config=cfg,
                horizon_models=self._horizon_models,
            )

    def decide(self, snapshot: Any) -> Iterable[Any]:
        """DecisionModule protocol: snapshot → opinion events."""
        self._bar_count += 1

        # Extract features from snapshot
        features = self._extract_features(snapshot)
        if features is None:
            return ()

        # Get current position
        positions = self._get_positions(snapshot)
        pos = positions.get(self.symbol)
        current_qty = Decimal("0")
        if pos is not None:
            qf = getattr(pos, "qty_f", None)
            current_qty = Decimal(str(qf)) if qf is not None else Decimal(str(getattr(pos, "qty", 0)))
        self._current_qty = current_qty

        # Track position state
        if current_qty > 0:
            self._position = 1.0
        elif current_qty < 0:
            self._position = -1.0
        else:
            self._position = 0.0

        # ML prediction + z-score normalization
        _constrained_from_rust = None  # Rust-constrained signal (if available)
        if self._multi_horizon and self._ensemble is not None:
            z = self._ensemble.predict(features)
        elif self._multi_horizon:
            z = self._predict_multi_horizon(features)
        else:
            pred = self._predict(features)
            if pred is None:
                return ()
            z = self._zscore_buf.push(pred)  # always feed for ExitManager
            # Use Rust bridge for unified z-score + discretize + min-hold
            # TIME BASIS: Backtest uses hour_key (ts-based) when delegating to Rust bridge,
            # matching live InferenceBridge behavior. The Python fallback (_discretize_signal)
            # uses bar-based counting, which diverges from live on sub-hourly bars.
            # Production always uses Rust bridge, so this only affects research scripts
            # running without _quant_hotpath.
            if self._rust_bridge is not None:
                trend_val = float("nan")
                if self._trend_follow:
                    tv = features.get(self._trend_indicator)
                    if tv is not None:
                        trend_val = float(tv)
                # Use proper hour_key from timestamp (matching live bridge.py:151)
                # so z-score aggregation and monthly gate accumulate per-hour,
                # not per-bar. For 1h bars this is equivalent to bar_count,
                # but for sub-hourly bars bar_count would inflate the z-score window.
                ts = self._get_timestamp_utc(snapshot)
                hour_key = int(ts.timestamp()) // 3600 if ts is not None else self._bar_count
                _constrained_from_rust = int(self._rust_bridge.apply_constraints(
                    self.symbol,
                    pred,
                    hour_key,
                    deadzone=self._deadzone,
                    min_hold=self._min_hold,
                    long_only=self._long_only,
                    trend_follow=self._trend_follow,
                    trend_val=trend_val,
                    trend_threshold=self._trend_threshold,
                    max_hold=self._max_hold,
                ))
        if z is None:
            return ()

        # Get market price for sizing
        close = self._get_close(snapshot)
        if close is None or close <= 0:
            return ()
        self._close_buf.append(close)

        # Update trailing stop price tracking
        if self._position != 0:
            self._exit_mgr.update_price(self.symbol, close)

        # Regime gate: get position scale factor
        # NOTE: Intentional divergence from live. Backtest uses feature-based
        # RegimeGate for position_scale. Live uses IC-based AlphaHealthMonitor
        # (see live_runner.py Gate 4). Both produce 0.0/0.5/1.0 scale factors.
        _regime_label, position_scale = self._regime_gate.evaluate(features)

        event_id = self._get_event_id(snapshot)

        # Extract hour for time filter
        hour_utc = self._get_hour_utc(snapshot)

        if _constrained_from_rust is not None:
            desired = _constrained_from_rust
        else:
            desired = self._discretize_signal(z)

        # When Rust bridge is active, trend_hold is already handled inside
        # apply_constraints(). Only apply Python trend_hold as fallback.
        if _constrained_from_rust is not None:
            trend_hold_active = False
        else:
            trend_hold_active = self._should_trend_hold(features, desired)

        # ── Exit logic ──
        events: List[Any] = []
        if self._position != 0:
            if not self._passes_monthly_gate(close, snapshot):
                bear_score = None
                if self._bear_model is not None:
                    ts = self._get_timestamp_utc(snapshot)
                    bear_sig = self._bear_model.predict(
                        symbol=self.symbol, ts=ts, features=features)
                    if bear_sig is not None and bear_sig.side == "long":
                        if self._bear_thresholds:
                            prob = 0.5 + bear_sig.strength
                            bear_score = 0.0
                            for thresh, s in self._bear_thresholds:
                                if prob > thresh:
                                    bear_score = s
                                    break
                        else:
                            bear_score = -1.0
                    else:
                        bear_score = 0.0

                if bear_score is not None and bear_score != 0:
                    # Bear model says stay in position — sync Rust hold state
                    # (matches live bridge.py:262-263)
                    if self._rust_bridge is not None:
                        cur_rust_pos = self._rust_bridge.get_position(self.symbol)
                        if bear_score != cur_rust_pos:
                            self._rust_bridge.set_position(self.symbol, bear_score, 1)
                    self._position = bear_score
                    return events
                else:
                    # No bear model or bear score is 0 → flatten
                    if self._rust_bridge is not None:
                        cur_rust_pos = self._rust_bridge.get_position(self.symbol)
                        if cur_rust_pos != 0.0:
                            self._rust_bridge.set_position(self.symbol, 0.0, 1)
                    side = "sell" if self._position > 0 else "buy"
                    events.extend(self._make_order(
                        side=side,
                        qty=abs(current_qty),
                        event_id=event_id,
                        reason="monthly_gate",
                    ))
                    self._exit_mgr.on_exit(self.symbol)
                    self._position = 0.0
                    return events

            # Match live constraint behavior: once the discretized score falls
            # back inside the deadzone, flatten unless trend-follow explicitly
            # extends the position.
            # When Rust bridge is active, min-hold is already enforced in
            # apply_constraints — desired stays non-zero until hold expires.
            if desired == 0 and not trend_hold_active:
                if _constrained_from_rust is not None or (self._bar_count - self._entry_bar) >= self._min_hold:
                    side = "sell" if self._position > 0 else "buy"
                    events.extend(self._make_order(
                        side=side,
                        qty=abs(current_qty),
                        event_id=event_id,
                        reason=f"signal_flat_z={z:.2f}",
                    ))
                    self._exit_mgr.on_exit(self.symbol)
                    self._position = 0.0
                    return events

            should_exit, reason = self._exit_mgr.check_exit(
                symbol=self.symbol,
                price=close,
                bar=self._bar_count,
                z_score=z,
                position=self._position,
            )
            if should_exit and trend_hold_active and reason.startswith("deadzone_fade"):
                should_exit = False
                reason = ""
            if should_exit:
                side = "sell" if self._position > 0 else "buy"
                events.extend(self._make_order(
                    side=side,
                    qty=abs(current_qty),
                    event_id=event_id,
                    reason=reason,
                ))
                self._exit_mgr.on_exit(self.symbol)
                self._position = 0.0

        # ── Entry logic ──
        if self._position == 0:
            # Check entry gates (z-cap + time filter)
            if not self._exit_mgr.allow_entry(z, hour_utc):
                return events

            if desired != 0 and not self._passes_monthly_gate(close, snapshot):
                if self._bear_model is not None:
                    ts = self._get_timestamp_utc(snapshot)
                    bear_sig = self._bear_model.predict(
                        symbol=self.symbol, ts=ts, features=features)
                    if bear_sig is not None and bear_sig.side == "long":
                        if self._bear_thresholds:
                            prob = 0.5 + bear_sig.strength
                            bear_entry_score = 0.0
                            for thresh, s in self._bear_thresholds:
                                if prob > thresh:
                                    bear_entry_score = s
                                    break
                        else:
                            bear_entry_score = -1.0
                        desired = int(bear_entry_score) if bear_entry_score != 0 else 0
                    else:
                        desired = 0
                else:
                    desired = 0

            if desired != 0:
                # Compute order qty: equity × risk_fraction × leverage × regime_scale / price
                # Vol-scale applied AFTER discretization to match live (bridge.py:265-272).
                vol_scale = 1.0
                if self._vol_target is not None:
                    vol_val = features.get(self._vol_feature)
                    if vol_val is not None and vol_val > 1e-8:
                        vol_scale = min(self._vol_target / float(vol_val), 1.0)
                notional = Decimal(str(
                    self._equity * self._risk_fraction * self._leverage * position_scale * vol_scale
                ))
                qty = notional / Decimal(str(close))
                # Adaptive rounding: use finer precision for high-price assets
                if close > 10000:
                    qty = qty.quantize(Decimal("0.00001"))  # 5dp for BTC-class
                else:
                    qty = qty.quantize(Decimal("0.001"))    # 3dp for ETH-class

                if qty > 0:
                    side = "buy" if desired > 0 else "sell"
                    events.extend(self._make_order(
                        side=side,
                        qty=qty,
                        event_id=event_id,
                        reason=f"entry_z={z:.2f}",
                    ))
                    self._position = float(desired)
                    self._entry_bar = self._bar_count
                    self._exit_mgr.on_entry(
                        self.symbol, close, self._bar_count, float(desired)
                    )

        # ── Independent short model (mirrors live bridge.py:287-330) ──
        if self._short_model is not None:
            ts = self._get_timestamp_utc(snapshot)
            short_sig = self._short_model.predict(
                symbol=self.symbol, ts=ts, features=features)
            short_score = 0.0
            if short_sig is not None and short_sig.strength is not None:
                raw = short_sig.strength
                if short_sig.side == "short":
                    raw = -raw
                elif short_sig.side == "flat":
                    raw = 0.0

                # Apply Rust constraint path if available (same as live)
                if self._rust_bridge is not None:
                    hour_key = int(ts.timestamp()) // 3600 if ts is not None else 0
                    short_score = self._rust_bridge.process_short_signal(
                        self.symbol, raw, hour_key,
                        self._deadzone, self._min_hold)
                else:
                    # Python fallback for short signal (deadzone + min-hold).
                    # Production always uses the Rust path above which
                    # includes the full constraint pipeline (deadzone +
                    # min-hold + trend-hold).  This fallback exists only
                    # for environments without _quant_hotpath.
                    if not hasattr(self, '_short_hold_count'):
                        self._short_hold_count = 0
                        self._prev_short_score = 0.0

                    if abs(raw) > self._deadzone:
                        short_score = raw
                        self._short_hold_count += 1
                    elif self._prev_short_score != 0.0:
                        if self._short_hold_count < self._min_hold:
                            short_score = self._prev_short_score
                            self._short_hold_count += 1
                        else:
                            self._short_hold_count = 0
                    else:
                        self._short_hold_count = 0
                    self._prev_short_score = short_score

                # Vol-adaptive sizing for short
                if short_score != 0.0 and self._vol_target is not None:
                    vol_val = features.get(self._vol_feature)
                    if vol_val is not None and vol_val > 1e-8:
                        scale = min(self._vol_target / float(vol_val), 1.0)
                        short_score *= scale

            features[self._short_score_key] = short_score
            self._last_short_score = short_score

        return events

    def _predict(self, features: Dict[str, float]) -> Optional[float]:
        """Run ensemble prediction on feature dict."""
        x = np.zeros((1, len(self._features)))
        for j, fname in enumerate(self._features):
            x[0, j] = features.get(fname, 0.0)

        lgbm_pred = float(self._lgbm.predict(x)[0])

        if self._xgb is not None:
            try:
                import xgboost as xgb
                xgb_pred = float(self._xgb.predict(xgb.DMatrix(x))[0])
                return self._lgbm_xgb_w * lgbm_pred + (1 - self._lgbm_xgb_w) * xgb_pred
            except Exception:
                pass

        return lgbm_pred

    def _predict_multi_horizon(self, features: Dict[str, float]) -> Optional[float]:
        """Run multi-horizon ensemble: predict per horizon, z-score each, average.

        Legacy path — used only when AdaptiveHorizonEnsemble is not available.
        """
        z_values = []
        for hm in self._horizon_models:
            x = np.zeros((1, len(hm["features"])))
            for j, fname in enumerate(hm["features"]):
                x[0, j] = features.get(fname, 0.0)

            pred = float(hm["lgbm"].predict(x)[0])
            if hm["xgb"] is not None:
                try:
                    import xgboost as xgb
                    xgb_pred = float(hm["xgb"].predict(xgb.DMatrix(x))[0])
                    pred = self._lgbm_xgb_w * pred + (1 - self._lgbm_xgb_w) * xgb_pred
                except Exception:
                    pass

            z = hm["zscore_buf"].push(pred)
            z_values.append(z)

        if not z_values:
            return None
        # Check if any horizon has warmed up
        if not any(hm["zscore_buf"].ready for hm in self._horizon_models):
            return 0.0
        return float(np.mean(z_values))

    def _extract_features(self, snapshot: Any) -> Optional[Dict[str, float]]:
        """Extract feature dict from pipeline snapshot."""
        # snapshot can be SimpleNamespace or dict
        if isinstance(snapshot, dict):
            feats = snapshot.get("features")
        else:
            feats = getattr(snapshot, "features", None)

        if feats is None:
            return None

        if isinstance(feats, Mapping):
            return {k: float(v) for k, v in feats.items()
                    if isinstance(v, (int, float, Decimal))}
        return None

    def _get_positions(self, snapshot: Any) -> Dict[str, Any]:
        if isinstance(snapshot, dict):
            return snapshot.get("positions") or {}
        return getattr(snapshot, "positions", {}) or {}

    def _get_close(self, snapshot: Any) -> Optional[float]:
        if isinstance(snapshot, dict):
            market = snapshot.get("market")
            if market is None:
                markets = snapshot.get("markets") or {}
                market = markets.get(self.symbol)
        else:
            market = getattr(snapshot, "market", None)
            if market is None:
                markets = getattr(snapshot, "markets", {}) or {}
                market = markets.get(self.symbol) if isinstance(markets, dict) else None

        if market is None:
            return None

        close = getattr(market, "close_f", None) or getattr(market, "close", None) \
            or getattr(market, "last_price", None)
        return float(close) if close is not None else None

    def _get_hour_utc(self, snapshot: Any) -> Optional[int]:
        """Extract UTC hour from snapshot (for time filter)."""
        dt = self._get_timestamp_utc(snapshot)
        return dt.hour if dt is not None else None

    def _get_timestamp_utc(self, snapshot: Any):
        """Extract UTC datetime from snapshot timestamp."""
        if isinstance(snapshot, dict):
            ts = snapshot.get("timestamp") or snapshot.get("open_time")
        else:
            ts = getattr(snapshot, "timestamp", None) or getattr(snapshot, "open_time", None)
        if ts is None:
            return None
        try:
            import datetime
            if isinstance(ts, (int, float)):
                # Assume milliseconds
                return datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc)
            return None
        except Exception:
            return None

    def _get_event_id(self, snapshot: Any) -> Optional[str]:
        if isinstance(snapshot, dict):
            return snapshot.get("event_id")
        return getattr(snapshot, "event_id", None)

    def _discretize_signal(self, z_score: float) -> int:
        if z_score > self._deadzone:
            return 1
        if not self._long_only and z_score < -self._deadzone:
            return -1
        return 0

    def _should_trend_hold(self, features: Dict[str, float], desired: int) -> bool:
        if not self._trend_follow or self._position == 0 or desired != 0:
            return False
        trend_val = features.get(self._trend_indicator)
        if trend_val is None:
            return False
        if self._position > 0:
            return float(trend_val) > self._trend_threshold
        return float(trend_val) < -self._trend_threshold

    def _passes_monthly_gate(self, close: float, snapshot: Any = None) -> bool:
        if not self._monthly_gate:
            return True
        # Use Rust monthly gate when available (matches live bridge.py:167-173)
        if self._rust_bridge is not None:
            ts = self._get_timestamp_utc(snapshot) if snapshot is not None else None
            hour_key = int(ts.timestamp()) // 3600 if ts is not None else self._bar_count
            return self._rust_bridge.check_monthly_gate(
                self.symbol, close, hour_key, self._monthly_gate_window)
        # Python fallback
        if len(self._close_buf) < self._monthly_gate_window:
            return True
        ma = sum(self._close_buf) / len(self._close_buf)
        return close > ma

    def _make_order(self, *, side: str, qty: Decimal,
                    event_id: Optional[str], reason: str) -> Sequence[Any]:
        """Generate IntentEvent + OrderEvent pair."""
        intent_id = _make_id("intent")
        order_id = _make_id("order")

        intent_h = EventHeader.new_root(
            event_type=EventType.INTENT,
            version=1,
            source=f"decision:{self._origin}",
            correlation_id=str(event_id) if event_id else None,
        )
        order_h = EventHeader.from_parent(
            parent=intent_h,
            event_type=EventType.ORDER,
            version=1,
            source=f"decision:{self._origin}",
        )

        return (
            IntentEvent(
                header=intent_h,
                intent_id=intent_id,
                symbol=self.symbol,
                side=side,
                target_qty=qty,
                reason_code=reason,
                origin=self._origin,
            ),
            OrderEvent(
                header=order_h,
                order_id=order_id,
                intent_id=intent_id,
                symbol=self.symbol,
                side=side,
                qty=qty,
                price=None,
            ),
        )

    def update_equity(self, new_equity: float) -> None:
        """Update equity for position sizing (call after each trade)."""
        self._equity = new_equity
