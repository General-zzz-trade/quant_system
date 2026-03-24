"""ML Signal Decision Module for event-driven backtest engine.

Bridges ML model predictions into backtest infrastructure.
Look-ahead bias prevention via EmbargoExecutionAdapter.
"""
from __future__ import annotations

import json
import logging
import pickle  # noqa: S301 - used for ML model loading (trusted local models only)
from collections import deque
from decimal import Decimal
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence

from decision.backtest_module_predict import (  # noqa: E402
    _ZScoreBuf,
    _resolve_primary_horizon_config,  # noqa: F401
    _resolve_primary_model_artifacts,
)


logger = logging.getLogger(__name__)


class MLSignalDecisionModule:
    """Decision module: ML predictions -> orders (LGBM+XGB+Ridge ensemble)."""

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
        bear_thresholds: Sequence[tuple[Any, ...]] | None = None,
        short_model: Any | None = None,
        short_score_key: str = "ml_short_score",
        zscore_window: int = 720,
        leverage: float = 2.0,
        v11_config: Any = None,
    ) -> None:
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
        self._ensemble_method = self._v11.ensemble_method

        # Multi-horizon or single model
        self._multi_horizon = self._config.get("multi_horizon", False)
        self._horizon_models: List[Dict[str, Any]] = []

        # Resolve params: v11_config > constructor args > config.json defaults
        cfg = self._v11
        warmup = cfg.zscore_warmup
        lgbm_xgb_w = cfg.lgbm_xgb_weight
        self._ridge_weight = cfg.ridge_weight
        self._lgbm_weight = cfg.lgbm_weight
        self._ridge = None
        self._ridge_features: Optional[List[str]] = None

        if self._multi_horizon:
            # Load per-horizon models
            for hcfg in self._config.get("horizon_models", []):
                lgbm_path = model_dir / hcfg["lgbm"]
                with open(lgbm_path, "rb") as f:
                    lgbm_data = pickle.load(f)
                xgb_model = None
                xgb_file = hcfg.get("xgb")
                xgb_path = model_dir / xgb_file if xgb_file else None
                if xgb_path is not None and xgb_path.exists():
                    with open(xgb_path, "rb") as f:
                        xgb_model = pickle.load(f)["model"]
                ridge_model = None
                ridge_features = hcfg.get("ridge_features")
                ridge_file = hcfg.get("ridge")
                ridge_path = model_dir / ridge_file if ridge_file else None
                if ridge_path is not None and ridge_path.exists():
                    with open(ridge_path, "rb") as f:
                        ridge_data = pickle.load(f)
                    ridge_model = ridge_data["model"] if isinstance(ridge_data, dict) else ridge_data
                    if ridge_features is None and isinstance(ridge_data, dict):
                        ridge_features = ridge_data.get("features")
                self._horizon_models.append({
                    "horizon": hcfg["horizon"],
                    "lgbm": lgbm_data["model"],
                    "xgb": xgb_model,
                    "ridge": ridge_model,
                    "ridge_features": ridge_features,
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
            # Single-model path: prefer config.json referenced artifacts, fall back to legacy v8 names.
            primary_artifacts = _resolve_primary_model_artifacts(model_dir, self._config)
            if primary_artifacts is None:
                raise FileNotFoundError(f"No loadable model artifact found in {model_dir}")
            with open(primary_artifacts["lgbm"], "rb") as f:
                lgbm_data = pickle.load(f)
            self._lgbm = lgbm_data["model"]
            self._features = primary_artifacts["features"] or lgbm_data["features"]
            self._xgb = None
            xgb_path = primary_artifacts["xgb"]
            if xgb_path is not None and xgb_path.exists():
                with open(xgb_path, "rb") as f:
                    self._xgb = pickle.load(f)["model"]
            ridge_path = primary_artifacts["ridge"]
            if ridge_path is not None and ridge_path.exists():
                with open(ridge_path, "rb") as f:
                    ridge_data = pickle.load(f)
                self._ridge = ridge_data["model"] if isinstance(ridge_data, dict) else ridge_data
                self._ridge_features = (
                    primary_artifacts["ridge_features"]
                    or (ridge_data.get("features") if isinstance(ridge_data, dict) else None)
                )

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
                from decision.backtest_module_exit_entry import handle_bear_model_exit
                should_return, events = handle_bear_model_exit(
                    self, features, snapshot, close, current_qty, event_id, events)
                if should_return:
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
                from decision.backtest_module_exit_entry import handle_bear_model_entry
                desired = handle_bear_model_entry(self, features, snapshot, desired)

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
            from decision.backtest_module_exit_entry import process_short_model
            process_short_model(self, features, snapshot)

        return events

    def _predict(self, features: Dict[str, float]) -> Optional[float]:
        from decision.backtest_module_helpers import predict_ensemble
        return predict_ensemble(self, features)

    def _predict_multi_horizon(self, features: Dict[str, float]) -> Optional[float]:
        from decision.backtest_module_helpers import predict_multi_horizon
        return predict_multi_horizon(self, features)

    def _extract_features(self, snapshot: Any) -> Optional[Dict[str, float]]:
        from decision.backtest_module_helpers import extract_features
        return extract_features(snapshot)

    def _get_positions(self, snapshot: Any) -> Dict[str, Any]:
        from decision.backtest_module_helpers import get_positions
        return get_positions(snapshot)

    def _get_close(self, snapshot: Any) -> Optional[float]:
        from decision.backtest_module_helpers import get_close
        return get_close(self.symbol, snapshot)

    def _get_hour_utc(self, snapshot: Any) -> Optional[int]:
        dt = self._get_timestamp_utc(snapshot)
        return dt.hour if dt is not None else None

    def _get_timestamp_utc(self, snapshot: Any) -> Any:
        from decision.backtest_module_helpers import get_timestamp_utc
        return get_timestamp_utc(snapshot)

    def _get_event_id(self, snapshot: Any) -> Optional[str]:
        from decision.backtest_module_helpers import get_event_id
        return get_event_id(snapshot)

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
        if self._rust_bridge is not None:
            ts = self._get_timestamp_utc(snapshot) if snapshot is not None else None
            hour_key = int(ts.timestamp()) // 3600 if ts is not None else self._bar_count
            return bool(self._rust_bridge.check_monthly_gate(
                self.symbol, close, hour_key, self._monthly_gate_window))
        if len(self._close_buf) < self._monthly_gate_window:
            return True
        ma = sum(self._close_buf) / len(self._close_buf)
        return close > ma

    def _make_order(self, *, side: str, qty: Decimal,
                    event_id: Optional[str], reason: str) -> Sequence[Any]:
        from decision.backtest_module_helpers import make_order_events
        return make_order_events(self.symbol, self._origin, side=side,
                                 qty=qty, event_id=event_id, reason=reason)

    def update_equity(self, new_equity: float) -> None:
        """Update equity for position sizing (call after each trade)."""
        self._equity = new_equity
