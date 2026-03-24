# decision/backtest_module_helpers.py
"""Helper methods for MLSignalDecisionModule prediction and snapshot extraction.

Extracted from backtest_module.py to reduce file size.
"""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from event.header import EventHeader
from event.types import EventType, IntentEvent, OrderEvent
from runner.backtest.adapter import _make_id

logger = logging.getLogger(__name__)


def predict_ensemble(
    module: Any,
    features: Dict[str, float],
) -> Optional[float]:
    """Run ensemble prediction on feature dict."""
    x = np.zeros((1, len(module._features)))
    for j, fname in enumerate(module._features):
        x[0, j] = features.get(fname, 0.0)

    lgbm_pred = float(module._lgbm.predict(x)[0])

    if module._ensemble_method == "ridge_primary" and module._ridge is not None:
        ridge_features = module._ridge_features or module._features
        rx = np.zeros((1, len(ridge_features)))
        for j, fname in enumerate(ridge_features):
            rx[0, j] = features.get(fname, 0.0)
        ridge_pred = float(module._ridge.predict(rx)[0])
        return float(module._ridge_weight * ridge_pred + module._lgbm_weight * lgbm_pred)

    if module._xgb is not None:
        try:
            import xgboost as xgb
            xgb_pred = float(module._xgb.predict(xgb.DMatrix(x))[0])
            return float(module._lgbm_xgb_w * lgbm_pred + (1 - module._lgbm_xgb_w) * xgb_pred)
        except Exception as e:
            logger.warning("XGBoost prediction failed, using LGBM only: %s", e)

    return lgbm_pred


def predict_multi_horizon(
    module: Any,
    features: Dict[str, float],
) -> Optional[float]:
    """Run multi-horizon ensemble: predict per horizon, z-score each, average."""
    z_values = []
    for hm in module._horizon_models:
        x = np.zeros((1, len(hm["features"])))
        for j, fname in enumerate(hm["features"]):
            x[0, j] = features.get(fname, 0.0)

        pred = float(hm["lgbm"].predict(x)[0])
        if module._ensemble_method == "ridge_primary" and hm.get("ridge") is not None:
            ridge_features = hm.get("ridge_features") or hm["features"]
            rx = np.zeros((1, len(ridge_features)))
            for j, fname in enumerate(ridge_features):
                rx[0, j] = features.get(fname, 0.0)
            ridge_pred = float(hm["ridge"].predict(rx)[0])
            pred = module._ridge_weight * ridge_pred + module._lgbm_weight * pred
        elif hm["xgb"] is not None:
            try:
                import xgboost as xgb
                xgb_pred = float(hm["xgb"].predict(xgb.DMatrix(x))[0])
                pred = module._lgbm_xgb_w * pred + (1 - module._lgbm_xgb_w) * xgb_pred
            except Exception as e:
                logger.warning("XGBoost multi-horizon prediction failed, using LGBM only: %s", e)

        z = hm["zscore_buf"].push(pred)
        z_values.append(z)

    if not z_values:
        return None
    if not any(hm["zscore_buf"].ready for hm in module._horizon_models):
        return 0.0
    return float(np.mean(z_values))


def extract_features(snapshot: Any) -> Optional[Dict[str, float]]:
    """Extract feature dict from pipeline snapshot."""
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


def get_positions(snapshot: Any) -> Dict[str, Any]:
    if isinstance(snapshot, dict):
        return snapshot.get("positions") or {}
    return getattr(snapshot, "positions", {}) or {}


def get_close(symbol: str, snapshot: Any) -> Optional[float]:
    if isinstance(snapshot, dict):
        market = snapshot.get("market")
        if market is None:
            markets = snapshot.get("markets") or {}
            market = markets.get(symbol)
    else:
        market = getattr(snapshot, "market", None)
        if market is None:
            markets = getattr(snapshot, "markets", {}) or {}
            market = markets.get(symbol) if isinstance(markets, dict) else None

    if market is None:
        return None

    close = getattr(market, "close_f", None) or getattr(market, "close", None) \
        or getattr(market, "last_price", None)
    return float(close) if close is not None else None


def get_timestamp_utc(snapshot: Any) -> Any:
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
            return datetime.datetime.fromtimestamp(ts / 1000, tz=datetime.timezone.utc)
        return None
    except Exception:
        return None


def get_event_id(snapshot: Any) -> Optional[str]:
    if isinstance(snapshot, dict):
        return snapshot.get("event_id")
    return getattr(snapshot, "event_id", None)


def make_order_events(
    symbol: str,
    origin: str,
    *,
    side: str,
    qty: Decimal,
    event_id: Optional[str],
    reason: str,
) -> Sequence[Any]:
    """Generate IntentEvent + OrderEvent pair."""
    intent_id = _make_id("intent")
    order_id = _make_id("order")

    intent_h = EventHeader.new_root(
        event_type=EventType.INTENT,
        version=1,
        source=f"decision:{origin}",
        correlation_id=str(event_id) if event_id else None,
    )
    order_h = EventHeader.from_parent(
        parent=intent_h,
        event_type=EventType.ORDER,
        version=1,
        source=f"decision:{origin}",
    )

    return (
        IntentEvent(
            header=intent_h,
            intent_id=intent_id,
            symbol=symbol,
            side=side,
            target_qty=qty,
            reason_code=reason,
            origin=origin,
        ),
        OrderEvent(
            header=order_h,
            order_id=order_id,
            intent_id=intent_id,
            symbol=symbol,
            side=side,
            qty=qty,
            price=None,
        ),
    )
