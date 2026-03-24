# engine/coordinator_handlers.py
"""Dispatcher event handlers extracted from EngineCoordinator.

Each handler is a standalone function that takes the coordinator (self)
as its first argument. This keeps coordinator.py under the 500-line limit
while preserving the same call semantics.
"""
from __future__ import annotations

import time as _time_mod
from datetime import datetime as _datetime_type
from typing import Any

from engine.pipeline import (
    PipelineInput, PipelineOutput, _detect_kind, _build_snapshot,
)


def _time_mod_time() -> int:
    return int(_time_mod.time())


def handle_pipeline_event(coord: Any, event: Any) -> None:
    """PIPELINE handler: factual events advance state."""
    if coord._tick_processors is not None:
        kind = _detect_kind(event)
        symbol = getattr(event, "symbol", coord._cfg.symbol_default)
        tp = coord._tick_processors.get(symbol)
        if tp is None:
            tp = next(iter(coord._tick_processors.values()))
        if kind == "MARKET":
            handle_market_tick_fast(coord, event, tp)
            return
        if kind == "FILL":
            tp.process_fill(event)
            return
        if kind == "FUNDING":
            tp.process_funding(event)
            return
        return

    # -- Original slow path --
    features = None
    if coord._cfg.feature_hook is not None:
        features = coord._cfg.feature_hook.on_event(event)

    inp = PipelineInput(
        event=event,
        event_index=0,
        symbol_default=coord._cfg.symbol_default,
        markets={},
        account=None,
        positions={},
        features=features,
    )

    assert coord._pipeline is not None
    out = coord._pipeline.apply(inp)

    with coord._lock:
        if out.snapshot is not None:
            coord._last_snapshot = out.snapshot

    if coord._cfg.on_pipeline_output is not None:
        if out.advanced or coord._cfg.emit_on_non_advanced:
            coord._cfg.on_pipeline_output(out)

    if coord._cfg.on_snapshot is not None and out.snapshot is not None:
        coord._cfg.on_snapshot(out.snapshot)

    # Decision trigger: only on MARKET event advancement
    if coord._decision_bridge is not None and out.advanced and out.snapshot is not None:
        if _detect_kind(event) == "MARKET":
            coord._decision_bridge.on_pipeline_output(out)


def handle_market_tick_fast(coord: Any, event: Any, tp: Any) -> None:
    """Fast path: single Rust call for features + predict + state + features dict."""
    symbol = getattr(event, "symbol", coord._cfg.symbol_default)
    close_f = float(getattr(event, "close", 0))
    volume = float(getattr(event, "volume", 0) or 0)
    high = float(getattr(event, "high", 0) or 0)
    low = float(getattr(event, "low", 0) or 0)
    open_ = float(getattr(event, "open", 0) or 0)

    ts = getattr(event, "ts", None)
    if isinstance(ts, _datetime_type):
        hour_key = int(ts.timestamp()) // 3600
        ts_str = ts.isoformat()
    elif isinstance(ts, str):
        ts_str = ts
        hour_key = _time_mod_time() // 3600
    else:
        ts_str = None
        hour_key = _time_mod_time() // 3600

    # Push external data (delegated to feature_hook sources via tick_processor)
    fh = coord._feature_hook
    if fh is not None:
        fh.push_external_data(symbol, event, tp)

    # Determine warmup status
    warmup_done = True
    if fh is not None:
        bar_count = fh.increment_bar_count(symbol)
        warmup_done = bar_count >= fh.warmup_bars

    # Single Rust call: features + predict + state + pre-built features dict
    result = tp.process_tick_full(
        symbol, close_f, volume, high, low, open_, hour_key,
        warmup_done=warmup_done, ts=ts_str,
    )

    # Use pre-built features dict from Rust (eliminates ~35us Python dict ops)
    features = result.features_dict

    # Cross-asset features (if enabled)
    if fh is not None and fh.cross_asset is not None:
        funding_rate = features.get("funding_rate")
        fh.cross_asset.on_bar(symbol, close=close_f, funding_rate=funding_rate,
                              high=high, low=low)
        cross_feats = fh.cross_asset.get_features(symbol)
        features.update(cross_feats)

    # Tag features with symbol for downstream hooks (alpha health, etc.)
    features["_symbol"] = symbol

    # Store last features in feature_hook for non-market event lookups
    if fh is not None:
        fh.set_last_features(symbol, features)

    # Skip Decimal conversion: pass Rust objects directly to snapshot/output.
    snapshot = _build_snapshot(
        raw_event=event,
        event_index=result.event_index,
        markets=result.markets,
        account=result.account,
        positions=result.positions,
        portfolio=result.portfolio,
        risk=result.risk,
        features=features,
        skip_convert=True,
    )

    coord._last_snapshot = snapshot

    # Build PipelineOutput for hooks + decision bridge
    out = PipelineOutput(
        markets=result.markets,
        account=result.account,
        positions=result.positions,
        portfolio=result.portfolio,
        risk=result.risk,
        features=features,
        event_index=result.event_index,
        last_event_id=result.last_event_id,
        last_ts=result.last_ts,
        snapshot=snapshot,
        advanced=True,
    )

    if coord._cfg.on_pipeline_output is not None:
        coord._cfg.on_pipeline_output(out)

    if coord._cfg.on_snapshot is not None:
        coord._cfg.on_snapshot(snapshot)

    if coord._decision_bridge is not None:
        coord._decision_bridge.on_pipeline_output(out)


def handle_decision_event(coord: Any, event: Any) -> None:
    """DECISION handler: v1.0 no-op (risk gates wired via gate_chain)."""
    return


def handle_execution_event(coord: Any, event: Any) -> None:
    """EXECUTION handler: all orders go through ExecutionBridge."""
    if coord._execution_bridge is None:
        raise RuntimeError("ExecutionBridge is not attached")
    coord._execution_bridge.handle_event(event)
