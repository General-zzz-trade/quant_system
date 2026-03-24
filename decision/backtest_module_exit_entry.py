"""Exit and entry logic for MLSignalDecisionModule.

Extracted from backtest_module.py to keep file under 500 lines.
Contains the short model processing logic.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def process_short_model(module, features: Dict[str, float], snapshot: Any) -> None:
    """Process independent short model signal (mirrors live bridge.py:287-330)."""
    ts = module._get_timestamp_utc(snapshot)
    short_sig = module._short_model.predict(
        symbol=module.symbol, ts=ts, features=features)
    short_score = 0.0
    if short_sig is not None and short_sig.strength is not None:
        raw = short_sig.strength
        if short_sig.side == "short":
            raw = -raw
        elif short_sig.side == "flat":
            raw = 0.0

        # Apply Rust constraint path if available (same as live)
        if module._rust_bridge is not None:
            hour_key = int(ts.timestamp()) // 3600 if ts is not None else 0
            short_score = module._rust_bridge.process_short_signal(
                module.symbol, raw, hour_key,
                module._deadzone, module._min_hold)
        else:
            # Python fallback for short signal (deadzone + min-hold).
            if not hasattr(module, '_short_hold_count'):
                module._short_hold_count = 0
                module._prev_short_score = 0.0

            if abs(raw) > module._deadzone:
                short_score = raw
                module._short_hold_count += 1
            elif module._prev_short_score != 0.0:
                if module._short_hold_count < module._min_hold:
                    short_score = module._prev_short_score
                    module._short_hold_count += 1
                else:
                    module._short_hold_count = 0
            else:
                module._short_hold_count = 0
            module._prev_short_score = short_score

        # Vol-adaptive sizing for short
        if short_score != 0.0 and module._vol_target is not None:
            vol_val = features.get(module._vol_feature)
            if vol_val is not None and vol_val > 1e-8:
                scale = min(module._vol_target / float(vol_val), 1.0)
                short_score *= scale

    features[module._short_score_key] = short_score
    module._last_short_score = short_score


def handle_bear_model_exit(module, features, snapshot, close, current_qty, event_id, events):
    """Handle bear model logic during exit when monthly gate fails.

    Returns (should_return_early, events).
    """
    bear_score = None
    if module._bear_model is not None:
        ts = module._get_timestamp_utc(snapshot)
        bear_sig = module._bear_model.predict(
            symbol=module.symbol, ts=ts, features=features)
        if bear_sig is not None and bear_sig.side == "long":
            if module._bear_thresholds:
                prob = 0.5 + bear_sig.strength
                bear_score = 0.0
                for thresh, s in module._bear_thresholds:
                    if prob > thresh:
                        bear_score = s
                        break
            else:
                bear_score = -1.0
        else:
            bear_score = 0.0

    if bear_score is not None and bear_score != 0:
        # Bear model says stay in position — sync Rust hold state
        if module._rust_bridge is not None:
            cur_rust_pos = module._rust_bridge.get_position(module.symbol)
            if bear_score != cur_rust_pos:
                module._rust_bridge.set_position(module.symbol, bear_score, 1)
        module._position = bear_score
        return True, events
    else:
        # No bear model or bear score is 0 -> flatten
        if module._rust_bridge is not None:
            cur_rust_pos = module._rust_bridge.get_position(module.symbol)
            if cur_rust_pos != 0.0:
                module._rust_bridge.set_position(module.symbol, 0.0, 1)
        side = "sell" if module._position > 0 else "buy"
        events.extend(module._make_order(
            side=side,
            qty=abs(current_qty),
            event_id=event_id,
            reason="monthly_gate",
        ))
        module._exit_mgr.on_exit(module.symbol)
        module._position = 0.0
        return True, events


def handle_bear_model_entry(module, features, snapshot, desired):
    """Handle bear model logic during entry when monthly gate fails.

    Returns adjusted desired signal.
    """
    if module._bear_model is not None:
        ts = module._get_timestamp_utc(snapshot)
        bear_sig = module._bear_model.predict(
            symbol=module.symbol, ts=ts, features=features)
        if bear_sig is not None and bear_sig.side == "long":
            if module._bear_thresholds:
                prob = 0.5 + bear_sig.strength
                bear_entry_score = 0.0
                for thresh, s in module._bear_thresholds:
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
    return desired
