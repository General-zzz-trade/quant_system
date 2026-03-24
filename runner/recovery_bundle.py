"""Atomic checkpoint bundle — save/restore all auxiliary state.

Extracted from recovery.py: save_all_auxiliary_state, restore_all_auxiliary_state.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from runner.recovery import (
    save_kill_switch_state,
    _CHECKPOINT_DIR_DEFAULT,
)
from runner.recovery_components import (
    save_inference_bridge_state,
    save_feature_hook_state,
    save_exit_manager_state,
    save_regime_gate_state,
    save_correlation_state,
    save_timeout_tracker_state,
    restore_kill_switch_state,
    restore_inference_bridge_state,
    restore_feature_hook_state,
    restore_exit_manager_state,
    restore_regime_gate_state,
    restore_correlation_state,
    restore_timeout_tracker_state,
)

logger = logging.getLogger(__name__)

_AUXILIARY_BUNDLE_FILE = "auxiliary_state_bundle.json"


def save_all_auxiliary_state(
    *,
    kill_switch: Any = None,
    inference_bridge: Any = None,
    feature_hook: Any = None,
    exit_manager: Any = None,
    regime_gate: Any = None,
    correlation_computer: Any = None,
    timeout_tracker: Any = None,
    drawdown_breaker: Any = None,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """PRIMARY ENTRY POINT: Atomically save all auxiliary component states."""
    save_kill_switch_state(kill_switch, data_dir=data_dir) if kill_switch else None
    save_inference_bridge_state(inference_bridge, data_dir=data_dir) if inference_bridge else None
    save_feature_hook_state(feature_hook, data_dir=data_dir) if feature_hook else None
    save_exit_manager_state(exit_manager, data_dir=data_dir) if exit_manager else None
    save_regime_gate_state(regime_gate, data_dir=data_dir) if regime_gate else None
    save_correlation_state(correlation_computer, data_dir=data_dir) if correlation_computer else None
    save_timeout_tracker_state(timeout_tracker, data_dir=data_dir) if timeout_tracker else None

    bundle: Dict[str, Any] = {}

    if inference_bridge is not None:
        bridges = inference_bridge if isinstance(inference_bridge, dict) else {"_default": inference_bridge}
        bridge_data = {}
        for key, bridge in bridges.items():
            if hasattr(bridge, "checkpoint"):
                try:
                    bridge_data[key] = bridge.checkpoint()
                except Exception as e:
                    logger.error("Failed to checkpoint inference bridge '%s': %s", key, e, exc_info=True)
        if bridge_data:
            bundle["inference_bridge"] = bridge_data

    if feature_hook is not None:
        bar_count = getattr(feature_hook, "_bar_count", None)
        if bar_count:
            bundle["feature_hook"] = {"bar_count": dict(bar_count)}

    for name, component in [
        ("exit_manager", exit_manager),
        ("regime_gate", regime_gate),
        ("correlation_computer", correlation_computer),
        ("timeout_tracker", timeout_tracker),
        ("drawdown_breaker", drawdown_breaker),
    ]:
        if component is not None and hasattr(component, "checkpoint"):
            try:
                bundle[name] = component.checkpoint()
            except Exception:
                logger.debug("Bundle checkpoint failed for %s", name)

    if not bundle:
        return

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _AUXILIARY_BUNDLE_FILE)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(bundle, f)
    os.replace(tmp_path, path)
    logger.info("Atomic auxiliary state bundle saved (%d components)", len(bundle))


def restore_all_auxiliary_state(
    *,
    kill_switch: Any = None,
    inference_bridge: Any = None,
    feature_hook: Any = None,
    exit_manager: Any = None,
    regime_gate: Any = None,
    correlation_computer: Any = None,
    timeout_tracker: Any = None,
    drawdown_breaker: Any = None,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> Dict[str, bool]:
    """PRIMARY ENTRY POINT: Restore all auxiliary component states."""
    results: Dict[str, bool] = {}

    bundle_path = os.path.join(data_dir, _AUXILIARY_BUNDLE_FILE)
    bundle: Optional[Dict[str, Any]] = None
    if os.path.exists(bundle_path):
        try:
            with open(bundle_path) as f:
                bundle = json.load(f)
        except Exception:
            logger.warning("Failed to read auxiliary bundle — falling back to individual files")
            bundle = None

    if kill_switch is not None:
        results["kill_switch"] = restore_kill_switch_state(kill_switch, data_dir=data_dir) > 0

    if inference_bridge is not None:
        if bundle and "inference_bridge" in bundle:
            bridges = inference_bridge if isinstance(inference_bridge, dict) else {"_default": inference_bridge}
            restored = False
            for key, bridge in bridges.items():
                data = bundle["inference_bridge"].get(key)
                if data is not None and hasattr(bridge, "restore"):
                    try:
                        bridge.restore(data)
                        restored = True
                    except Exception:
                        logger.exception("Bundle inference bridge restore failed for %s", key)
            results["inference_bridge"] = restored
        else:
            results["inference_bridge"] = restore_inference_bridge_state(inference_bridge, data_dir=data_dir)

    if feature_hook is not None:
        if bundle and "feature_hook" in bundle:
            bar_count = bundle["feature_hook"].get("bar_count", {})
            if bar_count and hasattr(feature_hook, "_bar_count"):
                feature_hook._bar_count = {k: int(v) for k, v in bar_count.items()}
                results["feature_hook"] = True
            else:
                results["feature_hook"] = False
        else:
            results["feature_hook"] = restore_feature_hook_state(feature_hook, data_dir=data_dir)

    if drawdown_breaker is not None:
        if bundle and "drawdown_breaker" in bundle and hasattr(drawdown_breaker, "restore_checkpoint"):
            try:
                drawdown_breaker.restore_checkpoint(bundle["drawdown_breaker"])
                results["drawdown_breaker"] = True
            except Exception:
                logger.exception("Bundle restore failed for drawdown_breaker")
                results["drawdown_breaker"] = False
        else:
            results["drawdown_breaker"] = False

    for name, component, fallback_fn in [
        ("exit_manager", exit_manager, restore_exit_manager_state),
        ("regime_gate", regime_gate, restore_regime_gate_state),
        ("correlation_computer", correlation_computer, restore_correlation_state),
        ("timeout_tracker", timeout_tracker, restore_timeout_tracker_state),
    ]:
        if component is not None:
            if bundle and name in bundle and hasattr(component, "restore"):
                try:
                    component.restore(bundle[name])
                    results[name] = True
                except Exception:
                    logger.exception("Bundle restore failed for %s", name)
                    results[name] = False
            else:
                results[name] = fallback_fn(component, data_dir=data_dir)

    return results
