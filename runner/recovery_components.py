# runner/recovery_components.py
"""Per-component checkpoint save/restore functions.

Extracted from recovery.py to reduce file size. These individual functions
are used as fallbacks by save_all_auxiliary_state/restore_all_auxiliary_state
when the atomic bundle file is not available.

Prefer save_all_auxiliary_state() / restore_all_auxiliary_state() over calling
these individually.
"""
from __future__ import annotations

import json
import logging

# Re-export from recovery.py for backward compat
from runner.recovery import restore_kill_switch_state  # noqa: F401
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR_DEFAULT = "data/live"


# ============================================================
# Inference Bridge Recovery
# ============================================================

_BRIDGE_CHECKPOINT_FILE = "inference_bridge_checkpoint.json"


def save_inference_bridge_state(
    inference_bridge: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist inference bridge z-score/signal state.

    .. deprecated:: Use save_all_auxiliary_state() for atomic saves.
    """
    if inference_bridge is None:
        return

    bridges: Dict[str, Any] = {}
    if isinstance(inference_bridge, dict):
        bridges = inference_bridge
    else:
        bridges = {"_default": inference_bridge}

    all_checkpoints: Dict[str, Any] = {}
    for key, bridge in bridges.items():
        if hasattr(bridge, "checkpoint"):
            try:
                all_checkpoints[key] = bridge.checkpoint()
            except Exception:
                logger.debug("Inference bridge checkpoint failed for %s", key)

    if not all_checkpoints:
        return

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _BRIDGE_CHECKPOINT_FILE)
    with open(path, "w") as f:
        json.dump(all_checkpoints, f)
    logger.info("Inference bridge state saved (%d bridge(s))", len(all_checkpoints))


def restore_inference_bridge_state(
    inference_bridge: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> bool:
    """Restore inference bridge state from checkpoint. Returns True if restored."""
    if inference_bridge is None:
        return False

    path = os.path.join(data_dir, _BRIDGE_CHECKPOINT_FILE)
    if not os.path.exists(path):
        return False

    try:
        with open(path) as f:
            all_checkpoints = json.load(f)
    except Exception:
        logger.exception("Failed to read inference bridge checkpoint")
        return False

    bridges: Dict[str, Any] = {}
    if isinstance(inference_bridge, dict):
        bridges = inference_bridge
    else:
        bridges = {"_default": inference_bridge}

    restored = False
    for key, bridge in bridges.items():
        data = all_checkpoints.get(key)
        if data is not None and hasattr(bridge, "restore"):
            try:
                bridge.restore(data)
                restored = True
                logger.info("Inference bridge state restored for %s", key)
            except Exception:
                logger.exception("Inference bridge restore failed for %s", key)

    return restored


# ============================================================
# Feature Hook Recovery
# ============================================================

_FEATURE_HOOK_FILE = "feature_hook_state.json"


def save_feature_hook_state(
    feature_hook: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist feature hook bar counts and engine rolling-window state."""
    if feature_hook is None:
        return

    bar_count = getattr(feature_hook, "_bar_count", None)
    if not bar_count:
        return

    state: Dict[str, Any] = {"bar_count": dict(bar_count)}

    # Checkpoint per-symbol RustFeatureEngine bar history
    engines = getattr(feature_hook, "_rust_engines", None)
    if engines:
        engine_checkpoints: Dict[str, str] = {}
        for symbol, engine in engines.items():
            if hasattr(engine, "checkpoint"):
                try:
                    engine_checkpoints[symbol] = engine.checkpoint()
                except Exception:
                    logger.debug("Engine checkpoint failed for %s", symbol)
        if engine_checkpoints:
            state["engine_checkpoints"] = engine_checkpoints

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _FEATURE_HOOK_FILE)
    with open(path, "w") as f:
        json.dump(state, f)
    logger.info("Feature hook state saved (bar_count=%s, engines=%d)",
                dict(bar_count), len(state.get("engine_checkpoints", {})))


def restore_feature_hook_state(
    feature_hook: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> bool:
    """Restore feature hook bar counts and engine rolling-window state."""
    if feature_hook is None:
        return False

    path = os.path.join(data_dir, _FEATURE_HOOK_FILE)
    if not os.path.exists(path):
        return False

    try:
        with open(path) as f:
            state = json.load(f)
    except Exception:
        logger.exception("Failed to read feature hook checkpoint")
        return False

    restored = False

    # Restore engine rolling-window state via checkpoint replay
    engine_checkpoints = state.get("engine_checkpoints", {})
    if engine_checkpoints and hasattr(feature_hook, "_rust_engines"):
        try:
            from _quant_hotpath import RustFeatureEngine as _RustFeatureEngine
        except ImportError:
            _RustFeatureEngine = None

        if _RustFeatureEngine is not None:
            for symbol, json_str in engine_checkpoints.items():
                try:
                    engine = _RustFeatureEngine()
                    n_replayed = engine.restore_checkpoint(json_str)
                    feature_hook._rust_engines[symbol] = engine
                    # Set bar_count from restored engine
                    if hasattr(feature_hook, "_bar_count"):
                        feature_hook._bar_count[symbol] = engine.bar_count
                    logger.info("Feature engine restored for %s (%d bars replayed)", symbol, n_replayed)
                    restored = True
                except Exception:
                    logger.exception("Feature engine restore failed for %s", symbol)

    # Fallback: restore bar_count only (for symbols without engine checkpoints)
    bar_count = state.get("bar_count", {})
    if bar_count and hasattr(feature_hook, "_bar_count"):
        for k, v in bar_count.items():
            if k not in feature_hook._bar_count:
                feature_hook._bar_count[k] = int(v)
        if not restored:
            logger.info("Feature hook bar_count restored (no engine checkpoints): %s", feature_hook._bar_count)
            restored = True

    return restored


# ============================================================
# Exit Manager Recovery
# ============================================================

_EXIT_MANAGER_FILE = "exit_manager_checkpoint.json"


def save_exit_manager_state(
    exit_manager: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist exit manager trailing stop state."""
    if exit_manager is None or not hasattr(exit_manager, "checkpoint"):
        return
    try:
        data = exit_manager.checkpoint()
    except Exception:
        logger.debug("Exit manager checkpoint failed")
        return
    if not data:
        return
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _EXIT_MANAGER_FILE)
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info("Exit manager state saved (%d position(s))", len(data))


def restore_exit_manager_state(
    exit_manager: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> bool:
    """Restore exit manager state from checkpoint. Returns True if restored."""
    if exit_manager is None or not hasattr(exit_manager, "restore"):
        return False
    path = os.path.join(data_dir, _EXIT_MANAGER_FILE)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read exit manager checkpoint")
        return False
    try:
        exit_manager.restore(data)
        logger.info("Exit manager state restored (%d position(s))", len(data))
        return True
    except Exception:
        logger.exception("Exit manager restore failed")
        return False


# ============================================================
# Regime Gate Recovery
# ============================================================

_REGIME_GATE_FILE = "regime_gate_checkpoint.json"


def save_regime_gate_state(
    regime_gate: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist regime gate buffer state."""
    if regime_gate is None or not hasattr(regime_gate, "checkpoint"):
        return
    try:
        data = regime_gate.checkpoint()
    except Exception:
        logger.debug("Regime gate checkpoint failed")
        return
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _REGIME_GATE_FILE)
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info("Regime gate state saved (%d bb_width entries)", len(data.get("bb_width_buf", [])))


def restore_regime_gate_state(
    regime_gate: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> bool:
    """Restore regime gate state from checkpoint. Returns True if restored."""
    if regime_gate is None or not hasattr(regime_gate, "restore"):
        return False
    path = os.path.join(data_dir, _REGIME_GATE_FILE)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read regime gate checkpoint")
        return False
    try:
        regime_gate.restore(data)
        logger.info("Regime gate state restored")
        return True
    except Exception:
        logger.exception("Regime gate restore failed")
        return False


# ============================================================
# Correlation Computer Recovery
# ============================================================

_CORRELATION_FILE = "correlation_checkpoint.json"


def save_correlation_state(
    correlation_computer: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist correlation computer state."""
    if correlation_computer is None or not hasattr(correlation_computer, "checkpoint"):
        return
    try:
        data = correlation_computer.checkpoint()
    except Exception:
        logger.debug("Correlation computer checkpoint failed")
        return
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _CORRELATION_FILE)
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info("Correlation state saved (%d symbols)", len(data.get("last_prices", {})))


def restore_correlation_state(
    correlation_computer: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> bool:
    """Restore correlation computer state from checkpoint. Returns True if restored."""
    if correlation_computer is None or not hasattr(correlation_computer, "restore"):
        return False
    path = os.path.join(data_dir, _CORRELATION_FILE)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read correlation checkpoint")
        return False
    try:
        correlation_computer.restore(data)
        logger.info("Correlation state restored (%d symbols)", len(data.get("last_prices", {})))
        return True
    except Exception:
        logger.exception("Correlation restore failed")
        return False


# ============================================================
# Timeout Tracker Recovery
# ============================================================

_TIMEOUT_TRACKER_FILE = "timeout_tracker_checkpoint.json"


def save_timeout_tracker_state(
    timeout_tracker: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist timeout tracker pending orders."""
    if timeout_tracker is None or not hasattr(timeout_tracker, "checkpoint"):
        return
    try:
        data = timeout_tracker.checkpoint()
    except Exception:
        logger.debug("Timeout tracker checkpoint failed")
        return
    if not data.get("pending"):
        return
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _TIMEOUT_TRACKER_FILE)
    with open(path, "w") as f:
        json.dump(data, f)
    logger.info("Timeout tracker state saved (%d pending)", len(data.get("pending", {})))


def restore_timeout_tracker_state(
    timeout_tracker: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> bool:
    """Restore timeout tracker state from checkpoint. Returns True if restored."""
    if timeout_tracker is None or not hasattr(timeout_tracker, "restore"):
        return False
    path = os.path.join(data_dir, _TIMEOUT_TRACKER_FILE)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        logger.exception("Failed to read timeout tracker checkpoint")
        return False
    try:
        timeout_tracker.restore(data)
        logger.info("Timeout tracker state restored (%d pending)", len(data.get("pending", {})))
        return True
    except Exception:
        logger.exception("Timeout tracker restore failed")
        return False
