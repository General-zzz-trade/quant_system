# runner/recovery.py
"""Recovery infrastructure — checkpoint, restore, and event recording for crash safety.

Provides:
  - PeriodicCheckpointer: saves state snapshots at regular intervals (not just shutdown)
  - KillSwitchPersistence: persists/restores kill switch state across restarts
  - EventRecorder: records market/fill/control events to event_log
  - InferenceBridgeRecovery: checkpoint/restore inference bridge z-score state
  - FeatureHookRecovery: checkpoint/restore feature hook bar counts
  - reconcile_and_heal(): startup reconciliation that corrects state (not just warns)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR_DEFAULT = "data/live"


# ============================================================
# 1. Periodic Checkpointer
# ============================================================

class PeriodicCheckpointer:
    """Periodically saves state snapshots to SqliteStateStore.

    Without this, snapshots are only saved on graceful shutdown.
    If the process crashes, all state since last shutdown is lost.
    """

    def __init__(
        self,
        *,
        state_store: Any,
        get_snapshot: Callable[[], Any],
        interval_sec: float = 60.0,
    ) -> None:
        self._store = state_store
        self._get_snapshot = get_snapshot
        self._interval = interval_sec
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="periodic-checkpoint",
        )
        self._thread.start()
        logger.info("PeriodicCheckpointer started (interval=%.0fs)", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # Final checkpoint on stop
        self._save_once()

    def _run(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            self._save_once()

    def _save_once(self) -> None:
        try:
            snapshot = self._get_snapshot()
            if snapshot is not None:
                self._store.save(snapshot)
        except Exception:
            logger.exception("Periodic checkpoint failed")


# ============================================================
# 2. Kill Switch Persistence
# ============================================================

_KILL_SWITCH_FILE = "kill_switch_state.json"


def save_kill_switch_state(
    kill_switch: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist active kill switch records to JSON file.

    .. deprecated:: Use save_all_auxiliary_state() for atomic saves.
    """
    records = kill_switch.active_records()
    if not records:
        # Remove file if no active kills
        path = os.path.join(data_dir, _KILL_SWITCH_FILE)
        if os.path.exists(path):
            os.remove(path)
        return

    entries = []
    for rec in records:
        entries.append({
            "scope": rec.scope.value,
            "key": rec.key,
            "mode": rec.mode.value,
            "triggered_at": rec.triggered_at,
            "ttl_seconds": rec.ttl_seconds,
            "source": rec.source,
            "reason": rec.reason,
            "tags": list(rec.tags),
        })

    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, _KILL_SWITCH_FILE)
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
    logger.info("Kill switch state saved: %d active record(s)", len(entries))


def restore_kill_switch_state(
    kill_switch: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> int:
    """Restore kill switch records from JSON file. Returns count restored.

    .. deprecated:: Use restore_all_auxiliary_state() for atomic restores.
        Individual restore functions are kept for backward compatibility
        and as fallback when bundle file doesn't exist.
    """
    from risk.kill_switch import KillMode, KillScope

    path = os.path.join(data_dir, _KILL_SWITCH_FILE)
    if not os.path.exists(path):
        return 0

    try:
        with open(path) as f:
            entries = json.load(f)
    except Exception:
        logger.exception("Failed to read kill switch state file")
        return 0

    restored = 0
    now = time.time()
    for entry in entries:
        scope = KillScope(entry["scope"])
        mode = KillMode(entry["mode"])
        triggered_at = entry.get("triggered_at", now)
        ttl = entry.get("ttl_seconds")

        # Skip expired TTL records
        if ttl is not None and (triggered_at + ttl) < now:
            continue

        kill_switch.trigger(
            scope=scope,
            key=entry["key"],
            mode=mode,
            reason=entry.get("reason", "restored_from_checkpoint"),
            source=entry.get("source", "recovery"),
            ttl_seconds=ttl,
            tags=tuple(entry.get("tags", ())),
            now_ts=triggered_at,
        )
        restored += 1

    if restored:
        logger.warning(
            "Restored %d kill switch record(s) from checkpoint — "
            "trading may be restricted", restored,
        )
    return restored


# ============================================================
# 3. Event Recorder
# ============================================================

class EventRecorder:
    """Records pipeline events to SQLiteEventLog for replay recovery.

    Wires into the coordinator's on_pipeline_output callback to capture
    market events and fill events as they flow through the pipeline.
    """

    def __init__(self, event_log: Any) -> None:
        self._log = event_log
        self._count = 0

    def on_pipeline_output(self, out: Any) -> None:
        """Called as on_pipeline_output hook — records the source event."""
        event = getattr(out, "event", None) or getattr(out, "raw_event", None)
        if event is None:
            return

        et = getattr(event, "event_type", None)
        et_str = getattr(et, "value", str(et) if et else "").lower()

        if et_str in ("market", "fill", "funding", "signal", "intent", "order", "risk", "control"):
            self._record_event(event, et_str)

    def record_fill(self, fill_event: Any) -> None:
        """Explicitly record a fill event."""
        self._record_event(fill_event, "fill")

    def record_market(self, market_event: Any) -> None:
        """Explicitly record a market event."""
        self._record_event(market_event, "market")

    def _record_event(self, event: Any, event_type: str) -> None:
        try:
            payload: Dict[str, Any] = {}
            symbol = getattr(event, "symbol", None)
            if symbol:
                payload["symbol"] = str(symbol)

            # Extract key fields based on type
            if event_type == "market":
                for field in ("open", "high", "low", "close", "volume"):
                    val = getattr(event, field, None)
                    if val is not None:
                        payload[field] = str(val)

            elif event_type == "fill":
                for field in ("side", "qty", "price", "fill_id", "order_id"):
                    val = getattr(event, field, None)
                    if val is not None:
                        payload[field] = str(val)

            elif event_type == "funding":
                rate = getattr(event, "rate", None)
                if rate is not None:
                    payload["rate"] = str(rate)

            elif event_type in ("signal", "intent"):
                for fld in ("side", "reason_code", "origin", "target_qty", "score"):
                    val = getattr(event, fld, None)
                    if val is not None:
                        payload[fld] = str(val)

            elif event_type == "order":
                for fld in ("side", "qty", "price", "order_id", "intent_id"):
                    val = getattr(event, fld, None)
                    if val is not None:
                        payload[fld] = str(val)

            elif event_type == "risk":
                for fld in ("rule_id", "level", "message"):
                    val = getattr(event, fld, None)
                    if val is not None:
                        payload[fld] = str(val)

            elif event_type == "control":
                for fld in ("command", "reason"):
                    val = getattr(event, fld, None)
                    if val is not None:
                        payload[fld] = str(val)

            ts = getattr(event, "ts", None)
            if ts is not None:
                payload["ts"] = str(ts)

            self._log.append(
                event_type=event_type,
                correlation_id=str(symbol) if symbol else None,
                payload=payload,
            )
            self._count += 1
        except Exception:
            logger.debug("Event recording failed", exc_info=True)

    @property
    def count(self) -> int:
        return self._count


# ============================================================
# 4. Inference Bridge Recovery
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
    """Restore inference bridge state from checkpoint. Returns True if restored.

    .. deprecated:: Use restore_all_auxiliary_state() for atomic restores.
        Individual restore functions are kept for backward compatibility
        and as fallback when bundle file doesn't exist.
    """
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
# 5. Feature Hook Recovery
# ============================================================

_FEATURE_HOOK_FILE = "feature_hook_state.json"


def save_feature_hook_state(
    feature_hook: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist feature hook bar counts and engine rolling-window state.

    Saves per-symbol RustFeatureEngine checkpoints (bar history) so that
    crash recovery rebuilds all rolling windows, not just bar_count.

    .. deprecated:: Use save_all_auxiliary_state() for atomic saves.
    """
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
    """Restore feature hook bar counts and engine rolling-window state.

    When engine checkpoints are available, replays stored bars through
    RustFeatureEngine to rebuild all rolling windows (EMA, RSI, ATR, etc.).
    Falls back to bar_count-only restore if no engine checkpoints exist.

    .. deprecated:: Use restore_all_auxiliary_state() for atomic restores.
        Individual restore functions are kept for backward compatibility
        and as fallback when bundle file doesn't exist.
    """
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
# 6. Exit Manager Recovery
# ============================================================

_EXIT_MANAGER_FILE = "exit_manager_checkpoint.json"


def save_exit_manager_state(
    exit_manager: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist exit manager trailing stop state.

    .. deprecated:: Use save_all_auxiliary_state() for atomic saves.
    """
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
    """Restore exit manager state from checkpoint. Returns True if restored.

    .. deprecated:: Use restore_all_auxiliary_state() for atomic restores.
        Individual restore functions are kept for backward compatibility
        and as fallback when bundle file doesn't exist.
    """
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
# 7. Regime Gate Recovery
# ============================================================

_REGIME_GATE_FILE = "regime_gate_checkpoint.json"


def save_regime_gate_state(
    regime_gate: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist regime gate buffer state.

    .. deprecated:: Use save_all_auxiliary_state() for atomic saves.
    """
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
    """Restore regime gate state from checkpoint. Returns True if restored.

    .. deprecated:: Use restore_all_auxiliary_state() for atomic restores.
        Individual restore functions are kept for backward compatibility
        and as fallback when bundle file doesn't exist.
    """
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
# 8. Correlation Computer Recovery
# ============================================================

_CORRELATION_FILE = "correlation_checkpoint.json"


def save_correlation_state(
    correlation_computer: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist correlation computer state.

    .. deprecated:: Use save_all_auxiliary_state() for atomic saves.
    """
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
    """Restore correlation computer state from checkpoint. Returns True if restored.

    .. deprecated:: Use restore_all_auxiliary_state() for atomic restores.
        Individual restore functions are kept for backward compatibility
        and as fallback when bundle file doesn't exist.
    """
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
# 9. Timeout Tracker Recovery
# ============================================================

_TIMEOUT_TRACKER_FILE = "timeout_tracker_checkpoint.json"


def save_timeout_tracker_state(
    timeout_tracker: Any,
    data_dir: str = _CHECKPOINT_DIR_DEFAULT,
) -> None:
    """Persist timeout tracker pending orders.

    .. deprecated:: Use save_all_auxiliary_state() for atomic saves.
    """
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
    """Restore timeout tracker state from checkpoint. Returns True if restored.

    .. deprecated:: Use restore_all_auxiliary_state() for atomic restores.
        Individual restore functions are kept for backward compatibility
        and as fallback when bundle file doesn't exist.
    """
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


# ============================================================
# 10. Atomic Checkpoint Bundle
# ============================================================

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
    """PRIMARY ENTRY POINT: Atomically save all auxiliary component states.

    Saves kill_switch, inference_bridge, feature_hook, exit_manager,
    regime_gate, correlation_computer, timeout_tracker, and drawdown_breaker
    to a single atomic JSON bundle (write-to-temp + os.replace() for POSIX
    atomicity). Also writes individual files for backward compatibility.

    This is the recommended save path — prefer this over individual
    save_*_state() functions to ensure cross-component consistency.
    """
    # Individual saves (backwards compat)
    save_kill_switch_state(kill_switch, data_dir=data_dir) if kill_switch else None
    save_inference_bridge_state(inference_bridge, data_dir=data_dir) if inference_bridge else None
    save_feature_hook_state(feature_hook, data_dir=data_dir) if feature_hook else None
    save_exit_manager_state(exit_manager, data_dir=data_dir) if exit_manager else None
    save_regime_gate_state(regime_gate, data_dir=data_dir) if regime_gate else None
    save_correlation_state(correlation_computer, data_dir=data_dir) if correlation_computer else None
    save_timeout_tracker_state(timeout_tracker, data_dir=data_dir) if timeout_tracker else None

    # Atomic bundle
    bundle: Dict[str, Any] = {}

    if inference_bridge is not None:
        bridges = inference_bridge if isinstance(inference_bridge, dict) else {"_default": inference_bridge}
        bridge_data = {}
        for key, bridge in bridges.items():
            if hasattr(bridge, "checkpoint"):
                try:
                    bridge_data[key] = bridge.checkpoint()
                except Exception:
                    pass
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
    """PRIMARY ENTRY POINT: Restore all auxiliary component states.

    Restores kill_switch, inference_bridge, feature_hook, exit_manager,
    regime_gate, correlation_computer, timeout_tracker, and drawdown_breaker.
    Prefers the atomic bundle file when available, falling back to individual
    checkpoint files per component for backward compatibility.

    Returns dict of component_name -> restored (bool).
    """
    results: Dict[str, bool] = {}

    # Try atomic bundle first
    bundle_path = os.path.join(data_dir, _AUXILIARY_BUNDLE_FILE)
    bundle: Optional[Dict[str, Any]] = None
    if os.path.exists(bundle_path):
        try:
            with open(bundle_path) as f:
                bundle = json.load(f)
        except Exception:
            logger.warning("Failed to read auxiliary bundle — falling back to individual files")
            bundle = None

    # Kill switch: always uses individual file (has its own format)
    if kill_switch is not None:
        results["kill_switch"] = restore_kill_switch_state(kill_switch, data_dir=data_dir) > 0

    # Inference bridge
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

    # Feature hook
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

    # Drawdown breaker (bundle-only, no individual file fallback)
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

    # Simple checkpoint/restore components
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


# ============================================================
# 11. Startup Reconciliation with Healing
# ============================================================

def reconcile_and_heal(
    coordinator: Any,
    venue_state: Dict[str, Any],
    symbols: Tuple[str, ...],
) -> List[str]:
    """Compare local state vs exchange, and HEAL position mismatches.

    Unlike _reconcile_startup() which only logs warnings, this function
    updates local state to match exchange reality when mismatches are found.

    Returns list of actions taken.
    """
    actions: List[str] = []
    local_view = coordinator.get_state_view()

    venue_positions = venue_state.get("positions", {})
    local_positions = local_view.get("positions", {})

    for sym in symbols:
        local_pos = local_positions.get(sym)
        venue_pos = venue_positions.get(sym)

        local_qty = float(getattr(local_pos, "qty", 0) if local_pos else 0)
        venue_qty = float(venue_pos.get("qty", 0) if isinstance(venue_pos, dict) else 0)

        if abs(local_qty - venue_qty) > 1e-8:
            # Heal: update local position to match exchange
            try:
                store = getattr(coordinator, "_store", None)
                if store is not None and hasattr(store, "set_position_qty"):
                    store.set_position_qty(sym, venue_qty)
                    actions.append(
                        f"{sym}: healed position {local_qty:.6f} → {venue_qty:.6f}"
                    )
                    logger.warning(
                        "Position healed: %s local=%.6f → venue=%.6f",
                        sym, local_qty, venue_qty,
                    )
                else:
                    # Fallback: log mismatch without healing (store doesn't support set_position_qty)
                    actions.append(
                        f"{sym}: position mismatch local={local_qty:.6f} venue={venue_qty:.6f} (no heal API)"
                    )
                    logger.warning(
                        "Position mismatch (no heal API): %s local=%.6f venue=%.6f",
                        sym, local_qty, venue_qty,
                    )
            except Exception:
                actions.append(
                    f"{sym}: position heal failed local={local_qty:.6f} venue={venue_qty:.6f}"
                )
                logger.exception("Position heal failed for %s", sym)

    # Check balance
    local_account = local_view.get("account")
    local_balance = float(getattr(local_account, "balance", 0) if local_account else 0)
    venue_balance = float(venue_state.get("balance", 0))
    if abs(local_balance - venue_balance) > 0.01:
        actions.append(
            f"Balance: local={local_balance:.2f} venue={venue_balance:.2f} (info only)"
        )
        logger.warning(
            "Balance mismatch: local=%.2f venue=%.2f — will converge on next funding/fill",
            local_balance, venue_balance,
        )

    return actions
