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


# Re-export per-component checkpoint functions from recovery_components.py
from runner.recovery_components import (  # noqa: F401, E402
    save_inference_bridge_state,
    restore_inference_bridge_state,
    save_feature_hook_state,
    restore_feature_hook_state,
    save_exit_manager_state,
    restore_exit_manager_state,
    save_regime_gate_state,
    restore_regime_gate_state,
    save_correlation_state,
    restore_correlation_state,
    save_timeout_tracker_state,
    restore_timeout_tracker_state,
)


# Atomic checkpoint bundle (extracted to runner/recovery_bundle.py)
from runner.recovery_bundle import (  # noqa: F401, E402
    save_all_auxiliary_state,
    restore_all_auxiliary_state,
)


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
