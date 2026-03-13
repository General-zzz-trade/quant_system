# runner/replay_runner.py
"""Replay runner — replays recorded events through the full live pipeline.

Supports two modes:
  1. State-only replay (default): market events → state updates only.
  2. Full-chain replay (capture_orders=True or decision_modules provided):
     market → features → decision → order → fill → position update.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from engine.replay import EventReplay, ReplayConfig


# ============================================================
# Result type
# ============================================================

@dataclass
class ReplayResult:
    """Result of a full-chain replay run."""
    events_processed: int
    order_log: List[Dict[str, Any]] = field(default_factory=list)
    captured_orders: List[Any] = field(default_factory=list)
    captured_signals: List[Any] = field(default_factory=list)
    final_state: Optional[Dict[str, Any]] = None
    account_snapshots: List[Dict[str, Any]] = field(default_factory=list)


# ============================================================
# JSONL event source
# ============================================================

class JsonlEventSource:
    """Reads events from a JSON Lines file, yielding decoded dicts.

    Each line must be a JSON object with at minimum an 'event_type' field.
    Compatible with the EventSource protocol (engine/replay.py).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._count: Optional[int] = None

    def __len__(self) -> int:
        if self._count is None:
            with self._path.open("r") as f:
                self._count = sum(1 for line in f if line.strip())
        return self._count

    def __iter__(self) -> Iterator[Any]:
        try:
            from event.codec import decode_event_json
            use_codec = True
        except Exception:
            use_codec = False

        with self._path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if use_codec:
                    try:
                        yield decode_event_json(line)
                        continue
                    except Exception:
                        pass
                # Fallback: yield raw dict
                yield json.loads(line)


# ============================================================
# Replay runner
# ============================================================

def run_replay(
    *,
    event_log_path: Path,
    symbol: Optional[str] = None,
    out_dir: Optional[Path] = None,
    decision_modules: Optional[Sequence[Any]] = None,
    capture_orders: bool = False,
    coordinator_config: Optional[CoordinatorConfig] = None,
    starting_balance: Optional[float] = None,
) -> ReplayResult:
    """Replay events from a JSONL event log through the engine coordinator.

    Args:
        event_log_path: Path to the JSONL event log file.
        symbol: Default symbol (default: "BTCUSDT").
        out_dir: Optional output directory for replay summary.
        decision_modules: Optional list of DecisionModule instances.
            When provided, wires DecisionBridge + ExecutionBridge to enable
            full-chain replay (signal → order → fill → position).
        capture_orders: If True and decision_modules is provided, captures
            all order and signal events emitted during replay.
        coordinator_config: Optional pre-built CoordinatorConfig (overrides symbol).

    Returns:
        ReplayResult with events_processed, order_log, captured events, and final state.
    """
    source = JsonlEventSource(event_log_path)
    symbol_default = symbol or "BTCUSDT"

    cfg = coordinator_config or CoordinatorConfig(
        symbol_default=symbol_default.upper(),
        currency="USDT",
    )

    coordinator = EngineCoordinator(cfg=cfg)

    # Full-chain wiring: decision modules + execution adapter
    replay_adapter = None
    captured_orders: List[Any] = []
    captured_signals: List[Any] = []

    if decision_modules is not None:
        from execution.sim.replay_adapter import ReplayExecutionAdapter

        # Price source reads last close from coordinator state
        def _price_source(sym: str) -> Optional[Decimal]:
            try:
                view = coordinator.get_state_view()
                markets = view.get("markets", {})
                mkt = markets.get(sym)
                if mkt is not None:
                    return Decimal(str(getattr(mkt, "close", 0)))
            except Exception:
                pass
            return None

        adapter_kwargs: Dict[str, Any] = {"price_source": _price_source}
        if starting_balance is not None:
            adapter_kwargs["starting_balance"] = starting_balance
        replay_adapter = ReplayExecutionAdapter(**adapter_kwargs)

        # Capturing emit: intercepts events by type before forwarding
        def _capturing_emit(ev: Any, *, actor: str = "replay") -> None:
            et = getattr(ev, "event_type", None)
            et_val = getattr(et, "value", str(et) if et else "").lower()
            if et_val in ("order",):
                captured_orders.append(ev)
            elif et_val in ("signal", "intent"):
                captured_signals.append(ev)
            coordinator.emit(ev, actor=actor)

        decision_bridge = DecisionBridge(
            dispatcher_emit=_capturing_emit,
            modules=list(decision_modules),
        )
        execution_bridge = ExecutionBridge(
            adapter=replay_adapter,
            dispatcher_emit=_capturing_emit,
        )

        coordinator.attach_decision_bridge(decision_bridge)
        coordinator.attach_execution_bridge(execution_bridge)

    coordinator.start()

    replay = EventReplay(
        dispatcher=coordinator.dispatcher,
        source=source,
        config=ReplayConfig(strict_order=False, actor="replay"),
    )

    processed = replay.run()
    final_state = dict(coordinator.get_state_view())
    coordinator.stop()

    result = ReplayResult(
        events_processed=processed,
        order_log=replay_adapter.order_log if replay_adapter else [],
        captured_orders=captured_orders,
        captured_signals=captured_signals,
        final_state=final_state,
        account_snapshots=replay_adapter.account_snapshots if replay_adapter else [],
    )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "events_processed": processed,
            "source": str(event_log_path),
            "orders": len(result.order_log),
            "signals": len(result.captured_signals),
        }
        (out_dir / "replay_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    return result


def run_replay_from_events(
    *,
    events: Sequence[Any],
    symbol: str = "BTCUSDT",
    decision_modules: Optional[Sequence[Any]] = None,
    coordinator_config: Optional[CoordinatorConfig] = None,
) -> ReplayResult:
    """Replay a sequence of in-memory events (no JSONL file needed).

    Same full-chain wiring as run_replay() but takes events directly.
    Useful for tests and programmatic replay.
    """
    cfg = coordinator_config or CoordinatorConfig(
        symbol_default=symbol.upper(),
        currency="USDT",
    )

    coordinator = EngineCoordinator(cfg=cfg)

    replay_adapter = None
    captured_orders: List[Any] = []
    captured_signals: List[Any] = []

    if decision_modules is not None:
        from execution.sim.replay_adapter import ReplayExecutionAdapter

        def _price_source(sym: str) -> Optional[Decimal]:
            try:
                view = coordinator.get_state_view()
                markets = view.get("markets", {})
                mkt = markets.get(sym)
                if mkt is not None:
                    return Decimal(str(getattr(mkt, "close", 0)))
            except Exception:
                pass
            return None

        replay_adapter = ReplayExecutionAdapter(price_source=_price_source)

        def _capturing_emit(ev: Any, *, actor: str = "replay") -> None:
            et = getattr(ev, "event_type", None)
            et_val = getattr(et, "value", str(et) if et else "").lower()
            if et_val in ("order",):
                captured_orders.append(ev)
            elif et_val in ("signal", "intent"):
                captured_signals.append(ev)
            coordinator.emit(ev, actor=actor)

        decision_bridge = DecisionBridge(
            dispatcher_emit=_capturing_emit,
            modules=list(decision_modules),
        )
        execution_bridge = ExecutionBridge(
            adapter=replay_adapter,
            dispatcher_emit=_capturing_emit,
        )

        coordinator.attach_decision_bridge(decision_bridge)
        coordinator.attach_execution_bridge(execution_bridge)

    coordinator.start()

    replay = EventReplay(
        dispatcher=coordinator.dispatcher,
        source=events,
        config=ReplayConfig(strict_order=False, actor="replay"),
    )

    processed = replay.run()
    final_state = dict(coordinator.get_state_view())
    coordinator.stop()

    return ReplayResult(
        events_processed=processed,
        order_log=replay_adapter.order_log if replay_adapter else [],
        captured_orders=captured_orders,
        captured_signals=captured_signals,
        final_state=final_state,
    )
