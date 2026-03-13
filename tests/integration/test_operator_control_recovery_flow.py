from __future__ import annotations

import pickle
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch
from decimal import Decimal
import time

from monitoring.alerts.base import Alert, Severity
from monitoring.alerts.manager import AlertManager
from monitoring.health_server import _HealthHandler
from execution.bridge.live_execution_bridge import LiveExecutionBridge
from execution.ingress.router import FillIngressRouter
from engine.coordinator import CoordinatorConfig, EngineCoordinator
from execution.reconcile.controller import ReconcileController
from execution.safety.timeout_tracker import OrderTimeoutTracker
from execution.state_machine.machine import OrderStateMachine
from execution.state_machine.transitions import OrderStatus
from execution.store.event_log import InMemoryEventLog
from risk.kill_switch import KillSwitch
from runner.control_plane import OperatorControlPlane
from runner.live_runner import LiveRunner, LiveRunnerConfig
import io
import json


class _RecordingSink:
    def __init__(self) -> None:
        self.alerts: list[object] = []

    def emit(self, alert: object) -> None:
        self.alerts.append(alert)


class _FakeTransport:
    def connect(self, url: str) -> None:
        pass

    def recv(self, timeout_s: float = 5.0):
        return None

    def close(self) -> None:
        pass


class _FakeVenueClient:
    def send_order(self, order_event: object) -> list[object]:
        return []


class _Ack:
    def __init__(self, *, ok: bool, status: str, error: str = "", command_id: str = "cmd-001") -> None:
        self.ok = ok
        self.status = status
        self.error = error
        self.command_id = command_id
        self.venue = "binance"
        self.result = {"price": "40000", "qty": "1.0", "fee": "0.1"} if ok else {}


class _SequencedBridge:
    def __init__(self, acks: list[_Ack]) -> None:
        self.acks = list(acks)

    def submit(self, _cmd: object) -> _Ack:
        return self.acks.pop(0)


def _create_stub_model_weights() -> bytes:
    return pickle.dumps({
        "model": None,
        "features": ("sma_20", "rsi_14"),
        "is_classifier": False,
    })


def _snapshot_state(sm: OrderStateMachine, order_id: str) -> dict:
    state = sm.get(order_id)
    assert state is not None
    return {
        "order_id": state.order_id,
        "client_order_id": state.client_order_id,
        "symbol": state.symbol,
        "side": state.side,
        "order_type": state.order_type,
        "qty": str(state.qty),
        "price": str(state.price) if state.price is not None else None,
        "transitions": [
            {
                "to_status": t.to_status.value,
                "ts_ms": t.ts_ms,
                "reason": t.reason,
            }
            for t in state.transitions
        ],
    }


def _restore_state(snapshot: dict) -> OrderStateMachine:
    sm = OrderStateMachine()
    sm.register(
        order_id=snapshot["order_id"],
        client_order_id=snapshot["client_order_id"],
        symbol=snapshot["symbol"],
        side=snapshot["side"],
        order_type=snapshot["order_type"],
        qty=Decimal(snapshot["qty"]),
        price=Decimal(snapshot["price"]) if snapshot["price"] is not None else None,
    )
    for item in snapshot["transitions"]:
        sm.transition(
            order_id=snapshot["order_id"],
            new_status=OrderStatus(item["to_status"]),
            ts_ms=int(item["ts_ms"]),
            reason=item["reason"],
        )
    return sm


def test_flush_drift_then_manual_halt_updates_status_and_alert_chain() -> None:
    report = SimpleNamespace(ok=False, should_halt=True, all_drifts=("position", "fill"))
    sink = _RecordingSink()
    runner = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
        reconcile_scheduler=SimpleNamespace(last_report=report, run_once=lambda: report),
        alert_manager=AlertManager(sink=sink),
    )
    plane = OperatorControlPlane(runner)

    flush_result = plane.execute({"command": "flush", "reason": "drift_review", "source": "ops"})
    halt_result = plane.execute({"command": "halt", "reason": "critical_drift", "source": "ops"})

    assert flush_result.accepted is True
    assert flush_result.outcome == "drift"
    assert halt_result.accepted is True
    assert halt_result.outcome == "hard_kill"
    assert halt_result.status is not None
    assert halt_result.status["kill_switch"]["mode"] == "hard_kill"
    assert [alert.title for alert in sink.alerts] == [
        "operator_flush",
        "execution-reconcile-drift",
        "operator_halt",
    ]


def test_health_control_endpoint_drives_runtime_and_records_history() -> None:
    report = SimpleNamespace(ok=False, should_halt=False, all_drifts=("position",))
    sink = _RecordingSink()
    runner = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
        reconcile_scheduler=SimpleNamespace(last_report=report, run_once=lambda: report),
        alert_manager=AlertManager(sink=sink),
    )
    plane = OperatorControlPlane(runner)

    handler_cls = type(
        "Handler",
        (_HealthHandler,),
        {
            "status_fn": staticmethod(lambda: {"status": "ok"}),
            "operator_fn": staticmethod(runner.operator_status),
            "control_history_fn": staticmethod(
                lambda: [
                    {
                        "command": rec.command,
                        "reason": rec.reason,
                        "source": rec.source,
                        "result": rec.result,
                        "ts": rec.ts.isoformat(),
                    }
                    for rec in runner.control_history
                ]
            ),
            "control_fn": staticmethod(lambda body: plane.execute(body).to_dict()),
            "auth_token": None,
        },
    )

    handler = handler_cls.__new__(handler_cls)
    handler.path = "/control"
    handler.headers = {"Content-Length": "54"}
    payload = json.dumps({"command": "flush", "reason": "manual_review", "source": "ops"}).encode("utf-8")
    handler.headers["Content-Length"] = str(len(payload))
    handler.rfile = io.BytesIO(payload)
    handler.wfile = io.BytesIO()
    response: dict[str, object] = {"code": None}
    handler.send_response = lambda code: response.__setitem__("code", code)
    handler.send_header = lambda _name, _value: None
    handler.end_headers = lambda: None

    handler.do_POST()

    body = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert response["code"] == 200
    assert body["accepted"] is True
    assert body["outcome"] == "drift"
    assert runner.control_history[-1].command == "flush"
    assert sink.alerts[-1].title == "execution-reconcile-drift"


def test_startup_mismatch_reduce_only_flush_and_ops_audit_snapshot(tmp_path) -> None:
    sink = _RecordingSink()
    config = LiveRunnerConfig(
        symbols=("BTCUSDT",),
        enable_monitoring=False,
        enable_reconcile=True,
        enable_persistent_stores=False,
        enable_structured_logging=False,
        enable_preflight=False,
        health_port=None,
    )
    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _FakeVenueClient()},
        transport=_FakeTransport(),
        fetch_venue_state=lambda: {
            "positions": {"BTCUSDT": {"qty": "1.0"}},
            "balances": {"USDT": "1000.0"},
        },
        alert_sink=sink,
    )
    runner.reconcile_scheduler.fetch_venue_state = lambda: {
        "positions": {"BTCUSDT":  Decimal("1.0")},
        "balances": {"USDT": Decimal("1000.0")},
    }

    runner.reduce_only(reason="startup_review", source="ops")
    report = runner.flush(reason="startup_flush")
    audit = runner.ops_audit_snapshot()

    assert report is not None
    assert report.ok is False
    assert audit["operator"]["kill_switch"]["mode"] == "reduce_only"
    assert audit["stream_status"] == "ok"
    assert audit["incident_state"] == "critical"
    assert audit["recommended_action"] == "halt"
    assert audit["control_history"][0]["command"] == "flush"
    assert audit["control_history"][1]["command"] == "reduce_only"
    assert any(a["title"] == "execution-reconcile-drift" for a in audit["execution_alerts"])


def test_retryable_reject_then_success_fill_is_visible_in_ops_audit() -> None:
    sink = _RecordingSink()
    runner = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
        alert_manager=AlertManager(sink=sink),
    )
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="recovery:test")
    bridge = LiveExecutionBridge(
        execution_bridge=_SequencedBridge([
            _Ack(ok=False, status="FAILED", error="retryable:TimeoutError:timeout", command_id="cmd-retry"),
            _Ack(ok=True, status="ACCEPTED", command_id="cmd-retry"),
        ]),
        dispatcher_emit=lambda event: router.ingest_canonical_fill(event),
        alert_manager=runner.alert_manager,
    )

    order = SimpleNamespace(
        event_type="order",
        order_id="ord-retry",
        symbol="BTCUSDT",
        side="buy",
        qty=1.0,
        price=40000.0,
        venue="binance",
        command_id="cmd-retry",
    )

    assert list(bridge.send_order(order)) == []
    fills = list(bridge.send_order(order))
    audit = runner.ops_audit_snapshot()

    assert len(fills) == 1
    categories = {row["meta"]["category"] for row in audit["execution_alerts"]}
    assert audit["incident_state"] == "normal"
    assert audit["recommended_action"] == "none"
    assert "execution_rejection" in categories
    assert "execution_fill" in categories


@patch("infra.model_signing.verify_file", return_value=True)
def test_health_ops_audit_endpoint_includes_model_actions(_mock_verify, tmp_path) -> None:
    from research.model_registry.artifact import ArtifactStore
    from research.model_registry.registry import ModelRegistry

    db_path = tmp_path / "models.db"
    artifact_root = tmp_path / "artifacts"
    registry = ModelRegistry(db_path)
    store = ArtifactStore(artifact_root)
    mv = registry.register(
        name="test_alpha",
        params={"n_estimators": 50},
        features=["sma_20", "rsi_14"],
        metrics={"sharpe": 1.2},
        tags=("lgbm",),
    )
    registry.promote(mv.model_id, reason="shadow_win", actor="ops")
    store.save(mv.model_id, "weights", _create_stub_model_weights())

    sink = _RecordingSink()
    config = LiveRunnerConfig(
        symbols=("BTCUSDT",),
        enable_monitoring=False,
        enable_reconcile=False,
        enable_persistent_stores=False,
        enable_structured_logging=False,
        enable_preflight=False,
        health_port=18081,
        model_registry_db=str(db_path),
        artifact_store_root=str(artifact_root),
        model_names=("test_alpha",),
    )
    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _FakeVenueClient()},
        transport=_FakeTransport(),
        alert_sink=sink,
    )
    runner.halt(reason="manual_halt", source="ops")

    handler_cls = type(
        "Handler",
        (_HealthHandler,),
        {
            "status_fn": staticmethod(lambda: {"status": "ok"}),
            "operator_fn": staticmethod(runner.operator_status),
            "control_history_fn": staticmethod(
                lambda: [
                    {
                        "command": rec.command,
                        "reason": rec.reason,
                        "source": rec.source,
                        "result": rec.result,
                        "ts": rec.ts.isoformat(),
                    }
                    for rec in runner.control_history
                ]
            ),
            "control_fn": staticmethod(lambda body: {"accepted": False}),
            "alerts_fn": staticmethod(runner.execution_alert_history),
            "ops_audit_fn": staticmethod(runner.ops_audit_snapshot),
            "auth_token": None,
        },
    )

    handler = handler_cls.__new__(handler_cls)
    handler.path = "/ops-audit"
    handler.headers = {"Content-Length": "2"}
    handler.rfile = io.BytesIO(b"{}")
    handler.wfile = io.BytesIO()
    response: dict[str, object] = {"code": None}
    handler.send_response = lambda code: response.__setitem__("code", code)
    handler.send_header = lambda _name, _value: None
    handler.end_headers = lambda: None

    handler.do_GET()

    body = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert response["code"] == 200
    assert body["operator"]["kill_switch"]["mode"] == "hard_kill"
    assert body["control_history"][0]["command"] == "halt"
    assert body["model_actions"][0]["model"] == "test_alpha"
    assert body["model_actions"][0]["reason"] == "shadow_win"
    assert body["model_status"][0]["name"] == "test_alpha"
    assert body["model_status"][0]["autoload_pending"] is False
    kinds = {row["kind"] for row in body["timeline"]}
    assert "control" in kinds
    assert "model_action" in kinds


@patch("infra.model_signing.verify_file", return_value=True)
def test_model_promote_pending_reload_is_visible_alongside_reduce_only_incident(_mock_verify, tmp_path) -> None:
    from research.model_registry.artifact import ArtifactStore
    from research.model_registry.registry import ModelRegistry

    db_path = tmp_path / "models.db"
    artifact_root = tmp_path / "artifacts"
    registry = ModelRegistry(db_path)
    store = ArtifactStore(artifact_root)

    mv1 = registry.register(
        name="test_alpha",
        params={"n_estimators": 50},
        features=["sma_20", "rsi_14"],
        metrics={"sharpe": 1.2},
        tags=("lgbm",),
    )
    store.save(mv1.model_id, "weights", _create_stub_model_weights())
    registry.promote(mv1.model_id, reason="initial_go", actor="ops")

    config = LiveRunnerConfig(
        symbols=("BTCUSDT",),
        enable_monitoring=False,
        enable_reconcile=False,
        enable_persistent_stores=False,
        enable_structured_logging=False,
        enable_preflight=False,
        model_registry_db=str(db_path),
        artifact_store_root=str(artifact_root),
        model_names=("test_alpha",),
    )
    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _FakeVenueClient()},
        transport=_FakeTransport(),
        alert_sink=_RecordingSink(),
    )
    runner.inference_bridge = SimpleNamespace(update_models=lambda models: None)

    mv2 = registry.register(
        name="test_alpha",
        params={"n_estimators": 80},
        features=["sma_20", "rsi_14"],
        metrics={"sharpe": 1.4},
        tags=("lgbm",),
    )
    store.save(mv2.model_id, "weights", _create_stub_model_weights())
    registry.promote(mv2.model_id, reason="shadow_win_v2", actor="ops")

    runner.reduce_only(reason="model_review", source="ops")
    audit = runner.ops_audit_snapshot()

    assert audit["incident_state"] == "degraded"
    assert audit["recommended_action"] == "reduce_only"
    assert audit["operator"]["kill_switch"]["mode"] == "reduce_only"
    assert audit["model_actions"][0]["to_model_id"] == mv2.model_id
    assert audit["model_actions"][0]["reason"] == "shadow_win_v2"
    assert audit["model_status"][0]["name"] == "test_alpha"
    assert audit["model_status"][0]["model_id"] == mv2.model_id
    assert audit["model_status"][0]["loaded_model_id"] == mv1.model_id
    assert audit["model_status"][0]["autoload_pending"] is True


@patch("infra.model_signing.verify_file", return_value=True)
def test_model_reload_closes_autoload_pending_and_records_reload_status(_mock_verify, tmp_path) -> None:
    from research.model_registry.artifact import ArtifactStore
    from research.model_registry.registry import ModelRegistry

    db_path = tmp_path / "models.db"
    artifact_root = tmp_path / "artifacts"
    registry = ModelRegistry(db_path)
    store = ArtifactStore(artifact_root)

    mv1 = registry.register(
        name="test_alpha",
        params={"n_estimators": 50},
        features=["sma_20", "rsi_14"],
        metrics={"sharpe": 1.2},
        tags=("lgbm",),
    )
    store.save(mv1.model_id, "weights", _create_stub_model_weights())
    registry.promote(mv1.model_id, reason="initial_go", actor="ops")

    config = LiveRunnerConfig(
        symbols=("BTCUSDT",),
        enable_monitoring=False,
        enable_reconcile=False,
        enable_persistent_stores=False,
        enable_structured_logging=False,
        enable_preflight=False,
        model_registry_db=str(db_path),
        artifact_store_root=str(artifact_root),
        model_names=("test_alpha",),
    )
    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _FakeVenueClient()},
        transport=_FakeTransport(),
        alert_sink=_RecordingSink(),
    )
    runner.inference_bridge = SimpleNamespace(update_models=lambda models: None)

    mv2 = registry.register(
        name="test_alpha",
        params={"n_estimators": 80},
        features=["sma_20", "rsi_14"],
        metrics={"sharpe": 1.4},
        tags=("lgbm",),
    )
    store.save(mv2.model_id, "weights", _create_stub_model_weights())
    registry.promote(mv2.model_id, reason="shadow_win_v2", actor="ops")

    names = tuple(config.model_names)
    new_models = runner.model_loader.reload_if_changed(names)
    assert new_models is not None
    runner.inference_bridge.update_models(new_models)
    runner._record_model_reload(
        outcome="reloaded",
        model_names=names,
        detail={"reloaded_count": len(new_models)},
    )

    audit = runner.ops_audit_snapshot()

    assert audit["model_status"][0]["model_id"] == mv2.model_id
    assert audit["model_status"][0]["loaded_model_id"] == mv2.model_id
    assert audit["model_status"][0]["autoload_pending"] is False
    assert audit["model_reload"]["outcome"] == "reloaded"
    assert audit["model_reload"]["model_names"] == ["test_alpha"]
    assert audit["model_reload"]["detail"]["reloaded_count"] == len(new_models)
    assert audit["model_alerts"][0]["title"] == "model_reload_reloaded"


@patch("infra.model_signing.verify_file", return_value=True)
def test_model_reload_failure_keeps_autoload_pending_and_records_failed_status(_mock_verify, tmp_path) -> None:
    from research.model_registry.artifact import ArtifactStore
    from research.model_registry.registry import ModelRegistry

    db_path = tmp_path / "models.db"
    artifact_root = tmp_path / "artifacts"
    registry = ModelRegistry(db_path)
    store = ArtifactStore(artifact_root)

    mv1 = registry.register(
        name="test_alpha",
        params={"n_estimators": 50},
        features=["sma_20", "rsi_14"],
        metrics={"sharpe": 1.2},
        tags=("lgbm",),
    )
    store.save(mv1.model_id, "weights", _create_stub_model_weights())
    registry.promote(mv1.model_id, reason="initial_go", actor="ops")

    config = LiveRunnerConfig(
        symbols=("BTCUSDT",),
        enable_monitoring=False,
        enable_reconcile=False,
        enable_persistent_stores=False,
        enable_structured_logging=False,
        enable_preflight=False,
        model_registry_db=str(db_path),
        artifact_store_root=str(artifact_root),
        model_names=("test_alpha",),
    )
    runner = LiveRunner.build(
        config,
        venue_clients={"binance": _FakeVenueClient()},
        transport=_FakeTransport(),
        alert_sink=_RecordingSink(),
    )

    mv2 = registry.register(
        name="test_alpha",
        params={"n_estimators": 80},
        features=["sma_20", "rsi_14"],
        metrics={"sharpe": 1.4},
        tags=("lgbm",),
    )
    store.save(mv2.model_id, "weights", _create_stub_model_weights())
    registry.promote(mv2.model_id, reason="shadow_win_v2", actor="ops")

    runner.reduce_only(reason="reload_failure_review", source="ops")
    names = tuple(config.model_names)
    runner._record_model_reload(
        outcome="failed",
        model_names=names,
        detail=None,
        error="model_hot_reload_failed",
    )

    audit = runner.ops_audit_snapshot()

    assert audit["incident_state"] == "degraded"
    assert audit["recommended_action"] == "reduce_only"
    assert audit["operator"]["kill_switch"]["mode"] == "reduce_only"
    assert audit["model_status"][0]["model_id"] == mv2.model_id
    assert audit["model_status"][0]["loaded_model_id"] == mv1.model_id
    assert audit["model_status"][0]["autoload_pending"] is True
    assert audit["model_reload"]["outcome"] == "failed"
    assert audit["model_reload"]["model_names"] == ["test_alpha"]
    assert audit["model_reload"]["detail"] is None
    assert audit["model_reload"]["error"] == "model_hot_reload_failed"
    assert audit["model_alerts"][0]["title"] == "model_reload_failed"
    kinds = {row["kind"] for row in audit["timeline"]}
    assert "control" in kinds
    assert "model_alert" in kinds
    assert "model_action" in kinds


@patch("infra.model_signing.verify_file", return_value=True)
def test_restart_rebuilds_ops_timeline_from_persistent_event_log_and_registry(_mock_verify, tmp_path) -> None:
    from research.model_registry.artifact import ArtifactStore
    from research.model_registry.registry import ModelRegistry

    db_path = tmp_path / "models.db"
    artifact_root = tmp_path / "artifacts"
    registry = ModelRegistry(db_path)
    store = ArtifactStore(artifact_root)

    mv1 = registry.register(
        name="test_alpha",
        params={"n_estimators": 50},
        features=["sma_20", "rsi_14"],
        metrics={"sharpe": 1.2},
        tags=("lgbm",),
    )
    store.save(mv1.model_id, "weights", _create_stub_model_weights())
    registry.promote(mv1.model_id, reason="shadow_win_v1", actor="ops")

    event_log = InMemoryEventLog()
    config = LiveRunnerConfig(
        symbols=("BTCUSDT",),
        enable_monitoring=False,
        enable_reconcile=False,
        enable_persistent_stores=False,
        enable_structured_logging=False,
        enable_preflight=False,
        model_registry_db=str(db_path),
        artifact_store_root=str(artifact_root),
        model_names=("test_alpha",),
    )

    first = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
        alert_manager=AlertManager(sink=_RecordingSink()),
        event_log=event_log,
    )
    first._config = config
    first.halt(reason="manual_halt", source="ops")
    first._emit_execution_incident(
        Alert(
            title="execution-timeout",
            message="timeout",
            severity=Severity.WARNING,
            source="execution:test",
            meta={"category": "execution_timeout", "routing_key": "binance:BTCUSDT:timeout"},
        )
    )
    first._record_model_reload(
        outcome="failed",
        model_names=("test_alpha",),
        detail=None,
        error="model_hot_reload_failed",
    )

    second = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
        event_log=event_log,
    )
    second._config = config

    audit = second.ops_audit_snapshot()
    timeline = audit["timeline"]
    kinds = {row["kind"] for row in timeline}

    assert "control" in kinds
    assert "execution_incident" in kinds
    assert "model_reload" in kinds
    assert "model_action" in kinds
    assert any(row["title"] == "operator_halt" for row in timeline)
    assert any(row["title"] == "execution-timeout" for row in timeline)
    assert any(row["title"] == "model_reload_failed" for row in timeline)
    assert any(
        row["kind"] == "model_action" and row.get("detail", {}).get("reason") == "shadow_win_v1"
        for row in timeline
    )

    handler_cls = type(
        "Handler",
        (_HealthHandler,),
        {
            "status_fn": staticmethod(lambda: {"status": "ok"}),
            "operator_fn": staticmethod(second.operator_status),
            "control_history_fn": staticmethod(
                lambda: [
                    {
                        "command": rec.command,
                        "reason": rec.reason,
                        "source": rec.source,
                        "result": rec.result,
                        "ts": rec.ts.isoformat(),
                    }
                    for rec in second.control_history
                ]
            ),
            "control_fn": staticmethod(lambda body: {"accepted": False}),
            "alerts_fn": staticmethod(second.execution_alert_history),
            "ops_audit_fn": staticmethod(second.ops_audit_snapshot),
            "auth_token": None,
        },
    )

    handler = handler_cls.__new__(handler_cls)
    handler.path = "/ops-audit"
    handler.headers = {"Content-Length": "2"}
    handler.rfile = io.BytesIO(b"{}")
    handler.wfile = io.BytesIO()
    response: dict[str, object] = {"code": None}
    handler.send_response = lambda code: response.__setitem__("code", code)
    handler.send_header = lambda _name, _value: None
    handler.end_headers = lambda: None

    handler.do_GET()

    body = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert response["code"] == 200
    endpoint_kinds = {row["kind"] for row in body["timeline"]}
    assert kinds == endpoint_kinds
    assert any(row["title"] == "operator_halt" for row in body["timeline"])
    assert any(row["title"] == "execution-timeout" for row in body["timeline"])
    assert any(row["title"] == "model_reload_failed" for row in body["timeline"])


def test_restart_reconnect_late_fill_reconcile_is_visible_via_ops_audit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(LiveRunner, "_apply_perf_tuning", staticmethod(lambda: None))

    class _FakeCoordinator:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class _FakeLoop:
        def start_background(self) -> None:
            pass

        def stop_background(self) -> None:
            pass

    class _FakeRuntime:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class _FakeUserStream:
        def __init__(self, owner: LiveRunner) -> None:
            self.owner = owner
            self.connect_calls = 0
            self.step_calls = 0
            self.close_calls = 0

        def connect(self) -> None:
            self.connect_calls += 1

        def step(self) -> None:
            self.step_calls += 1
            if self.step_calls == 1:
                raise RuntimeError("user stream dropped")
            self.owner._running = False

        def close(self) -> None:
            self.close_calls += 1

    sink = _RecordingSink()
    runner = LiveRunner(
        loop=_FakeLoop(),
        coordinator=_FakeCoordinator(),
        runtime=_FakeRuntime(),
        kill_switch=KillSwitch(),
        alert_manager=AlertManager(sink=sink),
    )
    runner._config = LiveRunnerConfig(venue="binance")
    runner.user_stream = _FakeUserStream(runner)

    now = [100.0]
    original = OrderStateMachine()

    def _cancel(cmd: SimpleNamespace) -> None:
        original.transition(
            order_id=cmd.order_id,
            new_status=OrderStatus.PENDING_CANCEL,
            ts_ms=2_000,
            reason="timeout_cancel",
        )

    tracker = OrderTimeoutTracker(
        timeout_sec=5.0,
        cancel_fn=_cancel,
        clock_fn=lambda: now[0],
    )
    original.register(
        order_id="o-restart-1",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.0"),
        price=Decimal("50000"),
    )
    original.transition(order_id="o-restart-1", new_status=OrderStatus.NEW, ts_ms=1_000)
    tracker.on_submit("o-restart-1", SimpleNamespace(order_id="o-restart-1"))
    runner.timeout_tracker = tracker

    real_sleep = time.sleep

    def _fast_sleep(seconds: float) -> None:
        if seconds >= 1.0:
            now[0] = 106.0
        real_sleep(0.01)

    monkeypatch.setattr("runner.live_runner.time.sleep", _fast_sleep)

    runner.start()

    checkpoint_path = tmp_path / "restart_late_fill.json"
    checkpoint_path.write_text(json.dumps(_snapshot_state(original, "o-restart-1")))
    restored = _restore_state(json.loads(checkpoint_path.read_text()))
    restored.transition(
        order_id="o-restart-1",
        new_status=OrderStatus.FILLED,
        filled_qty=Decimal("1.0"),
        avg_price=Decimal("50010"),
        ts_ms=3_000,
        reason="late_fill_after_restart",
    )
    report = ReconcileController().reconcile(
        venue="binance",
        local_positions={"BTCUSDT": Decimal("1.0")},
        venue_positions={"BTCUSDT": Decimal("1.0")},
        local_fill_ids={"fill-restart-1"},
        venue_fill_ids={"fill-restart-1"},
        fill_symbol="BTCUSDT",
    )
    audit = runner.ops_audit_snapshot()

    assert runner.user_stream.connect_calls >= 2
    assert runner.user_stream.step_calls >= 2
    assert audit["stream_status"] == "degraded"
    assert audit["incident_state"] == "degraded"
    assert audit["recommended_action"] == "review"
    assert any(a["title"] == "execution-timeout" for a in audit["execution_alerts"])
    assert report.ok is True
    assert restored.get("o-restart-1").status == OrderStatus.FILLED


def test_persistent_user_stream_failure_requires_manual_reduce_only_then_halt(monkeypatch) -> None:
    monkeypatch.setattr(LiveRunner, "_apply_perf_tuning", staticmethod(lambda: None))

    class _FakeCoordinator:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class _FakeLoop:
        def start_background(self) -> None:
            pass

        def stop_background(self) -> None:
            pass

    class _FakeRuntime:
        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    class _FailingUserStream:
        def __init__(self, owner: LiveRunner) -> None:
            self.owner = owner
            self.connect_calls = 0
            self.step_calls = 0
            self.close_calls = 0

        def connect(self) -> None:
            self.connect_calls += 1

        def step(self) -> None:
            self.step_calls += 1
            if self.step_calls >= 2:
                self.owner._running = False
            raise RuntimeError("persistent user stream failure")

        def close(self) -> None:
            self.close_calls += 1

    class _Scheduler:
        def __init__(self, report: object) -> None:
            self.last_report = report

        def run_once(self):
            return self.last_report

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    sink = _RecordingSink()
    report = SimpleNamespace(ok=False, should_halt=False, all_drifts=(SimpleNamespace(symbol="BTCUSDT"),), venue="binance")
    runner = LiveRunner(
        loop=_FakeLoop(),
        coordinator=_FakeCoordinator(),
        runtime=_FakeRuntime(),
        kill_switch=KillSwitch(),
        reconcile_scheduler=_Scheduler(report),
        alert_manager=AlertManager(sink=sink),
    )
    runner._config = LiveRunnerConfig(venue="binance")
    runner.user_stream = _FailingUserStream(runner)

    real_sleep = time.sleep

    def _fast_sleep(seconds: float) -> None:
        real_sleep(0.01)

    monkeypatch.setattr("runner.live_runner.time.sleep", _fast_sleep)

    runner.start()

    pre_control = runner.ops_audit_snapshot()
    assert pre_control["operator"]["kill_switch"] is None
    assert pre_control["stream_status"] == "down"
    assert pre_control["incident_state"] == "degraded"
    assert pre_control["recommended_action"] == "reduce_only"

    runner.reduce_only(reason="stream_degraded", source="ops")
    runner.flush(reason="stream_review")
    runner.halt(reason="stream_unrecoverable", source="ops")

    audit = runner.ops_audit_snapshot()

    assert runner.user_stream.connect_calls >= 2
    assert runner.user_stream.step_calls >= 2
    assert audit["stream_status"] == "down"
    assert audit["incident_state"] == "critical"
    assert audit["recommended_action"] == "halt"
    assert audit["last_incident_category"] in {"operator_control", "execution_reconcile"}
    assert audit["last_incident_ts"] is not None
    assert audit["operator"]["kill_switch"]["mode"] == "hard_kill"
    assert [row["command"] for row in audit["control_history"][:3]] == ["halt", "flush", "reduce_only"]
    assert any(a["title"] == "execution-reconcile-drift" for a in audit["execution_alerts"])
    assert any(a["meta"]["category"] == "execution_stream" for a in audit["execution_alerts"])


def test_checkpoint_restore_reduce_only_and_reconcile_overlap_stays_consistent(tmp_path) -> None:
    class _Scheduler:
        def __init__(self, report: object) -> None:
            self.last_report = report

        def run_once(self):
            return self.last_report

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    original = OrderStateMachine()
    original.register(
        order_id="o-restore-1",
        symbol="BTCUSDT",
        side="buy",
        order_type="LIMIT",
        qty=Decimal("1.0"),
        price=Decimal("50000"),
    )
    original.transition(order_id="o-restore-1", new_status=OrderStatus.NEW, ts_ms=1_000)
    original.transition(
        order_id="o-restore-1",
        new_status=OrderStatus.PENDING_CANCEL,
        ts_ms=2_000,
        reason="timeout_cancel",
    )

    checkpoint_path = tmp_path / "restore_overlap.json"
    checkpoint_path.write_text(json.dumps(_snapshot_state(original, "o-restore-1")))
    restored = _restore_state(json.loads(checkpoint_path.read_text()))

    sink = _RecordingSink()
    drift_report = SimpleNamespace(
        ok=False,
        should_halt=False,
        all_drifts=(SimpleNamespace(symbol="BTCUSDT"),),
        venue="binance",
    )
    clean_report = SimpleNamespace(
        ok=True,
        should_halt=False,
        all_drifts=(),
        venue="binance",
    )
    scheduler = _Scheduler(drift_report)
    runner = LiveRunner(
        loop=MagicMock(),
        coordinator=MagicMock(),
        runtime=MagicMock(),
        kill_switch=KillSwitch(),
        reconcile_scheduler=scheduler,
        alert_manager=AlertManager(sink=sink),
    )
    runner._config = LiveRunnerConfig(venue="binance")

    runner.reduce_only(reason="restore_review", source="ops")
    first_report = runner.flush(reason="post_restore_review")

    restored.transition(
        order_id="o-restore-1",
        new_status=OrderStatus.FILLED,
        filled_qty=Decimal("1.0"),
        avg_price=Decimal("50005"),
        ts_ms=3_000,
        reason="late_fill_after_restore",
    )
    scheduler.last_report = clean_report
    second_report = runner.flush(reason="post_late_fill_reconcile")
    audit = runner.ops_audit_snapshot()

    assert first_report.ok is False
    assert second_report.ok is True
    assert restored.get("o-restore-1").status == OrderStatus.FILLED
    assert audit["stream_status"] == "ok"
    assert audit["incident_state"] == "degraded"
    assert audit["recommended_action"] == "reduce_only"
    assert audit["operator"]["kill_switch"]["mode"] == "reduce_only"
    assert [row["command"] for row in audit["control_history"][:3]] == ["flush", "flush", "reduce_only"]
    assert any(a["meta"]["category"] == "execution_reconcile" for a in audit["execution_alerts"])
