"""Tests for the minimal HTTP health endpoint."""
from __future__ import annotations

import io
import json

from monitoring.health_server import HealthServer, _HealthHandler


def _dispatch(
    path: str,
    *,
    status_fn,
    operator_fn=None,
    control_history_fn=None,
    control_fn=None,
    alerts_fn=None,
    ops_audit_fn=None,
    auth_token: str | None = None,
    request_token: str | None = None,
    method: str = "GET",
    body: object = None,
) -> tuple[int, dict]:
    handler_cls = type(
        "Handler",
        (_HealthHandler,),
        {
            "status_fn": staticmethod(status_fn),
            "operator_fn": staticmethod(operator_fn) if operator_fn is not None else None,
            "control_history_fn": staticmethod(control_history_fn) if control_history_fn is not None else None,
            "control_fn": staticmethod(control_fn) if control_fn is not None else None,
            "alerts_fn": staticmethod(alerts_fn) if alerts_fn is not None else None,
            "ops_audit_fn": staticmethod(ops_audit_fn) if ops_audit_fn is not None else None,
            "auth_token": auth_token,
        },
    )
    handler = handler_cls.__new__(handler_cls)
    handler.path = path
    handler.headers = {}
    if request_token is not None:
        handler.headers["Authorization"] = f"Bearer {request_token}"
    payload = json.dumps(body if body is not None else {}).encode("utf-8")
    handler.headers["Content-Length"] = str(len(payload))
    handler.rfile = io.BytesIO(payload)
    handler.wfile = io.BytesIO()

    response: dict[str, object] = {"code": None}
    handler.send_response = lambda code: response.__setitem__("code", code)
    handler.send_header = lambda _name, _value: None
    handler.end_headers = lambda: None

    if method == "POST":
        handler.do_POST()
    else:
        handler.do_GET()
    decoded = json.loads(handler.wfile.getvalue().decode("utf-8"))
    return int(response["code"]), decoded


def test_health_ok():
    code, body = _dispatch("/health", status_fn=lambda: {"status": "ok"})
    assert code == 200
    assert body["status"] == "ok"


def test_health_critical():
    code, body = _dispatch("/health", status_fn=lambda: {"status": "critical"})
    assert code == 503
    assert body["status"] == "critical"


def test_status_endpoint():
    code, body = _dispatch("/status", status_fn=lambda: {"uptime": 42})
    assert code == 200
    assert body["uptime"] == 42


def test_operator_endpoint():
    code, body = _dispatch(
        "/operator",
        status_fn=lambda: {"status": "ok"},
        operator_fn=lambda: {"running": True, "stopped": False},
    )
    assert code == 200
    assert body["running"] is True


def test_control_history_endpoint():
    code, body = _dispatch(
        "/control-history",
        status_fn=lambda: {"status": "ok"},
        operator_fn=lambda: {"running": True},
        control_fn=lambda payload: {"accepted": True},
    )
    assert code == 404

    code, body = _dispatch(
        "/control-history",
        status_fn=lambda: {"status": "ok"},
        operator_fn=lambda: {"running": True},
        control_fn=lambda payload: {"accepted": True},
        control_history_fn=lambda: [{"command": "halt", "result": "hard_kill"}],
    )
    assert code == 200
    assert body["history"][0]["command"] == "halt"


def test_control_endpoint():
    code, body = _dispatch(
        "/control",
        status_fn=lambda: {"status": "ok"},
        control_fn=lambda payload: {"accepted": True, "command": payload["command"], "outcome": "hard_kill"},
        method="POST",
        body={"command": "halt", "reason": "manual_halt"},
    )
    assert code == 200
    assert body["accepted"] is True
    assert body["command"] == "halt"


def test_control_endpoint_invalid_json_object():
    code, body = _dispatch(
        "/control",
        status_fn=lambda: {"status": "ok"},
        control_fn=lambda payload: {"accepted": True},
        method="POST",
        body=["invalid"],
    )
    assert code == 400
    assert "invalid json" in body["error"]


def test_execution_alerts_endpoint():
    code, body = _dispatch(
        "/execution-alerts",
        status_fn=lambda: {"status": "ok"},
        alerts_fn=lambda: [{"title": "execution-timeout", "meta": {"category": "execution_timeout"}}],
    )
    assert code == 200
    assert body["alerts"][0]["title"] == "execution-timeout"


def test_ops_audit_endpoint():
    code, body = _dispatch(
        "/ops-audit",
        status_fn=lambda: {"status": "ok"},
        ops_audit_fn=lambda: {
            "operator": {"running": True},
            "control_history": [{"command": "halt"}],
            "execution_alerts": [{"title": "execution-timeout"}],
            "model_actions": [{"action": "promote"}],
        },
    )
    assert code == 200
    assert body["operator"]["running"] is True
    assert body["control_history"][0]["command"] == "halt"
    assert body["execution_alerts"][0]["title"] == "execution-timeout"
    assert body["model_actions"][0]["action"] == "promote"


def test_404_endpoint():
    code, body = _dispatch("/nonexistent", status_fn=lambda: {})
    assert code == 404
    assert body["error"] == "not found"


def test_health_requires_auth_token():
    code, body = _dispatch(
        "/health",
        status_fn=lambda: {"status": "ok"},
        auth_token="secret",
    )
    assert code == 401
    assert body["error"] == "unauthorized"

    ok_code, ok_body = _dispatch(
        "/health",
        status_fn=lambda: {"status": "ok"},
        auth_token="secret",
        request_token="secret",
    )
    assert ok_code == 200
    assert ok_body["status"] == "ok"


def test_server_start_stop_without_socket(monkeypatch):
    captured: dict[str, object] = {}

    class FakeHTTPServer:
        def __init__(self, address, handler):
            captured["address"] = address
            captured["handler"] = handler

        def serve_forever(self):
            captured["served"] = True

        def shutdown(self):
            captured["shutdown"] = True

    class FakeThread:
        def __init__(self, *, target, daemon):
            self._target = target
            self._daemon = daemon
            self._alive = False

        def start(self):
            captured["thread_target"] = self._target
            captured["thread_daemon"] = self._daemon
            self._alive = True
            self._target()
            self._alive = False

        def is_alive(self):
            return self._alive

        def join(self, timeout=None):
            captured["join_timeout"] = timeout

    monkeypatch.setattr("monitoring.health_server.HTTPServer", FakeHTTPServer)
    monkeypatch.setattr("monitoring.health_server.threading.Thread", FakeThread)

    server = HealthServer(
        port=19876,
        status_fn=lambda: {"status": "ok"},
        operator_fn=lambda: {"running": True},
        control_fn=lambda payload: {"accepted": True, "command": payload.get("command", "")},
        auth_token="secret",
    )
    server.start()
    server.stop()

    assert captured["address"] == ("127.0.0.1", 19876)
    assert captured["thread_daemon"] is True
    assert captured["served"] is True
    assert captured["shutdown"] is True
