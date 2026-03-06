"""Tests for the minimal HTTP health endpoint."""
from __future__ import annotations

import io
import json

from monitoring.health_server import HealthServer, _HealthHandler


def _dispatch(
    path: str,
    *,
    status_fn,
    auth_token: str | None = None,
    request_token: str | None = None,
) -> tuple[int, dict]:
    handler_cls = type(
        "Handler",
        (_HealthHandler,),
        {
            "status_fn": staticmethod(status_fn),
            "auth_token": auth_token,
        },
    )
    handler = handler_cls.__new__(handler_cls)
    handler.path = path
    handler.headers = {}
    if request_token is not None:
        handler.headers["Authorization"] = f"Bearer {request_token}"
    handler.wfile = io.BytesIO()

    response: dict[str, object] = {"code": None}
    handler.send_response = lambda code: response.__setitem__("code", code)
    handler.send_header = lambda _name, _value: None
    handler.end_headers = lambda: None

    handler.do_GET()
    payload = json.loads(handler.wfile.getvalue().decode("utf-8"))
    return int(response["code"]), payload


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

    server = HealthServer(port=19876, status_fn=lambda: {"status": "ok"}, auth_token="secret")
    server.start()
    server.stop()

    assert captured["address"] == ("127.0.0.1", 19876)
    assert captured["thread_daemon"] is True
    assert captured["served"] is True
    assert captured["shutdown"] is True
