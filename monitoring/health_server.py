"""Minimal HTTP health endpoint using stdlib only."""
from __future__ import annotations

import hmac
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Callable, Optional


class _HealthHandler(BaseHTTPRequestHandler):
    status_fn: Callable[[], dict]
    operator_fn: Optional[Callable[[], dict]] = None
    control_history_fn: Optional[Callable[[], list[dict[str, Any]]]] = None
    control_fn: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None
    alerts_fn: Optional[Callable[[], list[dict[str, Any]]]] = None
    ops_audit_fn: Optional[Callable[[], dict[str, Any]]] = None
    attribution_fn: Optional[Callable[[], dict[str, Any]]] = None
    auth_token: str | None = None

    def do_GET(self) -> None:
        if not self._is_authorized():
            self._json_response(401, {"error": "unauthorized"})
            return
        if self.path == "/health":
            data = self.status_fn()
            is_critical = data.get("critical") is True or data.get("status") == "critical"
            code = 503 if is_critical else 200
            self._json_response(code, data)
        elif self.path == "/status":
            self._json_response(200, self.status_fn())
        elif self.path == "/operator":
            if self.operator_fn is None:
                self._json_response(404, {"error": "operator unavailable"})
            else:
                self._json_response(200, self.operator_fn())
        elif self.path == "/control-history":
            if self.control_history_fn is None:
                self._json_response(404, {"error": "control history unavailable"})
            else:
                self._json_response(200, {"history": self.control_history_fn()})
        elif self.path == "/execution-alerts":
            if self.alerts_fn is None:
                self._json_response(404, {"error": "alerts unavailable"})
            else:
                self._json_response(200, {"alerts": self.alerts_fn()})
        elif self.path == "/ops-audit":
            if self.ops_audit_fn is None:
                self._json_response(404, {"error": "ops audit unavailable"})
            else:
                self._json_response(200, self.ops_audit_fn())
        elif self.path == "/attribution":
            if self.attribution_fn is None:
                self._json_response(404, {"error": "attribution unavailable"})
            else:
                self._json_response(200, self.attribution_fn())
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._is_authorized():
            self._json_response(401, {"error": "unauthorized"})
            return
        if self.path != "/control":
            self._json_response(404, {"error": "not found"})
            return
        if self.control_fn is None:
            self._json_response(404, {"error": "control unavailable"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json_response(400, {"error": "invalid content length"})
            return

        try:
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("control body must be object")
        except Exception as exc:
            self._json_response(400, {"error": f"invalid json: {exc}"})
            return

        result = self.control_fn(body)
        self._json_response(200 if bool(result.get("accepted")) else 400, result)

    def _json_response(self, code: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass  # suppress request logs

    def _is_authorized(self) -> bool:
        token = self.auth_token
        if not token:
            return True
        auth = self.headers.get("Authorization", "")
        return hmac.compare_digest(auth, f"Bearer {token}")


class HealthServer:
    def __init__(
        self,
        port: int,
        status_fn: Callable[[], dict],
        *,
        operator_fn: Optional[Callable[[], dict]] = None,
        control_history_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
        control_fn: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        alerts_fn: Optional[Callable[[], list[dict[str, Any]]]] = None,
        ops_audit_fn: Optional[Callable[[], dict[str, Any]]] = None,
        attribution_fn: Optional[Callable[[], dict[str, Any]]] = None,
        host: str = "127.0.0.1",
        auth_token: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._status_fn = status_fn
        self._operator_fn = operator_fn
        self._control_history_fn = control_history_fn
        self._control_fn = control_fn
        self._alerts_fn = alerts_fn
        self._ops_audit_fn = ops_audit_fn
        self._attribution_fn = attribution_fn
        self._auth_token = auth_token
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = type(
            "Handler",
            (_HealthHandler,),
            {
                "status_fn": staticmethod(self._status_fn),
                "operator_fn": staticmethod(self._operator_fn) if self._operator_fn is not None else None,
                "control_history_fn": staticmethod(self._control_history_fn) if self._control_history_fn is not None else None,
                "control_fn": staticmethod(self._control_fn) if self._control_fn is not None else None,
                "alerts_fn": staticmethod(self._alerts_fn) if self._alerts_fn is not None else None,
                "ops_audit_fn": staticmethod(self._ops_audit_fn) if self._ops_audit_fn is not None else None,
                "attribution_fn": staticmethod(self._attribution_fn) if self._attribution_fn is not None else None,
                "auth_token": self._auth_token,
            },
        )
        self._server = HTTPServer((self._host, self._port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            from infra.threading_utils import safe_join_thread

            safe_join_thread(self._thread, timeout=5.0)
            self._thread = None
