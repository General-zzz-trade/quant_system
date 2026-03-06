"""Minimal HTTP health endpoint using stdlib only."""
from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Callable, Dict


class _HealthHandler(BaseHTTPRequestHandler):
    status_fn: Callable[[], dict]
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
        else:
            self._json_response(404, {"error": "not found"})

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
        return auth == f"Bearer {token}"


class HealthServer:
    def __init__(
        self,
        port: int,
        status_fn: Callable[[], dict],
        *,
        host: str = "127.0.0.1",
        auth_token: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._status_fn = status_fn
        self._auth_token = auth_token
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        handler = type(
            "Handler",
            (_HealthHandler,),
            {
                "status_fn": staticmethod(self._status_fn),
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
