# tests/unit/test_health_server.py
"""Tests for the minimal HTTP health endpoint."""
from __future__ import annotations

import json
import urllib.request

from monitoring.health_server import HealthServer


def _get(port: int, path: str) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{port}{path}"
    try:
        resp = urllib.request.urlopen(url, timeout=2)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_ok(tmp_path):
    server = HealthServer(port=19876, status_fn=lambda: {"status": "ok"})
    server.start()
    try:
        code, body = _get(19876, "/health")
        assert code == 200
        assert body["status"] == "ok"
    finally:
        server.stop()


def test_health_critical(tmp_path):
    server = HealthServer(port=19877, status_fn=lambda: {"status": "critical"})
    server.start()
    try:
        code, body = _get(19877, "/health")
        assert code == 503
        assert body["status"] == "critical"
    finally:
        server.stop()


def test_status_endpoint(tmp_path):
    server = HealthServer(port=19878, status_fn=lambda: {"uptime": 42})
    server.start()
    try:
        code, body = _get(19878, "/status")
        assert code == 200
        assert body["uptime"] == 42
    finally:
        server.stop()


def test_404_endpoint(tmp_path):
    server = HealthServer(port=19879, status_fn=lambda: {})
    server.start()
    try:
        code, body = _get(19879, "/nonexistent")
        assert code == 404
    finally:
        server.stop()
