"""Tests for monitoring.health_server — lifecycle, /health, /status."""
import json
import urllib.request

import pytest

from monitoring.health_server import HealthServer

_PORT = 18932


def _fetch(path: str) -> tuple[int, dict]:
    url = f"http://127.0.0.1:{_PORT}{path}"
    try:
        resp = urllib.request.urlopen(url, timeout=2)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture()
def server():
    status = {"ok": True, "uptime": 42}
    srv = HealthServer(port=_PORT, status_fn=lambda: dict(status))
    srv.start()
    yield srv, status
    srv.stop()


def test_health_200(server):
    srv, _ = server
    code, body = _fetch("/health")
    assert code == 200
    assert body["ok"] is True


def test_status_full_dump(server):
    srv, status = server
    code, body = _fetch("/status")
    assert code == 200
    assert body == status


def test_health_503_critical_flag(server):
    srv, status = server
    status["critical"] = True
    code, body = _fetch("/health")
    assert code == 503


def test_health_503_status_critical(server):
    srv, status = server
    status.pop("critical", None)
    status["status"] = "critical"
    code, body = _fetch("/health")
    assert code == 503


def test_start_stop_lifecycle():
    srv = HealthServer(port=_PORT, status_fn=lambda: {"up": True})
    srv.start()
    code, _ = _fetch("/health")
    assert code == 200
    srv.stop()
    with pytest.raises(Exception):
        _fetch("/health")
