"""Root conftest: the project root on sys.path, the network guard and the order-transition guard every test
runs under.

(The event, engine and snapshot factories went with those packages, 2026-11-15.)
"""
import ipaddress
import socket
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class NetworkBlocked(RuntimeError):
    """Tests never contact a real exchange, Telegram or any other host (design 2026-11 M5, invariant T1)."""


_REAL_CONNECT, _REAL_CONNECT_EX = socket.socket.connect, socket.socket.connect_ex
_REAL_GETADDRINFO = socket.getaddrinfo


def _local(host) -> bool:
    if host in (None, "", "localhost"):
        return True
    if isinstance(host, bytes):
        host = host.decode()
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def _check(sock, address) -> None:
    if getattr(socket, "AF_UNIX", None) is not None and sock.family == socket.AF_UNIX:
        return
    host = address[0] if isinstance(address, tuple) else address
    if not _local(host):
        raise NetworkBlocked(f"a test tried to connect to {host!r}; tests never leave this host")


def _connect(self, address):
    _check(self, address)
    return _REAL_CONNECT(self, address)


def _connect_ex(self, address):
    _check(self, address)
    return _REAL_CONNECT_EX(self, address)


def _getaddrinfo(host, *args, **kwargs):
    if not _local(host):
        raise NetworkBlocked(f"a test tried to resolve {host!r}; tests never leave this host")
    return _REAL_GETADDRINFO(host, *args, **kwargs)


def pytest_configure(config):
    socket.socket.connect, socket.socket.connect_ex = _connect, _connect_ex
    socket.getaddrinfo = _getaddrinfo


@pytest.fixture(autouse=True)
def _order_transitions_stay_in_the_table():
    """Migration step 5: an order status change outside trading.domain.orders.TRANSITIONS fails the test.

    The executor only logs such a change in production (observation mode); here it must never happen.
    """
    from trading.application.execution import lifecycle
    lifecycle.TRANSITION_VIOLATIONS.clear()
    yield
    violations = list(lifecycle.TRANSITION_VIOLATIONS)
    lifecycle.TRANSITION_VIOLATIONS.clear()
    assert violations == [], f"order status transitions outside the table: {violations}"


@pytest.fixture(autouse=True)
def _host_lock_directory(tmp_path_factory, monkeypatch):
    """S6 (ADR-0016 point 5): every test gets its own existing host lock directory instead of the host's fixed
    /var/lib/quant/locks; a test about a missing or unusable directory sets its own."""
    try:
        from trading.adapters.files import host_lock
    except ImportError:                     # tests-first commit only: the module comes with the implementation
        return
    locks = tmp_path_factory.mktemp("locks")
    monkeypatch.setattr(host_lock, "LOCK_DIR", locks)
