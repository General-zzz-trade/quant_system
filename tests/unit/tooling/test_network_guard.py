"""The root conftest keeps every test on this host: no connection and no name lookup leaves it (invariant T1)."""
from __future__ import annotations

import socket
import urllib.request

import pytest

from tests.conftest import NetworkBlocked


@pytest.mark.parametrize("address", [("203.0.113.7", 443), ("2001:db8::1", 443, 0, 0)])
def test_a_connection_to_another_host_is_refused_before_it_is_made(address):
    family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        with pytest.raises(NetworkBlocked):
            sock.connect(address)
        with pytest.raises(NetworkBlocked):
            sock.connect_ex(address)


def test_a_name_lookup_and_an_http_request_are_refused():
    with pytest.raises(NetworkBlocked):
        socket.getaddrinfo("fapi.binance.com", 443)
    with pytest.raises(NetworkBlocked):
        urllib.request.urlopen("https://api.telegram.org/", timeout=1)


def test_loopback_still_works():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass
    assert socket.getaddrinfo("localhost", 80)
