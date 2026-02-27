"""Tests for ReconnectingWsTransport."""
from __future__ import annotations

from typing import List, Optional
from unittest.mock import patch

import pytest

from execution.adapters.binance.reconnecting_ws_transport import ReconnectingWsTransport


class FakeWsTransport:
    """Mock WsTransport for testing reconnection logic."""

    def __init__(self) -> None:
        self.connected = False
        self.connect_count = 0
        self.close_count = 0
        self.url: Optional[str] = None
        self._messages: List[str] = []
        self._fail_recv_count = 0
        self._sent: List[str] = []

    def connect(self, url: str) -> None:
        self.url = url
        self.connected = True
        self.connect_count += 1

    def recv(self, *, timeout_s: Optional[float] = None) -> str:
        if self._fail_recv_count > 0:
            self._fail_recv_count -= 1
            raise ConnectionError("disconnected")
        if self._messages:
            return self._messages.pop(0)
        return ""

    def close(self) -> None:
        self.connected = False
        self.close_count += 1

    def send(self, msg: str) -> None:
        self._sent.append(msg)

    def enqueue(self, *msgs: str) -> None:
        self._messages.extend(msgs)

    def set_fail_recv(self, count: int) -> None:
        self._fail_recv_count = count


class TestReconnectingBasic:
    def test_normal_recv(self) -> None:
        inner = FakeWsTransport()
        inner.enqueue("hello", "world")
        ws = ReconnectingWsTransport(inner=inner)
        ws.connect("wss://test.com")
        assert inner.connected
        assert ws.recv() == "hello"
        assert ws.recv() == "world"

    def test_close(self) -> None:
        inner = FakeWsTransport()
        ws = ReconnectingWsTransport(inner=inner)
        ws.connect("wss://test.com")
        ws.close()
        assert not inner.connected


class TestReconnection:
    @patch("execution.adapters.binance.reconnecting_ws_transport.time.sleep")
    def test_reconnects_on_failure(self, mock_sleep) -> None:
        inner = FakeWsTransport()
        ws = ReconnectingWsTransport(
            inner=inner, max_retries=3,
            base_delay_s=0.1, max_delay_s=1.0,
        )
        ws.connect("wss://test.com")
        assert inner.connect_count == 1

        # First recv fails, triggers reconnect, second recv succeeds
        inner.set_fail_recv(1)
        inner.enqueue("reconnected")
        msg = ws.recv()
        assert msg == "reconnected"
        assert inner.connect_count == 2  # reconnected once

    @patch("execution.adapters.binance.reconnecting_ws_transport.time.sleep")
    def test_exponential_backoff(self, mock_sleep) -> None:
        inner = FakeWsTransport()
        ws = ReconnectingWsTransport(
            inner=inner, max_retries=5,
            base_delay_s=1.0, max_delay_s=10.0,
        )
        ws.connect("wss://test.com")

        # Fail 3 times then succeed
        inner.set_fail_recv(3)
        inner.enqueue("ok")
        ws.recv()

        # Check sleep was called with increasing delays
        calls = [c[0][0] for c in mock_sleep.call_args_list]
        assert len(calls) >= 1
        # First delay should be base_delay_s
        assert calls[0] == 1.0
        # Second delay should be 2x
        if len(calls) >= 2:
            assert calls[1] == 2.0
        # Third should be 4x
        if len(calls) >= 3:
            assert calls[2] == 4.0

    @patch("execution.adapters.binance.reconnecting_ws_transport.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep) -> None:
        inner = FakeWsTransport()

        # Make connect always fail after initial connect
        original_connect = inner.connect

        def fail_connect(url):
            original_connect(url)
            inner._fail_recv_count = 999  # always fail recv

        inner.connect = fail_connect

        ws = ReconnectingWsTransport(
            inner=inner, max_retries=3,
            base_delay_s=0.01, max_delay_s=0.1,
        )
        ws.connect("wss://test.com")
        inner._fail_recv_count = 1

        with pytest.raises(ConnectionError, match="Failed to reconnect"):
            ws.recv()

    @patch("execution.adapters.binance.reconnecting_ws_transport.time.sleep")
    def test_resubscribe_on_reconnect(self, mock_sleep) -> None:
        inner = FakeWsTransport()
        ws = ReconnectingWsTransport(inner=inner, max_retries=3, base_delay_s=0.01)
        ws.connect("wss://test.com")

        # Subscribe to a channel
        ws.send_subscribe('{"method":"SUBSCRIBE","params":["btcusdt@trade"]}')

        # Trigger reconnect
        inner.set_fail_recv(1)
        inner.enqueue("data")
        ws.recv()

        # Check that subscription was re-sent after reconnect
        assert '{"method":"SUBSCRIBE","params":["btcusdt@trade"]}' in inner._sent

    @patch("execution.adapters.binance.reconnecting_ws_transport.time.sleep")
    def test_on_reconnect_callback(self, mock_sleep) -> None:
        inner = FakeWsTransport()
        callback_called = []

        ws = ReconnectingWsTransport(
            inner=inner, max_retries=3,
            base_delay_s=0.01,
            on_reconnect=lambda: callback_called.append(True),
        )
        ws.connect("wss://test.com")

        inner.set_fail_recv(1)
        inner.enqueue("data")
        ws.recv()

        assert len(callback_called) == 1

    @patch("execution.adapters.binance.reconnecting_ws_transport.time.sleep")
    def test_max_delay_cap(self, mock_sleep) -> None:
        inner = FakeWsTransport()
        ws = ReconnectingWsTransport(
            inner=inner, max_retries=10,
            base_delay_s=1.0, max_delay_s=5.0,
        )
        ws.connect("wss://test.com")

        inner.set_fail_recv(5)
        inner.enqueue("ok")
        ws.recv()

        calls = [c[0][0] for c in mock_sleep.call_args_list]
        # All delays should be <= max_delay
        for delay in calls:
            assert delay <= 5.0
