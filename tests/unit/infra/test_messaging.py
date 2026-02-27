"""Tests for the in-process message bus."""
from __future__ import annotations

import threading
from typing import List

import pytest

from infra.messaging.bus import InProcessMessageBus


class TestInProcessMessageBus:
    def test_publish_to_subscriber(self):
        bus = InProcessMessageBus()
        received: List[bytes] = []
        bus.subscribe("orders", received.append)
        bus.publish("orders", b"buy BTCUSDT")

        assert len(received) == 1
        assert received[0] == b"buy BTCUSDT"

    def test_multiple_subscribers(self):
        bus = InProcessMessageBus()
        r1: List[bytes] = []
        r2: List[bytes] = []
        bus.subscribe("fills", r1.append)
        bus.subscribe("fills", r2.append)
        bus.publish("fills", b"fill-1")

        assert len(r1) == 1
        assert len(r2) == 1

    def test_topic_isolation(self):
        bus = InProcessMessageBus()
        orders: List[bytes] = []
        fills: List[bytes] = []
        bus.subscribe("orders", orders.append)
        bus.subscribe("fills", fills.append)

        bus.publish("orders", b"msg1")
        bus.publish("fills", b"msg2")

        assert len(orders) == 1
        assert orders[0] == b"msg1"
        assert len(fills) == 1
        assert fills[0] == b"msg2"

    def test_no_subscribers_is_noop(self):
        bus = InProcessMessageBus()
        bus.publish("empty-topic", b"nobody listening")

    def test_unsubscribe_removes_all_handlers(self):
        bus = InProcessMessageBus()
        received: List[bytes] = []
        bus.subscribe("orders", received.append)
        bus.unsubscribe("orders")
        bus.publish("orders", b"after unsubscribe")

        assert len(received) == 0

    def test_unsubscribe_single_handler(self):
        bus = InProcessMessageBus()
        r1: List[bytes] = []
        r2: List[bytes] = []
        bus.subscribe("t", r1.append)
        bus.subscribe("t", r2.append)
        bus.unsubscribe_handler("t", r1.append)

        bus.publish("t", b"msg")
        assert len(r1) == 0
        assert len(r2) == 1


class TestMessageBusErrorHandling:
    def test_handler_error_does_not_crash_bus(self):
        bus = InProcessMessageBus()
        received: List[bytes] = []

        def bad_handler(msg: bytes) -> None:
            raise RuntimeError("handler broke")

        bus.subscribe("t", bad_handler)
        bus.subscribe("t", received.append)

        bus.publish("t", b"data")
        assert len(received) == 1

    def test_handler_error_does_not_affect_other_topics(self):
        bus = InProcessMessageBus()
        results: List[bytes] = []

        def bad(msg: bytes) -> None:
            raise ValueError("fail")

        bus.subscribe("bad-topic", bad)
        bus.subscribe("good-topic", results.append)

        bus.publish("bad-topic", b"x")
        bus.publish("good-topic", b"y")

        assert len(results) == 1
        assert results[0] == b"y"


class TestMessageBusThreadSafety:
    def test_concurrent_publish_subscribe(self):
        bus = InProcessMessageBus()
        received: List[bytes] = []
        lock = threading.Lock()

        def safe_append(msg: bytes) -> None:
            with lock:
                received.append(msg)

        bus.subscribe("concurrent", safe_append)

        threads = []
        for i in range(10):
            t = threading.Thread(
                target=bus.publish,
                args=("concurrent", f"msg-{i}".encode()),
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(received) == 10


class TestMessageBusLifecycle:
    def test_start_stop(self):
        bus = InProcessMessageBus()
        bus.start()
        bus.stop()

    def test_topics_property(self):
        bus = InProcessMessageBus()
        bus.subscribe("a", lambda m: None)
        bus.subscribe("b", lambda m: None)
        assert set(bus.topics) == {"a", "b"}
