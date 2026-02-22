# execution/tests/bridge/test_execution_bridge_idempotent_retry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, List, Dict

import pytest

from execution.bridge.execution_bridge import (
    ExecutionBridge,
    RetryPolicy,
    RateLimitConfig,
    CircuitBreakerConfig,
    RetryableVenueError,
    NonRetryableVenueError,
)
from execution.bridge.request_ids import RequestIdFactory
from execution.models.commands import make_submit_order_command, make_cancel_order_command


@dataclass
class FakeClock:
    t: float = 0.0
    def now(self) -> float:
        return self.t
    def advance(self, dt: float) -> None:
        self.t += dt


@dataclass
class FakeSleeper:
    clock: FakeClock
    sleeps: List[float]
    def sleep(self, sec: float) -> None:
        self.sleeps.append(sec)
        self.clock.advance(sec)


class FakeVenueClient:
    def __init__(self) -> None:
        self.submit_calls = 0
        self.cancel_calls = 0

        self._submit_fail_first: Optional[BaseException] = None
        self._submit_always_fail: Optional[BaseException] = None

        self._cancel_always_fail: Optional[BaseException] = None

    def set_submit_fail_first(self, e: BaseException) -> None:
        self._submit_fail_first = e

    def set_submit_always_fail(self, e: BaseException) -> None:
        self._submit_always_fail = e

    def set_cancel_always_fail(self, e: BaseException) -> None:
        self._cancel_always_fail = e

    def submit_order(self, cmd: Any) -> Mapping[str, Any]:
        self.submit_calls += 1

        if self._submit_always_fail is not None:
            raise self._submit_always_fail

        if self._submit_fail_first is not None and self.submit_calls == 1:
            raise self._submit_fail_first

        # emulate venue response
        return {"order_id": "v-1001", "client_order_id": getattr(cmd, "client_order_id")}

    def cancel_order(self, cmd: Any) -> Mapping[str, Any]:
        self.cancel_calls += 1

        if self._cancel_always_fail is not None:
            raise self._cancel_always_fail

        return {"canceled": True, "order_id": getattr(cmd, "order_id", None), "client_order_id": getattr(cmd, "client_order_id", None)}


def _mk_submit(*, rid: RequestIdFactory, logical_id: str) -> Any:
    return make_submit_order_command(
        rid=rid,
        actor="strategy:ema",
        venue="binance",
        symbol="BTCUSDT",
        strategy="ema",
        logical_id=logical_id,
        side="buy",
        order_type="limit",
        qty="1",
        price="100",
        tif="GTC",
    )


def test_bridge_submit_idempotent_dedup() -> None:
    clock = FakeClock()
    sleeps: List[float] = []
    sleeper = FakeSleeper(clock=clock, sleeps=sleeps)

    client = FakeVenueClient()
    bridge = ExecutionBridge(
        venue_clients={"binance": client},
        retry_policy=RetryPolicy(max_attempts=3, base_delay_sec=0.1, max_delay_sec=1.0, jitter_sec=0.0),
        rate_limits={},
        breaker_cfg=CircuitBreakerConfig(failure_threshold=99, window_sec=10, cooldown_sec=5),
        clock=clock,
        sleeper=sleeper,
    )

    rid = RequestIdFactory(namespace="qsys", run_id="run-001", deterministic=True)
    cmd = _mk_submit(rid=rid, logical_id="sig-1")

    a1 = bridge.submit(cmd)
    a2 = bridge.submit(cmd)

    assert a1.ok is True
    assert a2.ok is True
    assert a1.deduped is False
    assert a2.deduped is True
    assert client.submit_calls == 1
    assert sleeps == []


def test_bridge_submit_retry_then_success() -> None:
    clock = FakeClock()
    sleeps: List[float] = []
    sleeper = FakeSleeper(clock=clock, sleeps=sleeps)

    client = FakeVenueClient()
    client.set_submit_fail_first(TimeoutError("timeout"))

    bridge = ExecutionBridge(
        venue_clients={"binance": client},
        retry_policy=RetryPolicy(max_attempts=3, base_delay_sec=0.1, max_delay_sec=1.0, jitter_sec=0.0),
        rate_limits={},
        breaker_cfg=CircuitBreakerConfig(failure_threshold=99, window_sec=10, cooldown_sec=5),
        clock=clock,
        sleeper=sleeper,
    )

    rid = RequestIdFactory(namespace="qsys", run_id="run-001", deterministic=True)
    cmd = _mk_submit(rid=rid, logical_id="sig-2")

    ack = bridge.submit(cmd)

    assert ack.ok is True
    assert client.submit_calls == 2
    assert sleeps == [0.1]


def test_bridge_submit_fail_after_max_attempts_and_circuit_opens() -> None:
    clock = FakeClock()
    sleeps: List[float] = []
    sleeper = FakeSleeper(clock=clock, sleeps=sleeps)

    client = FakeVenueClient()
    client.set_submit_always_fail(TimeoutError("timeout"))

    bridge = ExecutionBridge(
        venue_clients={"binance": client},
        retry_policy=RetryPolicy(max_attempts=2, base_delay_sec=0.1, max_delay_sec=1.0, jitter_sec=0.0),
        rate_limits={},
        breaker_cfg=CircuitBreakerConfig(failure_threshold=2, window_sec=10, cooldown_sec=5),
        clock=clock,
        sleeper=sleeper,
    )

    rid = RequestIdFactory(namespace="qsys", run_id="run-001", deterministic=True)
    cmd1 = _mk_submit(rid=rid, logical_id="sig-3")

    ack1 = bridge.submit(cmd1)
    assert ack1.ok is False
    assert ack1.status == "FAILED"
    assert client.submit_calls == 2  # max_attempts=2
    assert sleeps == [0.1]  # 只有一次重试等待

    # 使用新 logical_id 避免命中幂等缓存，验证 circuit open 的 fail-fast
    cmd2 = _mk_submit(rid=rid, logical_id="sig-4")
    ack2 = bridge.submit(cmd2)
    assert ack2.ok is False
    assert ack2.status == "FAILED"
    assert "circuit_open" in (ack2.error or "")
    # circuit open 后不应再调用 client
    assert client.submit_calls == 2


def test_bridge_cancel_non_retryable_rejected() -> None:
    clock = FakeClock()
    sleeps: List[float] = []
    sleeper = FakeSleeper(clock=clock, sleeps=sleeps)

    client = FakeVenueClient()
    client.set_cancel_always_fail(NonRetryableVenueError("insufficient permission"))

    bridge = ExecutionBridge(
        venue_clients={"binance": client},
        retry_policy=RetryPolicy(max_attempts=3, base_delay_sec=0.1, max_delay_sec=1.0, jitter_sec=0.0),
        rate_limits={},
        breaker_cfg=CircuitBreakerConfig(failure_threshold=99, window_sec=10, cooldown_sec=5),
        clock=clock,
        sleeper=sleeper,
    )

    ccmd = make_cancel_order_command(
        actor="strategy:ema",
        venue="binance",
        symbol="BTCUSDT",
        client_order_id="qsys-run-ema-BTCUSDT-abc",
        reason="test",
    )

    ack = bridge.cancel(ccmd)
    assert ack.ok is False
    assert ack.status == "REJECTED"
    assert client.cancel_calls == 1
    assert sleeps == []
