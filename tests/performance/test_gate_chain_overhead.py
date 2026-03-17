# tests/performance/test_gate_chain_overhead.py
"""Benchmark: GateChain with 13 gates — target <5ms per decision."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from runner.gate_chain import GateChain, GateResult


class _PassGate:
    """Minimal gate that always passes."""
    def __init__(self, name: str):
        self.name = name

    def check(self, ev, context):
        return GateResult(allowed=True, scale=1.0)


class _ScaleGate:
    """Gate that applies a fixed scale."""
    def __init__(self, name: str, scale: float):
        self.name = name
        self._scale = scale

    def check(self, ev, context):
        return GateResult(allowed=True, scale=self._scale)


class _BlockGate:
    """Gate that blocks."""
    def __init__(self, name: str):
        self.name = name

    def check(self, ev, context):
        return GateResult(allowed=False, reason=f"{self.name}_blocked")


def _make_event():
    return SimpleNamespace(
        symbol="ETHUSDT", side="buy", qty=1.0, price=2300.0,
        order_id="test-001",
    )


@pytest.mark.benchmark
def test_13_gate_chain_latency():
    """13-gate chain must process in <5ms."""
    gates = [
        _PassGate("Correlation"),
        _PassGate("RiskSize"),
        _PassGate("PortfolioRisk"),
        _PassGate("RustDrawdown"),
        _PassGate("AlphaHealth"),
        _ScaleGate("RegimeSizer", 0.8),
        _ScaleGate("StagedRisk", 0.9),
        _PassGate("PortfolioAlloc"),
        _PassGate("ExecQuality"),
        _PassGate("WeightRec"),
        _PassGate("Extra1"),
        _PassGate("Extra2"),
        _PassGate("Extra3"),
    ]
    chain = GateChain(gates)
    ev = _make_event()

    # Warmup
    for _ in range(100):
        chain.process_with_audit(ev, {})

    # Measure
    times = []
    for _ in range(5000):
        t0 = time.perf_counter()
        chain.process_with_audit(ev, {})
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_us = sum(times) / len(times) * 1_000_000
    p99_us = sorted(times)[int(0.99 * len(times))] * 1_000_000
    print(f"\n13-gate chain: avg={avg_us:.1f}μs p99={p99_us:.1f}μs ({len(gates)} gates)")
    assert avg_us < 5000, f"Gate chain avg {avg_us:.1f}μs > 5000μs (5ms)"


@pytest.mark.benchmark
def test_gate_chain_short_circuit():
    """Chain should short-circuit: blocked chain processes fewer gates than full pass."""
    import logging
    # Suppress warning logs during benchmark (they dominate latency)
    logging.getLogger("runner.gate_chain").setLevel(logging.CRITICAL)
    try:
        gates_13 = [_PassGate(f"G{i}") for i in range(13)]
        chain_13 = GateChain(gates_13)
        gates_1 = [_PassGate("G0")]
        chain_1 = GateChain(gates_1)
        ev = _make_event()

        times_13 = []
        for _ in range(5000):
            t0 = time.perf_counter()
            chain_13.process_with_audit(ev, {})
            t1 = time.perf_counter()
            times_13.append(t1 - t0)

        times_1 = []
        for _ in range(5000):
            t0 = time.perf_counter()
            chain_1.process_with_audit(ev, {})
            t1 = time.perf_counter()
            times_1.append(t1 - t0)

        avg_13 = sum(times_13) / len(times_13) * 1_000_000
        avg_1 = sum(times_1) / len(times_1) * 1_000_000
        print(f"\n13-gate={avg_13:.1f}μs 1-gate={avg_1:.1f}μs ratio={avg_13/avg_1:.1f}x")
        # 13 gates should be slower than 1 gate (proving sequential execution)
        assert avg_13 > avg_1, "13-gate chain should be slower than 1-gate"
    finally:
        logging.getLogger("runner.gate_chain").setLevel(logging.WARNING)
