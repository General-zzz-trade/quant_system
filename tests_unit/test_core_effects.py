"""Tests for core.effects — Effects container and implementations."""
from __future__ import annotations

from core.clock import SimulatedClock
from core.effects import (
    DeterministicRandom,
    Effects,
    InMemoryMetrics,
    InMemoryPersist,
    StdLogger,
    StdRandom,
    live_effects,
    test_effects as make_test_effects,
)


class TestInMemoryMetrics:
    def test_counter_accumulates(self) -> None:
        m = InMemoryMetrics()
        m.counter("orders", 1)
        m.counter("orders", 3)
        assert m.snapshot()["orders"] == 4.0

    def test_gauge_overwrites(self) -> None:
        m = InMemoryMetrics()
        m.gauge("equity", 100.0)
        m.gauge("equity", 200.0)
        assert m.snapshot()["equity"] == 200.0

    def test_histogram(self) -> None:
        m = InMemoryMetrics()
        m.histogram("latency_ms", 12.5)
        assert m.snapshot()["latency_ms"] == 12.5

    def test_tags_create_separate_keys(self) -> None:
        m = InMemoryMetrics()
        m.counter("fills", 1, venue="binance")
        m.counter("fills", 2, venue="okx")
        snap = m.snapshot()
        assert len(snap) == 2


class TestInMemoryPersist:
    def test_save_and_load(self) -> None:
        p = InMemoryPersist()
        p.save_snapshot("state_v1", b"hello")
        assert p.load_snapshot("state_v1") == b"hello"

    def test_load_missing_returns_none(self) -> None:
        p = InMemoryPersist()
        assert p.load_snapshot("nonexistent") is None

    def test_overwrite(self) -> None:
        p = InMemoryPersist()
        p.save_snapshot("key", b"v1")
        p.save_snapshot("key", b"v2")
        assert p.load_snapshot("key") == b"v2"


class TestDeterministicRandom:
    def test_same_seed_same_sequence(self) -> None:
        r1 = DeterministicRandom(seed=123)
        r2 = DeterministicRandom(seed=123)
        seq1 = [r1.uniform(0, 1) for _ in range(10)]
        seq2 = [r2.uniform(0, 1) for _ in range(10)]
        assert seq1 == seq2

    def test_different_seed_different_sequence(self) -> None:
        r1 = DeterministicRandom(seed=1)
        r2 = DeterministicRandom(seed=2)
        v1 = r1.uniform(0, 1)
        v2 = r2.uniform(0, 1)
        assert v1 != v2

    def test_choice(self) -> None:
        r = DeterministicRandom(seed=42)
        items = [1, 2, 3, 4, 5]
        chosen = r.choice(items)
        assert chosen in items


class TestEffectsFactory:
    def test_live_effects(self) -> None:
        e = live_effects()
        assert isinstance(e, Effects)
        assert e.clock is not None
        assert e.log is not None

    def test_make_test_effects(self) -> None:
        e = make_test_effects()
        assert isinstance(e, Effects)
        assert isinstance(e.clock, SimulatedClock)
        assert isinstance(e.metrics, InMemoryMetrics)

    def test_test_effects_deterministic(self) -> None:
        e1 = make_test_effects(seed=99)
        e2 = make_test_effects(seed=99)
        v1 = e1.random.uniform(0, 1)
        v2 = e2.random.uniform(0, 1)
        assert v1 == v2
