"""Effect boundary — all side effects flow through injectable protocols.

The core business logic (reducers, signals, risk rules) is pure:
given identical inputs it produces identical outputs.  All I/O —
time, randomness, persistence, logging, metrics — is represented
as an ``Effect`` protocol and injected at the system boundary.

This means:
  * Unit tests use ``FakeEffects`` → deterministic, fast, no I/O.
  * Backtests use ``BacktestEffects`` → simulated clock, in-memory persistence.
  * Production uses ``LiveEffects`` → real clock, real DB, real metrics.
"""
from __future__ import annotations

import logging
import random as _random
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from core.clock import Clock, SimulatedClock, SystemClock


# ── Individual effect protocols ──────────────────────────

class LogEffect(Protocol):
    """Structured logging effect."""
    def debug(self, msg: str, **ctx: Any) -> None: ...
    def info(self, msg: str, **ctx: Any) -> None: ...
    def warning(self, msg: str, **ctx: Any) -> None: ...
    def error(self, msg: str, **ctx: Any) -> None: ...


class MetricsEffect(Protocol):
    """Metrics collection effect."""
    def counter(self, name: str, value: float = 1.0, **tags: str) -> None: ...
    def gauge(self, name: str, value: float, **tags: str) -> None: ...
    def histogram(self, name: str, value: float, **tags: str) -> None: ...


class PersistEffect(Protocol):
    """State persistence effect (WAL, snapshots, etc.)."""
    def save_snapshot(self, key: str, data: bytes) -> None: ...
    def load_snapshot(self, key: str) -> Optional[bytes]: ...


class RandomEffect(Protocol):
    """Controllable randomness."""
    def uniform(self, lo: float, hi: float) -> float: ...
    def choice(self, seq: list) -> Any: ...


# ── Composite effects container ──────────────────────────

@dataclass(frozen=True)
class Effects:
    """All side effects bundled — injected once at system startup."""
    clock: Clock
    log: LogEffect
    metrics: MetricsEffect
    persist: PersistEffect
    random: RandomEffect


# ── Production implementations ───────────────────────────

class StdLogger:
    """Wraps stdlib logging as a ``LogEffect``."""

    __slots__ = ("_log",)

    def __init__(self, name: str = "quant_system") -> None:
        self._log = logging.getLogger(name)

    def debug(self, msg: str, **ctx: Any) -> None:
        self._log.debug(msg, extra={"ctx": ctx})

    def info(self, msg: str, **ctx: Any) -> None:
        self._log.info(msg, extra={"ctx": ctx})

    def warning(self, msg: str, **ctx: Any) -> None:
        self._log.warning(msg, extra={"ctx": ctx})

    def error(self, msg: str, **ctx: Any) -> None:
        self._log.error(msg, extra={"ctx": ctx})


class NoopMetrics:
    """Metrics black-hole — used when no exporter is configured."""

    __slots__ = ()

    def counter(self, name: str, value: float = 1.0, **tags: str) -> None:
        pass

    def gauge(self, name: str, value: float, **tags: str) -> None:
        pass

    def histogram(self, name: str, value: float, **tags: str) -> None:
        pass


class InMemoryMetrics:
    """Metrics stored in a dict — for testing and backtests."""

    def __init__(self) -> None:
        self._data: dict[str, float] = {}

    def counter(self, name: str, value: float = 1.0, **tags: str) -> None:
        key = f"{name}:{tags}" if tags else name
        self._data[key] = self._data.get(key, 0.0) + value

    def gauge(self, name: str, value: float, **tags: str) -> None:
        key = f"{name}:{tags}" if tags else name
        self._data[key] = value

    def histogram(self, name: str, value: float, **tags: str) -> None:
        key = f"{name}:{tags}" if tags else name
        self._data[key] = value

    def snapshot(self) -> dict[str, float]:
        return dict(self._data)


class InMemoryPersist:
    """In-memory persistence — for testing and backtests."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def save_snapshot(self, key: str, data: bytes) -> None:
        self._store[key] = data

    def load_snapshot(self, key: str) -> Optional[bytes]:
        return self._store.get(key)


class StdRandom:
    """Production randomness from stdlib ``random``."""

    __slots__ = ("_rng",)

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = _random.Random(seed)

    def uniform(self, lo: float, hi: float) -> float:
        return self._rng.uniform(lo, hi)

    def choice(self, seq: list) -> Any:
        return self._rng.choice(seq)


class DeterministicRandom:
    """Fixed-seed random — for reproducible backtests and tests."""

    __slots__ = ("_rng",)

    def __init__(self, seed: int = 42) -> None:
        self._rng = _random.Random(seed)

    def uniform(self, lo: float, hi: float) -> float:
        return self._rng.uniform(lo, hi)

    def choice(self, seq: list) -> Any:
        return self._rng.choice(seq)


# ── Convenience factories ────────────────────────────────

def live_effects(*, log_name: str = "quant_system") -> Effects:
    """Production effects — real clock, real I/O."""
    return Effects(
        clock=SystemClock(),
        log=StdLogger(log_name),
        metrics=NoopMetrics(),
        persist=InMemoryPersist(),  # upgrade to DB-backed in Phase 3
        random=StdRandom(),
    )


def test_effects(*, seed: int = 42) -> Effects:
    """Deterministic effects for unit tests."""
    return Effects(
        clock=SimulatedClock(),
        log=StdLogger("test"),
        metrics=InMemoryMetrics(),
        persist=InMemoryPersist(),
        random=DeterministicRandom(seed),
    )
