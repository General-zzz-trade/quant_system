# execution/bridge/execution_bridge.py
# NOTE: This is the FULL execution bridge with retry, circuit breaker, and ack store.
# The production coordinator uses engine/execution_bridge.py (simple bridge) by default.
# This module is used by specific adapters that need advanced execution features.
from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Protocol

from execution.store.ack_store import InMemoryAckStore
from execution.store.interfaces import AckStore

logger = logging.getLogger(__name__)


# -----------------------------
# Models
# -----------------------------
@dataclass(frozen=True, slots=True)
class Ack:
    """
    统一返回：不把交易所细节泄漏到 engine。
    status: ACCEPTED / REJECTED / FAILED
    deduped: True 表示该 Ack 是重复调用命中幂等缓存返回的
    """
    status: str
    command_id: str
    idempotency_key: str
    venue: str
    symbol: str
    attempts: int
    deduped: bool = False
    result: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == "ACCEPTED"


# -----------------------------
# Errors
# -----------------------------
class RetryableVenueError(RuntimeError):
    """可重试的交易所/网络错误（例如 timeout/临时断链/临时 5xx）。"""


class NonRetryableVenueError(RuntimeError):
    """不可重试的错误（参数错误、权限错误、余额不足等）。"""


def is_retryable_exception(e: BaseException) -> bool:
    return isinstance(
        e,
        (
            TimeoutError,
            ConnectionError,
            RetryableVenueError,
        ),
    )


# -----------------------------
# Configs
# -----------------------------
@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_sec: float = 0.10
    max_delay_sec: float = 2.00
    jitter_sec: float = 0.05


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    rate_per_sec: float = 10.0
    burst: float = 10.0


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    failure_threshold: int = 8
    window_sec: float = 10.0
    cooldown_sec: float = 5.0
    max_consecutive_trips: int = 5  # permanent halt after N consecutive trips


# -----------------------------
# Small infrastructure
# -----------------------------
class Clock(Protocol):
    def now(self) -> float: ...


class Sleeper(Protocol):
    def sleep(self, sec: float) -> None: ...


@dataclass(slots=True)
class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


@dataclass(slots=True)
class RealSleeper:
    def sleep(self, sec: float) -> None:
        time.sleep(sec)


@dataclass(slots=True)
class TokenBucket:
    rate_per_sec: float
    burst: float
    clock: Clock

    tokens: float = field(init=False)
    last_ts: float = field(init=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.tokens = float(self.burst)
        self.last_ts = self.clock.now()
        self._lock = threading.Lock()

    def allow(self, n: float = 1.0) -> bool:
        with self._lock:
            now = self.clock.now()
            dt = max(0.0, now - self.last_ts)
            self.last_ts = now

            self.tokens = min(self.burst, self.tokens + dt * self.rate_per_sec)
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False


@dataclass(slots=True)
class CircuitBreaker:
    cfg: CircuitBreakerConfig
    clock: Clock
    _fail_ts: list[float] = field(default_factory=list)
    _open_until: float = 0.0
    _consecutive_trips: int = field(default=0, init=False)
    _permanently_open: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def is_open(self) -> bool:
        with self._lock:
            if self._permanently_open:
                return True
            return self.clock.now() < self._open_until

    @property
    def permanently_halted(self) -> bool:
        with self._lock:
            return self._permanently_open

    def record_success(self) -> None:
        with self._lock:
            self._prune()
            self._consecutive_trips = 0

    def record_failure(self) -> None:
        with self._lock:
            if self._permanently_open:
                return
            now = self.clock.now()
            self._fail_ts.append(now)
            self._prune()
            if len(self._fail_ts) >= self.cfg.failure_threshold:
                self._consecutive_trips += 1
                if self._consecutive_trips >= self.cfg.max_consecutive_trips:
                    self._permanently_open = True
                    logger.critical(
                        "CircuitBreaker permanently halted after %d consecutive trips",
                        self._consecutive_trips,
                    )
                else:
                    self._open_until = now + self.cfg.cooldown_sec

    def _prune(self) -> None:
        """Must be called with self._lock held."""
        now = self.clock.now()
        w = self.cfg.window_sec
        if not self._fail_ts:
            return
        self._fail_ts = [t for t in self._fail_ts if (now - t) <= w]


# -----------------------------
# Venue client protocol
# -----------------------------
class VenueClient(Protocol):
    def submit_order(self, cmd: Any) -> Mapping[str, Any]: ...
    def cancel_order(self, cmd: Any) -> Mapping[str, Any]: ...


# -----------------------------
# ExecutionBridge
# -----------------------------
@dataclass(slots=True)
class ExecutionBridge:
    venue_clients: Dict[str, VenueClient]

    # restart-safe idempotency store (default: in-memory)
    ack_store: AckStore = field(default_factory=InMemoryAckStore)

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limits: Dict[str, RateLimitConfig] = field(default_factory=dict)
    breaker_cfg: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    clock: Clock = field(default_factory=MonotonicClock)
    sleeper: Sleeper = field(default_factory=RealSleeper)

    # optional hook: for audit/logging
    on_ack: Optional[Callable[[Ack], None]] = None
    # backpressure: max pending commands when rate-limited (0 = fail-fast)
    pending_queue_size: int = 50

    _buckets: Dict[str, TokenBucket] = field(default_factory=dict, init=False, repr=False)
    _breakers: Dict[str, CircuitBreaker] = field(default_factory=dict, init=False, repr=False)
    _pending: Deque[tuple[Any, str]] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.ack_store, InMemoryAckStore):
            logger.warning(
                "ExecutionBridge using InMemoryAckStore — idempotency keys will be "
                "lost on restart. Pass SQLiteAckStore for production use."
            )

    @staticmethod
    def _ack_to_payload(a: Ack) -> Mapping[str, Any]:
        # Store only JSON-friendly primitives/mappings.
        return {
            "status": a.status,
            "command_id": a.command_id,
            "idempotency_key": a.idempotency_key,
            "venue": a.venue,
            "symbol": a.symbol,
            "attempts": int(a.attempts),
            "deduped": bool(a.deduped),
            "result": dict(a.result) if a.result is not None else None,
            "error": a.error,
        }

    @staticmethod
    def _payload_to_ack(p: Mapping[str, Any]) -> Ack:
        return Ack(
            status=str(p.get("status")),
            command_id=str(p.get("command_id")),
            idempotency_key=str(p.get("idempotency_key")),
            venue=str(p.get("venue")),
            symbol=str(p.get("symbol")),
            attempts=int(p.get("attempts", 0)),
            deduped=bool(p.get("deduped", False)),
            result=p.get("result"),
            error=p.get("error"),
        )

    def _bucket(self, venue: str) -> Optional[TokenBucket]:
        v = venue.lower()
        cfg = self.rate_limits.get(v)
        if not cfg:
            return None
        b = self._buckets.get(v)
        if b is None:
            b = TokenBucket(rate_per_sec=cfg.rate_per_sec, burst=cfg.burst, clock=self.clock)
            self._buckets[v] = b
        return b

    def _breaker(self, venue: str) -> CircuitBreaker:
        v = venue.lower()
        br = self._breakers.get(v)
        if br is None:
            br = CircuitBreaker(cfg=self.breaker_cfg, clock=self.clock)
            self._breakers[v] = br
        return br

    def submit(self, cmd: Any) -> Ack:
        return self._send(cmd=cmd, action="submit")

    def cancel(self, cmd: Any) -> Ack:
        return self._send(cmd=cmd, action="cancel")

    def drain_pending(self) -> list[Ack]:
        """Process queued commands that were rate-limited. Returns list of Acks."""
        results = []
        while self._pending:
            cmd, action = self._pending.popleft()
            ack = self._send(cmd=cmd, action=action)
            results.append(ack)
        return results

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def _send(self, *, cmd: Any, action: str) -> Ack:
        venue = str(getattr(cmd, "venue")).lower()
        symbol = str(getattr(cmd, "symbol")).upper()
        command_id = str(getattr(cmd, "command_id"))
        idem = str(getattr(cmd, "idempotency_key"))

        # ---- dedup ----
        prev_payload = self.ack_store.get(idem)
        if prev_payload is not None:
            prev = self._payload_to_ack(prev_payload)
            ack = Ack(
                status=prev.status,
                command_id=prev.command_id,
                idempotency_key=prev.idempotency_key,
                venue=prev.venue,
                symbol=prev.symbol,
                attempts=prev.attempts,
                deduped=True,
                result=prev.result,
                error=prev.error,
            )
            if self.on_ack:
                self.on_ack(ack)
            return ack

        # ---- circuit breaker ----
        br = self._breaker(venue)
        if br.is_open():
            ack = Ack(
                status="FAILED",
                command_id=command_id,
                idempotency_key=idem,
                venue=venue,
                symbol=symbol,
                attempts=0,
                deduped=False,
                result=None,
                error=f"circuit_open:{venue}",
            )
            self.ack_store.put(idem, self._ack_to_payload(ack))
            if self.on_ack:
                self.on_ack(ack)
            return ack

        # ---- rate limit with backpressure ----
        bucket = self._bucket(venue)
        if bucket is not None and not bucket.allow(1.0):
            if self.pending_queue_size > 0 and len(self._pending) < self.pending_queue_size:
                self._pending.append((cmd, action))
            else:
                ack = Ack(
                    status="FAILED",
                    command_id=command_id,
                    idempotency_key=idem,
                    venue=venue,
                    symbol=symbol,
                    attempts=0,
                    deduped=False,
                    result=None,
                    error=f"rate_limited:{venue}",
                )
                self.ack_store.put(idem, self._ack_to_payload(ack))
                if self.on_ack:
                    self.on_ack(ack)
                return ack
            # Wait briefly for token replenishment then retry
            self.sleeper.sleep(1.0 / max(bucket.rate_per_sec, 1.0))
            if bucket.allow(1.0):
                self._pending.pop()  # remove the one we just queued
            else:
                self._pending.pop()  # queued command must not survive a FAILED ack
                ack = Ack(
                    status="FAILED",
                    command_id=command_id,
                    idempotency_key=idem,
                    venue=venue,
                    symbol=symbol,
                    attempts=0,
                    deduped=False,
                    result=None,
                    error=f"rate_limited_queued:{venue}",
                )
                self.ack_store.put(idem, self._ack_to_payload(ack))
                if self.on_ack:
                    self.on_ack(ack)
                return ack

        # ---- dispatch to venue client with retry ----
        client = self.venue_clients.get(venue)
        if client is None:
            ack = Ack(
                status="FAILED",
                command_id=command_id,
                idempotency_key=idem,
                venue=venue,
                symbol=symbol,
                attempts=0,
                deduped=False,
                result=None,
                error=f"no_venue_client:{venue}",
            )
            self.ack_store.put(idem, self._ack_to_payload(ack))
            if self.on_ack:
                self.on_ack(ack)
            return ack

        attempts = 0
        last_err: Optional[str] = None
        rp = self.retry_policy

        for i in range(1, rp.max_attempts + 1):
            attempts = i
            try:
                if action == "submit":
                    res = client.submit_order(cmd)
                else:
                    res = client.cancel_order(cmd)

                br.record_success()
                ack = Ack(
                    status="ACCEPTED",
                    command_id=command_id,
                    idempotency_key=idem,
                    venue=venue,
                    symbol=symbol,
                    attempts=attempts,
                    deduped=False,
                    result=dict(res),
                    error=None,
                )
                self.ack_store.put(idem, self._ack_to_payload(ack))
                if self.on_ack:
                    self.on_ack(ack)
                return ack

            except NonRetryableVenueError as e:
                br.record_failure()
                last_err = f"non_retryable:{type(e).__name__}:{e}"
                ack = Ack(
                    status="REJECTED",
                    command_id=command_id,
                    idempotency_key=idem,
                    venue=venue,
                    symbol=symbol,
                    attempts=attempts,
                    deduped=False,
                    result=None,
                    error=last_err,
                )
                self.ack_store.put(idem, self._ack_to_payload(ack))
                if self.on_ack:
                    self.on_ack(ack)
                return ack

            except BaseException as e:
                if not is_retryable_exception(e):
                    br.record_failure()
                    last_err = f"unexpected_non_retryable:{type(e).__name__}:{e}"
                    ack = Ack(
                        status="FAILED",
                        command_id=command_id,
                        idempotency_key=idem,
                        venue=venue,
                        symbol=symbol,
                        attempts=attempts,
                        deduped=False,
                        result=None,
                        error=last_err,
                    )
                    self.ack_store.put(idem, self._ack_to_payload(ack))
                    if self.on_ack:
                        self.on_ack(ack)
                    return ack

                br.record_failure()
                last_err = f"retryable:{type(e).__name__}:{e}"

                if i >= rp.max_attempts:
                    ack = Ack(
                        status="FAILED",
                        command_id=command_id,
                        idempotency_key=idem,
                        venue=venue,
                        symbol=symbol,
                        attempts=attempts,
                        deduped=False,
                        result=None,
                        error=last_err,
                    )
                    self.ack_store.put(idem, self._ack_to_payload(ack))
                    if self.on_ack:
                        self.on_ack(ack)
                    return ack

                delay = min(rp.max_delay_sec, rp.base_delay_sec * (2 ** (i - 1)))
                if rp.jitter_sec > 0:
                    delay = delay + random.uniform(0, rp.jitter_sec)
                if delay > 0:
                    self.sleeper.sleep(delay)

        # 理论不可达
        ack = Ack(
            status="FAILED",
            command_id=command_id,
            idempotency_key=idem,
            venue=venue,
            symbol=symbol,
            attempts=attempts,
            deduped=False,
            result=None,
            error=last_err or "unknown",
        )
        self.ack_store.put(idem, self._ack_to_payload(ack))
        if self.on_ack:
            self.on_ack(ack)
        return ack
