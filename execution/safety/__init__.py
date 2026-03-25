"""execution.safety — Pre-execution safety guards (Domain 2).

All guards that run BEFORE an order reaches the venue:
  - CircuitBreaker: rapid error rate -> halt submissions
  - ExecutionKillSwitch: manual or automated kill
  - OrderLimiter: rate / notional / position limits
  - DuplicateGuard: idempotency dedup (+ persistent variant)
  - IntegrityChecker: message-level HMAC
  - OutOfOrderGuard: sequence gap detection
  - RiskGate: composite pre-trade risk check (Rust-delegated)
  - OrderTimeoutTracker: stale-order cancellation
"""
from execution.safety.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, BreakerState
from execution.safety.duplicate_guard import (
    DuplicateGuard, PersistentDedupGuard, PayloadCorruptionError, compute_digest,
)
from execution.safety.kill_switch import ExecutionKillSwitch
from execution.safety.limits import OrderLimiter, OrderLimitsConfig, LimitCheckResult
from execution.safety.message_integrity import IntegrityChecker, IntegrityError
from execution.safety.out_of_order_guard import OutOfOrderGuard, SequencedMessage
from execution.safety.risk_gate import RiskGate, RiskGateConfig
from execution.safety.timeout_tracker import OrderTimeoutTracker

__all__ = [
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "BreakerState",
    # Duplicate guard
    "DuplicateGuard",
    "PersistentDedupGuard",
    "PayloadCorruptionError",
    "compute_digest",
    # Kill switch
    "ExecutionKillSwitch",
    # Order limits
    "OrderLimiter",
    "OrderLimitsConfig",
    "LimitCheckResult",
    # Message integrity
    "IntegrityChecker",
    "IntegrityError",
    # Sequence guard
    "OutOfOrderGuard",
    "SequencedMessage",
    # Risk gate (Rust-delegated)
    "RiskGate",
    "RiskGateConfig",
    # Timeout tracker
    "OrderTimeoutTracker",
]
