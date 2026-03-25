"""execution.bridge — Venue-agnostic execution bridge (Domain 3: core plumbing).

Orchestrates order submission with retry, rate-limiting, circuit-breaking,
and error classification. The ExecutionBridge is the single entry point
for all order flow from the decision layer to any venue adapter.

Sub-modules:
  execution_bridge  ExecutionBridge, Ack, retry/rate-limit/CB internals
  request_ids       Idempotency key generation
  error_map         Venue error -> canonical category mapping
  error_policy      Policy-based error handling (retry vs. reject vs. alert)
  infra             Low-level bridge infrastructure helpers
"""
from execution.bridge.execution_bridge import (
    ExecutionBridge,
    Ack,
    VenueClient,
    RetryableVenueError,
    NonRetryableVenueError,
    RetryPolicy,
    RateLimitConfig,
    CircuitBreakerConfig,
    TokenBucket,
    CircuitBreaker,
)
from execution.bridge.request_ids import RequestIdFactory, make_idempotency_key
from execution.bridge.error_map import ErrorMapper, ErrorCategory, ErrorMapping
from execution.bridge.error_policy import ErrorPolicy, ErrorAction, PolicyDecision

__all__ = [
    # Core bridge
    "ExecutionBridge",
    "Ack",
    "VenueClient",
    # Error types
    "RetryableVenueError",
    "NonRetryableVenueError",
    # Retry / rate-limit / CB config
    "RetryPolicy",
    "RateLimitConfig",
    "CircuitBreakerConfig",
    "TokenBucket",
    "CircuitBreaker",
    # Request IDs
    "RequestIdFactory",
    "make_idempotency_key",
    # Error classification
    "ErrorMapper",
    "ErrorCategory",
    "ErrorMapping",
    "ErrorPolicy",
    "ErrorAction",
    "PolicyDecision",
]
