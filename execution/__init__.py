"""Execution layer — 4 logical domains.

Domain 1: adapters/       Exchange connectivity (Bybit, Binance, Hyperliquid, Polymarket)
Domain 2: safety/         Pre-execution safety guards (circuit breaker, kill switch, limits, dedup)
Domain 3: core plumbing   bridge/ + models/ + state_machine/ + store/ + ingress/
Domain 4: ops             reconcile/ + observability/ + latency/ + sim/

Top-level utilities: balance_utils, order_utils
"""
from execution.bridge.execution_bridge import ExecutionBridge
from execution.models.orders import CanonicalOrder
from execution.models.fills import CanonicalFill
from execution.balance_utils import get_total_and_free_balance
from execution.order_utils import reliable_close_position, clamp_notional

__all__ = [
    # --- Core plumbing (Domain 3) ---
    "ExecutionBridge",
    "CanonicalOrder",
    "CanonicalFill",
    # --- Top-level utilities ---
    "get_total_and_free_balance",
    "reliable_close_position",
    "clamp_notional",
]
