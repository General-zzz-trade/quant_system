"""Decision layer (institutional-grade).

Design goals:
- Pure decision (no IO): Snapshot -> DecisionOutput / intents / orders
- Deterministic and replayable (stable IDs, stable explain schema)
- Decoupled from execution (emits orders/intents that can be bridged)
"""
from decision.engine import DecisionEngine
from decision.types import (
    Candidate,
    DecisionExplain,
    DecisionOutput,
    OrderSpec,
    SignalResult,
    TargetPosition,
)

__all__ = [
    "DecisionEngine",
    "Candidate",
    "DecisionExplain",
    "DecisionOutput",
    "OrderSpec",
    "SignalResult",
    "TargetPosition",
]
