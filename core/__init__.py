"""Core foundation layer — types, clock, effects, errors.

This is the bottom of the dependency hierarchy.  Nothing in ``core/`` may
import from ``state/``, ``engine/``, ``event/``, ``risk/``, ``decision/``,
``execution/``, or ``infra/``.  All higher layers depend on ``core/``.
"""
from core.clock import Clock, SystemClock, SimulatedClock
from core.effects import Effects, live_effects, test_effects
from core.errors import QuantError, StateError, RiskError, ExecutionError
from core.types import Envelope

__all__ = [
    "Clock",
    "SystemClock",
    "SimulatedClock",
    "Effects",
    "live_effects",
    "test_effects",
    "QuantError",
    "StateError",
    "RiskError",
    "ExecutionError",
    "Envelope",
]
