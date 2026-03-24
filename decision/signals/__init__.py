# decision/signals
"""Signal models for decision engine.

Signal classification
---------------------
**Production** (used in live/backtest hot path):
  - ``WeightedEnsembleSignal`` — primary ensemble signal (ensemble.py)
  - ``NullSignal`` — no-op / flat signal (base.py)

**Infrastructure** (protocol & adaptive wrappers):
  - ``SignalModel`` — structural typing protocol all signals satisfy (base.py)
"""
from decision.signals.base import NullSignal, SignalModel
from decision.signals.ensemble import WeightedEnsembleSignal

__all__ = ["NullSignal", "SignalModel", "WeightedEnsembleSignal"]
