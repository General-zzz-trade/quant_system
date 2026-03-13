# decision/signals
"""Signal models for decision engine.

Signal classification
---------------------
**Production** (used in live/backtest hot path):
  - ``WeightedEnsembleSignal`` — primary ensemble signal (ensemble.py)
  - ``NullSignal`` — no-op / flat signal (base.py)

**Infrastructure** (protocol & adaptive wrappers):
  - ``SignalModel`` — structural typing protocol all signals satisfy (base.py)
  - ``AdaptiveEnsembleSignal`` — Rust-backed adaptive calibration (adaptive_ensemble.py)
  - ``DynamicEnsembleSignal`` — regime-aware dynamic weights (dynamic_ensemble.py)

**Research** (experimental / offline analysis):
  - ``feature_signal.py`` — raw-feature passthrough signal for research notebooks
"""
from decision.signals.base import NullSignal, SignalModel
from decision.signals.ensemble import WeightedEnsembleSignal

__all__ = ["NullSignal", "SignalModel", "WeightedEnsembleSignal"]
