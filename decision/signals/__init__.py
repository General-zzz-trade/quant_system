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


def __getattr__(name: str):  # noqa: ANN001
    """Lazy re-export from strategy.signals to break decision ↔ strategy cycle."""
    if name in ("NullSignal", "SignalModel"):
        from strategy.signals.base import NullSignal, SignalModel
        return {"NullSignal": NullSignal, "SignalModel": SignalModel}[name]
    if name == "WeightedEnsembleSignal":
        from strategy.signals.ensemble import WeightedEnsembleSignal
        return WeightedEnsembleSignal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["NullSignal", "SignalModel", "WeightedEnsembleSignal"]
