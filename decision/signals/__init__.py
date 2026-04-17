# decision/signals
"""Signal models for decision engine.

Production live entry: ``EnsemblePredictor`` (alpha_signal.py) used by
``AlphaDecisionModule`` in decision/modules/alpha.py.

Legacy ``WeightedEnsembleSignal`` and ``strategy/signals/{technical,ml,
statistical,factors}/`` trees were removed; nothing in the live path
referenced them.
"""


def __getattr__(name: str):  # noqa: ANN001
    if name in ("NullSignal", "SignalModel", "SignalResult"):
        from strategy.signals.base import NullSignal, SignalModel, SignalResult
        return {
            "NullSignal": NullSignal,
            "SignalModel": SignalModel,
            "SignalResult": SignalResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["NullSignal", "SignalModel", "SignalResult"]
