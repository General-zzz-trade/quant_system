# decision/signals/statistical
"""Statistical signals — lazy re-export to break decision ↔ strategy cycle."""


def __getattr__(name: str):  # noqa: ANN001
    if name == "CointegrationSignal":
        from strategy.signals.statistical.cointegration import CointegrationSignal
        return CointegrationSignal
    if name == "ZScoreSignal":
        from strategy.signals.statistical.zscore import ZScoreSignal
        return ZScoreSignal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CointegrationSignal", "ZScoreSignal"]
