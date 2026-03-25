# decision/signals/technical
"""Technical analysis signals — lazy re-export to break decision ↔ strategy cycle."""

_LAZY_MAP = {
    "BollingerBandSignal": "strategy.signals.technical.bollinger_band",
    "BreakoutSignal": "strategy.signals.technical.breakout",
    "GridSignal": "strategy.signals.technical.grid_signal",
    "MACrossSignal": "strategy.signals.technical.ma_cross",
    "MACDSignal": "strategy.signals.technical.macd_signal",
    "MeanReversionSignal": "strategy.signals.technical.mean_reversion",
    "RSISignal": "strategy.signals.technical.rsi_signal",
}


def __getattr__(name: str):  # noqa: ANN001
    mod_path = _LAZY_MAP.get(name)
    if mod_path is not None:
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_MAP.keys())
