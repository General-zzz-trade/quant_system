"""Classic factor-based trading signals — lazy re-export to break decision ↔ strategy cycle."""

_LAZY_MAP = {
    "MomentumSignal": "strategy.signals.factors.momentum",
    "CarrySignal": "strategy.signals.factors.carry",
    "VolatilitySignal": "strategy.signals.factors.volatility",
    "LiquiditySignal": "strategy.signals.factors.liquidity",
    "TrendStrengthSignal": "strategy.signals.factors.trend_strength",
    "VolumePriceDivergenceSignal": "strategy.signals.factors.volume_price_div",
}


def __getattr__(name: str):  # noqa: ANN001
    mod_path = _LAZY_MAP.get(name)
    if mod_path is not None:
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_LAZY_MAP.keys())
