"""Market regime detection.

Regime detectors transform market data or feature summaries into discrete labels.
The implementations are lightweight and designed for backtesting and live use.
"""

from .base import RegimeDetector, RegimeLabel
from .composite import CompositeRegimeDetector, CompositeRegimeLabel
from .param_router import RegimeParamRouter, RegimeParams
from .trend import TrendRegimeDetector
from .volatility import VolatilityRegimeDetector

__all__ = [
    "RegimeDetector",
    "RegimeLabel",
    "TrendRegimeDetector",
    "VolatilityRegimeDetector",
    "CompositeRegimeDetector",
    "CompositeRegimeLabel",
    "RegimeParamRouter",
    "RegimeParams",
]
