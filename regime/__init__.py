"""Market regime detection.

Regime detectors transform market data or feature summaries into discrete labels.
The implementations are lightweight and designed for backtesting and live use.
"""

from .base import RegimeDetector, RegimeLabel
from .trend import TrendRegimeDetector
from .volatility import VolatilityRegimeDetector
from .liquidity import LiquidityRegimeDetector
from .correlation import CorrelationRegimeDetector

__all__ = [
    "RegimeDetector",
    "RegimeLabel",
    "TrendRegimeDetector",
    "VolatilityRegimeDetector",
    "LiquidityRegimeDetector",
    "CorrelationRegimeDetector",
]
