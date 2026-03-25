# decision/signals/statistical
"""Statistical signals."""
from strategy.signals.statistical.cointegration import CointegrationSignal
from strategy.signals.statistical.zscore import ZScoreSignal

__all__ = ["CointegrationSignal", "ZScoreSignal"]
