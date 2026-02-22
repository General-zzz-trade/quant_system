# decision/signals/statistical
"""Statistical signals."""
from decision.signals.statistical.cointegration import CointegrationSignal
from decision.signals.statistical.zscore import ZScoreSignal

__all__ = ["CointegrationSignal", "ZScoreSignal"]
