# portfolio/risk_model/base.py
"""Risk model base protocol and estimate type."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class RiskEstimate:
    """风险模型估计结果。"""
    volatilities: Mapping[str, float]                 # symbol → annualized vol
    correlation: Mapping[str, Mapping[str, float]]    # symbol × symbol → corr
    covariance: Mapping[str, Mapping[str, float]]     # symbol × symbol → cov
    timestamp_ms: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def vol(self, symbol: str) -> float:
        return self.volatilities.get(symbol, 0.0)

    def corr(self, s1: str, s2: str) -> float:
        return self.correlation.get(s1, {}).get(s2, 1.0 if s1 == s2 else 0.0)

    def cov(self, s1: str, s2: str) -> float:
        return self.covariance.get(s1, {}).get(s2, 0.0)


class RiskModel(Protocol):
    """风险模型协议。"""
    name: str

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> RiskEstimate:
        """根据历史收益率序列估计风险参数。"""
        ...
