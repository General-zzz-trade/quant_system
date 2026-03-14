# portfolio/risk_model/factor/base.py
"""Factor model protocol."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class FactorModelResult:
    """因子模型估计结果。"""
    factor_returns: Mapping[str, Sequence[float]]       # factor → return series
    exposures: Mapping[str, Mapping[str, float]]        # symbol → factor → exposure
    factor_covariance: Mapping[str, Mapping[str, float]]  # factor × factor → cov
    specific_risk: Mapping[str, float]                  # symbol → idiosyncratic var


class FactorModel(Protocol):
    """因子风险模型协议。"""
    name: str

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> FactorModelResult:
        """估计因子模型。"""
        ...
