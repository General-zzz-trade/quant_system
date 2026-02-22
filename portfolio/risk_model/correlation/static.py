# portfolio/risk_model/correlation/static.py
"""Static (constant) correlation model."""
from __future__ import annotations

from typing import Mapping, Sequence


class StaticCorrelation:
    """静态相关性（固定常数或预设矩阵）。"""
    name: str = "static"

    def __init__(
        self,
        default_corr: float = 0.3,
        overrides: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.default_corr = default_corr
        self._overrides = dict(overrides) if overrides else {}

    def estimate(
        self,
        symbols: Sequence[str],
        returns: Mapping[str, Sequence[float]],
    ) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for s1 in symbols:
            row: dict[str, float] = {}
            for s2 in symbols:
                if s1 == s2:
                    row[s2] = 1.0
                else:
                    row[s2] = self._overrides.get(s1, {}).get(
                        s2, self.default_corr
                    )
            result[s1] = row
        return result
