"""Live derivatives inputs for Binance with the research lookup rules.

History comes from the ``data.downloads.binance_public`` files (metrics
archives are published daily); the most recent days come from the public REST
endpoints. Both feed one ``DerivativesContext``, so ``at(close_ms)`` applies
exactly the same causal rules in research and live trading.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from runner.offline.context import METRICS_LAG_MS, METRICS_MAX_AGE_MS, DerivativesContext

HOUR_MS = 3_600_000
FUNDING_INTERVAL_MS = 8 * HOUR_MS

logger = logging.getLogger(__name__)


class LiveDerivativesContext:
    def __init__(self, adapter: Any, symbol: str, data_dir: Path | None = None) -> None:
        self.adapter, self.symbol = adapter, symbol
        funding_csv = Path(data_dir) / f"{symbol}_funding.csv" if data_dir else None
        metrics_csv = Path(data_dir) / f"{symbol}_metrics.csv" if data_dir else None
        if funding_csv is not None and funding_csv.exists():
            self.context = DerivativesContext.from_files(funding_csv, metrics_csv)
        else:
            self.context = DerivativesContext([], [])

    def refresh(self) -> None:
        """Pull the latest settled funding and 5-minute positioning snapshots."""
        observed, rate = self.adapter.get_last_funding(self.symbol)
        # Same normalization as the research downloader: nominal settlement hour, <= 1s jitter.
        at = round(observed / HOUR_MS) * HOUR_MS
        if abs(observed - at) > 1000:
            raise ValueError(f"funding time {observed} is not within 1s of an hour boundary")
        if self.context.funding_end_ms is None or at > self.context.funding_end_ms:
            self.context.add_funding(at, rate)
        for row in self.adapter.get_positioning(self.symbol, limit=12):
            self.context.add_metrics(row["timestamp"], row)

    def at(self, close_ms: int) -> dict:
        return self.context.at(close_ms)

    def ready(self, close_ms: int, *, needs_metrics: bool) -> tuple[bool, str]:
        values = self.context.at(close_ms)
        if values["funding_rate"] is None:
            return False, "no settled funding rate known"
        if close_ms % FUNDING_INTERVAL_MS == 0 and (self.context.funding_end_ms or 0) < close_ms:
            return False, "funding settlement at this close is not published yet"
        if needs_metrics and values["open_interest"] is None:
            end = self.context.metrics_end_ms
            age = "none" if end is None else f"{(close_ms - end) / 60_000:.0f} min"
            return False, f"no positioning snapshot within {METRICS_MAX_AGE_MS // 60_000} min " \
                          f"(lag {METRICS_LAG_MS // 60_000} min); latest age {age}"
        return True, ""
