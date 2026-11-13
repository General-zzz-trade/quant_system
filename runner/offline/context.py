"""Causal derivatives inputs (funding, open interest, positioning ratios).

``DerivativesContext.at(close_ms)`` returns only observations that were
already public at a bar's close:

- funding: the last *settled* rate with ``funding_time <= close_ms``;
- metrics (5-minute OI / long-short snapshots): the last snapshot whose
  timestamp is at least ``METRICS_LAG_MS`` before the close. The lag keeps a
  snapshot stamped exactly at the close (and possibly published after it)
  out of that bar, and is applied identically in live trading.
"""
from __future__ import annotations

import bisect
import csv
import math
from pathlib import Path

METRICS_LAG_MS = 5 * 60_000
METRICS_MAX_AGE_MS = 30 * 60_000


class DerivativesContext:
    def __init__(self, funding: list[tuple[int, float]], metrics: list[tuple[int, dict[str, float]]]):
        funding = sorted(funding)
        metrics = sorted(metrics, key=lambda row: row[0])
        self._funding_ts = [t for t, _ in funding]
        self._funding = [r for _, r in funding]
        self._metric_ts = [t for t, _ in metrics]
        self._metrics = [m for _, m in metrics]

    @classmethod
    def from_files(cls, funding_csv: Path, metrics_csv: Path | None = None) -> "DerivativesContext":
        with Path(funding_csv).open(newline="") as stream:
            funding = [(int(r["funding_time"]), float(r["rate"])) for r in csv.DictReader(stream)]
        metrics = []
        if metrics_csv is not None and Path(metrics_csv).exists():
            with Path(metrics_csv).open(newline="") as stream:
                for row in csv.DictReader(stream):
                    values = {}
                    for key, column in (("open_interest", "sum_open_interest"),
                                        ("ls_ratio", "count_long_short_ratio"),
                                        ("top_trader_ls_ratio", "sum_toptrader_long_short_ratio")):
                        try:
                            value = float(row[column])
                        except (KeyError, TypeError, ValueError):
                            value = math.nan
                        values[key] = value if math.isfinite(value) and value > 0 else None
                    metrics.append((int(row["timestamp"]), values))
        return cls(funding, metrics)

    def at(self, close_ms: int) -> dict:
        out: dict = {"funding_rate": None, "open_interest": None, "ls_ratio": None, "top_trader_ls_ratio": None}
        i = bisect.bisect_right(self._funding_ts, close_ms) - 1
        if i >= 0:
            out["funding_rate"] = self._funding[i]
        j = bisect.bisect_right(self._metric_ts, close_ms - METRICS_LAG_MS) - 1
        if j >= 0 and close_ms - self._metric_ts[j] <= METRICS_MAX_AGE_MS:
            out.update(self._metrics[j])
        return out

    @property
    def metrics_start_ms(self) -> int | None:
        return self._metric_ts[0] if self._metric_ts else None

    @property
    def metrics_end_ms(self) -> int | None:
        return self._metric_ts[-1] if self._metric_ts else None

    @property
    def funding_end_ms(self) -> int | None:
        return self._funding_ts[-1] if self._funding_ts else None

    # Live extension: observations only ever arrive in time order.
    def add_funding(self, at_ms: int, rate: float) -> None:
        if self._funding_ts and at_ms <= self._funding_ts[-1]:
            if at_ms in self._funding_ts:
                return
            raise ValueError("funding observations must arrive in time order")
        self._funding_ts.append(int(at_ms))
        self._funding.append(float(rate))

    def add_metrics(self, at_ms: int, values: dict) -> None:
        if self._metric_ts and at_ms <= self._metric_ts[-1]:
            return  # already known (or older): history is immutable
        self._metric_ts.append(int(at_ms))
        self._metrics.append({k: values.get(k) for k in ("open_interest", "ls_ratio", "top_trader_ls_ratio")})
