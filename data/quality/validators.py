"""Bar data quality validators — OHLC consistency, time continuity, anomaly detection."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Sequence

from data.store import Bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of bar validation with errors, warnings, and statistics."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    stats: dict[str, Any]


class BarValidator:
    """Validates OHLCV bar data quality.

    Checks performed:
    1. OHLC consistency: high >= max(open, close), low <= min(open, close)
    2. Time continuity: bars sorted by ts, no duplicate timestamps
    3. Anomaly detection: Z-score on close-to-close returns exceeding threshold
    4. Volume non-negative (when present)
    """

    def __init__(
        self,
        *,
        zscore_threshold: float = 5.0,
        max_gap_seconds: int = 7200,
    ) -> None:
        self._zscore_threshold = zscore_threshold
        self._max_gap_seconds = max_gap_seconds

    def validate(self, bars: Sequence[Bar]) -> ValidationResult:
        """Run all validation checks on a sequence of bars."""
        errors: list[str] = []
        warnings: list[str] = []
        anomaly_count = 0

        if not bars:
            return ValidationResult(
                valid=True,
                errors=(),
                warnings=("empty bar sequence",),
                stats={"total_bars": 0, "anomalies": 0},
            )

        # -- Check OHLC consistency --
        for i, bar in enumerate(bars):
            ohlc_errors = self._check_ohlc(bar, i)
            errors.extend(ohlc_errors)

            vol_errors = self._check_volume(bar, i)
            errors.extend(vol_errors)

        # -- Check time continuity --
        time_errors, time_warnings = self._check_time_continuity(bars)
        errors.extend(time_errors)
        warnings.extend(time_warnings)

        # -- Anomaly detection on returns --
        anomaly_warnings, anomaly_count = self._check_return_anomalies(bars)
        warnings.extend(anomaly_warnings)

        stats: dict[str, Any] = {
            "total_bars": len(bars),
            "anomalies": anomaly_count,
            "ohlc_errors": sum(1 for e in errors if "OHLC" in e),
            "volume_errors": sum(1 for e in errors if "volume" in e.lower()),
            "time_errors": len(time_errors),
            "gap_warnings": len(time_warnings),
        }

        return ValidationResult(
            valid=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            stats=stats,
        )

    def _check_ohlc(self, bar: Bar, idx: int) -> list[str]:
        """Verify high >= max(open, close) and low <= min(open, close)."""
        errors: list[str] = []
        high_floor = max(bar.open, bar.close)
        low_ceil = min(bar.open, bar.close)

        if bar.high < high_floor:
            errors.append(
                f"bar[{idx}] OHLC: high {bar.high} < max(open, close) {high_floor} "
                f"at {bar.ts.isoformat()}"
            )
        if bar.low > low_ceil:
            errors.append(
                f"bar[{idx}] OHLC: low {bar.low} > min(open, close) {low_ceil} "
                f"at {bar.ts.isoformat()}"
            )
        if bar.high < bar.low:
            errors.append(
                f"bar[{idx}] OHLC: high {bar.high} < low {bar.low} "
                f"at {bar.ts.isoformat()}"
            )
        return errors

    def _check_volume(self, bar: Bar, idx: int) -> list[str]:
        """Verify volume is non-negative when present."""
        if bar.volume is not None and bar.volume < 0:
            return [
                f"bar[{idx}] negative volume {bar.volume} at {bar.ts.isoformat()}"
            ]
        return []

    def _check_time_continuity(
        self, bars: Sequence[Bar]
    ) -> tuple[list[str], list[str]]:
        """Check bars are sorted by ts with no duplicates, flag large gaps."""
        errors: list[str] = []
        warnings: list[str] = []

        for i in range(1, len(bars)):
            prev_ts = bars[i - 1].ts
            curr_ts = bars[i].ts

            if curr_ts < prev_ts:
                errors.append(
                    f"bar[{i}] out of order: {curr_ts.isoformat()} < {prev_ts.isoformat()}"
                )
            elif curr_ts == prev_ts:
                errors.append(
                    f"bar[{i}] duplicate timestamp: {curr_ts.isoformat()}"
                )
            else:
                gap_secs = (curr_ts - prev_ts).total_seconds()
                if gap_secs > self._max_gap_seconds:
                    warnings.append(
                        f"bar[{i}] gap of {gap_secs:.0f}s between "
                        f"{prev_ts.isoformat()} and {curr_ts.isoformat()}"
                    )

        return errors, warnings

    def _check_return_anomalies(
        self, bars: Sequence[Bar]
    ) -> tuple[list[str], int]:
        """Detect anomalous close-to-close returns using Z-score."""
        warnings: list[str] = []
        anomaly_count = 0

        if len(bars) < 3:
            return warnings, anomaly_count

        # Compute log returns
        returns: list[float] = []
        for i in range(1, len(bars)):
            prev_close = float(bars[i - 1].close)
            curr_close = float(bars[i].close)
            if prev_close > 0 and curr_close > 0:
                returns.append(math.log(curr_close / prev_close))
            else:
                returns.append(0.0)

        if len(returns) < 2:
            return warnings, anomaly_count

        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0.0:
            return warnings, anomaly_count

        for i, ret in enumerate(returns):
            zscore = abs((ret - mean) / std)
            if zscore > self._zscore_threshold:
                bar_idx = i + 1  # offset because returns start at bar[1]
                anomaly_count += 1
                warnings.append(
                    f"bar[{bar_idx}] anomalous return: z-score {zscore:.2f} "
                    f"(return {ret:.6f}) at {bars[bar_idx].ts.isoformat()}"
                )

        return warnings, anomaly_count
