"""Gap detection and filling for OHLCV bar time series."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List, Sequence

from data.store import Bar

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Gap:
    """A contiguous range of missing bars."""

    start: datetime
    end: datetime
    expected_bars: int
    reason: str  # "missing" | "maintenance" | "unknown"


@dataclass(frozen=True, slots=True)
class GapReport:
    """Summary of gaps detected for a symbol."""

    symbol: str
    gaps: tuple[Gap, ...]
    total_expected: int
    total_actual: int
    completeness_pct: float


class GapDetector:
    """Detect missing bars in a time series by walking expected timestamps."""

    def __init__(self, *, interval_seconds: int = 3600) -> None:
        self._interval = interval_seconds

    def detect(
        self, bars: Sequence[Bar], *, start: datetime, end: datetime
    ) -> GapReport:
        """Find all gaps between *start* and *end* for the given bars.

        Walks from *start* to *end* in steps of ``interval_seconds`` and
        identifies stretches where no bar exists.
        """
        if not bars:
            symbol = ""
        else:
            symbol = bars[0].symbol

        # Ensure UTC-aware
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        interval = timedelta(seconds=self._interval)

        # Build set of bar timestamps (truncated to interval boundary)
        bar_ts_set: set[datetime] = set()
        for bar in bars:
            ts = bar.ts
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            bar_ts_set.add(ts)

        # Walk expected timestamps
        gaps: list[Gap] = []
        total_expected = 0
        gap_start: datetime | None = None
        gap_count = 0
        current = start

        while current <= end:
            total_expected += 1

            if current not in bar_ts_set:
                if gap_start is None:
                    gap_start = current
                    gap_count = 1
                else:
                    gap_count += 1
            else:
                if gap_start is not None:
                    gap_end = current - interval
                    gaps.append(
                        Gap(
                            start=gap_start,
                            end=gap_end,
                            expected_bars=gap_count,
                            reason="missing",
                        )
                    )
                    gap_start = None
                    gap_count = 0

            current += interval

        # Close trailing gap
        if gap_start is not None:
            gaps.append(
                Gap(
                    start=gap_start,
                    end=end,
                    expected_bars=gap_count,
                    reason="missing",
                )
            )

        total_actual = total_expected - sum(g.expected_bars for g in gaps)
        completeness = (total_actual / total_expected * 100.0) if total_expected > 0 else 100.0

        report = GapReport(
            symbol=symbol,
            gaps=tuple(gaps),
            total_expected=total_expected,
            total_actual=total_actual,
            completeness_pct=round(completeness, 4),
        )

        logger.debug(
            "GapReport %s: %d gaps, %.2f%% complete",
            symbol,
            len(gaps),
            completeness,
        )
        return report


class GapFiller:
    """Fill gaps in bar data using forward-fill or linear interpolation."""

    def fill_forward(self, bars: List[Bar], gaps: Sequence[Gap]) -> List[Bar]:
        """For each gap, create synthetic bars using the last known close.

        Returns a new list containing original bars plus filled bars, sorted by ts.
        """
        if not bars or not gaps:
            return list(bars)

        sorted_bars = sorted(bars, key=lambda b: b.ts)
        filled: list[Bar] = list(sorted_bars)
        bar_ts_set = {b.ts for b in sorted_bars}
        symbol = sorted_bars[0].symbol
        exchange = sorted_bars[0].exchange

        for gap in gaps:
            # Find the bar just before this gap
            last_bar = self._find_bar_before(sorted_bars, gap.start)
            if last_bar is None:
                continue

            last_close = last_bar.close
            last_volume = Decimal("0")

            current = gap.start
            while current <= gap.end:
                ts = current if current.tzinfo else current.replace(tzinfo=timezone.utc)
                if ts not in bar_ts_set:
                    filled.append(
                        Bar(
                            ts=ts,
                            open=last_close,
                            high=last_close,
                            low=last_close,
                            close=last_close,
                            volume=last_volume,
                            symbol=symbol,
                            exchange=exchange,
                        )
                    )
                    bar_ts_set.add(ts)
                # Infer interval from the gap
                if gap.expected_bars > 1:
                    total_secs = (gap.end - gap.start).total_seconds()
                    step = timedelta(seconds=total_secs / (gap.expected_bars - 1))
                    if step.total_seconds() < 1:
                        break
                    current += step
                else:
                    break

        filled.sort(key=lambda b: b.ts)
        return filled

    def fill_linear(self, bars: List[Bar], gaps: Sequence[Gap]) -> List[Bar]:
        """Linear interpolation between the bars surrounding each gap.

        Returns a new list containing original bars plus interpolated bars, sorted by ts.
        """
        if not bars or not gaps:
            return list(bars)

        sorted_bars = sorted(bars, key=lambda b: b.ts)
        filled: list[Bar] = list(sorted_bars)
        bar_ts_set = {b.ts for b in sorted_bars}
        symbol = sorted_bars[0].symbol
        exchange = sorted_bars[0].exchange

        for gap in gaps:
            before = self._find_bar_before(sorted_bars, gap.start)
            after = self._find_bar_after(sorted_bars, gap.end)

            if before is None or after is None:
                continue

            start_close = float(before.close)
            end_close = float(after.close)
            n_steps = gap.expected_bars + 1  # intervals including before/after

            for step_i in range(1, gap.expected_bars + 1):
                frac = step_i / n_steps
                interp_close = Decimal(
                    str(round(start_close + frac * (end_close - start_close), 8))
                )

                total_secs = (gap.end - gap.start).total_seconds()
                if gap.expected_bars > 1:
                    dt_offset = total_secs * ((step_i - 1) / (gap.expected_bars - 1))
                else:
                    dt_offset = 0.0
                ts = gap.start + timedelta(seconds=dt_offset)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                if ts not in bar_ts_set:
                    filled.append(
                        Bar(
                            ts=ts,
                            open=interp_close,
                            high=interp_close,
                            low=interp_close,
                            close=interp_close,
                            volume=Decimal("0"),
                            symbol=symbol,
                            exchange=exchange,
                        )
                    )
                    bar_ts_set.add(ts)

        filled.sort(key=lambda b: b.ts)
        return filled

    @staticmethod
    def _find_bar_before(sorted_bars: List[Bar], ts: datetime) -> Bar | None:
        """Find the last bar with timestamp strictly before *ts*."""
        result = None
        for bar in sorted_bars:
            bar_ts = bar.ts if bar.ts.tzinfo else bar.ts.replace(tzinfo=timezone.utc)
            ref_ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if bar_ts < ref_ts:
                result = bar
            else:
                break
        return result

    @staticmethod
    def _find_bar_after(sorted_bars: List[Bar], ts: datetime) -> Bar | None:
        """Find the first bar with timestamp strictly after *ts*."""
        for bar in sorted_bars:
            bar_ts = bar.ts if bar.ts.tzinfo else bar.ts.replace(tzinfo=timezone.utc)
            ref_ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if bar_ts > ref_ts:
                return bar
        return None
