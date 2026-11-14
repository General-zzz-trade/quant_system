"""Drawdown latch arithmetic for UTC day and ISO week periods.

The baseline of a period is its high-water mark: the first observed equity of the period, raised by
every higher observation. A breach latches until the period ends. A stored limit can only tighten:
restarting with a looser configuration never relaxes a period that already has a limit.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

KINDS = ("day", "week")


def validate_limits(daily_pct: float, weekly_pct: float) -> dict[str, float]:
    if any(not math.isfinite(x) or not 0 < x < 100 for x in (daily_pct, weekly_pct)):
        raise ValueError("drawdown limits must be finite percentages between 0 and 100")
    return {"day": daily_pct, "week": weekly_pct}


def periods(now: float) -> dict[str, str]:
    """Period keys of a UTC timestamp: ISO date and ISO week."""
    dt = datetime.fromtimestamp(now, timezone.utc)
    year, week, _ = dt.isocalendar()
    return {"day": dt.date().isoformat(), "week": f"{year}-W{week:02d}"}


def advance(baseline: float, stored_limit: float, configured_limit: float,
            equity: float) -> tuple[float, float, bool]:
    """One observation of a period: (new high-water mark, effective limit, breached)."""
    if not math.isfinite(baseline) or baseline <= 0 or not math.isfinite(stored_limit) or not 0 < stored_limit < 100:
        raise ValueError("invalid persisted equity baseline")
    limit = min(stored_limit, configured_limit)
    peak = max(baseline, equity)
    return peak, limit, equity <= peak * (1 - limit / 100)
