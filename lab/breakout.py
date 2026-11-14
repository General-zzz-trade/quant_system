"""4h channel breakout, two-sided, one entry per signal (pre-registered 2026-11-14).

Decision at every 4h close (00, 04, ... 20 UTC), from complete 4h bars only:

- flat:  close > highest high of the previous ``entry_bars`` 4h bars -> long;
         close < lowest low of the previous ``entry_bars`` bars     -> short;
- long:  close < previous ``entry_bars`` low -> flip short;
         else close < previous ``exit_bars`` low -> flat;
- short: mirror image.

Every entry carries a protective stop at the decision close -/+ ``stop_atr`` x
ATR(``atr_bars``) and is sized once, at entry, by ``lab.sizing.SizingRule``
from the realized volatility of the last ``vol_bars`` 4h returns and that stop
distance. An open position is never resized. Stops are checked every hour by
the broker; channel exits only at 4h closes.

Definitions (fixed before any real-data run):
- true range uses the previous 4h close; ATR is its simple mean;
- realized volatility is the standard deviation of 4h log returns,
  annualized by sqrt(6 x 365);
- channels exclude the decision bar itself.
All features are finite rolling windows over complete 4h bars, so a trailing
window of ``window_hours()`` hourly bars gives the same values as full history.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from lab.data import aggregate


@dataclass(frozen=True)
class BreakoutSpec:
    entry_bars: int = 20
    exit_bars: int = 10
    atr_bars: int = 14
    stop_atr: float = 1.5
    vol_bars: int = 42             # 7 days of 4h bars
    bar_hours: int = 4
    size_headroom: float = 0.99    # orders use 99% of the rule's notional (price drift, flip costs)

    def __post_init__(self):
        if not 2 <= self.exit_bars <= self.entry_bars or self.atr_bars < 2 or self.vol_bars < 5:
            raise ValueError("invalid channel, ATR or volatility window")
        if not 0 < self.stop_atr < 10 or not 0 < self.size_headroom <= 1 or 24 % self.bar_hours:
            raise ValueError("invalid stop multiple, headroom or bar size")

    def record(self) -> dict:
        return asdict(self)

    def window_hours(self) -> int:
        """Hourly history needed for exact trailing-window evaluation (+2 bars of slack)."""
        return (max(self.entry_bars, self.atr_bars, self.vol_bars) + 3) * self.bar_hours


def features(hourly: pd.DataFrame, spec: BreakoutSpec) -> pd.DataFrame:
    """One row per complete 4h bar; ``available_at`` is its close time (ms)."""
    bars = aggregate(hourly, spec.bar_hours)
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev_close = close.shift()
    true_range = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = true_range.rolling(spec.atr_bars, min_periods=spec.atr_bars).mean()
    periods_per_year = 24 / spec.bar_hours * 365
    vol = np.log(close).diff().rolling(spec.vol_bars, min_periods=spec.vol_bars).std() * math.sqrt(periods_per_year)

    def channel(series, bars_, fn):
        return getattr(series.shift(1).rolling(bars_, min_periods=bars_), fn)()

    out = pd.DataFrame({
        "available_at": bars["available_at"], "close": close, "atr": atr, "vol": vol,
        "hi_entry": channel(high, spec.entry_bars, "max"), "lo_entry": channel(low, spec.entry_bars, "min"),
        "hi_exit": channel(high, spec.exit_bars, "max"), "lo_exit": channel(low, spec.exit_bars, "min"),
    })
    out["ready"] = np.isfinite(out[["atr", "vol", "hi_entry", "lo_entry", "hi_exit", "lo_exit"]]).all(axis=1)
    return out.reset_index(drop=True)


def desired_side(side: int, row) -> int:
    """Side wanted after this 4h close, given the current side (-1, 0, +1)."""
    close = row.close
    if side == 0:
        return 1 if close > row.hi_entry else -1 if close < row.lo_entry else 0
    if side > 0:
        return -1 if close < row.lo_entry else 0 if close < row.lo_exit else 1
    return 1 if close > row.hi_entry else 0 if close > row.hi_exit else -1


def stop_distance(row, spec: BreakoutSpec) -> float:
    """Protective stop distance as a fraction of the decision close."""
    return spec.stop_atr * float(row.atr) / float(row.close)


__all__ = ["BreakoutSpec", "features", "desired_side", "stop_distance"]
