"""Daily replays of US stocks and ETFs at the costs of Binance's US-equity perpetuals (owner 2026-11-18: "执行", the
development route for the US-stock perpetuals, step 1).

The perpetuals listed between 2025-12 and 2026-07, so a strategy's signals are researched on the underlyings' own
history (``data/downloads/us_equity_daily.py``: 10 years of daily bars, SPY's and GLD's NAV since 2003/2004) and the
perpetual supplies the costs. A replay goes through the shared broker (``lab.simulate``) with day-long bars:

- A decision is made at a day's close (the bar is indexed by the day's midnight UTC and is available a day later)
  and fills at the next trading day's open: a Friday decision fills at Monday's open, the weekend gap included.
- The lot rules are the perpetual's (``exchange_info.json`` of the perpetuals' snapshot); fees and slippage are the
  research broker's (``broker_config``), scaled by ``cost_multiplier``.
- Funding: the perpetual settles every day, the stock market trades on weekdays. Each bar charges the perpetual's
  mean daily funding rate (its mean settlement rate times its settlements a day, over the whole history in the
  perpetuals' snapshot; positive: longs pay) for every calendar day since the previous bar, times
  ``funding_multiplier``. One rate for the whole replay: the perpetual did not exist for most of it.
- NAV files (SPY, GLD) have one price a day and no volume: their bars are flat (open = high = low = close = NAV, so an
  exchange stop fills at the next NAV, which flatters stops) and their volume is ``NAV_VOLUME``, so the broker's
  participation cap never binds (at this capital it would not bind on SPY's real volume either).
- FRED index levels (the Nasdaq Composite since 1971, added 2026-11-18 for a long check) are flat bars like a NAV,
  scaled to end at 100, at the costs of the perpetual that tracks them (NASDAQCOM -> QQQUSDT).
- ``leverage`` as ``lab.um_lab.replay``: weights times the leverage, isolated margin at the largest exchange leverage
  the stop allows.

Not modelled: dividends (the underlyings' prices are not dividend-adjusted; a perpetual pays none either), the gap
between a perpetual's price and its underlying, and the perpetuals' own trading at night and on weekends.
"""
from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path

import numpy as np
import pandas as pd

from lab import um_lab
from lab.simulate import simulate
from lab.strategy import StrategySpec
from trading.domain.bars import Bar
from trading.domain.risk.exposure import MAX_TOTAL_LEVERAGE

DAY = 86_400_000
ROOT = Path(__file__).resolve().parents[1]
US_SNAPSHOT = ROOT / "data_files" / "us_equity_daily_2026-11-12"
PERPS_SNAPSHOT = ROOT / "data_files" / "binance_usdt_perps_2020_2026-11-12"
NAV_VOLUME = 1e9
# The underlying -> the perpetual whose costs it carries. GLD stands in for gold (XAUUSDT); the others share a name.
PERPETUAL = {"GLD": "XAUUSDT", "NASDAQCOM": "QQQUSDT"}


def perpetual(ticker: str) -> str:
    return PERPETUAL.get(ticker, f"{ticker}USDT")


def daily_frame(ticker: str, snapshot: Path = US_SNAPSHOT, *, nav: bool = False, index: bool = False) -> pd.DataFrame:
    """Daily bars indexed by the day's midnight UTC (ms): open, high, low, close, volume. ``index``: a FRED index level
    (``{ticker}_index.csv``), flat bars like a NAV, scaled so its last close is 100 (only returns matter, and the
    perpetual's lot step then fits the price)."""
    if index:
        raw = pd.read_csv(Path(snapshot) / f"{ticker}_index.csv").dropna(subset=["close"])
        level = raw["close"].astype(float).to_numpy()
        price = level * 100.0 / level[-1]
        df = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": NAV_VOLUME})
    elif nav:
        raw = pd.read_csv(Path(snapshot) / f"{ticker}_nav.csv").dropna(subset=["nav"])
        price = raw["nav"].astype(float).to_numpy()
        df = pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": NAV_VOLUME})
    else:
        raw = pd.read_csv(Path(snapshot) / f"{ticker}_1d.csv").dropna(subset=["close"])
        df = raw[["open", "high", "low", "close"]].astype(float).copy()
        df["volume"] = raw["volume"].fillna(0.0).astype(float).to_numpy()
    df.index = pd.Index((pd.to_datetime(raw["date"], utc=True).astype("int64") // 10**6).to_numpy(np.int64))
    return df.sort_index()


def daily_funding(perp: str, perps_snapshot: Path = PERPS_SNAPSHOT) -> float | None:
    """The perpetual's mean funding rate per day over its whole history in the snapshot; None without one."""
    path = Path(perps_snapshot) / f"{perp}_funding.csv.gz"
    if not path.is_file():
        return None
    fund = pd.read_csv(path).sort_values("funding_time")
    if len(fund) < 2:
        return None
    gap = float(np.median(np.diff(fund["funding_time"].to_numpy(np.int64))))
    return float(fund["rate"].mean()) * DAY / gap


def replay_daily(ticker: str, weights: pd.Series, *, us_snapshot: Path = US_SNAPSHOT,
                 perps_snapshot: Path = PERPS_SNAPSHOT, perp: str | None = None, starting_balance: str,
                 start: int | None = None, end: int | None = None, nav: bool = False, cost_multiplier: int = 1,
                 funding_multiplier: float = 1.0, stop_pct: str = "0", pause_ms: int | None = None,
                 reenter_after_stop: str = "pause", leverage: str = "1",
                 daily_rate: float | None = None, index: bool = False, frame: pd.DataFrame | None = None) -> dict:
    """Target weights (decided at each day's close, index = the day's midnight UTC) through the shared broker.
    Returns the equity at each bar's close (``curve``), the fills, the funding paid per bar (open time -> cashflow) and
    the calendar days each bar's funding covered. ``daily_rate`` replaces the perpetual's measured rate (a study's
    conservative or stress assumption); the perpetual's lot rules still apply."""
    perp = perp or perpetual(ticker)
    rate = daily_funding(perp, perps_snapshot) if daily_rate is None else daily_rate
    if rate is None:
        raise ValueError(f"no funding history for {perp}: its costs are unknown")
    lev = D(leverage)
    exchange = um_lab.exchange_leverage(stop_pct) if lev != 1 else 0
    if lev != 1 and (not 1 <= lev <= MAX_TOTAL_LEVERAGE or D(stop_pct) <= 0 or not lev <= exchange):
        raise ValueError(f"a replay at leverage {leverage} needs a stop that allows it (stop {stop_pct}: {exchange}x)")
    df = frame if frame is not None else daily_frame(ticker, us_snapshot, nav=nav, index=index)
    first = int(df.index[0]) if start is None else start
    bars = [Bar(int(t), D(repr(r.open)), D(repr(r.high)), D(repr(r.low)), D(repr(r.close)), D(repr(r.volume)),
                interval_ms=DAY) for t, r in zip(df.index, df.itertuples())]
    days = np.diff(df.index.to_numpy(np.int64), prepend=int(df.index[0]) - DAY) // DAY
    funding = {int(t): (D(repr(rate * int(n) * funding_multiplier)),) for t, n in zip(df.index, days)}
    w = weights.reindex(df.index).fillna(0.0).clip(-1.0, 1.0) * float(lev)
    targets = pd.DataFrame({"available_at": df.index.to_numpy(np.int64) + DAY, "target": w.to_numpy(float),
                            "ready": np.ones(len(df), dtype=bool)})
    result = simulate(ticker, bars, targets, funding, StrategySpec(execution="market"), start_ms=first, end_ms=end,
                      rules=um_lab.rules(perp, perps_snapshot), cost_multiplier=cost_multiplier,
                      starting_balance=starting_balance, stop_pct=stop_pct, pause_after_stop_ms=pause_ms,
                      reenter_after_stop=reenter_after_stop, leverage=leverage, exchange_leverage=exchange)
    curve = result.curve.set_index("at")
    paid = curve["funding"].diff().fillna(curve["funding"])
    return {"symbol": ticker, "perp": perp, "daily_funding_rate": rate, "curve": curve["equity"],
            "fills": [um_lab._fill(f) for f in result.fills],
            "funding": {int(at) - DAY: float(v) for at, v in paid.items()},
            "funding_days": {int(t): int(n) for t, n in zip(df.index, days)}}


__all__ = ["DAY", "NAV_VOLUME", "PERPETUAL", "daily_frame", "daily_funding", "perpetual", "replay_daily"]
