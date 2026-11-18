"""Daily replays of US stocks and ETFs at the perpetuals' costs (lab/us_equity.py).

Owner 2026-11-18: "执行" (the development route for the US-stock perpetuals: step 1, the research tool). The
underlyings' daily bars go through the shared broker (lab.simulate) as day-long bars; the costs are the perpetual's:
its lot rules, its funding charged for every calendar day held, fees and slippage. Synthetic data only."""
from __future__ import annotations

import json
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from lab import us_equity as us

DAY = us.DAY
FRI, MON = pd.Timestamp("2024-01-05", tz="UTC"), pd.Timestamp("2024-01-08", tz="UTC")


def _days(n=60, start="2023-11-01"):
    """Weekdays only, as the stock market trades."""
    return [d for d in pd.date_range(start, periods=n * 2, freq="D", tz="UTC") if d.dayofweek < 5][:n]


@pytest.fixture
def snaps(tmp_path):
    us_dir, perps = tmp_path / "us", tmp_path / "perps"
    us_dir.mkdir()
    perps.mkdir()
    days = _days()
    close = 100 * np.exp(np.linspace(0, 0.2, len(days)))
    pd.DataFrame({"date": [d.date().isoformat() for d in days], "open": close * 0.999, "high": close * 1.01,
                  "low": close * 0.99, "close": close, "volume": 1e6}).to_csv(us_dir / "QQQ_1d.csv", index=False)
    pd.DataFrame({"date": [d.date().isoformat() for d in days], "nav": close}).to_csv(us_dir / "SPY_nav.csv",
                                                                                   index=False)
    times = [int(pd.Timestamp("2026-06-01", tz="UTC").value // 10**6) + k * 8 * 3_600_000 for k in range(90)]
    pd.DataFrame({"funding_time": times, "rate": 0.0001}).to_csv(perps / "QQQUSDT_funding.csv.gz", index=False)
    (perps / "exchange_info.json").write_text(json.dumps({"QQQUSDT": {"step": "0.01", "min_notional": "5"}}))
    return us_dir, perps


def test_a_daily_frame_is_indexed_by_the_day_s_midnight_utc(snaps):
    us_dir, _ = snaps
    df = us.daily_frame("QQQ", us_dir)
    assert df.index[0] == int(pd.Timestamp("2023-11-01", tz="UTC").value // 10**6)
    assert (np.diff(df.index) % DAY == 0).all() and list(df.columns[:5]) == ["open", "high", "low", "close", "volume"]


def test_a_nav_file_becomes_flat_bars_whose_volume_never_binds(snaps):
    us_dir, _ = snaps
    df = us.daily_frame("SPY", us_dir, nav=True)
    assert (df["open"] == df["close"]).all() and (df["high"] == df["close"]).all()
    assert (df["volume"] == us.NAV_VOLUME).all()


def test_the_daily_funding_rate_is_the_perpetual_s_mean_settlement_times_settlements_a_day(snaps):
    _, perps = snaps
    assert us.daily_funding("QQQUSDT", perps) == pytest.approx(0.0003)     # 3 settlements a day of 0.01 %
    assert us.daily_funding("NONEUSDT", perps) is None


def _replay(snaps, weights, **kw):
    us_dir, perps = snaps
    df = us.daily_frame("QQQ", us_dir)
    return us.replay_daily("QQQ", weights.reindex(df.index).fillna(0.0), us_snapshot=us_dir, perps_snapshot=perps,
                           perp="QQQUSDT", starting_balance="1000", **kw), df


def test_a_friday_decision_fills_at_monday_s_open_and_pays_three_days_of_funding(snaps):
    us_dir, _ = snaps
    df = us.daily_frame("QQQ", us_dir)
    fri = int(FRI.value // 10**6)
    weights = pd.Series(np.where(df.index >= fri, 0.5, 0.0), index=df.index)
    r, _ = _replay(snaps, weights)
    first = r["fills"][0]
    assert first["at"] == int(MON.value // 10**6)
    assert D(first["price"]) >= D(str(df.loc[int(MON.value // 10**6), "open"]))   # bought at the open, with slippage
    paid = r["funding"]                                                    # bar open time -> cashflow
    tue = int(MON.value // 10**6) + DAY
    wed = tue + DAY
    assert paid[wed] == pytest.approx(paid[tue], rel=0.05) and paid[tue] < 0  # longs pay, one day at a time
    assert r["funding_days"][wed] == 1 and r["funding_days"][int(MON.value // 10**6) + 7 * DAY] == 3
    next_mon = int(MON.value // 10**6) + 7 * DAY
    assert paid[next_mon] == pytest.approx(3 * paid[next_mon + DAY], rel=0.05)   # the weekend's three days at once


def test_the_funding_multiplier_scales_the_funding_alone(snaps):
    us_dir, _ = snaps
    df = us.daily_frame("QQQ", us_dir)
    weights = pd.Series(0.8, index=df.index)
    base, _ = _replay(snaps, weights)
    double, _ = _replay(snaps, weights, funding_multiplier=2.0)
    assert sum(double["funding"].values()) == pytest.approx(2 * sum(base["funding"].values()), rel=0.05)


def test_a_nav_file_replays_and_fills(snaps):
    us_dir, perps = snaps
    df = us.daily_frame("SPY", us_dir, nav=True)
    r = us.replay_daily("SPY", pd.Series(0.5, index=df.index), us_snapshot=us_dir, perps_snapshot=perps,
                        perp="QQQUSDT", nav=True, starting_balance="1000")
    assert r["fills"] and float(r["fills"][0]["qty"]) > 0


def test_leverage_needs_a_stop_inside_the_liquidation(snaps):
    us_dir, _ = snaps
    df = us.daily_frame("QQQ", us_dir)
    with pytest.raises(ValueError, match="leverage"):
        _replay(snaps, pd.Series(0.5, index=df.index), leverage="2")


def test_doubling_the_costs_and_the_funding_costs_more(snaps):
    us_dir, _ = snaps
    df = us.daily_frame("QQQ", us_dir)
    weights = pd.Series(np.where(np.arange(len(df)) % 10 < 5, 0.8, 0.0), index=df.index)
    base, _ = _replay(snaps, weights)
    worse, _ = _replay(snaps, weights, cost_multiplier=2, funding_multiplier=2.0)
    assert worse["curve"].iloc[-1] < base["curve"].iloc[-1]


def test_leverage_two_holds_twice_the_notional(snaps):
    us_dir, _ = snaps
    df = us.daily_frame("QQQ", us_dir)
    weights = pd.Series(0.9, index=df.index)
    one, _ = _replay(snaps, weights)
    two, _ = _replay(snaps, weights, leverage="2", stop_pct="0.08")
    n = lambda r: float(D(r["fills"][0]["qty"]) * D(r["fills"][0]["price"]))   # noqa: E731
    assert n(two) == pytest.approx(2 * n(one), rel=0.02)


def test_later_prices_change_nothing_before(snaps):
    us_dir, perps = snaps
    df = us.daily_frame("QQQ", us_dir)
    weights = pd.Series(0.7, index=df.index)
    r, _ = _replay(snaps, weights)
    cut = df.index[30]
    later = us_dir.parent / "later"
    later.mkdir()
    raw = pd.read_csv(us_dir / "QQQ_1d.csv")
    raw.loc[30:, ["open", "high", "low", "close"]] *= 3
    raw.to_csv(later / "QQQ_1d.csv", index=False)
    again = us.replay_daily("QQQ", weights, us_snapshot=later, perps_snapshot=perps, perp="QQQUSDT",
                            starting_balance="1000")
    assert again["curve"].loc[:cut].equals(r["curve"].loc[:cut])


def test_an_unknown_perpetual_has_no_funding_and_is_refused(snaps):
    us_dir, perps = snaps
    df = us.daily_frame("QQQ", us_dir)
    with pytest.raises(ValueError, match="funding"):
        us.replay_daily("QQQ", pd.Series(0.5, index=df.index), us_snapshot=us_dir, perps_snapshot=perps,
                        perp="NONEUSDT", starting_balance="1000")


def test_a_given_daily_rate_replaces_the_measured_one(snaps):
    us_dir, _ = snaps
    df = us.daily_frame("QQQ", us_dir)
    weights = pd.Series(0.8, index=df.index)
    measured, _ = _replay(snaps, weights)
    stressed, _ = _replay(snaps, weights, daily_rate=0.20 / 365)          # above the 0.03 % a day measured
    assert stressed["daily_funding_rate"] == pytest.approx(0.20 / 365)
    assert sum(stressed["funding"].values()) < sum(measured["funding"].values()) < 0
