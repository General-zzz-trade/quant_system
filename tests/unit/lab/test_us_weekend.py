"""The weekend gap of Binance's US-stock perpetuals (lab/us_weekend.py).

Owner 2026-11-18: "先开始开发策略", then "两个都做" (the weekend gap first) and the stock perpetuals' history as
development data. The times of an event in New York time (daylight saving included), the breaks of the trading
calendar, the prices an event uses, the funding it pays and its net return. Synthetic data only."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from lab import us_weekend as wk

H = wk.H


def utc(text):
    return int(pd.Timestamp(text, tz="UTC").value // 10**6)


def test_the_session_is_new_york_time_with_daylight_saving():
    assert wk.session(date(2026, 1, 9)) == (utc("2026-01-09 14:30"), utc("2026-01-09 21:00"))   # winter: UTC-5
    assert wk.session(date(2026, 7, 10)) == (utc("2026-07-10 13:30"), utc("2026-07-10 20:00"))  # summer: UTC-4


def test_a_break_is_three_or_more_calendar_days_between_trading_days():
    days = [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8),
            date(2026, 7, 9), date(2026, 7, 10), date(2026, 7, 13), date(2026, 7, 14),
            date(2026, 7, 16)]              # 3 July: a holiday before a weekend; 15 July: a mid-week day off
    assert wk.breaks(days) == [(date(2026, 7, 2), date(2026, 7, 6)), (date(2026, 7, 10), date(2026, 7, 13))]


def _hourly(start, end, price=lambda t: 100.0):
    times = np.arange(utc(start), utc(end), H)
    close = np.array([price(t) for t in times])
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "quote_volume": 1e6}, index=times)


def test_the_times_of_a_summer_weekend():
    t = wk.times(date(2026, 7, 10), date(2026, 7, 13), exit_rule="open+1h")
    assert t["reference"] == utc("2026-07-10 20:00")          # the hourly close at Friday's 16:00 New York
    assert t["decision"] == utc("2026-07-13 13:00")           # the last hourly close 30 minutes or more before the open
    assert t["exit"] == utc("2026-07-13 15:00")               # the first hourly close an hour or more after the open
    assert wk.times(date(2026, 7, 10), date(2026, 7, 13), exit_rule="close")["exit"] == utc("2026-07-13 20:00")
    winter = wk.times(date(2026, 1, 9), date(2026, 1, 12), exit_rule="open+1h")
    assert (winter["decision"], winter["exit"]) == (utc("2026-01-12 14:00"), utc("2026-01-12 16:00"))


def test_an_event_reads_the_closes_and_fills_at_the_next_open():
    def price(t):
        if t < utc("2026-07-10 20:00"):
            return 100.0
        if t < utc("2026-07-13 13:00"):
            return 104.0                                       # the weekend moved it up 4 %
        return 101.0                                           # Monday brings it back
    bars = _hourly("2026-06-01", "2026-07-15", price)
    e = wk.event(bars, pd.Series(dtype=float), date(2026, 7, 10), date(2026, 7, 13), exit_rule="open+1h")
    assert e["move"] == pytest.approx(0.04)                   # close at the decision over the Friday reference
    assert e["entry"] == 101.0 and e["exit_price"] == 101.0   # the opens of the bars after the decision and the exit
    bars2 = bars.copy()
    bars2.loc[utc("2026-07-13 13:00"), "open"] = 103.0        # the entry fills at that bar's open
    assert wk.event(bars2, pd.Series(dtype=float), date(2026, 7, 10), date(2026, 7, 13),
                    exit_rule="open+1h")["entry"] == 103.0


def test_the_funding_inside_the_holding_is_counted_once_each():
    funding = pd.Series({utc("2026-07-13 08:00") + 1: 0.001, utc("2026-07-13 16:00"): 0.002,
                         utc("2026-07-12 16:00"): 0.009})
    assert wk.funding_between(funding, utc("2026-07-13 13:00"), utc("2026-07-13 20:00")) == pytest.approx(0.002)
    assert wk.funding_between(funding, utc("2026-07-12 16:00"), utc("2026-07-13 13:00")) == pytest.approx(0.001)


def test_the_net_return_fades_the_move_and_pays_costs_and_funding():
    e = {"move": 0.04, "entry": 100.0, "exit_price": 98.0, "funding": 0.001}
    # fade: short; the price fell 2 %; costs 2 x (5 + 3) bp; a short receives a positive funding rate
    assert wk.net_return(e, threshold=0.02, cost_bp=8) == pytest.approx(0.02 - 0.0016 + 0.001)
    assert wk.net_return(e, threshold=0.05, cost_bp=8) is None             # below the threshold: no trade
    up = {"move": -0.03, "entry": 100.0, "exit_price": 101.0, "funding": 0.001}
    assert wk.net_return(up, threshold=0.02, cost_bp=8) == pytest.approx(0.01 - 0.0016 - 0.001)   # long pays


def test_a_perpetual_that_does_not_track_its_stock_is_no_event():
    assert wk.tracks(100.0, 105.0) and not wk.tracks(100.0, 130.0)


def test_liquidity_is_the_median_daily_quote_volume_of_the_30_days_before():
    bars = _hourly("2026-05-01", "2026-10-01")                  # months of bars after the event
    t = utc("2026-07-10 20:00")
    assert wk.median_daily_quote(bars, t) == pytest.approx(24e6)
    late = bars.copy()
    late.loc[late.index >= t, "quote_volume"] = 1e12            # later volume changes nothing
    assert wk.median_daily_quote(late, t) == pytest.approx(24e6)
