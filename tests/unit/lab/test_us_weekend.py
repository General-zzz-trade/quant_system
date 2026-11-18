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


def test_an_overnight_is_two_trading_days_one_calendar_day_apart():
    days = [date(2026, 7, 8), date(2026, 7, 9), date(2026, 7, 10), date(2026, 7, 13), date(2026, 7, 14),
            date(2026, 7, 16)]
    assert wk.overnights(days) == [(date(2026, 7, 8), date(2026, 7, 9)), (date(2026, 7, 9), date(2026, 7, 10)),
                                   (date(2026, 7, 13), date(2026, 7, 14))]   # the weekend and the day off are not


def test_the_times_of_an_overnight():
    t = wk.times(date(2026, 7, 8), date(2026, 7, 9), exit_rule="close")
    assert (t["reference"], t["decision"], t["exit"]) == (utc("2026-07-08 20:00"), utc("2026-07-09 13:00"),
                                                         utc("2026-07-09 20:00"))


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


def _weekend_bars(entry_open, exit_open):
    """Friday 100, the weekend up to 104; on Monday the entry bar opens at entry_open, the exit bar at exit_open."""
    def price(t):
        if t < utc("2026-07-10 20:00"):
            return 100.0
        if t < utc("2026-07-13 13:00"):
            return 104.0
        return entry_open if t < utc("2026-07-13 15:00") else exit_open
    bars = _hourly("2026-06-01", "2026-07-15", price)
    bars["volume"] = 1e6
    return bars


RULES = wk.SymbolRules(wk.D("0.01"), wk.D("0.01"), wk.D("5"))
FUNDING = pd.Series({utc("2026-07-13 14:00"): 0.001})


def test_the_fade_goes_through_the_shared_broker_with_its_costs_and_the_funding():
    bars = _weekend_bars(100.0, 98.0)
    e = wk.event(bars, FUNDING, date(2026, 7, 10), date(2026, 7, 13), exit_rule="open+1h")
    assert wk.fade_side(e["move"]) == -1.0                              # up 4 %: short
    r = wk.replay(bars, FUNDING, e, rules=RULES)
    # short at 100, cover at 98: 2 %; fees 2 x 5 bp and slippage 2 x 3 bp; a short receives the positive rate
    assert r == pytest.approx(0.02 - 0.0016 + 0.001, abs=0.0005)


def test_doubled_costs_and_funding_cost_more_and_a_long_pays_the_funding():
    bars = _weekend_bars(100.0, 98.0)
    e = wk.event(bars, FUNDING, date(2026, 7, 10), date(2026, 7, 13), exit_rule="open+1h")
    base = wk.replay(bars, FUNDING, e, rules=RULES)
    worse = wk.replay(bars, FUNDING, e, rules=RULES, cost_multiplier=2)
    assert worse == pytest.approx(base - 0.0016, abs=0.0003)
    more_funding = wk.replay(bars, FUNDING, e, rules=RULES, funding_multiplier=2.0)
    assert more_funding == pytest.approx(base + 0.001, abs=0.0002)     # the short receives twice the rate
    down = {**e, "move": -0.04}                                         # a fall is faded with a long
    assert wk.fade_side(down["move"]) == 1.0
    assert wk.replay(bars, FUNDING, down, rules=RULES) == pytest.approx(-0.02 - 0.0016 - 0.001, abs=0.0005)


def test_an_entry_that_opens_beyond_the_decision_close_still_fills():
    """The broker refuses a fill above its 1x exposure cap: a target of a full 1 was rejected whenever the next open
    moved the notional over it, keeping only some entries (a biased sample). The fade holds 0.9, as the lab's ideas
    hold 0.95."""
    bars = _weekend_bars(105.0, 103.0)                                  # opens above the 104 close, then falls
    e = wk.event(bars, FUNDING, date(2026, 7, 10), date(2026, 7, 13), exit_rule="open+1h")
    assert wk.replay(bars, FUNDING, e, rules=RULES) == pytest.approx(2 / 105 - 0.0016 + 0.001, abs=0.0005)


def test_a_trade_needs_a_move_of_at_least_the_threshold():
    assert wk.trades_at({"move": 0.025}, threshold=0.02) and not wk.trades_at({"move": 0.015}, threshold=0.02)
    assert wk.trades_at({"move": -0.03}, threshold=0.03) and not wk.trades_at({"move": 0.0}, threshold=0.0)


def test_a_perpetual_that_does_not_track_its_stock_is_no_event():
    assert wk.tracks(100.0, 105.0) and not wk.tracks(100.0, 130.0)


def test_liquidity_is_the_median_daily_quote_volume_of_the_30_days_before():
    bars = _hourly("2026-05-01", "2026-10-01")                  # months of bars after the event
    t = utc("2026-07-10 20:00")
    assert wk.median_daily_quote(bars, t) == pytest.approx(24e6)
    late = bars.copy()
    late.loc[late.index >= t, "quote_volume"] = 1e12            # later volume changes nothing
    assert wk.median_daily_quote(late, t) == pytest.approx(24e6)
