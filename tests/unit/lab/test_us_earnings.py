"""Earnings drift on the US-stock perpetuals' underlyings (lab/us_earnings.py; owner 2026-11-18: "同意，按草案登记",
research/protocols/2026-11-18-us-earnings-drift.md). Parsing, events, signals, the holding weights, the event
returns from a sleeve's curve and the portfolio of open sleeves. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lab import us_earnings as ue

DAY = ue.DAY


def _frame(closes, start="2024-03-04"):
    days = [d for d in pd.date_range(start, periods=len(closes) * 2, freq="D", tz="UTC") if d.dayofweek < 5]
    days = days[:len(closes)]
    idx = pd.Index([int(d.value // 10**6) for d in days])
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx)


def test_eps_is_parsed_with_parentheses_as_negative():
    assert ue.parse_eps("$4.74") == 4.74 and ue.parse_eps("($0.08)") == -0.08 and ue.parse_eps("") is None
    assert ue.surprise(4.74, 4.21) == pytest.approx(0.53 / 4.21)
    assert ue.surprise(0.002, 0.0) == pytest.approx(0.002 / 0.01)              # a forecast near 0: at least 0.01
    assert ue.surprise(None, 1.0) is None


def test_an_event_reads_the_window_around_the_report_and_adjusts_by_qqq():
    stock = _frame([100, 100, 100, 110, 121, 121, 121, 121])                  # report on the 4th day (index 3)
    qqq = _frame([50, 50, 50, 50, 51, 51, 51, 51])
    day = pd.to_datetime(stock.index[3], unit="ms").date().isoformat()
    cal = pd.DataFrame([{"date": day, "symbol": "X", "eps": "$1.10", "epsForecast": "$1.00"},
                        {"date": "2024-03-09", "symbol": "X", "eps": "$1", "epsForecast": "$1"}])   # a Saturday
    evts, skipped = ue.events("X", cal, stock, qqq)
    assert skipped == 1 and len(evts) == 1
    e = evts[0]
    assert e["reaction"] == pytest.approx(121 / 100 - 1)                      # close of the day before to the day after
    assert e["reaction_adjusted"] == pytest.approx(0.21 - 0.02)
    assert e["surprise"] == pytest.approx(0.10) and e["decide"] == int(stock.index[4])


def test_the_signals_need_their_thresholds():
    e = {"surprise": 0.06, "reaction_adjusted": -0.04}
    assert ue.side(e, "surprise") == 1 and ue.side(e, "reaction") == -1
    assert ue.side({"surprise": 0.04, "reaction_adjusted": 0.02}, "surprise") == 0
    assert ue.side({"surprise": None, "reaction_adjusted": 0.02}, "reaction") == 0
    assert ue.side({"surprise": -0.05, "reaction_adjusted": 0.0}, "surprise") == -1          # 5 % counts


def test_a_held_event_has_the_weight_for_its_holding_days_only():
    index = pd.Index(range(0, 20 * DAY, DAY))
    w = ue.weights(index, [{"decide": 3 * DAY, "side": -1}], hold=5)
    assert list(w[w != 0].index) == [3 * DAY, 4 * DAY, 5 * DAY, 6 * DAY, 7 * DAY]   # decided at 3..7, out at 8
    assert (w[w != 0] == -0.9).all() and ue.EXPOSURE == 0.9          # under the broker's 1x exposure cap
    longs, shorts = ue.split(w)
    assert (longs == 0).all() and shorts.equals(w)


def test_an_event_return_is_the_sleeve_equity_from_before_the_entry_to_after_the_exit():
    index = pd.Index(range(0, 20 * DAY, DAY))
    curve = pd.Series(100.0, index=index + DAY)                               # equity at each bar's close
    curve.iloc[5:9] = 103.0                                                   # held: up 3 %
    curve.iloc[9:] = 102.9                                                    # the exit's bar: the fee is paid
    assert ue.event_return(curve, decide=3 * DAY, hold=5) == pytest.approx(0.029)


def test_the_portfolio_averages_the_open_sleeves_and_holds_cash_otherwise():
    a = pd.Series([0.0, 0.01, 0.02, 0.0], index=[0, 1, 2, 3])
    b = pd.Series([0.0, 0.03, 0.0, 0.0], index=[0, 1, 2, 3])
    p = ue.portfolio([a, b])
    assert list(p) == [0.0, pytest.approx(0.02), pytest.approx(0.02), 0.0]


def test_events_cluster_by_the_week_of_their_decision():
    monday = int(pd.Timestamp("2024-03-04", tz="UTC").value // 10**6)
    assert ue.week(monday) == ue.week(monday + 4 * DAY) != ue.week(monday + 7 * DAY)
