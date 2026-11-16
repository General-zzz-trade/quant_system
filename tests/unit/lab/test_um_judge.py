"""The lab's judge (lab/um_judge.py): periods, the yearly universe rule, the stop variants and a period's daily
returns. Synthetic snapshots only: no network, no real data."""
from __future__ import annotations

import csv
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from lab import um_judge as judge
from lab import um_lab as lab

H, DAY = lab.H, lab.DAY
S = lab.START                                                     # 2022-11-13, a UTC midnight
FOLDS = tuple((S + a * DAY, S + b * DAY) for a, b in ((50, 70), (70, 90), (90, 110)))
PERIODS = judge.periods(S, FOLDS, S + 130 * DAY)
DATES = [start for start, _ in PERIODS]


def write_contract(snapshot, symbol, *, first_day=0, days=130, price=100.0, quote=1e9, lows=None):
    """Hourly bars from day ``first_day`` after S. ``price`` and ``quote`` (daily quote volume) are numbers or
    functions of the day (a float, hour / 24); ``lows`` gives some hours (counted from S) a low as a share of the
    price, to trip a stop."""
    hours = np.arange(first_day * 24, days * 24)
    at = hours / 24
    p = np.array([price(d) for d in at]) if callable(price) else np.full(len(hours), float(price))
    q = np.array([quote(d) for d in at]) / 24 if callable(quote) else np.full(len(hours), quote / 24)
    low = np.array([(lows or {}).get(int(h), 0.99) for h in hours]) * p
    with gzip.open(snapshot / f"{symbol}_1h.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades",
                         "taker_buy_volume", "taker_buy_quote_volume"))
        writer.writerows((S + int(h) * H, a, a * 1.01, b, a, 1e9, c, 10, 5e8, c / 2)
                         for h, a, b, c in zip(hours, p, low, q))
    with gzip.open(snapshot / f"{symbol}_funding.csv.gz", "wt", newline="") as handle:
        handle.write("funding_time,rate\n")


def finish(snapshot, symbols, info=None):
    (snapshot / "manifest.json").write_text(json.dumps({"contracts": [
        {"symbol": s, "type": "perpetual", "margin": "USDT", "files": {"x": {}}} for s in symbols]}))
    (snapshot / "exchange_info.json").write_text(json.dumps(info or {}))
    for cached in (lab.frame, lab.manifest, lab.exchange_info):
        cached.cache_clear()
    return snapshot


# -- periods ---------------------------------------------------------------------------------------------------
def test_the_periods_are_the_history_the_folds_and_the_holdout():
    spans = ((30, 50), (50, 70), (70, 90), (90, 110), (110, 130))
    assert PERIODS == tuple((S + a * DAY, S + b * DAY) for a, b in spans)
    real = [[pd.Timestamp(t, unit="ms").strftime("%Y-%m-%d") for t in p] for p in judge.PERIODS]
    assert [p[0] for p in real] == ["2022-12-13", "2023-11-13", "2024-11-13", "2025-11-13", "2026-05-13"]   # ideas.md
    assert real[-1] == ["2026-05-13", "2026-11-13"]                                  # the holdout is the last


def test_periods_that_overlap_leave_a_gap_or_do_not_start_at_midnight_are_refused():
    with pytest.raises(ValueError, match="follow"):
        judge.periods(S, ((S + 50 * DAY, S + 70 * DAY), (S + 71 * DAY, S + 90 * DAY)), S + 130 * DAY)
    with pytest.raises(ValueError, match="follow"):
        judge.periods(S, ((S + 50 * DAY, S + 70 * DAY), (S + 69 * DAY, S + 90 * DAY)), S + 130 * DAY)
    with pytest.raises(ValueError, match="midnight"):
        judge.periods(S, ((S + 50 * DAY + H, S + 70 * DAY),), S + 130 * DAY)
    with pytest.raises(ValueError, match="empty"):
        judge.periods(S, ((S + 30 * DAY, S + 70 * DAY),), S + 130 * DAY)                # no history period


# -- the universe rule -----------------------------------------------------------------------------------------
INFO = {"BTCUSDT": {"step": "0.001", "min_notional": "100"}, "EEEUSDT": {"step": "1", "min_notional": "5"}}
QUOTES = {"BTCUSDT": 10e9, "USDCUSDT": 9e9, "EEEUSDT": 7e9, "AAAUSDT": 5e9, "BBBUSDT": 4e9, "CCCUSDT": 3e9}


def _universe_snapshot(path, after=None, cut=None):
    """Volumes: BTC > USDC > EEE > NEW (listed on day 35) > AAA > BBB > CCC; DDD is the smallest until day 60 and
    the largest alt after. ``after`` multiplies (price, quote) of every bar from day ``cut`` on, per symbol."""
    def scaled(symbol, value, column):
        factor = (after or {}).get(symbol, (1.0, 1.0))[column]
        return lambda d: value(d) * (factor if cut is not None and d >= cut else 1.0)
    path.mkdir()
    for symbol, quote in QUOTES.items():
        write_contract(path, symbol, price=scaled(symbol, lambda d: 100.0, 0),
                       quote=scaled(symbol, lambda d, q=quote: q, 1))
    write_contract(path, "DDDUSDT", price=scaled("DDDUSDT", lambda d: 100.0, 0),
                   quote=scaled("DDDUSDT", lambda d: 1e9 if d < 60 else 8e9, 1))
    write_contract(path, "NEWUSDT", first_day=35, price=scaled("NEWUSDT", lambda d: 100.0, 0),
                   quote=scaled("NEWUSDT", lambda d: 6e9, 1))
    return finish(path, [*QUOTES, "DDDUSDT", "NEWUSDT"], INFO)


def _lists(snapshot, slots=3, dates=DATES):
    return judge.universe(snapshot, dates, capital="300", slots=slots)


def test_the_universe_is_the_top_by_volume_that_a_slot_can_trade_without_stablecoins(tmp_path):
    lists = _lists(_universe_snapshot(tmp_path / "a"))
    # a slot is 100 USDT, so the smallest order may be 66.67: BTC's minimum is 100, one EEE lot is 100
    assert lists[0] == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    assert lists[1] == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]          # NEW has 15 days of history, 25 are needed
    assert lists[2] == ["NEWUSDT", "AAAUSDT", "BBBUSDT"]          # DDD: 20 of its last 30 days are still small
    assert lists[3] == lists[4] == ["DDDUSDT", "NEWUSDT", "AAAUSDT"]


def test_the_smallest_order_may_be_exactly_two_thirds_of_a_slot(tmp_path):
    snapshot = _universe_snapshot(tmp_path / "a")
    assert _lists(snapshot, slots=2)[0] == ["BTCUSDT", "EEEUSDT"]       # a slot is 150: an order of 100 is allowed
    assert judge.universe(snapshot, DATES, capital="299.99", slots=2)[0] == ["AAAUSDT", "BBBUSDT"]   # a cent less
    assert judge.universe(snapshot, DATES, capital="300", slots=2, stable=())[0] == ["BTCUSDT", "USDCUSDT"]


def test_a_list_uses_only_bars_from_before_its_date(tmp_path):
    base = _lists(_universe_snapshot(tmp_path / "a"))
    # from day 50 on: volumes turned upside down, EEE's price so low that a slot could trade it, BTC unchanged
    later = {"CCCUSDT": (1.0, 1000.0), "BBBUSDT": (1.0, 0.001), "AAAUSDT": (1.0, 0.0001), "EEEUSDT": (0.1, 1e6),
             "DDDUSDT": (3.0, 500.0)}
    changed = _lists(_universe_snapshot(tmp_path / "b", after=later, cut=50))
    assert changed[:2] == base[:2]                                 # the lists of day 30 and day 50: untouched
    assert changed[2] == ["EEEUSDT", "CCCUSDT", "DDDUSDT"]         # the change is real: later lists follow it
    # one bar earlier (the last bar before day 50) and the day-50 list sees it
    assert _lists(_universe_snapshot(tmp_path / "c", after=later, cut=49.95))[1] != base[1]


def test_the_universe_record_states_the_rule_and_the_dates():
    record = judge.universe_record([["AAAUSDT"]] * 5, DATES, capital="300", slots=10)
    assert record["rule"] == ("top 10 by median daily quote volume of the 30 days before each date; smallest order "
                              "<= 20 USDT; no stablecoin contracts")
    assert record["dates"][:2] == ["2022-12-13", "2023-01-02"] and len(record["lists"]) == 5


# -- stop variants ---------------------------------------------------------------------------------------------
def test_the_twelve_stop_variants_of_the_plan_and_the_live_one_first():
    v = judge.STOP_VARIANTS
    assert len(v) == len({x.name for x in v}) == 12
    assert {x.stop_pct for x in v} == {"0.08", "0.12", "0.15"}
    assert {(x.reenter, x.pause_hours) for x in v} == {("pause", 4), ("pause", 24), ("pause", 72),
                                                       ("signal_reset", None)}
    assert v[0] == judge.LIVE_STOP == judge.StopVariant(lab.LIVE.stop_pct, lab.LIVE.pause_hours)
    assert [x.name for x in v[:4]] == ["stop0.08_pause4h", "stop0.08_pause24h", "stop0.08_pause72h",
                                       "stop0.08_signal_reset"]


def test_a_stop_variant_gives_the_replay_its_terms():
    assert judge.StopVariant("0.12", 24).replay_terms() == {"stop_pct": "0.12", "pause_ms": 24 * H,
                                                            "reenter_after_stop": "pause"}
    assert judge.StopVariant("0.15", None, "signal_reset").replay_terms() == {
        "stop_pct": "0.15", "pause_ms": None, "reenter_after_stop": "signal_reset"}
    no_stop = judge.StopVariant("0", None)                                            # a diagnostic, never live
    assert no_stop.name == "nostop" and no_stop.replay_terms()["stop_pct"] == "0"


def test_stop_variants_that_make_no_sense_are_refused():
    for bad in (("0.08", 4, "signal_reset"), ("0", None, "signal_reset"), ("0", 4, "pause"), ("0.08", 4, "wait"),
                ("-0.08", 4, "pause")):
        with pytest.raises(ValueError):
            judge.StopVariant(*bad)


# -- one period's daily returns --------------------------------------------------------------------------------
START, END = PERIODS[1]
D0 = START // DAY


def _curve(values_by_hour, first_hour=1):
    """An equity curve as lab.replay returns it: one value per bar, at the bar's close."""
    return pd.Series(values_by_hour, index=[START + (first_hour + k) * H for k in range(len(values_by_hour))])


def test_a_period_has_one_return_per_day_from_the_fresh_capital_to_its_last_close():
    hours = 20 * 24
    whole = _curve([99.0 + 0.01 * k for k in range(hours)])               # 99 after the first hour: the entry cost
    daily = judge.period_daily([whole], START, END, slots=3, slot_capital=100.0)
    assert list(daily.index) == list(range(D0, D0 + 20))                  # every day of the period, once
    assert daily.iloc[0] == pytest.approx((whole.iloc[23] + 200) / 300 - 1)       # day one, from 3 x 100
    assert (1 + daily).prod() == pytest.approx((whole.iloc[-1] + 200) / 300)      # nothing is left out


def test_the_close_at_midnight_belongs_to_the_day_that_ended():
    values = [100.0] * (20 * 24)
    values[23:] = [110.0] * (len(values) - 23)                            # the bar from 23:00 to midnight gains 10
    daily = judge.period_daily([_curve(values)], START, END, slots=1, slot_capital=100.0)
    assert daily.loc[D0] == pytest.approx(0.10) and daily.loc[D0 + 1] == 0.0


def test_an_idle_slot_a_late_contract_and_a_finished_contract_are_cash_or_their_last_equity():
    late = _curve([100.0, 120.0], first_hour=3 * 24 + 1)                  # trades on day 3 only, then it is gone
    daily = judge.period_daily([late], START, END, slots=2, slot_capital=100.0)
    assert daily.loc[D0 + 3] == pytest.approx(0.10) and daily.drop(D0 + 3).abs().sum() == 0.0
    nothing = judge.period_daily([], START, END, slots=2, slot_capital=100.0)
    assert list(nothing.index) == list(range(D0, D0 + 20)) and nothing.abs().sum() == 0.0


def test_the_days_of_the_2026_11_16_run_drop_each_period_s_first_day():
    """``legacy`` keeps what research/lab/runs/2026-11-16/judge.py did, only to reproduce its numbers: a day ends
    at 23:00 and the period's first day (with the cost of entering) is not in the series."""
    hours = 20 * 24
    whole = _curve([99.0 + 0.01 * k for k in range(hours)])
    daily = judge.period_daily([whole], START, END, slots=3, slot_capital=100.0, legacy=True)
    old = lab.account_daily([whole], slots=3, slot_capital=100.0).reindex(pd.RangeIndex(D0 + 1, D0 + 20)).fillna(0.0)
    assert daily.equals(old) and list(daily.index) == list(range(D0 + 1, D0 + 20))
    assert (1 + daily).prod() == pytest.approx((whole.iloc[-2] + 200) / (whole.iloc[22] + 200))   # 23:00 to 23:00
