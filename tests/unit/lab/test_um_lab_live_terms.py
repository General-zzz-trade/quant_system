"""The lab harness under the live terms (lab/um_lab.py ``LiveTerms``, tech lead 2026-11-16): one strategy capital
split over a fixed symbol list, the live stop and pause, the whole account's returns, the order limits a strategy
needs and the pilot's hard stops. Synthetic data only."""
from __future__ import annotations

import pandas as pd
import pytest

from lab import um_lab as lab
from tests.unit.lab.test_um_lab import snapshot  # noqa: F401  (the synthetic snapshot fixture)

H, DAY = lab.H, lab.DAY


def test_the_account_holds_idle_slots_as_cash_and_keeps_a_finished_contract_s_last_equity():
    first = pd.Series([100.0, 110.0, 121.0, 121.0], index=[0, DAY, 2 * DAY, 3 * DAY])
    late = pd.Series([100.0, 90.0], index=[DAY, 2 * DAY])                 # lists on day 1, delisted after day 2
    daily = lab.account_daily([first, late], slots=3, slot_capital=100.0)  # the third slot never trades
    assert daily.loc[1] == pytest.approx(10 / 300)
    assert daily.loc[2] == pytest.approx((121 + 90 + 100) / 310 - 1)
    assert daily.loc[3] == pytest.approx(0.0)                             # the delisted slot stays at 90


def test_the_limits_a_strategy_needs_count_entries_per_day_and_hour():
    entries = [(0, 10.0), (60_000, 20.0), (2 * H, 5.0), (DAY + 5, 40.0)]
    need = lab.limits_needed(entries)
    assert need == {"entry_orders": 4, "max_orders_per_day": 3, "p95_orders_per_active_day": pytest.approx(2.9),
                    "max_daily_notional": 40.0, "max_order_notional": 40.0, "max_orders_in_one_hour": 2}
    assert lab.limits_needed([]) == {"entry_orders": 0}


def _f(at, side, qty, price, fee="0"):
    return {"at": at, "side": side, "qty": qty, "price": price, "fee": fee, "order": 1}


def test_round_trips_run_from_flat_to_flat_after_fees_and_split_a_fill_through_zero():
    fills = [_f(1, "buy", "2", "100", "0.1"), _f(2, "sell", "1", "110", "0.05"), _f(3, "sell", "1", "90", "0.05"),
             _f(4, "sell", "2", "100", "0.2"), _f(5, "buy", "4", "95", "0.4"), _f(6, "sell", "2", "95", "0")]
    trips = lab.round_trips(fills)
    assert [t[0] for t in trips] == [3, 5, 6]
    assert trips[0][1] == pytest.approx(-200 - 0.1 + 110 - 0.05 + 90 - 0.05)
    assert trips[1][1] == pytest.approx(200 - 0.2 - 2 * 95 - 0.4)          # short 2 closed by half of the buy
    assert trips[2][1] == pytest.approx(-2 * 95 + 2 * 95)                  # the other half (its fee went before)


def test_losing_streaks_count_again_after_each_stop():
    trips = [(1, -1.0), (2, -1.0), (3, -1.0), (4, -1.0), (5, 2.0), (6, -1.0), (7, -1.0), (8, -1.0)]
    assert lab.losing_streaks(trips, 3) == [3, 8]


def _flip(frame, ctx):
    return pd.Series([0.5 if (t // H) % 48 < 24 else -0.5 for t in frame.index], index=frame.index)


def _always(frame, ctx):
    return pd.Series(0.9, index=frame.index)


def test_a_live_run_shares_the_capital_and_reports_the_live_numbers(snapshot):  # noqa: F811
    df = lab.frame("BTCUSDT", snapshot)
    terms = lab.LiveTerms(capital="300", stop_pct="0.08", pause_hours=24)
    out = lab.run_idea(_flip, {}, ["BTCUSDT", "ALTUSDT"], processes=1, snapshot=snapshot, live=terms)
    live = out["live"]
    assert live["slots"] == 2 and live["slot_capital"] == 150.0
    assert live["limits_needed"]["entry_orders"] > 10 and live["round_trips"] > 5
    assert live["terms"]["pause_hours"] == 24
    btc = lab.replay("BTCUSDT", _flip(df, {}), snapshot=snapshot, starting_balance="150", stop_pct="0.08",
                     pause_ms=24 * H)
    assert btc["curve"].iloc[0] == pytest.approx(150, rel=0.01)           # half of the capital, not 5000


def test_a_slot_below_the_exchange_minimum_never_trades(snapshot):  # noqa: F811
    out = lab.run_idea(_always, {}, ["BTCUSDT", "ALTUSDT"], processes=1, snapshot=snapshot,
                       live=lab.LiveTerms(capital="8"))                     # 4 USDT per slot, minimum is 5
    assert out["fills"] == 0 and out["live"]["limits_needed"] == {"entry_orders": 0}


def test_the_default_terms_are_the_tech_lead_s():
    assert (lab.LIVE.capital, lab.LIVE.stop_pct, lab.LIVE.pause_hours) == ("300", "0.08", 4)   # 2026-11-16
