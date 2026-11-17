"""The lab's live terms for BTC and ETH only (owner 2026-11-17: "目前如何去改成做BTC和ETH，不再全市场这样做"; the lab's
terms: 150 USDT, the two symbols half each, positions up to 2 times each half).

``lab.um_lab.LIVE_BTC_ETH``: the fixed list is mainnet's (``MAINNET_SYMBOLS``); a replay at leverage 2 holds twice the
weight on isolated margin at the largest exchange leverage the stop allows (the runtime's liquidation rule). The
earlier terms (``LIVE``, 300 USDT over ten slots at 1x) stay as they were, for the runs that used them. Synthetic
data only.
"""
from __future__ import annotations

from decimal import Decimal as D

import pandas as pd
import pytest

from lab import um_judge as judge
from lab import um_lab as lab
from lab import um_monthly as monthly
from tests.unit.lab.test_um_judge import PERIODS
from tests.unit.lab.test_um_judge_runs import always, make, market  # noqa: F401  (a fixture)
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.spec import MAINNET_SYMBOLS


def test_the_terms_are_150_usdt_btc_and_eth_half_each_at_2x():
    terms = lab.LIVE_BTC_ETH
    assert (terms.capital, terms.leverage, terms.symbols) == ("150", "2", ("BTCUSDT", "ETHUSDT"))
    assert set(terms.symbols) == MAINNET_SYMBOLS
    assert (terms.stop_pct, terms.pause_hours, terms.loss_stop, terms.daily_stop) == ("0.08", 4, 0.15, 0.03)


def test_the_earlier_terms_are_unchanged():
    assert (lab.LIVE.capital, lab.LIVE.leverage, lab.LIVE.symbols) == ("300", "1", ())


@pytest.mark.parametrize("stop, lev", [("0.045", 10), ("0.08", 6), ("0.12", 4), ("0.15", 3)])
def test_the_exchange_leverage_is_the_largest_the_runtime_allows_for_the_stop(stop, lev):
    assert lab.exchange_leverage(stop) == lev
    check_stop_inside_liquidation(float(stop), lev, "ISOLATED")                 # the runtime takes it
    with pytest.raises(RuntimeError):
        check_stop_inside_liquidation(float(stop), lev + 1, "ISOLATED")         # and no more


def test_a_replay_at_2x_holds_twice_the_weight(market):  # noqa: F811
    start, end = PERIODS[1]
    weights = pd.Series(0.9, index=lab.frame("BTCUSDT", market).index)
    one = lab.replay("BTCUSDT", weights, start=start, end=end, snapshot=market, starting_balance="75")
    two = lab.replay("BTCUSDT", weights, start=start, end=end, snapshot=market, starting_balance="75", leverage="2",
                     stop_pct="0.08")
    first_one, first_two = one["fills"][0], two["fills"][0]
    notional = lambda f: float(D(f["qty"]) * D(f["price"]))                     # noqa: E731
    assert notional(first_one) == pytest.approx(0.9 * 75, rel=0.02)
    assert notional(first_two) == pytest.approx(1.8 * 75, rel=0.02)
    assert first_two["at"] == first_one["at"]


def test_a_leveraged_replay_needs_a_stop_inside_its_liquidation(market):  # noqa: F811
    weights = pd.Series(0.9, index=lab.frame("BTCUSDT", market).index)
    with pytest.raises(ValueError, match="stop"):
        lab.replay("BTCUSDT", weights, snapshot=market, starting_balance="75", leverage="2", stop_pct="0")


def test_the_live_run_splits_150_usdt_in_two_and_replays_at_2x(market, monkeypatch):  # noqa: F811
    calls, real = [], lab.replay

    def spy(symbol, weights, **kwargs):
        calls.append({"symbol": symbol, **kwargs})
        return real(symbol, weights, **kwargs)
    monkeypatch.setattr(lab, "replay", spy)
    terms = lab.LiveTerms(capital="150", leverage="2", symbols=("BTCUSDT", "UPUSDT"))
    out = lab.run_idea(always, {"w": 0.9}, list(terms.symbols), processes=1, snapshot=market, live=terms)
    assert {c["symbol"] for c in calls} == {"BTCUSDT", "UPUSDT"}
    assert all(c["starting_balance"] == "75" and c["leverage"] == "2" for c in calls)
    assert out["live"]["slots"] == 2 and out["live"]["slot_capital"] == 75.0


def test_the_judge_replays_at_its_terms_leverage(market, tmp_path, monkeypatch):  # noqa: F811
    calls, real = [], lab.replay

    def spy(symbol, weights, **kwargs):
        calls.append(kwargs)
        return real(symbol, weights, **kwargs)
    monkeypatch.setattr(lab, "replay", spy)
    terms = lab.LiveTerms(capital="150", leverage="2", symbols=("UPUSDT", "DOWNUSDT"))
    j = make(market, tmp_path, terms=terms, slots=2, lists=judge.fixed_lists(terms.symbols, PERIODS))
    j.run_config((judge.Config("always", {"w": 0.9}), [1]))
    assert calls and all(c["leverage"] == "2" and c["starting_balance"] == "75.0" for c in calls)


def test_fixed_lists_are_the_same_list_for_every_period():
    assert judge.fixed_lists(("BTCUSDT", "ETHUSDT"), PERIODS) == [["BTCUSDT", "ETHUSDT"]] * len(PERIODS)


def test_the_monthly_re_selection_trades_btc_and_eth_only(tmp_path):
    run = monthly.MonthlyRun.__new__(monthly.MonthlyRun)
    run.snapshot, run.run, run.periods = tmp_path, tmp_path, PERIODS
    lists = run.lists()
    assert lists == [["BTCUSDT", "ETHUSDT"]] * len(PERIODS)
    import json
    record = json.loads((tmp_path / "universe.json").read_text())
    assert record["lists"] == lists and "BTCUSDT and ETHUSDT" in record["rule"]
