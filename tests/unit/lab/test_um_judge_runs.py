"""The lab's judge (lab/um_judge.py): one configuration over periods, the cache and the registry. A synthetic
snapshot: no network, no real data.

The market: UP gains 2 % a day, DOWN loses 0.2 % a day, CRASH is flat with one bar whose low is 10 % down on day 55
(it trips an 8 % stop, not a 12 % one), BTC wanders with a small drift. Three slots of 100 USDT; each period lists
two contracts, so one slot is idle.
"""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from lab import um_judge as judge
from lab import um_lab as lab
from tests.unit.lab.test_um_judge import PERIODS, finish, write_contract

H, DAY = lab.H, lab.DAY
S = lab.START
CRASH_HOUR = 55 * 24 + 15
LISTS = (("UPUSDT", "DOWNUSDT"), ("UPUSDT", "CRASHUSDT"), ("UPUSDT", "DOWNUSDT"), ("UPUSDT", "CRASHUSDT"),
         ("UPUSDT", "DOWNUSDT"))


def always(df, ctx):
    return pd.Series(ctx["params"]["w"], index=df.index)


def trend(df, ctx):
    close = df["close"]
    return (close > close.shift(ctx["params"]["n"] * 24)).astype(float) * 0.9


def short(df, ctx):
    return pd.Series(-ctx["params"]["w"], index=df.index)


IDEAS = {"always": (lab.legacy(always, reads=()), [{"w": 0.5}, {"w": 0.9}]),        # they read no other symbol
         "trend": (lab.legacy(trend, reads=()), [{"n": 3}, {"n": 6}]),
         "short": (lab.legacy(short, reads=()), [{"w": 0.5}])}
FULL = judge.Config("always", {"w": 0.9})


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    path = tmp_path_factory.mktemp("judge_market")
    write_contract(path, "BTCUSDT", price=lambda d: 100 * math.exp(0.0005 * d + 0.02 * math.sin(2.3 * d)))
    write_contract(path, "UPUSDT", price=lambda d: 100 * math.exp(0.02 * d))
    write_contract(path, "DOWNUSDT", price=lambda d: 100 * math.exp(-0.002 * d))
    write_contract(path, "CRASHUSDT", lows={CRASH_HOUR: 0.90})
    yield finish(path, ["BTCUSDT", "UPUSDT", "DOWNUSDT", "CRASHUSDT"])
    lab.frame.cache_clear()


def make(market, tmp_path, **kwargs):
    settings = {"ideas": IDEAS, "lists": LISTS, "cache": tmp_path / "cache", "run_date": "2026-11-17",
                "snapshot": market, "slots": 3, "periods": PERIODS, "trials": tmp_path / "trials.jsonl", **kwargs}
    return judge.Judge(**settings)


def _replays(monkeypatch):
    """Record every replay the judge asks for."""
    calls, real = [], lab.replay

    def spy(symbol, weights, **kwargs):
        calls.append({"symbol": symbol, **kwargs})
        return real(symbol, weights, **kwargs)
    monkeypatch.setattr(lab, "replay", spy)
    return calls


def _rows(j):
    return [json.loads(line) for line in j.trials.read_text().splitlines()] if j.trials.exists() else []


# -- one configuration over periods ----------------------------------------------------------------------------
def test_a_period_starts_flat_with_fresh_slot_capital(market, tmp_path):
    j = make(market, tmp_path)
    assert j.slot_capital == 100.0
    out = j.run_config((FULL, (0, 1)))["periods"]
    start = PERIODS[1][0]
    first = [(at, notional) for at, notional in out[1]["entries"] if at == start + H]    # the first decision
    assert len(first) == 2                                        # both contracts enter at the first decision
    assert all(89.0 < notional <= 90.0 for _, notional in first)  # 0.9 of a fresh 100, not of UP's 144 of period 0
    assert out[0]["trips"] == []                                  # a position open at a period's end is not sold
    assert out[0]["fills"] == 2 and out[0]["stops"] == 0


def test_a_period_s_days_compound_to_the_account_s_value_at_its_last_close(market, tmp_path):
    j = make(market, tmp_path)
    out = j.run_config((FULL, (2,)))["periods"][2]
    start, end = PERIODS[2]
    ends = [lab.replay(s, always(lab.frame(s, market), {"params": FULL.params}), start=start, end=end,
                       snapshot=market, starting_balance="100.0", **judge.LIVE_STOP.replay_terms())["curve"].iloc[-1]
            for s in LISTS[2]]
    assert list(out["daily"].index) == list(range(start // DAY, end // DAY))
    assert (1 + out["daily"]).prod() == pytest.approx((sum(ends) + 100.0) / 300.0, rel=1e-12)
    free = 0.9 * (math.exp(0.02) + math.exp(-0.002) - 2) / 3      # day one's return without costs, held all day
    assert 0.8 * free < out["daily"].iloc[0] < free               # in from the second hour, and the entries cost


def test_the_days_of_consecutive_periods_join_without_overlap_or_gap(market, tmp_path):
    out = make(market, tmp_path).run_config((FULL, (0, 1, 2, 3)))["periods"]
    daily = judge.joined(out, (0, 1, 2, 3))
    assert list(daily.index) == list(range(PERIODS[0][0] // DAY, PERIODS[3][1] // DAY))   # each day exactly once
    assert list(judge.joined(out, (2, 3)).index) == list(range(PERIODS[2][0] // DAY, PERIODS[3][1] // DAY))
    old = make(market, tmp_path, legacy_days=True).run_config((FULL, (0, 1, 2, 3)))["periods"]
    missing = sorted(set(daily.index) - set(judge.joined(old, (0, 1, 2, 3)).index))
    assert missing == [PERIODS[k][0] // DAY for k in range(4)]    # the 2026-11-16 script had no first days


# -- the stop variant ------------------------------------------------------------------------------------------
def test_every_stop_variant_reaches_the_simulator(market, tmp_path, monkeypatch):
    seen, real = [], lab.simulate

    def spy(*args, **kwargs):
        seen.append((kwargs["stop_pct"], kwargs["pause_after_stop_ms"], kwargs["reenter_after_stop"]))
        return real(*args, **kwargs)
    monkeypatch.setattr(lab, "simulate", spy)
    j = make(market, tmp_path)
    for variant in judge.STOP_VARIANTS:
        del seen[:]
        j.run_config((judge.Config("always", {"w": 0.9}, variant), (1,)))
        pause = variant.pause_hours * H if variant.pause_hours else None
        assert seen == [(variant.stop_pct, pause, variant.reenter)] * 2          # UP and CRASH


def test_what_follows_a_stop_changes_the_result(market, tmp_path):
    j = make(market, tmp_path)

    def period(variant):
        return j.run_config((judge.Config("always", {"w": 0.9}, variant), (1,)))["periods"][1]
    paused, waiting = period(judge.StopVariant("0.08", 4)), period(judge.StopVariant("0.08", None, "signal_reset"))
    wide = period(judge.StopVariant("0.12", 4))
    assert (paused["stops"], waiting["stops"], wide["stops"]) == (1, 1, 0)
    stop_at = S + (CRASH_HOUR + 1) * H                            # the close of the bar the stop filled in
    assert [at for at, _ in paused["entries"] if at >= stop_at] == [stop_at + 4 * H]     # CRASH again after 4 hours
    assert [at for at, _ in waiting["entries"] if at >= stop_at] == []                   # the target never left 0.9
    assert len(paused["trips"]) == len(waiting["trips"]) == 1 and wide["trips"] == []


# -- the ideas' context ----------------------------------------------------------------------------------------
def test_a_cross_sectional_idea_gets_each_period_s_list_and_the_others_run_once_per_contract(market, tmp_path):
    calls = []

    def recording(df, ctx):
        btc = ctx["btc"] is not None and len(ctx["btc"]) == len(df)
        calls.append((ctx["symbol"], tuple(ctx["universe"]), ctx["params"]["w"], btc))
        return pd.Series(ctx["params"]["w"], index=df.index)
    ideas = {"xs": (lab.legacy(recording, cross=True), [{"w": 0.5}]), "own": (lab.legacy(recording), [{"w": 0.6}])}
    j = make(market, tmp_path, ideas=ideas)
    j.run_config((judge.Config("xs", {"w": 0.5}), (0, 1, 2)))
    # R7c: a cross-sectional strategy is given the period's list and nothing else (no BTCUSDT frame unasked)
    assert calls == [(s, LISTS[k], 0.5, False) for k in (0, 1) for s in LISTS[k]]   # period 2 has period 0's list
    del calls[:]
    j.run_config((judge.Config("own", {"w": 0.6}), (0, 1, 2)))
    # a per-symbol one is given its symbol and BTCUSDT, which a research function may read
    assert [c[0] for c in calls if c[0] != "BTCUSDT"] == ["UPUSDT", "DOWNUSDT", "CRASHUSDT"]   # UP once for 3 periods
    assert all(c[3] for c in calls)


# -- cache and registry ----------------------------------------------------------------------------------------
def test_run_all_registers_every_period_caches_it_and_only_runs_what_is_missing(market, tmp_path, monkeypatch):
    j = make(market, tmp_path, note="a test")
    calls = _replays(monkeypatch)
    slow = judge.Config("always", {"w": 0.9}, judge.StopVariant("0.12", None, "signal_reset"), cost=2)
    assert j.run_all([(FULL, (0, 1)), (slow, (1,))], processes=1) == 2
    rows = _rows(j)
    assert len(rows) == 3 and len(calls) == 6
    assert rows[0] == {"date": "2026-11-17", "idea": "always", "params": {"w": 0.9}, "stop": "stop0.08_pause4h",
                       "cost_multiplier": 1, "terms": "300 USDT, 3 slots, yearly list, stop0.08_pause4h",
                       "window": ["2022-12-13", "2023-01-02"], "holdout": False, "note": "a test",
                       "total_return": rows[0]["total_return"], "sharpe": rows[0]["sharpe"],
                       "max_drawdown": rows[0]["max_drawdown"], "round_trips": 0, "fills": 2}
    assert rows[0]["total_return"] == pytest.approx((1 + j.load(FULL)[0]["daily"]).prod() - 1)
    assert (rows[2]["stop"], rows[2]["cost_multiplier"], rows[2]["window"]) == (
        "stop0.12_signal_reset", 2, ["2023-01-02", "2023-01-22"])
    assert {c["cost_multiplier"] for c in calls[4:]} == {2} and {c["snapshot"] for c in calls} == {market}

    del calls[:]
    assert j.run_all([(FULL, (0, 1)), (slow, (1,))], processes=1) == 0             # all cached
    assert calls == [] and len(_rows(j)) == 3
    assert j.run_all([(FULL, (0, 1, 2))], processes=1) == 1                        # extended, not rerun
    assert {c["start"] for c in calls} == {PERIODS[2][0]} and len(_rows(j)) == 4
    assert sorted(j.load(FULL)) == [0, 1, 2] and sorted(j.load(slow)) == [1]


def test_two_workers_give_the_same_results_as_one(market, tmp_path):
    jobs = [(c, (0, 1)) for name in ("always", "trend") for c in make(market, tmp_path).candidates(name)]
    one, two = make(market, tmp_path / "one"), make(market, tmp_path / "two")
    assert one.run_all(jobs, processes=1) == two.run_all(jobs, processes=2) == 4
    for config, _ in jobs:
        a, b = one.load(config), two.load(config)
        assert all(a[k]["daily"].equals(b[k]["daily"]) and a[k]["trips"] == b[k]["trips"] for k in (0, 1))
    key = lambda r: (r["idea"], str(r["params"]), r["window"])                     # noqa: E731
    assert sorted(_rows(one), key=key) == sorted(_rows(two), key=key)


def test_a_cache_made_with_other_settings_is_refused(market, tmp_path):
    make(market, tmp_path).run_all([(FULL, (0,))], processes=1)
    for other in ({"slots": 4}, {"terms": lab.LiveTerms(capital="5000")}, {"legacy_days": True},
                  {"lists": (("UPUSDT",),) * 5}):
        j = make(market, tmp_path, **other)
        with pytest.raises(ValueError, match="other settings"):
            j.load(FULL)
        with pytest.raises(ValueError, match="other settings"):
            j.run_all([(FULL, (0, 1))], processes=1)
    assert len(_rows(make(market, tmp_path))) == 1                                 # nothing ran, nothing registered


def test_the_holdout_is_refused_outside_the_final_check(market, tmp_path, monkeypatch):
    j = make(market, tmp_path)
    calls = _replays(monkeypatch)
    assert j.holdout == 4 and list(j.selection) == [0, 1, 2, 3] and list(j.folds) == [1, 2, 3]
    with pytest.raises(ValueError, match="holdout"):
        j.run_all([(FULL, (0, 1)), (FULL, (4,))], processes=1)
    assert calls == [] and _rows(j) == []
    assert j.run_all([(FULL, (4,))], processes=1, holdout=True) == 1
    assert _rows(j)[0]["holdout"] is True and {c["end"] for c in calls} == {PERIODS[4][1]}


def test_the_settings_must_fit_together(market, tmp_path):
    with pytest.raises(ValueError, match="one symbol list per period"):
        make(market, tmp_path, lists=LISTS[:4])
    with pytest.raises(ValueError, match="more symbols than slots"):
        make(market, tmp_path, slots=1)


def test_the_cache_is_json_and_gives_back_exactly_what_was_computed(market, tmp_path):
    j = make(market, tmp_path)
    direct = j.run_config((FULL, (1,)))["periods"][1]
    j.run_all([(FULL, (1,))], processes=1)
    stored = j.load(FULL)[1]
    assert stored["daily"].equals(direct["daily"]) and list(stored["daily"].index) == list(direct["daily"].index)
    assert stored["trips"] == direct["trips"] != [] and stored["entries"] == direct["entries"]
    assert (stored["stops"], stored["fills"]) == (direct["stops"], direct["fills"]) == (1, 4)
    files = list(j.cache.iterdir())                               # no pickle in lab/ (test_artifact_loading)
    assert [f.suffix for f in files] == [".json"] and json.loads(files[0].read_text())["settings"]["slots"] == 3
