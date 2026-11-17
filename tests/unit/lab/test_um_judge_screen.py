"""The lab's judge (lab/um_judge.py): the walk-forward choice, the score table, the elimination rule and the final
check, as written in research/lab/runs/2026-11-16/ideas.md. Synthetic data only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lab import um_judge as judge
from lab import um_lab as lab
from tests.unit.lab.test_um_judge import PERIODS
from tests.unit.lab.test_um_judge_runs import IDEAS, always, make, market, short  # noqa: F401  (market: a fixture)

H, DAY = lab.H, lab.DAY
LIVE, WIDE = judge.LIVE_STOP, judge.StopVariant("0.12", 4)
WAIT = judge.StopVariant("0.08", None, "signal_reset")


# -- the walk-forward choice -----------------------------------------------------------------------------------
def _days(k, mean, phase=0.0):
    days = np.arange(PERIODS[k][0] // DAY, PERIODS[k][1] // DAY)
    return pd.Series(mean + 1e-4 * np.sin(days + phase), index=days)


def _run(params, means, stop=LIVE, phase=0.0):
    """A configuration and its periods: a mean daily return per period, k trades and k stops in period k."""
    periods = {k: {"daily": _days(k, m, phase), "trips": [(PERIODS[k][0] + i, 1.0) for i in range(k)],
                   "entries": [(PERIODS[k][0], 10.0 * k)], "stops": k, "fills": 2 * k} for k, m in enumerate(means)}
    return judge.Config("idea", params, stop), periods


EARLY = _run({"p": "early"}, (0.002, -0.002, -0.002, -0.002))
LATE = _run({"p": "late"}, (-0.002, 0.004, 0.004, 0.004), phase=1.0)


def test_each_fold_uses_the_best_sharpe_of_the_periods_before_it():
    wf = judge.walk_forward([EARLY, LATE], folds=(1, 2, 3))
    assert [c.params["p"] for c in wf["chosen"]] == ["early", "late", "late"]
    assert wf["final"].params == {"p": "late"}                    # the best on all four periods
    assert wf["daily"].equals(pd.concat([EARLY[1][1]["daily"], LATE[1][2]["daily"], LATE[1][3]["daily"]]))
    assert wf["daily"].loc[: PERIODS[1][1] // DAY - 1].sum() < 0  # the first fold lost, honestly
    assert list(wf["daily"].index) == list(range(PERIODS[1][0] // DAY, PERIODS[3][1] // DAY))
    assert len(wf["trips"]) == 1 + 2 + 3 and wf["stops"] == 6 and wf["entries"] == [
        (PERIODS[1][0], 10.0), (PERIODS[2][0], 20.0), (PERIODS[3][0], 30.0)]


def test_nothing_from_a_fold_or_after_it_changes_the_choice_for_that_fold():
    base = judge.walk_forward([EARLY, LATE], folds=(1, 2, 3))["chosen"]
    for fold in (1, 2, 3):
        def altered(run, means, fold=fold):
            config, periods = run
            later = _run(config.params, means)[1]
            return config, {k: (later[k] if k >= fold else periods[k]) for k in periods}
        # from the fold on, the two swap roles and far more strongly
        swapped = [altered(EARLY, (0.05,) * 4), altered(LATE, (-0.05,) * 4)]
        assert judge.walk_forward(swapped, folds=(1, 2, 3))["chosen"][fold - 1] == base[fold - 1]
    # and the data before a fold does choose: make period 0 bad for "early" and fold 1 changes hands
    flipped = [_run({"p": "early"}, (-0.01, -0.002, -0.002, -0.002)), LATE]
    assert judge.walk_forward(flipped, folds=(1, 2, 3))["chosen"][0].params == {"p": "late"}


def test_the_final_choice_uses_every_selection_period_the_last_fold_included():
    steady = _run({"p": "steady"}, (0.001, 0.001, 0.001, 0.001))
    broke = _run({"p": "broke"}, (0.004, 0.004, 0.004, -0.03), phase=1.0)         # the best until the last fold
    wf = judge.walk_forward([steady, broke], folds=(1, 2, 3))
    assert [c.params["p"] for c in wf["chosen"]] == ["broke", "broke", "broke"]   # the last fold could not know
    assert wf["final"].params == {"p": "steady"}                                  # what the holdout would get


def test_ties_go_to_the_larger_parameter_text_then_to_the_variant_listed_first():
    means = (0.001, 0.001, 0.001, 0.001)
    same = [_run({"n": 20}, means, LIVE), _run({"n": 20}, means, WIDE), _run({"n": 40}, means, WIDE),
            _run({"n": 40}, means, WAIT)]
    wf = judge.walk_forward(same, folds=(1, 2, 3))
    assert all(c.params == {"n": 40} and c.stop == WIDE for c in [*wf["chosen"], wf["final"]])
    assert judge.walk_forward(same[:2], folds=(1, 2, 3))["final"].stop == LIVE


def test_a_configuration_that_never_traded_before_a_fold_can_still_be_chosen_last():
    idle = _run({"p": "idle"}, (0.0, 0.01, 0.01, 0.01))
    idle[1][0]["daily"] = idle[1][0]["daily"] * 0.0               # no return at all in period 0: Sharpe 0
    loser = _run({"p": "loser"}, (-0.002, 0.0, 0.0, 0.0))
    assert judge.walk_forward([idle, loser], folds=(1, 2, 3))["chosen"][0].params == {"p": "idle"}


# -- the score table (ideas.md) --------------------------------------------------------------------------------
def _score(cagr=0.3, drawdown=0.2, trips=120, streak=4, bear=0.0, chop=0.0):
    money = [-1.0] * streak + [1.0] * (trips - streak)
    return judge.score({"cagr": cagr, "max_drawdown": drawdown}, [(i, m) for i, m in enumerate(money)],
                       {"bear": bear, "chop": chop})


@pytest.mark.parametrize("cagr,points", [(-0.2, 1), (0.0, 1), (1e-9, 2), (0.10, 2), (0.1001, 3), (0.25, 3),
                                         (0.2501, 4), (0.50, 4), (0.5001, 5)])
def test_score_return_bands(cagr, points):
    assert _score(cagr=cagr)["return"] == points


@pytest.mark.parametrize("drawdown,points", [(0.0, 5), (0.1499, 5), (0.15, 4), (0.25, 4), (0.2501, 3), (0.35, 3),
                                             (0.3501, 2), (0.50, 2), (0.5001, 1)])
def test_score_drawdown_bands(drawdown, points):
    assert _score(drawdown=drawdown)["drawdown"] == points        # "< 15 %" is 5, so 15 % itself is 4


@pytest.mark.parametrize("trips,points", [(0, 1), (29, 1), (30, 2), (49, 2), (50, 3), (99, 3), (100, 4), (199, 4),
                                          (200, 5)])
def test_score_trade_count_bands(trips, points):
    assert _score(trips=trips, streak=0)["trades"] == points


@pytest.mark.parametrize("streak,points", [(0, 5), (3, 5), (4, 4), (5, 4), (6, 3), (7, 3), (8, 2), (10, 2), (11, 1)])
def test_score_losing_streak_bands(streak, points):
    assert _score(streak=streak)["losing_streak"] == points


@pytest.mark.parametrize("bear,chop,points", [(-0.01, -0.01, 1), (-0.10, 0.05, 2), (0.05, -0.0501, 2),
                                              (-0.05, 0.05, 3), (0.2, -0.1, 3), (0.0, 0.0, 4), (0.0499, 0.3, 4),
                                              (0.05, 0.05, 5), (0.3, 0.06, 5)])
def test_score_bear_and_chop_bands(bear, chop, points):
    assert _score(bear=bear, chop=chop)["bear_and_chop"] == points


def test_the_score_is_the_sum_of_its_five_parts_and_the_streak_is_the_longest_run_of_losses():
    s = _score(cagr=0.6, drawdown=0.1, trips=200, streak=3, bear=0.1, chop=0.1)
    assert s == {"return": 5, "drawdown": 5, "trades": 5, "losing_streak": 5, "bear_and_chop": 5, "total": 25}
    assert _score(cagr=-1, drawdown=0.9, trips=11, streak=11, bear=-1, chop=-1)["total"] == 5
    trips = [(5, -1.0), (1, -1.0), (2, -1.0), (3, 1.0), (4, -1.0), (6, 0.0), (7, -1.0)]    # by time: L L W L L 0 L
    assert judge.longest_losing(trips) == 2


# -- elimination and the final check ---------------------------------------------------------------------------
def test_an_idea_survives_only_with_a_positive_return_and_a_sharpe_above_holding_btc():
    hold = {"sharpe": 0.88}
    assert judge.survives({"total_return": 0.01, "sharpe": 0.89}, hold)
    assert not judge.survives({"total_return": 0.0, "sharpe": 2.0}, hold)          # a return of 0 is out
    assert not judge.survives({"total_return": 0.5, "sharpe": 0.88}, hold)         # "not above" is out
    assert not judge.survives({"days": 0}, hold)                                   # never traded


def _checks(**changes):
    passing = {"holdout": {"total_return": 0.02, "sharpe": 1.0}, "holdout_hold": {"sharpe": 0.5}, "round_trips": 50,
               "doubled": {"total_return": 0.01, "sharpe": 0.9}, "walk_forward_hold": {"sharpe": 0.88},
               "selection_sharpe": {"a": 0.1, "b": 2.0}, "lookahead": []}
    return judge.final_checks(**{**passing, **changes})


def test_the_final_check_passes_only_when_all_five_hold():
    assert _checks() == {"unseen_data": True, "at_least_50_trades": True, "double_fees": True,
                         "neighbouring_settings": True, "no_lookahead": True, "runs_as_judged": True}
    assert not _checks(runs_as_judged=False)["runs_as_judged"]                     # R7 audit F9
    assert not _checks(holdout={"total_return": 0.0, "sharpe": 1.0})["unseen_data"]
    assert not _checks(holdout={"total_return": 0.02, "sharpe": 0.5})["unseen_data"]
    assert not _checks(holdout={"days": 0})["unseen_data"]
    assert not _checks(round_trips=49)["at_least_50_trades"]
    assert not _checks(doubled={"total_return": 0.0, "sharpe": 0.9})["double_fees"]
    assert not _checks(doubled={"total_return": 0.01, "sharpe": 0.88})["double_fees"]
    assert not _checks(selection_sharpe={"a": 0.0, "b": 2.0})["neighbouring_settings"]
    assert not _checks(lookahead=["2024-03-05 changes output"])["no_lookahead"]
    assert not _checks(lookahead=None)["no_lookahead"]                              # not checked is not passed


# -- a whole run on the synthetic market -----------------------------------------------------------------------
def pulse(df, ctx):
    """Long for the first ``hours`` of every UTC day: one round trip a day per contract."""
    return pd.Series(((df.index // H) % 24 < ctx["params"]["hours"]) * 0.9, index=df.index)


RUN_IDEAS = {"pulse": (lab.legacy(pulse, reads=()), [{"hours": 12}, {"hours": 8}]), "always": IDEAS["always"],
             "short": IDEAS["short"]}


def _run_judge(market, tmp_path, **kwargs):  # noqa: F811
    return make(market, tmp_path, ideas=RUN_IDEAS, lookahead=lambda name, params: [], **kwargs)


def _replay_ends(monkeypatch):
    ends, real = [], lab.replay

    def spy(symbol, weights, **kwargs):
        ends.append((kwargs["end"], kwargs["cost_multiplier"]))
        return real(symbol, weights, **kwargs)
    monkeypatch.setattr(lab, "replay", spy)
    return ends


def test_the_screen_walks_forward_scores_and_eliminates_without_reading_the_holdout(market, tmp_path, monkeypatch):  # noqa: F811,E501
    j = _run_judge(market, tmp_path)
    ends = _replay_ends(monkeypatch)
    screened = j.screen(variants=(LIVE, WIDE), processes=1)
    assert max(end for end, _ in ends) == PERIODS[3][1] and {c for _, c in ends} == {1}
    rows = [json.loads(line) for line in j.trials.read_text().splitlines()]
    assert len(rows) == (2 + 2 + 1) * 2 * 4 and not any(r["holdout"] for r in rows)   # candidates x periods
    assert {r["stop"] for r in rows} == {"stop0.08_pause4h", "stop0.12_pause4h"}

    hold, ideas = screened["buy_and_hold_btc"], screened["ideas"]
    assert hold["days"] == ideas["pulse"]["walk_forward"]["days"] == 60               # the same days as holding BTC
    assert [ideas[n]["survives"] for n in ("pulse", "always", "short")] == [True, True, False]
    assert ideas["short"]["walk_forward"]["total_return"] < 0
    pulse_row = ideas["pulse"]
    assert pulse_row["round_trips"] == 2 * 60 and pulse_row["score"]["trades"] == 4
    assert pulse_row["score"]["total"] == sum(v for k, v in pulse_row["score"].items() if k != "total")
    assert len(pulse_row["selection_sharpe"]) == 4 and len(pulse_row["chosen_per_fold"]) == 3
    for config in j.candidates("pulse", (LIVE, WIDE)):            # the Sharpe of all four selection periods
        whole = lab.summary(judge.joined(j.load(config), (0, 1, 2, 3)))["sharpe"]
        assert pulse_row["selection_sharpe"][config.key] == whole
    assert set(pulse_row["pilot"]) == {"losing_streak_stops", "per_30_days", "days_below_daily_stop",
                                       "loss_stop_share"}
    assert pulse_row["limits_needed"]["entry_orders"] == 2 * 60
    json.dumps(screened)                                                              # a run writes it to a file


def test_the_stop_variant_is_chosen_fold_by_fold_like_a_parameter(market, tmp_path):  # noqa: F811
    j = make(market, tmp_path)
    row = j.screen(["always"], variants=(LIVE, WIDE, WAIT), processes=1)["ideas"]["always"]
    # period 0 has no stop, so the variants tie there: fold 1 takes the one listed first
    assert row["chosen_per_fold"][0]["stop"] == {"stop_pct": "0.08", "pause_hours": 4, "reenter": "pause"}
    for fold, chosen in zip((1, 2, 3), row["chosen_per_fold"]):
        def before(config, fold=fold):
            return lab.summary(judge.joined(j.load(config), range(fold)))["sharpe"]
        best = max(before(c) for c in j.candidates("always", (LIVE, WIDE, WAIT)))
        assert before(judge.Config("always", chosen["params"], judge.StopVariant(**chosen["stop"]))) == best
    # CRASH's 10 % bar in period 1 trips the 8 % stop: from fold 2 on the 12 % stop is the better history
    assert [c["stop"]["stop_pct"] for c in row["chosen_per_fold"][1:]] == ["0.12", "0.12"]
    assert row["exchange_stops"] == 1                              # fold 1 ran with the 8 % stop


def test_the_final_check_reads_the_holdout_for_survivors_only_and_names_the_champion(market, tmp_path, monkeypatch):  # noqa: F811,E501
    j = _run_judge(market, tmp_path)
    screened = j.screen(processes=1)
    ends = _replay_ends(monkeypatch)
    out = j.final(screened, processes=1)
    rows = [json.loads(line) for line in j.trials.read_text().splitlines()]
    assert {r["idea"] for r in rows if r["holdout"]} == {"pulse", "always"}           # not the eliminated idea
    assert sum(r["holdout"] for r in rows) == 2                                       # one configuration each
    assert {end for end, c in ends if c == 2} == {PERIODS[k][1] for k in (1, 2, 3)}   # double fees: the folds only
    assert {c for end, c in ends if end == PERIODS[4][1]} == {1}

    assert set(out["final_check"]) == {"pulse", "always"}
    assert out["final_check"]["pulse"]["checks"] == {"unseen_data": True, "at_least_50_trades": True,
                                                     "double_fees": True, "neighbouring_settings": True,
                                                     "no_lookahead": True, "runs_as_judged": True}
    assert out["final_check"]["pulse"]["final_choice"] == screened["ideas"]["pulse"]["final_choice"]
    assert out["final_check"]["pulse"]["holdout"]["days"] == 20
    doubled = out["final_check"]["pulse"]["double_fees_walk_forward"]["total_return"]
    assert 0 < doubled < screened["ideas"]["pulse"]["walk_forward"]["total_return"]
    assert out["final_check"]["always"]["checks"]["at_least_50_trades"] is False
    assert out["ranking"] == ["pulse", "always"] and out["champion"] == "pulse"
    assert out["buy_and_hold"]["holdout_btc"]["days"] == 20
    json.dumps(out)


def test_without_a_lookahead_check_or_with_a_violation_there_is_no_champion(market, tmp_path):  # noqa: F811
    j = make(market, tmp_path, ideas=RUN_IDEAS)                                       # no check given
    out = j.final(j.screen(["pulse"], processes=1), processes=1)
    assert out["final_check"]["pulse"]["checks"]["no_lookahead"] is False and out["champion"] is None
    assert out["final_check"]["pulse"]["lookahead_violations"] == ["not checked"]
    seen = []

    def check(name, params):
        seen.append((name, params))
        return ["2023-01-05 changes output"]
    out = make(market, tmp_path, ideas=RUN_IDEAS, lookahead=check).final(j.screen(["pulse"], processes=1),
                                                                         processes=1)
    assert seen == [("pulse", out["final_check"]["pulse"]["final_choice"]["params"])]
    assert out["champion"] is None and out["final_check"]["pulse"]["passes"] is False


def test_with_no_survivor_the_holdout_stays_closed(market, tmp_path, monkeypatch):  # noqa: F811
    j = _run_judge(market, tmp_path)
    screened = j.screen(["short"], processes=1)
    ends = _replay_ends(monkeypatch)
    windows, real = [], lab.buy_and_hold
    monkeypatch.setattr(lab, "buy_and_hold", lambda *a, **k: windows.append(a) or real(*a, **k))
    out = j.final(screened, processes=1)
    assert ends == [] and windows == []
    assert out == {"buy_and_hold": None, "final_check": {}, "ranking": [], "champion": None}


def test_an_idea_that_reads_a_symbol_outside_its_lists_does_not_run_as_judged(market, tmp_path):  # noqa: F811
    """R7 audit F9: the runtime trades every symbol a strategy reads, so an idea judged reading BTCUSDT outside the
    lists cannot run as it was judged: the final check fails it, whatever its numbers."""
    reading = {**RUN_IDEAS, "pulse": (lab.legacy(pulse), RUN_IDEAS["pulse"][1])}          # reads BTCUSDT by default
    j = make(market, tmp_path, ideas=reading, lookahead=lambda name, params: [])
    out = j.final(j.screen(["pulse"], processes=1), processes=1)
    assert out["final_check"]["pulse"]["checks"]["runs_as_judged"] is False
    assert out["final_check"]["pulse"]["passes"] is False and out["champion"] is None
