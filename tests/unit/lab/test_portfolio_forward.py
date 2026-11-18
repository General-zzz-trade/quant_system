"""lab/portfolio_forward.py: the monthly forward looks of the frozen weak-signal portfolio (tech lead 2026-11-18:
"历史只作检查，前向序贯检验判定", "照做，序贯检验按夏普 1.5 设计"). Synthetic market, a frozen spec and a look
registry in a temporary directory; no network, no real data.

The market: UP gains 2 % a day, DOWN loses 0.2 % a day, BTC wanders with a small drift; 130 days from S. The forward
window starts on day 60 (test setup: the synthetic market lies in 2022-23).
"""
from __future__ import annotations

import json
import math

import pytest

from lab import forward, gates
from lab import portfolio_forward as pf
from lab import um_judge as judge
from lab import um_lab as lab
from lab import um_portfolio as portfolio
from tests.unit.lab.test_um_judge import finish, write_contract
from tests.unit.lab.test_um_judge_runs import IDEAS

DAY = lab.DAY
S = lab.START
START, END = S + 60 * DAY, S + 120 * DAY
STOP = {"stop_pct": "0.08", "pause_hours": 4, "reenter": "pause_then_reset"}
PRICES = {"BTCUSDT": lambda d: 100 * math.exp(0.0005 * d + 0.02 * math.sin(2.3 * d)),
          "UPUSDT": lambda d: 100 * math.exp(0.02 * d), "DOWNUSDT": lambda d: 100 * math.exp(-0.002 * d)}
FROZEN = [{"idea": name, "params": params, "stop": STOP} for name, params in
          (("F1_btc_trend_beta", {"n": 40}), ("E3_vol_target_trend", {"target": 0.02}), ("A1_tsmom_long", {"n": 40}),
           ("C5_trend_not_crowded", {"thr": 0.006}), ("A8_xs_momentum_top3", {"n": 30}))]


def _write(path, *, days=130, first_day=0, later=1.0):
    """``later`` scales every price from the end of the last test look on (day 120)."""
    path.mkdir(exist_ok=True)
    for symbol, price in PRICES.items():
        write_contract(path, symbol, first_day=first_day, days=days,
                       price=lambda d, p=price: p(d) * (later if d >= 120 else 1.0))
    return finish(path, list(PRICES))


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    yield _write(tmp_path_factory.mktemp("portfolio_market"))
    lab.frame.cache_clear()
    lab.snapshot_window.cache_clear()


@pytest.fixture(autouse=True)
def _synthetic_forward_start(monkeypatch):
    monkeypatch.setattr(forward, "FORWARD_START", START)


def _day(t):
    return pf._date(t)


def _setup(tmp_path, members, *, ends=(END, END + 30 * DAY), sd=0.02, symbols=("UPUSDT", "DOWNUSDT"),
           start=START, registered=None, max_looks=None):
    spec = {"study": "weak_test:frozen", "passed": False, "forward_start": _day(start), "members": members,
            "terms": {"capital": "150", "leverage": "2", "symbols": list(symbols)},
            "sequential": {"effect_sharpe": 1.5, "daily_sd": sd, "alpha": 0.05, "beta": 0.2,
                           "max_looks": max_looks or len(ends)}}
    path = tmp_path / "forward_spec.json"
    path.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    looks = tmp_path / "looks.jsonl"
    looks.write_text("".join(json.dumps({"study": "weak_test:frozen", "spec": str(path), "end": _day(e),
                                         "spec_sha256": forward.spec_digest(path),
                                         "registered": registered or _day(START - DAY)}) + "\n" for e in ends))
    return path, looks


def _look(path, looks, data, out, *, end=END, now=None, ideas=None):
    return pf.look(path, end=_day(end), data=data, out=out, looks=looks, ideas=ideas,
                   now_ms=now if now is not None else end + 2 * DAY)


# -- refusals ---------------------------------------------------------------------------------------------------
def test_a_look_that_is_not_registered_premature_or_late_registered_is_refused_before_anything(market, tmp_path):
    path, looks = _setup(tmp_path, FROZEN)
    with pytest.raises(PermissionError):
        _look(path, looks, market, tmp_path / "out", end=END - DAY)                     # not registered
    with pytest.raises(PermissionError):
        _look(path, looks, market, tmp_path / "out", now=END + DAY - 1)                # its end not fully passed
    (tmp_path / "late").mkdir()
    late, late_looks = _setup(tmp_path / "late", FROZEN, registered=_day(END))
    with pytest.raises(PermissionError):
        _look(late, late_looks, market, tmp_path / "out")                              # registered on its end
    path.write_text(path.read_text().replace('"leverage": "2"', '"leverage": "3"'))
    with pytest.raises(PermissionError):
        _look(path, looks, market, tmp_path / "out")                                   # the spec changed
    assert not (tmp_path / "out").exists()


def test_a_spec_not_frozen_or_whose_start_or_number_of_looks_is_not_the_registrys_is_refused(market, tmp_path):
    (tmp_path / "a").mkdir()
    path, looks = _setup(tmp_path / "a", FROZEN, start=START + DAY)
    with pytest.raises(ValueError, match="forward start"):
        _look(path, looks, market, tmp_path / "a" / "out")
    (tmp_path / "b").mkdir()
    path, looks = _setup(tmp_path / "b", FROZEN, max_looks=3)
    with pytest.raises(ValueError, match="2 looks are registered"):
        _look(path, looks, market, tmp_path / "b" / "out")
    (tmp_path / "c").mkdir()
    path, looks = _setup(tmp_path / "c", FROZEN)
    passed = path.read_text().replace('"passed": false', '"passed": true')
    path.write_text(passed)
    looks.write_text(looks.read_text().replace(json.loads(looks.read_text().splitlines()[0])["spec_sha256"],
                                               forward.spec_digest(path)))
    with pytest.raises(ValueError, match="not a frozen portfolio spec"):
        _look(path, looks, market, tmp_path / "c" / "out")


def test_data_that_is_not_whole_or_too_short_to_warm_up_is_refused(tmp_path):
    (tmp_path / "a").mkdir()
    path, looks = _setup(tmp_path / "a", FROZEN)
    old = _write(tmp_path / "old", days=110)                                            # ends before the look's end
    with pytest.raises(ValueError, match="not whole"):
        _look(path, looks, old, tmp_path / "a" / "out")
    short = _write(tmp_path / "short", first_day=55)                                    # 5 days before the start
    with pytest.raises(ValueError, match="not ready"):
        _look(path, looks, short, tmp_path / "a" / "out")
    assert not (tmp_path / "a" / "out").exists()


# -- the look ---------------------------------------------------------------------------------------------------
def test_a_look_replays_the_frozen_members_combines_them_and_is_evaluated_once(market, tmp_path):
    path, looks = _setup(tmp_path, FROZEN)
    result = _look(path, looks, market, tmp_path / "out")
    assert (result["look"], result["of"], result["days"]) == (1, 2, 60)
    assert result["spec_sha256"] == forward.spec_digest(path) and result["start"] == _day(START)
    assert [m["idea"] for m in result["members"]] == [m["idea"] for m in FROZEN]
    assert any(m["round_trips"] for m in result["members"])                          # the members trade
    # equal capital at the start, never rebalanced: the portfolio's growth is the members' mean growth
    growth = sum(1 + m["total_return"] for m in result["members"]) / len(FROZEN)
    assert 1 + result["portfolio"]["total_return"] == pytest.approx(growth, rel=1e-9)
    assert json.loads((tmp_path / "out" / "result.json").read_text()) == result
    assert "第 1/2 次" in (tmp_path / "out" / "README.md").read_text()
    with pytest.raises(FileExistsError):
        _look(path, looks, market, tmp_path / "out")


def test_each_member_is_the_judges_replay_of_its_configuration_over_the_forward_days(market, tmp_path):
    path, looks = _setup(tmp_path, FROZEN[:2])
    result = _look(path, looks, market, tmp_path / "out")
    j = judge.Judge(ideas=lab.lab_ideas(), lists=[["UPUSDT", "DOWNUSDT"]], cache=tmp_path, run_date="x",
                    snapshot=market, terms=lab.LiveTerms(capital="150", leverage="2", symbols=("UPUSDT", "DOWNUSDT")),
                    slots=2, periods=((START, END),), trials=tmp_path / "t.jsonl")
    dailies = [j.run_config((judge.Config(m["idea"], m["params"], judge.StopVariant(**STOP)), (0,)))["periods"][0]
               ["daily"] for m in FROZEN[:2]]
    assert [m["total_return"] for m in result["members"]] == pytest.approx([float((1 + d).prod() - 1) for d in dailies])
    daily = portfolio.combine(dailies)
    sd = 0.02
    expected = gates.sequential(daily.to_numpy(), effect=1.5 / math.sqrt(365) * sd, sd=sd)
    assert result["sequential"]["llr"] == pytest.approx(expected["llr"])
    assert result["sequential"]["decision"] == expected["decision"]


def test_bars_from_the_end_on_change_nothing(market, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    path, looks = _setup(tmp_path / "a", FROZEN)
    first = _look(path, looks, market, tmp_path / "a" / "out")
    other = _write(tmp_path / "other", later=0.5)                                       # halved from day 120 on
    second = _look(path, looks, other, tmp_path / "b" / "out")
    assert {k: v for k, v in first.items() if k != "data"} == {k: v for k, v in second.items() if k != "data"}


# -- the decision -----------------------------------------------------------------------------------------------
def _member(idea, w):
    return {"idea": idea, "params": {"w": w}, "stop": STOP}


def test_a_gaining_portfolio_is_an_effect_and_a_losing_one_no_effect_with_the_day_it_crossed(market, tmp_path):
    for name, member, symbols, decision in (("up", _member("always", 0.9), ("UPUSDT",), "effect"),
                                            ("down", _member("always", 0.9), ("DOWNUSDT",), "no effect")):
        (tmp_path / name).mkdir()
        path, looks = _setup(tmp_path / name, [member], sd=0.002, symbols=symbols)
        result = _look(path, looks, market, tmp_path / name / "out", ideas=IDEAS)
        s = result["sequential"]
        assert result["decision"] == s["decision"] == decision
        assert s["decided_on"] == _day(START + (s["at"] - 1) * DAY)
        assert (s["llr"] >= s["upper"]) if decision == "effect" else (s["llr"] <= s["lower"])


def test_inside_the_bounds_a_look_continues_and_the_last_look_has_no_conclusion(market, tmp_path):
    end = START + 3 * DAY
    for name, ends in (("open", (end, end + DAY)), ("last", (end - DAY, end))):
        (tmp_path / name).mkdir()
        path, looks = _setup(tmp_path / name, [_member("always", 0.5)], ends=ends, sd=0.5)
        result = _look(path, looks, market, tmp_path / name / "out", end=end, ideas=IDEAS)
        assert result["sequential"]["decision"] == "continue" and result["days"] == 3
        assert result["decision"] == ("continue" if name == "open" else "no conclusion")
        assert result["look"] == (1 if name == "open" else 2)
        assert result["sequential"]["decided_on"] is None
