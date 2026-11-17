"""scripts/draft_strategy_spec.py: a lab champion's card -> a schema-2 spec draft (tech lead 2026-11-17, the monthly
re-selection: a new spec is proposed, tracked forward 4 weeks, and published and marked by a person).

- The draft is the judge's final choice (strategy name and parameters), the run's latest symbol list, the lab's
  execution settings, ``allow_short`` when the strategy's targets on that list go below 0, and always ``passed: false``.
  It parses with the runtime's own ``parse_spec``.
- The runtime settings it needs: STOP_PCT, the largest exchange leverage that stop allows (half the isolated
  liquidation distance), the lab's capital and 1x strategy leverage. A stop rule the runtime does not have (a pause
  other than 4 hours, waiting for the signal to reset, a volatility stop, no stop) is listed as a gap.
- Refused: an idea that did not pass the final check (unless asked), one not in the results, one that reads a symbol
  outside the list.
"""
from __future__ import annotations

import json

import pytest

from lab import um_lab as lab
from scripts import draft_strategy_spec as draft
from tests.unit.lab.test_um_judge_runs import market  # noqa: F401  (a fixture)
from trading.domain.strategy.spec import parse_spec

LISTS = [["UPUSDT", "CRASHUSDT"], ["UPUSDT", "DOWNUSDT"]]
FIXED = {"stop_pct": "0.08", "pause_hours": 4, "reenter": "pause"}


def results(name="A1_tsmom_long", params=None, stop=None, passes=True):
    card = {"final_choice": {"params": params or {"n": 20}, "stop": stop or FIXED}, "passes": passes}
    return {"final": {"final_check": {name: card}, "champion": name if passes else None}}


def test_a_champion_becomes_a_schema_2_draft_that_the_runtime_parses(market):  # noqa: F811
    spec, settings, gaps = draft.draft(results(), LISTS, market)
    assert spec == {"schema": 2, "kind": "target_strategy",
                    "strategy": {"name": "A1_tsmom_long", "params": {"n": 20}},
                    "execution": {"band": spec["execution"]["band"], "execution": "market",
                                  "ttl_bars": spec["execution"]["ttl_bars"]},
                    "symbols": ["UPUSDT", "DOWNUSDT"], "allow_short": False, "passed": False}
    parsed = parse_spec(spec)
    assert parsed.strategy_name == "A1_tsmom_long" and parsed.symbols == ("UPUSDT", "DOWNUSDT")
    assert gaps == []
    assert settings == {"STOP_PCT": "0.08", "BINANCE_EXCHANGE_LEVERAGE (largest)": 6, "BINANCE_MARGIN_TYPE": "ISOLATED",
                        "STRATEGY_CAPITAL_USDT": lab.LIVE.capital, "STRATEGY_LEVERAGE": 1}


def test_the_execution_settings_are_the_lab_s(market):  # noqa: F811
    spec, _, _ = draft.draft(results(), LISTS, market)
    from lab.strategy import StrategySpec
    lab_spec = StrategySpec(execution="market")
    assert spec["execution"] == {"band": lab_spec.band, "execution": "market", "ttl_bars": lab_spec.ttl_bars}


def test_a_strategy_that_goes_short_on_the_list_gets_allow_short(market):  # noqa: F811
    spec, _, _ = draft.draft(results("A2_tsmom_ls", {"n": 20}), LISTS, market)
    assert spec["allow_short"] is True and parse_spec(spec).allow_short is True


def test_a_4_5_percent_stop_allows_10x():
    assert draft.largest_leverage("0.045") == 10 and draft.largest_leverage("0.08") == 6
    assert draft.largest_leverage("0.15") == 3


@pytest.mark.parametrize("stop, gap", [
    ({"stop_pct": "0.08", "pause_hours": 24, "reenter": "pause"}, "24"),
    ({"stop_pct": "0.08", "pause_hours": None, "reenter": "signal_reset"}, "signal"),
    ({"stop_pct": "0.045", "pause_hours": 4, "reenter": "pause", "vol_k": "2"}, "vol"),
    ({"stop_pct": "0", "pause_hours": None, "reenter": "pause"}, "no stop")])
def test_a_stop_rule_the_runtime_lacks_is_a_gap(market, stop, gap):  # noqa: F811
    spec, _, gaps = draft.draft(results(stop=stop), LISTS, market)
    assert len(gaps) == 1 and gap in gaps[0] and spec["passed"] is False


def test_an_idea_that_failed_the_final_check_is_refused_unless_asked(market):  # noqa: F811
    with pytest.raises(ValueError, match="no champion"):
        draft.draft(results(passes=False), LISTS, market)                    # nobody passed: name one
    with pytest.raises(ValueError, match="final check"):
        draft.draft(results(passes=False), LISTS, market, idea="A1_tsmom_long")
    spec, _, _ = draft.draft(results(passes=False), LISTS, market, idea="A1_tsmom_long", allow_failed=True)
    assert spec["passed"] is False


def test_an_idea_not_in_the_results_or_reading_outside_the_list_is_refused(market):  # noqa: F811
    with pytest.raises(ValueError, match="not in the final check"):
        draft.draft(results(), LISTS, market, idea="A6_voladj_mom")
    with pytest.raises(ValueError, match="BTCUSDT"):
        draft.draft(results("F1_btc_trend_beta", {"n": 20}), LISTS, market)


def test_main_writes_the_draft_and_its_notes_and_exits_2_on_a_gap(market, tmp_path):  # noqa: F811
    run = tmp_path / "run"
    run.mkdir()
    (run / "results.json").write_text(json.dumps(results(stop={"stop_pct": "0.08", "pause_hours": 24,
                                                                "reenter": "pause"})))
    (run / "universe.json").write_text(json.dumps({"lists": LISTS}))
    out = run / "spec_draft"
    with pytest.raises(SystemExit) as exit_:
        draft.main(["--run", str(run), "--snapshot", str(market), "--out", str(out)])
    assert exit_.value.code == 2
    assert json.loads((out / "final_spec.json").read_text())["passed"] is False
    notes = (out / "README.md").read_text()
    assert "STOP_PCT=0.08" in notes and "24" in notes and "未验证" in notes
    with pytest.raises(FileExistsError):                                 # a draft is written once
        draft.main(["--run", str(run), "--snapshot", str(market), "--out", str(out)])
