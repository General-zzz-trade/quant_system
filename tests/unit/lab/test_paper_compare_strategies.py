"""R7d of the runtime extension: the compare tool rebuilds research with the strategy the paper record names, every
symbol's targets from one call of it, as the runtime decides. A paper run of a registered lab idea - per symbol, and
cross-sectional on BTC and ETH - compares "一致" with research on the same bars; a record naming a strategy the
registry does not know cannot be compared."""
from __future__ import annotations

import json

import pytest

from lab import paper_compare
from lab.strategy import StrategySpec
from runner.binance_target import TargetRuntime
from tests.unit.lab.test_paper_compare import _bars, _document, _save_document, _snapshot
from tests.unit.lab.test_target_runtime import _world

# The world's warm-up is this overlay's window (33 days), longer than the ideas' below need.
WARM = StrategySpec(lookbacks=(20,), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


def _paper(tmp, name, params, live=60):
    with pytest.MonkeyPatch.context() as mp:
        for var in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "RISK_SCOPE"):
            mp.delenv(var, raising=False)
        mp.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
        world = _world(tmp, WARM)
        world["spec_path"].write_text(json.dumps({
            "schema": 2, "kind": "target_strategy", "strategy": {"name": name, "params": params},
            "execution": {"band": 0.05, "execution": "market", "ttl_bars": 1}, "symbols": ["BTCUSDT", "ETHUSDT"],
            "allow_short": True, "passed": False}))
        runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp / "paper", mode="paper",
                                market=world["adapter"], clock=lambda: world["clock"]["now"])
        assert runtime.sessions["BTCUSDT"].strategy.name == name
        runtime.warmup()
        _bars(world, runtime, world["warm"], live)
    _snapshot(world, tmp / "data")
    return tmp / "paper", tmp / "data"


@pytest.mark.parametrize("name,params", [("A1_tsmom_long", {"n": 20}), ("B1_xs_reversal_long", {"days": 1}),
                                         ("F1_btc_trend_beta", {"n": 10})])
def test_a_paper_run_of_a_registered_idea_compares_equal(tmp_path, name, params):
    paper, data = _paper(tmp_path, name, params)
    report = paper_compare.compare(paper, data)
    assert report.verdict == paper_compare.EQUAL, paper_compare.render(report)
    assert {r.symbol for r in report.symbols} == {"BTCUSDT", "ETHUSDT"}
    assert sum(r.matched for r in report.symbols) > 0                       # fills were compared, not none


def test_a_record_naming_an_unknown_strategy_cannot_be_compared(tmp_path):
    paper, data = _paper(tmp_path, "A1_tsmom_long", {"n": 20}, live=3)
    document = _document(paper)
    document["spec"] = {**document["spec"], "strategy": "no_such_strategy"}
    _save_document(paper, document)
    report = paper_compare.compare(paper, data)
    assert report.verdict == paper_compare.CANNOT and "不认识" in report.reason and "no_such_strategy" in report.reason


def test_the_record_s_execution_settings_are_the_research_ones():
    strategy, spec = paper_compare._strategy({"strategy": "A1_tsmom_long", "params": {"n": 20}, "band": 0.07,
                                              "execution": "maker", "ttl_bars": 3})
    assert strategy.name == "A1_tsmom_long" and strategy.params == {"n": 20}
    assert (spec.band, spec.execution, spec.ttl_bars) == (0.07, "maker", 3)
    overlay, spec = paper_compare._strategy(StrategySpec(band=0.07, execution="market").record())
    assert overlay.name == "overlay" and spec.band == 0.07 and overlay.spec == spec
