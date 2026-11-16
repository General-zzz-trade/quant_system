"""Fixes for the independent audit of R1–R4 (2026-11-16 evening; the findings are in the execution log).

1. Paper reports a symbol still behind a late close on the runtime clock (test_decide_together.py).
2. A schema-2 spec is parsed strictly: a field, a strategy key or an execution key the runtime does not understand,
   a value of the wrong type, or overlay execution settings that differ from the execution block, are refused.
   A schema-1 spec (the pilot's) loads as before.
3./4. The spec's strategy, symbols and shorts are in the start-up config event, and a paper record names its
   strategy (``TargetSpec.identity``): a paper directory does not resume under another strategy, and the compare tool
   refuses a record of a strategy it cannot rebuild.
5. An unknown funding interval still lets a position be reduced; the warm-up asks for funding by the symbol's interval.
7./8. An unknown interval is the symbol's issue (it blocks its entries and is alerted when it lasts) and says when it
   was last read (test_funding_interval.py).
10. Paper names a symbol it has no research lot rules for.
Symbol names of 5 characters (AUSDT, 4USDT: listed on Binance 2026-11-16) are accepted.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from trading.domain.strategy.overlay import StrategySpec
from trading.domain.strategy.spec import parse_spec

ROOT = Path(__file__).resolve().parents[3]
OVERLAY = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market")


def _v2(**over):
    doc = {"schema": 2, "kind": "target_strategy", "strategy": {"name": "overlay", "params": OVERLAY.record()},
           "execution": {"band": 0.05, "execution": "market", "ttl_bars": 1}, "symbols": ["BTCUSDT", "ETHUSDT"],
           "allow_short": False, "passed": False}
    doc.update(over)
    return doc


def _params(**over):
    return {"name": "overlay", "params": {**OVERLAY.record(), **over}}


@pytest.mark.parametrize("over,named", [
    ({"leverage": 50}, "does not understand: leverage"),
    ({"execution": {"band": 0.05, "execution": "market", "ttl_bars": 1, "stop_pct": 0.2}}, "exactly band"),
    ({"strategy": {**_params(), "notes": "x"}}, "only name and params"),
    ({"strategy": _params(long_only="false")}, "long_only"),
    ({"strategy": _params(funding_filter="no")}, "funding_filter"),
    ({"strategy": _params(max_weight=True)}, "max_weight"),
    ({"strategy": _params(vol_lookback=10.5)}, "vol_lookback"),
    ({"strategy": _params(vol_lookback=True)}, "vol_lookback"),
    ({"strategy": _params(lookbacks=[5.5, 10, 20])}, "lookbacks"),
    ({"strategy": _params(lookbacks=[])}, "lookbacks"),
    ({"strategy": _params(unknown_knob=1)}, "no parameter 'unknown_knob'"),
    ({"strategy": _params(execution="maker")}, "differs from the spec's execution block"),
    ({"strategy": _params(band=0.1)}, "differs"),
    ({"execution": {"band": 0.05, "execution": "market", "ttl_bars": 1.9}}, "wrong types"),
    ({"execution": {"band": 0.05, "execution": "market", "ttl_bars": True}}, "wrong types"),
    ({"execution": {"band": "0.05", "execution": "market", "ttl_bars": 1}}, "wrong types"),
    ({"execution": {"band": float("nan"), "execution": "market", "ttl_bars": 1}}, "wrong types"),
    ({"schema": 2.0}, "schema"),
    ({"schema": True}, "schema"),
    ({"passed": "no"}, "passed"),
])
def test_a_schema_2_spec_with_anything_the_runtime_does_not_understand_is_refused(over, named):
    doc = _v2(**over)
    with pytest.raises(ValueError, match=named):
        parse_spec(doc)


def test_a_schema_2_spec_without_passed_is_refused():
    doc = _v2()
    doc.pop("passed")
    with pytest.raises(ValueError, match="passed"):
        parse_spec(doc)


def test_overlay_params_may_leave_the_execution_settings_to_the_block():
    params = {k: v for k, v in OVERLAY.record().items() if k not in ("band", "execution", "ttl_bars")}
    spec = parse_spec(_v2(strategy={"name": "overlay", "params": params},
                          execution={"band": 0.07, "execution": "maker", "ttl_bars": 3}))
    assert (spec.execution.band, spec.execution.execution, spec.execution.ttl_bars) == (0.07, "maker", 3)
    assert spec.identity()["band"] == 0.07 and spec.identity()["execution"] == "maker"


def test_the_pilot_s_schema_1_spec_loads_as_before_and_its_identity_is_its_old_record():
    document = json.loads((ROOT / "research/results/overlay_pilot_2026-11-16/final_spec.json").read_text())
    spec = parse_spec(document)
    assert spec.strategy_name == "overlay" and spec.symbols == ("BTCUSDT", "ETHUSDT")
    assert spec.identity() == StrategySpec.from_record(document["spec"]).record()


def test_a_schema_1_spec_with_a_parameter_the_overlay_does_not_have_is_refused_by_name():
    document = {"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT", "ETHUSDT"],
                "spec": {**OVERLAY.record(), "unknown_knob": 1}, "passed": False}
    with pytest.raises(ValueError, match="refuses its params"):
        parse_spec(document)


@pytest.mark.parametrize("symbol,ok", [("AUSDT", True), ("4USDT", True), ("USDT", False), ("BTCUSDT", True)])
def test_five_character_symbols_are_usdt_perpetuals_too(symbol, ok):
    if ok:
        assert parse_spec(_v2(symbols=[symbol])).symbols == (symbol,)
    else:
        with pytest.raises(ValueError, match="USDT"):
            parse_spec(_v2(symbols=[symbol]))


class Toy:
    """A registered strategy other than the overlay, with plain parameters."""
    name, cross_sectional = "toy", False

    def __init__(self, params):
        self.params = params

    def references(self):
        return ()


@pytest.fixture
def toy(monkeypatch):
    from trading.domain.strategy import spec as spec_module
    from trading.domain.strategy.base import STRATEGIES
    monkeypatch.setitem(STRATEGIES, "toy", Toy)
    monkeypatch.setitem(spec_module.PARAMS, "toy", lambda p: dict(p))


def test_two_strategies_have_different_identities(toy):
    one = parse_spec(_v2(strategy={"name": "toy", "params": {"n": 20}})).identity()
    two = parse_spec(_v2(strategy={"name": "toy", "params": {"n": 40}})).identity()
    overlay = parse_spec(_v2()).identity()
    assert one != two and one == {"strategy": "toy", "params": {"n": 20}, "band": 0.05, "execution": "market",
                                  "ttl_bars": 1}
    assert "strategy" not in overlay


def _world(tmp_path, document):
    from tests.unit.lab.test_target_runtime import _world as world_of
    world = world_of(tmp_path, OVERLAY)
    world["spec_path"].write_text(json.dumps(document))
    return world


def test_the_compare_tool_refuses_a_record_of_another_strategy(tmp_path):
    from lab import paper_compare
    from tests.unit.lab.test_paper_compare import _bars, _document, _run, _save_document
    run = _run(tmp_path, lambda w, r: _bars(w, r, w["warm"], 3))
    document = _document(run["paper"])
    document["spec"] = {"strategy": "A1_tsmom_long", "params": {"n": 20}}
    _save_document(run["paper"], document)
    report = paper_compare.compare(run["paper"], run["data"])
    assert report.verdict == paper_compare.CANNOT and "A1_tsmom_long" in report.reason


def test_paper_names_a_symbol_it_has_no_lot_rules_for():
    from trading.infrastructure.paper import paper_exchange
    with pytest.raises(ValueError, match="SOLUSDT"):
        paper_exchange(OVERLAY, ["BTCUSDT", "SOLUSDT"])


def test_the_warm_up_asks_for_funding_by_the_symbol_s_interval(tmp_path, monkeypatch):
    from runner.binance_target import TargetRuntime
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    world = _world(tmp_path, _v2())
    for fake in world["fakes"].values():
        fake.funding_hours = {"BTCUSDT": 8, "ETHUSDT": 1}
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                            market=world["adapter"], clock=lambda: world["clock"]["now"])
    runtime.warmup()
    runtime.save_paper_state()
    world["clock"]["now"] += 24 * 3600                                 # a day without the runtime: 24 hours missed
    limits = {}
    real = world["adapter"].get_funding_history

    def recording(symbol, limit):
        limits[symbol] = limit
        return real(symbol, limit=limit)
    world["adapter"].get_funding_history = recording
    again = TargetRuntime(spec_path=world["spec_path"], state_dir=tmp_path / "paper", mode="paper",
                          market=world["adapter"], clock=lambda: world["clock"]["now"])
    again.warmup()
    base = OVERLAY.funding_window() + 30 + 1
    assert limits["ETHUSDT"] - base > limits["BTCUSDT"] - base >= 0
    assert limits["ETHUSDT"] - base == 23 or limits["ETHUSDT"] - base == 24     # one settlement an hour missed
