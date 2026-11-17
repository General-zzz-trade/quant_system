"""R2 of the runtime extension (docs/arch/runtime-lab-strategies-plan.md, owner 2026-11-16): the strategy spec names
the strategy, its parameters, its symbols (at most 10) and whether shorts are allowed.

``parse_spec`` turns a signed spec document into one record, ``TargetSpec``:
- schema 1 (``btc_eth_multitimeframe_target``, the pilot's spec) is the overlay strategy; shorts are allowed exactly
  when its ``long_only`` is false; it loads as before;
- schema 2 (``target_strategy``) names a strategy from ``STRATEGIES`` with its parameters, the execution settings
  (rebalancing band, market or maker, maker lifetime), the symbols and ``allow_short``.
Anything it cannot use is refused by name: the runtime never trades a spec it does not fully understand.
"""
from __future__ import annotations

import json

import pytest

from trading.domain.strategy.base import OverlayStrategy
from trading.domain.strategy.overlay import StrategySpec
from trading.domain.strategy.spec import MAX_SYMBOLS, TargetSpec, parse_spec

OVERLAY = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution="market",
                       long_only=True)


def _v1(**over):
    return {"schema": 1, "kind": "btc_eth_multitimeframe_target", "symbols": ["BTCUSDT", "ETHUSDT"],
            "spec": OVERLAY.record(), "passed": False, **over}


def _v2(**over):
    doc = {"schema": 2, "kind": "target_strategy", "strategy": {"name": "overlay", "params": OVERLAY.record()},
           "execution": {"band": 0.05, "execution": "market", "ttl_bars": 1},
           "symbols": ["ETHUSDT", "SOLUSDT", "XRPUSDT"], "allow_short": False, "passed": False}
    doc.update(over)
    return doc


def test_a_schema_1_spec_is_the_overlay_on_its_two_symbols():
    spec = parse_spec(_v1())
    assert isinstance(spec, TargetSpec) and spec.strategy_name == "overlay"
    assert spec.symbols == ("BTCUSDT", "ETHUSDT") and spec.allow_short is False
    assert spec.execution == OVERLAY                                         # band, execution mode, maker lifetime
    strategy = spec.strategy()
    assert isinstance(strategy, OverlayStrategy) and strategy.spec == OVERLAY
    both_ways = parse_spec(_v1(spec={**OVERLAY.record(), "long_only": False}))
    assert both_ways.allow_short is True


def test_a_schema_2_spec_names_its_strategy_symbols_and_shorts():
    spec = parse_spec(_v2())
    assert spec.strategy_name == "overlay" and spec.symbols == ("ETHUSDT", "SOLUSDT", "XRPUSDT")
    assert spec.allow_short is False
    assert (spec.execution.band, spec.execution.execution, spec.execution.ttl_bars) == (0.05, "market", 1)
    assert spec.strategy().spec == OVERLAY
    assert parse_spec(_v2(allow_short=True)).allow_short is True
    assert json.loads(json.dumps(spec.record()))["symbols"] == ["ETHUSDT", "SOLUSDT", "XRPUSDT"]


def test_ten_symbols_are_allowed_and_eleven_are_not():
    ten = [f"S{k}USDT" for k in range(MAX_SYMBOLS)]
    assert MAX_SYMBOLS == 10 and len(parse_spec(_v2(symbols=ten)).symbols) == 10
    with pytest.raises(ValueError, match="at most 10"):
        parse_spec(_v2(symbols=ten + ["S10USDT"]))


@pytest.mark.parametrize("over,named", [
    ({"symbols": []}, "symbol"),
    ({"symbols": ["ETHUSDT", "ETHUSDT"]}, "twice"),
    ({"symbols": ["ethusdt"]}, "ethusdt"),
    ({"symbols": ["ETHUSDC"]}, "ETHUSDC"),
    ({"symbols": ["ETHUSDT_250926"]}, "ETHUSDT_250926"),
    ({"symbols": ["ethUSDT"]}, "ethUSDT"),
    ({"symbols": ["ETH_USDT"]}, "ETH_USDT"),
    ({"symbols": "ETHUSDT"}, "list"),
    ({"strategy": {"name": "momentum_xs", "params": {}}}, "unknown strategy"),
    ({"strategy": {"name": "overlay", "params": {**OVERLAY.record(), "lookbacks": [1]}}}, "lookbacks"),
    ({"strategy": {"name": "overlay"}}, "needs its params"),
    ({"execution": {"band": 0.05, "execution": "twap", "ttl_bars": 1}}, "execution"),
    ({"execution": {"band": 1.5, "execution": "market", "ttl_bars": 1}}, "band"),
    ({"execution": {"band": 0.05, "execution": "market"}}, "ttl_bars"),
    ({"allow_short": "yes"}, "allow_short"),
    ({"allow_short": None}, "allow_short"),
    ({"schema": 3}, "schema"),
    ({"kind": "something_else"}, "kind"),
])
def test_what_cannot_be_used_is_refused_by_name(over, named):
    with pytest.raises(ValueError, match=named):
        parse_spec(_v2(**over))


def test_a_schema_1_spec_with_other_symbols_or_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        parse_spec(_v1(kind="target_strategy"))
    with pytest.raises(ValueError, match="symbol"):
        parse_spec(_v1(symbols=[]))


def test_the_runner_s_loader_accepts_both_schemas_and_refuses_the_rest(tmp_path, monkeypatch):
    import runner.binance_target as entry
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")
    monkeypatch.delenv("QUANT_MODEL_SIGN_KEY", raising=False)
    for name, doc in (("one.json", _v1()), ("two.json", _v2())):
        path = tmp_path / name
        path.write_text(json.dumps(doc))
        assert entry.load_spec(path)["schema"] == doc["schema"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_v2(symbols=[])))
    with pytest.raises(ValueError, match="symbol"):
        entry.load_spec(bad)
