"""S10 acceptance (added 2026-11-15): with maker entries and with both symbols trading at once, the execution path
sends research's orders bar by bar and holds research's position after every bar.

Paper is the execution path - gates, order journal, lifecycle, fill consumer, stops - on the simulated exchange, so
it is compared here with the research ``Sleeve`` fed the same bars and funding, bar by bar: the intents of each
symbol's decision (side, quantity, reduce-only) and the ledger's position. test_execution_matches_research.py does
the same for one symbol with market entries against the fake Binance venue. The run must see both symbols enter on
one bar (the orders-per-second limit then defers the second entry by 5 s of simulated time, S10 decision 2) and,
with maker entries, entries that rest, fill as maker and expire.
"""
from __future__ import annotations

import json
from decimal import Decimal as D

import pytest

from lab.data import frame_from_bars, funding_series
from lab.simulate import SYMBOL_RULES, Sleeve, broker_config
from lab.strategy import StrategySpec, compute_targets
from runner.binance_target import TargetRuntime
from tests.unit.lab.test_target_runtime import _advance, _world
from trading.domain.bars import bar_from_kline

BARS = 240
SYMBOLS = ("BTCUSDT", "ETHUSDT")


@pytest.fixture(autouse=True)
def _dev_signing(monkeypatch):
    for name in ("QUANT_MODEL_SIGN_KEY", "BINANCE_TESTNET", "RISK_SCOPE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("QUANT_ALLOW_UNSIGNED_MODELS", "1")


def _events(state_dir, seen):
    lines = (state_dir / "target_events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines[seen:]], len(lines)


@pytest.mark.parametrize("execution,ttl", [("maker", 1), ("maker", 4), ("market", 1)])
def test_the_execution_path_sends_research_s_orders_bar_by_bar_for_both_symbols(tmp_path, execution, ttl):
    spec = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, target_vol=0.6, band=0.05, execution=execution,
                        ttl_bars=ttl)
    world = _world(tmp_path, spec)
    state = tmp_path / "paper"
    runtime = TargetRuntime(spec_path=world["spec_path"], state_dir=state, mode="paper", market=world["adapter"],
                            clock=lambda: world["clock"]["now"])
    runtime.warmup()
    funding = {t: (D(str(r)), D("1")) for t, r in world["funding"]}
    bars, targets, research = {}, {}, {}
    for symbol in SYMBOLS:
        bars[symbol] = [bar_from_kline(r) for r in world["rows"][symbol]]
        frame = compute_targets(frame_from_bars(bars[symbol]), funding_series(funding), spec)
        targets[symbol] = dict(zip(frame["available_at"].tolist(), zip(frame["target"].tolist(),
                                                                        frame["ready"].tolist())))
        rules = SYMBOL_RULES[symbol]
        research[symbol] = Sleeve(symbol, spec, rules, broker_config(spec, rules))
    seen = len((state / "target_events.jsonl").read_text().splitlines())
    both_entered = deferred = 0
    for i in range(world["warm"], world["warm"] + BARS):
        expected = {}
        for symbol in SYMBOLS:
            bar = bars[symbol][i]
            research[symbol].on_bar(bar, funding[bar.open_time][0] if bar.open_time in funding else None)
            target, ready = targets[symbol].get(bar.available_at, (0.0, False))
            expected[symbol] = [(o.side, o.qty, o.reduce_only) for o in research[symbol].decide(bar, target, ready)]
        _advance(world, runtime, i)
        events, seen = _events(state, seen)
        for symbol in SYMBOLS:
            prefix = f"{symbol}:{bars[symbol][i].available_at}:"
            sent = [(e["side"], D(e["qty"]), e["reduce_only"]) for e in events
                    if e["kind"] == "intent" and e["order_id"].startswith(prefix)]
            assert sent == expected[symbol], f"{symbol} orders differ at bar {i}"
            position = runtime.sessions[symbol].position_qty()
            assert position == research[symbol].broker.position.qty, f"{symbol} position differs after bar {i}"
        both_entered += all(any(not o[2] for o in expected[s]) for s in SYMBOLS)
        deferred += sum(1 for e in events if e["kind"] == "blocked" and e.get("retry"))
    assert both_entered >= 1 and deferred >= 1           # two entries on one bar: the second waited 5 s
    divergences = [e for e in _events(state, 0)[0] if e["kind"] == "paper_divergence"]
    assert divergences == []                             # nothing the gates or stops did differs from research
    fills = [f for s in SYMBOLS for f in runtime.account.get_recent_fills(symbol=s, limit=10_000)]
    research_fills = [f for s in SYMBOLS for f in research[s].broker.fills]
    assert len(fills) == len(research_fills) > 0
    if execution == "maker":
        assert {f.liquidity for f in fills} == {"maker", "taker"}       # maker entries filled, reductions taker
        expired = [r for s in SYMBOLS for r in research[s].broker.orders
                   if r["status"] == "canceled" and not r["reduce_only"]]
        assert expired, "no maker entry expired unfilled"
