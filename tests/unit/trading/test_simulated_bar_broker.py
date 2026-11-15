"""The research broker is part of the simulated-exchange adapter (ADR-0013 item 1, design 2026-11 M5 step S10).

Research (``lab.simulate``), the paper sleeve, the closed breakout study and the old import path all use the one
class in trading/adapters/simulated/bar_broker.py, so paper and research fills stay identical by construction.
For the simulated exchange's order gateway the broker gains one public method, ``cancel``.
"""
from __future__ import annotations

import json
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace as NS

import lab.breakout_study as breakout
import lab.simulate as research
import runner.offline.broker as old_path
from lab.strategy import StrategySpec
from trading.adapters.simulated import bar_broker
from trading.adapters.simulated.bar_broker import BarBroker, ReplayConfig
from trading.domain.bars import HOUR, Bar
from trading.domain.orders import LIMIT_OFFSET_BPS
from trading.infrastructure import target

SPEC = StrategySpec(lookbacks=(5, 10, 20), vol_lookback=10, execution="maker")


def order(oid, *, side="buy", qty="1", price="100", reduce_only=False):
    return NS(order_id=oid, symbol="BTCUSDT", side=side, qty=D(qty), price=D(price), reduce_only=reduce_only)


def candle(start=0, **kwargs):
    values = {"open": D(101), "high": D(103), "low": D(98), "close": D(102), "volume": D(1000)}
    values.update({k: D(str(v)) for k, v in kwargs.items()})
    return Bar(start, **values)


def broker():
    out = BarBroker(ReplayConfig(starting_balance="10000", qty_step="0.1", min_notional="1", limit_offset_bps="0"))
    out.seen = []
    out.status = lambda oid, status: out.seen.append((oid, status))
    return out


def test_research_paper_the_breakout_study_and_the_old_path_use_the_adapters_broker():
    rules = research.SYMBOL_RULES["ETHUSDT"]
    sleeve = research.Sleeve("ETHUSDT", SPEC, rules, research.broker_config(SPEC, rules))   # the paper sleeve
    assert type(sleeve.broker) is bar_broker.BarBroker
    assert research.SYMBOL_RULES is bar_broker.SYMBOL_RULES and research.SymbolRules is bar_broker.SymbolRules
    assert research.broker_config is bar_broker.broker_config
    assert breakout.BarBroker is bar_broker.BarBroker and breakout.ReplayConfig is bar_broker.ReplayConfig
    assert (old_path.BarBroker, old_path.ReplayConfig, old_path.BrokerFill) == \
        (bar_broker.BarBroker, bar_broker.ReplayConfig, bar_broker.BrokerFill)
    assert all(getattr(old_path, name) is getattr(bar_broker, name) for name in old_path.__all__)


def test_research_and_execution_rest_a_maker_entry_at_the_same_offset():
    config = research.broker_config(SPEC, research.SYMBOL_RULES["BTCUSDT"])
    assert D(config.limit_offset_bps) == D(ReplayConfig().limit_offset_bps) == LIMIT_OFFSET_BPS
    assert target.LIMIT_OFFSET == LIMIT_OFFSET_BPS / 10000
    simulated = BarBroker(config)
    simulated.send_order(order("entry", qty="0.01", price="60000"))
    assert D(simulated.pending[0]["limit"]) == D("60000") * (1 - target.LIMIT_OFFSET)   # ExecutionService's price


def test_the_closed_breakout_study_keeps_the_broker_settings_it_recorded():
    """Its recorded result must stay reproducible; it now takes the offset from the one definition."""
    recorded = Path(__file__).resolve().parents[3] / "research/results/breakout_2026-11-14/final_spec.json"
    config = breakout.broker_config(research.SYMBOL_RULES[breakout.SYMBOL]).record()
    assert config == json.loads(recorded.read_text())["broker"]


def test_cancel_takes_a_resting_order_off_the_book_for_good():
    b = broker()
    b.send_order(order("keep"))
    b.send_order(order("drop"))
    assert b.cancel("drop") is True
    assert [row["internal_id"] for row in b.pending] == ["keep"]
    dropped = next(row for row in b.orders if row["internal_id"] == "drop")
    assert (dropped["status"], dropped["rejection"]) == ("canceled", "canceled by request")
    assert b.seen == [("drop", "canceled")]
    b.send_order(order("drop"))                     # a retry of the same order does not bring it back
    b.process_bar(candle())                         # trades through both limits (100 > low 98)
    assert [fill["order"] for fill in b.fills] == [b.orders[0]["id"]] and b.position.qty == 1
    assert b.seen == [("drop", "canceled"), ("keep", "filled")]


def test_cancel_takes_a_resting_reduction_off_the_book():
    b = broker()
    b.send_order(order("entry"))
    b.process_bar(candle())
    b.send_order(order("exit", side="sell", qty="0", reduce_only=True))
    assert b.cancel("exit") is True and not b.pending
    b.process_bar(candle(HOUR))
    assert b.position.qty == 1 and len(b.fills) == 1


def test_cancel_changes_nothing_when_the_order_is_not_resting():
    b = broker()
    b.send_order(order("done"))
    b.process_bar(candle())
    before = [dict(row) for row in b.orders]
    assert b.cancel("done") is False and b.cancel("never-sent") is False
    assert b.orders == before and not b.pending
    assert b.seen == [("done", "filled")]
