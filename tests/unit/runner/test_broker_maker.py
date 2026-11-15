"""Maker-mode accounting of the research broker (``runner/offline/broker.py``) and the funding reader.

Restored 2026-11-15 from the tests deleted with the ML offline replay in M4 (they tested kept code; M4 audit
finding 1). Assertions are unchanged; ``load_funding`` now lives in ``lab.data``.
"""
from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace as NS

import pytest

from data.quality.causal_bars import HOUR, Bar
from lab.data import load_funding
from runner.offline.broker import BarBroker, ReplayConfig


def order(oid="one", qty="1", side="buy", price="100"):
    return NS(order_id=oid, symbol="BTCUSDT", qty=D(qty), side=side, price=D(price))


def broker_config(**kwargs):
    return ReplayConfig(starting_balance="10000", qty_step="0.1", min_notional="1", **kwargs)


def candle(start=0, **kwargs):
    vals = {"open": D(101), "high": D(103), "low": D(98), "close": D(102), "volume": D(100)}
    vals.update({k: D(str(v)) for k, v in kwargs.items()})
    return Bar(start, **vals)


def test_no_same_bar_fill_or_touch_fill():
    broker = BarBroker(broker_config(limit_offset_bps="0"))
    broker.now = HOUR
    broker.send_order(order())
    broker.process_bar(candle(0))
    assert not broker.fills
    broker.process_bar(candle(HOUR, low=100))
    assert not broker.fills  # touching the limit does not establish queue execution
    broker.process_bar(candle(2 * HOUR))
    assert len(broker.fills) == 1
    assert broker.fills[0]["at"] == 3 * HOUR


def test_post_only_cross_rejected_after_gap():
    broker = BarBroker(broker_config(limit_offset_bps="0"))
    broker.send_order(order())
    broker.process_bar(candle(open=99))
    assert not broker.fills
    assert broker.orders[0]["status"] == "rejected"
    assert "post-only" in broker.orders[0]["rejection"]


def test_partial_fill_remainder_expires_and_callback_order():
    broker = BarBroker(broker_config(participation=".001", ttl_bars=2))
    seen = []
    broker.emit = lambda e: seen.append(("fill", str(e.qty)))
    broker.status = lambda oid, status: seen.append(("status", status))
    broker.send_order(order())
    broker.send_order(order())  # idempotent retry
    broker.process_bar(candle())
    broker.process_bar(candle(HOUR))
    assert broker.position.qty == D(".2")
    assert len(broker.orders) == 1 and not broker.pending
    assert seen == [("fill", "0.1"), ("fill", "0.1"), ("status", "canceled")]
    assert broker.orders[0]["remaining"] == "0.8"


def test_signed_funding_realized_pnl_fees_and_partial_close():
    broker = BarBroker(broker_config(limit_offset_bps="0", slippage_bps="0"))
    broker.send_order(order(side="sell", price="102"))
    broker.process_bar(candle(open=101, close=100))
    assert broker.position.qty == -1
    broker.settle_funding(HOUR, D(".001"), D(100))
    assert broker.funding == D(".1")
    close = order("close", qty=".4", side="buy")
    close.reduce_only = True
    broker.send_order(close)
    broker.process_bar(candle(HOUR, open=100, close=100))
    assert broker.position.qty == D("-.6")
    assert broker.position.realized_pnl == D(".8")
    assert broker.fees == D(".0404")
    assert broker.equity == D("10002.0596")


def test_close_cancels_entry_remainder_before_it_can_reopen():
    broker = BarBroker(broker_config(participation=".001", slippage_bps="0"))
    broker.send_order(order())
    broker.process_bar(candle())
    assert broker.position.qty == D(".1")
    broker.send_order(order("stop", qty="0", side="sell"))
    assert broker.orders[0]["status"] == "canceled"
    broker.process_bar(candle(HOUR))
    assert broker.position.qty == 0 and not broker.pending
    assert len(broker.fills) == 2


def test_funding_jitter_is_explicit_and_bounded(tmp_path):
    path = tmp_path / "funding.csv"
    path.write_text("funding_time,rate,mark_price,observed_funding_time\n0,0.0001,100,16\n")
    assert load_funding(path)[0] == (D(".0001"), D(100))
    path.write_text("funding_time,rate,mark_price,observed_funding_time\n0,0.0001,100,1001\n")
    with pytest.raises(ValueError, match="tolerance"):
        load_funding(path)


def test_a_paper_order_keeps_its_reason_and_reduce_only_flag():
    """The paper session's order factory passes every field to the broker's order record (M4 audit finding 2,
    mutant M12b: a dropped reason left paper records unexplained)."""
    from lab.orders import make_order
    broker = BarBroker(broker_config())
    broker.send_order(order())
    broker.send_order(make_order(order_id="exit", source="target.BTCUSDT", symbol="BTCUSDT", side="sell",
                                 qty=D("0.5"), price=D(100), reduce_only=True, reason="kill_switch"))
    row = broker.orders[-1]
    assert (row["reason"], row["reduce_only"], row["side"]) == ("kill_switch", True, "sell")
