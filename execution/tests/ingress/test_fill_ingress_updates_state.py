from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from engine.coordinator import EngineCoordinator, CoordinatorConfig
from execution.ingress.router import FillIngressRouter


def _mk_fill(
    *,
    digest: str = "d1",
    venue: str = "binance",
    symbol: str = "BTCUSDT",
    side: str = "buy",
    qty: float = 1.0,
    price: float = 100.0,
    fee: float = 0.1,
    order_id: str = "o-1",
    fill_id: str = "f-1",
    trade_id: str = "t-1",
) -> SimpleNamespace:
    return SimpleNamespace(
        venue=venue,
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        fee=fee,
        realized_pnl=0.0,
        margin_change=0.0,
        cash_delta=0.0,
        order_id=order_id,
        fill_id=fill_id,           # 幂等 key
        trade_id=trade_id,
        payload_digest=digest,      # mismatch 检测
        ts=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    )


def test_ingress_fill_updates_state_and_is_idempotent() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord, default_actor="venue:binance")

    # first accept -> position qty becomes +1
    assert router.ingest_canonical_fill(_mk_fill()) is True
    st1 = coord.get_state_view()
    pos1 = st1["positions"]["BTCUSDT"]
    assert pos1.qty == Decimal("1")

    # duplicate same key + same digest -> DROP
    assert router.ingest_canonical_fill(_mk_fill()) is False
    st2 = coord.get_state_view()
    pos2 = st2["positions"]["BTCUSDT"]
    assert pos2.qty == pos1.qty

    # duplicate same key + different digest -> FAIL FAST
    with pytest.raises(ValueError):
        router.ingest_canonical_fill(_mk_fill(digest="DIFF", qty=9.0))


def test_ingress_sell_reduces_position() -> None:
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    router = FillIngressRouter(coordinator=coord)

    # buy +1
    assert router.ingest_canonical_fill(_mk_fill(fill_id="f-buy", trade_id="t-buy", digest="b1", qty=1.0)) is True

    # sell 0.4
    assert router.ingest_canonical_fill(_mk_fill(fill_id="f-sell", trade_id="t-sell", digest="s1", side="sell", qty=0.4)) is True

    st = coord.get_state_view()
    pos = st["positions"]["BTCUSDT"]
    assert float(pos.qty) == pytest.approx(0.6)
