from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, List, Mapping

import pytest

from execution.adapters.binance.user_stream_processor_um import BinanceUmUserStreamProcessor


@dataclass
class _OrderRouter:
    calls: List[Any]
    def ingest_canonical_order(self, order: Any, *, actor=None) -> bool:
        self.calls.append(("order", actor, order))
        return True


@dataclass
class _FillRouter:
    calls: List[Any]
    def ingest_canonical_fill(self, fill: Any, *, actor=None) -> bool:
        self.calls.append(("fill", actor, fill))
        return True


class _OrderMapper:
    def map_um_user_stream_order_trade_update(self, payload: Mapping[str, Any]) -> Any:
        return {"kind": "order", "id": payload["o"].get("i")}


class _FillMapper:
    def map_um_user_stream_order_trade_update(self, payload: Mapping[str, Any]) -> Any:
        return {"kind": "fill", "trade_id": payload["o"].get("t")}


def _mk_trade_event() -> str:
    # 最小 UM ORDER_TRADE_UPDATE（TRADE）
    return json.dumps({
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "x": "TRADE",
            "i": 1001,
            "t": 90001,
            "l": "1",   # last filled qty
        }
    })


def _mk_new_order_event() -> str:
    return json.dumps({
        "e": "ORDER_TRADE_UPDATE",
        "o": {
            "x": "NEW",
            "i": 1002,
            "l": "0",
        }
    })


def test_trade_event_calls_order_and_fill():
    ocalls: List[Any] = []
    fcalls: List[Any] = []

    p = BinanceUmUserStreamProcessor(
        order_router=_OrderRouter(ocalls),
        fill_router=_FillRouter(fcalls),
        order_mapper=_OrderMapper(),
        fill_mapper=_FillMapper(),
        default_actor="venue:binance",
    )

    p.process_raw(_mk_trade_event())

    assert len(ocalls) == 1
    assert len(fcalls) == 1
    assert ocalls[0][0] == "order"
    assert fcalls[0][0] == "fill"


def test_non_trade_event_calls_only_order():
    ocalls: List[Any] = []
    fcalls: List[Any] = []

    p = BinanceUmUserStreamProcessor(
        order_router=_OrderRouter(ocalls),
        fill_router=_FillRouter(fcalls),
        order_mapper=_OrderMapper(),
        fill_mapper=_FillMapper(),
    )

    p.process_raw(_mk_new_order_event())

    assert len(ocalls) == 1
    assert len(fcalls) == 0


def test_invalid_json_fail_fast():
    ocalls: List[Any] = []
    fcalls: List[Any] = []

    p = BinanceUmUserStreamProcessor(
        order_router=_OrderRouter(ocalls),
        fill_router=_FillRouter(fcalls),
        order_mapper=_OrderMapper(),
        fill_mapper=_FillMapper(),
    )

    with pytest.raises(ValueError):
        p.process_raw("{bad json}")
