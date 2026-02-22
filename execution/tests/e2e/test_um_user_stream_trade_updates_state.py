from __future__ import annotations

from engine.coordinator import EngineCoordinator, CoordinatorConfig
from execution.adapters.binance.mapper_fill import BinanceFillMapper
from execution.adapters.binance.mapper_order import BinanceOrderMapper
from execution.adapters.binance.user_stream_processor_um import BinanceUmUserStreamProcessor
from execution.ingress.order_router import OrderIngressRouter
from execution.ingress.router import FillIngressRouter


def _mk_um_order_trade_update_trade(*, trade_id: int = 1, client_order_id: str = "CID-1") -> str:
    import json
    E = 1767225600000  # 2026-01-01T00:00:00Z ms
    return json.dumps({
        "e": "ORDER_TRADE_UPDATE",
        "fs": "UM",
        "E": E,
        "T": E,
        "i": "",
        "o": {
            "s": "BTCUSDT",
            "c": client_order_id,
            "S": "BUY",
            "o": "MARKET",
            "f": "GTC",
            "q": "1",
            "p": "0",
            "ap": "100",
            "sp": "0",
            "x": "TRADE",
            "X": "FILLED",
            "i": 8886774,
            "l": "1",
            "z": "1",
            "L": "100",
            "N": "USDT",
            "n": "0.1",
            "T": E,
            "t": trade_id,
            "b": "0",
            "a": "0",
            "m": False,
            "R": False,
            "ps": "LONG",
            "rp": "0",
            "gtd": 0,
        }
    })


def test_trade_update_advances_state_and_is_idempotent():
    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    fill_router = FillIngressRouter(coordinator=coord, default_actor="venue:binance")
    order_router = OrderIngressRouter(coordinator=coord, default_actor="venue:binance")

    p = BinanceUmUserStreamProcessor(
        order_router=order_router,
        fill_router=fill_router,
        order_mapper=BinanceOrderMapper(),
        fill_mapper=BinanceFillMapper(),
        default_actor="venue:binance",
    )

    raw = _mk_um_order_trade_update_trade(trade_id=1, client_order_id="CID-1")

    assert coord.get_state_view()["event_index"] == 0

    # 第一次：应更新
    p.process_raw(raw)
    st1 = coord.get_state_view()
    assert st1["event_index"] >= 1

    pos1 = st1["positions"]["BTCUSDT"]
    # 这里不强行写死 qty 格式（Decimal '1' vs '1.0' 你已处理过）
    assert float(pos1.qty) == 1.0

    # 第二次同 payload：必须幂等（不重复加仓、不重复推进事件）
    idx_before = st1["event_index"]
    p.process_raw(raw)
    st2 = coord.get_state_view()
    pos2 = st2["positions"]["BTCUSDT"]

    assert float(pos2.qty) == 1.0
    assert st2["event_index"] == idx_before
