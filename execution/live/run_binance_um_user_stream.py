from __future__ import annotations

import os
from dotenv import load_dotenv
load_dotenv()

import time

from engine.coordinator import EngineCoordinator, CoordinatorConfig
from execution.ingress.router import FillIngressRouter
from execution.ingress.order_router import OrderIngressRouter

from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig
from execution.adapters.binance.listen_key_um import BinanceUmListenKeyClient
from execution.adapters.binance.listen_key_manager import BinanceUmListenKeyManager, ListenKeyManagerConfig
from execution.adapters.binance.ws_user_stream_um import BinanceUmUserStreamWsClient, UserStreamWsConfig
from execution.adapters.binance.ws_transport_websocket_client import WebsocketClientTransport
from execution.adapters.binance.user_stream_processor_um import BinanceUmUserStreamProcessor
from execution.adapters.binance.mapper_fill import BinanceFillMapper
from execution.adapters.binance.mapper_order import BinanceOrderMapper


class TimeClock:
    def now(self) -> float:
        return time.time()


def main() -> None:
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing env vars: BINANCE_API_KEY / BINANCE_API_SECRET")

    rest_base = os.environ.get("BINANCE_UM_REST", "https://demo-fapi.binance.com")
    ws_base = os.environ.get("BINANCE_UM_WS")

    if not ws_base:
        ws_base = "wss://fstream.binancefuture.com/ws" if "demo-fapi" in rest_base else "wss://fstream.binance.com/ws"

    # testnet: https://demo-fapi.binance.com
    rest = BinanceRestClient(BinanceRestConfig(
        base_url=os.environ.get("BINANCE_UM_REST", "https://demo-fapi.binance.com"),
        api_key=api_key,
        api_secret=api_secret,
        timeout_s=10.0,
    ))

    coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default="BTCUSDT", starting_balance=0.0))
    fill_router = FillIngressRouter(coordinator=coord, default_actor="venue:binance")
    order_router = OrderIngressRouter(coordinator=coord, default_actor="venue:binance")

    processor = BinanceUmUserStreamProcessor(
        order_router=order_router,
        fill_router=fill_router,
        order_mapper=BinanceOrderMapper(),
        fill_mapper=BinanceFillMapper(),
        default_actor="venue:binance",
    )

    lk_client = BinanceUmListenKeyClient(rest=rest)
    lk_mgr = BinanceUmListenKeyManager(
        client=lk_client,
        clock=TimeClock(),
        cfg=ListenKeyManagerConfig(validity_sec=3600, renew_margin_sec=300),
    )

    ws = BinanceUmUserStreamWsClient(
        transport=WebsocketClientTransport(),
        listen_key_mgr=lk_mgr,
        processor=processor,
        cfg=UserStreamWsConfig(
            ws_base_url=os.environ.get("BINANCE_UM_WS", "wss://fstream.binance.com/ws"),
            recv_timeout_s=5.0,
        )
    )

    ws.connect()
    while True:
        ws.step()
        # 你可以在这里把 coord.get_state_view() 打日志或输出关键字段
        # print(coord.get_state_view()["event_index"])
        time.sleep(0.01)


if __name__ == "__main__":
    main()
