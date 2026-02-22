# execution/adapters/binance/listen_key_um.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from execution.adapters.binance.rest import (
    BinanceRestClient,
    BinanceNonRetryableError,
)


@dataclass(slots=True)
class BinanceUmListenKeyClient:
    """
    USDⓈ-M Futures User Data Stream listenKey REST:
    POST /fapi/v1/listenKey
    PUT  /fapi/v1/listenKey
    DELETE /fapi/v1/listenKey
    listenKey 60 分钟有效，PUT 延长 60 分钟。:contentReference[oaicite:4]{index=4}
    """
    rest: BinanceRestClient

    def create(self) -> str:
        res = self.rest.request_api_key(method="POST", path="/fapi/v1/listenKey", params=None)
        lk = res.get("listenKey")
        if not lk:
            raise RuntimeError(f"listenKey missing in response: {res}")
        return str(lk)

    def keepalive(self, listen_key: str) -> str:
        # 2024-04 后 PUT 可能返回 listenKey 字段（官方 changelog）:contentReference[oaicite:5]{index=5}
        res = self.rest.request_api_key(method="PUT", path="/fapi/v1/listenKey", params={"listenKey": listen_key})
        lk = res.get("listenKey") or listen_key
        return str(lk)

    def close(self, listen_key: str) -> None:
        self.rest.request_api_key(method="DELETE", path="/fapi/v1/listenKey", params={"listenKey": listen_key})

    @staticmethod
    def is_listen_key_missing_error(e: BaseException) -> bool:
        # 文档提到 -1125 "This listenKey does not exist." 时应 POST 重建 :contentReference[oaicite:6]{index=6}
        msg = str(e)
        return ("-1125" in msg) or ("listenKey does not exist" in msg) or ("This listenKey does not exist" in msg)
