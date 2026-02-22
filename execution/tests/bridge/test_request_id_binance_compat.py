from __future__ import annotations

import re

from execution.bridge.request_ids import RequestIdFactory


BINANCE_CLIENT_ID_RE = re.compile(r'^[\.A-Z\:/a-z0-9_-]{1,36}$')


def test_request_id_is_binance_new_client_order_id_compatible():
    rid = RequestIdFactory(namespace="qsys", run_id="run-001", deterministic=True)

    # 注意：这里必须调用“你实际用于 newClientOrderId 的生成函数”
    for i in range(50):
        x = rid.client_order_id(strategy="ema", symbol="BTCUSDT", logical_id=f"sig-{i}")

        assert len(x) <= 36, x
        assert BINANCE_CLIENT_ID_RE.fullmatch(x), x
