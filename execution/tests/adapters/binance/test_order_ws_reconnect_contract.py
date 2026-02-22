# execution/tests/adapters/binance/test_order_ws_reconnect_contract.py
from __future__ import annotations

import pytest

from execution.adapters.binance.mapper_order import BinanceOrderMapper


class _DupGuard:
    def __init__(self) -> None:
        self._seen: dict[str, str] = {}

    def check(self, key: str, digest: str) -> None:
        old = self._seen.get(key)
        if old is None:
            self._seen[key] = digest
            return
        if old != digest:
            raise ValueError("duplicate order_key with mismatched payload")


def test_ws_reconnect_duplicate_same_order_snapshot_is_idempotent() -> None:
    mapper = BinanceOrderMapper()

    raw1 = {
        "e": "executionReport",
        "E": 1700000000999,
        "s": "BTCUSDT",
        "S": "BUY",
        "i": 111,
        "c": "cli-001",
        "X": "NEW",
        "o": "LIMIT",
        "f": "GTC",
        "q": "1",
        "p": "43000",
        "z": "0",
        "T": 1700000001001,
    }
    raw2 = dict(raw1)
    raw2["E"] = 1700000001000  # 模拟重连补发/重复推送

    o1 = mapper.map_order(raw1)
    o2 = mapper.map_order(raw2)

    assert o1.order_key == o2.order_key
    assert o1.payload_digest == o2.payload_digest

    g = _DupGuard()
    g.check(o1.order_key, o1.payload_digest)
    g.check(o2.order_key, o2.payload_digest)  # 不应报错


def test_ws_reconnect_duplicate_same_key_but_payload_mismatch_must_fail_fast() -> None:
    mapper = BinanceOrderMapper()

    ok = {
        "e": "executionReport",
        "E": 1700000000999,
        "s": "BTCUSDT",
        "S": "BUY",
        "i": 111,
        "c": "cli-001",
        "X": "NEW",
        "o": "LIMIT",
        "f": "GTC",
        "q": "1",
        "p": "43000",
        "z": "0",
        "T": 1700000001001,
    }
    bad = dict(ok)
    bad["q"] = "9"  # qty 被篡改（同 order_id）

    o1 = mapper.map_order(ok)
    o2 = mapper.map_order(bad)

    assert o1.order_key == o2.order_key
    assert o1.payload_digest != o2.payload_digest

    g = _DupGuard()
    g.check(o1.order_key, o1.payload_digest)
    with pytest.raises(ValueError):
        g.check(o2.order_key, o2.payload_digest)
