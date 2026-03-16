# tests/adapters/binance/test_ws_reconnect_replay_contract.py
from __future__ import annotations

import pytest

from execution.adapters.binance.mapper_fill import BinanceFillMapper


class _DupGuard:
    """
    用于契约测试：同 fill_id 重复出现但 payload 不一致 => 数据损坏 => fail-fast
    """
    def __init__(self) -> None:
        self._seen: dict[str, str] = {}

    def check(self, fill_id: str, digest: str) -> None:
        old = self._seen.get(fill_id)
        if old is None:
            self._seen[fill_id] = digest
            return
        if old != digest:
            raise ValueError("duplicate fill_id with mismatched payload")
        # old == digest: ok (idempotent)


def test_ws_reconnect_duplicate_same_trade_is_idempotent() -> None:
    mapper = BinanceFillMapper()
    raw1 = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1700000000000,
        "o": {"s": "BTCUSDT", "S": "BUY", "i": 1, "t": 99, "l": "0.1", "L": "43000", "n": "0", "N": "USDT",
            "T": 1700000000123, "m": False},
    }
    raw2 = {
        # 模拟 WS 重连后补发同一笔成交
        "e": "ORDER_TRADE_UPDATE",
        "E": 1700000000001,
        "o": {"s": "BTCUSDT", "S": "BUY", "i": 1, "t": 99, "l": "0.1", "L": "43000", "n": "0", "N": "USDT",
            "T": 1700000000123, "m": False},
    }

    f1 = mapper.map_fill(raw1)
    f2 = mapper.map_fill(raw2)

    assert f1.fill_id == f2.fill_id
    assert f1.payload_digest == f2.payload_digest

    g = _DupGuard()
    g.check(f1.fill_id, f1.payload_digest)
    g.check(f2.fill_id, f2.payload_digest)  # 不应报错


def test_ws_reconnect_duplicate_same_trade_but_payload_mismatch_must_fail_fast() -> None:
    mapper = BinanceFillMapper()
    ok = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1700000000000,
        "o": {"s": "BTCUSDT", "S": "BUY", "i": 1, "t": 100, "l": "0.1", "L": "43000", "n": "0", "N": "USDT",
            "T": 1700000000123, "m": False},
    }
    bad = {
        # 同 trade_id / fill_id，但 qty 被篡改（或交易所bug/数据损坏）
        "e": "ORDER_TRADE_UPDATE",
        "E": 1700000000001,
        "o": {"s": "BTCUSDT", "S": "BUY", "i": 1, "t": 100, "l": "9.9", "L": "43000", "n": "0", "N": "USDT",
            "T": 1700000000123, "m": False},
    }

    f1 = mapper.map_fill(ok)
    f2 = mapper.map_fill(bad)

    assert f1.fill_id == f2.fill_id
    assert f1.payload_digest != f2.payload_digest  # 关键契约

    g = _DupGuard()
    g.check(f1.fill_id, f1.payload_digest)
    with pytest.raises(ValueError):
        g.check(f2.fill_id, f2.payload_digest)
