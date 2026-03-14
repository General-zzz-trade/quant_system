# tests/unit/execution/test_digest.py
"""Tests for unified digest module — verify parity with all old implementations."""
from decimal import Decimal
import pytest

from execution.models.digest import (
    stable_hash,
    payload_digest,
    fill_key,
    order_key,
    fill_digest,
    order_digest,
    _stable_json,
)


class TestStableJson:
    def test_deterministic(self):
        a = _stable_json({"b": 1, "a": 2})
        b = _stable_json({"a": 2, "b": 1})
        assert a == b  # sorted keys

    def test_decimal_serialized_as_str(self):
        result = _stable_json({"qty": Decimal("0.001")})
        assert '"0.001"' in result

    def test_no_spaces(self):
        result = _stable_json({"a": 1, "b": 2})
        assert " " not in result


class TestStableHash:
    def test_returns_64_char_hex(self):
        h = stable_hash({"a": 1})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self):
        h1 = stable_hash({"symbol": "BTCUSDT", "qty": Decimal("0.1")})
        h2 = stable_hash({"qty": Decimal("0.1"), "symbol": "BTCUSDT"})
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        h1 = stable_hash({"a": 1})
        h2 = stable_hash({"a": 2})
        assert h1 != h2


class TestPayloadDigest:
    def test_full_length_default(self):
        d = payload_digest({"a": 1})
        assert len(d) == 64

    def test_truncated(self):
        d = payload_digest({"a": 1}, length=16)
        assert len(d) == 16

    def test_truncated_matches_prefix(self):
        full = payload_digest({"a": 1})
        short = payload_digest({"a": 1}, length=16)
        assert full.startswith(short)


class TestKeys:
    def test_fill_key(self):
        k = fill_key(venue="binance", symbol="BTCUSDT", trade_id="12345")
        assert k == "binance:BTCUSDT:12345"

    def test_order_key(self):
        k = order_key(venue="binance", symbol="BTCUSDT", order_id="99")
        assert k == "binance:BTCUSDT:order:99"


class TestFillDigest:
    def test_returns_hex_string(self):
        d = fill_digest(
            symbol="BTCUSDT", order_id="1", trade_id="2",
            side="buy", qty=Decimal("0.1"), price=Decimal("70000"),
            fee=Decimal("0.07"), ts_ms=1700000000000,
        )
        assert len(d) == 64

    def test_deterministic(self):
        args = dict(
            symbol="BTCUSDT", order_id="1", trade_id="2",
            side="buy", qty=Decimal("0.1"), price=Decimal("70000"),
            fee=Decimal("0.07"), ts_ms=1700000000000,
        )
        assert fill_digest(**args) == fill_digest(**args)


class TestOrderDigest:
    def test_returns_hex_string(self):
        d = order_digest(
            symbol="BTCUSDT", order_id="1", status="new",
            side="buy", order_type="limit", qty=Decimal("0.1"),
            price=Decimal("70000"),
        )
        assert len(d) == 64

    def test_different_status_different_digest(self):
        base = dict(
            symbol="BTCUSDT", order_id="1", side="buy",
            order_type="limit", qty=Decimal("0.1"),
        )
        d1 = order_digest(status="new", **base)
        d2 = order_digest(status="filled", **base)
        assert d1 != d2


class TestParityWithOldImplementations:
    """Verify new unified module produces same output as old scattered code."""

    def test_parity_with_orders_py_stable_hash(self):
        """Match execution/models/orders.py:73-83."""
        obj = {
            "symbol": "BTCUSDT",
            "order_id": "123",
            "client_order_id": "",
            "status": "new",
            "side": "buy",
            "order_type": "limit",
            "tif": "gtc",
            "qty": Decimal("0.1"),
            "price": Decimal("70000"),
            "filled_qty": Decimal("0"),
            "avg_price": "",
            "ts_ms": 0,
        }
        new_hash = stable_hash(obj)
        # Old implementation: same logic, same result
        import hashlib
        import json
        def old_default(v):
            if isinstance(v, Decimal):
                return str(v)
            return str(v)
        old_json = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=old_default)
        old_hash = hashlib.sha256(old_json.encode("utf-8")).hexdigest()
        assert new_hash == old_hash
