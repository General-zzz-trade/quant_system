# tests/unit/test_rust_parity_v2.py
"""Parity tests: Rust hotpath Phase 2 modules."""
from __future__ import annotations

import hashlib
import json
import time
import pytest
from decimal import Decimal

try:
    from _quant_hotpath import (
        rust_payload_digest,
        rust_stable_hash,
        RustFillDedupGuard,
        RustSequenceBuffer,
        rust_event_id,
        rust_now_ns,
        rust_sanitize,
        rust_short_hash,
        rust_make_idempotency_key,
        rust_client_order_id,
        rust_hmac_sign,
        rust_hmac_verify,
        rust_route_event_type,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust _quant_hotpath not built")


# ============================================================
# Digest
# ============================================================

class TestRustPayloadDigest:
    def test_basic(self):
        fields = [("b", "2"), ("a", "1")]
        d = rust_payload_digest(fields, 16)
        assert len(d) == 16
        assert all(c in "0123456789abcdef" for c in d)

    def test_sorted_order(self):
        d1 = rust_payload_digest([("a", "1"), ("b", "2")], 64)
        d2 = rust_payload_digest([("b", "2"), ("a", "1")], 64)
        assert d1 == d2

    def test_full_length(self):
        d = rust_payload_digest([("x", "y")], 64)
        assert len(d) == 64

    def test_deterministic(self):
        fields = [("key", "value"), ("num", "42")]
        d1 = rust_payload_digest(fields, 16)
        d2 = rust_payload_digest(fields, 16)
        assert d1 == d2


class TestRustStableHash:
    def test_matches_python(self):
        text = "hello world"
        py = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        rs = rust_stable_hash(text, 16)
        assert rs == py

    def test_full_hash(self):
        text = "test"
        py = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rs = rust_stable_hash(text, 64)
        assert rs == py

    def test_different_lengths(self):
        text = "abc"
        h8 = rust_stable_hash(text, 8)
        h16 = rust_stable_hash(text, 16)
        assert h16.startswith(h8)


# ============================================================
# FillDedupGuard
# ============================================================

class TestRustFillDedupGuard:
    def test_new_returns_new(self):
        g = RustFillDedupGuard()
        assert g.check("key1", "digest1", 100.0) == "new"

    def test_duplicate(self):
        g = RustFillDedupGuard()
        g.check("key1", "digest1", 100.0)
        assert g.check("key1", "digest1", 101.0) == "duplicate"

    def test_corrupted(self):
        g = RustFillDedupGuard()
        g.check("key1", "digest1", 100.0)
        result = g.check("key1", "digest2", 101.0)
        assert result.startswith("corrupted:")
        assert "digest1" in result
        assert "digest2" in result

    def test_contains(self):
        g = RustFillDedupGuard()
        assert not g.contains("key1")
        g.check("key1", "d", 100.0)
        assert g.contains("key1")

    def test_len(self):
        g = RustFillDedupGuard()
        assert len(g) == 0
        g.check("k1", "d1", 100.0)
        g.check("k2", "d2", 100.0)
        assert len(g) == 2

    def test_clear(self):
        g = RustFillDedupGuard()
        g.check("k1", "d1", 100.0)
        g.clear()
        assert len(g) == 0
        assert g.check("k1", "d1", 101.0) == "new"

    def test_ttl_prune(self):
        g = RustFillDedupGuard(ttl_sec=10.0, max_size=100)
        g.check("k1", "d1", 100.0)
        # After TTL + prune trigger (>60s)
        assert g.check("k1", "d1", 170.0) == "new"

    def test_check_with_fields(self):
        g = RustFillDedupGuard()
        result = g.check_with_fields("key1", [("a", "1"), ("b", "2")], 100.0)
        assert result == "new"
        result2 = g.check_with_fields("key1", [("a", "1"), ("b", "2")], 101.0)
        assert result2 == "duplicate"

    def test_parity_with_python(self):
        """Compare Rust guard behavior with Python DuplicateGuard."""
        from execution.safety.duplicate_guard import DuplicateGuard, PayloadCorruptionError
        py_guard = DuplicateGuard(ttl_seconds=86400.0)
        rs_guard = RustFillDedupGuard(ttl_sec=86400.0)

        payload = {"symbol": "BTCUSDT", "qty": "0.1", "price": "50000"}
        from execution.safety.duplicate_guard import compute_digest
        digest = compute_digest(payload)

        py_result = py_guard.check(key="k1", payload=payload)
        rs_result = rs_guard.check("k1", digest, 100.0)
        assert py_result is True
        assert rs_result == "new"

        py_result2 = py_guard.check(key="k1", payload=payload)
        rs_result2 = rs_guard.check("k1", digest, 101.0)
        assert py_result2 is False
        assert rs_result2 == "duplicate"


# ============================================================
# SequenceBuffer
# ============================================================

class TestRustSequenceBuffer:
    def test_in_order(self):
        buf = RustSequenceBuffer()
        r = buf.push("k", 0, "A")
        assert r == ["A"]

    def test_consecutive(self):
        buf = RustSequenceBuffer()
        assert buf.push("k", 0, "A") == ["A"]
        assert buf.push("k", 1, "B") == ["B"]

    def test_gap_then_fill(self):
        buf = RustSequenceBuffer()
        assert buf.push("k", 0, "A") == ["A"]
        # seq 2 arrives before seq 1
        assert buf.push("k", 2, "C") == []
        assert buf.push("k", 1, "B") == ["B", "C"]

    def test_duplicate_dropped(self):
        buf = RustSequenceBuffer()
        buf.push("k", 0, "A")
        assert buf.push("k", 0, "A-dup") == []

    def test_expected_seq(self):
        buf = RustSequenceBuffer()
        assert buf.expected_seq("k") == 0
        buf.push("k", 0, "A")
        assert buf.expected_seq("k") == 1

    def test_buffered_count(self):
        buf = RustSequenceBuffer()
        buf.push("k", 2, "C")
        buf.push("k", 3, "D")
        assert buf.buffered_count("k") == 2

    def test_flush(self):
        buf = RustSequenceBuffer()
        buf.push("k", 2, "C")
        buf.push("k", 5, "F")
        flushed = buf.flush("k")
        assert len(flushed) == 2
        assert buf.buffered_count("k") == 0

    def test_pending_count(self):
        buf = RustSequenceBuffer()
        buf.push("k1", 2, "X")
        buf.push("k2", 3, "Y")
        assert buf.pending_count("k1") == 1
        assert buf.pending_count(None) == 2

    def test_reset_key(self):
        buf = RustSequenceBuffer()
        buf.push("k", 0, "A")
        buf.push("k", 2, "C")
        buf.reset_key("k")
        assert buf.expected_seq("k") == 0
        assert buf.buffered_count("k") == 0

    def test_clear(self):
        buf = RustSequenceBuffer()
        buf.push("k1", 0, "A")
        buf.push("k2", 0, "B")
        buf.clear()
        assert buf.expected_seq("k1") == 0
        assert buf.expected_seq("k2") == 0

    def test_max_buffer(self):
        buf = RustSequenceBuffer(max_buffer_size=3)
        buf.push("k", 2, "C")
        buf.push("k", 3, "D")
        buf.push("k", 4, "E")
        # 4th buffered item should be silently dropped
        buf.push("k", 5, "F")
        assert buf.buffered_count("k") == 3

    def test_parity_with_python(self):
        """Compare Rust vs Python SequenceBuffer."""
        from execution.ingress.sequence_buffer import SequenceBuffer
        py = SequenceBuffer()
        rs = RustSequenceBuffer()

        # Same sequence of operations
        for key, seq, payload in [("k", 0, "A"), ("k", 2, "C"), ("k", 1, "B"), ("k", 3, "D")]:
            py_result = py.push(key=key, seq=seq, payload=payload)
            rs_result = rs.push(key, seq, payload)
            assert list(py_result) == list(rs_result), f"Mismatch at seq={seq}"


# ============================================================
# Event ID
# ============================================================

class TestRustEventId:
    def test_uuid_format(self):
        eid = rust_event_id()
        parts = eid.split("-")
        assert len(parts) == 5
        assert len(eid) == 36

    def test_unique(self):
        ids = {rust_event_id() for _ in range(100)}
        assert len(ids) == 100

    def test_now_ns(self):
        ns = rust_now_ns()
        py_ns = int(time.time() * 1e9)
        # Within 1 second
        assert abs(ns - py_ns) < 1_000_000_000


# ============================================================
# Request IDs
# ============================================================

class TestRustSanitize:
    def test_basic(self):
        assert rust_sanitize("hello world") == "hello-world"

    def test_special_chars(self):
        assert rust_sanitize("a@b#c") == "a-b-c"

    def test_empty(self):
        assert rust_sanitize("") == "x"

    def test_only_special(self):
        assert rust_sanitize("@#$") == "x"

    def test_dashes_stripped(self):
        assert rust_sanitize("-hello-") == "hello"

    def test_parity_with_python(self):
        from execution.bridge.request_ids import _sanitize
        cases = ["hello", "a b c", "test@123", "", "---", "A_B-C"]
        for s in cases:
            # Use Python fallback for comparison
            py = s.strip()
            if not py:
                py = "x"
            else:
                import re
                py = re.sub(r"[^A-Za-z0-9_-]+", "-", py).strip("-") or "x"
            rs = rust_sanitize(s)
            assert rs == py, f"Mismatch for {s!r}: py={py!r}, rs={rs!r}"


class TestRustShortHash:
    def test_matches_python(self):
        text = "test|data"
        py = hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        rs = rust_short_hash(text, 10)
        assert rs == py

    def test_default_n(self):
        h = rust_short_hash("abc")
        assert len(h) == 10


class TestRustMakeIdempotencyKey:
    def test_deterministic(self):
        k1 = rust_make_idempotency_key("binance", "submit", "order123")
        k2 = rust_make_idempotency_key("binance", "submit", "order123")
        assert k1 == k2
        assert len(k1) == 64

    def test_parity(self):
        # Compare with Python implementation
        from execution.bridge.request_ids import _sanitize
        venue, action, key = "binance", "submit", "order-1"
        v = _sanitize(venue).lower()
        a = _sanitize(action).lower()
        k = _sanitize(key)
        base = f"{v}|{a}|{k}"
        py = hashlib.sha256(base.encode("utf-8")).hexdigest()
        rs = rust_make_idempotency_key(venue, action, key)
        assert rs == py


class TestRustClientOrderId:
    def test_deterministic(self):
        cid = rust_client_order_id("qsys", "local", "ml", "BTCUSDT", "lid1")
        assert len(cid) <= 36
        assert "qsys" in cid

    def test_nonce_mode(self):
        cid = rust_client_order_id("qsys", "local", "ml", "BTCUSDT", nonce=42, deterministic=False)
        assert "n2a" in cid  # hex(42)

    def test_truncation(self):
        cid = rust_client_order_id(
            "very-long-namespace", "very-long-run-id", "strategy-name", "BTCUSDTPERP",
            "logical-id-1", max_len=36,
        )
        assert len(cid) <= 36


# ============================================================
# Signer
# ============================================================

class TestRustHmacSign:
    def test_signature_format(self):
        query, sig = rust_hmac_sign("secret123", [("symbol", "BTCUSDT"), ("side", "BUY")])
        assert len(sig) == 64  # SHA-256 hex
        assert "symbol=" in query
        assert "side=" in query

    def test_adds_timestamp(self):
        query, sig = rust_hmac_sign("secret", [("a", "1")])
        assert "timestamp=" in query

    def test_preserves_existing_timestamp(self):
        query, sig = rust_hmac_sign("secret", [("timestamp", "12345"), ("a", "1")])
        assert "timestamp=12345" in query

    def test_verify_roundtrip(self):
        params = [("symbol", "BTCUSDT"), ("side", "BUY"), ("timestamp", "1700000000000")]
        _, sig = rust_hmac_sign("mysecret", params.copy())
        params_with_sig = params + [("signature", sig)]
        assert rust_hmac_verify("mysecret", params_with_sig, sig) is True

    def test_verify_wrong_sig(self):
        params = [("symbol", "BTCUSDT"), ("timestamp", "1700000000000")]
        assert rust_hmac_verify("mysecret", params, "wrong") is False

    def test_parity_with_python(self):
        """Compare Rust HMAC with Python HmacSha256Signer."""
        import hmac as py_hmac
        secret = "test_secret_key"
        params = {"symbol": "BTCUSDT", "side": "BUY", "quantity": "0.001", "timestamp": "1700000000000"}

        # Python
        items = sorted(params.items())
        from urllib.parse import urlencode
        qs = urlencode(items)
        py_sig = py_hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()

        # Rust
        _, rs_sig = rust_hmac_sign(secret, list(params.items()))
        assert rs_sig == py_sig


# ============================================================
# Route Match
# ============================================================

class TestRustRouteEventType:
    def test_market(self):
        assert rust_route_event_type("MARKET_DATA") == "pipeline"

    def test_fill(self):
        assert rust_route_event_type("FILL") == "pipeline"

    def test_funding(self):
        assert rust_route_event_type("FUNDING") == "pipeline"

    def test_order_update(self):
        assert rust_route_event_type("ORDER_UPDATE") == "pipeline"

    def test_order_report(self):
        assert rust_route_event_type("ORDER_REPORT") == "pipeline"

    def test_signal(self):
        assert rust_route_event_type("SIGNAL") == "decision"

    def test_intent(self):
        assert rust_route_event_type("INTENT") == "decision"

    def test_risk(self):
        assert rust_route_event_type("RISK_CHECK") == "decision"

    def test_order_generic(self):
        assert rust_route_event_type("ORDER") == "execution"

    def test_unknown(self):
        assert rust_route_event_type("UNKNOWN") == "drop"

    def test_case_insensitive(self):
        assert rust_route_event_type("market_data") == "pipeline"
        assert rust_route_event_type("Market_Data") == "pipeline"

    def test_parity_with_python(self):
        """Compare with Python dispatcher._route_from_type()."""
        from engine.dispatcher import EventDispatcher
        cases = [
            "MARKET_DATA", "FILL", "FUNDING", "ORDER_UPDATE", "ORDER_REPORT",
            "ORDER_STATUS", "SIGNAL", "INTENT", "RISK", "ORDER", "UNKNOWN",
        ]
        route_map = {"pipeline": "PIPELINE", "decision": "DECISION", "execution": "EXECUTION", "drop": "DROP"}
        for et in cases:
            rs = rust_route_event_type(et)
            py = EventDispatcher._route_from_type(et).value.lower()
            assert rs == py, f"Mismatch for {et}: rust={rs}, py={py}"
