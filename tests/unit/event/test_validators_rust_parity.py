"""Parity tests: RustEventValidator."""
import pytest

try:
    from _quant_hotpath import RustEventValidator
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not available")


class TestEventValidatorParity:
    def _make_event(self, event_id="evt1", timestamp=1000.0, event_type="market", stream="", **extra):
        d = {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_type": event_type,
        }
        if stream:
            d["stream"] = stream
        d.update(extra)
        return d

    def test_valid_event_passes(self):
        v = RustEventValidator()
        v.validate(self._make_event())

    def test_duplicate_rejects(self):
        v = RustEventValidator()
        v.validate(self._make_event(event_id="e1"))
        with pytest.raises(ValueError, match="duplicate"):
            v.validate(self._make_event(event_id="e1"))

    def test_nan_timestamp_rejects(self):
        v = RustEventValidator()
        with pytest.raises(ValueError):
            v.validate(self._make_event(timestamp=float('nan')))

    def test_negative_timestamp_rejects(self):
        v = RustEventValidator()
        with pytest.raises(ValueError):
            v.validate(self._make_event(timestamp=-1.0))

    def test_non_monotonic_rejects(self):
        v = RustEventValidator()
        v.validate(self._make_event(event_id="e1", timestamp=100.0, stream="s1"))
        with pytest.raises(ValueError, match="non-monotonic"):
            v.validate(self._make_event(event_id="e2", timestamp=50.0, stream="s1"))

    def test_different_streams_independent(self):
        v = RustEventValidator()
        v.validate(self._make_event(event_id="e1", timestamp=100.0, stream="s1"))
        # Different stream, earlier timestamp is fine
        v.validate(self._make_event(event_id="e2", timestamp=50.0, stream="s2"))

    def test_bounded_dedup(self):
        v = RustEventValidator(max_seen=100)
        for i in range(150):
            v.validate(self._make_event(event_id=f"e{i}", timestamp=float(1000 + i)))
        # Oldest should have been evicted -- count stays bounded
        assert v.seen_count() <= 100
        # Re-inserting an old evicted ID should work (not duplicate)
        v.validate(self._make_event(event_id="e0", timestamp=2000.0))

    def test_market_nan_close_rejects(self):
        v = RustEventValidator()
        with pytest.raises(ValueError, match="close"):
            v.validate(self._make_event(event_type="market", close=float('nan')))

    def test_fill_negative_qty_rejects(self):
        v = RustEventValidator()
        with pytest.raises(ValueError, match="qty"):
            v.validate(self._make_event(event_id="e2", event_type="fill", qty=-1.0, price=3000.0))

    def test_clear_seen(self):
        v = RustEventValidator()
        v.validate(self._make_event(event_id="e1"))
        v.clear_seen()
        # Should be able to re-use same ID after clear
        v.validate(self._make_event(event_id="e1", timestamp=2000.0))
