"""Test orderLinkId uniqueness under rapid-fire submission."""


class TestOrderLinkIdUniqueness:
    def test_same_timestamp_orders_produce_distinct_keys(self, monkeypatch):
        """Fixed timestamp should still produce unique orderLinkIds via the sequence suffix."""
        from execution.adapters.bybit.adapter import _make_order_link_id

        monkeypatch.setattr("execution.adapters.bybit.adapter._time.time", lambda: 1773834992.625)
        ids = set()
        for _ in range(100):
            ids.add(_make_order_link_id("ETHUSDT", "buy"))
        assert len(ids) == 100

    def test_order_link_id_format(self):
        """orderLinkId must start with 'qs_' and contain symbol+side."""
        from execution.adapters.bybit.adapter import _make_order_link_id

        oid = _make_order_link_id("BTCUSDT", "sell")
        assert oid.startswith("qs_")
        assert "BTCUSDT" in oid
        assert "s" in oid

    def test_order_link_id_length_within_bybit_limit(self):
        """Bybit orderLinkId max length is 36 characters."""
        from execution.adapters.bybit.adapter import _make_order_link_id

        oid = _make_order_link_id("ETHUSDT", "buy")
        assert len(oid) <= 36

    def test_long_symbol_still_fits_bybit_limit(self):
        from execution.adapters.bybit.adapter import _make_order_link_id

        oid = _make_order_link_id("1000000MOGUSDT", "buy")
        assert len(oid) <= 36
        assert oid.startswith("qs_1000000MOGUSDT_b_")
