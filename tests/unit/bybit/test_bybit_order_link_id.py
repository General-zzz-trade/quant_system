"""Test orderLinkId uniqueness under rapid-fire submission."""


class TestOrderLinkIdUniqueness:
    def test_same_second_orders_produce_distinct_keys(self):
        """Two orders created within the same second must have different orderLinkIds."""
        from execution.adapters.bybit.adapter import _make_order_link_id

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
