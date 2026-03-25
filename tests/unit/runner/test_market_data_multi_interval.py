"""Test multi-interval WS subscription configuration."""
from __future__ import annotations


class TestMultiIntervalConfig:
    def test_multi_interval_creates_separate_subscriptions(self):
        """Each interval should create a separate subscription entry."""
        from runner.builders.market_data_builder import _build_subscriptions

        subs = _build_subscriptions(
            symbols=("ETHUSDT",),
            kline_interval="60",
            multi_interval_symbols={"ETHUSDT": ["60", "15"]},
        )
        # Should have 2 subscriptions for ETHUSDT
        intervals = [s["interval"] for s in subs if s["symbol"] == "ETHUSDT"]
        assert "60" in intervals
        assert "15" in intervals

    def test_no_multi_interval_uses_default(self):
        """Without multi_interval config, use single default interval."""
        from runner.builders.market_data_builder import _build_subscriptions

        subs = _build_subscriptions(
            symbols=("ETHUSDT",),
            kline_interval="60",
            multi_interval_symbols=None,
        )
        assert len(subs) == 1
        assert subs[0]["interval"] == "60"

    def test_multi_interval_multiple_symbols(self):
        """Multi-interval for one symbol should not duplicate other symbols."""
        from runner.builders.market_data_builder import _build_subscriptions

        subs = _build_subscriptions(
            symbols=("ETHUSDT", "BTCUSDT"),
            kline_interval="60",
            multi_interval_symbols={"ETHUSDT": ["60", "15"]},
        )
        eth_intervals = [s["interval"] for s in subs if s["symbol"] == "ETHUSDT"]
        btc_intervals = [s["interval"] for s in subs if s["symbol"] == "BTCUSDT"]
        # ETHUSDT gets 2 subscriptions (60 + 15)
        assert sorted(eth_intervals) == ["15", "60"]
        # BTCUSDT gets only the default interval
        assert btc_intervals == ["60"]

    def test_subscription_entries_have_required_keys(self):
        """Each subscription dict must carry symbol and interval."""
        from runner.builders.market_data_builder import _build_subscriptions

        subs = _build_subscriptions(
            symbols=("BTCUSDT",),
            kline_interval="60",
            multi_interval_symbols=None,
        )
        assert len(subs) == 1
        sub = subs[0]
        assert "symbol" in sub
        assert "interval" in sub
        assert sub["symbol"] == "BTCUSDT"
        assert sub["interval"] == "60"

    def test_subscription_stream_name_format(self):
        """Stream name should follow <symbol.lower>@kline_<interval> convention."""
        from runner.builders.market_data_builder import _build_subscriptions

        subs = _build_subscriptions(
            symbols=("ETHUSDT",),
            kline_interval="60",
            multi_interval_symbols=None,
        )
        assert subs[0]["stream"] == "ethusdt@kline_60"

    def test_multi_interval_stream_names(self):
        """Multi-interval subscriptions should each have correct stream name."""
        from runner.builders.market_data_builder import _build_subscriptions

        subs = _build_subscriptions(
            symbols=("ETHUSDT",),
            kline_interval="60",
            multi_interval_symbols={"ETHUSDT": ["60", "15"]},
        )
        streams = {s["stream"] for s in subs}
        assert "ethusdt@kline_60" in streams
        assert "ethusdt@kline_15" in streams

    def test_no_duplicate_subscriptions(self):
        """If multi_interval_symbols lists the default interval, no duplicates."""
        from runner.builders.market_data_builder import _build_subscriptions

        subs = _build_subscriptions(
            symbols=("ETHUSDT",),
            kline_interval="60",
            multi_interval_symbols={"ETHUSDT": ["60"]},
        )
        # Only one entry for ETHUSDT@60 even though it appears in both places
        streams = [s["stream"] for s in subs]
        assert streams.count("ethusdt@kline_60") == 1
