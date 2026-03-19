"""Tests for MetricsCollector."""

from execution.market_maker.metrics import MetricsCollector


class TestMetricsCollector:
    def test_initial_snapshot(self):
        mc = MetricsCollector()
        s = mc.snapshot()
        assert s.total_fills == 0
        assert s.realised_pnl == 0.0

    def test_record_fill(self):
        mc = MetricsCollector()
        mc.record_fill("buy", 0.01, 2000.0, 0.0)
        mc.record_fill("sell", 0.01, 2001.0, 0.01)
        s = mc.snapshot()
        assert s.total_fills == 2
        assert s.buy_fills == 1
        assert s.sell_fills == 1
        assert abs(s.realised_pnl - 0.01) < 1e-8

    def test_maker_rebate(self):
        mc = MetricsCollector(maker_rebate_bps=-1.0)
        mc.record_fill("buy", 0.01, 2000.0, 0.0)  # $20 notional
        assert abs(mc.maker_rebate_earned - 0.002) < 1e-6  # $20 * 1bps

    def test_spread_tracking(self):
        mc = MetricsCollector()
        mc.record_spread(3.0)
        mc.record_spread(5.0)
        s = mc.snapshot()
        assert abs(s.avg_spread_bps - 4.0) < 1e-6

    def test_drawdown_tracking(self):
        mc = MetricsCollector()
        mc.record_fill("buy", 0.01, 2000.0, 0.10)   # win
        mc.record_fill("sell", 0.01, 2000.0, -0.20)  # loss
        s = mc.snapshot()
        assert s.max_drawdown < 0

    def test_quote_cancel_counts(self):
        mc = MetricsCollector()
        mc.record_quote()
        mc.record_quote()
        mc.record_cancel()
        s = mc.snapshot()
        assert s.quotes_sent == 2
        assert s.cancels_sent == 1
