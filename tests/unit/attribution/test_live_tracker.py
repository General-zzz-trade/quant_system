"""Tests for LiveSignalTracker."""
from attribution.live_tracker import LiveSignalTracker


class _MockFill:
    def __init__(self, origin: str, pnl: float):
        self.origin = origin
        self.realized_pnl = pnl


class TestLiveSignalTracker:
    def test_on_fill_accumulates_pnl(self):
        tracker = LiveSignalTracker()
        tracker.on_fill(_MockFill("h24_lgbm", 10.0))
        tracker.on_fill(_MockFill("h24_lgbm", -3.0))
        tracker.on_fill(_MockFill("h12_xgb", 5.0))

        status = tracker.get_status()
        assert status["origins"]["h24_lgbm"]["pnl"] == 7.0
        assert status["origins"]["h24_lgbm"]["trades"] == 2
        assert status["origins"]["h12_xgb"]["pnl"] == 5.0
        assert status["total_pnl"] == 12.0

    def test_record_pnl_direct(self):
        tracker = LiveSignalTracker()
        tracker.record_pnl("manual", 100.0, is_win=True)
        tracker.record_pnl("manual", -20.0, is_win=False)

        status = tracker.get_status()
        assert status["origins"]["manual"]["pnl"] == 80.0
        assert status["origins"]["manual"]["trades"] == 2
        assert status["origins"]["manual"]["win_rate"] == 50.0

    def test_win_rate_calculation(self):
        tracker = LiveSignalTracker()
        tracker.on_fill(_MockFill("sig", 10.0))
        tracker.on_fill(_MockFill("sig", 5.0))
        tracker.on_fill(_MockFill("sig", -1.0))
        tracker.on_fill(_MockFill("sig", -2.0))

        status = tracker.get_status()
        assert status["origins"]["sig"]["win_rate"] == 50.0

    def test_unknown_origin(self):
        tracker = LiveSignalTracker()

        class NoOriginFill:
            realized_pnl = 5.0

        tracker.on_fill(NoOriginFill())
        status = tracker.get_status()
        assert "unknown" in status["origins"]

    def test_export_metrics_with_prometheus(self):
        calls = []

        class MockPrometheus:
            def set_gauge(self, name, value, labels=None):
                calls.append((name, value, labels))

        tracker = LiveSignalTracker(prometheus=MockPrometheus())
        tracker.on_fill(_MockFill("h24_lgbm", 10.0))
        tracker.export_metrics()

        gauge_names = [c[0] for c in calls]
        assert "signal_pnl_by_origin" in gauge_names
        assert "signal_trades_by_origin" in gauge_names

    def test_get_status_empty(self):
        tracker = LiveSignalTracker()
        status = tracker.get_status()
        assert status["total_pnl"] == 0.0
        assert status["total_trades"] == 0
        assert status["origins"] == {}
