"""Test EngineLoop metrics counters."""


class TestEngineLoopMetrics:
    def test_loop_metrics_dataclass(self):
        from engine.loop import LoopMetrics
        m = LoopMetrics()
        assert m.drops == 0
        assert m.retries == 0
        assert m.errors == 0
        assert m.processed == 0

    def test_loop_metrics_increment(self):
        from engine.loop import LoopMetrics
        m = LoopMetrics()
        m.drops += 1
        m.retries += 3
        m.errors += 2
        m.processed += 100
        assert m.drops == 1
        assert m.retries == 3
        assert m.errors == 2
        assert m.processed == 100
