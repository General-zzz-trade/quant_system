"""Tests for CorrelationComputer checkpoint/restore persistence."""
from __future__ import annotations



from risk.correlation_computer import CorrelationComputer


class TestCorrelationPersistence:
    def test_checkpoint_restore_roundtrip(self):
        cc = CorrelationComputer(window=60)

        # Build up return history
        for i in range(30):
            cc.update("BTCUSDT", 40000.0 + i * 10)
            cc.update("ETHUSDT", 3000.0 + i * 5)

        # Get correlation before checkpoint
        corr1 = cc.portfolio_avg_correlation(["BTCUSDT", "ETHUSDT"])
        assert corr1 is not None

        data = cc.checkpoint()
        assert "BTCUSDT" in data["returns"]
        assert "ETHUSDT" in data["returns"]
        assert "BTCUSDT" in data["last_prices"]

        # Restore into fresh computer
        cc2 = CorrelationComputer(window=60)
        cc2.restore(data)

        corr2 = cc2.portfolio_avg_correlation(["BTCUSDT", "ETHUSDT"])
        assert corr2 is not None
        assert abs(corr1 - corr2) < 1e-10

    def test_restore_empty(self):
        cc = CorrelationComputer(window=60)
        cc.restore({})
        assert cc._returns == {}
        assert cc._last_prices == {}

    def test_restore_continues_updating(self):
        cc = CorrelationComputer(window=60)
        for i in range(10):
            cc.update("BTCUSDT", 40000.0 + i * 10)

        data = cc.checkpoint()
        cc2 = CorrelationComputer(window=60)
        cc2.restore(data)

        # Continue updating — should work normally
        cc2.update("BTCUSDT", 40200.0)
        assert len(cc2._returns["BTCUSDT"]) == 10  # 9 from restore + 1 new
