"""Tests for VolEstimator."""

from execution.market_maker.vol_estimator import VolEstimator


class TestVolEstimator:
    def test_not_ready_initially(self):
        ve = VolEstimator(min_trades=5)
        assert not ve.ready
        assert ve.volatility == 0.0

    def test_ready_after_min_trades(self):
        ve = VolEstimator(alpha=0.1, min_trades=5)
        for i in range(10):
            ve.on_trade(2000.0 + i * 0.1)
        assert ve.ready
        assert ve.volatility > 0

    def test_higher_moves_higher_vol(self):
        ve1 = VolEstimator(alpha=0.1, min_trades=5)
        ve2 = VolEstimator(alpha=0.1, min_trades=5)
        for i in range(50):
            ve1.on_trade(2000.0 + i * 0.01)   # small moves
            ve2.on_trade(2000.0 + i * 1.0)     # large moves
        assert ve2.volatility > ve1.volatility

    def test_constant_price_zero_vol(self):
        ve = VolEstimator(alpha=0.5, min_trades=3)
        for _ in range(10):
            ve.on_trade(2000.0)
        assert ve.volatility < 1e-10

    def test_negative_price_ignored(self):
        ve = VolEstimator(min_trades=3)
        ve.on_trade(2000.0)
        vol = ve.on_trade(-1.0)
        assert vol == 0.0  # not enough trades yet, and negative ignored

    def test_reset(self):
        ve = VolEstimator(alpha=0.1, min_trades=3)
        for i in range(10):
            ve.on_trade(2000.0 + i)
        assert ve.ready
        ve.reset()
        assert not ve.ready
        assert ve.volatility == 0.0
