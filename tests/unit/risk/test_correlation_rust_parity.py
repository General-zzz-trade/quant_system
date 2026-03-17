"""Parity tests: RustCorrelationComputer vs Python CorrelationComputer."""
import pytest

try:
    from _quant_hotpath import RustCorrelationComputer
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not available")


class TestCorrelationParity:
    def test_perfect_positive_correlation(self):
        c = RustCorrelationComputer(window=20)
        for i in range(25):
            c.update("A", 100.0 + i * 1.0)
            c.update("B", 200.0 + i * 2.0)
        corr = c.pairwise_correlation("A", "B")
        assert corr is not None
        assert abs(corr - 1.0) < 1e-8

    def test_perfect_negative_correlation(self):
        """Use push_return directly to get perfectly anti-correlated returns."""
        c = RustCorrelationComputer(window=20)
        for i in range(25):
            r = 0.01 * (i + 1)
            c.push_return("A", r)
            c.push_return("B", -r)
        corr = c.pairwise_correlation("A", "B")
        assert corr is not None
        assert abs(corr + 1.0) < 1e-8

    def test_avg_correlation(self):
        c = RustCorrelationComputer(window=20)
        for i in range(25):
            c.update("A", 100.0 + i)
            c.update("B", 200.0 + i * 0.5)
            c.update("C", 150.0 + i * 0.3)
        avg = c.avg_correlation(["A", "B", "C"])
        assert avg is not None
        assert 0.0 <= avg <= 1.0

    def test_exceeds_limit(self):
        c = RustCorrelationComputer(window=20)
        for i in range(25):
            c.update("A", 100.0 + i)
            c.update("B", 200.0 + i)  # perfect correlation
        assert c.exceeds_limit(["A", "B"], 0.5) is True
        assert c.exceeds_limit(["A", "B"], 1.5) is False

    def test_insufficient_data_returns_none(self):
        c = RustCorrelationComputer(window=20)
        c.update("A", 100.0)
        assert c.pairwise_correlation("A", "B") is None

    def test_invalid_close_raises(self):
        c = RustCorrelationComputer()
        with pytest.raises(ValueError):
            c.update("A", float('nan'))
        with pytest.raises(ValueError):
            c.update("A", -1.0)

    def test_window_eviction(self):
        c = RustCorrelationComputer(window=10)
        for i in range(20):
            c.update("A", 100.0 + i)
        assert c.data_points("A") == 10  # window=10, only keeps 10

    def test_checkpoint_restore(self):
        c = RustCorrelationComputer(window=20)
        for i in range(15):
            c.update("A", 100.0 + i)
            c.update("B", 200.0 + i * 0.5)
        ckpt = c.checkpoint()

        c2 = RustCorrelationComputer(window=20)
        c2.restore(ckpt)

        assert c2.symbol_count() == c.symbol_count()
        assert c2.data_points("A") == c.data_points("A")
