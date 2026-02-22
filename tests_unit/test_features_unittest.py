import unittest
from datetime import datetime, timedelta

from features.types import Bar
from features.technical import sma, ema, rsi, atr, returns


class TestFeatures(unittest.TestCase):
    def test_sma_len(self):
        s = sma([1, 2, 3, 4, 5], 3)
        self.assertEqual(len(s), 5)

    def test_ema_monotonic(self):
        e = ema([1, 1, 1, 1], 3)
        self.assertAlmostEqual(e[-1], 1.0)

    def test_rsi_bounds(self):
        s = rsi([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 14)
        v = s[-1]
        self.assertTrue(v is None or (0.0 <= v <= 100.0))

    def test_atr_len(self):
        t0 = datetime(2020, 1, 1)
        bars = [
            Bar(ts=t0 + timedelta(minutes=i), open=1, high=2, low=0.5, close=1.5, volume=10)
            for i in range(30)
        ]
        a = atr(bars, 14)
        self.assertEqual(len(a), len(bars))

    def test_returns_len(self):
        r = returns([1, 2, 4, 8])
        self.assertEqual(len(r), 4)


if __name__ == "__main__":
    unittest.main()
