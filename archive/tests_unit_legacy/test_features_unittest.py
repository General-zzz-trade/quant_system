import unittest
from datetime import datetime, timedelta

from features.types import Bar
from features.technical import sma, ema, rsi, atr, returns, macd, bollinger_bands


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


class TestMACD(unittest.TestCase):
    def test_output_length(self):
        prices = list(range(1, 50))
        ml, sl, hist = macd(prices, fast=5, slow=10, signal=3)
        self.assertEqual(len(ml), 49)
        self.assertEqual(len(sl), 49)
        self.assertEqual(len(hist), 49)

    def test_warmup_is_none(self):
        prices = list(range(1, 30))
        ml, sl, _ = macd(prices, fast=5, slow=10, signal=3)
        # Before slow window, macd_line should still have values (ema always returns values)
        # Signal line needs signal warmup on top of macd
        self.assertIsNone(sl[0])  # not enough data for signal line at start

    def test_trending_market(self):
        # Strictly rising prices → fast EMA > slow EMA → positive MACD
        prices = [float(i) for i in range(1, 60)]
        ml, sl, hist = macd(prices, fast=12, slow=26, signal=9)
        # Last MACD should be positive (fast > slow in trending up market)
        last_valid = next(v for v in reversed(ml) if v is not None)
        self.assertGreater(last_valid, 0)

    def test_invalid_windows_raise(self):
        with self.assertRaises(ValueError):
            macd([1, 2, 3], fast=10, slow=5)  # fast >= slow
        with self.assertRaises(ValueError):
            macd([1, 2, 3], fast=0, slow=5)   # zero window


class TestBollingerBands(unittest.TestCase):
    def test_output_length(self):
        prices = list(range(1, 30))
        u, m, l = bollinger_bands(prices, window=10)
        self.assertEqual(len(u), 29)
        self.assertEqual(len(m), 29)
        self.assertEqual(len(l), 29)

    def test_warmup_is_none(self):
        prices = list(range(1, 30))
        u, m, l = bollinger_bands(prices, window=10)
        # First 9 entries should be None (need full window)
        for i in range(9):
            self.assertIsNone(u[i])
        self.assertIsNotNone(u[9])  # 10th entry should be valid

    def test_upper_above_lower(self):
        prices = [100.0 + (i % 5) for i in range(30)]
        u, m, l = bollinger_bands(prices, window=10)
        for up, lo in zip(u, l):
            if up is not None:
                self.assertGreater(up, lo)

    def test_middle_is_sma(self):
        prices = [float(i) for i in range(1, 21)]
        _, m, _ = bollinger_bands(prices, window=10)
        # Middle band at index 9 should equal SMA of prices[0..9]
        expected_sma = sum(prices[0:10]) / 10
        self.assertAlmostEqual(m[9], expected_sma, places=8)

    def test_invalid_window_raises(self):
        with self.assertRaises(ValueError):
            bollinger_bands([1, 2, 3], window=0)
        with self.assertRaises(ValueError):
            bollinger_bands([1, 2, 3], window=10, num_std=0)


if __name__ == "__main__":
    unittest.main()
