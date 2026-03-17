"""Test multi-exchange funding spread features."""
from features.funding_spread import FundingSpreadComputer, FUNDING_SPREAD_FEATURES


class TestFundingSpreadComputer:
    def test_two_exchanges_computes_spread(self):
        comp = FundingSpreadComputer()
        feats = comp.update({"binance": 0.0001, "bybit": 0.0003})
        assert feats["funding_spread"] is not None
        assert abs(feats["funding_spread"] - 0.0002) < 1e-10

    def test_single_exchange_returns_none(self):
        comp = FundingSpreadComputer()
        feats = comp.update({"binance": 0.0001})
        assert all(v is None for v in feats.values())

    def test_skew_with_three_exchanges(self):
        comp = FundingSpreadComputer()
        feats = comp.update({"binance": 0.0001, "bybit": 0.0002, "okx": 0.0010})
        assert feats["funding_skew"] is not None
        # mean = 0.000433, median = 0.0002, skew > 0
        assert feats["funding_skew"] > 0

    def test_zscore_needs_warmup(self):
        comp = FundingSpreadComputer(zscore_window=720)
        for _ in range(100):
            feats = comp.update({"binance": 0.0001, "bybit": 0.0002})
        assert feats["funding_zscore_spread"] is None  # < 180 warmup

    def test_zscore_after_warmup(self):
        comp = FundingSpreadComputer(zscore_window=720)
        for i in range(200):
            # Vary spread slightly so std > 0 and zscore can be computed
            rate_b = 0.0001 + i * 1e-8
            feats = comp.update({"binance": rate_b, "bybit": 0.0003})
        assert feats["funding_zscore_spread"] is not None

    def test_nan_rates_excluded(self):
        comp = FundingSpreadComputer()
        feats = comp.update({"binance": 0.0001, "bybit": float("nan"), "okx": 0.0003})
        assert feats["funding_spread"] is not None
        assert abs(feats["funding_spread"] - 0.0002) < 1e-10

    def test_feature_names(self):
        assert len(FUNDING_SPREAD_FEATURES) == 3
