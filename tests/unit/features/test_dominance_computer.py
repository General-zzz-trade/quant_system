"""Test V14 dominance feature computer."""
from features.dominance_computer import DominanceComputer, _DOMINANCE_FEATURES


class TestDominanceComputer:
    def test_insufficient_history_returns_none(self):
        comp = DominanceComputer(window=75)
        feats = comp.update(60000.0, 3000.0)
        assert all(v is None for v in feats.values())
        assert set(feats.keys()) == set(_DOMINANCE_FEATURES)

    def test_after_warmup_returns_values(self):
        comp = DominanceComputer(window=75)
        for i in range(25):
            btc = 60000 + i * 100
            eth = 3000 + i * 5
            feats = comp.update(btc, eth)
        # After 25 bars, ratio_dev_20 and ratio_mom_10 should be non-None
        assert feats["btc_dom_ratio_dev_20"] is not None
        assert feats["btc_dom_ratio_mom_10"] is not None

    def test_zero_price_returns_none(self):
        comp = DominanceComputer()
        feats = comp.update(0.0, 3000.0)
        assert all(v is None for v in feats.values())

    def test_ratio_dev_direction(self):
        comp = DominanceComputer()
        # Feed constant ratio for 20 bars, then spike BTC
        for _ in range(20):
            comp.update(60000.0, 3000.0)  # ratio = 20
        feats = comp.update(66000.0, 3000.0)  # ratio = 22, above MA
        assert feats["btc_dom_ratio_dev_20"] is not None
        assert feats["btc_dom_ratio_dev_20"] > 0  # above MA

    def test_return_diff_features_after_warmup(self):
        comp = DominanceComputer(window=75)
        for i in range(25):
            comp.update(60000.0 + i * 10, 3000.0 + i * 1)
        feats = comp.update(60250.0, 3025.0)
        # After 26 bars (25 returns), 6h diff should be available
        assert feats["btc_dom_return_diff_6h"] is not None

    def test_return_diff_24h_requires_24_returns(self):
        comp = DominanceComputer(window=75)
        # Only 20 bars — not enough for 24h diff
        for i in range(20):
            comp.update(60000.0 + i * 10, 3000.0 + i)
        feats = comp.update(60200.0, 3020.0)
        assert feats["btc_dom_return_diff_24h"] is None

    def test_all_keys_always_present(self):
        comp = DominanceComputer(window=75)
        # Even on first bar, all 4 keys must be returned
        feats = comp.update(50000.0, 2500.0)
        assert set(feats.keys()) == set(_DOMINANCE_FEATURES)

    def test_negative_eth_price_returns_none(self):
        comp = DominanceComputer()
        feats = comp.update(60000.0, -1.0)
        assert all(v is None for v in feats.values())
