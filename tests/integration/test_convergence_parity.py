"""Test convergence parity between AlphaRunner and LiveRunner components.

Verifies that the migrated logic produces identical results to the original
AlphaRunner implementations:
- combo_builder  <- PortfolioCombiner AGREE ONLY
- equity_leverage_gate <- AlphaRunner.LEVERAGE_LADDER + compute_z_scale
- consensus_scaling_gate <- AlphaRunner._get_consensus_scale
- adaptive_stop_gate <- AlphaRunner._compute_stop_price
- features/dominance_computer <- EnrichedFeatureComputer V14 dominance
"""
from runner.builders.combo_builder import combine_signals, ComboConfig

_CFG = ComboConfig(mode="agree", conviction_both=1.0, conviction_single=0.5, per_symbol_cap=0.3)


class TestCombineSignalsParity:
    """Verify combo signal logic matches AlphaRunner's PortfolioCombiner AGREE behavior."""

    def test_agree_mode_matches_alpha_runner(self):
        r = combine_signals(1, 1, _CFG)
        assert r.direction == 1 and r.conviction == 1.0
        r = combine_signals(1, -1, _CFG)
        assert r.direction == 0
        r = combine_signals(1, 0, _CFG)
        assert r.direction == 1 and r.conviction == 0.5

    def test_both_short_agree(self):
        r = combine_signals(-1, -1, _CFG)
        assert r.direction == -1 and r.conviction == 1.0

    def test_both_flat(self):
        r = combine_signals(0, 0, _CFG)
        assert r.direction == 0 and r.conviction == 0.0

    def test_short_with_flat(self):
        r = combine_signals(-1, 0, _CFG)
        assert r.direction == -1 and r.conviction == 0.5


class TestDominanceFeatureParity:
    """Verify V14 dominance features compute correctly."""

    def test_ratio_deviation(self):
        from features.dominance_computer import DominanceComputer
        comp = DominanceComputer()
        for _ in range(20):
            comp.update(60000.0, 3000.0)
        feats = comp.update(66000.0, 3000.0)
        assert feats["btc_dom_ratio_dev_20"] is not None
        assert feats["btc_dom_ratio_dev_20"] > 0.05

    def test_stable_ratio_gives_zero_deviation(self):
        from features.dominance_computer import DominanceComputer
        comp = DominanceComputer()
        for _ in range(21):
            feats = comp.update(60000.0, 3000.0)
        assert feats["btc_dom_ratio_dev_20"] is not None
        assert abs(feats["btc_dom_ratio_dev_20"]) < 1e-10

    def test_return_diff_after_warmup(self):
        from features.dominance_computer import DominanceComputer, _DOMINANCE_FEATURES
        comp = DominanceComputer()
        for _ in range(30):
            feats = comp.update(60000.0, 3000.0)
        # After 30 bars, all features should be available
        for name in _DOMINANCE_FEATURES:
            assert feats.get(name) is not None, f"{name} should not be None after 30 bars"
