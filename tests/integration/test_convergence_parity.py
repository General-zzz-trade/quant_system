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


class TestEquityLeverageParity:
    """Verify equity leverage brackets match AlphaRunner's LEVERAGE_LADDER."""

    def test_leverage_brackets_match_alpha_runner(self):
        from runner.gates.equity_leverage_gate import _bracket_leverage
        assert _bracket_leverage(500) == 1.5
        assert _bracket_leverage(4999) == 1.5
        assert _bracket_leverage(5000) == 1.5
        assert _bracket_leverage(19999) == 1.5
        assert _bracket_leverage(20000) == 1.0
        assert _bracket_leverage(49999) == 1.0
        assert _bracket_leverage(50000) == 1.0
        assert _bracket_leverage(100000) == 1.0

    def test_bracket_boundary_at_zero(self):
        from runner.gates.equity_leverage_gate import _bracket_leverage
        assert _bracket_leverage(0) == 1.5


class TestZScaleParity:
    """Verify z-score scaling matches AlphaRunner's compute_z_scale."""

    def test_z_scale_thresholds_match(self):
        from runner.gates.equity_leverage_gate import _z_scale
        assert _z_scale(2.5) == 1.5
        assert _z_scale(-2.1) == 1.5
        assert _z_scale(1.5) == 1.0
        assert _z_scale(-1.2) == 1.0
        assert _z_scale(0.8) == 0.7
        assert _z_scale(0.3) == 0.5
        assert _z_scale(0.0) == 0.5

    def test_negative_z_mirrors_positive(self):
        from runner.gates.equity_leverage_gate import _z_scale
        assert _z_scale(-2.5) == _z_scale(2.5)
        assert _z_scale(-1.5) == _z_scale(1.5)
        assert _z_scale(-0.8) == _z_scale(0.8)


class TestConsensusScaleParity:
    """Verify consensus scaling matches AlphaRunner's _get_consensus_scale."""

    def test_consensus_scale_matches_alpha_runner(self):
        from runner.gates.consensus_scaling_gate import _consensus_scale
        assert _consensus_scale("ETH", 1, {"BTC": -1, "SUI": -1, "AXS": -1}) == 1.3
        assert _consensus_scale("ETH", 1, {"BTC": 1, "SUI": 1, "AXS": 1}) == 1.0
        assert _consensus_scale("ETH", 1, {"BTC": 1, "SUI": -1, "AXS": -1}) == 0.7
        assert _consensus_scale("ETH", 1, {}) == 1.0

    def test_self_key_excluded(self):
        from runner.gates.consensus_scaling_gate import _consensus_scale
        assert _consensus_scale("ETH", 1, {"ETH": -1}) == 1.0

    def test_short_side_contrarian(self):
        from runner.gates.consensus_scaling_gate import _consensus_scale
        assert _consensus_scale("ETH", -1, {"BTC": 1, "SUI": 1, "AXS": 1}) == 1.3


class TestAdaptiveStopParity:
    """Verify ATR stop phases match AlphaRunner's 3-phase logic."""

    def test_gate_exists_and_is_callable(self):
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate
        gate = AdaptiveStopGate()
        assert hasattr(gate, "on_new_position")
        assert hasattr(gate, "check_stop")
        assert hasattr(gate, "compute_stop_price")

    def test_new_position_long(self):
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate
        gate = AdaptiveStopGate()
        gate.on_new_position("ETH", side=1, entry_price=3000.0)
        # Should not stop immediately
        assert gate.check_stop("ETH", 3000.0) is False

    def test_stop_triggers_on_large_drop(self):
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate
        gate = AdaptiveStopGate()
        gate.on_new_position("ETH", side=1, entry_price=3000.0)
        # 10% drop should trigger stop (exceeds 5% hard floor)
        assert gate.check_stop("ETH", 2700.0) is True


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
