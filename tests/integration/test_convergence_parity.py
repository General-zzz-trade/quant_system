"""Test convergence parity between AlphaRunner and LiveRunner components.

Verifies that the migrated logic produces identical results to the original
AlphaRunner implementations:
- combo_builder  ← PortfolioCombiner AGREE ONLY
- equity_leverage_gate ← AlphaRunner.LEVERAGE_LADDER + compute_z_scale
- consensus_scaling_gate ← AlphaRunner._get_consensus_scale
- adaptive_stop_gate ← AlphaRunner._compute_stop_price
- features/dominance_computer ← EnrichedFeatureComputer V14 dominance
"""
class TestCombineSignalsParity:
    """Verify combo signal logic matches AlphaRunner's PortfolioCombiner AGREE behavior."""

    def test_agree_mode_matches_alpha_runner(self):
        """AGREE mode should match PortfolioCombiner's AGREE ONLY behavior."""
        from runner.builders.combo_builder import combine_signals, ComboConfig

        cfg = ComboConfig(mode="agree", conviction_both=1.0, conviction_single=0.5, per_symbol_cap=0.3)

        # Both long → trade at full conviction
        r = combine_signals(1, 1, cfg)
        assert r.direction == 1 and r.conviction == 1.0

        # Disagree → flat (AGREE ONLY)
        r = combine_signals(1, -1, cfg)
        assert r.direction == 0

        # One flat → trade at half conviction
        r = combine_signals(1, 0, cfg)
        assert r.direction == 1 and r.conviction == 0.5

    def test_both_short_agree(self):
        """Both short should give direction=-1 at full conviction."""
        from runner.builders.combo_builder import combine_signals, ComboConfig

        cfg = ComboConfig(mode="agree")
        r = combine_signals(-1, -1, cfg)
        assert r.direction == -1 and r.conviction == 1.0

    def test_both_flat(self):
        """Both flat → direction=0."""
        from runner.builders.combo_builder import combine_signals, ComboConfig

        cfg = ComboConfig(mode="agree")
        r = combine_signals(0, 0, cfg)
        assert r.direction == 0 and r.conviction == 0.0

    def test_short_with_flat(self):
        """Short + flat → short at half conviction."""
        from runner.builders.combo_builder import combine_signals, ComboConfig

        cfg = ComboConfig(mode="agree")
        r = combine_signals(-1, 0, cfg)
        assert r.direction == -1 and r.conviction == 0.5

    def test_default_config(self):
        """Default config should have conviction_both=1.0 and conviction_single=0.5."""
        from runner.builders.combo_builder import ComboConfig

        cfg = ComboConfig()
        assert cfg.conviction_both == 1.0
        assert cfg.conviction_single == 0.5
        assert cfg.per_symbol_cap == 0.3


class TestEquityLeverageParity:
    """Verify equity leverage brackets match AlphaRunner's LEVERAGE_LADDER."""

    def test_leverage_brackets_match_alpha_runner(self):
        """Brackets should match AlphaRunner lines 30-35."""
        from runner.gates.equity_leverage_gate import _bracket_leverage

        # AlphaRunner: $0-5K → 1.5x
        assert _bracket_leverage(500) == 1.5
        assert _bracket_leverage(4999) == 1.5

        # AlphaRunner: $5K-20K → 1.5x
        assert _bracket_leverage(5000) == 1.5
        assert _bracket_leverage(19999) == 1.5

        # AlphaRunner: $20K-50K → 1.0x
        assert _bracket_leverage(20000) == 1.0
        assert _bracket_leverage(49999) == 1.0

        # AlphaRunner: $50K+ → 1.0x
        assert _bracket_leverage(50000) == 1.0
        assert _bracket_leverage(100000) == 1.0

    def test_bracket_boundary_at_zero(self):
        """Zero equity should return lowest bracket leverage."""
        from runner.gates.equity_leverage_gate import _bracket_leverage

        assert _bracket_leverage(0) == 1.5


class TestZScaleParity:
    """Verify z-score scaling matches AlphaRunner's compute_z_scale."""

    def test_z_scale_thresholds_match(self):
        """Z-scale thresholds should match AlphaRunner lines 683-702."""
        from runner.gates.equity_leverage_gate import _z_scale

        # |z| > 2.0 → 1.5x
        assert _z_scale(2.5) == 1.5
        assert _z_scale(-2.1) == 1.5

        # |z| > 1.0 → 1.0x
        assert _z_scale(1.5) == 1.0
        assert _z_scale(-1.2) == 1.0

        # |z| > 0.5 → 0.7x
        assert _z_scale(0.8) == 0.7

        # |z| ≤ 0.5 → 0.5x
        assert _z_scale(0.3) == 0.5
        assert _z_scale(0.0) == 0.5

    def test_z_scale_exact_boundaries(self):
        """Boundary values: exactly 2.0, 1.0, 0.5."""
        from runner.gates.equity_leverage_gate import _z_scale

        # Boundaries use strict greater-than
        assert _z_scale(2.0) == 1.0   # not > 2.0 → next bucket
        assert _z_scale(1.0) == 0.7   # not > 1.0 → next bucket
        assert _z_scale(0.5) == 0.5   # not > 0.5 → bottom bucket

    def test_negative_z_mirrors_positive(self):
        """Negative z-scores should mirror positive (uses abs)."""
        from runner.gates.equity_leverage_gate import _z_scale

        assert _z_scale(-2.5) == _z_scale(2.5)
        assert _z_scale(-1.5) == _z_scale(1.5)
        assert _z_scale(-0.8) == _z_scale(0.8)
        assert _z_scale(-0.3) == _z_scale(0.3)


class TestConsensusScaleParity:
    """Verify consensus scaling matches AlphaRunner's _get_consensus_scale."""

    def test_consensus_scale_matches_alpha_runner(self):
        """Consensus logic should match AlphaRunner lines 628-680."""
        from runner.gates.consensus_scaling_gate import _consensus_scale

        # All disagree → 1.3x (contrarian boost)
        assert _consensus_scale("ETH", 1, {"BTC": -1, "SUI": -1, "AXS": -1}) == 1.3

        # All agree → 1.0x
        assert _consensus_scale("ETH", 1, {"BTC": 1, "SUI": 1, "AXS": 1}) == 1.0

        # 1/3 agree → 0.7x
        assert _consensus_scale("ETH", 1, {"BTC": 1, "SUI": -1, "AXS": -1}) == 0.7

        # No others → 1.0x (no data)
        assert _consensus_scale("ETH", 1, {}) == 1.0

    def test_flat_signal_returns_one(self):
        """Flat signal should always return 1.0 regardless of others."""
        from runner.gates.consensus_scaling_gate import _consensus_scale

        assert _consensus_scale("ETH", 0, {"BTC": 1, "SUI": -1, "AXS": -1}) == 1.0

    def test_self_key_excluded(self):
        """Symbol's own key in other_signals dict should be ignored."""
        from runner.gates.consensus_scaling_gate import _consensus_scale

        # Include own key in dict — should be treated as no-data case
        assert _consensus_scale("ETH", 1, {"ETH": -1}) == 1.0

    def test_contrarian_requires_two_or_more_others(self):
        """Contrarian boost only fires when n_total >= 2."""
        from runner.gates.consensus_scaling_gate import _consensus_scale

        # Only one other disagrees → not full contrarian boost, goes to 0.5
        result = _consensus_scale("ETH", 1, {"BTC": -1})
        # n_total=1, opposite_dir=1 == n_total, but n_total < 2 → not 1.3
        assert result != 1.3

    def test_short_side_contrarian(self):
        """Contrarian boost works for short signals too."""
        from runner.gates.consensus_scaling_gate import _consensus_scale

        # Short signal, all others are long → contrarian
        assert _consensus_scale("ETH", -1, {"BTC": 1, "SUI": 1, "AXS": 1}) == 1.3


class TestAdaptiveStopParity:
    """Verify ATR stop phases match AlphaRunner's 3-phase logic."""

    def test_initial_stop_distance(self):
        """Initial phase: stop distance = atr × 2.0 (absolute units).

        ATR must be small enough that the initial stop stays above the 5% hard
        floor. With entry=3000 and 5% floor at 2850, we need entry - atr*2 > 2850,
        i.e. atr < 75. Using atr=50 gives stop=2900, safely above floor.
        """
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate

        gate = AdaptiveStopGate()
        atr = 50.0  # 1.67% of 3000; stop=2900, above 5% floor of 2850
        gate.on_new_position("ETH", side="long", entry_price=3000.0, atr=atr)
        stop = gate.get_stop_price("ETH")
        assert stop is not None
        # Initial: entry - atr * 2.0 for long
        assert abs(stop - (3000.0 - atr * 2.0)) < 1.0

    def test_initial_stop_short(self):
        """Initial phase short: stop = entry + atr * 2.0."""
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate

        gate = AdaptiveStopGate()
        atr = 50.0
        gate.on_new_position("ETH", side="short", entry_price=3000.0, atr=atr)
        stop = gate.get_stop_price("ETH")
        assert stop is not None
        assert abs(stop - (3000.0 + atr * 2.0)) < 1.0

    def test_breakeven_transition(self):
        """After profit > atr × 1.0, stop moves to breakeven."""
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate

        gate = AdaptiveStopGate()
        atr = 100.0
        gate.on_new_position("ETH", side="long", entry_price=3000.0, atr=atr)

        # Price rises by more than atr * 1.0 (breakeven_atr=1.0)
        gate.update_price("ETH", 3150.0)
        stop = gate.get_stop_price("ETH")
        # Should be at or above entry (stop moved off initial position)
        assert stop is not None
        assert stop > 3000.0 - 1.0  # At or above entry

    def test_no_position_returns_none(self):
        """get_stop_price returns None for unknown symbol."""
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate

        gate = AdaptiveStopGate()
        assert gate.get_stop_price("ETH") is None

    def test_close_position_removes_stop(self):
        """close_position should remove stop tracking."""
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate

        gate = AdaptiveStopGate()
        gate.on_new_position("ETH", side="long", entry_price=3000.0, atr=100.0)
        assert gate.get_stop_price("ETH") is not None
        gate.close_position("ETH")
        assert gate.get_stop_price("ETH") is None

    def test_stop_never_widens_below_five_percent(self):
        """Hard floor: stop must never be more than 5% below entry."""
        from runner.gates.adaptive_stop_gate import AdaptiveStopGate

        # Very large ATR that would push stop past 5%
        gate = AdaptiveStopGate()
        gate.on_new_position("ETH", side="long", entry_price=3000.0, atr=500.0)
        stop = gate.get_stop_price("ETH")
        assert stop is not None
        assert stop >= 3000.0 * 0.95  # max 5% loss


class TestDominanceFeatureParity:
    """Verify V14 dominance features compute correctly."""

    def test_ratio_deviation(self):
        """BTC/ETH ratio deviation should reflect price divergence."""
        from features.dominance_computer import DominanceComputer

        comp = DominanceComputer()
        # 20 bars of stable ratio = 20 (btc=60000, eth=3000)
        for _ in range(20):
            comp.update(60000.0, 3000.0)

        # BTC jumps 10% → ratio increases
        feats = comp.update(66000.0, 3000.0)
        assert feats["btc_dom_ratio_dev_20"] is not None
        assert feats["btc_dom_ratio_dev_20"] > 0.05  # Significant positive deviation

    def test_insufficient_history_returns_none(self):
        """Features requiring 20+ bars should return None with fewer bars."""
        from features.dominance_computer import DominanceComputer

        comp = DominanceComputer()
        for _ in range(5):
            feats = comp.update(60000.0, 3000.0)

        assert feats["btc_dom_ratio_dev_20"] is None
        assert feats["btc_dom_ratio_dev_50"] is None
        assert feats["btc_dom_ratio_ret_24"] is None
        assert feats["btc_dom_ratio_ret_72"] is None

    def test_stable_ratio_gives_zero_deviation(self):
        """Constant BTC/ETH ratio should produce ~0 deviation."""
        from features.dominance_computer import DominanceComputer

        comp = DominanceComputer()
        for _ in range(21):
            feats = comp.update(60000.0, 3000.0)

        # Stable ratio → deviation near zero
        assert feats["btc_dom_ratio_dev_20"] is not None
        assert abs(feats["btc_dom_ratio_dev_20"]) < 1e-10

    def test_ret_24_requires_25_bars(self):
        """btc_dom_ratio_ret_24 needs 25 bars."""
        from features.dominance_computer import DominanceComputer

        comp = DominanceComputer()
        for i in range(24):
            feats = comp.update(60000.0, 3000.0)
        assert feats["btc_dom_ratio_ret_24"] is None

        feats = comp.update(60000.0, 3000.0)  # 25th bar
        assert feats["btc_dom_ratio_ret_24"] is not None
