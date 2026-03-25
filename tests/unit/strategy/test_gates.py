"""Tests for strategy/gates/ — all gate classes."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from runner.gate_chain import GateResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(symbol="BTCUSDT", **meta):
    """Create a minimal event-like object."""
    return SimpleNamespace(symbol=symbol, metadata=meta, price=0.0)


def _ev_with_signal(symbol="BTCUSDT", signal=1):
    return SimpleNamespace(symbol=symbol, metadata={"signal": signal}, price=0.0)


# ===========================================================================
# AdaptiveStopGate
# ===========================================================================

class TestAdaptiveStopGate:
    """Tests for AdaptiveStopGate — Python fallback path."""

    @pytest.fixture()
    def gate(self):
        with patch("strategy.gates.adaptive_stop_gate._HAS_RUST", False):
            from strategy.gates.adaptive_stop_gate import AdaptiveStopGate
            g = AdaptiveStopGate(atr_initial_mult=2.0, atr_trailing_mult=0.3, atr_fallback=0.015)
            g._use_rust = False
            g._rust = None
            return g

    def test_no_position_allows(self, gate):
        result = gate.check(_ev(), {"price": 100.0})
        assert result.allowed is True

    def test_on_new_position_records_state(self, gate):
        gate.on_new_position("BTCUSDT", 1, 100.0)
        state = gate._get_state("BTCUSDT")
        assert state.entry_price == 100.0
        assert state.side == 1

    def test_stop_not_triggered_within_range(self, gate):
        gate.on_new_position("BTCUSDT", 1, 100.0)
        # Price slightly below entry but above stop (fallback ATR=1.5%, initial_mult=2.0 → stop at ~97%)
        result = gate.check(_ev(), {"price": 98.0, "symbol": "BTCUSDT"})
        assert result.allowed is True

    def test_stop_triggered_on_large_drop(self, gate):
        gate.on_new_position("BTCUSDT", 1, 100.0)
        # Price drops below the 5% max loss floor → stop at 95.0
        result = gate.check(_ev(), {"price": 94.0, "symbol": "BTCUSDT"})
        assert result.allowed is False
        assert "stop_triggered" in result.reason

    def test_short_position_stop_triggered_on_rise(self, gate):
        gate.on_new_position("BTCUSDT", -1, 100.0)
        # Price rises above 5% max loss ceiling → stop at 105.0
        result = gate.check(_ev(), {"price": 106.0, "symbol": "BTCUSDT"})
        assert result.allowed is False

    def test_check_stop_no_position_returns_false(self, gate):
        assert gate.check_stop("BTCUSDT", 100.0) is False

    def test_check_stop_invalid_price(self, gate):
        assert gate.check_stop("BTCUSDT", float("nan")) is False
        assert gate.check_stop("BTCUSDT", -1.0) is False
        assert gate.check_stop("BTCUSDT", 0.0) is False

    def test_compute_stop_price_no_position(self, gate):
        assert gate.compute_stop_price("BTCUSDT", 100.0) == 0.0

    def test_compute_stop_price_long_initial_phase(self, gate):
        gate.on_new_position("BTCUSDT", 1, 100.0)
        stop = gate.compute_stop_price("BTCUSDT", 100.0)
        # Initial: entry * (1 - atr * initial_mult) with fallback ATR=0.015
        # = 100 * (1 - 0.015 * 2.0) = 100 * 0.97 = 97.0
        # But min distance (0.3% of 100 = 0.3) applies: 100 - 97 = 3 > 0.3 OK
        # Max loss floor: 100 * 0.95 = 95.0 → max(97.0, 95.0) = 97.0
        assert stop == pytest.approx(97.0, abs=0.01)

    def test_reset_symbol(self, gate):
        gate.on_new_position("BTCUSDT", 1, 100.0)
        gate.reset_symbol("BTCUSDT")
        assert gate.compute_stop_price("BTCUSDT", 100.0) == 0.0

    def test_get_phase_initial(self, gate):
        from strategy.gates.adaptive_stop_gate import StopPhase
        gate.on_new_position("BTCUSDT", 1, 100.0)
        assert gate.get_phase("BTCUSDT") == StopPhase.INITIAL

    def test_atr_buffer_updates(self, gate):
        gate.on_new_position("BTCUSDT", 1, 100.0)
        state = gate._states["BTCUSDT"]
        # Push enough ATR samples for meaningful ATR
        for i in range(20):
            state.push_true_range(102.0, 98.0, 100.0)
        assert len(state.atr_buffer) == 20
        atr = state.current_atr()
        assert atr > 0

    def test_symbol_state_reset(self):
        from strategy.gates.adaptive_stop_gate import _SymbolState
        s = _SymbolState()
        s.entry_price = 100.0
        s.side = 1
        s.peak_price = 105.0
        s.reset()
        assert s.entry_price == 0.0
        assert s.side == 0
        assert s.peak_price == 0.0

    def test_current_atr_insufficient_samples(self):
        from strategy.gates.adaptive_stop_gate import _SymbolState
        s = _SymbolState()
        # Less than 5 samples → fallback
        for _ in range(3):
            s.atr_buffer.append(0.01)
        assert s.current_atr(fallback=0.02) == 0.02


# ===========================================================================
# EquityLeverageGate
# ===========================================================================

class TestEquityLeverageGate:
    def test_default_small_equity_high_leverage(self):
        from strategy.gates.equity_leverage_gate import EquityLeverageGate
        gate = EquityLeverageGate(get_equity=lambda: 1000.0)
        result = gate.check(_ev(z_score=1.5), {"z_score": 1.5})
        assert result.allowed is True
        # equity 1000 → bracket 1.5x, z=1.5 → z_scale=1.0 → 1.5
        assert result.scale == pytest.approx(1.5)

    def test_large_equity_lower_leverage(self):
        from strategy.gates.equity_leverage_gate import EquityLeverageGate
        gate = EquityLeverageGate(get_equity=lambda: 30_000.0)
        result = gate.check(_ev(), {"z_score": 1.5})
        # equity 30K → bracket 1.0x, z=1.5 → z_scale=1.0 → 1.0
        assert result.scale == pytest.approx(1.0)

    def test_extreme_z_score(self):
        from strategy.gates.equity_leverage_gate import EquityLeverageGate
        gate = EquityLeverageGate(get_equity=lambda: 1000.0)
        result = gate.check(_ev(), {"z_score": 3.0})
        # z=3.0 → z_scale=1.5, equity=1000 → lev=1.5 → 2.25
        assert result.scale == pytest.approx(2.25)

    def test_weak_z_score(self):
        from strategy.gates.equity_leverage_gate import EquityLeverageGate
        gate = EquityLeverageGate(get_equity=lambda: 1000.0)
        result = gate.check(_ev(), {"z_score": 0.3})
        # z=0.3 → z_scale=0.5, lev=1.5 → 0.75
        assert result.scale == pytest.approx(0.75)

    def test_equity_from_context_fallback(self):
        from strategy.gates.equity_leverage_gate import EquityLeverageGate
        gate = EquityLeverageGate()  # no get_equity
        result = gate.check(_ev(), {"equity": 1000.0, "z_score": 1.5})
        assert result.scale == pytest.approx(1.5)

    def test_always_allowed(self):
        from strategy.gates.equity_leverage_gate import EquityLeverageGate
        gate = EquityLeverageGate(get_equity=lambda: 0.0)
        result = gate.check(_ev(), {})
        assert result.allowed is True


class TestZScale:
    def test_z_scale_brackets(self):
        from strategy.gates.equity_leverage_gate import _z_scale
        assert _z_scale(2.5) == 1.5
        assert _z_scale(1.5) == 1.0
        assert _z_scale(0.7) == 0.7
        assert _z_scale(0.2) == 0.5

    def test_z_scale_negative(self):
        from strategy.gates.equity_leverage_gate import _z_scale
        assert _z_scale(-2.5) == 1.5
        assert _z_scale(-0.3) == 0.5

    def test_z_scale_zero(self):
        from strategy.gates.equity_leverage_gate import _z_scale
        assert _z_scale(0.0) == 0.5


class TestBracketLeverage:
    def test_bracket_lookup(self):
        from strategy.gates.equity_leverage_gate import _bracket_leverage
        assert _bracket_leverage(100) == 1.5
        assert _bracket_leverage(10_000) == 1.5
        assert _bracket_leverage(25_000) == 1.0
        assert _bracket_leverage(100_000) == 1.0


# ===========================================================================
# ConsensusScalingGate
# ===========================================================================

class TestConsensusScalingGate:
    def test_flat_signal_returns_1(self):
        from strategy.gates.consensus_scaling_gate import ConsensusScalingGate
        gate = ConsensusScalingGate()
        result = gate.check(SimpleNamespace(symbol="BTCUSDT", signal=0), {"signal": 0})
        assert result.scale == 1.0

    def test_no_others_returns_1(self):
        from strategy.gates.consensus_scaling_gate import ConsensusScalingGate
        consensus = {"BTCUSDT": 1}
        gate = ConsensusScalingGate(consensus=consensus)
        ev = SimpleNamespace(symbol="BTCUSDT", signal=1)
        result = gate.check(ev, {"signal": 1})
        assert result.scale == 1.0

    def test_all_disagree_contrarian_boost(self):
        from strategy.gates.consensus_scaling_gate import ConsensusScalingGate
        consensus = {"BTCUSDT": 1, "ETHUSDT": -1, "SOLUSDT": -1}
        gate = ConsensusScalingGate(consensus=consensus)
        ev = SimpleNamespace(symbol="BTCUSDT", signal=1)
        result = gate.check(ev, {"signal": 1})
        assert result.scale == pytest.approx(1.3)

    def test_most_agree_returns_1(self):
        from strategy.gates.consensus_scaling_gate import ConsensusScalingGate
        consensus = {"BTCUSDT": 1, "ETHUSDT": 1, "SOLUSDT": 1, "DOGEUSDT": 1}
        gate = ConsensusScalingGate(consensus=consensus)
        ev = SimpleNamespace(symbol="BTCUSDT", signal=1)
        result = gate.check(ev, {"signal": 1})
        assert result.scale == 1.0

    def test_mixed_signals_reduced(self):
        from strategy.gates.consensus_scaling_gate import ConsensusScalingGate
        # 1 of 3 agree = 33% → 0.25-0.74 range → 0.7
        consensus = {"BTCUSDT": 1, "ETHUSDT": 1, "SOLUSDT": -1, "DOGEUSDT": -1}
        gate = ConsensusScalingGate(consensus=consensus)
        ev = SimpleNamespace(symbol="BTCUSDT", signal=1)
        result = gate.check(ev, {"signal": 1})
        assert result.scale == pytest.approx(0.7)


class TestConsensusScaleFunction:
    def test_no_active_others(self):
        from strategy.gates.consensus_scaling_gate import _consensus_scale
        assert _consensus_scale("BTC", 1, {"BTC": 1, "ETH": 0}) == 1.0

    def test_all_disagree(self):
        from strategy.gates.consensus_scaling_gate import _consensus_scale
        assert _consensus_scale("BTC", 1, {"BTC": 1, "ETH": -1, "SOL": -1}) == 1.3

    def test_low_agreement(self):
        from strategy.gates.consensus_scaling_gate import _consensus_scale
        # 0 of 4 agree → all disagree → 1.3
        result = _consensus_scale("BTC", 1, {"BTC": 1, "A": -1, "B": -1, "C": -1, "D": -1})
        assert result == 1.3


# ===========================================================================
# CarryCostGate
# ===========================================================================

class TestCarryCostGate:
    def test_disabled_returns_1(self):
        from strategy.gates.carry_cost_gate import CarryCostGate, CarryCostConfig
        gate = CarryCostGate(CarryCostConfig(enabled=False))
        result = gate.check(_ev_with_signal(signal=1), {"signal": 1})
        assert result.scale == 1.0

    def test_flat_signal_returns_1(self):
        from strategy.gates.carry_cost_gate import CarryCostGate
        gate = CarryCostGate()
        result = gate.check(_ev_with_signal(signal=0), {"signal": 0})
        assert result.scale == 1.0

    def test_favorable_carry_boosts(self):
        from strategy.gates.carry_cost_gate import CarryCostGate, CarryCostConfig
        cfg = CarryCostConfig(favorable_carry_pct=5.0, favorable_scale=1.15)
        gate = CarryCostGate(cfg)
        # Long with very negative funding → receive carry
        # funding_rate = -0.01 → annual = -0.01 * 3 * 365 * 100 = -1095%
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1, "funding_rate": -0.01, "basis": 0.0
        })
        assert result.scale == pytest.approx(1.15)

    def test_costly_carry_reduces(self):
        from strategy.gates.carry_cost_gate import CarryCostGate, CarryCostConfig
        cfg = CarryCostConfig(costly_carry_pct=10.0, costly_scale=0.7, extreme_carry_pct=30.0)
        gate = CarryCostGate(cfg)
        # Long with positive funding → pay carry
        # funding_rate = 0.001 → annual = 0.001 * 3 * 365 * 100 = 109.5%
        # net_carry_cost = 109.5 + 0 = 109.5 → extreme
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1, "funding_rate": 0.001, "basis": 0.0
        })
        # 109.5 > extreme(30) → extreme_scale
        assert result.scale == pytest.approx(0.4)

    def test_low_carry_no_change(self):
        from strategy.gates.carry_cost_gate import CarryCostGate
        gate = CarryCostGate()
        # Very small funding → neutral carry
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1, "funding_rate": 0.00001, "basis": 0.0
        })
        assert result.scale == 1.0


# ===========================================================================
# LiquidationCascadeGate
# ===========================================================================

class TestLiquidationCascadeGate:
    def test_disabled_returns_1(self):
        from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate, LiquidationCascadeConfig
        gate = LiquidationCascadeGate(LiquidationCascadeConfig(enabled=False))
        result = gate.check(_ev(), {"liquidation_volume_zscore_24": 5.0})
        assert result.allowed is True
        assert result.scale == 1.0

    def test_extreme_liq_zscore_blocks(self):
        from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
        gate = LiquidationCascadeGate()
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1, "liquidation_volume_zscore_24": 3.5
        })
        assert result.allowed is False

    def test_danger_plus_oi_unwind_blocks(self):
        from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
        gate = LiquidationCascadeGate()
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1,
            "liquidation_volume_zscore_24": 2.5,
            "oi_acceleration": -2.5,
        })
        assert result.allowed is False

    def test_danger_zscore_scales_down(self):
        from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
        gate = LiquidationCascadeGate()
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1,
            "liquidation_volume_zscore_24": 2.5,
            "oi_acceleration": 0.0,
        })
        assert result.allowed is True
        assert result.scale == pytest.approx(0.3)

    def test_caution_zscore_scales_down(self):
        from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
        gate = LiquidationCascadeGate()
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1,
            "liquidation_volume_zscore_24": 1.6,
            "oi_acceleration": 0.0,
        })
        assert result.allowed is True
        assert result.scale == pytest.approx(0.5)

    def test_normal_conditions_pass(self):
        from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
        gate = LiquidationCascadeGate()
        result = gate.check(_ev(), {
            "liquidation_volume_zscore_24": 0.5,
            "oi_acceleration": 0.0,
        })
        assert result.allowed is True
        assert result.scale == 1.0

    def test_contrarian_boost(self):
        from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
        gate = LiquidationCascadeGate()
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1,
            "liquidation_volume_zscore_24": 0.0,
            "oi_acceleration": 0.0,
            "liquidation_imbalance": -0.5,  # sell liquidations → go long = contrarian
        })
        assert result.allowed is True
        assert result.scale == pytest.approx(1.1)

    def test_stats_tracking(self):
        from strategy.gates.liquidation_cascade_gate import LiquidationCascadeGate
        gate = LiquidationCascadeGate()
        gate.check(_ev(), {"liquidation_volume_zscore_24": 0.0})
        gate.check(_ev(), {"liquidation_volume_zscore_24": 3.5, "signal": 1})
        stats = gate.stats
        assert stats["total_checks"] == 2
        assert stats["blocked"] == 1


# ===========================================================================
# MultiTFConfluenceGate
# ===========================================================================

class TestMultiTFConfluenceGate:
    def test_disabled_returns_1(self):
        from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate, MultiTFConfluenceConfig
        gate = MultiTFConfluenceGate(MultiTFConfluenceConfig(enabled=False))
        result = gate.check(_ev_with_signal(signal=1), {"signal": 1})
        assert result.scale == 1.0

    def test_flat_signal_returns_1(self):
        from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
        gate = MultiTFConfluenceGate()
        result = gate.check(_ev_with_signal(signal=0), {"signal": 0})
        assert result.scale == 1.0

    def test_aligned_trend_boosts(self):
        from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
        gate = MultiTFConfluenceGate()
        # Signal=1 (long), 4h indicators all bullish
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1,
            "tf4h_close_vs_ma20": 0.02,   # bullish
            "tf4h_rsi_14": 70.0,           # bullish
            "tf4h_macd_hist": 0.5,         # bullish
        })
        assert result.scale == pytest.approx(1.2)

    def test_opposed_trend_reduces(self):
        from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
        gate = MultiTFConfluenceGate()
        # Signal=1 (long), 4h indicators bearish
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1,
            "tf4h_close_vs_ma20": -0.02,
            "tf4h_rsi_14": 30.0,
            "tf4h_macd_hist": -0.5,
        })
        assert result.scale == pytest.approx(0.5)

    def test_neutral_4h_no_change(self):
        from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
        gate = MultiTFConfluenceGate()
        # Mixed 4h indicators — only 1 bullish, need 2
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1,
            "tf4h_close_vs_ma20": 0.02,   # bullish
            "tf4h_rsi_14": 50.0,           # neutral
            "tf4h_macd_hist": -0.1,        # bearish
        })
        assert result.scale == 1.0

    def test_no_4h_data_passes_through(self):
        from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
        gate = MultiTFConfluenceGate()
        result = gate.check(_ev_with_signal(signal=1), {"signal": 1})
        assert result.scale == 1.0

    def test_4h_model_signal_overrides_indicators(self):
        from strategy.gates.multi_tf_confluence_gate import MultiTFConfluenceGate
        gate = MultiTFConfluenceGate()
        # 4h model says bearish but indicators say bullish → model wins
        result = gate.check(_ev_with_signal(signal=1), {
            "signal": 1,
            "tf4h_model_signal": -1,
            "tf4h_close_vs_ma20": 0.02,
            "tf4h_rsi_14": 70.0,
            "tf4h_macd_hist": 0.5,
        })
        # model_signal=-1, signal=1 → opposed
        assert result.scale == pytest.approx(0.5)


# ===========================================================================
# GateEvaluator
# ===========================================================================

class TestGateEvaluator:
    def test_flat_signal_returns_1(self):
        from strategy.gates.evaluator import GateEvaluator
        evaluator = GateEvaluator(
            liq_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True)),
            mtf_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True)),
            carry_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True)),
            vpin_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True)),
        )
        scale = evaluator.evaluate(signal=0, feat_dict={}, consensus_signals={},
                                   runner_key="BTCUSDT", symbol="BTCUSDT")
        assert scale == 1.0

    def test_liq_gate_blocks(self):
        from strategy.gates.evaluator import GateEvaluator
        evaluator = GateEvaluator(
            liq_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=False, reason="cascade")),
            mtf_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True)),
            carry_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True)),
            vpin_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True)),
        )
        scale = evaluator.evaluate(signal=1, feat_dict={}, consensus_signals={},
                                   runner_key="BTCUSDT", symbol="BTCUSDT")
        assert scale == 0.0

    def test_scales_multiply(self):
        from strategy.gates.evaluator import GateEvaluator
        evaluator = GateEvaluator(
            liq_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True, scale=0.5)),
            mtf_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True, scale=0.8)),
            carry_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True, scale=1.0)),
            vpin_gate=SimpleNamespace(check=lambda ev, ctx: GateResult(allowed=True, scale=1.0)),
        )
        scale = evaluator.evaluate(signal=1, feat_dict={}, consensus_signals={},
                                   runner_key="BTCUSDT", symbol="BTCUSDT")
        assert scale == pytest.approx(0.4)
