# tests/unit/execution/test_advanced_algos.py
"""Tests for volume profile, adaptive TWAP, and Almgren-Chriss impact model."""
from __future__ import annotations

from decimal import Decimal

import pytest

from execution.algos.volume_profile import IntraDayVolumeProfile, VolumeBar
from execution.algos.adaptive_twap import (
    AdaptiveTWAPAlgo,
    AdaptiveTWAPConfig,
    MarketSnapshot,
)
from execution.algos.impact_model import (
    AlmgrenChrissModel,
    AlmgrenChrissParams,
    ImpactEstimate,
)


# ── IntraDayVolumeProfile tests ─────────────────────────────


class TestIntraDayVolumeProfile:
    def test_crypto_24h_normalized(self):
        profile = IntraDayVolumeProfile.crypto_24h()
        total = sum(b.weight for b in profile.bars)
        assert total == pytest.approx(1.0, abs=1e-6)
        assert len(profile.bars) == 24

    def test_uniform_profile(self):
        profile = IntraDayVolumeProfile.uniform()
        weights = [b.weight for b in profile.bars]
        assert all(w == pytest.approx(1 / 24) for w in weights)

    def test_from_historical(self):
        vols = [float(i + 1) for i in range(24)]
        profile = IntraDayVolumeProfile.from_historical(vols)
        assert len(profile.bars) == 24
        total = sum(b.weight for b in profile.bars)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_from_historical_wrong_length(self):
        with pytest.raises(ValueError, match="24"):
            IntraDayVolumeProfile.from_historical([1.0] * 12)

    def test_get_weight(self):
        profile = IntraDayVolumeProfile.uniform()
        assert profile.get_weight(0) == pytest.approx(1 / 24)
        assert profile.get_weight(12) == pytest.approx(1 / 24)

    def test_get_weights_normalized(self):
        profile = IntraDayVolumeProfile.crypto_24h()
        weights = profile.get_weights(n_slices=10, start_hour=8, duration_hours=8)
        assert len(weights) == 10
        assert sum(weights) == pytest.approx(1.0, abs=1e-6)

    def test_get_weights_zero_slices(self):
        profile = IntraDayVolumeProfile.uniform()
        assert profile.get_weights(n_slices=0) == []

    def test_peak_hours(self):
        profile = IntraDayVolumeProfile.crypto_24h()
        peaks = profile.peak_hours
        assert len(peaks) > 0
        # US session peak around 14-16 UTC should be peak
        assert any(14 <= h <= 16 for h in peaks)

    def test_crypto_has_session_peaks(self):
        profile = IntraDayVolumeProfile.crypto_24h()
        # Europe open (8-10 UTC) should be higher than overnight (22-24)
        europe_weight = profile.get_weight(9)
        overnight_weight = profile.get_weight(22)
        assert europe_weight > overnight_weight


# ── AdaptiveTWAPAlgo tests ──────────────────────────────────


class TestAdaptiveTWAPAlgo:
    @pytest.fixture
    def submit_fn(self):
        def _submit(symbol, side, qty):
            return Decimal("50000")
        return _submit

    def test_create_order(self, submit_fn):
        algo = AdaptiveTWAPAlgo(submit_fn=submit_fn)
        order = algo.create("BTCUSDT", "buy", Decimal("1.0"), n_slices=5)
        assert order.n_slices == 5
        assert len(order.slices) == 5
        total = sum(s.qty for s in order.slices)
        assert total == Decimal("1.0")

    def test_volume_weighted_slices(self, submit_fn):
        algo = AdaptiveTWAPAlgo(submit_fn=submit_fn)
        order = algo.create("BTCUSDT", "buy", Decimal("10.0"), n_slices=10)
        qtys = [s.qty for s in order.slices]
        total = sum(qtys)
        assert total == Decimal("10.0")
        assert len(qtys) == 10

    def test_tick_without_market_data(self, submit_fn):
        import time
        algo = AdaptiveTWAPAlgo(submit_fn=submit_fn)
        order = algo.create("BTCUSDT", "buy", Decimal("1.0"), n_slices=3)
        # Set all slices to be due now
        now = time.monotonic()
        for i, s in enumerate(order.slices):
            order.slices[i] = s.__class__(
                slice_idx=s.slice_idx, qty=s.qty,
                scheduled_at=now - 1,
            )

        result = algo.adaptive_tick(order, market=None)
        assert result is not None
        assert result.status == "executed"

    def test_wide_spread_reduces_qty(self, submit_fn):
        algo = AdaptiveTWAPAlgo(
            submit_fn=submit_fn,
            cfg=AdaptiveTWAPConfig(spread_threshold_bps=5.0),
        )
        base_qty = Decimal("1.0")
        market = MarketSnapshot(
            bid=50000, ask=50050,
            spread_bps=20.0,  # 4x threshold
            recent_volatility=0.001,
        )
        adjusted = algo._adapt_qty(base_qty, market)
        assert adjusted < base_qty

    def test_high_volume_increases_qty(self, submit_fn):
        algo = AdaptiveTWAPAlgo(submit_fn=submit_fn)
        base_qty = Decimal("1.0")
        market = MarketSnapshot(
            bid=50000, ask=50001,
            spread_bps=1.0,
            recent_volatility=0.001,
            volume_ratio=2.0,  # 2x expected
        )
        adjusted = algo._adapt_qty(base_qty, market)
        assert adjusted > base_qty

    def test_no_market_data_no_change(self, submit_fn):
        algo = AdaptiveTWAPAlgo(submit_fn=submit_fn)
        qty = Decimal("1.0")
        assert algo._adapt_qty(qty, None) == qty

    def test_high_volatility_reduces_qty(self, submit_fn):
        algo = AdaptiveTWAPAlgo(
            submit_fn=submit_fn,
            cfg=AdaptiveTWAPConfig(vol_threshold=0.005),
        )
        market = MarketSnapshot(
            bid=50000, ask=50001,
            spread_bps=1.0,
            recent_volatility=0.02,  # 4x threshold
        )
        adjusted = algo._adapt_qty(Decimal("1.0"), market)
        assert adjusted < Decimal("1.0")


# ── AlmgrenChrissModel tests ───────────────────────────────


class TestAlmgrenChrissModel:
    def test_default_params(self):
        model = AlmgrenChrissModel()
        assert model.params.sigma == 0.02

    def test_estimate_impact(self):
        model = AlmgrenChrissModel()
        est = model.estimate_impact(
            total_qty=10.0,
            price=50000.0,
        )
        assert est.permanent_impact_bps >= 0
        assert est.temporary_impact_bps >= 0
        assert est.total_impact_bps >= 0
        assert est.optimal_n_slices >= 3
        assert est.optimal_duration_sec > 0

    def test_larger_order_higher_impact(self):
        model = AlmgrenChrissModel()
        small = model.estimate_impact(total_qty=1.0, price=50000.0)
        large = model.estimate_impact(total_qty=100.0, price=50000.0)
        assert large.permanent_impact_bps >= small.permanent_impact_bps

    def test_custom_duration(self):
        model = AlmgrenChrissModel()
        est = model.estimate_impact(
            total_qty=10.0,
            price=50000.0,
            duration_sec=300,
            n_slices=5,
        )
        assert est.optimal_n_slices == 5
        assert est.optimal_duration_sec == 300.0

    def test_optimal_trajectory(self):
        model = AlmgrenChrissModel()
        traj = model.optimal_trajectory(
            total_qty=10.0,
            n_slices=5,
            duration_sec=300,
        )
        assert len(traj) == 5
        assert sum(traj) == pytest.approx(10.0, rel=1e-3)
        assert all(q >= 0 for q in traj)

    def test_trajectory_front_loaded(self):
        """Higher risk aversion → more front-loaded execution."""
        model = AlmgrenChrissModel(
            AlmgrenChrissParams(risk_aversion=1e-3),
        )
        traj = model.optimal_trajectory(10.0, 10, 600)
        # Front-loaded: first slice should be larger than last
        assert traj[0] > traj[-1]

    def test_trajectory_zero_slices(self):
        model = AlmgrenChrissModel()
        assert model.optimal_trajectory(10.0, 0, 300) == []

    def test_low_risk_aversion_uniform(self):
        """Low risk aversion → near-uniform trajectory."""
        model = AlmgrenChrissModel(
            AlmgrenChrissParams(risk_aversion=1e-15),
        )
        traj = model.optimal_trajectory(10.0, 5, 300)
        # Should be roughly uniform
        avg = sum(traj) / len(traj)
        assert all(abs(q - avg) < avg * 0.2 for q in traj)

    def test_high_vol_faster_execution(self):
        """Higher volatility = higher timing risk → execute faster."""
        low_vol = AlmgrenChrissModel(AlmgrenChrissParams(sigma=0.01))
        high_vol = AlmgrenChrissModel(AlmgrenChrissParams(sigma=0.05))

        est_low = low_vol.estimate_impact(10.0, 50000.0)
        est_high = high_vol.estimate_impact(10.0, 50000.0)

        # Higher vol → faster execution (shorter duration)
        assert est_high.optimal_duration_sec <= est_low.optimal_duration_sec
