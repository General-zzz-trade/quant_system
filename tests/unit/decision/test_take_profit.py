"""Tests for take-profit exit mechanisms.

Covers:
1. ATR trailing stop phase tuning (lowered floor, adjusted thresholds)
2. Profit-lock trailing (cap giveback at 50% of peak profit)
3. Z-fade profit exit (signal weakening while profitable)
4. Max hold enforcement
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from decision.modules.alpha import AlphaDecisionModule


# ── helpers ──────────────────────────────────────────────────────


def _make_module(
    symbol: str = "BTCUSDT",
    runner_key: str = "BTCUSDT",
    deadzone: float = 2.0,
    max_hold: int = 60,
) -> AlphaDecisionModule:
    predictor = MagicMock()
    predictor.predict.return_value = 0.5

    discretizer = MagicMock()
    discretizer.discretize.return_value = (1, 2.0)
    discretizer.deadzone = deadzone
    discretizer.min_hold = 6
    discretizer.max_hold = max_hold

    sizer = MagicMock()
    sizer.target_qty.return_value = Decimal("0.1")
    sizer.min_size = Decimal("0.001")

    mod = AlphaDecisionModule(
        symbol=symbol,
        runner_key=runner_key,
        predictor=predictor,
        discretizer=discretizer,
        sizer=sizer,
    )
    return mod


def _set_position(
    mod: AlphaDecisionModule,
    signal: int = 1,
    entry_price: float = 66000.0,
    trade_peak: float = 66000.0,
    atr_pct: float = 0.004,
    bars_held: int = 5,
    vol_ratio: float = 1.0,
):
    """Simulate a held position with given state.

    vol_ratio controls the adaptive vol_factor:
        0.5 → low-vol regime (tighter exits)
        1.0 → baseline
        2.0 → high-vol regime (wider exits)
    """
    mod._signal = signal
    mod._entry_price = entry_price
    mod._trade_peak = trade_peak
    mod._current_qty = Decimal("0.1")
    mod._closes = [entry_price] * 30
    mod._atr_buffer = [atr_pct] * 20
    mod._bars_processed = 800 + bars_held
    mod._last_trade_bar = 800
    # Seed vol_history so _vol_factor() returns vol_ratio
    target_vol = mod._vol_median * vol_ratio
    mod._vol_history = [target_vol] * 30


# ── ATR trailing stop tuning ────────────────────────────────────


class TestATRTrailingTuning:
    """ATR trailing stop with adjusted parameters."""

    def test_trailing_phase_enters_at_1x_atr(self):
        """Trailing phase activates at profit >= 1.0×ATR."""
        mod = _make_module()
        atr = 0.004  # 0.4%
        entry = 66000.0
        # Peak profit = 0.4% = 1.0 × ATR
        peak = entry * (1 + 1.0 * atr)
        _set_position(mod, entry_price=entry, trade_peak=peak, atr_pct=atr)

        # Close at peak (0% drawdown) — should not exit
        ok, reason = mod._check_force_exits(peak, 1.5)
        assert not ok

        # Drawdown of 0.35% from peak — triggers trailing (floor=0.3%)
        close = peak * (1 - 0.0035)
        ok, reason = mod._check_force_exits(close, 1.5)
        assert ok
        assert "atr_stop" in reason

    def test_floor_is_0_3_pct(self):
        """Floor is 0.3% — validated by backtest."""
        mod = _make_module()
        entry = 66000.0
        peak = entry * 1.005  # +0.5% profit (in trailing phase)
        _set_position(mod, entry_price=entry, trade_peak=peak, atr_pct=0.004)

        # 0.25% drawdown from peak — should NOT trigger (below 0.3% floor)
        close = peak * (1 - 0.0025)
        ok, reason = mod._check_force_exits(close, 1.5)
        assert not ok or "atr_stop" not in reason

        # 0.35% drawdown from peak — should trigger
        close2 = peak * (1 - 0.0035)
        ok2, reason2 = mod._check_force_exits(close2, 1.5)
        assert ok2
        assert "atr_stop" in reason2

    def test_initial_phase_wide_stop(self):
        """Initial phase (low profit) has wide stop = ATR × 1.2."""
        mod = _make_module()
        entry = 66000.0
        _set_position(mod, entry_price=entry, trade_peak=entry, atr_pct=0.004)

        # 0.35% drawdown — below initial stop (0.4% * 1.2 = 0.48%)
        close = entry * (1 - 0.0035)
        ok, reason = mod._check_force_exits(close, 1.5)
        assert not ok


# ── Profit-lock trailing ────────────────────────────────────────


class TestProfitLock:
    """Profit-lock: cap giveback at 50% of peak profit when profit > 2×ATR."""

    def test_profit_lock_tightens_stop_for_large_winners(self):
        """Profit-lock overrides ATR trailing when 50% of profit < ATR stop_dist.

        Low-vol scenario: ATR=0.1%, floor=0.3% dominates.
        profit = 0.4% >= 2×ATR (0.2%) → profit_lock active.
        stop_dist = max(ATR*0.2, 0.3%) = 0.3%
        max_giveback = 0.4% * 0.5 = 0.2% < 0.3% → profit_lock overrides!
        """
        mod = _make_module()
        atr = 0.001  # very low vol: 0.1%
        entry = 66000.0
        # profit = 0.4% = 4×ATR (above 2×ATR threshold)
        peak = entry * 1.004
        _set_position(mod, entry_price=entry, trade_peak=peak, atr_pct=atr)

        # stop_dist = max(0.001*0.2, 0.003) = 0.3% (floor)
        # profit_lock max_giveback = 0.004 * 0.5 = 0.002 = 0.20%
        # 0.20% < 0.30% → profit_lock overrides to 0.20%!
        # Drawdown at 0.25% from peak → triggers profit_lock (>0.20%)
        close = peak * (1 - 0.0025)  # 0.25% drawdown
        ok, reason = mod._check_force_exits(close, 1.5)
        assert ok
        assert "profit_lock" in reason

    def test_profit_lock_does_not_fire_below_threshold(self):
        """Profit < 2×ATR: profit-lock should not activate."""
        mod = _make_module()
        atr = 0.004
        entry = 66000.0
        # Peak profit = 0.5% = 1.25×ATR (below 2×ATR)
        peak = entry * 1.005
        _set_position(mod, entry_price=entry, trade_peak=peak, atr_pct=atr)

        # Give back 60% — but profit_lock threshold not met
        close = entry * 1.002
        ok, reason = mod._check_force_exits(close, 1.5)
        # May trigger atr_stop but NOT profit_lock
        if ok:
            assert "profit_lock" not in reason

    def test_profit_lock_no_override_when_atr_tighter(self):
        """When ATR stop_dist < 50% of profit, ATR stop fires normally."""
        mod = _make_module()
        atr = 0.004
        entry = 66000.0
        peak = entry * 1.010  # +1.0% = 2.5×ATR
        _set_position(mod, entry_price=entry, trade_peak=peak, atr_pct=atr)

        # stop_dist = max(0.004*0.2, 0.003) = 0.3% (floor)
        # profit_lock max_giveback = 1.0% * 0.5 = 0.5%
        # 0.5% > 0.3% → no override, ATR stop fires
        close = peak * (1 - 0.004)  # 0.4% drawdown > 0.3% stop
        ok, reason = mod._check_force_exits(close, 1.5)
        assert ok
        assert "atr_stop" in reason


# ── Z-fade profit exit ──────────────────────────────────────────


class TestZFadeTakeProfit:
    """Z-fade: exit when signal weakens and position is profitable."""

    def test_z_fade_triggers_when_profitable_and_z_weak(self):
        """Profit > ATR and z < 0.5×deadzone → z_fade_tp."""
        mod = _make_module(deadzone=2.0)
        atr = 0.004
        entry = 66000.0
        # Price above entry by 0.5% (> ATR=0.4%)
        close = entry * 1.005
        _set_position(mod, entry_price=entry, trade_peak=close, atr_pct=atr)

        # z = 0.8 < 0.5 * dz(2.0) = 1.0 → signal is weakening
        ok, reason = mod._check_force_exits(close, 0.8)
        assert ok
        assert "z_fade_tp" in reason

    def test_z_fade_does_not_fire_when_z_strong(self):
        """If z is still strong (> 0.5×dz), no z_fade exit."""
        mod = _make_module(deadzone=2.0)
        entry = 66000.0
        close = entry * 1.005
        _set_position(mod, entry_price=entry, trade_peak=close, atr_pct=0.004)

        # z = 1.5 > 0.5 * dz(2.0) = 1.0 → signal still strong
        ok, reason = mod._check_force_exits(close, 1.5)
        assert not ok or "z_fade_tp" not in reason

    def test_z_fade_does_not_fire_when_not_profitable(self):
        """If not in profit (close < entry), no z_fade exit."""
        mod = _make_module(deadzone=2.0)
        entry = 66000.0
        close = entry * 0.999  # small loss
        _set_position(mod, entry_price=entry, trade_peak=entry, atr_pct=0.004)

        ok, reason = mod._check_force_exits(close, 0.3)
        if ok:
            assert "z_fade_tp" not in reason

    def test_z_fade_with_eth_deadzone(self):
        """ETH has lower deadzone=1.2 → 0.5×dz = 0.6."""
        mod = _make_module(symbol="ETHUSDT", runner_key="ETHUSDT", deadzone=1.2)
        entry = 2000.0
        close = entry * 1.008  # +0.8% profit
        _set_position(mod, entry_price=entry, trade_peak=close, atr_pct=0.005)

        # z = 0.5 < 0.5 * dz(1.2) = 0.6 → trigger
        ok, reason = mod._check_force_exits(close, 0.5)
        assert ok
        assert "z_fade_tp" in reason


# ── Max hold ────────────────────────────────────────────────────


class TestMaxHold:
    """Max hold: force exit after max_hold bars."""

    def test_max_hold_triggers(self):
        """Position held > max_hold bars → forced exit."""
        mod = _make_module(max_hold=60)
        _set_position(mod, bars_held=65)

        # Close near entry, z still positive → no other exit reason
        ok, reason = mod._check_force_exits(mod._entry_price, 1.5)
        assert ok
        assert "max_hold" in reason
        assert "65>=60" in reason

    def test_max_hold_does_not_fire_early(self):
        """Position held < max_hold bars → no exit."""
        mod = _make_module(max_hold=60)
        _set_position(mod, bars_held=30)

        ok, reason = mod._check_force_exits(mod._entry_price, 1.5)
        assert not ok or "max_hold" not in reason

    def test_max_hold_zero_disabled(self):
        """max_hold=0 means disabled."""
        mod = _make_module(max_hold=0)
        _set_position(mod, bars_held=9999)

        ok, reason = mod._check_force_exits(mod._entry_price, 1.5)
        assert not ok or "max_hold" not in reason

    def test_max_hold_lower_priority_than_signal_exits(self):
        """Signal-based exits should fire before max_hold."""
        mod = _make_module(max_hold=60)
        _set_position(mod, bars_held=65)

        # z_reversal condition: z = -0.5 while long
        ok, reason = mod._check_force_exits(mod._entry_price, -0.5)
        assert ok
        # z_reversal should fire first (higher priority)
        assert "z_reversal" in reason


# ── Integration: exit priority ──────────────────────────────────


class TestExitPriority:
    """Verify exit mechanisms fire in correct priority order."""

    def test_atr_stop_before_profit_lock(self):
        """ATR stop is checked before profit-lock."""
        mod = _make_module()
        entry = 66000.0
        peak = entry * 1.015  # +1.5%
        _set_position(mod, entry_price=entry, trade_peak=peak, atr_pct=0.004)

        # Large drawdown triggers both ATR stop and profit-lock
        close = entry * 1.001
        ok, reason = mod._check_force_exits(close, 1.5)
        assert ok
        # ATR stop fires first
        assert "atr_stop" in reason

    def test_quick_loss_still_works(self):
        """Quick loss exit unchanged by new take-profit mechanisms."""
        mod = _make_module()
        entry = 66000.0
        _set_position(mod, entry_price=entry, trade_peak=entry, atr_pct=0.004)

        # 2.5% adverse move (> ceiling 2%)
        close = entry * 0.975
        ok, reason = mod._check_force_exits(close, 1.5)
        assert ok
        # Should be quick_loss or atr_stop
        assert "quick_loss" in reason or "atr_stop" in reason


# ── Adaptive vol_factor tests ───────────────────────────────────


class TestVolFactorAdaptive:
    """Exit parameters should adapt to market volatility regime."""

    def test_vol_factor_baseline(self):
        """vol_ratio=1.0 → vf=1.0, baseline parameters."""
        mod = _make_module()
        _set_position(mod, vol_ratio=1.0)
        vf = mod._vol_factor()
        assert 0.95 <= vf <= 1.05

    def test_vol_factor_low_vol(self):
        """vol_ratio=0.5 → vf clamped to 0.5."""
        mod = _make_module()
        _set_position(mod, vol_ratio=0.5)
        vf = mod._vol_factor()
        assert 0.45 <= vf <= 0.55

    def test_vol_factor_high_vol(self):
        """vol_ratio=2.0 → vf clamped to 2.0."""
        mod = _make_module()
        _set_position(mod, vol_ratio=2.0)
        vf = mod._vol_factor()
        assert 1.9 <= vf <= 2.05

    def test_vol_factor_clamp_floor(self):
        """Extreme low vol (ratio=0.1) clamped to 0.5."""
        mod = _make_module()
        _set_position(mod, vol_ratio=0.1)
        vf = mod._vol_factor()
        assert vf == 0.5

    def test_vol_factor_clamp_ceiling(self):
        """Extreme high vol (ratio=5.0) clamped to 2.0."""
        mod = _make_module()
        _set_position(mod, vol_ratio=5.0)
        vf = mod._vol_factor()
        assert vf == 2.0

    def test_ic_red_forces_conservative(self):
        """IC RED (scale=0.4) forces vf ≤ 0.7 even in high-vol."""
        mod = _make_module()
        _set_position(mod, vol_ratio=2.0)
        mod._ic_scale = 0.4  # RED
        vf = mod._vol_factor()
        assert vf <= 0.7

    def test_low_vol_tighter_trailing_stop(self):
        """Low-vol: trailing stop triggers at smaller drawdown."""
        mod = _make_module()
        entry = 66000.0
        peak = entry * 1.005  # +0.5% profit, in trailing phase
        atr = 0.004

        # Low-vol: floor = 0.001 + 0.002*0.5 = 0.002 (0.2%)
        _set_position(mod, entry_price=entry, trade_peak=peak,
                      atr_pct=atr, vol_ratio=0.5)
        close_low = peak * (1 - 0.0025)  # 0.25% drawdown
        ok_low, _ = mod._check_force_exits(close_low, 1.5)

        # High-vol: floor = 0.001 + 0.002*2.0 = 0.005 (0.5%)
        _set_position(mod, entry_price=entry, trade_peak=peak,
                      atr_pct=atr, vol_ratio=2.0)
        close_high = peak * (1 - 0.0025)  # same 0.25% drawdown
        ok_high, _ = mod._check_force_exits(close_high, 1.5)

        # Low-vol should exit, high-vol should hold
        assert ok_low, "Low-vol should trigger at 0.25% drawdown"
        assert not ok_high, "High-vol should hold at 0.25% drawdown"

    def test_high_vol_wider_quick_loss_ceiling(self):
        """High-vol quick_loss ceiling is wider than low-vol."""
        mod = _make_module()
        # Verify the ceiling formula:
        #   vf=0.5: ceiling = 0.01 + 0.01*0.5 = 1.5%
        #   vf=1.0: ceiling = 0.01 + 0.01*1.0 = 2.0%
        #   vf=2.0: ceiling = 0.01 + 0.01*2.0 = 3.0%
        _set_position(mod, vol_ratio=0.5)
        vf_low = mod._vol_factor()
        ceil_low = 0.01 + 0.01 * vf_low

        _set_position(mod, vol_ratio=2.0)
        vf_high = mod._vol_factor()
        ceil_high = 0.01 + 0.01 * vf_high

        assert ceil_low < ceil_high, "High-vol ceiling should be wider"
        assert abs(ceil_low - 0.015) < 0.002
        assert abs(ceil_high - 0.03) < 0.002

    def test_max_hold_scales_with_vol(self):
        """max_hold is longer in high-vol (vf×base)."""
        mod = _make_module(max_hold=60)

        # Low-vol (vf=0.5): max_hold = 60*0.5 = 30
        _set_position(mod, bars_held=40, vol_ratio=0.5)
        ok_low, reason_low = mod._check_force_exits(mod._entry_price, 1.5)

        # High-vol (vf=2.0): max_hold = 60*2.0 = 120
        _set_position(mod, bars_held=40, vol_ratio=2.0)
        ok_high, reason_high = mod._check_force_exits(mod._entry_price, 1.5)

        assert ok_low and "max_hold" in reason_low, "Low-vol: 40 >= 30 triggers"
        assert not ok_high or "max_hold" not in reason_high, "High-vol: 40 < 120 holds"
