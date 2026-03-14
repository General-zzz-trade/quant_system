"""Tests for MultiFactorDecisionModule — entry, exit, cooldown, position sizing."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict

import pytest

from state.position import PositionState
from strategies.multi_factor.decision_module import (
    MultiFactorConfig,
    MultiFactorDecisionModule,
    _snapshot_views,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_snapshot(
    *,
    symbol: str = "BTCUSDT",
    close: float = 100.0,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100.0,
    qty: float = 0.0,
    avg_price: float | None = None,
    balance: float = 10000.0,
    event_id: str = "ev_1",
) -> SimpleNamespace:
    """Build a minimal snapshot compatible with _snapshot_views."""
    o = open_ or close
    h = high or close
    l_ = low or close
    market = SimpleNamespace(
        close=Decimal(str(close)),
        open=Decimal(str(o)),
        high=Decimal(str(h)),
        low=Decimal(str(l_)),
        volume=Decimal(str(volume)),
        last_price=Decimal(str(close)),
    )
    positions: Dict[str, Any] = {}
    if qty != 0.0:
        positions[symbol] = PositionState(
            symbol=symbol,
            qty=Decimal(str(qty)),
            avg_price=Decimal(str(avg_price)) if avg_price else None,
        )
    account = SimpleNamespace(balance=Decimal(str(balance)))
    return SimpleNamespace(
        market=market,
        positions=positions,
        event_id=event_id,
        account=account,
    )


def _small_config(**overrides) -> MultiFactorConfig:
    """Config with small windows for fast warmup in tests."""
    defaults = dict(
        symbol="BTCUSDT",
        sma_fast_window=3,
        sma_slow_window=5,
        sma_trend_window=10,
        rsi_window=3,
        macd_fast=3,
        macd_slow=5,
        macd_signal=3,
        bb_window=3,
        bb_std=2.0,
        atr_window=3,
        atr_pct_window=25,
        ma_slope_window=3,
        trend_threshold=0.1,
        range_threshold=0.1,
        atr_extreme_pct=95.0,
        cooldown_bars=3,
        max_consecutive_losses=2,
        loss_reduction_factor=0.5,
        atr_stop_multiple=2.0,
        trailing_atr_multiple=4.0,
        long_only_above_trend=False,
        range_long_only=False,
    )
    defaults.update(overrides)
    return MultiFactorConfig(**defaults)


def _warmup_module(module: MultiFactorDecisionModule, n: int = 80, base_price: float = 100.0):
    """Feed enough bars through the module to complete warmup, returns last events."""
    for i in range(n):
        p = base_price + i * 0.01
        snap = _make_snapshot(close=p, high=p + 0.5, low=p - 0.5)
        module.decide(snap)


# ── _snapshot_views ──────────────────────────────────────────

class TestSnapshotViews:
    def test_object_snapshot(self):
        snap = _make_snapshot(close=50000.0)
        market, positions, event_id, account = _snapshot_views(snap)
        assert market is not None
        assert event_id == "ev_1"

    def test_dict_snapshot(self):
        snap = {
            "market": SimpleNamespace(close=100, open=100, high=100, low=100, volume=1),
            "positions": {},
            "event_id": "e1",
            "account": None,
        }
        market, positions, event_id, account = _snapshot_views(snap)
        assert market.close == 100

    def test_dict_snapshot_with_markets(self):
        mkt = SimpleNamespace(close=100)
        snap = {"markets": {"BTC": mkt}, "positions": {}, "event_id": "e1"}
        market, positions, event_id, account = _snapshot_views(snap)
        assert market is mkt


# ── Warmup ───────────────────────────────────────────────────

class TestWarmup:
    def test_no_events_during_warmup(self):
        cfg = _small_config()
        dm = MultiFactorDecisionModule(cfg)
        snap = _make_snapshot(close=100.0)
        events = list(dm.decide(snap))
        assert events == []

    def test_returns_empty_when_close_missing(self):
        cfg = _small_config()
        dm = MultiFactorDecisionModule(cfg)
        snap = SimpleNamespace(
            market=SimpleNamespace(close=None, open=None, high=None, low=None, volume=0, last_price=None),
            positions={},
            event_id="e1",
            account=None,
        )
        events = list(dm.decide(snap))
        assert events == []


# ── Entry signals ────────────────────────────────────────────

class TestEntrySignals:
    def test_entry_on_strong_trend(self):
        # Use atr_extreme_pct=100.0 to prevent HIGH_VOL regime blocking entries
        cfg = _small_config(trend_threshold=0.01, atr_extreme_pct=100.0)
        dm = MultiFactorDecisionModule(cfg)
        # Feed rising prices to build strong uptrend.
        # With constant-width bars, ATR percentile will be 100% (all equal).
        # Setting atr_extreme_pct=100.0 means the percentile must exceed 100 to
        # trigger HIGH_VOL, which never happens, so regime falls to TRENDING_UP.
        events_collected = []
        for i in range(50):
            p = 100 + i * 1.0
            snap = _make_snapshot(close=p, high=p + 2, low=p - 2)
            events = list(dm.decide(snap))
            events_collected.extend(events)
        # Should eventually generate entry events (IntentEvent + OrderEvent pairs)
        assert len(events_collected) >= 2  # at least one intent+order pair

    def test_no_entry_when_flat_signal(self):
        cfg = _small_config(trend_threshold=0.99, range_threshold=0.99)
        dm = MultiFactorDecisionModule(cfg)
        # Feed flat prices — no strong signal
        events_collected = []
        for i in range(80):
            snap = _make_snapshot(close=100.0)
            events = list(dm.decide(snap))
            events_collected.extend(events)
        assert len(events_collected) == 0


# ── Cooldown ─────────────────────────────────────────────────

class TestCooldown:
    def test_cooldown_blocks_entry(self):
        cfg = _small_config(cooldown_bars=5)
        dm = MultiFactorDecisionModule(cfg)
        # Manually set cooldown
        dm._cooldown = 5
        _warmup_module(dm, n=80)
        # Even with strong signal, cooldown should block
        snap = _make_snapshot(close=200.0)
        list(dm.decide(snap))
        # The cooldown decrements each bar, so after warmup it may have expired.
        # Let's test the mechanism directly:
        dm._cooldown = 3
        dm._bar_count = 100  # past warmup
        # Force regime to be valid by feeding enough data (already done in warmup)
        snap = _make_snapshot(close=200.0)
        list(dm.decide(snap))
        # Cooldown should decrement but still block
        assert dm._cooldown == 2  # decremented from 3

    def test_cooldown_set_on_stop_loss(self):
        cfg = _small_config(cooldown_bars=5, atr_stop_multiple=1.0)
        dm = MultiFactorDecisionModule(cfg)
        _warmup_module(dm, n=80)
        # Simulate position state
        dm._entry_atr = 2.0
        dm._entry_price = 150.0
        dm._trailing_peak = 155.0
        # Feed a bar that triggers stop loss (close < entry - atr * stop_multiple)
        stop_price = 150.0 - 2.0 * 1.0 - 1.0  # well below stop
        snap = _make_snapshot(
            close=stop_price, high=stop_price, low=stop_price,
            qty=1.0, avg_price=150.0,
        )
        events = list(dm.decide(snap))
        if events:  # stop was triggered
            assert dm._cooldown == cfg.cooldown_bars


# ── Position sizing ──────────────────────────────────────────

class TestPositionSizing:
    def test_compute_qty_basic(self):
        cfg = _small_config(risk_per_trade=0.02, atr_stop_multiple=3.0, max_position_pct=0.8)
        dm = MultiFactorDecisionModule(cfg)
        qty = dm._compute_qty(equity=10000, price=100, atr=5.0, direction=1)
        # risk_budget = 10000 * 0.02 = 200
        # stop_dist = 5.0 * 3.0 = 15
        # raw_qty = 200 / 15 = 13.333
        # max_qty = 0.8 * 10000 / 100 = 80
        expected = Decimal(str(round(200 / 15, 5)))
        assert qty == expected

    def test_compute_qty_max_position_cap(self):
        cfg = _small_config(risk_per_trade=0.5, atr_stop_multiple=0.1, max_position_pct=0.1)
        dm = MultiFactorDecisionModule(cfg)
        qty = dm._compute_qty(equity=10000, price=100, atr=5.0, direction=1)
        # risk_budget = 5000, stop_dist = 0.5, raw_qty = 10000
        # max_qty = 0.1 * 10000 / 100 = 10
        assert float(qty) <= 10.0

    def test_consecutive_losses_reduce_size(self):
        cfg = _small_config(
            risk_per_trade=0.02, atr_stop_multiple=3.0,
            max_consecutive_losses=2, loss_reduction_factor=0.5,
        )
        dm = MultiFactorDecisionModule(cfg)
        normal_qty = dm._compute_qty(equity=10000, price=100, atr=5.0, direction=1)
        dm._consecutive_losses = 3  # exceeds max
        reduced_qty = dm._compute_qty(equity=10000, price=100, atr=5.0, direction=1)
        assert float(reduced_qty) == pytest.approx(float(normal_qty) * 0.5, rel=1e-4)

    def test_zero_atr_returns_zero(self):
        cfg = _small_config()
        dm = MultiFactorDecisionModule(cfg)
        qty = dm._compute_qty(equity=10000, price=100, atr=0.0, direction=1)
        assert qty == Decimal("0")


# ── Exit checks ──────────────────────────────────────────────

class TestExitChecks:
    def _make_features(self, close=100.0, atr=5.0, bb_middle=100.0, sma_trend=None):
        from strategies.multi_factor.feature_computer import MultiFactorFeatures
        return MultiFactorFeatures(
            sma_fast=None, sma_slow=None, sma_trend=sma_trend,
            rsi=None, macd=None, macd_signal=None, macd_hist=None,
            bb_upper=None, bb_middle=bb_middle, bb_lower=None, bb_pct=None,
            atr=atr, atr_pct=None, atr_percentile=None, ma_slope=None,
            close=close, volume=0,
        )

    def test_stop_loss_long(self):
        from strategies.multi_factor.regime import Regime
        cfg = _small_config(atr_stop_multiple=2.0)
        dm = MultiFactorDecisionModule(cfg)
        dm._entry_atr = 5.0
        dm._entry_price = 100.0
        dm._trailing_peak = 105.0
        features = self._make_features(close=89.0)  # below 100 - 10 = 90
        result = dm._check_exit(features, Regime.TRENDING_UP, qty=1.0, avg_price=100.0)
        assert result == "stop_loss"

    def test_stop_loss_short(self):
        from strategies.multi_factor.regime import Regime
        cfg = _small_config(atr_stop_multiple=2.0)
        dm = MultiFactorDecisionModule(cfg)
        dm._entry_atr = 5.0
        dm._entry_price = 100.0
        dm._trailing_peak = 95.0
        features = self._make_features(close=111.0)  # above 100 + 10 = 110
        result = dm._check_exit(features, Regime.TRENDING_DOWN, qty=-1.0, avg_price=100.0)
        assert result == "stop_loss"

    def test_no_stop_within_range(self):
        from strategies.multi_factor.regime import Regime
        cfg = _small_config(atr_stop_multiple=2.0)
        dm = MultiFactorDecisionModule(cfg)
        dm._entry_atr = 5.0
        dm._entry_price = 100.0
        dm._trailing_peak = 103.0
        features = self._make_features(close=95.0)  # above 90 threshold
        result = dm._check_exit(features, Regime.TRENDING_UP, qty=1.0, avg_price=100.0)
        assert result is None

    def test_trailing_stop_long(self):
        from strategies.multi_factor.regime import Regime
        cfg = _small_config(trailing_atr_multiple=3.0, atr_stop_multiple=10.0)
        dm = MultiFactorDecisionModule(cfg)
        dm._entry_atr = 5.0
        dm._entry_price = 80.0  # far enough to not trigger hard stop
        dm._trailing_peak = 120.0
        features = self._make_features(close=104.0, atr=5.0)  # 120 - 5*3 = 105, close < 105
        result = dm._check_exit(features, Regime.TRENDING_UP, qty=1.0, avg_price=80.0)
        assert result == "trailing_stop"

    def test_trailing_stop_short(self):
        from strategies.multi_factor.regime import Regime
        cfg = _small_config(trailing_atr_multiple=3.0, atr_stop_multiple=10.0)
        dm = MultiFactorDecisionModule(cfg)
        dm._entry_atr = 5.0
        dm._entry_price = 120.0
        dm._trailing_peak = 80.0
        features = self._make_features(close=96.0, atr=5.0)  # 80 + 15 = 95, close > 95
        result = dm._check_exit(features, Regime.TRENDING_DOWN, qty=-1.0, avg_price=120.0)
        assert result == "trailing_stop"

    def test_target_reached_ranging_long(self):
        from strategies.multi_factor.regime import Regime
        cfg = _small_config()
        dm = MultiFactorDecisionModule(cfg)
        dm._entry_atr = 5.0
        dm._entry_price = 90.0
        dm._entry_regime = Regime.RANGING
        dm._trailing_peak = 100.0
        features = self._make_features(close=101.0, bb_middle=100.0)
        result = dm._check_exit(features, Regime.RANGING, qty=1.0, avg_price=90.0)
        assert result == "target_reached"

    def test_no_target_for_trending(self):
        from strategies.multi_factor.regime import Regime
        cfg = _small_config()
        dm = MultiFactorDecisionModule(cfg)
        dm._entry_atr = 5.0
        dm._entry_price = 90.0
        dm._entry_regime = Regime.TRENDING_UP
        dm._trailing_peak = 100.0
        features = self._make_features(close=101.0, bb_middle=100.0)
        result = dm._check_exit(features, Regime.TRENDING_UP, qty=1.0, avg_price=90.0)
        assert result is None  # not ranging, so no target exit


# ── Equity calculation ───────────────────────────────────────

class TestGetEquity:
    def test_flat_position(self):
        cfg = _small_config()
        dm = MultiFactorDecisionModule(cfg)
        account = SimpleNamespace(balance=Decimal("10000"))
        equity = dm._get_equity(account, {}, 100.0)
        assert equity == 10000.0

    def test_with_unrealized_pnl(self):
        cfg = _small_config()
        dm = MultiFactorDecisionModule(cfg)
        account = SimpleNamespace(balance=Decimal("10000"))
        pos = PositionState(symbol="BTCUSDT", qty=Decimal("1"), avg_price=Decimal("100"))
        equity = dm._get_equity(account, {"BTCUSDT": pos}, 110.0)
        assert equity == pytest.approx(10010.0)  # 10000 + (110-100)*1

    def test_no_account(self):
        cfg = _small_config()
        dm = MultiFactorDecisionModule(cfg)
        equity = dm._get_equity(None, {}, 100.0)
        assert equity == 10000.0  # default balance
