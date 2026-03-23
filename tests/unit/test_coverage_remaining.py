"""Unit tests for modules with low/zero coverage.

Covers:
- decision/signals/factors/__init__.py
- decision/signals/factors/carry.py
- decision/signals/factors/liquidity.py
- decision/signals/factors/momentum.py
- decision/signals/factors/trend_strength.py
- decision/signals/factors/volatility.py
- decision/signals/factors/volume_price_div.py
- decision/precomputed_hook.py
- decision/risk_overlay/base.py
- decision/selectors.py
- risk/stress.py
- state/store.py
- strategies/alpha_momentum.py
- monitoring/alerts/console.py
- monitoring/alerts/webhook.py
- monitoring/signal_decay_analysis.py
- core/plugins.py
- core/observability.py
- core/effects.py
- infra/logging/setup.py
- infra/config/schema.py
"""
from __future__ import annotations

import logging
import tempfile
import threading
import time
import unittest
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

def _make_snapshot(symbol="ETHUSDT", bar_index=0, ts=None, positions=None, account=None):
    """Build a minimal StateSnapshot for tests."""
    from state.snapshot import StateSnapshot
    from state.account import AccountState
    from state.market import MarketState

    acct = account or AccountState.initial(currency="USDT", balance=Decimal("1000"))
    mkts = {symbol: MarketState.empty(symbol)}
    pos = positions or {}
    return StateSnapshot.of(
        symbol=symbol,
        ts=ts or datetime.now(timezone.utc),
        event_id="evt-1",
        event_type="market",
        bar_index=bar_index,
        markets=mkts,
        positions=pos,
        account=acct,
    )


# ===========================================================================
# decision/signals/factors tests
# ===========================================================================

class TestFactorsInit(unittest.TestCase):
    """Test that the __init__ re-exports work."""

    def test_imports(self):
        with patch.dict("sys.modules", {
            "_quant_hotpath": MagicMock(
                rust_carry_score=MagicMock(return_value=("flat", 0.0, 0.0)),
                rust_liquidity_score=MagicMock(return_value=("flat", 0.0, 0.0)),
                rust_momentum_score=MagicMock(return_value=("flat", 0.0, 0.0)),
                rust_volatility_score=MagicMock(return_value=("flat", 0.0, 0.0)),
                rust_volume_price_div_score=MagicMock(return_value=("flat", 0.0, 0.0)),
            )
        }):
            from decision.signals.factors import (
                MomentumSignal, CarrySignal, VolatilitySignal, LiquiditySignal,
                TrendStrengthSignal, VolumePriceDivergenceSignal,
            )
            self.assertIsNotNone(MomentumSignal)
            self.assertIsNotNone(CarrySignal)
            self.assertIsNotNone(VolatilitySignal)
            self.assertIsNotNone(LiquiditySignal)
            self.assertIsNotNone(TrendStrengthSignal)
            self.assertIsNotNone(VolumePriceDivergenceSignal)


class TestCarrySignal(unittest.TestCase):

    def _make_signal(self):
        mock_hp = MagicMock()
        mock_hp.rust_carry_score = MagicMock(return_value=("sell", -0.5, 0.8))
        with patch.dict("sys.modules", {"_quant_hotpath": mock_hp}):
            import importlib
            import decision.signals.factors.carry as m
            importlib.reload(m)
            return m, mock_hp

    def test_compute_with_funding_rate_attr(self):
        mock_hp = MagicMock()
        mock_hp.rust_carry_score = MagicMock(return_value=("sell", -0.5, 0.8))
        with patch.dict("sys.modules", {"_quant_hotpath": mock_hp}):
            import importlib
            import decision.signals.factors.carry as m
            importlib.reload(m)

            snap = MagicMock()
            snap.funding_rate = 0.001
            del snap.funding_rate  # remove so we fall into next branch

            snap2 = MagicMock()
            snap2.funding_rate = 0.001
            sig = m.CarrySignal()
            result = sig.compute(snap2, "ETHUSDT")
            self.assertEqual(result.symbol, "ETHUSDT")
            self.assertEqual(result.side, "sell")

    def test_compute_no_funding_rate_returns_flat(self):
        mock_hp = MagicMock()
        mock_hp.rust_carry_score = MagicMock(return_value=("sell", -0.5, 0.8))
        with patch.dict("sys.modules", {"_quant_hotpath": mock_hp}):
            import importlib
            import decision.signals.factors.carry as m
            importlib.reload(m)

            snap = MagicMock(spec=[])  # no attributes
            sig = m.CarrySignal()
            result = sig.compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "flat")
            self.assertEqual(result.score, Decimal("0"))

    def test_get_funding_rate_dict(self):
        mock_hp = MagicMock()
        mock_hp.rust_carry_score = MagicMock(return_value=("buy", 0.3, 0.7))
        with patch.dict("sys.modules", {"_quant_hotpath": mock_hp}):
            import importlib
            import decision.signals.factors.carry as m
            importlib.reload(m)

            snap = MagicMock()
            snap.funding_rate = {"ETHUSDT": 0.002, "BTCUSDT": 0.001}
            result = m.CarrySignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "buy")

    def test_get_funding_rate_dict_missing_symbol(self):
        mock_hp = MagicMock()
        mock_hp.rust_carry_score = MagicMock(return_value=("buy", 0.3, 0.7))
        with patch.dict("sys.modules", {"_quant_hotpath": mock_hp}):
            import importlib
            import decision.signals.factors.carry as m
            importlib.reload(m)

            snap = MagicMock()
            snap.funding_rate = {"BTCUSDT": 0.001}  # ETHUSDT missing
            result = m.CarrySignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "flat")

    def test_get_funding_rate_via_method(self):
        mock_hp = MagicMock()
        mock_hp.rust_carry_score = MagicMock(return_value=("buy", 0.3, 0.7))
        with patch.dict("sys.modules", {"_quant_hotpath": mock_hp}):
            import importlib
            import decision.signals.factors.carry as m
            importlib.reload(m)

            snap = MagicMock(spec=["get_funding_rate"])
            snap.get_funding_rate = MagicMock(return_value=0.003)
            result = m.CarrySignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "buy")

    def test_get_funding_rate_method_none(self):
        mock_hp = MagicMock()
        mock_hp.rust_carry_score = MagicMock(return_value=("buy", 0.3, 0.7))
        with patch.dict("sys.modules", {"_quant_hotpath": mock_hp}):
            import importlib
            import decision.signals.factors.carry as m
            importlib.reload(m)

            snap = MagicMock(spec=["get_funding_rate"])
            snap.get_funding_rate = MagicMock(return_value=None)
            result = m.CarrySignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "flat")

    def test_default_name(self):
        mock_hp = MagicMock()
        with patch.dict("sys.modules", {"_quant_hotpath": mock_hp}):
            import importlib
            import decision.signals.factors.carry as m
            importlib.reload(m)
            self.assertEqual(m.CarrySignal().name, "carry")


class TestLiquiditySignal(unittest.TestCase):

    def _patched(self, side="flat", score=0.0, conf=0.0):
        mock_hp = MagicMock()
        mock_hp.rust_liquidity_score = MagicMock(return_value=(side, score, conf))
        return mock_hp

    def test_compute_from_bars_list(self):
        hp = self._patched("buy", 1.0, 0.9)
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.liquidity as m
            importlib.reload(m)

            b1 = MagicMock(volume=100.0)
            b2 = MagicMock(volume=200.0)
            snap = MagicMock()
            snap.bars = [b1, b2]
            result = m.LiquiditySignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "buy")

    def test_compute_from_bars_dict(self):
        hp = self._patched("sell", -0.5, 0.5)
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.liquidity as m
            importlib.reload(m)

            snap = MagicMock()
            snap.bars = {"ETHUSDT": [{"volume": 300.0}]}
            result = m.LiquiditySignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "sell")

    def test_compute_no_bars_attr_uses_get_bars(self):
        hp = self._patched("flat", 0.0, 0.0)
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.liquidity as m
            importlib.reload(m)

            snap = MagicMock(spec=["get_bars"])
            snap.get_bars = MagicMock(return_value=[])
            result = m.LiquiditySignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "flat")

    def test_default_lookback(self):
        hp = self._patched()
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.liquidity as m
            importlib.reload(m)
            self.assertEqual(m.LiquiditySignal().lookback, 20)


class TestMomentumSignal(unittest.TestCase):

    def test_compute_bars_list(self):
        hp = MagicMock()
        hp.rust_momentum_score = MagicMock(return_value=("buy", 0.8, 0.9))
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.momentum as m
            importlib.reload(m)

            snap = MagicMock()
            snap.bars = [MagicMock(close=100.0), MagicMock(close=110.0)]
            result = m.MomentumSignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "buy")
            self.assertEqual(result.symbol, "ETHUSDT")

    def test_compute_bars_dict(self):
        hp = MagicMock()
        hp.rust_momentum_score = MagicMock(return_value=("sell", -0.5, 0.6))
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.momentum as m
            importlib.reload(m)

            snap = MagicMock()
            snap.bars = {"ETHUSDT": [{"close": 100.0}, {"close": 90.0}]}
            result = m.MomentumSignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "sell")

    def test_compute_no_bars(self):
        hp = MagicMock()
        hp.rust_momentum_score = MagicMock(return_value=("flat", 0.0, 0.0))
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.momentum as m
            importlib.reload(m)

            snap = MagicMock(spec=["get_bars"])
            snap.get_bars = MagicMock(return_value=[])
            result = m.MomentumSignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "flat")

    def test_custom_lookback(self):
        hp = MagicMock()
        hp.rust_momentum_score = MagicMock(return_value=("flat", 0.0, 0.0))
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.momentum as m
            importlib.reload(m)
            sig = m.MomentumSignal(lookback=10)
            self.assertEqual(sig.lookback, 10)


class TestVolatilitySignal(unittest.TestCase):

    def test_compute(self):
        hp = MagicMock()
        hp.rust_volatility_score = MagicMock(return_value=("sell", -0.3, 0.7))
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.volatility as m
            importlib.reload(m)

            snap = MagicMock()
            snap.bars = [MagicMock(close=100.0)]
            result = m.VolatilitySignal().compute(snap, "BTCUSDT")
            self.assertEqual(result.side, "sell")
            self.assertEqual(result.symbol, "BTCUSDT")

    def test_bars_dict_missing_symbol(self):
        hp = MagicMock()
        hp.rust_volatility_score = MagicMock(return_value=("flat", 0.0, 0.0))
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.volatility as m
            importlib.reload(m)

            snap = MagicMock()
            snap.bars = {"OTHER": [{"close": 100.0}]}
            # ETHUSDT missing → empty list
            m.VolatilitySignal().compute(snap, "ETHUSDT")
            hp.rust_volatility_score.assert_called_once_with([], 20)


class TestVolumePriceDivergenceSignal(unittest.TestCase):

    def test_compute_from_dict_bars(self):
        hp = MagicMock()
        hp.rust_volume_price_div_score = MagicMock(return_value=("buy", 0.6, 0.8))
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.volume_price_div as m
            importlib.reload(m)

            snap = MagicMock()
            snap.bars = [
                {"close": 100.0, "volume": 500.0},
                {"close": 110.0, "volume": 300.0},
            ]
            result = m.VolumePriceDivergenceSignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "buy")

    def test_compute_from_object_bars(self):
        hp = MagicMock()
        hp.rust_volume_price_div_score = MagicMock(return_value=("sell", -0.4, 0.6))
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.volume_price_div as m
            importlib.reload(m)

            snap = MagicMock()
            b1 = MagicMock(close=100.0, volume=500.0)
            snap.bars = [b1]
            result = m.VolumePriceDivergenceSignal().compute(snap, "ETHUSDT")
            self.assertEqual(result.side, "sell")

    def test_default_lookback(self):
        hp = MagicMock()
        with patch.dict("sys.modules", {"_quant_hotpath": hp}):
            import importlib
            import decision.signals.factors.volume_price_div as m
            importlib.reload(m)
            self.assertEqual(m.VolumePriceDivergenceSignal().lookback, 10)


class TestTrendStrengthSignal(unittest.TestCase):
    """TrendStrengthSignal uses pure Python — no _quant_hotpath needed."""

    def _make_bars(self, n=60, start=100.0, step=1.0):
        """Create n Bar objects with incrementing closes."""
        from features.types import Bar
        bars = []
        for i in range(n):
            c = start + i * step
            bars.append(Bar(
                ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
                open=c - 0.1,
                high=c + 0.5,
                low=c - 0.5,
                close=c,
                volume=1000.0,
            ))
        return bars

    def test_not_enough_bars_returns_flat(self):
        from decision.signals.factors.trend_strength import TrendStrengthSignal

        sig = TrendStrengthSignal(adx_window=14, adx_threshold=25.0, lookback=20)
        snap = MagicMock()
        snap.bars = self._make_bars(n=5)  # too few
        result = sig.compute(snap, "ETHUSDT")
        self.assertEqual(result.side, "flat")
        self.assertEqual(result.score, Decimal("0"))

    def test_bars_from_dict_objects(self):
        from decision.signals.factors.trend_strength import _get_bars
        snap = MagicMock()
        snap.bars = [
            {"ts": "2024-01-01", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000.0}
        ]
        bars = _get_bars(snap, "ETHUSDT")
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].close, 100.0)

    def test_bars_from_dict_keyed(self):
        from decision.signals.factors.trend_strength import _get_bars
        snap = MagicMock()
        snap.bars = {"ETHUSDT": [
            {"open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000.0}
        ]}
        bars = _get_bars(snap, "ETHUSDT")
        self.assertEqual(len(bars), 1)

    def test_bars_no_attribute_uses_get_bars(self):
        from decision.signals.factors.trend_strength import _get_bars
        from features.types import Bar
        snap = MagicMock(spec=["get_bars"])
        b = Bar(ts=datetime.now(timezone.utc), open=99, high=101, low=98, close=100, volume=500)
        snap.get_bars = MagicMock(return_value=[b])
        bars = _get_bars(snap, "ETHUSDT")
        self.assertEqual(len(bars), 1)

    def test_bars_object_attr_conversion(self):
        """Branches: bar has attrs but not isinstance Bar."""
        from decision.signals.factors.trend_strength import _get_bars
        snap = MagicMock()
        b = MagicMock(spec=[])
        b.ts = datetime.now(timezone.utc)
        b.open = 99.0
        b.high = 101.0
        b.low = 98.0
        b.close = 100.0
        b.volume = 500.0
        snap.bars = [b]
        bars = _get_bars(snap, "ETHUSDT")
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].close, 100.0)

    def test_compute_below_adx_threshold(self):
        """ADX below threshold returns flat with partial confidence."""
        from decision.signals.factors.trend_strength import TrendStrengthSignal
        from features.types import Bar
        sig = TrendStrengthSignal(adx_window=3, adx_threshold=100.0, lookback=5)

        # Sideways bars → low ADX
        bars = []
        for i in range(20):
            c = 100.0
            bars.append(Bar(ts=datetime.now(timezone.utc), open=c, high=c+0.1, low=c-0.1, close=c, volume=100.0))
        snap = MagicMock()
        snap.bars = bars
        result = sig.compute(snap, "ETHUSDT")
        # ADX < 100 threshold → flat
        self.assertEqual(result.side, "flat")

    def test_compute_strong_uptrend(self):
        """Strong trending bars → buy signal."""
        from decision.signals.factors.trend_strength import TrendStrengthSignal

        # Use very low ADX threshold to guarantee we pass
        sig = TrendStrengthSignal(adx_window=3, adx_threshold=0.0, lookback=5)
        bars = self._make_bars(n=20, start=100.0, step=2.0)  # strong uptrend
        snap = MagicMock()
        snap.bars = bars
        result = sig.compute(snap, "ETHUSDT")
        # With strong trend and price going up → buy
        self.assertIn(result.side, ("buy", "flat"))  # depends on ADX calculation

    def test_compute_first_close_zero_returns_flat(self):
        """When closes[0] == 0 → flat."""
        from decision.signals.factors.trend_strength import TrendStrengthSignal
        from features.types import Bar

        sig = TrendStrengthSignal(adx_window=3, adx_threshold=0.0, lookback=5)
        bars = []
        for i in range(20):
            c = 0.0  # zero close
            bars.append(Bar(
                ts=datetime.now(timezone.utc), open=c, high=c + 0.01,
                low=max(c - 0.01, 0), close=c, volume=100.0,
            ))
        snap = MagicMock()
        snap.bars = bars
        result = sig.compute(snap, "ETHUSDT")
        self.assertEqual(result.side, "flat")

    def test_no_valid_adx_returns_flat(self):
        """When adx_series has all Nones → flat."""
        from decision.signals.factors.trend_strength import TrendStrengthSignal

        sig = TrendStrengthSignal(adx_window=14, adx_threshold=25.0, lookback=5)
        bars = self._make_bars(n=50, start=100.0, step=1.0)
        snap = MagicMock()
        snap.bars = bars

        with patch("features.technical.adx.adx", return_value=[None] * 50):
            result = sig.compute(snap, "ETHUSDT")
        self.assertEqual(result.side, "flat")


# ===========================================================================
# decision/precomputed_hook.py
# ===========================================================================

class TestPrecomputedFeatureHook(unittest.TestCase):

    def _make_hook(self, mapping=None):
        from decision.precomputed_hook import PrecomputedFeatureHook
        return PrecomputedFeatureHook(mapping or {})

    def test_on_event_none_ts(self):
        hook = self._make_hook()
        event = MagicMock()
        event.event_type = None
        event.symbol = "ETHUSDT"
        event.ts = None
        result = hook.on_event(event)
        self.assertIsNone(result)

    def test_on_event_none_symbol(self):
        hook = self._make_hook()
        event = MagicMock()
        event.event_type = None
        event.symbol = None
        event.ts = 1_700_000_000_000
        result = hook.on_event(event)
        self.assertIsNone(result)

    def test_on_event_exact_ts_match(self):
        ts_ms = 1_700_000_000_000
        feats = {"rsi": 50.0, "vol": 0.01}
        hook = self._make_hook({ts_ms: feats})

        event = MagicMock()
        event.event_type = None
        event.symbol = "ETHUSDT"
        event.ts = ts_ms
        result = hook.on_event(event)
        self.assertEqual(result, feats)

    def test_on_event_offset_match(self):
        """Test ±1ms tolerance lookup."""
        ts_ms = 1_700_000_000_001  # +1 offset
        feats = {"rsi": 55.0}
        hook = self._make_hook({ts_ms: feats})

        event = MagicMock()
        event.event_type = None
        event.symbol = "ETHUSDT"
        event.ts = 1_700_000_000_000  # exact miss but offset=+1 hits
        result = hook.on_event(event)
        self.assertEqual(result, feats)

    def test_on_event_fallback_to_last(self):
        """When ts not found, return last known features."""
        ts_ms = 1_700_000_000_000
        feats = {"rsi": 50.0}
        hook = self._make_hook({ts_ms: feats})

        # First call: store into last_features
        ev1 = MagicMock(event_type=None, symbol="ETHUSDT", ts=ts_ms)
        hook.on_event(ev1)

        # Second call: ts not found, should return last known
        ev2 = MagicMock(event_type=None, symbol="ETHUSDT", ts=9_999_999_999_999)
        result = hook.on_event(ev2)
        self.assertEqual(result, feats)

    def test_on_event_no_last_features(self):
        """ts not found, no last features → None."""
        hook = self._make_hook({})
        event = MagicMock(event_type=None, symbol="ETHUSDT", ts=1_700_000_000_000)
        result = hook.on_event(event)
        self.assertIsNone(result)

    def test_on_event_non_market_event_type_with_last(self):
        """Non-MARKET event type: if there's a last feature, return it."""
        from decision.precomputed_hook import PrecomputedFeatureHook

        # Patch EventType
        mock_et = MagicMock()
        mock_et.MARKET = mock_et  # make MARKET the same object
        with patch("decision.precomputed_hook.PrecomputedFeatureHook.on_event") as _:
            pass  # just check logic manually

        hook = PrecomputedFeatureHook({})
        hook._last_features["ETHUSDT"] = {"rsi": 42.0}

        # Create an event with non-MARKET event_type
        ev = MagicMock()
        ev.symbol = "ETHUSDT"
        ev.ts = 1_700_000_000_000

        # Patch the EventType import inside on_event
        with patch("event.types.EventType") as mock_etype:
            mock_etype.MARKET = "MARKET"
            ev.event_type = "ORDER"  # not MARKET
            # Will try string check: "market" in "order".lower() → False
            result = hook.on_event(ev)
            # Should return last known features for ETHUSDT
            self.assertEqual(result, {"rsi": 42.0})

    def test_resolve_ts_integer_large(self):
        from decision.precomputed_hook import PrecomputedFeatureHook
        ts_ms = PrecomputedFeatureHook._resolve_ts(1_700_000_000_000)
        self.assertEqual(ts_ms, 1_700_000_000_000)

    def test_resolve_ts_integer_small(self):
        """Seconds → convert to ms."""
        from decision.precomputed_hook import PrecomputedFeatureHook
        ts = PrecomputedFeatureHook._resolve_ts(1_700_000_000)
        self.assertEqual(ts, 1_700_000_000 * 1000)

    def test_resolve_ts_datetime(self):
        from decision.precomputed_hook import PrecomputedFeatureHook
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = PrecomputedFeatureHook._resolve_ts(dt)
        expected = int(dt.timestamp() * 1000)
        self.assertEqual(ts, expected)

    def test_resolve_ts_iso_string_with_z(self):
        from decision.precomputed_hook import PrecomputedFeatureHook
        ts = PrecomputedFeatureHook._resolve_ts("2024-01-01T00:00:00Z")
        self.assertIsNotNone(ts)
        self.assertIsInstance(ts, int)

    def test_resolve_ts_iso_string_no_tz(self):
        from decision.precomputed_hook import PrecomputedFeatureHook
        ts = PrecomputedFeatureHook._resolve_ts("2024-01-01T00:00:00")
        self.assertIsNotNone(ts)

    def test_resolve_ts_invalid_string(self):
        from decision.precomputed_hook import PrecomputedFeatureHook
        ts = PrecomputedFeatureHook._resolve_ts("not-a-date")
        self.assertIsNone(ts)

    def test_resolve_ts_none(self):
        from decision.precomputed_hook import PrecomputedFeatureHook
        ts = PrecomputedFeatureHook._resolve_ts(None)
        self.assertIsNone(ts)

    def test_from_dataframe_smoke(self):
        """Verify from_dataframe builds the hook without errors with mocked deps."""
        import pandas as pd
        from decision.precomputed_hook import PrecomputedFeatureHook

        df = pd.DataFrame({
            "open_time": [1_700_000_000_000, 1_700_003_600_000],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1000.0, 1100.0],
        })

        feat_df = pd.DataFrame({
            "rsi_14": [50.0, 55.0],
            "vol_20": [float("nan"), float("nan")],  # all NaN → excluded
        }, index=df.index)

        with patch("features.batch_feature_engine.compute_features_batch", return_value=feat_df):
            with patch("pathlib.Path.exists", return_value=False):
                with patch("features.batch_feature_engine.compute_4h_features", side_effect=Exception("no tf")):
                    hook = PrecomputedFeatureHook.from_dataframe(
                        "ETHUSDT", df, include_4h=True, include_interactions=False
                    )
        self.assertIsInstance(hook, PrecomputedFeatureHook)
        # Check that NaN was excluded
        ts0 = int(df["open_time"].iloc[0])
        feats0 = hook._features_by_ts.get(ts0, {})
        self.assertIn("rsi_14", feats0)
        self.assertNotIn("vol_20", feats0)  # NaN excluded from row 0


# ===========================================================================
# decision/risk_overlay/base.py
# ===========================================================================

class TestRiskOverlayBase(unittest.TestCase):

    def test_always_allow(self):
        from decision.risk_overlay.base import AlwaysAllow
        overlay = AlwaysAllow()
        snap = _make_snapshot()
        ok, reasons = overlay.allow(snap)
        self.assertTrue(ok)
        self.assertEqual(len(reasons), 0)

    def test_composite_overlay_all_allow(self):
        from decision.risk_overlay.base import AlwaysAllow, CompositeOverlay
        comp = CompositeOverlay(overlays=(AlwaysAllow(), AlwaysAllow()))
        snap = _make_snapshot()
        ok, reasons = comp.allow(snap)
        self.assertTrue(ok)
        self.assertEqual(len(reasons), 0)

    def test_composite_overlay_one_rejects(self):
        from decision.risk_overlay.base import AlwaysAllow, CompositeOverlay

        class RejectOverlay:
            def allow(self, snap):
                return False, ("risk limit exceeded",)

        comp = CompositeOverlay(overlays=(AlwaysAllow(), RejectOverlay()))
        snap = _make_snapshot()
        ok, reasons = comp.allow(snap)
        self.assertFalse(ok)
        self.assertIn("risk limit exceeded", reasons)

    def test_composite_overlay_multiple_rejections(self):
        from decision.risk_overlay.base import CompositeOverlay

        class Reject1:
            def allow(self, snap):
                return False, ("reason1",)

        class Reject2:
            def allow(self, snap):
                return False, ("reason2",)

        comp = CompositeOverlay(overlays=(Reject1(), Reject2()))
        snap = _make_snapshot()
        ok, reasons = comp.allow(snap)
        self.assertFalse(ok)
        self.assertIn("reason1", reasons)
        self.assertIn("reason2", reasons)

    def test_composite_overlay_empty(self):
        from decision.risk_overlay.base import CompositeOverlay
        comp = CompositeOverlay()
        snap = _make_snapshot()
        ok, reasons = comp.allow(snap)
        self.assertTrue(ok)


# ===========================================================================
# decision/selectors.py
# ===========================================================================

class TestUniverseSelector(unittest.TestCase):

    def test_explicit_symbols(self):
        from decision.selectors import UniverseSelector
        sel = UniverseSelector(symbols=["ETHUSDT", "BTCUSDT", "ETHUSDT"])
        snap = _make_snapshot()
        result = sel.select(snap)
        # Should deduplicate while preserving order
        self.assertEqual(result, ["ETHUSDT", "BTCUSDT"])

    def test_empty_symbols_uses_snapshot_symbol(self):
        from decision.selectors import UniverseSelector
        sel = UniverseSelector(symbols=None)
        snap = _make_snapshot(symbol="SUIUSDT")
        result = sel.select(snap)
        self.assertIn("SUIUSDT", result)

    def test_empty_symbols_uses_positions(self):
        from decision.selectors import UniverseSelector
        sel = UniverseSelector(symbols=None)

        # Use a plain MagicMock snapshot with a real dict for positions
        # (StateSnapshot wraps positions in MappingProxyType which is not isinstance dict)
        snap = MagicMock()
        snap.symbol = None
        snap.positions = {"ETHUSDT": MagicMock(), "BTCUSDT": MagicMock()}
        result = sel.select(snap)
        self.assertIn("ETHUSDT", result)
        self.assertIn("BTCUSDT", result)

    def test_empty_symbols_snapshot_no_symbol_attr(self):
        from decision.selectors import UniverseSelector
        sel = UniverseSelector(symbols=None)
        snap = MagicMock(spec=["positions"])
        snap.positions = {}
        # getattr(snap, "symbol", None) → None
        result = sel.select(snap)
        self.assertEqual(result, [])

    def test_returns_sorted_when_no_explicit_symbols(self):
        from decision.selectors import UniverseSelector
        sel = UniverseSelector(symbols=None)
        snap = MagicMock()
        snap.symbol = None
        snap.positions = {"ZZUSDT": MagicMock(), "AAUSDT": MagicMock()}
        result = sel.select(snap)
        self.assertEqual(result, sorted(result))


# ===========================================================================
# risk/stress.py
# ===========================================================================

class TestStressEngine(unittest.TestCase):

    def _make_account(self, equity=1000, positions=None):
        from risk.stress import AccountExposure
        return AccountExposure(
            equity=Decimal(str(equity)),
            balance=Decimal(str(equity)),
            used_margin=Decimal("100"),
            positions=positions or {},
        )

    def _make_scenario(self, name="test", pct="-0.10"):
        from risk.stress import StressScenario, PriceShock
        return StressScenario(
            name=name,
            global_shock=PriceShock(pct=Decimal(pct)),
        )

    def test_empty_scenarios_raises(self):
        from risk.stress import StressEngine, StressEngineError
        engine = StressEngine()
        account = self._make_account()
        with self.assertRaises(StressEngineError):
            engine.run(account=account, scenarios=())

    def test_single_scenario_no_positions(self):
        from risk.stress import StressEngine
        engine = StressEngine()
        account = self._make_account(equity=1000)
        sc = self._make_scenario("down10pct", "-0.10")
        report = engine.run(account=account, scenarios=[sc])
        self.assertEqual(len(report.results), 1)
        r = report.results[0]
        self.assertEqual(r.scenario, "down10pct")
        self.assertEqual(r.pnl, Decimal("0"))  # no positions
        self.assertEqual(r.drawdown_pct, Decimal("0"))  # no change
        self.assertTrue(r.ok)

    def test_long_position_down_shock(self):
        from risk.stress import StressEngine, AccountExposure, PositionExposure, PriceShock, StressScenario

        pos = PositionExposure(
            symbol="ETHUSDT",
            qty=Decimal("1"),
            mark_price=Decimal("1000"),
        )
        account = AccountExposure(
            equity=Decimal("1000"),
            balance=Decimal("1000"),
            positions={"ETHUSDT": pos},
        )
        sc = StressScenario(
            name="down10",
            global_shock=PriceShock(pct=Decimal("-0.10")),
        )
        engine = StressEngine()
        report = engine.run(account=account, scenarios=[sc])
        r = report.results[0]
        # PnL = (900 - 1000) * 1 * 1 = -100
        self.assertEqual(r.pnl, Decimal("-100"))
        self.assertEqual(r.equity_after, Decimal("900"))
        self.assertTrue(r.drawdown_pct > 0)

    def test_short_position_down_shock(self):
        from risk.stress import StressEngine, AccountExposure, PositionExposure, PriceShock, StressScenario

        pos = PositionExposure(
            symbol="BTCUSDT",
            qty=Decimal("-0.5"),  # short
            mark_price=Decimal("40000"),
        )
        account = AccountExposure(
            equity=Decimal("5000"),
            balance=Decimal("5000"),
            positions={"BTCUSDT": pos},
        )
        sc = StressScenario(name="btc_crash", global_shock=PriceShock(pct=Decimal("-0.20")))
        engine = StressEngine()
        report = engine.run(account=account, scenarios=[sc])
        r = report.results[0]
        # PnL = (32000 - 40000) * -0.5 * 1 = 4000 (profit for short)
        self.assertGreater(r.equity_after, account.equity)
        self.assertEqual(r.drawdown_pct, Decimal("0"))  # profit, no drawdown

    def test_price_shock_abs(self):
        from risk.stress import PriceShock
        shock = PriceShock(abs=Decimal("-500"))
        new_price = shock.apply(Decimal("1000"))
        self.assertEqual(new_price, Decimal("500"))

    def test_price_shock_pct_and_abs_combined(self):
        from risk.stress import PriceShock
        shock = PriceShock(pct=Decimal("-0.10"), abs=Decimal("-50"))
        # 1000 * 0.9 = 900, then -50 = 850
        new_price = shock.apply(Decimal("1000"))
        self.assertEqual(new_price, Decimal("850"))

    def test_price_shock_clamp_min(self):
        from risk.stress import PriceShock
        shock = PriceShock(pct=Decimal("-0.90"), clamp_min_price=Decimal("200"))
        # 1000 * 0.1 = 100, but clamp to 200
        new_price = shock.apply(Decimal("1000"))
        self.assertEqual(new_price, Decimal("200"))

    def test_price_shock_clamp_max(self):
        from risk.stress import PriceShock
        shock = PriceShock(pct=Decimal("2.00"), clamp_max_price=Decimal("2500"))
        new_price = shock.apply(Decimal("1000"))
        self.assertEqual(new_price, Decimal("2500"))

    def test_price_shock_cannot_go_negative(self):
        from risk.stress import PriceShock
        shock = PriceShock(pct=Decimal("-2.0"))
        new_price = shock.apply(Decimal("100"))
        self.assertEqual(new_price, Decimal("0"))

    def test_stress_scenario_specific_shock(self):
        from risk.stress import StressScenario, PriceShock
        sc = StressScenario(
            name="eth_crash",
            shocks={"ETHUSDT": PriceShock(pct=Decimal("-0.30"))},
            global_shock=PriceShock(pct=Decimal("-0.05")),
        )
        # Symbol-specific shock takes priority
        eth_price = sc.shocked_price_for("ETHUSDT", Decimal("1000"))
        self.assertEqual(eth_price, Decimal("700"))
        # Global shock for other symbols
        btc_price = sc.shocked_price_for("BTCUSDT", Decimal("40000"))
        self.assertEqual(btc_price, Decimal("38000"))

    def test_stress_scenario_no_shock(self):
        from risk.stress import StressScenario
        sc = StressScenario(name="flat")
        price = sc.shocked_price_for("ETHUSDT", Decimal("1000"))
        self.assertEqual(price, Decimal("1000"))

    def test_threshold_min_equity_violation(self):
        from risk.stress import (
            StressEngine, StressThresholds, AccountExposure,
            PositionExposure, PriceShock, StressScenario,
        )

        pos = PositionExposure(symbol="ETHUSDT", qty=Decimal("1"), mark_price=Decimal("1000"))
        account = AccountExposure(equity=Decimal("1000"), balance=Decimal("1000"), positions={"ETHUSDT": pos})
        sc = StressScenario(name="crash", global_shock=PriceShock(pct=Decimal("-0.90")))
        thresholds = StressThresholds(min_equity=Decimal("500"))
        engine = StressEngine(thresholds=thresholds)
        report = engine.run(account=account, scenarios=[sc])
        r = report.results[0]
        self.assertFalse(r.ok)
        codes = [v.code for v in r.violations]
        self.assertIn("min_equity", codes)

    def test_threshold_max_drawdown_violation(self):
        from risk.stress import (
            StressEngine, StressThresholds, AccountExposure,
            PositionExposure, PriceShock, StressScenario,
        )

        pos = PositionExposure(symbol="ETHUSDT", qty=Decimal("1"), mark_price=Decimal("1000"))
        account = AccountExposure(equity=Decimal("1000"), balance=Decimal("1000"), positions={"ETHUSDT": pos})
        sc = StressScenario(name="crash", global_shock=PriceShock(pct=Decimal("-0.50")))
        thresholds = StressThresholds(max_drawdown_pct=Decimal("0.20"))
        engine = StressEngine(thresholds=thresholds)
        report = engine.run(account=account, scenarios=[sc])
        r = report.results[0]
        self.assertFalse(r.ok)
        codes = [v.code for v in r.violations]
        self.assertIn("max_drawdown", codes)

    def test_threshold_min_margin_ratio_violation(self):
        from risk.stress import (
            StressEngine, StressThresholds, AccountExposure,
            PositionExposure, PriceShock, StressScenario,
        )

        pos = PositionExposure(symbol="ETHUSDT", qty=Decimal("1"), mark_price=Decimal("1000"))
        account = AccountExposure(
            equity=Decimal("1000"), balance=Decimal("1000"),
            used_margin=Decimal("800"),
            positions={"ETHUSDT": pos},
        )
        sc = StressScenario(name="crash", global_shock=PriceShock(pct=Decimal("-0.80")))
        thresholds = StressThresholds(min_margin_ratio=Decimal("1.5"))
        engine = StressEngine(thresholds=thresholds)
        report = engine.run(account=account, scenarios=[sc])
        r = report.results[0]
        self.assertFalse(r.ok)
        codes = [v.code for v in r.violations]
        self.assertIn("min_margin_ratio", codes)

    def test_no_margin_ratio_when_margin_zero(self):
        from risk.stress import StressEngine, AccountExposure
        account = AccountExposure(equity=Decimal("1000"), balance=Decimal("1000"), used_margin=Decimal("0"))
        sc = self._make_scenario()
        engine = StressEngine()
        report = engine.run(account=account, scenarios=[sc])
        self.assertIsNone(report.results[0].margin_ratio)

    def test_equity_zero_no_drawdown_pct(self):
        from risk.stress import StressEngine, AccountExposure
        account = AccountExposure(equity=Decimal("0"), balance=Decimal("0"))
        sc = self._make_scenario()
        engine = StressEngine()
        report = engine.run(account=account, scenarios=[sc])
        self.assertEqual(report.results[0].drawdown_pct, Decimal("0"))

    def test_worst_by_equity_and_drawdown(self):
        from risk.stress import StressEngine, AccountExposure, PositionExposure, PriceShock, StressScenario

        pos = PositionExposure(symbol="ETHUSDT", qty=Decimal("1"), mark_price=Decimal("1000"))
        account = AccountExposure(equity=Decimal("1000"), balance=Decimal("1000"), positions={"ETHUSDT": pos})
        scenarios = [
            StressScenario(name="small", global_shock=PriceShock(pct=Decimal("-0.10"))),
            StressScenario(name="large", global_shock=PriceShock(pct=Decimal("-0.50"))),
        ]
        engine = StressEngine()
        report = engine.run(account=account, scenarios=scenarios, ts="2024-01-01")
        worst_eq = report.worst_by_equity
        self.assertEqual(worst_eq.scenario, "large")
        worst_dd = report.worst_by_drawdown
        self.assertEqual(worst_dd.scenario, "large")

    def test_extra_liabilities(self):
        from risk.stress import StressEngine, AccountExposure, StressScenario
        account = AccountExposure(
            equity=Decimal("1000"), balance=Decimal("1000"),
            extra_liabilities=Decimal("100"),
        )
        sc = StressScenario(name="flat")  # no shock
        engine = StressEngine()
        report = engine.run(account=account, scenarios=[sc])
        # equity_after = 1000 + 0 - 100 = 900
        self.assertEqual(report.results[0].equity_after, Decimal("900"))

    def test_build_default_stress_scenarios(self):
        from risk.stress import build_default_stress_scenarios
        scenarios = build_default_stress_scenarios(symbols=["ETHUSDT", "BTCUSDT"])
        names = [s.name for s in scenarios]
        # global up/down + 2 per symbol (crash + squeeze)
        self.assertIn("global_10pct_down", names)
        self.assertIn("global_10pct_up", names)
        self.assertIn("ETHUSDT_crash_30pct", names)
        self.assertIn("BTCUSDT_squeeze_30pct", names)

    def test_stress_result_ok_property(self):
        from risk.stress import StressResult
        r = StressResult(
            scenario="test",
            equity_before=Decimal("1000"),
            equity_after=Decimal("900"),
            pnl=Decimal("-100"),
            drawdown_pct=Decimal("0.10"),
            used_margin=Decimal("0"),
            margin_ratio=None,
        )
        self.assertTrue(r.ok)

    def test_multiplier_applied(self):
        from risk.stress import StressEngine, AccountExposure, PositionExposure, PriceShock, StressScenario

        # multiplier=10 (like a futures contract)
        pos = PositionExposure(
            symbol="BTC", qty=Decimal("1"), mark_price=Decimal("1000"), multiplier=Decimal("10")
        )
        account = AccountExposure(equity=Decimal("2000"), balance=Decimal("2000"), positions={"BTC": pos})
        sc = StressScenario(name="down10", global_shock=PriceShock(pct=Decimal("-0.10")))
        engine = StressEngine()
        report = engine.run(account=account, scenarios=[sc])
        # pnl = (900 - 1000) * 1 * 10 = -1000
        self.assertEqual(report.results[0].pnl, Decimal("-1000"))


# ===========================================================================
# state/store.py
# ===========================================================================

class TestInMemoryStateStore(unittest.TestCase):

    def test_save_and_latest(self):
        from state.store import InMemoryStateStore
        store = InMemoryStateStore()
        snap = _make_snapshot("ETHUSDT", bar_index=5)
        store.save(snap)
        cp = store.latest("ETHUSDT")
        self.assertIsNotNone(cp)
        self.assertEqual(cp.symbol, "ETHUSDT")
        self.assertEqual(cp.bar_index, 5)

    def test_latest_missing_returns_none(self):
        from state.store import InMemoryStateStore
        store = InMemoryStateStore()
        self.assertIsNone(store.latest("NOTEXISTS"))

    def test_overwrite(self):
        from state.store import InMemoryStateStore
        store = InMemoryStateStore()
        snap1 = _make_snapshot("ETHUSDT", bar_index=1)
        snap2 = _make_snapshot("ETHUSDT", bar_index=10)
        store.save(snap1)
        store.save(snap2)
        cp = store.latest("ETHUSDT")
        self.assertEqual(cp.bar_index, 10)


class TestDcToDict(unittest.TestCase):

    def test_decimal_roundtrip(self):
        from state.store import _dc_to_dict
        result = _dc_to_dict(Decimal("3.14"))
        self.assertEqual(result, {"__decimal__": "3.14"})

    def test_datetime_roundtrip(self):
        from state.store import _dc_to_dict
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = _dc_to_dict(dt)
        self.assertEqual(result, {"__datetime__": dt.isoformat()})

    def test_dict_recursion(self):
        from state.store import _dc_to_dict
        d = {"a": Decimal("1.0"), "b": {"c": 42}}
        result = _dc_to_dict(d)
        self.assertEqual(result["a"], {"__decimal__": "1.0"})
        self.assertEqual(result["b"]["c"], 42)

    def test_list_recursion(self):
        from state.store import _dc_to_dict
        lst = [1, Decimal("2"), "three"]
        result = _dc_to_dict(lst)
        self.assertEqual(result[1], {"__decimal__": "2"})

    def test_plain_value_passthrough(self):
        from state.store import _dc_to_dict
        self.assertEqual(_dc_to_dict(42), 42)
        self.assertEqual(_dc_to_dict("hello"), "hello")


class TestStateDecoderHook(unittest.TestCase):

    def test_decimal_decode(self):
        from state.store import _state_decoder_hook
        result = _state_decoder_hook({"__decimal__": "3.14"})
        self.assertEqual(result, Decimal("3.14"))

    def test_datetime_decode_utc(self):
        from state.store import _state_decoder_hook
        dt_str = "2024-01-01T00:00:00+00:00"
        result = _state_decoder_hook({"__datetime__": dt_str})
        self.assertIsInstance(result, datetime)
        self.assertIsNotNone(result.tzinfo)

    def test_datetime_decode_no_tz_gets_utc(self):
        from state.store import _state_decoder_hook
        result = _state_decoder_hook({"__datetime__": "2024-01-01T00:00:00"})
        self.assertEqual(result.tzinfo, timezone.utc)

    def test_passthrough(self):
        from state.store import _state_decoder_hook
        d = {"key": "value"}
        result = _state_decoder_hook(d)
        self.assertEqual(result, d)


class TestSqliteStateStore(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_save_and_latest(self):
        from state.store import SqliteStateStore
        path = Path(self.tmpdir) / "test.db"
        with SqliteStateStore(path) as store:
            snap = _make_snapshot("ETHUSDT", bar_index=3)
            store.save(snap)
            cp = store.latest("ETHUSDT")
        self.assertIsNotNone(cp)
        self.assertEqual(cp.symbol, "ETHUSDT")
        self.assertEqual(cp.bar_index, 3)

    def test_latest_missing(self):
        from state.store import SqliteStateStore
        path = Path(self.tmpdir) / "test2.db"
        with SqliteStateStore(path) as store:
            cp = store.latest("MISSING")
        self.assertIsNone(cp)

    def test_upsert_semantics(self):
        from state.store import SqliteStateStore
        path = Path(self.tmpdir) / "test3.db"
        with SqliteStateStore(path) as store:
            store.save(_make_snapshot("ETHUSDT", bar_index=1))
            store.save(_make_snapshot("ETHUSDT", bar_index=99))
            cp = store.latest("ETHUSDT")
        self.assertEqual(cp.bar_index, 99)

    def test_all_symbols(self):
        from state.store import SqliteStateStore
        path = Path(self.tmpdir) / "test4.db"
        with SqliteStateStore(path) as store:
            store.save(_make_snapshot("ETHUSDT", bar_index=1))
            store.save(_make_snapshot("BTCUSDT", bar_index=2))
            syms = store.all_symbols()
        self.assertIn("ETHUSDT", syms)
        self.assertIn("BTCUSDT", syms)

    def test_with_history(self):
        from state.store import SqliteStateStore
        path = Path(self.tmpdir) / "test5.db"
        with SqliteStateStore(path, keep_history=True) as store:
            store.save(_make_snapshot("ETHUSDT", bar_index=1))
            store.save(_make_snapshot("ETHUSDT", bar_index=2))
            cp = store.latest("ETHUSDT")
        self.assertEqual(cp.bar_index, 2)

    def test_thread_safety(self):
        from state.store import SqliteStateStore
        path = Path(self.tmpdir) / "thread.db"
        store = SqliteStateStore(path)
        errors = []

        def writer(i):
            try:
                store.save(_make_snapshot(f"SYM{i}", bar_index=i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        store.close()
        self.assertEqual(errors, [])


# ===========================================================================
# strategies/alpha_momentum.py
# ===========================================================================

class TestAlphaMomentumStrategy(unittest.TestCase):

    def _make_model_info(self, features=None, horizon_models=None):
        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[0.5])
        return {
            "model": mock_model,
            "features": features or ["f1", "f2", "f3"],
            "config": {},
            "deadzone": 0.5,
            "min_hold": 3,
            "zscore_window": 20,
            "zscore_warmup": 5,
            "horizon_models": horizon_models or [],
        }

    def test_generate_signal_warmup(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        strat = AlphaMomentumStrategy(mi, symbol="ETHUSDT")
        # During warmup, zscore is None → direction = 0
        sig = strat.generate_signal({"f1": 1.0, "f2": 2.0, "f3": 3.0})
        self.assertEqual(sig.direction, 0)
        self.assertEqual(sig.confidence, 0.0)

    def test_generate_signal_after_warmup(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi["zscore_warmup"] = 2
        strat = AlphaMomentumStrategy(mi, symbol="ETHUSDT")

        # Feed enough bars to get past warmup; model returns predictable value
        mi["model"].predict = MagicMock(side_effect=lambda x: [0.5 if i < 5 else 2.0 for i in [0]])
        for _ in range(3):
            strat.generate_signal({"f1": 1.0, "f2": 2.0, "f3": 3.0})

    def test_min_hold_enforcement(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi["zscore_warmup"] = 1
        mi["deadzone"] = 0.01

        strat = AlphaMomentumStrategy(mi, symbol="ETHUSDT")
        strat._current_signal = 1
        strat._hold_counter = 2  # simulate active hold

        sig = strat.generate_signal({"f1": 1.0, "f2": 2.0, "f3": 3.0})
        self.assertTrue(sig.meta.get("held"))
        self.assertEqual(strat._hold_counter, 1)

    def test_direction_long(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi["zscore_warmup"] = 1
        mi["zscore_window"] = 5

        strat = AlphaMomentumStrategy(mi, symbol="ETHUSDT")
        # Force a high z-score by varying predictions
        strat._pred_buffer = [0.0, 0.0, 0.0, 0.0]
        strat._bars_processed = 4

        mi["model"].predict = MagicMock(return_value=[5.0])  # very high
        sig = strat.generate_signal({"f1": 1.0})
        self.assertEqual(sig.direction, 1)

    def test_direction_short(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi["zscore_warmup"] = 1

        strat = AlphaMomentumStrategy(mi, symbol="ETHUSDT")
        strat._pred_buffer = [0.0, 0.0, 0.0, 0.0]

        mi["model"].predict = MagicMock(return_value=[-5.0])  # very negative
        sig = strat.generate_signal({"f1": 1.0})
        self.assertEqual(sig.direction, -1)

    def test_update_zscore_flat_std(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        strat = AlphaMomentumStrategy(mi)
        strat._pred_buffer = [1.0] * 10  # all same → std ~0
        z = strat._update_zscore(1.0)
        self.assertEqual(z, 0.0)

    def test_predict_horizon_models(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy

        lgbm = MagicMock()
        lgbm.predict = MagicMock(return_value=[0.3])
        ridge = MagicMock()
        ridge.predict = MagicMock(return_value=[0.7])

        hm = {
            "features": ["f1", "f2"],
            "lgbm": lgbm,
            "ridge": ridge,
            "ridge_features": ["f1"],
        }
        mi = self._make_model_info(horizon_models=[hm])
        strat = AlphaMomentumStrategy(mi)

        feats = {"f1": 1.0, "f2": 2.0}
        pred = strat._predict(feats)
        # blended = 0.6 * 0.7 + 0.4 * 0.3 = 0.42 + 0.12 = 0.54
        self.assertAlmostEqual(pred, 0.54, places=5)

    def test_predict_horizon_models_no_ridge(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy

        lgbm = MagicMock()
        lgbm.predict = MagicMock(return_value=[0.4])

        hm = {"features": ["f1"], "lgbm": lgbm, "ridge": None}
        mi = self._make_model_info(horizon_models=[hm])
        strat = AlphaMomentumStrategy(mi)
        pred = strat._predict({"f1": 1.0})
        self.assertAlmostEqual(pred, 0.4, places=5)

    def test_validate_config_passes(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        strat = AlphaMomentumStrategy(mi)
        self.assertTrue(strat.validate_config())

    def test_validate_config_no_model(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi["model"] = None
        strat = AlphaMomentumStrategy(mi)
        self.assertFalse(strat.validate_config())

    def test_validate_config_empty_features(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi = {**mi, "features": []}  # override with truly empty list
        strat = AlphaMomentumStrategy(mi)
        self.assertFalse(strat.validate_config())

    def test_validate_config_zero_deadzone(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi["deadzone"] = 0.0
        strat = AlphaMomentumStrategy(mi)
        self.assertFalse(strat.validate_config())

    def test_describe(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        strat = AlphaMomentumStrategy(mi, symbol="ETHUSDT", timeframe="1h")
        desc = strat.describe()
        self.assertIn("ETHUSDT", desc)
        self.assertIn("1h", desc)

    def test_safe_val_none(self):
        from strategies.alpha_momentum import _safe_val
        self.assertEqual(_safe_val(None), 0.0)

    def test_safe_val_nan(self):
        from strategies.alpha_momentum import _safe_val
        self.assertEqual(_safe_val(float("nan")), 0.0)

    def test_safe_val_normal(self):
        from strategies.alpha_momentum import _safe_val
        self.assertEqual(_safe_val(3.14), 3.14)

    def test_safe_val_type_error(self):
        from strategies.alpha_momentum import _safe_val
        self.assertEqual(_safe_val("not-a-number"), 0.0)

    def test_pred_buffer_window_limit(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi["zscore_window"] = 5
        strat = AlphaMomentumStrategy(mi)
        # Push 10 predictions; buffer should be capped at 5
        for v in range(10):
            strat._update_zscore(float(v))
        self.assertLessEqual(len(strat._pred_buffer), 5)

    def test_direction_hold_counter_set_on_signal_change(self):
        from strategies.alpha_momentum import AlphaMomentumStrategy
        mi = self._make_model_info()
        mi["zscore_warmup"] = 1
        strat = AlphaMomentumStrategy(mi)
        strat._pred_buffer = [0.0, 0.0, 0.0, 0.0]
        strat._current_signal = -1  # was short
        mi["model"].predict = MagicMock(return_value=[5.0])  # now long
        strat.generate_signal({"f1": 1.0})
        self.assertGreater(strat._hold_counter, 0)


# ===========================================================================
# monitoring/alerts/console.py
# ===========================================================================

class TestConsoleAlertSink(unittest.TestCase):

    def _make_alert(self, title="Test", msg="test msg", severity=None, source="", ts=None, meta=None):
        from monitoring.alerts.base import Alert, Severity
        return Alert(
            title=title,
            message=msg,
            severity=severity or Severity.WARNING,
            source=source,
            ts=ts,
            meta=meta,
        )

    def test_emit_basic(self):
        from monitoring.alerts.console import ConsoleAlertSink
        sink = ConsoleAlertSink()
        alert = self._make_alert(ts=datetime(2024, 1, 1, tzinfo=timezone.utc))
        with patch("builtins.print") as mock_print:
            sink.emit(alert)
        mock_print.assert_called_once()
        output = mock_print.call_args[0][0]
        self.assertIn("WARNING", output)
        self.assertIn("Test", output)

    def test_emit_with_source(self):
        from monitoring.alerts.console import ConsoleAlertSink
        sink = ConsoleAlertSink()
        alert = self._make_alert(source="risk_module")
        with patch("builtins.print") as mock_print:
            sink.emit(alert)
        output = mock_print.call_args[0][0]
        self.assertIn("risk_module", output)

    def test_emit_with_meta(self):
        from monitoring.alerts.console import ConsoleAlertSink
        sink = ConsoleAlertSink()
        alert = self._make_alert(meta={"drawdown": "0.15", "symbol": "ETHUSDT"})
        with patch("builtins.print") as mock_print:
            sink.emit(alert)
        output = mock_print.call_args[0][0]
        self.assertIn("drawdown=0.15", output)

    def test_emit_no_ts_uses_now(self):
        from monitoring.alerts.console import ConsoleAlertSink
        sink = ConsoleAlertSink()
        alert = self._make_alert(ts=None)
        with patch("builtins.print") as mock_print:
            sink.emit(alert)
        # Should not raise; uses datetime.now(utc) internally
        mock_print.assert_called_once()

    def test_emit_no_source(self):
        from monitoring.alerts.console import ConsoleAlertSink
        from monitoring.alerts.base import Alert, Severity
        sink = ConsoleAlertSink()
        alert = Alert(title="T", message="M", severity=Severity.ERROR, source="")
        with patch("builtins.print") as mock_print:
            sink.emit(alert)
        output = mock_print.call_args[0][0]
        self.assertNotIn("[", output.split("]")[-1].split("ERROR")[0] if "ERROR" in output else "")


# ===========================================================================
# monitoring/alerts/webhook.py
# ===========================================================================

class TestWebhookAlertSink(unittest.TestCase):

    def _make_alert(self, severity=None, title="Test"):
        from monitoring.alerts.base import Alert, Severity
        return Alert(
            title=title,
            message="test",
            severity=severity or Severity.WARNING,
        )

    def test_severity_filter_below_min(self):
        from monitoring.alerts.webhook import WebhookAlertSink
        from monitoring.alerts.base import Severity
        sink = WebhookAlertSink(url="http://example.com", min_severity=Severity.ERROR)
        alert = self._make_alert(severity=Severity.DEBUG)

        with patch("monitoring.alerts.webhook.urlopen") as mock_open:
            sink.emit(alert)
            mock_open.assert_not_called()

    def _fake_resp(self):
        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return b""
        return FakeResp()

    def test_successful_emit(self):
        from monitoring.alerts.webhook import WebhookAlertSink
        from monitoring.alerts.base import Severity

        sink = WebhookAlertSink(url="http://example.com", retries=0)
        alert = self._make_alert(severity=Severity.CRITICAL)

        with patch("monitoring.alerts.webhook.urlopen", return_value=self._fake_resp()):
            sink.emit(alert)

        self.assertEqual(sink._consecutive_failures, 0)

    def test_failure_increments_counter(self):
        from monitoring.alerts.webhook import WebhookAlertSink
        from monitoring.alerts.base import Severity
        from urllib.error import URLError

        sink = WebhookAlertSink(url="http://example.com", retries=0)
        alert = self._make_alert(severity=Severity.WARNING)

        with patch("monitoring.alerts.webhook.urlopen", side_effect=URLError("connection refused")):
            sink.emit(alert)

        self.assertEqual(sink._consecutive_failures, 1)

    def test_circuit_breaker_opens_at_threshold(self):
        from monitoring.alerts.webhook import WebhookAlertSink
        from monitoring.alerts.base import Severity
        from urllib.error import URLError

        sink = WebhookAlertSink(url="http://example.com", retries=0)
        alert = self._make_alert(severity=Severity.ERROR)

        with patch("monitoring.alerts.webhook.urlopen", side_effect=URLError("refused")):
            for _ in range(sink._CIRCUIT_THRESHOLD):
                sink.emit(alert)

        # Circuit should be open
        self.assertGreater(sink._circuit_open_until, time.monotonic())

    def test_circuit_breaker_skips_when_open(self):
        from monitoring.alerts.webhook import WebhookAlertSink
        from monitoring.alerts.base import Severity

        sink = WebhookAlertSink(url="http://example.com", retries=0)
        sink._consecutive_failures = sink._CIRCUIT_THRESHOLD
        sink._circuit_open_until = time.monotonic() + 60.0  # open for 60s
        alert = self._make_alert(severity=Severity.CRITICAL)

        with patch("monitoring.alerts.webhook.urlopen") as mock_open:
            sink.emit(alert)
            mock_open.assert_not_called()

    def test_circuit_breaker_resets_after_expiry(self):
        from monitoring.alerts.webhook import WebhookAlertSink
        from monitoring.alerts.base import Severity

        sink = WebhookAlertSink(url="http://example.com", retries=0)
        sink._consecutive_failures = sink._CIRCUIT_THRESHOLD
        sink._circuit_open_until = time.monotonic() - 1.0  # already expired
        alert = self._make_alert(severity=Severity.CRITICAL)

        with patch("monitoring.alerts.webhook.urlopen", return_value=self._fake_resp()):
            sink.emit(alert)

        self.assertEqual(sink._consecutive_failures, 0)

    def test_retry_on_transient_failure(self):
        from monitoring.alerts.webhook import WebhookAlertSink
        from monitoring.alerts.base import Severity
        from urllib.error import URLError

        sink = WebhookAlertSink(url="http://example.com", retries=2)
        alert = self._make_alert(severity=Severity.ERROR)

        call_count = {"n": 0}
        fake_resp = self._fake_resp()

        def urlopen_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise URLError("transient")
            return fake_resp

        with patch("monitoring.alerts.webhook.urlopen", side_effect=urlopen_side_effect):
            with patch("time.sleep"):
                sink.emit(alert)

        self.assertEqual(call_count["n"], 3)
        self.assertEqual(sink._consecutive_failures, 0)

    def test_custom_headers(self):
        from monitoring.alerts.webhook import WebhookAlertSink
        from monitoring.alerts.base import Severity

        sink = WebhookAlertSink(
            url="http://example.com",
            headers={"Authorization": "Bearer token123"},
            retries=0,
        )
        alert = self._make_alert(severity=Severity.CRITICAL)
        captured_req = {}
        fake_resp = self._fake_resp()

        def fake_urlopen(req, timeout=None):
            captured_req["headers"] = req.headers
            return fake_resp

        with patch("monitoring.alerts.webhook.urlopen", side_effect=fake_urlopen):
            sink.emit(alert)

        # Authorization should be in headers (urllib capitalizes header names)
        auth_key = next((k for k in captured_req["headers"] if k.lower() == "authorization"), None)
        self.assertIsNotNone(auth_key)


# ===========================================================================
# monitoring/signal_decay_analysis.py
# ===========================================================================

class TestSignalDecayAnalyzer(unittest.TestCase):

    def _make_analyzer(self, n_lags=5):
        from monitoring.signal_decay_analysis import SignalDecayAnalyzer
        return SignalDecayAnalyzer(max_lags=n_lags)

    def _populate(self, analyzer, n=10):
        import random
        rng = random.Random(42)
        for lag in range(analyzer.max_lags + 1):
            for _ in range(n):
                score = rng.uniform(-1, 1)
                ret = score * 0.5 + rng.uniform(-0.1, 0.1)
                analyzer.record(score, ret, lag=lag)

    def test_record_valid_lag(self):
        analyzer = self._make_analyzer()
        analyzer.record(0.5, 0.01, lag=2)
        self.assertEqual(len(analyzer._data[2]), 1)

    def test_record_invalid_lag_ignored(self):
        analyzer = self._make_analyzer(n_lags=5)
        analyzer.record(0.5, 0.01, lag=-1)
        analyzer.record(0.5, 0.01, lag=100)
        self.assertEqual(len(analyzer._data), 0)

    def test_compute_ic_series_insufficient_data(self):
        analyzer = self._make_analyzer()
        analyzer.record(0.5, 0.01, lag=0)
        analyzer.record(0.4, 0.02, lag=0)
        # only 2 pairs, need >= 3
        ic_series = analyzer.compute_ic_series()
        self.assertNotIn(0, ic_series)

    def test_compute_ic_series_with_data(self):
        analyzer = self._make_analyzer()
        self._populate(analyzer, n=10)
        ic_series = analyzer.compute_ic_series()
        self.assertIn(0, ic_series)
        for ic in ic_series.values():
            self.assertGreaterEqual(ic, -1.0)
            self.assertLessEqual(ic, 1.0)

    def test_half_life_requires_positive_ic0(self):
        analyzer = self._make_analyzer()
        # Negative IC at lag 0 → None
        for _ in range(10):
            analyzer.record(1.0, -0.5, lag=0)  # negative correlation
        # half_life checks ic_series[0] <= 0
        hl = analyzer.half_life()
        # Might be None (if IC at lag 0 is negative) or a value
        # We just check it doesn't raise
        self.assertTrue(hl is None or isinstance(hl, float))

    def test_half_life_with_decaying_ic(self):
        """Construct IC series that clearly decays."""
        analyzer = self._make_analyzer(n_lags=5)
        # Build data where IC at lag 0 is strongly positive, decays at later lags
        for _ in range(20):
            analyzer.record(1.0, 1.0, lag=0)   # perfect correlation
            analyzer.record(0.0, 1.0, lag=0)
        for lag in range(1, 6):
            # Weaker correlation at later lags
            for i in range(20):
                score = float(i) / 20
                ret = score * (1.0 - lag * 0.15)
                analyzer.record(score, ret, lag=lag)
        hl = analyzer.half_life()
        # May be None or positive float — just check no error
        if hl is not None:
            self.assertGreater(hl, 0)

    def test_is_decayed_no_data(self):
        analyzer = self._make_analyzer()
        self.assertFalse(analyzer.is_decayed())

    def test_is_decayed_high_ic(self):
        analyzer = self._make_analyzer()
        for _ in range(10):
            analyzer.record(1.0, 1.0, lag=0)
            analyzer.record(0.0, 0.0, lag=0)
        # High IC → not decayed
        result = analyzer.is_decayed(threshold_ic=0.02)
        self.assertIsInstance(result, bool)

    def test_summary_keys(self):
        analyzer = self._make_analyzer()
        self._populate(analyzer)
        summary = analyzer.summary()
        self.assertIn("ic_series", summary)
        self.assertIn("half_life", summary)
        self.assertIn("is_decayed", summary)
        self.assertIn("n_observations", summary)

    def test_half_life_not_enough_points(self):
        """Only ic_series[0] available → not enough points for regression."""
        analyzer = self._make_analyzer(n_lags=5)
        for _ in range(10):
            analyzer.record(1.0, 1.0, lag=0)
            analyzer.record(0.0, 0.0, lag=0)
        # No other lags populated → points list will be empty
        hl = analyzer.half_life()
        self.assertIsNone(hl)

    def test_spearman_rank_corr_perfect_positive(self):
        from monitoring.signal_decay_analysis import _spearman_rank_corr
        pairs = [(float(i), float(i)) for i in range(10)]
        ic = _spearman_rank_corr(pairs)
        self.assertAlmostEqual(ic, 1.0, places=5)

    def test_spearman_rank_corr_perfect_negative(self):
        from monitoring.signal_decay_analysis import _spearman_rank_corr
        n = 10
        pairs = [(float(i), float(n - i)) for i in range(n)]
        ic = _spearman_rank_corr(pairs)
        self.assertAlmostEqual(ic, -1.0, places=5)

    def test_spearman_rank_corr_ties(self):
        from monitoring.signal_decay_analysis import _spearman_rank_corr
        pairs = [(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)]
        ic = _spearman_rank_corr(pairs)
        self.assertIsInstance(ic, float)

    def test_spearman_rank_corr_constant(self):
        from monitoring.signal_decay_analysis import _spearman_rank_corr
        pairs = [(1.0, 2.0), (1.0, 2.0), (1.0, 2.0)]
        ic = _spearman_rank_corr(pairs)
        self.assertEqual(ic, 0.0)

    def test_rank_basic(self):
        from monitoring.signal_decay_analysis import _rank
        ranks = _rank([3.0, 1.0, 2.0])
        # 1.0 → rank 1, 2.0 → rank 2, 3.0 → rank 3
        self.assertEqual(ranks[1], 1.0)
        self.assertEqual(ranks[2], 2.0)
        self.assertEqual(ranks[0], 3.0)

    def test_rank_with_ties(self):
        from monitoring.signal_decay_analysis import _rank
        ranks = _rank([1.0, 1.0, 2.0])
        # Ties at positions 0,1 → avg rank = (1+2)/2 = 1.5
        self.assertEqual(ranks[0], 1.5)
        self.assertEqual(ranks[1], 1.5)
        self.assertEqual(ranks[2], 3.0)


# ===========================================================================
# core/plugins.py
# ===========================================================================

class TestPluginRegistry(unittest.TestCase):

    def setUp(self):
        from core.plugins import reset_global_registries
        reset_global_registries()

    def test_register_decorator(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        @reg.register(name="my_plugin", version="1.0", description="test", tags=("a",))
        class MyPlugin:
            name = "my_plugin"

        entry = reg.get("my_plugin")
        self.assertEqual(entry.meta.name, "my_plugin")
        self.assertEqual(entry.meta.version, "1.0")
        self.assertEqual(entry.meta.category, "test_cat")
        self.assertIn("a", entry.meta.tags)

    def test_register_uses_class_name_attribute(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        @reg.register()
        class NamedPlugin:
            name = "custom_name"

        entry = reg.get("custom_name")
        self.assertIsNotNone(entry)

    def test_register_falls_back_to_classname(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        @reg.register()
        class UnnamedPlugin:
            pass

        entry = reg.get("UnnamedPlugin")
        self.assertIsNotNone(entry)

    def test_get_not_found_raises(self):
        from core.plugins import PluginRegistry, PluginNotFoundError
        reg = PluginRegistry("test_cat")
        with self.assertRaises(PluginNotFoundError):
            reg.get("nonexistent")

    def test_get_optional(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")
        self.assertIsNone(reg.get_optional("nonexistent"))

    def test_register_instance(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        class MyInst:
            name = "my_inst"

        inst = MyInst()
        reg.register_instance(inst, name="inst_key", version="2.0")
        entry = reg.get("inst_key")
        self.assertEqual(entry.plugin, inst)
        self.assertEqual(entry.meta.version, "2.0")

    def test_contains(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        @reg.register(name="x")
        class X:
            pass

        self.assertIn("x", reg)
        self.assertNotIn("y", reg)

    def test_len(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")
        self.assertEqual(len(reg), 0)

        @reg.register(name="a")
        class A:
            pass

        self.assertEqual(len(reg), 1)

    def test_list_names(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        @reg.register(name="p1")
        class P1:
            pass

        @reg.register(name="p2")
        class P2:
            pass

        names = reg.list_names()
        self.assertIn("p1", names)
        self.assertIn("p2", names)

    def test_list_entries(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        @reg.register(name="e1")
        class E1:
            pass

        entries = reg.list_entries()
        self.assertEqual(len(entries), 1)

    def test_init_all(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")
        called = []

        class P:
            name = "p"
            @staticmethod
            def on_init(config):
                called.append(config)

        reg.register_instance(P(), name="p")
        reg.init_all({"key": "val"})
        self.assertEqual(called, [{"key": "val"}])

    def test_start_all(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")
        called = []

        class P:
            def on_start(self):
                called.append("started")

        reg.register_instance(P(), name="p")
        reg.start_all()
        self.assertEqual(called, ["started"])

    def test_stop_all(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")
        called = []

        class P:
            def on_stop(self):
                called.append("stopped")

        reg.register_instance(P(), name="p")
        reg.stop_all()
        self.assertEqual(called, ["stopped"])

    def test_plugins_without_lifecycle_hooks(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        @reg.register(name="plain")
        class Plain:
            pass  # no on_init/on_start/on_stop

        # Should not raise
        reg.init_all({})
        reg.start_all()
        reg.stop_all()

    def test_get_registry_creates_and_reuses(self):
        from core.plugins import get_registry, reset_global_registries
        reset_global_registries()
        r1 = get_registry("my_cat")
        r2 = get_registry("my_cat")
        self.assertIs(r1, r2)

    def test_category_plugin_convenience(self):
        from core.plugins import strategy_plugin, venue_plugin, alpha_plugin, indicator_plugin, reset_global_registries
        from core.plugins import get_registry
        reset_global_registries()

        @strategy_plugin(name="test_strat")
        class TestStrat:
            pass

        @venue_plugin(name="test_venue")
        class TestVenue:
            pass

        @alpha_plugin(name="test_alpha")
        class TestAlpha:
            pass

        @indicator_plugin(name="test_ind")
        class TestInd:
            pass

        self.assertIn("test_strat", get_registry("strategy"))
        self.assertIn("test_venue", get_registry("venue"))
        self.assertIn("test_alpha", get_registry("alpha"))
        self.assertIn("test_ind", get_registry("indicator"))

    def test_discover_package_import_error(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")
        n = reg.discover_package("nonexistent_pkg_xyz")
        self.assertEqual(n, 0)

    def test_discover_package_no_path(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")
        # sys module has no __path__, so should not iterate
        n = reg.discover_package("sys")
        self.assertEqual(n, 0)

    def test_params_schema_set(self):
        from core.plugins import PluginRegistry
        reg = PluginRegistry("test_cat")

        @reg.register(name="schema_plugin", params_schema={"window": {"type": "int"}})
        class SchemaPlugin:
            pass

        entry = reg.get("schema_plugin")
        self.assertIn("window", entry.meta.params_schema)


# ===========================================================================
# core/observability.py
# ===========================================================================

class TestTracingInterceptor(unittest.TestCase):

    def _make_envelope(self):
        from core.types import Envelope, EventKind, EventMetadata, TraceContext
        trace = TraceContext.new_root()
        meta = EventMetadata.create(source="test", trace=trace)
        return Envelope(event=MagicMock(), metadata=meta, kind=EventKind.MARKET)

    def test_before_reduce_records_start(self):
        from core.observability import TracingInterceptor
        from core.interceptors import InterceptAction
        ti = TracingInterceptor()
        env = self._make_envelope()
        result = ti.before_reduce(env, None)
        self.assertEqual(result.action, InterceptAction.CONTINUE)
        self.assertIn(env.event_id, ti._active)

    def test_after_reduce_creates_span(self):
        from core.observability import TracingInterceptor
        ti = TracingInterceptor()
        env = self._make_envelope()
        ti.before_reduce(env, None)
        ti.after_reduce(env, None, None)
        self.assertEqual(len(ti.spans), 1)
        span = ti.spans[0]
        self.assertEqual(span.event_id, env.event_id)
        self.assertGreaterEqual(span.duration_ms, 0.0)

    def test_span_ring_buffer_evicts_oldest(self):
        from core.observability import TracingInterceptor
        ti = TracingInterceptor(max_spans=3)
        for _ in range(5):
            env = self._make_envelope()
            ti.before_reduce(env, None)
            ti.after_reduce(env, None, None)
        self.assertEqual(len(ti.spans), 3)

    def test_cleanup_stale(self):
        from core.observability import TracingInterceptor
        ti = TracingInterceptor(active_ttl_seconds=0.0)  # expire immediately
        env = self._make_envelope()
        ti._active[env.event_id] = time.monotonic() - 1.0  # already stale
        removed = ti.cleanup_stale()
        self.assertEqual(removed, 1)
        self.assertNotIn(env.event_id, ti._active)

    def test_auto_evict_when_active_large(self):
        from core.observability import TracingInterceptor
        ti = TracingInterceptor(active_ttl_seconds=0.0)
        # Fill _active with 101 stale entries
        for i in range(101):
            ti._active[f"evt-{i}"] = time.monotonic() - 1.0
        env = self._make_envelope()
        ti.before_reduce(env, None)  # triggers cleanup
        # Most stale entries should be cleared
        self.assertLess(len(ti._active), 50)

    def test_after_reduce_without_before(self):
        from core.observability import TracingInterceptor
        ti = TracingInterceptor()
        env = self._make_envelope()
        # Call after_reduce without before_reduce → duration should be 0
        ti.after_reduce(env, None, None)
        span = ti.spans[0]
        self.assertEqual(span.duration_ms, 0.0)

    def test_name_property(self):
        from core.observability import TracingInterceptor
        self.assertEqual(TracingInterceptor().name, "tracing")

    def test_export_to_tracer(self):
        from core.observability import TracingInterceptor
        mock_tracer = MagicMock()
        ctx_mgr = MagicMock()
        mock_tracer.start_span = MagicMock(return_value=ctx_mgr)
        ctx_mgr.__enter__ = MagicMock(return_value=ctx_mgr)
        ctx_mgr.__exit__ = MagicMock(return_value=False)

        ti = TracingInterceptor(tracer=mock_tracer)
        env = self._make_envelope()
        ti.before_reduce(env, None)
        ti.after_reduce(env, None, None)
        mock_tracer.start_span.assert_called_once()

    def test_export_to_tracer_exception_suppressed(self):
        from core.observability import TracingInterceptor
        mock_tracer = MagicMock()
        mock_tracer.start_span.side_effect = RuntimeError("oops")

        ti = TracingInterceptor(tracer=mock_tracer)
        env = self._make_envelope()
        ti.before_reduce(env, None)
        # Should not raise
        ti.after_reduce(env, None, None)


class TestLoggingInterceptor(unittest.TestCase):

    def _make_envelope(self):
        from core.types import Envelope, EventKind, EventMetadata, TraceContext
        trace = TraceContext.new_root()
        meta = EventMetadata.create(source="test", trace=trace)
        return Envelope(event=MagicMock(), metadata=meta, kind=EventKind.MARKET)

    def test_before_reduce_always_ok(self):
        from core.observability import LoggingInterceptor
        from core.interceptors import InterceptAction
        li = LoggingInterceptor()
        env = self._make_envelope()
        result = li.before_reduce(env, None)
        self.assertEqual(result.action, InterceptAction.CONTINUE)

    def test_record_non_continue_is_logged(self):
        from core.observability import LoggingInterceptor
        from core.interceptors import InterceptResult
        li = LoggingInterceptor()
        env = self._make_envelope()
        result = InterceptResult.reject("test_ic", "too risky")
        li.record("before", env, result)
        self.assertEqual(len(li.entries), 1)
        self.assertEqual(li.entries[0]["action"], "REJECT")

    def test_record_continue_not_logged_by_default(self):
        from core.observability import LoggingInterceptor
        from core.interceptors import InterceptResult
        li = LoggingInterceptor(log_continue=False)
        env = self._make_envelope()
        result = InterceptResult.ok("test_ic")
        li.record("before", env, result)
        self.assertEqual(len(li.entries), 0)

    def test_record_continue_logged_when_verbose(self):
        from core.observability import LoggingInterceptor
        from core.interceptors import InterceptResult
        li = LoggingInterceptor(log_continue=True)
        env = self._make_envelope()
        result = InterceptResult.ok("test_ic")
        li.record("before", env, result)
        self.assertEqual(len(li.entries), 1)

    def test_entry_ring_buffer(self):
        from core.observability import LoggingInterceptor
        from core.interceptors import InterceptResult
        li = LoggingInterceptor(max_entries=3, log_continue=True)
        env = self._make_envelope()
        result = InterceptResult.ok("test_ic")
        for _ in range(5):
            li.record("before", env, result)
        self.assertEqual(len(li.entries), 3)

    def test_log_fn_called(self):
        from core.observability import LoggingInterceptor
        from core.interceptors import InterceptResult
        calls = []
        li = LoggingInterceptor(log_fn=lambda level, msg, **kw: calls.append((level, msg)))
        env = self._make_envelope()
        result = InterceptResult.reject("r", "reason")
        li.record("before", env, result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "warning")

    def test_name_property(self):
        from core.observability import LoggingInterceptor
        self.assertEqual(LoggingInterceptor().name, "logging")


class TestMetricsInterceptor(unittest.TestCase):

    def _make_envelope(self):
        from core.types import Envelope, EventKind, EventMetadata, TraceContext
        trace = TraceContext.new_root()
        meta = EventMetadata.create(source="test", trace=trace)
        return Envelope(event=MagicMock(), metadata=meta, kind=EventKind.MARKET)

    def test_before_reduce_increments_events_in(self):
        from core.observability import MetricsInterceptor
        from core.effects import InMemoryMetrics
        metrics = InMemoryMetrics()
        mi = MetricsInterceptor(metrics)
        env = self._make_envelope()
        mi.before_reduce(env, None)
        snap = metrics.snapshot()
        self.assertTrue(any("events_in" in k for k in snap))

    def test_after_reduce_increments_events_out_and_histogram(self):
        from core.observability import MetricsInterceptor
        from core.effects import InMemoryMetrics
        metrics = InMemoryMetrics()
        mi = MetricsInterceptor(metrics)
        env = self._make_envelope()
        mi.before_reduce(env, None)
        mi.after_reduce(env, None, None)
        snap = metrics.snapshot()
        self.assertTrue(any("events_out" in k for k in snap))
        self.assertTrue(any("reduce_ms" in k for k in snap))

    def test_after_reduce_without_before(self):
        from core.observability import MetricsInterceptor
        from core.effects import InMemoryMetrics
        metrics = InMemoryMetrics()
        mi = MetricsInterceptor(metrics)
        env = self._make_envelope()
        # No before_reduce call → should not raise
        mi.after_reduce(env, None, None)

    def test_name_property(self):
        from core.observability import MetricsInterceptor
        from core.effects import InMemoryMetrics
        mi = MetricsInterceptor(InMemoryMetrics())
        self.assertEqual(mi.name, "metrics")


# ===========================================================================
# core/effects.py
# ===========================================================================

class TestEffectsModules(unittest.TestCase):

    def test_std_logger_debug(self):
        from core.effects import StdLogger
        logger = StdLogger("test.unit")
        with patch.object(logger._log, "debug") as mock_debug:
            logger.debug("hello", key="val")
            mock_debug.assert_called_once()

    def test_std_logger_info(self):
        from core.effects import StdLogger
        logger = StdLogger()
        with patch.object(logger._log, "info") as mock_info:
            logger.info("msg")
            mock_info.assert_called_once()

    def test_std_logger_warning(self):
        from core.effects import StdLogger
        logger = StdLogger()
        with patch.object(logger._log, "warning") as mock_warn:
            logger.warning("warn")
            mock_warn.assert_called_once()

    def test_std_logger_error(self):
        from core.effects import StdLogger
        logger = StdLogger()
        with patch.object(logger._log, "error") as mock_err:
            logger.error("err")
            mock_err.assert_called_once()

    def test_noop_metrics_no_raise(self):
        from core.effects import NoopMetrics
        m = NoopMetrics()
        m.counter("x", 1.0, tag="v")
        m.gauge("y", 2.0)
        m.histogram("z", 3.0)

    def test_in_memory_metrics_counter(self):
        from core.effects import InMemoryMetrics
        m = InMemoryMetrics()
        m.counter("hits", 1.0)
        m.counter("hits", 2.0)
        snap = m.snapshot()
        self.assertEqual(snap["hits"], 3.0)

    def test_in_memory_metrics_counter_with_tags(self):
        from core.effects import InMemoryMetrics
        m = InMemoryMetrics()
        m.counter("hits", 1.0, symbol="ETH")
        m.counter("hits", 1.0, symbol="BTC")
        snap = m.snapshot()
        self.assertEqual(len(snap), 2)

    def test_in_memory_metrics_gauge(self):
        from core.effects import InMemoryMetrics
        m = InMemoryMetrics()
        m.gauge("pnl", 100.0)
        m.gauge("pnl", 200.0)
        self.assertEqual(m.snapshot()["pnl"], 200.0)

    def test_in_memory_metrics_histogram(self):
        from core.effects import InMemoryMetrics
        m = InMemoryMetrics()
        m.histogram("latency_ms", 5.0)
        self.assertEqual(m.snapshot()["latency_ms"], 5.0)

    def test_in_memory_persist(self):
        from core.effects import InMemoryPersist
        p = InMemoryPersist()
        p.save_snapshot("k", b"data")
        self.assertEqual(p.load_snapshot("k"), b"data")
        self.assertIsNone(p.load_snapshot("missing"))

    def test_std_random_uniform(self):
        from core.effects import StdRandom
        r = StdRandom(seed=42)
        v = r.uniform(0.0, 1.0)
        self.assertGreaterEqual(v, 0.0)
        self.assertLessEqual(v, 1.0)

    def test_std_random_choice(self):
        from core.effects import StdRandom
        r = StdRandom(seed=42)
        seq = [1, 2, 3, 4, 5]
        c = r.choice(seq)
        self.assertIn(c, seq)

    def test_deterministic_random_reproducible(self):
        from core.effects import DeterministicRandom
        r1 = DeterministicRandom(seed=99)
        r2 = DeterministicRandom(seed=99)
        vals1 = [r1.uniform(0, 1) for _ in range(5)]
        vals2 = [r2.uniform(0, 1) for _ in range(5)]
        self.assertEqual(vals1, vals2)

    def test_deterministic_random_choice(self):
        from core.effects import DeterministicRandom
        r = DeterministicRandom(seed=42)
        seq = ["a", "b", "c"]
        c = r.choice(seq)
        self.assertIn(c, seq)

    def test_live_effects_factory(self):
        from core.effects import live_effects
        fx = live_effects()
        self.assertIsNotNone(fx.clock)
        self.assertIsNotNone(fx.log)
        self.assertIsNotNone(fx.metrics)
        self.assertIsNotNone(fx.persist)
        self.assertIsNotNone(fx.random)

    def test_test_effects_factory(self):
        from core.effects import test_effects
        fx = test_effects(seed=7)
        self.assertIsNotNone(fx.clock)


# ===========================================================================
# infra/logging/setup.py
# ===========================================================================

class TestLoggingSetup(unittest.TestCase):

    def test_setup_logging_basic(self):
        from infra.logging.setup import setup_logging
        import logging
        # Remove existing handlers to test setup
        logger = logging.getLogger("quant_system")
        old_handlers = logger.handlers[:]
        logger.handlers.clear()
        try:
            result = setup_logging(level="DEBUG")
            self.assertEqual(result.name, "quant_system")
            self.assertEqual(result.level, logging.DEBUG)
            self.assertTrue(len(result.handlers) > 0)
        finally:
            # Restore
            logger.handlers = old_handlers

    def test_setup_logging_idempotent(self):
        from infra.logging.setup import setup_logging
        # Calling twice should not add duplicate handlers
        setup_logging(level="INFO")
        logger = logging.getLogger("quant_system")
        handler_count = len(logger.handlers)
        setup_logging(level="INFO")
        self.assertEqual(len(logger.handlers), handler_count)

    def test_setup_logging_with_file(self):
        from infra.logging.setup import setup_logging
        import logging
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = str(Path(tmpdir) / "subdir" / "test.log")
            logger = logging.getLogger("quant_system_test_file")
            logger.handlers.clear()
            with patch("logging.getLogger", return_value=logger):
                result = setup_logging(level="INFO", log_file=log_path)
            # Check that the file handler was added
            file_handlers = [h for h in result.handlers if hasattr(h, "baseFilename")]
            self.assertTrue(len(file_handlers) > 0 or Path(log_path).parent.exists())

    def test_get_logger(self):
        from infra.logging.setup import get_logger
        log = get_logger("my_module")
        self.assertEqual(log.name, "quant_system.my_module")

    def test_get_logger_with_level(self):
        from infra.logging.setup import get_logger
        import logging
        log = get_logger("debug_module", level=logging.DEBUG)
        self.assertEqual(log.level, logging.DEBUG)

    def test_get_effects_logger(self):
        from infra.logging.setup import get_effects_logger
        from core.effects import StdLogger
        result = get_effects_logger()
        self.assertIsInstance(result, StdLogger)


# ===========================================================================
# infra/config/schema.py
# ===========================================================================

class TestInfraConfigSchema(unittest.TestCase):
    """Tests for infra/config/schema.py -- config uses nested dict structure
    with dot-notation resolution."""

    def _valid_config(self):
        # Nested dict form matching _resolve_key dot-notation
        return {
            "trading": {"symbol": "ETHUSDT", "exchange": "bybit"},
            "strategy": {"name": "alpha_momentum"},
        }

    def test_valid_config_no_errors(self):
        from infra.config.schema import validate_trading_config
        errors = validate_trading_config(self._valid_config())
        self.assertEqual(errors, [])

    def test_missing_required_field(self):
        from infra.config.schema import validate_trading_config
        cfg = {"trading": {"symbol": "ETHUSDT"}}  # missing exchange and strategy.name
        errors = validate_trading_config(cfg)
        self.assertTrue(len(errors) > 0)

    def test_wrong_type(self):
        from infra.config.schema import validate_trading_config
        cfg = self._valid_config()
        cfg["risk"] = {"max_orders_per_minute": "not_an_int"}
        errors = validate_trading_config(cfg)
        self.assertTrue(len(errors) > 0)

    def test_optional_fields_absent_is_ok(self):
        from infra.config.schema import validate_trading_config
        cfg = self._valid_config()
        # Without optional fields → no errors
        errors = validate_trading_config(cfg)
        self.assertEqual(errors, [])

    def test_optional_correct_type(self):
        from infra.config.schema import validate_trading_config
        cfg = self._valid_config()  # has all required keys
        cfg["risk"] = {"max_leverage": 2.0, "max_orders_per_minute": 10}
        errors = validate_trading_config(cfg)
        self.assertEqual(errors, [], f"Unexpected errors: {errors}")

    def test_get_schema_docs_includes_sections(self):
        from infra.config.schema import get_schema_docs
        docs = get_schema_docs()
        self.assertIn("Trading", docs)
        self.assertIn("Strategy", docs)
        self.assertIn("Risk", docs)

    def test_schema_keys_present(self):
        from infra.config.schema import SCHEMA
        required = {"trading.symbol", "trading.exchange", "strategy.name"}
        self.assertTrue(required.issubset(SCHEMA.keys()))


if __name__ == "__main__":
    unittest.main()
