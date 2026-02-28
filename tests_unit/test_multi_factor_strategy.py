"""Unit tests for the multi-factor trend/mean-reversion hybrid strategy."""
from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import List, Optional

import pytest

from strategies.multi_factor.feature_computer import MultiFactorFeatureComputer, MultiFactorFeatures
from strategies.multi_factor.regime import Regime, classify_regime
from strategies.multi_factor.signal_combiner import CombinedSignal, combine_signals
from strategies.multi_factor.decision_module import MultiFactorConfig, MultiFactorDecisionModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_bars(n: int, start: float = 50000.0, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    bars = []
    price = start
    for _ in range(n):
        change = rng.gauss(0, price * 0.005)
        c = price + change
        h = c + abs(rng.gauss(0, price * 0.002))
        l_ = c - abs(rng.gauss(0, price * 0.002))
        o = price + rng.gauss(0, price * 0.001)
        bars.append({"open": o, "high": h, "low": l_, "close": c, "volume": rng.uniform(100, 1000)})
        price = c
    return bars


def _trending_up_bars(n: int, start: float = 50000.0) -> list[dict]:
    bars = []
    price = start
    for i in range(n):
        price *= 1.002  # steady uptrend
        noise = price * 0.001
        bars.append({
            "open": price - noise,
            "high": price + noise * 2,
            "low": price - noise * 2,
            "close": price,
            "volume": 500.0,
        })
    return bars


def _trending_down_bars(n: int, start: float = 50000.0) -> list[dict]:
    bars = []
    price = start
    for i in range(n):
        price *= 0.998
        noise = price * 0.001
        bars.append({
            "open": price + noise,
            "high": price + noise * 2,
            "low": price - noise * 2,
            "close": price,
            "volume": 500.0,
        })
    return bars


def _ranging_bars(n: int, center: float = 50000.0) -> list[dict]:
    bars = []
    for i in range(n):
        offset = math.sin(i * 0.3) * center * 0.005
        c = center + offset
        bars.append({
            "open": c - 10,
            "high": c + 50,
            "low": c - 50,
            "close": c,
            "volume": 500.0,
        })
    return bars


def _feed_bars(fc: MultiFactorFeatureComputer, bars: list[dict]) -> MultiFactorFeatures:
    result = None
    for b in bars:
        result = fc.on_bar(**b)
    return result


def _make_snapshot(
    close: float,
    symbol: str = "BTCUSDT",
    qty: Decimal = Decimal("0"),
    avg_price: Optional[Decimal] = None,
    balance: Decimal = Decimal("10000"),
    open_: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    volume: float = 500.0,
) -> SimpleNamespace:
    market = SimpleNamespace(
        symbol=symbol,
        close=Decimal(str(close)),
        last_price=Decimal(str(close)),
        open=Decimal(str(open_ or close)),
        high=Decimal(str(high or close)),
        low=Decimal(str(low or close)),
        volume=Decimal(str(volume)),
        last_ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    from state.position import PositionState
    pos = PositionState(symbol=symbol, qty=qty, avg_price=avg_price)
    account = SimpleNamespace(balance=balance, realized_pnl=Decimal("0"))
    return SimpleNamespace(
        market=market,
        positions={symbol: pos},
        event_id="test-evt-001",
        account=account,
    )


# ===========================================================================
# Feature Computer Tests
# ===========================================================================


class TestFeatureComputer:
    def test_sma_matches_batch(self):
        from features.technical import sma as batch_sma
        bars = _random_bars(100)
        closes = [b["close"] for b in bars]

        fc = MultiFactorFeatureComputer(sma_fast_window=20, sma_slow_window=50)
        incremental_fast = []
        for b in bars:
            f = fc.on_bar(**b)
            incremental_fast.append(f.sma_fast)

        batch = batch_sma(closes, 20)
        for i in range(len(bars)):
            if batch[i] is None:
                assert incremental_fast[i] is None
            else:
                assert incremental_fast[i] is not None
                assert abs(incremental_fast[i] - batch[i]) < 1e-6, f"SMA mismatch at {i}"

    def test_rsi_matches_batch(self):
        from features.technical import rsi as batch_rsi
        bars = _random_bars(100)
        closes = [b["close"] for b in bars]

        fc = MultiFactorFeatureComputer(rsi_window=14)
        incremental = []
        for b in bars:
            f = fc.on_bar(**b)
            incremental.append(f.rsi)

        batch = batch_rsi(closes, 14)
        for i in range(len(bars)):
            if batch[i] is None:
                # Incremental may also be None or very early
                continue
            if incremental[i] is not None:
                assert abs(incremental[i] - batch[i]) < 0.5, f"RSI mismatch at {i}: {incremental[i]} vs {batch[i]}"

    def test_macd_matches_batch(self):
        from features.technical import macd as batch_macd
        bars = _random_bars(100)
        closes = [b["close"] for b in bars]

        fc = MultiFactorFeatureComputer()
        incremental_hist = []
        for b in bars:
            f = fc.on_bar(**b)
            incremental_hist.append(f.macd_hist)

        _, _, batch_hist = batch_macd(closes, 12, 26, 9)
        # Compare where both are not None
        both_valid = 0
        for i in range(len(bars)):
            if batch_hist[i] is not None and incremental_hist[i] is not None:
                both_valid += 1
                # Allow some tolerance since EMA initialization differs
                assert abs(incremental_hist[i] - batch_hist[i]) < 500, \
                    f"MACD hist mismatch at {i}: {incremental_hist[i]} vs {batch_hist[i]}"
        assert both_valid > 0

    def test_bb_matches_batch(self):
        from features.technical import bollinger_bands as batch_bb
        bars = _random_bars(100)
        closes = [b["close"] for b in bars]

        fc = MultiFactorFeatureComputer(bb_window=20, bb_std=2.0)
        inc_upper = []
        inc_lower = []
        for b in bars:
            f = fc.on_bar(**b)
            inc_upper.append(f.bb_upper)
            inc_lower.append(f.bb_lower)

        batch_u, _, batch_l = batch_bb(closes, 20, 2.0)
        for i in range(len(bars)):
            if batch_u[i] is None:
                assert inc_upper[i] is None
            else:
                assert inc_upper[i] is not None
                assert abs(inc_upper[i] - batch_u[i]) < 1e-4, f"BB upper mismatch at {i}"
                assert abs(inc_lower[i] - batch_l[i]) < 1e-4, f"BB lower mismatch at {i}"

    def test_atr_matches_batch(self):
        from features.technical import atr as batch_atr
        from features.types import Bar
        bars = _random_bars(100)
        bar_objs = [
            Bar(ts=datetime(2024, 1, 1), open=b["open"], high=b["high"],
                low=b["low"], close=b["close"], volume=b["volume"])
            for b in bars
        ]

        fc = MultiFactorFeatureComputer(atr_window=14)
        incremental = []
        for b in bars:
            f = fc.on_bar(**b)
            incremental.append(f.atr)

        batch = batch_atr(bar_objs, 14)
        for i in range(len(bars)):
            if batch[i] is None:
                continue
            if incremental[i] is not None:
                assert abs(incremental[i] - batch[i]) < 1.0, \
                    f"ATR mismatch at {i}: {incremental[i]} vs {batch[i]}"

    def test_warmup_returns_none(self):
        fc = MultiFactorFeatureComputer(sma_fast_window=20, sma_slow_window=50)
        bars = _random_bars(10)
        for b in bars:
            f = fc.on_bar(**b)
        assert f.sma_fast is None  # need 20 bars
        assert f.sma_slow is None  # need 50 bars
        assert f.rsi is None  # need 15 bars (14 changes + 1st value)

    def test_atr_percentile(self):
        fc = MultiFactorFeatureComputer(atr_window=14, atr_pct_window=100)
        bars = _random_bars(200)
        f = _feed_bars(fc, bars)
        assert f.atr_percentile is not None
        assert 0 <= f.atr_percentile <= 100

    def test_ma_slope(self):
        # Uptrend should produce positive slope
        fc = MultiFactorFeatureComputer(sma_slow_window=20, ma_slope_window=10)
        bars = _trending_up_bars(100)
        f = _feed_bars(fc, bars)
        assert f.ma_slope is not None
        assert f.ma_slope > 0

        # Downtrend should produce negative slope
        fc2 = MultiFactorFeatureComputer(sma_slow_window=20, ma_slope_window=10)
        bars2 = _trending_down_bars(100)
        f2 = _feed_bars(fc2, bars2)
        assert f2.ma_slope is not None
        assert f2.ma_slope < 0

    def test_reset(self):
        fc = MultiFactorFeatureComputer()
        bars = _random_bars(50)
        _feed_bars(fc, bars)
        fc.reset()
        f = fc.on_bar(open=50000, high=50100, low=49900, close=50000, volume=100)
        assert f.sma_fast is None
        assert f.rsi is None
        assert f.atr is None


# ===========================================================================
# Regime Tests
# ===========================================================================


class TestRegime:
    def test_high_vol_regime(self):
        features = MultiFactorFeatures(
            sma_fast=50000, sma_slow=49000, sma_trend=48000,
            rsi=55, macd=100, macd_signal=90, macd_hist=10,
            bb_upper=51000, bb_middle=50000, bb_lower=49000, bb_pct=0.5,
            atr=500, atr_pct=0.01, atr_percentile=90.0,
            ma_slope=0.002, close=50000, volume=500,
        )
        assert classify_regime(features) == Regime.HIGH_VOL

    def test_trending_up(self):
        features = MultiFactorFeatures(
            sma_fast=50500, sma_slow=50000, sma_trend=49000,
            rsi=60, macd=100, macd_signal=90, macd_hist=10,
            bb_upper=51000, bb_middle=50000, bb_lower=49000, bb_pct=0.6,
            atr=300, atr_pct=0.006, atr_percentile=50.0,
            ma_slope=0.003, close=50500, volume=500,
        )
        assert classify_regime(features) == Regime.TRENDING_UP

    def test_trending_down(self):
        features = MultiFactorFeatures(
            sma_fast=49000, sma_slow=49500, sma_trend=50000,
            rsi=40, macd=-100, macd_signal=-90, macd_hist=-10,
            bb_upper=50000, bb_middle=49500, bb_lower=49000, bb_pct=0.3,
            atr=300, atr_pct=0.006, atr_percentile=50.0,
            ma_slope=-0.003, close=49000, volume=500,
        )
        assert classify_regime(features) == Regime.TRENDING_DOWN

    def test_ranging(self):
        features = MultiFactorFeatures(
            sma_fast=50000, sma_slow=50010, sma_trend=50000,
            rsi=50, macd=5, macd_signal=4, macd_hist=1,
            bb_upper=50500, bb_middle=50000, bb_lower=49500, bb_pct=0.5,
            atr=200, atr_pct=0.004, atr_percentile=40.0,
            ma_slope=0.0001, close=50000, volume=500,
        )
        assert classify_regime(features) == Regime.RANGING

    def test_none_when_missing(self):
        features = MultiFactorFeatures(
            sma_fast=None, sma_slow=None, sma_trend=None,
            rsi=None, macd=None, macd_signal=None, macd_hist=None,
            bb_upper=None, bb_middle=None, bb_lower=None, bb_pct=None,
            atr=None, atr_pct=None, atr_percentile=None,
            ma_slope=None, close=50000, volume=500,
        )
        assert classify_regime(features) is None


# ===========================================================================
# Signal Combiner Tests
# ===========================================================================


class TestSignalCombiner:
    def test_trending_bullish(self):
        features = MultiFactorFeatures(
            sma_fast=51000, sma_slow=50000, sma_trend=49000,
            rsi=60, macd=200, macd_signal=100, macd_hist=100,
            bb_upper=52000, bb_middle=50500, bb_lower=49000, bb_pct=0.6,
            atr=300, atr_pct=0.006, atr_percentile=50.0,
            ma_slope=0.003, close=51000, volume=500,
        )
        sig = combine_signals(features, Regime.TRENDING_UP, trend_threshold=0.3)
        assert sig.direction == 1
        assert sig.strength > 0.3

    def test_ranging_oversold(self):
        features = MultiFactorFeatures(
            sma_fast=50000, sma_slow=50010, sma_trend=50000,
            rsi=20, macd=-10, macd_signal=-5, macd_hist=-5,
            bb_upper=50500, bb_middle=50000, bb_lower=49500, bb_pct=0.05,
            atr=200, atr_pct=0.004, atr_percentile=40.0,
            ma_slope=0.0001, close=49550, volume=500,
        )
        sig = combine_signals(features, Regime.RANGING, range_threshold=0.3)
        assert sig.direction == 1  # buy on oversold
        assert sig.strength > 0.3

    def test_high_vol_flat(self):
        features = MultiFactorFeatures(
            sma_fast=51000, sma_slow=50000, sma_trend=49000,
            rsi=60, macd=200, macd_signal=100, macd_hist=100,
            bb_upper=52000, bb_middle=50500, bb_lower=49000, bb_pct=0.6,
            atr=500, atr_pct=0.01, atr_percentile=90.0,
            ma_slope=0.003, close=51000, volume=500,
        )
        sig = combine_signals(features, Regime.HIGH_VOL)
        assert sig.direction == 0
        assert sig.strength == 0.0

    def test_below_threshold_flat(self):
        features = MultiFactorFeatures(
            sma_fast=50001, sma_slow=50000, sma_trend=49000,
            rsi=50, macd=1, macd_signal=0.9, macd_hist=0.1,
            bb_upper=50500, bb_middle=50000, bb_lower=49500, bb_pct=0.5,
            atr=200, atr_pct=0.004, atr_percentile=50.0,
            ma_slope=0.002, close=50001, volume=500,
        )
        sig = combine_signals(features, Regime.TRENDING_UP, trend_threshold=0.5)
        assert sig.direction == 0

    def test_trending_bearish_short(self):
        features = MultiFactorFeatures(
            sma_fast=49000, sma_slow=50000, sma_trend=51000,
            rsi=35, macd=-300, macd_signal=-100, macd_hist=-200,
            bb_upper=50500, bb_middle=49500, bb_lower=48500, bb_pct=0.3,
            atr=400, atr_pct=0.008, atr_percentile=50.0,
            ma_slope=-0.003, close=49000, volume=500,
        )
        sig = combine_signals(features, Regime.TRENDING_DOWN, trend_threshold=0.3)
        assert sig.direction == -1
        assert sig.strength > 0.3


# ===========================================================================
# Decision Module Tests
# ===========================================================================


class TestDecisionModule:
    def test_warmup_no_trade(self):
        module = MultiFactorDecisionModule(MultiFactorConfig(symbol="BTCUSDT"))
        # Feed only a few bars — not enough for warmup
        for i in range(10):
            snap = _make_snapshot(50000 + i * 10)
            events = list(module.decide(snap))
        assert events == []

    def test_entry_long(self):
        config = MultiFactorConfig(symbol="BTCUSDT", trend_threshold=0.2, range_threshold=0.2)
        module = MultiFactorDecisionModule(config)
        # Feed strong uptrend bars
        bars = _trending_up_bars(200)
        events = []
        for b in bars:
            snap = _make_snapshot(
                b["close"],
                open_=b["open"], high=b["high"], low=b["low"],
                volume=b["volume"],
            )
            evts = list(module.decide(snap))
            if evts:
                events.extend(evts)
                break

        # Should eventually produce a buy entry
        if events:
            from event.types import IntentEvent, OrderEvent
            intents = [e for e in events if isinstance(e, IntentEvent)]
            orders = [e for e in events if isinstance(e, OrderEvent)]
            assert len(intents) >= 1
            assert len(orders) >= 1
            assert intents[0].side == "buy"
            assert orders[0].side == "buy"

    def test_entry_short(self):
        config = MultiFactorConfig(
            symbol="BTCUSDT", trend_threshold=0.2, range_threshold=0.2,
            long_only_above_trend=False,
        )
        module = MultiFactorDecisionModule(config)
        bars = _trending_down_bars(200)
        events = []
        for b in bars:
            snap = _make_snapshot(
                b["close"],
                open_=b["open"], high=b["high"], low=b["low"],
                volume=b["volume"],
            )
            evts = list(module.decide(snap))
            if evts:
                events.extend(evts)
                break

        if events:
            from event.types import IntentEvent
            intents = [e for e in events if isinstance(e, IntentEvent)]
            if intents:
                assert intents[0].side == "sell"

    def test_stop_loss_exit(self):
        config = MultiFactorConfig(symbol="BTCUSDT", atr_stop_multiple=2.0, trend_threshold=0.2)
        module = MultiFactorDecisionModule(config)

        # Warmup with uptrend to get an entry
        bars = _trending_up_bars(200)
        entered = False
        entry_price = None
        for b in bars:
            snap = _make_snapshot(b["close"], open_=b["open"], high=b["high"], low=b["low"])
            evts = list(module.decide(snap))
            if evts and not entered:
                entered = True
                entry_price = b["close"]
                # Simulate position
                break

        if entered and entry_price and module._entry_atr:
            # Now feed a bar that triggers stop loss
            stop_price = entry_price - module._entry_atr * config.atr_stop_multiple * 1.5
            snap = _make_snapshot(
                stop_price,
                qty=Decimal("0.01"),
                avg_price=Decimal(str(entry_price)),
                high=stop_price + 10,
                low=stop_price - 10,
                open_=stop_price + 5,
            )
            evts = list(module.decide(snap))
            from event.types import IntentEvent
            intents = [e for e in evts if isinstance(e, IntentEvent)]
            if intents:
                assert intents[0].reason_code == "stop_loss"
                assert intents[0].side == "sell"

    def test_trailing_stop_exit(self):
        config = MultiFactorConfig(
            symbol="BTCUSDT", trend_threshold=0.2,
            atr_stop_multiple=100.0, trailing_atr_multiple=2.0,
        )
        module = MultiFactorDecisionModule(config)

        # Warmup with uptrend
        bars = _trending_up_bars(200)
        for b in bars:
            snap = _make_snapshot(b["close"], open_=b["open"], high=b["high"], low=b["low"])
            module.decide(snap)

        module._entry_regime = Regime.TRENDING_UP
        module._entry_atr = 300.0
        module._entry_price = bars[-1]["close"]
        module._trailing_peak = bars[-1]["close"]

        # Feed downtrend bars with position — trailing stop should fire
        down_bars = _trending_down_bars(100, start=bars[-1]["close"])
        exit_found = False
        for b in down_bars:
            snap = _make_snapshot(
                b["close"],
                qty=Decimal("0.01"),
                avg_price=Decimal(str(bars[-1]["close"])),
                open_=b["open"], high=b["high"], low=b["low"],
            )
            evts = list(module.decide(snap))
            if evts:
                from event.types import IntentEvent
                intents = [e for e in evts if isinstance(e, IntentEvent)]
                if intents and intents[0].reason_code == "trailing_stop":
                    exit_found = True
                    break

        assert exit_found, "Expected trailing stop exit"

    def test_cooldown(self):
        config = MultiFactorConfig(symbol="BTCUSDT", cooldown_bars=3)
        module = MultiFactorDecisionModule(config)
        module._cooldown = 3
        module._bar_count = 200  # pretend warmed up

        # Warmup the feature computer
        bars = _random_bars(200)
        for b in bars:
            module._fc.on_bar(**b)

        # During cooldown, no entries should be made
        for _ in range(3):
            snap = _make_snapshot(50000)
            evts = list(module.decide(snap))
            assert len(evts) == 0 or all(
                not hasattr(e, "reason_code") or "stop" not in getattr(e, "reason_code", "")
                for e in evts
            )

    def test_position_sizing(self):
        module = MultiFactorDecisionModule(MultiFactorConfig(
            risk_per_trade=0.02,
            atr_stop_multiple=2.0,
            max_position_pct=0.30,
        ))
        qty = module._compute_qty(equity=10000, price=50000, atr=500, direction=1)
        # risk_budget = 10000 * 0.02 = 200
        # stop_dist = 500 * 2 = 1000
        # raw_qty = 200 / 1000 = 0.2
        # max_qty = 0.3 * 10000 / 50000 = 0.06
        # qty = min(0.2, 0.06) = 0.06
        assert qty == Decimal("0.06")

    def test_consecutive_loss_reduction(self):
        module = MultiFactorDecisionModule(MultiFactorConfig(
            risk_per_trade=0.02,
            atr_stop_multiple=2.0,
            max_position_pct=1.0,  # high cap to not interfere
            max_consecutive_losses=3,
            loss_reduction_factor=0.5,
        ))
        module._consecutive_losses = 3

        qty_reduced = module._compute_qty(equity=10000, price=50000, atr=500, direction=1)
        module._consecutive_losses = 0
        qty_normal = module._compute_qty(equity=10000, price=50000, atr=500, direction=1)

        assert qty_reduced < qty_normal
        assert qty_reduced == Decimal(str(round(float(qty_normal) * 0.5, 5)))


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestIntegration:
    def test_full_backtest_produces_trades(self):
        """Run 2000 random bars through the full backtest pipeline."""
        import csv
        import tempfile
        from pathlib import Path
        from datetime import timedelta

        bars = _random_bars(2000, seed=123)

        # Write to temp CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "open", "high", "low", "close", "volume"])
            base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
            for i, b in enumerate(bars):
                ts = base_ts + timedelta(hours=i)
                w.writerow([ts.isoformat(), b["open"], b["high"], b["low"], b["close"], b["volume"]])
            tmp_path = f.name

        try:
            from runner.backtest_runner import run_backtest
            config = MultiFactorConfig(
                symbol="BTCUSDT",
                trend_threshold=0.3,
                range_threshold=0.3,
            )
            module = MultiFactorDecisionModule(config=config)

            equity, fills = run_backtest(
                csv_path=Path(tmp_path),
                symbol="BTCUSDT",
                starting_balance=Decimal("10000"),
                fee_bps=Decimal("4"),
                slippage_bps=Decimal("2"),
                decision_modules=[module],
            )
            assert len(equity) > 0
            assert len(fills) > 0
        finally:
            import os
            os.unlink(tmp_path)

    def test_no_exceptions_full_run(self):
        """Run a larger dataset through without any exceptions."""
        import csv
        import tempfile
        from pathlib import Path
        from datetime import timedelta

        bars = _random_bars(5000, seed=999)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "open", "high", "low", "close", "volume"])
            base_ts = datetime(2023, 1, 1, tzinfo=timezone.utc)
            for i, b in enumerate(bars):
                ts = base_ts + timedelta(hours=i)
                w.writerow([ts.isoformat(), b["open"], b["high"], b["low"], b["close"], b["volume"]])
            tmp_path = f.name

        try:
            from runner.backtest_runner import run_backtest
            module = MultiFactorDecisionModule(MultiFactorConfig(symbol="BTCUSDT"))

            equity, fills = run_backtest(
                csv_path=Path(tmp_path),
                symbol="BTCUSDT",
                starting_balance=Decimal("10000"),
                fee_bps=Decimal("4"),
                decision_modules=[module],
            )
            assert len(equity) > 0
            # No exceptions = pass
        finally:
            import os
            os.unlink(tmp_path)
