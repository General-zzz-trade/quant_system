"""Tests for RSI(5) 5-minute Polymarket strategy."""
from __future__ import annotations

import os
import pytest
import numpy as np

from polymarket.strategies.rsi_5m import RSI5mStrategy, RSI5mSignal
from polymarket.strategies.runner_5m import PolymarketRSI5mRunner, Runner5mConfig


class TestRSI5mStrategy:
    """Unit tests for RSI5mStrategy."""

    def test_rsi_generates_up_signal_on_oversold(self):
        """Feed declining prices so RSI drops below 25 -> expect 'up' signal."""
        strategy = RSI5mStrategy(period=5, oversold=25.0, overbought=75.0, min_bars=6)

        # Start at 100, drop steadily to push RSI below 25
        prices = [100.0]
        for i in range(1, 30):
            # Mostly down moves with tiny bounces
            if i % 7 == 0:
                prices.append(prices[-1] + 0.1)
            else:
                prices.append(prices[-1] - 1.0)

        signal = None
        for p in prices:
            result = strategy.update(p)
            if result is not None:
                signal = result

        assert signal is not None, "Expected an oversold signal"
        assert signal.direction == "up"
        assert signal.rsi_value < 25.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_rsi_generates_down_signal_on_overbought(self):
        """Feed rising prices so RSI goes above 75 -> expect 'down' signal."""
        strategy = RSI5mStrategy(period=5, oversold=25.0, overbought=75.0, min_bars=6)

        # Start at 100, rise steadily to push RSI above 75
        prices = [100.0]
        for i in range(1, 30):
            if i % 7 == 0:
                prices.append(prices[-1] - 0.1)
            else:
                prices.append(prices[-1] + 1.0)

        signal = None
        for p in prices:
            result = strategy.update(p)
            if result is not None:
                signal = result

        assert signal is not None, "Expected an overbought signal"
        assert signal.direction == "down"
        assert signal.rsi_value > 75.0
        assert 0.0 <= signal.confidence <= 1.0

    def test_rsi_no_signal_in_neutral(self):
        """Normal oscillating prices should not trigger a signal."""
        strategy = RSI5mStrategy(period=5, oversold=25.0, overbought=75.0, min_bars=6)

        # Alternating up/down of similar magnitude -> RSI stays near 50
        prices = [100.0]
        for i in range(1, 50):
            if i % 2 == 0:
                prices.append(prices[-1] + 1.0)
            else:
                prices.append(prices[-1] - 1.0)

        signals = []
        for p in prices:
            result = strategy.update(p)
            if result is not None:
                signals.append(result)

        assert len(signals) == 0, f"Expected no signals in neutral RSI, got {len(signals)}"

    def test_rsi_needs_warmup(self):
        """First min_bars bars should never produce a signal."""
        strategy = RSI5mStrategy(period=5, oversold=25.0, overbought=75.0, min_bars=10)

        # Even with extreme prices, no signal during warmup
        prices = [100.0 - i * 5.0 for i in range(11)]

        signals_before_warmup = []
        for i, p in enumerate(prices[:10]):
            result = strategy.update(p)
            if result is not None:
                signals_before_warmup.append((i, result))

        assert len(signals_before_warmup) == 0, "Should not signal during warmup"

    def test_rsi_reset_clears_state(self):
        """After reset, strategy should behave as freshly constructed."""
        strategy = RSI5mStrategy(period=5, oversold=25.0, overbought=75.0, min_bars=6)

        # Feed some data
        for i in range(20):
            strategy.update(100.0 + i)

        assert strategy.bar_count > 0

        strategy.reset()

        assert strategy.bar_count == 0
        assert strategy.current_rsi == 50.0
        # First update after reset should return None (sets prev_close)
        assert strategy.update(100.0) is None

    def test_signal_is_frozen_dataclass(self):
        """RSI5mSignal should be immutable."""
        signal = RSI5mSignal(direction="up", rsi_value=20.0, confidence=0.5, timestamp_ms=123)
        assert signal.direction == "up"
        assert signal.timestamp_ms == 123
        with pytest.raises(AttributeError):
            signal.direction = "down"  # type: ignore[misc]


class TestPolymarketRSI5mRunner:
    """Tests for the runner / backtest."""

    def test_backtest_returns_stats(self):
        """Run backtest on small synthetic data and check stats structure."""
        config = Runner5mConfig(dry_run=True, bet_size_usd=10.0)
        runner = PolymarketRSI5mRunner(config)

        # Create data with a trend reversal to trigger signals
        n = 200
        prices = []
        p = 100.0
        for i in range(n):
            if i < 50:
                p -= 0.5  # downtrend
            elif i < 100:
                p += 0.5  # uptrend
            elif i < 150:
                p -= 0.5  # downtrend
            else:
                p += 0.5  # uptrend
            prices.append(p)

        result = runner.run_backtest(prices)

        assert "total_bets" in result
        assert "wins" in result
        assert "accuracy" in result
        assert "total_pnl" in result
        assert "pnl_per_bet" in result
        assert "days" in result
        assert "bets_per_day" in result
        assert result["total_bets"] >= 0
        if result["total_bets"] > 0:
            assert 0 <= result["accuracy"] <= 1

    def test_on_kline_dry_run(self):
        """on_kline in dry_run mode should return bet dict on signal."""
        config = Runner5mConfig(dry_run=True)
        runner = PolymarketRSI5mRunner(config)

        # Feed declining prices to trigger oversold
        bets = []
        p = 100.0
        for i in range(50):
            p -= 1.0
            bet = runner.on_kline(p, timestamp_ms=i * 60000)
            if bet is not None:
                bets.append(bet)

        assert len(bets) > 0
        assert bets[0]["direction"] == "up"
        assert "rsi" in bets[0]
        assert "confidence" in bets[0]
        assert bets[0]["size_usd"] == 10.0

    @pytest.mark.skipif(
        not os.path.exists("/quant_system/data_files/BTCUSDT_1m.csv"),
        reason="BTC 1m data file not available",
    )
    def test_backtest_accuracy_matches_expected(self):
        """Run backtest on real BTC 1m data; verify accuracy ~52-53%."""
        import pandas as pd

        df = pd.read_csv("/quant_system/data_files/BTCUSDT_1m.csv", nrows=200000)
        closes = df["close"].values
        timestamps = df["open_time"].values

        config = Runner5mConfig(dry_run=True, bet_size_usd=10.0)
        runner = PolymarketRSI5mRunner(config)
        result = runner.run_backtest(closes, timestamps)

        assert result["total_bets"] > 1000, (
            f"Expected >1000 bets on 200k bars, got {result['total_bets']}"
        )
        assert 0.51 <= result["accuracy"] <= 0.55, (
            f"Expected accuracy 51-55%, got {result['accuracy']:.4f}"
        )
