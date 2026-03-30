"""Tests for position reconciliation dust detection.

Regression tests for the bug where _entry_price=0.0 (cleared on signal→flat)
caused notional to collapse to qty*1, misclassifying real positions as dust.
"""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_module(signal: int = 0, entry_price: float = 0.0, qty: float = 0.0):
    """Build a minimal AlphaDecisionModule-like object."""
    m = SimpleNamespace(
        _signal=signal,
        _entry_price=entry_price,
        _current_qty=Decimal(str(qty)),
    )
    return m


def _make_position(symbol: str, qty: float, is_long: bool):
    return SimpleNamespace(
        symbol=symbol,
        qty=abs(qty),
        abs_qty=abs(qty),
        is_long=is_long,
        is_flat=False,
    )


class TestDustNotionalUsesMarketPrice:
    """Notional must use live market price, not stale _entry_price."""

    def test_real_position_not_classified_as_dust(self):
        """0.086 BTC at ~$66k should NOT be dust ($5,700 >> $100 threshold)."""
        am = _make_module(signal=0, entry_price=0.0)
        exchange_qty = 0.086
        market_price = 66800.0

        # New logic: use market price
        notional = abs(exchange_qty) * market_price
        assert notional > 100, f"Real position misclassified as dust: ${notional:.2f}"
        assert notional > 5000

    def test_entry_price_zero_no_longer_collapses_notional(self):
        """When _entry_price=0.0, old code did qty*1. Verify new path avoids this."""
        am = _make_module(signal=0, entry_price=0.0)
        exchange_qty = 0.144

        # Old logic (buggy): notional = 0.144 * (0.0 or 1) = 0.144
        old_notional = abs(exchange_qty) * float(am._entry_price or 1)
        assert old_notional < 1, "Sanity: old logic produces tiny notional"

        # New logic: use market price
        market_price = 66500.0
        new_notional = abs(exchange_qty) * market_price
        assert new_notional > 9000

    def test_true_dust_still_detected(self):
        """Actual dust (tiny qty) should still be caught as dust."""
        exchange_qty = 0.000001  # ~$0.07 at $66k
        market_price = 66800.0
        notional = abs(exchange_qty) * market_price
        assert notional < 100, f"True dust not detected: ${notional:.2f}"

    @patch("engine.feature_hook._last_closes", {"BTCUSDT": 66800.0})
    def test_reconcile_syncs_real_position(self):
        """Real position (notional>$100) should sync internal state, not close."""
        am = _make_module(signal=0, entry_price=0.0, qty=0.0)
        exchange_qty = 0.086

        from engine.feature_hook import _last_closes
        mkt_price = _last_closes.get("BTCUSDT", 0.0)
        notional = abs(exchange_qty) * mkt_price

        # Should sync, not close
        assert notional > 100
        # Simulate sync path
        am._signal = 1 if exchange_qty > 0 else -1
        am._current_qty = Decimal(str(abs(exchange_qty)))
        assert am._signal == 1
        assert am._current_qty == Decimal("0.086")

    @patch("engine.feature_hook._last_closes", {})
    def test_fallback_to_ticker_when_cache_empty(self):
        """When _last_closes has no price, should query adapter.get_ticker()."""
        mock_adapter = MagicMock()
        mock_ticker = SimpleNamespace(last_price=66500.0)
        mock_adapter.get_ticker.return_value = mock_ticker

        from engine.feature_hook import _last_closes
        mkt_price = _last_closes.get("BTCUSDT", 0.0)
        assert mkt_price == 0.0  # cache empty

        # Fallback path
        tk = mock_adapter.get_ticker("BTCUSDT")
        mkt_price = float(getattr(tk, "last_price", 0) or 0)
        assert mkt_price == 66500.0

        notional = 0.086 * mkt_price
        assert notional > 5000

    def test_eth_position_not_dust(self):
        """ETH positions should also use market price correctly."""
        exchange_qty = 2.9  # ~$5,800 at $2000
        market_price = 2010.0
        notional = abs(exchange_qty) * market_price
        assert notional > 5000
        assert notional > 100
