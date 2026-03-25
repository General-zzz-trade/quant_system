"""Tests for PortfolioCombiner — AGREE ONLY mode signal combining."""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

from execution.models.balances import BalanceSnapshot, CanonicalBalance


class MockAdapter:
    def __init__(self, equity=1000.0):
        self.orders = []
        self._equity = equity

    def send_market_order(self, symbol, side, qty, reduce_only=False):
        self.orders.append({"symbol": symbol, "side": side, "qty": qty, "reduce_only": reduce_only})
        return {"orderId": f"mock_{len(self.orders)}", "status": "Filled", "retCode": 0}

    def get_balances(self):
        return {"USDT": type("B", (), {"total": self._equity, "available": self._equity})()}

    def get_ticker(self, symbol):
        return {"fundingRate": "0.0001", "lastPrice": 100.0}

    def close_position(self, symbol):
        self.orders.append({"symbol": symbol, "action": "close"})
        return {"status": "ok"}

    def get_positions(self, symbol=None):
        return []


def _canonical_usdt_snapshot(*, total: str, free: str) -> BalanceSnapshot:
    total_dec = Decimal(total)
    free_dec = Decimal(free)
    locked_dec = total_dec - free_dec
    return BalanceSnapshot(
        venue="bybit",
        ts_ms=0,
        balances=(
            CanonicalBalance.from_free_locked(
                venue="bybit",
                asset="USDT",
                free=free_dec,
                locked=locked_dec,
                ts_ms=0,
            ),
        ),
    )


def _make_combiner(adapter=None, equity=1000.0, dry_run=False, min_size=0.01):
    if adapter is None:
        adapter = MockAdapter(equity=equity)
    from scripts.ops.portfolio_combiner import PortfolioCombiner
    return PortfolioCombiner(
        adapter=adapter, symbol="ETHUSDT",
        weights={"1h": 0.5, "15m": 0.5},
        threshold=0.3, dry_run=dry_run, min_size=min_size,
    ), adapter


class TestPortfolioCombinerAgreeMode:

    def test_both_agree_long(self):
        pc, adapter = _make_combiner()
        pc.update_signal("1h", 1, 2000.0)
        result = pc.update_signal("15m", 1, 2000.0)
        assert result is not None
        assert result["to"] == 1
        assert pc._current_position == 1
        # Should have placed a buy order
        buy_orders = [o for o in adapter.orders if o.get("side") == "buy"]
        assert len(buy_orders) == 1

    def test_both_agree_short(self):
        pc, adapter = _make_combiner()
        pc.update_signal("1h", -1, 2000.0)
        result = pc.update_signal("15m", -1, 2000.0)
        assert result is not None
        assert result["to"] == -1
        assert pc._current_position == -1
        sell_orders = [o for o in adapter.orders if o.get("side") == "sell"]
        assert len(sell_orders) == 1

    def test_disagree_goes_flat(self):
        pc, adapter = _make_combiner()
        # First get into a position (both agree long)
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        # Now 15m flips short -> disagree -> flat
        result = pc.update_signal("15m", -1, 2100.0)
        assert result is not None
        assert result["to"] == 0
        assert pc._current_position == 0

    def test_one_signal_change_no_trade(self):
        """First signal changes but second is still 0 -> disagree -> flat (no change from flat)."""
        pc, _ = _make_combiner()
        result = pc.update_signal("1h", 1, 2000.0)
        # 1h=+1, 15m=0 -> disagree -> desired=0, current=0 -> no change
        assert result is None

    def test_same_signal_no_action(self):
        pc, _ = _make_combiner()
        pc.update_signal("1h", 1, 2000.0)
        # Send same signal again
        result = pc.update_signal("1h", 1, 2000.0)
        assert result is None

    def test_unknown_runner_key(self):
        pc, _ = _make_combiner()
        result = pc.update_signal("unknown_runner", 1, 2000.0)
        assert result is None

    def test_close_records_pnl(self):
        pc, _ = _make_combiner()
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        # Now disagree to close with profit
        result = pc.update_signal("15m", -1, 2200.0)
        assert "closed_pnl" in result
        assert result["closed_pnl"] > 0
        assert pc._trade_count == 1
        assert pc._win_count == 1

    def test_conviction_both_agree(self):
        """When both agree, conviction = 1.0 -> full size (capped by MAX_ORDER_NOTIONAL)."""
        pc, adapter = _make_combiner(equity=1000.0)
        pc.update_signal("1h", 1, 100.0)
        pc.update_signal("15m", 1, 100.0)
        # Full conviction: equity(1000) * leverage(10) * conviction(1.0) / price(100) = 100
        # But capped by 30%: 1000 * 0.30 * 10 / 100 = 30
        assert pc._position_size <= 30.01  # rounding tolerance

    def test_dry_run_no_orders(self):
        pc, adapter = _make_combiner(dry_run=True)
        pc.update_signal("1h", 1, 2000.0)
        result = pc.update_signal("15m", 1, 2000.0)
        assert result is not None
        assert result["to"] == 1
        # No orders should have been sent
        assert len(adapter.orders) == 0

    def test_position_size_capped(self):
        """Size capped by safety notional limit (150% of equity)."""
        pc, _ = _make_combiner(equity=10000.0)
        pc.update_signal("1h", 1, 100.0)
        pc.update_signal("15m", 1, 100.0)
        # Safety cap: 10000 * 1.50 = $15000 → 15000/100 = 150
        # 30% equity cap: 10000 * 0.30 * lev(10) / 100 = 300 (higher)
        # Capped at safety limit
        from strategy.config import get_max_order_notional
        max_size = get_max_order_notional(10000.0) / 100.0
        assert pc._position_size <= max_size + 0.01

    def test_get_status(self):
        pc, _ = _make_combiner()
        status = pc.get_status()
        assert "position" in status
        assert "signals" in status
        assert "pnl" in status
        assert "trades" in status
        assert "size" in status
        assert status["position"] == 0

    def test_transition_long_to_short(self):
        pc, adapter = _make_combiner()
        # Both agree long
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        assert pc._current_position == 1
        # 1h flips to short -> disagree -> goes flat first
        r1 = pc.update_signal("1h", -1, 2100.0)
        assert r1 is not None
        assert r1["to"] == 0
        # Now 15m also flips to short -> both agree short
        result = pc.update_signal("15m", -1, 2100.0)
        assert result is not None
        assert result["from"] == 0
        assert result["to"] == -1
        assert pc._current_position == -1
        # Should have sell order
        sell_orders = [o for o in adapter.orders if o.get("side") == "sell"]
        assert len(sell_orders) >= 1

    def test_flat_to_long_to_flat(self):
        """Full lifecycle: flat -> long -> flat."""
        pc, adapter = _make_combiner()
        # Go long
        pc.update_signal("1h", 1, 2000.0)
        r1 = pc.update_signal("15m", 1, 2000.0)
        assert r1["to"] == 1
        # Go flat (disagree)
        r2 = pc.update_signal("15m", 0, 2100.0)
        assert r2["to"] == 0
        assert pc._current_position == 0
        assert pc._position_size == 0.0

    def test_pnl_win_tracking(self):
        pc, _ = _make_combiner()
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        # Close with profit
        pc.update_signal("1h", 0, 2200.0)
        assert pc._win_count == 1
        assert pc._total_pnl > 0

    def test_pnl_loss_tracking(self):
        pc, _ = _make_combiner()
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        # Close with loss
        pc.update_signal("1h", 0, 1800.0)
        assert pc._trade_count == 1
        assert pc._win_count == 0
        assert pc._total_pnl < 0

    def test_close_then_reopen_same_direction(self):
        """Close a position and reopen in same direction tracks correctly."""
        pc, _ = _make_combiner()
        # Open long
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        # Close (disagree)
        pc.update_signal("15m", 0, 2100.0)
        assert pc._trade_count == 1
        # Reopen long
        pc.update_signal("15m", 1, 2200.0)
        assert pc._current_position == 1
        assert pc._entry_price == 2200.0

    def test_margin_skip_uses_canonical_free_balance(self):
        adapter = MockAdapter(equity=1000.0)
        adapter.get_balances = lambda: _canonical_usdt_snapshot(total="1000", free="0")
        pc, _adapter = _make_combiner(adapter=adapter)

        pc.update_signal("1h", 1, 2000.0)
        result = pc.update_signal("15m", 1, 2000.0)

        assert result is not None
        assert pc._current_position == 0
        assert pc._position_size == 0.0
        assert adapter.orders == []

    def test_trade_info_uses_actual_fill_price(self):
        adapter = MockAdapter(equity=1000.0)
        adapter.get_recent_fills = lambda symbol=None: [type("F", (), {"price": 1995.5})()]
        pc, _adapter = _make_combiner(adapter=adapter)

        with patch("scripts.ops.portfolio_combiner.time.sleep", return_value=None):
            pc.update_signal("1h", 1, 2000.0)
            result = pc.update_signal("15m", 1, 2000.0)

        assert result is not None
        assert result["fill_price"] == 1995.5
        assert pc._entry_price == 1995.5

    def test_force_flat_closes_position_and_clears_signals(self):
        pc, adapter = _make_combiner()

        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        forced = pc.force_flat(1990.0, reason="portfolio_killed")

        assert forced is not None
        assert forced["action"] == "forced_flat"
        assert forced["reason"] == "portfolio_killed"
        assert pc._current_position == 0
        assert pc._position_size == 0.0
        assert pc._signals == {"1h": 0, "15m": 0}
        close_orders = [o for o in adapter.orders if o.get("action") == "close"]
        assert len(close_orders) == 1

    def test_force_flat_clears_stale_signals_without_position(self):
        pc, _adapter = _make_combiner(dry_run=True)
        pc._signals["1h"] = 1

        forced = pc.force_flat(2000.0, reason="portfolio_killed")

        assert forced is not None
        assert forced["action"] == "forced_flat"
        assert pc._current_position == 0
        assert pc._signals == {"1h": 0, "15m": 0}

    def test_force_flat_reports_failure_when_close_fails(self):
        adapter = MockAdapter(equity=1000.0)
        adapter.close_position = lambda symbol: {"status": "error", "retCode": 10001}
        pc, _adapter = _make_combiner(adapter=adapter)

        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        forced = pc.force_flat(1990.0, reason="portfolio_killed")

        assert forced is not None
        assert forced["action"] == "forced_flat_failed"
        assert pc._current_position == 1
