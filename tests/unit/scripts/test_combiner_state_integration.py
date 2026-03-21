"""Integration tests for PortfolioCombiner + StateStore interaction.

Validates that COMBO fills correctly flow into RustStateStore,
that close failures do NOT record phantom fills, and that
PortfolioManager stays in sync with combiner trades.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch


from execution.models.balances import BalanceSnapshot, CanonicalBalance
from scripts.ops.pnl_tracker import PnLTracker
from scripts.ops.portfolio_combiner import PortfolioCombiner
from scripts.ops.portfolio_manager import PortfolioManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockAdapter:
    """Controllable adapter for testing order flows."""

    def __init__(self, equity: float = 1000.0, close_fails: bool = False):
        self.orders: list[dict] = []
        self._equity = equity
        self._close_fails = close_fails
        self._fill_price: float | None = None  # set to override fill lookup

    def send_market_order(self, symbol, side, qty, reduce_only=False):
        self.orders.append({
            "symbol": symbol, "side": side, "qty": qty,
            "reduce_only": reduce_only,
        })
        return {"orderId": f"mock_{len(self.orders)}", "status": "Filled", "retCode": 0}

    def get_balances(self):
        return _canonical_snapshot(total=str(self._equity), free=str(self._equity))

    def close_position(self, symbol):
        if self._close_fails:
            return {"status": "error", "retCode": 10001}
        self.orders.append({"symbol": symbol, "action": "close"})
        return {"status": "ok", "retCode": 0}

    def get_positions(self, symbol=None):
        return []

    def get_recent_fills(self, symbol=None):
        if self._fill_price is not None:
            return [type("Fill", (), {"price": self._fill_price})()]
        return []


class FakeStateStore:
    """Minimal in-process state store tracking fills for assertions."""

    SCALE = 100_000_000

    def __init__(self):
        self.fills: list[dict] = []
        self._position: dict[str, dict] = {}  # symbol -> {qty, side, price}

    def process_event(self, fill_event, symbol: str):
        side = fill_event.side
        qty = fill_event.qty
        price = fill_event.price
        self.fills.append({
            "symbol": symbol, "side": side, "qty": qty, "price": price,
        })
        # Simplified position tracking
        pos = self._position.get(symbol, {"qty": 0.0, "price": 0.0})
        if side == "buy":
            pos["qty"] += qty
        else:
            pos["qty"] -= qty
        pos["price"] = price
        if abs(pos["qty"]) < 1e-10:
            pos = {"qty": 0.0, "price": 0.0}
        self._position[symbol] = pos

    def get_position(self, symbol: str) -> dict:
        return self._position.get(symbol, {"qty": 0.0, "price": 0.0})


def _canonical_snapshot(*, total: str, free: str) -> BalanceSnapshot:
    total_dec = Decimal(total)
    free_dec = Decimal(free)
    locked_dec = total_dec - free_dec
    return BalanceSnapshot(
        venue="bybit", ts_ms=0,
        balances=(
            CanonicalBalance.from_free_locked(
                venue="bybit", asset="USDT",
                free=free_dec, locked=locked_dec, ts_ms=0,
            ),
        ),
    )


class FakeRustFillEvent:
    """Stand-in for RustFillEvent when _quant_hotpath is not importable."""

    def __init__(self, symbol, side, qty, price, realized_pnl=0.0, ts="0"):
        self.symbol = symbol
        self.side = side
        self.qty = qty
        self.price = price
        self.realized_pnl = realized_pnl
        self.ts = ts


def _make_combiner(
    adapter=None, equity=1000.0, dry_run=False, min_size=0.01,
    state_store=None, pnl_tracker=None,
) -> tuple[PortfolioCombiner, MockAdapter, FakeStateStore]:
    if adapter is None:
        adapter = MockAdapter(equity=equity)
    if state_store is None:
        state_store = FakeStateStore()
    combiner = PortfolioCombiner(
        adapter=adapter, symbol="ETHUSDT",
        weights={"1h": 0.5, "15m": 0.5},
        threshold=0.3, dry_run=dry_run, min_size=min_size,
        pnl_tracker=pnl_tracker,
        state_store=state_store,
    )
    return combiner, adapter, state_store


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@patch("scripts.ops.portfolio_combiner._RustFillEvent", FakeRustFillEvent)
@patch("scripts.ops.portfolio_combiner._HAS_RUST_FILL", True)
@patch("scripts.ops.portfolio_combiner.time.sleep", return_value=None)
class TestCombinerStateIntegration:
    """PortfolioCombiner + StateStore full interaction tests."""

    def test_open_position_updates_state_store(self, _sleep):
        """COMBO signal -> open -> StateStore.get_position() reflects new position."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 2005.0

        pc.update_signal("1h", 1, 2000.0)
        result = pc.update_signal("15m", 1, 2000.0)

        assert result is not None
        assert result["to"] == 1
        pos = ss.get_position("ETHUSDT")
        assert pos["qty"] > 0, "StateStore should show long position after open"
        assert len(ss.fills) == 1
        assert ss.fills[0]["side"] == "buy"

    def test_close_position_updates_state_store(self, _sleep):
        """COMBO signal -> close -> StateStore.get_position() goes to zero."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 2005.0

        # Open long
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)

        # Close (disagree)
        pc.update_signal("15m", 0, 2100.0)

        pos = ss.get_position("ETHUSDT")
        assert abs(pos["qty"]) < 1e-8, "StateStore should be flat after close"
        # Should have 2 fills: open buy + close sell
        assert len(ss.fills) == 2
        assert ss.fills[0]["side"] == "buy"
        assert ss.fills[1]["side"] == "sell"

    def test_close_failure_does_not_update_state(self, _sleep):
        """close_position failure -> StateStore keeps original position (no phantom fill)."""
        adapter = MockAdapter(equity=1000.0, close_fails=True)
        adapter._fill_price = 2005.0
        pc, adapter, ss = _make_combiner(adapter=adapter)

        # Open long (this succeeds)
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)

        fills_after_open = len(ss.fills)
        pos_after_open = ss.get_position("ETHUSDT")
        assert pos_after_open["qty"] > 0

        # Try to close (will fail)
        pc.update_signal("15m", 0, 2100.0)

        # StateStore should NOT have recorded a close fill
        assert len(ss.fills) == fills_after_open, \
            "No new fills should be recorded when close fails"
        pos_after_fail = ss.get_position("ETHUSDT")
        assert pos_after_fail["qty"] == pos_after_open["qty"], \
            "Position qty must not change on failed close"

        # Internal combiner state should also be preserved
        assert pc._current_position == 1, \
            "Combiner should still think it has a long position"

    def test_close_failure_does_not_record_phantom_pnl(self, _sleep):
        """close_position failure -> PnL tracker is NOT updated (phantom fill bug)."""
        adapter = MockAdapter(equity=1000.0, close_fails=True)
        adapter._fill_price = 2005.0
        pnl = PnLTracker()
        pc, adapter, ss = _make_combiner(adapter=adapter, pnl_tracker=pnl)

        # Open
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)

        assert pnl.trade_count == 0

        # Try to close (fails)
        pc.update_signal("15m", 0, 2100.0)

        assert pnl.trade_count == 0, \
            "PnL should NOT record a trade when close fails"
        assert pnl.total_pnl == 0.0, \
            "PnL should stay at zero when close fails"

    def test_state_store_reflects_fill_price(self, _sleep):
        """StateStore entry_price is actual fill price, not bar close."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 1995.5  # different from bar close of 2000

        pc.update_signal("1h", 1, 2000.0)
        result = pc.update_signal("15m", 1, 2000.0)

        assert result is not None
        assert result.get("fill_price") == 1995.5
        assert pc._entry_price == 1995.5
        # StateStore should have actual fill price
        assert ss.fills[0]["price"] == 1995.5

    def test_combiner_and_portfolio_manager_sync(self, _sleep):
        """After COMBO trade, PM.record_position() is called (integration wiring)."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 2005.0
        pm = PortfolioManager(adapter=adapter, dry_run=True)

        # Open long via combiner
        pc.update_signal("1h", 1, 2000.0)
        combo_trade = pc.update_signal("15m", 1, 2000.0)

        assert combo_trade is not None
        # Simulate the wiring from run_bybit_alpha.py
        desired = combo_trade.get("to", 0)
        if desired != 0:
            entry_price = combo_trade.get("fill_price", combo_trade["price"])
            pm.record_position(
                "ETHUSDT",
                pc._position_size * desired,
                entry_price,
                "COMBO",
            )
        else:
            pm.record_position("ETHUSDT", 0, 0, "COMBO")

        pm_status = pm.get_status()
        assert "ETHUSDT" in pm_status["positions"]
        assert pm_status["positions"]["ETHUSDT"]["qty"] > 0
        assert pm_status["positions"]["ETHUSDT"]["source"] == "COMBO"

    def test_signal_disagreement_closes_position(self, _sleep):
        """Two runners disagree -> close position -> StateStore goes flat."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 2005.0

        # Both agree long
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        assert pc._current_position == 1

        # 15m flips to short -> disagreement -> flat
        result = pc.update_signal("15m", -1, 2100.0)
        assert result is not None
        assert result["to"] == 0
        assert pc._current_position == 0

        pos = ss.get_position("ETHUSDT")
        assert abs(pos["qty"]) < 1e-8, "StateStore should be flat on disagreement"

    def test_both_agree_opens_position(self, _sleep):
        """Two runners both +1 -> open long -> StateStore has position."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 2005.0

        # First signal alone doesn't open
        r1 = pc.update_signal("1h", 1, 2000.0)
        assert r1 is None  # no trade (only one signal)
        assert len(ss.fills) == 0

        # Second signal agrees -> trade
        r2 = pc.update_signal("15m", 1, 2000.0)
        assert r2 is not None
        assert r2["to"] == 1
        assert len(ss.fills) == 1

        pos = ss.get_position("ETHUSDT")
        assert pos["qty"] > 0

    def test_dry_run_does_not_update_state_store(self, _sleep):
        """dry_run=True -> no fill -> StateStore unchanged."""
        pc, adapter, ss = _make_combiner(dry_run=True)

        pc.update_signal("1h", 1, 2000.0)
        result = pc.update_signal("15m", 1, 2000.0)

        assert result is not None
        assert result["to"] == 1
        # No orders sent
        assert len(adapter.orders) == 0
        # No fills in state store
        assert len(ss.fills) == 0
        pos = ss.get_position("ETHUSDT")
        assert abs(pos["qty"]) < 1e-8, "StateStore must stay flat in dry_run"

    def test_open_close_reopen_state_consistency(self, _sleep):
        """Full lifecycle: open -> close -> reopen tracks correctly."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 2005.0

        # Open long
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        pos1 = ss.get_position("ETHUSDT")
        assert pos1["qty"] > 0

        # Close (disagree)
        pc.update_signal("15m", 0, 2100.0)
        pos2 = ss.get_position("ETHUSDT")
        assert abs(pos2["qty"]) < 1e-8

        # Reopen long
        pc.update_signal("15m", 1, 2200.0)
        pos3 = ss.get_position("ETHUSDT")
        assert pos3["qty"] > 0
        assert len(ss.fills) == 3  # open + close + reopen

    def test_close_then_reverse_state_store(self, _sleep):
        """Long -> short reversal properly records both close and open fills."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 2005.0

        # Open long
        pc.update_signal("1h", 1, 2000.0)
        pc.update_signal("15m", 1, 2000.0)
        assert pc._current_position == 1

        # Disagree -> flat
        pc.update_signal("1h", -1, 2100.0)
        assert pc._current_position == 0

        # Both agree short
        pc.update_signal("15m", -1, 2100.0)
        assert pc._current_position == -1

        pos = ss.get_position("ETHUSDT")
        assert pos["qty"] < 0, "StateStore should show short position"
        # Fills: open buy, close sell, open sell
        assert len(ss.fills) == 3

    def test_pm_record_position_zero_on_close(self, _sleep):
        """PM.record_position called with 0 qty when combiner closes."""
        pc, adapter, ss = _make_combiner()
        adapter._fill_price = 2005.0
        pm = PortfolioManager(adapter=adapter, dry_run=True)

        # Open
        pc.update_signal("1h", 1, 2000.0)
        combo_trade = pc.update_signal("15m", 1, 2000.0)
        pm.record_position("ETHUSDT", pc._position_size, 2005.0, "COMBO")
        assert "ETHUSDT" in pm.get_status()["positions"]

        # Close
        combo_trade = pc.update_signal("15m", 0, 2100.0)
        assert combo_trade["to"] == 0
        pm.record_position("ETHUSDT", 0, 0, "COMBO")

        assert "ETHUSDT" not in pm.get_status()["positions"], \
            "PM should remove position when qty_signed=0"
