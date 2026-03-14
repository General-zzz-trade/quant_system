"""Tests for attribution/signal_attribution and attribution/report."""
from __future__ import annotations

import pytest

from attribution.signal_attribution import (
    attribute_by_signal,
)
from attribution.report import AttributionReport, build_report


# ── Helpers ──────────────────────────────────────────────────

def _make_chain(origin, symbol="BTC/USDT", buy_price=100.0, sell_price=110.0, qty=1.0):
    """Create a simple intent→order→fill chain with one buy+sell round-trip."""
    intents = [{"intent_id": f"i_{origin}", "origin": origin, "symbol": symbol, "side": "buy"}]
    orders = [{"order_id": f"o_{origin}", "intent_id": f"i_{origin}", "symbol": symbol, "side": "buy"}]
    fills = [
        {"fill_id": f"fb_{origin}", "order_id": f"o_{origin}", "symbol": symbol, "side": "buy",
         "qty": qty, "price": buy_price, "fee": 0.0},
        {"fill_id": f"fs_{origin}", "order_id": f"o_{origin}", "symbol": symbol, "side": "sell",
         "qty": qty, "price": sell_price, "fee": 0.0},
    ]
    return intents, orders, fills


# ── attribute_by_signal ──────────────────────────────────────

class TestAttributeBySignal:
    def test_single_origin_attribution(self):
        intents, orders, fills = _make_chain("momentum_alpha")
        report = attribute_by_signal(intents, orders, fills)
        assert "momentum_alpha" in report.by_signal
        sig = report.by_signal["momentum_alpha"]
        assert sig.realized_pnl == pytest.approx(10.0)
        assert sig.trade_count == 2

    def test_unattributed_fills(self):
        intents = []
        orders = []
        fills = [
            {"fill_id": "f1", "order_id": "orphan", "symbol": "BTC/USDT",
             "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.0},
        ]
        report = attribute_by_signal(intents, orders, fills, current_prices={"BTC/USDT": 105.0})
        assert len(report.by_signal) == 0
        assert report.unattributed_pnl == pytest.approx(5.0)  # unrealized

    def test_multiple_origins(self):
        i1, o1, f1 = _make_chain("alpha_A", buy_price=100.0, sell_price=120.0)
        i2, o2, f2 = _make_chain("alpha_B", buy_price=50.0, sell_price=40.0)
        report = attribute_by_signal(i1 + i2, o1 + o2, f1 + f2)
        assert report.by_signal["alpha_A"].realized_pnl == pytest.approx(20.0)
        assert report.by_signal["alpha_B"].realized_pnl == pytest.approx(-10.0)

    def test_win_rate(self):
        intents = [{"intent_id": "i1", "origin": "sig", "symbol": "BTC/USDT", "side": "buy"}]
        orders = [{"order_id": "o1", "intent_id": "i1"}]
        fills = [
            {"fill_id": "f1", "order_id": "o1", "symbol": "BTC/USDT", "side": "buy",
             "qty": 1.0, "price": 100.0, "fee": 0.0},
            {"fill_id": "f2", "order_id": "o1", "symbol": "BTC/USDT", "side": "sell",
             "qty": 1.0, "price": 110.0, "fee": 0.0},
        ]
        report = attribute_by_signal(intents, orders, fills)
        assert report.by_signal["sig"].win_rate == pytest.approx(1.0)

    def test_win_rate_partial(self):
        """Two round-trips: one win, one loss → win_rate 0.5."""
        intents = [{"intent_id": "i1", "origin": "sig", "symbol": "BTC/USDT", "side": "buy"}]
        orders = [{"order_id": "o1", "intent_id": "i1"}]
        fills = [
            # Win: buy 100, sell 110
            {"fill_id": "f1", "order_id": "o1", "symbol": "BTC/USDT", "side": "buy",
             "qty": 1.0, "price": 100.0, "fee": 0.0},
            {"fill_id": "f2", "order_id": "o1", "symbol": "BTC/USDT", "side": "sell",
             "qty": 1.0, "price": 110.0, "fee": 0.0},
            # Loss: buy 110, sell 100
            {"fill_id": "f3", "order_id": "o1", "symbol": "BTC/USDT", "side": "buy",
             "qty": 1.0, "price": 110.0, "fee": 0.0},
            {"fill_id": "f4", "order_id": "o1", "symbol": "BTC/USDT", "side": "sell",
             "qty": 1.0, "price": 100.0, "fee": 0.0},
        ]
        report = attribute_by_signal(intents, orders, fills)
        assert report.by_signal["sig"].win_rate == pytest.approx(0.5)

    def test_pnl_identity_sum_equals_total(self):
        i1, o1, f1 = _make_chain("alpha_A", buy_price=100.0, sell_price=115.0)
        # Add unattributed fill
        orphan = {"fill_id": "orphan", "order_id": "missing", "symbol": "ETH/USDT",
                  "side": "buy", "qty": 2.0, "price": 50.0, "fee": 1.0}
        report = attribute_by_signal(i1, o1, f1 + [orphan], current_prices={"ETH/USDT": 55.0})
        signal_pnl_sum = sum(
            s.realized_pnl + s.unrealized_pnl - s.fee_cost
            for s in report.by_signal.values()
        )
        assert report.total_pnl == pytest.approx(signal_pnl_sum + report.unattributed_pnl)

    def test_empty_inputs(self):
        report = attribute_by_signal([], [], [])
        assert len(report.by_signal) == 0
        assert report.total_pnl == 0.0
        assert report.unattributed_pnl == 0.0


# ── build_report ─────────────────────────────────────────────

class TestBuildReport:
    def test_composition(self):
        fills = [
            {"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 100.0, "fee": 0.5},
            {"symbol": "BTC/USDT", "side": "sell", "qty": 1.0, "price": 110.0, "fee": 0.5},
        ]
        report = build_report(fills, initial_equity=10_000.0)
        assert isinstance(report, AttributionReport)
        assert report.pnl.realized_pnl == pytest.approx(10.0)
        assert report.pnl.fee_cost == pytest.approx(1.0)
        assert report.cost.fee_bps > 0
        # net_return_pct = total_pnl / equity * 100
        assert report.net_return_pct == pytest.approx(report.pnl.total_pnl / 10_000.0 * 100)

    def test_with_period_label(self):
        report = build_report([], initial_equity=1000.0, period_label="2024-Q1")
        assert report.period_label == "2024-Q1"

    def test_zero_equity(self):
        report = build_report([], initial_equity=0.0)
        assert report.net_return_pct == 0.0

    def test_with_reference_prices(self):
        fills = [{"symbol": "BTC/USDT", "side": "buy", "qty": 1.0, "price": 101.0, "fee": 0.0}]
        ref = {"BTC/USDT": 100.0}
        report = build_report(fills, initial_equity=10_000.0, reference_prices=ref)
        assert report.cost.slippage_bps > 0
