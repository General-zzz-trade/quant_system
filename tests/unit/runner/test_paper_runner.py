"""Tests for runner/paper_runner.py — PaperPortfolio and _SimpleMAStrategy."""
from __future__ import annotations

from decimal import Decimal


from runner.paper_runner import PaperPortfolio, _SimpleMAStrategy


# ── PaperPortfolio ──────────────────────────────────────────


class TestPaperPortfolio:
    def _make(
        self,
        balance: str = "10000",
        fee_bps: str = "4",
        slippage_bps: str = "0",
    ) -> PaperPortfolio:
        return PaperPortfolio(
            balance=Decimal(balance),
            fee_bps=Decimal(fee_bps),
            slippage_bps=Decimal(slippage_bps),
        )

    def test_buy_deducts_balance(self):
        p = self._make(fee_bps="0", slippage_bps="0")
        fill = p.execute("buy", Decimal("1"), Decimal("100"))
        assert fill["side"] == "buy"
        assert Decimal(fill["balance"]) == Decimal("9900")
        assert Decimal(fill["position_qty"]) == Decimal("1")

    def test_buy_with_fee(self):
        p = self._make(fee_bps="10", slippage_bps="0")  # 0.1%
        p.execute("buy", Decimal("1"), Decimal("1000"))
        # fee = 1 * 1000 * 10/10000 = 1.0
        # balance = 10000 - 1000 - 1 = 8999
        assert p._balance == Decimal("8999")
        assert p._total_fees == Decimal("1")

    def test_sell_adds_balance(self):
        p = self._make(fee_bps="0", slippage_bps="0")
        # Open long then close
        p.execute("buy", Decimal("1"), Decimal("100"))
        fill = p.execute("sell", Decimal("1"), Decimal("110"))
        # Balance: 10000 - 100 + 110 = 10010
        assert Decimal(fill["balance"]) == Decimal("10010")
        assert Decimal(fill["position_qty"]) == Decimal("0")

    def test_realized_pnl_on_close(self):
        p = self._make(fee_bps="0", slippage_bps="0")
        p.execute("buy", Decimal("1"), Decimal("100"))
        p.execute("sell", Decimal("1"), Decimal("120"))
        assert p._realized_pnl == Decimal("20")

    def test_position_tracking_add(self):
        p = self._make(fee_bps="0", slippage_bps="0")
        p.execute("buy", Decimal("1"), Decimal("100"))
        p.execute("buy", Decimal("1"), Decimal("200"))
        assert p._position_qty == Decimal("2")
        assert p._avg_price == Decimal("150")

    def test_short_position(self):
        p = self._make(fee_bps="0", slippage_bps="0")
        p.execute("sell", Decimal("1"), Decimal("100"))
        assert p._position_qty == Decimal("-1")
        assert p._avg_price == Decimal("100")

    def test_unrealized_pnl(self):
        p = self._make(fee_bps="0", slippage_bps="0")
        p.execute("buy", Decimal("1"), Decimal("100"))
        assert p.unrealized_pnl(Decimal("110")) == Decimal("10")

    def test_unrealized_pnl_flat(self):
        p = self._make()
        assert p.unrealized_pnl(Decimal("100")) == Decimal("0")

    def test_equity(self):
        p = self._make(fee_bps="0", slippage_bps="0")
        p.execute("buy", Decimal("1"), Decimal("100"))
        # balance = 9900, unrealized at 110 = 10, equity = 9910
        eq = p.equity(Decimal("110"))
        assert eq == Decimal("9910")

    def test_summary_keys(self):
        p = self._make()
        s = p.summary(Decimal("100"))
        expected_keys = {
            "balance", "position_qty", "avg_price",
            "unrealized_pnl", "realized_pnl", "total_fees",
            "equity", "trade_count",
        }
        assert set(s.keys()) == expected_keys

    def test_slippage_applied(self):
        p = self._make(fee_bps="0", slippage_bps="100")  # 1%
        fill = p.execute("buy", Decimal("1"), Decimal("100"))
        # exec_price = 100 * 1.01 = 101
        assert Decimal(fill["exec_price"]) == Decimal("101")

    def test_trade_count_increments(self):
        p = self._make(fee_bps="0", slippage_bps="0")
        p.execute("buy", Decimal("1"), Decimal("100"))
        p.execute("sell", Decimal("1"), Decimal("100"))
        assert p._trade_count == 2


# ── _SimpleMAStrategy ──────────────────────────────────────


class TestSimpleMAStrategy:
    def test_no_signal_during_warmup(self):
        s = _SimpleMAStrategy(symbol="BTCUSDT", window=3)
        assert s.on_price(Decimal("100")) is None
        assert s.on_price(Decimal("101")) is None

    def test_buy_signal_above_ma(self):
        s = _SimpleMAStrategy(symbol="BTCUSDT", window=3)
        s.on_price(Decimal("100"))
        s.on_price(Decimal("100"))
        # Third price completes window; MA = (100+100+110)/3 = 103.33, close=110 > MA
        sig = s.on_price(Decimal("110"))
        assert sig == "buy"

    def test_sell_signal_below_ma(self):
        s = _SimpleMAStrategy(symbol="BTCUSDT", window=3)
        s.on_price(Decimal("100"))
        s.on_price(Decimal("100"))
        sig = s.on_price(Decimal("90"))
        # MA = (100+100+90)/3 = 96.67, close=90 < MA
        assert sig == "sell"

    def test_no_signal_at_ma(self):
        s = _SimpleMAStrategy(symbol="BTCUSDT", window=3)
        s.on_price(Decimal("100"))
        s.on_price(Decimal("100"))
        sig = s.on_price(Decimal("100"))
        # MA = 100, close = 100, no cross
        assert sig is None

    def test_window_slides(self):
        s = _SimpleMAStrategy(symbol="BTCUSDT", window=2)
        s.on_price(Decimal("100"))
        s.on_price(Decimal("100"))
        # Window full: [100, 100], MA=100
        # Add higher price, window slides to [100, 120]
        sig = s.on_price(Decimal("120"))
        assert sig == "buy"
        assert len(s._closes) == 2
