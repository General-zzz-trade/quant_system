"""Parity tests: RustPortfolioAllocator."""
import pytest

try:
    from _quant_hotpath import RustPortfolioAllocator
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust not available")


class TestAllocatorParity:
    def test_basic_allocation(self):
        alloc = RustPortfolioAllocator()
        result = alloc.allocate(
            {"ETHUSDT": 0.5, "BTCUSDT": 0.3},
            {"ETHUSDT": 0.0, "BTCUSDT": 0.0},
            {"ETHUSDT": 3000.0, "BTCUSDT": 60000.0},
            10000.0,
        )
        assert "trades" in result
        assert "diagnostics" in result
        trades = result["trades"]
        assert len(trades) >= 1

    def test_zero_equity_raises(self):
        alloc = RustPortfolioAllocator()
        with pytest.raises(ValueError):
            alloc.allocate(
                {"ETHUSDT": 0.5}, {"ETHUSDT": 0.0},
                {"ETHUSDT": 3000.0}, 0.0,
            )

    def test_nan_equity_raises(self):
        alloc = RustPortfolioAllocator()
        with pytest.raises(ValueError):
            alloc.allocate(
                {"ETHUSDT": 0.5}, {"ETHUSDT": 0.0},
                {"ETHUSDT": 3000.0}, float('nan'),
            )

    def test_leverage_cap(self):
        alloc = RustPortfolioAllocator(max_gross_leverage=1.0)
        result = alloc.allocate(
            {"ETHUSDT": 2.0},  # 2x weight -> 2x leverage requested
            {"ETHUSDT": 0.0},
            {"ETHUSDT": 3000.0},
            10000.0,
        )
        trades = result["trades"]
        if trades:
            # Total notional should not exceed equity * 1.0
            total_notional = sum(abs(t.get("notional_delta", t.get("qty_delta", 0) * 3000.0)) for t in trades)
            assert total_notional <= 10000.0 * 1.1  # allow small rounding

    def test_notional_per_symbol_cap(self):
        alloc = RustPortfolioAllocator(max_notional_per_symbol=2000.0)
        result = alloc.allocate(
            {"ETHUSDT": 1.0},
            {"ETHUSDT": 0.0},
            {"ETHUSDT": 3000.0},
            10000.0,
        )
        trades = result["trades"]
        for t in trades:
            notional = abs(t.get("qty_delta", 0)) * 3000.0
            assert notional <= 2000.0 * 1.01  # allow tiny rounding

    def test_lot_size_rounding(self):
        alloc = RustPortfolioAllocator()
        alloc.set_lot_size("ETHUSDT", 0.01)
        result = alloc.allocate(
            {"ETHUSDT": 0.5},
            {"ETHUSDT": 0.0},
            {"ETHUSDT": 3000.0},
            10000.0,
        )
        trades = result["trades"]
        for t in trades:
            qty = abs(t["qty_delta"])
            # qty should be multiple of 0.01
            assert abs(qty * 100 - round(qty * 100)) < 1e-9

    def test_scale_order(self):
        alloc = RustPortfolioAllocator(max_notional_per_symbol=5000.0, max_gross_leverage=3.0)
        scale = alloc.scale_order("ETHUSDT", 5.0, 10000.0, 3000.0)
        # 5 * 3000 = 15000 > 5000 cap -> scale = 5000/15000 = 0.333
        assert abs(scale - 1.0/3) < 0.01

    def test_reduce_only_detection(self):
        alloc = RustPortfolioAllocator()
        result = alloc.allocate(
            {"ETHUSDT": 0.0},   # target: flat
            {"ETHUSDT": 1.0},   # current: long 1 ETH
            {"ETHUSDT": 3000.0},
            10000.0,
        )
        trades = result["trades"]
        for t in trades:
            if t["symbol"] == "ETHUSDT":
                assert t["reduce_only"] is True
