"""Tests for portfolio allocator: TargetWeightAllocator, EqualWeightAllocator,
and constraint application (gross leverage, turnover, max notional, Rust parity).
"""
from __future__ import annotations

from decimal import Decimal
from typing import Mapping

import pytest

from portfolio.allocator import (
    TargetWeightAllocator,
    EqualWeightAllocator,
    PortfolioConstraints,
    AllocatorError,
)

# Force Python path (no Rust delegate) for deterministic tests
import portfolio.allocator as _alloc_mod
_alloc_mod._RUST_ALLOCATOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers: mock account & price provider
# ---------------------------------------------------------------------------

class _Prices:
    def __init__(self, prices: dict[str, float]):
        self._prices = prices

    def price(self, symbol: str) -> Decimal:
        return Decimal(str(self._prices[symbol]))


class _Account:
    def __init__(self, equity: float, positions: dict[str, float] | None = None):
        self._equity = Decimal(str(equity))
        self._positions = {k: Decimal(str(v)) for k, v in (positions or {}).items()}

    @property
    def equity(self) -> Decimal:
        return self._equity

    @property
    def positions_qty(self) -> Mapping[str, Decimal]:
        return self._positions


# ---------------------------------------------------------------------------
# TargetWeightAllocator: basic weight -> qty
# ---------------------------------------------------------------------------

class TestTargetWeightAllocator:
    def test_basic_weight_to_qty(self):
        """50% weight on BTC at $50k, equity $10k => qty = 0.1 BTC."""
        alloc = TargetWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["BTCUSDT"],
            account=_Account(10_000.0),
            prices=_Prices({"BTCUSDT": 50_000.0}),
            constraints=PortfolioConstraints(),
            inputs={"target_weights": {"BTCUSDT": 0.5}},
        )
        assert len(plan.targets) == 1
        t = plan.targets[0]
        assert t.symbol == "BTCUSDT"
        assert t.target_qty == pytest.approx(Decimal("0.1"), abs=Decimal("0.0001"))
        assert t.target_notional == pytest.approx(Decimal("5000"), abs=Decimal("1"))

    def test_missing_weights_raises(self):
        alloc = TargetWeightAllocator()
        with pytest.raises(AllocatorError):
            alloc.allocate(
                ts=0, symbols=["BTCUSDT"],
                account=_Account(10_000),
                prices=_Prices({"BTCUSDT": 50_000}),
                constraints=PortfolioConstraints(),
                inputs=None,
            )

    def test_zero_equity_raises(self):
        alloc = TargetWeightAllocator()
        with pytest.raises(AllocatorError):
            alloc.allocate(
                ts=0, symbols=["BTCUSDT"],
                account=_Account(0),
                prices=_Prices({"BTCUSDT": 50_000}),
                constraints=PortfolioConstraints(),
                inputs={"target_weights": {"BTCUSDT": 0.5}},
            )

    def test_short_weight_disallowed(self):
        """Short weight clipped to 0 when allow_short=False."""
        alloc = TargetWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["BTCUSDT"],
            account=_Account(10_000),
            prices=_Prices({"BTCUSDT": 50_000}),
            constraints=PortfolioConstraints(allow_short=False),
            inputs={"target_weights": {"BTCUSDT": -0.5}},
        )
        assert plan.targets[0].target_qty == Decimal("0")

    def test_multi_symbol_weights(self):
        """Two symbols with different weights produce correct qty split."""
        alloc = TargetWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["BTCUSDT", "ETHUSDT"],
            account=_Account(10_000),
            prices=_Prices({"BTCUSDT": 50_000, "ETHUSDT": 2_000}),
            constraints=PortfolioConstraints(),
            inputs={"target_weights": {"BTCUSDT": 0.3, "ETHUSDT": 0.7}},
        )
        by_sym = {t.symbol: t for t in plan.targets}
        # BTC: 0.3 * 10000 / 50000 = 0.06
        assert by_sym["BTCUSDT"].target_qty == pytest.approx(Decimal("0.06"), abs=Decimal("0.001"))
        # ETH: 0.7 * 10000 / 2000 = 3.5
        assert by_sym["ETHUSDT"].target_qty == pytest.approx(Decimal("3.5"), abs=Decimal("0.01"))


# ---------------------------------------------------------------------------
# EqualWeightAllocator
# ---------------------------------------------------------------------------

class TestEqualWeightAllocator:
    def test_two_symbols_equal(self):
        alloc = EqualWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["BTCUSDT", "ETHUSDT"],
            account=_Account(10_000),
            prices=_Prices({"BTCUSDT": 50_000, "ETHUSDT": 2_000}),
            constraints=PortfolioConstraints(),
        )
        w0 = plan.targets[0].target_weight
        w1 = plan.targets[1].target_weight
        assert w0 == w1
        assert w0 == Decimal("0.5")

    def test_three_symbols_equal(self):
        alloc = EqualWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["A", "B", "C"],
            account=_Account(9_000),
            prices=_Prices({"A": 100, "B": 200, "C": 300}),
            constraints=PortfolioConstraints(),
        )
        weights = [t.target_weight for t in plan.targets]
        expected = Decimal("1") / Decimal("3")
        for w in weights:
            assert w == expected

    def test_empty_symbols_raises(self):
        alloc = EqualWeightAllocator()
        with pytest.raises(AllocatorError):
            alloc.allocate(
                ts=0, symbols=[],
                account=_Account(10_000),
                prices=_Prices({}),
                constraints=PortfolioConstraints(),
            )


# ---------------------------------------------------------------------------
# Gross leverage cap
# ---------------------------------------------------------------------------

class TestGrossLeverageCap:
    def test_leverage_scaled_down(self):
        """2x weight on single symbol, cap at 1.0 => scaled to ~1.0x."""
        alloc = TargetWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["BTCUSDT"],
            account=_Account(10_000),
            prices=_Prices({"BTCUSDT": 50_000}),
            constraints=PortfolioConstraints(max_gross_leverage=Decimal("1.0")),
            inputs={"target_weights": {"BTCUSDT": 2.0}},
        )
        t = plan.targets[0]
        # notional should be capped to ~10000 (equity * 1.0)
        assert float(t.target_notional) <= 10_001

    def test_no_cap_means_no_scaling(self):
        alloc = TargetWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["BTCUSDT"],
            account=_Account(10_000),
            prices=_Prices({"BTCUSDT": 50_000}),
            constraints=PortfolioConstraints(),  # no cap
            inputs={"target_weights": {"BTCUSDT": 3.0}},
        )
        # Should not be scaled: notional = 3 * 10000 = 30000
        assert float(plan.targets[0].target_notional) == pytest.approx(30_000, abs=1)


# ---------------------------------------------------------------------------
# Turnover cap
# ---------------------------------------------------------------------------

class TestTurnoverCap:
    def test_turnover_limits_change(self):
        """Large position change is throttled by turnover cap."""
        alloc = TargetWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["BTCUSDT"],
            account=_Account(10_000, {"BTCUSDT": 0.0}),
            prices=_Prices({"BTCUSDT": 50_000}),
            constraints=PortfolioConstraints(turnover_cap=Decimal("0.1")),
            inputs={"target_weights": {"BTCUSDT": 1.0}},
        )
        # Target notional would be 10000, but turnover cap 10% of equity = 1000
        assert float(plan.targets[0].target_notional) <= 1_100


# ---------------------------------------------------------------------------
# Max notional per symbol
# ---------------------------------------------------------------------------

class TestMaxNotionalPerSymbol:
    def test_notional_clamped(self):
        alloc = TargetWeightAllocator()
        plan = alloc.allocate(
            ts=0,
            symbols=["BTCUSDT"],
            account=_Account(100_000),
            prices=_Prices({"BTCUSDT": 50_000}),
            constraints=PortfolioConstraints(max_notional_per_symbol=Decimal("5000")),
            inputs={"target_weights": {"BTCUSDT": 0.5}},
        )
        # 50% of 100k = 50k notional, capped to 5000
        assert float(plan.targets[0].target_notional) <= 5_001


# ---------------------------------------------------------------------------
# Rust parity (run only if Rust available)
# ---------------------------------------------------------------------------

class TestRustPythonParity:
    @pytest.fixture(autouse=True)
    def _check_rust(self):
        try:
            from _quant_hotpath import rust_allocate_portfolio  # noqa: F401
            self._has_rust = True
        except ImportError:
            self._has_rust = False

    def test_rust_python_same_result(self):
        if not self._has_rust:
            pytest.skip("Rust not available")

        from portfolio.allocator_constraints import _rust_allocate_targets

        symbols = ["BTCUSDT", "ETHUSDT"]
        weights = {s: Decimal("0.5") for s in symbols}
        equity = Decimal("10000")
        prices = _Prices({"BTCUSDT": 50_000, "ETHUSDT": 2_000})
        current_qty: dict[str, Decimal] = {s: Decimal("0") for s in symbols}
        constraints = PortfolioConstraints()

        rust_targets = _rust_allocate_targets(
            symbols=symbols,
            target_weights=weights,
            equity=equity,
            prices=prices,
            current_qty=current_qty,
            constraints=constraints,
        )

        # Python path
        alloc = TargetWeightAllocator()
        py_plan = alloc.allocate(
            ts=0, symbols=symbols,
            account=_Account(10_000),
            prices=prices,
            constraints=constraints,
            inputs={"target_weights": {s: 0.5 for s in symbols}},
        )
        py_targets = {t.symbol: t for t in py_plan.targets}

        for s in symbols:
            assert float(rust_targets[s].target_qty) == pytest.approx(
                float(py_targets[s].target_qty), rel=0.01,
            ), f"Qty mismatch for {s}"
