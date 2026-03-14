"""Tests for Decision allocators and position sizing modules."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace


from decision.types import Candidate
from decision.allocators.base import EqualWeightAllocator
from decision.allocators.single_asset import SingleAssetAllocator
from decision.allocators.constraints import AllocationConstraints
from decision.sizing.fixed_fraction import FixedFractionSizer
from decision.sizing.base import VolatilityAdjustedSizer


# ── Test helpers ──────────────────────────────────────────────

def _candidate(symbol: str, score: float, side: str = "buy") -> Candidate:
    return Candidate(symbol=symbol, score=Decimal(str(score)), side=side)


def _snapshot(close: float = 40000.0, balance: float = 10000.0,
              features: dict | None = None):
    """Minimal snapshot for sizer tests."""
    return SimpleNamespace(
        account=SimpleNamespace(
            balance=Decimal(str(balance)),
            unrealized_pnl=Decimal("0"),
        ),
        market=SimpleNamespace(close=close),
        features=features or {},
    )


# ── EqualWeightAllocator ─────────────────────────────────────

class TestEqualWeightAllocator:
    def test_single_candidate(self):
        alloc = EqualWeightAllocator()
        result = alloc.allocate([_candidate("BTC", 0.8)])
        assert result == {"BTC": Decimal("1")}

    def test_two_candidates(self):
        alloc = EqualWeightAllocator()
        result = alloc.allocate([_candidate("BTC", 0.8), _candidate("ETH", 0.5)])
        assert result["BTC"] == Decimal("0.5")
        assert result["ETH"] == Decimal("0.5")

    def test_three_candidates(self):
        alloc = EqualWeightAllocator()
        candidates = [_candidate(s, 0.5) for s in ["BTC", "ETH", "SOL"]]
        result = alloc.allocate(candidates)
        for w in result.values():
            assert abs(w - Decimal("1") / Decimal("3")) < Decimal("0.001")

    def test_empty_candidates(self):
        alloc = EqualWeightAllocator()
        assert alloc.allocate([]) == {}


# ── SingleAssetAllocator ─────────────────────────────────────

class TestSingleAssetAllocator:
    def test_picks_highest_score(self):
        alloc = SingleAssetAllocator()
        candidates = [_candidate("BTC", 0.3), _candidate("ETH", 0.9), _candidate("SOL", 0.5)]
        result = alloc.allocate(candidates)
        assert list(result.keys()) == ["ETH"]
        assert result["ETH"] == Decimal("1")

    def test_picks_highest_abs_score(self):
        alloc = SingleAssetAllocator()
        candidates = [_candidate("BTC", 0.3, "buy"), _candidate("ETH", -0.9, "sell")]
        result = alloc.allocate(candidates)
        assert "ETH" in result

    def test_single_candidate(self):
        alloc = SingleAssetAllocator()
        result = alloc.allocate([_candidate("BTC", 0.5)])
        assert result == {"BTC": Decimal("1")}

    def test_empty_candidates(self):
        alloc = SingleAssetAllocator()
        assert alloc.allocate([]) == {}


# ── AllocationConstraints ─────────────────────────────────────

class TestAllocationConstraints:
    def test_max_positions_clips(self):
        constraints = AllocationConstraints(max_positions=2)
        weights = {"BTC": Decimal("0.5"), "ETH": Decimal("0.3"), "SOL": Decimal("0.2")}
        result = constraints.apply(weights)
        assert len(result) == 2
        assert "BTC" in result
        assert "ETH" in result

    def test_max_positions_one(self):
        constraints = AllocationConstraints(max_positions=1)
        weights = {"BTC": Decimal("0.6"), "ETH": Decimal("0.4")}
        result = constraints.apply(weights)
        assert len(result) == 1
        assert "BTC" in result

    def test_renormalizes_weights(self):
        constraints = AllocationConstraints(max_positions=2)
        weights = {"BTC": Decimal("0.5"), "ETH": Decimal("0.3"), "SOL": Decimal("0.2")}
        result = constraints.apply(weights)
        total = sum(abs(w) for w in result.values())
        assert abs(total - Decimal("1")) < Decimal("0.01")

    def test_empty_weights(self):
        constraints = AllocationConstraints(max_positions=3)
        assert constraints.apply({}) == {}


# ── FixedFractionSizer ────────────────────────────────────────

class TestFixedFractionSizer:
    def test_basic_sizing(self):
        sizer = FixedFractionSizer(fraction=Decimal("0.02"))
        snap = _snapshot(close=40000.0, balance=100000.0)
        qty = sizer.target_qty(snap, "BTCUSDT", Decimal("1"))
        # expected: 100000 * 0.02 * 1 / 40000 = 0.05
        assert abs(qty - Decimal("0.05")) < Decimal("0.001")

    def test_half_weight(self):
        sizer = FixedFractionSizer(fraction=Decimal("0.02"))
        snap = _snapshot(close=40000.0, balance=100000.0)
        qty = sizer.target_qty(snap, "BTCUSDT", Decimal("0.5"))
        # expected: 100000 * 0.02 * 0.5 / 40000 = 0.025
        assert abs(qty - Decimal("0.025")) < Decimal("0.001")

    def test_lot_size_rounding(self):
        sizer = FixedFractionSizer(fraction=Decimal("0.02"), lot_size=Decimal("0.01"))
        snap = _snapshot(close=40000.0, balance=100000.0)
        qty = sizer.target_qty(snap, "BTCUSDT", Decimal("1"))
        # Should be rounded to lot_size
        assert qty % Decimal("0.01") == 0

    def test_zero_price(self):
        sizer = FixedFractionSizer()
        snap = _snapshot(close=0.0)
        qty = sizer.target_qty(snap, "BTCUSDT", Decimal("1"))
        assert qty == Decimal("0")


# ── VolatilityAdjustedSizer ───────────────────────────────────

class TestVolatilityAdjustedSizer:
    def test_basic_sizing(self):
        sizer = VolatilityAdjustedSizer(risk_fraction=Decimal("0.10"))
        snap = _snapshot(close=100.0, balance=100000.0, features={"atr": 5.0})
        qty = sizer.target_qty(snap, "ETHUSDT", Decimal("1"))
        # expected: (100000 * 0.10 * 1) / (5 * 100) = 10000 / 500 = 20.0
        assert qty > 0

    def test_uses_default_volatility(self):
        sizer = VolatilityAdjustedSizer(
            risk_fraction=Decimal("0.02"),
            default_volatility=Decimal("0.02"),
        )
        snap = _snapshot(close=40000.0, balance=100000.0)  # no atr feature
        qty = sizer.target_qty(snap, "BTCUSDT", Decimal("1"))
        assert qty > 0

    def test_zero_price_returns_zero(self):
        sizer = VolatilityAdjustedSizer()
        snap = _snapshot(close=0.0)
        qty = sizer.target_qty(snap, "BTCUSDT", Decimal("1"))
        assert qty == Decimal("0")
