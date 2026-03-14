# tests/unit/portfolio/test_allocator.py
"""Allocator and candidate generator unit tests."""
from __future__ import annotations

from decimal import Decimal

from decision.allocators.base import EqualWeightAllocator
from decision.candidates.base import PassthroughCandidates
from decision.signals.base import NullSignal
from decision.types import Candidate, SignalResult


# ---------------------------------------------------------------------------
# Tests: EqualWeightAllocator
# ---------------------------------------------------------------------------

class TestEqualWeightAllocator:
    def test_single_candidate(self) -> None:
        alloc = EqualWeightAllocator()
        c = Candidate(symbol="BTCUSDT", score=Decimal("0.8"), side="buy")
        weights = alloc.allocate([c])
        assert weights["BTCUSDT"] == Decimal("1")

    def test_two_candidates(self) -> None:
        alloc = EqualWeightAllocator()
        cs = [
            Candidate(symbol="BTCUSDT", score=Decimal("0.8"), side="buy"),
            Candidate(symbol="ETHUSDT", score=Decimal("0.6"), side="buy"),
        ]
        weights = alloc.allocate(cs)
        assert weights["BTCUSDT"] == Decimal("0.5")
        assert weights["ETHUSDT"] == Decimal("0.5")

    def test_three_candidates(self) -> None:
        alloc = EqualWeightAllocator()
        cs = [
            Candidate(symbol="BTCUSDT", score=Decimal("0.8"), side="buy"),
            Candidate(symbol="ETHUSDT", score=Decimal("0.6"), side="buy"),
            Candidate(symbol="SOLUSDT", score=Decimal("0.4"), side="sell"),
        ]
        weights = alloc.allocate(cs)
        expected_w = Decimal("1") / Decimal("3")
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            assert weights[sym] == expected_w

    def test_empty_candidates(self) -> None:
        alloc = EqualWeightAllocator()
        weights = alloc.allocate([])
        assert weights == {}


# ---------------------------------------------------------------------------
# Tests: PassthroughCandidates
# ---------------------------------------------------------------------------

class TestPassthroughCandidates:
    def test_positive_score_is_buy(self) -> None:
        gen = PassthroughCandidates()
        signals = [SignalResult(symbol="BTCUSDT", side="buy", score=Decimal("0.7"))]
        candidates = gen.generate(signals)
        assert len(candidates) == 1
        assert candidates[0].side == "buy"
        assert candidates[0].symbol == "BTCUSDT"

    def test_negative_score_is_sell(self) -> None:
        gen = PassthroughCandidates()
        signals = [SignalResult(symbol="BTCUSDT", side="sell", score=Decimal("-0.5"))]
        candidates = gen.generate(signals)
        assert len(candidates) == 1
        assert candidates[0].side == "sell"

    def test_flat_signal_filtered(self) -> None:
        gen = PassthroughCandidates()
        signals = [SignalResult(symbol="BTCUSDT", side="flat", score=Decimal("0"))]
        candidates = gen.generate(signals)
        assert len(candidates) == 0

    def test_zero_score_filtered(self) -> None:
        gen = PassthroughCandidates()
        signals = [SignalResult(symbol="BTCUSDT", side="buy", score=Decimal("0"))]
        candidates = gen.generate(signals)
        assert len(candidates) == 0

    def test_multiple_signals_mixed(self) -> None:
        gen = PassthroughCandidates()
        signals = [
            SignalResult(symbol="BTCUSDT", side="buy", score=Decimal("0.8")),
            SignalResult(symbol="ETHUSDT", side="flat", score=Decimal("0")),
            SignalResult(symbol="SOLUSDT", side="sell", score=Decimal("-0.3")),
        ]
        candidates = gen.generate(signals)
        assert len(candidates) == 2
        syms = {c.symbol for c in candidates}
        assert syms == {"BTCUSDT", "SOLUSDT"}

    def test_meta_preserved(self) -> None:
        gen = PassthroughCandidates()
        signals = [SignalResult(symbol="BTCUSDT", side="buy", score=Decimal("1"), meta={"model": "test"})]
        candidates = gen.generate(signals)
        assert candidates[0].meta == {"model": "test"}


# ---------------------------------------------------------------------------
# Tests: NullSignal
# ---------------------------------------------------------------------------

class TestNullSignal:
    def test_null_signal_returns_flat(self) -> None:
        ns = NullSignal()
        result = ns.compute(snapshot=None, symbol="BTCUSDT")
        assert result.side == "flat"
        assert result.score == Decimal("0")
        assert result.confidence == Decimal("0")
        assert result.symbol == "BTCUSDT"

    def test_null_signal_name(self) -> None:
        ns = NullSignal()
        assert ns.name == "null"

    def test_null_signal_custom_name(self) -> None:
        ns = NullSignal(name="custom_null")
        assert ns.name == "custom_null"
