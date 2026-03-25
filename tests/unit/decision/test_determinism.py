"""Determinism tests for DecisionEngine.

Verifies that DecisionEngine.run() produces bit-identical output when given
identical input, regardless of call count. Any non-determinism here would make
live vs backtest debugging impossible.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any


from decision.config import DecisionConfig
from decision.engine import DecisionEngine
from decision.types import SignalResult
from strategy.signals.base import NullSignal
from state.snapshot import StateSnapshot


# ── Stub types ───────────────────────────────────────────────

@dataclass(frozen=True)
class StubMarket:
    close: Decimal = Decimal("50000")
    last_price: Decimal = Decimal("50000")
    volume_24h: float = 1e9
    bid: Decimal = Decimal("49999")
    ask: Decimal = Decimal("50001")


@dataclass(frozen=True)
class StubPosition:
    qty: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    side: str = "flat"


@dataclass(frozen=True)
class StubAccount:
    equity: Decimal = Decimal("10000")
    balance: Decimal = Decimal("10000")
    balance_f: float = 10000.0
    unrealized_pnl: Decimal = Decimal("0")
    unrealized_pnl_f: float = 0.0
    available_balance: Decimal = Decimal("10000")
    total_margin: Decimal = Decimal("0")
    leverage: int = 1


@dataclass(frozen=True)
class StubRisk:
    halted: bool = False
    blocked: bool = False


def _make_snapshot(
    symbol: str = "BTCUSDT",
    close: Decimal = Decimal("50000"),
    equity: Decimal = Decimal("10000"),
    position_qty: Decimal = Decimal("0"),
) -> StateSnapshot:
    market = StubMarket(close=close, last_price=close)
    position = StubPosition(qty=position_qty)
    account = StubAccount(equity=equity, available_balance=equity)
    return StateSnapshot(
        symbol=symbol,
        ts=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
        event_id="test-event-001",
        event_type="kline",
        bar_index=100,
        markets={symbol: market},
        positions={symbol: position},
        account=account,
        risk=StubRisk(),
        features={"ml_score": 0.8, "close": 50000.0},
    )


# ── Signal models for testing ───────────────────────────────

@dataclass(frozen=True, slots=True)
class FixedSignal:
    """Deterministic signal model that returns a fixed score."""
    name: str = "fixed"
    _score: Decimal = Decimal("0.8")
    _side: str = "buy"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        return SignalResult(
            symbol=symbol,
            side=self._side,
            score=self._score,
            confidence=Decimal("0.9"),
        )


@dataclass(frozen=True, slots=True)
class FeatureReadingSignal:
    """Signal model that reads features from snapshot (tests dict ordering)."""
    name: str = "feature_reader"

    def compute(self, snapshot: Any, symbol: str) -> SignalResult:
        feats = snapshot.features or {}
        ml_score = feats.get("ml_score", 0.0)
        score = Decimal(str(round(ml_score, 4)))
        side = "buy" if score > 0 else ("sell" if score < 0 else "flat")
        return SignalResult(
            symbol=symbol,
            side=side,
            score=score,
            confidence=Decimal("1"),
        )


# ── Core determinism tests ───────────────────────────────────

class TestDecisionEngineDeterminism:
    """Verify DecisionEngine.run() is deterministic."""

    def test_null_signal_deterministic(self):
        """NullSignal → 100 identical runs → identical output."""
        cfg = DecisionConfig(symbols=("BTCUSDT",), strategy_id="test")
        engine = DecisionEngine(cfg=cfg, signal_model=NullSignal())
        snapshot = _make_snapshot()

        results = [engine.run(snapshot).to_dict() for _ in range(100)]

        for i, r in enumerate(results[1:], 1):
            assert r == results[0], f"Run {i} differs from run 0"

    def test_fixed_signal_deterministic(self):
        """FixedSignal → 100 identical runs → identical output."""
        cfg = DecisionConfig(
            symbols=("BTCUSDT",),
            strategy_id="test",
            max_positions=1,
            risk_fraction=Decimal("0.02"),
        )
        engine = DecisionEngine(cfg=cfg, signal_model=FixedSignal())
        snapshot = _make_snapshot()

        results = [engine.run(snapshot).to_dict() for _ in range(100)]

        for i, r in enumerate(results[1:], 1):
            assert r == results[0], f"Run {i} differs from run 0"

    def test_feature_reading_signal_deterministic(self):
        """Signal that reads features → 100 runs → identical output."""
        cfg = DecisionConfig(
            symbols=("BTCUSDT",),
            strategy_id="test",
            max_positions=1,
        )
        engine = DecisionEngine(cfg=cfg, signal_model=FeatureReadingSignal())
        snapshot = _make_snapshot()

        results = [engine.run(snapshot).to_dict() for _ in range(100)]

        for i, r in enumerate(results[1:], 1):
            assert r == results[0], f"Run {i} differs from run 0"

    def test_multi_symbol_deterministic(self):
        """Multiple symbols → deterministic ordering."""
        symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
        cfg = DecisionConfig(
            symbols=symbols,
            strategy_id="multi",
            max_positions=3,
        )
        engine = DecisionEngine(cfg=cfg, signal_model=FixedSignal())

        markets = {s: StubMarket() for s in symbols}
        positions = {s: StubPosition() for s in symbols}
        snapshot = StateSnapshot(
            symbol="BTCUSDT",
            ts=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            event_id="test-multi",
            event_type="kline",
            bar_index=100,
            markets=markets,
            positions=positions,
            account=StubAccount(),
            risk=StubRisk(),
            features={"ml_score": 0.5},
        )

        results = [engine.run(snapshot).to_dict() for _ in range(100)]

        for i, r in enumerate(results[1:], 1):
            assert r == results[0], f"Run {i} differs from run 0"

    def test_risk_halted_deterministic(self):
        """Risk halt → deterministic empty output."""
        cfg = DecisionConfig(symbols=("BTCUSDT",), strategy_id="test")
        engine = DecisionEngine(cfg=cfg, signal_model=FixedSignal())

        snapshot = StateSnapshot(
            symbol="BTCUSDT",
            ts=datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc),
            event_id="test-halt",
            event_type="kline",
            bar_index=100,
            markets={"BTCUSDT": StubMarket()},
            positions={"BTCUSDT": StubPosition()},
            account=StubAccount(),
            risk=StubRisk(halted=True),
        )

        results = [engine.run(snapshot).to_dict() for _ in range(100)]

        for i, r in enumerate(results[1:], 1):
            assert r == results[0], f"Run {i} differs from run 0"
        # Verify it's actually halted
        assert results[0]["targets"] == []
        assert results[0]["orders"] == []

    def test_output_structure_stable(self):
        """Verify output dict keys are stable across runs."""
        cfg = DecisionConfig(symbols=("BTCUSDT",), strategy_id="test")
        engine = DecisionEngine(cfg=cfg, signal_model=FixedSignal())
        snapshot = _make_snapshot()

        out1 = engine.run(snapshot)
        out2 = engine.run(snapshot)

        d1, d2 = out1.to_dict(), out2.to_dict()
        assert set(d1.keys()) == set(d2.keys())
        assert d1["strategy_id"] == d2["strategy_id"] == "test"
        assert d1["ts"] == d2["ts"]

    def test_order_id_deterministic(self):
        """Order IDs (based on stable_hash) should be identical across runs."""
        cfg = DecisionConfig(
            symbols=("BTCUSDT",),
            strategy_id="test",
            max_positions=1,
            risk_fraction=Decimal("0.02"),
        )
        engine = DecisionEngine(cfg=cfg, signal_model=FixedSignal())
        snapshot = _make_snapshot()

        out1 = engine.run(snapshot)
        out2 = engine.run(snapshot)

        if out1.orders and out2.orders:
            assert out1.orders[0].order_id == out2.orders[0].order_id
            assert out1.orders[0].intent_id == out2.orders[0].intent_id
