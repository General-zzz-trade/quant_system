"""Tests for decision signal strategies: MACrossSignal, BreakoutSignal, ZScoreSignal."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Mapping, Optional

import pytest

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from state.snapshot import StateSnapshot

from decision.signals.technical.ma_cross import MACrossSignal
from decision.signals.technical.breakout import BreakoutSignal
from decision.signals.statistical.zscore import ZScoreSignal

# ─── Snapshot helpers ─────────────────────────────────────────────────────────

_TS = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
_SYMBOL = "BTCUSDT"


@dataclass
class _SnapshotStub:
    """A minimal snapshot stub that supports an injected `features` dict.

    StateSnapshot uses __slots__ and frozen=True so we cannot attach
    arbitrary attributes.  This stub satisfies the duck-typed interface
    expected by signal strategies (snapshot.features, snapshot.market).
    """
    features: Optional[Mapping[str, Any]] = None
    market: Optional[MarketState] = None


def _make_snapshot(
    *,
    last_price: Decimal = Decimal("50000"),
    close: Optional[Decimal] = None,
    high: Optional[Decimal] = None,
    low: Optional[Decimal] = None,
    balance: Decimal = Decimal("10000"),
    positions: Optional[dict] = None,
    ts: Optional[datetime] = None,
) -> StateSnapshot:
    """Create a real StateSnapshot (no features — use _SnapshotStub for features)."""
    market = MarketState(
        symbol=_SYMBOL,
        last_price=last_price,
        close=close if close is not None else last_price,
        high=high,
        low=low,
    )
    account = AccountState.initial(currency="USDT", balance=balance)
    return StateSnapshot.of(
        symbol=_SYMBOL,
        ts=ts or _TS,
        event_id="e1",
        event_type="bar",
        bar_index=0,
        markets={_SYMBOL: market},
        positions=positions or {},
        account=account,
    )


def _stub_with_features(features: Dict[str, Any]) -> _SnapshotStub:
    return _SnapshotStub(features=features)


def _stub_with_market(
    *,
    close: Decimal,
    high: Decimal,
    low: Decimal,
    features: Optional[Dict[str, Any]] = None,
) -> _SnapshotStub:
    market = MarketState(
        symbol=_SYMBOL,
        last_price=close,
        close=close,
        high=high,
        low=low,
    )
    return _SnapshotStub(features=features, market=market)


# ─── MACrossSignal tests ──────────────────────────────────────────────────────

class TestMACrossSignal:
    def setup_method(self) -> None:
        self.signal = MACrossSignal()
        self.symbol = _SYMBOL

    def test_buy_when_fast_above_slow(self) -> None:
        stub = _stub_with_features({"ma_fast": 105, "ma_slow": 100})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "buy"
        assert result.score == Decimal("1")

    def test_sell_when_fast_below_slow(self) -> None:
        stub = _stub_with_features({"ma_fast": 95, "ma_slow": 100})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "sell"
        assert result.score == Decimal("-1")

    def test_flat_when_equal(self) -> None:
        stub = _stub_with_features({"ma_fast": 100, "ma_slow": 100})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"

    def test_missing_features_returns_flat(self) -> None:
        # Snapshot with no features attribute at all
        stub = _SnapshotStub(features=None)
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"
        assert result.confidence == Decimal("0")

    def test_confidence_is_one_on_signal(self) -> None:
        stub = _stub_with_features({"ma_fast": 110, "ma_slow": 100})
        result = self.signal.compute(stub, self.symbol)
        assert result.confidence == Decimal("1")

    def test_custom_keys(self) -> None:
        signal = MACrossSignal(fast_key="ema_fast", slow_key="ema_slow")
        stub = _stub_with_features({"ema_fast": 200, "ema_slow": 190})
        result = signal.compute(stub, self.symbol)
        assert result.side == "buy"

    def test_non_mapping_features_returns_flat(self) -> None:
        # features is a list, not a Mapping
        stub = _SnapshotStub(features=["ma_fast", "ma_slow"])  # type: ignore[arg-type]
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"
        assert result.confidence == Decimal("0")

    def test_missing_key_returns_flat(self) -> None:
        # features dict exists but keys are absent
        stub = _stub_with_features({"some_other_key": 100})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"
        assert result.confidence == Decimal("0")

    def test_result_symbol_matches_input(self) -> None:
        stub = _stub_with_features({"ma_fast": 105, "ma_slow": 100})
        result = self.signal.compute(stub, "ETHUSDT")
        assert result.symbol == "ETHUSDT"

    def test_meta_contains_fast_and_slow(self) -> None:
        stub = _stub_with_features({"ma_fast": 105, "ma_slow": 100})
        result = self.signal.compute(stub, self.symbol)
        assert result.meta is not None
        assert "fast" in result.meta
        assert "slow" in result.meta


# ─── BreakoutSignal tests ─────────────────────────────────────────────────────

class TestBreakoutSignal:
    def setup_method(self) -> None:
        self.signal = BreakoutSignal()
        self.symbol = _SYMBOL

    def test_buy_on_breakout_high(self) -> None:
        # close >= high → buy breakout
        stub = _stub_with_market(close=Decimal("110"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "buy"
        assert result.score == Decimal("1")

    def test_buy_when_close_above_high(self) -> None:
        stub = _stub_with_market(close=Decimal("115"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "buy"

    def test_sell_on_breakout_low(self) -> None:
        # close <= low → sell breakout
        stub = _stub_with_market(close=Decimal("90"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "sell"
        assert result.score == Decimal("-1")

    def test_sell_when_close_below_low(self) -> None:
        stub = _stub_with_market(close=Decimal("85"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "sell"

    def test_flat_when_no_breakout(self) -> None:
        stub = _stub_with_market(close=Decimal("100"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"

    def test_no_market_returns_flat(self) -> None:
        stub = _SnapshotStub(market=None)
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"
        assert result.confidence == Decimal("0")

    def test_buy_confidence_is_one(self) -> None:
        stub = _stub_with_market(close=Decimal("110"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, self.symbol)
        assert result.confidence == Decimal("1")

    def test_sell_confidence_is_one(self) -> None:
        stub = _stub_with_market(close=Decimal("90"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, self.symbol)
        assert result.confidence == Decimal("1")

    def test_flat_confidence_is_low(self) -> None:
        stub = _stub_with_market(close=Decimal("100"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, self.symbol)
        assert result.confidence == Decimal("0.2")

    def test_result_symbol_matches_input(self) -> None:
        stub = _stub_with_market(close=Decimal("110"), high=Decimal("110"), low=Decimal("90"))
        result = self.signal.compute(stub, "ETHUSDT")
        assert result.symbol == "ETHUSDT"


# ─── ZScoreSignal tests ───────────────────────────────────────────────────────

class TestZScoreSignal:
    def setup_method(self) -> None:
        self.signal = ZScoreSignal(threshold=Decimal("1.0"))
        self.symbol = _SYMBOL

    def test_buy_when_zscore_below_negative_threshold(self) -> None:
        # z <= -1.0 → mean reversion buy
        stub = _stub_with_features({"zscore": -2.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "buy"

    def test_sell_when_zscore_above_positive_threshold(self) -> None:
        # z >= 1.0 → mean reversion sell
        stub = _stub_with_features({"zscore": 2.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "sell"

    def test_flat_within_thresholds(self) -> None:
        stub = _stub_with_features({"zscore": 0.5})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"

    def test_flat_at_zero(self) -> None:
        stub = _stub_with_features({"zscore": 0.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"

    def test_sell_at_exact_threshold(self) -> None:
        stub = _stub_with_features({"zscore": 1.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "sell"

    def test_buy_at_exact_negative_threshold(self) -> None:
        stub = _stub_with_features({"zscore": -1.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "buy"

    def test_buy_score_positive(self) -> None:
        stub = _stub_with_features({"zscore": -2.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.score > Decimal("0")

    def test_sell_score_negative(self) -> None:
        stub = _stub_with_features({"zscore": 2.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.score < Decimal("0")

    def test_missing_features_returns_flat(self) -> None:
        stub = _SnapshotStub(features=None)
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"
        assert result.confidence == Decimal("0")

    def test_missing_zscore_key_returns_flat(self) -> None:
        stub = _stub_with_features({"other_key": 5.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.side == "flat"

    def test_custom_threshold(self) -> None:
        signal = ZScoreSignal(threshold=Decimal("2.0"))
        stub_below = _stub_with_features({"zscore": 1.5})
        result_below = signal.compute(stub_below, self.symbol)
        assert result_below.side == "flat"

        stub_above = _stub_with_features({"zscore": 2.5})
        result_above = signal.compute(stub_above, self.symbol)
        assert result_above.side == "sell"

    def test_custom_key(self) -> None:
        signal = ZScoreSignal(key="spread_z", threshold=Decimal("1.0"))
        stub = _stub_with_features({"spread_z": -1.5})
        result = signal.compute(stub, self.symbol)
        assert result.side == "buy"

    def test_confidence_is_nonzero_on_signal(self) -> None:
        stub = _stub_with_features({"zscore": 2.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.confidence > Decimal("0")

    def test_result_symbol_matches_input(self) -> None:
        stub = _stub_with_features({"zscore": 2.0})
        result = self.signal.compute(stub, "ETHUSDT")
        assert result.symbol == "ETHUSDT"

    def test_meta_contains_z_value(self) -> None:
        stub = _stub_with_features({"zscore": 2.0})
        result = self.signal.compute(stub, self.symbol)
        assert result.meta is not None
        assert "z" in result.meta
