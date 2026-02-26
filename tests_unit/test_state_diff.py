"""Tests for state.diff — structured snapshot diffing."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from state.account import AccountState
from state.diff import FieldDelta, SnapshotDiff, compute_diff
from state.market import MarketState
from state.position import PositionState
from state.snapshot import StateSnapshot


def _make_snapshot(
    *,
    symbol: str = "BTCUSDT",
    last_price: Decimal = Decimal("50000"),
    balance: Decimal = Decimal("10000"),
    positions: dict | None = None,
    bar_index: int = 0,
) -> StateSnapshot:
    market = MarketState(symbol=symbol, last_price=last_price, close=last_price)
    account = AccountState.initial(currency="USDT", balance=balance)
    return StateSnapshot.of(
        symbol=symbol,
        ts=datetime(2024, 1, 1, tzinfo=timezone.utc),
        event_id="e1",
        event_type="bar",
        bar_index=bar_index,
        markets={symbol: market},
        positions=positions or {},
        account=account,
    )


class TestComputeDiff:
    def test_identical_snapshots_no_changes(self) -> None:
        s = _make_snapshot()
        diff = compute_diff(s, s)
        assert diff.changed is False
        assert diff.market.changed is False
        assert diff.account.changed is False

    def test_market_price_change_detected(self) -> None:
        old = _make_snapshot(last_price=Decimal("50000"))
        new = _make_snapshot(last_price=Decimal("51000"))
        diff = compute_diff(old, new)
        assert diff.changed is True
        assert diff.market.changed is True
        assert any(d.field == "last_price" for d in diff.market.deltas)

    def test_account_balance_change(self) -> None:
        old = _make_snapshot(balance=Decimal("10000"))
        new = _make_snapshot(balance=Decimal("10500"))
        diff = compute_diff(old, new)
        assert diff.changed is True
        assert diff.account.changed is True
        assert any(d.field == "balance" for d in diff.account.deltas)

    def test_position_added(self) -> None:
        old = _make_snapshot(positions={})
        new = _make_snapshot(positions={
            "BTCUSDT": PositionState(symbol="BTCUSDT", qty=Decimal("1")),
        })
        diff = compute_diff(old, new)
        assert diff.changed is True
        # positions is a tuple of PositionDiff
        added = [p for p in diff.positions if p.action == "added"]
        assert any(p.symbol == "BTCUSDT" for p in added)

    def test_position_removed(self) -> None:
        old = _make_snapshot(positions={
            "BTCUSDT": PositionState(symbol="BTCUSDT", qty=Decimal("1")),
        })
        new = _make_snapshot(positions={})
        diff = compute_diff(old, new)
        assert diff.changed is True
        removed = [p for p in diff.positions if p.action == "removed"]
        assert any(p.symbol == "BTCUSDT" for p in removed)

    def test_position_changed(self) -> None:
        old = _make_snapshot(positions={
            "BTCUSDT": PositionState(symbol="BTCUSDT", qty=Decimal("1")),
        })
        new = _make_snapshot(positions={
            "BTCUSDT": PositionState(symbol="BTCUSDT", qty=Decimal("2")),
        })
        diff = compute_diff(old, new)
        assert diff.changed is True
        changed = [p for p in diff.positions if p.action == "changed"]
        assert any(p.symbol == "BTCUSDT" for p in changed)

    def test_position_unchanged(self) -> None:
        pos = {"BTCUSDT": PositionState(symbol="BTCUSDT", qty=Decimal("1"))}
        old = _make_snapshot(positions=pos)
        new = _make_snapshot(positions=pos)
        diff = compute_diff(old, new)
        unchanged = [p for p in diff.positions if p.action == "unchanged"]
        assert any(p.symbol == "BTCUSDT" for p in unchanged)

    def test_to_dict_serializable(self) -> None:
        old = _make_snapshot(last_price=Decimal("50000"))
        new = _make_snapshot(last_price=Decimal("51000"))
        diff = compute_diff(old, new)
        d = diff.to_dict()
        assert isinstance(d, dict)
        assert "market" in d
        assert "changed" in d
        assert "summary" in d

    def test_summary_text(self) -> None:
        old = _make_snapshot(last_price=Decimal("50000"), balance=Decimal("10000"))
        new = _make_snapshot(last_price=Decimal("51000"), balance=Decimal("10500"))
        diff = compute_diff(old, new)
        assert "market" in diff.summary
        assert "account" in diff.summary
