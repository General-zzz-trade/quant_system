"""Tests for funding rate integration in backtest."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.loaders.funding_rate import (
    FundingRateRecord,
    funding_schedule_for_bars,
    generate_constant_funding,
    iter_funding_events,
)


class TestFundingScheduleForBars:
    def test_matches_settlement_to_nearest_bar(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        bar_ts = [start + timedelta(hours=i) for i in range(25)]
        records = [
            FundingRateRecord(ts=start + timedelta(hours=8), symbol="BTCUSDT",
                            funding_rate=Decimal("0.0001"), mark_price=Decimal("40000")),
        ]
        schedule = funding_schedule_for_bars(bar_ts, records)
        assert start + timedelta(hours=8) in schedule
        assert schedule[start + timedelta(hours=8)].funding_rate == Decimal("0.0001")

    def test_empty_records_returns_empty(self) -> None:
        bar_ts = [datetime(2024, 1, 1, tzinfo=timezone.utc)]
        assert funding_schedule_for_bars(bar_ts, []) == {}

    def test_multiple_settlements(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        bar_ts = [start + timedelta(hours=i) for i in range(25)]
        records = generate_constant_funding("BTCUSDT", start, start + timedelta(hours=24))
        schedule = funding_schedule_for_bars(bar_ts, records)
        # Should have settlements at 0h, 8h, 16h, 24h
        assert len(schedule) >= 3


class TestIterFundingEvents:
    def test_yields_events_for_open_positions(self) -> None:
        records = [
            FundingRateRecord(
                ts=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
                symbol="BTCUSDT",
                funding_rate=Decimal("0.0001"),
                mark_price=Decimal("40000"),
            ),
        ]
        positions = {"BTCUSDT": Decimal("1.0")}
        events = list(iter_funding_events(records, positions))
        assert len(events) == 1
        assert events[0]["position_qty"] == Decimal("1.0")
        assert events[0]["funding_rate"] == Decimal("0.0001")

    def test_skips_zero_position(self) -> None:
        records = [
            FundingRateRecord(
                ts=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
                symbol="BTCUSDT",
                funding_rate=Decimal("0.0001"),
                mark_price=Decimal("40000"),
            ),
        ]
        positions = {"BTCUSDT": Decimal("0")}
        events = list(iter_funding_events(records, positions))
        assert len(events) == 0

    def test_funding_payment_direction(self) -> None:
        """Positive rate + long position → pay (negative impact on balance)."""
        records = [
            FundingRateRecord(
                ts=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
                symbol="BTCUSDT",
                funding_rate=Decimal("0.0001"),
                mark_price=Decimal("40000"),
            ),
        ]
        # Long position
        positions = {"BTCUSDT": Decimal("1.0")}
        events = list(iter_funding_events(records, positions))
        assert len(events) == 1
        # payment = qty * mark * rate = 1 * 40000 * 0.0001 = 4 USDT
        # Positive payment = account pays

    def test_short_position_receives(self) -> None:
        """Positive rate + short position → receive."""
        records = [
            FundingRateRecord(
                ts=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
                symbol="BTCUSDT",
                funding_rate=Decimal("0.0001"),
                mark_price=Decimal("40000"),
            ),
        ]
        positions = {"BTCUSDT": Decimal("-1.0")}
        events = list(iter_funding_events(records, positions))
        assert len(events) == 1
        ev = events[0]
        # payment = -1 * 40000 * 0.0001 = -4 → account receives


class TestGenerateConstantFunding:
    def test_generates_every_8h(self) -> None:
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
        records = generate_constant_funding("BTCUSDT", start, end)
        assert len(records) >= 3
        for r in records:
            assert r.symbol == "BTCUSDT"
            assert r.funding_rate == Decimal("0.0001")
