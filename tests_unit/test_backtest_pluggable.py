from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from runner.backtest_runner import run_backtest, MovingAverageCrossModule


def _write_csv(path: Path, rows: int = 30) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        base = 100
        for i in range(rows):
            day = 1 + i // 24
            hour = i % 24
            ts = f"2024-01-{day:02d}T{hour:02d}:00:00Z"
            p = base + (i % 5) - 2
            w.writerow([ts, p, p + 1, p - 1, p, 100])


class AlwaysBuyModule:
    """Trivial module that buys once then holds."""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self._bought = False

    def decide(self, snapshot: Any) -> Iterable[Any]:
        if self._bought:
            return ()
        self._bought = True

        from event.header import EventHeader
        from event.types import EventType, IntentEvent, OrderEvent
        from runner.backtest.adapter import _make_id

        intent_id = _make_id("intent")
        order_id = _make_id("order")
        ih = EventHeader.new_root(event_type=EventType.INTENT, version=1, source="test")
        oh = EventHeader.from_parent(parent=ih, event_type=EventType.ORDER, version=1, source="test")

        return [
            IntentEvent(header=ih, intent_id=intent_id, symbol=self.symbol, side="buy",
                        target_qty=Decimal("0.01"), reason_code="test_buy", origin="test"),
            OrderEvent(header=oh, order_id=order_id, intent_id=intent_id, symbol=self.symbol,
                       side="buy", qty=Decimal("0.01"), price=None),
        ]


def test_pluggable_custom_module(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    module = AlwaysBuyModule("BTCUSDT")
    eq, fills = run_backtest(
        csv_path=csv_path,
        symbol="BTCUSDT",
        starting_balance=Decimal("10000"),
        fee_bps=Decimal("0"),
        decision_modules=[module],
    )
    assert len(eq) > 0
    assert module._bought
    assert len(fills) >= 1


def test_backward_compat_default_module(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    eq, fills = run_backtest(
        csv_path=csv_path,
        symbol="BTCUSDT",
        starting_balance=Decimal("10000"),
        ma_window=5,
        order_qty=Decimal("0.01"),
        fee_bps=Decimal("0"),
    )
    assert len(eq) > 0


def test_pluggable_empty_modules(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path)

    eq, fills = run_backtest(
        csv_path=csv_path,
        symbol="BTCUSDT",
        starting_balance=Decimal("10000"),
        fee_bps=Decimal("0"),
        decision_modules=[],
    )
    assert len(eq) > 0
    assert len(fills) == 0
