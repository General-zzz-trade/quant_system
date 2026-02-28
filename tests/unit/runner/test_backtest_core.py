"""Tests for runner/backtest_runner.py — MovingAverageCrossModule, walk-forward, metrics."""
from __future__ import annotations

import csv
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest

from runner.backtest_runner import (
    MovingAverageCrossModule,
    WalkForwardWindow,
    _snapshot_views,
    run_backtest,
)
from runner.backtest.metrics import (
    EquityPoint,
    _max_drawdown,
    _build_summary,
    _build_trades_from_fills,
)
from state.position import PositionState


# ── Helpers ─────────────────────────────────────────────────


def _make_snapshot(
    close: float,
    position_qty: float = 0.0,
    avg_price: float | None = None,
    event_id: str | None = None,
) -> SimpleNamespace:
    market = SimpleNamespace(close=Decimal(str(close)), last_price=Decimal(str(close)))
    pos = PositionState(
        symbol="BTCUSDT",
        qty=Decimal(str(position_qty)),
        avg_price=Decimal(str(avg_price)) if avg_price is not None else None,
    )
    return SimpleNamespace(
        market=market,
        positions={"BTCUSDT": pos},
        event_id=event_id,
    )


def _write_csv(tmp_path: Path, rows: list[tuple], name: str = "test.csv") -> Path:
    p = tmp_path / name
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for row in rows:
            w.writerow(row)
    return p


def _gen_bars(n: int, start_price: float = 100.0, trend: float = 0.1) -> list[tuple]:
    bars = []
    for i in range(n):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
        # Create a simple trending price
        price = start_price + i * trend
        bars.append((
            f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00+00:00",
            str(price), str(price + 1), str(price - 1), str(price), "100",
        ))
    return bars


# ── _snapshot_views ─────────────────────────────────────────


class TestSnapshotViews:
    def test_namespace_snapshot(self):
        snap = _make_snapshot(close=100.0)
        market, positions, event_id = _snapshot_views(snap)
        assert Decimal(str(market.close)) == Decimal("100")
        assert "BTCUSDT" in positions

    def test_dict_snapshot(self):
        snap = {
            "market": SimpleNamespace(close=Decimal("200")),
            "positions": {"BTCUSDT": PositionState.empty("BTCUSDT")},
            "event_id": "e123",
        }
        market, positions, event_id = _snapshot_views(snap)
        assert market.close == Decimal("200")
        assert event_id == "e123"

    def test_dict_with_markets_key(self):
        snap = {
            "markets": {"BTCUSDT": SimpleNamespace(close=Decimal("300"))},
            "positions": {},
        }
        market, positions, event_id = _snapshot_views(snap)
        assert market.close == Decimal("300")

    def test_dict_no_market_raises(self):
        with pytest.raises(RuntimeError, match="missing market"):
            _snapshot_views({"positions": {}})

    def test_unsupported_type_raises(self):
        with pytest.raises(RuntimeError, match="unsupported snapshot type"):
            _snapshot_views(42)


# ── MovingAverageCrossModule ────────────────────────────────


class TestMovingAverageCrossModule:
    def test_no_events_during_warmup(self):
        mod = MovingAverageCrossModule(symbol="BTCUSDT", window=3, order_qty=Decimal("0.01"))
        snap = _make_snapshot(close=100)
        events = list(mod.decide(snap))
        assert events == []

    def test_no_events_second_bar(self):
        mod = MovingAverageCrossModule(symbol="BTCUSDT", window=3, order_qty=Decimal("0.01"))
        mod.decide(_make_snapshot(close=100))
        events = list(mod.decide(_make_snapshot(close=101)))
        assert events == []

    def test_open_long_above_ma(self):
        mod = MovingAverageCrossModule(symbol="BTCUSDT", window=3, order_qty=Decimal("0.01"))
        mod.decide(_make_snapshot(close=100))
        mod.decide(_make_snapshot(close=100))
        # Third bar: MA = (100+100+110)/3=103.33, close=110 > MA, no position => open long
        events = list(mod.decide(_make_snapshot(close=110)))
        assert len(events) == 2  # IntentEvent + OrderEvent
        assert events[0].side == "buy"
        assert events[1].side == "buy"
        assert events[1].qty == Decimal("0.01")

    def test_close_long_below_ma(self):
        mod = MovingAverageCrossModule(symbol="BTCUSDT", window=3, order_qty=Decimal("0.01"))
        mod.decide(_make_snapshot(close=100))
        mod.decide(_make_snapshot(close=100))
        # Open long
        mod.decide(_make_snapshot(close=110))
        # Now price drops below MA
        # closes: [100, 110, 80], MA=96.67, close=80 < MA, has long pos => close
        events = list(mod.decide(_make_snapshot(close=80, position_qty=0.01, avg_price=110)))
        assert len(events) == 2
        assert events[0].side == "sell"

    def test_no_signal_when_already_flat_below_ma(self):
        mod = MovingAverageCrossModule(symbol="BTCUSDT", window=3, order_qty=Decimal("0.01"))
        mod.decide(_make_snapshot(close=100))
        mod.decide(_make_snapshot(close=100))
        # MA=(100+100+90)/3=96.67, close=90 < MA, no position => no event (module only goes long)
        events = list(mod.decide(_make_snapshot(close=90)))
        assert events == []

    def test_no_market_close_returns_empty(self):
        mod = MovingAverageCrossModule(symbol="BTCUSDT", window=3, order_qty=Decimal("0.01"))
        snap = SimpleNamespace(
            market=SimpleNamespace(close=None, last_price=None),
            positions={},
            event_id=None,
        )
        events = list(mod.decide(snap))
        assert events == []


# ── Metrics ────────────────────────────────────────────────


class TestMaxDrawdown:
    def test_no_drawdown(self):
        eq = [Decimal("100"), Decimal("110"), Decimal("120")]
        assert _max_drawdown(eq) == Decimal("0")

    def test_simple_drawdown(self):
        eq = [Decimal("100"), Decimal("80"), Decimal("90")]
        mdd = _max_drawdown(eq)
        assert mdd == Decimal("0.2")  # (100-80)/100

    def test_empty_list(self):
        assert _max_drawdown([]) == Decimal("0")


class TestBuildSummary:
    def test_empty_equity(self):
        s = _build_summary(equity=[], trades=[], csv_path=Path("x.csv"), symbol="BTCUSDT")
        assert s["bars"] == 0
        assert s["trades"] == 0

    def test_basic_summary(self):
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
        equity = [
            EquityPoint(ts=ts1, close=Decimal("100"), position_qty=Decimal("0"),
                        avg_price=None, balance=Decimal("10000"), realized=Decimal("0"),
                        unrealized=Decimal("0"), equity=Decimal("10000")),
            EquityPoint(ts=ts2, close=Decimal("110"), position_qty=Decimal("0"),
                        avg_price=None, balance=Decimal("10100"), realized=Decimal("100"),
                        unrealized=Decimal("0"), equity=Decimal("10100")),
        ]
        s = _build_summary(equity=equity, trades=[], csv_path=Path("x.csv"), symbol="BTCUSDT")
        assert s["bars"] == 2
        assert s["symbol"] == "BTCUSDT"
        assert Decimal(s["return"]) == Decimal("0.01")


# ── Run backtest integration ───────────────────────────────


class TestRunBacktest:
    def test_empty_csv(self, tmp_path: Path):
        csv_path = _write_csv(tmp_path, [])
        equity, fills = run_backtest(
            csv_path=csv_path,
            symbol="BTCUSDT",
            starting_balance=Decimal("10000"),
            fee_bps=Decimal("0"),
        )
        assert equity == []
        assert fills == []

    def test_basic_backtest(self, tmp_path: Path):
        bars = _gen_bars(30, start_price=100, trend=0.5)
        csv_path = _write_csv(tmp_path, bars)
        equity, fills = run_backtest(
            csv_path=csv_path,
            symbol="BTCUSDT",
            starting_balance=Decimal("10000"),
            ma_window=5,
            order_qty=Decimal("0.01"),
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
            embargo_bars=0,
        )
        assert len(equity) > 0

    def test_output_written(self, tmp_path: Path):
        bars = _gen_bars(30, start_price=100, trend=1.0)
        csv_path = _write_csv(tmp_path, bars)
        out_dir = tmp_path / "output"
        run_backtest(
            csv_path=csv_path,
            symbol="BTCUSDT",
            starting_balance=Decimal("10000"),
            ma_window=5,
            order_qty=Decimal("0.01"),
            fee_bps=Decimal("4"),
            out_dir=out_dir,
            embargo_bars=0,
        )
        assert (out_dir / "equity_curve.csv").exists()
        assert (out_dir / "fills.csv").exists()
        assert (out_dir / "summary.json").exists()


class TestBuildTradesFromFills:
    def test_roundtrip_trade(self):
        ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
        ts2 = datetime(2024, 1, 2, tzinfo=timezone.utc).isoformat()
        fills = [
            {"ts": ts1, "symbol": "BTCUSDT", "side": "buy", "qty": "1", "price": "100", "fee": "0.1"},
            {"ts": ts2, "symbol": "BTCUSDT", "side": "sell", "qty": "1", "price": "110", "fee": "0.1"},
        ]
        trades = _build_trades_from_fills(fills)
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTCUSDT"
        assert trades[0]["side"] == "long"
        assert Decimal(trades[0]["gross_pnl"]) == Decimal("10")
