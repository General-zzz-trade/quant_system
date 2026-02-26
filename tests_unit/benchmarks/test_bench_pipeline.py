"""Performance benchmark: StatePipeline throughput."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pytest

from state.account import AccountState
from state.market import MarketState
from state.position import PositionState
from engine.pipeline import PipelineConfig, PipelineInput, StatePipeline


@dataclass
class _FillEvent:
    symbol: str = "BTCUSDT"
    side: str = "buy"
    qty: Decimal = Decimal("0.1")
    price: Decimal = Decimal("50000")
    event_type: str = "fill"
    ts: Optional[datetime] = None
    fee: Optional[Decimal] = None
    realized_pnl: Optional[Decimal] = None
    cash_delta: Optional[Decimal] = None
    margin_change: Optional[Decimal] = None


@pytest.mark.benchmark
def test_pipeline_throughput():
    """Pipeline should process >1,000 fill events/sec."""
    n_events = 10_000
    pipeline = StatePipeline(config=PipelineConfig(
        build_snapshot_on_change_only=True,
        fail_on_missing_symbol=False,
    ))

    events = []
    for i in range(n_events):
        side = "buy" if i % 2 == 0 else "sell"
        events.append(_FillEvent(
            side=side,
            qty=Decimal("0.01"),
            price=Decimal("50000") + Decimal(str(i % 100)),
            ts=datetime(2024, 6, 1, tzinfo=timezone.utc),
        ))

    market = MarketState.empty("BTCUSDT")
    account = AccountState.initial(currency="USDT", balance=Decimal("100000"))
    positions: dict[str, PositionState] = {}
    idx = 0

    start = time.perf_counter()
    for ev in events:
        inp = PipelineInput(
            event=ev,
            event_index=idx,
            symbol_default="BTCUSDT",
            market=market,
            account=account,
            positions=positions,
        )
        out = pipeline.apply(inp)
        market = out.market
        account = out.account
        positions = dict(out.positions)
        idx = out.event_index
    elapsed = time.perf_counter() - start

    throughput = n_events / elapsed
    print(f"\nPipeline throughput: {throughput:,.0f} events/sec ({elapsed:.3f}s for {n_events:,} events)")
    assert throughput > 1_000, f"Pipeline too slow: {throughput:.0f} events/sec"
