from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from decision.intents.target_position import TargetPositionIntentBuilder
from decision.types import TargetPosition
from state import AccountState
from state import MarketState
from state import PositionState
from state.snapshot import StateSnapshot

_SCALE = 100_000_000


def _snapshot(pos_qty: str = '0') -> StateSnapshot:
    ts = datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
    ts_str = ts.isoformat()
    return StateSnapshot(
        symbol='BTCUSDT',
        ts=ts,
        event_id='e-1',
        event_type='market',
        bar_index=1,
        markets={
            'BTCUSDT': MarketState(
                symbol='BTCUSDT',
                last_price=100 * _SCALE,
                open=100 * _SCALE,
                high=101 * _SCALE,
                low=99 * _SCALE,
                close=100 * _SCALE,
                volume=1 * _SCALE,
                last_ts=ts_str,
            )
        },
        positions={
            'BTCUSDT': PositionState(
                symbol='BTCUSDT',
                qty=int(Decimal(pos_qty) * _SCALE),
                avg_price=None,
                last_price=100 * _SCALE,
                last_ts=ts_str,
            )
        },
        account=AccountState(
            currency='USDT',
            balance=1000 * _SCALE,
            margin_used=0,
            margin_available=1000 * _SCALE,
            realized_pnl=0,
            unrealized_pnl=0,
            fees_paid=0,
            last_ts=ts_str,
        ),
        portfolio=None,
        risk=None,
    )


class TestTargetPositionIntentBuilder:
    def test_builds_buy_delta(self) -> None:
        builder = TargetPositionIntentBuilder()
        order = builder.build(_snapshot('0.5'), TargetPosition(symbol='BTCUSDT', target_qty=Decimal('1.25')))

        assert order is not None
        assert order.side == 'buy'
        assert order.qty == Decimal('0.75')
        assert order.meta == {'reason_code': 'signal', 'origin': 'decision'}

    def test_builds_sell_delta(self) -> None:
        builder = TargetPositionIntentBuilder()
        order = builder.build(_snapshot('2'), TargetPosition(symbol='BTCUSDT', target_qty=Decimal('0.5')))

        assert order is not None
        assert order.side == 'sell'
        assert order.qty == Decimal('1.5')

    def test_returns_none_for_zero_delta(self) -> None:
        builder = TargetPositionIntentBuilder()
        order = builder.build(_snapshot('1'), TargetPosition(symbol='BTCUSDT', target_qty=Decimal('1')))

        assert order is None
