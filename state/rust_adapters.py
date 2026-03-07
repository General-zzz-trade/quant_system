from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from state._util import ensure_utc, to_decimal
from state.account import AccountState
from state.errors import ReducerError
from state.market import MarketState
from state.position import PositionState
from state.reducers.base import ReducerResult

from _quant_hotpath import (
    RustAccountReducer,
    RustAccountState,
    RustMarketReducer,
    RustMarketState,
    RustPositionReducer,
    RustPositionState,
)

HAS_RUST_STATE_REDUCERS = True

# ---------------------------------------------------------------------------
# Fixed-point scale: 10^8 (8 decimal places, matches crypto exchange precision)
# ---------------------------------------------------------------------------
_SCALE = 100_000_000
_DECIMAL_SCALE = Decimal("100000000")


def _decimal_to_i64(value: Optional[Decimal]) -> Optional[int]:
    if value is None:
        return None
    return int(value * _DECIMAL_SCALE)


def _decimal_to_i64_required(value: Decimal) -> int:
    return int(value * _DECIMAL_SCALE)


def _i64_to_decimal(value: Optional[int], *, allow_none: bool = False) -> Optional[Decimal]:
    if value is None:
        return None if allow_none else Decimal("0")
    return Decimal(value) / _DECIMAL_SCALE


def _ts_to_rust(ts: Optional[datetime]) -> Optional[str]:
    if ts is None:
        return None
    return ensure_utc(ts).isoformat()


def _ts_from_rust(ts: Optional[str]) -> Optional[datetime]:
    if ts is None:
        return None
    raw = str(ts).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as e:  # pragma: no cover - Rust emits ISO strings
        raise ReducerError(f"invalid rust timestamp: {ts!r}") from e
    return ensure_utc(parsed)


# ---------------------------------------------------------------------------
# Market state converters
# ---------------------------------------------------------------------------

def market_to_rust(state: MarketState) -> Any:
    return RustMarketState(
        symbol=state.symbol,
        last_price=_decimal_to_i64(state.last_price),
        open=_decimal_to_i64(state.open),
        high=_decimal_to_i64(state.high),
        low=_decimal_to_i64(state.low),
        close=_decimal_to_i64(state.close),
        volume=_decimal_to_i64(state.volume),
        last_ts=_ts_to_rust(state.last_ts),
    )


def market_from_rust(state: Any) -> MarketState:
    return MarketState(
        symbol=str(state.symbol),
        last_price=_i64_to_decimal(state.last_price, allow_none=True),
        open=_i64_to_decimal(state.open, allow_none=True),
        high=_i64_to_decimal(state.high, allow_none=True),
        low=_i64_to_decimal(state.low, allow_none=True),
        close=_i64_to_decimal(state.close, allow_none=True),
        volume=_i64_to_decimal(state.volume, allow_none=True),
        last_ts=_ts_from_rust(getattr(state, "last_ts", None)),
    )


# ---------------------------------------------------------------------------
# Position state converters
# ---------------------------------------------------------------------------

def position_to_rust(state: PositionState) -> Any:
    return RustPositionState(
        symbol=state.symbol,
        qty=_decimal_to_i64_required(state.qty),
        avg_price=_decimal_to_i64(state.avg_price),
        last_price=_decimal_to_i64(state.last_price),
        last_ts=_ts_to_rust(state.last_ts),
    )


def position_from_rust(state: Any) -> PositionState:
    return PositionState(
        symbol=str(state.symbol),
        qty=_i64_to_decimal(state.qty) or Decimal("0"),
        avg_price=_i64_to_decimal(state.avg_price, allow_none=True),
        last_price=_i64_to_decimal(state.last_price, allow_none=True),
        last_ts=_ts_from_rust(getattr(state, "last_ts", None)),
    )


# ---------------------------------------------------------------------------
# Account state converters
# ---------------------------------------------------------------------------

def account_to_rust(state: AccountState) -> Any:
    return RustAccountState(
        currency=state.currency,
        balance=_decimal_to_i64_required(state.balance),
        margin_used=_decimal_to_i64_required(state.margin_used),
        margin_available=_decimal_to_i64_required(state.margin_available),
        realized_pnl=_decimal_to_i64_required(state.realized_pnl),
        unrealized_pnl=_decimal_to_i64_required(state.unrealized_pnl),
        fees_paid=_decimal_to_i64_required(state.fees_paid),
        last_ts=_ts_to_rust(state.last_ts),
    )


def account_from_rust(state: Any) -> AccountState:
    return AccountState(
        currency=str(state.currency),
        balance=_i64_to_decimal(state.balance) or Decimal("0"),
        margin_used=_i64_to_decimal(state.margin_used) or Decimal("0"),
        margin_available=_i64_to_decimal(state.margin_available) or Decimal("0"),
        realized_pnl=_i64_to_decimal(state.realized_pnl) or Decimal("0"),
        unrealized_pnl=_i64_to_decimal(state.unrealized_pnl) or Decimal("0"),
        fees_paid=_i64_to_decimal(state.fees_paid) or Decimal("0"),
        last_ts=_ts_from_rust(getattr(state, "last_ts", None)),
    )


# ---------------------------------------------------------------------------
# Coerce note helper
# ---------------------------------------------------------------------------

def _coerce_note(value: Any) -> Optional[str]:
    if value is None:
        return None
    note = str(value)
    return note or None


# ---------------------------------------------------------------------------
# Reducer adapters (used by slow path when fast path is disabled)
# ---------------------------------------------------------------------------

class RustMarketReducerAdapter:
    def __init__(self) -> None:
        self._inner = RustMarketReducer()

    def reduce(self, state: MarketState, event: Any) -> ReducerResult[MarketState]:
        try:
            res = self._inner.reduce(market_to_rust(state), event)
        except Exception as e:
            raise ReducerError(str(e)) from e
        return ReducerResult(
            state=market_from_rust(res.state),
            changed=bool(res.changed),
            note=_coerce_note(getattr(res, "note", None)),
        )


class RustPositionReducerAdapter:
    def __init__(self) -> None:
        self._inner = RustPositionReducer()

    def reduce(self, state: PositionState, event: Any) -> ReducerResult[PositionState]:
        try:
            res = self._inner.reduce(position_to_rust(state), event)
        except Exception as e:
            raise ReducerError(str(e)) from e
        return ReducerResult(
            state=position_from_rust(res.state),
            changed=bool(res.changed),
            note=_coerce_note(getattr(res, "note", None)),
        )


class RustAccountReducerAdapter:
    def __init__(self) -> None:
        self._inner = RustAccountReducer()

    def reduce(self, state: AccountState, event: Any) -> ReducerResult[AccountState]:
        try:
            res = self._inner.reduce(account_to_rust(state), event)
        except Exception as e:
            raise ReducerError(str(e)) from e
        return ReducerResult(
            state=account_from_rust(res.state),
            changed=bool(res.changed),
            note=_coerce_note(getattr(res, "note", None)),
        )
