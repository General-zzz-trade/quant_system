from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from state._util import ensure_utc, to_decimal
from state.account import AccountState
from state.errors import ReducerError
from state.market import MarketState
from state.portfolio import PortfolioState
from state.position import PositionState
from state.risk import RiskLimits, RiskState

from _quant_hotpath import (
    RustAccountState,
    RustMarketState,
    RustPortfolioState,
    RustPositionState,
    RustRiskLimits,
    RustRiskState,
)

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
# Portfolio / risk converters
# ---------------------------------------------------------------------------

def portfolio_to_rust(state: PortfolioState) -> Any:
    return RustPortfolioState(
        total_equity=str(state.total_equity),
        cash_balance=str(state.cash_balance),
        realized_pnl=str(state.realized_pnl),
        unrealized_pnl=str(state.unrealized_pnl),
        fees_paid=str(state.fees_paid),
        gross_exposure=str(state.gross_exposure),
        net_exposure=str(state.net_exposure),
        leverage=(str(state.leverage) if state.leverage is not None else None),
        margin_used=str(state.margin_used),
        margin_available=str(state.margin_available),
        margin_ratio=(str(state.margin_ratio) if state.margin_ratio is not None else None),
        symbols=list(state.symbols),
        last_ts=_ts_to_rust(state.last_ts),
    )


def portfolio_from_rust(state: Any) -> PortfolioState:
    return PortfolioState(
        total_equity=to_decimal(state.total_equity),
        cash_balance=to_decimal(state.cash_balance),
        realized_pnl=to_decimal(state.realized_pnl),
        unrealized_pnl=to_decimal(state.unrealized_pnl),
        fees_paid=to_decimal(state.fees_paid),
        gross_exposure=to_decimal(state.gross_exposure),
        net_exposure=to_decimal(state.net_exposure),
        leverage=to_decimal(state.leverage) if getattr(state, "leverage", None) is not None else None,
        margin_used=to_decimal(state.margin_used),
        margin_available=to_decimal(state.margin_available),
        margin_ratio=to_decimal(state.margin_ratio) if getattr(state, "margin_ratio", None) is not None else None,
        symbols=tuple(getattr(state, "symbols", ()) or ()),
        last_ts=_ts_from_rust(getattr(state, "last_ts", None)),
    )


def risk_limits_to_rust(limits: RiskLimits) -> Any:
    return RustRiskLimits(
        max_leverage=str(limits.max_leverage),
        max_position_notional=(
            str(limits.max_position_notional)
            if limits.max_position_notional is not None
            else None
        ),
        max_drawdown_pct=str(limits.max_drawdown_pct),
        block_on_equity_le_zero=bool(limits.block_on_equity_le_zero),
    )


def risk_to_rust(state: RiskState) -> Any:
    return RustRiskState(
        blocked=bool(state.blocked),
        halted=bool(state.halted),
        level=state.level,
        message=state.message,
        flags=list(state.flags),
        equity_peak=str(state.equity_peak),
        drawdown_pct=str(state.drawdown_pct),
        last_ts=_ts_to_rust(state.last_ts),
    )


def risk_from_rust(state: Any) -> RiskState:
    return RiskState(
        blocked=bool(state.blocked),
        halted=bool(state.halted),
        level=getattr(state, "level", None),
        message=getattr(state, "message", None),
        flags=tuple(getattr(state, "flags", ()) or ()),
        equity_peak=to_decimal(state.equity_peak),
        drawdown_pct=to_decimal(state.drawdown_pct),
        last_ts=_ts_from_rust(getattr(state, "last_ts", None)),
    )


