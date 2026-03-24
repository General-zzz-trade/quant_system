"""JSON serialization helpers for Rust state types → plain dicts and back.

Used by SqliteStateStore and InMemoryStateStore for checkpoint persistence.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from _quant_hotpath import (  # type: ignore[import-untyped]
    RustMarketState,
    RustPositionState,
    RustAccountState,
    RustPortfolioState,
    RustRiskState,
)

from state.snapshot import StateSnapshot

# Fixed-point scale: 10^8 (matches Rust Fd8)
_SCALE = 100_000_000


# ---------------------------------------------------------------------------
# Rust state types → plain dicts
# ---------------------------------------------------------------------------

def _rust_market_to_dict(obj: RustMarketState) -> dict:
    return {
        "symbol": obj.symbol,
        "last_price": obj.last_price,
        "open": obj.open,
        "high": obj.high,
        "low": obj.low,
        "close": obj.close,
        "volume": obj.volume,
        "last_ts": obj.last_ts,
    }


def _rust_position_to_dict(obj: RustPositionState) -> dict:
    return {
        "symbol": obj.symbol,
        "qty": obj.qty,
        "avg_price": obj.avg_price,
        "last_price": obj.last_price,
        "last_ts": obj.last_ts,
    }


def _rust_account_to_dict(obj: RustAccountState) -> dict:
    return {
        "currency": obj.currency,
        "balance": obj.balance,
        "margin_used": obj.margin_used,
        "margin_available": obj.margin_available,
        "realized_pnl": obj.realized_pnl,
        "unrealized_pnl": obj.unrealized_pnl,
        "fees_paid": obj.fees_paid,
        "last_ts": obj.last_ts,
    }


def _rust_portfolio_to_dict(obj: RustPortfolioState) -> dict:
    return {
        "total_equity": obj.total_equity,
        "cash_balance": obj.cash_balance,
        "realized_pnl": obj.realized_pnl,
        "unrealized_pnl": obj.unrealized_pnl,
        "fees_paid": obj.fees_paid,
        "gross_exposure": obj.gross_exposure,
        "net_exposure": obj.net_exposure,
        "leverage": obj.leverage,
        "margin_used": obj.margin_used,
        "margin_available": obj.margin_available,
        "margin_ratio": obj.margin_ratio,
        "symbols": list(obj.symbols),
        "last_ts": obj.last_ts,
    }


def _rust_risk_to_dict(obj: RustRiskState) -> dict:
    return {
        "blocked": obj.blocked,
        "halted": obj.halted,
        "level": obj.level,
        "message": obj.message,
        "flags": list(obj.flags),
        "equity_peak": obj.equity_peak,
        "drawdown_pct": obj.drawdown_pct,
        "last_ts": obj.last_ts,
    }


def _dc_to_dict(obj: Any) -> Any:
    """Recursively convert Rust state types / dataclasses to plain dicts."""
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, Decimal):
        return {"__decimal__": str(obj)}
    if isinstance(obj, datetime):
        return {"__datetime__": obj.isoformat()}
    if isinstance(obj, RustMarketState):
        return _rust_market_to_dict(obj)
    if isinstance(obj, RustPositionState):
        return _rust_position_to_dict(obj)
    if isinstance(obj, RustAccountState):
        return _rust_account_to_dict(obj)
    if isinstance(obj, RustPortfolioState):
        return _rust_portfolio_to_dict(obj)
    if isinstance(obj, RustRiskState):
        return _rust_risk_to_dict(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {f: _dc_to_dict(getattr(obj, f)) for f in obj.__dataclass_fields__}
    if isinstance(obj, dict):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    # MappingProxyType and other Mapping-like objects
    if hasattr(obj, "items") and callable(getattr(obj, "items")):
        return {k: _dc_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dc_to_dict(v) for v in obj]
    return obj


class _StateEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return {"__decimal__": str(obj)}
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        if isinstance(obj, (RustMarketState, RustPositionState, RustAccountState,
                            RustPortfolioState, RustRiskState)):
            return _dc_to_dict(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return _dc_to_dict(obj)
        if hasattr(obj, "items") and callable(getattr(obj, "items")):
            return {k: _dc_to_dict(v) for k, v in obj.items()}
        return super().default(obj)


def _state_decoder_hook(d: dict[str, Any]) -> Any:
    if "__decimal__" in d:
        return Decimal(d["__decimal__"])
    if "__datetime__" in d:
        dt = datetime.fromisoformat(d["__datetime__"])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return d


def _serialize_snapshot(snap: StateSnapshot) -> str:
    return json.dumps(snap, cls=_StateEncoder, ensure_ascii=False)


def _deserialize_snapshot(blob: str) -> StateSnapshot:
    raw = json.loads(blob, object_hook=_state_decoder_hook)
    return _reconstruct_snapshot(raw)


# ---------------------------------------------------------------------------
# Value coercion helpers (float/Decimal → Rust i64 fixed-point)
# ---------------------------------------------------------------------------

def _to_opt_i64(val: Any) -> Optional[int]:
    """Convert a value to Optional[int] for Rust i64 fields.

    Handles: None, int, Decimal (from old checkpoints), float.
    """
    if val is None:
        return None
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, Decimal):
        return int(val * _SCALE)
    if isinstance(val, float):
        return int(val * _SCALE)
    return int(val)


def _to_i64(val: Any, default: int = 0) -> int:
    """Convert a value to int for Rust i64 fields (non-optional)."""
    if val is None:
        return default
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    if isinstance(val, Decimal):
        return int(val * _SCALE)
    if isinstance(val, float):
        return int(val * _SCALE)
    return int(val)


def _to_str(val: Any, default: str = "0") -> str:
    """Convert a value to string for Rust String fields (Portfolio/Risk)."""
    if val is None:
        return default
    return str(val)


def _to_opt_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    return str(val)


# ---------------------------------------------------------------------------
# Reconstruct Rust state types from plain dicts
# ---------------------------------------------------------------------------

def _reconstruct_market(md: dict) -> RustMarketState:
    return RustMarketState(
        symbol=str(md["symbol"]),
        last_price=_to_opt_i64(md.get("last_price")),
        open=_to_opt_i64(md.get("open")),
        high=_to_opt_i64(md.get("high")),
        low=_to_opt_i64(md.get("low")),
        close=_to_opt_i64(md.get("close")),
        volume=_to_opt_i64(md.get("volume")),
        last_ts=md.get("last_ts"),
    )


def _reconstruct_position(pd: dict) -> RustPositionState:
    return RustPositionState(
        symbol=str(pd["symbol"]),
        qty=_to_i64(pd.get("qty", 0)),
        avg_price=_to_opt_i64(pd.get("avg_price")),
        last_price=_to_opt_i64(pd.get("last_price")),
        last_ts=pd.get("last_ts"),
    )


def _reconstruct_account(ad: dict) -> RustAccountState:
    return RustAccountState(
        currency=str(ad.get("currency", "USDT")),
        balance=_to_i64(ad.get("balance", 0)),
        margin_used=_to_i64(ad.get("margin_used", 0)),
        margin_available=_to_i64(ad.get("margin_available", 0)),
        realized_pnl=_to_i64(ad.get("realized_pnl", 0)),
        unrealized_pnl=_to_i64(ad.get("unrealized_pnl", 0)),
        fees_paid=_to_i64(ad.get("fees_paid", 0)),
        last_ts=ad.get("last_ts"),
    )


def _reconstruct_portfolio(pd: dict) -> RustPortfolioState:
    symbols = pd.get("symbols", [])
    if isinstance(symbols, tuple):
        symbols = list(symbols)
    return RustPortfolioState(
        total_equity=_to_str(pd.get("total_equity")),
        cash_balance=_to_str(pd.get("cash_balance")),
        realized_pnl=_to_str(pd.get("realized_pnl")),
        unrealized_pnl=_to_str(pd.get("unrealized_pnl")),
        fees_paid=_to_str(pd.get("fees_paid")),
        gross_exposure=_to_str(pd.get("gross_exposure")),
        net_exposure=_to_str(pd.get("net_exposure")),
        leverage=_to_opt_str(pd.get("leverage")),
        margin_used=_to_str(pd.get("margin_used")),
        margin_available=_to_str(pd.get("margin_available")),
        margin_ratio=_to_opt_str(pd.get("margin_ratio")),
        symbols=symbols,
        last_ts=pd.get("last_ts"),
    )


def _reconstruct_risk(rd: dict) -> RustRiskState:
    flags = rd.get("flags", [])
    if isinstance(flags, tuple):
        flags = list(flags)
    return RustRiskState(
        blocked=bool(rd.get("blocked", False)),
        halted=bool(rd.get("halted", False)),
        level=rd.get("level"),
        message=rd.get("message"),
        flags=flags,
        equity_peak=_to_str(rd.get("equity_peak")),
        drawdown_pct=_to_str(rd.get("drawdown_pct")),
        last_ts=rd.get("last_ts"),
    )


def _reconstruct_snapshot(d: dict[str, Any]) -> StateSnapshot:
    # Multi-symbol compat: new format "markets" dict, old format "market" single object
    markets: Dict[str, Any] = {}
    if "markets" in d and isinstance(d["markets"], dict):
        for sym, md in d["markets"].items():
            if isinstance(md, dict):
                md.pop("__dc__", None)
                if "symbol" not in md:
                    md["symbol"] = sym
                markets[sym] = _reconstruct_market(md)
    elif "market" in d:
        market_d = d["market"]
        if isinstance(market_d, dict):
            market_d.pop("__dc__", None)
            sym = market_d.get("symbol", d.get("symbol", "UNKNOWN"))
            if "symbol" not in market_d:
                market_d["symbol"] = sym
            markets[sym] = _reconstruct_market(market_d)

    account_d = d["account"]
    if isinstance(account_d, dict):
        account_d.pop("__dc__", None)
        account = _reconstruct_account(account_d)
    else:
        account = account_d  # Already a Rust type

    positions: Dict[str, Any] = {}
    raw_pos = d.get("positions") or {}
    for sym, pos_d in raw_pos.items():
        if isinstance(pos_d, dict):
            pos_d.pop("__dc__", None)
            if "symbol" not in pos_d:
                pos_d["symbol"] = sym
            positions[sym] = _reconstruct_position(pos_d)

    portfolio = None
    if d.get("portfolio") is not None:
        pd = d["portfolio"]
        if isinstance(pd, dict):
            pd.pop("__dc__", None)
            portfolio = _reconstruct_portfolio(pd)
        else:
            portfolio = pd  # Already a Rust type

    risk = None
    if d.get("risk") is not None:
        rd = d["risk"]
        if isinstance(rd, dict):
            rd.pop("__dc__", None)
            risk = _reconstruct_risk(rd)
        else:
            risk = rd  # Already a Rust type

    return StateSnapshot.of(
        symbol=d["symbol"],
        ts=d.get("ts"),
        event_id=d.get("event_id"),
        event_type=d.get("event_type", ""),
        bar_index=d.get("bar_index", 0),
        markets=markets,
        positions=positions,
        account=account,
        portfolio=portfolio,
        risk=risk,
    )
