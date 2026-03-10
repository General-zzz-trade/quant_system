# engine/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

try:
    from state.snapshot import StateSnapshot
    _HAS_STRICT_SNAPSHOT = True
except Exception:  # pragma: no cover
    StateSnapshot = Any  # type: ignore
    _HAS_STRICT_SNAPSHOT = False

from _quant_hotpath import rust_detect_event_kind as _rust_detect_kind
from _quant_hotpath import rust_normalize_to_facts as _rust_normalize

from state.rust_adapters import (
    account_from_rust,
    market_from_rust,
    portfolio_from_rust,
    position_from_rust,
    risk_from_rust,
)

from _quant_hotpath import (
    RustMarketState as _RustMarketState,
    RustPositionState as _RustPositionState,
    RustAccountState as _RustAccountState,
)


# ============================================================
# Errors
# ============================================================

class PipelineError(RuntimeError):
    """pipeline 内部错误"""


class FactNormalizationError(PipelineError):
    """事件无法规范化为事实事件"""


# ============================================================
# Contracts
# ============================================================

@dataclass(frozen=True, slots=True)
class PipelineInput:
    """pipeline 的输入载体（只读）"""
    event: Any
    event_index: int
    symbol_default: str

    markets: Mapping[str, Any]
    account: Any
    positions: Mapping[str, Any]

    portfolio: Any = None
    risk: Any = None
    features: Optional[Mapping[str, Any]] = None

    @property
    def market(self) -> Any:
        if self.symbol_default in self.markets:
            return self.markets[self.symbol_default]
        return next(iter(self.markets.values()))


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """pipeline 输出（只读）"""
    markets: Mapping[str, Any]
    account: Any
    positions: Mapping[str, Any]

    portfolio: Any
    risk: Any
    features: Optional[Mapping[str, Any]]

    event_index: int
    last_event_id: Optional[str]
    last_ts: Optional[Any]

    snapshot: Optional[Any]
    advanced: bool

    @property
    def market(self) -> Any:
        return next(iter(self.markets.values()))


# ============================================================
# Utilities
# ============================================================

def _detect_kind(event: Any) -> str:
    return _rust_detect_kind(event)


def normalize_to_facts(event: Any) -> List[Any]:
    try:
        return _rust_normalize(event)
    except RuntimeError as e:
        raise FactNormalizationError(str(e)) from e


class _LazyConvertMapping(Mapping):
    """Lazily converts Rust state types to Python on first access per key."""

    __slots__ = ("_raw", "_converted", "_converter", "_check_type")

    def __init__(self, raw: Mapping[str, Any], converter, check_type) -> None:
        self._raw = raw
        self._converted: dict = {}
        self._converter = converter
        self._check_type = check_type

    def __getitem__(self, key: str) -> Any:
        if key in self._converted:
            return self._converted[key]
        val = self._raw[key]
        if isinstance(val, self._check_type):
            val = self._converter(val)
        self._converted[key] = val
        return val

    def __iter__(self):
        return iter(self._raw)

    def __len__(self):
        return len(self._raw)


def _build_snapshot(
    *,
    raw_event: Any,
    event_index: int,
    markets: Mapping[str, Any],
    account: Any,
    positions: Mapping[str, Any],
    portfolio: Any,
    risk: Any,
    features: Optional[Mapping[str, Any]] = None,
    skip_convert: bool = False,
) -> Any:
    header = getattr(raw_event, "header", None)
    event_id = getattr(header, "event_id", None)
    ts = getattr(header, "ts", None)
    if event_id is not None and not isinstance(event_id, str):
        event_id = None

    if skip_convert:
        # Fast path: pass Rust objects directly (duck-typed, consumers use _f fields)
        snapshot_markets = markets
        snapshot_account = account
        snapshot_positions = positions
    else:
        snapshot_markets = _LazyConvertMapping(markets, market_from_rust, _RustMarketState)
        snapshot_account = account_from_rust(account) if isinstance(account, _RustAccountState) else account
        snapshot_positions = _LazyConvertMapping(positions, position_from_rust, _RustPositionState)

    if _HAS_STRICT_SNAPSHOT:
        event_type_str = getattr(raw_event, "event_type", "unknown")
        if hasattr(event_type_str, "value"):
            event_type_str = event_type_str.value
        return StateSnapshot.of(
            symbol=getattr(raw_event, "symbol", ""),
            ts=ts,
            event_id=event_id,
            event_type=str(event_type_str),
            bar_index=event_index,
            markets=snapshot_markets,
            positions=snapshot_positions,
            account=snapshot_account,
            portfolio=portfolio,
            risk=risk,
            features=features,
        )

    return {
        "event_id": event_id,
        "event_index": event_index,
        "ts": ts,
        "markets": snapshot_markets,
        "account": snapshot_account,
        "positions": snapshot_positions,
        "portfolio": portfolio,
        "risk": risk,
        "features": features,
    }


def _export_store_state(store: Any) -> Tuple[Mapping[str, Any], Any, Mapping[str, Any], Any, Any, Any, Any, Any]:
    markets = store.get_markets()
    positions = store.get_positions()
    account = store.get_account()
    portfolio = portfolio_from_rust(store.get_portfolio())
    risk = risk_from_rust(store.get_risk())
    return (
        markets, account, positions, portfolio, risk,
        store.event_index,
        store.last_event_id,
        store.last_ts,
    )


# ============================================================
# Pipeline
# ============================================================

@dataclass(frozen=True, slots=True)
class PipelineConfig:
    build_snapshot_on_change_only: bool = True
    fail_on_missing_symbol: bool = False
    default_symbol_fallback: bool = True


class StatePipeline:
    """StatePipeline — state 的唯一写通道。

    State lives on the Rust heap via RustStateStore. No per-event
    Python↔Rust Decimal conversion. Only exports on demand.
    """

    def __init__(
        self,
        *,
        config: Optional[PipelineConfig] = None,
        store: Any = None,
    ) -> None:
        if store is None:
            raise ValueError("StatePipeline requires a RustStateStore")
        self._cfg = config or PipelineConfig()
        self._store = store

    @property
    def config(self) -> PipelineConfig:
        return self._cfg

    def apply(self, inp: PipelineInput) -> PipelineOutput:
        store = self._store
        result = store.process_event(inp.event, inp.symbol_default)

        if not result.advanced:
            return PipelineOutput(
                markets=inp.markets,
                account=inp.account,
                positions=inp.positions,
                portfolio=inp.portfolio,
                risk=inp.risk,
                features=inp.features,
                event_index=int(inp.event_index),
                last_event_id=getattr(inp, "last_event_id", None),
                last_ts=getattr(inp, "last_ts", None),
                snapshot=None,
                advanced=False,
            )

        next_index = int(result.event_index)

        need_export = (
            (not self._cfg.build_snapshot_on_change_only)
            or result.changed
            or inp.portfolio is None
            or inp.risk is None
        )

        if need_export:
            (
                markets, account, positions, portfolio, risk,
                bundled_event_index, bundled_last_event_id, bundled_last_ts,
            ) = _export_store_state(store)

            snapshot: Optional[Any] = _build_snapshot(
                raw_event=inp.event,
                event_index=next_index,
                markets=markets,
                account=account,
                positions=positions,
                portfolio=portfolio,
                risk=risk,
                features=inp.features,
            )

            return PipelineOutput(
                markets=markets,
                account=account,
                positions=positions,
                portfolio=portfolio,
                risk=risk,
                features=inp.features,
                event_index=int(bundled_event_index),
                last_event_id=bundled_last_event_id,
                last_ts=bundled_last_ts,
                snapshot=snapshot,
                advanced=True,
            )

        return PipelineOutput(
            markets=inp.markets,
            account=inp.account,
            positions=inp.positions,
            portfolio=inp.portfolio,
            risk=inp.risk,
            features=inp.features,
            event_index=next_index,
            last_event_id=store.last_event_id,
            last_ts=store.last_ts,
            snapshot=None,
            advanced=True,
        )
