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

from _quant_hotpath import (  # type: ignore[import-untyped]
    rust_detect_event_kind as _rust_detect_kind,
    rust_normalize_to_facts as _rust_normalize,
    rust_detect_kernel_event_kind as _rust_detect_kernel_kind,
    rust_normalize_kernel_event_to_facts as _rust_normalize_kernel,
    rust_pipeline_apply,  # noqa: F401 — re-exported for callers
    RustProcessResult,
    RustMarketReducer,
    RustPositionReducer,
    RustAccountReducer,
    RustPortfolioReducer,
    RustRiskReducer,
)

import logging as _logging

_pipeline_logger = _logging.getLogger(__name__)

# Reducer registry: maps event kind → the Rust reducer responsible for that
# state transition.  Used for audit logging and type dispatch documentation.
REDUCER_REGISTRY: dict[str, type] = {
    "MARKET": RustMarketReducer,
    "FILL": RustPositionReducer,
    "ACCOUNT": RustAccountReducer,
    "PORTFOLIO": RustPortfolioReducer,
    "RISK": RustRiskReducer,
}



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
    return str(_rust_detect_kind(event))


def normalize_to_facts(event: Any) -> List[Any]:
    try:
        return list(_rust_normalize(event))
    except RuntimeError as e:
        raise FactNormalizationError(str(e)) from e


def detect_kernel_event_kind(payload_json: str) -> str:
    """Classify a kernel-level event (binary path JSON payload).

    Falls back to the standard detector on failure so callers always
    get a usable kind string.
    """
    try:
        return str(_rust_detect_kernel_kind(payload_json))
    except Exception:
        _pipeline_logger.debug("Kernel event kind detection failed, payload truncated: %s", payload_json[:120])
        return "UNKNOWN"


def normalize_kernel_event_to_facts(payload_json: str) -> str:
    """Normalize a kernel-level event payload to a fact representation.

    Used on the binary (standalone Rust trader) path where events arrive
    as JSON strings rather than Python objects.
    """
    return str(_rust_normalize_kernel(payload_json))


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
            markets=markets,
            positions=positions,
            account=account,
            portfolio=portfolio,
            risk=risk,
            features=features,
        )

    return {
        "event_id": event_id,
        "event_index": event_index,
        "ts": ts,
        "markets": markets,
        "account": account,
        "positions": positions,
        "portfolio": portfolio,
        "risk": risk,
        "features": features,
    }


def _export_store_state(store: Any) -> Tuple[Mapping[str, Any], Any, Mapping[str, Any], Any, Any, Any, Any, Any]:
    markets = store.get_markets()
    positions = store.get_positions()
    account = store.get_account()
    portfolio = store.get_portfolio()
    risk = store.get_risk()
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
        result: RustProcessResult = store.process_event(inp.event, inp.symbol_default)

        if result.advanced:
            _pipeline_logger.debug(
                "State advanced: index=%d kind=%s changed=%s",
                result.event_index, result.kind, result.changed,
            )

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
