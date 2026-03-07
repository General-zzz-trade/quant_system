# engine/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple

# -------------------------
# state 侧（你的 state 层）
# -------------------------
from state.market import MarketState
from state.account import AccountState
from state.position import PositionState

from state.reducers.base import Reducer
from state.reducers.market import MarketReducer
from state.reducers.account import AccountReducer
from state.reducers.position import PositionReducer

# snapshot：可选依赖（你实现齐 portfolio/risk 后自动升级为严格快照）
try:
    from state.snapshot import StateSnapshot
    _HAS_STRICT_SNAPSHOT = True
except Exception:  # pragma: no cover
    StateSnapshot = Any  # type: ignore
    _HAS_STRICT_SNAPSHOT = False


# -------------------------
# Rust acceleration (PyO3) — hard dependency
# -------------------------
from _quant_hotpath import rust_detect_event_kind as _rust_detect_kind
from _quant_hotpath import rust_normalize_to_facts as _rust_normalize

from state.rust_adapters import (
    RustAccountReducerAdapter,
    RustMarketReducerAdapter,
    RustPositionReducerAdapter,
    account_from_rust,
    derive_portfolio_and_risk,
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

# -------------------------
# event 侧（你的 event 层）
# -------------------------
try:
    from event.types import EventType  # 可能存在
except Exception:  # pragma: no cover
    EventType = None  # type: ignore


# ============================================================
# Errors
# ============================================================

class PipelineError(RuntimeError):
    """pipeline 内部错误（不等价于 reducer 错误）"""


class FactNormalizationError(PipelineError):
    """事件无法规范化为事实事件"""


# ============================================================
# Contracts
# ============================================================

@dataclass(frozen=True, slots=True)
class PipelineInput:
    """
    pipeline 的输入载体（只读）

    State fields accept both Python dataclass types (MarketState etc.)
    and Rust PyO3 types (RustMarketState etc.) — duck-typed.
    """
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
        """Backward compat: 返回 symbol_default 对应的 MarketState。"""
        if self.symbol_default in self.markets:
            return self.markets[self.symbol_default]
        return next(iter(self.markets.values()))


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """
    pipeline 输出（只读）

    State fields hold Rust types (RustMarketState etc.) when using
    fast/store paths, Python types when using slow reducer path.
    """
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
        """Backward compat: 返回第一个 MarketState。"""
        return next(iter(self.markets.values()))


class ReducerTriplet(Protocol):
    """
    允许你未来替换 reducers（例如多市场、多账户、多资产）
    """
    def market(self) -> MarketReducer: ...
    def account(self) -> AccountReducer: ...
    def position(self) -> PositionReducer: ...


# ============================================================
# Utilities: event header / id / ts
# ============================================================

def _event_id_ts(event: Any) -> Tuple[Optional[str], Optional[Any]]:
    header = getattr(event, "header", None)
    event_id = getattr(header, "event_id", None)
    ts = getattr(header, "ts", None)
    return (event_id if isinstance(event_id, str) else None, ts)


# ============================================================
# Utilities: detect & normalize
# ============================================================

def _detect_kind(event: Any) -> str:
    """
    返回：MARKET / FILL / ORDER / SIGNAL / INTENT / RISK / CONTROL / FUNDING / UNKNOWN
    """
    return _rust_detect_kind(event)



def normalize_to_facts(event: Any) -> List[Any]:
    """
    event -> reducers 可消费的 facts（非空才会推进 event_index）
    """
    try:
        return _rust_normalize(event)
    except RuntimeError as e:
        raise FactNormalizationError(str(e)) from e



# ============================================================
# Snapshot builder (policy)
# ============================================================

def _build_snapshot(
    *,
    raw_event: Any,
    event_index: int,
    markets: Mapping[str, MarketState],
    account: AccountState,
    positions: Mapping[str, PositionState],
    portfolio: Any,
    risk: Any,
    features: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    统一快照生成点（顶级纪律：snapshot 生成必须集中，不可散落）

    - 若 state.snapshot.build_snapshot 可用：生成严格 StateSnapshot
    - 否则：生成最小可审计 dict（仍然冻结为只读对象由外层 store/codec 负责）
    """
    event_id, ts = _event_id_ts(raw_event)
    snapshot_markets = {
        sym: market_from_rust(state) if isinstance(state, _RustMarketState) else state
        for sym, state in markets.items()
    }
    snapshot_account = account_from_rust(account) if isinstance(account, _RustAccountState) else account
    snapshot_positions = {
        sym: position_from_rust(state) if isinstance(state, _RustPositionState) else state
        for sym, state in positions.items()
    }

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
    bundle = dict(store.export_state())
    markets = dict(bundle["markets"])
    positions = dict(bundle["positions"])
    account = bundle["account"]
    portfolio = portfolio_from_rust(bundle["portfolio"])
    risk = risk_from_rust(bundle["risk"])
    return (
        markets,
        account,
        positions,
        portfolio,
        risk,
        bundle.get("event_index"),
        bundle.get("last_event_id"),
        bundle.get("last_ts"),
    )


# ============================================================
# Pipeline (frozen v1.0)
# ============================================================

@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """
    pipeline 配置（只包含运行制度，不包含策略参数）
    """
    build_snapshot_on_change_only: bool = True
    fail_on_missing_symbol: bool = False  # 若事实事件缺 symbol 是否视为错误
    default_symbol_fallback: bool = True  # 允许用 symbol_default 兜底


class StatePipeline:
    """
    StatePipeline —— "state 的唯一写通道"（冻结版 v1.0）

    目标：
    - 把 core.py 中"事实归一化 + reducer 链 + snapshot"抽离为独立制度模块
    - 让 replay 与 live 共享同一 pipeline

    冻结铁律：
    - pipeline 不做调度
    - pipeline 不做决策
    - pipeline 不做 IO
    - pipeline 只在"事实事件"驱动时推进 event_index
    """

    def __init__(
        self,
        *,
        market_reducer: Optional[Reducer[MarketState]] = None,
        account_reducer: Optional[Reducer[AccountState]] = None,
        position_reducer: Optional[Reducer[PositionState]] = None,
        config: Optional[PipelineConfig] = None,
        store: Optional[Any] = None,
    ) -> None:
        self._mr = market_reducer or RustMarketReducerAdapter()
        self._ar = account_reducer or RustAccountReducerAdapter()
        self._pr = position_reducer or RustPositionReducerAdapter()
        self._cfg = config or PipelineConfig()
        self._store = store

    @property
    def config(self) -> PipelineConfig:
        return self._cfg

    def _derive_state(
        self,
        *,
        raw_event: Any,
        inp: PipelineInput,
        markets: Mapping[str, Any],
        account: Any,
        positions: Mapping[str, Any],
        recompute: bool,
    ) -> Tuple[Any, Any]:
        if not recompute and inp.portfolio is not None and inp.risk is not None:
            return inp.portfolio, inp.risk
        return derive_portfolio_and_risk(
            event=raw_event,
            symbol_default=inp.symbol_default,
            markets=markets,
            account=account,
            positions=positions,
            prior_portfolio=inp.portfolio,
            prior_risk=inp.risk,
        )

    def apply(self, inp: PipelineInput) -> PipelineOutput:
        """
        应用 pipeline：event → facts → reducers → snapshot

        返回：
        - advanced=True 表示 event_index 推进了 1（仅当 facts 非空）
        - snapshot：可选（按 config 控制）
        """
        raw_event = inp.event

        # Store path (default): state on Rust heap, no per-event Python↔Rust conversion
        if self._store is not None:
            return self._apply_store_path(inp, raw_event)

        # Slow path: only reached when custom Python reducers are injected (tests).
        facts = normalize_to_facts(raw_event)

        # 非事实事件：不改变任何 state，也不推进 event_index
        if not facts:
            return PipelineOutput(
                markets=inp.markets,
                account=inp.account,
                positions=inp.positions,
                portfolio=inp.portfolio,
                risk=inp.risk,
                features=inp.features,
                event_index=int(inp.event_index),
                last_event_id=inp.last_event_id if hasattr(inp, "last_event_id") else None,  # type: ignore
                last_ts=inp.last_ts if hasattr(inp, "last_ts") else None,  # type: ignore
                snapshot=None,
                advanced=False,
            )

        # 复制可变容器（MarketState/PositionState 本身 frozen）
        markets: Dict[str, MarketState] = dict(inp.markets)
        account = inp.account
        positions: Dict[str, PositionState] = dict(inp.positions)

        any_changed = False

        for fev in facts:
            # symbol 归一化
            sym = getattr(fev, "symbol", None)
            if not isinstance(sym, str) or not sym:
                if self._cfg.fail_on_missing_symbol:
                    raise FactNormalizationError("事实事件缺少 symbol")
                if self._cfg.default_symbol_fallback:
                    sym = inp.symbol_default
                else:
                    sym = inp.symbol_default

            # market reducer（按 symbol 路由）
            if sym not in markets:
                markets[sym] = MarketState.empty(symbol=sym)
            m_res = self._mr.reduce(markets[sym], fev)
            markets[sym] = m_res.state
            any_changed = any_changed or bool(m_res.changed)

            # account reducer（只有 FILL/FUNDING 会改变）
            a_res = self._ar.reduce(account, fev)
            account = a_res.state
            any_changed = any_changed or bool(a_res.changed)

            # position reducer（按 symbol）
            if sym not in positions:
                positions[sym] = PositionState.empty(symbol=sym)
            p_res = self._pr.reduce(positions[sym], fev)
            positions[sym] = p_res.state
            any_changed = any_changed or bool(p_res.changed)

        # 推进 event_index（只对事实事件推进）
        next_index = int(inp.event_index) + 1
        event_id, ts = _event_id_ts(raw_event)
        portfolio, risk = self._derive_state(
            raw_event=raw_event,
            inp=inp,
            markets=markets,
            account=account,
            positions=positions,
            recompute=any_changed or inp.portfolio is None or inp.risk is None,
        )

        # snapshot 生成策略
        snapshot: Optional[Any] = None
        if (not self._cfg.build_snapshot_on_change_only) or any_changed:
            snapshot = _build_snapshot(
                raw_event=raw_event,
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
            event_index=next_index,
            last_event_id=event_id,
            last_ts=ts,
            snapshot=snapshot,
            advanced=True,
        )

    def _apply_store_path(self, inp: PipelineInput, raw_event: Any) -> PipelineOutput:
        """
        Store path: state lives on Rust heap via RustStateStore.
        No Python↔Rust Decimal conversion per event.
        Only exports state on demand (decision cycles / snapshot).
        """
        store = self._store
        result = store.process_event(raw_event, inp.symbol_default)

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

        # Export when snapshot is needed or when coordinator has not yet received
        # the derived state carried by the Rust store.
        need_export = (
            (not self._cfg.build_snapshot_on_change_only)
            or result.changed
            or inp.portfolio is None
            or inp.risk is None
        )

        if need_export:
            (
                markets,
                account,
                positions,
                portfolio,
                risk,
                bundled_event_index,
                bundled_last_event_id,
                bundled_last_ts,
            ) = _export_store_state(store)

            snapshot: Optional[Any] = _build_snapshot(
                raw_event=raw_event,
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

        # Advanced but no snapshot needed — return lightweight output
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

