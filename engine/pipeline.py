# engine/pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

# -------------------------
# state 侧（你的 state 层）
# -------------------------
from state.market import MarketState
from state.account import AccountState
from state.position import PositionState

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

    注意：pipeline 并不依赖 engine/core 的具体实现，
    只关心：
    - 当前 state 组合
    - 当前事件
    - 当前 event_index（用于快照标定）
    """
    event: Any
    event_index: int
    symbol_default: str

    markets: Mapping[str, MarketState]
    account: AccountState
    positions: Mapping[str, PositionState]

    # 这两项属于派生事实（可为空），pipeline 不做决策，只负责传递/快照
    portfolio: Any = None
    risk: Any = None
    features: Optional[Mapping[str, Any]] = None

    @property
    def market(self) -> MarketState:
        """Backward compat: 返回 symbol_default 对应的 MarketState。"""
        if self.symbol_default in self.markets:
            return self.markets[self.symbol_default]
        return next(iter(self.markets.values()))


@dataclass(frozen=True, slots=True)
class PipelineOutput:
    """
    pipeline 输出（只读）

    - advanced: 是否推进了 event_index（仅当"事实事件"驱动了 state 更新时才推进）
    """
    markets: Mapping[str, MarketState]
    account: AccountState
    positions: Mapping[str, PositionState]

    portfolio: Any
    risk: Any
    features: Optional[Mapping[str, Any]]

    event_index: int
    last_event_id: Optional[str]
    last_ts: Optional[Any]

    snapshot: Optional[Any]
    advanced: bool

    @property
    def market(self) -> MarketState:
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
    et = getattr(event, "event_type", None)
    if et is not None:
        try:
            et_val = et.value if hasattr(et, "value") else et
            et_u = str(et_val).upper()
            if "MARKET" in et_u:
                return "MARKET"
            if "FILL" in et_u:
                return "FILL"
            if "FUNDING" in et_u:
                return "FUNDING"
            if "ORDER" in et_u:
                return "ORDER"
            if "INTENT" in et_u:
                return "INTENT"
            if "SIGNAL" in et_u:
                return "SIGNAL"
            if "RISK" in et_u:
                return "RISK"
            if "CONTROL" in et_u:
                return "CONTROL"
        except Exception:
            pass

    name = getattr(event, "EVENT_TYPE", None)
    if isinstance(name, str) and name:
        n = name.lower()
        if "market" in n:
            return "MARKET"
        if "fill" in n:
            return "FILL"
        if "funding" in n:
            return "FUNDING"
        if "order" in n:
            return "ORDER"
        if "intent" in n:
            return "INTENT"
        if "signal" in n:
            return "SIGNAL"
        if "risk" in n:
            return "RISK"
        if "control" in n:
            return "CONTROL"

    return "UNKNOWN"



def _to_float(x: Any, *, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        import logging
        logging.getLogger(__name__).warning("_to_float: conversion failed for %r, using default=%s", x, default)
        return default


def normalize_to_facts(event: Any) -> List[Any]:
    """
    event -> reducers 可消费的 facts（非空才会推进 event_index）
    """
    kind = _detect_kind(event)
    header = getattr(event, "header", None)

    out: List[Any] = []

    # -------------------------
    # MARKET（行情）：透传给 MarketReducer 更新 MarketState
    # -------------------------
    if kind == "MARKET":
        out.append(
            SimpleNamespace(
                event_type="market",
                header=header,
                symbol=getattr(event, "symbol", None),
                open=getattr(event, "open", None),
                high=getattr(event, "high", None),
                low=getattr(event, "low", None),
                close=getattr(event, "close", None),
                volume=getattr(event, "volume", None),
                ts=getattr(event, "ts", None),
            )
        )
        return out

    # -------------------------
    # FUNDING（资金费率结算）：透传给 AccountReducer 更新余额
    # -------------------------
    if kind == "FUNDING":
        out.append(
            SimpleNamespace(
                event_type="funding",
                header=header,
                symbol=getattr(event, "symbol", None),
                funding_rate=getattr(event, "funding_rate", None),
                mark_price=getattr(event, "mark_price", None),
                ts=getattr(event, "ts", None),
            )
        )
        return out

    # -------------------------
    # ORDER（订单回报/状态）：作为事实事件推进 event_index（reducers 可忽略）
    # -------------------------
    if kind == "ORDER":
        out.append(
            SimpleNamespace(
                event_type="ORDER_UPDATE",
                header=header,
                symbol=getattr(event, "symbol", None),
                venue=getattr(event, "venue", None),
                order_id=getattr(event, "order_id", None),
                client_order_id=getattr(event, "client_order_id", None),
                status=getattr(event, "status", None),
                side=getattr(event, "side", None),
                order_type=getattr(event, "order_type", None),
                tif=getattr(event, "tif", None),
                qty=getattr(event, "qty", None),
                price=getattr(event, "price", None),
                filled_qty=getattr(event, "filled_qty", None),
                avg_price=getattr(event, "avg_price", None),
                order_key=getattr(event, "order_key", None),
                payload_digest=getattr(event, "payload_digest", None),
            )
        )
        return out

    # -------------------------
    # FILL（成交）：必须带 side，qty/quantity 保持绝对值，side 决定方向
    # -------------------------
    if kind == "FILL":
        raw_qty = getattr(event, "qty", None)
        if raw_qty is None:
            raw_qty = getattr(event, "quantity", None)

        side = getattr(event, "side", None)
        if side is None:
            raise FactNormalizationError("FILL 事实事件缺少 side")

        side_s = getattr(side, "value", None) if hasattr(side, "value") else str(side)
        side_s = str(side_s).strip().lower()
        if side_s in ("buy", "long"):
            side_norm = "buy"
        elif side_s in ("sell", "short"):
            side_norm = "sell"
        else:
            raise FactNormalizationError(f"不支持的 fill side: {side_s!r}")

        qty = abs(_to_float(raw_qty, default=0.0))

        out.append(
            SimpleNamespace(
                event_type="FILL",
                header=header,
                symbol=getattr(event, "symbol", None),
                side=side_norm,
                qty=qty,
                quantity=qty,
                price=_to_float(getattr(event, "price", None), default=0.0),
                fee=_to_float(getattr(event, "fee", None), default=0.0),
                realized_pnl=_to_float(getattr(event, "realized_pnl", None), default=0.0),
                margin_change=_to_float(getattr(event, "margin_change", None), default=0.0),
            )
        )
        return out

    # 其他事件：默认不驱动 state
    return out



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
        market_reducer: Optional[MarketReducer] = None,
        account_reducer: Optional[AccountReducer] = None,
        position_reducer: Optional[PositionReducer] = None,
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self._mr = market_reducer or MarketReducer()
        self._ar = account_reducer or AccountReducer()
        self._pr = position_reducer or PositionReducer()
        self._cfg = config or PipelineConfig()

    @property
    def config(self) -> PipelineConfig:
        return self._cfg

    def apply(self, inp: PipelineInput) -> PipelineOutput:
        """
        应用 pipeline：event → facts → reducers → snapshot

        返回：
        - advanced=True 表示 event_index 推进了 1（仅当 facts 非空）
        - snapshot：可选（按 config 控制）
        """
        raw_event = inp.event
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

        # snapshot 生成策略
        snapshot: Optional[Any] = None
        if (not self._cfg.build_snapshot_on_change_only) or any_changed:
            snapshot = _build_snapshot(
                raw_event=raw_event,
                event_index=next_index,
                markets=markets,
                account=account,
                positions=positions,
                portfolio=inp.portfolio,
                risk=inp.risk,
                features=inp.features,
            )

        return PipelineOutput(
            markets=markets,
            account=account,
            positions=positions,
            portfolio=inp.portfolio,
            risk=inp.risk,
            features=inp.features,
            event_index=next_index,
            last_event_id=event_id,
            last_ts=ts,
            snapshot=snapshot,
            advanced=True,
        )
