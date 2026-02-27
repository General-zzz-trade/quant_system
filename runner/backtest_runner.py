from __future__ import annotations

import json
from math import sqrt
from statistics import median, stdev

import argparse
import csv
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from engine.coordinator import CoordinatorConfig, EngineCoordinator
from engine.decision_bridge import DecisionBridge
from engine.execution_bridge import ExecutionBridge
from event.header import EventHeader
from event.types import EventType, MarketEvent, IntentEvent, OrderEvent
from state.position import PositionState


# ============================================================
# CSV IO
# ============================================================

_TS_COLS: Tuple[str, ...] = (
    "ts",
    "timestamp",
    "time",
    "datetime",
    "date",
    "open_time",
    "open time",
)

_O_COLS: Tuple[str, ...] = ("open", "o")
_H_COLS: Tuple[str, ...] = ("high", "h")
_L_COLS: Tuple[str, ...] = ("low", "l")
_C_COLS: Tuple[str, ...] = ("close", "c", "price")
_V_COLS: Tuple[str, ...] = ("volume", "vol", "v")


def _to_key(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _parse_ts(raw: Any) -> datetime:
    if raw is None:
        raise ValueError("missing timestamp")

    s = str(raw).strip()
    if not s:
        raise ValueError("empty timestamp")

    if s.isdigit():
        n = int(s)
        if n >= 1_000_000_000_000:
            return datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
        return datetime.fromtimestamp(float(n), tz=timezone.utc)

    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    raise ValueError(f"unsupported timestamp format: {raw!r}")


def _dec(x: Any) -> Decimal:
    if x is None:
        raise ValueError("missing numeric")
    s = str(x).strip()
    if not s:
        raise ValueError("empty numeric")
    return Decimal(s)


@dataclass(frozen=True, slots=True)
class OhlcvBar:
    ts: datetime
    o: Decimal
    h: Decimal
    l: Decimal
    c: Decimal
    v: Optional[Decimal]


def iter_ohlcv_csv(path: Path) -> Iterator[OhlcvBar]:
    header_like = {_to_key(x) for x in _TS_COLS} | {_to_key("open_time")} | {_to_key("open time")} | {_to_key("ts")} | {_to_key("timestamp")}

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")

        cols = {_to_key(c): c for c in reader.fieldnames}

        def pick(candidates: Sequence[str]) -> str:
            for c in candidates:
                k = _to_key(c)
                if k in cols:
                    return cols[k]
            raise ValueError(f"missing required column, candidates={candidates}")

        ts_col = pick(_TS_COLS)
        o_col = pick(_O_COLS)
        h_col = pick(_H_COLS)
        l_col = pick(_L_COLS)
        c_col = pick(_C_COLS)

        v_col: Optional[str] = None
        for c in _V_COLS:
            k = _to_key(c)
            if k in cols:
                v_col = cols[k]
                break

        for idx, row in enumerate(reader, start=1):
            raw_ts = row.get(ts_col)
            if raw_ts is None:
                continue

            s = str(raw_ts).strip()
            if not s:
                continue

            # 跳过被拼进来的表头行，比如 open_time
            if _to_key(s) in header_like:
                continue

            try:
                ts = _parse_ts(raw_ts)
            except ValueError as e:
                # 再兜一层，遇到表头文本直接跳过
                if _to_key(s) in header_like:
                    continue
                raise ValueError(f"bad timestamp at row {idx}: {raw_ts!r}") from e

            o = _dec(row.get(o_col))
            h = _dec(row.get(h_col))
            l = _dec(row.get(l_col))
            c = _dec(row.get(c_col))
            v = _dec(row.get(v_col)) if v_col and row.get(v_col) not in (None, "") else None
            yield OhlcvBar(ts=ts, o=o, h=h, l=l, c=c, v=v)


# ============================================================
# Backtest execution adapter
# ============================================================


def _sign(side: str) -> int:
    s = str(side).strip().lower()
    if s in ("buy", "long"):
        return 1
    if s in ("sell", "short"):
        return -1
    raise ValueError(f"unsupported side: {side!r}")


def _make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class BacktestExecutionAdapter:
    """Minimal paper execution adapter.

    Takes OrderEvent, produces fill events immediately at a chosen price.
    Maintains its own lightweight position state to compute realized PnL.
    """

    def __init__(
        self,
        *,
        price_source: Callable[[str], Optional[Decimal]],
        ts_source: Callable[[], Optional[datetime]],
        fee_bps: Decimal = Decimal("0"),
        slippage_bps: Decimal = Decimal("0"),
        source: str = "paper",
        on_fill: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._price_source = price_source
        self._ts_source = ts_source
        self._fee_bps = Decimal(str(fee_bps))
        self._slippage_bps = Decimal(str(slippage_bps))
        self._source = source
        self._on_fill = on_fill

        self._pos_qty: Dict[str, Decimal] = {}
        self._avg_px: Dict[str, Optional[Decimal]] = {}

    def send_order(self, order_event: Any) -> List[Any]:
        sym = str(getattr(order_event, "symbol")).upper()
        side = str(getattr(order_event, "side"))
        qty = Decimal(str(getattr(order_event, "qty")))
        if qty <= 0:
            return []

        px: Optional[Decimal]
        raw_price = getattr(order_event, "price", None)
        if raw_price is not None:
            px = Decimal(str(raw_price))
        else:
            px = self._price_source(sym)

        if px is None:
            raise RuntimeError(f"no price available for {sym}")

        # Apply slippage: buy fills at a higher price, sell fills at a lower price
        if self._slippage_bps > 0:
            slippage_mult = self._slippage_bps / Decimal("10000")
            if _sign(side) > 0:  # buy
                px = px * (Decimal("1") + slippage_mult)
            else:  # sell
                px = px * (Decimal("1") - slippage_mult)

        signed = qty * Decimal(_sign(side))
        prev_qty = self._pos_qty.get(sym, Decimal("0"))
        prev_avg = self._avg_px.get(sym, None)

        fee = (px * qty) * (self._fee_bps / Decimal("10000"))
        realized = Decimal("0")

        if prev_qty != 0 and prev_avg is not None and (prev_qty > 0) != (signed > 0):
            closed = min(abs(prev_qty), abs(signed))
            sign_prev = Decimal("1") if prev_qty > 0 else Decimal("-1")
            realized = (px - prev_avg) * closed * sign_prev

        new_qty = prev_qty + signed
        new_avg: Optional[Decimal]

        if new_qty == 0:
            new_avg = None
        else:
            if prev_qty == 0 or (prev_qty > 0) == (signed > 0):
                base_qty = abs(prev_qty)
                add_qty = abs(signed)
                base_avg = prev_avg if prev_avg is not None else px
                new_avg = (base_avg * base_qty + px * add_qty) / (base_qty + add_qty)
            else:
                if (prev_qty > 0 and new_qty > 0) or (prev_qty < 0 and new_qty < 0):
                    new_avg = prev_avg
                else:
                    new_avg = px

        self._pos_qty[sym] = new_qty
        self._avg_px[sym] = new_avg

        parent = getattr(order_event, "header", None)
        if isinstance(parent, EventHeader):
            h = EventHeader.from_parent(parent=parent, event_type=EventType.FILL, version=1, source=self._source)
        else:
            h = EventHeader.new_root(event_type=EventType.FILL, version=1, source=self._source)

        fill = SimpleNamespace(
            header=h,
            event_type=EventType.FILL,
            ts=self._ts_source(),
            symbol=sym,
            side=side,
            qty=qty,
            price=px,
            fee=fee,
            realized_pnl=realized,
            cash_delta=0.0,
            margin_change=0.0,
        )

        if self._on_fill is not None:
            self._on_fill(fill)

        return [fill]


# ============================================================
# Decision module
# ============================================================


class MovingAverageCrossModule:
    def __init__(self, *, symbol: str, window: int, order_qty: Decimal, origin: str = "ma_cross") -> None:
        self.symbol = symbol.upper()
        self.window = int(window)
        self.order_qty = Decimal(str(order_qty))
        self.origin = origin
        self._closes: List[Decimal] = []

    def decide(self, snapshot: Any) -> Iterable[Any]:
        market, positions, event_id = _snapshot_views(snapshot)
        close = getattr(market, "close", None) or getattr(market, "last_price", None)
        if close is None:
            return ()

        close_d = Decimal(str(close))
        self._closes.append(close_d)
        if len(self._closes) > self.window:
            self._closes.pop(0)
        if len(self._closes) < self.window:
            return ()

        ma = sum(self._closes) / Decimal(str(self.window))

        pos = positions.get(self.symbol) or PositionState.empty(self.symbol)
        qty = getattr(pos, "qty", Decimal("0"))

        want_long = close_d > ma

        events: List[Any] = []
        if qty == 0 and want_long:
            events.extend(self._open_long(event_id=event_id))
        elif qty > 0 and (not want_long):
            events.extend(self._close_long(qty=qty, event_id=event_id))

        return events

    def _open_long(self, *, event_id: Optional[str]) -> Sequence[Any]:
        intent_id = _make_id("intent")
        order_id = _make_id("order")

        intent_h = EventHeader.new_root(
            event_type=EventType.INTENT,
            version=1,
            source=f"decision:{self.origin}",
            correlation_id=str(event_id) if event_id else None,
        )
        order_h = EventHeader.from_parent(
            parent=intent_h,
            event_type=EventType.ORDER,
            version=1,
            source=f"decision:{self.origin}",
        )

        return (
            IntentEvent(
                header=intent_h,
                intent_id=intent_id,
                symbol=self.symbol,
                side="buy",
                target_qty=self.order_qty,
                reason_code="ma_cross_long",
                origin=self.origin,
            ),
            OrderEvent(
                header=order_h,
                order_id=order_id,
                intent_id=intent_id,
                symbol=self.symbol,
                side="buy",
                qty=self.order_qty,
                price=None,
            ),
        )

    def _close_long(self, *, qty: Decimal, event_id: Optional[str]) -> Sequence[Any]:
        intent_id = _make_id("intent")
        order_id = _make_id("order")

        intent_h = EventHeader.new_root(
            event_type=EventType.INTENT,
            version=1,
            source=f"decision:{self.origin}",
            correlation_id=str(event_id) if event_id else None,
        )
        order_h = EventHeader.from_parent(
            parent=intent_h,
            event_type=EventType.ORDER,
            version=1,
            source=f"decision:{self.origin}",
        )

        q = abs(qty)
        return (
            IntentEvent(
                header=intent_h,
                intent_id=intent_id,
                symbol=self.symbol,
                side="sell",
                target_qty=q,
                reason_code="ma_cross_exit",
                origin=self.origin,
            ),
            OrderEvent(
                header=order_h,
                order_id=order_id,
                intent_id=intent_id,
                symbol=self.symbol,
                side="sell",
                qty=q,
                price=None,
            ),
        )


def _snapshot_views(snapshot: Any) -> Tuple[Any, Mapping[str, Any], Optional[str]]:
    if hasattr(snapshot, "market") and hasattr(snapshot, "positions"):
        market = getattr(snapshot, "market")
        positions = getattr(snapshot, "positions")
        event_id = getattr(snapshot, "event_id", None)
        return market, positions, event_id

    if isinstance(snapshot, dict):
        market = snapshot.get("market")
        if market is None:
            # multi-symbol: pick first market from "markets" dict
            markets = snapshot.get("markets") or {}
            market = next(iter(markets.values()), None) if markets else None
        positions = snapshot.get("positions") or {}
        event_id = snapshot.get("event_id")
        if market is None:
            raise RuntimeError("snapshot missing market/markets")
        return market, positions, event_id

    raise RuntimeError(f"unsupported snapshot type: {type(snapshot).__name__}")


# ============================================================
# Metrics
# ============================================================


@dataclass(frozen=True, slots=True)
class EquityPoint:
    ts: datetime
    close: Decimal
    position_qty: Decimal
    avg_price: Optional[Decimal]
    balance: Decimal
    realized: Decimal
    unrealized: Decimal
    equity: Decimal


def _max_drawdown(equity: Sequence[Decimal]) -> Decimal:
    peak = None
    mdd = Decimal("0")
    for x in equity:
        if peak is None or x > peak:
            peak = x
        if peak is None or peak == 0:
            continue
        dd = (peak - x) / peak
        if dd > mdd:
            mdd = dd
    return mdd

def _parse_fill_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    if s.isdigit():
        n = int(s)
        # ns
        if n >= 1_000_000_000_000_000_000:
            return datetime.fromtimestamp(n / 1_000_000_000.0, tz=timezone.utc)
        # us
        if n >= 1_000_000_000_000_000:
            return datetime.fromtimestamp(n / 1_000_000.0, tz=timezone.utc)
        # ms
        if n >= 1_000_000_000_000:
            return datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
        # s
        if n >= 1_000_000_000:
            return datetime.fromtimestamp(float(n), tz=timezone.utc)

    # ISO
    try:
        t = s
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _safe_dec(x: Any, default: str = "0") -> Decimal:
    try:
        if x is None:
            return Decimal(default)
        s = str(x).strip()
        if not s:
            return Decimal(default)
        return Decimal(s)
    except Exception:
        return Decimal(default)


def _build_trades_from_fills(fills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def _key(r: Dict[str, Any]):
        t = _parse_fill_ts(r.get("ts"))
        return t or datetime.min.replace(tzinfo=timezone.utc)

    rows = sorted(fills, key=_key)

    trades: List[Dict[str, Any]] = []
    state: Dict[str, Dict[str, Any]] = {}

    def _pos_sign(q: Decimal) -> int:
        return 1 if q > 0 else (-1 if q < 0 else 0)

    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        if not sym:
            continue

        ts = _parse_fill_ts(r.get("ts"))
        side = str(r.get("side") or "").lower()
        qty = _safe_dec(r.get("qty"))
        px = _safe_dec(r.get("price"))
        fee = _safe_dec(r.get("fee"))
        if qty <= 0 or px <= 0 or side not in ("buy", "sell"):
            continue

        signed = qty if side == "buy" else -qty

        st = state.setdefault(
            sym,
            {
                "pos_qty": Decimal("0"),
                "avg_px": None,
                "open": None,  # {"trade_id","side","entry_ts","entry_price","qty","fees_open"}
            },
        )

        pos_qty: Decimal = st["pos_qty"]
        avg_px: Optional[Decimal] = st["avg_px"]
        open_tr: Optional[Dict[str, Any]] = st["open"]

        prev_sign = _pos_sign(pos_qty)
        signed_sign = _pos_sign(signed)

        # 开仓
        if pos_qty == 0:
            st["pos_qty"] = signed
            st["avg_px"] = px
            st["open"] = {
                "trade_id": _make_id("trade"),
                "side": "long" if signed > 0 else "short",
                "entry_ts": ts.isoformat() if ts else "",
                "entry_price": px,
                "qty": abs(signed),
                "fees_open": fee,
            }
            continue

        # 同向加仓
        if prev_sign != 0 and prev_sign == signed_sign:
            new_qty = pos_qty + signed
            base_qty = abs(pos_qty)
            add_qty = abs(signed)
            base_avg = avg_px if avg_px is not None else px
            new_avg = (base_avg * base_qty + px * add_qty) / (base_qty + add_qty)

            st["pos_qty"] = new_qty
            st["avg_px"] = new_avg

            if open_tr is None:
                open_tr = {
                    "trade_id": _make_id("trade"),
                    "side": "long" if new_qty > 0 else "short",
                    "entry_ts": ts.isoformat() if ts else "",
                    "entry_price": base_avg,
                    "qty": abs(new_qty),
                    "fees_open": fee,
                }
                st["open"] = open_tr
            else:
                open_tr["qty"] = abs(new_qty)
                open_tr["fees_open"] = _safe_dec(open_tr.get("fees_open")) + fee
            continue

        # 反向成交，平仓或反手
        if avg_px is None:
            avg_px = px

        abs_before = abs(pos_qty)
        abs_signed = abs(signed)
        closed_qty = min(abs_before, abs_signed)

        sign_prev = Decimal("1") if pos_qty > 0 else Decimal("-1")
        gross_pnl = (px - avg_px) * closed_qty * sign_prev

        if open_tr is None:
            open_tr = {
                "trade_id": _make_id("trade"),
                "side": "long" if pos_qty > 0 else "short",
                "entry_ts": "",
                "entry_price": avg_px,
                "qty": abs_before,
                "fees_open": Decimal("0"),
            }

        # 平仓与开仓共享同一笔成交的 fee，按成交数量比例拆分
        fee_close = fee
        fee_open_part = Decimal("0")
        if abs_signed > 0:
            close_ratio = closed_qty / abs_signed
            fee_close = fee * close_ratio
            fee_open_part = fee - fee_close

        open_fee_total = _safe_dec(open_tr.get("fees_open"))
        alloc_open_fee = Decimal("0")
        if abs_before > 0:
            alloc_open_fee = open_fee_total * (closed_qty / abs_before)

        trade_fee = alloc_open_fee + fee_close
        net_pnl = gross_pnl - trade_fee

        entry_px = _safe_dec(open_tr.get("entry_price"))
        entry_ts = open_tr.get("entry_ts") or ""
        exit_ts = ts.isoformat() if ts else ""

        dur = ""
        if entry_ts and exit_ts:
            try:
                t0 = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(exit_ts.replace("Z", "+00:00"))
                if t0.tzinfo is None:
                    t0 = t0.replace(tzinfo=timezone.utc)
                if t1.tzinfo is None:
                    t1 = t1.replace(tzinfo=timezone.utc)
                dur = str(int((t1 - t0).total_seconds()))
            except Exception:
                dur = ""

        ret = Decimal("0")
        denom = entry_px * closed_qty
        if denom != 0:
            ret = gross_pnl / denom

        trades.append(
            {
                "trade_id": open_tr.get("trade_id") or _make_id("trade"),
                "symbol": sym,
                "side": open_tr.get("side") or ("long" if pos_qty > 0 else "short"),
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "qty": str(closed_qty),
                "entry_price": str(entry_px),
                "exit_price": str(px),
                "gross_pnl": str(gross_pnl),
                "fees": str(trade_fee),
                "net_pnl": str(net_pnl),
                "return": str(ret),
                "duration_sec": dur,
            }
        )

        # 更新剩余仓位与 open 记录
        new_qty = pos_qty + signed

        # 完全平掉
        if new_qty == 0:
            st["pos_qty"] = Decimal("0")
            st["avg_px"] = None
            st["open"] = None
            continue

        # 仍然保留原方向，代表部分平仓
        if (pos_qty > 0 and new_qty > 0) or (pos_qty < 0 and new_qty < 0):
            st["pos_qty"] = new_qty
            st["avg_px"] = avg_px  # 平仓不改变剩余仓位的 avg_px

            remain_qty = abs_before - closed_qty
            remain_open_fee = open_fee_total - alloc_open_fee
            open_tr["qty"] = remain_qty
            open_tr["fees_open"] = remain_open_fee
            st["open"] = open_tr
            continue

        # 反手开新仓
        st["pos_qty"] = new_qty
        st["avg_px"] = px
        st["open"] = {
            "trade_id": _make_id("trade"),
            "side": "long" if new_qty > 0 else "short",
            "entry_ts": ts.isoformat() if ts else "",
            "entry_price": px,
            "qty": abs(new_qty),
            "fees_open": fee_open_part,
        }

    return trades



def _json_safe(x: Any) -> Any:
    if isinstance(x, Decimal):
        return str(x)
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_json_safe(v) for v in x]
    return x


def _build_summary(
    *,
    equity: List[EquityPoint],
    trades: List[Dict[str, Any]],
    csv_path: Path,
    symbol: str,
) -> Dict[str, Any]:
    if not equity:
        return {
            "symbol": symbol,
            "csv": str(csv_path),
            "bars": 0,
            "start_equity": "0",
            "end_equity": "0",
            "return": "0",
            "max_drawdown": "0",
            "trades": 0,
        }

    start_eq = equity[0].equity
    end_eq = equity[-1].equity
    ret = (end_eq - start_eq) / start_eq if start_eq != 0 else Decimal("0")
    mdd = _max_drawdown([x.equity for x in equity])

    start_ts = equity[0].ts
    end_ts = equity[-1].ts

    # time span
    total_seconds = (end_ts - start_ts).total_seconds()
    days = total_seconds / 86400.0 if total_seconds > 0 else 0.0
    years = total_seconds / (365.0 * 24.0 * 3600.0) if total_seconds > 0 else 0.0

    cagr = ""
    if years > 0 and start_eq > 0:
        try:
            cagr_v = (float(end_eq / start_eq) ** (1.0 / years)) - 1.0
            cagr = str(cagr_v)
        except Exception:
            cagr = ""

    # trade stats
    pnls = [Decimal(t["net_pnl"]) for t in trades if t.get("net_pnl") not in (None, "")]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    gross_profit = sum(wins, Decimal("0"))
    gross_loss = sum(losses, Decimal("0"))

    profit_factor = ""
    if gross_loss != 0:
        profit_factor = str(gross_profit / abs(gross_loss))

    win_rate = ""
    if pnls:
        win_rate = str(Decimal(len(wins)) / Decimal(len(pnls)))

    avg_trade_pnl = ""
    median_trade_pnl = ""
    best_trade_pnl = ""
    worst_trade_pnl = ""
    if pnls:
        avg_trade_pnl = str(sum(pnls, Decimal("0")) / Decimal(len(pnls)))
        median_trade_pnl = str(Decimal(str(median([float(p) for p in pnls]))))
        best_trade_pnl = str(max(pnls))
        worst_trade_pnl = str(min(pnls))

    avg_win = ""
    avg_loss = ""
    win_loss_ratio = ""
    if wins:
        avg_win = str(sum(wins, Decimal("0")) / Decimal(len(wins)))
    if losses:
        avg_loss = str(sum(losses, Decimal("0")) / Decimal(len(losses)))
    if wins and losses:
        try:
            win_loss_ratio = str(abs(Decimal(avg_win)) / abs(Decimal(avg_loss)))
        except Exception:
            win_loss_ratio = ""

    total_fees = sum((_safe_dec(t.get("fees")) for t in trades), Decimal("0"))

    # Sharpe ratio (annualized, using daily equity returns)
    sharpe = ""
    if len(equity) >= 2:
        eq_vals = [float(p.equity) for p in equity]
        daily_rets = [(eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1] for i in range(1, len(eq_vals)) if eq_vals[i - 1] != 0]
        if len(daily_rets) >= 2:
            try:
                mean_r = sum(daily_rets) / len(daily_rets)
                std_r = stdev(daily_rets)
                if std_r > 0:
                    # Annualize: bars_per_year depends on bar frequency
                    # Use actual time span to estimate
                    if years > 0 and len(daily_rets) > 0:
                        bars_per_year = len(daily_rets) / years
                    else:
                        bars_per_year = 252.0  # fallback: daily
                    sharpe = str(round(mean_r / std_r * sqrt(bars_per_year), 4))
            except Exception:
                sharpe = ""

    # Calmar ratio (CAGR / max_drawdown)
    calmar = ""
    if cagr and mdd > 0:
        try:
            calmar = str(round(float(cagr) / float(mdd), 4))
        except Exception:
            calmar = ""

    # duration stats
    durs = []
    for t in trades:
        d = str(t.get("duration_sec") or "").strip()
        if d.isdigit():
            durs.append(int(d))

    avg_duration_sec = ""
    median_duration_sec = ""
    p95_duration_sec = ""
    if durs:
        avg_duration_sec = str(int(sum(durs) / len(durs)))
        median_duration_sec = str(int(median(durs)))
        durs_sorted = sorted(durs)
        k = int(0.95 * (len(durs_sorted) - 1))
        p95_duration_sec = str(durs_sorted[k])

    trades_per_day = ""
    if days > 0:
        trades_per_day = str(float(len(trades)) / days)

    # max consecutive losses
    max_loss_streak = 0
    cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            if cur > max_loss_streak:
                max_loss_streak = cur
        else:
            cur = 0

    return {
        "symbol": symbol,
        "csv": str(csv_path),
        "bars": len(equity),
        "start_ts": start_ts.isoformat(),
        "end_ts": end_ts.isoformat(),
        "start_equity": str(start_eq),
        "end_equity": str(end_eq),
        "return": str(ret),
        "max_drawdown": str(mdd),
        "years": str(years) if years > 0 else "",
        "cagr": cagr,

        "trades": len(trades),
        "trades_per_day": trades_per_day,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_pnl": avg_trade_pnl,
        "median_trade_pnl": median_trade_pnl,
        "best_trade_pnl": best_trade_pnl,
        "worst_trade_pnl": worst_trade_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "max_consecutive_losses": str(max_loss_streak),
        "total_fees": str(total_fees),

        "avg_duration_sec": avg_duration_sec,
        "median_duration_sec": median_duration_sec,
        "p95_duration_sec": p95_duration_sec,

        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
    }

# ============================================================
# Runner
# ============================================================


def run_backtest(
    *,
    csv_path: Path,
    symbol: str,
    starting_balance: Decimal,
    ma_window: int,
    order_qty: Decimal,
    fee_bps: Decimal,
    slippage_bps: Decimal = Decimal("0"),
    out_dir: Optional[Path] = None,
    embargo_bars: int = 1,
    funding_csv: Optional[str] = None,
) -> Tuple[List[EquityPoint], List[Dict[str, Any]]]:
    symbol_u = symbol.upper()

    equity: List[EquityPoint] = []
    fills: List[Dict[str, Any]] = []

    def _on_fill(ev: Any) -> None:
        ts_val = ""
        ev_ts = getattr(ev, "ts", None)
        if isinstance(ev_ts, datetime):
            ts_val = ev_ts.isoformat()
        else:
            ts_ns = getattr(getattr(ev, "header", None), "ts_ns", None)
            ts_val = str(ts_ns) if ts_ns is not None else ""

        fills.append(
            {
                "ts": ts_val,
                "symbol": getattr(ev, "symbol", None),
                "side": getattr(ev, "side", None),
                "qty": getattr(ev, "qty", None),
                "price": getattr(ev, "price", None),
                "fee": getattr(ev, "fee", None),
                "realized_pnl": getattr(ev, "realized_pnl", None),
                "event_id": getattr(getattr(ev, "header", None), "event_id", None),
                "root_event_id": getattr(getattr(ev, "header", None), "root_event_id", None),
            }
        )

    def _on_pipeline(out: Any) -> None:
        m = out.market
        a = out.account
        positions = out.positions
        ts = getattr(m, "last_ts", None)
        close = getattr(m, "close", None) or getattr(m, "last_price", None)
        if ts is None or close is None:
            return

        pos = positions.get(symbol_u) or PositionState.empty(symbol_u)
        qty = getattr(pos, "qty", Decimal("0"))
        avg = getattr(pos, "avg_price", None)

        unreal = Decimal("0")
        if qty != 0 and avg is not None:
            unreal = (Decimal(str(close)) - avg) * qty

        bal = getattr(a, "balance", Decimal("0"))
        realized = getattr(a, "realized_pnl", Decimal("0"))
        eq = bal + unreal

        equity.append(
            EquityPoint(
                ts=ts,
                close=Decimal(str(close)),
                position_qty=qty,
                avg_price=avg,
                balance=bal,
                realized=realized,
                unrealized=unreal,
                equity=eq,
            )
        )

    coordinator = EngineCoordinator(
        cfg=CoordinatorConfig(
            symbol_default=symbol_u,
            currency="USDT",
            starting_balance=float(starting_balance),
            on_pipeline_output=_on_pipeline,
        )
    )

    def _emit(ev: Any) -> None:
        coordinator.emit(ev, actor="backtest")

    def _price(sym: str) -> Optional[Decimal]:
        view = coordinator.get_state_view()
        m = view.get("market")
        if m is None:
            return None
        px = getattr(m, "close", None) or getattr(m, "last_price", None)
        return px

    def _ts() -> Optional[datetime]:
        view = coordinator.get_state_view()
        m = view.get("market")
        return getattr(m, "last_ts", None) if m is not None else None

    from execution.sim.embargo import EmbargoExecutionAdapter

    base_adapter = BacktestExecutionAdapter(
        price_source=_price,
        ts_source=_ts,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        source="paper",
        on_fill=_on_fill,
    )

    embargo_adapter = EmbargoExecutionAdapter(inner=base_adapter, embargo_bars=embargo_bars)
    exec_bridge = ExecutionBridge(adapter=embargo_adapter, dispatcher_emit=_emit)

    decision_module = MovingAverageCrossModule(symbol=symbol_u, window=ma_window, order_qty=order_qty)
    decision_bridge = DecisionBridge(dispatcher_emit=_emit, modules=[decision_module])

    coordinator.attach_execution_bridge(exec_bridge)
    coordinator.attach_decision_bridge(decision_bridge)

    # Build funding schedule if funding CSV provided
    funding_schedule: Dict[datetime, Any] = {}
    if funding_csv is not None:
        from data.loaders.funding_rate import load_funding_csv, funding_schedule_for_bars
        funding_records = load_funding_csv(funding_csv, symbol=symbol_u)
        bar_list = list(iter_ohlcv_csv(csv_path))
        bar_timestamps = [b.ts for b in bar_list]
        funding_schedule = funding_schedule_for_bars(bar_timestamps, funding_records)
    else:
        bar_list = list(iter_ohlcv_csv(csv_path))

    coordinator.start()

    for i, bar in enumerate(bar_list):
        # Execute embargoed orders from previous bars
        if embargo_bars > 0:
            embargo_adapter.on_bar(i)
        # Set current bar index for any new orders generated this bar
        embargo_adapter.set_bar(i)

        h = EventHeader.new_root(event_type=EventType.MARKET, version=MarketEvent.VERSION, source="csv")
        ev = MarketEvent(
            header=h,
            ts=bar.ts,
            symbol=symbol_u,
            open=bar.o,
            high=bar.h,
            low=bar.l,
            close=bar.c,
            volume=bar.v if bar.v is not None else Decimal("0"),
        )
        coordinator.emit(ev, actor="replay")

        # Funding rate settlement
        if bar.ts in funding_schedule:
            fr = funding_schedule[bar.ts]
            pos_qty = base_adapter._pos_qty.get(symbol_u, Decimal("0"))
            if pos_qty != 0:
                fh = EventHeader.new_root(event_type=EventType.FILL, version=1, source="funding")
                funding_ev = SimpleNamespace(
                    header=fh,
                    event_type="funding",
                    ts=bar.ts,
                    symbol=symbol_u,
                    funding_rate=fr.funding_rate,
                    mark_price=fr.mark_price,
                    position_qty=pos_qty,
                )
                coordinator.emit(funding_ev, actor="funding")

    # Flush remaining embargoed orders on last bar
    if embargo_bars > 0 and bar_list:
        embargo_adapter.on_bar(len(bar_list))

    coordinator.stop()

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        eq_path = out_dir / "equity_curve.csv"
        with eq_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ts", "close", "qty", "avg_price", "balance", "realized", "unrealized", "equity"])
            for r in equity:
                w.writerow(
                    [
                        r.ts.isoformat(),
                        str(r.close),
                        str(r.position_qty),
                        str(r.avg_price) if r.avg_price is not None else "",
                        str(r.balance),
                        str(r.realized),
                        str(r.unrealized),
                        str(r.equity),
                    ]
                )

        fills_path = out_dir / "fills.csv"
        with fills_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["ts", "symbol", "side", "qty", "price", "fee", "realized_pnl", "event_id", "root_event_id"],
            )
            w.writeheader()
            for row in fills:
                w.writerow(row)

        trades = _build_trades_from_fills(fills)

        trades_path = out_dir / "trades.csv"
        with trades_path.open("w", newline="") as f:
            fieldnames = [
                "trade_id",
                "symbol",
                "side",
                "entry_ts",
                "exit_ts",
                "qty",
                "entry_price",
                "exit_price",
                "gross_pnl",
                "fees",
                "net_pnl",
                "return",
                "duration_sec",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for t in trades:
                w.writerow(t)

        summary = _build_summary(equity=equity, trades=trades, csv_path=csv_path, symbol=symbol_u)
        summary_path = out_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(summary), f, ensure_ascii=False, indent=2)

    return equity, fills


# ============================================================
# Walk-Forward Validation
# ============================================================


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One window of a walk-forward test."""
    window_idx: int
    train_bars: int
    test_bars: int
    test_summary: Dict[str, Any]


def run_walk_forward(
    *,
    csv_path: Path,
    symbol: str,
    starting_balance: Decimal,
    ma_window: int,
    order_qty: Decimal,
    fee_bps: Decimal,
    slippage_bps: Decimal = Decimal("0"),
    train_size: int = 500,
    test_size: int = 100,
    out_dir: Optional[Path] = None,
) -> List[WalkForwardWindow]:
    """Walk-forward validation: split bars into train/test windows.

    Only the test period result is recorded. Train period is used to warm up
    the strategy's state. This prevents look-ahead bias in parameter tuning.

    Returns list of WalkForwardWindow, one per test period.
    """
    all_bars = list(iter_ohlcv_csv(csv_path))
    if len(all_bars) < train_size + test_size:
        raise ValueError(
            f"Not enough bars for walk-forward: need {train_size + test_size}, got {len(all_bars)}"
        )

    results: List[WalkForwardWindow] = []
    window_idx = 0
    start = 0

    while start + train_size + test_size <= len(all_bars):
        test_start = start + train_size
        test_end = test_start + test_size
        test_window_bars = all_bars[start:test_end]  # includes train for warmup
        test_only_bars = all_bars[test_start:test_end]

        # Build a minimal in-memory CSV for this window
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, newline=""
        ) as tmp:
            writer = csv.writer(tmp)
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])
            for bar in test_window_bars:
                writer.writerow([
                    bar.ts.isoformat(),
                    str(bar.o),
                    str(bar.h),
                    str(bar.l),
                    str(bar.c),
                    str(bar.v) if bar.v is not None else "0",
                ])
            tmp_path = tmp.name

        try:
            window_out = (out_dir / f"window_{window_idx:03d}") if out_dir else None
            equity, fills = run_backtest(
                csv_path=Path(tmp_path),
                symbol=symbol,
                starting_balance=starting_balance,
                ma_window=ma_window,
                order_qty=order_qty,
                fee_bps=fee_bps,
                slippage_bps=slippage_bps,
                out_dir=window_out,
            )

            # Only keep equity/fills from the test period (skip warmup bars)
            test_equity = equity[train_size:] if len(equity) > train_size else equity
            test_fills_raw = fills  # fills are already from the whole window; approximate

            trades = _build_trades_from_fills(test_fills_raw)
            summary = _build_summary(
                equity=test_equity,
                trades=trades,
                csv_path=csv_path,
                symbol=symbol,
            )
            summary["window_idx"] = window_idx
            summary["train_bars"] = train_size
            summary["test_bars"] = len(test_only_bars)
            summary["test_start_ts"] = test_only_bars[0].ts.isoformat() if test_only_bars else ""
            summary["test_end_ts"] = test_only_bars[-1].ts.isoformat() if test_only_bars else ""

            results.append(WalkForwardWindow(
                window_idx=window_idx,
                train_bars=train_size,
                test_bars=len(test_only_bars),
                test_summary=summary,
            ))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        start += test_size
        window_idx += 1

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        wf_path = out_dir / "walk_forward_summary.json"
        with wf_path.open("w", encoding="utf-8") as f:
            json.dump(
                [_json_safe(w.test_summary) for w in results],
                f,
                ensure_ascii=False,
                indent=2,
            )

    return results


# ============================================================
# Default runnable CLI
# ============================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pick_default_csv(root: Path) -> Optional[Path]:
    fixed = root / "data" / "binance" / "ohlcv" / "BTCUSDT_1m_ohlcv.csv"
    if fixed.exists() and fixed.stat().st_size > 0:
        return fixed

    ohlcv_dir = root / "data" / "binance" / "ohlcv"
    if not ohlcv_dir.exists():
        return None

    candidates = [p for p in ohlcv_dir.glob("*_ohlcv.csv") if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _infer_symbol_from_csv_name(csv_path: Path) -> Optional[str]:
    name = csv_path.stem
    if "_1m_" in name:
        return name.split("_")[0]
    if "-1m-" in name:
        return name.split("-")[0]
    if "_" in name:
        return name.split("_")[0]
    return None


def _default_out_dir(root: Path, symbol: str) -> Path:
    return root / "out" / f"{symbol.lower()}_default"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="Path to OHLCV CSV")
    p.add_argument("--symbol", default=None, help="Symbol, e.g. BTCUSDT")
    p.add_argument("--starting-balance", default="10000", help="Starting balance")
    p.add_argument("--ma", type=int, default=20, help="Moving average window")
    p.add_argument("--qty", default="0.01", help="Order quantity")
    p.add_argument("--fee-bps", default="0", help="Fee bps per fill (e.g. 4 = 0.04%%)")
    p.add_argument("--slippage-bps", default="0", help="Slippage bps per fill (e.g. 2 = 0.02%%)")
    p.add_argument("--out", default=None, help="Output directory for csv logs")
    return p


def parse_args(argv: Optional[List[str]] = None):
    args = build_arg_parser().parse_args(argv)
    root = _project_root()

    if args.csv is None:
        picked = _pick_default_csv(root)
        if picked is None:
            print("Missing --csv and no default CSV found.")
            print(f"Expected: {root / 'data' / 'binance' / 'ohlcv' / 'BTCUSDT_1m_ohlcv.csv'}")
            print("Or put any *_ohlcv.csv under: data/binance/ohlcv/")
            raise SystemExit(2)
        args.csv = str(picked)

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = (root / csv_path).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        raise SystemExit(2)
    args.csv = str(csv_path)

    if args.symbol is None:
        args.symbol = _infer_symbol_from_csv_name(csv_path) or "BTCUSDT"

    if args.out is None or str(args.out).strip() == "":
        args.out = str(_default_out_dir(root, args.symbol))
    else:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            args.out = str((root / out_path).resolve())

    return args


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out) if args.out else None

    eq, _ = run_backtest(
        csv_path=csv_path,
        symbol=args.symbol,
        starting_balance=Decimal(str(args.starting_balance)),
        ma_window=int(args.ma),
        order_qty=Decimal(str(args.qty)),
        fee_bps=Decimal(str(args.fee_bps)),
        slippage_bps=Decimal(str(args.slippage_bps)),
        out_dir=out_dir,
    )

    if not eq:
        print("No equity points produced. Check CSV columns and data.")
        return

    start = eq[0].equity
    end = eq[-1].equity
    ret = (end - start) / start if start != 0 else Decimal("0")
    mdd = _max_drawdown([x.equity for x in eq])

    summary_path = (out_dir / "summary.json") if out_dir else None
    summary = None
    if summary_path and summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = None

    print(f"csv={csv_path}")
    print(f"symbol={args.symbol}")
    print(f"out={out_dir}")
    print(f"bars={len(eq)}")
    print(f"start_equity={start}")
    print(f"end_equity={end}")
    print(f"return={ret}")
    print(f"max_drawdown={mdd}")

    if isinstance(summary, dict):
        print(f"trades={summary.get('trades')}")
        print(f"trades_per_day={summary.get('trades_per_day')}")
        print(f"win_rate={summary.get('win_rate')}")
        print(f"profit_factor={summary.get('profit_factor')}")
        print(f"avg_trade_pnl={summary.get('avg_trade_pnl')}")
        print(f"median_trade_pnl={summary.get('median_trade_pnl')}")
        print(f"max_consecutive_losses={summary.get('max_consecutive_losses')}")
        print(f"total_fees={summary.get('total_fees')}")
        print(f"avg_duration_sec={summary.get('avg_duration_sec')}")
        print(f"p95_duration_sec={summary.get('p95_duration_sec')}")



if __name__ == "__main__":
    main()
