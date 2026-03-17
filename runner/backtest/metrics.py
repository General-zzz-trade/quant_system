from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from math import sqrt
from pathlib import Path
from statistics import median, stdev
from typing import Any, Dict, List, Optional, Sequence

from .adapter import _make_id


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
        if n >= 1_000_000_000_000_000_000:
            return datetime.fromtimestamp(n / 1_000_000_000.0, tz=timezone.utc)
        if n >= 1_000_000_000_000_000:
            return datetime.fromtimestamp(n / 1_000_000.0, tz=timezone.utc)
        if n >= 1_000_000_000_000:
            return datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
        if n >= 1_000_000_000:
            return datetime.fromtimestamp(float(n), tz=timezone.utc)

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
    def _key(r: Dict[str, Any]) -> datetime:
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
                "open": None,
            },
        )

        pos_qty: Decimal = st["pos_qty"]
        avg_px: Optional[Decimal] = st["avg_px"]
        open_tr: Optional[Dict[str, Any]] = st["open"]

        prev_sign = _pos_sign(pos_qty)
        signed_sign = _pos_sign(signed)

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

        new_qty = pos_qty + signed

        if new_qty == 0:
            st["pos_qty"] = Decimal("0")
            st["avg_px"] = None
            st["open"] = None
            continue

        if (pos_qty > 0 and new_qty > 0) or (pos_qty < 0 and new_qty < 0):
            st["pos_qty"] = new_qty
            st["avg_px"] = avg_px

            remain_qty = abs_before - closed_qty
            remain_open_fee = open_fee_total - alloc_open_fee
            open_tr["qty"] = remain_qty
            open_tr["fees_open"] = remain_open_fee
            st["open"] = open_tr
            continue

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

    sharpe = ""
    if len(equity) >= 2:
        eq_vals = [float(p.equity) for p in equity]
        daily_rets = [(eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1] for i in range(1,
            len(eq_vals)) if eq_vals[i - 1] != 0]
        if len(daily_rets) >= 2:
            try:
                mean_r = sum(daily_rets) / len(daily_rets)
                std_r = stdev(daily_rets)
                if std_r > 0:
                    if years > 0 and len(daily_rets) > 0:
                        bars_per_year = len(daily_rets) / years
                    else:
                        bars_per_year = 252.0
                    sharpe = str(round(mean_r / std_r * sqrt(bars_per_year), 4))
            except Exception:
                sharpe = ""

    calmar = ""
    if cagr and mdd > 0:
        try:
            calmar = str(round(float(cagr) / float(mdd), 4))
        except Exception:
            calmar = ""

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
