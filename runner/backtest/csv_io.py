from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence, Tuple


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
    l: Decimal  # noqa: E741
    c: Decimal
    v: Optional[Decimal]


def iter_ohlcv_csv(path: Path) -> Iterator[OhlcvBar]:
    header_like = (
        {_to_key(x) for x in _TS_COLS} | {_to_key("open_time")}
        | {_to_key("open time")} | {_to_key("ts")} | {_to_key("timestamp")}
    )

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

            if _to_key(s) in header_like:
                continue

            try:
                ts = _parse_ts(raw_ts)
            except ValueError as e:
                if _to_key(s) in header_like:
                    continue
                raise ValueError(f"bad timestamp at row {idx}: {raw_ts!r}") from e

            o = _dec(row.get(o_col))
            h = _dec(row.get(h_col))
            l = _dec(row.get(l_col))  # noqa: E741
            c = _dec(row.get(c_col))
            v = _dec(row.get(v_col)) if v_col and row.get(v_col) not in (None, "") else None
            yield OhlcvBar(ts=ts, o=o, h=h, l=l, c=c, v=v)
