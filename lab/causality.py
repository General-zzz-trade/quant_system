"""Look-ahead check shared by tests and studies (design 2026-11 M5, step S12).

The rule: change every input that becomes known after a time ``t``, or remove it; every output known at or
before ``t`` must stay bit-for-bit identical. ``violations(fn, inputs, cuts)`` calls ``fn(inputs)`` once as
given and, for every cut, once with the later inputs changed and once with them removed. A test that checks many
functions on the same inputs builds the altered inputs once (``variants``) and calls ``check`` for each.

A new feature, signal or target function passes this on synthetic data before its first real-data run. The
registry in tests/unit/lab/test_lookahead.py runs it on every such function research and the forward tests use.

When a row is known (inputs and outputs are read by the same rules):
- a frame with an ``available_at`` column (hourly bars, aggregates, targets): that column;
- a frame with a ``timestamp`` column (Binance metrics): ``timestamp + METRIC_LAG``, the 5 minutes lab.pair
  waits before it uses a row;
- any other frame or series: its index, which must be epoch milliseconds;
- a dict keyed by epoch milliseconds (funding ``{settlement: (rate, mark)}``, decisions): the key;
- a list of bars: ``bar.available_at``.
Dicts with other keys, lists and tuples are read item by item. Any other input (a spec, a number, an array of
decision times) stays as it is; any other output is refused, because nothing says when it is known.

A changed value is multiplied by a random factor in [0.5, 0.9] or [1.1, 2.0], so prices stay positive; an integer
column other than a time column is increased by 1 to 999 instead. Time columns never change. After a
change the outputs are compared bit for bit. A removal shortens arrays, and OpenBLAS then rounds some rows of a
matrix product differently: Ridge and frozen-model predictions moved by up to one unit in the last place of the
largest prediction (S12). After a removal, finite floats may therefore differ by at most ``REMOVAL_ULPS`` units
in the last place of the largest value in their column; everything else is still compared bit for bit.

A check is refused as vacuous when a cut has no output known by it or no input known after it, or when no change
at any cut reaches the output.
"""
from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from lab.pair import METRIC_LAG

EPOCH_MIN = 10 ** 11            # 1973-03-03 in ms; smaller integers are positions, not times
TIME_COLUMNS = frozenset({"open_time", "available_at", "timestamp", "close_time", "funding_time"})
BAR_FIELDS = ("open", "high", "low", "close", "volume")
REMOVAL_ULPS = 4                # see the module docstring


def _is_time_keyed(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(
        isinstance(k, (int, np.integer)) and not isinstance(k, bool) and k >= EPOCH_MIN for k in value)


def _is_bar_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        hasattr(b, "available_at") and hasattr(b, "open_time") for b in value)


def _epoch(values: Any, what: str) -> np.ndarray:
    values = np.asarray(values)
    if values.dtype.kind not in "iu" or (len(values) and values.min() < EPOCH_MIN):
        raise TypeError(f"{what} is not in epoch milliseconds: cannot tell when the rows are known")
    return values.astype(np.int64)


def known_times(value: Any) -> np.ndarray | None:
    """When each row or item of ``value`` is known (ms); None when it has no time axis."""
    if isinstance(value, pd.DataFrame):
        if "available_at" in value.columns:
            return _epoch(value["available_at"].to_numpy(), "available_at")
        if "timestamp" in value.columns:
            return _epoch(value["timestamp"].to_numpy(), "timestamp") + METRIC_LAG
        return _epoch(value.index.to_numpy(), "the index")
    if isinstance(value, pd.Series):
        return _epoch(value.index.to_numpy(), "the index")
    if _is_time_keyed(value):
        return np.fromiter((int(k) for k in value), dtype=np.int64, count=len(value))
    if _is_bar_list(value):
        return np.array([b.available_at for b in value], dtype=np.int64)
    return None


def _factors(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.exp(rng.choice([-1.0, 1.0], n) * rng.uniform(0.1, 0.7, n))


def _scaled(value: Any, rng: np.random.Generator) -> Any:
    if isinstance(value, (tuple, list)):
        return type(value)(_scaled(v, rng) for v in value)
    if isinstance(value, Decimal):
        return value * Decimal(str(round(float(_factors(rng, 1)[0]), 6)))
    if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        return float(value) * float(_factors(rng, 1)[0])
    return value


def _changed_frame(value: pd.DataFrame | pd.Series, later: np.ndarray, rng: np.random.Generator):
    out = value.copy()
    rows = np.flatnonzero(later)
    columns = [None] if isinstance(out, pd.Series) else [c for c in out.columns if c not in TIME_COLUMNS]
    for column in columns:
        series = out if column is None else out[column]
        kind = series.dtype.kind
        if kind == "f":
            new = series.to_numpy(copy=True)
            new[rows] = new[rows] * _factors(rng, len(rows))
        elif kind in "iu":
            new = series.to_numpy(copy=True)
            new[rows] = new[rows] + rng.integers(1, 1000, len(rows))
        else:
            continue
        if column is None:
            out = pd.Series(new, index=out.index, name=out.name)
        else:
            out[column] = new
    return out


def _alter(value: Any, t: int, remove: bool, rng: np.random.Generator) -> tuple[Any, int]:
    """(``value`` with everything known after ``t`` changed or removed, how many rows or items that was)."""
    if isinstance(value, dict) and not _is_time_keyed(value):
        out, count = {}, 0
        for key, item in value.items():
            out[key], n = _alter(item, t, remove, rng)
            count += n
        return out, count
    if isinstance(value, (list, tuple)) and not _is_bar_list(value):
        parts = [_alter(item, t, remove, rng) for item in value]
        return type(value)(p[0] for p in parts), sum(p[1] for p in parts)
    times = known_times(value)
    if times is None:
        return value, 0
    later = times > t
    count = int(later.sum())
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return (value.iloc[np.flatnonzero(~later)] if remove else _changed_frame(value, later, rng)), count
    if isinstance(value, dict):
        if remove:
            return {k: v for k, v in value.items() if k <= t}, count
        return {k: (_scaled(v, rng) if k > t else v) for k, v in value.items()}, count
    if remove:
        return [b for b, late in zip(value, later) if not late], count
    factors = iter(_factors(rng, count * len(BAR_FIELDS)).round(6).tolist())
    return [replace(b, **{f: getattr(b, f) * Decimal(str(next(factors))) for f in BAR_FIELDS})
            if late else b for b, late in zip(value, later)], count


def known_at(value: Any, t: int) -> Any:
    """The part of an output known at or before ``t``; refuses an output without a time axis."""
    if isinstance(value, dict) and not _is_time_keyed(value):
        return {key: known_at(item, t) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not _is_bar_list(value):
        return type(value)(known_at(item, t) for item in value)
    times = known_times(value)
    if times is None:
        raise TypeError(f"cannot tell when a {type(value).__name__} output is known; return a timed frame or series")
    keep = times <= t
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.iloc[np.flatnonzero(keep)]
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if k <= t}
    return [b for b, k in zip(value, keep) if k]


def _rows(value: Any) -> int:
    if isinstance(value, dict) and not _is_time_keyed(value):
        return sum(_rows(v) for v in value.values())
    if isinstance(value, (list, tuple)) and not _is_bar_list(value):
        return sum(_rows(v) for v in value)
    return len(value)


def bits(value: Any) -> Any:
    """A comparable image of ``value``: two images are equal exactly when the values are bit-for-bit equal."""
    if isinstance(value, pd.DataFrame):
        return ("frame", tuple(map(str, value.columns)), bits(value.index),
                tuple(bits(value.iloc[:, i]) for i in range(value.shape[1])))
    if isinstance(value, pd.Series):
        return ("series", repr(value.name), bits(value.index), bits(value.to_numpy()))
    if isinstance(value, pd.Index):
        return ("index", bits(np.asarray(value)))
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ("objects", value.shape, tuple(bits(v) for v in value.ravel()))
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, dict):
        return ("dict", tuple((bits(k), bits(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(bits(v) for v in value))
    if isinstance(value, (float, np.floating)):
        return ("float", struct.pack("<d", float(value)))
    return (type(value).__name__, repr(value))


def _equal(a: Any, b: Any, ulps: int) -> bool:
    """Bit-for-bit equal; with ``ulps``, finite floats may also differ by that many units in the last place of
    the largest value in their array (rounding of a sum whose terms are about that size)."""
    if bits(a) == bits(b):
        return True
    if not ulps:
        return False
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        return repr(a.name) == repr(b.name) and bits(a.index) == bits(b.index) and \
            _equal(a.to_numpy(), b.to_numpy(), ulps)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape \
            and a.dtype == b.dtype == np.float64:
        odd = ~(np.isfinite(a) & np.isfinite(b))
        x, y = a[~odd], b[~odd]
        unit = np.spacing(max(np.abs(x).max(), np.abs(y).max())) if len(x) else 0.0
        return bits(a[odd]) == bits(b[odd]) and bool(np.all(np.abs(x - y) <= ulps * unit))
    return False


def _where(a: Any, b: Any, path: str, ulps: int) -> list[str]:
    """Paths (keys, columns) at which two outputs differ."""
    if _equal(a, b, ulps):
        return []
    if isinstance(a, dict) and isinstance(b, dict) and not _is_time_keyed(a) and a.keys() == b.keys():
        return [p for k in a for p in _where(a[k], b[k], f"{path}[{k!r}]", ulps)]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) \
            and not _is_bar_list(a):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _where(x, y, f"{path}[{i}]", ulps)]
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame) and list(a.columns) == list(b.columns) \
            and bits(a.index) == bits(b.index):
        return [f"{path}[{c!r}]" for i, c in enumerate(a.columns) if not _equal(a.iloc[:, i], b.iloc[:, i], ulps)]
    return [path]


def _when(t: int) -> str:
    return pd.Timestamp(t, unit="ms").strftime("%Y-%m-%d %H:%M UTC")


@dataclass(frozen=True)
class Variant:
    """The inputs with everything known after ``cut`` changed (``removed`` False) or removed."""
    cut: int
    removed: bool
    inputs: Any


def variants(inputs: Any, cuts: Sequence[int], *, seed: int = 0) -> list[Variant]:
    """For every cut, the inputs with everything known after it changed, then removed. Raises ValueError when
    no input is known after a cut (the check would be vacuous)."""
    rng = np.random.default_rng(seed)
    out = []
    for t in cuts:
        for remove in (False, True):
            altered, count = _alter(inputs, t, remove, rng)
            if not count:
                raise ValueError(f"no input is known after {_when(t)}: the check would be vacuous")
            out.append(Variant(int(t), remove, altered))
    return out


def check(fn: Callable[[Any], Any], inputs: Any, probes: Sequence[Variant]) -> list[str]:
    """Where ``fn`` looks ahead: one line per variant and output part whose rows known by the cut differ from
    ``fn(inputs)``. An empty list means no look-ahead was found. Raises ValueError when the check would be
    vacuous: no output known at a cut, or no change that ever reaches the output."""
    base = fn(inputs)
    image = bits(base)
    found, reached = [], False
    for probe in probes:
        before = known_at(base, probe.cut)
        if not _rows(before):
            raise ValueError(f"no output is known at {_when(probe.cut)}: the check would be vacuous")
        out = fn(probe.inputs)
        reached = reached or (not probe.removed and bits(out) != image)
        how = "removing" if probe.removed else "changing"
        found += [f"{how} the inputs known after {_when(probe.cut)} changes {p}"
                  for p in _where(before, known_at(out, probe.cut), "output", REMOVAL_ULPS if probe.removed else 0)]
    if not reached:
        raise ValueError("changing the inputs known after the cuts never changes the output: the check is vacuous")
    return found


def violations(fn: Callable[[Any], Any], inputs: Any, cuts: Sequence[int], *, seed: int = 0) -> list[str]:
    """``check`` over the ``variants`` of these cuts: the one call a study needs."""
    return check(fn, inputs, variants(inputs, cuts, seed=seed))


__all__ = ["violations", "variants", "check", "Variant", "known_at", "known_times", "bits", "METRIC_LAG"]
