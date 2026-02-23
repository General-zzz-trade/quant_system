"""Structured snapshot diffing for observability and audit.

Compares two ``StateSnapshot`` instances and produces a detailed,
JSON-serializable diff that shows exactly what changed in market,
account, and position state between two event boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True, slots=True)
class FieldDelta:
    """A single field-level change."""
    field: str
    old: Any
    new: Any

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "old": _serialize(self.old), "new": _serialize(self.new)}


@dataclass(frozen=True, slots=True)
class CategoryDiff:
    """Diff for one state category (market, account, position)."""
    category: str
    changed: bool
    deltas: Tuple[FieldDelta, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "changed": self.changed,
            "deltas": [d.to_dict() for d in self.deltas],
        }


@dataclass(frozen=True, slots=True)
class PositionDiff:
    """Diff for a single symbol's position."""
    symbol: str
    action: str  # "added" | "removed" | "changed" | "unchanged"
    deltas: Tuple[FieldDelta, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "deltas": [d.to_dict() for d in self.deltas],
        }


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    """Structured diff between two StateSnapshots."""
    changed: bool
    market: CategoryDiff
    account: CategoryDiff
    positions: Tuple[PositionDiff, ...]
    meta: CategoryDiff  # event_id, ts, bar_index changes

    @property
    def summary(self) -> str:
        parts = []
        if self.market.changed:
            parts.append(f"market({len(self.market.deltas)})")
        if self.account.changed:
            parts.append(f"account({len(self.account.deltas)})")
        pos_changed = [p for p in self.positions if p.action != "unchanged"]
        if pos_changed:
            parts.append(f"positions({len(pos_changed)})")
        return ", ".join(parts) if parts else "no changes"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "changed": self.changed,
            "summary": self.summary,
            "market": self.market.to_dict(),
            "account": self.account.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "meta": self.meta.to_dict(),
        }


def compute_diff(old: Any, new: Any) -> SnapshotDiff:
    """Compute a structured diff between two StateSnapshots.

    Both arguments should be ``StateSnapshot`` instances (or compatible).
    """
    market_diff = _diff_dataclass(old.market, new.market, "market")
    account_diff = _diff_dataclass(old.account, new.account, "account")
    positions_diff = _diff_positions(
        getattr(old, "positions", {}),
        getattr(new, "positions", {}),
    )
    meta_diff = _diff_meta(old, new)

    changed = (
        market_diff.changed
        or account_diff.changed
        or any(p.action != "unchanged" for p in positions_diff)
        or meta_diff.changed
    )

    return SnapshotDiff(
        changed=changed,
        market=market_diff,
        account=account_diff,
        positions=positions_diff,
        meta=meta_diff,
    )


# ── Internal helpers ─────────────────────────────────────

def _diff_dataclass(old: Any, new: Any, category: str) -> CategoryDiff:
    """Compare two frozen dataclass instances field-by-field."""
    if old is None and new is None:
        return CategoryDiff(category=category, changed=False)
    if old is None or new is None:
        return CategoryDiff(
            category=category,
            changed=True,
            deltas=(FieldDelta(field="__present__", old=old is not None, new=new is not None),),
        )

    deltas = []
    fields = getattr(old, "__dataclass_fields__", None)
    if fields:
        field_names = list(fields.keys())
    else:
        field_names = list(getattr(old, "__slots__", []))

    for name in field_names:
        old_val = getattr(old, name, None)
        new_val = getattr(new, name, None)
        if old_val != new_val:
            deltas.append(FieldDelta(field=name, old=old_val, new=new_val))

    return CategoryDiff(
        category=category,
        changed=len(deltas) > 0,
        deltas=tuple(deltas),
    )


def _diff_positions(
    old_positions: Mapping[str, Any],
    new_positions: Mapping[str, Any],
) -> Tuple[PositionDiff, ...]:
    """Diff position maps: detect added, removed, changed, unchanged."""
    all_symbols = sorted(set(old_positions) | set(new_positions))
    diffs = []

    for sym in all_symbols:
        old_pos = old_positions.get(sym)
        new_pos = new_positions.get(sym)

        if old_pos is None:
            diffs.append(PositionDiff(symbol=sym, action="added"))
        elif new_pos is None:
            diffs.append(PositionDiff(symbol=sym, action="removed"))
        else:
            cat = _diff_dataclass(old_pos, new_pos, f"position:{sym}")
            if cat.changed:
                diffs.append(PositionDiff(symbol=sym, action="changed", deltas=cat.deltas))
            else:
                diffs.append(PositionDiff(symbol=sym, action="unchanged"))

    return tuple(diffs)


def _diff_meta(old: Any, new: Any) -> CategoryDiff:
    """Diff snapshot metadata fields (event_id, ts, bar_index)."""
    deltas = []
    for name in ("event_id", "ts", "bar_index", "event_type"):
        old_val = getattr(old, name, None)
        new_val = getattr(new, name, None)
        if old_val != new_val:
            deltas.append(FieldDelta(field=name, old=old_val, new=new_val))
    return CategoryDiff(category="meta", changed=len(deltas) > 0, deltas=tuple(deltas))


def _serialize(value: Any) -> Any:
    """Make a value JSON-friendly."""
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
