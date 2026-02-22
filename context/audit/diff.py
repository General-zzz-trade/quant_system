# context/audit/diff.py
"""Snapshot diffing — compare two context snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from context.context import ContextSnapshot


@dataclass(frozen=True, slots=True)
class FieldChange:
    """单个字段变化。"""
    field: str
    old_value: Any
    new_value: Any


@dataclass(frozen=True, slots=True)
class SnapshotDiff:
    """两个快照之间的差异。"""
    from_id: str
    to_id: str
    changes: tuple[FieldChange, ...]

    @property
    def has_changes(self) -> bool:
        return len(self.changes) > 0


def snapshot_diff(old: ContextSnapshot, new: ContextSnapshot) -> SnapshotDiff:
    """比较两个 ContextSnapshot 的差异。"""
    changes: list[FieldChange] = []

    if old.ts != new.ts:
        changes.append(FieldChange("ts", old.ts, new.ts))
    if old.bar_index != new.bar_index:
        changes.append(FieldChange("bar_index", old.bar_index, new.bar_index))
    if old.context_id != new.context_id:
        changes.append(FieldChange("context_id", old.context_id, new.context_id))

    # meta diff
    old_meta = dict(old.meta) if old.meta else {}
    new_meta = dict(new.meta) if new.meta else {}
    all_keys = set(old_meta.keys()) | set(new_meta.keys())
    for k in sorted(all_keys):
        ov = old_meta.get(k)
        nv = new_meta.get(k)
        if ov != nv:
            changes.append(FieldChange(f"meta.{k}", ov, nv))

    return SnapshotDiff(
        from_id=old.snapshot_id,
        to_id=new.snapshot_id,
        changes=tuple(changes),
    )
