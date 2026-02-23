"""State schema versioning and compatibility checks.

Tracks schema versions and provides a migration registry for
forward-compatible state recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

from state.errors import SchemaVersionError

# Current schema version — bump on breaking state changes.
STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SchemaVersion:
    """Describes a state schema version."""
    version: int
    description: str = ""


# ── Compatibility ────────────────────────────────────────

class Compatibility:
    """Schema compatibility checker."""

    FULL = "full"           # exact match required
    BACKWARD = "backward"   # current can read older versions
    NONE = "none"           # incompatible

    @staticmethod
    def check(stored: int, current: int) -> str:
        """Check compatibility between stored and current schema versions."""
        if stored == current:
            return Compatibility.FULL
        if stored < current:
            return Compatibility.BACKWARD
        return Compatibility.NONE


def check_compatibility(stored_version: int, current_version: int = STATE_SCHEMA_VERSION) -> None:
    """Raise if stored version is incompatible with current.

    Forward-incompatible versions (stored > current) always raise.
    Backward-compatible versions (stored < current) are accepted —
    the migration registry handles upgrades.
    """
    compat = Compatibility.check(stored_version, current_version)
    if compat == Compatibility.NONE:
        raise SchemaVersionError(stored_version, current_version)


# ── Migration Registry ───────────────────────────────────

MigrationFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class MigrationRegistry:
    """Registered migrations from version N to N+1.

    Usage::

        registry = MigrationRegistry()
        registry.register(1, 2, migrate_v1_to_v2)
        migrated = registry.migrate(data, from_version=1, to_version=2)
    """

    def __init__(self) -> None:
        self._migrations: Dict[Tuple[int, int], MigrationFn] = {}

    def register(self, from_v: int, to_v: int, fn: MigrationFn) -> None:
        """Register a migration function from version ``from_v`` to ``to_v``."""
        if to_v != from_v + 1:
            raise ValueError(f"migrations must be sequential: {from_v} → {to_v}")
        self._migrations[(from_v, to_v)] = fn

    def migrate(
        self,
        data: Dict[str, Any],
        *,
        from_version: int,
        to_version: int = STATE_SCHEMA_VERSION,
    ) -> Dict[str, Any]:
        """Apply sequential migrations to bring data up to ``to_version``."""
        current = from_version
        result = dict(data)

        while current < to_version:
            key = (current, current + 1)
            fn = self._migrations.get(key)
            if fn is None:
                raise SchemaVersionError(current, to_version)
            result = fn(result)
            current += 1

        return result

    def has_path(self, from_version: int, to_version: int) -> bool:
        """Check if a migration path exists."""
        current = from_version
        while current < to_version:
            if (current, current + 1) not in self._migrations:
                return False
            current += 1
        return True
