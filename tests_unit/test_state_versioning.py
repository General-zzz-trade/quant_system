"""Tests for state.versioning — schema version tracking and migration."""
from __future__ import annotations

import pytest

from state.versioning import (
    Compatibility,
    MigrationRegistry,
    SchemaVersion,
    check_compatibility,
)


class TestSchemaVersion:
    def test_create(self) -> None:
        v = SchemaVersion(version=2, description="upgrade")
        assert v.version == 2
        assert v.description == "upgrade"

    def test_default_description(self) -> None:
        v = SchemaVersion(version=1)
        assert v.description == ""


class TestCompatibility:
    def test_same_version_is_full(self) -> None:
        result = Compatibility.check(1, 1)
        assert result == Compatibility.FULL

    def test_older_stored_is_backward(self) -> None:
        result = Compatibility.check(1, 2)
        assert result == Compatibility.BACKWARD

    def test_newer_stored_is_none(self) -> None:
        result = Compatibility.check(2, 1)
        assert result == Compatibility.NONE


class TestCheckCompatibility:
    def test_same_version_ok(self) -> None:
        # Should not raise
        check_compatibility(1, 1)

    def test_older_stored_ok(self) -> None:
        # Backward compatible — should not raise
        check_compatibility(1, 2)

    def test_future_stored_raises(self) -> None:
        with pytest.raises(Exception):
            check_compatibility(2, 1)


class TestMigrationRegistry:
    def test_register_and_migrate(self) -> None:
        reg = MigrationRegistry()
        reg.register(1, 2, lambda data: {**data, "v2_field": True})
        result = reg.migrate({"name": "test"}, from_version=1, to_version=2)
        assert result["v2_field"] is True
        assert result["name"] == "test"

    def test_multi_step_migration(self) -> None:
        reg = MigrationRegistry()
        reg.register(1, 2, lambda d: {**d, "step1": True})
        reg.register(2, 3, lambda d: {**d, "step2": True})
        result = reg.migrate({"name": "test"}, from_version=1, to_version=3)
        assert result["step1"] is True
        assert result["step2"] is True

    def test_has_path(self) -> None:
        reg = MigrationRegistry()
        reg.register(1, 2, lambda d: d)
        reg.register(2, 3, lambda d: d)
        assert reg.has_path(1, 3) is True
        assert reg.has_path(1, 4) is False

    def test_no_migration_needed(self) -> None:
        reg = MigrationRegistry()
        data = {"name": "test"}
        result = reg.migrate(data, from_version=1, to_version=1)
        assert result == data

    def test_non_sequential_raises(self) -> None:
        reg = MigrationRegistry()
        with pytest.raises(ValueError, match="sequential"):
            reg.register(1, 3, lambda d: d)
