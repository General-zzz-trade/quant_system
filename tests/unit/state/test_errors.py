"""Tests for state.errors."""
from __future__ import annotations

import pytest

from infra.errors import StateError
from state.errors import (
    ProjectionError,
    ReducerError,
    SchemaVersionError,
    SnapshotConsistencyError,
)


class TestStateErrors:
    def test_reducer_error_is_state_error(self):
        err = ReducerError("bad reduce")
        assert isinstance(err, StateError)
        assert str(err) == "bad reduce"

    def test_projection_error_is_state_error(self):
        err = ProjectionError("projection failed")
        assert isinstance(err, StateError)

    def test_snapshot_consistency_error_is_state_error(self):
        err = SnapshotConsistencyError("negative balance")
        assert isinstance(err, StateError)

    def test_schema_version_error_fields(self):
        err = SchemaVersionError(stored=1, current=3)
        assert err.stored == 1
        assert err.current == 3
        assert "stored=1" in str(err)
        assert "current=3" in str(err)

    def test_schema_version_error_is_state_error(self):
        err = SchemaVersionError(stored=2, current=5)
        assert isinstance(err, StateError)

    def test_errors_catchable_as_state_error(self):
        with pytest.raises(StateError):
            raise ReducerError("test")
