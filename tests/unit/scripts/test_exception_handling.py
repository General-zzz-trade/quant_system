"""Tests for trading exception hierarchy."""
from __future__ import annotations

import pytest


class TestExceptionHierarchy:

    def test_import(self):
        from infra.errors import (
            TradingError, VenueError, InsufficientMargin,
            OrderRejected, ModelError, DataError, ReconcileError,
        )

    def test_hierarchy(self):
        from infra.errors import (
            TradingError, VenueError, InsufficientMargin,
            OrderRejected, ModelError, DataError, ReconcileError,
        )
        assert issubclass(VenueError, TradingError)
        assert issubclass(InsufficientMargin, VenueError)
        assert issubclass(OrderRejected, VenueError)
        assert issubclass(ModelError, TradingError)
        assert issubclass(DataError, TradingError)
        assert issubclass(ReconcileError, TradingError)

    def test_insufficient_margin_attrs(self):
        from infra.errors import InsufficientMargin
        e = InsufficientMargin(needed=1000.0, available=500.0)
        assert e.needed == 1000.0
        assert e.available == 500.0
        assert "1000" in str(e)

    def test_order_rejected_attrs(self):
        from infra.errors import OrderRejected
        e = OrderRejected("test", reason="Qty invalid", order_params={"qty": 0.001})
        assert e.reason == "Qty invalid"
        assert e.order_params["qty"] == 0.001

    def test_reconcile_error_attrs(self):
        from infra.errors import ReconcileError
        e = ReconcileError(local_pos=1.0, exchange_pos=0.0)
        assert e.local_pos == 1.0
        assert e.exchange_pos == 0.0

    def test_catch_venue_catches_margin(self):
        from infra.errors import VenueError, InsufficientMargin
        try:
            raise InsufficientMargin("low balance")
        except VenueError:
            pass  # Should be caught

    def test_catch_trading_catches_all(self):
        from infra.errors import (
            TradingError, VenueError, ModelError, DataError,
        )
        for exc_cls in [VenueError, ModelError, DataError]:
            try:
                raise exc_cls("test")
            except TradingError:
                pass  # Should be caught

    def test_venue_error_not_caught_by_model(self):
        from infra.errors import VenueError, ModelError
        with pytest.raises(VenueError):
            try:
                raise VenueError("network timeout")
            except ModelError:
                pass

    def test_default_messages(self):
        from infra.errors import InsufficientMargin, OrderRejected, ReconcileError
        e1 = InsufficientMargin(needed=500, available=100)
        assert "500" in str(e1) and "100" in str(e1)
        e2 = OrderRejected(reason="too small")
        assert "too small" in str(e2)
        e3 = ReconcileError(local_pos=1.0, exchange_pos=0.0)
        assert "1.0" in str(e3)

    def test_exception_with_no_args(self):
        from infra.errors import VenueError, ModelError, DataError
        for cls in [VenueError, ModelError, DataError]:
            e = cls()
            assert isinstance(e, Exception)

    def test_alpha_runner_uses_venue_error(self):
        """Verify AlphaRunner uses VenueError from exceptions module."""
        from runner.alpha_runner import VenueError
        from infra.errors import VenueError as VE
        assert VenueError is VE
