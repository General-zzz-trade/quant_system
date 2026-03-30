"""Tests for close order dust position prevention.

Verifies that make_close_order emits qty=0 so execution adapter
queries exchange for actual position, preventing dust residuals.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from decision.modules.alpha_orders import make_close_order


class TestCloseOrderDustPrevention:
    """Close orders should use qty=0 to delegate sizing to exchange."""

    def test_close_order_emits_zero_qty(self):
        """make_close_order should emit qty=0 regardless of _current_qty."""
        orders = make_close_order(
            symbol="BTCUSDT",
            runner_key="BTCUSDT",
            price=66000.0,
            old_signal=1,  # was LONG
            reason="quick_loss",
            current_qty=Decimal("0.115"),
            min_size=Decimal("0.001"),
        )
        assert len(orders) == 1
        assert orders[0].qty == Decimal("0")
        assert orders[0].side == "sell"

    def test_close_short_emits_zero_qty(self):
        """Close SHORT should also emit qty=0."""
        orders = make_close_order(
            symbol="ETHUSDT",
            runner_key="ETHUSDT",
            price=1990.0,
            old_signal=-1,  # was SHORT
            reason="z_reversal",
            current_qty=Decimal("4.5"),
            min_size=Decimal("0.01"),
        )
        assert len(orders) == 1
        assert orders[0].qty == Decimal("0")
        assert orders[0].side == "buy"

    def test_close_with_zero_tracked_qty_still_works(self):
        """Even if _current_qty is somehow 0, close order is valid."""
        orders = make_close_order(
            symbol="BTCUSDT",
            runner_key="BTCUSDT",
            price=66000.0,
            old_signal=1,
            reason="signal_change",
            current_qty=Decimal("0"),
            min_size=Decimal("0.001"),
        )
        assert len(orders) == 1
        assert orders[0].qty == Decimal("0")


class TestExecutionAdapterCloseDispatch:
    """Execution adapter should call reliable_close_position when qty=0."""

    def test_zero_qty_triggers_reliable_close(self):
        """When order qty=0, adapter should use reliable_close_position."""
        from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter

        mock_adapter = MagicMock()
        mock_adapter.close_position.return_value = {"status": "closed"}
        mock_adapter.get_positions.return_value = []

        exec_adapter = BinanceExecutionAdapter(mock_adapter)

        # Create order event with qty=0
        order = MagicMock()
        order.symbol = "BTCUSDT"
        order.side = "sell"
        order.qty = Decimal("0")
        order.time_in_force = "GTC"

        exec_adapter.send_order(order)

        # Should have called close_position (via reliable_close_position)
        mock_adapter.close_position.assert_called_once_with("BTCUSDT")


class TestBatchPredOverride:
    """AlphaDecisionModule should use batch prediction when set."""

    def test_batch_pred_override_used(self):
        """_batch_pred_override should replace incremental prediction."""
        from decision.modules.alpha import AlphaDecisionModule
        from unittest.mock import MagicMock

        predictor = MagicMock()
        predictor.predict.return_value = 0.05  # incremental
        discretizer = MagicMock()
        discretizer.discretize.return_value = (0, 0.0)
        discretizer.deadzone = 1.2
        sizer = MagicMock()

        module = AlphaDecisionModule(
            symbol="BTCUSDT",
            runner_key="BTCUSDT",
            predictor=predictor,
            discretizer=discretizer,
            sizer=sizer,
            leverage=10.0,
        )

        # Set batch override
        module._batch_pred_override = 0.001  # batch value

        # Create snapshot
        snap = MagicMock()
        mkt = MagicMock()
        mkt.close_f = 66000.0
        mkt.close = 66000.0
        mkt.high = 66500.0
        mkt.low = 65500.0
        snap.markets = {"BTCUSDT": mkt}
        snap.features = {"close": 66000.0}

        # Warmup the module
        module._bars_processed = 800
        module._audit_enabled = False

        module.decide(snap)

        # predictor.predict should NOT have been called (override used)
        predictor.predict.assert_not_called()

        # discretizer should have received the batch prediction
        discretizer.discretize.assert_called_once()
        args = discretizer.discretize.call_args
        assert args[0][0] == 0.001  # batch pred passed to discretize

    def test_override_cleared_after_use(self):
        """_batch_pred_override should be None after decide() uses it."""
        from decision.modules.alpha import AlphaDecisionModule

        predictor = MagicMock()
        predictor.predict.return_value = 0.05
        discretizer = MagicMock()
        discretizer.discretize.return_value = (0, 0.0)
        discretizer.deadzone = 1.2
        sizer = MagicMock()

        module = AlphaDecisionModule(
            symbol="BTCUSDT",
            runner_key="BTCUSDT",
            predictor=predictor,
            discretizer=discretizer,
            sizer=sizer,
            leverage=10.0,
        )

        module._batch_pred_override = 0.001
        module._bars_processed = 800
        module._audit_enabled = False

        snap = MagicMock()
        mkt = MagicMock()
        mkt.close_f = 66000.0
        mkt.close = 66000.0
        mkt.high = 66500.0
        mkt.low = 65500.0
        snap.markets = {"BTCUSDT": mkt}
        snap.features = {"close": 66000.0}

        module.decide(snap)

        assert module._batch_pred_override is None

    def test_no_override_uses_predictor(self):
        """Without override, should use predictor.predict()."""
        from decision.modules.alpha import AlphaDecisionModule

        predictor = MagicMock()
        predictor.predict.return_value = 0.05
        discretizer = MagicMock()
        discretizer.discretize.return_value = (0, 0.0)
        discretizer.deadzone = 1.2
        sizer = MagicMock()

        module = AlphaDecisionModule(
            symbol="BTCUSDT",
            runner_key="BTCUSDT",
            predictor=predictor,
            discretizer=discretizer,
            sizer=sizer,
            leverage=10.0,
        )

        module._bars_processed = 800
        module._audit_enabled = False

        snap = MagicMock()
        mkt = MagicMock()
        mkt.close_f = 66000.0
        mkt.close = 66000.0
        mkt.high = 66500.0
        mkt.low = 65500.0
        snap.markets = {"BTCUSDT": mkt}
        snap.features = {"close": 66000.0}

        module.decide(snap)

        predictor.predict.assert_called_once()
