from __future__ import annotations

from runner.builders.engine_builder import build_coordinator_and_pipeline


class _Config:
    symbols = ("BTCUSDT",)
    currency = "USDT"
    enable_portfolio_risk = False
    max_position_notional = 25_000.0
    max_order_notional = 100.0
    max_open_orders = 3
    max_portfolio_notional = 40_000.0


class _KillSwitch:
    def is_killed(self):
        return None


class _OrderStateMachine:
    def active_orders(self):
        return []


def test_engine_builder_propagates_runtime_risk_limits(monkeypatch) -> None:
    monkeypatch.setattr("runner.gate_chain.build_gate_chain", lambda **kwargs: object())

    class _EmitHandler:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("runner.emit_handler.LiveEmitHandler", _EmitHandler)

    _coordinator, risk_gate, _portfolio_aggregator, _emit_handler, _emit, _event_recorder_ref = (
        build_coordinator_and_pipeline(
            _Config(),
            symbol_default="BTCUSDT",
            hook=None,
            feat_hook=None,
            tick_processors=None,
            _update_correlation=lambda snapshot: None,
            correlation_gate=None,
            kill_switch=_KillSwitch(),
            order_state_machine=_OrderStateMachine(),
            timeout_tracker=None,
            attribution_tracker=None,
            live_signal_tracker=None,
            alpha_health_monitor=None,
            regime_sizer=None,
            staged_risk=None,
            portfolio_allocator=None,
            fetch_margin=None,
            report=type("_Report", (), {"record": lambda *args, **kwargs: None})(),
        )
    )

    assert risk_gate.config.max_position_notional == 25_000.0
    assert risk_gate.config.max_order_notional == 100.0
    assert risk_gate.config.max_open_orders == 3
    assert risk_gate.config.max_portfolio_notional == 40_000.0

