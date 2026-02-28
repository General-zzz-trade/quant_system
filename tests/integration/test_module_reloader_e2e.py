"""End-to-end tests for ModuleReloader + DecisionBridge hot-swap integration."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

from engine.decision_bridge import DecisionBridge
from engine.module_reloader import ModuleReloader, ReloaderConfig


class StubDecisionModule:
    """Decision module that records calls and returns nothing."""

    def __init__(self, name: str = "stub") -> None:
        self.name = name
        self.call_count = 0

    def decide(self, snapshot: Any) -> list:
        self.call_count += 1
        return []


def test_trigger_reload_calls_callback():
    """trigger_reload() invokes the on_reload callback with the trigger reason."""
    triggers: List[str] = []

    reloader = ModuleReloader(
        config=ReloaderConfig(enable_sighup=False),
        on_reload=lambda t: triggers.append(t),
    )
    reloader.start()

    reloader.trigger_reload()
    assert len(triggers) == 1
    assert triggers[0] == "manual"

    reloader.trigger_reload()
    assert len(triggers) == 2

    reloader.stop()
    assert not reloader.is_running


def test_swap_modules_via_decision_bridge():
    """DecisionBridge.swap_modules() replaces old modules and returns them."""
    emitted: List[Any] = []
    old_mod = StubDecisionModule("old")
    new_mod = StubDecisionModule("new")

    bridge = DecisionBridge(
        dispatcher_emit=lambda ev: emitted.append(ev),
        modules=[old_mod],
    )

    # Verify old module is active
    from engine.pipeline import PipelineOutput
    snapshot = SimpleNamespace(
        symbol="BTCUSDT",
        bar_index=0,
        event_type="MARKET",
        markets={"BTCUSDT": SimpleNamespace(close=40000.0)},
    )
    out = PipelineOutput(
        markets={"BTCUSDT": SimpleNamespace(close=40000.0)},
        account=SimpleNamespace(),
        positions={},
        portfolio=None,
        risk=None,
        features=None,
        event_index=0,
        last_event_id=None,
        last_ts=None,
        snapshot=snapshot,
        advanced=True,
    )
    bridge.on_pipeline_output(out)
    assert old_mod.call_count == 1
    assert new_mod.call_count == 0

    # Swap
    returned = bridge.swap_modules([new_mod])
    assert returned == [old_mod]

    # Verify new module is now active
    bridge.on_pipeline_output(out)
    assert old_mod.call_count == 1  # unchanged
    assert new_mod.call_count == 1


def test_reloader_with_bridge_swap():
    """ModuleReloader on_reload callback can trigger DecisionBridge.swap_modules()."""
    old_mod = StubDecisionModule("old")
    new_mod = StubDecisionModule("new")
    swap_log: List[str] = []

    bridge = DecisionBridge(
        dispatcher_emit=lambda ev: None,
        modules=[old_mod],
    )

    def on_reload(trigger: str) -> None:
        swap_log.append(trigger)
        bridge.swap_modules([new_mod])

    reloader = ModuleReloader(
        config=ReloaderConfig(enable_sighup=False),
        on_reload=on_reload,
    )
    reloader.start()

    # Trigger reload
    reloader.trigger_reload()
    assert swap_log == ["manual"]
    assert bridge.modules == [new_mod]

    reloader.stop()
