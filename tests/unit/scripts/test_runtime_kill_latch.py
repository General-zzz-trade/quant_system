from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ops.runtime_kill_latch import (
    PersistentKillSwitch,
    RuntimeKillLatch,
    build_bybit_kill_latch,
    require_kill_latch_clear,
)


class _FakeKillSwitch:
    def __init__(self) -> None:
        self.calls = []
        self.armed = False

    def arm(self, scope, key, mode, reason, **kwargs):
        self.calls.append((scope, key, mode, reason, kwargs))
        self.armed = True
        return {"ok": True}

    def is_armed(self):
        return self.armed


def _fake_adapter(*, api_key: str = "demo-key"):
    cfg = SimpleNamespace(
        api_key=api_key,
        api_secret="secret",
        base_url="https://api-demo.bybit.com",
        account_type="UNIFIED",
        category="linear",
    )
    return SimpleNamespace(_config=cfg)


def test_runtime_kill_latch_round_trip_and_clear(tmp_path):
    latch = RuntimeKillLatch(tmp_path / "alpha.json")

    record = latch.arm(reason="daily_loss", payload={"symbol": "ETHUSDT"})

    assert latch.is_armed() is True
    stored = latch.read()
    assert stored["reason"] == "daily_loss"
    assert stored["symbol"] == "ETHUSDT"
    assert stored["pid"] == record["pid"]
    assert latch.clear() is True
    assert latch.is_armed() is False
    assert latch.read() is None


def test_require_kill_latch_clear_rejects_armed_latch(tmp_path):
    latch = RuntimeKillLatch(tmp_path / "alpha.json")
    latch.arm(reason="PM drawdown 24.4%")

    with pytest.raises(RuntimeError, match="persistent kill latch armed"):
        require_kill_latch_clear(latch, runtime_name="bybit-alpha.service")


def test_persistent_kill_switch_arms_underlying_switch_and_latch(tmp_path):
    latch = RuntimeKillLatch(tmp_path / "alpha.json")
    kill_switch = _FakeKillSwitch()
    wrapped = PersistentKillSwitch(
        kill_switch,
        latch=latch,
        service_name="bybit-alpha.service",
        scope_name="portfolio",
    )

    result = wrapped.arm("global", "*", "halt", "PM drawdown 24.4%", source="PortfolioManager")

    assert result == {"ok": True}
    assert kill_switch.calls == [
        ("global", "*", "halt", "PM drawdown 24.4%", {"source": "PortfolioManager"})
    ]
    stored = latch.read()
    assert stored["service"] == "bybit-alpha.service"
    assert stored["scope_name"] == "portfolio"
    assert stored["reason"] == "PM drawdown 24.4%"


def test_build_bybit_kill_latch_scopes_account_and_service(tmp_path):
    latch = build_bybit_kill_latch(
        adapter=_fake_adapter(api_key="key-1"),
        service_name="bybit-mm.service",
        scope_name="ETHUSDT",
        latch_dir=str(tmp_path),
    )

    assert latch.path.parent == Path(tmp_path)
    assert "bybit-mm.service" in latch.path.name
    assert "ETHUSDT" in latch.path.name
