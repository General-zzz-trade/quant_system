"""Port contracts: the existing implementations conform to trading.application.ports (migration step 3).

Two checks per port: the implementation has every port method and accepts every parameter name the
port declares; and a failing venue makes reads raise instead of returning an empty answer.
"""
from __future__ import annotations

import inspect
from decimal import Decimal as D

import pytest

from execution.adapters.binance import BinanceConfig
from execution.adapters.binance.adapter import BinanceAdapter
from execution.adapters.binance.rest import BinanceRetryableError
from execution.safety.limits import OrderLimiter
from runner.binance_main import EventJournal
from shared.drawdown_guard import DrawdownGuard
from shared.execution_journal import ExecutionJournal
from shared.exposure_gate import AccountExposureGate
from shared.kill_file import KillFile
from tests.unit.binance.fake_venue import FakeBinance, make_bars
from trading.application import ports


def missing_members(impl, protocol) -> list[str]:
    problems = []
    for name, member in vars(protocol).items():
        if name.startswith("_") or not callable(member):
            continue
        method = getattr(impl, name, None)
        if method is None:
            problems.append(name)
            continue
        have = inspect.signature(method).parameters
        open_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in have.values())
        for param in list(inspect.signature(member).parameters)[1:]:
            if param not in have and not open_kwargs and param != "fields":
                problems.append(f"{name}({param})")
    return problems


def _adapter(client=None):
    fake = client or FakeBinance(make_bars(10), clock=lambda: 1_800_000_000.0)
    adapter = BinanceAdapter(BinanceConfig(api_key="k", api_secret="s"), client=fake)
    return adapter


@pytest.mark.parametrize("protocol", [ports.MarketData, ports.ExchangeAccount, ports.OrderGateway,
                                      ports.StopOrderGateway])
def test_binance_adapter_implements_the_exchange_ports(protocol):
    adapter = _adapter()
    assert missing_members(adapter, protocol) == []
    assert isinstance(adapter, protocol)


def test_state_and_risk_components_implement_their_ports(tmp_path):
    cases = [
        (ExecutionJournal(tmp_path / "orders.sqlite", "s"), ports.OrderJournal),
        (KillFile(tmp_path / "kill.json"), ports.KillSwitch),
        (DrawdownGuard(tmp_path / "dd.sqlite", "s"), ports.DrawdownGuard),
        (OrderLimiter(), ports.OrderLimits),
        (AccountExposureGate(_adapter()), ports.ExposureReader),
        (EventJournal(tmp_path / "events.jsonl"), ports.EventLog),
    ]
    for impl, protocol in cases:
        assert missing_members(impl, protocol) == [], (type(impl).__name__, protocol.__name__)
        assert isinstance(impl, protocol)


def test_signature_check_catches_a_missing_parameter():
    class Partial:
        def get_klines(self, symbol, interval="60", limit=200):
            return []

        def get_funding_history(self, symbol, *, limit=1000):
            return []

    assert missing_members(Partial(), ports.MarketData) == ["get_klines(now_ms)"]


class _Down:
    """A venue whose every request fails like a network outage."""

    def request_public(self, *, method, path, params=None):
        raise BinanceRetryableError("Network error: down")

    request_signed = request_public


@pytest.mark.parametrize("call", [
    lambda a: a.get_klines("BTCUSDT", "60", 5, now_ms=1_800_000_000_000),
    lambda a: a.get_funding_history("BTCUSDT", limit=3),
    lambda a: a.get_account_equity(),
    lambda a: a.get_positions("BTCUSDT", strict=True),
    lambda a: a.get_open_orders("BTCUSDT", strict=True),
    lambda a: a.get_order("BTCUSDT", order_id="1"),
    lambda a: a.get_recent_fills(symbol="BTCUSDT", order_id="1", limit=100),
    lambda a: a.get_algo_order(client_algo_id="x"),
    lambda a: a.get_ticker("BTCUSDT"),
])
def test_reads_raise_when_the_venue_is_down(call):
    with pytest.raises(Exception):
        call(_adapter(_Down()))


def test_exposure_gate_blocks_when_the_account_cannot_be_read():
    allowed, reason = AccountExposureGate(_adapter(_Down())).check_would_exceed(
        symbol="BTCUSDT", venue="binance", side="buy", proposed_notional_usd=float(D("100")))
    assert not allowed and "exposure check failed" in reason
