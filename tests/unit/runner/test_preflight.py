"""Tests for runner/preflight.py — PreflightChecker pre-trade validation."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from runner.preflight import (
    PreflightCheck,
    PreflightChecker,
    PreflightError,
    PreflightResult,
)


# ── Helpers ─────────────────────────────────────────────────


def _mock_client(
    *,
    ping_ok: bool = True,
    can_trade: bool = True,
    symbols: list[dict[str, str]] | None = None,
    balance: list[dict[str, Any]] | None = None,
    positions: list[dict[str, Any]] | None = None,
    cred_error: Exception | None = None,
    balance_error: Exception | None = None,
) -> MagicMock:
    client = MagicMock()

    if ping_ok:
        client.request_public.side_effect = lambda method, path, **kw: (
            {} if path == "/fapi/v1/ping"
            else {"symbols": symbols or []} if path == "/fapi/v1/exchangeInfo"
            else {}
        )
    else:
        def _fail_public(method, path, **kw):
            if path == "/fapi/v1/ping":
                raise ConnectionError("timeout")
            return {"symbols": symbols or []}
        client.request_public.side_effect = _fail_public

    def _signed(method, path, **kw):
        if cred_error and path == "/fapi/v2/account":
            raise cred_error
        if path == "/fapi/v2/account":
            return {"canTrade": can_trade}
        if balance_error and path == "/fapi/v2/balance":
            raise balance_error
        if path == "/fapi/v2/balance":
            return balance if balance is not None else []
        if path == "/fapi/v2/positionRisk":
            return positions if positions is not None else []
        return {}

    client.request_signed.side_effect = _signed
    return client


# ── Tests ───────────────────────────────────────────────────


class TestPreflightCheck:
    def test_passed_check(self):
        c = PreflightCheck(name="test", passed=True, message="ok")
        assert c.passed
        assert c.name == "test"

    def test_failed_check_with_detail(self):
        c = PreflightCheck(name="bal", passed=False, message="low", detail={"amount": 5})
        assert not c.passed
        assert c.detail == {"amount": 5}


class TestPreflightError:
    def test_error_message_lists_failures(self):
        checks = [
            PreflightCheck(name="a", passed=True, message="ok"),
            PreflightCheck(name="b", passed=False, message="bad"),
        ]
        result = PreflightResult(passed=False, checks=tuple(checks))
        err = PreflightError(result)
        assert "b: bad" in str(err)
        assert "a: ok" not in str(err)


class TestPreflightChecker:
    def test_all_checks_pass(self):
        client = _mock_client(
            symbols=[{"symbol": "BTCUSDT", "status": "TRADING"}],
            balance=[{"asset": "USDT", "availableBalance": "5000"}],
        )
        checker = PreflightChecker(client)
        result = checker.run_all(["BTCUSDT"], min_balance=100)
        assert result.passed
        assert len(result.checks) == 5
        assert all(c.passed for c in result.checks)

    def test_connectivity_failure_early_stop(self):
        client = _mock_client(ping_ok=False)
        checker = PreflightChecker(client)
        result = checker.run_all(["BTCUSDT"])
        assert not result.passed
        assert len(result.checks) == 1
        assert result.checks[0].name == "connectivity"
        assert not result.checks[0].passed

    def test_credentials_trading_disabled(self):
        client = _mock_client(can_trade=False)
        checker = PreflightChecker(client)
        c = checker.check_credentials()
        assert not c.passed
        assert "trading disabled" in c.message

    def test_credentials_exception(self):
        client = _mock_client(cred_error=PermissionError("forbidden"))
        checker = PreflightChecker(client)
        c = checker.check_credentials()
        assert not c.passed
        assert "forbidden" in c.message

    def test_symbol_not_available(self):
        client = _mock_client(symbols=[{"symbol": "ETHUSDT", "status": "TRADING"}])
        checker = PreflightChecker(client)
        c = checker.check_symbols(["BTCUSDT"])
        assert not c.passed
        assert "missing" in c.message

    def test_symbol_not_trading(self):
        client = _mock_client(symbols=[{"symbol": "BTCUSDT", "status": "BREAK"}])
        checker = PreflightChecker(client)
        c = checker.check_symbols(["BTCUSDT"])
        assert not c.passed
        assert "not trading" in c.message

    def test_low_balance(self):
        client = _mock_client(balance=[{"asset": "USDT", "availableBalance": "50"}])
        checker = PreflightChecker(client)
        c = checker.check_balance(min_balance=100.0)
        assert not c.passed
        assert "50.00" in c.message
        assert c.detail["usdt_available"] == 50.0

    def test_sufficient_balance(self):
        client = _mock_client(balance=[{"asset": "USDT", "availableBalance": "5000"}])
        checker = PreflightChecker(client)
        c = checker.check_balance(min_balance=100.0)
        assert c.passed
        assert c.detail["usdt_available"] == 5000.0

    def test_existing_positions_info(self):
        client = _mock_client(
            positions=[{"symbol": "BTCUSDT", "positionAmt": "0.05"}],
        )
        checker = PreflightChecker(client)
        c = checker.check_positions(["BTCUSDT"])
        assert c.passed
        assert "BTCUSDT=0.05" in c.message

    def test_no_positions(self):
        client = _mock_client(positions=[])
        checker = PreflightChecker(client)
        c = checker.check_positions(["BTCUSDT"])
        assert c.passed
        assert "No existing positions" in c.message

    def test_run_all_fails_on_any_failure(self):
        client = _mock_client(
            symbols=[{"symbol": "BTCUSDT", "status": "TRADING"}],
            balance=[{"asset": "USDT", "availableBalance": "10"}],
        )
        checker = PreflightChecker(client)
        result = checker.run_all(["BTCUSDT"], min_balance=1000)
        assert not result.passed
        assert len(result.checks) == 5
        failed = [c for c in result.checks if not c.passed]
        assert len(failed) == 1
        assert failed[0].name == "balance"

    def test_balance_no_usdt_asset(self):
        client = _mock_client(balance=[{"asset": "BNB", "availableBalance": "100"}])
        checker = PreflightChecker(client)
        c = checker.check_balance(min_balance=50.0)
        assert not c.passed

    def test_positions_check_ignores_zero_qty(self):
        client = _mock_client(
            positions=[{"symbol": "BTCUSDT", "positionAmt": "0"}],
        )
        checker = PreflightChecker(client)
        c = checker.check_positions(["BTCUSDT"])
        assert c.passed
        assert "No existing positions" in c.message
