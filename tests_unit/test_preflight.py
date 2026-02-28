# tests_unit/test_preflight.py
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from runner.preflight import PreflightChecker, PreflightCheck, PreflightResult, PreflightError


def _make_client(**overrides):
    client = MagicMock()

    # Default: everything succeeds
    client.request_public.return_value = {}
    client.request_signed.return_value = {
        "canTrade": True,
        "totalMarginBalance": "1000.0",
    }

    for k, v in overrides.items():
        setattr(client, k, v)
    return client


class TestCheckConnectivity:
    def test_pass(self):
        client = _make_client()
        checker = PreflightChecker(client)
        result = checker.check_connectivity()
        assert result.passed is True
        assert result.name == "connectivity"

    def test_fail_on_exception(self):
        client = _make_client()
        client.request_public.side_effect = RuntimeError("timeout")
        checker = PreflightChecker(client)
        result = checker.check_connectivity()
        assert result.passed is False
        assert "unreachable" in result.message.lower()


class TestCheckCredentials:
    def test_pass(self):
        client = _make_client()
        client.request_signed.return_value = {"canTrade": True}
        checker = PreflightChecker(client)
        result = checker.check_credentials()
        assert result.passed is True

    def test_fail_trading_disabled(self):
        client = _make_client()
        client.request_signed.return_value = {"canTrade": False}
        checker = PreflightChecker(client)
        result = checker.check_credentials()
        assert result.passed is False
        assert "disabled" in result.message.lower()

    def test_fail_on_exception(self):
        client = _make_client()
        client.request_signed.side_effect = RuntimeError("invalid key")
        checker = PreflightChecker(client)
        result = checker.check_credentials()
        assert result.passed is False


class TestCheckSymbols:
    def test_pass(self):
        client = _make_client()
        client.request_public.return_value = {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING"},
                {"symbol": "ETHUSDT", "status": "TRADING"},
            ]
        }
        checker = PreflightChecker(client)
        result = checker.check_symbols(["BTCUSDT", "ETHUSDT"])
        assert result.passed is True

    def test_fail_missing_symbol(self):
        client = _make_client()
        client.request_public.return_value = {
            "symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}]
        }
        checker = PreflightChecker(client)
        result = checker.check_symbols(["BTCUSDT", "XYZUSDT"])
        assert result.passed is False
        assert "missing" in result.message.lower()

    def test_fail_not_trading(self):
        client = _make_client()
        client.request_public.return_value = {
            "symbols": [{"symbol": "BTCUSDT", "status": "HALT"}]
        }
        checker = PreflightChecker(client)
        result = checker.check_symbols(["BTCUSDT"])
        assert result.passed is False
        assert "not trading" in result.message.lower()


class TestCheckBalance:
    def test_pass_no_minimum(self):
        client = _make_client()
        client.request_signed.return_value = [
            {"asset": "USDT", "availableBalance": "500.0"},
        ]
        checker = PreflightChecker(client)
        result = checker.check_balance(min_balance=0.0)
        assert result.passed is True
        assert result.detail["usdt_available"] == 500.0

    def test_fail_below_minimum(self):
        client = _make_client()
        client.request_signed.return_value = [
            {"asset": "USDT", "availableBalance": "10.0"},
        ]
        checker = PreflightChecker(client)
        result = checker.check_balance(min_balance=100.0)
        assert result.passed is False


class TestCheckPositions:
    def test_no_positions(self):
        client = _make_client()
        client.request_signed.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0"},
        ]
        checker = PreflightChecker(client)
        result = checker.check_positions(["BTCUSDT"])
        assert result.passed is True
        assert "no existing" in result.message.lower()

    def test_existing_positions(self):
        client = _make_client()
        client.request_signed.return_value = [
            {"symbol": "BTCUSDT", "positionAmt": "0.5"},
        ]
        checker = PreflightChecker(client)
        result = checker.check_positions(["BTCUSDT"])
        assert result.passed is True
        assert "existing" in result.message.lower()


class TestRunAll:
    def test_all_pass(self):
        client = _make_client()
        # ping
        client.request_public.return_value = {}
        # exchangeInfo (called for check_symbols)
        def _public_dispatch(*, method, path, params=None):
            if "exchangeInfo" in path:
                return {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}]}
            return {}
        client.request_public.side_effect = _public_dispatch

        # signed calls: account, balance, positionRisk
        def _signed_dispatch(*, method, path, params=None):
            if "account" in path:
                return {"canTrade": True}
            if "balance" in path:
                return [{"asset": "USDT", "availableBalance": "1000"}]
            if "positionRisk" in path:
                return [{"symbol": "BTCUSDT", "positionAmt": "0"}]
            return {}
        client.request_signed.side_effect = _signed_dispatch

        checker = PreflightChecker(client)
        result = checker.run_all(symbols=("BTCUSDT",))
        assert result.passed is True
        assert len(result.checks) == 5

    def test_connectivity_fail_stops_early(self):
        client = _make_client()
        client.request_public.side_effect = RuntimeError("unreachable")
        checker = PreflightChecker(client)
        result = checker.run_all(symbols=("BTCUSDT",))
        assert result.passed is False
        assert len(result.checks) == 1

    def test_preflight_error_message(self):
        client = _make_client()
        client.request_public.side_effect = RuntimeError("down")
        checker = PreflightChecker(client)
        result = checker.run_all(symbols=("BTCUSDT",))
        with pytest.raises(PreflightError) as exc_info:
            raise PreflightError(result)
        assert "connectivity" in str(exc_info.value).lower()
