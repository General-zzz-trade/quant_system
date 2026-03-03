"""Unit tests for scripts/testnet_smoke.py — no network required."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from scripts.testnet_smoke import (
    CheckResult,
    PhaseResult,
    format_report,
    phase1_public_rest,
    phase2_authenticated_rest,
    phase3_order_lifecycle,
    phase5_feature_pipeline,
)


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_all_pass(self) -> None:
        p = PhaseResult(title="Phase 1: Test")
        p.add("Check A", True, "ok")
        p.add("Check B", True, "good")
        report = format_report([p])
        assert "[PASS] Check A: ok" in report
        assert "[PASS] Check B: good" in report
        assert "2/2 PASSED" in report

    def test_mixed_results(self) -> None:
        p = PhaseResult(title="Phase 1: Test")
        p.add("Good", True, "ok")
        p.add("Bad", False, "broken")
        report = format_report([p])
        assert "[PASS] Good: ok" in report
        assert "[FAIL] Bad: broken" in report
        assert "1/2 PASSED" in report

    def test_multi_phase(self) -> None:
        p1 = PhaseResult(title="Phase 1: A")
        p1.add("C1", True, "ok")
        p2 = PhaseResult(title="Phase 2: B")
        p2.add("C2", True, "ok")
        p2.add("C3", False, "nope")
        report = format_report([p1, p2])
        assert "Phase 1: A" in report
        assert "Phase 2: B" in report
        assert "2/3 PASSED" in report

    def test_header_and_footer(self) -> None:
        report = format_report([PhaseResult(title="Phase 1: X")])
        assert "BINANCE TESTNET SMOKE TEST" in report
        assert "RESULT:" in report


# ---------------------------------------------------------------------------
# PhaseResult
# ---------------------------------------------------------------------------


class TestPhaseResult:
    def test_passed_all_true(self) -> None:
        p = PhaseResult(title="T")
        p.add("a", True, "")
        p.add("b", True, "")
        assert p.passed is True

    def test_passed_one_false(self) -> None:
        p = PhaseResult(title="T")
        p.add("a", True, "")
        p.add("b", False, "")
        assert p.passed is False

    def test_empty_is_passed(self) -> None:
        assert PhaseResult(title="T").passed is True


# ---------------------------------------------------------------------------
# Phase 1: Public REST
# ---------------------------------------------------------------------------


class TestPhase1:
    def _make_client(self, ping_ok: bool = True, time_resp: dict | None = None,
                     exchange_resp: dict | None = None) -> MagicMock:
        client = MagicMock()
        responses = {}
        if ping_ok:
            responses["/fapi/v1/ping"] = {}
        if time_resp is not None:
            responses["/fapi/v1/time"] = time_resp
        if exchange_resp is not None:
            responses["/fapi/v1/exchangeInfo"] = exchange_resp

        def public_side_effect(*, method, path, params=None):
            if not ping_ok and path == "/fapi/v1/ping":
                raise ConnectionError("unreachable")
            return responses.get(path, {})

        client.request_public.side_effect = public_side_effect
        return client

    def test_all_pass(self) -> None:
        client = self._make_client(
            time_resp={"serverTime": int(__import__("time").time() * 1000)},
            exchange_resp={"symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}]},
        )
        urls = SimpleNamespace(rest_base="https://testnet.binancefuture.com")
        result = phase1_public_rest(client, urls)
        assert result.passed
        assert len(result.checks) == 3

    def test_ping_fail_short_circuits(self) -> None:
        client = self._make_client(ping_ok=False)
        urls = SimpleNamespace(rest_base="https://testnet.binancefuture.com")
        result = phase1_public_rest(client, urls)
        assert not result.passed
        assert len(result.checks) == 1

    def test_time_drift(self) -> None:
        client = self._make_client(
            time_resp={"serverTime": 0},  # huge drift
            exchange_resp={"symbols": [{"symbol": "BTCUSDT", "status": "TRADING"}]},
        )
        urls = SimpleNamespace(rest_base="https://testnet.binancefuture.com")
        result = phase1_public_rest(client, urls)
        time_check = result.checks[1]
        assert not time_check.passed

    def test_btcusdt_missing(self) -> None:
        client = self._make_client(
            time_resp={"serverTime": int(__import__("time").time() * 1000)},
            exchange_resp={"symbols": [{"symbol": "ETHUSDT", "status": "TRADING"}]},
        )
        urls = SimpleNamespace(rest_base="https://testnet.binancefuture.com")
        result = phase1_public_rest(client, urls)
        assert not result.checks[2].passed


# ---------------------------------------------------------------------------
# Phase 2: Authenticated REST
# ---------------------------------------------------------------------------


class TestPhase2:
    def test_all_pass(self) -> None:
        client = MagicMock()
        client.request_signed.side_effect = [
            {"canTrade": True},
            [{"asset": "USDT", "availableBalance": "1000.00"}],
            [],
        ]
        result = phase2_authenticated_rest(client)
        assert result.passed
        assert len(result.checks) == 3
        assert "1000.00" in result.checks[1].detail

    def test_cant_trade(self) -> None:
        client = MagicMock()
        client.request_signed.side_effect = [
            {"canTrade": False},
            [{"asset": "USDT", "availableBalance": "500.00"}],
            [],
        ]
        result = phase2_authenticated_rest(client)
        assert not result.checks[0].passed
        assert not result.passed

    def test_account_exception_short_circuits(self) -> None:
        client = MagicMock()
        client.request_signed.side_effect = ConnectionError("auth failed")
        result = phase2_authenticated_rest(client)
        assert not result.checks[0].passed
        assert len(result.checks) == 1


# ---------------------------------------------------------------------------
# Phase 3: Order lifecycle
# ---------------------------------------------------------------------------


class TestPhase3:
    def test_full_lifecycle(self) -> None:
        client = MagicMock()
        client.request_public.return_value = {"price": "50000.00"}
        client.request_signed.side_effect = [
            {"orderId": 999, "status": "NEW"},           # submit
            [{"orderId": 999}],                            # open orders
            {"orderId": 999, "status": "CANCELED"},        # cancel
            [],                                            # verify empty
        ]
        result = phase3_order_lifecycle(client)
        assert result.passed
        assert len(result.checks) == 4
        assert "999" in result.checks[0].detail

    def test_price_fetch_fails(self) -> None:
        client = MagicMock()
        client.request_public.side_effect = ConnectionError("nope")
        result = phase3_order_lifecycle(client)
        assert not result.passed
        assert len(result.checks) == 1


# ---------------------------------------------------------------------------
# Phase 5: Feature pipeline
# ---------------------------------------------------------------------------


class TestPhase5:
    def _make_klines(self, n: int = 50) -> list:
        return [
            [0, "100", "105", "95", "102", "1000", 0, "50000", "200", "500", "0", "0"]
            for _ in range(n)
        ]

    def test_warmup_and_features(self) -> None:
        client = MagicMock()
        client.request_public.return_value = self._make_klines(50)
        result = phase5_feature_pipeline(client)
        warmup_check = result.checks[0]
        assert warmup_check.passed
        assert "50 bars" in warmup_check.detail
        feature_check = result.checks[1]
        assert feature_check.passed

    def test_insufficient_bars(self) -> None:
        client = MagicMock()
        client.request_public.return_value = self._make_klines(3)
        result = phase5_feature_pipeline(client)
        assert not result.checks[0].passed
