# runner/preflight.py
"""Pre-flight checks for live trading — validates connectivity, credentials, symbols, balance."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """Raised when pre-flight checks fail."""

    def __init__(self, result: PreflightResult) -> None:
        self.result = result
        failed = [c for c in result.checks if not c.passed]
        msg = "Pre-flight checks failed:\n" + "\n".join(
            f"  - {c.name}: {c.message}" for c in failed
        )
        super().__init__(msg)


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    name: str
    passed: bool
    message: str
    detail: Optional[Dict[str, Any]] = None


@dataclass(frozen=True, slots=True)
class PreflightResult:
    passed: bool
    checks: tuple[PreflightCheck, ...]


class PreflightChecker:
    """Runs pre-flight checks against a Binance REST client."""

    def __init__(self, rest_client: Any) -> None:
        self._client = rest_client

    def check_connectivity(self) -> PreflightCheck:
        try:
            self._client.request_public(method="GET", path="/fapi/v1/ping")
            return PreflightCheck(name="connectivity", passed=True, message="API reachable")
        except Exception as e:
            return PreflightCheck(
                name="connectivity", passed=False,
                message=f"API unreachable: {e}",
            )

    def check_credentials(self) -> PreflightCheck:
        try:
            resp = self._client.request_signed(method="GET", path="/fapi/v2/account")
            can_trade = resp.get("canTrade", False)
            if not can_trade:
                return PreflightCheck(
                    name="credentials", passed=False,
                    message="API key valid but trading disabled",
                    detail={"canTrade": can_trade},
                )
            return PreflightCheck(
                name="credentials", passed=True,
                message="API key valid, trading enabled",
            )
        except Exception as e:
            return PreflightCheck(
                name="credentials", passed=False,
                message=f"Credential check failed: {e}",
            )

    def check_symbols(self, symbols: Sequence[str]) -> PreflightCheck:
        try:
            resp = self._client.request_public(method="GET", path="/fapi/v1/exchangeInfo")
            exchange_symbols = {
                s["symbol"]: s["status"]
                for s in resp.get("symbols", [])
            }
            missing = []
            not_trading = []
            for sym in symbols:
                if sym not in exchange_symbols:
                    missing.append(sym)
                elif exchange_symbols[sym] != "TRADING":
                    not_trading.append(f"{sym}({exchange_symbols[sym]})")

            if missing or not_trading:
                parts = []
                if missing:
                    parts.append(f"missing: {missing}")
                if not_trading:
                    parts.append(f"not trading: {not_trading}")
                return PreflightCheck(
                    name="symbols", passed=False,
                    message="; ".join(parts),
                    detail={"missing": missing, "not_trading": not_trading},
                )
            return PreflightCheck(
                name="symbols", passed=True,
                message=f"All {len(symbols)} symbols valid and TRADING",
            )
        except Exception as e:
            return PreflightCheck(
                name="symbols", passed=False,
                message=f"Symbol check failed: {e}",
            )

    def check_balance(self, min_balance: float = 0.0) -> PreflightCheck:
        try:
            resp = self._client.request_signed(method="GET", path="/fapi/v2/balance")
            usdt_balance = 0.0
            for asset in resp if isinstance(resp, list) else []:
                if asset.get("asset") == "USDT":
                    usdt_balance = float(asset.get("availableBalance", 0))
                    break

            if usdt_balance < min_balance:
                return PreflightCheck(
                    name="balance", passed=False,
                    message=f"USDT balance {usdt_balance:.2f} < minimum {min_balance:.2f}",
                    detail={"usdt_available": usdt_balance},
                )
            return PreflightCheck(
                name="balance", passed=True,
                message=f"USDT available: {usdt_balance:.2f}",
                detail={"usdt_available": usdt_balance},
            )
        except Exception as e:
            return PreflightCheck(
                name="balance", passed=False,
                message=f"Balance check failed: {e}",
            )

    def check_positions(self, symbols: Sequence[str]) -> PreflightCheck:
        try:
            resp = self._client.request_signed(method="GET", path="/fapi/v2/positionRisk")
            open_positions = []
            for pos in resp if isinstance(resp, list) else []:
                sym = pos.get("symbol", "")
                amt = float(pos.get("positionAmt", 0))
                if sym in symbols and amt != 0.0:
                    open_positions.append(f"{sym}={amt}")

            if open_positions:
                return PreflightCheck(
                    name="positions", passed=True,
                    message=f"Existing positions: {', '.join(open_positions)}",
                    detail={"open_positions": open_positions},
                )
            return PreflightCheck(
                name="positions", passed=True,
                message="No existing positions",
            )
        except Exception as e:
            return PreflightCheck(
                name="positions", passed=False,
                message=f"Position check failed: {e}",
            )

    def run_all(
        self,
        symbols: Sequence[str],
        min_balance: float = 0.0,
    ) -> PreflightResult:
        checks = [
            self.check_connectivity(),
            self.check_credentials(),
            self.check_symbols(symbols),
            self.check_balance(min_balance),
            self.check_positions(symbols),
        ]
        # Stop early if connectivity fails
        if not checks[0].passed:
            return PreflightResult(passed=False, checks=tuple(checks[:1]))

        passed = all(c.passed for c in checks)
        return PreflightResult(passed=passed, checks=tuple(checks))
