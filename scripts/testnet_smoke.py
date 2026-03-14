"""Binance Testnet end-to-end smoke test.

Validates each layer of the system against the real testnet API:
  Phase 1: Public REST endpoints (no API key)
  Phase 2: Authenticated REST endpoints
  Phase 3: Order lifecycle (place → verify → cancel → verify)
  Phase 4: WebSocket market data
  Phase 5: Feature pipeline integration

Usage:
  # Full test (needs BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET):
  python3 -m scripts.testnet_smoke

  # Public endpoints only (no API key needed):
  python3 -m scripts.testnet_smoke --public-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class PhaseResult:
    title: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def add(self, name: str, passed: bool, detail: str) -> CheckResult:
        r = CheckResult(name=name, passed=passed, detail=detail)
        self.checks.append(r)
        return r


def format_report(phases: List[PhaseResult]) -> str:
    lines: List[str] = []
    sep = "=" * 60
    lines.append(sep)
    lines.append("  BINANCE TESTNET SMOKE TEST")
    lines.append(sep)
    lines.append("")

    total = 0
    passed = 0
    for phase in phases:
        lines.append(f"{phase.title}")
        for c in phase.checks:
            tag = "PASS" if c.passed else "FAIL"
            lines.append(f"  [{tag}] {c.name}: {c.detail}")
            total += 1
            if c.passed:
                passed += 1
        lines.append("")

    lines.append(sep)
    lines.append(f"  RESULT: {passed}/{total} PASSED")
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase implementations
# ---------------------------------------------------------------------------


def _build_rest_client(urls: Any) -> Any:
    from execution.adapters.binance.rest import BinanceRestClient, BinanceRestConfig

    api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
    api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
    cfg = BinanceRestConfig(
        base_url=urls.rest_base,
        api_key=api_key,
        api_secret=api_secret,
    )
    return BinanceRestClient(cfg)


def phase1_public_rest(client: Any, urls: Any) -> PhaseResult:
    """Phase 1: Public REST API — no API key needed."""
    phase = PhaseResult(title="Phase 1: Public REST API")

    # Ping
    try:
        t0 = time.monotonic()
        client.request_public(method="GET", path="/fapi/v1/ping")
        ms = (time.monotonic() - t0) * 1000
        phase.add("Ping", True, f"testnet reachable ({ms:.0f}ms)")
    except Exception as e:
        phase.add("Ping", False, str(e))
        return phase  # no point continuing

    # Time sync
    try:
        resp = client.request_public(method="GET", path="/fapi/v1/time")
        server_ms = resp.get("serverTime", 0)
        delta = abs(int(time.time() * 1000) - server_ms)
        ok = delta < 2000
        phase.add("Time sync", ok, f"delta={delta}ms {'(< 2000ms)' if ok else '(>= 2000ms!)'}")
    except Exception as e:
        phase.add("Time sync", False, str(e))

    # Exchange info — BTCUSDT
    try:
        resp = client.request_public(method="GET", path="/fapi/v1/exchangeInfo")
        symbols = {s["symbol"]: s["status"] for s in resp.get("symbols", [])}
        status = symbols.get("BTCUSDT")
        if status == "TRADING":
            phase.add("Exchange info", True, "BTCUSDT TRADING")
        elif status:
            phase.add("Exchange info", False, f"BTCUSDT status={status}")
        else:
            phase.add("Exchange info", False, "BTCUSDT not found")
    except Exception as e:
        phase.add("Exchange info", False, str(e))

    return phase


def phase2_authenticated_rest(client: Any) -> PhaseResult:
    """Phase 2: Authenticated REST API — needs API key."""
    phase = PhaseResult(title="Phase 2: Authenticated REST API")

    # Account
    try:
        resp = client.request_signed(method="GET", path="/fapi/v2/account")
        can_trade = resp.get("canTrade", False)
        phase.add("Account", bool(can_trade), f"canTrade={can_trade}")
    except Exception as e:
        phase.add("Account", False, str(e))
        return phase

    # Balance
    try:
        resp = client.request_signed(method="GET", path="/fapi/v2/balance")
        usdt_bal = 0.0
        for asset in (resp if isinstance(resp, list) else []):
            if asset.get("asset") == "USDT":
                usdt_bal = float(asset.get("availableBalance", 0))
                break
        phase.add("Balance", True, f"USDT={usdt_bal:.2f}")
    except Exception as e:
        phase.add("Balance", False, str(e))

    # Positions
    try:
        resp = client.request_signed(method="GET", path="/fapi/v2/positionRisk")
        open_pos = []
        for pos in (resp if isinstance(resp, list) else []):
            amt = float(pos.get("positionAmt", 0))
            if amt != 0.0:
                open_pos.append(f"{pos.get('symbol')}={amt}")
        if open_pos:
            phase.add("Positions", True, f"{len(open_pos)} open: {', '.join(open_pos)}")
        else:
            phase.add("Positions", True, "none open")
    except Exception as e:
        phase.add("Positions", False, str(e))

    return phase


def phase3_order_lifecycle(client: Any) -> PhaseResult:
    """Phase 3: Order lifecycle — place, verify, cancel, verify."""
    from types import SimpleNamespace

    from execution.adapters.binance.order_gateway_um import BinanceUmFuturesOrderGateway

    phase = PhaseResult(title="Phase 3: Order Lifecycle")
    gateway = BinanceUmFuturesOrderGateway(rest=client)

    # Get current price
    try:
        resp = client.request_public(
            method="GET", path="/fapi/v1/ticker/price",
            params={"symbol": "BTCUSDT"},
        )
        price = float(resp.get("price", 0))
        if price <= 0:
            phase.add("Get price", False, "invalid price")
            return phase
    except Exception as e:
        phase.add("Get price", False, str(e))
        return phase

    # Place limit buy far below market — ensure notional >= $100
    # Round price to tick size 0.10 for BTCUSDT
    limit_price = round(price * 0.5 / 0.10) * 0.10
    limit_price = round(limit_price, 1)
    import math
    min_qty = max(0.001, math.ceil(110 / limit_price * 1000) / 1000)
    order_id: Optional[int] = None
    try:
        cmd = SimpleNamespace(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            qty=min_qty,
            price=limit_price,
            time_in_force="GTC",
            reduce_only=False,
            request_id=None,
        )
        resp = gateway.submit_order(cmd)
        order_id = resp.get("orderId")
        phase.add("Limit buy placed", bool(order_id), f"orderId={order_id}")
    except Exception as e:
        phase.add("Limit buy placed", False, str(e))
        return phase

    # Verify open orders
    try:
        resp = client.request_signed(
            method="GET", path="/fapi/v1/openOrders",
            params={"symbol": "BTCUSDT"},
        )
        count = len(resp) if isinstance(resp, list) else 0
        found = any(o.get("orderId") == order_id for o in (resp if isinstance(resp, list) else []))
        phase.add("Open orders", found, f"{count} order(s) found")
    except Exception as e:
        phase.add("Open orders", False, str(e))

    # Cancel
    try:
        cmd_cancel = SimpleNamespace(symbol="BTCUSDT", order_id=order_id, client_order_id=None)
        gateway.cancel_order(cmd_cancel)
        phase.add("Cancel", True, "order cancelled")
    except Exception as e:
        phase.add("Cancel", False, str(e))

    # Verify cancelled
    try:
        resp = client.request_signed(
            method="GET", path="/fapi/v1/openOrders",
            params={"symbol": "BTCUSDT"},
        )
        remaining = len(resp) if isinstance(resp, list) else 0
        still_there = any(o.get("orderId") == order_id for o in (resp if isinstance(resp, list) else []))
        phase.add("Verify", not still_there, f"{remaining} open orders")
    except Exception as e:
        phase.add("Verify", False, str(e))

    return phase


def phase4_websocket(urls: Any) -> PhaseResult:
    """Phase 4: WebSocket market data — connect and receive kline."""
    from execution.adapters.binance.ws_transport_websocket_client import WebsocketClientTransport
    from execution.adapters.binance.kline_processor import KlineProcessor

    phase = PhaseResult(title="Phase 4: WebSocket Market Data")
    ws_url = f"{urls.ws_market_stream}?streams=btcusdt@kline_1m"
    transport = WebsocketClientTransport()

    try:
        transport.connect(ws_url)
        phase.add("Connected", True, ws_url)
    except Exception as e:
        phase.add("Connected", False, str(e))
        return phase

    KlineProcessor(only_closed=False)
    deadline = time.monotonic() + 30
    received = False
    detail = "timeout after 30s"
    try:
        while time.monotonic() < deadline:
            raw = transport.recv(timeout_s=5.0)
            if not raw:
                continue
            # Accept any valid kline message (open or closed)
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            data = payload.get("data", payload)
            if str(data.get("e", "")).strip() == "kline":
                k = data.get("k", {})
                sym = str(data.get("s", "")).upper()
                close_price = k.get("c", "?")
                vol = k.get("v", "?")
                received = True
                detail = f"{sym} close={close_price} vol={vol}"
                break
    finally:
        transport.close()

    phase.add("Kline received", received, detail)
    return phase


def phase5_feature_pipeline(client: Any) -> PhaseResult:
    """Phase 5: Feature pipeline — warm up with REST klines, run inference."""
    from features.enriched_computer import EnrichedFeatureComputer

    phase = PhaseResult(title="Phase 5: Feature Pipeline")

    # Fetch historical klines for warmup
    try:
        resp = client.request_public(
            method="GET", path="/fapi/v1/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": 50},
        )
        if not isinstance(resp, list) or len(resp) < 10:
            phase.add("Warmup", False, f"only {len(resp) if isinstance(resp, list) else 0} bars")
            return phase
        phase.add("Warmup", True, f"{len(resp)} bars loaded via REST")
    except Exception as e:
        phase.add("Warmup", False, str(e))
        return phase

    # Feed into EnrichedFeatureComputer
    computer = EnrichedFeatureComputer()
    features: Dict[str, Optional[float]] = {}
    for bar in resp:
        # Binance kline format: [open_time, open, high, low, close, volume, close_time,
        #   quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore]
        features = computer.on_bar(
            "BTCUSDT",
            close=float(bar[4]),
            volume=float(bar[5]),
            high=float(bar[2]),
            low=float(bar[3]),
            open_=float(bar[1]),
            trades=float(bar[8]),
            taker_buy_volume=float(bar[9]),
            quote_volume=float(bar[7]),
        )

    non_none = sum(1 for v in features.values() if v is not None)
    total = len(features)
    phase.add("Features", non_none > 0, f"{non_none}/{total} non-None after warmup")

    # Optional: ML inference if model exists
    model_path = Path("models/lgbm_alpha.txt")
    if model_path.exists():
        try:
            from datetime import datetime, timezone
            from alpha.models.lgbm_alpha import LGBMAlphaModel

            model = LGBMAlphaModel()
            model.load(str(model_path))
            signal = model.predict(
                symbol="BTCUSDT",
                ts=datetime.now(timezone.utc),
                features=features,
            )
            score = getattr(signal, "strength", None) if signal else None
            phase.add("ML inference", True, f"ml_score={score}")
        except Exception as e:
            phase.add("ML inference", False, str(e))
    else:
        phase.add("ML inference", True, "skipped (no model file)")

    return phase


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Phase 6: Safety Integration Validation
# ---------------------------------------------------------------------------


def phase6_safety_integration() -> PhaseResult:
    """Validate RiskGate, OrderStateMachine, GracefulShutdown wiring."""
    phase = PhaseResult(title="Phase 6: Safety Integration")

    # Test RiskGate
    try:
        from execution.safety.risk_gate import RiskGate, RiskGateConfig

        gate = RiskGate(
            config=RiskGateConfig(
                max_position_notional=1000.0,
                max_order_notional=500.0,
                max_open_orders=5,
                max_portfolio_notional=2000.0,
            ),
            get_positions=lambda: {},
            is_killed=lambda: False,
        )

        # Create a mock order that should pass
        class _MockOrder:
            symbol = "BTCUSDT"
            qty = 0.001
            price = 50000.0
            order_type = "LIMIT"

        result = gate.check(_MockOrder())
        phase.add("RiskGate allows valid order", result.allowed,
                   f"allowed={result.allowed} reason={result.reason}")

        # Create a mock order that exceeds notional
        class _BigOrder:
            symbol = "BTCUSDT"
            qty = 100.0
            price = 50000.0
            order_type = "LIMIT"

        result2 = gate.check(_BigOrder())
        phase.add("RiskGate blocks oversized order", not result2.allowed,
                   f"allowed={result2.allowed} reason={result2.reason}")

        # Test kill switch rejection
        gate_killed = RiskGate(
            config=RiskGateConfig(),
            get_positions=lambda: {},
            is_killed=lambda: True,
        )
        result3 = gate_killed.check(_MockOrder())
        phase.add("RiskGate blocks when kill switch active", not result3.allowed,
                   f"allowed={result3.allowed} reason={result3.reason}")

    except Exception as e:
        phase.add("RiskGate validation", False, str(e))

    # Test OrderStateMachine
    try:
        from execution.state_machine.machine import OrderStateMachine
        from execution.state_machine.transitions import OrderStatus

        from decimal import Decimal
        osm = OrderStateMachine()
        state = osm.register(
            order_id="test-001",
            client_order_id="coid-001",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            qty=Decimal("0.001"),
            price=Decimal("50000.0"),
        )
        phase.add("OSM register order", state.status == OrderStatus.PENDING_NEW,
                   f"status={state.status}")

        state2 = osm.transition(order_id="test-001", new_status=OrderStatus.NEW)
        phase.add("OSM transition to NEW", state2.status == OrderStatus.NEW,
                   f"status={state2.status}")

        state3 = osm.transition(order_id="test-001", new_status=OrderStatus.FILLED,
                                filled_qty=Decimal("0.001"))
        phase.add("OSM transition to FILLED (terminal)",
                   state3.is_terminal, f"terminal={state3.is_terminal}")

        phase.add("OSM active count after fill",
                   osm.active_count() == 0, f"active={osm.active_count()}")

    except Exception as e:
        phase.add("OrderStateMachine validation", False, str(e))

    # Test TimeoutTracker
    try:
        from execution.safety.timeout_tracker import OrderTimeoutTracker

        tracker = OrderTimeoutTracker(timeout_sec=0.1)
        tracker.on_submit("order-timeout-test")
        phase.add("TimeoutTracker submit", tracker.pending_count == 1,
                   f"pending={tracker.pending_count}")

        time.sleep(0.2)  # Wait for timeout
        timed_out = tracker.check_timeouts()
        phase.add("TimeoutTracker detects timeout",
                   "order-timeout-test" in timed_out,
                   f"timed_out={timed_out}")

    except Exception as e:
        phase.add("TimeoutTracker validation", False, str(e))

    # Test GracefulShutdown sequence
    try:
        from runner.graceful_shutdown import GracefulShutdown, ShutdownConfig

        steps_executed: List[str] = []

        shutdown = GracefulShutdown(
            config=ShutdownConfig(pending_order_timeout_sec=0.5),
            stop_new_orders=lambda: steps_executed.append("stop_new"),
            wait_pending=lambda: (steps_executed.append("wait"), True)[1],
            cancel_all=lambda: steps_executed.append("cancel"),
            reconcile=lambda: steps_executed.append("reconcile"),
            save_snapshot=lambda p: steps_executed.append("save"),
            cleanup=lambda: steps_executed.append("cleanup"),
        )
        shutdown.execute()

        expected = ["stop_new", "wait", "cancel", "reconcile", "save", "cleanup"]
        phase.add("GracefulShutdown 6-step sequence",
                   steps_executed == expected,
                   f"expected={expected} got={steps_executed}")

    except Exception as e:
        phase.add("GracefulShutdown validation", False, str(e))

    return phase


# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance testnet end-to-end smoke test")
    parser.add_argument(
        "--public-only", action="store_true",
        help="Only run public endpoint tests (no API key needed)",
    )
    args = parser.parse_args()

    from execution.adapters.binance.urls import resolve_binance_urls

    urls = resolve_binance_urls(testnet=True)
    client = _build_rest_client(urls)

    phases: List[PhaseResult] = []

    phases.append(phase1_public_rest(client, urls))
    if not phases[-1].passed:
        print(format_report(phases))
        sys.exit(1)

    if not args.public_only:
        api_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
        if not api_key:
            print("ERROR: BINANCE_TESTNET_API_KEY not set. Use --public-only or export keys.")
            sys.exit(1)

        phases.append(phase2_authenticated_rest(client))
        phases.append(phase3_order_lifecycle(client))
        phases.append(phase4_websocket(urls))
        phases.append(phase5_feature_pipeline(client))
        phases.append(phase6_safety_integration())
    else:
        phases.append(phase4_websocket(urls))

    print(format_report(phases))

    total = sum(len(p.checks) for p in phases)
    passed = sum(1 for p in phases for c in p.checks if c.passed)
    sys.exit(0 if total == passed else 1)


if __name__ == "__main__":
    main()
