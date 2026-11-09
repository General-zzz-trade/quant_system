"""Offline data-to-signal contract through the production alpha builder.

Synthetic public-response fixtures, model objects and balances are explicit.
This check does not validate online feeds, trained model artifacts, account
access, exchange settlement or deployment readiness.
"""
from __future__ import annotations

from contextlib import chdir
from datetime import datetime, timezone
from decimal import Decimal
import csv
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from runner.startup_check import no_network

_HOUR_MS = 3_600_000
_BASE_MS = 1_704_067_200_000
_WARMUP_BARS = 8
_SYMBOL = "BTCUSDT"
_RUNNER = "OFFLINE_PAPER_CONTRACT"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("Offline paper contract failed: " + message)


class _SyntheticModel:
    """A fixture model; the real ensemble and discretizer remain in use."""

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, rows):
        self.calls += len(rows)
        return [float(row[0]) / 100 for row in rows]


class _SyntheticVenue:
    venue = "bybit"

    def __init__(self, bars: list[dict]) -> None:
        self.bars = bars
        self.placements = 0
        self.balance_reads = 0

    def get_balances(self):
        self.balance_reads += 1
        return {"USDT": SimpleNamespace(total=Decimal("10000"))}

    def get_klines(self, symbol, *, interval, limit):
        _require(symbol == _SYMBOL and interval == "60", "unexpected historical data request")
        return list(reversed(self.bars[-limit:]))

    def send_market_order(self, *args, **kwargs):
        self.placements += 1
        raise AssertionError("Signal contract must never submit an order")


class _FixtureRiskGate:
    def check(self, event):
        from shared.entry_risk import GateResult
        return GateResult(True)


def _historical_files(root: Path) -> None:
    root.mkdir()
    for suffix, column, value in (
        ("open_interest", "sum_open_interest", lambda i: 1000 + i),
        ("ls_ratio", "long_short_ratio", lambda i: 1 + i / 100),
    ):
        with (root / f"{_SYMBOL}_{suffix}.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("timestamp", column))
            # Include a live-time CSV row too: the check must use the different
            # public-cache observation once the real warmup phase has ended.
            for index in range(_WARMUP_BARS + 1):
                writer.writerow((_BASE_MS + index * _HOUR_MS, value(index)))


def check_paper_contract(*, inject_stale_data: bool = False) -> dict[str, object]:
    """Check real native builder wiring with deterministic, network-free inputs.

    ``inject_stale_data`` is a failure-injection hook for the regression suite.
    It must make this function fail before reporting the positive contract.
    """
    with no_network(), TemporaryDirectory(prefix="quant-paper-contract-") as scratch, chdir(scratch):
        from data.oi_cache import BinanceOICache, OI_PATH, LS_PATH, RECEIPT_TTL_MS
        from event.header import EventHeader
        from event.types import EventType, MarketEvent, OrderEvent
        from runner.builders import alpha_builder
        from runner.warmup import warmup

        root = Path(scratch)
        data_root = root / "data_files"
        _historical_files(data_root)
        observed_ms = _BASE_MS + (_WARMUP_BARS + 1) * _HOUR_MS + 60_000
        clock = [observed_ms / 1000]
        transport_error = [False]
        requests = []

        def fetch(path, params):
            if transport_error[0]:
                raise TimeoutError("Synthetic public data timeout")
            _require(path in {OI_PATH, LS_PATH}, "unexpected public data endpoint")
            _require(params["symbol"] == _SYMBOL and params["period"] == "1h", "data contract changed")
            requests.append(path)
            field = "sumOpenInterest" if path == OI_PATH else "longShortRatio"
            return [{"symbol": _SYMBOL, "timestamp": _BASE_MS + index * _HOUR_MS,
                     field: str(2000 + index if path == OI_PATH else 2 + index / 10)}
                    for index in range(2, 10)]

        cache = BinanceOICache(_SYMBOL, fetch=fetch, clock=lambda: clock[0])
        _require(cache.refresh(), "synthetic public observations were not accepted")
        bars = [{"time": _BASE_MS + index * _HOUR_MS, "open": 100 + index,
                 "high": 102 + index, "low": 99 + index, "close": 101 + index, "volume": 10 + index}
                for index in range(_WARMUP_BARS)]
        venue, model = _SyntheticVenue(bars), _SyntheticModel()
        info = {"zscore_window": 4, "zscore_warmup": 2,
                "horizon_models": [{"features": ["close"], "lgbm": model, "ic": 1.0}],
                "config": {}, "deadzone": 0.5, "min_hold": 1, "max_hold": 10}
        config = {"symbol": _SYMBOL, "interval": "60", "warmup": _WARMUP_BARS,
                  "size": 0.001, "step": 0.001, "max_qty": 1}
        with (
            patch.object(alpha_builder, "DATA_DIR", data_root),
            patch.dict(alpha_builder.SYMBOL_CONFIG, {_RUNNER: config}),
            patch.dict(os.environ, {"RISK_SCOPE": "offline-paper-contract",
                                    "ORDER_JOURNAL_PATH": str(root / "orders.sqlite")}),
        ):
            # The actual execution boundary is attached so stale-data rejection
            # is exercised there. Its mock venue has no network/account client;
            # durable execution remains unreconciled and cannot place orders.
            coord, module = alpha_builder.build_coordinator(
                _SYMBOL, _RUNNER, info, venue, dry_run=False, oi_cache=cache,
                risk_gate=_FixtureRiskGate(),
            )
            audit_path = root / "signal-audit.jsonl"
            module._audit._path = audit_path
            module._audit_enabled = False
            readiness = coord._data_readiness
            readiness.clock = lambda: clock[0]
            try:
                processed = warmup(coord, venue, _SYMBOL, "60", _WARMUP_BARS)
                _require(processed == _WARMUP_BARS, "historical warmup did not complete")
                _require(readiness.valid_warmup_bars == _WARMUP_BARS, "OI/LS warmup did not advance")
                _require(not readiness.snapshot()["ready"], "warmup incorrectly marked live data ready")
                readiness.set_live()
                module._audit_enabled = True
                coord.start()
                live_ms = _BASE_MS + _WARMUP_BARS * _HOUR_MS
                if inject_stale_data:
                    clock[0] += (RECEIPT_TTL_MS + 1) / 1000
                live_ts = datetime.fromtimestamp(live_ms / 1000, timezone.utc)
                market_header = EventHeader(
                    event_id="paper-contract-live-market", event_type=EventType.MARKET,
                    version=1, ts_ns=live_ms * 1_000_000, source="offline-paper-contract",
                )
                coord.emit(MarketEvent(
                    header=market_header, ts=live_ts, symbol=_SYMBOL, open=Decimal("108"),
                    high=Decimal("110"), low=Decimal("107"), close=Decimal("109"), volume=Decimal("18"),
                ), actor="offline-paper-contract")
                fresh = readiness.snapshot()
                _require(fresh["ready"], "live fixture data is not ready: " + fresh["reason"])
                features = module._last_features
                _require(features["close"] == 109, "native market feature was not propagated")
                _require(abs(float(features["ls_ratio"]) - 2.8) < 1e-9,
                         "live feature did not use the causal cache observation")
                _require(model.calls >= _WARMUP_BARS + 1, "actual ensemble did not evaluate fixture model")
                audits = [json.loads(line) for line in audit_path.read_text().splitlines()]
                signals = [record for record in audits if record.get("type") == "signal"]
                _require(bool(signals), "alpha decision produced no observable signal evaluation")
                _require(all(math.isfinite(float(record["z_score"])) for record in signals),
                         "signal normalization did not produce finite values")
                order = OrderEvent(
                    header=EventHeader.from_parent(parent=market_header, event_type=EventType.ORDER,
                                                   version=1, source="offline-paper-contract"),
                    order_id="paper-contract-blocked-entry", intent_id="paper-contract-intent",
                    symbol=_SYMBOL, side="buy",
                    qty=Decimal("0.01"), price=Decimal("109"),
                )
                gate = coord._execution_bridge.risk_gate
                _require(gate.check(order).allowed, "fresh data failed the fixture execution gate")
                clock[0] += (RECEIPT_TTL_MS + 1) / 1000
                stale = readiness.snapshot()
                _require(not stale["ready"] and not gate.check(order).allowed,
                         "expired data continued to permit entries")
                coord._execution_bridge.handle_event(order)
                transport_error[0] = True
                _require(not cache.refresh(), "transport failure incorrectly refreshed data")
                failed = readiness.snapshot()
                _require(not failed["ready"] and not gate.check(order).allowed,
                         "failed data refresh continued to permit entries")
                coord._execution_bridge.handle_event(order)
                _require(venue.placements == 0, "signal-only check attempted an order")
                _require(coord._store.get_position(_SYMBOL).qty == 0, "signal-only check altered position")
                return {"status": "passed", "scope": "offline-paper-data-signal-contract",
                        "historical_bars": processed, "live_bars": 1, "model_evaluations": model.calls,
                        "signal_evaluations": len(signals), "data_ready_before_fault": True,
                        "stale_data_blocked": True, "failed_refresh_blocked": True,
                        "venue_order_calls": venue.placements, "public_fixture_requests": len(requests),
                        "model_kind": "synthetic fixture", "online_feed_verified": False,
                        "trained_models_verified": False, "account_verified": False,
                        "exchange_settlement_verified": False}
            finally:
                coord.stop()
                module._audit.close()
                cache.stop()
