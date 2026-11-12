"""Public-data shadow process: local observations, no account or execution client.

This is not a fill simulator. One-hour order intents are recorded as unsubmitted;
the existing four-hour signal-only behavior is preserved. No wallet/PnL claims
can be derived from the explicitly synthetic starting balance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any

from runner.run_manifest import RunManifest, load_run_manifest

logger = logging.getLogger(__name__)


class ShadowJournal:
    """Separate append-only observations; never writes execution facts."""

    def __init__(self, path: Path, manifest: RunManifest) -> None:
        from shared.runtime_state import _path
        self.path = _path(path)
        if self.path.exists() and not self.path.is_file():
            raise ValueError("Shadow observations must be a regular file")
        self.manifest = manifest

    def write(self, kind: str, **fields: Any) -> None:
        record = {"ts": time.time(), "mode": "shadow", "type": kind,
                  "manifest_digest": self.manifest.digest, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _market_event(symbol: str, bar: dict) -> Any:
    from event.header import EventHeader
    from event.types import EventType, MarketEvent
    return MarketEvent(
        header=EventHeader.new_root(event_type=EventType.MARKET, version=1, source="public_shadow"),
        ts=datetime.fromtimestamp(int(bar["start"]) / 1000, tz=timezone.utc), symbol=symbol,
        **{key: Decimal(str(bar[key])) for key in ("open", "high", "low", "close", "volume")},
    )


def _batch_prediction(symbol: str, model_dir: str, timestamp: int) -> float | None:
    from runner.batch_predictor import invalidate_cache, predict_latest
    invalidate_cache(symbol)
    return predict_latest(symbol, model_dir, as_of_ms=timestamp)


def _reset_after_warmup(module: Any) -> None:
    # Same startup reset as alpha_main, without restoring account positions.
    module._pending_orders.clear()
    module._pending_reductions.clear()
    module._pending_exit_reasons.clear()
    module._signal = 0
    module._current_qty = Decimal("0")
    module._entry_price = module._trade_peak = 0.0
    module._entry_bar = 0
    module._last_trade_bar = module._bars_processed
    module._consecutive_stops = module._stop_pause_until = 0
    reset_hold = getattr(module._discretizer._bridge, "reset_hold", None)
    if callable(reset_hold):
        reset_hold(module._symbol)


class ShadowSession:
    """Single-threaded market polling with injectable public transport for tests."""

    def __init__(self, manifest: RunManifest, public: Any, monitor: Any, journal: ShadowJournal) -> None:
        if not manifest.files_validated:
            raise ValueError("Shadow startup requires validated local inputs")
        if (manifest.project_root != Path.cwd().absolute()
                or manifest.models_root != manifest.project_root / "models_v8"
                or manifest.data_root != manifest.project_root / "data_files"):
            raise ValueError("Shadow must run from its project root with standard model/data roots")
        self.manifest, self.public, self.monitor, self.journal = manifest, public, monitor, journal
        self.coordinators: dict[str, Any] = {}
        self.modules: dict[str, Any] = {}
        self.caches: dict[str, Any] = {}
        self.cursors: dict[str, int] = {}
        self.warming = True

    def _observe_intent(self, key: str, event: Any) -> None:
        if not self.warming:
            self.journal.write("intent", runner_key=key, order_id=event.order_id,
                               symbol=event.symbol, side=str(event.side), qty=str(event.qty),
                               disposition="not_submitted", balance_source="synthetic")
        # Release pending intent bookkeeping through the existing terminal-status
        # callback. No fill callback, account balance, or confirmed position update.
        self.modules[key].on_order_status(event.order_id, "blocked")

    def start(self) -> None:
        from alpha.model_loader_prod import load_model
        from data.oi_cache import BinanceOICache
        from decision.modules.alpha import _OfflineAudit
        from engine.dispatcher import Route
        from runner.builders.alpha_builder import build_coordinator

        self.journal.write("session_start", runners=list(self.manifest.runner_keys),
                           disabled_runners=list(self.manifest.disabled_runner_keys),
                           shadow_balance=str(self.manifest.shadow_balance), balance_source="synthetic")
        for symbol in self.manifest.symbols:
            self.caches[symbol] = BinanceOICache(symbol, fetch=self.public.fetch_oi)
        for spec in self.manifest.runners:
            model = load_model(spec.model_path)
            coord, module = build_coordinator(
                symbol=spec.symbol, runner_key=spec.key, model_info=model, adapter=self.public, dry_run=True,
                oi_cache=self.caches[spec.symbol], starting_balance=self.manifest.shadow_balance,
                repair_inputs=False,
            )
            self.coordinators[spec.key], self.modules[spec.key] = coord, module
            if coord._execution_bridge is not None:
                raise RuntimeError("Shadow coordinator must not have an execution bridge")
            # The ordinary audit mixes virtual entries and account-facing records.
            # This process owns only its dedicated observation journal instead.
            module._audit = _OfflineAudit()
            module._audit_enabled = False
            coord.dispatcher.register(route=Route.EXECUTION,
                                      handler=lambda event, key=spec.key: self._observe_intent(key, event))
            bars = self.public.get_closed_bars(spec.symbol, interval=spec.interval, limit=spec.warmup)
            if len(bars) != spec.warmup:
                raise RuntimeError("Incomplete public warmup for " + spec.key)
            # Include the latest closed bar in warmup. Between boundaries that
            # bar can already be too old for live readiness; waiting on it as a
            # pending live bar would permanently obstruct later fresh bars.
            for bar in bars:
                coord.emit(_market_event(spec.symbol, bar), actor="warmup")
            _reset_after_warmup(module)
            module._audit_enabled = True  # This flag also gates an existing strategy entry condition.
            coord._data_readiness.set_live()
            if spec.interval == "60":
                coord._data_readiness.set_external_status("batch_inputs", False, "awaiting validated prediction")
            self.cursors[spec.key] = int(bars[-1]["start"])
            coord.start()
        consensus = {key: 0 for key in self.modules}
        for module in self.modules.values():
            module.set_consensus(consensus)
        self.warming = False
        self.monitor.bind(self.coordinators, self.modules)

    def cycle(self) -> dict | None:
        for cache in self.caches.values():
            cache.refresh()  # Failed refresh withdraws readiness; never uses stale values as ready.
        for spec in self.manifest.runners:
            coord, module = self.coordinators[spec.key], self.modules[spec.key]
            if coord._execution_bridge is not None:
                raise RuntimeError("Shadow coordinator acquired an execution bridge")
            bars = self.public.get_closed_bars(spec.symbol, interval=spec.interval, limit=8,
                                               after_ms=self.cursors[spec.key])
            for bar in bars:
                ts = int(bar["start"])
                readiness = coord._data_readiness
                readiness.set_bar_ts(ts)
                if int(readiness.clock() * 1000) - ts > spec.interval_ms + 300_000:
                    self.journal.write("inputs_blocked", runner_key=spec.key, bar_ts_ms=ts,
                                       reason="closed bar expired; fresh warmup required")
                    raise RuntimeError("Shadow closed bar expired; restart with fresh warmup")
                if spec.interval == "60":
                    prediction = _batch_prediction(spec.symbol, spec.model_dir, ts)
                    readiness.set_external_status("batch_inputs", prediction is not None,
                                                  "batch prediction inputs or model unavailable")
                    if prediction is not None:
                        module._batch_pred_override = prediction
                inputs = readiness.snapshot()
                if inputs["ready"] is not True:
                    self.journal.write("inputs_blocked", runner_key=spec.key, bar_ts_ms=ts, inputs=inputs)
                    break  # Retry this bar when local inputs recover; do not skip it silently.
                coord.emit(_market_event(spec.symbol, bar), actor="shadow")
                self.cursors[spec.key] = ts
                self.journal.write("signal_observation", runner_key=spec.key, bar_ts_ms=ts,
                                   inputs=inputs, target_signal=module._target_signal,
                                   strategy_signal=module._signal, balance_source="synthetic",
                                   signal_only=module._signal_only)
        return self.monitor.publish("running")

    def close(self) -> None:
        for coord in self.coordinators.values():
            coord.stop()
        for cache in self.caches.values():
            cache.stop()


def run(manifest: RunManifest, *, max_cycles: int | None = None, stop: threading.Event | None = None) -> None:
    from runner.public_market_data import BybitPublicMarketData
    from runner.runtime_monitor import RuntimeMonitor
    from shared.runtime_state import MANIFEST_NAME, RuntimeStateLease, _path
    from shared.runtime_status import STATUS_NAME, RuntimeStatusWriter, read_json

    state = _path(Path(os.environ.get("RUNTIME_STATE_DIR", "data/runtime")))
    if state != manifest.project_root / "data/runtime":
        raise ValueError("Shadow state must use the project's data/runtime directory")
    journal_path = Path(os.environ.get("ORDER_JOURNAL_PATH", str(state / "orders.sqlite")))
    if _path(Path(os.environ.get("DRAWDOWN_STATE_PATH", str(state / "drawdown.sqlite")))) != state / "drawdown.sqlite":
        raise ValueError("Shadow drawdown state must use its managed runtime directory")
    scope = os.environ.get("RISK_SCOPE", "")
    stop = stop if stop is not None else threading.Event()
    with RuntimeStateLease(state, scope, journal_path=journal_path):
        identity = read_json(state / MANIFEST_NAME)
        release = {key: os.environ.get(name, "") for key, name in (
            ("revision", "QUANT_RELEASE_REVISION"), ("image_id", "QUANT_RELEASE_IMAGE_ID"),
            ("input_digest", "QUANT_RELEASE_INPUT_DIGEST"),
        )}
        writer = RuntimeStatusWriter(
            state / STATUS_NAME, scope=scope, state_id=identity["state_id"], mode="shadow",
            runner_keys=list(manifest.runner_keys), execution_runner_keys=[],
            manifest_digest=manifest.digest, release=release if any(release.values()) else {},
        )
        monitor = RuntimeMonitor(writer, execution=False, execution_runner_keys=[])
        session = ShadowSession(manifest, BybitPublicMarketData(manifest.symbols), monitor,
                                ShadowJournal(state / "shadow-observations.jsonl", manifest))
        monitor.publish("starting")
        try:
            session.start()
            cycles = 0
            while not stop.is_set():
                session.cycle()
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    break
                stop.wait(10)
        finally:
            try:
                monitor.publish("stopping")
            finally:
                session.close()
                monitor.publish("stopped")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("config/shadow.json"))
    parser.add_argument("--dry-run", action="store_true", required=True,
                        help="Required acknowledgement: observations only; no account or execution")
    parser.add_argument("--max-cycles", type=int, help="Stop after this many public polling cycles")
    args = parser.parse_args(argv)
    if args.max_cycles is not None and args.max_cycles < 1:
        parser.error("--max-cycles must be positive")
    manifest = load_run_manifest(args.manifest)
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run(manifest, max_cycles=args.max_cycles, stop=stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
