"""Binance USDⓈ-M runtime: research-identical decisions, durable execution.

    python -m runner.binance_main --models-dir models_binance --symbols BTCUSDT ETHUSDT          # paper
    python -m runner.binance_main --models-dir models_binance --enable-execution                 # testnet
    BINANCE_TESTNET=0 python -m runner.binance_main ... --enable-execution                       # mainnet

Design (each point mirrors a defect found in the 2026-11-13 audit):

- Decisions are built exactly like ``runner.offline.replay``: same frozen
  research model (signed ``model.json``), same incremental features, same
  ``AlphaDecisionModule`` configuration, 4h bars aggregated locally from closed
  1h bars and published BEFORE the 1h decision at a shared close. Each symbol
  runs independently (no cross-symbol rules), as in research.
- Market data always comes from Binance mainnet PUBLIC endpoints (the research
  data source). Closed bars are pulled by REST from the last processed bar, so
  a disconnect can never skip a bar and everything runs in ONE thread.
- Paper mode (default) fills with the research broker model (``BarBroker``).
  Execution mode uses durable intents (``BinanceExecutionAdapter``), exchange
  side protective stops, a 3x gross exposure cap, a fail-closed kill-switch
  file, a peak-based drawdown latch that also flattens, and exit retries.
- Mainnet execution additionally requires each model's research gate to have
  passed (``gate.json``); testnet execution is allowed for engineering tests.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import time
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

logger = logging.getLogger("binance_main")

HOUR = 3_600_000
FOUR_HOURS = 4 * HOUR
WARMUP_BARS = 3000           # same bounded warmup as the research replays
POLL_SECONDS = 5.0
RECONCILE_SECONDS = 5.0
EQUITY_SECONDS = 60.0
EQUITY_MAX_AGE = 120.0
MAX_ENTRY_DELAY_MS = 10 * 60_000   # entries only from bars processed within 10 min of close
FUNDING_WAIT_MS = 2 * 60_000       # wait this long for a settlement published at the close
EXIT_RETRY_SECONDS = 30.0
EXIT_RETRY_LIMIT = 5
TERMINAL_UNFILLED = {"blocked", "not_found", "rejected", "canceled", "expired", "mmp_canceled"}


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def load_model_dir(model_dir: Path, symbol: str, now_ms: int) -> tuple[dict, dict, dict]:
    """Signed research artifacts for one symbol: (bundle, final_config, gate)."""
    from infra.model_signing import load_verified_json
    from runner.offline.models import validate_bundle

    base = Path(model_dir) / symbol
    bundle = load_verified_json(base / "model.json")
    protocol = load_verified_json(base / "protocol.json")
    gate = load_verified_json(base / "gate.json")
    if protocol.get("symbol") != symbol or protocol.get("final_model_sha256") != bundle.get("sha256"):
        raise ValueError(f"{symbol}: protocol does not describe this model")
    validate_bundle(bundle, now_ms, symbol=symbol)
    if "final_config" not in protocol:
        raise ValueError(f"{symbol}: protocol has no frozen final_config")
    return bundle, protocol["final_config"], gate


# ----------------------------------------------------------------------------
# Small gates and helpers
# ----------------------------------------------------------------------------
class EquitySource:
    def __init__(self, clock=time.time) -> None:
        self.value, self.at, self.clock = D(0), 0.0, clock

    def update(self, equity: D) -> None:
        self.value, self.at = D(str(equity)), self.clock()

    def current(self) -> D:
        if self.value <= 0 or self.clock() - self.at > EQUITY_MAX_AGE:
            return D(0)  # stale equity sizes to zero
        return self.value


class EntryReadinessGate:
    """Blocks new exposure unless the bar that produced it had complete, fresh inputs."""

    def __init__(self, session: "SymbolSession", inner: Any) -> None:
        self.session, self.inner = session, inner

    def check(self, event: Any) -> Any:
        from shared.entry_risk import GateResult
        try:
            qty = D(str(event.qty))
            reducing = qty == 0 or getattr(event, "reduce_only", False) is True
            if not reducing and not self.session.entry_ready:
                return GateResult(False, "inputs: " + self.session.entry_reason)
            return self.inner.check(event)
        except Exception as exc:
            return GateResult(False, f"entry gate failed: {type(exc).__name__}")


class DecisionAdapter:
    """Injects the account equity used for sizing (as the research replay does)."""

    def __init__(self, module: Any, equity: Any) -> None:
        self.module, self.equity = module, equity

    def decide(self, snapshot: Any):
        equity = self.equity()
        return self.module.decide(replace(snapshot, account=NS(balance=equity, balance_f=float(equity))))

    def on_fill(self, event: Any) -> None:
        self.module.on_fill(event)


class EventJournal:
    """Append-only JSONL of signals, intents, fills and issues (observability only)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, **fields: Any) -> None:
        row = {"ts": time.time(), "kind": kind, **fields}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")


def bar_from_kline(row: dict):
    from data.quality.causal_bars import Bar
    start = int(row["start"])
    if start % HOUR:
        raise ValueError(f"kline open time {start} is not hour-aligned")
    extras = None
    if all(k in row for k in ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume")):
        extras = {k: D(str(row[k])) for k in ("quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume")}
    return Bar(start, *(D(str(row[k])) for k in ("open", "high", "low", "close", "volume")), extras=extras)


# ----------------------------------------------------------------------------
# One symbol: 4h signal runner + 1h trading runner
# ----------------------------------------------------------------------------
class SymbolSession:
    def __init__(self, symbol: str, bundle: dict, cfg: Any, *, mode: str, context: Any, needs_metrics: bool,
                 equity: Any, journal: EventJournal, leverage: float, step: D,
                 execution: dict | None = None, paper_dir: Path | None = None) -> None:
        from _quant_hotpath import RustInferenceBridge
        from decision.modules.alpha import AlphaDecisionModule
        from decision.signals.alpha_signal import SignalDiscretizer
        from decision.sizing.adaptive import AdaptivePositionSizer
        from engine.coordinator import CoordinatorConfig, EngineCoordinator
        from engine.decision_bridge import DecisionBridge
        from engine.execution_bridge import ExecutionBridge
        from runner.offline.models import FrozenPredictor
        from runner.offline.replay import FeatureHook

        self.symbol, self.cfg, self.mode, self.journal = symbol, cfg, mode, journal
        self.context, self.needs_metrics = context, needs_metrics
        self.entry_ready, self.entry_reason = False, "no live bar processed yet"
        self.cursor: int | None = None      # open time of the last processed 1h bar
        self.recent: list = []               # last closed 1h bars for 4h aggregation
        self.exit_retry: dict | None = None
        self.consensus: dict = {}
        self.hooks, self.modules, self.coords = {}, {}, {}
        equity_fn = (lambda: self.broker.equity) if mode == "paper" else equity.current
        for interval in ("4h", "1h"):
            self.hooks[interval] = FeatureHook(symbol, context.context if context is not None else None)
            key = f"{symbol}_4h" if interval == "4h" else symbol
            module = AlphaDecisionModule(
                symbol=symbol, runner_key=key, predictor=FrozenPredictor(bundle["models"][interval]),
                discretizer=SignalDiscretizer(RustInferenceBridge(cfg.zscore_window, cfg.zscore_warmup),
                                              symbol=symbol, deadzone=cfg.deadzone, min_hold=cfg.min_hold,
                                              max_hold=cfg.max_hold),
                sizer=AdaptivePositionSizer(key, step_size=float(step), min_size=float(step)),
                leverage=leverage, signal_only=interval == "4h", venue="binance", offline=True)
            module.set_consensus(self.consensus)
            module._execution_ready = False  # no orders during warmup
            self.modules[interval] = module
            coord = EngineCoordinator(cfg=CoordinatorConfig(symbol_default=symbol, symbols=(symbol,),
                                                            starting_balance=1.0, feature_hook=self.hooks[interval]))
            coord.attach_decision_bridge(DecisionBridge(self._emitter(coord, interval),
                                                        [DecisionAdapter(module, equity_fn)]))
            self.coords[interval] = coord
        module, coord = self.modules["1h"], self.coords["1h"]
        self.broker = self.executor = self.stops = None
        if mode == "paper":
            self._build_paper(module, coord, paper_dir, ExecutionBridge)
        else:
            self._build_execution(module, coord, execution, ExecutionBridge)
        for c in self.coords.values():
            c.start()

    def _emitter(self, coord, interval):
        from event.types import EventType

        def emit(event):
            kind = getattr(event, "event_type", None)
            if kind == EventType.SIGNAL:
                self.journal.write("signal", symbol=self.symbol, tf=interval, side=event.side,
                                   strength=str(event.strength), live=self.cursor is not None and self.entry_ready)
            elif kind == EventType.ORDER:
                self.journal.write("intent", symbol=self.symbol, order_id=event.order_id, side=event.side,
                                   qty=str(event.qty), price=str(event.price), reason=getattr(event, "reason", None))
            coord.emit(event)
        return emit

    def _build_paper(self, module, coord, paper_dir, ExecutionBridge):
        from execution.safety.limits import OrderLimiter, OrderLimitsConfig
        from runner.offline.broker import BarBroker
        from runner.offline.replay import HistoricalRisk
        from shared.drawdown_guard import DrawdownGuard
        from shared.entry_risk import CompositeRiskGate

        self.broker = BarBroker(self.cfg)
        guard = DrawdownGuard(Path(paper_dir) / f"paper-risk-{self.symbol}.sqlite", f"paper-{self.symbol}",
                              daily_pct=self.cfg.daily_loss_pct, weekly_pct=self.cfg.weekly_loss_pct,
                              max_age_seconds=1)
        self._paper_guard = guard
        risk = HistoricalRisk(self.broker, guard)
        gate = CompositeRiskGate(OrderLimiter(OrderLimitsConfig()), portfolio_reader=risk,
                                 drawdown_guard=risk, venue="paper")
        coord.attach_execution_bridge(ExecutionBridge(self.broker, coord.emit, risk_gate=EntryReadinessGate(self, gate),
                                                      max_retries=0, on_order_status=self.broker.status_update))
        self.broker.status = module.on_order_status
        self.broker.emit = coord.emit

    def _build_execution(self, module, coord, execution, ExecutionBridge):
        from engine.durable_fills import DurableFillConsumer
        from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
        from execution.adapters.binance.protective_stops import ProtectiveStopManager

        self.executor = BinanceExecutionAdapter(
            execution["account"], owner=f"alpha.{self.symbol}", scope=execution["scope"],
            journal_path=execution["journal_path"], limit_offset_bps=D(str(self.cfg.limit_offset_bps)),
            ttl_bars=int(self.cfg.ttl_bars))
        bridge = ExecutionBridge(self.executor, coord.emit, risk_gate=EntryReadinessGate(self, execution["gate"]),
                                 on_order_status=self._on_status, max_retries=0)
        coord.attach_execution_bridge(bridge)
        self.executor._status_callback = self._on_status
        coord._durable_fill_consumer = DurableFillConsumer(coord, module, self.executor)
        self.stops = ProtectiveStopManager(execution["account"], self.executor, self.symbol,
                                           stop_pct=execution["stop_pct"])

    # -- order status: detect exits that did not complete -------------------
    def _on_status(self, order_id: str, status: str) -> None:
        module = self.modules["1h"]
        was_reduction = order_id in module._pending_reductions
        reason = module._pending_exit_reasons.get(order_id)
        module.on_order_status(order_id, status)
        if was_reduction and status in TERMINAL_UNFILLED:
            self.exit_retry = self.exit_retry or {"reason": reason or "exit", "attempts": 0, "next": 0.0}
            self.journal.write("exit_unfilled", symbol=self.symbol, order_id=order_id, status=status)

    # -- bars ---------------------------------------------------------------
    def process_bar(self, bar, *, live: bool, now_ms: int) -> None:
        """Feed one CLOSED 1h bar (in order); publishes a completed 4h bar first."""
        from data.quality.causal_bars import aggregate_closed
        from runner.offline.replay import market_event

        if self.cursor is not None and bar.open_time != self.cursor + HOUR:
            raise RuntimeError(f"{self.symbol}: non-contiguous bar {bar.open_time} after {self.cursor}")
        if self.broker is not None:
            self.broker.now, self.broker.mark = bar.open_time, bar.open
            if self.context is not None and bar.open_time % (8 * HOUR) == 0 and live:
                rate = self.context.context.at(bar.open_time)["funding_rate"]
                if rate is not None:
                    self.broker.settle_funding(bar.open_time, D(str(rate)), bar.open)
            self.broker.process_bar(bar)
            self._paper_guard.observe(float(self.broker.equity), now=self.broker.now / 1000)
        # Entry readiness for THIS bar: live, fresh, complete inputs.
        if live:
            reasons = []
            if now_ms - bar.available_at > MAX_ENTRY_DELAY_MS:
                reasons.append("bar processed too late")
            if self.context is not None:
                ok, why = self.context.ready(bar.available_at, needs_metrics=self.needs_metrics)
                if not ok:
                    reasons.append(why)
            self.entry_ready, self.entry_reason = not reasons, "; ".join(reasons)
            if self.mode == "paper":
                self.modules["1h"]._execution_ready = True
        self.recent = (self.recent + [bar])[-4:]
        if bar.available_at % FOUR_HOURS == 0 and len(self.recent) == 4:
            groups = aggregate_closed(self.recent)
            if groups and groups[-1].available_at == bar.available_at:
                self.hooks["4h"].bar = groups[-1]
                self.coords["4h"].emit(market_event(groups[-1], "4h", self.symbol), actor="live" if live else "warmup")
        self.hooks["1h"].bar = bar
        self.coords["1h"].emit(market_event(bar, "1h", self.symbol), actor="live" if live else "warmup")
        self.cursor = bar.open_time

    # -- actions ------------------------------------------------------------
    def close_position(self, reason: str) -> bool:
        module = self.modules["1h"]
        if module._signal == 0 or module._pending_reductions:
            return False
        price = float(self.recent[-1].close) if self.recent else 0.0
        for order in module._make_close_order(price, module._signal, reason):
            self.journal.write("intent", symbol=self.symbol, order_id=order.order_id, side=order.side,
                               qty="0", reason=reason)
            self.coords["1h"].emit(order, actor="risk")
        return True


# ----------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------
class BinanceRuntime:
    def __init__(self, *, symbols: list[str], models_dir: Path, state_dir: Path, mode: str,
                 market: Any, account: Any = None, data_dir: Path | None = None,
                 clock=time.time, testnet: bool = True, stop_pct: D = D("0.04"),
                 paper_config: dict | None = None) -> None:
        from runner.offline.broker import ReplayConfig
        from runner.binance_context import LiveDerivativesContext
        from runner.offline.models import FEATURE_SETS, normalize_spec

        if mode not in ("paper", "execution"):
            raise ValueError("mode must be paper or execution")
        self.mode, self.market, self.account, self.clock = mode, market, account, clock
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.journal = EventJournal(self.state_dir / "binance_events.jsonl")
        self.equity = EquitySource(clock)
        self.sessions: dict[str, SymbolSession] = {}
        self._last_reconcile = self._last_equity = 0.0
        now_ms = int(clock() * 1000)
        execution = None
        if mode == "execution":
            execution = self._build_execution_context(symbols, testnet, stop_pct)
        for symbol in symbols:
            bundle, final_config, gate = load_model_dir(models_dir, symbol, now_ms)
            if mode == "execution" and not testnet and gate.get("passed") is not True:
                raise RuntimeError(f"{symbol}: research gate not passed; mainnet execution refused "
                                   f"({'; '.join(gate.get('reasons', []))})")
            cfg = ReplayConfig(**{**final_config, **(paper_config or {})}) if mode == "paper" \
                else ReplayConfig(**final_config)
            spec = normalize_spec(bundle.get("spec"))
            needs_metrics = any(f not in FEATURE_SETS["ohlcv"] for f in FEATURE_SETS[spec["feature_set"]])
            # Research replays always carried the derivatives context (and funding cash flows).
            context = LiveDerivativesContext(market, symbol, data_dir)
            step = account.filters(symbol).step_size if account is not None else D(cfg.qty_step)
            self.sessions[symbol] = SymbolSession(
                symbol, bundle, cfg, mode=mode, context=context, needs_metrics=needs_metrics,
                equity=self.equity, journal=self.journal, leverage=float(execution["leverage"]) if execution else 1.0,
                step=step, execution=execution, paper_dir=self.state_dir)
        self.kill = execution["kill"] if execution else None
        self.drawdown = execution["drawdown"] if execution else None

    def _build_execution_context(self, symbols, testnet, stop_pct) -> dict:
        from execution.safety.limits import OrderLimiter, OrderLimitsConfig
        from shared.drawdown_guard import DrawdownGuard
        from shared.entry_risk import CompositeRiskGate
        from shared.exposure_gate import MAX_TOTAL_LEVERAGE, AccountExposureGate
        from shared.kill_file import KillFile
        from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter  # noqa: F401

        scope = os.environ.get("RISK_SCOPE", "").strip()
        if not scope:
            raise RuntimeError("Execution requires RISK_SCOPE (account + environment identity)")
        if (("testnet" in scope) != testnet) or ("binance" not in scope):
            raise RuntimeError("RISK_SCOPE must name binance and match the environment (contain 'testnet' "
                               "exactly when BINANCE_TESTNET=1)")
        journal_path = self.state_dir / "orders.sqlite"
        kill = KillFile(self.state_dir / "kill_switch.json")
        drawdown = DrawdownGuard(self.state_dir / "drawdown.sqlite", scope,
                                 daily_pct=float(os.environ.get("MAX_DAILY_DRAWDOWN_PCT", "3")),
                                 weekly_pct=float(os.environ.get("MAX_WEEKLY_DRAWDOWN_PCT", "8")))
        limiter = OrderLimiter(OrderLimitsConfig(
            max_order_notional=D(os.environ.get("BINANCE_MAX_ORDER_NOTIONAL", "1000")),
            max_daily_notional=D(os.environ.get("BINANCE_MAX_DAILY_NOTIONAL", "10000")),
            max_daily_orders=int(os.environ.get("BINANCE_MAX_DAILY_ORDERS", "50")),
            max_orders_per_second=1.0))
        from shared.execution_journal import ExecutionJournal
        exposure = AccountExposureGate(self.account, ExecutionJournal(journal_path, scope),
                                       max_leverage=min(MAX_TOTAL_LEVERAGE,
                                                        D(os.environ.get("BINANCE_MAX_LEVERAGE", "3"))))
        gate = CompositeRiskGate(limiter, kill_switch=kill, drawdown_guard=drawdown, portfolio_reader=exposure,
                                 venue="binance")
        return {"account": self.account, "scope": scope, "journal_path": journal_path, "gate": gate,
                "kill": kill, "drawdown": drawdown, "stop_pct": stop_pct, "leverage": exposure.max_leverage}

    # -- lifecycle ----------------------------------------------------------
    def warmup(self) -> None:
        now_ms = int(self.clock() * 1000)
        for symbol, session in self.sessions.items():
            if session.context is not None:
                session.context.refresh()
            rows = self.market.get_klines(symbol, "60", WARMUP_BARS, now_ms=now_ms)
            bars = [bar_from_kline(r) for r in rows]
            if len(bars) < WARMUP_BARS:
                raise RuntimeError(f"{symbol}: incomplete warmup ({len(bars)}/{WARMUP_BARS} bars)")
            for bar in bars:
                session.process_bar(bar, live=False, now_ms=now_ms)
            self.journal.write("warmup", symbol=symbol, bars=len(bars), last_open=session.cursor)
        if self.mode == "execution":
            self.refresh_equity()
            self.reconcile(restore=True)

    def refresh_equity(self) -> None:
        if self.account is None:
            return
        try:
            equity = self.account.get_account_equity()
            self.equity.update(D(str(equity)))
            allowed, reason = self.drawdown.observe(float(equity))
            if not allowed:
                logger.warning("New entries paused: %s", reason)
        except Exception:
            self.drawdown.invalidate("exchange equity unavailable")
            logger.warning("Equity refresh failed; entries blocked", exc_info=True)
        self._last_equity = self.clock()

    def reconcile(self, *, restore: bool = False) -> None:
        from runner.execution_recovery import reconcile_execution
        for symbol, session in self.sessions.items():
            if session.executor is None:
                continue
            try:
                issues = reconcile_execution(session.coords["1h"], session.modules["1h"], restore=restore)
                issues += session.stops.sync()
            except Exception as exc:
                session.modules["1h"]._execution_ready = False
                issues = [f"reconciliation failed: {type(exc).__name__}: {exc}"]
                logger.exception("%s reconciliation failed; entries blocked", symbol)
            for issue in issues:
                self.journal.write("issue", symbol=symbol, issue=issue)
            if any("stop_breached" in issue for issue in issues):
                session.close_position("stop_breached")
        self._last_reconcile = self.clock()

    def risk_actions(self) -> None:
        """Flatten on kill(flatten) or drawdown latch; retry exits that did not complete."""
        if self.mode != "execution":
            return
        kill = self.kill.state()
        flatten = kill.active and kill.flatten
        latched = self.drawdown.latched()
        now = self.clock()
        for session in self.sessions.values():
            module = session.modules["1h"]
            if module._signal != 0 and (flatten or latched) and not module._pending_orders:
                if session.close_position("kill_switch" if flatten else "drawdown_latched"):
                    self.journal.write("flatten", symbol=session.symbol, kill=flatten, drawdown=latched)
            retry = session.exit_retry
            if retry is None:
                continue
            if module._signal == 0:
                session.exit_retry = None
            elif not module._pending_orders and now >= retry["next"]:
                if retry["attempts"] >= EXIT_RETRY_LIMIT:
                    self.journal.write("exit_retry_exhausted", symbol=session.symbol)
                    self.kill.write(active=True, flatten=False,
                                    reason=f"{session.symbol}: exit failed {EXIT_RETRY_LIMIT} times")
                    session.exit_retry = None
                    continue
                retry["attempts"] += 1
                retry["next"] = now + EXIT_RETRY_SECONDS
                session.close_position(f"{retry['reason']}_retry")

    def poll_bars(self) -> int:
        processed = 0
        now_ms = int(self.clock() * 1000)
        for symbol, session in self.sessions.items():
            expected_open = session.cursor + HOUR
            if now_ms < expected_open + HOUR:
                continue  # next bar has not closed yet
            if session.context is not None:
                session.context.refresh()
                close = expected_open + HOUR
                waiting = close % (8 * HOUR) == 0 and (session.context.context.funding_end_ms or 0) < close
                if waiting and now_ms - close < FUNDING_WAIT_MS:
                    continue  # settlement at this close not published yet; retry shortly
            missing = (now_ms // HOUR * HOUR - expected_open) // HOUR
            rows = self.market.get_klines(symbol, "60", int(min(max(missing, 1) + 2, 1500)), now_ms=now_ms)
            for row in rows:
                bar = bar_from_kline(row)
                if bar.open_time <= session.cursor:
                    continue
                session.process_bar(bar, live=True, now_ms=now_ms)
                processed += 1
        return processed

    def status(self) -> dict:
        out = {"mode": self.mode, "at": self.clock(), "equity": str(self.equity.value), "symbols": {}}
        if self.kill is not None:
            out["kill"] = self.kill.state().__dict__
            out["drawdown_latched"] = self.drawdown.latched()
        for symbol, s in self.sessions.items():
            m = s.modules["1h"]
            out["symbols"][symbol] = {
                "last_bar_open": s.cursor, "entry_ready": s.entry_ready, "entry_reason": s.entry_reason,
                "signal": m._signal, "target_signal": m._target_signal, "qty": str(m._current_qty),
                "pending": sorted(m._pending_orders), "execution_ready": m._execution_ready,
                "protective_stop": s.stops.has_active_stop() if s.stops else None,
                "paper_equity": str(s.broker.equity) if s.broker else None}
        return out

    def cycle(self) -> None:
        now = self.clock()
        if self.mode == "execution" and now - self._last_equity >= EQUITY_SECONDS:
            self.refresh_equity()  # sizing needs equity no older than EQUITY_MAX_AGE
        self.poll_bars()
        if self.mode == "execution":
            if self.clock() - self._last_reconcile >= RECONCILE_SECONDS:
                self.reconcile()
            self.risk_actions()
        tmp = self.state_dir / "binance_status.json.tmp"
        tmp.write_text(json.dumps(self.status(), default=str, indent=2))
        os.replace(tmp, self.state_dir / "binance_status.json")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_adapters(*, execution: bool, testnet: bool, symbols: list[str]):
    from execution.adapters.binance import BinanceAdapter, BinanceConfig
    market = BinanceAdapter(BinanceConfig(testnet=False))  # public mainnet data only (no keys)
    account = None
    if execution:
        account = BinanceAdapter(BinanceConfig.from_env(testnet=testnet))
        if not account.connect(symbols=tuple(symbols), configure_leverage=True):
            raise RuntimeError("Binance account connection failed")
        blockers = account.execution_blockers()
        if blockers:
            raise RuntimeError("Execution blocked: " + "; ".join(blockers))
    return market, account


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=Path("data/runtime/binance"))
    parser.add_argument("--data-dir", type=Path, default=Path("data_files/binance_public"),
                        help="binance_public download dir (derivatives history for warmup)")
    parser.add_argument("--stop-pct", type=D, default=D("0.04"), help="Exchange-side protective stop distance")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="execution", action="store_false", help="Paper trading (default)")
    mode.add_argument("--enable-execution", dest="execution", action="store_true")
    parser.set_defaults(execution=False)
    parser.add_argument("--max-cycles", type=int)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    testnet = os.environ.get("BINANCE_TESTNET", "1").strip().lower() not in ("0", "false", "no", "off")
    if args.execution and not testnet:
        logger.warning("MAINNET EXECUTION requested: research gates will be enforced")
    from shared.execution_lease import ExecutionLease
    state_dir = args.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    lease = ExecutionLease(state_dir / "orders.sqlite",
                           os.environ.get("RISK_SCOPE", "paper") if args.execution else f"paper-{os.getpid()}")
    market, account = build_adapters(execution=args.execution, testnet=testnet, symbols=args.symbols)
    runtime = BinanceRuntime(symbols=args.symbols, models_dir=args.models_dir, state_dir=state_dir,
                             mode="execution" if args.execution else "paper", market=market, account=account,
                             data_dir=args.data_dir, testnet=testnet, stop_pct=args.stop_pct)
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    try:
        runtime.warmup()
        cycles = 0
        while not stop["flag"]:
            runtime.cycle()
            cycles += 1
            if args.max_cycles is not None and cycles >= args.max_cycles:
                break
            time.sleep(POLL_SECONDS)
    finally:
        lease.close()


if __name__ == "__main__":
    main()


__all__ = ["BinanceRuntime", "SymbolSession", "load_model_dir", "bar_from_kline", "math"]
