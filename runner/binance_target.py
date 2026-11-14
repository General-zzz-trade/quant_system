"""Binance BTC/ETH target-position runtime (1h clock, 4h/daily features).

    python -m runner.binance_target --spec models_binance/strategy/final_spec.json            # paper
    python -m runner.binance_target --spec ... --enable-execution                             # testnet

Each closed 1h bar, per symbol:
1. advance the sleeve (paper: research BarBroker; execution: consume confirmed
   exchange fills durably),
2. compute the target weight with ``lab.strategy.latest_target`` on a trailing
   window (exactly the research value),
3. rebalance with ``lab.rebalance.plan`` (band, reduce-only vs entry),
4. send intents through the gates (inputs fresh, kill switch, drawdown,
   3x gross exposure, order limits) to the durable Binance executor.

Paper mode reproduces ``lab.simulate`` fills exactly (tested). Mainnet
execution requires the frozen spec's research gate to have passed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
from collections import deque
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from lab.data import frame_from_bars
from lab.orders import make_order
from lab.rebalance import plan
from lab.simulate import SYMBOL_RULES, Sleeve, SymbolRules, broker_config
from lab.strategy import StrategySpec, latest_target
from runner.binance_main import EquitySource, EventJournal, bar_from_kline

logger = logging.getLogger("binance_target")

HOUR = 3_600_000
FUNDING_INTERVAL = 8 * HOUR
POLL_SECONDS = 5.0
RECONCILE_SECONDS = 5.0
EQUITY_SECONDS = 60.0
MAX_ENTRY_DELAY_MS = 10 * 60_000
FUNDING_WAIT_MS = 2 * 60_000
MARKET_BACKOFF_MAX = 300.0
PAPER_STATE = "paper_state.json"
# BarBroker attributes that change while trading (config and callbacks are rebuilt from the spec).
_BROKER_STATE = ("fees", "funding", "mark", "now", "orders", "fills", "funding_log", "margin", "stop_price",
                 "liquidations")


def _encode(value: Any) -> Any:
    if isinstance(value, D):
        return {"__decimal__": str(value)}
    raise TypeError(f"cannot persist {type(value).__name__}")


def _decode(obj: dict) -> Any:
    return D(obj["__decimal__"]) if obj.keys() == {"__decimal__"} else obj


def load_spec(path: Path) -> dict:
    from infra.model_signing import load_verified_json
    spec = load_verified_json(path)
    if spec.get("kind") != "btc_eth_multitimeframe_target" or spec.get("schema") != 1:
        raise ValueError("not a frozen BTC/ETH target-strategy spec")
    StrategySpec.from_record(spec["spec"])  # validates parameters
    return spec


class TargetFillConsumer:
    """Durable consumption of confirmed fills into the owner's position checkpoint."""

    def __init__(self, executor: Any) -> None:
        from shared.strategy_position import StrategyPosition
        self.executor, self.journal, self.owner = executor, executor.journal, executor.owner
        saved = self.journal.checkpoint(self.owner)
        self.position = StrategyPosition.from_dict(saved["position"]) if saved else StrategyPosition()
        self.revision = saved["revision"] if saved else None

    def consume(self, event: Any) -> bool:
        from shared.strategy_position import StrategyPosition
        fact = self.executor.fill_fact(event.fill_id)
        if (str(event.order_id) != fact["intent_id"] or event.symbol != fact["symbol"] or event.side != fact["side"]
                or event.qty != D(fact["qty"]) or event.price != D(fact["price"])):
            raise ValueError("Event differs from its durable exchange fact")
        before = self.position.to_dict()

        def apply() -> dict:
            self.position.apply(SimpleNamespace(**fact))
            return {"position": self.position.to_dict(), "revision": event.fill_id, "strategy": {}}

        def rollback() -> None:
            self.position = StrategyPosition.from_dict(before)

        changed = self.journal.consume(self.owner, event.fill_id, apply, rollback, expected_revision=self.revision)
        if changed:
            self.revision = event.fill_id
        return changed


class TargetSession:
    def __init__(self, symbol: str, spec: StrategySpec, *, mode: str, journal: EventJournal,
                 rules: SymbolRules, execution: dict | None = None, allocation: D = D("0.5")) -> None:
        self.symbol, self.spec, self.mode, self.journal, self.rules = symbol, spec, mode, journal, rules
        self.allocation = allocation
        self.bars: deque = deque(maxlen=spec.window_hours() + 48)
        self.funding: dict[int, float] = {}
        self.cursor: int | None = None
        self.entry_ready, self.entry_reason = False, "no live bar yet"
        self.last_target: tuple[float, bool] = (0.0, False)
        self.sleeve = self.executor = self.consumer = self.stops = None
        self.gate = None
        if mode == "paper":
            self.sleeve = Sleeve(symbol, spec, rules, broker_config(spec, rules))
        else:
            from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
            from execution.adapters.binance.protective_stops import ProtectiveStopManager
            from runner.binance_main import EntryReadinessGate
            self.executor = BinanceExecutionAdapter(
                execution["account"], owner=f"target.{symbol}", scope=execution["scope"],
                journal_path=execution["journal_path"], entry_mode=spec.execution, ttl_bars=spec.ttl_bars)
            self.executor._status_callback = self._on_status
            self.consumer = TargetFillConsumer(self.executor)
            self.stops = ProtectiveStopManager(execution["account"], self.executor, symbol,
                                               stop_pct=execution["stop_pct"])
            self.gate = EntryReadinessGate(self, execution["gate"])
            self.equity = execution["equity"]
        self.pending: dict[str, bool] = {}   # working order id -> reduce_only

    # -- data ---------------------------------------------------------------
    def funding_series(self, upto: int | None = None) -> pd.Series:
        """Settlements known at ``upto`` (all when None); a catch-up must not see later ones."""
        times = sorted(t for t in self.funding if upto is None or t <= upto)[-self.spec.funding_window():]
        return pd.Series([self.funding[t] for t in times], index=pd.Index(times, dtype="int64"), dtype=float)

    def _target(self) -> tuple[float, bool]:
        return latest_target(frame_from_bars(list(self.bars)), self.funding_series(self.bars[-1].available_at),
                             self.spec)

    # -- paper persistence --------------------------------------------------
    def paper_snapshot(self) -> dict:
        b = self.sleeve.broker
        return {"cursor": self.cursor, "config": b.config.record(), "orders_sent": self.sleeve.orders,
                "position": b.position.to_dict(), "pending": [row["id"] for row in b.pending],
                "broker": {key: getattr(b, key) for key in _BROKER_STATE}}

    def restore_paper(self, snap: dict) -> None:
        from shared.strategy_position import StrategyPosition
        b = self.sleeve.broker
        if snap["config"] != b.config.record():
            raise RuntimeError(f"{self.symbol}: saved paper state used another broker configuration")
        for key in _BROKER_STATE:
            setattr(b, key, snap["broker"][key])
        b.position = StrategyPosition.from_dict(snap["position"])
        by_id = {row["id"]: row for row in b.orders}
        b.pending = [by_id[i] for i in snap["pending"]]   # pending rows are the same objects as order rows
        self.sleeve.orders = snap["orders_sent"]
        self._logged = len(b.fills)

    # -- execution callbacks ------------------------------------------------
    def _on_status(self, order_id: str, status: str) -> None:
        if status in {"filled", "canceled", "mmp_canceled", "rejected", "expired", "blocked", "not_found"}:
            self.pending.pop(order_id, None)

    def position_qty(self) -> D:
        if self.sleeve is not None:
            return self.sleeve.broker.position.qty
        return self.consumer.position.qty

    def sleeve_equity(self) -> D:
        if self.sleeve is not None:
            return self.sleeve.broker.equity
        return self.equity.current() * self.allocation

    def consume_fills(self) -> None:
        for event in self.executor.reconcile_pending():
            if self.consumer.consume(event):
                self.journal.write("fill", symbol=self.symbol, order_id=event.order_id, side=event.side,
                                   qty=str(event.qty), price=str(event.price))

    def submit(self, intents, bar, reason: str) -> None:
        for k, intent in enumerate(intents):
            order = make_order(order_id=f"{self.symbol}:{bar.available_at}:{reason}:{k}",
                               source=f"target.{self.symbol}", symbol=self.symbol, side=intent.side, qty=intent.qty,
                               price=bar.close, reduce_only=intent.reduce_only, reason=reason)
            self.journal.write("intent", symbol=self.symbol, order_id=order.order_id, side=intent.side,
                               qty=str(intent.qty), reduce_only=intent.reduce_only, reason=reason)
            if self.sleeve is not None:
                self.sleeve.broker.send_order(order)
                continue
            check = self.gate.check(order)
            if not check.allowed:
                self.journal.write("blocked", symbol=self.symbol, order_id=order.order_id, reason=check.reason)
                continue
            self.pending[order.order_id] = intent.reduce_only
            self.executor.send_order(order)
            self.executor.notify_status()
            self.consume_fills()

    # -- the hourly step ----------------------------------------------------
    def process_bar(self, bar, *, live: bool, now_ms: int) -> None:
        if self.cursor is not None and bar.open_time != self.cursor + HOUR:
            raise RuntimeError(f"{self.symbol}: non-contiguous bar {bar.open_time} after {self.cursor}")
        self.bars.append(bar)
        self.cursor = bar.open_time
        if not live:
            return
        if self.sleeve is not None:
            rate = self.funding.get(bar.open_time)
            self.sleeve.on_bar(bar, D(str(rate)) if rate is not None else None)
        reasons = []
        if now_ms - bar.available_at > MAX_ENTRY_DELAY_MS:
            reasons.append("bar processed too late")
        latest_settlement = bar.available_at // FUNDING_INTERVAL * FUNDING_INTERVAL
        if latest_settlement not in self.funding:
            reasons.append("funding settlement at or before this close not published yet")
        self.entry_ready, self.entry_reason = not reasons, "; ".join(reasons)
        target, ready = self._target()
        self.last_target = (target, ready)
        if self.sleeve is not None:
            # Paper: exactly the research sleeve step (no readiness gate in research).
            self.sleeve.decide(bar, target, ready)
            for fill in self.sleeve.broker.fills[getattr(self, "_logged", 0):]:
                self.journal.write("paper_fill", symbol=self.symbol, **fill)
            self._logged = len(self.sleeve.broker.fills)
            self.journal.write("paper_equity", symbol=self.symbol, bar_open=bar.open_time,
                               equity=str(self.sleeve.broker.equity), qty=str(self.sleeve.broker.position.qty),
                               target=target)
            return
        if not ready:
            return
        intents = plan(target, equity=self.sleeve_equity(), price=bar.close, current_qty=self.position_qty(),
                       entry_pending=any(not reduce_only for reduce_only in self.pending.values()),
                       band=self.spec.band, step=self.rules.step, min_qty=self.rules.min_qty,
                       min_notional=self.rules.min_notional)
        self.submit(intents, bar, "rebalance")

    def flatten(self, reason: str) -> None:
        qty = self.position_qty()
        if qty == 0 or self.pending or not self.bars:
            return
        from lab.rebalance import Intent
        self.submit([Intent("sell" if qty > 0 else "buy", abs(qty), True)], self.bars[-1], reason)


class TargetRuntime:
    def __init__(self, *, spec_path: Path, state_dir: Path, mode: str, market: Any, account: Any = None,
                 clock=time.time, testnet: bool = True, stop_pct: D = D("0.08"), reset_paper: bool = False) -> None:
        if mode not in ("paper", "execution"):
            raise ValueError("mode must be paper or execution")
        self.frozen = load_spec(spec_path)
        self.spec = StrategySpec.from_record(self.frozen["spec"])
        if mode == "execution" and not testnet and self.frozen.get("passed") is not True:
            raise RuntimeError("research gate not passed; mainnet execution refused")
        self.mode, self.market, self.account, self.clock = mode, market, account, clock
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.journal = EventJournal(self.state_dir / "target_events.jsonl")
        self.equity = EquitySource(clock)
        self._last_equity = self._last_reconcile = 0.0
        self._market_failures, self._market_retry_at = 0, 0.0
        self.reset_paper, self._saved_cursors = reset_paper, None
        execution = self._execution_context(testnet, stop_pct) if mode == "execution" else None
        symbols = self.frozen["symbols"]
        allocation = D(1) / D(len(symbols))
        self.sessions = {}
        for symbol in symbols:
            if account is not None:
                f = account.filters(symbol)
                rules = SymbolRules(f.step_size, f.min_qty, max(f.min_notional, SYMBOL_RULES[symbol].min_notional))
            else:
                rules = SYMBOL_RULES[symbol]
            self.sessions[symbol] = TargetSession(symbol, self.spec, mode=mode, journal=self.journal, rules=rules,
                                                  execution=execution, allocation=allocation)
        self.kill = execution["kill"] if execution else None
        self.drawdown = execution["drawdown"] if execution else None

    def _execution_context(self, testnet: bool, stop_pct: D) -> dict:
        from runner.binance_main import check_stop_inside_liquidation
        check_stop_inside_liquidation(stop_pct)
        from execution.safety.limits import OrderLimiter, OrderLimitsConfig
        from shared.drawdown_guard import DrawdownGuard
        from shared.entry_risk import CompositeRiskGate
        from shared.execution_journal import ExecutionJournal
        from shared.exposure_gate import MAX_TOTAL_LEVERAGE, AccountExposureGate
        from shared.kill_file import KillFile
        scope = os.environ.get("RISK_SCOPE", "").strip()
        if not scope or "binance" not in scope or (("testnet" in scope) != testnet):
            raise RuntimeError("RISK_SCOPE must name binance and contain 'testnet' exactly when BINANCE_TESTNET=1")
        journal_path = self.state_dir / "orders.sqlite"
        kill = KillFile(self.state_dir / "kill_switch.json")
        drawdown = DrawdownGuard(self.state_dir / "drawdown.sqlite", scope,
                                 daily_pct=float(os.environ.get("MAX_DAILY_DRAWDOWN_PCT", "5")),
                                 weekly_pct=float(os.environ.get("MAX_WEEKLY_DRAWDOWN_PCT", "12")))
        limiter = OrderLimiter(OrderLimitsConfig(
            max_order_notional=D(os.environ.get("BINANCE_MAX_ORDER_NOTIONAL", "5000")),
            max_daily_notional=D(os.environ.get("BINANCE_MAX_DAILY_NOTIONAL", "50000")),
            max_daily_orders=int(os.environ.get("BINANCE_MAX_DAILY_ORDERS", "50")), max_orders_per_second=1.0))
        cap = min(MAX_TOTAL_LEVERAGE, D(os.environ.get("BINANCE_MAX_LEVERAGE", "3")))
        exposure = AccountExposureGate(self.account, ExecutionJournal(journal_path, scope), max_leverage=cap)
        gate = CompositeRiskGate.for_execution(limiter, kill_switch=kill, drawdown_guard=drawdown,
                                               portfolio_reader=exposure,
                                               venue="binance")
        return {"account": self.account, "scope": scope, "journal_path": journal_path, "gate": gate, "kill": kill,
                "drawdown": drawdown, "stop_pct": stop_pct, "equity": self.equity}

    # -- lifecycle ------------------------------------------------------------
    def warmup(self) -> None:
        """Load history; a paper run resumes from its saved state and replays the bars it missed."""
        now_ms = int(self.clock() * 1000)
        saved = self._load_paper_state() if self.mode == "paper" else {}
        for symbol, session in self.sessions.items():
            snap = saved.get(symbol)
            resume = snap["cursor"] if snap else None
            gap = 0 if resume is None else max(0, (now_ms // HOUR * HOUR - HOUR - resume) // HOUR)
            limit = self.spec.funding_window() + 30 + gap // 8 + 1
            for at, rate in self.market.get_funding_history(symbol, limit=limit):
                session.funding[round(at / HOUR) * HOUR] = rate
            rows = self.market.get_klines(symbol, "60", session.bars.maxlen + gap, now_ms=now_ms)
            if snap is not None:
                if not any(row["start"] == resume for row in rows):
                    raise RuntimeError(f"{symbol}: saved paper state at {resume} cannot be resumed from the available "
                                       "market data; start a new paper record with --reset-paper-state")
                session.restore_paper(snap)
            for row in rows:
                bar = bar_from_kline(row)
                session.process_bar(bar, live=resume is not None and bar.open_time > resume, now_ms=now_ms)
            if len(session.bars) < self.spec.window_hours():
                raise RuntimeError(f"{symbol}: incomplete warmup ({len(session.bars)}/{self.spec.window_hours()})")
            self.journal.write("warmup", symbol=symbol, bars=len(session.bars), last_open=session.cursor,
                               resumed_from=resume)
        if self.mode == "paper":
            self.save_paper_state()
        if self.mode == "execution":
            self.refresh_equity()
            self.reconcile()

    def refresh_equity(self) -> None:
        try:
            equity = self.account.get_account_equity()
            self.equity.update(D(str(equity)))
            self.drawdown.observe(float(equity))
        except Exception:
            self.drawdown.invalidate("exchange equity unavailable")
            logger.warning("Equity refresh failed; entries blocked", exc_info=True)
        self._last_equity = self.clock()

    def reconcile(self) -> None:
        for symbol, session in self.sessions.items():
            try:
                session.consume_fills()
                session.executor.notify_status()
                issues = session.executor.reconcile_account(symbol) + session.stops.sync()
            except Exception as exc:
                session.executor._ready = False
                issues = [f"reconciliation failed: {type(exc).__name__}: {exc}"]
                logger.exception("%s reconciliation failed; entries blocked", symbol)
            for issue in issues:
                self.journal.write("issue", symbol=symbol, issue=issue)
            if any("stop_breached" in i for i in issues):
                session.flatten("stop_breached")
        self._last_reconcile = self.clock()

    def risk_actions(self) -> None:
        kill = self.kill.state()
        if (kill.active and kill.flatten) or self.drawdown.latched():
            for session in self.sessions.values():
                session.flatten("kill_switch" if kill.active else "drawdown_latched")

    def poll(self) -> int:
        """Process newly closed bars. A market-data failure is logged and retried with backoff instead of
        ending the process, so reconciliation, stops and risk actions keep running in the same cycle."""
        if self.clock() < self._market_retry_at:
            return 0
        from execution.adapters.binance.rest import BinanceRestError
        try:
            processed = self._poll_market()
        except (BinanceRestError, OSError) as exc:
            self._market_failures += 1
            delay = min(POLL_SECONDS * 2 ** (self._market_failures - 1), MARKET_BACKOFF_MAX)
            self._market_retry_at = self.clock() + delay
            self.journal.write("issue", issue=f"market data unavailable: {type(exc).__name__}: {exc}",
                               failures=self._market_failures, retry_in=delay)
            logger.warning("Market data unavailable (%d in a row); retry in %.0fs: %s",
                           self._market_failures, delay, exc)
            return 0
        self._market_failures = 0
        return processed

    def _poll_market(self) -> int:
        processed = 0
        now_ms = int(self.clock() * 1000)
        for symbol, session in self.sessions.items():
            close = session.cursor + 2 * HOUR
            if now_ms < close:
                continue
            missing = (now_ms // HOUR * HOUR - (session.cursor + HOUR)) // HOUR
            for at, rate in self.market.get_funding_history(symbol, limit=3 + max(missing, 0) // 8):
                session.funding[round(at / HOUR) * HOUR] = rate
            settlement = close // FUNDING_INTERVAL * FUNDING_INTERVAL
            if settlement not in session.funding and now_ms - close < FUNDING_WAIT_MS:
                continue
            for row in self.market.get_klines(symbol, "60", int(min(max(missing, 1) + 2, 1500)), now_ms=now_ms):
                bar = bar_from_kline(row)
                if bar.open_time > session.cursor:
                    session.process_bar(bar, live=True, now_ms=now_ms)
                    processed += 1
        return processed

    def _paper_path(self) -> Path:
        return self.state_dir / PAPER_STATE

    def _load_paper_state(self) -> dict:
        path = self._paper_path()
        if self.reset_paper and path.exists():
            archived = path.with_name(f"paper_state.{int(self.clock())}.json")
            os.replace(path, archived)
            self.journal.write("paper_reset", archived=archived.name)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(), object_hook=_decode)
        if (data.get("schema") != 1 or data.get("spec") != self.spec.record()
                or set(data["symbols"]) != set(self.sessions)):
            raise RuntimeError("saved paper state belongs to another spec or symbol set; "
                               "start a new paper record with --reset-paper-state")
        return data["symbols"]

    def save_paper_state(self) -> None:
        """Atomically persist every paper sleeve after the bars it has processed."""
        cursors = {symbol: s.cursor for symbol, s in self.sessions.items()}
        if cursors == self._saved_cursors:
            return
        data = {"schema": 1, "spec": self.spec.record(),
                "symbols": {symbol: s.paper_snapshot() for symbol, s in self.sessions.items()}}
        path = self._paper_path()
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, default=_encode)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        self._saved_cursors = cursors

    def status(self) -> dict:
        out = {"mode": self.mode, "at": self.clock(), "spec_passed_gate": self.frozen.get("passed"),
               "market_failures": self._market_failures, "symbols": {}}
        if self.kill is not None:
            out["kill"] = self.kill.state().__dict__
            out["drawdown_latched"] = self.drawdown.latched()
            out["equity"] = str(self.equity.value)
        for symbol, s in self.sessions.items():
            out["symbols"][symbol] = {
                "last_bar_open": s.cursor, "target": s.last_target[0], "target_ready": s.last_target[1],
                "entry_ready": s.entry_ready, "entry_reason": s.entry_reason, "position": str(s.position_qty()),
                "sleeve_equity": str(s.sleeve_equity()),
                "protective_stop": s.stops.has_active_stop() if s.stops else None}
        return out

    def cycle(self) -> None:
        if self.mode == "execution" and self.clock() - self._last_equity >= EQUITY_SECONDS:
            self.refresh_equity()
        self.poll()
        if self.mode == "paper":
            self.save_paper_state()
        if self.mode == "execution":
            if self.clock() - self._last_reconcile >= RECONCILE_SECONDS:
                self.reconcile()
            self.risk_actions()
        tmp = self.state_dir / "target_status.json.tmp"
        tmp.write_text(json.dumps(self.status(), default=str, indent=2))
        os.replace(tmp, self.state_dir / "target_status.json")


def main(argv: list[str] | None = None) -> None:
    from runner.binance_main import build_adapters
    from shared.execution_lease import ExecutionLease
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=Path("data/runtime/binance_target"))
    parser.add_argument("--stop-pct", type=D, default=D("0.08"), help="Exchange-side catastrophic stop distance")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", dest="execution", action="store_false", help="Paper trading (default)")
    mode.add_argument("--enable-execution", dest="execution", action="store_true")
    parser.set_defaults(execution=False)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--reset-paper-state", action="store_true",
                        help="Archive the saved paper state and start a new paper record")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    testnet = os.environ.get("BINANCE_TESTNET", "1").strip().lower() not in ("0", "false", "no", "off")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    scope = os.environ.get("RISK_SCOPE", "paper") if args.execution else "paper-target"
    lease = ExecutionLease(args.state_dir / "orders.sqlite", scope)
    symbols = json.loads(args.spec.read_text())["symbols"]
    market, account = build_adapters(execution=args.execution, testnet=testnet, symbols=symbols)
    runtime = TargetRuntime(spec_path=args.spec, state_dir=args.state_dir,
                            mode="execution" if args.execution else "paper", market=market, account=account,
                            testnet=testnet, stop_pct=args.stop_pct, reset_paper=args.reset_paper_state)
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
