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
from decimal import Decimal as D
from pathlib import Path
from typing import Any

from lab.orders import make_order
from lab.simulate import SYMBOL_RULES, Sleeve, SymbolRules, broker_config
from lab.strategy import StrategySpec
from runner.binance_main import EntryReadinessGate, EquitySource, EventJournal
from trading.adapters.files.state_files import JsonPaperStore, JsonStatusFile
from trading.application import runtime as _runtime
from trading.application.execution.fills import TargetFillConsumer
from trading.application.runtime import POLL_SECONDS
from trading.application.session import BROKER_STATE as _BROKER_STATE
from trading.application.session import MAX_ENTRY_DELAY_MS, ExecutionParts, TargetSession

logger = logging.getLogger("binance_target")

PAPER_STATE = "paper_state.json"


def load_spec(path: Path) -> dict:
    from infra.model_signing import load_verified_json
    spec = load_verified_json(path)
    if spec.get("kind") != "btc_eth_multitimeframe_target" or spec.get("schema") != 1:
        raise ValueError("not a frozen BTC/ETH target-strategy spec")
    StrategySpec.from_record(spec["spec"])  # validates parameters
    return spec


class TargetRuntime(_runtime.TargetRuntime):
    """Composition for the legacy entry point (migration step 10): the loop itself is
    trading.application.runtime.TargetRuntime; this class builds what it is given."""

    def __init__(self, *, spec_path: Path, state_dir: Path, mode: str, market: Any, account: Any = None,
                 clock=time.time, testnet: bool = True, stop_pct: D = D("0.08"), reset_paper: bool = False) -> None:
        from execution.adapters.binance.rest import BinanceRestError
        if mode not in ("paper", "execution"):
            raise ValueError("mode must be paper or execution")
        self.frozen = load_spec(spec_path)
        spec = StrategySpec.from_record(self.frozen["spec"])
        if mode == "execution" and not testnet and self.frozen.get("passed") is not True:
            raise RuntimeError("research gate not passed; mainnet execution refused")
        self.account, self.clock = account, clock
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        journal = EventJournal(self.state_dir / "target_events.jsonl")
        self.equity = EquitySource(clock)
        execution = self._execution_context(testnet, stop_pct) if mode == "execution" else None
        symbols = self.frozen["symbols"]
        allocation = D(1) / D(len(symbols))
        sessions = {}
        for symbol in symbols:
            if account is not None:
                f = account.filters(symbol)
                rules = SymbolRules(f.step_size, f.min_qty, max(f.min_notional, SYMBOL_RULES[symbol].min_notional))
            else:
                rules = SYMBOL_RULES[symbol]
            common = dict(journal=journal, rules=rules, make_order=make_order, allocation=allocation)
            if execution is None:
                sessions[symbol] = TargetSession(symbol, spec, sleeve=Sleeve(symbol, spec, rules,
                                                                             broker_config(spec, rules)), **common)
                continue
            from execution.adapters.binance.execution_adapter import BinanceExecutionAdapter
            from execution.adapters.binance.protective_stops import ProtectiveStopManager
            executor = BinanceExecutionAdapter(
                execution["account"], owner=f"target.{symbol}", scope=execution["scope"],
                journal_path=execution["journal_path"], entry_mode=spec.execution, ttl_bars=spec.ttl_bars)
            parts = ExecutionParts(executor, TargetFillConsumer(executor),
                                   ProtectiveStopManager(execution["account"], executor, symbol,
                                                         stop_pct=execution["stop_pct"]), None, execution["equity"])
            session = TargetSession(symbol, spec, execution=parts, **common)
            executor._status_callback = session._on_status
            session.gate = EntryReadinessGate(session, execution["gate"])
            sessions[symbol] = session
        super().__init__(
            spec=spec, mode=mode, market=market, sessions=sessions, journal=journal, equity=self.equity, clock=clock,
            status_sink=JsonStatusFile(self.state_dir / "target_status.json"),
            market_errors=(BinanceRestError, OSError), account=account,
            kill=execution["kill"] if execution else None, drawdown=execution["drawdown"] if execution else None,
            paper_store=JsonPaperStore(self.state_dir / PAPER_STATE, clock=clock) if mode == "paper" else None,
            reset_paper=reset_paper, spec_passed=self.frozen.get("passed"))

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


__all__ = ["MAX_ENTRY_DELAY_MS", "PAPER_STATE", "POLL_SECONDS", "TargetFillConsumer", "TargetRuntime",
           "TargetSession", "_BROKER_STATE", "load_spec", "main"]


if __name__ == "__main__":
    main()
