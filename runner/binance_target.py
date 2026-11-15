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
import logging
import signal
import time
from decimal import Decimal as D
from pathlib import Path
from typing import Any

from lab.orders import make_order
from lab.simulate import SYMBOL_RULES, Sleeve, broker_config
from trading.adapters.binance.rest import BinanceNonRetryableError, BinanceRestError
from trading.adapters.files.event_log import JsonlEventLog
from trading.adapters.files.kill_switch import KillFile
from trading.adapters.files.lease import ExecutionLease
from trading.adapters.files.state_files import JsonPaperStore, JsonStatusFile
from trading.adapters.sqlite.paper_store import SqlitePaperStore
from trading.adapters.sqlite.schema import STATE_DB
from trading.application import runtime as _runtime
from trading.application.equity import EquitySource
from trading.application.execution.fills import TargetFillConsumer
from trading.application.alerts import NotifyingEventLog
from trading.application.runtime import POLL_SECONDS
from trading.application.session import BROKER_STATE as _BROKER_STATE
from trading.application.session import MAX_ENTRY_DELAY_MS, TargetSession
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure import target
from trading.infrastructure.bootstrap import build_adapters
from trading.infrastructure.paper_store import MirroredPaperStore
from trading.infrastructure.state_dir import is_managed, kill_file_path
from trading.infrastructure.state_dir import verify as verify_state_dir
from trading.infrastructure.settings import (exchange_settings, execution_scope, notifier_from_env, risk_limits,
                                             testnet_from_env)
from trading.domain.events import EventKind

logger = logging.getLogger("binance_target")

PAPER_STATE = "paper_state.json"


def load_spec(path: Path) -> dict:
    from infra.model_signing import load_verified_json
    spec = load_verified_json(path)
    if spec.get("kind") != "btc_eth_multitimeframe_target" or spec.get("schema") != 1:
        raise ValueError("not a frozen BTC/ETH target-strategy spec")
    StrategySpec.from_record(spec["spec"])  # validates parameters
    return spec


def preflight(spec_path: Path, *, execution: bool, testnet: bool, stop_pct: D, state_dir: Path | None = None) -> dict:
    """Every start-up check that needs no exchange: the spec's signature and kind, the research gate for
    mainnet, the risk scope, the drawdown ceiling, the stop distance and - for execution - a state directory a
    person initialized for this scope (step 13c). Run before any account is connected (invariant L6). Since
    ADR-0010 connecting is read-only; the account itself is checked next, by ``refuse_foreign_holdings``."""
    frozen = load_spec(spec_path)
    if execution:
        if not testnet and frozen.get("passed") is not True:
            raise RuntimeError("research gate not passed; mainnet execution refused")
        scope = execution_scope(testnet=testnet)
        risk_limits()                             # the drawdown ceiling (step 13b), before any account access
        check_stop_inside_liquidation(float(stop_pct), *exchange_settings())
        if state_dir is not None:
            verify_state_dir(state_dir, scope)
    return frozen



class _TcaLog:
    """The EventLog port over the TCA file next to the order journal (one append per fill). Moved here from the
    execution adapter shell (ADR-0006, 2026-11-15); the writer itself is still the legacy monitoring.tca."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, kind: EventKind, **fields: Any) -> None:
        from monitoring.tca import TCALogger
        tca = TCALogger("binance", path=self._path)
        try:
            tca.record_fill(**fields)
        finally:
            tca.close()

class TargetRuntime(_runtime.TargetRuntime):
    """The legacy entry point's constructor. The loop is trading.application.runtime.TargetRuntime and the
    sessions are wired in trading.infrastructure.target (migration step 10); this class loads the signed
    spec and supplies the parts that still live in the legacy packages: the research paper sleeve, the
    order limiter, the kill file and the transaction-cost log."""

    def __init__(self, *, spec_path: Path, state_dir: Path, mode: str, market: Any, account: Any = None,
                 clock=time.time, testnet: bool = True, stop_pct: D = D("0.08"), reset_paper: bool = False,
                 notifier: Any = None) -> None:
        if mode not in ("paper", "execution"):
            raise ValueError("mode must be paper or execution")
        self.frozen = load_spec(spec_path)
        spec = StrategySpec.from_record(self.frozen["spec"])
        if mode == "execution":
            target.refuse_unverified_mainnet(testnet=testnet, spec_passed=self.frozen.get("passed"))
        self.account, self.clock = account, clock
        alert_prefix = ""
        if mode == "execution":                   # alerts reach a person in execution mode (step 12)
            alert_prefix = execution_scope(testnet=testnet)
            notifier = notifier if notifier is not None else notifier_from_env()
        else:
            notifier = None
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        journal: Any = JsonlEventLog(self.state_dir / "target_events.jsonl")
        if notifier is not None:
            journal = NotifyingEventLog(journal, notifier, prefix=alert_prefix)
        self.equity = EquitySource(clock)
        symbols = self.frozen["symbols"]
        kill = drawdown = pauses = None
        if mode == "execution":
            wiring, kill = self._execution(spec, symbols, journal, testnet, stop_pct)
            sessions, drawdown, pauses = wiring.sessions, wiring.drawdown, wiring.pauses
        else:
            sessions = target.paper_sessions(
                spec, symbols, journal=journal, rules=SYMBOL_RULES,     # paper equals research
                make_order=make_order, sleeve=lambda symbol, lot: Sleeve(symbol, spec, lot, broker_config(spec, lot)))
        super().__init__(
            spec=spec, mode=mode, market=market, sessions=sessions, journal=journal, equity=self.equity, clock=clock,
            status_sink=JsonStatusFile(self.state_dir / "target_status.json"),
            market_errors=(BinanceRestError, OSError), final_market_errors=(BinanceNonRetryableError,),
            account=account, kill=kill, drawdown=drawdown,
            paper_store=MirroredPaperStore(
                SqlitePaperStore(self.state_dir / "paper_state.sqlite", clock=clock),
                JsonPaperStore(self.state_dir / PAPER_STATE, clock=clock)) if mode == "paper" else None,
            reset_paper=reset_paper, spec_passed=self.frozen.get("passed"), pauses=pauses, notifier=notifier,
            alert_prefix=alert_prefix)

    def _execution(self, spec, symbols, journal, testnet: bool, stop_pct: D):
        scope = execution_scope(testnet=testnet)
        limits = risk_limits()
        # In a managed directory a missing kill file reads as active (ADR-0005 5b); the runtime never writes it.
        kill = KillFile(kill_file_path(self.state_dir), missing_is_active=is_managed(self.state_dir))
        inputs = target.ExecutionInputs(account=self.account, scope=scope, stop_pct=stop_pct, limits=limits,
                                        kill=kill,
                                        tca_log=_TcaLog(self.state_dir / "tca_binance.jsonl"))
        return target.execution_sessions(spec, symbols, state_dir=self.state_dir, journal=journal,
                                         equity=self.equity, inputs=inputs, clock=self.clock), kill


def main(argv: list[str] | None = None) -> None:
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
    testnet = testnet_from_env()
    frozen = preflight(args.spec, execution=args.execution, testnet=testnet, stop_pct=args.stop_pct,
                       state_dir=args.state_dir)
    # The same validated, stripped scope the journal and the manual tools use (audit after step 12b, finding 7).
    scope = execution_scope(testnet=testnet) if args.execution else "paper-target"
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lease = ExecutionLease(args.state_dir / STATE_DB, scope)
    market, account = build_adapters(execution=args.execution, testnet=testnet, symbols=frozen["symbols"])
    if args.execution:       # ADR-0010: read-only, before the runtime is built; the runtime never sets leverage
        target.refuse_foreign_holdings(account, state_dir=args.state_dir, scope=scope, symbols=frozen["symbols"])
    runtime = TargetRuntime(spec_path=args.spec, state_dir=args.state_dir,
                            mode="execution" if args.execution else "paper", market=market, account=account,
                            testnet=testnet, stop_pct=args.stop_pct, reset_paper=args.reset_paper_state)
    stop = {"flag": False}
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.update(flag=True))
    try:
        while not stop["flag"] and not runtime.warmup_or_hold():   # market data down: exits keep working
            time.sleep(POLL_SECONDS)
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
