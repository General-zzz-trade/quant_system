"""Binance BTC/ETH target-position runtime (1h clock, 4h/daily features).

    python -m runner.binance_target --spec models_binance/strategy/final_spec.json            # paper
    python -m runner.binance_target --spec ... --enable-execution                             # testnet

Each closed 1h bar, per symbol:
1. advance the window (paper: the simulated exchange matches the bar and the
   exit side consumes its fills first, as research does),
2. compute the target weight with the function research uses on a trailing
   window (exactly the research value),
3. rebalance with ``trading.domain.rebalance.plan`` (band, reduce-only vs entry),
4. send intents through the gates (inputs fresh, kill switch, drawdown,
   3x gross exposure, order limits) to the durable executor.

Paper mode runs the same execution path on the simulated exchange (ADR-0013
plan B) and reproduces ``lab.simulate`` fills exactly (tested). Mainnet
execution requires the frozen spec's research gate to have passed.
"""
from __future__ import annotations

import argparse
import logging
import signal
import time
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
from trading.application.event_guard import GuardedEventLog
from trading.application.runtime import POLL_SECONDS
from trading.adapters.simulated.exchange import BROKER_STATE as _BROKER_STATE
from trading.application.session import MAX_ENTRY_DELAY_MS, TargetSession
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure import target
from trading.infrastructure.bootstrap import build_adapters
from trading.infrastructure.config import risk_limits
from trading.infrastructure.modes import policy_for
from trading.infrastructure.paper import (PAPER_STATE, BarTimeClock, DivergencePauses, PaperDivergenceLog,
                                          SymbolVenueEquity, paper_exchange, paper_state_db)
from trading.infrastructure.paper_store import MirroredPaperStore
from trading.infrastructure.state_dir import is_managed, kill_file_path
from trading.infrastructure.state_dir import verify as verify_state_dir
from trading.infrastructure.config import RuntimeConfig, Secrets, load_config, make_notifier, with_credentials
from trading.domain.events import EventKind

logger = logging.getLogger("binance_target")


def load_spec(path: Path) -> dict:
    from infra.model_signing import load_verified_json
    spec = load_verified_json(path)
    if spec.get("kind") != "btc_eth_multitimeframe_target" or spec.get("schema") != 1:
        raise ValueError("not a frozen BTC/ETH target-strategy spec")
    StrategySpec.from_record(spec["spec"])  # validates parameters
    return spec


def preflight(spec_path: Path, *, execution: bool, testnet: bool, stop_pct: D, state_dir: Path | None = None,
              config: RuntimeConfig | None = None) -> dict:
    """Every start-up check that needs no exchange: the spec's signature and kind, the research gate for
    mainnet, the risk scope, the drawdown ceiling, the stop distance and - for execution - a state directory a
    person initialized for this scope (step 13c). Run before any account is connected (invariant L6). Since
    ADR-0010 connecting is read-only; the account itself is checked next, by ``refuse_foreign_holdings``."""
    frozen = load_spec(spec_path)
    if execution:
        net = config.testnet if config is not None else testnet
        if not net and frozen.get("passed") is not True:
            raise RuntimeError("research gate not passed; mainnet execution refused")
        if config is None:      # the scope and the drawdown ceiling (step 13b) are checked by load_config
            config, _ = load_config(execution=True, testnet=testnet, stop_pct=stop_pct)
        check_stop_inside_liquidation(float(config.stop_pct), config.leverage or 0, config.margin)
        if state_dir is not None:
            verify_state_dir(state_dir, config.scope)
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
    spec and picks the mode's policy and parts: in paper the simulated exchange (S10 part 3)."""

    def __init__(self, *, spec_path: Path, state_dir: Path, mode: str, market: Any, account: Any = None,
                 clock=time.time, testnet: bool = True, stop_pct: D = D("0.08"), reset_paper: bool = False,
                 notifier: Any = None, config: RuntimeConfig | None = None, secrets: Secrets | None = None) -> None:
        policy = policy_for(mode)
        if config is None:      # read once, here, when the entry point did not (tests construct the runtime)
            config, secrets = load_config(execution=mode == "execution", testnet=testnet, stop_pct=stop_pct)
        self.config = config
        self.frozen = load_spec(spec_path)
        spec = StrategySpec.from_record(self.frozen["spec"])
        if mode == "execution":
            target.refuse_unverified_mainnet(testnet=config.testnet, spec_passed=self.frozen.get("passed"))
        self.account, self.clock = account, clock
        alert_prefix = ""
        if mode == "execution":                   # alerts reach a person in execution mode (step 12)
            alert_prefix = config.scope
            notifier = notifier if notifier is not None else make_notifier(secrets or Secrets(None, None))
        else:
            notifier = None
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        # A failing event log never stops trading; it blocks every entry until it works again (S3 audit, finding 3).
        events = GuardedEventLog(JsonlEventLog(self.state_dir / "target_events.jsonl"))
        journal: Any = events
        if notifier is not None:
            journal = NotifyingEventLog(journal, notifier, prefix=alert_prefix)
        self.equity = EquitySource(clock)
        symbols = self.frozen["symbols"]
        account_check = venue = gate_clock = paper_store = None
        if mode == "execution":
            wiring, kill = self._execution(spec, symbols, journal)
            assert config.leverage is not None
            account_check = target.account_checker(account, state_dir=self.state_dir, scope=config.scope,
                                                   symbols=symbols, leverage=config.leverage, margin=config.margin)
        else:
            journal = PaperDivergenceLog(journal)
            venue, gate_clock = paper_exchange(spec, symbols), BarTimeClock()
            paper_store, wiring, kill = self._paper(spec, symbols, journal, venue, gate_clock, reset_paper)
            account = venue
        self.exchange = venue
        super().__init__(
            spec=spec, policy=policy, market=market, sessions=wiring.sessions, journal=journal, equity=self.equity,
            clock=clock, status_sink=JsonStatusFile(self.state_dir / "target_status.json"),
            market_errors=(BinanceRestError, OSError), final_market_errors=(BinanceNonRetryableError,),
            account=account, kill=kill, drawdown=wiring.drawdown, paper_store=paper_store,
            spec_passed=self.frozen.get("passed"), pauses=wiring.pauses, notifier=notifier,
            alert_prefix=alert_prefix, event_log_failure=events.failure, config_digest=config.digest(),
            account_check=account_check, venue=venue, gate_clock=gate_clock)
        # ADR-0017: what this run uses, recorded once, without any secret.
        journal.write(EventKind.CONFIG, **config.record(), digest=config.digest())
        logger.info("Configuration %s: %s", config.digest()[:12], config.record())

    def _paper(self, spec, symbols, journal, venue, gate_clock, reset: bool):
        """Plan B (ADR-0013): the execution path over the simulated exchange. No credentialed adapter, no signed
        request, no account setting from the environment: the code's default limits (S10 decision 3), the venue's
        unlevered account for the stop check, a kill file that is not managed unless the directory is."""
        _, aside = paper_state_db(self.state_dir, reset=reset, stamp_ms=int(self.clock() * 1000))
        store = MirroredPaperStore(SqlitePaperStore(self.state_dir / "paper_state.sqlite", clock=self.clock),
                                   JsonPaperStore(self.state_dir / PAPER_STATE, clock=self.clock))
        archived = store.archive() if reset else None
        if archived or aside:
            journal.write(EventKind.PAPER_RESET, archived=archived, state_db=aside)
        kill = KillFile(kill_file_path(self.state_dir), missing_is_active=is_managed(self.state_dir))
        inputs = target.ExecutionInputs(account=venue, scope=self.config.scope, stop_pct=self.config.stop_pct,
                                        limits=risk_limits({}), kill=kill, leverage=1, margin=None)
        parts = target.PaperParts(symbol_equity=lambda symbol: SymbolVenueEquity(venue, symbol),
                                  gate_clock=gate_clock, sleeper=lambda seconds: None,
                                  pauses=lambda pauses: DivergencePauses(pauses, journal))
        wiring = target.execution_sessions(spec, symbols, state_dir=self.state_dir, journal=journal,
                                           equity=self.equity, inputs=inputs, clock=self.clock,
                                           policy=policy_for("paper"), paper=parts)
        for symbol, session in wiring.sessions.items():
            session.sleeve = SimpleNamespace(broker=venue.broker(symbol))    # read-only, for people and tests
        return store, wiring, kill

    def _execution(self, spec, symbols, journal):
        config = self.config
        assert config.limits is not None and config.leverage is not None
        # In a managed directory a missing kill file reads as active (ADR-0005 5b); the runtime never writes it.
        kill = KillFile(kill_file_path(self.state_dir), missing_is_active=is_managed(self.state_dir))
        inputs = target.ExecutionInputs(account=self.account, scope=config.scope, stop_pct=config.stop_pct,
                                        limits=config.limits, kill=kill, leverage=config.leverage, margin=config.margin,
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
    # ADR-0017: the configuration is read here, once; everything after receives it.
    config, secrets = load_config(execution=args.execution, stop_pct=args.stop_pct)
    testnet = config.testnet
    frozen = preflight(args.spec, execution=args.execution, testnet=testnet, stop_pct=config.stop_pct,
                       state_dir=args.state_dir, config=config)
    # The same validated, stripped scope the journal and the manual tools use (audit after step 12b, finding 7).
    scope = config.scope
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lease = ExecutionLease(args.state_dir / STATE_DB, scope)
    if args.execution:       # the credentials after every local check, before the account (as before ADR-0017)
        config, secrets = with_credentials(config, secrets)
    market, account = build_adapters(execution=args.execution, testnet=testnet, symbols=frozen["symbols"],
                                     credentials=secrets.binance)
    if args.execution:       # ADR-0010: read-only, before the runtime is built; the runtime never sets leverage
        assert config.leverage is not None
        target.refuse_foreign_holdings(account, state_dir=args.state_dir, scope=scope, symbols=frozen["symbols"],
                                       leverage=config.leverage, margin=config.margin)
    runtime = TargetRuntime(spec_path=args.spec, state_dir=args.state_dir,
                            mode="execution" if args.execution else "paper", market=market, account=account,
                            testnet=testnet, stop_pct=config.stop_pct, reset_paper=args.reset_paper_state,
                            config=config, secrets=secrets)
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
