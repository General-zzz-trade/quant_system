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
import os
import signal
import time
from decimal import Decimal as D
from pathlib import Path
from typing import Any

from lab.orders import make_order
from lab.simulate import SYMBOL_RULES, Sleeve, broker_config
from trading.adapters.binance.rest import BinanceRestError
from trading.adapters.files.event_log import JsonlEventLog
from trading.adapters.files.state_files import JsonPaperStore, JsonStatusFile
from trading.adapters.sqlite.paper_store import SqlitePaperStore
from trading.application import runtime as _runtime
from trading.application.equity import EquitySource
from trading.application.execution.fills import TargetFillConsumer
from trading.application.runtime import POLL_SECONDS
from trading.application.session import BROKER_STATE as _BROKER_STATE
from trading.application.session import MAX_ENTRY_DELAY_MS, TargetSession
from trading.domain.risk.liquidation import check_stop_inside_liquidation
from trading.domain.strategy.overlay import StrategySpec
from trading.infrastructure import target
from trading.infrastructure.bootstrap import build_adapters
from trading.infrastructure.paper_store import MirroredPaperStore
from trading.infrastructure.settings import exchange_settings, execution_scope, risk_limits

logger = logging.getLogger("binance_target")

PAPER_STATE = "paper_state.json"


def load_spec(path: Path) -> dict:
    from infra.model_signing import load_verified_json
    spec = load_verified_json(path)
    if spec.get("kind") != "btc_eth_multitimeframe_target" or spec.get("schema") != 1:
        raise ValueError("not a frozen BTC/ETH target-strategy spec")
    StrategySpec.from_record(spec["spec"])  # validates parameters
    return spec


def preflight(spec_path: Path, *, execution: bool, testnet: bool, stop_pct: D) -> dict:
    """Every start-up check that needs no exchange: the spec's signature and kind, the research gate for
    mainnet, the risk scope and the stop distance. Run before any account is connected (invariant L6):
    connecting sets leverage and margin type on the exchange."""
    frozen = load_spec(spec_path)
    if execution:
        if not testnet and frozen.get("passed") is not True:
            raise RuntimeError("research gate not passed; mainnet execution refused")
        execution_scope(testnet=testnet)
        check_stop_inside_liquidation(float(stop_pct), *exchange_settings())
    return frozen


class TargetRuntime(_runtime.TargetRuntime):
    """The legacy entry point's constructor. The loop is trading.application.runtime.TargetRuntime and the
    sessions are wired in trading.infrastructure.target (migration step 10); this class loads the signed
    spec and supplies the parts that still live in the legacy packages: the research paper sleeve, the
    order limiter, the kill file and the transaction-cost log."""

    def __init__(self, *, spec_path: Path, state_dir: Path, mode: str, market: Any, account: Any = None,
                 clock=time.time, testnet: bool = True, stop_pct: D = D("0.08"), reset_paper: bool = False) -> None:
        if mode not in ("paper", "execution"):
            raise ValueError("mode must be paper or execution")
        self.frozen = load_spec(spec_path)
        spec = StrategySpec.from_record(self.frozen["spec"])
        if mode == "execution":
            target.refuse_unverified_mainnet(testnet=testnet, spec_passed=self.frozen.get("passed"))
        self.account, self.clock = account, clock
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        journal = JsonlEventLog(self.state_dir / "target_events.jsonl")
        self.equity = EquitySource(clock)
        symbols = self.frozen["symbols"]
        kill = drawdown = None
        if mode == "execution":
            wiring, kill = self._execution(spec, symbols, journal, testnet, stop_pct)
            sessions, drawdown = wiring.sessions, wiring.drawdown
        else:
            sessions = target.paper_sessions(
                spec, symbols, journal=journal, rules=SYMBOL_RULES,     # paper equals research
                make_order=make_order, sleeve=lambda symbol, lot: Sleeve(symbol, spec, lot, broker_config(spec, lot)))
        super().__init__(
            spec=spec, mode=mode, market=market, sessions=sessions, journal=journal, equity=self.equity, clock=clock,
            status_sink=JsonStatusFile(self.state_dir / "target_status.json"),
            market_errors=(BinanceRestError, OSError), account=account, kill=kill, drawdown=drawdown,
            paper_store=MirroredPaperStore(
                SqlitePaperStore(self.state_dir / "paper_state.sqlite", clock=clock),
                JsonPaperStore(self.state_dir / PAPER_STATE, clock=clock)) if mode == "paper" else None,
            reset_paper=reset_paper, spec_passed=self.frozen.get("passed"))

    def _execution(self, spec, symbols, journal, testnet: bool, stop_pct: D):
        from execution.adapters.binance.execution_adapter import _TcaLog
        from execution.safety.limits import OrderLimiter, OrderLimitsConfig
        from shared.kill_file import KillFile
        scope = execution_scope(testnet=testnet)
        limits = risk_limits()
        kill = KillFile(self.state_dir / "kill_switch.json")
        limiter = OrderLimiter(OrderLimitsConfig(
            max_order_notional=limits.max_order_notional, max_daily_notional=limits.max_daily_notional,
            max_daily_orders=limits.max_daily_orders, max_orders_per_second=1.0))
        inputs = target.ExecutionInputs(account=self.account, scope=scope, stop_pct=stop_pct, limits=limits,
                                        limiter=limiter, kill=kill,
                                        tca_log=_TcaLog(self.state_dir / "tca_binance.jsonl"))
        return target.execution_sessions(spec, symbols, state_dir=self.state_dir, journal=journal,
                                         equity=self.equity, inputs=inputs), kill


def main(argv: list[str] | None = None) -> None:
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
    scope = os.environ.get("RISK_SCOPE", "paper") if args.execution else "paper-target"
    frozen = preflight(args.spec, execution=args.execution, testnet=testnet, stop_pct=args.stop_pct)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    lease = ExecutionLease(args.state_dir / "orders.sqlite", scope)
    market, account = build_adapters(execution=args.execution, testnet=testnet, symbols=frozen["symbols"])
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
