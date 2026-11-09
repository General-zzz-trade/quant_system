"""Bind atomic portfolio admission to identified runtime state and strict reads.

All-scope collection runs under the reservation database's write lock. It only
queries accounts; initialization never opens a network connection.
"""
from __future__ import annotations

from contextlib import ExitStack
from decimal import Decimal
import os
from pathlib import Path
import sqlite3
import time
from typing import Any

from shared.portfolio_reservations import (
    AccountEvidence, ConsumerPosition, PortfolioReservationStore, PositionEvidence,
    ScopeRegistration, SubmissionEvidence,
)
from shared.runtime_state import MANIFEST_NAME, validate_runtime_state, _path, _json

_ENDPOINTS = {
    "https://api-demo.bybit.com": "demo",
    "https://api-testnet.bybit.com": "testnet",
    "https://api.bybit.com": "live",
}


def bybit_environment(base_url: str) -> str:
    try:
        return _ENDPOINTS[base_url]
    except KeyError:
        raise ValueError("Portfolio risk requires an exact recognized Bybit endpoint") from None


def runtime_registration(scope: str, state_dir: Path, environment: str) -> ScopeRegistration:
    if environment not in _ENDPOINTS.values():
        raise ValueError("Unknown risk environment")
    validate_runtime_state(state_dir, scope, require_writable=False)
    manifest = _json((Path(state_dir) / MANIFEST_NAME).read_text())
    return ScopeRegistration(scope, "bybit", environment, manifest["state_id"])


class PortfolioRiskBinding:
    def __init__(self, store: PortfolioReservationStore, registration: ScopeRegistration) -> None:
        if registration not in store.registrations:
            raise ValueError("Runtime account/state is not declared in the risk group")
        self.store, self.registration = store, registration

    def reserve(self, event: Any, *, quantity: Decimal, notional: Decimal):
        r = self.registration
        return self.store.reserve(
            scope=r.scope, venue=r.venue, environment=r.environment, state_id=r.state_id,
            intent_id=str(getattr(event, "order_id", "") or ""), symbol=str(event.symbol),
            side=str(event.side).lower(), quantity=quantity, notional=notional,
        )


def execution_risk_preflight(venue: str, environ: Any = None) -> PortfolioRiskBinding:
    """Run before account construction; no JSON exposure fallback or lost-db reset."""
    env = os.environ if environ is None else environ
    if venue != "bybit":
        raise ValueError("Execution requires a validated atomic risk collector; currently Bybit only")
    required = ("RISK_SCOPE", "RUNTIME_STATE_DIR", "PORTFOLIO_RISK_PATH", "PORTFOLIO_RISK_GROUP")
    if any(not env.get(name, "").strip() for name in required):
        raise ValueError("Execution requires explicit runtime state and shared portfolio risk identity")
    store = PortfolioReservationStore(_path(Path(env["PORTFOLIO_RISK_PATH"])), group=env["PORTFOLIO_RISK_GROUP"])
    from strategy.config import PORTFOLIO_NOTIONAL_CAPS_USD
    if store.caps != {s: Decimal(str(v)) for s, v in PORTFOLIO_NOTIONAL_CAPS_USD.items()}:
        raise ValueError("Risk store caps differ from the configured strategy portfolio caps")
    registration = runtime_registration(
        env["RISK_SCOPE"], Path(env["RUNTIME_STATE_DIR"]),
        bybit_environment(env.get("BYBIT_BASE_URL", "https://api-demo.bybit.com")),
    )
    return PortfolioRiskBinding(store, registration)


def collect_accounts(store: PortfolioReservationStore, accounts: dict[str, tuple[Any, Path]],
                     *, clock=time.time) -> tuple[AccountEvidence, ...]:
    """Only called by store.reconcile, which holds the global admission lock.

    Lock every local journal before reading any account. Unknown orders, orphan
    reservations or mismatching consumer positions prevent baseline replacement.
    """
    if set(accounts) != {r.scope for r in store.registrations}:
        raise ValueError("Risk reconciliation requires every declared account")
    with ExitStack() as stack:
        connections = {}
        for r in sorted(store.registrations, key=lambda item: item.scope):
            adapter, state_dir = accounts[r.scope]
            cfg = adapter._config
            if (adapter.venue != r.venue or cfg.category != "linear" or cfg.account_type != "UNIFIED"
                    or bybit_environment(cfg.base_url) != r.environment
                    or runtime_registration(r.scope, state_dir, r.environment) != r):
                raise ValueError("Account adapter/runtime identity differs from the risk registry")
            path = _path(Path(state_dir) / "orders.sqlite")
            db = sqlite3.connect(f"{path.as_uri()}?mode=rw", uri=True, timeout=10)
            stack.callback(db.close)
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            connections[r.scope] = db
        result = []
        for r in store.registrations:
            adapter, _ = accounts[r.scope]
            db = connections[r.scope]
            started = clock()
            positions = adapter.get_positions(strict=True)
            open_orders = adapter.get_open_orders(strict=True)
            position_rows = []
            seen_symbols = set()
            for p in positions:
                qty, price = Decimal(str(p.qty)), Decimal(str(p.mark_price))
                if (p.venue != r.venue or not qty.is_finite() or not price.is_finite() or price <= 0
                        or qty == 0 or p.symbol in seen_symbols):
                    raise ValueError("Malformed or hedge-mode portfolio position")
                seen_symbols.add(p.symbol)
                position_rows.append(PositionEvidence(p.symbol, "buy" if qty > 0 else "sell",
                                                      abs(qty), abs(qty) * price))
            orders = [dict(row) for row in db.execute("SELECT * FROM submissions WHERE scope=?", (r.scope,))]
            submissions = []
            owner_symbols: dict[str, set[str]] = {}
            for order in orders:
                if not order["owner"] or order["settled"] not in (0, 1) or not order["parent"]:
                    raise ValueError("Legacy/unattributed journal requires explicit recovery")
                owner_symbols.setdefault(order["owner"], set()).add(order["symbol"])
                submissions.append(SubmissionEvidence(order["intent_id"], order["symbol"], order["side"],
                                                      Decimal(order["qty"]), bool(order["settled"]), order["status"]))
            consumers = []
            for row in db.execute("SELECT owner,checkpoint FROM execution_consumers WHERE scope=?", (r.scope,)):
                symbols = owner_symbols.get(row["owner"], set())
                if len(symbols) != 1:
                    raise ValueError("Consumer must own exactly one known symbol")
                checkpoint = _json(row["checkpoint"])
                if not checkpoint.get("revision"):
                    raise ValueError("Consumer revision is missing")
                consumers.append(ConsumerPosition(row["owner"], next(iter(symbols)),
                                                  Decimal(checkpoint["position"]["qty"])))
            unconsumed = tuple(row[0] for row in db.execute(
                "SELECT fill_id FROM execution_fills WHERE scope=? AND delivered=0", (r.scope,)))
            result.append(AccountEvidence(r.scope, r.venue, r.environment, r.state_id, started,
                                         tuple(position_rows), tuple(submissions), tuple(consumers),
                                         tuple(o.order_id for o in open_orders), unconsumed))
        return tuple(result)


def refresh_portfolio(store: PortfolioReservationStore, accounts: dict[str, tuple[Any, Path]]) -> int:
    return store.reconcile(lambda: collect_accounts(store, accounts))


def load_registry(path: Path) -> list[dict[str, Any]]:
    data = _json(Path(path).read_text())
    rows = data.get("accounts")
    if not isinstance(rows, list) or not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Registry requires explicit accounts")
    if len({row["scope"] for row in rows}) != len(rows):
        raise ValueError("Duplicate account scope")
    return rows
