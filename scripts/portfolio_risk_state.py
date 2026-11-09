"""Explicit risk initialization and read-only-account reconciliation.

Examples (never invoked automatically by CI):
  python -m scripts.portfolio_risk_state init --registry accounts.json --group G --path risk.sqlite
  python -m scripts.portfolio_risk_state reconcile --registry accounts.json --group G --path risk.sqlite

Registry accounts each declare scope, state_dir, environment and credential_prefix.
Only the reconcile command queries private account GET endpoints. Neither command
places orders, configures leverage or resets an existing baseline.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import os
from pathlib import Path
import re

from runner.portfolio_risk import load_registry, refresh_portfolio, runtime_registration
from shared.portfolio_reservations import PortfolioReservationStore


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "inspect", "reconcile"))
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--registry", type=Path)
    args = parser.parse_args(argv)
    if args.action != "inspect" and args.registry is None:
        parser.error("init and reconcile require --registry")
    rows = load_registry(args.registry) if args.registry else []
    if args.action == "init":
        from strategy.config import PORTFOLIO_NOTIONAL_CAPS_USD, PORTFOLIO_RISK_MAX_STALE_SEC
        registrations = [runtime_registration(row["scope"], Path(row["state_dir"]), row["environment"])
                         for row in rows]
        store = PortfolioReservationStore.initialize(
            args.path, group=args.group, registrations=registrations,
            caps={s: Decimal(str(v)) for s, v in PORTFOLIO_NOTIONAL_CAPS_USD.items()},
            max_age_seconds=PORTFOLIO_RISK_MAX_STALE_SEC,
        )
    else:
        store = PortfolioReservationStore(args.path, group=args.group)
    if args.action == "reconcile":
        from execution.adapters.bybit.adapter import BybitAdapter
        from execution.adapters.bybit.config import BybitConfig
        urls = {"demo": "https://api-demo.bybit.com", "testnet": "https://api-testnet.bybit.com",
                "live": "https://api.bybit.com"}
        accounts = {}
        registrations = {r.scope: r for r in store.registrations}
        # Validate every identity and credential binding before the first request.
        for row in rows:
            scope = row["scope"]
            expected = runtime_registration(scope, Path(row["state_dir"]), row["environment"])
            if registrations.get(scope) != expected:
                raise ValueError("Reconciler registry differs from durable risk group")
            prefix = row["credential_prefix"]
            if not isinstance(prefix, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", prefix):
                raise ValueError("Invalid credential environment prefix")
            key, secret = os.environ.get(prefix + "_API_KEY"), os.environ.get(prefix + "_API_SECRET")
            if not key or not secret:
                raise ValueError("Declared account credentials unavailable")
            config = BybitConfig(api_key=key, api_secret=secret, base_url=urls[row["environment"]])
            accounts[scope] = (BybitAdapter(config), Path(row["state_dir"]))
        refresh_portfolio(store, accounts)
    print(json.dumps(store.snapshot(), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
