#!/usr/bin/env python3
"""Guardrails for the compose deploy workflow scope.

This module makes two truths executable:
1. ``scripts/deploy.sh`` only manages compose services.
2. The GitHub deploy workflow must fail fast when the diff touches
   host-managed trading artifacts, because compose deploy does not update
   ``bybit-alpha.service`` / ``bybit-mm.service`` on the host.
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable


COMPOSE_DEPLOY_SERVICES: tuple[str, ...] = (
    "quant-paper",
    "quant-live",
    "quant-framework",
)

_HOST_MANAGED_EXACT: frozenset[str] = frozenset(
    {
        "infra/systemd/bybit-alpha.service",
        "infra/systemd/bybit-mm.service",
        "scripts/run_bybit_alpha.py",
        "scripts/ops/run_bybit_alpha.py",
        "scripts/ops/alpha_runner.py",
        "scripts/run_bybit_mm.py",
    }
)

_HOST_MANAGED_PREFIXES: tuple[str, ...] = (
    "execution/market_maker/",
)


def unknown_compose_services(services: Iterable[str]) -> list[str]:
    allowed = set(COMPOSE_DEPLOY_SERVICES)
    return [svc for svc in services if svc not in allowed]


def host_managed_changes(paths: Iterable[str]) -> list[str]:
    hits: list[str] = []
    for raw_path in paths:
        path = raw_path.strip()
        if not path:
            continue
        if path in _HOST_MANAGED_EXACT or any(path.startswith(prefix) for prefix in _HOST_MANAGED_PREFIXES):
            hits.append(path)
    return hits


def _validate_services(services: list[str]) -> int:
    unknown = unknown_compose_services(services)
    if not unknown:
        print("compose deploy scope OK:", " ".join(services))
        return 0

    allowed = ", ".join(COMPOSE_DEPLOY_SERVICES)
    sys.stderr.write(
        "scripts/deploy.sh only manages compose services: "
        f"{allowed}\n"
        f"unsupported targets: {', '.join(unknown)}\n"
        "Host systemd services must be synced/restarted separately.\n"
    )
    return 1


def _guard_changed_files_from_stdin() -> int:
    changed = [line.rstrip("\n") for line in sys.stdin]
    host_hits = host_managed_changes(changed)
    if not host_hits:
        print("deploy scope OK: no host-managed runtime artifacts changed")
        return 0

    sys.stderr.write(
        "deploy workflow blocked: host-managed trading artifacts changed, "
        "but this workflow only redeploys compose services.\n"
        "sync the host systemd path separately for:\n"
    )
    for path in host_hits:
        sys.stderr.write(f"  - {path}\n")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate deploy scope boundaries.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-services", nargs="+", metavar="SERVICE")
    group.add_argument("--guard-changed-files-stdin", action="store_true")
    args = parser.parse_args(argv)

    if args.validate_services is not None:
        return _validate_services(list(args.validate_services))
    return _guard_changed_files_from_stdin()


if __name__ == "__main__":
    raise SystemExit(main())
