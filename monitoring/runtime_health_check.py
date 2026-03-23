#!/usr/bin/env python3
"""Fail-fast runtime health checks for the active Bybit host services.

This script encodes the production runbook's stronger success criteria:
`systemd active` alone is not enough. A healthy runtime must also show fresh
business logs and recent runtime/account evidence.

Usage:
    python3 -m scripts.ops.runtime_health_check
    python3 -m scripts.ops.runtime_health_check --service alpha
    python3 -m scripts.ops.runtime_health_check --service mm --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

from execution.adapters.bybit.adapter import BybitAdapter
from execution.adapters.bybit.config import BybitConfig
from runner.kill_latch import build_bybit_kill_latch


DEFAULT_ALPHA_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "ETHUSDT_15m",
)
DEFAULT_MM_SYMBOL = "ETHUSDT"
DEFAULT_ENV_FILE = ".env"
MAX_LOG_LINES = 4000
ALPHA_KILL_PATTERNS = (
    "'killed': True",
    '"killed": true',
    "DRAWDOWN KILL",
)

_TIMESTAMP_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:,\d{3})?"
)


@dataclass(frozen=True)
class ServiceSpec:
    key: str
    service_name: str
    log_path: str
    description: str
    default_symbols: tuple[str, ...]
    log_max_age_s: int
    activity_window_s: int
    activity_markers: tuple[str, ...]


ALPHA_SPEC = ServiceSpec(
    key="alpha",
    service_name="bybit-alpha.service",
    log_path="logs/bybit_alpha.log",
    description="directional alpha",
    default_symbols=DEFAULT_ALPHA_SYMBOLS,
    log_max_age_s=600,
    activity_window_s=900,
    activity_markers=(
        "WS HEARTBEAT",
        "Opened ",
        " CLOSE ",
        "COMBO OPEN",
        "COMBO CLOSE",
        "orderLinkId",
    ),
)

MM_SPEC = ServiceSpec(
    key="mm",
    service_name="bybit-mm.service",
    log_path="logs/bybit_mm.log",
    description="market maker",
    default_symbols=(DEFAULT_MM_SYMBOL,),
    log_max_age_s=180,
    activity_window_s=300,
    activity_markers=(
        "FILL ",
        "METRICS ",
        "tick=",
        "Market maker running",
        "WS subscribed",
    ),
)

SERVICE_SPECS = {
    ALPHA_SPEC.key: ALPHA_SPEC,
    MM_SPEC.key: MM_SPEC,
}


def parse_log_timestamp(line: str) -> Optional[datetime]:
    match = _TIMESTAMP_RE.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def tail_lines(path: Path, *, max_lines: int = MAX_LOG_LINES) -> list[str]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        return list(deque(handle, maxlen=max_lines))


def read_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            data[key] = value
    return data


def merge_runtime_env(env_file: str = DEFAULT_ENV_FILE) -> dict[str, str]:
    merged = dict(read_env_file(Path(env_file)))
    merged.update({k: v for k, v in os.environ.items() if v})
    return merged


def venue_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    seen: set[str] = set()
    for symbol in symbols:
        venue_symbol = symbol.split("_", 1)[0].upper()
        if venue_symbol and venue_symbol not in seen:
            seen.add(venue_symbol)
            normalized.append(venue_symbol)
    return tuple(normalized)


def service_state(service_name: str) -> str:
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return "systemctl_not_found"
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def summarize_log_health(
    *,
    spec: ServiceSpec,
    log_path: str,
    now: Optional[datetime] = None,
    log_max_age_s: Optional[int] = None,
    activity_window_s: Optional[int] = None,
) -> dict[str, Any]:
    current_time = now or datetime.now()
    max_age = spec.log_max_age_s if log_max_age_s is None else log_max_age_s
    activity_window = (
        spec.activity_window_s if activity_window_s is None else activity_window_s
    )
    path = Path(log_path)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "fresh": False,
        "last_ts": None,
        "last_age_s": None,
        "marker_counts": {marker: 0 for marker in spec.activity_markers},
        "kill_active": False,
        "kill_detail": None,
    }
    if not path.exists():
        return result

    lines = tail_lines(path)
    if not lines:
        return result

    cutoff = current_time - timedelta(seconds=activity_window)
    last_ts: Optional[datetime] = None
    marker_counts: Counter[str] = Counter()

    for line in lines:
        ts = parse_log_timestamp(line)
        if ts is not None:
            last_ts = ts
        if ts is None or ts < cutoff:
            continue
        if spec.key == ALPHA_SPEC.key and any(pattern in line for pattern in ALPHA_KILL_PATTERNS):
            result["kill_active"] = True
            result["kill_detail"] = line.strip()
        for marker in spec.activity_markers:
            if marker in line:
                marker_counts[marker] += 1

    result["marker_counts"] = {
        marker: int(marker_counts.get(marker, 0))
        for marker in spec.activity_markers
    }
    if last_ts is None:
        return result

    age_s = max(0.0, (current_time - last_ts).total_seconds())
    result["last_ts"] = last_ts.isoformat(sep=" ")
    result["last_age_s"] = round(age_s, 3)
    result["fresh"] = age_s <= max_age
    return result


def build_bybit_adapter(env_file: str = DEFAULT_ENV_FILE) -> tuple[Optional[BybitAdapter], str]:
    env = merge_runtime_env(env_file)
    api_key = env.get("BYBIT_API_KEY", "")
    api_secret = env.get("BYBIT_API_SECRET", "")
    if not api_key or not api_secret:
        return None, "missing_credentials"
    config = BybitConfig(
        api_key=api_key,
        api_secret=api_secret,
        base_url=env.get("BYBIT_BASE_URL", "https://api-demo.bybit.com"),
    )
    return BybitAdapter(config), "ok"


def summarize_account_truth(
    *,
    symbols: Iterable[str],
    activity_window_s: int,
    env_file: str = DEFAULT_ENV_FILE,
    adapter: Optional[BybitAdapter] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    if adapter is None:
        adapter, status = build_bybit_adapter(env_file)
        if adapter is None:
            return {"status": "skipped", "reason": status}
    current_time = now or datetime.now()
    since_ms = int((current_time - timedelta(seconds=activity_window_s)).timestamp() * 1000)
    v_symbols = venue_symbols(symbols)
    try:
        balances = adapter.get_balances()
        usdt_balance = balances.get("USDT")
        positions = adapter.get_positions()
        open_orders = []
        recent_fills = []
        for symbol in v_symbols:
            open_orders.extend(adapter.get_open_orders(symbol=symbol))
            recent_fills.extend(
                fill for fill in adapter.get_recent_fills(symbol=symbol)
                if fill.ts_ms >= since_ms
            )
        return {
            "status": "ok",
            "reason": "",
            "venue_symbols": list(v_symbols),
            "usdt_total": str(usdt_balance.total) if usdt_balance else None,
            "positions": len(positions),
            "open_orders": len(open_orders),
            "recent_fills": len(recent_fills),
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}


def runtime_evidence_present(
    spec: ServiceSpec,
    log_health: dict[str, Any],
    account_truth: dict[str, Any],
) -> bool:
    marker_counts = log_health.get("marker_counts", {})
    if any(int(count) > 0 for count in marker_counts.values()):
        return True
    if spec.key == ALPHA_SPEC.key:
        # Active alpha shares an account with other strategies, so account-side
        # activity is not trustworthy as sole liveness evidence.
        return False
    if account_truth.get("status") != "ok":
        return False
    return any(
        int(account_truth.get(field, 0)) > 0
        for field in ("positions", "open_orders", "recent_fills")
    )


def summarize_kill_latch(
    *,
    spec: ServiceSpec,
    symbols: Iterable[str],
    env_file: str = DEFAULT_ENV_FILE,
    adapter: Optional[BybitAdapter] = None,
) -> dict[str, Any]:
    if adapter is None:
        adapter, status = build_bybit_adapter(env_file)
        if adapter is None:
            return {"status": "skipped", "reason": status, "armed": False, "path": None, "record": None}
    scope_name = "portfolio" if spec.key == ALPHA_SPEC.key else venue_symbols(symbols)[0]
    try:
        latch = build_bybit_kill_latch(
            adapter=adapter,
            service_name=spec.service_name,
            scope_name=scope_name,
        )
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "armed": False, "path": None, "record": None}
    record = latch.read()
    return {
        "status": "ok",
        "reason": "",
        "armed": latch.is_armed(),
        "path": str(latch.path),
        "record": record,
    }


def evaluate_service_health(
    *,
    spec: ServiceSpec,
    log_path: Optional[str] = None,
    symbols: Optional[Iterable[str]] = None,
    log_max_age_s: Optional[int] = None,
    activity_window_s: Optional[int] = None,
    env_file: str = DEFAULT_ENV_FILE,
    require_account: bool = False,
    now: Optional[datetime] = None,
    adapter: Optional[BybitAdapter] = None,
) -> dict[str, Any]:
    selected_symbols = tuple(symbols or spec.default_symbols)
    activity_window = spec.activity_window_s if activity_window_s is None else activity_window_s
    result = {
        "service": spec.service_name,
        "runtime": spec.description,
        "symbols": list(selected_symbols),
        "service_state": service_state(spec.service_name),
    }

    log_health = summarize_log_health(
        spec=spec,
        log_path=log_path or spec.log_path,
        now=now,
        log_max_age_s=log_max_age_s,
        activity_window_s=activity_window,
    )
    result["log"] = log_health
    kill_latch = summarize_kill_latch(
        spec=spec,
        symbols=selected_symbols,
        env_file=env_file,
        adapter=adapter,
    )
    result["kill_latch"] = kill_latch
    account_truth = summarize_account_truth(
        symbols=selected_symbols,
        activity_window_s=activity_window,
        env_file=env_file,
        adapter=adapter,
        now=now,
    )
    result["account"] = account_truth

    problems: list[str] = []
    if result["service_state"] != "active":
        problems.append(f"service_state={result['service_state']}")
    if not log_health.get("exists"):
        problems.append("log_missing")
    elif not log_health.get("fresh"):
        problems.append("log_stale")
    if not runtime_evidence_present(spec, log_health, account_truth):
        problems.append("no_recent_runtime_evidence")
    if log_health.get("kill_active"):
        problems.append("portfolio_killed")
    if kill_latch.get("armed"):
        problems.append("persistent_kill_latched")
    if require_account and account_truth.get("status") != "ok":
        problems.append(f"account_{account_truth.get('status', 'unknown')}")

    result["ok"] = not problems
    result["problems"] = problems
    return result


def format_service_health(result: dict[str, Any]) -> str:
    status = "PASS" if result.get("ok") else "FAIL"
    log_health = result.get("log", {})
    account = result.get("account", {})
    kill_latch = result.get("kill_latch", {})
    marker_counts = log_health.get("marker_counts", {})
    marker_summary = ", ".join(
        f"{marker}={count}" for marker, count in marker_counts.items()
    ) or "(none)"
    lines = [
        f"[{status}] {result.get('service')} ({result.get('runtime')})",
        f"  state={result.get('service_state')}",
        (
            "  log="
            f"{log_health.get('path')} fresh={log_health.get('fresh')} "
            f"last_ts={log_health.get('last_ts')} age_s={log_health.get('last_age_s')}"
        ),
        f"  log_markers={marker_summary}",
        (
            "  account="
            f"{account.get('status')} positions={account.get('positions', 0)} "
            f"open_orders={account.get('open_orders', 0)} "
            f"recent_fills={account.get('recent_fills', 0)} "
            f"usdt_total={account.get('usdt_total')}"
        ),
        (
            "  kill_latch="
            f"{kill_latch.get('status')} armed={kill_latch.get('armed')} "
            f"path={kill_latch.get('path')}"
        ),
    ]
    if log_health.get("kill_active"):
        lines.append(f"  alpha_kill_active=True detail={log_health.get('kill_detail')}")
    if kill_latch.get("armed"):
        lines.append(f"  persistent_kill_detail={kill_latch.get('record')}")
    if result.get("problems"):
        lines.append(f"  problems={', '.join(result['problems'])}")
    return "\n".join(lines)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-fast runtime health check for active Bybit host services",
    )
    parser.add_argument(
        "--service",
        choices=("all", "alpha", "mm"),
        default="all",
        help="Which active host service to check",
    )
    parser.add_argument(
        "--alpha-log",
        default=ALPHA_SPEC.log_path,
        help=f"Directional alpha log path (default: {ALPHA_SPEC.log_path})",
    )
    parser.add_argument(
        "--mm-log",
        default=MM_SPEC.log_path,
        help=f"Market-maker log path (default: {MM_SPEC.log_path})",
    )
    parser.add_argument(
        "--alpha-symbols",
        nargs="*",
        default=list(DEFAULT_ALPHA_SYMBOLS),
        help="Directional alpha runtime symbols",
    )
    parser.add_argument(
        "--mm-symbol",
        default=DEFAULT_MM_SYMBOL,
        help="Market-maker venue symbol",
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help=f"Optional env file for Bybit credentials (default: {DEFAULT_ENV_FILE})",
    )
    parser.add_argument(
        "--require-account",
        action="store_true",
        help="Fail when account truth cannot be queried",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON",
    )
    return parser.parse_args(argv)


def collect_runtime_health(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected = ("alpha", "mm") if args.service == "all" else (args.service,)
    results = []
    for service_key in selected:
        if service_key == "alpha":
            results.append(
                evaluate_service_health(
                    spec=ALPHA_SPEC,
                    log_path=args.alpha_log,
                    symbols=args.alpha_symbols,
                    env_file=args.env_file,
                    require_account=args.require_account,
                )
            )
        else:
            results.append(
                evaluate_service_health(
                    spec=MM_SPEC,
                    log_path=args.mm_log,
                    symbols=(args.mm_symbol,),
                    env_file=args.env_file,
                    require_account=args.require_account,
                )
            )
    return results


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    results = collect_runtime_health(args)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "ok": all(r.get("ok") for r in results),
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("RUNTIME HEALTH CHECK")
        for result in results:
            print(format_service_health(result))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
