#!/usr/bin/env python3
"""Grafana dashboard auto-import script.

Usage:
    python scripts/grafana_import.py --url http://localhost:3000 --token <api-key>
    python scripts/grafana_import.py --export /path/to/output.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.dashboards.grafana_config import DashboardConfig, GrafanaDashboardGenerator


def generate_dashboard(strategies: tuple[str, ...] = ()) -> dict:
    """Generate the full dashboard JSON."""
    config = DashboardConfig(
        title="Quant Trading System",
        refresh_interval="10s",
        time_range="24h",
        strategies=strategies,
        datasource="Prometheus",
        uid="quant-system-main",
    )
    return GrafanaDashboardGenerator(config).generate()


def export_to_file(path: Path, strategies: tuple[str, ...] = ()) -> None:
    """Export dashboard JSON to a file."""
    data = generate_dashboard(strategies)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Dashboard exported to {path}")


def import_to_grafana(url: str, token: str, strategies: tuple[str, ...] = ()) -> None:
    """Import dashboard into Grafana via HTTP API."""
    import urllib.request
    import urllib.error

    data = generate_dashboard(strategies)
    payload = json.dumps({"dashboard": data["dashboard"], "overwrite": True}).encode()

    api_url = f"{url.rstrip('/')}/api/dashboards/db"
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(f"Dashboard imported: {result.get('url', 'OK')}")
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        print(f"Import failed ({e.code}): {body}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grafana dashboard import/export")
    parser.add_argument("--url", help="Grafana URL (e.g., http://localhost:3000)")
    parser.add_argument("--token", help="Grafana API token")
    parser.add_argument("--export", help="Export dashboard JSON to file path")
    parser.add_argument("--strategies", default="", help="Comma-separated strategy names")
    args = parser.parse_args()

    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())

    if args.export:
        export_to_file(Path(args.export), strategies)
    elif args.url and args.token:
        import_to_grafana(args.url, args.token, strategies)
    else:
        # Default: export to deploy/grafana/dashboards/
        default_path = Path(__file__).resolve().parents[1] / "deploy" / "grafana" / "dashboards" / "trading.json"
        export_to_file(default_path, strategies)


if __name__ == "__main__":
    main()
