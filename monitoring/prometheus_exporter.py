#!/usr/bin/env python3
"""Lightweight Prometheus exporter for the quant-system observability stack.

Serves ``GET /metrics`` on ``QUANT_METRICS_PORT`` (default 9102) in the
Prometheus text exposition format (version 0.0.4).  No external deps —
we emit the text manually, so the system doesn't need ``prometheus_client``
installed on the box.

Metrics exposed:
- quant_service_up{service=...}              — 1/0 systemd active
- quant_service_memory_bytes{service=...}    — memory usage in bytes
- quant_service_uptime_seconds{service=...}  — seconds since ActiveEnter
- quant_signal_zscore{venue=,symbol=}        — latest z-score per venue/sym
- quant_signal_direction{venue=,symbol=}     — -1/0/+1 latest discretised
- quant_position_qty{venue=,symbol=}         — signed qty (coin units)
- quant_position_notional_usd{venue=,symbol=}— abs(qty) * price
- quant_position_is_testnet{venue=,symbol=}  — 1/0 sim-money flag
- quant_portfolio_long_usd{symbol=}          — cross-venue long notional
- quant_portfolio_short_usd{symbol=}         — cross-venue short notional
- quant_portfolio_cap_usd{symbol=}           — per-symbol enforcement cap
- quant_portfolio_cap_usage{symbol=,side=}   — fraction of cap used
- quant_model_ic_training{model=,horizon=}   — training IC
- quant_model_ic_live{model=,horizon=,window=} — live IC per window
- quant_model_ic_status{model=}              — 1 GREEN / 0 YELLOW / -1 RED
- quant_okx_ramp_level                        — current cap level 0..4
- quant_okx_ramp_cap_usd                      — current $ cap value
- quant_trades_24h{venue=}                    — # closed trades last 24h
- quant_pnl_24h_usd{venue=}                   — $ PnL last 24h

Run::

    python3 -m monitoring.prometheus_exporter
    # or via systemd: prometheus-exporter.service

Grafana prometheus datasource URL: http://localhost:9102/metrics
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, "/quant_system")

logger = logging.getLogger(__name__)

DEFAULT_PORT = int(os.environ.get("QUANT_METRICS_PORT", "9102"))

# File locations (read-only)
PORTFOLIO_RISK_PATH = Path("/quant_system/data/runtime/portfolio_risk.json")
IC_HEALTH_PATH = Path("/quant_system/data/runtime/ic_health.json")
OKX_RAMP_STATE_PATH = Path("/quant_system/data/runtime/okx_ramp_state.json")
AUDIT_DIR = Path("/quant_system/data/runtime")


# ── Helpers ───────────────────────────────────────────────────────
def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt_metric(name: str, value: float, **labels: Any) -> str:
    if labels:
        labels_str = ",".join(f'{k}="{_escape(str(v))}"' for k, v in labels.items())
        return f"{name}{{{labels_str}}} {value}"
    return f"{name} {value}"


def _read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        logger.debug("read %s failed: %s", path, exc)
    return None


def _systemd_show(unit: str) -> dict[str, str]:
    try:
        out = subprocess.check_output(
            ["systemctl", "show", unit,
             "--property=ActiveState,MainPID,ActiveEnterTimestampMonotonic,MemoryCurrent"],
            text=True, timeout=3,
        )
        return {
            k: v for k, v in
            (line.split("=", 1) for line in out.splitlines() if "=" in line)
        }
    except Exception:
        return {}


# ── Metric collectors ─────────────────────────────────────────────
def _collect_services(lines: list[str]) -> None:
    for service, unit in (("binance-alpha", "binance-alpha.service"),
                          ("okx-alpha", "okx-alpha.service")):
        props = _systemd_show(unit)
        active = 1 if props.get("ActiveState") == "active" else 0
        lines.append(_fmt_metric("quant_service_up", active, service=service))

        mem_bytes = props.get("MemoryCurrent", "0")
        try:
            mem = int(mem_bytes)
            # systemd returns [not set] as 18446744073709551615 (UINT64_MAX)
            if mem > 0 and mem < 2**63:
                lines.append(_fmt_metric(
                    "quant_service_memory_bytes", mem, service=service,
                ))
        except (TypeError, ValueError):
            pass


def _collect_portfolio(lines: list[str]) -> None:
    snap = _read_json(PORTFOLIO_RISK_PATH)
    if not snap:
        return
    for sym, data in (snap.get("symbols") or {}).items():
        long_usd = float(data.get("total_long_notional_usd") or 0)
        short_usd = float(data.get("total_short_notional_usd") or 0)
        cap = float(data.get("max_total_notional_usd") or 0)

        lines.append(_fmt_metric("quant_portfolio_long_usd", long_usd, symbol=sym))
        lines.append(_fmt_metric("quant_portfolio_short_usd", short_usd, symbol=sym))
        lines.append(_fmt_metric("quant_portfolio_cap_usd", cap, symbol=sym))

        if cap > 0:
            lines.append(_fmt_metric(
                "quant_portfolio_cap_usage", long_usd / cap,
                symbol=sym, side="long",
            ))
            lines.append(_fmt_metric(
                "quant_portfolio_cap_usage", short_usd / cap,
                symbol=sym, side="short",
            ))

        # Per-venue positions
        for venue, v_data in (data.get("venues") or {}).items():
            qty = float(v_data.get("qty") or 0)
            notional = float(v_data.get("notional_usd") or 0)
            is_testnet = 1 if v_data.get("is_testnet") else 0
            lines.append(_fmt_metric(
                "quant_position_qty", qty, venue=venue, symbol=sym,
            ))
            lines.append(_fmt_metric(
                "quant_position_notional_usd", notional, venue=venue, symbol=sym,
            ))
            lines.append(_fmt_metric(
                "quant_position_is_testnet", is_testnet, venue=venue, symbol=sym,
            ))


def _collect_signals(lines: list[str]) -> None:
    # Scan all per-venue audit files for the most recent signal per symbol
    for venue in ("binance", "okx", "bybit"):
        path = AUDIT_DIR / f"decision_audit_{venue}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue
        latest: dict[str, dict] = {}
        try:
            for line in path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") != "signal":
                    continue
                sym = e.get("symbol", "")
                ts = e.get("ts", 0)
                if ts >= latest.get(sym, {}).get("ts", 0):
                    latest[sym] = e
        except Exception:
            continue
        for sym, e in latest.items():
            z = float(e.get("z_score") or 0)
            sig = int(e.get("signal") or 0)
            lines.append(_fmt_metric(
                "quant_signal_zscore", z, venue=venue, symbol=sym,
            ))
            lines.append(_fmt_metric(
                "quant_signal_direction", sig, venue=venue, symbol=sym,
            ))


def _collect_ic_health(lines: list[str]) -> None:
    data = _read_json(IC_HEALTH_PATH)
    if not data:
        return
    STATUS_MAP = {"GREEN": 1, "YELLOW": 0, "RED": -1}
    for m in data.get("models") or []:
        name = m.get("model") or m.get("symbol") or "?"
        status_str = m.get("overall_status", "YELLOW")
        lines.append(_fmt_metric(
            "quant_model_ic_status", STATUS_MAP.get(status_str, 0),
            model=name,
        ))
        train_ic = m.get("training_avg_ic")
        if train_ic is not None:
            lines.append(_fmt_metric(
                "quant_model_ic_training", float(train_ic), model=name,
            ))
        for h in m.get("horizons") or []:
            h_id = str(h.get("horizon", "?"))
            t_ic = h.get("training_ic")
            if t_ic is not None:
                lines.append(_fmt_metric(
                    "quant_model_ic_training", float(t_ic),
                    model=name, horizon=h_id,
                ))
            for win, win_data in (h.get("windows") or {}).items():
                ic_val = win_data.get("ic")
                if ic_val is not None:
                    lines.append(_fmt_metric(
                        "quant_model_ic_live", float(ic_val),
                        model=name, horizon=h_id, window=win,
                    ))


def _collect_okx_ramp(lines: list[str]) -> None:
    state = _read_json(OKX_RAMP_STATE_PATH)
    if not state:
        return
    lvl = int(state.get("current_level") or 0)
    cap = float(state.get("current_cap") or 0)
    lines.append(_fmt_metric("quant_okx_ramp_level", lvl))
    lines.append(_fmt_metric("quant_okx_ramp_cap_usd", cap))


def _collect_24h_pnl(lines: list[str]) -> None:
    try:
        from monitoring.daily_pnl_alert import build_venue_summaries
        per_venue = build_venue_summaries()
        for venue, summary in per_venue.items():
            trades = int(summary.get("n_trades", 0))
            pnl = float(summary.get("total_pnl", 0))
            lines.append(_fmt_metric("quant_trades_24h", trades, venue=venue))
            lines.append(_fmt_metric("quant_pnl_24h_usd", pnl, venue=venue))
    except Exception as exc:
        logger.debug("24h pnl collect failed: %s", exc)


def render_metrics() -> str:
    """Render all metrics in Prometheus text format."""
    lines: list[str] = []
    # Emit a single HELP/TYPE header per metric family the first time
    # (Grafana doesn't require these, but scrapers are happier with them).
    header_lines = [
        "# HELP quant_service_up Service systemd ActiveState (1=active)",
        "# TYPE quant_service_up gauge",
        "# HELP quant_signal_zscore Latest per-symbol z-score",
        "# TYPE quant_signal_zscore gauge",
        "# HELP quant_position_qty Current position size (signed coin units)",
        "# TYPE quant_position_qty gauge",
        "# HELP quant_portfolio_cap_usage Fraction of portfolio cap used",
        "# TYPE quant_portfolio_cap_usage gauge",
        "# HELP quant_model_ic_status Model IC health (1=GREEN, 0=YELLOW, -1=RED)",
        "# TYPE quant_model_ic_status gauge",
        "# HELP quant_okx_ramp_level Current OKX notional cap ramp level (0-4)",
        "# TYPE quant_okx_ramp_level gauge",
    ]
    lines.extend(header_lines)

    for collector in (
        _collect_services,
        _collect_portfolio,
        _collect_signals,
        _collect_ic_health,
        _collect_okx_ramp,
        _collect_24h_pnl,
    ):
        try:
            collector(lines)
        except Exception as exc:
            logger.warning("collector %s failed: %s", collector.__name__, exc)

    # Heartbeat: current UTC epoch so Grafana can detect a dead exporter
    lines.append(_fmt_metric("quant_exporter_last_scrape_unixtime", int(time.time())))
    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler convention
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return
        body = render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # silence default stdout logging
        return


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--dump", action="store_true",
                        help="Print metrics once to stdout and exit")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.dump:
        print(render_metrics())
        return 0

    server = HTTPServer((args.bind, args.port), MetricsHandler)
    logger.info("quant_prometheus_exporter listening on http://%s:%d/metrics",
                args.bind, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
