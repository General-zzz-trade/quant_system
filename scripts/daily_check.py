#!/usr/bin/env python3
"""Daily trading system health check.

Outputs a structured report to stdout (and optionally Telegram).
Run via systemd timer or cron at 09:00 JST daily.

Usage:
    python3 -m scripts.daily_check
    python3 -m scripts.daily_check --telegram
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
sys.path.insert(0, "/quant_system")


def _run(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, timeout=30).strip()
    except Exception:
        return ""


def check_service() -> dict:
    status = _run("systemctl is-active binance-alpha.service")
    uptime = _run("systemctl show binance-alpha.service -p ActiveEnterTimestamp --value")
    return {"status": status, "since": uptime}


def check_positions(log_path: str) -> dict:
    """Extract current positions from last MONITOR lines."""
    positions = {}
    try:
        lines = _run(f"tail -20 {log_path}").split("\n")
        for line in reversed(lines):
            if "MONITOR" not in line:
                continue
            m = re.search(r"MONITOR (\w+): \$([0-9.]+).*?(pos=(\w+) pnl=([+-]?[0-9.]+)%)?", line)
            if m:
                sym = m.group(1)
                if sym not in positions:
                    positions[sym] = {
                        "price": float(m.group(2)),
                        "position": m.group(4) or "FLAT",
                        "pnl_pct": float(m.group(5)) if m.group(5) else 0.0,
                    }
    except Exception:
        pass
    return positions


def check_errors(log_path: str) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    errors = int(_run(f"grep -c 'Error emitting' {log_path}") or "0")
    reconciles = int(_run(f"grep -c 'RECONCILE' {log_path} | tail -1") or "0")
    today_errors = int(_run(f"grep '{today}' {log_path} | grep -c 'Error emitting'") or "0")
    today_reconciles = int(_run(f"grep '{today}' {log_path} | grep -c 'RECONCILE'") or "0")
    return {
        "total_errors": errors,
        "today_errors": today_errors,
        "total_reconciles": reconciles,
        "today_reconciles": today_reconciles,
    }


def check_trades(log_path: str) -> dict:
    today = datetime.now().strftime("%Y-%m-%d")
    lines = _run(f"grep '{today}' {log_path} | grep 'Binance'").split("\n")
    buys = sum(1 for ln in lines if "buy" in ln.lower() and ln.strip())
    sells = sum(1 for ln in lines if "sell" in ln.lower() and ln.strip())
    return {"today_buys": buys, "today_sells": sells, "today_total": buys + sells}


def check_models() -> dict:
    models = {}
    for m in ["BTCUSDT_gate_v2", "ETHUSDT_gate_v2", "BTCUSDT_4h", "ETHUSDT_4h"]:
        try:
            with open(f"models_v8/{m}/config.json") as f:
                c = json.load(f)
            models[m] = {
                "sharpe": c.get("metrics", {}).get("sharpe", 0),
                "deadzone": c.get("deadzone", 0),
                "train_date": c.get("train_date", "?"),
            }
        except Exception:
            models[m] = {"error": "config missing"}
    return models


def check_data_freshness() -> dict:
    files = {}
    for f in ["BTCUSDT_1h", "ETHUSDT_1h", "BTCUSDT_funding", "BTCUSDT_ls_ratio"]:
        path = f"data_files/{f}.csv"
        if os.path.exists(path):
            age_h = (time.time() - os.path.getmtime(path)) / 3600
            files[f] = {"age_hours": round(age_h, 1), "ok": age_h < 12}
        else:
            files[f] = {"age_hours": -1, "ok": False}
    return files


def check_market() -> dict:
    import pandas as pd
    import numpy as np
    result = {}
    for sym in ["BTCUSDT", "ETHUSDT"]:
        try:
            df = pd.read_csv(f"data_files/{sym}_1h.csv").tail(25)
            c = df["close"].values
            result[sym] = {
                "price": round(c[-1], 2),
                "ret_24h_pct": round((c[-1] / c[0] - 1) * 100, 2),
                "vol_pct": round(float(np.std(np.diff(np.log(c)))) * 100, 2),
            }
        except Exception:
            result[sym] = {"error": "data unavailable"}
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--telegram", action="store_true")
    args = parser.parse_args()

    log_path = "/quant_system/logs/binance_alpha.log"
    now = datetime.now().strftime("%Y-%m-%d %H:%M JST")

    svc = check_service()
    pos = check_positions(log_path)
    errs = check_errors(log_path)
    trades = check_trades(log_path)
    models = check_models()
    data = check_data_freshness()
    market = check_market()

    report = []
    report.append(f"=== Daily Trading Report ({now}) ===")
    report.append("")

    # Service
    svc_icon = "OK" if svc["status"] == "active" else "DOWN"
    report.append(f"Service: {svc_icon} (since {svc['since']})")

    # Positions
    report.append("")
    for sym, p in pos.items():
        if p["position"] != "FLAT":
            report.append(f"{sym}: {p['position']} @ ${p['price']:,.1f} PnL={p['pnl_pct']:+.2f}%")
        else:
            report.append(f"{sym}: FLAT @ ${p['price']:,.1f}")

    # Trades
    report.append(f"\nToday: {trades['today_total']} trades ({trades['today_buys']}B/{trades['today_sells']}S)")

    # Errors
    if errs["today_errors"] > 0:
        report.append(f"WARN: {errs['today_errors']} emit errors today")
    if errs["today_reconciles"] > 0:
        report.append(f"WARN: {errs['today_reconciles']} reconcile syncs today")

    # Models
    report.append("\nModels:")
    for m, info in models.items():
        if "error" in info:
            report.append(f"  {m}: ERROR")
        else:
            report.append(f"  {m}: Sharpe={info['sharpe']} dz={info['deadzone']} trained={info['train_date']}")

    # Data
    stale = [f for f, d in data.items() if not d["ok"]]
    if stale:
        report.append(f"\nDATA STALE: {', '.join(stale)}")

    # Market
    report.append("\nMarket:")
    for sym, m in market.items():
        if "error" not in m:
            report.append(f"  {sym}: ${m['price']:,.1f} 24h={m['ret_24h_pct']:+.2f}% vol={m['vol_pct']:.2f}%")

    text = "\n".join(report)
    print(text)

    if args.telegram:
        try:
            from monitoring.notify import send_telegram
            send_telegram(text)
            print("\n(Sent to Telegram)")
        except Exception as e:
            print(f"\n(Telegram failed: {e})")


if __name__ == "__main__":
    main()
