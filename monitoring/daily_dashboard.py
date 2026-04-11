"""Daily observation dashboard — unified view of trading health.

Prints a comprehensive status report covering:
- Service uptime + memory
- Today's trades and PnL
- IC health per model
- Active positions
- Disk + system health
- Recent signals

Usage:
    python3 -m monitoring.daily_dashboard
    python3 -m monitoring.daily_dashboard --json  # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from monitoring.daily_pnl_alert import build_daily_summary
from monitoring.watchdog import check_disk_space


def _service_status(unit: str) -> dict[str, Any]:
    """Query systemd for service uptime + memory."""
    result: dict[str, Any] = {"unit": unit}
    try:
        out = subprocess.check_output(
            ["systemctl", "show", unit, "--property=ActiveState,ActiveEnterTimestamp,MemoryCurrent"],
            text=True, timeout=5,
        )
        for line in out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                result[k] = v.strip()
    except Exception as e:
        result["error"] = str(e)
    return result


def _ic_health_summary() -> dict[str, Any]:
    """Read latest IC health report."""
    path = Path("data/runtime/ic_health.json")
    if not path.exists():
        return {"error": "no ic_health.json"}
    try:
        data = json.loads(path.read_text())
        out: dict[str, Any] = {"models": [], "evaluated_at": data.get("timestamp", "?")}
        for m in data.get("models", []):
            out["models"].append({
                "name": m.get("model", m.get("symbol", "?")),
                "status": m.get("overall_status", "?"),
            })
        return out
    except Exception as e:
        return {"error": str(e)}


def _recent_signals(n: int = 10) -> list[dict]:
    """Last N signal events from decision audit (all venues merged, ts-sorted)."""
    from monitoring.decision_audit import audit_path_for
    merged: list[dict] = []
    for venue in ("binance", "okx", "bybit"):
        path = audit_path_for(venue)
        if not path.exists():
            continue
        try:
            for line in path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("type") != "signal":
                    continue
                if "venue" not in e:
                    e["venue"] = venue
                merged.append(e)
        except Exception:
            continue
    # Fall back to the legacy shared file if no per-venue data
    if not merged:
        legacy = Path("data/runtime/decision_audit.jsonl")
        if legacy.exists():
            try:
                for line in legacy.read_text().splitlines():
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("type") == "signal":
                        merged.append(e)
            except Exception:
                pass
    merged.sort(key=lambda e: e.get("ts", 0))
    return merged[-n:]


def _load_env_file() -> dict[str, str]:
    """Parse /quant_system/.env without touching os.environ (side-effect free)."""
    env_path = Path(".env")
    env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def _positions_for_binance(env: dict[str, str]) -> dict[str, Any]:
    try:
        from execution.adapters.binance.adapter import BinanceAdapter
        from execution.adapters.binance.config import BinanceConfig
        # Dashboard reads whichever key set is configured.  The runner uses
        # BINANCE_TESTNET_API_* for testnet; live uses BINANCE_API_*.
        key = (
            env.get("BINANCE_TESTNET_API_KEY")
            or env.get("BINANCE_API_KEY")
            or os.environ.get("BINANCE_TESTNET_API_KEY", "")
            or os.environ.get("BINANCE_API_KEY", "")
        )
        secret = (
            env.get("BINANCE_TESTNET_API_SECRET")
            or env.get("BINANCE_API_SECRET")
            or os.environ.get("BINANCE_TESTNET_API_SECRET", "")
            or os.environ.get("BINANCE_API_SECRET", "")
        )
        if not key or not secret:
            return {"error": "no credentials"}
        # BinanceConfig doesn't take base_url; it resolves from the testnet flag.
        # The dashboard's binance queries follow whatever the runner is using —
        # infer testnet=True if only BINANCE_TESTNET_* is set.
        is_testnet = bool(env.get("BINANCE_TESTNET_API_KEY") or os.environ.get("BINANCE_TESTNET_API_KEY"))
        adapter = BinanceAdapter(BinanceConfig(api_key=key, api_secret=secret, testnet=is_testnet))
        positions = adapter.get_positions()
        open_pos = [
            {"symbol": p.symbol, "side": p.side, "qty": float(p.qty), "entry": float(p.entry_price)}
            for p in positions if float(getattr(p, "qty", 0)) != 0
        ]
        return {"count": len(open_pos), "positions": open_pos}
    except Exception as e:
        return {"error": str(e)[:80]}


def _positions_for_okx(env: dict[str, str]) -> dict[str, Any]:
    try:
        from execution.adapters.okx.adapter import OkxAdapter
        from execution.adapters.okx.config import OkxConfig
        key = env.get("OKX_API_KEY", os.environ.get("OKX_API_KEY", ""))
        secret = env.get("OKX_API_SECRET", os.environ.get("OKX_API_SECRET", ""))
        passphrase = env.get("OKX_API_PASSPHRASE", os.environ.get("OKX_API_PASSPHRASE", ""))
        if not (key and secret and passphrase):
            return {"error": "no credentials"}
        base_url = env.get("OKX_BASE_URL", os.environ.get("OKX_BASE_URL", "https://www.okx.com"))
        adapter = OkxAdapter(OkxConfig(
            api_key=key, api_secret=secret, passphrase=passphrase, base_url=base_url,
        ))
        if not adapter.connect():
            return {"error": "connect failed"}
        positions = adapter.get_positions()
        open_pos = [
            {"symbol": p.symbol, "side": p.side, "qty": float(p.qty), "entry": float(p.entry_price)}
            for p in positions if float(getattr(p, "qty", 0)) != 0
        ]
        # Also fetch equity for context
        bal = adapter.get_balances()
        usdt = bal.get("USDT")
        equity = float(usdt.total) if usdt else None
        return {"count": len(open_pos), "positions": open_pos, "equity_usdt": equity}
    except Exception as e:
        return {"error": str(e)[:80]}


def _open_positions_all() -> dict[str, dict[str, Any]]:
    """Query every configured venue for current positions."""
    env = _load_env_file()
    return {
        "binance": _positions_for_binance(env),
        "okx": _positions_for_okx(env),
    }


def build_dashboard() -> dict[str, Any]:
    """Build complete multi-venue dashboard report."""
    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "services": {
            "binance": _service_status("binance-alpha.service"),
            "okx": _service_status("okx-alpha.service"),
        },
        "disk": check_disk_space(),
        "ic_health": _ic_health_summary(),
        "today_pnl": build_daily_summary(),
        "positions": _open_positions_all(),
        "recent_signals": _recent_signals(10),
    }


def _fmt_memory(bytes_str: str) -> str:
    try:
        b = int(bytes_str)
        return f"{b / (1024*1024):.0f}MB"
    except (ValueError, TypeError):
        return bytes_str


def print_dashboard(report: dict[str, Any]) -> None:
    """Human-readable terminal output."""
    print(f"\n{'='*66}")
    print(f"  交易系统状态仪表板  {report['timestamp']}")
    print(f"{'='*66}\n")

    # Services (multi-venue)
    services = report.get("services", {})
    print("🔧 服务状态:")
    for venue, svc in services.items():
        state = svc.get("ActiveState", "?")
        started = svc.get("ActiveEnterTimestamp", "?")[:25] if svc.get("ActiveEnterTimestamp") else "?"
        mem = _fmt_memory(svc.get("MemoryCurrent", "0"))
        icon = "✅" if state == "active" else "⚠️"
        print(f"   {icon} {venue:<9} {state.upper():<10} mem={mem:<8} since={started}")
    print()

    # Disk
    disk = report["disk"]
    if disk.get("status") != "unknown":
        icon = "✅" if disk["status"] == "ok" else ("⚠️" if disk["status"] == "warning" else "🚨")
        print(f"{icon} 磁盘: {disk.get('used_pct', '?')}% 使用 ({disk.get('free_gb', '?')}GB 空闲)")
    print()

    # IC Health
    ic = report["ic_health"]
    if "models" in ic:
        print(f"📊 模型 IC 健康 (评估: {ic.get('evaluated_at', '?')[:16]}):")
        for m in ic["models"]:
            status = m["status"]
            icon = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(status, "⚫")
            print(f"   {icon} {m['name']:<25} {status}")
    print()

    # Today's PnL
    pnl = report["today_pnl"]
    if pnl.get("n_trades", 0) > 0:
        total = pnl["total_pnl"]
        icon = "💰" if total >= 0 else "📉"
        print(f"{icon} 今日交易 (最近24h):")
        print(f"   交易数: {pnl['n_trades']} ({pnl['n_wins']}胜/{pnl['n_losses']}负)")
        print(f"   胜率:   {pnl['win_rate']}%")
        print(f"   总PnL:  ${total:+.2f}")
        print(f"   最佳:   ${pnl['best']:+.2f}   最差: ${pnl['worst']:+.2f}")
        for sym, stats in pnl.get("by_symbol", {}).items():
            print(f"   {sym}: {stats['n']} 笔, ${stats['pnl']:+.2f}")
    else:
        print(f"💤 今日无交易 (信号评估: {pnl.get('n_signals', 0)} 次)")
    print()

    # Positions (per venue)
    positions = report.get("positions", {})
    total_open = 0
    print("📦 当前持仓:")
    for venue, pos in positions.items():
        if "error" in pos:
            print(f"   ⚠️  {venue:<9} {pos['error']}")
            continue
        count = pos.get("count", 0)
        total_open += count
        eq = pos.get("equity_usdt")
        eq_str = f" (equity=${eq:.0f})" if eq is not None else ""
        if count == 0:
            print(f"   📭 {venue:<9} flat{eq_str}")
        else:
            print(f"   📬 {venue:<9} {count} open{eq_str}")
            for p in pos["positions"]:
                print(f"        {p['symbol']}: {p['side']} {p['qty']} @ ${p['entry']:.2f}")
    print()

    # Recent signals
    sigs = report.get("recent_signals", [])
    if sigs:
        print("📡 最近信号 (全 venue 合并, 最后10):")
        for s in sigs[-10:]:
            ts = datetime.fromtimestamp(s["ts"]).strftime("%H:%M")
            venue = s.get("venue", "?")[:3]
            rk = s.get("runner_key", "?")
            z = s.get("z_score", 0)
            sig = s.get("signal", 0)
            sig_str = "+1 LONG" if sig == 1 else ("-1 SHORT" if sig == -1 else "0 FLAT")
            print(f"   {ts} [{venue}] [{rk:<12}] z={z:+.2f} → {sig_str}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    report = build_dashboard()

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_dashboard(report)


if __name__ == "__main__":
    main()
