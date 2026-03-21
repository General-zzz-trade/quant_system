#!/usr/bin/env python3
"""One-command server setup — deploy quant system to a fresh Ubuntu server.

Handles:
  1. System dependencies (Python 3.12, build tools)
  2. Pip packages (from frozen requirements)
  3. Rust crate build
  4. Systemd services + timers
  5. Crontab setup
  6. Data directory structure
  7. Environment validation

Usage:
    # On a fresh server:
    git clone git@github.com:General-zzz-trade/quant_system.git /quant_system
    cd /quant_system
    python3 scripts/ops/server_setup.py --env-file .env

    # Verify:
    python3 scripts/ops/server_setup.py --check
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, check=check, cwd=str(PROJECT_ROOT))


def setup_directories() -> None:
    """Create required data directories."""
    print("\n[1/7] Creating directories...")
    dirs = [
        "data/runtime/checkpoints",
        "data/runtime/kills",
        "data/options",
        "data/onchain",
        "data/polymarket",
        "data_files",
        "logs",
        "reports/weekly",
        ".cache/features",
        ".cache/hpo",
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"    ✓ {d}")


def setup_pip() -> None:
    """Install Python packages."""
    print("\n[2/7] Installing pip packages...")
    req_file = PROJECT_ROOT / "requirements.lock.txt"
    if req_file.exists():
        run(f"{sys.executable} -m pip install -r {req_file} --break-system-packages -q")
    else:
        run(f"{sys.executable} -m pip install -e '.[live,data,ml,monitoring,dev,test]' --break-system-packages -q")
    print("    ✓ Packages installed")


def setup_rust() -> None:
    """Build Rust crate."""
    print("\n[3/7] Building Rust crate...")
    run("make rust")
    # Copy .so to project root
    run(
        "cp $(python3 -c \"import _quant_hotpath, os;"
        " print(os.path.dirname(_quant_hotpath.__file__))\")"
        "/*.so _quant_hotpath/ 2>/dev/null || true"
    )
    run("python3 -c \"import _quant_hotpath; print(len(dir(_quant_hotpath)),'exports')\"")



def setup_systemd() -> None:
    """Install systemd services and timers."""
    print("\n[4/7] Installing systemd services...")
    services = list((PROJECT_ROOT / "infra/systemd").glob("*.service"))
    timers = list((PROJECT_ROOT / "infra/systemd").glob("*.timer"))

    for f in services + timers:
        dst = Path(f"/etc/systemd/system/{f.name}")
        shutil.copy2(str(f), str(dst))
        print(f"    ✓ {f.name}")

    run("sudo systemctl daemon-reload")

    # Enable timers
    for t in timers:
        run(f"sudo systemctl enable {t.name}", check=False)
        run(f"sudo systemctl start {t.name}", check=False)
    print("    ✓ Timers enabled")


def setup_crontab() -> None:
    """Install crontab entries."""
    print("\n[5/7] Setting up crontab...")
    P = "/usr/bin/python3"
    D = "/quant_system"
    L = f"{D}/logs"
    jobs = [
        f"0 * * * * cd {D} && {P} -m scripts.ops.demo_tracker >> {L}/demo_tracker.log 2>&1",
        f"0 */6 * * * cd {D} && {P} -m scripts.data.download_oi_data --symbols ETHUSDT BTCUSDT --days 30 >> {L}/oi_download.log 2>&1",  # noqa: E501
        f"15 */6 * * * cd {D} && {P} -m scripts.data.download_liquidations --symbol BTCUSDT >> {L}/liq_proxy.log 2>&1",  # noqa: E501
        f"0 4 * * * cd {D} && {P} -m scripts.ops.daily_reconciliation --log-file logs/bybit_alpha.log --days 1 >> {L}/reconciliation.log 2>&1",  # noqa: E501
        f"30 * * * * cd {D} && {P} -m scripts.data.download_deribit_options --currency BTC --once >> {L}/options_collector.log 2>&1",  # noqa: E501
        f"0 1 * * 0 cd {D} && {P} -m scripts.ops.auto_bug_scan --severity warning >> {L}/bug_scan.log 2>&1",  # noqa: E501
        f"0 3 * * 0 cd {D} && {P} -m scripts.ops.weekly_report >> {L}/weekly_report.log 2>&1",
        f"0 5 * * * bash {D}/scripts/ops/backup.sh >> {L}/backup.log 2>&1",
    ]
    cron_text = "# === Quant System ===\n" + "\n".join(jobs) + "\n"
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cron", delete=False) as f:
        f.write(cron_text)
        tmp = f.name
    run(f"crontab {tmp}")
    Path(tmp).unlink()
    print(f"    ✓ {len(jobs)} cron jobs installed")


def setup_env(env_file: str | None) -> None:
    """Validate or create .env file."""
    print("\n[6/7] Environment setup...")
    env_path = PROJECT_ROOT / ".env"
    if env_file and Path(env_file).exists():
        shutil.copy2(env_file, str(env_path))
        print(f"    ✓ Copied {env_file} → .env")
    elif env_path.exists():
        print("    ✓ .env exists")
    else:
        example = PROJECT_ROOT / ".env.example"
        if example.exists():
            shutil.copy2(str(example), str(env_path))
            print("    ⚠ Created .env from .env.example — edit with real keys!")
        else:
            print("    ✗ No .env or .env.example found!")


def freeze_requirements() -> None:
    """Save current pip freeze for reproducibility."""
    print("\n[7/7] Freezing requirements...")
    lock = PROJECT_ROOT / "requirements.lock.txt"
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True, text=True,
    )
    lock.write_text(result.stdout)
    n = len(result.stdout.strip().splitlines())
    print(f"    ✓ {n} packages → requirements.lock.txt")


def check_deployment() -> None:
    """Verify deployment health."""
    print("\n=== Deployment Health Check ===")
    checks = [
        ("Python 3.12", "python3 --version"),
        ("Rust crate", 'python3 -c "import _quant_hotpath; print(len(dir(_quant_hotpath)),\'exports\')"'),
        (".env exists", "test -f .env && echo OK || echo MISSING"),
        ("Models exist", "ls models_v8/BTCUSDT_gate_v2/config.json && echo OK"),
        ("Checkpoints", "ls data/runtime/checkpoints/*.json 2>/dev/null && echo OK || echo 'No checkpoints yet'"),
        ("bybit-alpha", "systemctl is-active bybit-alpha.service 2>/dev/null || echo inactive"),
        ("health-watchdog", "systemctl is-active health-watchdog.timer 2>/dev/null || echo inactive"),
        ("auto-retrain", "systemctl is-active auto-retrain.timer 2>/dev/null || echo inactive"),
        ("data-refresh", "systemctl is-active data-refresh.timer 2>/dev/null || echo inactive"),
        ("Cron jobs", "crontab -l 2>/dev/null | grep -c quant_system"),
    ]
    for name, cmd in checks:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
        output = result.stdout.strip().split("\n")[-1] if result.stdout.strip() else "FAIL"
        status = "✓" if "OK" in output or "active" in output or "3.12" in output else "⚠"
        print(f"  {status} {name}: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Server setup for quant system")
    parser.add_argument("--check", action="store_true", help="Check deployment health only")
    parser.add_argument("--env-file", default=None, help="Path to .env file to copy")
    parser.add_argument("--skip-rust", action="store_true", help="Skip Rust build")
    parser.add_argument("--skip-systemd", action="store_true", help="Skip systemd setup")
    args = parser.parse_args()

    if args.check:
        check_deployment()
        return

    print("=" * 60)
    print("  Quant System Server Setup")
    print("=" * 60)

    setup_directories()
    setup_pip()
    if not args.skip_rust:
        setup_rust()
    if not args.skip_systemd:
        setup_systemd()
    setup_crontab()
    setup_env(args.env_file)
    freeze_requirements()

    print("\n" + "=" * 60)
    print("  Setup complete! Next steps:")
    print("  1. Edit .env with your API keys")
    print("  2. Download data: python3 -m scripts.data_refresh")
    print("  3. Start trading: sudo systemctl start bybit-alpha.service")
    print("  4. Verify: python3 scripts/ops/server_setup.py --check")
    print("=" * 60)


if __name__ == "__main__":
    main()
