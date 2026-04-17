#!/usr/bin/env python3
"""Pre-live trading checklist — run before service startup.

Validates environment, models, security, safety constants, Rust build,
and exchange connectivity. Exit code 0 = all passed, 1 = failures present.

Supports: Bybit, OKX (auto-detected from environment).
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_PROJECT_ROOT)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


results: list[bool] = []

# ── Detect venue ─────────────────────────────────────────────────────
venue = os.environ.get("VENUE", "").lower()
has_okx = bool(os.environ.get("OKX_API_KEY"))
has_bybit = bool(os.environ.get("BYBIT_API_KEY"))
if not venue:
    venue = "okx" if has_okx else "bybit"

# ── 1. Environment ────────────────────────────────────────────────────
print(f"\n[Environment — {venue.upper()}]")
if venue == "okx":
    results.append(check("OKX_API_KEY set", has_okx))
    results.append(check("OKX_API_SECRET set", bool(os.environ.get("OKX_API_SECRET"))))
    results.append(check("OKX_API_PASSPHRASE set", bool(os.environ.get("OKX_API_PASSPHRASE"))))
    base_url = os.environ.get("OKX_BASE_URL", "")
    is_live = "okx.com" in base_url and "demo" not in base_url
    results.append(check("OKX_BASE_URL", bool(base_url), base_url or "(not set)"))
else:
    results.append(check("BYBIT_API_KEY set", has_bybit))
    results.append(check("BYBIT_API_SECRET set", bool(os.environ.get("BYBIT_API_SECRET"))))
    base_url = os.environ.get("BYBIT_BASE_URL", "")
    is_live = base_url == "https://api.bybit.com"
    results.append(check("BYBIT_BASE_URL", bool(base_url), base_url or "(not set)"))

# ── 2. Models ─────────────────────────────────────────────────────────
print("\n[Models]")
REQUIRED_MODELS = ["BTCUSDT_gate_v2", "ETHUSDT_gate_v2", "BTCUSDT_4h", "ETHUSDT_4h"]
for model in REQUIRED_MODELS:
    config_path = Path(f"models_v8/{model}/config.json")
    results.append(check(f"{model} config.json exists", config_path.exists()))

# Try loading each model via production loader
try:
    from alpha.model_loader_prod import load_model
    for model in REQUIRED_MODELS:
        model_dir = Path(f"models_v8/{model}")
        try:
            loaded = load_model(model_dir)
            n_horizons = len(loaded.get("horizon_models", []))
            results.append(check(f"{model} loadable", True, f"{n_horizons} horizon(s)"))
        except Exception as e:
            results.append(check(f"{model} loadable", False, str(e)))
except ImportError as e:
    results.append(check("alpha.model_loader_prod importable", False, str(e)))

# ── 3. Security ───────────────────────────────────────────────────────
print("\n[Security]")
sign_key = os.environ.get("QUANT_MODEL_SIGN_KEY")
allow_unsigned = os.environ.get("QUANT_ALLOW_UNSIGNED_MODELS", "").lower() in ("1", "true")
sign_ok = bool(sign_key) or not is_live or allow_unsigned
results.append(check("QUANT_MODEL_SIGN_KEY set", sign_ok,
                      "bypassed (QUANT_ALLOW_UNSIGNED_MODELS)" if allow_unsigned and not sign_key
                      else ("required for live" if is_live else "optional for demo")))

# ── 4. Safety constants ──────────────────────────────────────────────
print("\n[Safety]")
try:
    from strategy.config import (
        LEVERAGE_LADDER,
        MAX_ORDER_NOTIONAL_PCT,
        SYMBOL_CONFIG,
    )
    results.append(check(
        "MAX_ORDER_NOTIONAL_PCT <= 2.5",
        MAX_ORDER_NOTIONAL_PCT <= 2.5,
        f"value={MAX_ORDER_NOTIONAL_PCT}",
    ))
    lev = LEVERAGE_LADDER[0][1] if LEVERAGE_LADDER else 0
    results.append(check(
        f"Leverage = {lev}x",
        lev <= 10.0,
        f"{'live' if is_live else 'demo'} mode",
    ))
    active_symbols = [k for k in SYMBOL_CONFIG if "15m" not in k]
    results.append(check(
        "Active SYMBOL_CONFIG entries",
        len(active_symbols) >= 2,
        f"{', '.join(active_symbols)}",
    ))
except ImportError as e:
    results.append(check("strategy.config importable", False, str(e)))

# ── 5. Rust build ────────────────────────────────────────────────────
print("\n[Rust]")
try:
    import _quant_hotpath
    n_exports = len(dir(_quant_hotpath))
    results.append(check("_quant_hotpath importable", True, f"{n_exports} exports"))
    results.append(check("_quant_hotpath >= 200 exports", n_exports >= 200, f"got {n_exports}"))
except ImportError as e:
    results.append(check("_quant_hotpath importable", False, str(e)))

# ── 6. Connectivity (with timeout) ───────────────────────────────────
print("\n[Connectivity]")
import signal

def _timeout_handler(signum, frame):
    raise TimeoutError("connectivity check timed out")

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(10)  # 10s max for connectivity check

try:
    if venue == "okx" and has_okx:
        from execution.adapters.okx.adapter import OkxAdapter
        from execution.adapters.okx.config import OkxConfig
        cfg = OkxConfig(
            api_key=os.environ["OKX_API_KEY"],
            api_secret=os.environ["OKX_API_SECRET"],
            passphrase=os.environ["OKX_API_PASSPHRASE"],
            base_url=os.environ.get("OKX_BASE_URL", "https://www.okx.com"),
        )
        adapter = OkxAdapter(cfg)
        adapter.connect()
        positions = adapter.get_positions()
        results.append(check("OKX connectivity", True, f"{len(positions)} positions"))
    elif venue == "bybit" and has_bybit:
        from execution.adapters.bybit.config import BybitConfig
        from execution.adapters.bybit.adapter import BybitAdapter
        cfg = BybitConfig(
            api_key=os.environ["BYBIT_API_KEY"],
            api_secret=os.environ["BYBIT_API_SECRET"],
            base_url=base_url or "https://api-demo.bybit.com",
        )
        adapter = BybitAdapter(cfg)
        connected = adapter.connect()
        results.append(check("Bybit connectivity", connected))
    else:
        results.append(check("Exchange connectivity", False, "no API keys set — skipped"))
except TimeoutError:
    results.append(check("Exchange connectivity", False, "timed out (10s)"))
except Exception as e:
    results.append(check("Exchange connectivity", False, str(e)[:100]))
finally:
    signal.alarm(0)

# ── 7. Systemd services ─────────────────────────────────────────────
print("\n[Systemd]")
service_files = [
    f"infra/systemd/{'okx' if venue == 'okx' else 'bybit'}-alpha.service",
    "infra/systemd/health-watchdog.service",
    "infra/systemd/health-watchdog.timer",
    "infra/systemd/data-refresh.service",
    "infra/systemd/daily-retrain.service",
]
for sf in service_files:
    results.append(check(f"{sf} exists", Path(sf).exists()))

# ── 8. Z-score checkpoints ──────────────────────────────────────────
print("\n[Z-score]")
zscore_dir = Path("data/runtime/zscore_checkpoints")
if zscore_dir.exists():
    checkpoints = list(zscore_dir.glob("*.json"))
    results.append(check("Z-score checkpoints exist", len(checkpoints) > 0,
                         f"{len(checkpoints)} files"))
else:
    results.append(check("Z-score checkpoints dir", False, "missing"))

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
passed = sum(results)
total = len(results)
failed = total - passed
print(f"Results: {passed}/{total} passed, {failed} failed")
if passed == total:
    print("ALL CHECKS PASSED — ready for trading")
else:
    print("SOME CHECKS FAILED — review before going live")
sys.exit(0 if passed == total else 1)
