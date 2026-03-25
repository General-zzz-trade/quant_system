#!/usr/bin/env python3
"""Pre-live trading checklist — run before switching to api.bybit.com.

Validates environment, models, security, safety constants, Rust build,
and exchange connectivity. Exit code 0 = all passed, 1 = failures present.
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

# ── 1. Environment ────────────────────────────────────────────────────
print("\n[Environment]")
results.append(check("BYBIT_API_KEY set", bool(os.environ.get("BYBIT_API_KEY"))))
results.append(check("BYBIT_API_SECRET set", bool(os.environ.get("BYBIT_API_SECRET"))))
base_url = os.environ.get("BYBIT_BASE_URL", "")
results.append(check("BYBIT_BASE_URL is live", base_url == "https://api.bybit.com", base_url or "(not set)"))

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
results.append(check("QUANT_MODEL_SIGN_KEY set", bool(sign_key),
                      "required for live — unsigned models rejected"))

# ── 4. Safety constants ──────────────────────────────────────────────
print("\n[Safety]")
try:
    from strategy.config import (
        LEVERAGE_LADDER,
        MAX_ORDER_NOTIONAL_PCT,
        SYMBOL_CONFIG,
        _IS_LIVE,
    )
    results.append(check(
        "MAX_ORDER_NOTIONAL_PCT <= 2.5",
        MAX_ORDER_NOTIONAL_PCT <= 2.5,
        f"value={MAX_ORDER_NOTIONAL_PCT}",
    ))
    is_live_env = base_url == "https://api.bybit.com"
    results.append(check(
        "_IS_LIVE matches BYBIT_BASE_URL",
        _IS_LIVE == is_live_env,
        f"_IS_LIVE={_IS_LIVE}, url_is_live={is_live_env}",
    ))
    lev = LEVERAGE_LADDER[0][1] if LEVERAGE_LADDER else 0
    results.append(check(
        "Leverage <= 3x for live" if _IS_LIVE else "Leverage (demo mode, info only)",
        lev <= 3.0 if _IS_LIVE else True,
        f"leverage={lev}x",
    ))
    active_symbols = [k for k in SYMBOL_CONFIG if "15m" not in k]
    results.append(check(
        "Active SYMBOL_CONFIG entries",
        len(active_symbols) >= 4,
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

# ── 6. Connectivity ──────────────────────────────────────────────────
print("\n[Connectivity]")
api_key = os.environ.get("BYBIT_API_KEY")
api_secret = os.environ.get("BYBIT_API_SECRET")
if api_key and api_secret:
    try:
        from execution.adapters.bybit.config import BybitConfig
        from execution.adapters.bybit.adapter import BybitAdapter

        cfg = BybitConfig(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url or "https://api-demo.bybit.com",
        )
        adapter = BybitAdapter(cfg)
        connected = adapter.connect()
        results.append(check("Bybit connect()", connected))
        if connected:
            snap = adapter.get_balances()
            usdt = snap.get("USDT")
            if usdt:
                results.append(check("USDT balance available", float(usdt.total) > 0,
                                     f"total={usdt.total}"))
            else:
                results.append(check("USDT balance available", False, "no USDT in snapshot"))
    except Exception as e:
        results.append(check("Bybit connection", False, str(e)))
else:
    results.append(check("Bybit connection", False, "API key/secret not set — skipped"))

# ── 7. Systemd services ─────────────────────────────────────────────
print("\n[Systemd]")
service_files = [
    "infra/systemd/bybit-alpha.service",
    "infra/systemd/health-watchdog.service",
    "infra/systemd/health-watchdog.timer",
    "infra/systemd/data-refresh.service",
    "infra/systemd/data-refresh.timer",
    "infra/systemd/daily-retrain.service",
    "infra/systemd/daily-retrain.timer",
]
for sf in service_files:
    results.append(check(f"{sf} exists", Path(sf).exists()))

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'=' * 55}")
passed = sum(results)
total = len(results)
failed = total - passed
print(f"Results: {passed}/{total} passed, {failed} failed")
if passed == total:
    print("ALL CHECKS PASSED — ready for live trading")
else:
    print("SOME CHECKS FAILED — review before going live")
sys.exit(0 if passed == total else 1)
