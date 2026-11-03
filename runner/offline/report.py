"""Portable experiment artifacts, accounting checks and reproducible checkpoints."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
from importlib.metadata import version
from decimal import Decimal as D
from pathlib import Path

from data.quality.causal_bars import HOUR, dataset_manifest, fingerprint


def load_funding(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            at, rate, mark = int(row["funding_time"]), D(row["rate"]), D(row["mark_price"])
            observed = int(row.get("observed_funding_time") or at)
            if not 0 <= observed - at <= 1000:
                raise ValueError("funding timestamp normalization exceeds the declared 1-second tolerance")
            if at in result or at % (8 * HOUR) or not rate.is_finite() or not mark.is_finite() or mark <= 0:
                raise ValueError("invalid/duplicate funding record; this runner requires UTC 8h settlements")
            result[at] = (rate, mark)
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def code_identity():
    root = Path(__file__).resolve().parents[2]
    paths = []
    for name in ("runner", "decision", "engine", "event", "features", "state", "execution", "shared", "risk",
                 "data/quality",
                 "strategy", "alpha", "portfolio", "regime", "core", "monitoring"):
        paths.extend(p for p in (root / name).rglob("*") if p.suffix in {".py", ".rs"} and p.is_file())
    values = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}
    import _quant_hotpath
    native = Path(_quant_hotpath.__file__)
    if native.is_file():
        values["native_extension"] = hashlib.sha256(native.read_bytes()).hexdigest()
    # Wheels expose a Python package which re-exports the compiled submodule.
    # Hash the actual shared library as well as __init__.py in that layout.
    for folder in getattr(_quant_hotpath, "__path__", ()):
        for compiled in sorted(Path(folder).glob("*")):
            if compiled.suffix in {".so", ".pyd"}:
                values[f"native_extension/{compiled.name}"] = hashlib.sha256(compiled.read_bytes()).hexdigest()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    return {"source_sha256": fingerprint(values), "git_revision": revision, "python": platform.python_version(),
            "numpy": version("numpy"), "pandas": version("pandas"),
            "platform": platform.system(), "machine": platform.machine()}


def input_identity(bars, bundle, funding, config, start_ms, mode):
    return fingerprint({"dataset": dataset_manifest(bars), "model": bundle, "config": config.record(),
                        "funding": [[at, str(rate), str(mark)] for at, (rate, mark) in sorted(funding.items())],
                        "start_ms": start_ms, "mode": mode, "code": code_identity()})


def metrics(result):
    initial = D(result["config"]["starting_balance"])
    peak = initial
    max_dd = D(0)
    exposures = []
    daily = {}
    for row in result["equity"]:
        equity = D(row["equity"])
        reconciled = initial + D(row["realized_pnl"]) - D(row["fees"]) + D(row["funding"]) + D(row["unrealized_pnl"])
        if abs(equity - reconciled) > D("1e-18"):
            raise ValueError("equity reconciliation failed")
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak)
        exposures.append(float(abs(D(row["qty"])) * D(row["mark"]) / equity) if equity > 0 else 0)
        # A bar ending at 00:00 closes the PREVIOUS UTC day.
        daily[(row["at"] - 1) // (24 * HOUR)] = float(equity)
    values = list(daily.values())
    returns = [values[i] / values[i - 1] - 1 for i in range(1, len(values)) if values[i - 1] > 0]
    sharpe = None
    if len(returns) >= 30:
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        if variance > 0:
            sharpe = mean / math.sqrt(variance) * math.sqrt(365)
    final = result["final"]
    return {"net_return_pct": float((D(final["equity"]) / initial - 1) * 100),
            "max_drawdown_pct": float(max_dd * 100), "daily_sharpe": sharpe,
            "sharpe_observations": len(returns), "fills": len(result["fills"]),
            "orders": len(result["orders"]), "fees_usdt": final["fees"], "funding_usdt": final["funding"],
            "final_equity_usdt": final["equity"], "open_qty": final["qty"],
            "average_gross_exposure": sum(exposures) / len(exposures) if exposures else 0,
            "accounting_reconciled": True, "valuation": "marked equity; open positions not forcibly closed"}


def write_result(out, result):
    out = Path(out)
    write_json(out / "replay.json", result)
    write_json(out / "metrics.json", metrics(result))
    for key in ("orders", "fills", "equity", "funding", "signals", "risk_events"):
        rows = result[key]
        if not rows:
            continue
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / f"{key}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def checkpoint_replay(bars, bundle, funding, *, start_ms, config, checkpoint, stop_after=None, resume=False):
    from .replay import run_replay
    identity = input_identity(bars, bundle, funding, config, start_ms, "strategy")
    if resume:
        saved = json.loads(Path(checkpoint).read_text())
        if saved["input_sha256"] != identity:
            raise ValueError("checkpoint input/code/config/model/funding mismatch")
        prefix = run_replay(bars, bundle, funding, start_ms=start_ms, config=config, stop_after=saved["processed_bars"])
        if prefix["result_sha256"] != saved["result_sha256"]:
            raise ValueError("checkpoint replay differs from committed prefix")
    result = run_replay(bars, bundle, funding, start_ms=start_ms, config=config, stop_after=stop_after)
    write_json(checkpoint, {"schema": 1, "input_sha256": identity, "processed_bars": result["processed_bars"],
                            "result_sha256": result["result_sha256"],
                            "recovery": "deterministic reconstruction from immutable input; no live orders"})
    return result
