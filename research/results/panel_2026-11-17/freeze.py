"""Freeze the panel study's models for the forward looks (owner 2026-11-17, "执行").

The same settings as run.py, fit once on every row known before the forward window (2026-11-13; no setting is chosen
here), saved as JSON and LightGBM text files, with a spec whose SHA-256 the registered looks name
(research/protocols/2026-11-17-panel-forward.md).
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
from lab.um_lab import ms  # noqa: E402

FORWARD_START = ms("2026-11-13")
FROZEN = HERE / "frozen"
SPEC = HERE / "forward_spec.json"
MODELS = {"trend_14": ("binary", ("logistic", "lgbm")), "fwd_vol_7": ("regression", ("ridge", "lgbm")),
          "fwd_vol_14": ("regression", ("ridge", "lgbm")), "ret_14_scaled": ("regression", ("ridge", "lgbm"))}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if SPEC.exists():
        raise SystemExit(f"{SPEC} exists: a frozen spec is never rewritten")
    R.LIMIT = FORWARD_START                                          # every row known before the forward window
    times = np.arange(R.EVAL_START + R.DAY, FORWARD_START + 1, R.DAY, dtype=np.int64)
    panel, y, info = R.build_panel(times)
    panel["log_rv_30"] = np.log(panel["rv_30"])
    columns = list(R.ps.PANEL_FEATURES) + ["log_rv_30"]
    x = panel[columns].astype(np.float32)
    t = panel.index.to_numpy(dtype=np.int64)
    models = {}
    for name, (kind, names) in MODELS.items():
        target = y[name].to_numpy(float)
        for model in names:
            frozen = R.ps.fit_final(x, target, t, kind=kind, model=model)
            path = R.ps.save_frozen(frozen, FROZEN / f"{name}_{model}")
            files = [path] + ([path.with_name(path.name[:-len(".model.txt")] + ".meta.json")]
                              if path.name.endswith(".model.txt") else [])
            models[f"{name} {model}"] = {f.name: sha(f) for f in files}
            print(name, model, "rows", int(np.isfinite(target).sum()), flush=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    spec = {"study": "panel_2026-11-17:frozen", "frozen_at": "2026-11-17", "code_commit": commit,
            "snapshot": R.SNAP.name, "rows_known_before": "2026-11-13T00:00:00Z", "panel": info,
            "universe": f"top {R.TOP} crypto USDT perpetuals per day by the 30-day median quote volume; BTC and ETH"
                        " always; tradfi (runs/2026-11-17b/tradfi_contracts.json) and stablecoin contracts left out",
            "tradfi_sha256": sha(ROOT / "research/lab/runs/2026-11-17b/tradfi_contracts.json"),
            "features": columns, "targets": {k: v[0] for k, v in MODELS.items()}, "models": models,
            "protocol": "research/protocols/2026-11-17-panel-forward.md"}
    SPEC.write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
    print("spec sha256", sha(SPEC))


if __name__ == "__main__":
    main()
