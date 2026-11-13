#!/usr/bin/env python3
"""Publish a finished research study as signed runtime model artifacts.

    QUANT_MODEL_SIGN_KEY=... python3 -m scripts.publish_research_model \
        --study .cache/research/BTCUSDT --out models_binance

Copies model.json, protocol.json and gate.json into <out>/<SYMBOL>/ and writes
an HMAC signature next to each (runner.binance_main refuses unsigned files in
live mode). Refuses studies without a frozen final model. The gate result is
copied as-is: a failed gate keeps mainnet execution refused.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from infra.model_signing import _get_key, sign_file


def publish(study: Path, out: Path) -> Path:
    protocol = json.loads((study / "protocol.json").read_text())
    model = json.loads((study / "model.json").read_text())
    if protocol.get("final_model_sha256") != model.get("sha256"):
        raise SystemExit("study has no frozen final model matching its protocol")
    if not (study / "gate.json").exists():
        raise SystemExit("study has no gate.json")
    if _get_key() is None:
        raise SystemExit("QUANT_MODEL_SIGN_KEY is required to publish runtime artifacts")
    target = out / protocol["symbol"]
    target.mkdir(parents=True, exist_ok=True)
    for name in ("model.json", "protocol.json", "gate.json"):
        shutil.copyfile(study / name, target / name)
        sign_file(target / name)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models_binance"))
    args = parser.parse_args()
    target = publish(args.study, args.out)
    gate = json.loads((target / "gate.json").read_text())
    print(f"published {target} (gate passed={gate.get('passed')})")


if __name__ == "__main__":
    main()
