#!/usr/bin/env python3
"""Publish a frozen lab study spec for runner.binance_target (signed).

    QUANT_MODEL_SIGN_KEY=... python3 -m scripts.publish_strategy_spec \
        --study research/results/lab_2026-11-13 --out models_binance/strategy

The gate result inside final_spec.json is copied unchanged: a failed gate keeps
mainnet execution refused; paper and testnet runs are allowed.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from infra.model_signing import _get_key, sign_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models_binance/strategy"))
    args = parser.parse_args()
    spec = json.loads((args.study / "final_spec.json").read_text())
    if spec.get("kind") != "btc_eth_multitimeframe_target":
        raise SystemExit("not a lab target-strategy spec")
    if _get_key() is None:
        raise SystemExit("QUANT_MODEL_SIGN_KEY is required to publish runtime artifacts")
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "final_spec.json"
    shutil.copyfile(args.study / "final_spec.json", target)
    sign_file(target)
    print(f"published {target} (gate passed={spec.get('passed')})")


if __name__ == "__main__":
    main()
