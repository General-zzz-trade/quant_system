"""Shim: delegates 4h training to train_4h_production.py.

Called by alpha.retrain.multi_tf.retrain_4h_symbols() with:
    train_symbol(symbol, interval="4h", horizons=[6, 12, 24])

The production script lives at /home/ubuntu/dev/scripts/training/train_4h_production.py
and is invoked via subprocess with --out-dir pointing to the expected model directory.
Returns True if the production script's config.json reports passed=True.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

_PRODUCTION_SCRIPT = "/home/ubuntu/dev/scripts/training/train_4h_production.py"
_MODEL_DIR_TEMPLATE = "models_v8/{symbol}_4h"


def train_symbol(
    symbol: str,
    interval: str = "4h",
    horizons: Optional[List[int]] = None,
) -> bool:
    """Train a 4h model for the given symbol.

    Args:
        symbol: e.g. "BTCUSDT"
        interval: ignored (always "4h")
        horizons: ignored (production script uses its own horizon logic)

    Returns:
        True if training passed all production checks, False otherwise.
    """
    out_dir = Path(_MODEL_DIR_TEMPLATE.format(symbol=symbol))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        _PRODUCTION_SCRIPT,
        "--symbol", symbol,
        "--out-dir", str(out_dir),
    ]

    logger.info("Running 4h production training: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd="/quant_system",
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
        )
    except subprocess.TimeoutExpired:
        logger.error("4h training timed out for %s", symbol)
        return False

    if result.returncode != 0:
        logger.error("4h training process failed for %s (rc=%d)\nstderr: %s",
                      symbol, result.returncode, result.stderr[-2000:] if result.stderr else "")
        return False

    # Log stdout (training progress)
    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logger.info("  [4h] %s", line)

    # Check if training passed production checks
    config_path = out_dir / "config.json"
    if not config_path.exists():
        logger.error("4h training did not produce config.json for %s", symbol)
        return False

    with open(config_path) as f:
        config = json.load(f)

    passed = config.get("passed", False)
    metrics = config.get("metrics", {})
    sharpe = metrics.get("sharpe", 0)
    ic = metrics.get("ic", 0)

    if passed:
        logger.info("4h %s PASSED: Sharpe=%.2f, IC=%.4f", symbol, sharpe, ic)
    else:
        logger.warning("4h %s FAILED checks: Sharpe=%.2f, IC=%.4f", symbol, sharpe, ic)

    return bool(passed)
