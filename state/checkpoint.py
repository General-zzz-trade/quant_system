"""Checkpoint manager for AlphaRunner — save/restore engine + inference state.

Extracted from AlphaRunner to separate persistence concerns.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from _quant_hotpath import RustCheckpointStore  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_DIR = Path("data/runtime/checkpoints")


class CheckpointManager:
    """Manages checkpoint save/restore for individual runners."""

    def __init__(self, checkpoint_dir: Path = _DEFAULT_DIR):
        self._dir = checkpoint_dir
        # Rust-side in-memory checkpoint cache for fast state snapshots
        self._rust_store = RustCheckpointStore()

    def save(
        self,
        runner_key: str,
        engine_ckpt: Any,
        inference_ckpt: Any,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        """Save checkpoint to disk.

        Args:
            runner_key: Unique runner identifier (e.g. "BTCUSDT_4h").
            engine_ckpt: Engine checkpoint data (JSON string from Rust).
            inference_ckpt: Inference checkpoint data (dict or JSON string).
            extra: Additional state to persist (bars_processed, buffers, etc).
        """
        self._dir.mkdir(parents=True, exist_ok=True)

        if isinstance(inference_ckpt, dict):
            inference_ckpt = json.dumps(inference_ckpt)

        ckpt: Dict[str, Any] = {
            "engine": engine_ckpt,
            "inference": inference_ckpt,
        }
        if extra:
            ckpt.update(extra)

        path = self._dir / f"{runner_key}.json"
        # Atomic write: write to temp file then rename (prevents corruption on crash)
        tmp_path = path.with_suffix(".json.tmp")
        try:
            data = json.dumps(ckpt, default=str, allow_nan=False)
        except ValueError:
            # NaN/Inf in data — sanitize
            data = json.dumps(ckpt, default=str)
            data = re.sub(r'\bNaN\b', 'null', data)
            data = re.sub(r'\bInfinity\b', 'null', data)
            data = re.sub(r'\b-Infinity\b', 'null', data)
        tmp_path.write_text(data)
        tmp_path.replace(path)  # atomic on same filesystem

        logger.debug("%s checkpoint saved", runner_key)

    def restore(self, runner_key: str) -> Optional[Dict[str, Any]]:
        """Restore checkpoint from disk.

        Returns:
            Dict with checkpoint data, or None if no checkpoint exists.
        """
        path = self._dir / f"{runner_key}.json"
        if not path.exists():
            return None

        try:
            ckpt = json.loads(path.read_text())
            # Normalize inference data
            if isinstance(ckpt.get("inference"), str):
                ckpt["inference"] = json.loads(ckpt["inference"])

            # Validate: checkpoint with bars > 0 but empty closes is corrupted
            # Only check if "closes" key is present (some checkpoints don't have it)
            bars = ckpt.get("bars_processed", 0)
            if bars > 0 and "closes" in ckpt and len(ckpt["closes"]) == 0:
                logger.warning(
                    "%s checkpoint corrupted: bars=%d but closes empty — forcing full warmup",
                    runner_key, bars,
                )
                return None

            logger.info("%s checkpoint restored", runner_key)
            return ckpt
        except Exception as e:
            logger.warning("%s checkpoint restore failed: %s", runner_key, e)
            return None

    def exists(self, runner_key: str) -> bool:
        """Check if a checkpoint exists for the given runner."""
        return (self._dir / f"{runner_key}.json").exists()

    def delete(self, runner_key: str) -> bool:
        """Delete a checkpoint. Returns True if deleted."""
        path = self._dir / f"{runner_key}.json"
        if path.exists():
            path.unlink()
            return True
        return False
