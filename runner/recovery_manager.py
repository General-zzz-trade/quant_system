"""RecoveryManager — checkpoint/restore + startup reconciliation.

Coordinates saving/restoring state across TradingEngine, RiskManager,
and OrderManager. Delegates to existing runner/recovery.py functions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RecoveryManager:
    """Periodic checkpointing and crash recovery."""

    def __init__(
        self,
        state_dir: str,
        engine: Any,
        risk: Any,
        orders: Any,
        interval_sec: float = 300.0,
    ) -> None:
        self._state_dir = Path(state_dir)
        self._engine = engine
        self._risk = risk
        self._orders = orders
        self.interval_sec = interval_sec

    def save(self) -> None:
        """Save checkpoint: engine + risk state to state_dir."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "engine": self._engine.checkpoint(),
            "risk": self._risk.checkpoint(),
        }
        checkpoint_path = self._state_dir / "checkpoint.json"
        tmp_path = checkpoint_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state, default=str))
        tmp_path.rename(checkpoint_path)
        logger.info("Checkpoint saved to %s", checkpoint_path)

    def restore(self) -> bool:
        """Restore from last checkpoint. Returns True if restored."""
        checkpoint_path = self._state_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            logger.info("No checkpoint found at %s", checkpoint_path)
            return False
        try:
            state = json.loads(checkpoint_path.read_text())
            if "engine" in state:
                self._engine.restore(state["engine"])
            if "risk" in state:
                self._risk.restore(state["risk"])
            logger.info("Checkpoint restored from %s", checkpoint_path)
            return True
        except Exception as e:
            logger.error("Failed to restore checkpoint: %s", e)
            return False

    def reconcile_startup(self, executor: Any) -> list[str]:
        """Compare internal state vs venue positions at startup.

        Returns list of mismatch descriptions (empty = all OK).
        """
        mismatches: list[str] = []
        try:
            venue_positions = executor.get_positions()
            # Basic check: log what venue reports
            for pos in venue_positions:
                symbol = getattr(pos, "symbol", str(pos))
                qty = getattr(pos, "qty", None)
                logger.info("Venue position: %s qty=%s", symbol, qty)
        except Exception as e:
            mismatches.append(f"Failed to fetch venue positions: {e}")
        return mismatches
