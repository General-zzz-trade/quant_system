"""TradingEngine — features + inference + model hot-reload.

Wraps FeatureComputeHook and RustInferenceBridge behind a clean interface.
Owns SIGHUP model reload logic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TradingEngine:
    """Features → ML prediction → trading signal + model hot-reload."""

    def __init__(
        self,
        feature_hook: Any,
        inference_bridge: Any,
        symbols: list[str],
        model_dir: str = "models_v8",
    ) -> None:
        self.feature_hook = feature_hook
        self.inference_bridge = inference_bridge
        self.symbols = symbols
        self.model_dir = model_dir

    def on_bar(self, symbol: str, bar: dict) -> float | None:
        """Process one bar through feature computation + ML inference.

        Returns prediction score (nonzero) or None (no signal / insufficient features).
        """
        features = self.feature_hook.on_bar(symbol, bar)
        if features is None:
            return None
        prediction = self.inference_bridge.predict(symbol, features)
        if prediction == 0.0:
            return None
        return prediction

    def reload_models(self) -> dict[str, str]:
        """Hot-reload models from model_dir. Called on SIGHUP.

        Scans model_dir for per-symbol model directories, loads .pkl files,
        and updates the inference bridge.

        Returns {symbol: status} for each attempted reload.
        """
        results: dict[str, str] = {}
        model_path = Path(self.model_dir)
        if not model_path.exists():
            logger.warning("Model dir %s does not exist", self.model_dir)
            return {s: "no_model_dir" for s in self.symbols}

        for symbol in self.symbols:
            try:
                # Look for {symbol}_gate_v2/ directory pattern
                candidates = list(model_path.glob(f"{symbol}_gate_v2/lgbm_v8.pkl"))
                if not candidates:
                    candidates = list(model_path.glob(f"{symbol}*/lgbm*.pkl"))
                if candidates:
                    pkl_path = candidates[0]
                    self.inference_bridge.reload_model(symbol, str(pkl_path))
                    results[symbol] = "reloaded"
                    logger.info("Reloaded model for %s from %s", symbol, pkl_path)
                else:
                    results[symbol] = "no_model_found"
            except Exception as e:
                results[symbol] = f"error: {e}"
                logger.error("Failed to reload model for %s: %s", symbol, e)

        return results

    def checkpoint(self) -> dict:
        """Serialize feature_hook + inference_bridge state for crash recovery."""
        state: dict[str, Any] = {}
        if hasattr(self.feature_hook, "checkpoint"):
            state["feature_hook"] = self.feature_hook.checkpoint()
        if hasattr(self.inference_bridge, "get_params"):
            state["inference"] = self.inference_bridge.get_params()
        return state

    def restore(self, state: dict) -> None:
        """Restore from checkpoint."""
        if "feature_hook" in state and hasattr(self.feature_hook, "restore_checkpoint"):
            self.feature_hook.restore_checkpoint(state["feature_hook"])
        if "inference" in state and hasattr(self.inference_bridge, "update_params"):
            self.inference_bridge.update_params(state["inference"])
