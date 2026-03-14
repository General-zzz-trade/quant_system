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

    def reload_models(self, model_loader: Any = None) -> dict[str, str]:
        """Hot-reload models from model_dir. Called on SIGHUP.

        Two paths (mirrors LiveRunner._handle_model_reload):
        1. ModelRegistry-based: model_loader.reload_if_changed() → bridge.update_models()
        2. Direct file reload: scan model_dir/{sym}_gate_v2/*.pkl → load → push

        Returns {symbol: status} for each attempted reload.
        """
        # Path 1: ModelRegistry-based reload
        if model_loader is not None:
            try:
                new_models = model_loader.reload_if_changed()
                if new_models is not None:
                    self.inference_bridge.update_models(new_models)
                    return {s: "registry_reloaded" for s in self.symbols}
                return {s: "registry_noop" for s in self.symbols}
            except Exception as e:
                logger.exception("ModelRegistry reload failed")
                return {s: f"registry_error: {e}" for s in self.symbols}

        # Path 2: Direct file reload
        results: dict[str, str] = {}
        model_path = Path(self.model_dir)
        if not model_path.exists():
            logger.warning("Model dir %s does not exist", self.model_dir)
            return {s: "no_model_dir" for s in self.symbols}

        all_models = []
        for symbol in self.symbols:
            try:
                sym_dir = model_path / f"{symbol}_gate_v2"
                if not sym_dir.exists():
                    # Try broader match
                    candidates = list(model_path.glob(f"{symbol}*/lgbm*.pkl"))
                else:
                    candidates = list(sym_dir.glob("*.pkl"))

                if candidates:
                    try:
                        from alpha.models.lgbm_alpha import LGBMAlphaModel
                        for pkl in candidates:
                            m = LGBMAlphaModel(name=f"{symbol}_{pkl.stem}")
                            m.load(pkl)
                            all_models.append(m)
                        results[symbol] = f"reloaded:{len(candidates)}"
                    except ImportError:
                        # Fallback: single model reload via bridge
                        self.inference_bridge.reload_model(symbol, str(candidates[0]))
                        results[symbol] = "reloaded"
                else:
                    results[symbol] = "no_model_found"
            except Exception as e:
                results[symbol] = f"error: {e}"
                logger.error("Failed to reload model for %s: %s", symbol, e)

        # Push all loaded models to bridge
        if all_models and hasattr(self.inference_bridge, "update_models"):
            bridge = self.inference_bridge
            if isinstance(bridge, dict):
                for sym in self.symbols:
                    b = bridge.get(sym)
                    if b is not None:
                        sym_models = [m for m in all_models if sym in m.name]
                        if sym_models:
                            b.update_models(sym_models)
            else:
                bridge.update_models(all_models)
            logger.info("Direct model reload: %d model(s) from disk", len(all_models))

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
