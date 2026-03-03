# alpha/model_loader.py
"""ProductionModelLoader — loads production models from ModelRegistry + ArtifactStore."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class ProductionModelLoader:
    """Load production models from ModelRegistry + ArtifactStore.

    Usage:
        loader = ProductionModelLoader(registry, artifact_store)
        models = loader.load_production_models(["alpha_btc", "alpha_eth"])
    """

    def __init__(self, registry: Any, artifact_store: Any) -> None:
        self._registry = registry
        self._store = artifact_store
        self._loaded_ids: Dict[str, str] = {}  # name -> model_id

    def load_production_models(self, model_names: Sequence[str]) -> List[Any]:
        """Load production models by name. Returns list of AlphaModel instances."""
        models = []
        for name in model_names:
            model = self._load_one(name)
            if model is not None:
                models.append(model)
        return models

    def reload_if_changed(self, model_names: Sequence[str]) -> Optional[List[Any]]:
        """Check if production model_ids changed; reload if so. Returns None if unchanged."""
        changed = False
        for name in model_names:
            version = self._registry.get_production(name)
            if version is None:
                continue
            prev_id = self._loaded_ids.get(name)
            if prev_id != version.model_id:
                changed = True
                break

        if not changed:
            return None

        logger.info("Production model change detected, reloading")
        return self.load_production_models(model_names)

    def _load_one(self, name: str) -> Optional[Any]:
        version = self._registry.get_production(name)
        if version is None:
            logger.warning("No production model for '%s'", name)
            return None

        weights = self._store.load(version.model_id, "weights")
        if weights is None:
            logger.error("No weights artifact for model %s (id=%s)", name, version.model_id)
            return None

        model_type = self._detect_type(version)
        model = self._instantiate(model_type, name, version, weights)
        if model is not None:
            self._loaded_ids[name] = version.model_id
            logger.info(
                "Loaded production model %s v%d (id=%s, type=%s)",
                name, version.version, version.model_id, model_type,
            )
        return model

    def _detect_type(self, version: Any) -> str:
        tags = set(getattr(version, "tags", ()))
        if "lgbm" in tags:
            return "lgbm"
        if "xgb" in tags:
            return "xgb"
        name_lower = version.name.lower()
        if "lgbm" in name_lower:
            return "lgbm"
        if "xgb" in name_lower:
            return "xgb"
        return "lgbm"  # default

    def _instantiate(self, model_type: str, name: str, version: Any, weights: bytes) -> Optional[Any]:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(weights)
            tmp_path = Path(f.name)

        try:
            if model_type == "lgbm":
                from alpha.models.lgbm_alpha import LGBMAlphaModel
                model = LGBMAlphaModel(name=name)
                model.load(tmp_path)
            elif model_type == "xgb":
                from alpha.models.xgb_alpha import XGBAlphaModel
                model = XGBAlphaModel(name=name)
                model.load(tmp_path)
            else:
                logger.error("Unknown model type: %s", model_type)
                return None
            return model
        except Exception:
            logger.exception("Failed to load model %s (type=%s)", name, model_type)
            return None
        finally:
            tmp_path.unlink(missing_ok=True)
