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

    def inspect_production_model(self, name: str) -> Dict[str, Any]:
        version = self._registry.get_production(name)
        if version is None:
            return {"name": name, "available": False, "reason": "no_production_model"}

        has_weights = self._store.load(version.model_id, "weights") is not None
        has_signature = self._store.load(version.model_id, "weights.sig") is not None
        loaded_model_id = self._loaded_ids.get(name)
        return {
            "name": name,
            "available": True,
            "model_id": version.model_id,
            "version": version.version,
            "tags": tuple(getattr(version, "tags", ()) or ()),
            "has_weights": has_weights,
            "has_signature": has_signature,
            "loaded_model_id": loaded_model_id,
            "autoload_pending": loaded_model_id not in (None, version.model_id),
        }

    def inspect_production_models(self, model_names: Sequence[str]) -> List[Dict[str, Any]]:
        return [self.inspect_production_model(name) for name in model_names]

    def _load_one(self, name: str) -> Optional[Any]:
        version = self._registry.get_production(name)
        if version is None:
            logger.warning("No production model for '%s'", name)
            return None

        weights = self._store.load(version.model_id, "weights")
        if weights is None:
            logger.error("No weights artifact for model %s (id=%s)", name, version.model_id)
            return None

        # Verify artifact integrity (SHA-256 digest) if digest is available
        if hasattr(self._store, 'verify_digest'):
            integrity = self._store.verify_digest(version.model_id)
            if integrity is False:
                logger.error(
                    "Artifact integrity check FAILED for %s (id=%s) — digest mismatch",
                    name, version.model_id,
                )
                return None
            elif integrity is None:
                logger.debug("No digest stored for %s — skipping integrity check", name)

        weights_sig = self._store.load(version.model_id, "weights.sig")

        model_type = self._detect_type(version)
        model = self._instantiate(model_type, name, version, weights, weights_sig)
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

    def _instantiate(
        self,
        model_type: str,
        name: str,
        version: Any,
        weights: bytes,
        weights_sig: bytes | None,
    ) -> Optional[Any]:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir = Path(tmp_dir_obj.name)
        tmp_path = tmp_dir / "weights.pkl"
        tmp_path.write_bytes(weights)
        if weights_sig is not None:
            (tmp_path.with_suffix(tmp_path.suffix + ".sig")).write_bytes(weights_sig)
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
            if not self._features_match(version, model):
                logger.error(
                    "Feature schema mismatch for %s (id=%s)",
                    name, version.model_id,
                )
                return None
            # Validate features against the centralized production catalog
            try:
                from features.feature_catalog import validate_model_features
                feat_names = getattr(model, "feature_names", None)
                if feat_names:
                    catalog_warnings = validate_model_features(feat_names, model_name=name)
                    for w in catalog_warnings:
                        logger.warning(w)
            except Exception:
                pass  # catalog validation is advisory — never block
            return model
        except Exception:
            logger.exception("Failed to load model %s (type=%s)", name, model_type)
            return None
        finally:
            tmp_dir_obj.cleanup()

    @staticmethod
    def _features_match(version: Any, model: Any) -> bool:
        expected = tuple(getattr(version, "features", ()) or ())
        actual = tuple(getattr(model, "feature_names", ()) or ())
        if not expected and not actual:
            return True  # Both empty — legacy model, allow
        if not expected or not actual:
            logger.warning(
                "Feature schema incomplete: expected=%d actual=%d — allowing load but model may misbehave",
                len(expected), len(actual),
            )
            return True  # One side empty — warn but allow (backward compat)
        return expected == actual
