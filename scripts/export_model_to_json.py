#!/usr/bin/env python3
"""Export LightGBM/XGBoost models from .pkl to .json for Rust native inference.

Usage:
    python3 scripts/export_model_to_json.py models_v8/BTCUSDT_gate_v2/lgbm_v8.pkl
    python3 scripts/export_model_to_json.py models_v8/BTCUSDT_gate_v2/xgb_v8.pkl
    python3 scripts/export_model_to_json.py models_v8/  # Export all models in directory

Output: .json file alongside the .pkl file.

NOTE: Uses pickle to load existing signed model artifacts. All .pkl files are
HMAC-signed and verified in production via infra/model_signing.py.
"""
from __future__ import annotations

import json
import pickle  # noqa: S403 — signed model artifacts, verified by HMAC
import sys
from pathlib import Path


def export_lgbm(pkl_path: Path) -> Path:
    import tempfile

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)  # noqa: S301

    model = data["model"]
    features = list(data["features"])
    is_classifier = data.get("is_classifier", False)

    if hasattr(model, "booster_"):
        booster = model.booster_
    else:
        booster = model

    # Use text model format for exact NaN handling via decision_type bitmask.
    # dump_model()'s default_left doesn't capture missing_type=None behavior
    # (LightGBM replaces NaN with 0.0 when feature had no NaN in training).
    tmp = tempfile.mktemp(suffix=".txt")
    booster.save_model(tmp)
    with open(tmp) as f:
        text_lines = f.readlines()
    Path(tmp).unlink()

    num_trees = booster.num_trees()
    trees = []
    for tree_idx in range(num_trees):
        tree_data = _parse_lgbm_text_tree(text_lines, tree_idx)
        if tree_data is None:
            continue

        split_feature = list(map(int, tree_data["split_feature"].split()))
        threshold = [
            1e308 if v == "inf" else -1e308 if v == "-inf" else float(v)
            for v in tree_data["threshold"].split()
        ]
        decision_type = list(map(int, tree_data["decision_type"].split()))
        left_child = list(map(int, tree_data["left_child"].split()))
        right_child = list(map(int, tree_data["right_child"].split()))
        leaf_value = list(map(float, tree_data["leaf_value"].split()))

        n_internal = len(split_feature)
        nodes = []
        for i in range(n_internal):
            dt = decision_type[i]
            # decision_type bitmask: bit 1 (2) = default_left, bit 3 (8) = NaN in training
            has_nan_training = bool(dt & 8)
            default_left = bool(dt & 2)

            lc = left_child[i]
            rc = right_child[i]
            # Negative child = leaf: -(leaf_idx + 1)
            left_ref = lc if lc >= 0 else {"leaf_idx": -(lc + 1)}
            right_ref = rc if rc >= 0 else {"leaf_idx": -(rc + 1)}

            nodes.append({
                "type": "split",
                "feature": split_feature[i],
                "threshold": threshold[i],
                "default_left": default_left,
                "nan_as_zero": not has_nan_training,
                "left": left_ref,
                "right": right_ref,
            })
        for val in leaf_value:
            nodes.append({"type": "leaf", "value": val})

        # Remap child references: internal nodes [0..n_internal), leaves [n_internal..)
        for node in nodes:
            if node["type"] == "split":
                lref = node["left"]
                rref = node["right"]
                if isinstance(lref, dict):
                    node["left"] = n_internal + lref["leaf_idx"]
                if isinstance(rref, dict):
                    node["right"] = n_internal + rref["leaf_idx"]

        trees.append({"nodes": nodes})

    output = {
        "format": "lgbm",
        "features": features,
        "num_features": len(features),
        "num_trees": len(trees),
        "is_classifier": is_classifier,
        "trees": trees,
    }

    out_path = pkl_path.with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  {pkl_path.name} -> {out_path.name} ({len(trees)} trees, {size_mb:.1f} MB)")
    return out_path


def _parse_lgbm_text_tree(text_lines: list[str], tree_idx: int) -> dict | None:
    """Parse a single tree from LightGBM text model format."""
    in_tree = False
    tree_data = {}
    for line in text_lines:
        line = line.strip()
        if line == f"Tree={tree_idx}":
            in_tree = True
            continue
        if in_tree and line.startswith("Tree="):
            break
        if in_tree and "=" in line:
            key, val = line.split("=", 1)
            tree_data[key] = val
    if not tree_data or "split_feature" not in tree_data:
        return None
    return tree_data


def _flatten_xgb_tree(node: dict) -> list[dict]:
    """Flatten XGBoost recursive tree structure to a node list."""
    nodes = []

    def _walk(n: dict) -> int:
        idx = len(nodes)
        if "leaf" in n:
            nodes.append({"type": "leaf", "value": n["leaf"]})
            return idx

        feature = int(n["split"].lstrip("f"))
        threshold = n["split_condition"]
        yes_id = n["yes"]
        missing_id = n.get("missing", yes_id)
        default_left = (missing_id == yes_id)

        nodes.append(None)  # placeholder
        left_idx = _walk(n["children"][0])
        right_idx = _walk(n["children"][1])

        nodes[idx] = {
            "type": "split",
            "feature": feature,
            "threshold": threshold,
            "default_left": default_left,
            "left": left_idx,
            "right": right_idx,
        }
        return idx

    _walk(node)
    return nodes


def export_xgb(pkl_path: Path) -> Path:
    import tempfile

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)  # noqa: S301

    model = data["model"]
    features = list(data["features"])
    is_classifier = data.get("is_classifier", False)

    # Use native JSON save for full precision
    tmp = tempfile.mktemp(suffix=".json")
    model.save_model(tmp)
    with open(tmp) as f:
        full_model = json.load(f)
    Path(tmp).unlink()

    config = json.loads(model.save_config())
    base_score_raw = config["learner"]["learner_model_param"]["base_score"]
    if isinstance(base_score_raw, list):
        base_score = float(base_score_raw[0])
    elif isinstance(base_score_raw, str) and base_score_raw.startswith("["):
        base_score = float(base_score_raw.strip("[]"))
    else:
        base_score = float(base_score_raw)

    # Extract trees from native format (full float precision)
    gbtree = full_model["learner"]["gradient_booster"]["model"]["trees"]
    trees = []
    for tree_data in gbtree:
        num_nodes = int(tree_data["tree_param"]["num_nodes"])
        left_children = tree_data["left_children"]
        right_children = tree_data["right_children"]
        split_indices = tree_data["split_indices"]
        split_conditions = tree_data["split_conditions"]
        default_left_flags = tree_data["default_left"]
        base_weights = tree_data["base_weights"]

        nodes = []
        for i in range(num_nodes):
            if left_children[i] == -1:  # leaf node
                nodes.append({"type": "leaf", "value": base_weights[i]})
            else:
                nodes.append({
                    "type": "split",
                    "feature": split_indices[i],
                    "threshold": split_conditions[i],
                    "default_left": bool(default_left_flags[i]),
                    "left": left_children[i],
                    "right": right_children[i],
                })
        trees.append({"nodes": nodes})

    output = {
        "format": "xgb",
        "features": features,
        "num_features": len(features),
        "num_trees": len(trees),
        "is_classifier": is_classifier,
        "base_score": base_score,
        "trees": trees,
    }

    out_path = pkl_path.with_suffix(".json")
    with open(out_path, "w") as f:
        json.dump(output, f)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  {pkl_path.name} -> {out_path.name} ({len(trees)} trees, {size_mb:.1f} MB)")
    return out_path


def export_file(pkl_path: Path) -> None:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)  # noqa: S301

    model = data["model"]

    # XGBoost Booster has get_dump; LightGBM Booster has dump_model (no args)
    if hasattr(model, "get_dump"):
        export_xgb(pkl_path)
        return

    if hasattr(model, "dump_model") or hasattr(model, "booster_"):
        export_lgbm(pkl_path)
        return

    print(f"  SKIP {pkl_path.name}: unknown model type {type(model)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/export_model_to_json.py <model.pkl or directory>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_dir():
        pkl_files = sorted(target.rglob("*.pkl"))
        print(f"Found {len(pkl_files)} .pkl files in {target}")
        for p in pkl_files:
            try:
                export_file(p)
            except Exception as e:
                print(f"  ERROR {p.name}: {e}")
    else:
        export_file(target)


if __name__ == "__main__":
    main()
