#!/usr/bin/env python3
"""Construct every registered numerical method against a real feature map.

This checks package wiring and constructor compatibility without starting a
training run or loading any dataset rows. Pass a prepared feature_map.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "method"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_map", type=Path)
    parser.add_argument("--embedding-dim", type=int, default=16)
    args = parser.parse_args()

    import numemb_plugin  # noqa: F401: registers bundled methods
    from fuxictr.features import FeatureMap
    from method.registry import SUPPORTED_NUM_EMBEDDINGS, build_numerical_embedding

    path = args.feature_map.resolve()
    with path.open(encoding="utf-8") as stream:
        snapshot = json.load(stream)
    feature_map = FeatureMap(snapshot["dataset_id"], str(path.parent))
    feature_map.load(str(path), {"embedding_dim": args.embedding_dim})
    numeric_features = [
        name for name, spec in feature_map.features.items()
        if spec.get("type") == "numeric"
    ]
    if not numeric_features:
        raise SystemExit("feature map has no numeric fields")
    # Construct only two fields to keep CPU smoke checks cheap while retaining
    # the actual training statistics, bins and context-field catalog.
    numeric_features = numeric_features[:2]
    built = []
    for method in SUPPORTED_NUM_EMBEDDINGS:
        params = {}
        if method == "log_squared_bucket":
            params["vocabularies"] = {
                name: {"__PAD__": 0, "__OOV__": 1, "1": 2}
                for name in numeric_features
            }
        module = build_numerical_embedding(
            method, numeric_features, feature_map, args.embedding_dim, params
        )
        built.append(f"{method}: {type(module).__name__}")
    print("Constructed all registered methods:")
    print("\n".join(built))


if __name__ == "__main__":
    main()
