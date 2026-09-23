#!/usr/bin/env python3
"""Import the frozen method registry and all paired FuxiCTR backbones."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "method"))

import numemb_plugin  # noqa: E402,F401
from fuxictr.pytorch.layers.embeddings.numerical_embedding import list_numerical_embeddings  # noqa: E402
from backbone.run_expid import resolve_model_class  # noqa: E402


def main() -> None:
    snapshot = ROOT / "reproduction/results/tuned_accuracy/config_snapshot"
    protocol = json.loads((snapshot / "protocol.json").read_text())
    used = set()
    for slug in ("criteo", "automl_a", "automl_e"):
        configuration = yaml.safe_load((snapshot / slug / "model_config.yaml").read_text())
        for name, params in configuration.items():
            if name.startswith("Tuned_"):
                used.add(params["numeric_embedding"])
    missing = used - set(list_numerical_embeddings())
    if missing:
        raise RuntimeError(f"Frozen numerical methods not registered: {sorted(missing)}")
    resolved = {}
    for name in protocol["backbones"]:
        resolved[name] = resolve_model_class(name).__name__
    if any(name != class_name for name, class_name in resolved.items()):
        raise RuntimeError(f"Backbone class mismatch: {resolved}")
    print(json.dumps({"status": "pass", "registered_methods_used": sorted(used),
                      "backbones": resolved}, indent=2))


if __name__ == "__main__":
    main()
