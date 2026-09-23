#!/usr/bin/env python3
"""Mechanically split frozen benchmark classes out of the legacy source.

This one-time source migration preserves each class definition verbatim. Run
only when rebuilding the repository layout from reproduction/legacy/source.
"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reproduction/legacy/source/numerical_embeddings.py"
TREE = ast.parse(SOURCE.read_text())
NODES = {node.name: node for node in TREE.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
LINES = SOURCE.read_text().splitlines()


def source(name: str) -> str:
    node = NODES[name]
    return "\n".join(LINES[node.lineno - 1:node.end_lineno]) + "\n"


PRELUDE = """\
\"\"\"Frozen benchmark implementation, split verbatim from the experiment source.\"\"\"

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

"""

MODULES = {
    "field/embedding.py": ["FieldEmbedding"],
    "autodis/embedding.py": ["AutoDisEmbedding"],
    "quantile_bucket/embedding.py": ["QuantileBucketEmbedding"],
    "log_squared_bucket/embedding.py": ["LogSquaredBucketEmbedding"],
    "narydis/common.py": ["_to_nary_digits", "_normalized_digit_weights"],
    "narydis/lite.py": ["NaryDisLiteEmbedding"],
    "narydis/full.py": ["NaryDisEmbedding"],
    "bspline/embedding.py": ["BSplineEmbedding"],
    "deer/embedding.py": ["DEEREmbedding"],
    "dae/embedding.py": ["DistributionAwareEmbedding"],
    "daes/embedding.py": ["DAESEmbedding"],
    "common.py": ["StandardizedInputEmbedding", "_resample_bin_edges",
                  "_load_bins", "_load_standardization", "_maybe_standardize_embedding"],
}


def main() -> None:
    for relative, names in MODULES.items():
        target = ROOT / "method" / relative
        if target.exists():
            raise FileExistsError(f"Refuse to overwrite an extracted method: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        header = PRELUDE
        if relative == "narydis/lite.py":
            header += "from .common import _to_nary_digits, _normalized_digit_weights\n\n"
        elif relative == "narydis/full.py":
            header += "from .common import _to_nary_digits\n\n"
        elif relative == "deer/embedding.py":
            header += "from .engineering import precise_index as _deer_flat_index\n\n"
        elif relative == "common.py":
            header += "from .vendor import rtdl_num_embeddings as rtdl_num\n\n"
        target.write_text(header + "\n".join(source(name) for name in names))
        if relative != "common.py":
            package = target.parent / "__init__.py"
            if not package.exists():
                package.write_text('"""Numerical embedding family."""\n')
        print(f"EXTRACT {relative}: {', '.join(names)}")


if __name__ == "__main__":
    main()
