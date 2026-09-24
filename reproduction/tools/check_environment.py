#!/usr/bin/env python3
"""Read-only environment and source-integrity check for the benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKBONES = {
    "DNN": "DNN/DNN.py",
    "DeepFM": "DeepFM/DeepFM.py",
    "WideDeep": "WideDeep/WideDeep.py",
    "NFM": "NFM/NFM.py",
    "PNN": "PNN/PNN.py",
    "DCN": "DCN/DCN.py",
    "DCNv2": "DCNv2/DCNv2.py",
    "AutoInt": "AutoInt/AutoInt.py",
    "FiBiNET": "FiBiNET/FiBiNET.py",
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--allow-missing-packages", action="store_true",
                        help="Only check code layout; useful on an analysis-only machine")
    args = parser.parse_args()

    version_file = ROOT / "fuxictr/version.py"
    version_match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", version_file.read_text()) if version_file.is_file() else None
    framework_version = version_match.group(1) if version_match else None
    backbone_files = {name: (ROOT / "backbone" / relative).is_file()
                      for name, relative in BACKBONES.items()}
    integrations = {}
    for name in ("feature_embedding.py", "numerical_embedding.py"):
        patch = ROOT / "reproduction/legacy/integration/fuxictr_2_3_10" / name
        deployed = ROOT / "fuxictr/pytorch/layers/embeddings" / name
        integrations[name] = patch.is_file() and deployed.is_file() and digest(patch) == digest(deployed)
    packages = {name: importlib.util.find_spec(name) is not None
                for name in ("numpy", "pandas", "torch", "pyarrow", "yaml", "sklearn",
                             "packaging", "keras_preprocessing", "h5py", "tqdm", "polars")}
    packages["rtdl_num_embeddings"] = (
        importlib.util.find_spec("rtdl_num_embeddings") is not None
        or (ROOT / "method/vendor/rtdl_num_embeddings.py").is_file()
    )
    cuda = None
    torch_version = None
    if packages["torch"]:
        import torch
        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
    issues = []
    if sys.version_info < (3, 10):
        issues.append("Python 3.10 or newer is required")
    if framework_version != "2.3.10":
        issues.append(f"Expected bundled FuxiCTR 2.3.10, found {framework_version!r}")
    if not all(backbone_files.values()):
        issues.append("Missing backbone source: " + ", ".join(name for name, ok in backbone_files.items() if not ok))
    if not all(integrations.values()):
        issues.append("Bundled FuxiCTR integration files differ from their frozen patch copies")
    if not (ROOT / "method/numemb_plugin.py").is_file():
        issues.append("Missing method/numemb_plugin.py entry point required by frozen configurations")
    if not args.allow_missing_packages and not all(packages.values()):
        issues.append("Missing Python packages: " + ", ".join(name for name, ok in packages.items() if not ok))
    if args.require_cuda and not cuda:
        issues.append("CUDA was required but is unavailable")
    report = {
        "status": "pass" if not issues else "fail",
        "python": sys.version.split()[0],
        "fuxictr": framework_version,
        "torch": torch_version,
        "cuda_available": cuda,
        "backbones": backbone_files,
        "integration_matches": integrations,
        "packages": packages,
        "issues": issues,
    }
    print(json.dumps(report, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
