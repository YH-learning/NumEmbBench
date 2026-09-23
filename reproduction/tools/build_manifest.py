#!/usr/bin/env python3
"""Record checksums for the static inputs in this reproducibility copy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = {"manifest.json", ".DS_Store"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main() -> None:
    files = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name in EXCLUDED or path.name.startswith("._"):
            continue
        relative = path.relative_to(ROOT)
        if any(part in {"reproduced", "analysis_outputs", "figures", "outputs", ".venv"} for part in relative.parts):
            continue
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        files[str(relative)] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    payload = {"format": 1, "static_files": files}
    (ROOT / "reproduction/manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Recorded {len(files)} static files")


if __name__ == "__main__":
    main()
