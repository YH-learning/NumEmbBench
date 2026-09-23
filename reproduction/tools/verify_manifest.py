#!/usr/bin/env python3
"""Check the static-file SHA-256 manifest of the copied artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "reproduction/manifest.json").read_text())
    for relative, expected in manifest["static_files"].items():
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise RuntimeError(f"Checksum mismatch: {relative}")
    print(f"PASS {len(manifest['static_files'])} static files")


if __name__ == "__main__":
    main()
