#!/usr/bin/env python3
"""Materialize the shared train-z-score datasets for NumEmb benchmark V2.

All numerical columns are transformed with statistics stored in the source
feature map.  Those statistics were computed from the training split.  The
same affine transform is then applied to train, validation, and test.  Labels,
categorical ids, split membership, and row order are copied unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def feature_items(feature_map: dict) -> list[tuple[str, dict]]:
    items = []
    for entry in feature_map["features"]:
        if len(entry) != 1:
            raise ValueError(f"Malformed feature entry: {entry}")
        items.append(next(iter(entry.items())))
    return items


def zscore_metadata(spec: dict) -> tuple[dict, float, float]:
    out = copy.deepcopy(spec)
    mean = float(spec["numeric_mean"])
    raw_std = float(spec["numeric_std"])
    scale = raw_std if raw_std > 1e-12 else 1.0
    out["raw_numeric_pre_transform"] = spec.get("numeric_pre_transform", "identity")
    out["raw_numeric_mean"] = mean
    out["raw_numeric_std"] = raw_std
    out["raw_numeric_min"] = float(spec["numeric_min"])
    out["raw_numeric_max"] = float(spec["numeric_max"])
    out["numeric_pre_transform"] = "train_zscore_v2"
    out["numeric_storage_dtype"] = "float32"
    out["numeric_mean"] = 0.0
    out["numeric_std"] = 0.0 if raw_std <= 1e-12 else 1.0
    out["numeric_min"] = (float(spec["numeric_min"]) - mean) / scale
    out["numeric_max"] = (float(spec["numeric_max"]) - mean) / scale
    out["numeric_bin_edges"] = [
        (float(value) - mean) / scale for value in spec["numeric_bin_edges"]
    ]
    return out, mean, scale


def log_squared_tokens(values: np.ndarray) -> np.ndarray:
    safe = np.nan_to_num(values.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    integer = np.trunc(safe).astype(np.int64)
    transformed = np.floor(np.square(np.log(np.maximum(integer, 1)))).astype(np.int64)
    return np.where(integer > 2, transformed, integer)


def sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def transform_split(
    source: Path,
    destination: Path,
    numeric: dict[str, tuple[float, float]],
    vocabulary_tokens: dict[str, set[int]] | None,
    batch_size: int,
) -> dict:
    parquet = pq.ParquetFile(source)
    writer = None
    rows = 0
    started = time.time()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            for name, (mean, scale) in numeric.items():
                index = table.schema.get_field_index(name)
                if index < 0:
                    raise KeyError(f"{name} is missing from {source}")
                raw = table.column(index).combine_chunks().to_numpy(zero_copy_only=False)
                standardized = ((raw.astype(np.float64, copy=False) - mean) / scale).astype(np.float32)
                table = table.set_column(index, name, pa.array(standardized, type=pa.float32()))
                if vocabulary_tokens is not None:
                    tokens = log_squared_tokens(standardized)
                    vocabulary_tokens[name].update(int(x) for x in np.unique(tokens) if int(x) > 0)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return {
        "source": str(source),
        "destination": str(destination),
        "rows": rows,
        "source_rows": parquet.metadata.num_rows,
        "elapsed_seconds": time.time() - started,
        "bytes": destination.stat().st_size,
    }


def build_dataset(dataset_id: str, source_dir: Path, output_root: Path, batch_size: int) -> dict:
    output_dir = output_root / dataset_id
    if output_dir.exists():
        raise RuntimeError(f"Refuse to overwrite existing V2 dataset: {output_dir}")
    output_dir.mkdir(parents=True)

    source_map = json.loads((source_dir / "feature_map.json").read_text())
    target_map = copy.deepcopy(source_map)
    target_map["dataset_id"] = dataset_id
    numeric = {}
    new_features = []
    stats = {}
    for name, spec in feature_items(source_map):
        if spec.get("type") == "numeric":
            transformed, mean, scale = zscore_metadata(spec)
            numeric[name] = (mean, scale)
            stats[name] = {
                "source_mean": mean,
                "source_std": float(spec["numeric_std"]),
                "applied_scale": scale,
                "mean": transformed["numeric_mean"],
                "std": transformed["numeric_std"],
                "min": transformed["numeric_min"],
                "max": transformed["numeric_max"],
                "bin_edges": transformed["numeric_bin_edges"],
            }
            spec = transformed
        new_features.append({name: spec})
    if not numeric:
        raise ValueError(f"No numerical features found in {source_dir}")
    target_map["features"] = new_features

    vocab_tokens = {name: set() for name in numeric}
    split_records = {}
    for split in ("train", "valid", "test"):
        split_records[split] = transform_split(
            source_dir / f"{split}.parquet",
            output_dir / f"{split}.parquet",
            numeric,
            vocab_tokens if split == "train" else None,
            batch_size,
        )
        if split_records[split]["rows"] != split_records[split]["source_rows"]:
            raise RuntimeError(f"Row-count mismatch for {dataset_id}/{split}")

    vocabularies = {}
    for name, values in vocab_tokens.items():
        vocabulary = {"__PAD__": 0, "__OOV__": 1}
        vocabulary.update({str(value): index for index, value in enumerate(sorted(values), start=2)})
        vocabularies[name] = vocabulary

    (output_dir / "feature_map.json").write_text(json.dumps(target_map, indent=2) + "\n")
    (output_dir / "numeric_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output_dir / "log_squared_bucket_vocab.json").write_text(
        json.dumps(vocabularies, indent=2) + "\n"
    )
    for name in ("feature_vocab.json",):
        source = source_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)

    manifest = {
        "dataset_id": dataset_id,
        "version": 2,
        "protocol": "all numerical columns use train-only z-score; all other values and row order unchanged",
        "source_dataset": str(source_dir),
        "numeric_features": list(numeric),
        "splits": split_records,
        "files": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            manifest["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=262144)
    parser.add_argument("--criteo-source", type=Path, required=True)
    parser.add_argument("--automl-a-source", type=Path, required=True)
    parser.add_argument("--automl-e-source", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    top = {"version": 2, "created_at_unix": time.time(), "datasets": {}}
    sources = {
        "Criteo_x4_NumEmbZScoreV2": args.criteo_source,
        "AutoML_A_ZScoreV2": args.automl_a_source,
        "AutoML_E_ZScoreV2": args.automl_e_source,
    }
    for dataset_id, source_dir in sources.items():
        if not (source_dir / "feature_map.json").is_file():
            raise FileNotFoundError(f"Missing source feature map: {source_dir}")
        print(f"START dataset={dataset_id} source={source_dir}", flush=True)
        top["datasets"][dataset_id] = build_dataset(
            dataset_id, source_dir, args.output_root, args.batch_size
        )
        print(f"DONE dataset={dataset_id}", flush=True)
    (args.output_root.parent / "dataset_manifest.json").write_text(
        json.dumps(top, indent=2) + "\n"
    )
    print(f"COMPLETE root={args.output_root}", flush=True)


if __name__ == "__main__":
    main()
