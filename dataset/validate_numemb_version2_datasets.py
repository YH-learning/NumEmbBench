#!/usr/bin/env python3
"""Audit Version-2 dataset invariants without scanning full tables again."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def features(path: Path) -> dict:
    payload = json.loads((path / "feature_map.json").read_text())
    return {name: spec for entry in payload["features"] for name, spec in entry.items()}


def sample_indices(parquet: pq.ParquetFile) -> list[int]:
    values = {0, max(0, parquet.num_row_groups // 2), max(0, parquet.num_row_groups - 1)}
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--criteo-source", type=Path, required=True)
    parser.add_argument("--automl-a-source", type=Path, required=True)
    parser.add_argument("--automl-e-source", type=Path, required=True)
    args = parser.parse_args()
    pairs = {
        "Criteo_x4_NumEmbZScoreV2": args.criteo_source,
        "AutoML_A_ZScoreV2": args.automl_a_source,
        "AutoML_E_ZScoreV2": args.automl_e_source,
    }
    report = {"status": "pass", "datasets": {}}
    for dataset_id, source_dir in pairs.items():
        target_dir = args.dataset_root / dataset_id
        source_features = features(source_dir)
        target_features = features(target_dir)
        numeric = [name for name, spec in source_features.items() if spec.get("type") == "numeric"]
        dataset_report = {"splits": {}, "numeric_features": numeric}
        for split in ("train", "valid", "test"):
            source = pq.ParquetFile(source_dir / f"{split}.parquet")
            target = pq.ParquetFile(target_dir / f"{split}.parquet")
            assert source.metadata.num_rows == target.metadata.num_rows
            assert source.schema_arrow.names == target.schema_arrow.names
            max_abs_error = 0.0
            checked_rows = 0
            for group in sample_indices(source):
                raw = source.read_row_group(group)
                transformed = target.read_row_group(group)
                assert raw.num_rows == transformed.num_rows
                take = np.unique(np.linspace(0, raw.num_rows - 1, min(64, raw.num_rows), dtype=int))
                for name in raw.schema.names:
                    left = raw[name].combine_chunks().to_numpy(zero_copy_only=False)[take]
                    right = transformed[name].combine_chunks().to_numpy(zero_copy_only=False)[take]
                    if name in numeric:
                        spec = source_features[name]
                        mean = float(spec["numeric_mean"])
                        std = float(spec["numeric_std"])
                        scale = std if std > 1e-12 else 1.0
                        expected = ((left.astype(np.float64) - mean) / scale).astype(np.float32)
                        error = float(np.max(np.abs(expected - right.astype(np.float32))))
                        max_abs_error = max(max_abs_error, error)
                        assert np.allclose(expected, right, rtol=0.0, atol=1e-6)
                    else:
                        assert np.array_equal(left, right)
                checked_rows += len(take)
            dataset_report["splits"][split] = {
                "rows": source.metadata.num_rows,
                "sampled_rows": checked_rows,
                "max_numeric_abs_error": max_abs_error,
                "categorical_labels_exact": True,
                "row_order_preserved": True,
            }
        vocab = json.loads((target_dir / "log_squared_bucket_vocab.json").read_text())
        assert sorted(vocab) == sorted(numeric)
        assert all(value.get("__PAD__") == 0 and value.get("__OOV__") == 1 for value in vocab.values())
        dataset_report["log_squared_vocabulary_sizes"] = {name: len(value) for name, value in vocab.items()}
        report["datasets"][dataset_id] = dataset_report
        print("PASS", dataset_id, flush=True)
    destination = args.dataset_root / "validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("COMPLETE", destination, flush=True)


if __name__ == "__main__":
    main()
