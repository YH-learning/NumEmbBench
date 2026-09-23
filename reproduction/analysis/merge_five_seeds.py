#!/usr/bin/env python3
"""Combine frozen three-seed records with a completed 2029/2030 extension.

This writes a new versioned result directory; the frozen three-seed results
and the paper table are never changed by this script.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "results/tuned_accuracy/per_seed.csv"
DATASETS = {"criteo": "Criteo", "automl_a": "AutoML-A", "automl_e": "AutoML-E"}
METRICS = ("val_auc", "val_logloss", "test_auc", "test_logloss")
SEEDS = {2026, 2027, 2028, 2029, 2030}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def merge_rows(original: list[dict[str, str]], extension: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    records: dict[tuple[str, str, str, int], dict] = {}
    for source, rows, expected_seeds in (("frozen", original, {2026, 2027, 2028}),
                                         ("extension", extension, {2029, 2030})):
        for row in rows:
            slug = row.get("dataset_slug") or row.get("slug")
            if slug not in DATASETS:
                raise ValueError(f"Unknown dataset slug: {slug}")
            seed = int(row["seed"])
            if seed not in expected_seeds:
                raise ValueError(f"Unexpected {source} seed {seed}")
            key = (slug, row["backbone"], row["method"], seed)
            if key in records:
                raise ValueError(f"Duplicate result: {key}")
            records[key] = {"dataset": DATASETS[slug], "dataset_slug": slug,
                            "backbone": row["backbone"], "method": row["method"],
                            "seed": seed, "source": source,
                            **{metric: float(row[metric]) for metric in METRICS}}
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in records.values():
        groups[(row["dataset"], row["backbone"], row["method"])].append(row)
    if len(groups) != 486 or len(records) != 2430:
        raise ValueError(f"Incomplete matrix: {len(groups)} cells, {len(records)} runs; expected 486/2430")
    for key, rows in groups.items():
        if {row["seed"] for row in rows} != SEEDS:
            raise ValueError(f"Incomplete five-seed cell: {key}")
    per_seed = sorted(records.values(), key=lambda r: (r["dataset"], r["backbone"], r["method"], r["seed"]))
    summary = []
    for (dataset, backbone, method), rows in sorted(groups.items()):
        result = {"dataset": dataset, "backbone": backbone, "method": method, "n_seeds": 5}
        for metric in METRICS:
            values = [row[metric] for row in rows]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_std"] = statistics.stdev(values)
        summary.append(result)
    by_backbone = defaultdict(list)
    for row in summary:
        by_backbone[(row["dataset"], row["backbone"])].append(row)
    for rows in by_backbone.values():
        field = next(row["test_auc_mean"] for row in rows if row["method"] == "field")
        for rank, row in enumerate(sorted(rows, key=lambda r: (-r["test_auc_mean"], r["method"])), 1):
            row["rank_by_test_auc"] = rank
            row["field_test_auc_mean"] = field
            row["delta_test_auc_vs_field"] = row["test_auc_mean"] - field
    return per_seed, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True,
                        help="per_seed.csv from a complete 2029/2030 run")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    per_seed, summary = merge_rows(read_csv(ORIGINAL), read_csv(args.extension))
    if args.output_dir.exists():
        raise FileExistsError(f"Refuse to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_seed.csv", per_seed)
    write_csv(args.output_dir / "summary.csv", summary)
    print(f"READY {args.output_dir}: {len(per_seed)} runs, {len(summary)} cells, 5 seeds")


if __name__ == "__main__":
    main()
