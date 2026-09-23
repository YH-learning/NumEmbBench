#!/usr/bin/env python3
"""Recompute the 486 three-seed cells and paper headline from released runs."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACCURACY = ROOT / "results/tuned_accuracy"
OUTPUT = ROOT / "reproduced"
METRICS = ("val_auc", "val_logloss", "test_auc", "test_logloss")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
        raise AssertionError(f"{label}: recomputed={actual}, released={expected}")


def main() -> None:
    per_seed = rows(ACCURACY / "per_seed.csv")
    summary = rows(ACCURACY / "summary.csv")
    protocol = json.loads((ACCURACY / "config_snapshot/protocol.json").read_text())
    if len(per_seed) != protocol["expected_tasks"]:
        raise AssertionError(f"Expected {protocol['expected_tasks']} runs, found {len(per_seed)}")
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_seed:
        key = (row["dataset"], row["backbone"], row["method"])
        grouped[key].append(row)
    if len(grouped) != 486 or len(summary) != 486:
        raise AssertionError("Expected 486 dataset/backbone/method cells")
    summary_by_key = {(r["dataset"], r["backbone"], r["method"]): r for r in summary}
    if set(grouped) != set(summary_by_key):
        raise AssertionError("Per-seed and summary cell keys differ")
    output = []
    for key, run_rows in sorted(grouped.items()):
        seeds = sorted(int(row["seed"]) for row in run_rows)
        if seeds != protocol["seeds"]:
            raise AssertionError(f"Incomplete seeds for {key}: {seeds}")
        released = summary_by_key[key]
        computed: dict[str, str | float | int] = dict(zip(("dataset", "backbone", "method"), key))
        computed["n_seeds"] = len(run_rows)
        for metric in METRICS:
            values = [float(row[metric]) for row in run_rows]
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            close(mean, float(released[f"{metric}_mean"]), f"{key} {metric} mean")
            close(std, float(released[f"{metric}_std"]), f"{key} {metric} std")
            computed[f"{metric}_mean"] = mean
            computed[f"{metric}_std"] = std
        output.append(computed)
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "recomputed_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    rank = rows(ACCURACY / "method_robustness.csv")
    deer = next((r for r in rank if r.get("method") == "deer"), None)
    report = {
        "status": "pass",
        "per_seed_runs": len(per_seed),
        "cells": len(grouped),
        "datasets": sorted({key[0] for key in grouped}),
        "backbones": sorted({key[1] for key in grouped}),
        "methods": sorted({key[2] for key in grouped}),
        "deer_robustness_record": deer,
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
