#!/usr/bin/env python3
"""Summarize one new benchmark run without mixing it with frozen paper data."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from run_benchmark import valid_result


def metrics_from_result(path: Path) -> dict[str, float]:
    lines = path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError(f"Expected exactly one result row: {path}")
    line = lines[0]
    output = {}
    for split in ("val", "test"):
        match = re.search(rf"\[{split}\] (.*?)(?:,\[|$)", line)
        if match is None:
            raise ValueError(f"Missing {split} metrics: {path}")
        for metric in ("AUC", "logloss"):
            number = re.search(rf"(?:^| - ){metric}: ([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", match.group(1))
            if number is None:
                raise ValueError(f"Missing {split} {metric}: {path}")
            output[f"{split}_{metric.lower()}"] = float(number.group(1))
    return output


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory produced by reproduction/tools/run_benchmark.py")
    parser.add_argument("--require-complete", action="store_true", help="Fail if any selected task is missing or invalid")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "task_manifest.json").read_text())
    tasks = manifest["selected_tasks"]
    per_seed = []
    missing = []
    for task in tasks:
        path = run_dir / task["slug"] / "results" / f"{task['task']}.csv"
        if not valid_result(path, task):
            missing.append(task["task"])
            continue
        per_seed.append({key: task[key] for key in ("slug", "dataset_id", "backbone", "method", "seed", "experiment", "task")}
                        | metrics_from_result(path))
    fields = ["slug", "dataset_id", "backbone", "method", "seed", "experiment", "task",
              "val_auc", "val_logloss", "test_auc", "test_logloss"]
    write_csv(run_dir / "per_seed.csv", per_seed, fields)
    groups = defaultdict(list)
    for row in per_seed:
        groups[(row["slug"], row["backbone"], row["method"])].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        row = {"slug": key[0], "backbone": key[1], "method": key[2],
               "n_seeds": len(group), "seeds": ";".join(str(item["seed"]) for item in sorted(group, key=lambda item: item["seed"]))}
        for metric in ("val_auc", "val_logloss", "test_auc", "test_logloss"):
            values = [item[metric] for item in group]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(row)
    summary_fields = ["slug", "backbone", "method", "n_seeds", "seeds"] + [
        f"{metric}_{stat}" for metric in ("val_auc", "val_logloss", "test_auc", "test_logloss")
        for stat in ("mean", "std")
    ]
    write_csv(run_dir / "summary.csv", summary, summary_fields)
    (run_dir / "summary_status.json").write_text(json.dumps({
        "selected_tasks": len(tasks), "complete_tasks": len(per_seed), "missing_or_invalid_tasks": missing,
        "cells_with_results": len(summary), "all_complete": not missing,
    }, indent=2) + "\n")
    print(f"SUMMARY complete={len(per_seed)}/{len(tasks)} cells={len(summary)} run_dir={run_dir}")
    if missing and args.require_complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
