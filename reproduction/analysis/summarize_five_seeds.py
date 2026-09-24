#!/usr/bin/env python3
"""Summarize the completed five-seed benchmark without altering frozen inputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import friedmanchisquare, wilcoxon


METRICS = ("val_auc", "val_logloss", "test_auc", "test_logloss")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    per_seed = read_csv(result_dir / "per_seed.csv")
    summary = read_csv(result_dir / "summary.csv")
    if len(per_seed) != 2430 or len(summary) != 486:
        raise ValueError("Expected 2,430 runs and 486 five-seed cells")
    if {int(row["n_seeds"]) for row in summary} != {5}:
        raise ValueError("Summary does not contain five seeds per cell")

    by_cell: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in summary:
        by_cell[(row["dataset"], row["backbone"])].append(row)
    if len(by_cell) != 27 or any(len(rows) != 18 for rows in by_cell.values()):
        raise ValueError("Expected 27 complete dataset-backbone cells")
    methods = sorted({row["method"] for row in summary})
    ranks: dict[str, list[int]] = defaultdict(list)
    deltas: dict[str, list[float]] = defaultdict(list)
    wins: Counter[str] = Counter()
    top3: Counter[str] = Counter()
    values_by_method: dict[str, list[float]] = defaultdict(list)
    for key in sorted(by_cell):
        ordered = sorted(by_cell[key], key=lambda row: (-float(row["test_auc_mean"]), row["method"]))
        field_auc = float(next(row["test_auc_mean"] for row in ordered if row["method"] == "field"))
        for rank, row in enumerate(ordered, 1):
            method = row["method"]
            auc = float(row["test_auc_mean"])
            ranks[method].append(rank)
            deltas[method].append(auc - field_auc)
            values_by_method[method].append(auc)
            if rank == 1:
                wins[method] += 1
            if rank <= 3:
                top3[method] += 1

    robustness = []
    for method in methods:
        robustness.append({
            "method": method,
            "cells": 27,
            "average_rank": statistics.mean(ranks[method]),
            "median_rank": statistics.median(ranks[method]),
            "worst_rank": max(ranks[method]),
            "wins": wins[method],
            "top3_count": top3[method],
            "mean_delta_test_auc_vs_field": statistics.mean(deltas[method]),
            "mean_test_auc": statistics.mean(values_by_method[method]),
        })
    robustness.sort(key=lambda row: (row["average_rank"], row["method"]))
    write_csv(result_dir / "method_robustness.csv", robustness)

    ordered_cells = [by_cell[key] for key in sorted(by_cell)]
    matrix = {
        method: [float(next(row["test_auc_mean"] for row in cell if row["method"] == method))
                 for cell in ordered_cells]
        for method in methods
    }
    omnibus = friedmanchisquare(*(matrix[method] for method in methods))
    reference = robustness[0]["method"]
    comparisons = []
    for method in methods:
        if method == reference:
            continue
        differences = [a - b for a, b in zip(matrix[reference], matrix[method])]
        raw_p = float(wilcoxon(differences, alternative="two-sided").pvalue)
        comparisons.append({
            "reference_method": reference,
            "comparator": method,
            "mean_auc_difference_reference_minus_comparator": statistics.mean(differences),
            "wins": sum(diff > 0 for diff in differences),
            "ties": sum(diff == 0 for diff in differences),
            "losses": sum(diff < 0 for diff in differences),
            "p_raw": raw_p,
        })
    comparisons.sort(key=lambda row: (row["p_raw"], row["comparator"]))
    running_adjustment = 0.0
    for index, row in enumerate(comparisons):
        running_adjustment = max(running_adjustment, min(1.0, (len(comparisons) - index) * row["p_raw"]))
        row["p_holm"] = running_adjustment
        row["significant_0_05"] = running_adjustment < 0.05
    write_csv(result_dir / "holm_posthoc.csv", comparisons)
    friedman = {
        "test": "Friedman test over 27 dataset-backbone blocks",
        "blocks": 27,
        "methods": len(methods),
        "statistic": float(omnibus.statistic),
        "p_value": float(omnibus.pvalue),
        "posthoc_reference": reference,
        "posthoc": "paired two-sided Wilcoxon signed-rank with Holm correction",
    }
    (result_dir / "friedman.json").write_text(json.dumps(friedman, indent=2) + "\n", encoding="utf-8")

    best = robustness[0]
    lines = [
        "# Five-seed NumEmb benchmark",
        "",
        "## Protocol and provenance",
        "",
        "The 2026–2028 records are the frozen three-seed results. Seeds 2029 and 2030 were",
        "trained with the reorganized benchmark code and the same frozen configurations.",
        "The 972-run extension completed without failures; all 486 cells have exactly five",
        "seeds, for 2,430 per-seed records. This directory does not overwrite the original",
        "three-seed records. Rankings use mean test AUC within each of the 27 matched",
        "dataset–backbone cells; standard deviations in `summary.csv` are sample SDs.",
        "",
        "## Main result",
        "",
        f"**{reference}** has the best average rank ({best['average_rank']:.3f}),",
        f"{best['wins']}/27 wins, and {best['top3_count']}/27 top-three placements.",
        f"The Friedman statistic is {omnibus.statistic:.3f} (p={omnibus.pvalue:.3e}).",
        "Paired Wilcoxon tests use the top-ranked method as the reference and Holm",
        "correction across 17 comparisons; see `holm_posthoc.csv`.",
        "",
        "## Cross-cell robustness",
        "",
        "| Method | Mean rank | Wins / 27 | Top 3 / 27 | Worst rank | Mean test AUC delta vs Field |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in robustness:
        lines.append(f"| {row['method']} | {row['average_rank']:.3f} | {row['wins']} | "
                     f"{row['top3_count']} | {row['worst_rank']} | "
                     f"{row['mean_delta_test_auc_vs_field']:+.6f} |")
    lines.extend(["", "## Files", "", "- `per_seed.csv`: all 2,430 seed-level outcomes.",
                  "- `summary.csv`: 486 five-seed mean/SD cells with ranks and Field deltas.",
                  "- `extension_per_seed.csv`: the separately validated 972-run extension.",
                  "- `method_robustness.csv`, `friedman.json`, `holm_posthoc.csv`: derived analyses.", ""])
    (result_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"FIVE_SEED_READY reference={reference} mean_rank={best['average_rank']:.3f} "
          f"wins={best['wins']}/27 p={omnibus.pvalue:.3e}")


if __name__ == "__main__":
    main()
