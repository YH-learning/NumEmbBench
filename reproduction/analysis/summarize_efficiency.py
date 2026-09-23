#!/usr/bin/env python3
"""Aggregate the frozen-tuned efficiency matrix without third-party packages."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EFF = ROOT / "results/tuned_efficiency"
OUT = ROOT / "analysis_outputs/tuned_efficiency"
ACCURACY = ROOT / "results/tuned_accuracy/summary.csv"
TAXONOMY = ROOT / "analysis_outputs/taxonomy_method_summary.csv"

KEYS = ["dataset", "backbone", "method"]
ID_COLS = set(KEYS + ["round", "device"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(rows: list[dict[str, str]], keys: list[str], metrics: list[str]) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output: list[dict[str, object]] = []
    for group_key, group_rows in sorted(groups.items()):
        record: dict[str, object] = dict(zip(keys, group_key))
        for metric in metrics:
            values = [float(row[metric]) for row in group_rows]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = std(values)
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pareto(rows: list[dict[str, object]], cost_cols: list[str]) -> set[str]:
    keep: set[str] = set()
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            weak = float(other["mean_test_auc"]) >= float(row["mean_test_auc"])
            weak = weak and all(float(other[c]) <= float(row[c]) for c in cost_cols)
            strict = float(other["mean_test_auc"]) > float(row["mean_test_auc"])
            strict = strict or any(float(other[c]) < float(row[c]) for c in cost_cols)
            if weak and strict:
                dominated = True
                break
        if not dominated:
            keep.add(str(row["method"]))
    return keep


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        values: list[str] = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                if key.endswith("bytes_mean") or key == "parameters_numeric_mean":
                    values.append(f"{value:,.0f}")
                else:
                    values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    raw = read_csv(EFF / "summary.csv")
    if len(raw) != 486:
        raise ValueError(f"Expected 486 measurements, found {len(raw)}")
    OUT.mkdir(parents=True, exist_ok=True)
    metrics = [column for column in raw[0] if column not in ID_COLS]

    cell = aggregate(raw, KEYS, metrics)
    write_csv(OUT / "cell_mean_std.csv", cell)
    overall = aggregate(raw, ["method"], metrics)

    measured_methods = {row["method"] for row in raw}
    accuracy_rows = [row for row in read_csv(ACCURACY) if row["method"] in measured_methods]
    if len(accuracy_rows) != 486:
        raise ValueError(f"Expected 486 accuracy cells, found {len(accuracy_rows)}")
    accuracy = {
        row["method"]: row
        for row in aggregate(accuracy_rows, ["method"], ["test_auc_mean", "rank_by_test_auc"])
    }
    taxonomy = {
        row["method"]: row
        for row in read_csv(TAXONOMY)
        if row["protocol"] == "Shared z-score"
    }
    field_by_cell_round = {
        (row["dataset"], row["backbone"], row["round"]): row
        for row in raw
        if row["method"] == "field"
    }
    ratio_metrics = [
        "train_median_ms",
        "infer_b4096_median_ms",
        "embedding_b4096_median_ms",
        "train_peak_allocated_bytes",
    ]
    ratios: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in raw:
        baseline = field_by_cell_round[(row["dataset"], row["backbone"], row["round"])]
        for metric in ratio_metrics:
            ratios[row["method"]][metric].append(float(row[metric]) / float(baseline[metric]))

    merged: list[dict[str, object]] = []
    for record in overall:
        method = str(record["method"])
        acc = accuracy[method]
        meta = taxonomy[method]
        if abs(float(meta["mean_test_auc"]) - float(acc["test_auc_mean_mean"])) > 1e-10:
            raise ValueError(f"Accuracy mismatch for {method}")
        combined: dict[str, object] = dict(record)
        combined.update(
            display_name=meta["display_name"],
            family=meta["family"],
            mean_test_auc=float(acc["test_auc_mean_mean"]),
            mean_rank=float(acc["rank_by_test_auc_mean"]),
        )
        for metric in ratio_metrics:
            combined[f"{metric}_x_field"] = mean(ratios[method][metric])
        combined["peak_memory_mib_mean"] = float(combined["train_peak_allocated_bytes_mean"]) / 2**20
        merged.append(combined)

    train_front = pareto(merged, ["train_median_ms_mean"])
    infer_front = pareto(merged, ["infer_b4096_median_ms_mean"])
    joint_front = pareto(
        merged,
        ["train_median_ms_mean", "infer_b4096_median_ms_mean", "parameters_numeric_mean"],
    )
    for row in merged:
        method = str(row["method"])
        row["pareto_auc_train"] = method in train_front
        row["pareto_auc_infer"] = method in infer_front
        row["pareto_auc_train_infer_params"] = method in joint_front
    merged.sort(key=lambda row: float(row["mean_rank"]))
    write_csv(OUT / "method_efficiency_pareto.csv", merged)

    md = [
        "# Frozen-Tuned Numerical Embedding Efficiency Matrix",
        "",
        "486/486 successful measurements: 18 methods x 3 datasets x 3 representative backbones x 3 randomized rounds.",
        "All measurements use the Stage-2 validation-frozen configurations, one idle GPU, eager FP32, real batches, and a resident model. Disk I/O is excluded.",
        "Paired multipliers are computed against Field in the same dataset, backbone, and randomized round.",
        "",
        markdown_table(
            merged,
            [
                ("display_name", "Method"),
                ("mean_test_auc", "Mean AUC"),
                ("mean_rank", "Rank"),
                ("parameters_numeric_mean", "Numeric params"),
                ("train_median_ms_mean", "Train ms"),
                ("train_median_ms_x_field", "Train x Field"),
                ("infer_b4096_median_ms_mean", "Infer ms"),
                ("infer_b4096_median_ms_x_field", "Infer x Field"),
                ("embedding_b4096_median_ms_mean", "Embed ms"),
                ("peak_memory_mib_mean", "Peak MiB"),
                ("pareto_auc_train", "Train front"),
                ("pareto_auc_infer", "Infer front"),
            ],
        ),
        "",
        "The train and inference fronts are defined separately by maximizing mean test AUC and minimizing the corresponding latency. The joint flag in the CSV additionally minimizes numerical parameters.",
    ]
    (OUT / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
