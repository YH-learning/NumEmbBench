#!/usr/bin/env python3
"""Generate the complete per-dataset, per-backbone result tables."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/tuned_accuracy/summary.csv"
OUTPUT = ROOT / "reproduced/full_cell_results.tex"

DATASETS = ["Criteo", "AutoML-A", "AutoML-E"]
BACKBONES = ["DNN", "DeepFM", "WideDeep", "NFM", "PNN", "DCN", "DCNv2", "AutoInt", "FiBiNET"]
FAMILIES = [
    ("Pointwise projection", [
        ("field", "Field"), ("linear", "Linear"), ("linear_relu", "Linear-ReLU")]),
    ("Discretization", [
        ("quantile_bucket", "Quantile"), ("log_squared_bucket", "LogBucket"), ("autodis", "AutoDis")]),
    ("Fixed basis", [
        ("periodic", "Periodic"), ("bspline", "B-spline")]),
    ("Piecewise basis", [
        ("ple_a", "PLE-A"), ("ple_b", "PLE-B"), ("ple_relu", "PLE-ReLU"), ("plr", "PLR")]),
    ("Adaptive coordinates", [
        ("narydis_lite", "NaryDis-Lite"), ("narydis_full", "NaryDis"), ("deer", "DEER")]),
    ("Distribution/context adaptive", [
        ("dae", "DAE"), ("daes_gate", "DAES-Gate"), ("daes_tran", "DAES-Tran")]),
]
METHODS = [method for _, family_methods in FAMILIES for method in family_methods]
FAMILY_LINES = {
    "Pointwise projection": ("Pointwise", "projection"),
    "Discretization": ("Discretization",),
    "Fixed basis": ("Fixed basis",),
    "Piecewise basis": ("Piecewise basis",),
    "Adaptive coordinates": ("Adaptive", "coordinates"),
    "Distribution/context adaptive": ("Distribution/", "context adaptive"),
}


def fmt(value: str, best: float, lower_is_better: bool) -> str:
    number = float(value)
    rendered = f"{number:.4f}"
    is_best = abs(number - best) < 5e-7
    return rf"\textbf{{{rendered}}}" if is_best else rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    with args.summary.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(r["dataset"], r["backbone"], r["method"]): r for r in rows}

    seeds = {int(r["n_seeds"]) for r in rows}
    if len(rows) != 486 or len(seeds) != 1:
        raise ValueError("Expected 486 complete cells with one common seed count")
    n_seeds = seeds.pop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    blocks: list[str] = [
        r"\begin{table*}[t]",
        rf"\caption{{Complete test results for all 18 methods, three datasets, and nine backbones, averaged over {n_seeds} seeds. The first column gives each method's full numerical-embedding family. AUC and logloss (LL) are reported separately for every cell; bold marks the best value in each metric column.}}",
        r"\label{tab:full-results}",
        r"\centering",
        r"\renewcommand{\arraystretch}{0.68}",
        r"\setlength{\tabcolsep}{1.85pt}",
        r"\resizebox{0.94\textwidth}{!}{%",
        r"\begin{tabular}{llrrrrrrrrrrrrrrrrrr}",
        r"\toprule",
    ]
    for dataset_index, dataset in enumerate(DATASETS):
        best_auc = {
            backbone: max(float(lookup[(dataset, backbone, method)]["test_auc_mean"]) for method, _ in METHODS)
            for backbone in BACKBONES
        }
        best_ll = {
            backbone: min(float(lookup[(dataset, backbone, method)]["test_logloss_mean"]) for method, _ in METHODS)
            for backbone in BACKBONES
        }

        blocks.extend(
            [
                rf"\multicolumn{{20}}{{c}}{{\textbf{{{dataset}}}}} \\",
                r"\midrule",
                "Family & Method & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{backbone}}}" for backbone in BACKBONES) + r" \\",
                r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-14}\cmidrule(lr){15-16}\cmidrule(lr){17-18}\cmidrule(lr){19-20}",
                " &  & " + " & ".join(["AUC", "LL"] * len(BACKBONES)) + r" \\",
                r"\midrule",
            ]
        )

        for family_index, (family_name, family_methods) in enumerate(FAMILIES):
            family_text = r"\\".join(FAMILY_LINES[family_name])
            family_cell = rf"\multirow{{{len(family_methods)}}}{{*}}{{\shortstack[l]{{{family_text}}}}}"
            for method_index, (method, display_name) in enumerate(family_methods):
                values: list[str] = []
                for backbone in BACKBONES:
                    row = lookup[(dataset, backbone, method)]
                    values.append(fmt(row["test_auc_mean"], best_auc[backbone], lower_is_better=False))
                    values.append(fmt(row["test_logloss_mean"], best_ll[backbone], lower_is_better=True))
                blocks.append((family_cell if method_index == 0 else "") + " & " + display_name + " & " + " & ".join(values) + r" \\")
            if family_index != len(FAMILIES) - 1:
                blocks.append(r"\midrule")
        if dataset_index != len(DATASETS) - 1:
            blocks.extend([r"\midrule", r"\addlinespace[2pt]"])

    blocks.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
            "",
        ]
    )

    args.output.write_text("\n".join(blocks), encoding="utf-8")


if __name__ == "__main__":
    main()
