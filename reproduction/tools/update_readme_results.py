"""Render the frozen benchmark matrix into the repository README.

The marked README block is generated from summary.csv, never hand-edited.
Use --check in CI to detect drift without changing files.
"""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
SUMMARY = ROOT / "reproduction/results/tuned_accuracy/summary.csv"
START = "<!-- BENCHMARK_RESULTS_START -->"
END = "<!-- BENCHMARK_RESULTS_END -->"
DATASETS = ("Criteo", "AutoML-A", "AutoML-E")
BACKBONES = ("DNN", "DeepFM", "WideDeep", "NFM", "PNN", "DCN", "DCNv2", "AutoInt", "FiBiNET")
METHODS = (
    "field", "linear", "linear_relu", "quantile_bucket", "log_squared_bucket",
    "autodis", "ple_a", "ple_b", "ple_relu", "periodic", "plr",
    "narydis_lite", "bspline", "narydis_full", "deer", "dae",
    "daes_gate", "daes_tran",
)


def load_rows():
    with SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(d, b, m) for d in DATASETS for b in BACKBONES for m in METHODS}
    found = {(r["dataset"], r["backbone"], r["method"]) for r in rows}
    if len(rows) != 486 or found != expected or any(r["n_seeds"] != "3" for r in rows):
        raise ValueError("Frozen result matrix must contain exactly 486 unique three-seed cells")
    return {(r["dataset"], r["backbone"], r["method"]): r for r in rows}


def matrix(rows, dataset, metric):
    lines = ["| Method | " + " | ".join(BACKBONES) + " |",
             "|:--|" + "|".join(":--:" for _ in BACKBONES) + "|"]
    for method in METHODS:
        cells = []
        for backbone in BACKBONES:
            row = rows[(dataset, backbone, method)]
            cells.append(f'{float(row[f"test_{metric}_mean"]):.6f} ± {float(row[f"test_{metric}_std"]):.6f}')
        lines.append("| `" + method + "` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render(rows):
    parts = [
        START,
        "## Frozen benchmark results",
        "",
        "The tables report **test AUC** and **test logloss** as mean ± standard deviation over seeds 2026/2027/2028; higher AUC and lower logloss are better. Each cell is a separately trained method–backbone–dataset combination. These are the tuned shared-z-score V2 results, not the earlier raw-feature matrix. The training protocol is identical across methods within a dataset/backbone; method-specific settings are frozen in the supplied configurations. No values are pooled across different datasets or backbones.",
        "",
    ]
    for dataset in DATASETS:
        parts.extend([
            f"### {dataset} — test AUC",
            "",
            matrix(rows, dataset, "auc"),
            "",
            "<details>",
            f"<summary>{dataset} — test logloss (all 18 methods × 9 backbones)</summary>",
            "",
            matrix(rows, dataset, "logloss"),
            "",
            "</details>",
            "",
        ])
    parts.extend([
        "The [per-seed records](reproduction/results/tuned_accuracy/per_seed.csv) contain all 1,458 individual runs, including validation metrics, best/stopping epoch and elapsed time. The [full-precision summary](reproduction/results/tuned_accuracy/summary.csv) includes both test metrics, ranks and Field deltas. See the [protocol](docs/PROTOCOL.md) and [fidelity notes](docs/BASELINE_PROVENANCE.md) before interpreting differences as a property of the original papers.",
        END,
    ])
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if README differs; do not write")
    args = parser.parse_args()
    content = README.read_text(encoding="utf-8")
    if content.count(START) != 1 or content.count(END) != 1:
        raise ValueError("README must contain exactly one start and end results marker")
    before, rest = content.split(START, 1)
    _, after = rest.split(END, 1)
    updated = before + render(load_rows()) + after
    if args.check:
        if updated != content:
            raise SystemExit("README result tables differ from frozen summary.csv")
        print("README result tables match frozen summary.csv")
    else:
        README.write_text(updated, encoding="utf-8")
        print("Updated README result tables from frozen summary.csv")


if __name__ == "__main__":
    main()
