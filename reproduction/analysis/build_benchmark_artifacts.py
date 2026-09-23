#!/usr/bin/env python3
"""Build benchmark-only artifacts from the frozen native and z-score matrices.

ScalarLens/Fast is intentionally excluded so that these outputs describe the
benchmark finding rather than the method paper.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PAPER_DIR = SCRIPT_PATH.parents[1]
LOCAL_RESULTS = PAPER_DIR / "results"
if LOCAL_RESULTS.is_dir():
    # Anonymous artifact layout: all required records are self-contained.
    ROOT = PAPER_DIR
    ARTIFACT_DIR = PAPER_DIR / "analysis_outputs"
    FIGURE_DIR = PAPER_DIR / "figures"
    NATIVE_PATH = LOCAL_RESULTS / "native_summary_with_method_row.csv"
    ZSCORE_PATH = LOCAL_RESULTS / "zscore_summary_with_method_row.csv"
else:
    # Development repository layout.
    ROOT = SCRIPT_PATH.parents[3]
    ARTIFACT_DIR = PAPER_DIR / "artifacts"
    FIGURE_DIR = PAPER_DIR / "figures"
    NATIVE_PATH = ROOT / "reports/paper_main_matrix_20260905/summary_513.csv"
    ZSCORE_PATH = ROOT / "reports/numemb_benchmark_version_2_zscore_20260911/summary.csv"

DATASET_MAP = {
    "AutoML_A": "AutoML-A",
    "AutoML_E": "AutoML-E",
    "Criteo_x4_NumEmbRaw": "Criteo",
}

DISPLAY = {
    "field": "Field",
    "linear": "Linear",
    "linear_relu": "Linear-ReLU",
    "quantile_bucket": "Quantile",
    "log_squared_bucket": "LogBucket",
    "autodis": "AutoDis",
    "ple_a": "PLE-A",
    "ple_b": "PLE-B",
    "ple_relu": "PLE-ReLU",
    "periodic": "Periodic",
    "plr": "PLR",
    "narydis_lite": "NaryDis-Lite",
    "bspline": "B-spline",
    "narydis_full": "NaryDis",
    "deer": "DEER",
    "dae": "DAE",
    "daes_gate": "DAES-Gate",
    "daes_tran": "DAES-Tran",
}


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    native = pd.read_csv(NATIVE_PATH)
    zscore = pd.read_csv(ZSCORE_PATH)
    native["dataset"] = native["dataset"].replace(DATASET_MAP)
    native = native[native["method"] != "Fast"].copy()
    zscore = zscore[zscore["method"] != "Fast"].copy()
    expected = set(DISPLAY)
    assert set(native["method"]) == expected
    assert set(zscore["method"]) == expected
    for frame in (native, zscore):
        frame["rank"] = frame.groupby(["dataset", "backbone"])[
            "test_auc_mean"
        ].rank(method="min", ascending=False)
    return native, zscore


def aggregate(frame: pd.DataFrame, protocol: str, gain_col: str) -> pd.DataFrame:
    out = (
        frame.groupby("method")
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            worst_rank=("rank", "max"),
            wins=("rank", lambda s: int((s == 1).sum())),
            top3=("rank", lambda s: int((s <= 3).sum())),
            failed_cells=("test_auc_mean", lambda s: int((s < 0.55).sum())),
            mean_gain_vs_field=(gain_col, "mean"),
        )
        .reset_index()
    )
    out.insert(0, "protocol", protocol)
    out["display_name"] = out["method"].map(DISPLAY)
    return out.sort_values(["mean_rank", "method"])


def corr(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(x.to_numpy(), y.to_numpy())[0, 1])


def cell_comparison(native: pd.DataFrame, zscore: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (dataset, backbone), ncell in native.groupby(["dataset", "backbone"]):
        zcell = zscore[(zscore["dataset"] == dataset) & (zscore["backbone"] == backbone)]
        joined = ncell[["method", "rank", "test_auc_mean"]].merge(
            zcell[["method", "rank", "test_auc_mean"]],
            on="method",
            suffixes=("_native", "_zscore"),
            validate="one_to_one",
        )
        nbest = joined.loc[joined["test_auc_mean_native"].idxmax(), "method"]
        zbest = joined.loc[joined["test_auc_mean_zscore"].idxmax(), "method"]
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "native_winner": nbest,
                "zscore_winner": zbest,
                "winner_agrees": bool(nbest == zbest),
                "rank_correlation": corr(joined["rank_native"], joined["rank_zscore"]),
                "native_failed_cells": int((joined["test_auc_mean_native"] < 0.55).sum()),
                "zscore_failed_cells": int((joined["test_auc_mean_zscore"] < 0.55).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "backbone"])


def metric_consistency(frame: pd.DataFrame, protocol: str) -> pd.DataFrame:
    """Audit whether AUC and logloss induce the same conclusions."""
    rows: list[dict[str, object]] = []
    for (dataset, backbone), cell in frame.groupby(["dataset", "backbone"]):
        auc_rank = cell["test_auc_mean"].rank(method="average", ascending=False)
        ll_rank = cell["test_logloss_mean"].rank(method="average", ascending=True)
        auc_winner = cell.loc[cell["test_auc_mean"].idxmax(), "method"]
        ll_winner = cell.loc[cell["test_logloss_mean"].idxmin(), "method"]
        rows.append(
            {
                "protocol": protocol,
                "dataset": dataset,
                "backbone": backbone,
                "auc_winner": auc_winner,
                "logloss_winner": ll_winner,
                "winner_agrees": bool(auc_winner == ll_winner),
                "auc_logloss_rank_correlation": float(auc_rank.corr(ll_rank, method="spearman")),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "backbone"])


def plot_rank_shift(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    pivot = summary.pivot(index="method", columns="protocol", values="mean_rank")
    fig, ax = plt.subplots(figsize=(4.6, 4.25))
    highlighted = {
        "deer": (6, -6),
        "ple_a": (-34, -4),
        "daes_gate": (6, -11),
        "bspline": (6, 7),
        "quantile_bucket": (6, 7),
        "ple_b": (6, -10),
        "narydis_full": (6, 5),
        "field": (6, -9),
        "log_squared_bucket": (-55, -2),
    }
    for method, row in pivot.iterrows():
        x = row["Native scale"]
        y = row["Shared z-score"]
        key = method == "deer"
        color = "#246B73" if key else "#A8B2BC"
        ax.scatter(x, y, s=46 if key else 27, color=color, edgecolor="white",
                   linewidth=0.55, zorder=3 if key else 2)
        if method in highlighted:
            dx, dy = highlighted[method]
            ax.annotate(DISPLAY[method], (x, y), xytext=(dx, dy),
                        textcoords="offset points", fontsize=7.2,
                        color="#172126", ha="left", va="center",
                        arrowprops={"arrowstyle": "-", "color": "#9AA6AF",
                                    "lw": 0.55, "shrinkA": 1.5, "shrinkB": 2.0})

    ax.plot([1, 18], [1, 18], color="#CBD2D8", lw=1.0, ls="--", zorder=0)
    ax.text(12.0, 11.35, "unchanged mean rank", fontsize=6.8, color="#7A858E",
            rotation=37, ha="center")
    ax.set_xlim(1, 18)
    ax.set_ylim(18, 1)
    ax.set_xlabel("Mean rank on native scales", fontsize=8.5)
    ax.set_ylabel("Mean rank with shared z-score", fontsize=8.5)
    ax.grid(color="#E5E8EB", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=7.5, length=2.5)
    fig.tight_layout(pad=0.6)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "protocol_rank_shift.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "protocol_rank_shift.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_cell_stability(cells: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    """Show where protocol changes are mild versus conclusion-changing."""
    backbone_order = ["DNN", "DeepFM", "WideDeep", "NFM", "PNN", "DCN", "DCNv2", "AutoInt", "FiBiNET"]
    dataset_order = ["AutoML-A", "AutoML-E", "Criteo"]
    corr_grid = cells.pivot(index="dataset", columns="backbone", values="rank_correlation").loc[
        dataset_order, backbone_order
    ]
    agree_grid = cells.pivot(index="dataset", columns="backbone", values="winner_agrees").loc[
        dataset_order, backbone_order
    ]

    fig, ax = plt.subplots(figsize=(7.0, 2.15))
    image = ax.imshow(corr_grid.to_numpy(), cmap="RdBu", vmin=-0.45, vmax=0.65, aspect="auto")
    for row in range(len(dataset_order)):
        for col in range(len(backbone_order)):
            rho = corr_grid.iloc[row, col]
            marker = "=" if agree_grid.iloc[row, col] else "\u2260"
            color = "white" if abs(rho) > 0.42 else "#152027"
            ax.text(col, row - 0.09, f"{rho:.2f}", ha="center", va="center", fontsize=7.0, color=color)
            ax.text(col, row + 0.26, marker, ha="center", va="center", fontsize=7.4,
                    fontweight="bold", color=color)
    ax.set_xticks(range(len(backbone_order)), backbone_order, rotation=34, ha="right", fontsize=7.2)
    ax.set_yticks(range(len(dataset_order)), dataset_order, fontsize=7.6)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.018)
    cbar.set_label("Spearman rank correlation", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.8, length=2)
    ax.set_xlabel(r"Cell label: rank correlation; $=$ same winner, $\ne$ changed winner", fontsize=7.4)
    fig.tight_layout(pad=0.45)
    fig.savefig(FIGURE_DIR / "cell_protocol_stability.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "cell_protocol_stability.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    native, zscore = load()
    native_summary = aggregate(native, "Native scale", "test_auc_gain_vs_field")
    zscore_summary = aggregate(zscore, "Shared z-score", "delta_test_auc_vs_field")
    summary = pd.concat([native_summary, zscore_summary], ignore_index=True)
    cells = cell_comparison(native, zscore)
    metrics = pd.concat(
        [metric_consistency(native, "Native scale"), metric_consistency(zscore, "Shared z-score")],
        ignore_index=True,
    )

    summary.to_csv(ARTIFACT_DIR / "benchmark_summary.csv", index=False)
    cells.to_csv(ARTIFACT_DIR / "cell_protocol_comparison.csv", index=False)
    metrics.to_csv(ARTIFACT_DIR / "metric_consistency.csv", index=False)

    native_failures = int((native["test_auc_mean"] < 0.55).sum())
    zscore_failures = int((zscore["test_auc_mean"] < 0.55).sum())
    winner_agreement = int(cells["winner_agrees"].sum())
    rng = np.random.default_rng(2027)
    boot_change = []
    boot_corr = []
    for _ in range(10_000):
        sample = cells.iloc[rng.integers(0, len(cells), len(cells))]
        boot_change.append(float(1.0 - sample["winner_agrees"].mean()))
        boot_corr.append(float(sample["rank_correlation"].mean()))
    metrics = {
        "methods": len(DISPLAY),
        "datasets": int(native["dataset"].nunique()),
        "backbones": int(native["backbone"].nunique()),
        "cells": len(cells),
        "seeds": 3,
        "training_runs": len(DISPLAY) * len(cells) * 3 * 2,
        "native_failed_cells_below_0_55_auc": native_failures,
        "zscore_failed_cells_below_0_55_auc": zscore_failures,
        "winner_agreement_cells": winner_agreement,
        "winner_change_cells": len(cells) - winner_agreement,
        "winner_change_fraction": float(1.0 - cells["winner_agrees"].mean()),
        "winner_change_fraction_bootstrap_95ci": [
            float(np.quantile(boot_change, 0.025)),
            float(np.quantile(boot_change, 0.975)),
        ],
        "mean_rank_correlation": float(cells["rank_correlation"].mean()),
        "mean_rank_correlation_bootstrap_95ci": [
            float(np.quantile(boot_corr, 0.025)),
            float(np.quantile(boot_corr, 0.975)),
        ],
        "median_rank_correlation": float(cells["rank_correlation"].median()),
        "minimum_rank_correlation": float(cells["rank_correlation"].min()),
        "maximum_rank_correlation": float(cells["rank_correlation"].max()),
        "native_auc_logloss_mean_rank_correlation": float(
            metrics.loc[metrics["protocol"] == "Native scale", "auc_logloss_rank_correlation"].mean()
        ),
        "native_auc_logloss_winner_agreement_cells": int(
            metrics.loc[metrics["protocol"] == "Native scale", "winner_agrees"].sum()
        ),
        "zscore_auc_logloss_mean_rank_correlation": float(
            metrics.loc[metrics["protocol"] == "Shared z-score", "auc_logloss_rank_correlation"].mean()
        ),
        "zscore_auc_logloss_winner_agreement_cells": int(
            metrics.loc[metrics["protocol"] == "Shared z-score", "winner_agrees"].sum()
        ),
    }
    (ARTIFACT_DIR / "headline_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    plot_rank_shift(summary)
    plot_cell_stability(cells)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
