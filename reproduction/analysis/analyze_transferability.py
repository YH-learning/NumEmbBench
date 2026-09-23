#!/usr/bin/env python3
"""Evaluate whether a numerical-embedding recommendation transfers.

This analysis never selects on the target cell. It uses mean ranks on the
remaining backbones or datasets, chooses one representation, and then records
its target rank and AUC regret relative to the oracle winner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_benchmark_artifacts import ARTIFACT_DIR, DISPLAY, load


def add_protocol(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    out = frame.copy()
    out["protocol"] = name
    return out


def select_method(training: pd.DataFrame) -> str:
    scores = training.groupby("method")["rank"].mean().sort_values()
    return str(scores.index[0])


def evaluate_backbone_holdout(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (protocol, dataset), group in frame.groupby(["protocol", "dataset"]):
        for backbone in sorted(group["backbone"].unique()):
            train = group[group["backbone"] != backbone]
            test = group[group["backbone"] == backbone]
            chosen = select_method(train)
            chosen_row = test[test["method"] == chosen].iloc[0]
            oracle_auc = float(test["test_auc_mean"].max())
            rows.append(
                {
                    "holdout_type": "backbone",
                    "protocol": protocol,
                    "dataset": dataset,
                    "held_out": backbone,
                    "selected_method": chosen,
                    "selected_display": DISPLAY[chosen],
                    "heldout_rank": float(chosen_row["rank"]),
                    "auc_regret": oracle_auc - float(chosen_row["test_auc_mean"]),
                }
            )
    return rows


def evaluate_dataset_holdout(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for protocol, group in frame.groupby("protocol"):
        for dataset in sorted(group["dataset"].unique()):
            train = group[group["dataset"] != dataset]
            test = group[group["dataset"] == dataset]
            chosen = select_method(train)
            heldout = test[test["method"] == chosen]
            rows.append(
                {
                    "holdout_type": "dataset",
                    "protocol": protocol,
                    "dataset": dataset,
                    "held_out": dataset,
                    "selected_method": chosen,
                    "selected_display": DISPLAY[chosen],
                    "heldout_rank": float(heldout["rank"].mean()),
                    "auc_regret": float(
                        test.groupby("backbone")["test_auc_mean"].max().mean()
                        - heldout["test_auc_mean"].mean()
                    ),
                }
            )
    return rows


def evaluate_cross_protocol(native: pd.DataFrame, zscore: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_name, source, target_name, target in (
        ("Native scale", native, "Shared z-score", zscore),
        ("Shared z-score", zscore, "Native scale", native),
    ):
        for (dataset, backbone), source_cell in source.groupby(["dataset", "backbone"]):
            target_cell = target[
                (target["dataset"] == dataset) & (target["backbone"] == backbone)
            ]
            chosen = str(source_cell.loc[source_cell["test_auc_mean"].idxmax(), "method"])
            chosen_target = target_cell[target_cell["method"] == chosen].iloc[0]
            rows.append(
                {
                    "source_protocol": source_name,
                    "target_protocol": target_name,
                    "dataset": dataset,
                    "backbone": backbone,
                    "selected_method": chosen,
                    "selected_display": DISPLAY[chosen],
                    "target_rank": float(chosen_target["rank"]),
                    "target_auc_regret": float(
                        target_cell["test_auc_mean"].max()
                        - chosen_target["test_auc_mean"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    native, zscore = load()
    combined = pd.concat(
        [add_protocol(native, "Native scale"), add_protocol(zscore, "Shared z-score")],
        ignore_index=True,
    )
    holdout = pd.DataFrame(
        evaluate_backbone_holdout(combined) + evaluate_dataset_holdout(combined)
    )
    cross = evaluate_cross_protocol(native, zscore)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(ARTIFACT_DIR / "recommendation_holdout.csv", index=False)
    cross.to_csv(ARTIFACT_DIR / "cross_protocol_selection.csv", index=False)

    summary = (
        holdout.groupby(["holdout_type", "protocol"])
        .agg(
            evaluations=("heldout_rank", "size"),
            mean_heldout_rank=("heldout_rank", "mean"),
            median_heldout_rank=("heldout_rank", "median"),
            top3_rate=("heldout_rank", lambda s: float((s <= 3).mean())),
            mean_auc_regret=("auc_regret", "mean"),
            max_auc_regret=("auc_regret", "max"),
        )
        .reset_index()
    )
    cross_summary = (
        cross.groupby(["source_protocol", "target_protocol"])
        .agg(
            evaluations=("target_rank", "size"),
            mean_target_rank=("target_rank", "mean"),
            top3_rate=("target_rank", lambda s: float((s <= 3).mean())),
            mean_auc_regret=("target_auc_regret", "mean"),
            max_auc_regret=("target_auc_regret", "max"),
        )
        .reset_index()
    )
    summary.to_csv(ARTIFACT_DIR / "recommendation_holdout_summary.csv", index=False)
    cross_summary.to_csv(ARTIFACT_DIR / "cross_protocol_selection_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(cross_summary.to_string(index=False))


if __name__ == "__main__":
    main()
