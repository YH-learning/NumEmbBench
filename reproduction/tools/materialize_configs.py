#!/usr/bin/env python3
"""Relocate the frozen benchmark configurations without changing hyperparameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "reproduction/results/tuned_accuracy/config_snapshot"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--seeds", help="Comma-separated run seeds; default preserves the frozen three-seed protocol")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refuse to overwrite configuration directory: {output_root}")
    protocol = json.loads((SNAPSHOT / "protocol.json").read_text())
    if args.seeds is not None:
        try:
            seeds = [int(value.strip()) for value in args.seeds.split(",")]
        except ValueError as error:
            parser.error(f"Invalid --seeds: {error}")
        if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
            parser.error("--seeds must contain unique nonnegative integers")
        protocol["seeds"] = seeds
        protocol["expected_tasks"] = len(protocol["datasets"]) * len(protocol["backbones"]) * len(protocol["methods"]) * len(seeds)
        protocol["name"] = f"{protocol['name']}_seeds_{'_'.join(map(str, seeds))}"
    output_root.mkdir(parents=True)
    for slug in ("criteo", "automl_a", "automl_e"):
        source = SNAPSHOT / slug
        target = output_root / slug
        target.mkdir()
        manifest = json.loads((source / "manifest.json").read_text())
        if args.seeds is not None:
            manifest["seeds"] = seeds
        dataset_id = manifest["dataset_id"]
        dataset_dir = dataset_root / dataset_id
        for name in ("train", "valid", "test"):
            path = dataset_dir / f"{name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
        vocab = dataset_dir / "log_squared_bucket_vocab.json"
        if not vocab.is_file():
            raise FileNotFoundError(vocab)
        data_text = (source / "dataset_config.yaml").read_text()
        model_text = (source / "model_config.yaml").read_text()
        old_root = "__DATASET_ROOT__"
        old_checkpoint = "__CHECKPOINT_ROOT__/" + slug
        if old_root not in data_text or old_root not in model_text:
            raise ValueError(f"Frozen dataset path missing in {source}")
        if old_checkpoint not in model_text:
            raise ValueError(f"Frozen checkpoint path missing in {source}")
        data_text = data_text.replace(old_root, str(dataset_root))
        model_text = model_text.replace(old_root, str(dataset_root))
        model_text = model_text.replace(old_checkpoint, str(args.checkpoint_root.resolve() / slug))
        (target / "dataset_config.yaml").write_text(data_text)
        (target / "model_config.yaml").write_text(model_text)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output_root / "frozen_method_configs.json").write_text(
        (SNAPSHOT / "frozen_method_configs.json").read_text()
    )
    print(f"READY {output_root} expected_tasks={protocol['expected_tasks']}")


if __name__ == "__main__":
    main()
