#!/usr/bin/env python3
"""Run one benchmark cell or a filtered matrix from frozen configurations.

This launcher never creates or rewrites datasets/configurations. A run writes
only below the explicitly selected output and checkpoint roots.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def comma_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    if not items:
        raise ValueError("Empty selection is not allowed")
    return items


def read_protocol(config_root: Path) -> tuple[dict, list[dict]]:
    protocol_path = config_root / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    manifests = []
    for path in sorted(config_root.glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        manifest["config_dir"] = str(path.parent.resolve())
        manifests.append(manifest)
    if not manifests:
        raise RuntimeError(f"No dataset manifests found in {config_root}")
    return protocol, manifests


def select_tasks(args: argparse.Namespace, protocol: dict, manifests: list[dict]) -> list[dict]:
    selected = {
        "datasets": comma_set(args.datasets),
        "backbones": comma_set(args.backbones),
        "methods": comma_set(args.methods),
        "seeds": comma_set(args.seeds),
    }
    available = {
        "datasets": {manifest["slug"] for manifest in manifests},
        "backbones": set(protocol["backbones"]),
        "methods": set(protocol["methods"]),
        "seeds": {str(seed) for manifest in manifests for seed in manifest["seeds"]},
    }
    for key, values in selected.items():
        if values is not None and values - available[key]:
            raise ValueError(f"Unknown {key}: {sorted(values - available[key])}; available={sorted(available[key])}")

    tasks = []
    for manifest in manifests:
        slug = manifest["slug"]
        if selected["datasets"] is not None and slug not in selected["datasets"]:
            continue
        prefix = f"Tuned_{slug}_"
        for experiment in manifest["experiments"]:
            if not experiment.startswith(prefix):
                raise ValueError(f"Unexpected experiment ID: {experiment}")
            suffix = experiment[len(prefix):]
            backbone = next((name for name in protocol["backbones"] if suffix.startswith(name + "_")), None)
            if backbone is None:
                raise ValueError(f"Cannot parse backbone from {experiment}")
            method = suffix[len(backbone) + 1:]
            if method not in available["methods"]:
                raise ValueError(f"Cannot parse method from {experiment}")
            if selected["backbones"] is not None and backbone not in selected["backbones"]:
                continue
            if selected["methods"] is not None and method not in selected["methods"]:
                continue
            for seed in manifest["seeds"]:
                if selected["seeds"] is not None and str(seed) not in selected["seeds"]:
                    continue
                tasks.append({
                    "dataset_id": manifest["dataset_id"],
                    "slug": slug,
                    "config_dir": manifest["config_dir"],
                    "experiment": experiment,
                    "backbone": backbone,
                    "method": method,
                    "seed": int(seed),
                    "task": f"{experiment}__seed_{seed}",
                })
    if not tasks:
        raise ValueError("Selection matches no tasks")
    # Preserve fair coverage in partial runs: interleave datasets.
    by_dataset = {slug: deque(task for task in tasks if task["slug"] == slug)
                  for slug in sorted({task["slug"] for task in tasks})}
    ordered = []
    while any(by_dataset.values()):
        for slug in by_dataset:
            if by_dataset[slug]:
                ordered.append(by_dataset[slug].popleft())
    return ordered


def valid_result(path: Path, task: dict) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    lines = path.read_text(errors="replace").splitlines()
    for line in lines:
        if f"[exp_id] {task['experiment']}," not in line:
            continue
        if f"[dataset_id] {task['dataset_id']}," not in line:
            continue
        if "[val] " not in line or ",[test] " not in line:
            continue
        validation = line.split("[val] ", 1)[1].split(",[test] ", 1)[0]
        test = line.split(",[test] ", 1)[1]
        for segment in (validation, test):
            values = {}
            for metric in ("AUC", "logloss"):
                match = re.search(rf"(?:^| - ){metric}: ([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)", segment)
                if match is None:
                    break
                values[metric] = float(match.group(1))
            if len(values) != 2 or not all(math.isfinite(value) for value in values.values()):
                break
            if not (0.0 <= values["AUC"] <= 1.0 and values["logloss"] >= 0.0):
                break
        else:
            return True
    return False


def command_for(task: dict, args: argparse.Namespace, gpu: int, result: Path, checkpoint: Path) -> list[str]:
    return [
        sys.executable, str(args.project_root / "backbone/run_expid.py"),
        "--config", task["config_dir"],
        "--expid", task["experiment"],
        "--dataset-id", task["dataset_id"],
        "--gpu", str(gpu),
        "--seed", str(task["seed"]),
        "--model-root", str(checkpoint),
        "--results-file", str(result),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "outputs/checkpoints")
    parser.add_argument("--datasets", help="Comma-separated dataset slugs: criteo,automl_a,automl_e")
    parser.add_argument("--backbones", help="Comma-separated backbone names")
    parser.add_argument("--methods", help="Comma-separated method names")
    parser.add_argument("--seeds", help="Comma-separated seeds")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs; use -1 for CPU smoke tests")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="Validate selection and show commands without writes")
    parser.add_argument("--resume", action="store_true", help="Skip only complete, matching result files")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_name) or args.run_name in {".", ".."}:
        parser.error("--run-name must be a simple name containing letters, digits, dots, underscores, or hyphens")
    args.project_root = args.project_root.resolve()
    if not (args.project_root / "backbone/run_expid.py").is_file():
        parser.error(f"Missing training entry point: {args.project_root / 'backbone/run_expid.py'}")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    try:
        gpus = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        if not gpus or len(gpus) != len(set(gpus)) or any(gpu < -1 for gpu in gpus):
            raise ValueError("GPU IDs must be unique integers >= -1")
        protocol, manifests = read_protocol(args.config_root.resolve())
        tasks = select_tasks(args, protocol, manifests)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))

    if not args.dry_run:
        for manifest in manifests:
            for filename in ("dataset_config.yaml", "model_config.yaml"):
                config = Path(manifest["config_dir"]) / filename
                if not config.is_file():
                    parser.error(f"Missing configuration: {config}")
                if "__DATASET_ROOT__" in config.read_text() or "__CHECKPOINT_ROOT__" in config.read_text():
                    parser.error("Configuration contains unresolved paths; run reproduction/tools/materialize_configs.py first")

    run_dir = args.run_root.resolve() / args.run_name
    checkpoint_root = args.checkpoint_root.resolve() / args.run_name
    if args.dry_run:
        sample = tasks[0]
        result = run_dir / sample["slug"] / "results" / f"{sample['task']}.csv"
        checkpoint = checkpoint_root / sample["slug"] / sample["experiment"] / f"seed_{sample['seed']}"
        print(json.dumps({"tasks": len(tasks), "datasets": sorted({t["slug"] for t in tasks}),
                          "backbones": sorted({t["backbone"] for t in tasks}),
                          "methods": sorted({t["method"] for t in tasks}),
                          "seeds": sorted({t["seed"] for t in tasks}),
                          "example_command": command_for(sample, args, gpus[0], result, checkpoint)}, indent=2))
        return

    pending = deque()
    skipped = 0
    for task in tasks:
        result = run_dir / task["slug"] / "results" / f"{task['task']}.csv"
        if result.exists():
            if args.resume and valid_result(result, task):
                skipped += 1
                continue
            parser.error(f"Existing result would be overwritten or is incomplete: {result}")
        pending.append(task)
    if not pending:
        print(f"COMPLETE all {skipped} selected results already valid")
        return

    for task in tasks:
        (run_dir / task["slug"] / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / task["slug"] / "results").mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    (run_dir / "task_manifest.json").write_text(json.dumps({
        "run_name": args.run_name,
        "config_root": str(args.config_root.resolve()),
        "selected_tasks": tasks,
        "gpus": gpus,
        "skipped_valid_results": skipped,
    }, indent=2) + "\n")
    print(f"MATRIX selected={len(tasks)} pending={len(pending)} skipped={skipped} gpus={gpus}", flush=True)

    active: dict[int, dict] = {}
    failures: list[dict] = []
    completed = skipped
    completed_by_dataset: Counter[str] = Counter()
    interrupted = False
    started_at = time.time()

    def stop(signum: int, _frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        print(f"SIGNAL {signum}; terminating active children", flush=True)
        for record in active.values():
            if record["process"].poll() is None:
                record["process"].terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(gpu: int, task: dict) -> None:
        log = run_dir / task["slug"] / "logs" / f"{task['task']}.log"
        result = run_dir / task["slug"] / "results" / f"{task['task']}.csv"
        checkpoint = checkpoint_root / task["slug"] / task["experiment"] / f"seed_{task['seed']}"
        stream = log.open("w")
        process = subprocess.Popen(command_for(task, args, gpu, result, checkpoint),
                                   cwd=args.project_root, stdout=stream, stderr=subprocess.STDOUT)
        active[gpu] = {"process": process, "stream": stream, "task": task,
                       "log": log, "result": result, "started": time.time()}
        print(f"START {datetime.now().isoformat(timespec='seconds')} task={task['task']} gpu={gpu} pid={process.pid}", flush=True)

    last_status = 0.0
    while pending or active:
        if not interrupted:
            for gpu in gpus:
                if gpu not in active and pending:
                    launch(gpu, pending.popleft())
        time.sleep(args.poll_seconds)
        for gpu in list(active):
            record = active[gpu]
            rc = record["process"].poll()
            if rc is None:
                continue
            record["stream"].close()
            active.pop(gpu)
            task = record["task"]
            ok = rc == 0 and valid_result(record["result"], task)
            state = "DONE" if ok else "FAILED"
            print(f"{state} {datetime.now().isoformat(timespec='seconds')} task={task['task']} gpu={gpu} rc={rc} elapsed={time.time()-record['started']:.1f}", flush=True)
            if ok:
                completed += 1
                completed_by_dataset[task["slug"]] += 1
            else:
                failures.append({**task, "return_code": rc, "log": str(record["log"]),
                                 "result": str(record["result"])})
        if time.time() - last_status >= 60:
            print(f"STATUS completed={completed}/{len(tasks)} active={len(active)} pending={len(pending)} failed={len(failures)} by_dataset={dict(completed_by_dataset)}", flush=True)
            last_status = time.time()
        if interrupted and not active:
            break

    (run_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (run_dir / "elapsed_seconds").write_text(f"{time.time()-started_at:.3f}\n")
    print(f"FINISH completed={completed}/{len(tasks)} failed={len(failures)} run_dir={run_dir}", flush=True)
    if interrupted:
        raise SystemExit(130)
    if failures or completed != len(tasks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
