#!/usr/bin/env python3
"""Run the complete Version-2 matrix with a shared dynamic GPU queue."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/tmp/CTR-NumEmb-runs"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("/tmp/CTR-NumEmb-checkpoints"))
    parser.add_argument("--gpus", default=",".join(map(str, range(16))))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    config_root = args.config_root.resolve()
    project_root = args.project_root.resolve()
    run_dir = args.run_root.resolve() / args.run_name
    checkpoint_root = args.checkpoint_root.resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    dataset_manifests = []
    for manifest_path in sorted(config_root.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        manifest["config_dir"] = str(manifest_path.parent.resolve())
        dataset_manifests.append(manifest)
    if len(dataset_manifests) != 3:
        raise RuntimeError(f"Expected 3 dataset manifests, found {len(dataset_manifests)}")

    per_dataset = {}
    expected_total = 0
    for manifest in dataset_manifests:
        items = []
        slug = manifest["slug"]
        for experiment in manifest["experiments"]:
            for seed in manifest["seeds"]:
                task = f"{experiment}__seed_{seed}"
                result = run_dir / slug / "results" / f"{task}.csv"
                if not nonempty(result):
                    items.append({
                        "dataset_id": manifest["dataset_id"],
                        "slug": slug,
                        "config_dir": manifest["config_dir"],
                        "experiment": experiment,
                        "seed": seed,
                        "task": task,
                    })
                expected_total += 1
        per_dataset[slug] = deque(items)
        (run_dir / slug / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / slug / "results").mkdir(parents=True, exist_ok=True)

    # Interleave datasets so early partial results cover all three corpora.
    pending = deque()
    while any(per_dataset.values()):
        for slug in sorted(per_dataset):
            if per_dataset[slug]:
                pending.append(per_dataset[slug].popleft())
    existing = expected_total - len(pending)
    snapshot = {
        "run_name": args.run_name,
        "version": 2,
        "config_root": str(config_root),
        "checkpoint_root": str(checkpoint_root),
        "gpus": gpus,
        "expected_tasks": expected_total,
        "existing_tasks": existing,
        "datasets": dataset_manifests,
    }
    (run_dir / "task_manifest.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(f"MATRIX expected={expected_total} pending={len(pending)} existing={existing} gpus={gpus}", flush=True)

    active = {}
    failures = []
    done = existing
    completed_by_dataset = Counter()
    started_at = time.time()
    interrupted = False

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = True
        print(f"SIGNAL signum={signum}; terminating active children", flush=True)
        for record in active.values():
            if record["process"].poll() is None:
                record["process"].terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def launch(gpu: int, item: dict) -> None:
        log_path = run_dir / item["slug"] / "logs" / f'{item["task"]}.log'
        result_path = run_dir / item["slug"] / "results" / f'{item["task"]}.csv'
        checkpoint = checkpoint_root / item["slug"] / item["experiment"] / f'seed_{item["seed"]}'
        command = [
            sys.executable, str(project_root / "backbone/run_expid.py"),
            "--config", item["config_dir"], "--expid", item["experiment"],
            "--dataset-id", item["dataset_id"], "--gpu", str(gpu),
            "--seed", str(item["seed"]), "--model-root", str(checkpoint),
            "--results-file", str(result_path),
        ]
        stream = log_path.open("w")
        process = subprocess.Popen(command, cwd=project_root, stdout=stream, stderr=subprocess.STDOUT)
        active[gpu] = {"process": process, "stream": stream, "item": item, "log": log_path, "result": result_path, "started": time.time()}
        print(f"START time={datetime.now().isoformat(timespec='seconds')} dataset={item['slug']} task={item['task']} gpu={gpu} pid={process.pid}", flush=True)

    last_status = 0.0
    while pending or active:
        if not interrupted:
            for gpu in gpus:
                if gpu not in active and pending:
                    launch(gpu, pending.popleft())
        time.sleep(args.poll_seconds)
        for gpu in list(active):
            record = active[gpu]
            return_code = record["process"].poll()
            if return_code is None:
                continue
            record["stream"].close()
            active.pop(gpu)
            item = record["item"]
            ok = return_code == 0 and nonempty(record["result"])
            state = "DONE" if ok else "FAILED"
            elapsed = time.time() - record["started"]
            print(f"{state} time={datetime.now().isoformat(timespec='seconds')} dataset={item['slug']} task={item['task']} gpu={gpu} rc={return_code} elapsed={elapsed:.1f}", flush=True)
            if ok:
                done += 1
                completed_by_dataset[item["slug"]] += 1
            else:
                failures.append({**item, "return_code": return_code, "log": str(record["log"]), "result": str(record["result"])})
        if time.time() - last_status >= 60:
            print(f"STATUS time={datetime.now().isoformat(timespec='seconds')} completed={done}/{expected_total} active={len(active)} pending={len(pending)} failed={len(failures)} by_dataset={dict(completed_by_dataset)}", flush=True)
            last_status = time.time()
        if interrupted and not active:
            break

    (run_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    (run_dir / "elapsed_seconds").write_text(f"{time.time() - started_at:.3f}\n")
    print(f"FINISH completed={done}/{expected_total} failed={len(failures)} run_dir={run_dir}", flush=True)
    if interrupted:
        raise SystemExit(130)
    if failures or done != expected_total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
