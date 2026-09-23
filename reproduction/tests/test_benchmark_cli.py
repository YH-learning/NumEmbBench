"""Fast, dependency-free tests for the public benchmark entry points."""

import subprocess
import sys
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from reproduction.tools.run_benchmark import read_protocol, select_tasks, valid_result


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "reproduction/results/tuned_accuracy/config_snapshot"


class BenchmarkEntrypointTests(unittest.TestCase):
    def test_launcher_runs_and_resumes_isolated_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            (project / "backbone").mkdir()
            (project / "backbone/run_expid.py").write_text(
                "import argparse\n"
                "from pathlib import Path\n"
                "p=argparse.ArgumentParser()\n"
                "for x in ('config','expid','dataset-id','gpu','seed','model-root','results-file'): p.add_argument('--'+x)\n"
                "a=p.parse_args()\n"
                "Path(a.results_file).write_text('[exp_id] '+a.expid+',[dataset_id] '+a.dataset_id+',[val] AUC: 0.800000 - logloss: 0.400000,[test] AUC: 0.810000 - logloss: 0.390000\\n')\n"
            )
            configs = root / "configs"
            dataset = configs / "criteo"
            dataset.mkdir(parents=True)
            (configs / "protocol.json").write_text(json.dumps({"backbones": ["DCNv2"], "methods": ["field"]}))
            (dataset / "manifest.json").write_text(json.dumps({
                "slug": "criteo", "dataset_id": "Fixture", "experiments": ["Tuned_criteo_DCNv2_field"], "seeds": [2026]
            }))
            for name in ("dataset_config.yaml", "model_config.yaml"):
                (dataset / name).write_text("resolved: /tmp/fixture\n")
            command = [sys.executable, str(ROOT / "reproduction/tools/run_benchmark.py"),
                       "--config-root", str(configs), "--project-root", str(project),
                       "--run-root", str(root / "runs"), "--checkpoint-root", str(root / "weights"),
                       "--run-name", "fixture", "--gpus", "-1", "--poll-seconds", "0.01"]
            first = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            self.assertIn("DONE", first.stdout)
            result = root / "runs/fixture/criteo/results/Tuned_criteo_DCNv2_field__seed_2026.csv"
            self.assertTrue(result.is_file())
            summarized = subprocess.run([sys.executable, str(ROOT / "reproduction/tools/summarize_run.py"),
                                         str(root / "runs/fixture"), "--require-complete"],
                                        text=True, capture_output=True)
            self.assertEqual(summarized.returncode, 0, summarized.stdout + summarized.stderr)
            rows = (root / "runs/fixture/summary.csv").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertIn("0.81", rows[1])
            resumed = subprocess.run(command + ["--resume"], text=True, capture_output=True)
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            self.assertIn("already valid", resumed.stdout)

    def test_frozen_matrix_cardinality_and_selection(self):
        protocol, manifests = read_protocol(SNAPSHOT)
        all_args = SimpleNamespace(datasets=None, backbones=None, methods=None, seeds=None)
        self.assertEqual(len(select_tasks(all_args, protocol, manifests)), 1458)
        one = SimpleNamespace(datasets="criteo", backbones="DCNv2", methods="dae", seeds="2026")
        tasks = select_tasks(one, protocol, manifests)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["experiment"], "Tuned_criteo_DCNv2_dae")

    def test_result_validation_rejects_incomplete_rows(self):
        task = {"experiment": "Tuned_criteo_DCNv2_dae", "dataset_id": "Criteo_x4_NumEmbZScoreV2"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.csv"
            output.write_text("partial result\n")
            self.assertFalse(valid_result(output, task))
            output.write_text("[exp_id] Tuned_criteo_DCNv2_dae,[dataset_id] Criteo_x4_NumEmbZScoreV2,[val] AUC: 0.8 - logloss: 0.4,[test] AUC: 0.8 - logloss: 0.4\n")
            self.assertTrue(valid_result(output, task))
            output.write_text("[exp_id] Tuned_criteo_DCNv2_dae,[dataset_id] Criteo_x4_NumEmbZScoreV2,[val] AUC: 0.8 - logloss: 0.4,[test] AUC: nan - logloss: 0.4\n")
            self.assertFalse(valid_result(output, task))
            output.write_text("[exp_id] Tuned_criteo_DCNv2_dae,[dataset_id] Criteo_x4_NumEmbZScoreV2,[val] AUC: 0.8 - logloss: 0.4,[test] AUC: 0.8\n")
            self.assertFalse(valid_result(output, task))

    def test_materialized_configs_contain_no_placeholders(self):
        protocol, manifests = read_protocol(SNAPSHOT)
        self.assertEqual(protocol["expected_tasks"], 1458)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            data = temporary / "datasets"
            for manifest in manifests:
                folder = data / manifest["dataset_id"]
                folder.mkdir(parents=True)
                for filename in ("train.parquet", "valid.parquet", "test.parquet", "log_squared_bucket_vocab.json"):
                    (folder / filename).write_bytes(b"fixture")
            configs = temporary / "configs"
            command = [sys.executable, str(ROOT / "reproduction/tools/materialize_configs.py"),
                       "--dataset-root", str(data), "--output-root", str(configs),
                       "--checkpoint-root", str(temporary / "checkpoints")]
            subprocess.run(command, check=True, capture_output=True, text=True)
            for manifest in manifests:
                folder = configs / manifest["slug"]
                for filename in ("dataset_config.yaml", "model_config.yaml"):
                    content = (folder / filename).read_text()
                    self.assertNotIn("__DATASET_ROOT__", content)
                    self.assertNotIn("__CHECKPOINT_ROOT__", content)
                    self.assertIn(str(data), content)

    def test_five_seed_extension_preserves_frozen_configurations(self):
        _, manifests = read_protocol(SNAPSHOT)
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            data = temporary / "datasets"
            for manifest in manifests:
                folder = data / manifest["dataset_id"]
                folder.mkdir(parents=True)
                for filename in ("train.parquet", "valid.parquet", "test.parquet", "log_squared_bucket_vocab.json"):
                    (folder / filename).write_bytes(b"fixture")
            configs = temporary / "five_seed_configs"
            command = [sys.executable, str(ROOT / "reproduction/tools/materialize_configs.py"),
                       "--dataset-root", str(data), "--output-root", str(configs),
                       "--checkpoint-root", str(temporary / "checkpoints"),
                       "--seeds", "2026,2027,2028,2029,2030"]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            protocol, expanded = read_protocol(configs)
            self.assertEqual(protocol["expected_tasks"], 2430)
            self.assertEqual(len(select_tasks(SimpleNamespace(datasets=None, backbones=None,
                                                              methods=None, seeds="2029,2030"),
                                              protocol, expanded)), 972)
            for manifest in expanded:
                self.assertEqual(manifest["seeds"], [2026, 2027, 2028, 2029, 2030])
                self.assertEqual(
                    (configs / manifest["slug"] / "model_config.yaml").read_text(),
                    (SNAPSHOT / manifest["slug"] / "model_config.yaml").read_text()
                    .replace("__DATASET_ROOT__", str(data.resolve()))
                    .replace("__CHECKPOINT_ROOT__/" + manifest["slug"],
                             str((temporary / "checkpoints" / manifest["slug"]).resolve()))
                )


if __name__ == "__main__":
    unittest.main()
