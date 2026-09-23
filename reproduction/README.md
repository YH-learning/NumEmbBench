# Frozen paper-result reproduction

This directory is intentionally separate from the runnable benchmark code.
It contains released per-seed records, frozen configuration snapshots,
efficiency measurements, analysis programs, and generated paper artifacts.
No dataset archive or trained checkpoint is included.

From the repository root, two standard-library commands verify the **1,458**
per-seed records, **486** method/dataset/backbone cells, and rebuild the full
LaTeX results table:

```bash
python reproduction/analysis/verify_key_metrics.py
python reproduction/analysis/build_full_result_tables.py
```

Their outputs are in `reproduction/reproduced/`. For the remaining
figures, transferability analysis, and efficiency summaries, install the
analysis dependencies and run:

```bash
python -m pip install -r reproduction/requirements.txt
python reproduction/analysis/build_benchmark_artifacts.py
python reproduction/analysis/analyze_transferability.py
python reproduction/analysis/summarize_efficiency.py
```

Outputs are written under `reproduction/analysis_outputs/` and
`reproduction/figures/`; frozen measurement inputs under `results/`
remain unchanged. `legacy/run_full_matrix_legacy.py` is retained as a historical
launcher record. New training should use `reproduction/tools/run_benchmark.py` from the
repository root instead.

The native-scale and shared-z-score records represent **different dataset
protocols**. The tuned accuracy and efficiency records also come from
different experimental stages. Do not infer a new model-level speed/accuracy
comparison by joining them without checking their configuration and run
metadata. `../docs/BASELINE_PROVENANCE.md` documents where a paper-aligned
reimplementation or benchmark adapter was used instead of authors' exact
source code.
