# Datasets

This directory contains only converters and validators, not dataset files.
See [`../docs/DATASETS.md`](../docs/DATASETS.md) for the exact AutoML split,
Criteo reference-data prerequisite, and train-only z-score protocol.

| Script | Output |
|---|---|
| `prepare_automl_lifelong.py` | Raw-numerical AutoML-A/E FuxiCTR Parquet splits |
| `prepare_criteo_x4_v220_semantics.py` | Criteo categorical reference from aligned processed input |
| `prepare_criteo_x4_numemb_raw.py` | Criteo raw-numerical FuxiCTR Parquet splits |
| `prepare_numemb_version2_datasets.py` | Shared train-z-score dataset versions |
| `validate_numemb_version2_datasets.py` | Invariant checks for the three V2 datasets |

Use external dataset roots; do not commit converted Parquet files to Git.
