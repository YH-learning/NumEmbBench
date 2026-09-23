# Dataset preparation and invariants

The training launcher does not download or rewrite datasets. Prepare them
first, validate them, and then materialize the YAML paths. Keep dataset files
outside the Git repository. The expected layout for each FuxiCTR-ready source
is `feature_map.json` plus `train.parquet`, `valid.parquet`, `test.parquet`.
All categorical vocabularies and numerical statistics must be learned from
**train only**; validation/test labels must never affect preprocessing.

## AutoML-A and AutoML-E

Extract `AutoML3_input_data_1.zip` into a directory containing the challenge
task folders. `dataset/prepare_automl_lifelong.py` converts the labeled
chronological batches with this fixed split:

| FuxiCTR split | Challenge batches |
|---|---|
| train | train1 + test1 + test2 |
| valid | test3 |
| test | test4 |

Unlabeled test5–test9 batches are excluded. Only Numerical and Categorical
fields are kept; Time and Multi-value fields are excluded. The resulting
field counts are A: 23 numerical + 51 categorical, E: 6 numerical + 25
categorical. The converter's input is the **extracted** archive, not the ZIP
file itself.

```bash
python dataset/prepare_automl_lifelong.py \
  --source-root /data/AutoML3_input_data_1 \
  --output-root /data/AutoML_FuxiCTR \
  --tasks A,E
```

It writes `/data/AutoML_FuxiCTR/AutoML_A` and `AutoML_E`. Inspect the source
task mapping and the generated manifest before using a differently packaged
AutoML archive.

## Criteo_x4_NumEmbRaw

The Criteo converter combines untouched numerical I1–I13 values from the
original split CSVs with the validated, low-frequency-filtered, non-hashed
categorical C1–C26 IDs and labels from a reference Parquet dataset. Alignment
is by **row order**, so the raw CSVs and reference Parquet must have exactly
matching split membership and ordering. The converter checks this rather than
silently merging inconsistent sources.

```bash
python dataset/prepare_criteo_x4_numemb_raw.py \
  --source-root /data/Criteo_x4 \
  --reference-root /data/Criteo_x4/Criteo_x4_001_v220_semantics \
  --output-root /data/Criteo_x4/Criteo_x4_NumEmbRaw
```

`dataset/prepare_criteo_x4_v220_semantics.py` documents the categorical
filtering semantics and can rebuild the reference from an *already aligned*
FuxiCTR-processed Criteo source. It is not a standalone original-download
converter. The validated categorical reference is an **external prerequisite**
for this release; do not treat Criteo raw preparation as one-command complete
from the public download. No Criteo records or category vocabularies are
redistributed here.

## Shared z-score version used by the tuned matrix

`dataset/prepare_numemb_version2_datasets.py` transforms each numerical
column with train-only `numeric_mean` and `numeric_std` in the source feature
map, preserving categorical IDs, labels, split membership, and row order. It
also creates the train-derived vocabulary used by `log_squared_bucket`.

```bash
python dataset/prepare_numemb_version2_datasets.py \
  --criteo-source /data/Criteo_x4/Criteo_x4_NumEmbRaw \
  --automl-a-source /data/AutoML_FuxiCTR/AutoML_A \
  --automl-e-source /data/AutoML_FuxiCTR/AutoML_E \
  --output-root /data/numemb_v2/datasets
```

Then run `dataset/validate_numemb_version2_datasets.py` with the same three
source paths and `--dataset-root /data/numemb_v2/datasets`. It checks row
counts, schemas, sampled transformed values, categorical/label invariants,
and bucket-vocabulary structure. Preserve its `validation.json` and the
converter's `dataset_manifest.json` with a new training run. To claim exact
reproduction, additionally compare source checksums and the frozen protocol.

No preparation script overwrites an existing raw output directory by default.
Use a fresh destination when changing source data or preprocessing options.
