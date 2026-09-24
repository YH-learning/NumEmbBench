# Frozen experiment protocol

The primary tuned matrix is the Cartesian product of three datasets
(Criteo, AutoML-A, AutoML-E), nine backbones, 18 method configurations, and
three seeds (2026, 2027, 2028): **1,458 training tasks** or **486 three-seed
cells**. The authoritative frozen protocol is
`reproduction/results/tuned_accuracy/config_snapshot/protocol.json`;
the YAML files in the adjacent dataset folders specify backbone- and
method-specific details.

The completed five-seed extension adds seeds 2029 and 2030 using the same
frozen configurations, for 972 additional tasks and 2,430 total runs. Its
versioned outputs are under `reproduction/results/five_seed_20260924/`;
the original three-seed matrix remains intact under
`reproduction/results/tuned_accuracy/`. The five-seed ranking is derived
from all five seed outcomes, not from a mixture of three- and five-seed cells.

| Fixed setting | Value |
|---|---|
| Numerical embedding output dimension | 16 |
| Batch size | 4096 |
| Learning rate | 0.001 |
| Maximum epochs | 100 |
| Early-stopping monitor | validation AUC, maximize |
| Early-stopping patience | 2 |
| Data format | Parquet |
| Loading | non-streaming |
| Training shuffle | global shuffle enabled |
| Evaluation | validation for model selection; test after training |

The shared-z-score datasets are used for the tuned accuracy matrix. Published
native-scale results are separately archived under `reproduction/results`.
Do not mix their outcomes as if they were the same protocol. The frozen
method-specific hyperparameters were selected without test metrics, but the
result CSVs record **test** AUC/logloss. Report the protocol and seed count
alongside every comparison.

The one/batch launcher selects tasks from these manifests, not by inventing
configurations. A filtered run changes only which cells are executed. It does
not change hyperparameters, dataset processing, or the early-stopping rule.
Its `--dry-run` mode is useful for auditing a proposed selection. One GPU ID
is assigned to each concurrent process; it is **not** data-parallel training
of a single model. Use `--gpus -1` only for small CPU smoke tests.

Outputs are deliberately isolated by run name and task. A training result is
accepted only if the process exits successfully and its result file has the
expected experiment/dataset IDs and validation and test AUC/logloss. The
launcher records failures but never automatically retrains them. `--resume`
skips complete matching results and rejects incomplete existing files;
inspect logs and select a new run name before retrying a failed protocol.
Use `reproduction/tools/summarize_run.py <run_dir> --require-complete` to create a fresh
run's `per_seed.csv` and `summary.csv`; the summary reports the number of
seeds in each cell and uses sample standard deviation for multi-seed cells.
It never joins new runs with the frozen paper measurements implicitly.

If any dataset, framework source, method implementation, optimizer, or tuning
choice changes, call the result a **new experiment version**. Do not append
its cells to the frozen paper matrix without explicitly distinguishing it.
