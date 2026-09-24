# Five-seed NumEmb benchmark

## Protocol and provenance

The 2026–2028 records are the frozen three-seed results. Seeds 2029 and 2030 were
trained with the reorganized benchmark code and the same frozen configurations.
The 972-run extension completed without failures; all 486 cells have exactly five
seeds, for 2,430 per-seed records. This directory does not overwrite the original
three-seed records. Rankings use mean test AUC within each of the 27 matched
dataset–backbone cells; standard deviations in `summary.csv` are sample SDs.

## Main result

**deer** has the best average rank (3.407),
14/27 wins, and 19/27 top-three placements.
The Friedman statistic is 157.038 (p=1.018e-24).
Paired Wilcoxon tests use the top-ranked method as the reference and Holm
correction across 17 comparisons; see `holm_posthoc.csv`.

## Cross-cell robustness

| Method | Mean rank | Wins / 27 | Top 3 / 27 | Worst rank | Mean test AUC delta vs Field |
|---|---:|---:|---:|---:|---:|
| deer | 3.407 | 14 | 19 | 16 | +0.002616 |
| ple_b | 5.296 | 4 | 14 | 16 | +0.001550 |
| dae | 7.148 | 0 | 5 | 15 | +0.001154 |
| daes_gate | 7.185 | 0 | 10 | 18 | +0.000968 |
| ple_a | 7.222 | 3 | 7 | 17 | +0.000024 |
| quantile_bucket | 7.741 | 1 | 2 | 18 | +0.001077 |
| bspline | 7.926 | 0 | 9 | 14 | +0.001467 |
| autodis | 8.037 | 0 | 3 | 17 | +0.001154 |
| plr | 8.963 | 0 | 2 | 16 | +0.000776 |
| daes_tran | 9.556 | 0 | 1 | 18 | +0.000257 |
| narydis_full | 10.630 | 0 | 1 | 18 | -0.000645 |
| field | 10.741 | 3 | 3 | 18 | +0.000000 |
| periodic | 10.889 | 2 | 3 | 17 | +0.000060 |
| linear_relu | 11.926 | 0 | 1 | 18 | -0.003341 |
| linear | 12.259 | 0 | 0 | 18 | -0.000337 |
| ple_relu | 12.481 | 0 | 1 | 17 | -0.000557 |
| narydis_lite | 13.444 | 0 | 0 | 17 | -0.004396 |
| log_squared_bucket | 16.148 | 0 | 0 | 18 | -0.013224 |

## Files

- `per_seed.csv`: all 2,430 seed-level outcomes.
- `summary.csv`: 486 five-seed mean/SD cells with ranks and Field deltas.
- `extension_per_seed.csv`: the separately validated 972-run extension.
- `method_robustness.csv`, `friedman.json`, `holm_posthoc.csv`: derived analyses.
