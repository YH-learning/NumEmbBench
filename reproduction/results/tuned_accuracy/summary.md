# Tuned Numerical Embedding Benchmark: Final Stage 3

## Protocol and completion

All 1,458 tasks completed successfully: 3 datasets × 9 backbones × 18 methods × 3 seeds. Method configurations were frozen using validation results from Stages 1–2; Stage 3 performs no further tuning. All methods share train-only z-score inputs, embedding dimension 16, batch size 4096, Adam at 0.001, AUC monitoring, and patience 2.

## Main result

The best cross-cell average-rank method is **deer** (average rank 3.185; 12/27 wins; 19/27 top-three placements).
The Friedman omnibus test rejects equal performance across methods (χ²=155.304, p=2.231e-24). Post-hoc comparisons use deer as the predeclared reference selected by average rank and apply Holm correction across all 17 paired tests.

## Cross-cell robustness

| method | average_rank | wins | top3_count | worst_rank | mean_delta_test_auc_vs_field |
|---|---|---|---|---|---|
| deer | 3.185 | 12 | 19 | 14.0 | +0.002683 |
| ple_b | 6.593 | 3 | 11 | 17.0 | +0.001310 |
| ple_a | 6.889 | 2 | 8 | 17.0 | +0.000044 |
| quantile_bucket | 7.000 | 1 | 3 | 17.0 | +0.001314 |
| daes_gate | 7.222 | 1 | 9 | 18.0 | +0.000984 |
| autodis | 7.296 | 0 | 4 | 16.0 | +0.001275 |
| dae | 7.407 | 0 | 4 | 14.0 | +0.001043 |
| plr | 8.704 | 0 | 6 | 17.0 | +0.000823 |
| bspline | 8.778 | 4 | 5 | 17.0 | +0.001493 |
| daes_tran | 9.963 | 0 | 2 | 18.0 | +0.000068 |

## Top methods by dataset

| dataset | method | average_rank | wins | mean_test_auc | mean_delta_vs_field |
|---|---|---|---|---|---|
| AutoML-A | bspline | 2.444 | 4 | 0.707625 | +0.004609 |
| AutoML-A | deer | 2.889 | 3 | 0.707893 | +0.004876 |
| AutoML-A | plr | 6.333 | 0 | 0.705070 | +0.002053 |
| AutoML-A | ple_b | 6.667 | 2 | 0.705295 | +0.002278 |
| AutoML-A | autodis | 6.667 | 0 | 0.705131 | +0.002114 |
| AutoML-E | deer | 5.444 | 1 | 0.838480 | +0.000212 |
| AutoML-E | ple_a | 5.778 | 2 | 0.838434 | +0.000167 |
| AutoML-E | field | 6.111 | 3 | 0.838267 | +0.000000 |
| AutoML-E | quantile_bucket | 6.667 | 1 | 0.838210 | -0.000057 |
| AutoML-E | dae | 6.889 | 0 | 0.838125 | -0.000142 |
| Criteo | deer | 1.222 | 8 | 0.810481 | +0.002961 |
| Criteo | daes_gate | 3.222 | 0 | 0.810037 | +0.002517 |
| Criteo | ple_b | 3.333 | 1 | 0.809961 | +0.002442 |
| Criteo | narydis_full | 4.778 | 0 | 0.809822 | +0.002303 |
| Criteo | ple_a | 5.222 | 0 | 0.809770 | +0.002250 |

## Tuning gains over the default benchmark

| method | mean_auc_gain | improved_cells | mean_rank_change |
|---|---|---|---|
| narydis_full | +0.003951 | 24 | -3.185 |
| bspline | +0.000926 | 21 | -1.815 |
| daes_tran | +0.000540 | 19 | -1.407 |
| dae | +0.000473 | 21 | -1.889 |
| narydis_lite | +0.000454 | 17 | -1.593 |
| deer | +0.000328 | 20 | -1.037 |
| daes_gate | +0.000202 | 16 | -0.556 |
| linear | +0.000002 | 1 | -0.593 |
| plr | +0.000000 | 1 | -0.519 |
| ple_b | +0.000000 | 1 | -0.556 |
| linear_relu | +0.000000 | 1 | -0.741 |
| ple_relu | +0.000000 | 1 | -0.667 |
| field | +0.000000 | 1 | -0.407 |
| ple_a | +0.000000 | 0 | -0.852 |
| log_squared_bucket | -0.000007 | 1 | -0.889 |
| autodis | -0.000110 | 12 | +0.074 |
| quantile_bucket | -0.000133 | 9 | +0.222 |
| periodic | -0.000317 | 8 | +0.259 |

## Holm-corrected post-hoc comparisons

| reference_method | comparator | mean_auc_difference_reference_minus_comparator | wins | ties | losses | p_holm | significant_0_05 |
|---|---|---|---|---|---|---|---|
| deer | narydis_full | +0.003535 | 27 | 0 | 0 | 2.533e-07 | True |
| deer | log_squared_bucket | +0.016316 | 27 | 0 | 0 | 2.533e-07 | True |
| deer | narydis_lite | +0.006914 | 26 | 0 | 1 | 4.470e-07 | True |
| deer | autodis | +0.001409 | 25 | 0 | 2 | 8.970e-06 | True |
| deer | daes_tran | +0.002616 | 25 | 0 | 2 | 1.065e-05 | True |
| deer | linear_relu | +0.008108 | 24 | 0 | 3 | 1.967e-05 | True |
| deer | ple_relu | +0.003383 | 23 | 0 | 4 | 3.393e-05 | True |
| deer | linear | +0.003065 | 24 | 0 | 3 | 1.135e-04 | True |
| deer | field | +0.002683 | 23 | 0 | 4 | 1.434e-04 | True |
| deer | plr | +0.001860 | 22 | 0 | 5 | 1.503e-04 | True |
| deer | dae | +0.001640 | 25 | 0 | 2 | 2.466e-04 | True |
| deer | periodic | +0.002692 | 22 | 0 | 5 | 4.374e-04 | True |
| deer | quantile_bucket | +0.001369 | 24 | 0 | 3 | 6.266e-04 | True |
| deer | daes_gate | +0.001699 | 24 | 0 | 3 | 1.068e-03 | True |
| deer | bspline | +0.001191 | 21 | 0 | 6 | 4.588e-03 | True |
| deer | ple_a | +0.002639 | 19 | 0 | 8 | 9.225e-03 | True |
| deer | ple_b | +0.001374 | 19 | 0 | 8 | 1.208e-02 | True |

## Artifacts

- `per_seed.csv`: 1,458 individual runs.
- `summary.csv`: 486 three-seed dataset–backbone–method cells.
- `backbone_ranking.csv`: full within-cell ranking.
- `method_robustness.csv`: cross-cell ranks, wins, top-three counts, and Field-relative gains.
- `default_vs_tuned.csv`: matched default-versus-tuned results on the same standardized datasets.
- `friedman.json` and `holm_posthoc.csv`: omnibus and corrected paired tests.
- `config_snapshot/`: frozen Stage-3 configuration snapshot. No checkpoint or model weight is archived.
