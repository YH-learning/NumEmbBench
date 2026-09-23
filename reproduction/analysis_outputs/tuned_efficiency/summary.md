# Frozen-Tuned Numerical Embedding Efficiency Matrix

486/486 successful measurements: 18 methods x 3 datasets x 3 representative backbones x 3 randomized rounds.
All measurements use the Stage-2 validation-frozen configurations, one idle GPU, eager FP32, real batches, and a resident model. Disk I/O is excluded.
Paired multipliers are computed against Field in the same dataset, backbone, and randomized round.

| Method | Mean AUC | Rank | Numeric params | Train ms | Train x Field | Infer ms | Infer x Field | Embed ms | Peak MiB | Train front | Infer front |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DEER | 0.7856 | 3.1852 | 4,032 | 8.7284 | 0.7551 | 2.3715 | 1.1952 | 0.8514 | 526.7428 | True | True |
| PLE-B | 0.7842 | 6.5926 | 4,032 | 12.3316 | 1.0272 | 2.2173 | 1.1040 | 0.7469 | 510.6997 | False | True |
| PLE-A | 0.7830 | 6.8889 | 3,808 | 12.2860 | 1.0243 | 2.1826 | 1.0905 | 0.7243 | 509.7890 | False | False |
| Quantile | 0.7842 | 7.0000 | 2,133 | 14.7261 | 1.2247 | 2.4850 | 1.2252 | 1.0542 | 509.3754 | False | False |
| DAES-Gate | 0.7839 | 7.2222 | 4,496 | 14.7606 | 1.2755 | 2.5624 | 1.3068 | 1.0988 | 516.4302 | False | False |
| AutoDis | 0.7842 | 7.2963 | 7,392 | 12.4542 | 1.0407 | 2.2373 | 1.1164 | 0.7540 | 511.5594 | False | False |
| DAE | 0.7840 | 7.4074 | 36,736 | 12.7906 | 1.0750 | 2.2615 | 1.1341 | 0.7945 | 515.0118 | False | False |
| PLR | 0.7838 | 8.7037 | 7,616 | 12.3674 | 1.0324 | 2.1725 | 1.0827 | 0.7133 | 512.2655 | False | True |
| B-spline | 0.7844 | 8.7778 | 7,168 | 12.7316 | 1.0690 | 2.6788 | 1.3704 | 1.1889 | 511.4824 | False | False |
| DAES-Tran | 0.7830 | 9.9630 | 60,944 | 15.2259 | 1.3168 | 2.6368 | 1.3425 | 1.2149 | 518.9644 | False | False |
| Field | 0.7829 | 10.4074 | 224 | 12.0583 | 1.0000 | 2.0301 | 1.0000 | 0.6170 | 509.2905 | False | True |
| Periodic | 0.7829 | 10.6667 | 3,920 | 12.3134 | 1.0263 | 2.1568 | 1.0723 | 0.6966 | 510.8528 | False | False |
| NaryDis | 0.7821 | 10.8148 | 32,303 | 16.6067 | 1.4445 | 4.5587 | 2.4341 | 3.1849 | 792.4961 | False | False |
| Linear-ReLU | 0.7775 | 12.1481 | 448 | 12.1207 | 1.0075 | 2.0570 | 1.0163 | 0.6343 | 510.4549 | False | False |
| Linear | 0.7826 | 12.1852 | 448 | 12.0774 | 1.0024 | 2.0411 | 1.0080 | 0.6244 | 509.2950 | False | False |
| PLE-ReLU | 0.7822 | 12.2593 | 4,032 | 12.3214 | 1.0275 | 2.2172 | 1.1062 | 0.7527 | 510.4882 | False | False |
| NaryDis-Lite | 0.7787 | 13.0000 | 19,040 | 14.5338 | 1.2540 | 4.2218 | 2.2578 | 2.7270 | 513.8900 | False | False |
| LogBucket | 0.7693 | 16.4815 | 3,040 | 15.8331 | 1.3239 | 3.2244 | 1.6000 | 1.7269 | 509.4088 | False | False |

The train and inference fronts are defined separately by maximizing mean test AUC and minimizing the corresponding latency. The joint flag in the CSV additionally minimizes numerical parameters.
