# NumEmb Benchmark for CTR Prediction

A runnable numerical-embedding benchmark built on a **frozen FuxiCTR 2.3.10
source snapshot**. It contains 18 numerical embedding configurations, nine CTR
backbones, three datasets, and a unified training and analysis protocol. The
repository contains the model code, integration layer, configuration templates,
data converters, launchers, tests, and published per-seed records. It does not
contain dataset archives or model checkpoints.

The benchmark code and paper artifacts have different purposes: the root of
this repository is for **new experiments**; [`reproduction/`](reproduction/README.md)
holds the frozen result records and the scripts that reconstruct the paper's
tables and figures without retraining.

## Method catalog and paper links

The 18 evaluated configurations belong to six methodological families. A
paper link identifies the intellectual source, **not** an assertion that this
repository contains the authors' official code. The implementation and its
fidelity boundary are listed in the last two columns; consult the
[provenance notes](docs/BASELINE_PROVENANCE.md) before comparing papers.

| Family | Benchmark configuration | Paper or source | Implementation | Provenance |
|:--|:--|:--|:--|:--|
| Pointwise projection | `field` | — (standard per-field embedding baseline) | [`method/field/`](method/field/) | Local baseline |
| Pointwise projection | `linear` | [Gorishniy et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) (compared baseline) | [`method/linear/`](method/linear/) | RTDL adapter |
| Pointwise projection | `linear_relu` | [Gorishniy et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) (compared baseline) | [`method/linear/`](method/linear/) | RTDL adapter |
| Discretization | `quantile_bucket` | [Gorishniy et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) (related quantile binning) | [`method/quantile_bucket/`](method/quantile_bucket/) | Benchmark baseline, not a separate paper method |
| Discretization | `log_squared_bucket` | [FuxiCTR Criteo preprocessing recipe](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/datasets/criteo.py) | [`method/log_squared_bucket/`](method/log_squared_bucket/) | Local adaptation of a data transform |
| Discretization | `autodis` | [Guo et al., KDD 2021](https://doi.org/10.1145/3447548.3467077) | [`method/autodis/`](method/autodis/) | Local reimplementation |
| Piecewise basis | `ple_a` | [Gorishniy et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) | [`method/ple/`](method/ple/) | RTDL adapter, variant A |
| Piecewise basis | `ple_b` | [Gorishniy et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) | [`method/ple/`](method/ple/) | RTDL adapter, variant B |
| Piecewise basis | `ple_relu` | [Gorishniy et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) | [`method/ple/`](method/ple/) | RTDL adapter, ReLU variant |
| Fixed basis | `periodic` | [Gorishniy et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) | [`method/periodic/`](method/periodic/) | RTDL adapter |
| Fixed basis | `plr` | [Gorishniy et al., NeurIPS 2022](https://proceedings.nips.cc/paper_files/paper/2022/hash/9e9f0ffc3d836836ca96cbf8fe14b105-Abstract-Conference.html) | [`method/periodic/`](method/periodic/) | RTDL adapter |
| Fixed basis | `bspline` | [Shtoff et al., TMLR 2024](https://arxiv.org/abs/2305.14528) (related spline-basis work) | [`method/bspline/`](method/bspline/) | Local B-spline baseline; **not** the paper's full model |
| Adaptive coordinates | `narydis_lite` | [Chen et al., CIKM 2022](https://doi.org/10.1145/3511808.3557090) (inspiration) | [`method/narydis/`](method/narydis/) | Simplified engineering variant; **not** paper NaryDis |
| Adaptive coordinates | `narydis_full` | [Chen et al., CIKM 2022](https://doi.org/10.1145/3511808.3557090) | [`method/narydis/`](method/narydis/) | Local paper-aligned reimplementation |
| Adaptive coordinates | `deer` | [Cheng et al., CIKM 2022](https://doi.org/10.1145/3511808.3557587) | [`method/deer/`](method/deer/) | Local paper-aligned reimplementation |
| Distribution/context adaptive | `dae` | [Shen et al., CIKM 2023](https://doi.org/10.1145/3583780.3615212) | [`method/dae/`](method/dae/) | Local paper-aligned reimplementation |
| Distribution/context adaptive | `daes_gate` | [Liu et al., arXiv 2026](https://arxiv.org/abs/2602.03223) | [`method/daes/`](method/daes/) | Local reimplementation; contextual gate variant |
| Distribution/context adaptive | `daes_tran` | [Liu et al., arXiv 2026](https://arxiv.org/abs/2602.03223) | [`method/daes/`](method/daes/) | Local reimplementation; contextual transform variant |

<!-- BENCHMARK_RESULTS_START -->
## Five-seed benchmark results

The tables report **test AUC** and **test logloss** as mean ± sample standard deviation over seeds 2026–2030; higher AUC and lower logloss are better. Each cell is a separately trained method–backbone–dataset combination. These are the tuned shared-z-score V2 results, not the earlier raw-feature matrix. The training protocol is identical across methods within a dataset/backbone; method-specific settings are frozen in the supplied configurations. No values are pooled across different datasets or backbones. The original 2026–2028 records are preserved separately; the 2029–2030 extension was rerun using the reorganized repository and strictly validated before merging.

### Criteo — test AUC

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.808547 ± 0.000137 | 0.805612 ± 0.000105 | 0.807940 ± 0.000101 | 0.800859 ± 0.000249 | 0.809236 ± 0.000052 | 0.808320 ± 0.000262 | 0.809366 ± 0.000159 | 0.808884 ± 0.000137 | 0.809041 ± 0.000137 |
| `linear` | 0.808557 ± 0.000153 | 0.804107 ± 0.000301 | 0.807740 ± 0.000164 | 0.800957 ± 0.000329 | 0.809057 ± 0.000158 | 0.808212 ± 0.000226 | 0.809372 ± 0.000162 | 0.808791 ± 0.000157 | 0.809532 ± 0.000168 |
| `linear_relu` | 0.809299 ± 0.000075 | 0.805298 ± 0.000243 | 0.808539 ± 0.000125 | 0.802236 ± 0.000287 | 0.809550 ± 0.000073 | 0.809162 ± 0.000118 | 0.810780 ± 0.000088 | 0.809242 ± 0.000092 | 0.810240 ± 0.000142 |
| `quantile_bucket` | 0.810141 ± 0.000154 | 0.807588 ± 0.000162 | 0.809323 ± 0.000130 | 0.805219 ± 0.000156 | 0.811167 ± 0.000126 | 0.810146 ± 0.000063 | 0.811599 ± 0.000134 | 0.810543 ± 0.000182 | 0.810819 ± 0.000091 |
| `log_squared_bucket` | 0.793593 ± 0.000173 | 0.791339 ± 0.000182 | 0.793022 ± 0.000162 | 0.789960 ± 0.000213 | 0.794490 ± 0.000054 | 0.793696 ± 0.000135 | 0.794681 ± 0.000039 | 0.793924 ± 0.000159 | 0.793739 ± 0.000112 |
| `autodis` | 0.810180 ± 0.000045 | 0.807696 ± 0.000233 | 0.809494 ± 0.000112 | 0.804316 ± 0.000143 | 0.810955 ± 0.000145 | 0.810281 ± 0.000086 | 0.811546 ± 0.000143 | 0.810491 ± 0.000188 | 0.810434 ± 0.000136 |
| `ple_a` | 0.810442 ± 0.000122 | 0.806510 ± 0.000271 | 0.809634 ± 0.000299 | 0.804936 ± 0.000322 | 0.811453 ± 0.000136 | 0.810349 ± 0.000112 | 0.812189 ± 0.000067 | 0.810480 ± 0.000098 | 0.811642 ± 0.000186 |
| `ple_b` | 0.810614 ± 0.000219 | 0.806528 ± 0.000322 | 0.810042 ± 0.000126 | 0.805249 ± 0.000147 | 0.811751 ± 0.000124 | 0.810709 ± 0.000150 | 0.812501 ± 0.000128 | 0.810756 ± 0.000148 | 0.811658 ± 0.000183 |
| `ple_relu` | 0.808484 ± 0.000127 | 0.804006 ± 0.000162 | 0.807888 ± 0.000064 | 0.800818 ± 0.000078 | 0.809124 ± 0.000186 | 0.808256 ± 0.000093 | 0.809590 ± 0.000216 | 0.808801 ± 0.000089 | 0.809452 ± 0.000092 |
| `periodic` | 0.809050 ± 0.000245 | 0.802303 ± 0.000453 | 0.807965 ± 0.000519 | 0.802763 ± 0.000408 | 0.809688 ± 0.000142 | 0.808847 ± 0.000289 | 0.810744 ± 0.000163 | 0.808646 ± 0.000297 | 0.810025 ± 0.000245 |
| `plr` | 0.809532 ± 0.000271 | 0.806437 ± 0.000240 | 0.809037 ± 0.000111 | 0.803683 ± 0.000055 | 0.810264 ± 0.000145 | 0.809630 ± 0.000269 | 0.811283 ± 0.000051 | 0.809873 ± 0.000178 | 0.810276 ± 0.000126 |
| `narydis_lite` | 0.809257 ± 0.000087 | 0.806555 ± 0.000149 | 0.808484 ± 0.000153 | 0.801733 ± 0.000261 | 0.809923 ± 0.000093 | 0.809155 ± 0.000255 | 0.810952 ± 0.000129 | 0.809601 ± 0.000056 | 0.809576 ± 0.000222 |
| `bspline` | 0.809178 ± 0.000151 | 0.806471 ± 0.000257 | 0.808537 ± 0.000243 | 0.801349 ± 0.000235 | 0.809510 ± 0.000066 | 0.809256 ± 0.000041 | 0.810396 ± 0.000114 | 0.809355 ± 0.000287 | 0.809697 ± 0.000233 |
| `narydis_full` | 0.810407 ± 0.000153 | 0.807989 ± 0.000167 | 0.809689 ± 0.000159 | 0.805309 ± 0.000093 | 0.811229 ± 0.000110 | 0.810365 ± 0.000081 | 0.811949 ± 0.000070 | 0.810623 ± 0.000137 | 0.811008 ± 0.000120 |
| `deer` | 0.810942 ± 0.000234 | 0.808506 ± 0.000168 | 0.810347 ± 0.000225 | 0.805916 ± 0.000172 | 0.812181 ± 0.000081 | 0.810963 ± 0.000137 | 0.812560 ± 0.000106 | 0.811353 ± 0.000108 | 0.811565 ± 0.000117 |
| `dae` | 0.810326 ± 0.000191 | 0.807839 ± 0.000118 | 0.809540 ± 0.000126 | 0.805022 ± 0.000125 | 0.810757 ± 0.000208 | 0.810315 ± 0.000150 | 0.811646 ± 0.000198 | 0.810407 ± 0.000123 | 0.810386 ± 0.000320 |
| `daes_gate` | 0.810632 ± 0.000075 | 0.807624 ± 0.000308 | 0.809775 ± 0.000267 | 0.805750 ± 0.000196 | 0.811666 ± 0.000093 | 0.810511 ± 0.000197 | 0.812218 ± 0.000156 | 0.810659 ± 0.000081 | 0.811410 ± 0.000042 |
| `daes_tran` | 0.810471 ± 0.000161 | 0.807007 ± 0.000337 | 0.809662 ± 0.000191 | 0.805811 ± 0.000242 | 0.811063 ± 0.000124 | 0.810223 ± 0.000150 | 0.812087 ± 0.000172 | 0.810025 ± 0.000231 | 0.811289 ± 0.000225 |

<details>
<summary>Criteo — test logloss (all 18 methods × 9 backbones)</summary>

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.443378 ± 0.000236 | 0.446052 ± 0.000116 | 0.444037 ± 0.000177 | 0.450759 ± 0.000223 | 0.442551 ± 0.000106 | 0.443602 ± 0.000234 | 0.443297 ± 0.000282 | 0.443001 ± 0.000147 | 0.442901 ± 0.000302 |
| `linear` | 0.443279 ± 0.000096 | 0.447949 ± 0.000416 | 0.444256 ± 0.000250 | 0.450632 ± 0.000407 | 0.442926 ± 0.000446 | 0.443992 ± 0.000461 | 0.442814 ± 0.000204 | 0.443162 ± 0.000275 | 0.442829 ± 0.000457 |
| `linear_relu` | 0.442671 ± 0.000197 | 0.446662 ± 0.000519 | 0.443609 ± 0.000301 | 0.449283 ± 0.000206 | 0.442892 ± 0.000304 | 0.442911 ± 0.000229 | 0.441361 ± 0.000145 | 0.442674 ± 0.000239 | 0.442101 ± 0.000416 |
| `quantile_bucket` | 0.441740 ± 0.000212 | 0.444113 ± 0.000181 | 0.442813 ± 0.000080 | 0.446640 ± 0.000206 | 0.440676 ± 0.000261 | 0.441828 ± 0.000212 | 0.440497 ± 0.000111 | 0.441355 ± 0.000183 | 0.441418 ± 0.000253 |
| `log_squared_bucket` | 0.456619 ± 0.000285 | 0.458595 ± 0.000051 | 0.457340 ± 0.000226 | 0.460022 ± 0.000141 | 0.455600 ± 0.000078 | 0.456450 ± 0.000269 | 0.455794 ± 0.000065 | 0.456211 ± 0.000220 | 0.456568 ± 0.000266 |
| `autodis` | 0.441804 ± 0.000128 | 0.444064 ± 0.000360 | 0.442719 ± 0.000314 | 0.447530 ± 0.000261 | 0.441073 ± 0.000235 | 0.441731 ± 0.000163 | 0.440599 ± 0.000089 | 0.441387 ± 0.000240 | 0.441889 ± 0.000256 |
| `ple_a` | 0.441482 ± 0.000176 | 0.445151 ± 0.000324 | 0.442326 ± 0.000157 | 0.446971 ± 0.000346 | 0.440520 ± 0.000357 | 0.441693 ± 0.000573 | 0.439898 ± 0.000151 | 0.441334 ± 0.000219 | 0.440171 ± 0.000190 |
| `ple_b` | 0.441293 ± 0.000290 | 0.445494 ± 0.000461 | 0.442057 ± 0.000211 | 0.446570 ± 0.000412 | 0.440110 ± 0.000225 | 0.441403 ± 0.000404 | 0.439649 ± 0.000128 | 0.441203 ± 0.000220 | 0.440453 ± 0.000460 |
| `ple_relu` | 0.443336 ± 0.000251 | 0.448359 ± 0.000521 | 0.444068 ± 0.000168 | 0.450646 ± 0.000144 | 0.442864 ± 0.000358 | 0.443958 ± 0.000279 | 0.442747 ± 0.000198 | 0.443190 ± 0.000196 | 0.442798 ± 0.000096 |
| `periodic` | 0.442782 ± 0.000103 | 0.449854 ± 0.000535 | 0.443964 ± 0.000463 | 0.448914 ± 0.000613 | 0.442479 ± 0.000278 | 0.443091 ± 0.000265 | 0.441438 ± 0.000102 | 0.443096 ± 0.000293 | 0.442200 ± 0.000394 |
| `plr` | 0.442327 ± 0.000424 | 0.445341 ± 0.000305 | 0.443268 ± 0.000538 | 0.447861 ± 0.000076 | 0.441518 ± 0.000241 | 0.442146 ± 0.000161 | 0.440895 ± 0.000199 | 0.442085 ± 0.000288 | 0.442061 ± 0.000230 |
| `narydis_lite` | 0.442702 ± 0.000253 | 0.445350 ± 0.000656 | 0.443770 ± 0.000307 | 0.449726 ± 0.000266 | 0.442015 ± 0.000445 | 0.442615 ± 0.000227 | 0.441173 ± 0.000122 | 0.442189 ± 0.000166 | 0.442542 ± 0.000211 |
| `bspline` | 0.442976 ± 0.000366 | 0.445135 ± 0.000230 | 0.443451 ± 0.000160 | 0.450193 ± 0.000306 | 0.442515 ± 0.000198 | 0.442717 ± 0.000236 | 0.441684 ± 0.000174 | 0.442529 ± 0.000359 | 0.442434 ± 0.000386 |
| `narydis_full` | 0.441463 ± 0.000172 | 0.443776 ± 0.000340 | 0.442521 ± 0.000219 | 0.446595 ± 0.000129 | 0.440723 ± 0.000214 | 0.441778 ± 0.000319 | 0.440116 ± 0.000093 | 0.441290 ± 0.000212 | 0.441302 ± 0.000107 |
| `deer` | 0.441012 ± 0.000190 | 0.443246 ± 0.000139 | 0.441727 ± 0.000184 | 0.445890 ± 0.000393 | 0.439709 ± 0.000159 | 0.441089 ± 0.000355 | 0.439505 ± 0.000074 | 0.440581 ± 0.000296 | 0.440497 ± 0.000252 |
| `dae` | 0.441501 ± 0.000168 | 0.443997 ± 0.000331 | 0.442631 ± 0.000329 | 0.446799 ± 0.000157 | 0.441181 ± 0.000318 | 0.441613 ± 0.000230 | 0.440583 ± 0.000081 | 0.441629 ± 0.000452 | 0.441887 ± 0.000261 |
| `daes_gate` | 0.441190 ± 0.000133 | 0.444027 ± 0.000275 | 0.442571 ± 0.000517 | 0.446481 ± 0.000348 | 0.440430 ± 0.000605 | 0.441470 ± 0.000166 | 0.440010 ± 0.000304 | 0.441480 ± 0.000456 | 0.440778 ± 0.000107 |
| `daes_tran` | 0.441412 ± 0.000092 | 0.445248 ± 0.000622 | 0.442371 ± 0.000158 | 0.446020 ± 0.000440 | 0.440787 ± 0.000145 | 0.441877 ± 0.000598 | 0.440081 ± 0.000172 | 0.441729 ± 0.000205 | 0.440984 ± 0.000319 |

</details>

### AutoML-A — test AUC

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.705373 ± 0.003709 | 0.707362 ± 0.002104 | 0.706679 ± 0.001765 | 0.685684 ± 0.002202 | 0.701123 ± 0.002405 | 0.706328 ± 0.003629 | 0.707815 ± 0.002190 | 0.711125 ± 0.004104 | 0.697856 ± 0.002133 |
| `linear` | 0.705202 ± 0.002525 | 0.696621 ± 0.002220 | 0.704462 ± 0.003806 | 0.688098 ± 0.001887 | 0.705086 ± 0.004097 | 0.705124 ± 0.002111 | 0.705612 ± 0.000971 | 0.706388 ± 0.002221 | 0.709702 ± 0.001956 |
| `linear_relu` | 0.704402 ± 0.002392 | 0.620534 ± 0.110044 | 0.703103 ± 0.002043 | 0.681220 ± 0.005598 | 0.703976 ± 0.002419 | 0.703702 ± 0.002407 | 0.704733 ± 0.001388 | 0.708129 ± 0.001776 | 0.708030 ± 0.002043 |
| `quantile_bucket` | 0.705848 ± 0.004411 | 0.707815 ± 0.004377 | 0.706654 ± 0.001801 | 0.691034 ± 0.002246 | 0.705448 ± 0.003100 | 0.704476 ± 0.001347 | 0.705887 ± 0.001123 | 0.709395 ± 0.003503 | 0.704635 ± 0.002356 |
| `log_squared_bucket` | 0.680373 ± 0.001572 | 0.679816 ± 0.002785 | 0.678955 ± 0.001492 | 0.664459 ± 0.002523 | 0.679852 ± 0.005436 | 0.679063 ± 0.001694 | 0.680510 ± 0.002593 | 0.686235 ± 0.004213 | 0.679758 ± 0.002064 |
| `autodis` | 0.706104 ± 0.001496 | 0.706521 ± 0.002940 | 0.706714 ± 0.003520 | 0.688733 ± 0.002049 | 0.706539 ± 0.001631 | 0.703898 ± 0.002610 | 0.707807 ± 0.002396 | 0.713194 ± 0.001733 | 0.704699 ± 0.002073 |
| `ple_a` | 0.707781 ± 0.000647 | 0.676109 ± 0.006661 | 0.701084 ± 0.003724 | 0.689830 ± 0.003627 | 0.709955 ± 0.001337 | 0.706504 ± 0.002243 | 0.705145 ± 0.001636 | 0.707176 ± 0.004115 | 0.703641 ± 0.004034 |
| `ple_b` | 0.707859 ± 0.003124 | 0.701602 ± 0.004926 | 0.705071 ± 0.003151 | 0.694628 ± 0.002567 | 0.707121 ± 0.005630 | 0.705489 ± 0.004529 | 0.710423 ± 0.003110 | 0.707980 ± 0.001020 | 0.708732 ± 0.003583 |
| `ple_relu` | 0.704894 ± 0.002995 | 0.697282 ± 0.003338 | 0.702315 ± 0.004149 | 0.687636 ± 0.002718 | 0.704075 ± 0.004552 | 0.701186 ± 0.005367 | 0.707174 ± 0.003732 | 0.703533 ± 0.003553 | 0.709080 ± 0.003769 |
| `periodic` | 0.704993 ± 0.002306 | 0.698186 ± 0.002810 | 0.702394 ± 0.002322 | 0.692085 ± 0.001808 | 0.703785 ± 0.003348 | 0.702545 ± 0.001468 | 0.708409 ± 0.004153 | 0.708599 ± 0.001341 | 0.711009 ± 0.002757 |
| `plr` | 0.706291 ± 0.003718 | 0.707045 ± 0.001760 | 0.706715 ± 0.001449 | 0.686111 ± 0.004498 | 0.702881 ± 0.004497 | 0.704528 ± 0.002667 | 0.709330 ± 0.002437 | 0.711636 ± 0.002703 | 0.706135 ± 0.001131 |
| `narydis_lite` | 0.692024 ± 0.001272 | 0.689984 ± 0.001700 | 0.689370 ± 0.000831 | 0.676440 ± 0.002741 | 0.691442 ± 0.001886 | 0.689290 ± 0.003117 | 0.689135 ± 0.001722 | 0.704789 ± 0.004655 | 0.686747 ± 0.005836 |
| `bspline` | 0.707855 ± 0.001398 | 0.709627 ± 0.002660 | 0.708675 ± 0.002843 | 0.692636 ± 0.001766 | 0.706030 ± 0.002256 | 0.706754 ± 0.001270 | 0.708955 ± 0.002159 | 0.713278 ± 0.003252 | 0.710015 ± 0.002519 |
| `narydis_full` | 0.703093 ± 0.001652 | 0.699571 ± 0.002324 | 0.700096 ± 0.001328 | 0.688579 ± 0.001475 | 0.701565 ± 0.001932 | 0.701667 ± 0.003665 | 0.700127 ± 0.001572 | 0.708860 ± 0.003348 | 0.695722 ± 0.004743 |
| `deer` | 0.709122 ± 0.001259 | 0.710289 ± 0.002423 | 0.709865 ± 0.002409 | 0.692416 ± 0.003294 | 0.709010 ± 0.001202 | 0.708880 ± 0.001213 | 0.708162 ± 0.001873 | 0.715844 ± 0.001303 | 0.706304 ± 0.002511 |
| `dae` | 0.704276 ± 0.001603 | 0.706874 ± 0.000476 | 0.707444 ± 0.000424 | 0.690355 ± 0.003441 | 0.705147 ± 0.002574 | 0.704508 ± 0.003288 | 0.705027 ± 0.000787 | 0.713014 ± 0.002781 | 0.703419 ± 0.001269 |
| `daes_gate` | 0.705402 ± 0.002576 | 0.703445 ± 0.003977 | 0.704290 ± 0.003496 | 0.689356 ± 0.007786 | 0.705139 ± 0.001068 | 0.705317 ± 0.002723 | 0.706324 ± 0.002497 | 0.711173 ± 0.002321 | 0.709715 ± 0.001874 |
| `daes_tran` | 0.704641 ± 0.002774 | 0.700570 ± 0.002368 | 0.702327 ± 0.001518 | 0.689914 ± 0.003637 | 0.701787 ± 0.003532 | 0.702173 ± 0.001862 | 0.703057 ± 0.001098 | 0.709080 ± 0.001676 | 0.706499 ± 0.003023 |

<details>
<summary>AutoML-A — test logloss (all 18 methods × 9 backbones)</summary>

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.237337 ± 0.006969 | 0.237975 ± 0.002785 | 0.237546 ± 0.002276 | 0.228446 ± 0.003935 | 0.239029 ± 0.002409 | 0.243385 ± 0.007708 | 0.243958 ± 0.002907 | 0.213059 ± 0.005280 | 0.244234 ± 0.003321 |
| `linear` | 0.246009 ± 0.007995 | 0.230526 ± 0.006034 | 0.237464 ± 0.009109 | 0.221738 ± 0.004678 | 0.245876 ± 0.005303 | 0.238661 ± 0.010252 | 0.237567 ± 0.003257 | 0.227824 ± 0.007699 | 0.242209 ± 0.006357 |
| `linear_relu` | 0.243332 ± 0.007600 | 13.756528 ± 18.516464 | 0.235730 ± 0.008018 | 0.225515 ± 0.003769 | 0.245911 ± 0.009048 | 0.239768 ± 0.010817 | 0.239440 ± 0.002108 | 0.229128 ± 0.007990 | 0.247612 ± 0.003526 |
| `quantile_bucket` | 0.244890 ± 0.006842 | 0.240521 ± 0.003695 | 0.242665 ± 0.003349 | 0.228910 ± 0.004740 | 0.240581 ± 0.007800 | 0.243198 ± 0.006575 | 0.242660 ± 0.006912 | 0.220443 ± 0.010619 | 0.243404 ± 0.012249 |
| `log_squared_bucket` | 0.241538 ± 0.007413 | 0.250466 ± 0.007374 | 0.252459 ± 0.007595 | 0.235089 ± 0.005244 | 0.251636 ± 0.006862 | 0.251607 ± 0.006359 | 0.249776 ± 0.006717 | 0.224711 ± 0.009755 | 0.244789 ± 0.018080 |
| `autodis` | 0.241161 ± 0.003067 | 0.241341 ± 0.006532 | 0.239547 ± 0.003973 | 0.229841 ± 0.003585 | 0.242058 ± 0.006742 | 0.245668 ± 0.003391 | 0.240702 ± 0.006209 | 0.212394 ± 0.006418 | 0.235025 ± 0.015038 |
| `ple_a` | 0.244329 ± 0.004832 | 0.238381 ± 0.013178 | 0.242631 ± 0.007978 | 0.222274 ± 0.008695 | 0.235672 ± 0.006349 | 0.241261 ± 0.005184 | 0.229832 ± 0.004006 | 0.239781 ± 0.008391 | 0.231774 ± 0.003259 |
| `ple_b` | 0.242331 ± 0.008249 | 0.222648 ± 0.008659 | 0.233598 ± 0.006142 | 0.222216 ± 0.006121 | 0.240766 ± 0.008721 | 0.247859 ± 0.012475 | 0.234886 ± 0.003681 | 0.243307 ± 0.006576 | 0.231752 ± 0.006417 |
| `ple_relu` | 0.243479 ± 0.007208 | 0.222375 ± 0.008287 | 0.236514 ± 0.007659 | 0.227643 ± 0.004428 | 0.239807 ± 0.008539 | 0.247833 ± 0.010918 | 0.234818 ± 0.004216 | 0.245303 ± 0.006555 | 0.235554 ± 0.008088 |
| `periodic` | 0.250264 ± 0.004052 | 0.237378 ± 0.028181 | 0.244000 ± 0.014215 | 0.231593 ± 0.012318 | 0.232820 ± 0.003805 | 0.250357 ± 0.010299 | 0.233427 ± 0.002528 | 0.246794 ± 0.010415 | 0.236267 ± 0.007464 |
| `plr` | 0.242837 ± 0.005050 | 0.237724 ± 0.006252 | 0.244305 ± 0.005494 | 0.229946 ± 0.008341 | 0.251075 ± 0.008461 | 0.245445 ± 0.004092 | 0.236674 ± 0.002948 | 0.225148 ± 0.004957 | 0.237120 ± 0.008306 |
| `narydis_lite` | 0.246437 ± 0.004913 | 0.240034 ± 0.006412 | 0.239876 ± 0.007997 | 0.231629 ± 0.005922 | 0.241473 ± 0.007486 | 0.253543 ± 0.011770 | 0.242973 ± 0.004895 | 0.214691 ± 0.006480 | 0.238561 ± 0.024247 |
| `bspline` | 0.247675 ± 0.005166 | 0.240455 ± 0.002683 | 0.241786 ± 0.003595 | 0.225916 ± 0.003540 | 0.242897 ± 0.006630 | 0.241020 ± 0.008580 | 0.244795 ± 0.004165 | 0.217618 ± 0.007057 | 0.232745 ± 0.014184 |
| `narydis_full` | 0.243086 ± 0.001743 | 0.236778 ± 0.004079 | 0.237445 ± 0.005037 | 0.223186 ± 0.004632 | 0.244651 ± 0.004249 | 0.243118 ± 0.009399 | 0.244536 ± 0.010102 | 0.211731 ± 0.006545 | 0.238380 ± 0.020808 |
| `deer` | 0.238705 ± 0.003427 | 0.236906 ± 0.007772 | 0.237148 ± 0.008577 | 0.223310 ± 0.003293 | 0.241565 ± 0.004303 | 0.239545 ± 0.007231 | 0.238252 ± 0.003835 | 0.217453 ± 0.005412 | 0.248206 ± 0.004498 |
| `dae` | 0.240209 ± 0.004400 | 0.240555 ± 0.004146 | 0.239185 ± 0.003225 | 0.224422 ± 0.004310 | 0.241749 ± 0.004629 | 0.242812 ± 0.002974 | 0.239787 ± 0.007108 | 0.212013 ± 0.009296 | 0.244070 ± 0.005203 |
| `daes_gate` | 0.238234 ± 0.013549 | 0.238414 ± 0.009340 | 0.242233 ± 0.007940 | 0.223127 ± 0.006635 | 0.246282 ± 0.008138 | 0.243579 ± 0.009245 | 0.239989 ± 0.003659 | 0.227949 ± 0.008998 | 0.231092 ± 0.005437 |
| `daes_tran` | 0.243095 ± 0.007792 | 0.238655 ± 0.004899 | 0.240311 ± 0.005317 | 0.226773 ± 0.013464 | 0.247530 ± 0.006609 | 0.245210 ± 0.003157 | 0.239318 ± 0.006305 | 0.231091 ± 0.012534 | 0.228999 ± 0.021904 |

</details>

### AutoML-E — test AUC

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.835811 ± 0.002633 | 0.839183 ± 0.001259 | 0.838725 ± 0.001445 | 0.839495 ± 0.000633 | 0.838350 ± 0.001966 | 0.833769 ± 0.004255 | 0.837542 ± 0.001365 | 0.843991 ± 0.001708 | 0.834083 ± 0.001268 |
| `linear` | 0.837770 ± 0.001046 | 0.837790 ± 0.001863 | 0.837157 ± 0.001200 | 0.838385 ± 0.000943 | 0.838280 ± 0.002493 | 0.836580 ± 0.001460 | 0.837260 ± 0.000813 | 0.838799 ± 0.001272 | 0.834348 ± 0.003752 |
| `linear_relu` | 0.838067 ± 0.000739 | 0.836920 ± 0.000629 | 0.837258 ± 0.001106 | 0.836691 ± 0.001833 | 0.839131 ± 0.002016 | 0.836158 ± 0.001676 | 0.837257 ± 0.001233 | 0.839838 ± 0.001323 | 0.834396 ± 0.004825 |
| `quantile_bucket` | 0.838682 ± 0.002083 | 0.838272 ± 0.001598 | 0.837482 ± 0.001752 | 0.839632 ± 0.000595 | 0.838571 ± 0.001780 | 0.835038 ± 0.003228 | 0.835697 ± 0.001768 | 0.842365 ± 0.000648 | 0.833713 ± 0.003071 |
| `log_squared_bucket` | 0.836642 ± 0.002838 | 0.837249 ± 0.001144 | 0.837404 ± 0.000872 | 0.838460 ± 0.000432 | 0.838485 ± 0.001900 | 0.835184 ± 0.004252 | 0.836287 ± 0.002646 | 0.841406 ± 0.001094 | 0.832467 ± 0.004636 |
| `autodis` | 0.836439 ± 0.001262 | 0.837486 ± 0.001359 | 0.837831 ± 0.001675 | 0.839749 ± 0.000264 | 0.838252 ± 0.001869 | 0.836697 ± 0.001601 | 0.837000 ± 0.003272 | 0.843123 ± 0.000740 | 0.833073 ± 0.002932 |
| `ple_a` | 0.837537 ± 0.002710 | 0.838456 ± 0.000716 | 0.837888 ± 0.001641 | 0.837496 ± 0.001310 | 0.839521 ± 0.001330 | 0.837800 ± 0.001811 | 0.839347 ± 0.000768 | 0.839455 ± 0.001642 | 0.836398 ± 0.002110 |
| `ple_b` | 0.836580 ± 0.002010 | 0.837741 ± 0.003095 | 0.837800 ± 0.002026 | 0.837849 ± 0.000395 | 0.839554 ± 0.000448 | 0.837811 ± 0.001873 | 0.837563 ± 0.000987 | 0.841065 ± 0.001428 | 0.835273 ± 0.002629 |
| `ple_relu` | 0.836687 ± 0.001591 | 0.838291 ± 0.002265 | 0.836928 ± 0.001905 | 0.837822 ± 0.000522 | 0.838466 ± 0.000840 | 0.837379 ± 0.001965 | 0.837559 ± 0.000855 | 0.840144 ± 0.001173 | 0.836195 ± 0.000892 |
| `periodic` | 0.837916 ± 0.001326 | 0.837506 ± 0.001585 | 0.837332 ± 0.001109 | 0.837478 ± 0.000297 | 0.837868 ± 0.001409 | 0.838025 ± 0.000585 | 0.837855 ± 0.001564 | 0.839537 ± 0.001156 | 0.834163 ± 0.002009 |
| `plr` | 0.838141 ± 0.001987 | 0.838789 ± 0.000866 | 0.836975 ± 0.000537 | 0.838200 ± 0.001037 | 0.838882 ± 0.001072 | 0.837027 ± 0.001326 | 0.837533 ± 0.000507 | 0.840516 ± 0.001677 | 0.832302 ± 0.003806 |
| `narydis_lite` | 0.836866 ± 0.001182 | 0.837679 ± 0.002967 | 0.837644 ± 0.002841 | 0.837974 ± 0.000623 | 0.838160 ± 0.000682 | 0.835679 ± 0.002137 | 0.836300 ± 0.002058 | 0.841780 ± 0.001053 | 0.832856 ± 0.002955 |
| `bspline` | 0.837267 ± 0.000960 | 0.837600 ± 0.002403 | 0.837206 ± 0.003242 | 0.839689 ± 0.000296 | 0.839129 ± 0.003280 | 0.837137 ± 0.001617 | 0.837259 ± 0.001350 | 0.841671 ± 0.002740 | 0.833186 ± 0.001605 |
| `narydis_full` | 0.837137 ± 0.003427 | 0.836560 ± 0.001209 | 0.836474 ± 0.000987 | 0.839259 ± 0.000702 | 0.837961 ± 0.002985 | 0.835795 ± 0.002021 | 0.836755 ± 0.000629 | 0.841736 ± 0.001279 | 0.831159 ± 0.002825 |
| `deer` | 0.837414 ± 0.002294 | 0.839009 ± 0.001397 | 0.838505 ± 0.001219 | 0.839962 ± 0.000448 | 0.837884 ± 0.001925 | 0.836662 ± 0.003520 | 0.837478 ± 0.001114 | 0.842371 ± 0.001644 | 0.835228 ± 0.002644 |
| `dae` | 0.838402 ± 0.001496 | 0.838268 ± 0.002137 | 0.837986 ± 0.001701 | 0.839269 ± 0.000643 | 0.838908 ± 0.001367 | 0.836828 ± 0.001885 | 0.836322 ± 0.002159 | 0.842529 ± 0.001194 | 0.834439 ± 0.001466 |
| `daes_gate` | 0.838262 ± 0.001196 | 0.835855 ± 0.002114 | 0.835654 ± 0.002647 | 0.838754 ± 0.000935 | 0.838866 ± 0.002477 | 0.834940 ± 0.004295 | 0.837809 ± 0.001495 | 0.842062 ± 0.001437 | 0.831637 ± 0.003217 |
| `daes_tran` | 0.838244 ± 0.001375 | 0.837305 ± 0.004231 | 0.837146 ± 0.003294 | 0.838820 ± 0.000987 | 0.837753 ± 0.001877 | 0.837103 ± 0.001773 | 0.836147 ± 0.001965 | 0.841397 ± 0.001457 | 0.833438 ± 0.001979 |

<details>
<summary>AutoML-E — test logloss (all 18 methods × 9 backbones)</summary>

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.071235 ± 0.002200 | 0.070623 ± 0.000862 | 0.070598 ± 0.000863 | 0.071002 ± 0.000688 | 0.070010 ± 0.000943 | 0.071037 ± 0.000941 | 0.070117 ± 0.001030 | 0.068724 ± 0.000858 | 0.072786 ± 0.000691 |
| `linear` | 0.071234 ± 0.001007 | 0.070429 ± 0.000818 | 0.071128 ± 0.000938 | 0.070917 ± 0.000632 | 0.071896 ± 0.001561 | 0.071392 ± 0.001083 | 0.071491 ± 0.001376 | 0.070647 ± 0.000891 | 0.072515 ± 0.001618 |
| `linear_relu` | 0.070803 ± 0.000878 | 0.071565 ± 0.001436 | 0.070938 ± 0.000941 | 0.071131 ± 0.001153 | 0.071349 ± 0.001424 | 0.071264 ± 0.001204 | 0.070964 ± 0.001723 | 0.070134 ± 0.000797 | 0.072533 ± 0.001510 |
| `quantile_bucket` | 0.070249 ± 0.000833 | 0.070493 ± 0.001030 | 0.070665 ± 0.001015 | 0.070524 ± 0.001063 | 0.070725 ± 0.001282 | 0.071687 ± 0.001162 | 0.070558 ± 0.000905 | 0.069247 ± 0.000761 | 0.072013 ± 0.001045 |
| `log_squared_bucket` | 0.070532 ± 0.000939 | 0.070762 ± 0.001316 | 0.070792 ± 0.001284 | 0.070548 ± 0.000744 | 0.071304 ± 0.001601 | 0.071343 ± 0.000627 | 0.070941 ± 0.001216 | 0.068451 ± 0.000531 | 0.073253 ± 0.001511 |
| `autodis` | 0.071011 ± 0.000717 | 0.071122 ± 0.001552 | 0.070746 ± 0.001447 | 0.070814 ± 0.000945 | 0.070277 ± 0.001163 | 0.071085 ± 0.001473 | 0.070359 ± 0.000457 | 0.069229 ± 0.001303 | 0.073012 ± 0.001587 |
| `ple_a` | 0.071379 ± 0.000624 | 0.070838 ± 0.001381 | 0.071940 ± 0.000998 | 0.070686 ± 0.001028 | 0.071158 ± 0.000810 | 0.071541 ± 0.001920 | 0.071548 ± 0.000852 | 0.070111 ± 0.001455 | 0.072429 ± 0.001599 |
| `ple_b` | 0.071211 ± 0.001003 | 0.070982 ± 0.002290 | 0.071223 ± 0.001239 | 0.071147 ± 0.001526 | 0.071812 ± 0.001651 | 0.070954 ± 0.000965 | 0.071184 ± 0.000599 | 0.069563 ± 0.001076 | 0.072719 ± 0.001789 |
| `ple_relu` | 0.070780 ± 0.000891 | 0.070726 ± 0.001234 | 0.070985 ± 0.000906 | 0.071111 ± 0.001275 | 0.071136 ± 0.000848 | 0.070788 ± 0.000590 | 0.071969 ± 0.001229 | 0.070682 ± 0.001119 | 0.072197 ± 0.000868 |
| `periodic` | 0.070312 ± 0.001056 | 0.070402 ± 0.001582 | 0.071020 ± 0.001032 | 0.070057 ± 0.000665 | 0.071446 ± 0.000823 | 0.070954 ± 0.001714 | 0.072000 ± 0.001331 | 0.071544 ± 0.001254 | 0.072381 ± 0.001767 |
| `plr` | 0.070301 ± 0.000434 | 0.070262 ± 0.001047 | 0.070544 ± 0.000961 | 0.070773 ± 0.001050 | 0.070494 ± 0.000667 | 0.070885 ± 0.001276 | 0.070607 ± 0.001004 | 0.069186 ± 0.001237 | 0.073987 ± 0.001543 |
| `narydis_lite` | 0.071201 ± 0.000731 | 0.071204 ± 0.001076 | 0.071114 ± 0.000781 | 0.070717 ± 0.001069 | 0.070299 ± 0.000764 | 0.070748 ± 0.000582 | 0.069799 ± 0.001155 | 0.068700 ± 0.000896 | 0.072154 ± 0.002355 |
| `bspline` | 0.071451 ± 0.001341 | 0.070750 ± 0.000494 | 0.070628 ± 0.000546 | 0.070861 ± 0.000506 | 0.071598 ± 0.001761 | 0.070536 ± 0.000696 | 0.069887 ± 0.000717 | 0.070270 ± 0.001930 | 0.072815 ± 0.001360 |
| `narydis_full` | 0.071708 ± 0.002174 | 0.070583 ± 0.000487 | 0.070549 ± 0.000535 | 0.070909 ± 0.000579 | 0.070726 ± 0.001019 | 0.071274 ± 0.001850 | 0.070384 ± 0.000863 | 0.069375 ± 0.000646 | 0.073661 ± 0.001170 |
| `deer` | 0.070763 ± 0.000757 | 0.070193 ± 0.000502 | 0.070307 ± 0.000766 | 0.070548 ± 0.001037 | 0.070576 ± 0.001039 | 0.071929 ± 0.001776 | 0.070680 ± 0.000742 | 0.068218 ± 0.000905 | 0.072358 ± 0.000903 |
| `dae` | 0.070755 ± 0.001294 | 0.070942 ± 0.001280 | 0.071017 ± 0.000788 | 0.071517 ± 0.000987 | 0.070521 ± 0.001599 | 0.070775 ± 0.001062 | 0.070403 ± 0.000795 | 0.068950 ± 0.000871 | 0.072851 ± 0.001070 |
| `daes_gate` | 0.070222 ± 0.000587 | 0.071662 ± 0.001042 | 0.071381 ± 0.001358 | 0.070764 ± 0.001123 | 0.071125 ± 0.000385 | 0.071552 ± 0.001980 | 0.070387 ± 0.001015 | 0.068611 ± 0.001039 | 0.073208 ± 0.001549 |
| `daes_tran` | 0.069933 ± 0.000893 | 0.070313 ± 0.002132 | 0.070128 ± 0.001521 | 0.071076 ± 0.000692 | 0.070788 ± 0.000591 | 0.070713 ± 0.000643 | 0.070623 ± 0.000751 | 0.069935 ± 0.001862 | 0.073094 ± 0.001697 |

</details>

The [five-seed records](reproduction/results/five_seed_20260924/per_seed.csv) contain all 2,430 individual runs; the [full-precision summary](reproduction/results/five_seed_20260924/summary.csv) includes both test metrics, ranks and Field deltas. The [original three-seed records](reproduction/results/tuned_accuracy/per_seed.csv), including best/stopping epoch and elapsed time, remain unchanged. See the [five-seed analysis](reproduction/results/five_seed_20260924/summary.md), [protocol](docs/PROTOCOL.md), and [fidelity notes](docs/BASELINE_PROVENANCE.md) before interpreting differences as a property of the original papers.
<!-- BENCHMARK_RESULTS_END -->

## Repository layout

| Path | Purpose |
|---|---|
| `dataset/` | AutoML/Criteo converters and shared-z-score preparation/validation. |
| `fuxictr/` | Exact FuxiCTR 2.3.10 Python package snapshot used with this benchmark, including numerical-feature integration. |
| `method/` | Separate packages for each numerical-embedding family, plus the plugin entry point and frozen RTDL dependency. See [`docs/METHODS.md`](docs/METHODS.md). |
| `backbone/` | The nine FuxiCTR backbone model classes used in the experiments. |
| `reproduction/` | Frozen measurements and configurations, paper analysis, benchmark launchers, tests and archived source. |
| `docs/` | Detailed usage, protocol, provenance and third-party licenses. |
| `requirements.txt` | Training dependencies; analysis-only dependencies are in `reproduction/requirements.txt`. |
| `README.md`, `LICENSE` | Entry guide and the selected Apache-2.0 license. |

The numerical method is selected by `numerical_embedding` in the YAML model
configuration. `backbone/run_expid.py` loads a native backbone from `backbone/`, whose
FuxiCTR feature layer invokes `method/numemb_plugin.py` and `method/registry.py`.
The training entry point adds `method/` to the Python import path, preserving
the frozen plugin name `numemb_plugin` without an extra root-level shim.
All methods return a tensor of shape `[batch, numerical_fields, embedding_dim]`.
The repository root itself is the paired FuxiCTR root, so no copy into a
separate FuxiCTR checkout is needed. [`docs/FRAMEWORK.md`](docs/FRAMEWORK.md)
explains the integration and exact-version boundary.

## 1. Environment

Recommended: Linux, Python **3.10**, a PyTorch-compatible CUDA runtime, and
one or more NVIDIA GPUs. The recorded machine used Python 3.10, FuxiCTR 2.3.10,
PyTorch 2.6.0, NumPy 1.24.0, pandas 2.0.1, and PyArrow 24.0.0. These are
observed versions, not a claim that every other version has been tested. CPU
execution (`--gpus -1`) is intended only for small smoke tests.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a PyTorch build compatible with your CUDA driver if needed.
python -m pip install -r requirements.txt
python reproduction/tools/check_environment.py --require-cuda
python reproduction/tools/smoke_imports.py
python -m unittest discover -s reproduction/tests -v
```

The exact FuxiCTR and RTDL source snapshots are bundled, so do **not** install
a second `fuxictr` package over this repository. For result-only inspection,
PyTorch/GPU are unnecessary; use `reproduction/README.md` instead. See
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for platform details, resource
planning, and troubleshooting.

## 2. Prepare datasets

The frozen tuned protocol uses `Criteo_x4_NumEmbZScoreV2`,
`AutoML_A_ZScoreV2`, and `AutoML_E_ZScoreV2`. Each FuxiCTR-ready dataset has
`feature_map.json`, `train.parquet`, `valid.parquet`, and `test.parquet`.
Numerical statistics and bin edges are fitted on the training split only.
Dataset archives are not redistributed; see [`docs/DATASETS.md`](docs/DATASETS.md)
for source layouts, split rules, Criteo's categorical-reference prerequisite,
and validation checks.

If you already have the three FuxiCTR-ready **raw numerical** datasets, produce
the tuned shared-z-score variants as follows:

```bash
python dataset/prepare_numemb_version2_datasets.py \
  --criteo-source /data/Criteo_x4_NumEmbRaw \
  --automl-a-source /data/AutoML_A \
  --automl-e-source /data/AutoML_E \
  --output-root /data/numemb_v2/datasets

python dataset/validate_numemb_version2_datasets.py \
  --criteo-source /data/Criteo_x4_NumEmbRaw \
  --automl-a-source /data/AutoML_A \
  --automl-e-source /data/AutoML_E \
  --dataset-root /data/numemb_v2/datasets
```

The validator checks split row counts, schema, sampled numerical transforms,
unchanged categorical IDs/labels, and the bucket vocabularies. It does not
replace a source-data checksum audit.

To check that every method constructor works against a prepared feature map
without loading training rows, run:

```bash
python reproduction/tools/smoke_methods.py /data/numemb_v2/datasets/Criteo_x4_NumEmbZScoreV2/feature_map.json
```

## 3. Materialize the frozen experiment configurations

The checked-in templates are under
`reproduction/results/tuned_accuracy/config_snapshot/`. They contain
`__DATASET_ROOT__` and `__CHECKPOINT_ROOT__` placeholders. This command
creates relocated copies without changing hyperparameters:

```bash
python reproduction/tools/materialize_configs.py \
  --dataset-root /data/numemb_v2/datasets \
  --output-root /data/numemb_v2/configs \
  --checkpoint-root /data/numemb_v2/checkpoints
```

It refuses to overwrite an existing configuration directory. Use a new output
directory for a revised protocol rather than silently changing a frozen run.

## 4. Run one experiment

Use the same launcher for a single cell and a batch. First preview the command
without creating any file:

```bash
python reproduction/tools/run_benchmark.py \
  --config-root /data/numemb_v2/configs \
  --run-name criteo_dcnv2_dae_s2026 \
  --datasets criteo --backbones DCNv2 --methods dae --seeds 2026 \
  --gpus 0 --dry-run
```

Remove `--dry-run` to train. The launcher stores a manifest, per-task log, result
file, and elapsed time under `outputs/runs/<run-name>/`; model weights go under
`outputs/checkpoints/<run-name>/`. Both roots can be changed via `--run-root`
and `--checkpoint-root`. The launcher refuses an existing result rather than
overwriting it.

After a single or batch run, create seed-level and mean/standard-deviation
tables from **that run only**:

```bash
python reproduction/tools/summarize_run.py outputs/runs/criteo_dcnv2_dae_s2026 --require-complete
```

This writes `per_seed.csv`, `summary.csv`, and `summary_status.json` beside the
run manifest. Omit `--require-complete` for a clearly marked partial summary.

## 5. Run a filtered or full batch

For a filtered example (six tasks: two methods × three seeds):

```bash
python reproduction/tools/run_benchmark.py \
  --config-root /data/numemb_v2/configs \
  --run-name criteo_dcnv2_pair \
  --datasets criteo --backbones DCNv2 \
  --methods field,dae --seeds 2026,2027,2028 \
  --gpus 0,1 --dry-run
```

Remove `--dry-run` after checking the plan. Omit `--datasets`, `--backbones`,
`--methods`, and `--seeds` to select the entire frozen **1,458-task** matrix.
There is one training process per listed GPU, with a dynamic queue and dataset
interleaving. No failed task is silently retried. Inspect `failures.json` and
the corresponding log before deciding on a retry. `--resume` skips only a
nonempty result containing matching experiment/dataset identifiers and both
validation and test AUC/logloss; it rejects incomplete results instead of
overwriting them. For a new protocol, use a new `--run-name`.

To reproduce the published five-seed extension without rerunning or overwriting
the frozen 1,458 three-seed records, materialize a separate five-seed configuration and
select only seeds 2029 and 2030 (972 new tasks):

```bash
python reproduction/tools/materialize_configs.py \
  --dataset-root /data/numemb_v2/datasets \
  --output-root /data/numemb_v2/configs_five_seed \
  --checkpoint-root /data/numemb_v2/checkpoints \
  --seeds 2026,2027,2028,2029,2030
python reproduction/tools/run_benchmark.py \
  --config-root /data/numemb_v2/configs_five_seed \
  --run-name numemb_seed2029_2030 --seeds 2029,2030 \
  --gpus 0,1,2,3 --dry-run
```

Remove `--dry-run` to execute. After all 972 tasks pass, summarize the
completed run, then merge it with the original per-seed records into a *new*
directory:

```bash
python reproduction/tools/summarize_run.py outputs/runs/numemb_seed2029_2030 --require-complete
python reproduction/analysis/merge_five_seeds.py \
  --extension outputs/runs/numemb_seed2029_2030/per_seed.csv \
  --output-dir outputs/five_seed_results
```

The merge checks every method–backbone–dataset cell has exactly seeds
2026–2030. The completed [five-seed version](reproduction/results/five_seed_20260924/summary.md)
contains 2,430 runs, including the validated 972-run extension. Never replace
the published table with a partial run.

The default frozen protocol is: nine backbones, 18 methods, three datasets,
seeds 2026/2027/2028, embedding dimension 16, batch size 4096, learning rate
0.001, at most 100 epochs, validation-AUC early stopping with patience 2,
non-streaming Parquet, and global shuffle. Method-specific hyperparameters
are in the frozen YAML and `frozen_method_configs.json`, not inferred from
method defaults. Read [`docs/PROTOCOL.md`](docs/PROTOCOL.md) before comparing
against published results.

## 6. Verify and reproduce published analyses

The following checks require only Python's standard library:

```bash
python reproduction/tools/verify_manifest.py
python reproduction/analysis/verify_key_metrics.py
python reproduction/analysis/build_full_result_tables.py
python reproduction/tools/update_readme_results.py --check
```

They validate the original 1,458 per-seed records and rebuild the published
486-cell five-seed main table from the versioned summary. Run
`python reproduction/analysis/summarize_five_seeds.py reproduction/results/five_seed_20260924` to regenerate its method ranking and
statistical tests (requires SciPy).
Additional figures and cross-protocol analyses use
`reproduction/requirements.txt`; instructions and output locations are in
[`reproduction/README.md`](reproduction/README.md).

## Provenance, limits, and licensing

The bundled FuxiCTR tree is the **deployed 2.3.10 source snapshot**, not a
purported upstream Git commit: the deployment directory had no `.git` history.
The framework integration and nine backbone files are included so the exact
architecture used here is inspectable. `reproduction/tools/check_environment.py` compares
the archived integration copies in `reproduction/legacy/integration/` byte-for-byte
with the bundled framework. The 11 benchmark class definitions are checked
against their frozen source ASTs by `tests/test_method_sources.py`.

The snapshot was verified for structure and frozen-result analysis. The
reorganized code reproduced selected original runs exactly and completed the
972-task extension for seeds 2029–2030; it has **not** rerun the original
1,458 tasks after packaging. The Criteo raw conversion still requires an externally provided,
validated categorical-reference Parquet dataset. Baseline fidelity boundaries
are documented in [`docs/BASELINE_PROVENANCE.md`](docs/BASELINE_PROVENANCE.md).
Original benchmark code is licensed under Apache-2.0 (`LICENSE`, `NOTICE`).
Third-party Apache-2.0 notices are retained separately in
`docs/licenses/` and source headers.
Research extensions should follow [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) so a new
method or dataset cannot silently change a frozen comparison.
