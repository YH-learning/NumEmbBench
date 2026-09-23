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
## Frozen benchmark results

The tables report **test AUC** and **test logloss** as mean ± standard deviation over seeds 2026/2027/2028; higher AUC and lower logloss are better. Each cell is a separately trained method–backbone–dataset combination. These are the tuned shared-z-score V2 results, not the earlier raw-feature matrix. The training protocol is identical across methods within a dataset/backbone; method-specific settings are frozen in the supplied configurations. No values are pooled across different datasets or backbones.

### Criteo — test AUC

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.808614 ± 0.000112 | 0.805585 ± 0.000068 | 0.807916 ± 0.000111 | 0.800776 ± 0.000286 | 0.809232 ± 0.000043 | 0.808222 ± 0.000319 | 0.809407 ± 0.000116 | 0.808957 ± 0.000052 | 0.808965 ± 0.000119 |
| `linear` | 0.808590 ± 0.000121 | 0.804056 ± 0.000216 | 0.807698 ± 0.000207 | 0.800850 ± 0.000334 | 0.809029 ± 0.000101 | 0.808343 ± 0.000034 | 0.809431 ± 0.000133 | 0.808685 ± 0.000047 | 0.809531 ± 0.000156 |
| `linear_relu` | 0.809290 ± 0.000078 | 0.805251 ± 0.000165 | 0.808477 ± 0.000118 | 0.802198 ± 0.000218 | 0.809563 ± 0.000093 | 0.809217 ± 0.000099 | 0.810781 ± 0.000114 | 0.809218 ± 0.000062 | 0.810245 ± 0.000184 |
| `quantile_bucket` | 0.810221 ± 0.000064 | 0.807663 ± 0.000075 | 0.809347 ± 0.000175 | 0.805226 ± 0.000181 | 0.811192 ± 0.000170 | 0.810142 ± 0.000033 | 0.811521 ± 0.000074 | 0.810479 ± 0.000154 | 0.810870 ± 0.000083 |
| `log_squared_bucket` | 0.793598 ± 0.000242 | 0.791287 ± 0.000198 | 0.792926 ± 0.000087 | 0.789879 ± 0.000213 | 0.794503 ± 0.000025 | 0.793751 ± 0.000145 | 0.794673 ± 0.000046 | 0.793855 ± 0.000178 | 0.793764 ± 0.000143 |
| `autodis` | 0.810175 ± 0.000062 | 0.807752 ± 0.000150 | 0.809497 ± 0.000151 | 0.804220 ± 0.000071 | 0.810887 ± 0.000158 | 0.810234 ± 0.000057 | 0.811591 ± 0.000096 | 0.810439 ± 0.000245 | 0.810376 ± 0.000155 |
| `ple_a` | 0.810509 ± 0.000088 | 0.806568 ± 0.000355 | 0.809744 ± 0.000079 | 0.805043 ± 0.000387 | 0.811387 ± 0.000037 | 0.810366 ± 0.000079 | 0.812185 ± 0.000082 | 0.810470 ± 0.000099 | 0.811654 ± 0.000260 |
| `ple_b` | 0.810540 ± 0.000272 | 0.806460 ± 0.000338 | 0.810029 ± 0.000171 | 0.805252 ± 0.000207 | 0.811688 ± 0.000108 | 0.810722 ± 0.000118 | 0.812519 ± 0.000070 | 0.810705 ± 0.000100 | 0.811738 ± 0.000064 |
| `ple_relu` | 0.808429 ± 0.000144 | 0.803917 ± 0.000122 | 0.807927 ± 0.000028 | 0.800764 ± 0.000037 | 0.809106 ± 0.000104 | 0.808273 ± 0.000116 | 0.809747 ± 0.000038 | 0.808779 ± 0.000045 | 0.809446 ± 0.000126 |
| `periodic` | 0.809068 ± 0.000336 | 0.802274 ± 0.000612 | 0.808061 ± 0.000267 | 0.802517 ± 0.000311 | 0.809753 ± 0.000150 | 0.808738 ± 0.000346 | 0.810762 ± 0.000183 | 0.808681 ± 0.000404 | 0.810056 ± 0.000044 |
| `plr` | 0.809707 ± 0.000131 | 0.806397 ± 0.000239 | 0.809029 ± 0.000140 | 0.803699 ± 0.000030 | 0.810333 ± 0.000154 | 0.809466 ± 0.000205 | 0.811258 ± 0.000031 | 0.809799 ± 0.000204 | 0.810345 ± 0.000100 |
| `narydis_lite` | 0.809256 ± 0.000123 | 0.806617 ± 0.000173 | 0.808523 ± 0.000188 | 0.801714 ± 0.000304 | 0.809911 ± 0.000128 | 0.809301 ± 0.000152 | 0.811011 ± 0.000050 | 0.809575 ± 0.000051 | 0.809700 ± 0.000104 |
| `bspline` | 0.809231 ± 0.000182 | 0.806452 ± 0.000360 | 0.808407 ± 0.000188 | 0.801411 ± 0.000271 | 0.809485 ± 0.000059 | 0.809242 ± 0.000038 | 0.810318 ± 0.000045 | 0.809418 ± 0.000313 | 0.809661 ± 0.000322 |
| `narydis_full` | 0.810381 ± 0.000006 | 0.807916 ± 0.000175 | 0.809688 ± 0.000222 | 0.805325 ± 0.000109 | 0.811247 ± 0.000150 | 0.810377 ± 0.000075 | 0.811945 ± 0.000093 | 0.810564 ± 0.000149 | 0.810957 ± 0.000062 |
| `deer` | 0.811062 ± 0.000092 | 0.808447 ± 0.000186 | 0.810359 ± 0.000278 | 0.805931 ± 0.000164 | 0.812138 ± 0.000072 | 0.810952 ± 0.000192 | 0.812538 ± 0.000131 | 0.811342 ± 0.000117 | 0.811558 ± 0.000134 |
| `dae` | 0.810408 ± 0.000163 | 0.807844 ± 0.000133 | 0.809614 ± 0.000107 | 0.804996 ± 0.000165 | 0.810751 ± 0.000219 | 0.810370 ± 0.000083 | 0.811507 ± 0.000067 | 0.810423 ± 0.000160 | 0.810550 ± 0.000162 |
| `daes_gate` | 0.810584 ± 0.000044 | 0.807680 ± 0.000079 | 0.809851 ± 0.000319 | 0.805759 ± 0.000140 | 0.811665 ± 0.000131 | 0.810476 ± 0.000183 | 0.812250 ± 0.000162 | 0.810661 ± 0.000113 | 0.811403 ± 0.000056 |
| `daes_tran` | 0.810549 ± 0.000132 | 0.807130 ± 0.000326 | 0.809681 ± 0.000215 | 0.805915 ± 0.000275 | 0.811018 ± 0.000114 | 0.810220 ± 0.000158 | 0.812159 ± 0.000091 | 0.810145 ± 0.000227 | 0.811372 ± 0.000037 |

<details>
<summary>Criteo — test logloss (all 18 methods × 9 backbones)</summary>

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.443300 ± 0.000150 | 0.446104 ± 0.000034 | 0.444142 ± 0.000095 | 0.450804 ± 0.000283 | 0.442612 ± 0.000078 | 0.443719 ± 0.000223 | 0.443350 ± 0.000276 | 0.442938 ± 0.000136 | 0.443040 ± 0.000282 |
| `linear` | 0.443230 ± 0.000081 | 0.447892 ± 0.000391 | 0.444356 ± 0.000235 | 0.450582 ± 0.000344 | 0.442892 ± 0.000222 | 0.443805 ± 0.000176 | 0.442727 ± 0.000221 | 0.443284 ± 0.000261 | 0.442767 ± 0.000439 |
| `linear_relu` | 0.442692 ± 0.000223 | 0.446828 ± 0.000467 | 0.443798 ± 0.000117 | 0.449251 ± 0.000216 | 0.442847 ± 0.000336 | 0.442834 ± 0.000180 | 0.441410 ± 0.000179 | 0.442706 ± 0.000312 | 0.442059 ± 0.000474 |
| `quantile_bucket` | 0.441735 ± 0.000247 | 0.444058 ± 0.000104 | 0.442797 ± 0.000109 | 0.446701 ± 0.000235 | 0.440645 ± 0.000343 | 0.441875 ± 0.000259 | 0.440511 ± 0.000007 | 0.441354 ± 0.000212 | 0.441257 ± 0.000166 |
| `log_squared_bucket` | 0.456679 ± 0.000386 | 0.458578 ± 0.000032 | 0.457384 ± 0.000305 | 0.460033 ± 0.000191 | 0.455582 ± 0.000096 | 0.456344 ± 0.000171 | 0.455758 ± 0.000050 | 0.456326 ± 0.000214 | 0.456429 ± 0.000207 |
| `autodis` | 0.441821 ± 0.000177 | 0.443952 ± 0.000038 | 0.442722 ± 0.000256 | 0.447438 ± 0.000168 | 0.441133 ± 0.000246 | 0.441817 ± 0.000096 | 0.440580 ± 0.000115 | 0.441364 ± 0.000266 | 0.441831 ± 0.000318 |
| `ple_a` | 0.441548 ± 0.000193 | 0.445072 ± 0.000392 | 0.442271 ± 0.000064 | 0.446889 ± 0.000302 | 0.440615 ± 0.000391 | 0.441912 ± 0.000673 | 0.439874 ± 0.000059 | 0.441377 ± 0.000289 | 0.440215 ± 0.000254 |
| `ple_b` | 0.441464 ± 0.000227 | 0.445636 ± 0.000530 | 0.442038 ± 0.000294 | 0.446568 ± 0.000580 | 0.440216 ± 0.000236 | 0.441394 ± 0.000257 | 0.439629 ± 0.000026 | 0.441317 ± 0.000157 | 0.440333 ± 0.000144 |
| `ple_relu` | 0.443493 ± 0.000184 | 0.448712 ± 0.000265 | 0.443979 ± 0.000123 | 0.450660 ± 0.000196 | 0.442902 ± 0.000345 | 0.443863 ± 0.000180 | 0.442670 ± 0.000232 | 0.443318 ± 0.000074 | 0.442779 ± 0.000131 |
| `periodic` | 0.442795 ± 0.000139 | 0.449742 ± 0.000717 | 0.443998 ± 0.000171 | 0.449270 ± 0.000428 | 0.442509 ± 0.000383 | 0.443138 ± 0.000231 | 0.441441 ± 0.000061 | 0.442986 ± 0.000353 | 0.442253 ± 0.000402 |
| `plr` | 0.442041 ± 0.000118 | 0.445401 ± 0.000414 | 0.443345 ± 0.000712 | 0.447879 ± 0.000071 | 0.441399 ± 0.000170 | 0.442233 ± 0.000131 | 0.440966 ± 0.000210 | 0.442116 ± 0.000334 | 0.441907 ± 0.000062 |
| `narydis_lite` | 0.442849 ± 0.000209 | 0.445147 ± 0.000437 | 0.443698 ± 0.000406 | 0.449710 ± 0.000210 | 0.441860 ± 0.000270 | 0.442517 ± 0.000250 | 0.441164 ± 0.000128 | 0.442241 ± 0.000210 | 0.442581 ± 0.000237 |
| `bspline` | 0.442911 ± 0.000299 | 0.445168 ± 0.000287 | 0.443527 ± 0.000157 | 0.450131 ± 0.000349 | 0.442645 ± 0.000054 | 0.442691 ± 0.000232 | 0.441790 ± 0.000063 | 0.442420 ± 0.000250 | 0.442516 ± 0.000523 |
| `narydis_full` | 0.441497 ± 0.000117 | 0.443878 ± 0.000439 | 0.442499 ± 0.000304 | 0.446676 ± 0.000079 | 0.440660 ± 0.000172 | 0.441881 ± 0.000321 | 0.440167 ± 0.000080 | 0.441351 ± 0.000275 | 0.441353 ± 0.000059 |
| `deer` | 0.440994 ± 0.000120 | 0.443284 ± 0.000180 | 0.441748 ± 0.000255 | 0.445835 ± 0.000545 | 0.439795 ± 0.000132 | 0.441303 ± 0.000277 | 0.439497 ± 0.000103 | 0.440656 ± 0.000390 | 0.440407 ± 0.000284 |
| `dae` | 0.441417 ± 0.000170 | 0.444030 ± 0.000366 | 0.442515 ± 0.000175 | 0.446832 ± 0.000206 | 0.441167 ± 0.000389 | 0.441583 ± 0.000283 | 0.440551 ± 0.000096 | 0.441728 ± 0.000608 | 0.441838 ± 0.000275 |
| `daes_gate` | 0.441256 ± 0.000137 | 0.444034 ± 0.000071 | 0.442316 ± 0.000440 | 0.446297 ± 0.000174 | 0.440196 ± 0.000172 | 0.441533 ± 0.000090 | 0.439827 ± 0.000196 | 0.441680 ± 0.000510 | 0.440715 ± 0.000092 |
| `daes_tran` | 0.441418 ± 0.000095 | 0.445136 ± 0.000505 | 0.442441 ± 0.000074 | 0.445833 ± 0.000305 | 0.440750 ± 0.000135 | 0.441639 ± 0.000132 | 0.440124 ± 0.000221 | 0.441624 ± 0.000203 | 0.440771 ± 0.000104 |

</details>

### AutoML-A — test AUC

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.703336 ± 0.003446 | 0.707370 ± 0.002019 | 0.706175 ± 0.001165 | 0.685146 ± 0.001748 | 0.700758 ± 0.003006 | 0.707684 ± 0.002246 | 0.708278 ± 0.001743 | 0.711173 ± 0.005803 | 0.697229 ± 0.002695 |
| `linear` | 0.705240 ± 0.001421 | 0.696461 ± 0.000854 | 0.703464 ± 0.004841 | 0.688583 ± 0.001978 | 0.706207 ± 0.004381 | 0.704203 ± 0.002226 | 0.705450 ± 0.001319 | 0.707568 ± 0.000740 | 0.709862 ± 0.002711 |
| `linear_relu` | 0.703693 ± 0.000209 | 0.567508 ± 0.116927 | 0.702172 ± 0.002228 | 0.682481 ± 0.007298 | 0.704896 ± 0.002199 | 0.703206 ± 0.001321 | 0.703913 ± 0.001151 | 0.708134 ± 0.002275 | 0.708188 ± 0.001884 |
| `quantile_bucket` | 0.707641 ± 0.001230 | 0.707104 ± 0.005958 | 0.706286 ± 0.002282 | 0.690196 ± 0.002471 | 0.706535 ± 0.002121 | 0.704303 ± 0.001238 | 0.705362 ± 0.001211 | 0.710571 ± 0.004014 | 0.706151 ± 0.000897 |
| `log_squared_bucket` | 0.679897 ± 0.001925 | 0.681702 ± 0.001377 | 0.679514 ± 0.001688 | 0.663692 ± 0.002948 | 0.677516 ± 0.006175 | 0.678472 ± 0.000920 | 0.681739 ± 0.002790 | 0.684469 ± 0.002294 | 0.678911 ± 0.002319 |
| `autodis` | 0.706870 ± 0.001251 | 0.706112 ± 0.003942 | 0.706209 ± 0.004657 | 0.689761 ± 0.000930 | 0.706674 ± 0.001058 | 0.704292 ± 0.002238 | 0.708299 ± 0.002894 | 0.714041 ± 0.001422 | 0.703919 ± 0.002474 |
| `ple_a` | 0.708169 ± 0.000486 | 0.674625 ± 0.008671 | 0.701134 ± 0.004373 | 0.688798 ± 0.004708 | 0.710298 ± 0.000270 | 0.706256 ± 0.001730 | 0.704907 ± 0.001304 | 0.707676 ± 0.003776 | 0.704730 ± 0.003014 |
| `ple_b` | 0.707193 ± 0.001742 | 0.703067 ± 0.002907 | 0.704710 ± 0.002741 | 0.694622 ± 0.003544 | 0.710518 ± 0.001687 | 0.702547 ± 0.001833 | 0.709764 ± 0.001673 | 0.708146 ± 0.001381 | 0.707084 ± 0.003604 |
| `ple_relu` | 0.704312 ± 0.001368 | 0.696081 ± 0.000601 | 0.703811 ± 0.001643 | 0.687974 ± 0.002934 | 0.706635 ± 0.002298 | 0.697950 ± 0.003055 | 0.707083 ± 0.002530 | 0.704362 ± 0.004445 | 0.707339 ± 0.003763 |
| `periodic` | 0.704934 ± 0.003083 | 0.699386 ± 0.002778 | 0.701437 ± 0.002367 | 0.691888 ± 0.002069 | 0.703206 ± 0.002121 | 0.702384 ± 0.002019 | 0.706715 ± 0.004264 | 0.709340 ± 0.001097 | 0.710257 ± 0.002064 |
| `plr` | 0.707934 ± 0.002399 | 0.707838 ± 0.000939 | 0.706691 ± 0.001952 | 0.686161 ± 0.004790 | 0.704689 ± 0.005302 | 0.704325 ± 0.003416 | 0.709961 ± 0.001554 | 0.712483 ± 0.003422 | 0.705547 ± 0.001098 |
| `narydis_lite` | 0.691566 ± 0.001542 | 0.690922 ± 0.001469 | 0.689717 ± 0.000927 | 0.677858 ± 0.000313 | 0.690422 ± 0.001320 | 0.688816 ± 0.003342 | 0.690198 ± 0.001232 | 0.704371 ± 0.005774 | 0.689122 ± 0.006720 |
| `bspline` | 0.707840 ± 0.001888 | 0.710843 ± 0.002908 | 0.710007 ± 0.003084 | 0.693613 ± 0.000874 | 0.707024 ± 0.001982 | 0.706239 ± 0.001427 | 0.710201 ± 0.001872 | 0.712025 ± 0.003120 | 0.710835 ± 0.002910 |
| `narydis_full` | 0.702578 ± 0.002111 | 0.698954 ± 0.003042 | 0.700021 ± 0.001538 | 0.689624 ± 0.000492 | 0.700948 ± 0.001427 | 0.701506 ± 0.004900 | 0.700596 ± 0.001938 | 0.707459 ± 0.003359 | 0.697132 ± 0.005935 |
| `deer` | 0.709603 ± 0.001465 | 0.710274 ± 0.001234 | 0.709470 ± 0.001420 | 0.692262 ± 0.004639 | 0.709533 ± 0.001140 | 0.708541 ± 0.000571 | 0.708337 ± 0.001747 | 0.716565 ± 0.001188 | 0.706449 ± 0.001137 |
| `dae` | 0.703488 ± 0.001477 | 0.707192 ± 0.000273 | 0.707163 ± 0.000244 | 0.690807 ± 0.002163 | 0.704252 ± 0.001631 | 0.704363 ± 0.002969 | 0.705023 ± 0.000287 | 0.711751 ± 0.000665 | 0.703759 ± 0.001629 |
| `daes_gate` | 0.706819 ± 0.001054 | 0.704450 ± 0.004927 | 0.705794 ± 0.003671 | 0.688574 ± 0.010776 | 0.705217 ± 0.001369 | 0.704050 ± 0.002025 | 0.706882 ± 0.003360 | 0.710974 ± 0.003243 | 0.709580 ± 0.002484 |
| `daes_tran` | 0.704987 ± 0.001179 | 0.699814 ± 0.002841 | 0.702034 ± 0.001511 | 0.689453 ± 0.004713 | 0.702886 ± 0.004495 | 0.700992 ± 0.001012 | 0.703236 ± 0.001480 | 0.709472 ± 0.001918 | 0.707559 ± 0.002155 |

<details>
<summary>AutoML-A — test logloss (all 18 methods × 9 backbones)</summary>

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.240559 ± 0.007420 | 0.239127 ± 0.002685 | 0.238405 ± 0.001574 | 0.228581 ± 0.005201 | 0.240508 ± 0.001057 | 0.240854 ± 0.003826 | 0.243254 ± 0.001724 | 0.210391 ± 0.004834 | 0.244328 ± 0.003898 |
| `linear` | 0.244650 ± 0.010989 | 0.231880 ± 0.005398 | 0.243560 ± 0.002383 | 0.222084 ± 0.006574 | 0.247541 ± 0.006772 | 0.238748 ± 0.010744 | 0.235609 ± 0.002593 | 0.231075 ± 0.008806 | 0.238653 ± 0.005669 |
| `linear_relu` | 0.241271 ± 0.009570 | 22.780852 ± 19.501917 | 0.240432 ± 0.001156 | 0.226343 ± 0.004855 | 0.247032 ± 0.012598 | 0.238306 ± 0.009692 | 0.237952 ± 0.000660 | 0.233976 ± 0.006115 | 0.245787 ± 0.003124 |
| `quantile_bucket` | 0.241648 ± 0.007339 | 0.238265 ± 0.001300 | 0.240646 ± 0.002507 | 0.228700 ± 0.006670 | 0.238715 ± 0.009264 | 0.244777 ± 0.005085 | 0.245949 ± 0.001473 | 0.215836 ± 0.011820 | 0.240666 ± 0.015499 |
| `log_squared_bucket` | 0.242395 ± 0.010040 | 0.249027 ± 0.004013 | 0.252687 ± 0.006125 | 0.235026 ± 0.006693 | 0.252943 ± 0.009249 | 0.253516 ± 0.006964 | 0.248695 ± 0.008038 | 0.224018 ± 0.011299 | 0.249234 ± 0.015815 |
| `autodis` | 0.239710 ± 0.001593 | 0.238418 ± 0.004822 | 0.238480 ± 0.002453 | 0.230202 ± 0.004949 | 0.238115 ± 0.000780 | 0.245052 ± 0.002402 | 0.244405 ± 0.003877 | 0.211646 ± 0.005956 | 0.240828 ± 0.013446 |
| `ple_a` | 0.242215 ± 0.005314 | 0.234825 ± 0.012484 | 0.246825 ± 0.006521 | 0.216374 ± 0.003252 | 0.234944 ± 0.005120 | 0.239264 ± 0.004762 | 0.230707 ± 0.002694 | 0.241487 ± 0.011257 | 0.231753 ± 0.004043 |
| `ple_b` | 0.241211 ± 0.008066 | 0.219134 ± 0.006372 | 0.234300 ± 0.006662 | 0.222977 ± 0.005515 | 0.238081 ± 0.009275 | 0.256366 ± 0.005410 | 0.235250 ± 0.005157 | 0.246997 ± 0.005154 | 0.234807 ± 0.005108 |
| `ple_relu` | 0.241117 ± 0.006390 | 0.219514 ± 0.007391 | 0.235678 ± 0.006612 | 0.226211 ± 0.004445 | 0.237430 ± 0.010069 | 0.255113 ± 0.005309 | 0.235431 ± 0.005583 | 0.250011 ± 0.000994 | 0.238952 ± 0.008405 |
| `periodic` | 0.251252 ± 0.005036 | 0.253827 ± 0.022214 | 0.246248 ± 0.019427 | 0.227295 ± 0.010348 | 0.235289 ± 0.002249 | 0.248883 ± 0.013745 | 0.234501 ± 0.002135 | 0.240434 ± 0.006257 | 0.236148 ± 0.009744 |
| `plr` | 0.245732 ± 0.004277 | 0.238081 ± 0.008736 | 0.243684 ± 0.007669 | 0.232908 ± 0.010172 | 0.248920 ± 0.008277 | 0.248130 ± 0.001603 | 0.236712 ± 0.003991 | 0.222479 ± 0.004720 | 0.237649 ± 0.006943 |
| `narydis_lite` | 0.248084 ± 0.005235 | 0.244076 ± 0.003661 | 0.244671 ± 0.006065 | 0.230610 ± 0.007774 | 0.242850 ± 0.006401 | 0.253634 ± 0.015979 | 0.244179 ± 0.006094 | 0.215178 ± 0.009115 | 0.235125 ± 0.024886 |
| `bspline` | 0.247147 ± 0.005761 | 0.241210 ± 0.003293 | 0.242193 ± 0.004844 | 0.223732 ± 0.002268 | 0.239807 ± 0.001706 | 0.237703 ± 0.009681 | 0.246227 ± 0.005112 | 0.218331 ± 0.003135 | 0.231213 ± 0.015912 |
| `narydis_full` | 0.243617 ± 0.000910 | 0.237452 ± 0.005573 | 0.237930 ± 0.005481 | 0.224352 ± 0.006149 | 0.243878 ± 0.001358 | 0.246549 ± 0.009507 | 0.247997 ± 0.007476 | 0.213698 ± 0.008432 | 0.242310 ± 0.021133 |
| `deer` | 0.240344 ± 0.003628 | 0.241461 ± 0.006340 | 0.242319 ± 0.006794 | 0.223558 ± 0.004625 | 0.239245 ± 0.003629 | 0.239541 ± 0.009466 | 0.239401 ± 0.002803 | 0.220392 ± 0.002546 | 0.248163 ± 0.004030 |
| `dae` | 0.242000 ± 0.002330 | 0.240015 ± 0.005768 | 0.240479 ± 0.003539 | 0.223580 ± 0.003500 | 0.243477 ± 0.005390 | 0.244428 ± 0.002354 | 0.235213 ± 0.004434 | 0.215143 ± 0.011581 | 0.244982 ± 0.003941 |
| `daes_gate` | 0.241780 ± 0.017662 | 0.235381 ± 0.008762 | 0.240877 ± 0.010406 | 0.224689 ± 0.007790 | 0.248188 ± 0.009428 | 0.240681 ± 0.011809 | 0.238002 ± 0.003361 | 0.225794 ± 0.011961 | 0.234608 ± 0.003448 |
| `daes_tran` | 0.241603 ± 0.007492 | 0.238372 ± 0.006900 | 0.238202 ± 0.005826 | 0.228865 ± 0.013307 | 0.246984 ± 0.007420 | 0.246091 ± 0.003051 | 0.237607 ± 0.008027 | 0.239139 ± 0.008320 | 0.234207 ± 0.026015 |

</details>

### AutoML-E — test AUC

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.835622 ± 0.003300 | 0.839390 ± 0.000810 | 0.839473 ± 0.000717 | 0.839781 ± 0.000668 | 0.838852 ± 0.001412 | 0.835067 ± 0.003809 | 0.837451 ± 0.001908 | 0.844351 ± 0.001054 | 0.834418 ± 0.001014 |
| `linear` | 0.838334 ± 0.000461 | 0.837729 ± 0.002329 | 0.836723 ± 0.001474 | 0.837787 ± 0.000656 | 0.837883 ± 0.003206 | 0.837156 ± 0.001262 | 0.836863 ± 0.000817 | 0.839319 ± 0.001158 | 0.833880 ± 0.005006 |
| `linear_relu` | 0.838311 ± 0.000681 | 0.836947 ± 0.000833 | 0.836643 ± 0.000753 | 0.836028 ± 0.002220 | 0.838467 ± 0.002542 | 0.836998 ± 0.000966 | 0.836663 ± 0.001275 | 0.840678 ± 0.000925 | 0.833602 ± 0.006625 |
| `quantile_bucket` | 0.839906 ± 0.001739 | 0.838410 ± 0.002165 | 0.837633 ± 0.002455 | 0.839654 ± 0.000822 | 0.839679 ± 0.000460 | 0.834374 ± 0.003864 | 0.836050 ± 0.002092 | 0.842387 ± 0.000646 | 0.835801 ± 0.001491 |
| `log_squared_bucket` | 0.835553 ± 0.003309 | 0.837059 ± 0.001019 | 0.837485 ± 0.001067 | 0.838358 ± 0.000358 | 0.837408 ± 0.001028 | 0.833530 ± 0.005087 | 0.835254 ± 0.002832 | 0.840917 ± 0.000935 | 0.831431 ± 0.006053 |
| `autodis` | 0.836708 ± 0.001372 | 0.837761 ± 0.001474 | 0.838623 ± 0.000956 | 0.839719 ± 0.000189 | 0.837311 ± 0.001044 | 0.837151 ± 0.001110 | 0.838174 ± 0.000971 | 0.843076 ± 0.000995 | 0.833772 ± 0.001342 |
| `ple_a` | 0.838312 ± 0.001377 | 0.838852 ± 0.000491 | 0.838010 ± 0.001821 | 0.837373 ± 0.001204 | 0.839362 ± 0.001705 | 0.838146 ± 0.001853 | 0.839418 ± 0.000841 | 0.840231 ± 0.000313 | 0.836200 ± 0.000813 |
| `ple_b` | 0.836482 ± 0.001885 | 0.836514 ± 0.003477 | 0.836878 ± 0.001428 | 0.837637 ± 0.000377 | 0.839474 ± 0.000221 | 0.837518 ± 0.002586 | 0.838209 ± 0.000613 | 0.840653 ± 0.001421 | 0.833919 ± 0.001750 |
| `ple_relu` | 0.836972 ± 0.001348 | 0.837426 ± 0.002274 | 0.836038 ± 0.001353 | 0.838067 ± 0.000417 | 0.838409 ± 0.000728 | 0.837184 ± 0.002688 | 0.837990 ± 0.000867 | 0.839829 ± 0.001267 | 0.836497 ± 0.000795 |
| `periodic` | 0.838477 ± 0.001312 | 0.837475 ± 0.002124 | 0.837206 ± 0.001121 | 0.837319 ± 0.000277 | 0.837963 ± 0.001898 | 0.837994 ± 0.000789 | 0.837827 ± 0.002016 | 0.839927 ± 0.001018 | 0.835336 ± 0.000866 |
| `plr` | 0.838386 ± 0.002328 | 0.839002 ± 0.000576 | 0.837154 ± 0.000351 | 0.837867 ± 0.001253 | 0.838279 ± 0.000881 | 0.837727 ± 0.001141 | 0.837351 ± 0.000180 | 0.839564 ± 0.001491 | 0.830462 ± 0.004023 |
| `narydis_lite` | 0.836711 ± 0.000546 | 0.837196 ± 0.004063 | 0.837312 ± 0.003831 | 0.837969 ± 0.000469 | 0.837780 ± 0.000619 | 0.836869 ± 0.001339 | 0.836167 ± 0.002699 | 0.841395 ± 0.001230 | 0.835004 ± 0.000138 |
| `bspline` | 0.837413 ± 0.001327 | 0.836570 ± 0.002639 | 0.835899 ± 0.003786 | 0.839583 ± 0.000310 | 0.838527 ± 0.004255 | 0.836702 ± 0.002022 | 0.836619 ± 0.001439 | 0.843195 ± 0.001204 | 0.832770 ± 0.001478 |
| `narydis_full` | 0.835105 ± 0.002765 | 0.836560 ± 0.001700 | 0.836835 ± 0.001168 | 0.839177 ± 0.000974 | 0.837238 ± 0.001911 | 0.835009 ± 0.002388 | 0.836848 ± 0.000869 | 0.841406 ± 0.001589 | 0.830836 ± 0.003565 |
| `deer` | 0.837563 ± 0.000779 | 0.838559 ± 0.001718 | 0.838451 ± 0.001715 | 0.840161 ± 0.000267 | 0.837410 ± 0.001232 | 0.837511 ± 0.003607 | 0.837583 ± 0.001321 | 0.843282 ± 0.001070 | 0.835798 ± 0.002744 |
| `dae` | 0.839379 ± 0.000812 | 0.837804 ± 0.001922 | 0.837797 ± 0.001641 | 0.838900 ± 0.000497 | 0.838433 ± 0.001251 | 0.837390 ± 0.002098 | 0.836620 ± 0.000914 | 0.843259 ± 0.000257 | 0.833545 ± 0.000828 |
| `daes_gate` | 0.838621 ± 0.001229 | 0.834665 ± 0.001695 | 0.834350 ± 0.002297 | 0.838816 ± 0.001316 | 0.839916 ± 0.000383 | 0.836880 ± 0.002732 | 0.837384 ± 0.000992 | 0.842054 ± 0.001777 | 0.830449 ± 0.001373 |
| `daes_tran` | 0.837640 ± 0.001520 | 0.836814 ± 0.004712 | 0.836830 ± 0.003717 | 0.838659 ± 0.001198 | 0.837150 ± 0.001835 | 0.835971 ± 0.001147 | 0.835400 ± 0.001944 | 0.841865 ± 0.000886 | 0.832108 ± 0.001044 |

<details>
<summary>AutoML-E — test logloss (all 18 methods × 9 backbones)</summary>

| Method | DNN | DeepFM | WideDeep | NFM | PNN | DCN | DCNv2 | AutoInt | FiBiNET |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| `field` | 0.071847 ± 0.002312 | 0.070632 ± 0.001027 | 0.070511 ± 0.000889 | 0.070607 ± 0.000379 | 0.069925 ± 0.000915 | 0.070609 ± 0.000480 | 0.069648 ± 0.000935 | 0.068172 ± 0.000345 | 0.073141 ± 0.000686 |
| `linear` | 0.070793 ± 0.001002 | 0.070501 ± 0.001148 | 0.071323 ± 0.001237 | 0.071179 ± 0.000707 | 0.072514 ± 0.001851 | 0.071880 ± 0.000877 | 0.072095 ± 0.001445 | 0.070437 ± 0.000752 | 0.071854 ± 0.001767 |
| `linear_relu` | 0.070478 ± 0.000810 | 0.070710 ± 0.001131 | 0.071194 ± 0.001101 | 0.071443 ± 0.001328 | 0.071904 ± 0.001698 | 0.071597 ± 0.001398 | 0.071814 ± 0.001664 | 0.069741 ± 0.000599 | 0.072083 ± 0.001846 |
| `quantile_bucket` | 0.070250 ± 0.000961 | 0.070749 ± 0.001175 | 0.070878 ± 0.001022 | 0.070396 ± 0.000908 | 0.071177 ± 0.001517 | 0.072121 ± 0.001408 | 0.071017 ± 0.000908 | 0.069287 ± 0.000876 | 0.071507 ± 0.000260 |
| `log_squared_bucket` | 0.070505 ± 0.001137 | 0.070107 ± 0.001221 | 0.070189 ± 0.001267 | 0.070336 ± 0.000510 | 0.070934 ± 0.000789 | 0.071763 ± 0.000278 | 0.070448 ± 0.001411 | 0.068096 ± 0.000137 | 0.074116 ± 0.000896 |
| `autodis` | 0.071217 ± 0.000755 | 0.070123 ± 0.000463 | 0.069856 ± 0.000385 | 0.070270 ± 0.000781 | 0.070561 ± 0.001541 | 0.070916 ± 0.001423 | 0.070409 ± 0.000166 | 0.069064 ± 0.001561 | 0.072624 ± 0.001097 |
| `ple_a` | 0.071354 ± 0.000594 | 0.070771 ± 0.001816 | 0.071539 ± 0.001037 | 0.070546 ± 0.001399 | 0.071607 ± 0.000728 | 0.071298 ± 0.002625 | 0.071792 ± 0.001106 | 0.070224 ± 0.001837 | 0.072171 ± 0.001811 |
| `ple_b` | 0.071317 ± 0.001390 | 0.071909 ± 0.002687 | 0.071308 ± 0.001734 | 0.071126 ± 0.001012 | 0.072045 ± 0.001873 | 0.070966 ± 0.001260 | 0.071336 ± 0.000762 | 0.070199 ± 0.000892 | 0.073552 ± 0.001828 |
| `ple_relu` | 0.070610 ± 0.001033 | 0.070974 ± 0.001635 | 0.070967 ± 0.001266 | 0.071036 ± 0.000501 | 0.071415 ± 0.000747 | 0.070577 ± 0.000632 | 0.072525 ± 0.001322 | 0.071354 ± 0.000807 | 0.072452 ± 0.001089 |
| `periodic` | 0.070580 ± 0.000984 | 0.069965 ± 0.001681 | 0.071020 ± 0.001172 | 0.069688 ± 0.000435 | 0.071911 ± 0.000601 | 0.069960 ± 0.000988 | 0.071807 ± 0.001825 | 0.071667 ± 0.000826 | 0.071168 ± 0.000782 |
| `plr` | 0.070411 ± 0.000398 | 0.069611 ± 0.000542 | 0.070161 ± 0.001130 | 0.070254 ± 0.000953 | 0.070420 ± 0.000776 | 0.071462 ± 0.001364 | 0.070611 ± 0.000959 | 0.068493 ± 0.000989 | 0.074649 ± 0.001765 |
| `narydis_lite` | 0.071460 ± 0.000649 | 0.071703 ± 0.001157 | 0.071518 ± 0.000765 | 0.070916 ± 0.001394 | 0.070407 ± 0.001059 | 0.070768 ± 0.000636 | 0.069729 ± 0.001122 | 0.068163 ± 0.000335 | 0.070996 ± 0.000372 |
| `bspline` | 0.070623 ± 0.000662 | 0.070899 ± 0.000589 | 0.070819 ± 0.000669 | 0.070921 ± 0.000646 | 0.072471 ± 0.001827 | 0.070123 ± 0.000363 | 0.069866 ± 0.000567 | 0.069238 ± 0.000368 | 0.072914 ± 0.001421 |
| `narydis_full` | 0.073156 ± 0.001100 | 0.070549 ± 0.000669 | 0.070479 ± 0.000733 | 0.070909 ± 0.000543 | 0.070424 ± 0.000558 | 0.070624 ± 0.002275 | 0.070505 ± 0.001178 | 0.069612 ± 0.000501 | 0.074133 ± 0.001366 |
| `deer` | 0.070942 ± 0.000277 | 0.070067 ± 0.000597 | 0.069996 ± 0.000638 | 0.070747 ± 0.001412 | 0.071007 ± 0.001095 | 0.071721 ± 0.002474 | 0.070682 ± 0.001007 | 0.068213 ± 0.001110 | 0.072144 ± 0.000459 |
| `dae` | 0.070071 ± 0.000506 | 0.071268 ± 0.001641 | 0.071227 ± 0.000905 | 0.071284 ± 0.001318 | 0.071276 ± 0.001662 | 0.071137 ± 0.001208 | 0.070452 ± 0.000322 | 0.068482 ± 0.000617 | 0.073177 ± 0.000848 |
| `daes_gate` | 0.070580 ± 0.000373 | 0.072165 ± 0.000877 | 0.072015 ± 0.001384 | 0.071112 ± 0.001417 | 0.071053 ± 0.000448 | 0.070481 ± 0.001568 | 0.071074 ± 0.000529 | 0.068872 ± 0.001377 | 0.073949 ± 0.000889 |
| `daes_tran` | 0.070258 ± 0.000936 | 0.070161 ± 0.002248 | 0.069919 ± 0.001695 | 0.071349 ± 0.000415 | 0.070570 ± 0.000683 | 0.071071 ± 0.000568 | 0.070957 ± 0.000840 | 0.069124 ± 0.000489 | 0.073764 ± 0.001860 |

</details>

The [per-seed records](reproduction/results/tuned_accuracy/per_seed.csv) contain all 1,458 individual runs, including validation metrics, best/stopping epoch and elapsed time. The [full-precision summary](reproduction/results/tuned_accuracy/summary.csv) includes both test metrics, ranks and Field deltas. See the [protocol](docs/PROTOCOL.md) and [fidelity notes](docs/BASELINE_PROVENANCE.md) before interpreting differences as a property of the original papers.
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

To extend the published three seeds to five without rerunning or overwriting
the frozen 1,458 records, materialize a separate five-seed configuration and
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
2026–2030. Do not replace the published table with a partial run.

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

They validate 1,458 per-seed records and rebuild the 486-cell main table.
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

The snapshot was verified for structure and frozen-result analysis; a fresh
1,458-task training run has **not** been carried out after this packaging
change. The Criteo raw conversion still requires an externally provided,
validated categorical-reference Parquet dataset. Baseline fidelity boundaries
are documented in [`docs/BASELINE_PROVENANCE.md`](docs/BASELINE_PROVENANCE.md).
Original benchmark code is licensed under Apache-2.0 (`LICENSE`, `NOTICE`).
Third-party Apache-2.0 notices are retained separately in
`docs/licenses/` and source headers.
Research extensions should follow [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) so a new
method or dataset cannot silently change a frozen comparison.
