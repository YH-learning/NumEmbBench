# Numerical embedding implementations

`method/` is the maintained benchmark implementation tree. Each family has
its own directory and source file; `method/registry.py` constructs modules
from the frozen YAML parameters and registers them with the bundled FuxiCTR
feature layer through `method/numemb_plugin.py`. Every module emits a tensor of
shape `[batch, numerical_fields, embedding_dim]`.

| Frozen experiment label | FuxiCTR `numeric_embedding` | Implementation |
|---|---|---|
| `field` | `field` | `method/field/embedding.py` |
| `linear`, `linear_relu` | Same names | `method/linear/embedding.py` (frozen RTDL source adapter) |
| `quantile_bucket` | Same name | `method/quantile_bucket/embedding.py` |
| `log_squared_bucket` | Same name | `method/log_squared_bucket/embedding.py` |
| `autodis` | Same name | `method/autodis/embedding.py` |
| `ple_a`, `ple_b` | `piecewise_linear`, versions A/B | `method/ple/embedding.py` (frozen RTDL source adapter) |
| `ple_relu` | `piecewise_linear_relu` | `method/ple/embedding.py` |
| `periodic`, `plr` | Same names | `method/periodic/embedding.py` (frozen RTDL source adapter) |
| `narydis_lite` | Same name | `method/narydis/lite.py` |
| `narydis_full` | Same name | `method/narydis/full.py` |
| `bspline` | Same name | `method/bspline/embedding.py` |
| `deer` | Same name | `method/deer/embedding.py`; interpolation kernels in the same directory |
| `dae` | Same name | `method/dae/embedding.py` |
| `daes_gate`, `daes_tran` | Same names | `method/daes/embedding.py`, gate/transform selected by the registry |

Common normalization and bin-edge helpers live in `method/common.py`;
NaryDis digit helpers live in `method/narydis/common.py`. The 11 extracted
class definitions are AST-identical to the frozen implementation in
`reproduction/legacy/source/numerical_embeddings.py`; the test in
`tests/test_method_sources.py` enforces this. The new registry supports only
the 17 implementation names needed by the **18 evaluated configurations**.
Exploratory variants are excluded from this benchmark release and are not
silently exposed as baselines.

DAES uses the categorical context fields specified by the frozen YAML. In
this benchmark reproduction, selected categorical embeddings are mean-pooled
to form its context vector, because the published description does not fully
specify that pooling step. NaryDis-Lite is a separate engineering variant,
not the paper-aligned NaryDis reproduction. See
[`BASELINE_PROVENANCE.md`](BASELINE_PROVENANCE.md) for all fidelity limits.
