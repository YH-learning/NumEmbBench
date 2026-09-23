# Numerical embedding methods

The maintained method implementations are in one family per directory:

| Family | File(s) | Evaluated names |
|---|---|---|
| Field | `field/embedding.py` | `field` |
| Linear | `linear/embedding.py` | `linear`, `linear_relu` |
| AutoDis | `autodis/embedding.py` | `autodis` |
| Quantile bucket | `quantile_bucket/embedding.py` | `quantile_bucket` |
| Log-squared bucket | `log_squared_bucket/embedding.py` | `log_squared_bucket` |
| Piecewise linear | `ple/embedding.py` | `ple_a`, `ple_b`, `ple_relu` |
| Periodic | `periodic/embedding.py` | `periodic`, `plr` |
| NaryDis | `narydis/lite.py`, `narydis/full.py` | `narydis_lite`, `narydis_full` |
| B-spline | `bspline/embedding.py` | `bspline` |
| DEER | `deer/embedding.py`, `deer/engineering.py` | `deer` |
| DAE | `dae/embedding.py` | `dae` |
| DAES | `daes/embedding.py` | `daes_gate`, `daes_tran` |
| Frozen RTDL dependency | `vendor/rtdl_num_embeddings.py` | imported by linear, PLE, and periodic adapters |

`registry.py` translates frozen YAML parameters into constructors. Common
normalization and bin-edge utilities live in `common.py`. The upstream RTDL
implementation remains under its original license; see
[`../docs/licenses/`](../docs/licenses/) and the detailed
[`../docs/METHODS.md`](../docs/METHODS.md).
