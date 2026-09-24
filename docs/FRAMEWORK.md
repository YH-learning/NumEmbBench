# FuxiCTR integration and source provenance

This repository is intentionally self-contained at the **source-code** level:
`fuxictr/` is the FuxiCTR Python package deployed for the experiments, and
`backbone/` contains the nine exact backbone class files. The deployed
package reports `fuxictr.__version__ == 2.3.10`. The source directory had no
Git metadata, so there is **no verified upstream commit ID**. Do not replace
this snapshot with an arbitrary FuxiCTR tag and assume identical behavior.

## How the pieces connect

1. `backbone/run_expid.py` loads one experiment from a materialized YAML configuration.
2. `FeatureMap.load(..., params)` attaches `numeric_embedding`, its parameters,
   and plugin module names to the feature map.
3. `backbone/run_expid.py` loads the specified backbone class from `backbone/`.
4. The backbone constructs the bundled FuxiCTR `FeatureEmbedding` layer.
5. `fuxictr/pytorch/layers/embeddings/feature_embedding.py` groups numeric
   features and invokes the registry in the adjacent `numerical_embedding.py`.
6. The `method/numemb_plugin.py` entry point imports `method/registry.py`, which
   registers the 17 implementation names used by 18 frozen configurations.
   Each class lives in its own `method/<family>/` package and emits
   `[batch, numeric_fields, embedding_dim]`.
7. The numerical and categorical embeddings enter the same native backbone.

The copies in `reproduction/legacy/integration/fuxictr_2_3_10/` document the two integration
points. `reproduction/tools/check_environment.py` checks that those copies match the
bundled FuxiCTR files byte-for-byte. The RTDL 0.0.12 numerical primitives are
also included under `method/vendor/`; the benchmark always imports this frozen
copy, independent of a globally installed RTDL package.

## Which backbone files are bundled?

| Backbone | Source file |
|---|---|
| DNN | `backbone/DNN/DNN.py` |
| DeepFM | `backbone/DeepFM/DeepFM.py` |
| WideDeep | `backbone/WideDeep/WideDeep.py` |
| NFM | `backbone/NFM/NFM.py` |
| PNN | `backbone/PNN/PNN.py` |
| DCN | `backbone/DCN/DCN.py` |
| DCNv2 | `backbone/DCNv2/DCNv2.py` |
| AutoInt | `backbone/AutoInt/AutoInt.py` |
| FiBiNET | `backbone/FiBiNET/FiBiNET.py` |

The bundled framework is the supported route. If a project must use an
**external** FuxiCTR checkout, pin it to 2.3.10 and compare the *entire*
framework and backbone snapshot, not merely the two named integration files:
the deployed copy may contain additional local fixes. External-checkout
compatibility has not been regression-tested by this release. Keep the
framework's original Apache-2.0 notices when redistributing it.

## Verify the integration

```bash
python reproduction/tools/check_environment.py --allow-missing-packages  # source-layout audit
python reproduction/tools/check_environment.py --require-cuda           # full training environment
```

The first command is suitable for an analysis-only laptop. The second checks
Python packages and CUDA. A pass checks presence and snapshot integrity, not
numerical equivalence of a newly trained model. Use an actual smoke run before
the full matrix.
