# CTR backbones

These are the nine native FuxiCTR model class files paired with the bundled
2.3.10 framework snapshot. Their nested paths mirror the deployed FuxiCTR
model zoo and are intentionally not rewritten. `run_expid.py` resolves model
classes by name from this directory; the numerical method is injected through
the common FuxiCTR feature layer, rather than modifying each backbone.

Supported classes: DNN, DeepFM, WideDeep, NFM, PNN, DCN, DCNv2, AutoInt,
and FiBiNET. See [`../docs/FRAMEWORK.md`](../docs/FRAMEWORK.md) for exact
paths and integration details.
