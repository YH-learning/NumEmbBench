# Contributing to NumEmb Benchmark

Keep the frozen paper records immutable. New methods, preprocessing choices,
or backbone changes must have a new experiment version and a separate output
directory; do not replace `reproduction/results/` or modify the frozen
YAML in place.

## Adding a numerical embedding

1. Implement a PyTorch module that accepts the numerical fields in the order
   supplied by the FuxiCTR feature map and returns a finite tensor of shape
   `[batch, numerical_fields, embedding_dim]`.
2. Put its implementation in a dedicated `method/<family>/` package, register
   its builder in `method/registry.py`, and document the
   input range, initialization, trainable parameters, and any auxiliary loss.
3. Give the method a separate YAML experiment ID. Do not overwrite an
   existing method label with changed semantics.
4. Add unit tests for shape, finite forward/backward gradients, zero and
   missing values, and CPU execution. Check category/context routing if used.
5. Start with one dataset/backbone/seed and `--dry-run`; then run the full
   planned matrix using a new `--run-name`. Keep logs and manifests with the
   reported results.

## Source and evaluation checks

```bash
python -m unittest discover -s reproduction/tests -v
python reproduction/tools/check_environment.py --allow-missing-packages
python reproduction/tools/verify_manifest.py
python reproduction/analysis/verify_key_metrics.py
```

After changing any static release file, regenerate and verify the checksum
manifest with `python reproduction/tools/build_manifest.py` followed by
`python reproduction/tools/verify_manifest.py`. The manifest intentionally excludes
generated figures, analysis outputs, caches, and training outputs.

When reporting a new run, identify the dataset version, source commit or
snapshot checksum, configuration directory, seeds, early-stopping metric,
hardware, successful/failed task counts, and whether values came from fresh
training or frozen records. Baseline names do not imply official author code;
see `BASELINE_PROVENANCE.md` before making fidelity claims.
