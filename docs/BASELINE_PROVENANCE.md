# Baseline provenance and fidelity boundaries

This repository opens the benchmark-side implementations and adapters used
to produce the reported matrices. It does **not** claim that every file is an
official implementation released by the corresponding paper authors.

| Benchmark label | Source used in this artifact | Fidelity boundary |
|---|---|---|
| Field, Linear, Linear-ReLU | Field: local PyTorch implementation; Linear variants: RTDL adapters | Deterministic baselines with the common 16-dimensional output interface. |
| Quantile, PLE-A, PLE-B, PLE-ReLU, Periodic, PLR | Thin adapters around `rtdl-num-embeddings==0.0.12` | Upstream package provides the numerical embedding primitives; the artifact provides FuxiCTR integration and common-shape adapters. |
| AutoDis | Local vectorized PyTorch reimplementation | Implements soft meta-embedding assignment described in the KDD 2021 paper; it is not copied from an official author repository. |
| NaryDis-Lite | Local engineering variant | Preserves multi-base digit encoding but intentionally removes hierarchical attention and contrastive regularization. It is always reported separately from NaryDis. |
| NaryDis | Local paper-aligned PyTorch reimplementation | Implements hierarchical intra/inter-base aggregation and exposes the auxiliary contrastive loss. |
| DEER | Local paper-aligned PyTorch reimplementation | Implements learnable ordered intervals and interpolation. No official repository was used. |
| DAE | Local paper-aligned PyTorch reimplementation | Implements distribution-aware soft assignment from streaming batch statistics. |
| DAES-Gate, DAES-Tran | Local paper-aligned PyTorch reimplementation | Implements quantile-space encoding and contextual gate/transform modulation. The paper does not fully specify categorical-context pooling; this artifact mean-pools the selected field embeddings and documents that choice in code. |
| B-spline, LogBucket | Local PyTorch implementation | Standard spline basis and the documented Criteo log-squared vocabulary transform, respectively. |

## Framework and upstream dependencies

- FuxiCTR supplies data loading, categorical embedding, backbone, optimization,
  and metric infrastructure. The deployed FuxiCTR 2.3.10 Python package and
  nine selected backbone source files are bundled under their original
  Apache-2.0 notices. The exact deployed tree has no verified Git commit.
- `reproduction/legacy/integration/fuxictr_2_3_10/feature_embedding.py` retains the original
  FuxiCTR/Huawei Apache-2.0 header; the bundled counterpart is byte-identical.
- The actual `rtdl-num-embeddings` 0.0.12 implementation is vendored under
  `method/vendor/` with its Apache-2.0 notice. A global RTDL installation is unnecessary.
- PyTorch, NumPy, pandas, scikit-learn, PyArrow, and Matplotlib remain governed
  by their upstream licenses.

## Release boundary

Dataset archives and model checkpoints are intentionally excluded. Per-seed
outcomes, analysis code, configuration generators, implementation snapshots,
and SHA-256 hashes are included. Original benchmark code is Apache-2.0 under
the root `LICENSE`; third-party packages retain their own notices and are not
relicensed by the benchmark notice.
