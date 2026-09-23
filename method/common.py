"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .vendor import rtdl_num_embeddings as rtdl_num

class StandardizedInputEmbedding(nn.Module):
    """Apply a fixed train-set z-score before a numerical embedding.

    Several affine numerical tokenizers assume controlled input scale. The
    statistics stored in ``feature_map.json`` are computed from the training
    split, so this adapter is deterministic and does not inspect validation or
    test data. Constant fields are centered and divided by one.
    """

    def __init__(
        self,
        embedding: nn.Module,
        means: torch.Tensor,
        scales: torch.Tensor,
    ):
        super().__init__()
        if means.ndim != 1 or scales.shape != means.shape:
            raise ValueError("means and scales must be same-length vectors")
        if not torch.isfinite(means).all() or not torch.isfinite(scales).all():
            raise ValueError("standardization statistics must be finite")
        if not torch.all(scales > 0):
            raise ValueError("standardization scales must be positive")
        self.embedding = embedding
        self.register_buffer("means", means.float())
        self.register_buffer("scales", scales.float())

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.means.numel():
            raise ValueError(
                f"Expected [batch, {self.means.numel()}], got {tuple(x.shape)}"
            )
        return (x - self.means) / self.scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.standardize(x))

def _resample_bin_edges(bins: List[torch.Tensor], n_bins: int) -> torch.Tensor:
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    resampled = []
    for edges in bins:
        values = F.interpolate(
            edges.view(1, 1, -1),
            size=n_bins + 1,
            mode="linear",
            align_corners=True,
        ).view(-1)
        if not torch.all(values[1:] > values[:-1]):
            lower, upper = edges[0], edges[-1]
            values = torch.linspace(lower.item(), upper.item(), n_bins + 1)
        resampled.append(values)
    return torch.stack(resampled)

def _load_bins(
    numeric_features: Iterable[str], feature_map, device: Optional[torch.device] = None
) -> List[torch.Tensor]:
    bins = []
    for feature in numeric_features:
        edges = feature_map.features[feature].get("numeric_bin_edges")
        if edges is None:
            raise ValueError(
                f"Feature {feature!r} has no numeric_bin_edges. "
                "Run dataset/prepare_avazu_num.py first."
            )
        tensor = torch.as_tensor(edges, dtype=torch.float32, device=device)
        if tensor.ndim != 1 or len(tensor) < 2 or not torch.all(tensor[1:] > tensor[:-1]):
            raise ValueError(f"Invalid numeric_bin_edges for feature {feature!r}: {edges}")
        bins.append(tensor)
    return bins

def _load_standardization(
    numeric_features: Iterable[str], feature_map
) -> Tuple[torch.Tensor, torch.Tensor]:
    means = []
    scales = []
    for feature in numeric_features:
        spec = feature_map.features[feature]
        mean = spec.get("numeric_mean")
        std = spec.get("numeric_std")
        if mean is None or std is None:
            raise ValueError(
                f"Feature {feature!r} lacks train-set numeric_mean/numeric_std"
            )
        mean = float(mean)
        std = float(std)
        if not np.isfinite(mean) or not np.isfinite(std) or std < 0:
            raise ValueError(
                f"Invalid standardization statistics for {feature!r}: "
                f"mean={mean}, std={std}"
            )
        means.append(mean)
        scales.append(std if std > 1e-12 else 1.0)
    return torch.tensor(means), torch.tensor(scales)

def _maybe_standardize_embedding(
    module: nn.Module,
    numeric_features: Iterable[str],
    feature_map,
    enabled: bool,
) -> nn.Module:
    if not enabled:
        return module
    means, scales = _load_standardization(numeric_features, feature_map)
    return StandardizedInputEmbedding(module, means, scales)
