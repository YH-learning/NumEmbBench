"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class QuantileBucketEmbedding(nn.Module):
    """Hard quantile-bucket lookup baseline with feature-specific tables."""

    def __init__(
        self,
        bins: List[torch.Tensor],
        d_embedding: int,
        embedding_init_std: float = 1e-4,
    ):
        super().__init__()
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(len(edges) - 1, d_embedding) for edges in bins]
        )
        for table in self.embeddings:
            nn.init.normal_(table.weight, std=embedding_init_std)
        for i, edges in enumerate(bins):
            self.register_buffer(f"edges_{i}", edges.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, table in enumerate(self.embeddings):
            edges = getattr(self, f"edges_{i}")
            index = torch.bucketize(x[:, i].contiguous(), edges[1:-1], right=False)
            outputs.append(table(index))
        return torch.stack(outputs, dim=1)
