"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class AutoDisEmbedding(nn.Module):
    """Vectorized AutoDis for multiple numerical fields.

    For field ``j``, the module computes
    ``h=LeakyReLU(x*w)``, ``p=softmax((h@W+alpha*h)/tau)`` and ``p@M``.
    ``M`` is the field-specific table of meta-embeddings.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        n_meta_embeddings: int = 16,
        temperature: float = 0.1,
        residual_weight: float = 1.0,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        if n_meta_embeddings < 2:
            raise ValueError("n_meta_embeddings must be at least 2")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.residual_weight = float(residual_weight)
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
        self.proj_weight = nn.Parameter(torch.empty(n_features, n_meta_embeddings))
        self.proj_matrix = nn.Parameter(
            torch.empty(n_features, n_meta_embeddings, n_meta_embeddings)
        )
        self.meta_embeddings = nn.Parameter(
            torch.empty(n_features, n_meta_embeddings, d_embedding)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj_weight)
        for matrix in self.proj_matrix:
            nn.init.xavier_uniform_(matrix)
        nn.init.normal_(self.meta_embeddings, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.leaky_relu(x.unsqueeze(-1) * self.proj_weight)
        logits = torch.einsum("fnm,bfm->bfn", self.proj_matrix, hidden)
        logits = logits + self.residual_weight * hidden
        assignment = torch.softmax(logits / self.temperature, dim=-1)
        return torch.einsum("bfn,fnd->bfd", assignment, self.meta_embeddings)
