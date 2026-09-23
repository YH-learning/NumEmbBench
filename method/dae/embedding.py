"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class DistributionAwareEmbedding(nn.Module):
    """DAE reproduction using expected batch order statistics.

    During training, equally spaced order statistics from each batch update a
    running expectation of feature quantiles. Distances to those quantiles are
    transformed by a field-specific MLP into meta-embedding weights. Evaluation
    freezes the accumulated statistics. The original authors did not release
    code, so update behavior is explicit and checkpointed here.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        n_quantiles: int = 16,
        hidden_dim: int = 32,
        temperature: float = 1.0,
        momentum: Optional[float] = None,
        normalize_distances: bool = False,
    ):
        super().__init__()
        if n_quantiles < 2:
            raise ValueError("n_quantiles must be at least two")
        if hidden_dim < 1 or temperature <= 0:
            raise ValueError("Invalid DAE network settings")
        if momentum is not None and not 0 < momentum <= 1:
            raise ValueError("momentum must be in (0, 1]")
        self.n_quantiles = int(n_quantiles)
        self.temperature = float(temperature)
        self.momentum = None if momentum is None else float(momentum)
        self.normalize_distances = bool(normalize_distances)
        self.register_buffer(
            "quantile_levels", torch.linspace(0.0, 1.0, self.n_quantiles)
        )
        self.register_buffer(
            "running_quantiles", torch.zeros(n_features, self.n_quantiles)
        )
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))
        self.first_weight = nn.Parameter(
            torch.empty(n_features, self.n_quantiles, hidden_dim)
        )
        self.first_bias = nn.Parameter(torch.zeros(n_features, hidden_dim))
        self.second_weight = nn.Parameter(
            torch.empty(n_features, hidden_dim, self.n_quantiles)
        )
        self.second_bias = nn.Parameter(torch.zeros(n_features, self.n_quantiles))
        self.meta_embeddings = nn.Parameter(
            torch.empty(n_features, self.n_quantiles, d_embedding)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for field_weight in self.first_weight:
            nn.init.xavier_uniform_(field_weight)
        for field_weight in self.second_weight:
            nn.init.xavier_uniform_(field_weight)
        nn.init.zeros_(self.first_bias)
        nn.init.zeros_(self.second_bias)
        nn.init.normal_(self.meta_embeddings, std=1e-4)

    @torch.no_grad()
    def _batch_order_statistics(self, x: torch.Tensor) -> torch.Tensor:
        quantiles = torch.quantile(
            x.detach().float(), self.quantile_levels.to(x.device), dim=0
        )
        return quantiles.transpose(0, 1)

    @torch.no_grad()
    def _update_quantiles(self, batch_quantiles: torch.Tensor) -> None:
        if self.num_updates.item() == 0:
            self.running_quantiles.copy_(batch_quantiles)
        elif self.momentum is None:
            count = self.num_updates.to(dtype=batch_quantiles.dtype) + 1.0
            self.running_quantiles.add_(
                (batch_quantiles - self.running_quantiles) / count
            )
        else:
            self.running_quantiles.lerp_(batch_quantiles, self.momentum)
        self.num_updates.add_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_quantiles(self._batch_order_statistics(x))
            quantiles = self.running_quantiles
        elif self.num_updates.item() > 0:
            quantiles = self.running_quantiles
        else:
            quantiles = self._batch_order_statistics(x)
        distances = torch.abs(x.unsqueeze(-1) - quantiles.unsqueeze(0))
        if self.normalize_distances:
            scale = (quantiles[:, -1] - quantiles[:, 0]).clamp_min(1e-6)
            distances = distances / scale.view(1, -1, 1)
        hidden = F.relu(
            torch.einsum("bfq,fqh->bfh", distances, self.first_weight)
            + self.first_bias
        )
        logits = (
            torch.einsum("bfh,fhq->bfq", hidden, self.second_weight)
            + self.second_bias
        )
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return torch.einsum("bfq,fqd->bfd", weights, self.meta_embeddings)
