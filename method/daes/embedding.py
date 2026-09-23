"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class DAESEmbedding(nn.Module):
    """Paper-aligned DAES reproduction for streaming numerical features.

    The module implements quantile-space thermometer encoding and the gate/tran
    modulation equations from Liu et al. (arXiv:2602.03223). FuxiCTR consumes
    streams in batches, so the reservoir is updated with an exact batch-boundary
    merge: conditional on the previous uniform reservoir, a hypergeometric draw
    selects how many current-batch rows enter the new uniform reservoir. This
    has the same reservoir distribution at batch boundaries as item-wise
    reservoir sampling while avoiding Python work per sample.

    The paper selects three categorical fields but does not specify how their
    embeddings are combined into the single context vector used by its
    equations. This reproduction mean-pools the selected field embeddings.
    """

    requires_context_embeddings = True

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        context_features: Sequence[str],
        n_meta_embeddings: int = 10,
        reservoir_size: int = 100000,
        beta: float = 0.5,
        modulation: str = "gate",
        quantile_refresh_interval: int = 1,
        embedding_init_std: float = 1e-4,
    ):
        super().__init__()
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if n_meta_embeddings < 2:
            raise ValueError("n_meta_embeddings must be at least two")
        if reservoir_size < n_meta_embeddings + 1:
            raise ValueError(
                "reservoir_size must exceed the number of meta-embeddings"
            )
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if modulation not in {"gate", "tran"}:
            raise ValueError("modulation must be 'gate' or 'tran'")
        if quantile_refresh_interval < 1:
            raise ValueError("quantile_refresh_interval must be positive")
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")

        self.n_features = int(n_features)
        self.d_embedding = int(d_embedding)
        self.context_features = tuple(str(name) for name in context_features)
        self.n_meta_embeddings = int(n_meta_embeddings)
        self.reservoir_size = int(reservoir_size)
        self.beta = float(beta)
        self.modulation = modulation
        self.quantile_refresh_interval = int(quantile_refresh_interval)

        self.meta_embeddings = nn.Parameter(
            torch.empty(
                self.n_features,
                self.n_meta_embeddings,
                self.d_embedding,
            )
        )
        self.default_context = nn.Parameter(torch.empty(self.d_embedding))
        if self.modulation == "gate":
            self.modulation_weight = nn.Parameter(
                torch.empty(
                    self.n_features,
                    self.n_meta_embeddings,
                    self.d_embedding,
                )
            )
        else:
            self.modulation_weight = nn.Parameter(
                torch.empty(
                    self.n_features,
                    self.n_meta_embeddings,
                    self.n_meta_embeddings,
                    self.d_embedding,
                )
            )

        self.register_buffer(
            "reservoir",
            torch.zeros(self.reservoir_size, self.n_features),
        )
        self.register_buffer("num_seen", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "cached_quantiles",
            torch.zeros(self.n_features, self.n_meta_embeddings + 1),
        )
        self.register_buffer(
            "quantiles_initialized", torch.zeros((), dtype=torch.bool)
        )
        self.register_buffer(
            "batches_since_refresh", torch.zeros((), dtype=torch.long)
        )
        self.reset_parameters(float(embedding_init_std))

    def reset_parameters(self, embedding_init_std: float = 1e-4) -> None:
        nn.init.normal_(self.meta_embeddings, std=embedding_init_std)
        nn.init.normal_(self.default_context, std=embedding_init_std)
        if self.modulation == "gate":
            for weight in self.modulation_weight:
                nn.init.xavier_uniform_(weight)
        else:
            for feature_index in range(self.n_features):
                nn.init.xavier_uniform_(
                    self.modulation_weight[feature_index].reshape(
                        self.n_meta_embeddings * self.n_meta_embeddings,
                        self.d_embedding,
                    )
                )

    @torch.no_grad()
    def _update_reservoir(self, x: torch.Tensor) -> bool:
        """Merge one batch into an exactly uniform fixed-size reservoir."""

        batch = x.detach().float()
        batch_size = int(batch.size(0))
        seen = int(self.num_seen.item())
        position = 0
        changed = False

        if seen < self.reservoir_size:
            take = min(self.reservoir_size - seen, batch_size)
            if take:
                self.reservoir[seen : seen + take].copy_(batch[:take])
                seen += take
                position += take
                changed = True

        remaining = batch_size - position
        if remaining:
            selected = int(
                np.random.hypergeometric(
                    ngood=remaining,
                    nbad=seen,
                    nsample=self.reservoir_size,
                )
            )
            if selected:
                new_rows = np.random.choice(
                    remaining, size=selected, replace=False
                )
                old_slots = np.random.choice(
                    self.reservoir_size, size=selected, replace=False
                )
                new_rows = torch.as_tensor(
                    new_rows + position,
                    dtype=torch.long,
                    device=batch.device,
                )
                old_slots = torch.as_tensor(
                    old_slots,
                    dtype=torch.long,
                    device=batch.device,
                )
                self.reservoir.index_copy_(
                    0, old_slots, batch.index_select(0, new_rows)
                )
                changed = True
            seen += remaining

        self.num_seen.fill_(seen)
        return changed

    @torch.no_grad()
    def _batch_quantiles(self, samples: torch.Tensor) -> torch.Tensor:
        levels = torch.linspace(
            0.0,
            1.0,
            self.n_meta_embeddings + 1,
            device=samples.device,
        )
        return torch.quantile(
            samples.detach().float(), levels, dim=0
        ).transpose(0, 1)

    @torch.no_grad()
    def _refresh_quantiles(self) -> None:
        filled = min(int(self.num_seen.item()), self.reservoir_size)
        if filled < 1:
            return
        self.cached_quantiles.copy_(
            self._batch_quantiles(self.reservoir[:filled])
        )
        self.quantiles_initialized.fill_(True)
        self.batches_since_refresh.zero_()

    def _quantile_coordinates(
        self, x: torch.Tensor, quantiles: torch.Tensor
    ) -> torch.Tensor:
        """Equation (8): cumulative thermometer encoding in quantile space."""

        lower = quantiles[:, 0]
        upper = quantiles[:, -1]
        clipped = torch.maximum(torch.minimum(x, upper), lower)
        interval = (
            clipped.unsqueeze(-1) >= quantiles[:, 1:-1].unsqueeze(0)
        ).sum(dim=-1)
        expanded = quantiles.unsqueeze(0).expand(x.size(0), -1, -1)
        left = expanded.gather(2, interval.unsqueeze(-1)).squeeze(-1)
        right = expanded.gather(
            2, (interval + 1).unsqueeze(-1)
        ).squeeze(-1)
        width = right - left
        fraction = torch.where(
            width.abs() > 1e-12,
            (clipped - left) / width.clamp_min(1e-12),
            torch.zeros_like(clipped),
        ).clamp(0.0, 1.0)
        positions = torch.arange(
            self.n_meta_embeddings, device=x.device
        ).view(1, 1, -1)
        raw = (positions < interval.unsqueeze(-1)).to(x.dtype)
        raw = raw + (
            positions == interval.unsqueeze(-1)
        ).to(x.dtype) * fraction.unsqueeze(-1)
        return raw

    def _context_vector(
        self,
        x: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if not context_embeddings:
            return self.default_context.unsqueeze(0).expand(x.size(0), -1)
        vectors = []
        for feature in self.context_features:
            embedding = context_embeddings.get(feature)
            if embedding is None:
                continue
            if embedding.ndim != 2 or embedding.size(0) != x.size(0):
                raise ValueError(
                    f"Invalid context embedding shape for {feature!r}: "
                    f"{tuple(embedding.shape)}"
                )
            if embedding.size(1) != self.d_embedding:
                raise ValueError(
                    f"Context embedding {feature!r} has dim "
                    f"{embedding.size(1)}, expected {self.d_embedding}"
                )
            vectors.append(embedding)
        if not vectors:
            return self.default_context.unsqueeze(0).expand(x.size(0), -1)
        return torch.stack(vectors, dim=1).mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        context_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if self.training:
            changed = self._update_reservoir(x)
            self.batches_since_refresh.add_(1)
            if (
                changed
                and (
                    not bool(self.quantiles_initialized.item())
                    or self.batches_since_refresh.item()
                    >= self.quantile_refresh_interval
                )
            ):
                self._refresh_quantiles()

        if bool(self.quantiles_initialized.item()):
            quantiles = self.cached_quantiles
        else:
            quantiles = self._batch_quantiles(x)

        raw = self._quantile_coordinates(x, quantiles)
        context = self._context_vector(x, context_embeddings)
        if self.modulation == "gate":
            modulation = torch.sigmoid(
                torch.einsum(
                    "bd,fmd->bfm", context, self.modulation_weight
                )
            )
            transformed = raw * modulation
        else:
            transformed = torch.sigmoid(
                torch.einsum(
                    "bd,fijd,bfj->bfi",
                    context,
                    self.modulation_weight,
                    raw,
                )
            )
        weights = self.beta * transformed + (1.0 - self.beta) * raw
        return torch.einsum(
            "bfm,fmd->bfd", weights, self.meta_embeddings
        )
