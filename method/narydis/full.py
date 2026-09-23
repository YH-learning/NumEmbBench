"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import _to_nary_digits

class NaryDisEmbedding(nn.Module):
    """Paper-aligned NaryDis with hierarchical attention and InfoNCE.

    Equations (3)--(10) of Chen et al. (CIKM 2022) are implemented: gated
    intra-ary pooling, field/value-aware inter-ary attention, and an optional
    contrastive regularizer exposed through ``get_auxiliary_loss``.
    """

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        d_embedding: int,
        bases: Sequence[int] = tuple(range(2, 11)),
        max_digits: int = 12,
        multiplier: float = 100.0,
        signed: bool = True,
        temperature: float = 1.0,
        min_temperature: float = 0.1,
        temperature_decay: float = 5e-5,
        attention_hidden_dims: Sequence[int] = (80, 40),
        ssl_hidden_dim: int = 64,
        ssl_temperature: float = 0.1,
        ssl_weight: float = 0.0,
        ssl_max_samples: int = 512,
        mask_zero_digits: bool = True,
    ):
        super().__init__()
        self.bases = tuple(int(base) for base in bases)
        if len(self.bases) < 2 or any(base < 2 for base in self.bases):
            raise ValueError("narydis_full requires at least two valid bases")
        if temperature <= 0 or min_temperature <= 0:
            raise ValueError("NaryDis temperatures must be positive")
        if ssl_temperature <= 0 or ssl_weight < 0:
            raise ValueError("Invalid NaryDis SSL settings")
        self.max_digits = int(max_digits)
        self.multiplier = float(multiplier)
        self.signed = bool(signed)
        self.initial_temperature = float(temperature)
        self.min_temperature = float(min_temperature)
        self.temperature_decay = float(temperature_decay)
        self.ssl_temperature = float(ssl_temperature)
        self.ssl_weight = float(ssl_weight)
        self.ssl_max_samples = int(ssl_max_samples)
        if self.ssl_max_samples < 0:
            raise ValueError("ssl_max_samples must be non-negative")
        self.mask_zero_digits = bool(mask_zero_digits)
        self.register_buffer("lower", lower.float())
        self.register_buffer("upper", upper.float())
        self.register_buffer("training_steps", torch.zeros((), dtype=torch.long))
        n_features = len(lower)
        sequence_length = self.max_digits + int(self.signed)
        self.positional_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(n_features, sequence_length, d_embedding)
                )
                for _ in self.bases
            ]
        )
        self.intra_ary_gates = nn.ParameterList(
            [
                nn.Parameter(torch.empty(n_features, sequence_length))
                for _ in self.bases
            ]
        )
        self.field_vectors = nn.Parameter(torch.empty(n_features, d_embedding))
        hidden_dims = tuple(int(value) for value in attention_hidden_dims)
        if len(hidden_dims) != 2 or min(hidden_dims) < 1:
            raise ValueError("attention_hidden_dims must contain two positive values")
        self.inter_ary_attention = nn.Sequential(
            nn.Linear(3 * d_embedding, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )
        self.ssl_projector = nn.Sequential(
            nn.Linear(d_embedding, ssl_hidden_dim),
            nn.ReLU(),
            nn.Linear(ssl_hidden_dim, ssl_hidden_dim),
            nn.ReLU(),
            nn.Linear(ssl_hidden_dim, ssl_hidden_dim),
        )
        self._last_auxiliary_loss: Optional[torch.Tensor] = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embeddings in self.positional_embeddings:
            nn.init.normal_(embeddings, std=1e-4)
        for gates in self.intra_ary_gates:
            nn.init.ones_(gates)
        nn.init.normal_(self.field_vectors, std=1e-4)
        for module in (self.inter_ary_attention, self.ssl_projector):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def _current_temperature(self) -> float:
        annealed = self.initial_temperature - self.temperature_decay * int(
            self.training_steps.item()
        )
        return max(self.min_temperature, annealed)

    def _intra_ary_representations(self, x: torch.Tensor) -> torch.Tensor:
        representations = []
        temperature = self._current_temperature()
        for base, embeddings, gates in zip(
            self.bases, self.positional_embeddings, self.intra_ary_gates
        ):
            digits = _to_nary_digits(
                x, base, self.max_digits, self.multiplier, self.signed
            )
            logits = digits * gates.unsqueeze(0) / temperature
            if self.mask_zero_digits:
                active = digits > 0
                empty = ~active.any(dim=-1, keepdim=True)
                fallback = torch.zeros_like(active)
                fallback[..., 0] = True
                active = active | (empty & fallback)
                logits = logits.masked_fill(~active, torch.finfo(logits.dtype).min)
            weights = torch.softmax(logits, dim=-1)
            representations.append(
                torch.einsum("bfl,fld->bfd", weights, embeddings)
            )
        return torch.stack(representations, dim=2)

    def _contrastive_loss(self, ary_embeddings: torch.Tensor) -> torch.Tensor:
        if self.ssl_max_samples and ary_embeddings.size(0) > self.ssl_max_samples:
            sample = torch.randperm(
                ary_embeddings.size(0), device=ary_embeddings.device
            )[: self.ssl_max_samples]
            ary_embeddings = ary_embeddings[sample]
        pair = torch.randperm(len(self.bases), device=ary_embeddings.device)[:2]
        first = self.ssl_projector(ary_embeddings[:, :, pair[0], :])
        second = self.ssl_projector(ary_embeddings[:, :, pair[1], :])
        first = F.normalize(first, dim=-1)
        second = F.normalize(second, dim=-1)
        logits = torch.einsum("bfd,cfd->fbc", first, second)
        logits = logits / self.ssl_temperature
        batch_size, n_features = first.shape[:2]
        labels = torch.arange(batch_size, device=first.device).repeat(n_features)
        return F.cross_entropy(logits.reshape(n_features * batch_size, batch_size), labels)

    def get_auxiliary_loss(self) -> torch.Tensor:
        if self._last_auxiliary_loss is None:
            return self.field_vectors.new_zeros(())
        return self._last_auxiliary_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ary_embeddings = self._intra_ary_representations(x)
        normalized_x = (
            (x - self.lower) / (self.upper - self.lower).clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        value_aware = normalized_x.unsqueeze(-1) * self.field_vectors.unsqueeze(0)
        expanded_value = value_aware.unsqueeze(2).expand_as(ary_embeddings)
        attention_input = torch.cat(
            [ary_embeddings, expanded_value, ary_embeddings * expanded_value], dim=-1
        )
        scores = self.inter_ary_attention(attention_input).squeeze(-1)
        weights = torch.softmax(scores, dim=2)
        output = (weights.unsqueeze(-1) * ary_embeddings).sum(dim=2)
        if self.training and self.ssl_weight > 0:
            self._last_auxiliary_loss = self.ssl_weight * self._contrastive_loss(
                ary_embeddings
            )
        else:
            self._last_auxiliary_loss = output.new_zeros(())
        if self.training:
            self.training_steps.add_(1)
        return output
