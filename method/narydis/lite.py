"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import _to_nary_digits, _normalized_digit_weights

class NaryDisLiteEmbedding(nn.Module):
    """Efficient multi-base N-ary encoding inspired by NaryDis and MNSE.

    This engineering variant keeps deterministic hybrid-base encoding and
    field/base-specific positional embeddings, but replaces hierarchical
    attention and contrastive regularization with normalized pooling.
    """

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        bases: Sequence[int] = (2, 3, 5, 7, 10),
        max_digits: int = 12,
        multiplier: float = 100.0,
        signed: bool = True,
        pooling: str = "mean",
    ):
        super().__init__()
        self.bases = tuple(int(base) for base in bases)
        if not self.bases or any(base < 2 for base in self.bases):
            raise ValueError("bases must contain integers greater than one")
        if pooling not in ("mean", "sum"):
            raise ValueError("pooling must be 'mean' or 'sum'")
        self.max_digits = int(max_digits)
        self.multiplier = float(multiplier)
        self.signed = bool(signed)
        self.pooling = pooling
        sequence_length = self.max_digits + int(self.signed)
        self.positional_embeddings = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(n_features, sequence_length, d_embedding)
                )
                for _ in self.bases
            ]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embeddings in self.positional_embeddings:
            nn.init.normal_(embeddings, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ary_representations = []
        for base, embeddings in zip(self.bases, self.positional_embeddings):
            digits = _to_nary_digits(
                x, base, self.max_digits, self.multiplier, self.signed
            )
            weights = _normalized_digit_weights(digits)
            ary_representations.append(
                torch.einsum("bfl,fld->bfd", weights, embeddings)
            )
        stacked = torch.stack(ary_representations, dim=2)
        if self.pooling == "sum":
            return stacked.sum(dim=2)
        return stacked.mean(dim=2)
