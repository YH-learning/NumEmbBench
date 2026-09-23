"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class BSplineEmbedding(nn.Module):
    """B-spline function-basis encoding followed by learned coefficients.

    Inputs are clipped to each feature's training range, mapped to ``[0, 1]``,
    expanded with a clamped uniform B-spline basis, and aggregated with a
    field-specific coefficient embedding table.
    """

    def __init__(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        d_embedding: int,
        n_basis: int = 8,
        degree: int = 3,
    ):
        super().__init__()
        self.n_basis = int(n_basis)
        self.degree = int(degree)
        if self.degree < 0:
            raise ValueError("degree must be non-negative")
        if self.n_basis <= self.degree:
            raise ValueError("n_basis must be greater than degree")
        self.register_buffer("lower", lower.float())
        self.register_buffer("upper", upper.float())
        n_internal = self.n_basis - self.degree - 1
        internal = (
            torch.linspace(0.0, 1.0, n_internal + 2)[1:-1]
            if n_internal > 0
            else torch.empty(0)
        )
        knots = torch.cat(
            [
                torch.zeros(self.degree + 1),
                internal,
                torch.ones(self.degree + 1),
            ]
        )
        self.register_buffer("knots", knots)
        self.coefficients = nn.Parameter(
            torch.empty(len(lower), self.n_basis, d_embedding)
        )
        nn.init.normal_(self.coefficients, std=1e-4)

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        denominator = (self.upper - self.lower).clamp_min(1e-12)
        unit_x = ((x - self.lower) / denominator).clamp(0.0, 1.0)
        expanded = unit_x.unsqueeze(-1)
        basis = (
            (expanded >= self.knots[:-1]) & (expanded < self.knots[1:])
        ).to(dtype=x.dtype)
        for order in range(1, self.degree + 1):
            left_denominator = self.knots[order:-1] - self.knots[: -order - 1]
            right_denominator = self.knots[order + 1 :] - self.knots[1:-order]
            left_scale = torch.where(
                left_denominator > 0,
                (expanded - self.knots[: -order - 1])
                / left_denominator.clamp_min(1e-12),
                torch.zeros_like(left_denominator),
            )
            right_scale = torch.where(
                right_denominator > 0,
                (self.knots[order + 1 :] - expanded)
                / right_denominator.clamp_min(1e-12),
                torch.zeros_like(right_denominator),
            )
            basis = left_scale * basis[..., :-1] + right_scale * basis[..., 1:]
        endpoint = torch.zeros_like(basis)
        endpoint[..., -1] = 1.0
        return torch.where((unit_x >= 1.0).unsqueeze(-1), endpoint, basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfn,fnd->bfd", self.basis(x), self.coefficients)
