"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def _to_nary_digits(
    x: torch.Tensor,
    base: int,
    max_digits: int,
    multiplier: float,
    signed: bool,
) -> torch.Tensor:
    """Convert values to little-endian base-n digits without preprocessing state."""

    if base < 2:
        raise ValueError("All N-ary bases must be at least 2")
    if max_digits < 1:
        raise ValueError("max_digits must be positive")
    safe_x = torch.nan_to_num(
        x.detach() * multiplier,
        nan=0.0,
        posinf=9.0e18,
        neginf=-9.0e18,
    ).clamp(min=-9.0e18, max=9.0e18)
    integer = torch.round(safe_x).long()
    remainder = integer.abs()
    digits = []
    for _ in range(max_digits):
        digits.append(torch.remainder(remainder, base))
        remainder = torch.div(remainder, base, rounding_mode="floor")
    encoded = torch.stack(digits, dim=-1).to(dtype=x.dtype)
    if signed:
        encoded = torch.cat(
            [encoded, (integer < 0).to(dtype=x.dtype).unsqueeze(-1)], dim=-1
        )
    return encoded

def _normalized_digit_weights(digits: torch.Tensor) -> torch.Tensor:
    denominator = digits.sum(dim=-1, keepdim=True)
    weights = digits / denominator.clamp_min(1.0)
    fallback = torch.zeros_like(weights)
    fallback[..., 0] = 1.0
    return torch.where(denominator > 0, weights, fallback)
