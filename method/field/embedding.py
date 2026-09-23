"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class FieldEmbedding(nn.Module):
    """FuxiCTR's historical numeric path: one vector scaled by each value."""

    def __init__(self, n_features: int, d_embedding: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_embedding))
        nn.init.normal_(self.weight, std=1e-4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight
