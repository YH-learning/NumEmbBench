"""PLE adapter around the frozen RTDL 0.0.12 implementation."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from ..vendor.rtdl_num_embeddings import PiecewiseLinearEmbeddings


def build_piecewise_linear_embedding(
    bins: Sequence[torch.Tensor],
    d_embedding: int,
    params: Mapping,
    *,
    relu: bool,
) -> nn.Module:
    return PiecewiseLinearEmbeddings(
        bins=bins,
        d_embedding=d_embedding,
        activation=relu,
        version=str(params.get("version", "B")),
    )
