"""Periodic/PLR adapter around the frozen RTDL 0.0.12 implementation."""

from __future__ import annotations

from typing import Mapping

from torch import nn

from ..vendor.rtdl_num_embeddings import PeriodicEmbeddings


def build_periodic_embedding(
    n_features: int,
    d_embedding: int,
    params: Mapping,
    *,
    plr: bool,
) -> nn.Module:
    module = PeriodicEmbeddings(
        n_features=n_features,
        d_embedding=d_embedding,
        n_frequencies=int(params.get("n_frequencies", 16)),
        frequency_init_scale=float(params.get("frequency_init_scale", 0.05)),
        activation=plr,
        lite=bool(params.get("lite", False)),
    )
    embedding_init_std = params.get("embedding_init_std", 1e-4 if plr else None)
    if embedding_init_std is not None:
        embedding_init_std = float(embedding_init_std)
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        nn.init.normal_(module.linear.weight, std=embedding_init_std)
        if module.linear.bias is not None:
            nn.init.zeros_(module.linear.bias)
    return module
