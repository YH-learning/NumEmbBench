"""Frozen benchmark implementation, split verbatim from the experiment source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class LogSquaredBucketEmbedding(nn.Module):
    """Criteo log-squared bucketing followed by exact train-vocabulary lookup.

    The transform matches the RecZoo/FuxiCTR Criteo_x4 recipe: integer values
    greater than two become floor(log(x) squared); all other integers are kept.
    Each transformed token is then mapped through a field-specific vocabulary.
    Non-positive, rare, unseen, and non-finite values use that field's OOV id.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        vocabularies: Dict[str, Dict[str, int]],
        d_embedding: int,
        embedding_init_std: float = 1e-4,
    ):
        super().__init__()
        self.numeric_features = tuple(numeric_features)
        if embedding_init_std <= 0:
            raise ValueError("embedding_init_std must be positive")
        self.embeddings = nn.ModuleList()
        for i, feature in enumerate(self.numeric_features):
            if feature not in vocabularies:
                raise ValueError(f"Missing vocabulary for numeric feature {feature!r}")
            vocabulary = vocabularies[feature]
            if "__OOV__" not in vocabulary:
                raise ValueError(f"Vocabulary for {feature!r} has no __OOV__ id")
            token_to_id = {
                int(token): int(index)
                for token, index in vocabulary.items()
                if token not in {"__PAD__", "__OOV__"} and int(token) > 0
            }
            min_token = min(token_to_id, default=0)
            max_token = max(token_to_id, default=0)
            oov_id = int(vocabulary["__OOV__"])
            lookup = torch.full(
                (max_token - min_token + 1,), oov_id, dtype=torch.long
            )
            for token, index in token_to_id.items():
                lookup[token - min_token] = index
            self.register_buffer(f"lookup_{i}", lookup)
            self.register_buffer(
                f"min_token_{i}", torch.tensor(min_token, dtype=torch.long)
            )
            self.register_buffer(
                f"oov_id_{i}", torch.tensor(oov_id, dtype=torch.long)
            )
            vocab_size = max(int(index) for index in vocabulary.values()) + 1
            table = nn.Embedding(vocab_size, d_embedding, padding_idx=0)
            nn.init.normal_(table.weight, std=embedding_init_std)
            with torch.no_grad():
                table.weight[0].zero_()
            self.embeddings.append(table)

    @staticmethod
    def bucket_tokens(x: torch.Tensor) -> torch.Tensor:
        safe = torch.nan_to_num(
            x.detach().double(), nan=0.0, posinf=0.0, neginf=0.0
        )
        integer = torch.trunc(safe).long()
        positive = integer > 2
        transformed = torch.floor(
            torch.log(integer.clamp_min(1).double()).square()
        ).long()
        return torch.where(positive, transformed, integer)

    def bucket_indices(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != len(self.numeric_features):
            raise ValueError(
                f"Expected [batch, {len(self.numeric_features)}], got {tuple(x.shape)}"
            )
        tokens = self.bucket_tokens(x)
        indices = []
        for i in range(len(self.numeric_features)):
            lookup = getattr(self, f"lookup_{i}")
            min_token = getattr(self, f"min_token_{i}")
            oov_id = getattr(self, f"oov_id_{i}")
            positions = tokens[:, i] - min_token
            valid = (positions >= 0) & (positions < lookup.numel())
            safe_positions = positions.clamp(0, lookup.numel() - 1)
            mapped = lookup[safe_positions]
            indices.append(torch.where(valid, mapped, oov_id))
        return torch.stack(indices, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices = self.bucket_indices(x)
        return torch.stack(
            [table(indices[:, i]) for i, table in enumerate(self.embeddings)],
            dim=1,
        )
