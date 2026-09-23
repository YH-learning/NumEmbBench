"""FuxiCTR plugin registry for the 18 frozen benchmark configurations.

The model classes live in separate method-family packages. This builder keeps
their frozen hyperparameter defaults and feature-map behavior in one auditable
place; it does not expose exploratory variants from the legacy source.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn

from fuxictr.pytorch.layers.embeddings.numerical_embedding import register_numerical_embedding

from .autodis.embedding import AutoDisEmbedding
from .bspline.embedding import BSplineEmbedding
from .common import _load_bins, _load_standardization, _maybe_standardize_embedding, _resample_bin_edges
from .dae.embedding import DistributionAwareEmbedding
from .daes.embedding import DAESEmbedding
from .deer.embedding import DEEREmbedding
from .field.embedding import FieldEmbedding
from .log_squared_bucket.embedding import LogSquaredBucketEmbedding
from .linear.embedding import LinearEmbeddings, LinearReLUEmbeddings
from .narydis.full import NaryDisEmbedding
from .narydis.lite import NaryDisLiteEmbedding
from .periodic.embedding import build_periodic_embedding
from .ple.embedding import build_piecewise_linear_embedding
from .quantile_bucket.embedding import QuantileBucketEmbedding


SUPPORTED_NUM_EMBEDDINGS = (
    "field", "linear", "linear_relu", "autodis", "piecewise_linear",
    "piecewise_linear_relu", "periodic", "plr", "narydis_lite",
    "narydis_full", "bspline", "deer", "dae", "daes_gate", "daes_tran",
    "quantile_bucket", "log_squared_bucket",
)


def build_numerical_embedding(
    method: str,
    numeric_features: List[str],
    feature_map,
    d_embedding: int,
    params: Optional[Dict] = None,
) -> nn.Module:
    """Build a frozen benchmark tokenizer with output [batch, fields, dim]."""

    params = dict(params or {})
    method = method.lower()
    if method not in SUPPORTED_NUM_EMBEDDINGS:
        raise ValueError(f"Unsupported numerical embedding {method!r}; choose from {SUPPORTED_NUM_EMBEDDINGS}")
    standardize_inputs = bool(params.get("standardize_inputs", False))
    n_features = len(numeric_features)
    if method == "field":
        return _maybe_standardize_embedding(FieldEmbedding(n_features, d_embedding), numeric_features, feature_map, standardize_inputs)
    if method == "linear":
        return _maybe_standardize_embedding(LinearEmbeddings(n_features, d_embedding), numeric_features, feature_map, standardize_inputs)
    if method == "linear_relu":
        return _maybe_standardize_embedding(LinearReLUEmbeddings(n_features, d_embedding), numeric_features, feature_map, standardize_inputs)
    if method == "autodis":
        return AutoDisEmbedding(
            n_features=n_features, d_embedding=d_embedding,
            n_meta_embeddings=int(params.get("n_meta_embeddings", 16)),
            temperature=float(params.get("temperature", 0.1)),
            residual_weight=float(params.get("residual_weight", 1.0)),
            leaky_relu_slope=float(params.get("leaky_relu_slope", 0.01)),
        )
    if method == "narydis_lite":
        return NaryDisLiteEmbedding(
            n_features=n_features, d_embedding=d_embedding,
            bases=params.get("bases", (2, 3, 5, 7, 10)),
            max_digits=int(params.get("max_digits", 12)),
            multiplier=float(params.get("multiplier", 100.0)),
            signed=bool(params.get("signed", True)),
            pooling=str(params.get("pooling", "mean")),
        )
    if method == "dae":
        momentum = params.get("momentum")
        return DistributionAwareEmbedding(
            n_features=n_features, d_embedding=d_embedding,
            n_quantiles=int(params.get("n_quantiles", 16)),
            hidden_dim=int(params.get("hidden_dim", 32)),
            temperature=float(params.get("temperature", 1.0)),
            momentum=None if momentum is None else float(momentum),
            normalize_distances=bool(params.get("normalize_distances", False)),
        )
    if method in ("daes_gate", "daes_tran"):
        context_features = params.get("context_features")
        if context_features is None:
            n_context_fields = int(params.get("n_context_fields", 3))
            categorical = [
                (name, int(spec.get("vocab_size", 0)))
                for name, spec in feature_map.features.items()
                if spec.get("type") == "categorical"
            ]
            categorical.sort(key=lambda item: (item[1], item[0]))
            context_features = [name for name, _ in categorical[:n_context_fields]]
        elif isinstance(context_features, str):
            context_features = [name.strip() for name in context_features.split(",") if name.strip()]
        missing = [
            name for name in context_features
            if name not in feature_map.features or feature_map.features[name].get("type") != "categorical"
        ]
        if missing:
            raise ValueError(f"DAES context_features must be categorical: {missing}")
        return DAESEmbedding(
            n_features=n_features, d_embedding=d_embedding,
            context_features=context_features,
            n_meta_embeddings=int(params.get("n_meta_embeddings", 10)),
            reservoir_size=int(params.get("reservoir_size", 100000)),
            beta=float(params.get("beta", 0.5)),
            modulation="gate" if method == "daes_gate" else "tran",
            quantile_refresh_interval=int(params.get("quantile_refresh_interval", 1)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
        )
    if method in ("periodic", "plr"):
        module = build_periodic_embedding(n_features, d_embedding, params, plr=method == "plr")
        return _maybe_standardize_embedding(module, numeric_features, feature_map, standardize_inputs)
    if method == "log_squared_bucket":
        vocabularies = params.get("vocabularies")
        if vocabularies is None:
            vocab_path = params.get("vocab_path")
            if not vocab_path:
                raise ValueError("log_squared_bucket requires vocab_path or vocabularies")
            vocab_path = Path(vocab_path)
            if not vocab_path.is_absolute():
                vocab_path = Path(feature_map.data_dir) / vocab_path
            with vocab_path.open() as stream:
                vocabularies = json.load(stream)
        return LogSquaredBucketEmbedding(
            numeric_features=numeric_features, vocabularies=vocabularies,
            d_embedding=d_embedding,
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
        )

    bins = _load_bins(numeric_features, feature_map)
    if method == "bspline":
        return BSplineEmbedding(
            lower=torch.stack([edges[0] for edges in bins]),
            upper=torch.stack([edges[-1] for edges in bins]),
            d_embedding=d_embedding,
            n_basis=int(params.get("n_basis", 8)),
            degree=int(params.get("degree", 3)),
        )
    if method == "narydis_full":
        return NaryDisEmbedding(
            lower=torch.stack([edges[0] for edges in bins]),
            upper=torch.stack([edges[-1] for edges in bins]),
            d_embedding=d_embedding,
            bases=params.get("bases", tuple(range(2, 11))),
            max_digits=int(params.get("max_digits", 12)),
            multiplier=float(params.get("multiplier", 100.0)),
            signed=bool(params.get("signed", True)),
            temperature=float(params.get("temperature", 1.0)),
            min_temperature=float(params.get("min_temperature", 0.1)),
            temperature_decay=float(params.get("temperature_decay", 5e-5)),
            attention_hidden_dims=params.get("attention_hidden_dims", (80, 40)),
            ssl_hidden_dim=int(params.get("ssl_hidden_dim", 64)),
            ssl_temperature=float(params.get("ssl_temperature", 0.1)),
            ssl_weight=float(params.get("ssl_weight", 0.0)),
            ssl_max_samples=int(params.get("ssl_max_samples", 512)),
            mask_zero_digits=bool(params.get("mask_zero_digits", True)),
        )
    if method == "deer":
        n_bins = int(params.get("n_bins", 16))
        boundary_scheme = str(params.get("boundary_scheme", "quantile")).lower()
        if boundary_scheme == "quantile":
            initial_edges = _resample_bin_edges(bins, n_bins)
        elif boundary_scheme == "fixed":
            min_value = float(params.get("min_value", 0.0))
            max_value = float(params.get("max_value", 1.0))
            if max_value <= min_value:
                raise ValueError("max_value must be greater than min_value")
            initial_edges = torch.linspace(min_value, max_value, n_bins + 1).repeat(n_features, 1)
        else:
            raise ValueError("boundary_scheme must be either 'quantile' or 'fixed'")
        noise = params.get("boundary_init_noise")
        module = DEEREmbedding(
            initial_edges=initial_edges, d_embedding=d_embedding,
            min_width=float(params.get("min_width", 1e-4)),
            width_temperature=float(params.get("width_temperature", 1.0)),
            embedding_init_std=float(params.get("embedding_init_std", 1e-4)),
            boundary_init_noise=None if noise is None else float(noise),
        )
        module.set_execution_mode(str(params.get("deer_execution_mode", "auto")))
        return module
    if standardize_inputs:
        means, scales = _load_standardization(numeric_features, feature_map)
        bins = [(edges - means[index]) / scales[index] for index, edges in enumerate(bins)]
    if method == "quantile_bucket":
        return _maybe_standardize_embedding(
            QuantileBucketEmbedding(bins, d_embedding,
                                    embedding_init_std=float(params.get("embedding_init_std", 1e-4))),
            numeric_features, feature_map, standardize_inputs,
        )
    module = build_piecewise_linear_embedding(
        bins, d_embedding, params, relu=method == "piecewise_linear_relu"
    )
    return _maybe_standardize_embedding(module, numeric_features, feature_map, standardize_inputs)


def _registry_builder(method: str):
    def builder(*, numeric_features, feature_map, d_embedding, params):
        return build_numerical_embedding(method, numeric_features, feature_map, d_embedding, params)
    return builder


for _method in SUPPORTED_NUM_EMBEDDINGS:
    register_numerical_embedding(_method, _registry_builder(_method), override=True)
