"""FuxiCTR plugin entry point for the modular benchmark methods."""

from method.registry import SUPPORTED_NUM_EMBEDDINGS, build_numerical_embedding

__all__ = ["SUPPORTED_NUM_EMBEDDINGS", "build_numerical_embedding"]
