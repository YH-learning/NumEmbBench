"""Dependency-free checks that split method classes retain frozen source ASTs."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "reproduction/legacy/source/numerical_embeddings.py"
MAPPING = {
    "FieldEmbedding": "method/field/embedding.py",
    "StandardizedInputEmbedding": "method/common.py",
    "AutoDisEmbedding": "method/autodis/embedding.py",
    "QuantileBucketEmbedding": "method/quantile_bucket/embedding.py",
    "LogSquaredBucketEmbedding": "method/log_squared_bucket/embedding.py",
    "NaryDisLiteEmbedding": "method/narydis/lite.py",
    "BSplineEmbedding": "method/bspline/embedding.py",
    "NaryDisEmbedding": "method/narydis/full.py",
    "DEEREmbedding": "method/deer/embedding.py",
    "DistributionAwareEmbedding": "method/dae/embedding.py",
    "DAESEmbedding": "method/daes/embedding.py",
}


def definition(path: Path, name: str) -> ast.AST:
    return next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)


class SplitSourceTests(unittest.TestCase):
    def test_benchmark_classes_match_frozen_ast(self):
        for name, relative in MAPPING.items():
            with self.subTest(name=name):
                original = definition(LEGACY, name)
                extracted = definition(ROOT / relative, name)
                self.assertEqual(ast.dump(original), ast.dump(extracted))


if __name__ == "__main__":
    unittest.main()
