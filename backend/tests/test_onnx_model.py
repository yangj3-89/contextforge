"""Checks against the real all-MiniLM-L6-v2 ONNX model (skipped if not downloaded)."""

import numpy as np
import pytest

from app.core.config import Settings
from app.embeddings.onnx_minilm import OnnxMiniLMEmbedder, model_files_present

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def embedder():
    model_dir = Settings().model_dir
    if not model_files_present(model_dir):
        pytest.skip("ONNX model not downloaded (python scripts/download_model.py)")
    return OnnxMiniLMEmbedder(model_dir, auto_download=False)


def test_matches_sentence_transformers_reference(embedder):
    # Reference value published for sentence-transformers/all-MiniLM-L6-v2.
    v = embedder.embed(["A man is eating food.", "A man is eating a piece of bread."])
    assert float(v[0] @ v[1]) == pytest.approx(0.7553, abs=1e-3)


def test_embeddings_are_normalized_and_batched(embedder):
    texts = [f"sentence {i}" for i in range(70)]
    v = embedder.embed(texts)
    assert v.shape == (70, 384)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert embedder.embed([]).shape == (0, 384)
