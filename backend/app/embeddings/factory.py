from __future__ import annotations

from app.core.config import Settings
from app.embeddings.base import Embedder


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        from app.embeddings.hashing import HashingEmbedder

        return HashingEmbedder(settings.embedding_dim)
    if settings.embedding_backend == "sentence-transformers":
        from app.embeddings.sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(settings.embedding_model)
    from app.embeddings.onnx_minilm import OnnxMiniLMEmbedder

    return OnnxMiniLMEmbedder(settings.model_dir, auto_download=settings.model_auto_download)
