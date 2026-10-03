"""Deterministic feature-hashing embedder.

Not a semantic model: it is a dependency-free stand-in used by unit tests and
as an explicit opt-in (``CF_EMBEDDING_BACKEND=hashing``) when no model can be
downloaded. Word unigrams and character trigrams are hashed into a fixed number
of buckets, so texts sharing vocabulary have positive cosine similarity.
"""

from __future__ import annotations

import hashlib

import numpy as np

from app.core.text import lexical_terms
from app.embeddings.base import l2_normalize


class HashingEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dim, 1.0 if (value >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in lexical_terms(text):
                idx, sign = self._bucket("w:" + term)
                out[row, idx] += 2.0 * sign
                padded = f"#{term}#"
                for i in range(len(padded) - 2):
                    idx, sign = self._bucket("c:" + padded[i : i + 3])
                    out[row, idx] += 0.5 * sign
        return l2_normalize(out)
