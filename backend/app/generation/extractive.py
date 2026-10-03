"""Model-free baseline generator: returns the evidence sentences most similar to the query.

It cannot hallucinate (every sentence is copied from evidence and cited), which
makes it a useful default and a reference point for LLM-based generation.
"""

from __future__ import annotations

import numpy as np

from app.core.text import split_sentences
from app.embeddings.base import Embedder
from app.generation.base import Evidence


class ExtractiveGenerator:
    name = "extractive"

    def __init__(self, embedder: Embedder, max_sentences: int = 3, min_similarity: float = 0.2) -> None:
        self.embedder = embedder
        self.max_sentences = max_sentences
        self.min_similarity = min_similarity

    def generate(self, query: str, evidence: list[Evidence]) -> str:
        candidates: list[tuple[int, str]] = []
        for e in evidence:
            for sentence in split_sentences(e.text):
                if len(sentence.split()) >= 4:
                    candidates.append((e.marker, sentence))
        if not candidates:
            return "INSUFFICIENT_EVIDENCE"
        vectors = self.embedder.embed([query] + [s for _, s in candidates])
        sims = vectors[1:] @ vectors[0]
        order = np.argsort(-sims, kind="stable")
        picked: list[tuple[int, str]] = []
        for idx in order[: self.max_sentences]:
            if sims[idx] >= self.min_similarity or not picked:
                picked.append(candidates[int(idx)])
        return " ".join(f"{sentence} [{marker}]" for marker, sentence in picked)
