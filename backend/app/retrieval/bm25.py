"""Okapi BM25 over pre-analyzed terms (used by the in-memory backend).

idf uses the non-negative Lucene variant ln(1 + (N - df + 0.5) / (df + 0.5)),
the same formula the PostgreSQL backend evaluates in SQL.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class BM25Index:
    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.avgdl = 1.0
        self.df: dict[str, int] = {}
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._doc_terms: list[set[str]] = []
        self._index_of: dict[str, int] = {}

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def build(self, documents: list[tuple[str, list[str]]]) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_ids, self.doc_len, self._doc_terms = [], [], []
        for idx, (doc_id, terms) in enumerate(documents):
            counts = Counter(terms)
            self.doc_ids.append(doc_id)
            self.doc_len.append(len(terms))
            self._doc_terms.append(set(counts))
            for term, tf in counts.items():
                postings[term].append((idx, tf))
        self._postings = dict(postings)
        self.df = {term: len(plist) for term, plist in self._postings.items()}
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 1.0
        self._index_of = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def _term_score(self, tf: int, dl: int) -> float:
        denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
        return tf * (self.k1 + 1.0) / denom

    def _accumulate(self, terms: list[str]) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for term in dict.fromkeys(terms):
            plist = self._postings.get(term)
            if not plist:
                continue
            idf = self.idf(term)
            for idx, tf in plist:
                scores[idx] += idf * self._term_score(tf, self.doc_len[idx])
        return scores

    def search(self, terms: list[str], k: int) -> list[tuple[str, float]]:
        scores = self._accumulate(terms)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [(self.doc_ids[idx], score) for idx, score in ranked]

    def scores_for(self, terms: list[str], doc_ids: list[str]) -> dict[str, float]:
        wanted = {self._index_of[d] for d in doc_ids if d in self._index_of}
        scores = self._accumulate(terms)
        return {self.doc_ids[i]: scores.get(i, 0.0) for i in wanted}
