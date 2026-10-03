"""Retrieval engine: candidate generation -> scoring -> fusion/rerank -> confidence gate.

    query
      |-- lexical retriever  (BM25)                 top-N candidates
      |-- vector retriever   (cosine, MiniLM)       top-N candidates
      '-- entity retriever   (linked entities + 1-hop expansion)   [hybrid_entity]
                     |
            union of candidates; every active signal is computed for every
            candidate (no "missing = 0" artefacts from list truncation)
                     |
            per-query normalization (vector, lexical); entity score is absolute
                     |
            reranker (weighted alpha/beta/gamma, or RRF)  ->  top-k
                     |
            confidence estimator  ->  success | insufficient_evidence

Every result carries its component scores and full provenance.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.config import RETRIEVAL_MODES, Settings, SignalWeights
from app.core.text import lexical_terms, query_terms
from app.db.repository import Repository
from app.embeddings.base import Embedder
from app.entities.graph import EntityGraph
from app.entities.linking import LinkedEntity, LinkResult, QueryEntityLinker
from app.models.domain import Chunk, Document, ScoredCandidate
from app.retrieval.confidence import (
    ABSTAIN_MESSAGE,
    ConfidenceEstimator,
    ConfidenceInputs,
    ConfidenceReport,
)
from app.retrieval.entity_scorer import EntityPlan, EntityScorer
from app.retrieval.normalization import normalize
from app.retrieval.reranker import Reranker, RRFReranker, WeightedReranker


@dataclass
class EntityContext:
    graph: EntityGraph
    linker: QueryEntityLinker
    scorer: EntityScorer


@dataclass
class RankedResult:
    rank: int
    chunk_id: str
    document_id: str
    filename: str
    source_type: str
    title: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    metadata: dict[str, Any]
    vector_score: float | None
    lexical_score: float | None
    entity_score: float | None
    vector_similarity: float | None
    lexical_raw: float | None
    final_score: float
    retrieved_by: list[str]
    matched_entities: list[str]


@dataclass
class SearchOutcome:
    query: str
    mode: str
    status: str
    confidence: ConfidenceReport
    results: list[RankedResult]
    weights: SignalWeights
    query_entities: list[LinkedEntity] = field(default_factory=list)
    unknown_entities: list[str] = field(default_factory=list)
    candidates_considered: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def abstained(self) -> bool:
        return self.status != "success"

    @property
    def message(self) -> str | None:
        return ABSTAIN_MESSAGE if self.abstained else None


def idf_weighted_coverage(q_terms: list[str], chunk_terms: set[str], df: dict[str, int], n_docs: int) -> float:
    if not q_terms:
        return 0.0
    weights = {t: math.log(1.0 + (n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5)) for t in q_terms}
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return sum(w for t, w in weights.items() if t in chunk_terms) / total


class RetrievalEngine:
    def __init__(
        self,
        repo: Repository,
        embedder: Embedder,
        settings: Settings,
        entity_context: Callable[[], EntityContext | None],
        reranker: Reranker | None = None,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.settings = settings
        self._entity_context = entity_context
        if reranker is None:
            r = settings.retrieval
            reranker = RRFReranker(r.rrf_k) if r.fusion == "rrf" else WeightedReranker()
        self.reranker = reranker
        self.confidence = ConfidenceEstimator(settings.confidence)

    def default_weights(self, mode: str) -> SignalWeights:
        r = self.settings.retrieval
        if mode == "lexical_only":
            return SignalWeights(lexical=1.0)
        if mode == "vector_only":
            return SignalWeights(vector=1.0)
        if mode == "hybrid":
            return SignalWeights(vector=r.hybrid_weights.vector, lexical=r.hybrid_weights.lexical)
        return r.hybrid_entity_weights

    @staticmethod
    def _mask(weights: SignalWeights, mode: str) -> SignalWeights:
        vector = weights.vector if mode != "lexical_only" else 0.0
        lexical = weights.lexical if mode != "vector_only" else 0.0
        entity = weights.entity if mode == "hybrid_entity" else 0.0
        if vector + lexical + entity <= 0:
            raise ValueError(f"weights {weights} leave no active signal for mode {mode}")
        return SignalWeights(vector=vector, lexical=lexical, entity=entity)

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        mode: str = "hybrid_entity",
        top_k: int | None = None,
        weights: SignalWeights | None = None,
        threshold: float | None = None,
        candidate_pool: int | None = None,
    ) -> SearchOutcome:
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown mode '{mode}', expected one of {RETRIEVAL_MODES}")
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        top_k = top_k or self.settings.retrieval.default_top_k
        pool = candidate_pool or self.settings.retrieval.candidate_pool
        weights = self._mask(weights or self.default_weights(mode), mode)
        uses_vector = weights.vector > 0
        uses_lexical = weights.lexical > 0
        uses_entity = weights.entity > 0
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        def lap(name: str, t0: float) -> float:
            now = time.perf_counter()
            timings[name] = round((now - t0) * 1000, 3)
            return now

        candidates: dict[str, ScoredCandidate] = {}

        def add(hits: list[tuple[str, float]], source: str) -> None:
            for cid, score in hits:
                cand = candidates.setdefault(cid, ScoredCandidate(cid))
                cand.sources.add(source)
                setattr(cand, f"{source}_raw", score)

        t = time.perf_counter()
        lexical_hits: list[tuple[str, float]] = []
        if uses_lexical:
            lexical_hits = self.repo.lexical_search(query, pool)
            add(lexical_hits, "lexical")
            t = lap("lexical_ms", t)

        embedding = None
        vector_hits: list[tuple[str, float]] = []
        if uses_vector:
            embedding = self.embedder.embed([query])[0]
            t = lap("embed_ms", t)
            vector_hits = self.repo.vector_search(embedding, pool)
            add(vector_hits, "vector")
            t = lap("vector_ms", t)

        links = LinkResult()
        plan = EntityPlan()
        ctx = self._entity_context() if uses_entity else None
        if ctx is not None:
            links = ctx.linker.link(query)
            plan = ctx.scorer.plan(links, query)
            add(ctx.scorer.candidates(plan, pool), "entity")
            t = lap("entity_ms", t)

        # Fill in every active signal for every candidate.
        ids = sorted(candidates)
        if uses_lexical:
            missing = [cid for cid in ids if candidates[cid].lexical_raw is None]
            for cid, score in self.repo.lexical_scores(query, missing).items():
                candidates[cid].lexical_raw = score
            for cid in missing:
                if candidates[cid].lexical_raw is None:
                    candidates[cid].lexical_raw = 0.0
        if uses_vector and embedding is not None:
            missing = [cid for cid in ids if candidates[cid].vector_raw is None]
            for cid, score in self.repo.vector_scores(embedding, missing).items():
                candidates[cid].vector_raw = score
        if ctx is not None:
            for cid, (score, matched) in ctx.scorer.score_chunks(plan, ids).items():
                candidates[cid].entity_raw = score
                candidates[cid].entity_score = score
                candidates[cid].matched_entities = matched
        t = lap("scoring_ms", t)

        method = self.settings.retrieval.normalization
        if uses_vector:
            normed = normalize({c: v.vector_raw for c, v in candidates.items() if v.vector_raw is not None}, method)
            for cid, value in normed.items():
                candidates[cid].vector_score = value
        if uses_lexical:
            normed = normalize({c: v.lexical_raw or 0.0 for c, v in candidates.items()}, method)
            for cid, value in normed.items():
                candidates[cid].lexical_score = value

        ranked = self.reranker.rerank(query, list(candidates.values()), {}, weights)
        top = ranked[:top_k]
        t = lap("rerank_ms", t)

        chunks = self.repo.get_chunks([c.chunk_id for c in top])
        documents: dict[str, Document | None] = {}
        for chunk in chunks.values():
            if chunk.document_id not in documents:
                documents[chunk.document_id] = self.repo.get_document(chunk.document_id)

        # A named span in the query is "unknown" if some of its terms occur nowhere in the corpus.
        unknown = [span for span in links.unmatched if self.repo.unknown_terms(span)] if ctx else []

        report = self.confidence.evaluate(
            self._confidence_inputs(query, mode, top, chunks, lexical_hits, vector_hits, plan, ctx, unknown, uses_lexical),
            threshold=threshold,
        )
        t = lap("confidence_ms", t)

        results = [self._to_result(i + 1, c, chunks, documents) for i, c in enumerate(top) if c.chunk_id in chunks]
        timings["total_ms"] = round((time.perf_counter() - t_start) * 1000, 3)
        return SearchOutcome(
            query=query,
            mode=mode,
            status=report.status,
            confidence=report,
            results=results,
            weights=weights.normalized(),
            query_entities=links.linked,
            unknown_entities=unknown,
            candidates_considered=len(candidates),
            timings_ms=timings,
        )

    # ----------------------------------------------------------------- helpers
    def _confidence_inputs(
        self,
        query: str,
        mode: str,
        top: list[ScoredCandidate],
        chunks: dict[str, Chunk],
        lexical_hits: list[tuple[str, float]],
        vector_hits: list[tuple[str, float]],
        plan: EntityPlan,
        ctx: EntityContext | None,
        unknown: list[str],
        uses_lexical: bool,
    ) -> ConfidenceInputs:
        coverage: list[float | None] = [None] * len(top)
        if uses_lexical and top:
            q_terms = query_terms(query)
            df, n_docs = self.repo.term_document_frequencies(q_terms) if q_terms else ({}, 0)
            for i, cand in enumerate(top[:3]):
                chunk = chunks.get(cand.chunk_id)
                if chunk is not None:
                    coverage[i] = idf_weighted_coverage(q_terms, set(lexical_terms(chunk.search_text)), df, n_docs)
        entities_in_top: set[str] = set()
        if ctx is not None:
            for cand in top[:3]:
                entities_in_top |= ctx.graph.chunk_entities.get(cand.chunk_id, set())
        return ConfidenceInputs(
            mode=mode,
            final_scores=[c.final_score for c in top],
            top_vector_raw=[c.vector_raw for c in top],
            top_coverage=coverage,
            top_documents=[chunks[c.chunk_id].document_id if c.chunk_id in chunks else "" for c in top],
            lexical_top=[cid for cid, _ in lexical_hits] if mode in ("hybrid", "hybrid_entity") else [],
            vector_top=[cid for cid, _ in vector_hits] if mode in ("hybrid", "hybrid_entity") else [],
            top_chunk_ids=[c.chunk_id for c in top],
            linked_entities=plan.direct,
            entities_in_top=entities_in_top,
            unknown_entities=unknown,
        )

    @staticmethod
    def _to_result(
        rank: int, c: ScoredCandidate, chunks: dict[str, Chunk], documents: dict[str, Document | None]
    ) -> RankedResult:
        chunk = chunks[c.chunk_id]
        doc = documents.get(chunk.document_id)

        def r(x: float | None) -> float | None:
            return None if x is None else round(x, 4)

        return RankedResult(
            rank=rank,
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=doc.filename if doc else "",
            source_type=doc.source_type if doc else "",
            title=doc.title if doc else "",
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            metadata=chunk.metadata,
            vector_score=r(c.vector_score),
            lexical_score=r(c.lexical_score),
            entity_score=r(c.entity_score),
            vector_similarity=r(c.vector_raw),
            lexical_raw=r(c.lexical_raw),
            final_score=round(c.final_score, 4),
            retrieved_by=sorted(c.sources),
            matched_entities=c.matched_entities,
        )
