from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_container
from app.core.config import SignalWeights
from app.core.container import Container
from app.retrieval.engine import SearchOutcome
from app.schemas.api import (
    AnswerRequest,
    AnswerResponse,
    CitationOut,
    ConfidenceOut,
    LinkedEntityOut,
    SearchRequest,
    SearchResponse,
    SearchResultOut,
)

router = APIRouter(tags=["search"])

_LOCATION_KEYS = ("section", "heading_path", "page_start", "page_end", "json_path", "record_index")


def run_search(req: SearchRequest, container: Container) -> SearchOutcome:
    weights = None
    if req.weights is not None:
        try:
            weights = SignalWeights(**req.weights.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if container.repo.counts()["chunks"] == 0:
        raise HTTPException(409, "the index is empty; upload documents first")
    try:
        return container.engine.search(req.query, req.mode, req.top_k, weights, req.threshold)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def to_response(req: SearchRequest, outcome: SearchOutcome, container: Container) -> SearchResponse:
    show = (not outcome.abstained) or req.include_evidence_on_abstain
    results = [
        SearchResultOut(
            rank=r.rank,
            text=r.text,
            source=r.filename,
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            chunk_index=r.chunk_index,
            title=r.title,
            source_type=r.source_type,
            char_start=r.char_start,
            char_end=r.char_end,
            location={k: r.metadata[k] for k in _LOCATION_KEYS if k in r.metadata},
            retrieval_mode=outcome.mode,
            vector_score=r.vector_score,
            lexical_score=r.lexical_score,
            entity_score=r.entity_score,
            final_score=r.final_score,
            vector_similarity=r.vector_similarity,
            lexical_raw=r.lexical_raw,
            retrieved_by=r.retrieved_by,
            matched_entities=r.matched_entities,
        )
        for r in outcome.results
    ] if show else []
    threshold = req.threshold if req.threshold is not None else container.settings.confidence.threshold
    w = outcome.weights
    return SearchResponse(
        query=outcome.query,
        mode=outcome.mode,
        status=outcome.status,
        confidence=outcome.confidence.confidence,
        message=outcome.message,
        results=results,
        confidence_detail=ConfidenceOut(
            score=outcome.confidence.confidence,
            threshold=threshold,
            signals=outcome.confidence.signals,
            reasons=outcome.confidence.reasons,
            gates_triggered=outcome.confidence.gates_triggered,
        ),
        weights={"alpha_vector": round(w.vector, 4), "beta_lexical": round(w.lexical, 4), "gamma_entity": round(w.entity, 4)},
        query_entities=[
            LinkedEntityOut(
                entity_id=e.entity_id, canonical_name=e.canonical_name, entity_type=e.entity_type,
                surface=e.surface, confidence=e.confidence, method=e.method,
            )
            for e in outcome.query_entities
        ],
        unknown_entities=outcome.unknown_entities,
        candidates_considered=outcome.candidates_considered,
        timings_ms=outcome.timings_ms,
    )


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, container: Container = Depends(get_container)) -> SearchResponse:
    return to_response(req, run_search(req, container), container)


@router.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, container: Container = Depends(get_container)) -> AnswerResponse:
    outcome = run_search(req, container)
    result = container.answers.answer(outcome)
    return AnswerResponse(
        query=outcome.query,
        status=result.status,
        answer=result.answer,
        message=result.message,
        confidence=outcome.confidence.confidence,
        generator=result.generator,
        citations=[
            CitationOut(marker=e.marker, chunk_id=e.chunk_id, document_id=e.document_id, source=e.source, text=e.text)
            for e in result.cited
        ],
        unsupported_citations=result.unsupported_markers,
        search=to_response(req, outcome, container),
    )
