from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_container
from app.core.container import Container
from app.models.domain import ENTITY_TYPES, Entity, Mention
from app.schemas.api import (
    DecisionOut,
    EntityDetail,
    EntitySummary,
    GraphEdge,
    GraphNode,
    GraphResponse,
    MentionOut,
    RelationshipOut,
    ResolvedClusterOut,
    ResolveRequest,
    ResolveResponse,
)

router = APIRouter(prefix="/entities", tags=["entities"])


def _summary(e: Entity) -> EntitySummary:
    return EntitySummary(
        entity_id=e.id, canonical_name=e.canonical_name, entity_type=e.entity_type,
        aliases=e.aliases, mention_count=e.mention_count, document_count=e.document_count,
    )


@router.get("", response_model=list[EntitySummary])
def list_entities(
    entity_type: str | None = Query(None, description=f"One of {', '.join(ENTITY_TYPES)}"),
    q: str | None = Query(None, description="Case-insensitive substring filter on names/aliases"),
    limit: int = Query(200, ge=1, le=2000),
    container: Container = Depends(get_container),
) -> list[EntitySummary]:
    entities = container.repo.list_entities(entity_type.upper() if entity_type else None)
    if q:
        needle = q.lower()
        entities = [e for e in entities if needle in e.canonical_name.lower() or any(needle in a.lower() for a in e.aliases)]
    return [_summary(e) for e in entities[:limit]]


@router.get("/graph", response_model=GraphResponse)
def graph(
    min_weight: float = Query(0.5, ge=0, le=1),
    limit: int = Query(80, ge=1, le=500),
    container: Container = Depends(get_container),
) -> GraphResponse:
    g = container.graph
    edges = []
    for src, neighbors in g.adjacency.items():
        for dst, edge in neighbors.items():
            if src < dst and edge.weight >= min_weight:
                typed = sorted(t for t in edge.relation_types if t != "related_to") or ["related_to"]
                edges.append(GraphEdge(source=src, target=dst, type=typed[0], weight=round(edge.weight, 3)))
    edges = sorted(edges, key=lambda e: -e.weight)[:limit]
    node_ids = {e.source for e in edges} | {e.target for e in edges}
    nodes = [
        GraphNode(id=eid, label=g.entities[eid].canonical_name, type=g.entities[eid].entity_type, mentions=g.entities[eid].mention_count)
        for eid in sorted(node_ids)
    ]
    return GraphResponse(nodes=nodes, edges=edges)


@router.get("/{entity_id}", response_model=EntityDetail)
def get_entity(entity_id: str, container: Container = Depends(get_container)) -> EntityDetail:
    entity = container.repo.get_entity(entity_id)
    if entity is None:
        raise HTTPException(404, f"entity '{entity_id}' not found")
    mentions = [m for m in container.repo.all_mentions() if m.entity_id == entity_id]
    chunks = container.repo.get_chunks(sorted({m.chunk_id for m in mentions}))
    filenames: dict[str, str | None] = {}
    for doc_id in {m.document_id for m in mentions}:
        doc = container.repo.get_document(doc_id)
        filenames[doc_id] = doc.filename if doc else None

    def context(m: Mention) -> str:
        chunk = chunks.get(m.chunk_id)
        if chunk is None:
            return ""
        lo, hi = max(0, m.char_start - 80), min(len(chunk.text), m.char_end + 80)
        return ("..." if lo else "") + chunk.text[lo:hi].replace("\n", " ") + ("..." if hi < len(chunk.text) else "")

    names = {e.id: e.canonical_name for e in container.repo.list_entities()}
    relationships = [
        RelationshipOut(
            relationship_id=r.id, source_entity_id=r.source_entity_id, source_name=names.get(r.source_entity_id, "?"),
            target_entity_id=r.target_entity_id, target_name=names.get(r.target_entity_id, "?"),
            relationship_type=r.relationship_type, confidence=r.confidence,
            supporting_chunk_id=r.supporting_chunk_id, evidence=r.evidence, method=r.method,
        )
        for r in container.repo.all_relationships()
        if entity_id in (r.source_entity_id, r.target_entity_id)
    ]
    relationships.sort(key=lambda r: (-r.confidence, r.relationship_type))
    return EntityDetail(
        **_summary(entity).model_dump(),
        mentions=[
            MentionOut(
                mention_id=m.id, surface=m.surface, chunk_id=m.chunk_id, document_id=m.document_id,
                source=filenames.get(m.document_id), extractor=m.extractor, context=context(m),
            )
            for m in mentions[:200]
        ],
        relationships=relationships[:200],
    )


@router.post("/resolve", response_model=ResolveResponse)
def resolve(req: ResolveRequest | None = None, container: Container = Depends(get_container)) -> ResolveResponse:
    if req is None or not req.mentions:
        report = container.rebuild_entities()
        decisions = container.repo.resolution_decisions()
        entities = container.repo.list_entities()
        clusters = [
            ResolvedClusterOut(canonical_name=e.canonical_name, entity_type=e.entity_type, members=[e.canonical_name, *e.aliases])
            for e in entities
            if e.aliases
        ]
        return ResolveResponse(
            stats={**report.stats, "elapsed_ms": report.elapsed_ms},
            clusters=clusters,
            decisions=[DecisionOut(**asdict(d)) for d in decisions if d.decision != "distinct"][:500],
        )

    # Stateless what-if resolution over the supplied mentions only.
    for m in req.mentions:
        if m.entity_type.upper() not in ENTITY_TYPES:
            raise HTTPException(422, f"unknown entity_type '{m.entity_type}'")
    mentions = [
        Mention(
            id=f"adhoc{i}", chunk_id=f"adhoc:{m.document}", document_id=m.document, surface=m.surface,
            entity_type=m.entity_type.upper(), char_start=0, char_end=len(m.surface), extractor="api", confidence=1.0,
        )
        for i, m in enumerate(req.mentions)
    ]
    result = container.resolver.resolve(mentions)
    members: dict[str, list[str]] = defaultdict(list)
    for m in mentions:
        eid = result.assignments.get(m.id)
        if eid and m.surface not in members[eid]:
            members[eid].append(m.surface)
    clusters = [
        ResolvedClusterOut(canonical_name=e.canonical_name, entity_type=e.entity_type, members=members[e.id])
        for e in result.entities
    ]
    return ResolveResponse(
        stats=result.stats, clusters=clusters, decisions=[DecisionOut(**asdict(d)) for d in result.decisions]
    )
