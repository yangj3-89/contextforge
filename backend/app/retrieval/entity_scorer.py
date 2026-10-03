"""Entity-aware retrieval signal (the gamma term in hybrid_entity).

Given the entities linked in the query:

* **direct coverage** - fraction of (confidence-weighted) query entities that a
  chunk mentions;
* **relationship expansion** - entities one hop away in the relationship graph
  (e.g. the project a queried person works on) contribute
  ``link_confidence * edge_weight * decay`` when a chunk mentions them.
  ``decay`` is ``hinted_hop_decay`` when the query names the neighbor's type
  ("the *project* led by Alice Chen", "the *company* Hannah runs") and
  ``hop_decay`` otherwise: the type word marks the neighbor as the bridge the
  question is really about.

    entity_score = 1 - (1 - direct_coverage) * (1 - expansion)

Scores are absolute in [0, 1] (not normalized per query), so a chunk that
only touches a weak 1-hop neighbor can never outrank one that mentions the
queried entity itself on this signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import EntityRetrievalSettings
from app.entities.graph import EntityGraph
from app.entities.linking import LinkResult

_EXPANSION_SKIP_TYPES = frozenset({"DATE"})

# Query words that indicate which entity type a relational question is asking through.
TYPE_HINTS: dict[str, frozenset[str]] = {
    "PROJECT": frozenset({"project", "projects", "initiative", "pilot", "program", "programme", "product", "platform"}),
    "ORGANIZATION": frozenset({
        "company", "companies", "organization", "organisation", "org", "vendor", "vendors", "employer",
        "firm", "startup", "customer", "client", "partner", "hospital", "supplier",
    }),
    "PERSON": frozenset({"who", "whom", "person", "people", "lead", "leader", "owner", "manager", "engineer", "sponsor", "author"}),
    "TECHNOLOGY": frozenset({
        "database", "language", "library", "framework", "runtime", "tool", "tools", "technology",
        "technologies", "stack", "broker", "queue", "cache", "caching", "solver", "engine", "store",
    }),
    "LOCATION": frozenset({"where", "city", "country", "office", "located", "based", "depot", "site"}),
    "EVENT": frozenset({"event", "conference", "summit", "offsite", "meeting", "talk", "session"}),
}
_WORD = re.compile(r"[a-z]+")


def hinted_types(query: str) -> set[str]:
    words = set(_WORD.findall(query.lower()))
    return {t for t, hints in TYPE_HINTS.items() if words & hints}


@dataclass
class EntityPlan:
    direct: dict[str, float] = field(default_factory=dict)
    expanded: dict[str, float] = field(default_factory=dict)
    via: dict[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.direct


class EntityScorer:
    def __init__(self, graph: EntityGraph, settings: EntityRetrievalSettings) -> None:
        self.graph = graph
        self.s = settings

    def plan(self, links: LinkResult, query: str = "") -> EntityPlan:
        plan = EntityPlan(direct=links.confidences())
        if self.s.max_hops < 1:
            return plan
        hints = hinted_types(query)
        for qid, conf in plan.direct.items():
            for nid, edge in self.graph.neighbors(qid).items():
                if nid in plan.direct:
                    continue
                entity = self.graph.entities.get(nid)
                if entity is None or entity.entity_type in _EXPANSION_SKIP_TYPES:
                    continue
                decay = self.s.hinted_hop_decay if entity.entity_type in hints else self.s.hop_decay
                weight = conf * edge.weight * decay
                if weight > plan.expanded.get(nid, 0.0):
                    plan.expanded[nid] = weight
                    plan.via[nid] = qid
        strongest = sorted(plan.expanded.items(), key=lambda kv: -kv[1])[: self.s.max_neighbors]
        plan.expanded = dict(strongest)
        plan.via = {k: v for k, v in plan.via.items() if k in plan.expanded}
        return plan

    def score_chunks(self, plan: EntityPlan, chunk_ids: list[str]) -> dict[str, tuple[float, list[str]]]:
        total = sum(plan.direct.values())
        out: dict[str, tuple[float, list[str]]] = {}
        if total <= 0:
            return {cid: (0.0, []) for cid in chunk_ids}
        for cid in chunk_ids:
            mentioned = self.graph.chunk_entities.get(cid, set())
            direct_hits = [e for e in mentioned if e in plan.direct]
            coverage = sum(plan.direct[e] for e in direct_hits) / total
            expansion_hits = [e for e in mentioned if e in plan.expanded]
            expansion = max((plan.expanded[e] for e in expansion_hits), default=0.0)
            score = 1.0 - (1.0 - min(1.0, coverage)) * (1.0 - expansion)
            matched = [self.graph.entities[e].canonical_name for e in direct_hits]
            matched += [
                f"{self.graph.entities[e].canonical_name} (via {self.graph.entities[plan.via[e]].canonical_name})"
                for e in expansion_hits
            ]
            out[cid] = (score, matched)
        return out

    def candidates(self, plan: EntityPlan, k: int) -> list[tuple[str, float]]:
        pool: set[str] = set()
        for eid in list(plan.direct) + list(plan.expanded):
            pool |= self.graph.entity_chunks.get(eid, set())
        scored = self.score_chunks(plan, sorted(pool))
        ranked = sorted(((cid, s) for cid, (s, _) in scored.items() if s > 0), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]
