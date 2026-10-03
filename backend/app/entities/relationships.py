"""Relationship extraction between resolved entities.

Relations are inferred per sentence (or per JSON record) from entity type pairs
plus lexical trigger phrases. Confidence is graded by how direct the evidence is:

* trigger phrase *between* the two mentions        -> ``pattern_between``
* trigger phrase elsewhere in the same sentence    -> ``pattern_sentence``
* JSON record whose field name acts as the trigger  -> ``record``
* no trigger, same sentence                        -> ``related_to`` co-occurrence
* *context* entity (from the chunk's title/section header, e.g. the project a
  design-doc section belongs to) + an entity in the sentence + trigger
                                                    -> ``context`` (discounted)

Each relationship row keeps its supporting chunk and the evidence sentence, so
every graph edge is traceable to source text. This is deliberately simple and
noisy; downstream use weights edges by confidence rather than trusting them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.core.text import sentence_spans
from app.models.domain import Chunk, Mention, Relationship


@dataclass(frozen=True)
class RelationPattern:
    source_type: str
    target_type: str
    relation: str
    triggers: tuple[str, ...]

    def regex(self) -> re.Pattern[str]:
        alternatives = "|".join(re.escape(t) for t in sorted(self.triggers, key=len, reverse=True))
        return re.compile(rf"(?<![\w])(?:{alternatives})(?![\w])", re.IGNORECASE)


RELATION_PATTERNS: tuple[RelationPattern, ...] = (
    RelationPattern(
        "PERSON", "PROJECT", "works_on",
        ("leads", "led by", "lead", "leading", "works on", "working on", "worked on", "owns",
         "owner", "maintains", "maintainer", "contributes", "contributor", "assigned", "assignee",
         "building", "builds", "drives", "driving", "manages", "managing", "heads", "tech lead",
         "tech_lead", "joined", "oversees", "responsible for", "author", "project", "projects",
         "on the", "pm", "engineer", "sponsor", "sponsored by", "product manager", "runs", "run"),
    ),
    RelationPattern(
        "PERSON", "ORGANIZATION", "affiliated_with",
        ("at", "joined", "joins", "works for", "working for", "employed", "employee", "ceo", "cto",
         "vp", "director", "head of", "engineer", "scientist", "from", "of", "organization",
         "company", "employer", "hired", "moved to", "founder", "co-founder", "contact",
         "account", "manager", "leaves", "left"),
    ),
    RelationPattern(
        "PROJECT", "TECHNOLOGY", "uses",
        ("uses", "use", "using", "used", "built on", "built with", "runs on", "running on",
         "migrated to", "migrating to", "migrate to", "powered by", "written in", "adopted", "adopts",
         "relies on", "backed by", "stores", "on top of", "stack", "deployed on", "integrates",
         "chose", "choose", "selected", "replaced", "moved to", "via", "with", "serves", "technologies"),
    ),
    RelationPattern(
        "ORGANIZATION", "PROJECT", "owns",
        ("develops", "developed", "launched", "owns", "runs", "funds", "funded", "sponsors",
         "built", "building", "internal", "initiative", "project", "team", "organization", "'s"),
    ),
    RelationPattern(
        "ORGANIZATION", "LOCATION", "located_in",
        ("based in", "headquartered", "office", "offices", "located", "hq", "depot", "site", "in"),
    ),
    RelationPattern(
        "PERSON", "LOCATION", "located_in",
        ("based in", "located", "relocated", "lives in", "works from", "office", "location", "in"),
    ),
    RelationPattern(
        "ORGANIZATION", "ORGANIZATION", "partners_with",
        ("customer", "client", "vendor", "partner", "partnership", "contract", "supplier", "acquired"),
    ),
    RelationPattern(
        "PERSON", "EVENT", "participates_in",
        ("attended", "attend", "attending", "presented", "presenting", "speaker", "spoke", "hosted",
         "organized", "organizer", "keynote", "session"),
    ),
)

_COMPILED = [(p, p.regex()) for p in RELATION_PATTERNS]
RELATION_TYPES = tuple(sorted({p.relation for p in RELATION_PATTERNS} | {"related_to"}))

CONF_BETWEEN = 0.85
CONF_RECORD = 0.75
CONF_SENTENCE = 0.65
CONF_COOCCURRENCE = 0.35
CONTEXT_DISCOUNT = 0.8
"""Relations anchored on a header (context) entity get CONF_SENTENCE * CONTEXT_DISCOUNT."""
MAX_COOCCURRENCE_ENTITIES = 6
"""Units mentioning more entities than this only yield trigger-backed relations."""
SKIP_TYPES = frozenset({"DATE"})


def _relationship_id(source: str, target: str, relation: str, chunk_id: str) -> str:
    return "rel_" + hashlib.sha1(f"{source}|{target}|{relation}|{chunk_id}".encode()).hexdigest()[:16]


@dataclass(slots=True)
class _Anchor:
    entity_id: str
    entity_type: str
    start: int
    end: int


def _units(chunk: Chunk) -> list[tuple[int, int]]:
    if chunk.metadata.get("kind") in ("record", "header"):
        return [(0, len(chunk.text))]
    return sentence_spans(chunk.text) or [(0, len(chunk.text))]


def extract_relationships(
    chunk: Chunk,
    mentions: list[Mention],
    entity_types: dict[str, str],
    context_entities: list[str] | None = None,
) -> list[Relationship]:
    """Infer relations inside one chunk.

    ``mentions`` are this chunk's text mentions with ``entity_id`` set; types come
    from resolved entities. ``context_entities`` are entity IDs found in the
    chunk's contextual header (document title / section path).
    """
    anchors = [
        _Anchor(m.entity_id, entity_types[m.entity_id], m.char_start, m.char_end)
        for m in mentions
        if m.entity_id and m.entity_id in entity_types and entity_types[m.entity_id] not in SKIP_TYPES
    ]
    context = [
        c for c in dict.fromkeys(context_entities or [])
        if c in entity_types and entity_types[c] not in SKIP_TYPES
    ]
    is_record = chunk.metadata.get("kind") == "record"
    found: dict[tuple[str, str, str], Relationship] = {}

    def keep(rel: tuple[str, str, str, float, str], sentence: str) -> None:
        source, target, relation, confidence, method = rel
        key = (source, target, relation)
        if key in found and found[key].confidence >= confidence:
            return
        found[key] = Relationship(
            id=_relationship_id(source, target, relation, chunk.id),
            source_entity_id=source,
            target_entity_id=target,
            relationship_type=relation,
            confidence=confidence,
            supporting_chunk_id=chunk.id,
            evidence=sentence.strip()[:500],
            method=method,
        )

    for unit_start, unit_end in _units(chunk):
        unit = [a for a in anchors if unit_start <= a.start < unit_end]
        if not unit:
            continue
        sentence = chunk.text[unit_start:unit_end]
        distinct = {a.entity_id for a in unit}
        for i, a in enumerate(unit):
            for b in unit[i + 1 :]:
                if a.entity_id != b.entity_id:
                    rel = _classify(a, b, sentence, chunk.text, unit_start, is_record, len(distinct))
                    if rel is not None:
                        keep(rel, sentence)
        for cid in context:
            if cid in distinct:
                continue
            for a in unit:
                rel = _classify_context(cid, entity_types[cid], a, sentence)
                if rel is not None:
                    keep(rel, sentence)
    return list(found.values())


def _classify_context(
    context_id: str, context_type: str, a: _Anchor, sentence: str
) -> tuple[str, str, str, float, str] | None:
    for pattern, regex in _COMPILED:
        if (context_type, a.entity_type) == (pattern.source_type, pattern.target_type):
            src, tgt = context_id, a.entity_id
        elif (a.entity_type, context_type) == (pattern.source_type, pattern.target_type):
            src, tgt = a.entity_id, context_id
        else:
            continue
        if regex.search(sentence):
            return src, tgt, pattern.relation, round(CONF_SENTENCE * CONTEXT_DISCOUNT, 4), "context"
        return None
    return None


def _classify(
    a: _Anchor, b: _Anchor, sentence: str, text: str, unit_start: int, is_record: bool, n_entities: int
) -> tuple[str, str, str, float, str] | None:
    for pattern, regex in _COMPILED:
        if (a.entity_type, b.entity_type) == (pattern.source_type, pattern.target_type):
            src, tgt = a, b
        elif (b.entity_type, a.entity_type) == (pattern.source_type, pattern.target_type):
            src, tgt = b, a
        else:
            continue
        lo, hi = sorted((a, b), key=lambda x: x.start)
        between = text[lo.end : hi.start]
        if is_record:
            if regex.search(sentence):
                return src.entity_id, tgt.entity_id, pattern.relation, CONF_RECORD, "record"
        elif regex.search(between):
            return src.entity_id, tgt.entity_id, pattern.relation, CONF_BETWEEN, "pattern_between"
        elif regex.search(sentence):
            return src.entity_id, tgt.entity_id, pattern.relation, CONF_SENTENCE, "pattern_sentence"
        break
    if n_entities > MAX_COOCCURRENCE_ENTITIES:
        return None
    src, tgt = sorted((a, b), key=lambda x: x.entity_id)
    return src.entity_id, tgt.entity_id, "related_to", CONF_COOCCURRENCE, "cooccurrence"
