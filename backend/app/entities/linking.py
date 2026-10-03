"""Query-time entity linking: map spans of a query onto resolved entities.

Matching, in order of preference (longest n-gram first, non-overlapping):

1. exact alias match on normalized or compact form ("Open AI" -> OpenAI),
2. fuzzy alias match for capitalized n-grams (typos), via rapidfuzz,
3. partial name match on a single distinctive token ("Okafor" -> Samuel Okafor).

An alias shared by several entities splits its confidence across them, which
is how ambiguity ("Alice" with two Alices in the corpus) propagates into
scoring instead of being silently resolved.

The linker also reports *unmatched* entity mentions: named spans in the query
(found by the extractor) that correspond to nothing in the index. A query about
"Project Nimbus" over a corpus that never mentions it is a strong abstention
signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from app.core.config import EntityRetrievalSettings
from app.core.text import STOPWORDS
from app.entities.extraction import CompositeExtractor
from app.entities.graph import EntityGraph
from app.entities.normalization import basic_normalize

_TOKEN_RE = re.compile(r"[A-Za-z0-9][\w.&'-]*")
_QUESTION_WORDS = frozenset({"who", "what", "which", "where", "when", "why", "how", "does", "did", "is", "are"})
_LINKABLE_UNMATCHED_TYPES = frozenset({"PERSON", "ORGANIZATION", "PROJECT", "EVENT", "LOCATION"})


@dataclass(slots=True)
class LinkedEntity:
    entity_id: str
    canonical_name: str
    entity_type: str
    surface: str
    confidence: float
    method: str  # exact | compact | fuzzy | partial


@dataclass
class LinkResult:
    linked: list[LinkedEntity] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)

    def confidences(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for link in self.linked:
            out[link.entity_id] = max(out.get(link.entity_id, 0.0), link.confidence)
        return out


class QueryEntityLinker:
    def __init__(
        self,
        graph: EntityGraph,
        settings: EntityRetrievalSettings,
        extractor: CompositeExtractor | None = None,
        max_ngram: int = 5,
    ) -> None:
        self.graph = graph
        self.s = settings
        self.extractor = extractor
        self.max_ngram = max_ngram
        self._fuzzy_choices = list(graph.exact_index.keys())

    def link(self, query: str, exact_only: bool = False) -> LinkResult:
        tokens = [(m.group(0).rstrip(".'"), m.start(), m.start() + len(m.group(0).rstrip(".'"))) for m in _TOKEN_RE.finditer(query)]
        taken: list[tuple[int, int]] = []
        result = LinkResult()

        def free(start: int, end: int) -> bool:
            return all(end <= s or start >= e for s, e in taken)

        def accept(ids: set[str], surface: str, start: int, end: int, base: float, method: str) -> None:
            share = base / len(ids)
            for eid in sorted(ids):
                entity = self.graph.entities[eid]
                result.linked.append(
                    LinkedEntity(eid, entity.canonical_name, entity.entity_type, surface, round(share, 4), method)
                )
            taken.append((start, end))

        for n in range(min(self.max_ngram, len(tokens)), 0, -1):
            for i in range(len(tokens) - n + 1):
                window = tokens[i : i + n]
                start, end = window[0][1], window[-1][2]
                if not free(start, end):
                    continue
                surface = query[start:end]
                key = basic_normalize(surface)
                if not key or all(t in STOPWORDS or t in _QUESTION_WORDS for t in key.split()):
                    continue
                if n == 1 and (len(key) < 3 or key in STOPWORDS):
                    continue
                ids = set(self.graph.exact_index.get(key) or self.graph.compact_index.get(key.replace(" ", "")) or ())
                if ids:
                    if n == 1:
                        # A bare first name / surname / head word may also complete to
                        # full names ("Alice" -> "Alice Chen" or "Alice Moreno").
                        ids |= self.graph.token_index.get(key, set())
                    accept(ids, surface, start, end, 1.0, "exact")
                    continue
                if exact_only:
                    continue
                if n <= 4 and surface[:1].isupper() and len(key) >= 5 and self._fuzzy_choices:
                    match = process.extractOne(key, self._fuzzy_choices, scorer=fuzz.ratio)
                    if match and match[1] >= self.s.fuzzy_link_threshold:
                        accept(set(self.graph.exact_index[match[0]]), surface, start, end, 0.9 * match[1] / 100, "fuzzy")
                        continue

        # Partial names: single distinctive tokens ("Okafor", "Brightwater").
        for token, start, end in [] if exact_only else tokens:
            if not free(start, end):
                continue
            key = basic_normalize(token)
            if len(key) < 3 or key in STOPWORDS or key in _QUESTION_WORDS:
                continue
            ids = self.graph.token_index.get(key)
            if ids:
                accept(set(ids), token, start, end, self.s.partial_name_confidence, "partial")

        result.linked = [link for link in result.linked if link.confidence >= self.s.min_link_confidence]
        if self.extractor is not None and not exact_only:
            for span in self.extractor.extract(query):
                if span.entity_type in _LINKABLE_UNMATCHED_TYPES and free(span.start, span.end):
                    first = basic_normalize(span.surface).split()[:1]
                    if first and first[0] in _QUESTION_WORDS:
                        continue
                    result.unmatched.append(span.surface)
        return result
