"""Entity resolution: decide which mentions refer to the same real-world entity.

Pipeline (all steps are deterministic and log their decisions):

1. **Type harmonization** - the same surface form labelled with different
   types by different extractors (spaCy says ORG, a rule says PROJECT) is
   assigned the type with the highest summed extractor confidence.
2. **Surface grouping** - mentions sharing ``(type, normalized name)`` form a
   group ("OpenAI Inc." and "OpenAI" normalize identically).
3. **Within-document coreference** - a partial person mention ("Alice",
   "Dr. Chen", "A. Chen") is attached to a full name *in the same document*
   when exactly one compatible full name occurs there.
4. **Cross-document pairwise matching** - candidate pairs come from blocking
   keys (shared tokens, name prefixes, surnames, known alias groups) and are
   scored by a Fellegi-Sunter-style log-linear model with hand-set weights:

       score = sigmoid(bias + sum_i w_i * f_i)

   Features: compact-form equality, known technology alias, Jaro-Winkler
   string similarity, surname match with compatible first name/initial, token-subset relation,
   context-embedding similarity, document co-occurrence, uniqueness or
   ambiguity of a short-form completion, and hard conflicts (different
   surnames, different numbers).
5. **Constrained clustering** - pairs above ``auto_merge_threshold`` are
   merged in descending score order with union-find, refusing any merge that
   would place two conflicting groups in one cluster (prevents A~B~C chains
   when A and C clearly differ). Pairs in the ambiguous band are recorded but
   kept separate.

The weights are not learned; with labelled pairs they could be fit by
logistic regression without changing the interface.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz.distance import JaroWinkler

from app.core.config import ResolutionSettings
from app.entities.gazetteer import ORG_SUFFIXES, PERSON_TITLES, TECHNOLOGY_CANONICAL
from app.entities.normalization import (
    basic_normalize,
    harmonization_key,
    is_title_form,
    name_tokens,
)
from app.models.domain import Entity, Mention, ResolutionDecision

EXACT_ONLY_TYPES = frozenset({"DATE"})
_DIGITS = set("0123456789")


@dataclass
class SurfaceGroup:
    gid: str
    entity_type: str
    tokens: list[str]
    surfaces: Counter = field(default_factory=Counter)
    mention_ids: list[str] = field(default_factory=list)
    doc_ids: set[str] = field(default_factory=set)
    chunk_ids: set[str] = field(default_factory=set)
    context: np.ndarray | None = None

    @property
    def normalized(self) -> str:
        return " ".join(self.tokens)

    @property
    def compact(self) -> str:
        return "".join(self.tokens)

    @property
    def is_partial_person(self) -> bool:
        if self.entity_type != "PERSON":
            return False
        return len(self.tokens) == 1 or (len(self.tokens) == 2 and len(self.tokens[0]) == 1)

    @property
    def is_full(self) -> bool:
        return len(self.tokens) >= 2 and not self.is_partial_person

    def tech_canonicals(self) -> set[str]:
        return {
            TECHNOLOGY_CANONICAL[s.lower()] for s in self.surfaces if s.lower() in TECHNOLOGY_CANONICAL
        }


@dataclass
class ResolutionResult:
    entities: list[Entity]
    assignments: dict[str, str]
    decisions: list[ResolutionDecision]
    stats: dict[str, int]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _first_names_compatible(a: str, b: str) -> bool:
    """'a' ~ 'alice' (initial), 'sam' ~ 'samuel' (prefix), 'alice' ~ 'alice'."""
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    return a.startswith(b) or b.startswith(a)


def person_compatible(short: list[str], full: list[str]) -> bool:
    """Can the (possibly partial) person name ``short`` refer to ``full``?"""
    if len(full) < 2:
        return False
    if len(short) == 1:
        token = short[0]
        return token == full[-1] or _first_names_compatible(token, full[0]) and len(token) > 1
    if short[-1] != full[-1]:
        return False
    return _first_names_compatible(short[0], full[0])


class EntityResolver:
    def __init__(self, settings: ResolutionSettings) -> None:
        self.s = settings

    # ----------------------------------------------------------------- public
    def resolve(
        self,
        mentions: list[Mention],
        chunk_embeddings: dict[str, np.ndarray] | None = None,
    ) -> ResolutionResult:
        chunk_embeddings = chunk_embeddings or {}
        decisions: list[ResolutionDecision] = []
        types = self._harmonize_types(mentions, decisions)
        coref_links = self._within_document_coreference(mentions, types, decisions)
        unlinked = [m for m in mentions if m.id not in coref_links]
        groups = self._build_groups(unlinked, types, chunk_embeddings)
        self._completion_cache: dict[str, list[SurfaceGroup]] = {}
        parent: dict[str, str] = {g: g for g in groups}

        scored = self._score_pairs(groups, decisions)
        self._cluster(groups, parent, scored, decisions)

        clusters: dict[str, list[SurfaceGroup]] = defaultdict(list)
        for gid, group in groups.items():
            clusters[self._find(parent, gid)].append(group)

        entities: list[Entity] = []
        assignments: dict[str, str] = {}
        entity_of_group: dict[str, str] = {}
        for members in clusters.values():
            entity = self._make_entity(members)
            entities.append(entity)
            for group in members:
                entity_of_group[group.gid] = entity.id
                for mid in group.mention_ids:
                    assignments[mid] = entity.id
        # Partial mentions attached by within-document coreference follow their target group.
        for mention_id, target_gid in coref_links.items():
            if target_gid in entity_of_group:
                assignments[mention_id] = entity_of_group[target_gid]
        self._refresh_counts(entities, assignments, mentions)

        stats = {
            "mentions": len(mentions),
            "surface_groups": len(groups),
            "entities": len(entities),
            "coreference_links": len(coref_links),
            "merged_pairs": sum(1 for d in decisions if d.decision == "merge" and "within_document" not in d.features),
            "ambiguous_pairs": sum(1 for d in decisions if d.decision == "ambiguous"),
            "blocked_conflicts": sum(1 for d in decisions if d.decision == "blocked_conflict"),
            "type_harmonized": sum(1 for d in decisions if d.decision == "type_harmonized"),
        }
        return ResolutionResult(entities, assignments, decisions, stats)

    def score_pair(self, a: SurfaceGroup, b: SurfaceGroup, groups: dict[str, SurfaceGroup]) -> tuple[float, dict[str, float]]:
        features = self._features(a, b, groups)
        s = self.s
        logit = (
            s.bias
            + s.w_compact_equal * features["compact_equal"]
            + s.w_known_alias * features["known_alias"]
            + s.w_string * features["string"]
            + s.w_surname_match * features["surname_match"]
            + s.w_token_subset * features["token_subset"]
            + s.w_context * features["context"]
            + s.w_doc_cooccur * features["doc_cooccur"]
            + s.w_unique_completion * features["unique_completion"]
            + s.w_ambiguous_completion * features["ambiguous_completion"]
            + s.w_conflict * features["conflict"]
        )
        return _sigmoid(logit), features

    # ---------------------------------------------------------------- stages
    def _harmonize_types(self, mentions: list[Mention], decisions: list[ResolutionDecision]) -> dict[str, str]:
        votes: dict[str, Counter] = defaultdict(Counter)
        for m in mentions:
            if m.entity_type in EXACT_ONLY_TYPES:
                continue
            votes[harmonization_key(m.surface)][m.entity_type] += m.confidence
        chosen = {key: c.most_common(1)[0][0] for key, c in votes.items() if c}
        types: dict[str, str] = {}
        reported: set[str] = set()
        for m in mentions:
            key = harmonization_key(m.surface)
            new_type = chosen.get(key, m.entity_type)
            types[m.id] = new_type
            if new_type != m.entity_type and key not in reported and len(votes[key]) > 1:
                reported.add(key)
                decisions.append(
                    ResolutionDecision(
                        left=m.surface,
                        right=m.surface,
                        entity_type=new_type,
                        score=1.0,
                        decision="type_harmonized",
                        features={t: round(v, 3) for t, v in votes[key].items()},
                        reason=f"conflicting labels {dict(votes[key])}; kept {new_type}",
                    )
                )
        return types

    def _build_groups(
        self, mentions: list[Mention], types: dict[str, str], chunk_embeddings: dict[str, np.ndarray]
    ) -> dict[str, SurfaceGroup]:
        groups: dict[str, SurfaceGroup] = {}
        for m in mentions:
            entity_type = types[m.id]
            tokens = name_tokens(m.surface, entity_type)
            if not tokens:
                continue
            gid = f"{entity_type}::{' '.join(tokens)}"
            group = groups.get(gid)
            if group is None:
                group = groups[gid] = SurfaceGroup(gid, entity_type, tokens)
            group.surfaces[m.surface] += 1
            group.mention_ids.append(m.id)
            group.doc_ids.add(m.document_id)
            group.chunk_ids.add(m.chunk_id)
        for group in groups.values():
            vectors = [chunk_embeddings[c] for c in group.chunk_ids if c in chunk_embeddings]
            if vectors:
                mean = np.mean(vectors, axis=0)
                norm = np.linalg.norm(mean)
                group.context = mean / norm if norm > 0 else None
        return groups

    def _within_document_coreference(
        self,
        mentions: list[Mention],
        types: dict[str, str],
        decisions: list[ResolutionDecision],
    ) -> dict[str, str]:
        """Attach partial person mentions to the unique compatible full name in their document.

        Returns mention_id -> group id of the full name it refers to.
        """
        full_by_doc: dict[str, dict[str, tuple[list[str], str]]] = defaultdict(dict)
        partial: list[tuple[Mention, list[str]]] = []
        for m in mentions:
            if types.get(m.id) != "PERSON":
                continue
            tokens = name_tokens(m.surface, "PERSON")
            if not tokens:
                continue
            is_partial = len(tokens) == 1 or len(tokens[0]) == 1 or is_title_form(m.surface)
            if is_partial:
                partial.append((m, tokens))
            else:
                gid = f"PERSON::{' '.join(tokens)}"
                full_by_doc[m.document_id][gid] = (tokens, m.surface)

        links: dict[str, str] = {}
        logged: set[tuple[str, str]] = set()
        for m, tokens in partial:
            compatible = [
                (gid, surface)
                for gid, (full_tokens, surface) in full_by_doc.get(m.document_id, {}).items()
                if person_compatible(tokens, full_tokens)
            ]
            # Variants of one name ("Sam Okafor", "Samuel Okafor") count as one candidate.
            distinct = {gid.rsplit(" ", 1)[-1] for gid, _ in compatible}
            if compatible and len(distinct) == 1 and len(compatible) == 1:
                gid, surface = compatible[0]
                links[m.id] = gid
                if (m.surface, m.document_id) not in logged:
                    logged.add((m.surface, m.document_id))
                    decisions.append(
                        ResolutionDecision(
                            left=m.surface,
                            right=surface,
                            entity_type="PERSON",
                            score=1.0,
                            decision="merge",
                            features={"within_document": 1.0},
                            reason="within-document coreference: unique compatible full name in the same document",
                        )
                    )
        return links

    def _blocking_keys(self, g: SurfaceGroup) -> set[str]:
        keys = {f"p:{g.compact[:4]}"}
        keys.update(f"t:{t}" for t in g.tokens if len(t) >= 3)
        if g.entity_type == "PERSON":
            keys.add(f"last:{g.tokens[-1]}")
            keys.add(f"first:{g.tokens[0][:1]}")
        keys.update(f"kb:{c}" for c in g.tech_canonicals())
        return keys

    def _score_pairs(
        self, groups: dict[str, SurfaceGroup], decisions: list[ResolutionDecision]
    ) -> list[tuple[float, str, str, dict[str, float]]]:
        blocks: dict[tuple[str, str], list[str]] = defaultdict(list)
        for gid, g in groups.items():
            if g.entity_type in EXACT_ONLY_TYPES:
                continue
            for key in self._blocking_keys(g):
                blocks[(g.entity_type, key)].append(gid)
        seen: set[tuple[str, str]] = set()
        scored: list[tuple[float, str, str, dict[str, float]]] = []
        for members in blocks.values():
            if len(members) > 200:  # degenerate block (e.g. first-initial key); skip
                continue
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = sorted((members[i], members[j]))
                    if (a, b) in seen:
                        continue
                    seen.add((a, b))
                    score, feats = self.score_pair(groups[a], groups[b], groups)
                    scored.append((score, a, b, feats))
        for score, a, b, feats in sorted(scored, reverse=True):
            if score >= self.s.auto_merge_threshold:
                continue  # logged during clustering (merge or blocked_conflict)
            if score >= self.s.candidate_threshold:
                decision, reason = "ambiguous", "score in ambiguous band; kept separate"
            elif score >= 0.2:
                decision, reason = "distinct", "score below candidate threshold"
            else:
                continue
            decisions.append(self._decision(groups[a], groups[b], score, feats, decision, reason))
        return scored

    def _cluster(
        self,
        groups: dict[str, SurfaceGroup],
        parent: dict[str, str],
        scored: list[tuple[float, str, str, dict[str, float]]],
        decisions: list[ResolutionDecision],
    ) -> None:
        conflicts: set[tuple[str, str]] = {(a, b) for _, a, b, f in scored if f["conflict"] > 0}
        members: dict[str, set[str]] = {gid: {gid} for gid in groups}
        for score, a, b, feats in sorted(scored, reverse=True):
            if score < self.s.auto_merge_threshold:
                break
            ra, rb = self._find(parent, a), self._find(parent, b)
            if ra == rb:
                continue
            clash = any(
                (min(x, y), max(x, y)) in conflicts for x in members[ra] for y in members[rb]
            )
            if clash:
                decisions.append(
                    self._decision(
                        groups[a], groups[b], score, feats, "blocked_conflict",
                        "merge would join groups with a hard conflict (cannot-link)",
                    )
                )
                continue
            parent[rb] = ra
            members[ra] |= members.pop(rb)
            decisions.append(
                self._decision(groups[a], groups[b], score, feats, "merge", "score above auto-merge threshold")
            )

    # --------------------------------------------------------------- features
    def _features(self, a: SurfaceGroup, b: SurfaceGroup, groups: dict[str, SurfaceGroup]) -> dict[str, float]:
        s = self.s
        sa, sb = set(a.tokens), set(b.tokens)
        known_alias = bool(a.tech_canonicals() & b.tech_canonicals())
        jw = JaroWinkler.similarity(a.normalized, b.normalized)
        string = max(0.0, (jw - s.string_floor) / (1.0 - s.string_floor))
        subset = 1.0 if (sa < sb or sb < sa) else 0.0

        surname_match = 0.0
        conflict = 0.0
        if a.entity_type == "PERSON":
            if len(a.tokens) >= 2 and len(b.tokens) >= 2:
                if a.tokens[-1] == b.tokens[-1] and _first_names_compatible(a.tokens[0], b.tokens[0]):
                    surname_match = 1.0  # "A. Chen" ~ "Alice Chen", "Sam Okafor" ~ "Samuel Okafor"
                else:
                    conflict = 1.0  # different surnames, or incompatible first names
            short, full = self._short_full(a, b)
            if short.is_partial_person and full.is_full:
                if person_compatible(short.tokens, full.tokens):
                    subset = 1.0
                    if short.tokens[-1] == full.tokens[-1]:
                        surname_match = 1.0  # "Chen" / "Dr. Chen" ~ "Alice Chen"
                else:
                    conflict = 1.0
        elif len(a.tokens) >= 2 and len(b.tokens) >= 2 and not subset and a.compact != b.compact and jw < 0.9:
            conflict = 1.0  # e.g. "Quillon Labs" vs "Quillon Health"
        digits_a = {t for t in a.tokens if set(t) & _DIGITS}
        digits_b = {t for t in b.tokens if set(t) & _DIGITS}
        if digits_a and digits_b and digits_a != digits_b and not known_alias:
            conflict = 1.0
        if known_alias:
            conflict = 0.0

        context = 0.0
        if a.context is not None and b.context is not None:
            cos = float(a.context @ b.context)
            context = min(1.0, max(0.0, (cos - s.context_floor) / (s.context_ceiling - s.context_floor)))

        unique_completion = ambiguous_completion = 0.0
        if subset and not conflict:
            short, _ = self._short_full(a, b)
            distinct = self._distinct_completions(short, groups)
            if len(distinct) == 1:
                unique_completion = 1.0
            elif len(distinct) > 1:
                ambiguous_completion = 1.0

        return {
            "compact_equal": 1.0 if a.compact == b.compact else 0.0,
            "known_alias": 1.0 if known_alias else 0.0,
            "string": round(string, 4),
            "surname_match": surname_match,
            "token_subset": subset,
            "context": round(context, 4),
            "doc_cooccur": 1.0 if a.doc_ids & b.doc_ids else 0.0,
            "unique_completion": unique_completion,
            "ambiguous_completion": ambiguous_completion,
            "conflict": conflict,
        }

    @staticmethod
    def _short_full(a: SurfaceGroup, b: SurfaceGroup) -> tuple[SurfaceGroup, SurfaceGroup]:
        if a.is_partial_person != b.is_partial_person:
            return (a, b) if a.is_partial_person else (b, a)
        return (a, b) if len(a.tokens) <= len(b.tokens) else (b, a)

    def _distinct_completions(self, short: SurfaceGroup, groups: dict[str, SurfaceGroup]) -> set[str]:
        """Distinct full-form entities a short form could expand to.

        Variants of one name collapse: people by (surname, first initial), other
        types by keeping only minimal completions ("brightwater health" absorbs
        "brightwater health system").
        """
        if short.gid in self._completion_cache:
            completions = self._completion_cache[short.gid]
        else:
            completions = []
            for g in groups.values():
                if g.entity_type != short.entity_type or g.gid == short.gid or not g.is_full:
                    continue
                if short.entity_type == "PERSON":
                    if person_compatible(short.tokens, g.tokens):
                        completions.append(g)
                elif set(short.tokens) < set(g.tokens):
                    completions.append(g)
            self._completion_cache[short.gid] = completions
        if short.entity_type == "PERSON":
            return {f"{c.tokens[-1]}:{c.tokens[0][:1]}" for c in completions}
        token_sets = [set(c.tokens) for c in completions]
        return {
            c.compact
            for c, ts in zip(completions, token_sets, strict=True)
            if not any(other < ts for other in token_sets)
        }

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _find(parent: dict[str, str], x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    @staticmethod
    def _decision(
        a: SurfaceGroup, b: SurfaceGroup, score: float, feats: dict[str, float], decision: str, reason: str
    ) -> ResolutionDecision:
        return ResolutionDecision(
            left=a.surfaces.most_common(1)[0][0],
            right=b.surfaces.most_common(1)[0][0],
            entity_type=a.entity_type,
            score=round(score, 4),
            decision=decision,
            features=feats,
            reason=reason,
        )

    @staticmethod
    def _make_entity(members: list[SurfaceGroup]) -> Entity:
        entity_type = members[0].entity_type
        surfaces: Counter = Counter()
        for g in members:
            surfaces.update(g.surfaces)
        canonical = None
        if entity_type == "TECHNOLOGY":
            canonicals = Counter()
            for g in members:
                for c in g.tech_canonicals():
                    canonicals[c] += sum(g.surfaces.values())
            if canonicals:
                canonical = canonicals.most_common(1)[0][0]
        if canonical is None:
            full = [g for g in members if not g.is_partial_person] or members
            full_surfaces: Counter = Counter()
            for g in full:
                full_surfaces.update(g.surfaces)
            # Prefer "Project X" for projects, title-free person names and suffix-free
            # organization names ("OpenAI" over "OpenAI Inc."), then the most
            # frequent, then the longest surface (deterministic tie-break on the string).
            def rank(kv: tuple[str, int]) -> tuple:
                surface, count = kv
                tokens = basic_normalize(surface).split()
                titled = entity_type == "PERSON" and bool(tokens) and tokens[0] in PERSON_TITLES
                suffixed = entity_type == "ORGANIZATION" and len(tokens) > 1 and tokens[-1] in ORG_SUFFIXES
                project_form = entity_type == "PROJECT" and surface.startswith("Project ")
                return (titled or suffixed, not project_form, -count, -len(surface), surface)

            canonical = sorted(full_surfaces.items(), key=rank)[0][0]
        key = f"{entity_type}:{basic_normalize(canonical)}"
        entity_id = "ent_" + hashlib.sha1(key.encode()).hexdigest()[:12]
        aliases = sorted({s for s in surfaces if s != canonical})
        return Entity(
            id=entity_id,
            canonical_name=canonical,
            entity_type=entity_type,
            aliases=aliases,
            mention_count=0,
            document_count=0,
            metadata={"surface_groups": len(members)},
        )

    @staticmethod
    def _refresh_counts(entities: list[Entity], assignments: dict[str, str], mentions: list[Mention]) -> None:
        by_id = {e.id: e for e in entities}
        docs: dict[str, set[str]] = defaultdict(set)
        surfaces: dict[str, set[str]] = defaultdict(set)
        counts: Counter = Counter()
        for m in mentions:
            eid = assignments.get(m.id)
            if eid:
                counts[eid] += 1
                docs[eid].add(m.document_id)
                surfaces[eid].add(m.surface)
        for eid, entity in by_id.items():
            entity.mention_count = counts[eid]
            entity.document_count = len(docs[eid])
            entity.aliases = sorted((set(entity.aliases) | surfaces[eid]) - {entity.canonical_name})
        # Drop entities that lost all mentions (e.g. partial groups fully absorbed by coreference).
        entities[:] = [e for e in entities if e.mention_count > 0]
