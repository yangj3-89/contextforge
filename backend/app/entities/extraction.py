"""Entity mention extraction.

Extractors produce candidate spans; :class:`CompositeExtractor` merges them,
resolving overlaps by priority (precise rules beat statistical NER) and then by
span length. Two extractors ship:

* :class:`RuleBasedExtractor` - lexicon + regex patterns: technologies,
  "Project X" codenames, titled/initialed person names, organization names
  with corporate head words, ISO/quarter/month dates, named events, and
  key/value fields in JSON records ("owner: ...", "organization: ...").
* :class:`SpacyExtractor` - spaCy ``en_core_web_sm`` NER for PERSON / ORG /
  GPE / LOC / DATE / EVENT. Optional; the system degrades to rules only.

A third pass, :func:`known_name_mentions`, runs at entity-rebuild time: names
discovered anywhere in the corpus (e.g. "Project Halcyon") are matched as bare
mentions everywhere else (e.g. "Halcyon"), a corpus-adaptive gazetteer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from app.entities.gazetteer import (
    ACRONYM_STOPLIST,
    CITIES,
    NON_NAME_WORDS,
    ORG_HEAD_WORDS,
    TECHNOLOGY_ALIASES,
)
from app.entities.normalization import basic_normalize

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExtractedSpan:
    surface: str
    entity_type: str
    start: int
    end: int
    extractor: str
    confidence: float
    priority: int


class EntityExtractor(Protocol):
    name: str

    def extract_many(self, texts: list[str]) -> list[list[ExtractedSpan]]: ...


def _word_boundary(pattern: str) -> str:
    return rf"(?<![\w-]){pattern}(?![\w-])"


_CAP = r"[A-Z][a-zA-Z0-9]+"
# Capitalized only because they start a sentence; trimmed from the front of ORG/EVENT spans.
_COMMON_LEADING = frozenset(
    {
        "At", "In", "On", "For", "From", "With", "By", "And", "But", "When", "After", "Before",
        "During", "Since", "While", "If", "Both", "Meanwhile", "Today", "Yesterday", "Also",
        "Then", "Per", "Via", "Unlike", "Like", "Once", "Until", "As", "To", "Of",
        "Why", "How", "What", "Which", "Who", "Where", "Does", "Did", "Is", "Was", "Are",
    }
) | NON_NAME_WORDS - {"Data", "Health", "Labs", "Systems"}
_MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|November|December|"
    "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)


def _trim_leading(surface: str, start: int) -> tuple[str, int]:
    tokens = surface.split(" ")
    while len(tokens) > 1 and tokens[0] in _COMMON_LEADING:
        start += len(tokens[0]) + 1
        tokens = tokens[1:]
    return " ".join(tokens), start


class RuleBasedExtractor:
    name = "rules"

    def __init__(self) -> None:
        variants = sorted(
            {v for vs in TECHNOLOGY_ALIASES.values() for v in vs}, key=len, reverse=True
        )
        self._tech_re = re.compile(
            _word_boundary("(?:" + "|".join(re.escape(v) for v in variants) + ")")
        )
        org_heads = "|".join(re.escape(h) for h in sorted(ORG_HEAD_WORDS, key=len, reverse=True))
        self._patterns: list[tuple[re.Pattern[str], str, int, float, int]] = [
            # (pattern, type, capture group, confidence, priority)
            (re.compile(rf"\bProject\s+({_CAP})"), "PROJECT", 0, 0.95, 90),
            (re.compile(rf"\b({_CAP})\s+(?:project|initiative|pilot|programme|program)\b"), "PROJECT", 1, 0.85, 85),
            (re.compile(rf"\bcodenamed?\s+\"?({_CAP})"), "PROJECT", 1, 0.9, 88),
            (re.compile(r"(?m)^project(?:_name)?:\s*(?:Project\s+)?([A-Z][\w-]+)\s*$"), "PROJECT", 1, 0.9, 88),
            (re.compile(r"\b(?:Dr|Prof|Mr|Mrs|Ms)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"), "PERSON", 0, 0.9, 80),
            (re.compile(r"\b[A-Z]\.\s?[A-Z][a-z]{2,}\b"), "PERSON", 0, 0.85, 80),
            (
                re.compile(
                    r"(?m)^(?i:name|owner|lead|tech_lead|tech lead|author|assignee|reporter|contact|manager|"
                    r"sponsor|from|to|user|speaker|host|product manager|incident commander)(?:\.name)?:\s*"
                    r"([A-Z][a-z]+(?:\s+[A-Z][a-z'-]+){1,2})(?=\s*(?:$|[(,;]))"
                ),
                "PERSON",
                1,
                0.9,
                82,
            ),
            (
                re.compile(
                    r"(?m)^(?:organization|organisation|company|employer|vendor|customer|account|"
                    r"client|org|partner)(?:\.name)?:\s*([A-Z][\w&.,' -]+?)\s*$"
                ),
                "ORGANIZATION",
                1,
                0.9,
                82,
            ),
            (
                re.compile(rf"\b(?:{_CAP}\s){{0,2}}{_CAP}\s(?:{org_heads})(?:,?\s(?:Inc|Ltd|LLC)\.?)?(?![\w])"),
                "ORGANIZATION",
                0,
                0.8,
                70,
            ),
            (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "DATE", 0, 0.95, 75),
            (re.compile(r"\bQ[1-4]\s\d{4}\b"), "DATE", 0, 0.95, 75),
            (re.compile(rf"\b(?:{_MONTHS})\.?\s(?:\d{{1,2}},\s)?\d{{4}}\b"), "DATE", 0, 0.9, 75),
            (
                re.compile(
                    rf"\b(?:{_CAP}\s){{1,3}}(?:Summit|Offsite|Conference|Hackathon|Hack Week|Retreat|Kickoff|Expo)(?:\s\d{{4}})?\b"
                ),
                "EVENT",
                0,
                0.8,
                72,
            ),
            (
                re.compile(_word_boundary("(?:" + "|".join(re.escape(c) for c in CITIES) + ")")),
                "LOCATION",
                0,
                0.8,
                60,
            ),
        ]

    def extract(self, text: str) -> list[ExtractedSpan]:
        spans: list[ExtractedSpan] = []
        for m in self._tech_re.finditer(text):
            spans.append(ExtractedSpan(m.group(0), "TECHNOLOGY", m.start(), m.end(), self.name, 0.95, 100))
        for pattern, entity_type, group, conf, priority in self._patterns:
            for m in pattern.finditer(text):
                surface = m.group(group).strip().rstrip(".,;")
                start = m.start(group)
                if entity_type in ("ORGANIZATION", "EVENT", "PERSON"):
                    surface, start = _trim_leading(surface, start)
                if not surface or len(surface.split()) == 0:
                    continue
                last = surface.split()[-1]
                if entity_type == "PROJECT" and last in NON_NAME_WORDS:
                    continue
                if entity_type == "PERSON" and last in _COMMON_LEADING:
                    continue
                if entity_type in ("ORGANIZATION", "EVENT") and len(surface.split()) < 2:
                    continue
                spans.append(
                    ExtractedSpan(surface, entity_type, start, start + len(surface), self.name, conf, priority)
                )
        return spans

    def extract_many(self, texts: list[str]) -> list[list[ExtractedSpan]]:
        return [self.extract(t) for t in texts]


_SPACY_LABELS = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "DATE": "DATE",
    "EVENT": "EVENT",
}
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")


def _ner_segments(text: str) -> list[tuple[int, str]]:
    """Split text into (offset, segment) runs of consecutive non-blank, non-heading lines."""
    segments: list[tuple[int, str]] = []
    start: int | None = None
    pos = 0
    for line in text.split("\n") + [""]:
        boundary = not line.strip() or bool(_HEADING_LINE_RE.match(line))
        if boundary and start is not None:
            segments.append((start, text[start : pos - 1]))
            start = None
        elif not boundary and start is None:
            start = pos
        pos += len(line) + 1
    return segments
_RELATIVE_DATE_RE = re.compile(
    r"^(?:today|yesterday|tomorrow|now|daily|weekly|monthly|annually|quarterly|"
    r"(?:this|last|next|each|every|the)\b.*|\d+\s+(?:days?|weeks?|months?|years?).*|"
    r"(?:a|one|two|three|four|five|six)\s+(?:days?|weeks?|months?|years?).*)$",
    re.IGNORECASE,
)


def _at_line_start(text: str, start: int) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    prefix = text[line_start:start].strip()
    return prefix in ("", "-", "*") or prefix.endswith(":")


class SpacyExtractor:
    name = "spacy"

    def __init__(self, model: str = "en_core_web_sm") -> None:
        import spacy

        self._nlp = spacy.load(model, disable=["lemmatizer", "parser"])

    @staticmethod
    def _clean(surface: str, start: int) -> tuple[str, int]:
        if "\n" in surface:
            # Spans occasionally cross a line break ("Team\n\nSamantha Reyes" when a heading
            # precedes a paragraph); keep the longest line, with its offset.
            offset, best = 0, ""
            pos = 0
            for segment in surface.split("\n"):
                if len(segment.strip()) > len(best):
                    best, offset = segment.strip(), pos + (len(segment) - len(segment.lstrip()))
                pos += len(segment) + 1
            surface, start = best, start + offset
        if surface.lower().startswith("the "):
            surface, start = surface[4:], start + 4
        surface = surface.removesuffix("'s").removesuffix("’s").rstrip(" .,;:")
        return surface, start

    def extract_many(self, texts: list[str]) -> list[list[ExtractedSpan]]:
        # NER runs per paragraph, excluding Markdown heading lines: the small model's
        # decisions are context-sensitive (a preceding "## Team" heading can make it miss
        # the name that follows), and paragraphs are the natural unit anyway. Offsets are
        # mapped back to the original text.
        segments = [(i, offset, seg) for i, text in enumerate(texts) for offset, seg in _ner_segments(text)]
        results: list[list[ExtractedSpan]] = [[] for _ in texts]
        docs = self._nlp.pipe([seg for _, _, seg in segments], batch_size=64)
        for (i, offset, _), doc in zip(segments, docs, strict=True):
            for ent in doc.ents:
                entity_type = _SPACY_LABELS.get(ent.label_)
                if entity_type is None:
                    continue
                surface, start = self._clean(ent.text, ent.start_char)
                if len(surface) < 2 or not any(ch.isalpha() for ch in surface):
                    continue
                if surface.upper() in ACRONYM_STOPLIST:
                    continue
                if " " not in surface and entity_type != "DATE" and _at_line_start(doc.text, start):
                    # Single capitalized tokens opening a line ("Collapse ...", "Diego,") are
                    # usually ordinary words or salutations, not names. Real names there are
                    # recovered by the corpus-level known-name pass.
                    continue
                if entity_type == "DATE":
                    if _RELATIVE_DATE_RE.match(surface) or not re.search(r"\d", surface):
                        continue
                elif not surface[0].isupper():
                    continue
                start += offset
                results[i].append(ExtractedSpan(surface, entity_type, start, start + len(surface), self.name, 0.6, 50))
        return results


def resolve_overlaps(spans: Iterable[ExtractedSpan]) -> list[ExtractedSpan]:
    """Keep the highest-priority (then longest) span among overlapping candidates."""
    accepted: list[ExtractedSpan] = []
    for span in sorted(spans, key=lambda s: (-s.priority, -(s.end - s.start), s.start)):
        if all(span.end <= a.start or span.start >= a.end for a in accepted):
            accepted.append(span)
    return sorted(accepted, key=lambda s: s.start)


class CompositeExtractor:
    def __init__(self, extractors: list[EntityExtractor]) -> None:
        self.extractors = extractors
        self.name = "+".join(e.name for e in extractors)

    def extract_many(self, texts: list[str]) -> list[list[ExtractedSpan]]:
        merged: list[list[ExtractedSpan]] = [[] for _ in texts]
        for extractor in self.extractors:
            for i, spans in enumerate(extractor.extract_many(texts)):
                merged[i].extend(spans)
        return [resolve_overlaps(spans) for spans in merged]

    def extract(self, text: str) -> list[ExtractedSpan]:
        return self.extract_many([text])[0]


def build_extractor(kind: str = "auto") -> CompositeExtractor:
    extractors: list[EntityExtractor] = [RuleBasedExtractor()]
    if kind in ("auto", "spacy"):
        try:
            extractors.append(SpacyExtractor())
        except Exception as exc:  # ImportError or missing model (OSError)
            if kind == "spacy":
                raise
            logger.warning("spaCy unavailable (%s); using rule-based extraction only", exc)
    return CompositeExtractor(extractors)


# ------------------------------------------------------------ known-name pass
@dataclass(slots=True)
class KnownName:
    surface: str
    entity_type: str


def derive_known_names(mentions: Iterable[tuple[str, str]]) -> list[KnownName]:
    """From (surface, type) pairs, derive bare forms worth matching corpus-wide.

    "Project Halcyon" -> "Halcyon"; "Quillon Labs" -> "Quillon"; full PERSON
    names are kept as-is so mentions spaCy missed elsewhere are recovered.
    """
    out: dict[tuple[str, str], KnownName] = {}
    for surface, entity_type in mentions:
        tokens = surface.split()
        if entity_type == "PROJECT":
            bare = [t for t in tokens if t.lower() not in ("project", "initiative", "program", "pilot")]
            if len(bare) == 1 and bare[0] not in NON_NAME_WORDS and len(bare[0]) >= 3:
                out[(bare[0], "PROJECT")] = KnownName(bare[0], "PROJECT")
        elif entity_type == "ORGANIZATION" and len(tokens) >= 2:
            head = tokens[0]
            if head not in NON_NAME_WORDS and len(head) >= 4 and head[0].isupper() and head.isalpha():
                out[(head, "ORGANIZATION")] = KnownName(head, "ORGANIZATION")
            out[(surface, "ORGANIZATION")] = KnownName(surface, "ORGANIZATION")
        elif entity_type == "PERSON" and len(tokens) >= 2 and all(t[:1].isupper() for t in tokens):
            if "." not in surface:
                out[(surface, "PERSON")] = KnownName(surface, "PERSON")
                # Bare first names and surnames of known people ("Marta", "Okafor").
                for token in (tokens[0], tokens[-1]):
                    if len(token) >= 3 and token.isalpha() and token not in NON_NAME_WORDS:
                        out[(token, "PERSON")] = KnownName(token, "PERSON")
    return list(out.values())


def known_name_mentions(text: str, names: list[KnownName], existing: list[tuple[int, int]]) -> list[ExtractedSpan]:
    """Find case-sensitive whole-word occurrences of known names not already covered."""
    spans: list[ExtractedSpan] = []
    taken = list(existing)
    for name in sorted(names, key=lambda n: len(n.surface), reverse=True):
        for m in re.finditer(_word_boundary(re.escape(name.surface)), text):
            if all(m.end() <= s or m.start() >= e for s, e in taken):
                spans.append(
                    ExtractedSpan(name.surface, name.entity_type, m.start(), m.end(), "known_name", 0.8, 40)
                )
                taken.append((m.start(), m.end()))
    return spans


def normalized_surface_key(surface: str) -> str:
    return basic_normalize(surface)
