"""Name normalization used by entity resolution and query-time entity linking."""

from __future__ import annotations

import re
import unicodedata

from app.entities.gazetteer import ORG_SUFFIXES, PERSON_TITLES, PROJECT_WORDS

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def basic_normalize(text: str) -> str:
    """Lowercase, ASCII-fold, '&' -> 'and', drop punctuation, collapse whitespace."""
    text = ascii_fold(text).lower().replace("&", " and ")
    text = text.replace("'s ", " ").removesuffix("'s")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def strip_affixes(tokens: list[str], entity_type: str) -> list[str]:
    """Remove type-specific noise tokens (corporate suffixes, titles, 'project')."""
    tokens = list(tokens)
    if tokens and tokens[0] == "the":
        tokens = tokens[1:]
    if entity_type == "ORGANIZATION":
        while len(tokens) > 1 and tokens[-1] in ORG_SUFFIXES:
            tokens.pop()
    elif entity_type == "PERSON":
        while len(tokens) > 1 and tokens[0] in PERSON_TITLES:
            tokens = tokens[1:]
    elif entity_type == "PROJECT":
        filtered = [t for t in tokens if t not in PROJECT_WORDS]
        tokens = filtered or tokens
    return tokens


def name_tokens(surface: str, entity_type: str) -> list[str]:
    return strip_affixes(basic_normalize(surface).split(), entity_type)


def normalize_name(surface: str, entity_type: str) -> str:
    """Canonical comparison form, e.g. 'OpenAI Inc.' -> 'openai', 'Dr. Alice Chen' -> 'alice chen'."""
    return " ".join(name_tokens(surface, entity_type))


def compact_form(surface: str, entity_type: str) -> str:
    """Whitespace-free form so 'Open AI' and 'OpenAI' compare equal."""
    return normalize_name(surface, entity_type).replace(" ", "")


def harmonization_key(surface: str) -> str:
    """Type-agnostic key used to reconcile conflicting type labels for one surface form."""
    tokens = basic_normalize(surface).split()
    for entity_type in ("ORGANIZATION", "PERSON", "PROJECT"):
        tokens = strip_affixes(tokens, entity_type)
    return "".join(tokens)


def is_title_form(surface: str) -> bool:
    """'Dr. Chen' style: a title followed by a single name token."""
    tokens = basic_normalize(surface).split()
    return len(tokens) == 2 and tokens[0] in PERSON_TITLES


def initials(tokens: list[str]) -> str:
    return "".join(t[0] for t in tokens if t)
