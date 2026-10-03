"""Text utilities shared by ingestion, lexical retrieval, and entity processing.

The lexical analyzer intentionally mirrors PostgreSQL's ``english`` text-search
configuration (Snowball stemmer + the Snowball English stop list) so that the
in-memory BM25 backend and the PostgreSQL backend index roughly the same terms.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

# Snowball English stop list (the same list PostgreSQL ships as english.stop).
STOPWORDS: frozenset[str] = frozenset(
    """
    i me my myself we our ours ourselves you your yours yourself yourselves he him his
    himself she her hers herself it its itself they them their theirs themselves what
    which who whom this that these those am is are was were be been being have has had
    having do does did doing a an the and but if or because as until while of at by for
    with about against between into through during before after above below to from up
    down in out on off over under again further then once here there when where why how
    all any both each few more most other some such no nor not only own same so than too
    very s t can will just don should now
    """.split()
)

_STEMMER = snowballstemmer.stemmer("english")

# Words, optionally joined by - _ . / into compounds (e.g. "QL-5031", "retrieval.top_k").
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-_./][A-Za-z0-9]+)*")
_COMPOUND_SPLIT_RE = re.compile(r"[-_./]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+\n")
_TOKEN_APPROX_RE = re.compile(r"\w+|[^\w\s]")


def normalize_text(text: str) -> str:
    """Normalize raw extracted text without changing its meaning.

    NFKC folds compatibility characters (ligatures, full-width forms, non-breaking
    spaces); line endings are unified, control characters removed, trailing
    whitespace stripped and runs of blank lines collapsed.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = _CONTROL_RE.sub("", text)
    text = _TRAILING_WS_RE.sub("\n", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def approx_token_count(text: str) -> int:
    """Approximate subword token count (words + punctuation).

    Close enough to WordPiece counts for chunk sizing without loading a tokenizer.
    """
    return len(_TOKEN_APPROX_RE.findall(text))


@lru_cache(maxsize=65536)
def stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def lexical_terms(text: str) -> list[str]:
    """Analyze text into index terms (lowercased, stop-word filtered, stemmed).

    Compound identifiers such as ``QL-5031`` produce the whole compound plus its
    parts, matching PostgreSQL's parser behaviour for hyphenated words.
    """
    terms: list[str] = []
    for match in _WORD_RE.finditer(text.replace("’", "'")):
        token = match.group(0).lower()
        if _COMPOUND_SPLIT_RE.search(token):
            terms.append(token)
            parts = [p for p in _COMPOUND_SPLIT_RE.split(token) if p]
        else:
            parts = [token]
        for part in parts:
            if part in STOPWORDS:
                continue
            if len(part) == 1 and not part.isdigit():
                continue
            terms.append(stem(part))
    return terms


def query_terms(text: str) -> list[str]:
    """Unique lexical terms of a query, in order of first appearance."""
    seen: dict[str, None] = {}
    for term in lexical_terms(text):
        seen.setdefault(term, None)
    return list(seen)


def plain_words(text: str) -> list[str]:
    """Surface words (no stemming) used to build PostgreSQL tsqueries."""
    return [m.group(0) for m in _WORD_RE.finditer(text)]


_ABBREVIATIONS = (
    "dr", "mr", "mrs", "ms", "prof", "inc", "corp", "ltd", "co", "jr", "sr", "st", "vs",
    "etc", "e.g", "i.e", "approx", "no", "fig", "dept", "est", "u.s",
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_LIST_OR_KEY_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}\s|\d{1,2}:\d{2}\s|[\w .-]{1,40}:\s)")
WRAP_MIN_LINE_LENGTH = 60
"""A line shorter than this is a complete line (agenda item, heading), not hard-wrapped prose."""


def _merge_wrapped_lines(text: str) -> list[str]:
    """Join hard-wrapped lines into logical lines; keep list items / key:value / short lines separate."""
    logical: list[str] = []
    previous_raw = ""
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if logical and logical[-1] != "":
                logical.append("")
            previous_raw = ""
            continue
        wrapped_continuation = (
            logical
            and logical[-1]
            and not _LIST_OR_KEY_RE.match(line)
            and not previous_raw.endswith((".", "!", "?", ":", ";"))
            and not _LIST_OR_KEY_RE.match(previous_raw)
            and len(previous_raw) >= WRAP_MIN_LINE_LENGTH  # the *raw* previous line looks wrapped
        )
        if wrapped_continuation:
            logical[-1] = f"{logical[-1]} {line}"
        else:
            logical.append(line)
        previous_raw = line
    return [line for line in logical if line]


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter that protects common abbreviations and initials."""
    sentences: list[str] = []
    for line in _merge_wrapped_lines(text):
        start = 0
        for match in _SENTENCE_BOUNDARY_RE.finditer(line):
            candidate = line[start : match.start()].rstrip()
            last_word = candidate.rsplit(" ", 1)[-1].rstrip(".").lower()
            # Initials ("A. Chen") and abbreviations ("Dr. Chen", "Inc.") are not boundaries.
            if len(last_word) == 1 or last_word in _ABBREVIATIONS:
                continue
            if candidate:
                sentences.append(candidate)
            start = match.end()
        tail = line[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans (start, end) of sentences located inside ``text``.

    Sentences are located by searching forward, so spans always index the
    original string even after hard-wrapped lines were joined.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in split_sentences(text):
        first_word = sentence.split(" ", 1)[0]
        idx = text.find(first_word, cursor)
        if idx < 0:
            continue
        # Walk forward over the sentence's words to find its end in the original text.
        end = idx
        for word in sentence.split(" "):
            pos = text.find(word, end)
            if pos < 0:
                break
            end = pos + len(word)
        spans.append((idx, end))
        cursor = end
    return spans
