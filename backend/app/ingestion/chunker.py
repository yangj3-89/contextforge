"""Structure-aware chunking.

Strategy:
  1. Blocks flagged ``breaks_before`` (top-level headings, PDF pages, JSON
     records) always start a new chunk, so chunks never straddle sections.
  2. Consecutive blocks inside a section are packed greedily up to
     ``max_tokens``.
  3. A single block larger than ``max_tokens`` is split on sentence
     boundaries with ``overlap_sentences`` of overlap; a trailing fragment
     smaller than ``min_tokens`` is folded into the previous piece.

Every chunk's ``text`` is an exact slice ``document.text[char_start:char_end]``,
which is what makes provenance verifiable. The text that is embedded/indexed
(``search_text``) additionally carries a short contextual header (document
title and section path) because a bare paragraph often omits the subject it
is about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.config import ChunkingSettings
from app.core.text import approx_token_count, sentence_spans
from app.ingestion.parsers import Block, ParsedDocument


@dataclass(slots=True)
class ChunkSpec:
    chunk_index: int
    text: str
    search_text: str
    token_count: int
    char_start: int
    char_end: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _Piece:
    start: int
    end: int
    tokens: int
    metadata: dict[str, Any]


def _split_long_block(block: Block, settings: ChunkingSettings) -> list[_Piece]:
    spans = sentence_spans(block.text) or [(0, len(block.text))]
    sentences = [
        (block.char_start + s, block.char_start + e, approx_token_count(block.text[s:e]))
        for s, e in spans
    ]
    pieces: list[_Piece] = []
    i = 0
    while i < len(sentences):
        j, tokens = i, 0
        while j < len(sentences) and (tokens + sentences[j][2] <= settings.max_tokens or j == i):
            tokens += sentences[j][2]
            j += 1
        pieces.append(_Piece(sentences[i][0], sentences[j - 1][1], tokens, block.metadata))
        if j >= len(sentences):
            break
        # Overlap: restart a few sentences back, but always make progress.
        i = max(j - settings.overlap_sentences, i + 1)
    if len(pieces) > 1 and pieces[-1].tokens < settings.min_tokens:
        last = pieces.pop()
        prev = pieces[-1]
        pieces[-1] = _Piece(prev.start, last.end, prev.tokens + last.tokens, prev.metadata)
    return pieces


def _context_header(title: str, metadata: dict[str, Any]) -> str:
    path = [h for h in metadata.get("heading_path", []) if h and h != title]
    return " > ".join([title, *path]) if title else " > ".join(path)


def chunk_document(doc: ParsedDocument, settings: ChunkingSettings) -> list[ChunkSpec]:
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    current_tokens = 0

    def close() -> None:
        nonlocal current, current_tokens
        if current:
            groups.append(current)
        current, current_tokens = [], 0

    for block in doc.blocks:
        tokens = approx_token_count(block.text)
        if block.breaks_before:
            close()
        if tokens > settings.max_tokens:
            pieces = _split_long_block(block, settings)
            if current and current_tokens < settings.min_tokens:
                # Fold a short lead-in (typically the section heading) into the first piece
                # instead of emitting a heading-only chunk.
                first = pieces[0]
                pieces[0] = _Piece(current[0].start, first.end, first.tokens + current_tokens, current[0].metadata)
                current, current_tokens = [], 0
            close()
            groups.extend([piece] for piece in pieces)
            continue
        if current and current_tokens + tokens > settings.max_tokens:
            close()
        current.append(_Piece(block.char_start, block.char_end, tokens, block.metadata))
        current_tokens += tokens
    close()

    specs: list[ChunkSpec] = []
    for group in groups:
        start, end = group[0].start, group[-1].end
        text = doc.text[start:end]
        metadata: dict[str, Any] = {k: v for k, v in group[0].metadata.items() if k != "page"}
        pages = sorted({p.metadata["page"] for p in group if "page" in p.metadata})
        if pages:
            metadata["page_start"], metadata["page_end"] = pages[0], pages[-1]
        header = _context_header(doc.title, group[0].metadata)
        specs.append(
            ChunkSpec(
                chunk_index=len(specs),
                text=text,
                search_text=f"{header}\n{text}" if header else text,
                token_count=approx_token_count(text),
                char_start=start,
                char_end=end,
                metadata=metadata,
            )
        )
    return specs
