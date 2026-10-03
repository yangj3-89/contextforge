"""Ingestion: bytes -> parse -> chunk -> embed -> extract mentions -> store.

Ingestion is idempotent per (filename, content hash): re-uploading an identical
file is reported as a duplicate instead of creating a second copy. Document and
chunk IDs are derived deterministically from the filename and content, so the
same corpus always produces the same IDs (useful for evaluation datasets and
for diffing runs).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import ChunkingSettings
from app.db.repository import Repository
from app.embeddings.base import Embedder
from app.entities.extraction import CompositeExtractor
from app.ingestion.chunker import chunk_document
from app.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    ParseError,
    UnsupportedFormatError,
    parse_document,
)
from app.models.domain import Chunk, Document, Mention

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    filename: str
    status: str  # "indexed" | "duplicate" | "error"
    document_id: str | None = None
    chunks: int = 0
    mentions: int = 0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def document_id_for(filename: str, content_hash: str) -> str:
    return "doc_" + hashlib.sha256(f"{filename}\x00{content_hash}".encode()).hexdigest()[:16]


class IngestionService:
    def __init__(
        self,
        repo: Repository,
        embedder: Embedder,
        extractor: CompositeExtractor,
        chunking: ChunkingSettings,
    ) -> None:
        self.repo = repo
        self.embedder = embedder
        self.extractor = extractor
        self.chunking = chunking

    def ingest_bytes(self, filename: str, data: bytes, metadata: dict[str, Any] | None = None) -> IngestResult:
        filename = Path(filename).name
        content_hash = hashlib.sha256(data).hexdigest()
        duplicate = self.repo.find_duplicate(filename, content_hash)
        if duplicate is not None:
            return IngestResult(filename, "duplicate", duplicate.id, duplicate.chunk_count)
        try:
            parsed = parse_document(filename, data)
        except (UnsupportedFormatError, ParseError) as exc:
            return IngestResult(filename, "error", error=str(exc))

        doc_id = document_id_for(filename, content_hash)
        specs = chunk_document(parsed, self.chunking)
        if not specs:
            return IngestResult(filename, "error", error="document produced no chunks")
        embeddings = self.embedder.embed([s.search_text for s in specs])
        chunks = [
            Chunk(
                id=f"{doc_id}:{spec.chunk_index:04d}",
                document_id=doc_id,
                chunk_index=spec.chunk_index,
                text=spec.text,
                search_text=spec.search_text,
                token_count=spec.token_count,
                char_start=spec.char_start,
                char_end=spec.char_end,
                metadata=spec.metadata,
                embedding=embeddings[i],
            )
            for i, spec in enumerate(specs)
        ]
        mentions: list[Mention] = []
        for chunk, spans in zip(chunks, self.extractor.extract_many([c.text for c in chunks]), strict=True):
            for n, span in enumerate(spans):
                mentions.append(
                    Mention(
                        id=f"{chunk.id}:m{n:03d}",
                        chunk_id=chunk.id,
                        document_id=doc_id,
                        surface=span.surface,
                        entity_type=span.entity_type,
                        char_start=span.start,
                        char_end=span.end,
                        extractor=span.extractor,
                        confidence=span.confidence,
                    )
                )
        document = Document(
            id=doc_id,
            filename=filename,
            source_type=parsed.source_type,
            content_hash=content_hash,
            created_at=datetime.now(UTC),
            text=parsed.text,
            title=parsed.title,
            metadata={**parsed.metadata, **(metadata or {}), "size_bytes": len(data)},
            chunk_count=len(chunks),
        )
        self.repo.add_document(document, chunks, mentions)
        logger.info("Indexed %s: %d chunks, %d mentions", filename, len(chunks), len(mentions))
        return IngestResult(filename, "indexed", doc_id, len(chunks), len(mentions))

    def ingest_directory(self, directory: Path) -> list[IngestResult]:
        results = []
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                results.append(self.ingest_bytes(path.name, path.read_bytes(), {"source_path": str(path.relative_to(directory))}))
        return results
