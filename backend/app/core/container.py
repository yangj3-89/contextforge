"""Dependency container: builds and wires every component from Settings.

The API, CLI scripts, evaluation harness and tests all construct the system
through this one class, so they exercise identical wiring.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.core.config import Settings
from app.db.factory import build_repository
from app.db.repository import Repository
from app.embeddings.base import Embedder
from app.embeddings.factory import build_embedder
from app.entities.extraction import CompositeExtractor, build_extractor
from app.entities.graph import EntityGraph
from app.entities.linking import QueryEntityLinker
from app.entities.resolution import EntityResolver
from app.entities.service import EntityService, RebuildReport
from app.generation.service import AnswerService, build_generator
from app.ingestion.pipeline import IngestionService, IngestResult
from app.retrieval.engine import EntityContext, RetrievalEngine
from app.retrieval.entity_scorer import EntityScorer

logger = logging.getLogger(__name__)


class Container:
    def __init__(
        self,
        settings: Settings,
        repo: Repository | None = None,
        embedder: Embedder | None = None,
        extractor: CompositeExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo or build_repository(settings)
        self.embedder = embedder or build_embedder(settings)
        self.extractor = extractor or build_extractor(settings.entity_extractor)
        self.resolver = EntityResolver(settings.resolution)
        self.entity_service = EntityService(self.repo, self.resolver, settings.entity_retrieval.header_context)
        self.ingestion = IngestionService(self.repo, self.embedder, self.extractor, settings.chunking)
        self._write_lock = threading.RLock()
        self._entity_ctx: EntityContext | None = None
        self.last_rebuild: RebuildReport | None = None
        self.engine = RetrievalEngine(self.repo, self.embedder, settings, self.entity_context)
        self.answers = AnswerService(
            build_generator(settings.generation, self.embedder), settings.generation.max_evidence
        )
        self._set_graph(self.entity_service.load_graph())

    # ---------------------------------------------------------------- entities
    def _set_graph(self, graph: EntityGraph) -> None:
        linker = QueryEntityLinker(graph, self.settings.entity_retrieval, self.extractor)
        scorer = EntityScorer(graph, self.settings.entity_retrieval)
        self._entity_ctx = EntityContext(graph, linker, scorer)

    def entity_context(self) -> EntityContext | None:
        return self._entity_ctx

    @property
    def graph(self) -> EntityGraph:
        assert self._entity_ctx is not None
        return self._entity_ctx.graph

    def rebuild_entities(self) -> RebuildReport:
        with self._write_lock:
            graph, report = self.entity_service.rebuild()
            self._set_graph(graph)
            self.last_rebuild = report
            return report

    # --------------------------------------------------------------- ingestion
    def ingest_files(self, files: list[tuple[str, bytes]]) -> list[IngestResult]:
        with self._write_lock:
            results = [self.ingestion.ingest_bytes(name, data) for name, data in files]
            if any(r.status == "indexed" for r in results):
                self.repo.finalize_ingest()
                self.rebuild_entities()
            return results

    def ingest_directory(self, directory: Path) -> list[IngestResult]:
        with self._write_lock:
            results = self.ingestion.ingest_directory(directory)
            self.repo.finalize_ingest()
            self.rebuild_entities()
            return results

    def delete_document(self, document_id: str) -> bool:
        with self._write_lock:
            deleted = self.repo.delete_document(document_id)
            if deleted:
                self.repo.finalize_ingest()
                self.rebuild_entities()
            return deleted

    def reset(self) -> None:
        with self._write_lock:
            self.repo.reset()
            self.rebuild_entities()
