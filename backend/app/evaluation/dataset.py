"""Evaluation dataset format and relevance resolution.

Relevance is labelled with *evidence strings* plus source filenames rather than
hard-coded chunk IDs:

    {
      "id": "q001",
      "query": "What error code did the Halcyon outage produce?",
      "category": "lexical",
      "answerable": true,
      "split": "test",
      "relevant_sources": ["halcyon_incident_postmortem.md"],
      "evidence": ["surfaced as error QL-5031"]
    }

At evaluation time every chunk of a relevant source whose text contains an
evidence string (case/whitespace-insensitive) becomes a relevant chunk. This
keeps labels valid when chunking parameters change, and the resolved
``relevant_document_ids`` / ``relevant_chunk_ids`` are written into the
results for auditability. Explicit ``relevant_chunk_ids`` are also accepted.
An evidence string that matches no chunk is a hard error, so a stale label
can never silently count as a miss.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.db.repository import Repository


class EvalQuery(BaseModel):
    id: str
    query: str
    category: str
    answerable: bool = True
    split: Literal["dev", "test"] = "test"
    relevant_sources: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> EvalQuery:
        if self.answerable and not (self.evidence or self.relevant_chunk_ids):
            raise ValueError(f"{self.id}: answerable queries need evidence or relevant_chunk_ids")
        if not self.answerable and (self.evidence or self.relevant_chunk_ids):
            raise ValueError(f"{self.id}: unanswerable queries must not list evidence")
        return self


class EvalDataset(BaseModel):
    name: str
    description: str = ""
    version: str = "1"
    queries: list[EvalQuery]

    @model_validator(mode="after")
    def _unique_ids(self) -> EvalDataset:
        ids = [q.id for q in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate query ids in dataset")
        return self


class DatasetIntegrityError(ValueError):
    pass


@dataclass
class ResolvedRelevance:
    document_ids: set[str]
    chunk_ids: set[str]


def load_dataset(path: Path) -> EvalDataset:
    return EvalDataset.model_validate(json.loads(path.read_text()))


_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def resolve_relevance(dataset: EvalDataset, repo: Repository) -> dict[str, ResolvedRelevance]:
    docs = repo.list_documents()
    by_filename: dict[str, list[str]] = {}
    for d in docs:
        by_filename.setdefault(d.filename, []).append(d.id)
    chunk_cache: dict[str, list[tuple[str, str]]] = {}

    def doc_chunks(doc_id: str) -> list[tuple[str, str]]:
        if doc_id not in chunk_cache:
            chunk_cache[doc_id] = [(c.id, _norm(c.text)) for c in repo.get_document_chunks(doc_id)]
        return chunk_cache[doc_id]

    resolved: dict[str, ResolvedRelevance] = {}
    errors: list[str] = []
    for q in dataset.queries:
        if not q.answerable:
            resolved[q.id] = ResolvedRelevance(set(), set())
            continue
        doc_ids = set(q.relevant_document_ids)
        for fname in q.relevant_sources:
            if fname not in by_filename:
                errors.append(f"{q.id}: source '{fname}' is not indexed")
                continue
            doc_ids.update(by_filename[fname])
        chunk_ids = set(q.relevant_chunk_ids)
        search_docs = doc_ids or {d.id for d in docs}
        for ev in q.evidence:
            needle = _norm(ev)
            hits = [cid for did in sorted(search_docs) for cid, text in doc_chunks(did) if needle in text]
            if not hits:
                errors.append(f"{q.id}: evidence not found in any chunk: {ev!r}")
            chunk_ids.update(hits)
        if chunk_ids and not doc_ids:
            doc_ids = {cid.split(":")[0] for cid in chunk_ids}
        resolved[q.id] = ResolvedRelevance(doc_ids, chunk_ids)
    if errors:
        raise DatasetIntegrityError("dataset does not match the index:\n  " + "\n  ".join(errors))
    return resolved
