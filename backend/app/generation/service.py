"""Retrieve -> gate -> (optionally) generate a cited answer."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import GenerationSettings
from app.embeddings.base import Embedder
from app.generation.base import Evidence, GenerationError, Generator, cited_markers
from app.generation.extractive import ExtractiveGenerator
from app.generation.ollama import OllamaGenerator
from app.retrieval.engine import SearchOutcome


@dataclass
class AnswerOutcome:
    status: str  # success | insufficient_evidence | generation_error
    answer: str | None
    generator: str
    evidence: list[Evidence] = field(default_factory=list)
    cited: list[Evidence] = field(default_factory=list)
    unsupported_markers: list[int] = field(default_factory=list)
    message: str | None = None


def build_generator(settings: GenerationSettings, embedder: Embedder) -> Generator:
    if settings.backend == "ollama":
        return OllamaGenerator(settings.ollama_url, settings.ollama_model, settings.timeout_seconds)
    return ExtractiveGenerator(embedder)


class AnswerService:
    def __init__(self, generator: Generator, max_evidence: int = 5) -> None:
        self.generator = generator
        self.max_evidence = max_evidence

    def answer(self, outcome: SearchOutcome) -> AnswerOutcome:
        if outcome.abstained:
            # The gate failed: do not call the model at all.
            return AnswerOutcome("insufficient_evidence", None, self.generator.name, message=outcome.message)
        evidence = [
            Evidence(i + 1, r.chunk_id, r.document_id, r.filename, r.text)
            for i, r in enumerate(outcome.results[: self.max_evidence])
        ]
        try:
            text = self.generator.generate(outcome.query, evidence)
        except GenerationError as exc:
            return AnswerOutcome("generation_error", None, self.generator.name, evidence, message=str(exc))
        if "INSUFFICIENT_EVIDENCE" in text:
            return AnswerOutcome(
                "insufficient_evidence", None, self.generator.name, evidence,
                message="The generator judged the retrieved evidence insufficient.",
            )
        by_marker = {e.marker: e for e in evidence}
        markers = cited_markers(text)
        cited = [by_marker[m] for m in markers if m in by_marker]
        unsupported = [m for m in markers if m not in by_marker]
        return AnswerOutcome("success", text, self.generator.name, evidence, cited, unsupported)
