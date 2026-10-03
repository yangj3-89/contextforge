import httpx
import pytest

from app.core.container import Container
from app.embeddings.hashing import HashingEmbedder
from app.generation.base import Evidence, GenerationError, build_prompt, cited_markers
from app.generation.extractive import ExtractiveGenerator
from app.generation.ollama import OllamaGenerator
from app.generation.service import AnswerService

EVIDENCE = [
    Evidence(1, "c1", "d1", "a.md", "Halcyon stores embeddings in PostgreSQL with pgvector. It runs on Kubernetes."),
    Evidence(2, "c2", "d2", "b.txt", "The on-call rotation starts on Tuesday at ten in the morning."),
]


def test_extractive_generator_copies_and_cites_sentences():
    answer = ExtractiveGenerator(HashingEmbedder()).generate("Where are embeddings stored?", EVIDENCE)
    assert "PostgreSQL" in answer and "[1]" in answer
    assert cited_markers(answer)[0] == 1


def test_prompt_numbers_evidence():
    prompt = build_prompt("q?", EVIDENCE)
    assert "[1] (source: a.md)" in prompt and "INSUFFICIENT_EVIDENCE" in prompt


def test_ollama_generator_uses_api_and_handles_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        return httpx.Response(200, json={"response": "Embeddings live in PostgreSQL [1] and also [7]."})

    gen = OllamaGenerator("http://ollama:11434", "llama3.2:3b", client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert "PostgreSQL [1]" in gen.generate("q", EVIDENCE)

    failing = OllamaGenerator(
        "http://ollama:11434", "m", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    )
    with pytest.raises(GenerationError):
        failing.generate("q", EVIDENCE)


class _RecordingGenerator:
    name = "recording"

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, query, evidence):
        self.calls += 1
        return self.text


def test_answer_service_flags_unsupported_citations(indexed: Container):
    outcome = indexed.engine.search("Which database stores Halcyon embeddings?", "hybrid")
    gen = _RecordingGenerator("PostgreSQL [1], invented fact [9].")
    result = AnswerService(gen).answer(outcome)
    assert result.status == "success"
    assert [c.marker for c in result.cited] == [1]
    assert result.unsupported_markers == [9]


def test_answer_service_never_calls_generator_when_abstaining(indexed: Container):
    outcome = indexed.engine.search("What is the boiling point of water at sea level?", "hybrid", threshold=0.99)
    assert outcome.abstained
    gen = _RecordingGenerator("should not be used [1]")
    result = AnswerService(gen).answer(outcome)
    assert result.status == "insufficient_evidence" and gen.calls == 0


def test_answer_service_respects_model_refusal(indexed: Container):
    outcome = indexed.engine.search("Which database stores Halcyon embeddings?", "hybrid")
    result = AnswerService(_RecordingGenerator("INSUFFICIENT_EVIDENCE")).answer(outcome)
    assert result.status == "insufficient_evidence" and result.answer is None
