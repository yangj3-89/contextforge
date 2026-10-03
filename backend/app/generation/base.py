"""Answer generation interface (optional layer on top of retrieval).

Retrieval never depends on generation. Generators receive only evidence that
already passed the confidence gate, and every citation marker they emit is
checked against that evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Evidence:
    marker: int
    chunk_id: str
    document_id: str
    source: str
    text: str


class Generator(Protocol):
    name: str

    def generate(self, query: str, evidence: list[Evidence]) -> str: ...


class GenerationError(RuntimeError):
    pass


_CITATION_RE = re.compile(r"\[(\d+)\]")


def cited_markers(answer: str) -> list[int]:
    return sorted({int(m) for m in _CITATION_RE.findall(answer)})


def build_prompt(query: str, evidence: list[Evidence]) -> str:
    blocks = "\n\n".join(f"[{e.marker}] (source: {e.source})\n{e.text}" for e in evidence)
    return (
        "Answer the question using ONLY the numbered evidence below. Cite every factual claim "
        "with its evidence number in square brackets, e.g. [1]. If the evidence does not contain "
        "the answer, reply exactly: INSUFFICIENT_EVIDENCE.\n\n"
        f"Evidence:\n{blocks}\n\nQuestion: {query}\nAnswer:"
    )
