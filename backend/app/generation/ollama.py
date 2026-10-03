"""Local LLM generation through an Ollama server (http://localhost:11434)."""

from __future__ import annotations

import httpx

from app.generation.base import Evidence, GenerationError, build_prompt


class OllamaGenerator:
    def __init__(self, base_url: str, model: str, timeout: float = 60.0, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"ollama:{model}"
        self._client = client or httpx.Client(timeout=timeout)

    def generate(self, query: str, evidence: list[Evidence]) -> str:
        payload = {
            "model": self.model,
            "prompt": build_prompt(query, evidence),
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            response = self._client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise GenerationError(f"Ollama request failed: {exc}") from exc
        text = response.json().get("response", "").strip()
        if not text:
            raise GenerationError("Ollama returned an empty response")
        return text
