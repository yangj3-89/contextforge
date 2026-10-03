from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import get_container
from app.core.container import Container
from app.schemas.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(container: Container = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        storage_backend=container.repo.backend_name,
        embedding_model=container.embedder.name,
        entity_extractor=container.extractor.name,
        generation_backend=container.answers.generator.name,
        counts=container.repo.counts(),
    )
