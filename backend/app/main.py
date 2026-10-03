"""FastAPI application factory.

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import documents, entities, evaluation, health, search
from app.core.config import get_settings
from app.core.container import Container


def create_app(container: Container | None = None) -> FastAPI:
    settings = container.settings if container else get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if getattr(app.state, "container", None) is None:
            app.state.container = Container(settings)
        container: Container = app.state.container
        if settings.bootstrap_dir and container.repo.counts()["documents"] == 0:
            results = container.ingest_directory(settings.bootstrap_dir)
            logging.getLogger(__name__).info("Bootstrapped %d files from %s", len(results), settings.bootstrap_dir)
        yield

    app = FastAPI(
        title="ContextForge",
        version=__version__,
        description="Provenance-aware hybrid retrieval and entity resolution with confidence-aware abstention.",
        lifespan=lifespan,
    )
    app.state.container = container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (health.router, documents.router, search.router, entities.router, evaluation.router):
        app.include_router(router)
    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
