from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_container
from app.core.container import Container
from app.models.domain import Document
from app.schemas.api import ChunkOut, DocumentDetail, DocumentSummary, UploadItem, UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


def _summary(doc: Document) -> DocumentSummary:
    return DocumentSummary(
        document_id=doc.id,
        filename=doc.filename,
        title=doc.title,
        source_type=doc.source_type,
        content_hash=doc.content_hash,
        created_at=doc.created_at,
        chunk_count=doc.chunk_count,
        metadata=doc.metadata,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(files: list[UploadFile] = File(...), container: Container = Depends(get_container)) -> UploadResponse:
    limit = container.settings.max_upload_mb * 1024 * 1024
    payload: list[tuple[str, bytes]] = []
    rejected: list[UploadItem] = []
    for f in files:
        data = await f.read()
        name = f.filename or "upload"
        if len(data) > limit:
            rejected.append(UploadItem(filename=name, status="error", error=f"file exceeds {container.settings.max_upload_mb} MB"))
        elif not data:
            rejected.append(UploadItem(filename=name, status="error", error="empty file"))
        else:
            payload.append((name, data))
    results = container.ingest_files(payload) if payload else []
    items = [
        UploadItem(
            filename=r.filename, status=r.status, document_id=r.document_id,
            chunks=r.chunks, mentions=r.mentions, error=r.error,
        )
        for r in results
    ] + rejected
    stats = container.last_rebuild.stats if container.last_rebuild and any(r.status == "indexed" for r in results) else None
    return UploadResponse(items=items, entity_stats=stats)


@router.get("", response_model=list[DocumentSummary])
def list_documents(container: Container = Depends(get_container)) -> list[DocumentSummary]:
    return [_summary(d) for d in container.repo.list_documents()]


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(document_id: str, container: Container = Depends(get_container)) -> DocumentDetail:
    doc = container.repo.get_document(document_id)
    if doc is None:
        raise HTTPException(404, f"document '{document_id}' not found")
    chunks = [
        ChunkOut(
            chunk_id=c.id, chunk_index=c.chunk_index, text=c.text, token_count=c.token_count,
            char_start=c.char_start, char_end=c.char_end, metadata=c.metadata,
        )
        for c in container.repo.get_document_chunks(document_id)
    ]
    return DocumentDetail(**_summary(doc).model_dump(), chunks=chunks)


@router.delete("/{document_id}", status_code=204)
def delete_document(document_id: str, container: Container = Depends(get_container)) -> None:
    if not container.delete_document(document_id):
        raise HTTPException(404, f"document '{document_id}' not found")
