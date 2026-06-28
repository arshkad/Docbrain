"""Document ingestion endpoints — file upload and raw text."""

import io
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel
from app.schemas import TextIngestRequest, IngestResponse, DocumentListResponse
from app.ingestion import ingest_document, extract_text
from app.database import get_or_create_collection, chroma_client
from app.config import settings
from app.analytics import log_event

router = APIRouter()


@router.post("/upload", response_model=IngestResponse, status_code=201)
async def upload_file(
    collection_name: str = Form(..., description="Target collection name"),
    file: UploadFile = File(...),
    tags: str | None = Form(None, description="Comma-separated tags, e.g. 'legal,2024,urgent'"),
):
    """
    Upload a document file (PDF, TXT, MD, CSV) to a collection.

    The document is automatically:
    1. Parsed (text extracted from PDF if needed)
    2. Chunked with semantic overlap
    3. Embedded and stored in the vector database

    Max file size: 20MB
    """
    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(413, f"File too large: {size_mb:.1f}MB (max {settings.max_file_size_mb}MB)")

    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            415,
            f"Unsupported file type '{ext}'. Allowed: {settings.allowed_extensions}"
        )
    # Extract text
    try:
        text = extract_text(file.filename, io.BytesIO(content))
    except Exception as e:
        raise HTTPException(422, f"Failed to extract text: {e}")

    if len(text.strip()) < 50:
        raise HTTPException(422, "Document appears to be empty or unreadable")

    # Build metadata
    extra_meta = {"size_mb": str(round(size_mb, 3))}
    if tags:
        for i, tag in enumerate(tags.split(",")):
            extra_meta[f"tag_{i}"] = tag.strip()

    # Ingest
    try:
        result = ingest_document(collection_name, file.filename, text, extra_meta)
        log_event("upload", collection_name=collection_name, filename=file.filename,
                   chunks_used=result.get("chunks_stored"), success=True)
        return IngestResponse(**result)
    except Exception as e:
        log_event("upload", collection_name=collection_name, filename=file.filename, success=False)
        raise HTTPException(500, f"Ingestion failed: {e}")
