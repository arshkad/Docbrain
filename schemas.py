"""Pydantic models for API request/response validation."""

from pydantic import BaseModel, Field
from typing import Any


# ─── Collections ──────────────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str = Field(..., pattern=r"^[a-z0-9_-]{3,50}$",
                      description="Lowercase alphanumeric, hyphens and underscores only")
    description: str | None = Field(None, max_length=200)

class CollectionResponse(BaseModel):
    name: str
    document_chunks: int
    metadata: dict

class CollectionList(BaseModel):
    collections: list[CollectionResponse]
    total: int

# ─── Documents ────────────────────────────────────────────────────────────────

class TextIngestRequest(BaseModel):
    collection_name: str
    filename: str = Field(..., description="Logical name for this document, e.g. 'q3-report.txt'")
    content: str = Field(..., min_length=50, description="Raw text content to ingest")
    metadata: dict[str, str] | None = Field(None, description="Optional tags/labels")

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_stored: int
    total_tokens: int
    collection: str
    classification: dict[str, Any] | None = Field(
        None, description="Auto-predicted document type from the local PyTorch classifier, if trained"
    )
    storage_backend: str | None = Field(
        None, description="Where the original file bytes were stored: 's3', 'local', or null if storage failed"
    )
    storage_warning: str | None = Field(
        None, description="Present if original-file storage failed; ingestion/search still succeeded"
    )

class DocumentListResponse(BaseModel):
    collection: str
    documents: list[dict[str, Any]]
    total_documents: int

class ClassifyResponse(BaseModel):
    filename: str
    predicted_type: str
    confidence: float
    top_k: list[dict[str, Any]]
    model: str

class DownloadUrlResponse(BaseModel):
    filename: str
    backend: str = Field(..., description="'s3' or 'local'")
    size_bytes: int
    url: str | None = Field(None, description="Presigned S3 URL; null when backend is 'local'")
    expires_in_seconds: int | None = None
