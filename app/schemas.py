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

class DocumentListResponse(BaseModel):
    collection: str
    documents: list[dict[str, Any]]
    total_documents: int