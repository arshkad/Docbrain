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