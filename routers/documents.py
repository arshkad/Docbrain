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

@router.post("/text", response_model=IngestResponse, status_code=201)
async def ingest_text(body: TextIngestRequest):
    """
    Ingest raw text directly (no file upload needed).

    Useful for ingesting content from databases, APIs, or web scraping.
    """
    try:
        result = ingest_document(
            body.collection_name,
            body.filename,
            body.content,
            body.metadata,
        )
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(500, f"Ingestion failed: {e}")


@router.get("/{collection_name}", response_model=DocumentListResponse)
async def list_documents(
    collection_name: str,
    limit: int = Query(50, ge=1, le=200),
):
    """
    List all documents in a collection with their chunk counts and metadata.
    """
    try:
        col = chroma_client.get_collection(collection_name)
    except Exception:
        raise HTTPException(404, f"Collection '{collection_name}' not found")

    # Get all metadata to group by document
    results = col.get(include=["metadatas"], limit=10000)
    metadatas = results.get("metadatas") or []

    # Group chunks by document
    docs: dict[str, dict] = {}
    for meta in metadatas:
        filename = meta.get("filename", "unknown")
        if filename not in docs:
            docs[filename] = {
                "filename": filename,
                "doc_id": meta.get("doc_id", ""),
                "chunks": 0,
                "total_tokens": 0,
                "tags": [],
            }
        docs[filename]["chunks"] += 1
        docs[filename]["total_tokens"] += int(meta.get("token_count", 0))
        # Collect tags
        for k, v in meta.items():
            if k.startswith("tag_"):
                tag = str(v)
                if tag not in docs[filename]["tags"]:
                    docs[filename]["tags"].append(tag)

    doc_list = list(docs.values())[:limit]

    return DocumentListResponse(
        collection=collection_name,
        documents=doc_list,
        total_documents=len(docs),
    )

@router.delete("/{collection_name}/{filename}")
async def delete_document(collection_name: str, filename: str):
    """
    Delete all chunks of a specific document from a collection.
    """
    try:
        col = chroma_client.get_collection(collection_name)
    except Exception:
        raise HTTPException(404, f"Collection '{collection_name}' not found")

    # Find all chunk IDs for this document
    results = col.get(where={"filename": filename}, include=["metadatas"])
    ids = results.get("ids") or []

    if not ids:
        raise HTTPException(404, f"Document '{filename}' not found in collection")

    col.delete(ids=ids)
    return {"message": f"Deleted '{filename}' ({len(ids)} chunks) from '{collection_name}'"}
    
# ─── Bulk Operations & Tagging ────────────────────────────────────────────────

class BulkDeleteRequest(BaseModel):
    collection_name: str
    filenames: list[str]


class TagUpdateRequest(BaseModel):
    collection_name: str
    filename: str
    tags: list[str]


@router.post("/bulk-delete")
async def bulk_delete(body: BulkDeleteRequest):
    """
    Delete multiple documents from a collection in one call.
    Returns per-file results so partial failures are visible.
    """
    try:
        col = chroma_client.get_collection(body.collection_name)
    except Exception:
        raise HTTPException(404, f"Collection '{body.collection_name}' not found")

    results = []
    for filename in body.filenames:
        found = col.get(where={"filename": filename}, include=["metadatas"])
        ids = found.get("ids") or []
        if ids:
            col.delete(ids=ids)
            results.append({"filename": filename, "deleted_chunks": len(ids), "success": True})
        else:
            results.append({"filename": filename, "deleted_chunks": 0, "success": False})

    return {
        "results": results,
        "total_deleted": sum(r["deleted_chunks"] for r in results),
        "files_processed": len(results),
    }
