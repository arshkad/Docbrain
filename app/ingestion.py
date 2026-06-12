"""
Document ingestion pipeline.
Handles parsing (PDF/text), chunking, and storing in the vector database.
"""

import re
import uuid
import hashlib
from pathlib import Path
from typing import BinaryIO

from PyPDF2 import PdfReader

from app.config import settings
from app.database import get_or_create_collection

# ─── Text Extraction ─────────────────────────────────────────────────────────

def extract_text_from_pdf(file: BinaryIO) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def extract_text_from_txt(file: BinaryIO) -> str:
    """Read plain text / markdown / CSV files."""
    raw = file.read()
    # Try UTF-8 first, fall back to latin-1
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def extract_text(filename: str, file: BinaryIO) -> str:
    """Dispatch to the correct extractor based on file extension."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file)
    elif ext in {".txt", ".md", ".csv"}:
        return extract_text_from_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# ─── Chunking ─────────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.35))


def chunk_text(
    text: str,
    chunk_size: int = settings.chunk_size,
    overlap: int = settings.chunk_overlap,
) -> list[str]:
    """
    Sentence-aware chunking with token-based sizing.

    Strategy:
    1. Split into sentences
    2. Greedily fill chunks up to chunk_size tokens
    3. Add overlap by backtracking into the previous chunk
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        # Single sentence exceeds chunk — split by words
        if sentence_tokens > chunk_size:
            words = sentence.split()
            word_buffer = []
            word_tokens = 0
            for word in words:
                wt = count_tokens(word)
                if word_tokens + wt > chunk_size and word_buffer:
                    chunks.append(" ".join(word_buffer))
                    # overlap: keep last N tokens worth of words
                    overlap_words = []
                    overlap_t = 0
                    for w in reversed(word_buffer):
                        if overlap_t + count_tokens(w) > overlap:
                            break
                        overlap_words.insert(0, w)
                        overlap_t += count_tokens(w)
                    word_buffer = overlap_words + [word]
                    word_tokens = overlap_t + wt
                else:
                    word_buffer.append(word)
                    word_tokens += wt
            if word_buffer:
                current_sentences.append(" ".join(word_buffer))
                current_tokens += word_tokens
            continue

        if current_tokens + sentence_tokens > chunk_size and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Overlap: retain recent sentences up to `overlap` tokens
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                st = count_tokens(s)
                if overlap_tokens + st > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += st

            current_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_tokens + sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
    
# ─── Ingestion ─────────────────────────────────────────────────────────────────

def file_hash(content: str) -> str:
    """SHA-256 fingerprint for deduplication."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def ingest_document(
    collection_name: str,
    filename: str,
    content: str,
    extra_metadata: dict | None = None,
) -> dict:
    """
    Chunk and store a document in the vector database.

    Returns:
        Summary dict with chunk count and document ID.
    """
    collection = get_or_create_collection(collection_name)

    doc_id = f"{Path(filename).stem}_{file_hash(content)}"
    chunks = chunk_text(content)

    if not chunks:
        raise ValueError("No text could be extracted from document")

    # Build parallel lists for ChromaDB batch upsert
    ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = []
    for i, chunk in enumerate(chunks):
        meta = {
            "doc_id": doc_id,
            "filename": filename,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "token_count": count_tokens(chunk),
            **(extra_metadata or {}),
        }
        metadatas.append(meta)

    collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_stored": len(chunks),
        "total_tokens": sum(count_tokens(c) for c in chunks),
        "collection": collection_name,
    }
