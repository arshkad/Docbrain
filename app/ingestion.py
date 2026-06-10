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
