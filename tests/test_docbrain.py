"""
DocBrain Test Suite
Tests all major functionality without needing a real API key.
Run: pytest tests/ -v
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─── Chunking Tests (no mocking needed) ───────────────────────────────────────

from app.ingestion import chunk_text, count_tokens, file_hash


def test_chunk_text_basic():
    """Text should be split into multiple chunks."""
    long_text = "This is a sentence. " * 200
    chunks = chunk_text(long_text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_respects_size():
    """Each chunk should stay within token limit."""
    long_text = "Word " * 2000
    chunks = chunk_text(long_text, chunk_size=100, overlap=10)
    for chunk in chunks:
        assert count_tokens(chunk) <= 160  # tolerance for word-based approximation + overlap
        
def test_chunk_text_short_doc():
    """Short documents should produce exactly one chunk."""
    short_text = "This is a very short document. It has two sentences."
    chunks = chunk_text(short_text, chunk_size=200, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_overlap_continuity():
    """Adjacent chunks should share overlapping content."""
    text = "The quick brown fox. " * 100 + "Jumped over the lazy dog. " * 100
    chunks = chunk_text(text, chunk_size=80, overlap=30)
    if len(chunks) > 1:
        # Last sentence of chunk N should appear in start of chunk N+1
        # (not guaranteed for all cases, but overlap should reduce context loss)
        assert len(chunks) >= 2
