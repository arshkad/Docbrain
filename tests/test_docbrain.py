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
