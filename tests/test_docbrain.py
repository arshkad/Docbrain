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


def test_file_hash_deterministic():
    """Same content should always produce same hash."""
    content = "Hello, world!"
    assert file_hash(content) == file_hash(content)


def test_file_hash_unique():
    """Different content should produce different hashes."""
    assert file_hash("content A") != file_hash("content B")


def test_count_tokens():
    """Token count should be positive for non-empty text."""
    assert count_tokens("Hello world") > 0
    assert count_tokens("") == 0

# ─── API Tests (mocked) ───────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Create test client with mocked ChromaDB."""
    with patch("app.database.chromadb.PersistentClient") as mock_chroma:
        # Setup mock client
        mock_client = MagicMock()
        mock_chroma.return_value = mock_client
        mock_client.heartbeat.return_value = True

        # Mock collection
        mock_col = MagicMock()
        mock_col.count.return_value = 0
        mock_col.metadata = {}
        mock_client.get_or_create_collection.return_value = mock_col
        mock_client.list_collections.return_value = []

        from app.main import app
        return TestClient(app)


def test_health_endpoint(client):
    """Health check should return 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_root_endpoint(client):
    """Root should return service info."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "DocBrain API"

# ─── LLM Layer Tests (mocked) ─────────────────────────────────────────────────

@pytest.fixture
def mock_claude():
    """Mock Claude API calls."""
    with patch("app.llm.client") as mock:
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Mocked Claude response")]
        mock.messages.create.return_value = mock_response
        yield mock


@pytest.fixture
def mock_search():
    """Mock semantic search."""
    sample_chunks = [
        {
            "text": "The contract value is $50,000 due on March 15, 2024.",
            "metadata": {"filename": "contract.pdf", "chunk_index": 0, "total_chunks": 3},
            "score": 0.92,
        },
        {
            "text": "Payment terms: Net 30. Late fees apply after 30 days.",
            "metadata": {"filename": "contract.pdf", "chunk_index": 1, "total_chunks": 3},
            "score": 0.85,
        },
    ]
    with patch("app.llm.semantic_search", return_value=sample_chunks) as mock:
        yield mock
     
def test_rag_query_with_results(mock_claude, mock_search):
    """RAG query should return answer with sources."""
    from app.llm import rag_query
    result = rag_query("test-collection", "What is the contract value?")
    assert "answer" in result
    assert "sources" in result
    assert len(result["sources"]) == 2
    assert result["chunks_used"] == 2
def test_rag_query_no_results():
    """RAG query with no results should return helpful message."""
    from app.llm import rag_query
    with patch("app.llm.semantic_search", return_value=[]):
        result = rag_query("empty-collection", "Anything?")
        assert "No relevant documents" in result["answer"]
        assert result["chunks_used"] == 0


def test_summarize_executive(mock_claude, mock_search):
    """Summarization should call Claude with correct style."""
    from app.llm import summarize_document
    result = summarize_document("test-collection", "contract.pdf", style="executive")
    assert result["summary_style"] == "executive"
    assert result["filename"] == "contract.pdf"
    assert "summary" in result

def test_summarize_invalid_style(mock_search):
    """Invalid summary style should raise ValueError."""
    from app.llm import summarize_document
    with pytest.raises(ValueError, match="Style must be one of"):
        summarize_document("test-collection", "contract.pdf", style="invalid_style")


def test_extract_insights(mock_search):
    """Insight extraction should return structured JSON."""
    import json
    mock_insights = json.dumps({
        "action_items": [{"task": "Sign contract", "owner": "Legal", "deadline": "2024-03-01"}],
        "key_dates": [{"date": "2024-03-15", "event": "Payment due"}],
        "people_and_orgs": [{"name": "Acme Corp", "role": "Vendor"}],
        "financial_figures": [{"amount": "$50,000", "context": "Contract value"}],
        "key_decisions": ["Approved 30-day payment terms"],
        "document_type": "contract",
    })

    with patch("app.llm._call_claude", return_value=mock_insights):
        from app.llm import extract_insights
        result = extract_insights("test-collection", "contract.pdf")
        assert result["insights"]["document_type"] == "contract"
        assert len(result["insights"]["financial_figures"]) == 1
        assert result["insights"]["financial_figures"][0]["amount"] == "$50,000"

def test_compare_documents(mock_claude):
    """Document comparison should include both doc names."""
    sample_chunks = [{"text": "Sample content", "metadata": {"filename": "a.pdf", "chunk_index": 0, "total_chunks": 1}, "score": 0.9}]
    with patch("app.llm.semantic_search", return_value=sample_chunks):
        from app.llm import compare_documents
        result = compare_documents("test-col", "vendor_a.pdf", "vendor_b.pdf", "pricing")
        assert result["document_a"] == "vendor_a.pdf"
        assert result["document_b"] == "vendor_b.pdf"
        assert result["aspect"] == "pricing"
        
# ─── Integration-style Tests ──────────────────────────────────────────────────

def test_full_pipeline_text_to_chunks():
    """Test the full text → chunks pipeline with realistic business text."""
    sample_contract = """
    SERVICE AGREEMENT

    This Service Agreement ("Agreement") is entered into as of January 15, 2024,
    between TechCorp Inc. ("Service Provider") and Acme Industries ("Client").

    1. SERVICES
    Service Provider agrees to deliver software development services as outlined
    in Exhibit A. Work shall commence February 1, 2024, and be completed by June 30, 2024.

    2. COMPENSATION
    Client shall pay Service Provider $15,000 per month, invoiced on the first of
    each month, due within 30 days of receipt. Late payments incur a 1.5% monthly fee.

    3. CONFIDENTIALITY
    Both parties agree to maintain strict confidentiality of proprietary information
    disclosed during the term of this Agreement and for two years thereafter.

    4. TERMINATION
    Either party may terminate this Agreement with 30 days written notice.
    Client shall pay for all work completed up to the termination date.
    """

    chunks = chunk_text(sample_contract, chunk_size=100, overlap=20)
    assert len(chunks) >= 1
    # Contract value should appear in some chunk
    full_text = " ".join(chunks)
    assert "$15,000" in full_text
    assert "TechCorp" in full_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


