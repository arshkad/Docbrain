"""
Tests for: analytics tracking, conversation history (multi-turn RAG),
bulk document operations, and tagging.
Run: pytest tests/test_new_features.py -v
"""

import sys
import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ─── Analytics: isolate each test with a fresh temp DB ────────────────────────

@pytest.fixture
def analytics_module():
    """Reload the analytics module pointed at a throwaway SQLite file per test."""
    import importlib
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("app.config.settings.chroma_persist_dir", os.path.join(tmpdir, "chroma")):
            from app import analytics
            importlib.reload(analytics)
            analytics.DB_PATH = __import__("pathlib").Path(tmpdir) / "analytics.db"
            analytics.init_db()
            yield analytics


def test_init_db_creates_table(analytics_module):
    """init_db should be idempotent and create the events table."""
    analytics_module.init_db()  # call twice — must not raise
    stats = analytics_module.get_summary_stats()
    assert stats["total_queries"] == 0
    assert stats["total_events"] == 0


def test_log_event_basic(analytics_module):
    """A logged query event should show up in summary stats."""
    analytics_module.log_event(
        "query", collection_name="legal", filename="contract.pdf",
        question="what are the terms?", latency_ms=320, chunks_used=4, success=True,
    )
    stats = analytics_module.get_summary_stats()
    assert stats["total_queries"] == 1
    assert stats["avg_latency_ms"] == 320
    assert stats["success_rate"] == 100.0


def test_log_event_never_raises(analytics_module):
    """Analytics failures must be swallowed, never break the caller."""
    with patch.object(analytics_module, "_conn", side_effect=Exception("db locked")):
        analytics_module.log_event("query", collection_name="x")  # should not raise


def test_success_rate_with_failures(analytics_module):
    """Mixed success/failure events should compute the correct percentage."""
    analytics_module.log_event("query", collection_name="x", success=True)
    analytics_module.log_event("query", collection_name="x", success=True)
    analytics_module.log_event("query", collection_name="x", success=False)
    stats = analytics_module.get_summary_stats()
    assert stats["success_rate"] == pytest.approx(66.7, abs=0.1)


def test_timeseries_includes_today(analytics_module):
    """The most recent day in the timeseries must be today, with correct count."""
    analytics_module.log_event("query", collection_name="x", success=True)
    series = analytics_module.get_query_volume_timeseries(days=5)
    assert len(series) == 5
    assert series[-1]["count"] == 1  # today's bucket has our event
    
def test_timeseries_fills_gaps(analytics_module):
    """Days with no events should appear as 0, not be skipped."""
    series = analytics_module.get_query_volume_timeseries(days=7)
    assert len(series) == 7
    assert all(d["count"] == 0 for d in series)


def test_top_documents_ranking(analytics_module):
    """Documents queried more often should rank higher."""
    for _ in range(3):
        analytics_module.log_event("query", filename="popular.pdf", success=True)
    analytics_module.log_event("query", filename="rare.pdf", success=True)
    top = analytics_module.get_top_documents()
    assert top[0]["filename"] == "popular.pdf"
    assert top[0]["count"] == 3


def test_event_breakdown(analytics_module):
    """Breakdown should group by event_type correctly."""
    analytics_module.log_event("query", success=True)
    analytics_module.log_event("upload", success=True)
    analytics_module.log_event("upload", success=True)
    breakdown = {b["type"]: b["count"] for b in analytics_module.get_event_breakdown()}
    assert breakdown["upload"] == 2
    assert breakdown["query"] == 1


def test_recent_queries_ordering(analytics_module):
    """Recent queries should return most recent first."""
    analytics_module.log_event("query", question="first question", success=True)
    analytics_module.log_event("query", question="second question", success=True)
    recent = analytics_module.get_recent_queries(limit=10)
    assert recent[0]["question"] == "second question"
