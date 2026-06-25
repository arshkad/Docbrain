"""
Analytics layer for DocBrain.
Lightweight SQLite event log — tracks queries, uploads, and latency
without needing an external analytics service.
"""

import sqlite3
import time
import json
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timedelta, UTC

from app.config import settings

DB_PATH = Path(settings.chroma_persist_dir).parent / "analytics.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    """Create tables if they don't exist. Safe to call on every startup."""
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,      -- 'query' | 'upload' | 'summarize' | 'insights' | 'compare'
                collection_name TEXT,
                filename TEXT,
                question TEXT,
                latency_ms INTEGER,
                chunks_used INTEGER,
                success INTEGER DEFAULT 1,
                created_at TEXT NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_events_collection ON events(collection_name)")


def log_event(
    event_type: str,
    collection_name: str | None = None,
    filename: str | None = None,
    question: str | None = None,
    latency_ms: int | None = None,
    chunks_used: int | None = None,
    success: bool = True,
):
    """Record a single usage event. Never raises — analytics must not break the app."""
    try:
        with _conn() as c:
            c.execute(
                """INSERT INTO events
                   (event_type, collection_name, filename, question, latency_ms, chunks_used, success, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (event_type, collection_name, filename, question, latency_ms,
                 chunks_used, int(success), datetime.now(UTC).isoformat()),
            )
    except Exception:
        pass  # analytics failures should never break the main request

@contextmanager
def timed_event(event_type: str, **meta):
    """
    Context manager that times a block and logs it as an event.

    Usage:
        with timed_event("query", collection_name="x", question="y") as result:
            result["chunks_used"] = len(chunks)
            ...do work...
    """
    start = time.time()
    result = {"success": True}
    try:
        yield result
    except Exception:
        result["success"] = False
        raise
    finally:
        latency_ms = int((time.time() - start) * 1000)
        log_event(
            event_type=event_type,
            latency_ms=latency_ms,
            success=result.get("success", True),
            chunks_used=result.get("chunks_used"),
            **meta,
        )
        
# ─── Aggregation Queries ───────────────────────────────────────────────────────

def get_summary_stats(days: int = 30) -> dict:
    """High-level KPI numbers for the dashboard header."""
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    with _conn() as c:
        total_queries = c.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='query' AND created_at >= ?", (since,)
        ).fetchone()[0]

        total_uploads = c.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='upload' AND created_at >= ?", (since,)
        ).fetchone()[0]

        avg_latency = c.execute(
            "SELECT AVG(latency_ms) FROM events WHERE event_type='query' AND created_at >= ? AND success=1", (since,)
        ).fetchone()[0]

        success_rate = c.execute(
            """SELECT
                 SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*)
               FROM events WHERE created_at >= ?""", (since,)
        ).fetchone()[0]

        active_collections = c.execute(
            "SELECT COUNT(DISTINCT collection_name) FROM events WHERE created_at >= ? AND collection_name IS NOT NULL", (since,)
        ).fetchone()[0]

        total_events = c.execute(
            "SELECT COUNT(*) FROM events WHERE created_at >= ?", (since,)
        ).fetchone()[0]

    return {
        "total_queries": total_queries,
        "total_uploads": total_uploads,
        "avg_latency_ms": round(avg_latency) if avg_latency else 0,
        "success_rate": round((success_rate or 1.0) * 100, 1),
        "active_collections": active_collections,
        "total_events": total_events,
        "period_days": days,
    }