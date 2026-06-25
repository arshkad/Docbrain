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
