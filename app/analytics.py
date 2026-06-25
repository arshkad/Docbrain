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
