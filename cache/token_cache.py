"""SQLite-based token cache for storing and reusing captcha tokens."""

import time
import logging
import sqlite3
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS token_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sitekey TEXT NOT NULL,
    domain TEXT NOT NULL,
    captcha_type TEXT NOT NULL,
    token TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    used INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cache_lookup ON token_cache(sitekey, domain, captcha_type, used, expires_at);
"""


class TokenCache:
    """Store and retrieve captcha tokens. Uses in-memory SQLite for hot cache + disk for persistence."""

    def __init__(self, db_path: str | None = None, use_memory: bool | None = None):
        self.db_path = db_path or config.cache_db_path
        self._use_memory = use_memory if use_memory is not None else config.cache_use_memory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._mem_conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._use_memory:
            if self._mem_conn is None:
                self._mem_conn = sqlite3.connect(":memory:")
                self._mem_conn.executescript(_CREATE_TABLE)
            return self._mem_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        # Always init disk DB
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_CREATE_TABLE)
        # Init memory DB if needed
        if self._use_memory:
            self._get_conn()

    def store(self, sitekey: str, domain: str, captcha_type: str, token: str,
              ttl_seconds: int | None = None):
        """Store a token in cache (memory + disk)."""
        now = time.time()
        ttl = ttl_seconds or config.token_ttl_seconds
        expires_at = now + ttl
        params = (sitekey, domain, captcha_type, token, now, expires_at)
        insert_sql = (
            "INSERT INTO token_cache (sitekey, domain, captcha_type, token, created_at, expires_at) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )

        conn = self._get_conn()
        conn.execute(insert_sql, params)
        conn.commit()

        # Also persist to disk if using memory
        if self._use_memory:
            with sqlite3.connect(self.db_path) as disk_conn:
                disk_conn.execute(insert_sql, params)

        logger.info(f"Cached token for {domain}/{sitekey[:8]}... (TTL={ttl}s)")

    def get(self, sitekey: str, domain: str, captcha_type: str) -> str | None:
        """Retrieve a valid unused token from cache (<1ms from memory)."""
        now = time.time()

        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, token FROM token_cache "
            "WHERE sitekey = ? AND domain = ? AND captcha_type = ? "
            "AND used = 0 AND expires_at > ? "
            "ORDER BY created_at ASC LIMIT 1",
            (sitekey, domain, captcha_type, now)
        ).fetchone()

        if row:
            token_id, token = row
            conn.execute("UPDATE token_cache SET used = 1 WHERE id = ?", (token_id,))
            conn.commit()
            logger.info(f"Cache hit for {domain}/{sitekey[:8]}...")
            return token

        return None

    def get_available_count(self, sitekey: str, domain: str, captcha_type: str) -> int:
        """Count available unused tokens."""
        now = time.time()
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM token_cache "
            "WHERE sitekey = ? AND domain = ? AND captcha_type = ? "
            "AND used = 0 AND expires_at > ?",
            (sitekey, domain, captcha_type, now)
        ).fetchone()
        return row[0] if row else 0

    def cleanup_expired(self):
        """Remove expired tokens from both memory and disk."""
        now = time.time()
        conn = self._get_conn()
        deleted = conn.execute(
            "DELETE FROM token_cache WHERE expires_at < ?", (now,)
        ).rowcount
        conn.commit()
        if self._use_memory:
            with sqlite3.connect(self.db_path) as disk_conn:
                disk_conn.execute("DELETE FROM token_cache WHERE expires_at < ?", (now,))
        if deleted:
            logger.info(f"Cleaned up {deleted} expired tokens")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        now = time.time()
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM token_cache").fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM token_cache WHERE used = 0 AND expires_at > ?",
            (now,)
        ).fetchone()[0]
        used = conn.execute(
            "SELECT COUNT(*) FROM token_cache WHERE used = 1"
        ).fetchone()[0]
        expired = conn.execute(
            "SELECT COUNT(*) FROM token_cache WHERE expires_at < ?",
            (now,)
        ).fetchone()[0]

        return {
            "total": total,
            "active": active,
            "used": used,
            "expired": expired,
            "backend": "memory" if self._use_memory else "disk",
        }
