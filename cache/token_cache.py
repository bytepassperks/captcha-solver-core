"""SQLite-based token cache for storing and reusing captcha tokens."""

import time
import logging
import sqlite3
from pathlib import Path

from captcha_solver_core.config import config

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
    """Store and retrieve captcha tokens to avoid re-solving."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.cache_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_CREATE_TABLE)

    def store(self, sitekey: str, domain: str, captcha_type: str, token: str,
              ttl_seconds: int | None = None):
        """Store a token in the cache."""
        now = time.time()
        ttl = ttl_seconds or config.token_ttl_seconds
        expires_at = now + ttl

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO token_cache (sitekey, domain, captcha_type, token, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sitekey, domain, captcha_type, token, now, expires_at)
            )

        logger.info(f"Cached token for {domain}/{sitekey[:8]}... (TTL={ttl}s)")

    def get(self, sitekey: str, domain: str, captcha_type: str) -> str | None:
        """Retrieve a valid unused token from cache."""
        now = time.time()

        with sqlite3.connect(self.db_path) as conn:
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
                logger.info(f"Cache hit for {domain}/{sitekey[:8]}...")
                return token

        return None

    def get_available_count(self, sitekey: str, domain: str, captcha_type: str) -> int:
        """Count available unused tokens."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM token_cache "
                "WHERE sitekey = ? AND domain = ? AND captcha_type = ? "
                "AND used = 0 AND expires_at > ?",
                (sitekey, domain, captcha_type, now)
            ).fetchone()
            return row[0] if row else 0

    def cleanup_expired(self):
        """Remove expired tokens."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            deleted = conn.execute(
                "DELETE FROM token_cache WHERE expires_at < ?", (now,)
            ).rowcount
            if deleted:
                logger.info(f"Cleaned up {deleted} expired tokens")

    def get_stats(self) -> dict:
        """Get cache statistics."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
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
        }
