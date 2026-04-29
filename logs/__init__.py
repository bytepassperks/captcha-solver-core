"""Logging and telemetry module."""

import json
import time
import logging
import sqlite3
from pathlib import Path
from datetime import datetime

from config import config

_LOG_DB = config.logs_dir / "telemetry.db"

_CREATE_TELEMETRY_TABLE = """
CREATE TABLE IF NOT EXISTS solve_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    captcha_type TEXT,
    engine_used TEXT,
    success INTEGER,
    confidence REAL,
    solve_time_ms INTEGER,
    pageurl TEXT,
    sitekey TEXT,
    error TEXT
);
"""


def setup_logging():
    """Configure logging for the captcha solver system."""
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler()]

    if config.log_to_file:
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(config.logs_dir / "solver.log")
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=log_format,
        handlers=handlers,
    )


def _get_db():
    Path(_LOG_DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_LOG_DB))
    conn.executescript(_CREATE_TELEMETRY_TABLE)
    return conn


def log_solve(captcha_type: str, engine_used: str, success: bool,
              confidence: float, solve_time_ms: int,
              pageurl: str = "", sitekey: str = "", error: str = ""):
    """Log a solve attempt to the telemetry database."""
    try:
        conn = _get_db()
        conn.execute(
            "INSERT INTO solve_logs (timestamp, captcha_type, engine_used, success, "
            "confidence, solve_time_ms, pageurl, sitekey, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.utcnow().isoformat(),
                captcha_type,
                engine_used,
                int(success),
                confidence,
                solve_time_ms,
                pageurl,
                sitekey,
                error,
            )
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to log telemetry: {e}")


def get_stats(hours: int = 24) -> dict:
    """Get solve statistics for the last N hours."""
    try:
        conn = _get_db()
        cutoff = datetime.utcnow().timestamp() - (hours * 3600)
        cutoff_iso = datetime.utcfromtimestamp(cutoff).isoformat()

        total = conn.execute(
            "SELECT COUNT(*) FROM solve_logs WHERE timestamp > ?", (cutoff_iso,)
        ).fetchone()[0]

        success = conn.execute(
            "SELECT COUNT(*) FROM solve_logs WHERE timestamp > ? AND success = 1",
            (cutoff_iso,)
        ).fetchone()[0]

        by_type = {}
        rows = conn.execute(
            "SELECT captcha_type, COUNT(*), SUM(success), AVG(solve_time_ms) "
            "FROM solve_logs WHERE timestamp > ? GROUP BY captcha_type",
            (cutoff_iso,)
        ).fetchall()
        for row in rows:
            by_type[row[0]] = {
                "total": row[1],
                "success": row[2],
                "success_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0,
                "avg_solve_ms": round(row[3]) if row[3] else 0,
            }

        by_engine = {}
        rows = conn.execute(
            "SELECT engine_used, COUNT(*), SUM(success), AVG(solve_time_ms) "
            "FROM solve_logs WHERE timestamp > ? GROUP BY engine_used",
            (cutoff_iso,)
        ).fetchall()
        for row in rows:
            by_engine[row[0]] = {
                "total": row[1],
                "success": row[2],
                "success_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0,
                "avg_solve_ms": round(row[3]) if row[3] else 0,
            }

        conn.close()

        return {
            "period_hours": hours,
            "total_solves": total,
            "successful": success,
            "success_rate": round(success / total * 100, 1) if total > 0 else 0,
            "by_captcha_type": by_type,
            "by_engine": by_engine,
        }
    except Exception as e:
        return {"error": str(e)}
