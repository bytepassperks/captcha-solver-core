"""Continuous captcha benchmarking scheduler — tests detection and solving against real demo endpoints."""

import asyncio
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx

from config import config
from detector.captcha_detector import detect_from_html, CaptchaType
from router.dispatcher import Dispatcher, SolveRequest, SolveResult
from logs import log_solve

logger = logging.getLogger(__name__)

DEFAULT_TARGETS = [
    "https://www.google.com/recaptcha/api2/demo",
    "https://accounts.hcaptcha.com/demo",
    "https://demo.turnstile.workers.dev",
    "https://recaptcha-demo.appspot.com/recaptcha-v2-checkbox.php",
    "https://recaptcha-demo.appspot.com/recaptcha-v3-request-scores.php",
]

LOG_FILE = config.logs_dir / "random_site_tests.json"
HISTORY_FILE = config.logs_dir / "benchmark_history.json"


class RandomSiteRunner:
    """Continuous captcha benchmarking engine with rolling performance tracking."""

    def __init__(self, dispatcher: Dispatcher, targets: list[str] | None = None):
        self.dispatcher = dispatcher
        self.targets = list(targets or DEFAULT_TARGETS)
        self._running = False
        self._task: asyncio.Task | None = None
        self._rolling_stats: dict = {
            "avg_latency_per_engine": {},
            "avg_success_per_engine": {},
            "cache_hit_rate": 0.0,
            "token_harvest_success_rate": 0.0,
        }

    def _is_domain_allowed(self, url: str) -> bool:
        domain = urlparse(url).netloc
        if not config.allowed_test_domains:
            return True
        return any(
            domain == d or domain.endswith("." + d)
            for d in config.allowed_test_domains
        )

    async def _fetch_html(self, url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
                })
                return resp.text
        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return ""

    async def _test_single(self, url: str) -> dict:
        """Run detection + solve against a single target URL."""
        entry = {
            "url": url,
            "timestamp": datetime.utcnow().isoformat(),
            "captcha_type": None,
            "sitekey": None,
            "engine_selected": None,
            "engine_race_winner": None,
            "fallback_chain": [],
            "latency_ms": 0,
            "success": False,
            "confidence": 0.0,
            "browser_profile_used": None,
            "error": None,
            "detection_success": False,
        }

        if not self._is_domain_allowed(url):
            entry["error"] = "Domain not in allowed list"
            return entry

        html = await self._fetch_html(url)
        if not html:
            entry["error"] = "Failed to fetch page HTML"
            return entry

        detection = detect_from_html(html)
        entry["captcha_type"] = detection.captcha_type.value
        entry["sitekey"] = detection.sitekey
        entry["detection_success"] = detection.captcha_type != CaptchaType.NONE

        if detection.captcha_type == CaptchaType.NONE:
            entry["error"] = "No captcha detected on page"
            return entry

        solve_req = SolveRequest(
            captcha_type=detection.captcha_type.value,
            pageurl=url,
            sitekey=detection.sitekey,
        )

        result = await self.dispatcher.solve(solve_req, detection)
        entry["engine_selected"] = result.engine_used
        entry["engine_race_winner"] = result.engine_used if result.success else None
        entry["latency_ms"] = result.solve_time_ms
        entry["success"] = result.success
        entry["confidence"] = result.confidence
        entry["error"] = result.error

        log_solve(
            detection.captcha_type.value, result.engine_used, result.success,
            result.confidence, result.solve_time_ms, url, detection.sitekey or "",
            result.error or "",
        )

        return entry

    def _append_log(self, entry: dict):
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        entries = []
        if LOG_FILE.exists():
            try:
                entries = json.loads(LOG_FILE.read_text())
            except Exception:
                entries = []
        entries.append(entry)
        entries = entries[-500:]
        LOG_FILE.write_text(json.dumps(entries, indent=2))

    def _update_rolling_stats(self, results: list[dict]):
        """Update rolling performance averages from benchmark results."""
        engine_latencies: dict[str, list] = {}
        engine_successes: dict[str, list] = {}

        for r in results:
            eng = r.get("engine_selected")
            if not eng:
                continue
            if r["latency_ms"] > 0:
                engine_latencies.setdefault(eng, []).append(r["latency_ms"])
            engine_successes.setdefault(eng, []).append(1 if r["success"] else 0)

        for eng, lats in engine_latencies.items():
            self._rolling_stats["avg_latency_per_engine"][eng] = int(sum(lats) / len(lats))
        for eng, succs in engine_successes.items():
            self._rolling_stats["avg_success_per_engine"][eng] = round(sum(succs) / len(succs) * 100, 1)

    def _append_history(self, summary: dict):
        """Append benchmark summary to rolling 7-day history file."""
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if HISTORY_FILE.exists():
            try:
                history = json.loads(HISTORY_FILE.read_text())
            except Exception:
                history = []

        history.append({
            "timestamp": summary["timestamp"],
            "tests_run": summary["tests_run"],
            "detections_successful": summary["detections_successful"],
            "solves_successful": summary["solves_successful"],
            "avg_latency_ms": summary["avg_latency_ms"],
            "fastest_engine": summary.get("fastest_engine"),
            "slowest_engine": summary.get("slowest_engine"),
        })

        # Keep 7 days (28 entries at 6h intervals)
        cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
        history = [h for h in history if h.get("timestamp", "") > cutoff]
        HISTORY_FILE.write_text(json.dumps(history, indent=2))

    async def run_once(self) -> dict:
        """Run a single evaluation pass against all targets (shuffled)."""
        targets = self.targets.copy()
        random.shuffle(targets)

        results = []
        for url in targets:
            entry = await self._test_single(url)
            results.append(entry)
            self._append_log(entry)
            await asyncio.sleep(random.uniform(1, 3))

        tests_run = len(results)
        detections_ok = sum(1 for r in results if r["detection_success"])
        solves_ok = sum(1 for r in results if r["success"])
        latencies = [r["latency_ms"] for r in results if r["latency_ms"] > 0]
        avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0

        # Find fastest/slowest engines
        engine_lats: dict[str, list] = {}
        for r in results:
            eng = r.get("engine_selected")
            if eng and r["latency_ms"] > 0:
                engine_lats.setdefault(eng, []).append(r["latency_ms"])

        fastest_engine = None
        slowest_engine = None
        if engine_lats:
            avg_by_eng = {e: sum(l) / len(l) for e, l in engine_lats.items()}
            fastest_engine = min(avg_by_eng, key=avg_by_eng.get)
            slowest_engine = max(avg_by_eng, key=avg_by_eng.get)

        self._update_rolling_stats(results)

        summary = {
            "tests_run": tests_run,
            "detections_successful": detections_ok,
            "solves_successful": solves_ok,
            "avg_latency_ms": avg_latency,
            "fastest_engine": fastest_engine,
            "slowest_engine": slowest_engine,
            "rolling_stats": self._rolling_stats,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._append_history(summary)

        logger.info(
            f"Benchmark complete: {tests_run} tested, "
            f"{detections_ok} detected, {solves_ok} solved, "
            f"avg latency {avg_latency}ms"
        )
        return summary

    async def _scheduler_loop(self, interval_hours: float = 6):
        """Run tests on a repeating schedule."""
        self._running = True
        logger.info(f"Benchmark scheduler started (interval={interval_hours}h)")

        while self._running:
            try:
                await self.run_once()
            except Exception as e:
                logger.error(f"Scheduled benchmark failed: {e}")

            jitter = random.uniform(-300, 300)
            sleep_seconds = max(60, interval_hours * 3600 + jitter)
            logger.info(f"Next benchmark in {sleep_seconds / 3600:.1f}h")
            await asyncio.sleep(sleep_seconds)

    def start_scheduler(self, interval_hours: float = 6):
        """Start the scheduled benchmark as a background task."""
        if self._task is not None and not self._task.done():
            logger.warning("Benchmark scheduler already running")
            return
        self._task = asyncio.create_task(
            self._scheduler_loop(interval_hours),
            name="benchmark_scheduler",
        )

    def stop_scheduler(self):
        """Stop the scheduled benchmark."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        logger.info("Benchmark scheduler stopped")

    def get_latest_results(self, count: int = 20) -> list[dict]:
        """Get the most recent test results from the log file."""
        if not LOG_FILE.exists():
            return []
        try:
            entries = json.loads(LOG_FILE.read_text())
            return entries[-count:]
        except Exception:
            return []

    def get_history(self, days: int = 7) -> list[dict]:
        """Get the rolling performance history."""
        if not HISTORY_FILE.exists():
            return []
        try:
            history = json.loads(HISTORY_FILE.read_text())
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            return [h for h in history if h.get("timestamp", "") > cutoff]
        except Exception:
            return []

    def get_rolling_stats(self) -> dict:
        """Get the current rolling performance statistics."""
        return self._rolling_stats
