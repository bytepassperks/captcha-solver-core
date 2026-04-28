"""Randomized captcha evaluation runner — continuously tests detection and solving against demo endpoints."""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
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


class RandomSiteRunner:
    """Runs randomized captcha evaluation against configured target pool."""

    def __init__(self, dispatcher: Dispatcher, targets: list[str] | None = None):
        self.dispatcher = dispatcher
        self.targets = list(targets or DEFAULT_TARGETS)
        self._running = False
        self._task: asyncio.Task | None = None

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
            "fallback_chain": [],
            "solve_time_ms": 0,
            "success": False,
            "confidence": 0.0,
            "error": None,
            "detection_success": False,
        }

        if not self._is_domain_allowed(url):
            entry["error"] = f"Domain not in allowed list"
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
        entry["solve_time_ms"] = result.solve_time_ms
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
        # Keep last 500 entries
        entries = entries[-500:]
        LOG_FILE.write_text(json.dumps(entries, indent=2))

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
        latencies = [r["solve_time_ms"] for r in results if r["solve_time_ms"] > 0]
        avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0

        summary = {
            "tests_run": tests_run,
            "detections_successful": detections_ok,
            "solves_successful": solves_ok,
            "avg_latency_ms": avg_latency,
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.info(
            f"Random site run complete: {tests_run} tested, "
            f"{detections_ok} detected, {solves_ok} solved, "
            f"avg latency {avg_latency}ms"
        )
        return summary

    async def _scheduler_loop(self, interval_hours: float = 6):
        """Run tests on a repeating schedule."""
        self._running = True
        logger.info(f"Random site test scheduler started (interval={interval_hours}h)")

        while self._running:
            try:
                await self.run_once()
            except Exception as e:
                logger.error(f"Scheduled test run failed: {e}")

            jitter = random.uniform(-300, 300)
            sleep_seconds = max(60, interval_hours * 3600 + jitter)
            logger.info(f"Next random site test in {sleep_seconds / 3600:.1f}h")
            await asyncio.sleep(sleep_seconds)

    def start_scheduler(self, interval_hours: float = 6):
        """Start the scheduled test runner as a background task."""
        if self._task is not None and not self._task.done():
            logger.warning("Scheduler already running")
            return
        self._task = asyncio.create_task(
            self._scheduler_loop(interval_hours),
            name="random_site_test_scheduler",
        )

    def stop_scheduler(self):
        """Stop the scheduled test runner."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        logger.info("Random site test scheduler stopped")

    def get_latest_results(self, count: int = 20) -> list[dict]:
        """Get the most recent test results from the log file."""
        if not LOG_FILE.exists():
            return []
        try:
            entries = json.loads(LOG_FILE.read_text())
            return entries[-count:]
        except Exception:
            return []
