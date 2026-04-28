"""FastAPI server exposing the captcha solver as an API — v2.1.0."""

import asyncio
import base64
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import config
from detector.captcha_detector import detect_from_html, CaptchaType
from router.dispatcher import Dispatcher, SolveRequest, SolveResult, get_cached_detection
from engines.ocr_engine import OCREngine
from engines.vision_engine import VisionEngine
from engines.audio_engine import AudioEngine
from engines.token_engine import TokenEngine
from engines.behavior_engine import BehaviorEngine
from cache.token_cache import TokenCache
from browser.persistent_runner import PersistentBrowserRunner
from browser.browser_pool import BrowserPool
from engines.preharvest_daemon import PreharvestDaemon
from scheduler.profile_scheduler import ProfileScheduler
from test_runner.random_site_runner import RandomSiteRunner
from logs import setup_logging, log_solve, get_stats

logger = logging.getLogger(__name__)

# Globals
cache = TokenCache()
browser_pool = BrowserPool()
browser_runner = PersistentBrowserRunner()
scheduler = ProfileScheduler()
daemon = PreharvestDaemon(cache=cache)


def _build_engines() -> dict:
    return {
        "ocr": OCREngine(),
        "vision": VisionEngine(),
        "audio": AudioEngine(),
        "token_harvest": TokenEngine(browser_runner=browser_runner),
        "behavior": BehaviorEngine(browser_runner=browser_runner),
    }


dispatcher = Dispatcher(engines=_build_engines())
dispatcher.set_stats_provider(get_stats)
test_runner = RandomSiteRunner(dispatcher=dispatcher)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Captcha Solver Core v2.1.0 starting up...")

    # Speed Boost: YOLO + CLIP warm-start at boot
    logger.info("Warm-starting YOLO + CLIP models...")
    try:
        vision_engine = dispatcher.engines.get("vision")
        if vision_engine:
            vision_engine.warm_start()
            logger.info("YOLO + CLIP models warm-started successfully")
    except Exception as e:
        logger.warning(f"Model warm-start failed (will lazy-load on first request): {e}")

    # Start browser pool (adaptive scaling enabled)
    try:
        await browser_pool.start()
        logger.info(f"Browser pool started: {browser_pool.active_count} contexts (max={browser_pool.max_size})")
    except Exception as e:
        logger.warning(f"Browser pool start failed: {e}")

    # Start scheduler in background
    scheduler_task = asyncio.create_task(scheduler.run_scheduler())

    # Start random site benchmark scheduler
    test_runner.start_scheduler(interval_hours=config.test_runner_interval_hours)

    yield

    # Shutdown
    scheduler.stop()
    scheduler_task.cancel()
    test_runner.stop_scheduler()
    await daemon.stop()
    await browser_pool.stop()
    await browser_runner.stop()
    logger.info("Captcha Solver Core shut down.")


app = FastAPI(
    title="Captcha Solver Core",
    description="Modular captcha solving API with engine racing, adaptive routing, and continuous benchmarking",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ───────────────────────────────────────

class SolveRequestModel(BaseModel):
    captcha_type: str = Field("auto", description="Captcha type: auto, text, image_grid, recaptcha_v2, hcaptcha, turnstile")
    pageurl: str = Field("", description="URL of the page with captcha")
    sitekey: str | None = Field(None, description="Captcha sitekey (for reCAPTCHA/hCaptcha/Turnstile)")
    captcha_image_base64: str | None = Field(None, description="Base64-encoded captcha image")
    captcha_image_url: str | None = Field(None, description="URL to captcha image")
    extra: dict | None = Field(None, description="Additional params (target_label, audio_url, etc.)")


class SolveResponseModel(BaseModel):
    success: bool
    token: str | None = None
    engine_used: str = ""
    confidence: float = 0.0
    solve_time_ms: int = 0
    error: str | None = None
    cached: bool = False


class DetectRequestModel(BaseModel):
    html: str = Field(..., description="HTML source to analyze")
    url: str = Field("", description="Page URL for context")


class DetectResponseModel(BaseModel):
    captcha_type: str
    sitekey: str | None = None
    iframe_src: str | None = None
    confidence: float = 0.0
    cached: bool = False


class HarvestTargetModel(BaseModel):
    sitekey: str
    domain: str
    captcha_type: str
    pageurl: str
    min_tokens: int = 3
    max_tokens: int = 10


# ─── Endpoints ───────────────────────────────────────────────────────

@app.post("/solve", response_model=SolveResponseModel)
async def solve_captcha(req: SolveRequestModel):
    """Solve a captcha. Main endpoint."""

    # Check cache first
    if req.sitekey and req.pageurl:
        from urllib.parse import urlparse
        domain = urlparse(req.pageurl).netloc
        cached_token = cache.get(req.sitekey, domain, req.captcha_type)
        if cached_token:
            log_solve(req.captcha_type, "cache", True, 1.0, 0, req.pageurl, req.sitekey or "")
            return SolveResponseModel(
                success=True,
                token=cached_token,
                engine_used="cache",
                confidence=1.0,
                solve_time_ms=0,
                cached=True,
            )

    # Build solve request
    image_data = None
    if req.captcha_image_base64:
        image_data = base64.b64decode(req.captcha_image_base64)

    solve_req = SolveRequest(
        captcha_type=req.captcha_type,
        pageurl=req.pageurl,
        sitekey=req.sitekey,
        captcha_image=image_data,
        captcha_image_url=req.captcha_image_url,
        extra=req.extra,
    )

    detection = None
    result = await dispatcher.solve(solve_req, detection)

    log_solve(
        req.captcha_type, result.engine_used, result.success,
        result.confidence, result.solve_time_ms,
        req.pageurl, req.sitekey or "", result.error or ""
    )

    # Cache successful tokens
    if result.success and result.token and req.sitekey and req.pageurl:
        from urllib.parse import urlparse
        domain = urlparse(req.pageurl).netloc
        cache.store(req.sitekey, domain, req.captcha_type, result.token)

    return SolveResponseModel(
        success=result.success,
        token=result.token,
        engine_used=result.engine_used,
        confidence=result.confidence,
        solve_time_ms=result.solve_time_ms,
        error=result.error,
    )


@app.post("/solve/image", response_model=SolveResponseModel)
async def solve_image_captcha(file: UploadFile = File(...), captcha_type: str = "text"):
    """Solve a captcha from an uploaded image file."""
    image_data = await file.read()

    solve_req = SolveRequest(
        captcha_type=captcha_type,
        captcha_image=image_data,
    )

    result = await dispatcher.solve(solve_req)

    log_solve(captcha_type, result.engine_used, result.success,
              result.confidence, result.solve_time_ms)

    return SolveResponseModel(
        success=result.success,
        token=result.token,
        engine_used=result.engine_used,
        confidence=result.confidence,
        solve_time_ms=result.solve_time_ms,
        error=result.error,
    )


@app.post("/detect", response_model=DetectResponseModel)
async def detect_captcha(req: DetectRequestModel):
    """Detect captcha type from HTML source. Uses domain cache for repeat lookups."""
    # Check detector cache first
    from_cache = False
    if req.url and config.detector_cache_enabled:
        from urllib.parse import urlparse
        domain = urlparse(req.url).netloc
        cached_type = get_cached_detection(domain)
        if cached_type:
            from_cache = True
            return DetectResponseModel(
                captcha_type=cached_type.value,
                confidence=0.95,
                cached=True,
            )

    result = detect_from_html(req.html)
    return DetectResponseModel(
        captcha_type=result.captcha_type.value,
        sitekey=result.sitekey,
        iframe_src=result.iframe_src,
        confidence=result.confidence,
        cached=False,
    )


@app.post("/harvest/add")
async def add_harvest_target(target: HarvestTargetModel):
    """Add a target for the token pre-harvest daemon."""
    daemon.add_target(
        sitekey=target.sitekey,
        domain=target.domain,
        captcha_type=target.captcha_type,
        pageurl=target.pageurl,
        min_tokens=target.min_tokens,
        max_tokens=target.max_tokens,
    )
    return {"status": "added", "targets": len(daemon.targets)}


@app.post("/harvest/start")
async def start_harvest():
    """Start the token pre-harvest daemon."""
    await daemon.start()
    return {"status": "started", "targets": len(daemon.targets)}


@app.post("/harvest/stop")
async def stop_harvest():
    """Stop the token pre-harvest daemon."""
    await daemon.stop()
    return {"status": "stopped"}


@app.get("/harvest/status")
async def harvest_status():
    """Get pre-harvest daemon status."""
    return daemon.get_status()


@app.get("/cache/stats")
async def cache_stats():
    """Get token cache statistics."""
    return cache.get_stats()


@app.post("/cache/cleanup")
async def cache_cleanup():
    """Clean up expired tokens from cache."""
    cache.cleanup_expired()
    return cache.get_stats()


@app.get("/stats")
async def solver_stats(hours: int = 24):
    """Get solver telemetry statistics."""
    return get_stats(hours)


@app.post("/run_random_tests")
async def run_random_tests():
    """Trigger a single random-site test evaluation pass. Returns results."""
    summary = await test_runner.run_once()
    return summary


@app.post("/benchmark/run")
async def run_benchmark():
    """Run a full benchmark evaluation pass. Returns summary with fastest/slowest engines."""
    summary = await test_runner.run_once()
    return {
        "runs_completed": summary["tests_run"],
        "detections_successful": summary["detections_successful"],
        "solves_successful": summary["solves_successful"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "fastest_engine": summary.get("fastest_engine"),
        "slowest_engine": summary.get("slowest_engine"),
        "rolling_stats": summary.get("rolling_stats", {}),
    }


@app.get("/benchmark/history")
async def benchmark_history(days: int = 7):
    """Get rolling 7-day benchmark performance history."""
    return test_runner.get_history(days)


@app.get("/benchmark/rolling")
async def benchmark_rolling_stats():
    """Get current rolling performance statistics per engine."""
    return test_runner.get_rolling_stats()


@app.get("/test_runner/results")
async def test_runner_results(count: int = 20):
    """Get recent random-site test results."""
    return test_runner.get_latest_results(count)


@app.get("/test_runner/status")
async def test_runner_status():
    """Get test runner scheduler status."""
    return {
        "scheduler_running": test_runner._running,
        "targets": test_runner.targets,
        "interval_hours": config.test_runner_interval_hours,
    }


@app.get("/pool/status")
async def pool_status():
    """Get browser pool status (including adaptive scaling info)."""
    return {
        "active": browser_pool.active_count,
        "busy": browser_pool.busy_count,
        "pool_size": browser_pool.size,
        "max_size": browser_pool.max_size,
        "queue_depth": browser_pool.queue_depth,
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "version": "2.1.0",
        "engines": list(dispatcher.engines.keys()),
        "cache": cache.get_stats(),
        "browser_pool": {
            "active": browser_pool.active_count,
            "busy": browser_pool.busy_count,
            "max_size": browser_pool.max_size,
            "queue_depth": browser_pool.queue_depth,
        },
        "features": {
            "engine_racing": config.engine_racing_enabled,
            "arbitration_window_ms": config.arbitration_window_ms,
            "adaptive_routing": config.adaptive_routing_enabled,
            "detector_cache": config.detector_cache_enabled,
            "memory_cache": config.cache_use_memory,
            "whisper_model": config.whisper_model,
            "browser_pool_size": config.browser_pool_size,
            "browser_pool_max_size": config.browser_pool_max_size,
            "clip_preloaded": True,
        },
    }


@app.get("/")
async def root():
    return {
        "service": "Captcha Solver Core",
        "version": "2.1.0",
        "endpoints": {
            "POST /solve": "Solve a captcha (main endpoint)",
            "POST /solve/image": "Solve from uploaded image",
            "POST /detect": "Detect captcha type from HTML (cached by domain)",
            "POST /benchmark/run": "Run full benchmark evaluation",
            "GET /benchmark/history": "Rolling 7-day benchmark history",
            "GET /benchmark/rolling": "Rolling engine performance stats",
            "POST /run_random_tests": "Run random site captcha tests",
            "GET /test_runner/results": "Get recent test results",
            "GET /test_runner/status": "Test runner scheduler status",
            "POST /harvest/add": "Add pre-harvest target",
            "POST /harvest/start": "Start token pre-harvest daemon",
            "GET /harvest/status": "Pre-harvest daemon status",
            "GET /cache/stats": "Token cache statistics",
            "GET /stats": "Solver telemetry",
            "GET /pool/status": "Browser pool status (adaptive scaling)",
            "GET /health": "Health check",
        },
    }
